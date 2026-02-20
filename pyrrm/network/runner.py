"""
CatchmentNetworkRunner: upstream-to-downstream network calibration.

Supports two sequential calibration strategies:
  - incremental (default): calibrate each node upstream-first; at junctions
    the local RR model and (optionally) link routing parameters are
    calibrated jointly against observed flow.
  - residual: like incremental but link routing parameters are fixed
    (not calibrated). Only the local RR model is calibrated at each node.

Both strategies process nodes in topological (upstream-to-downstream) order,
with DAG-aware parallel scheduling exploiting the network topology via
wavefronts. Each node's calibration is configured by layered defaults +
per-node overrides.

Example:
    >>> from pyrrm.network import CatchmentNetwork, CatchmentNetworkRunner
    >>> network = CatchmentNetwork.from_csv('topology.csv', 'links.csv', data_dir='./data')
    >>> runner = CatchmentNetworkRunner(network, default_model_class='GR4J')
    >>> result = runner.run()
    >>> print(result.to_dataframe())
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Dict, List, Tuple, Optional, Any, Set, Type, TYPE_CHECKING,
)
import copy
import logging
import pickle
import time
import warnings

import numpy as np
import pandas as pd

from pyrrm.calibration.runner import CalibrationRunner, CalibrationResult
from pyrrm.calibration.report import CalibrationReport
from pyrrm.calibration.objective_functions import ObjectiveFunction, NSE
from pyrrm.parallel import create_backend, ParallelBackend
from pyrrm.network.topology import (
    CatchmentNode, NetworkLink, CatchmentNetwork, NodeCalibrationConfig,
)

if TYPE_CHECKING:
    from pyrrm.models.base import BaseRainfallRunoffModel
    from pyrrm.routing.base import BaseRouter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resolved configuration (after merging defaults + per-node overrides)
# ---------------------------------------------------------------------------

@dataclass
class ResolvedNodeConfig:
    """Effective calibration configuration for a node after merging layers."""
    model_class: str
    model_params: Optional[Dict[str, Any]]
    parameter_bounds: Optional[Dict[str, Tuple]]
    objective: ObjectiveFunction
    flow_transformation: Optional[Any]
    algorithm: Dict[str, Any]
    warmup_period: int


# ---------------------------------------------------------------------------
# NetworkCalibrationResult
# ---------------------------------------------------------------------------

@dataclass
class NetworkCalibrationResult:
    """Aggregated results from a network calibration run.

    Attributes:
        node_results: Per-node CalibrationReport (RR parameters).
        node_simulations: Per-node total outflow (local + routed upstream).
        aggregated_flows: Per-junction routed upstream sum.
        link_routing_params: (upstream, downstream) -> calibrated routing params.
        link_routed_flows: (upstream, downstream) -> routed flow on each link.
        strategy: Calibration strategy used.
        network: The CatchmentNetwork (topology reference).
        runtime_seconds: Total wall-clock time.
        failures: Per-node error messages for failed calibrations.
    """
    node_results: Dict[str, CalibrationReport] = field(default_factory=dict)
    node_simulations: Dict[str, np.ndarray] = field(default_factory=dict)
    aggregated_flows: Dict[str, np.ndarray] = field(default_factory=dict)
    link_routing_params: Dict[Tuple[str, str], Dict] = field(default_factory=dict)
    link_routed_flows: Dict[Tuple[str, str], np.ndarray] = field(default_factory=dict)
    strategy: str = 'incremental'
    network: Optional[CatchmentNetwork] = field(default=None, repr=False)
    runtime_seconds: float = 0.0
    failures: Dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        """Summary table: one row per node with RR params, objective, etc."""
        rows = []
        for nid, report in self.node_results.items():
            row = {
                'node_id': nid,
                'model': report.model_config.get('class', '') if hasattr(report, 'model_config') and report.model_config else '',
                'objective_name': report.result.objective_name,
                'best_objective': report.result.best_objective,
                'runtime_seconds': report.result.runtime_seconds,
                'n_params': len(report.result.best_parameters),
                'method': report.result.method,
                'success': report.result.success,
            }
            for pname, pval in report.result.best_parameters.items():
                row[f'param_{pname}'] = pval
            rows.append(row)
        for nid, err in self.failures.items():
            rows.append({
                'node_id': nid,
                'best_objective': np.nan,
                'success': False,
                'error': err,
            })
        return pd.DataFrame(rows)

    def link_summary(self) -> pd.DataFrame:
        """One row per link with calibrated routing parameters."""
        rows = []
        for (uid, did), params in self.link_routing_params.items():
            row = {'upstream_id': uid, 'downstream_id': did}
            row.update(params)
            rows.append(row)
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("NetworkCalibrationResult saved to %s", path)

    @classmethod
    def load(cls, path: str) -> 'NetworkCalibrationResult':
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected NetworkCalibrationResult, got {type(obj).__name__}")
        return obj

    def __repr__(self) -> str:
        n_ok = len(self.node_results)
        n_fail = len(self.failures)
        return (
            f"NetworkCalibrationResult({n_ok} nodes calibrated, "
            f"{n_fail} failed, strategy='{self.strategy}', "
            f"{self.runtime_seconds:.1f}s)"
        )


# ---------------------------------------------------------------------------
# Model factory helper
# ---------------------------------------------------------------------------

class _JunctionModelWrapper:
    """Wraps a local RR model so its run() output includes routed upstream.

    This makes a junction's calibration look identical to a headwater
    calibration from CalibrationRunner's perspective. The wrapper delegates
    all parameter/bounds methods to the underlying model.
    """

    def __init__(self, local_model: Any, routed_upstream: np.ndarray):
        self._model = local_model
        self._routed = routed_upstream

    def run(self, inputs: pd.DataFrame) -> pd.DataFrame:
        self._model.reset()
        result_df = self._model.run(inputs)
        local_flow = result_df['flow'].values
        n = min(len(local_flow), len(self._routed))
        total_flow = local_flow.copy()
        total_flow[:n] += self._routed[:n]
        out = result_df.copy()
        out['flow'] = total_flow
        return out

    def get_parameter_bounds(self):
        return self._model.get_parameter_bounds()

    def set_parameters(self, params):
        return self._model.set_parameters(params)

    def reset(self):
        return self._model.reset()

    @property
    def parameters(self):
        return self._model.parameters

    @parameters.setter
    def parameters(self, val):
        self._model.parameters = val

    def __getattr__(self, name):
        return getattr(self._model, name)


def _create_model(model_class: str, model_params: Optional[Dict] = None):
    """Instantiate a model by class name."""
    from pyrrm.calibration.batch import get_model_class
    cls = get_model_class(model_class)
    return cls(**(model_params or {}))


def _create_router(link: NetworkLink):
    """Create a BaseRouter instance for a link."""
    method = (link.routing_method or 'none').lower()
    if method == 'none':
        return None
    if method == 'muskingum':
        from pyrrm.routing.muskingum import NonlinearMuskingumRouter
        params = link.routing_params or {}
        return NonlinearMuskingumRouter(
            K=params.get('routing_K', 5.0),
            m=params.get('routing_m', 0.8),
            n_subreaches=int(params.get('routing_n_subreaches', 3)),
        )
    raise ValueError(f"Unknown routing method: '{method}'")


# ---------------------------------------------------------------------------
# CatchmentNetworkRunner
# ---------------------------------------------------------------------------

class CatchmentNetworkRunner:
    """DAG-aware upstream-to-downstream calibration of a catchment network.

    Args:
        network: The CatchmentNetwork DAG.
        default_model_class: Default RR model name (e.g. 'GR4J').
        default_model_params: Default model constructor kwargs.
        default_objective: Default objective function.
        default_flow_transformation: Default flow transformation.
        default_algorithm: Default calibration algorithm config.
        default_warmup_period: Default warmup days.
        output_dir: Directory for per-node result persistence.
        strategy: 'incremental' or 'residual'.
        link_routing: Whether to apply and optionally calibrate link routing.
        backend: Parallelisation backend name.
        max_workers: Number of workers for multiprocessing/ray.
    """

    def __init__(
        self,
        network: CatchmentNetwork,
        default_model_class: str = 'GR4J',
        default_model_params: Optional[Dict[str, Any]] = None,
        default_objective: Optional[ObjectiveFunction] = None,
        default_flow_transformation: Optional[Any] = None,
        default_algorithm: Optional[Dict[str, Any]] = None,
        default_warmup_period: int = 365,
        output_dir: str = './network_results',
        strategy: str = 'incremental',
        link_routing: bool = False,
        backend: str = 'sequential',
        max_workers: Optional[int] = None,
    ):
        self.network = network
        self.default_model_class = default_model_class
        self.default_model_params = default_model_params
        self.default_objective = default_objective or NSE()
        self.default_flow_transformation = default_flow_transformation
        self.default_algorithm = default_algorithm or {
            'method': 'sceua_direct', 'max_evals': 20000,
        }
        self.default_warmup_period = default_warmup_period
        self.output_dir = Path(output_dir)
        self.strategy = strategy.lower()
        self.link_routing = link_routing
        self.backend_name = backend
        self.max_workers = max_workers

        if self.strategy not in ('incremental', 'residual'):
            raise ValueError(
                f"Unknown strategy: '{self.strategy}'. "
                f"Choose from: 'incremental', 'residual'"
            )

    # ------------------------------------------------------------------
    # Configuration resolution
    # ------------------------------------------------------------------
    def _resolve_config(self, node: CatchmentNode) -> ResolvedNodeConfig:
        """Merge network defaults with per-node overrides."""
        nc = node.calibration_config or NodeCalibrationConfig()
        return ResolvedNodeConfig(
            model_class=nc.model_class or self.default_model_class,
            model_params=nc.model_params or self.default_model_params,
            parameter_bounds=nc.parameter_bounds,
            objective=nc.objective or copy.deepcopy(self.default_objective),
            flow_transformation=nc.flow_transformation or self.default_flow_transformation,
            algorithm=nc.algorithm or dict(self.default_algorithm),
            warmup_period=nc.warmup_period or self.default_warmup_period,
        )

    def get_resolved_configs(self) -> Dict[str, ResolvedNodeConfig]:
        """Preview the effective config for each node."""
        return {
            nid: self._resolve_config(node)
            for nid, node in self.network.nodes.items()
        }

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------
    def run(self, resume: bool = True) -> NetworkCalibrationResult:
        """Execute the network calibration.

        Args:
            resume: If True, skip nodes whose results already exist
                in ``output_dir``.

        Returns:
            NetworkCalibrationResult.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self._run_sequential(resume)

    # ------------------------------------------------------------------
    # Sequential strategies (incremental / residual)
    # ------------------------------------------------------------------
    def _run_sequential(self, resume: bool) -> NetworkCalibrationResult:
        """Run incremental or residual strategy with wavefront scheduling."""
        t0 = time.time()
        wavefronts = self.network.wavefronts()

        node_results: Dict[str, CalibrationReport] = {}
        node_simulations: Dict[str, np.ndarray] = {}
        aggregated_flows: Dict[str, np.ndarray] = {}
        link_routing_params: Dict[Tuple[str, str], Dict] = {}
        link_routed_flows: Dict[Tuple[str, str], np.ndarray] = {}
        failures: Dict[str, str] = {}

        if resume:
            for nid in self.network.node_ids:
                pkl_path = self.output_dir / f"{nid}.pkl"
                sim_path = self.output_dir / f"{nid}_sim.npy"
                if pkl_path.exists():
                    try:
                        report = CalibrationReport.load(str(pkl_path))
                        node_results[nid] = report
                        if sim_path.exists():
                            node_simulations[nid] = np.load(str(sim_path))
                        logger.info("Resumed node: %s", nid)
                    except Exception as e:
                        logger.warning("Failed to resume %s: %s", nid, e)

        backend = create_backend(self.backend_name, self.max_workers)
        try:
            for wf_idx, wf_node_ids in enumerate(wavefronts):
                remaining = [nid for nid in wf_node_ids if nid not in node_results]
                if not remaining:
                    logger.info("Wavefront %d/%d: all %d nodes already done",
                                wf_idx + 1, len(wavefronts), len(wf_node_ids))
                    continue

                logger.info(
                    "Wavefront %d/%d: calibrating %d nodes: %s",
                    wf_idx + 1, len(wavefronts), len(remaining), remaining,
                )

                def _calibrate(nid: str) -> Tuple[str, CalibrationReport, np.ndarray, Dict, Dict]:
                    return self._calibrate_node(
                        nid, node_simulations, link_routing_params, link_routed_flows,
                    )

                wf_tasks = [(nid, nid) for nid in remaining]
                wf_results = backend.map(
                    _calibrate,
                    remaining,
                    task_ids=remaining,
                    on_error=lambda tid, exc: failures.__setitem__(tid, str(exc)),
                )

                for nid, result_tuple in wf_results.items():
                    _, report, sim, lrp, lrf = result_tuple
                    node_results[nid] = report
                    node_simulations[nid] = sim

                    for key, params in lrp.items():
                        link_routing_params[key] = params
                    for key, flow in lrf.items():
                        link_routed_flows[key] = flow

                    try:
                        report.save(str(self.output_dir / f"{nid}.pkl"))
                        np.save(str(self.output_dir / f"{nid}_sim.npy"), sim)
                    except Exception as e:
                        logger.warning("Failed to save results for %s: %s", nid, e)

        finally:
            backend.shutdown()

        for nid in node_simulations:
            us_ids = self.network.upstream_ids(nid)
            if us_ids:
                routed_sum = np.zeros_like(node_simulations[nid])
                for uid in us_ids:
                    key = (uid, nid)
                    if key in link_routed_flows:
                        n = min(len(routed_sum), len(link_routed_flows[key]))
                        routed_sum[:n] += link_routed_flows[key][:n]
                    elif uid in node_simulations:
                        n = min(len(routed_sum), len(node_simulations[uid]))
                        routed_sum[:n] += node_simulations[uid][:n]
                aggregated_flows[nid] = routed_sum

        elapsed = time.time() - t0
        return NetworkCalibrationResult(
            node_results=node_results,
            node_simulations=node_simulations,
            aggregated_flows=aggregated_flows,
            link_routing_params=link_routing_params,
            link_routed_flows=link_routed_flows,
            strategy=self.strategy,
            network=self.network,
            runtime_seconds=elapsed,
            failures=failures,
        )

    # ------------------------------------------------------------------
    # Calibrate a single node (for incremental / residual)
    # ------------------------------------------------------------------
    def _calibrate_node(
        self,
        node_id: str,
        upstream_sims: Dict[str, np.ndarray],
        existing_link_params: Dict[Tuple[str, str], Dict],
        existing_link_flows: Dict[Tuple[str, str], np.ndarray],
    ) -> Tuple[str, CalibrationReport, np.ndarray, Dict, Dict]:
        """Calibrate one node and return (node_id, report, simulation, link_params, link_flows)."""
        node = self.network.get_node(node_id)
        cfg = self._resolve_config(node)

        if node.inputs is None:
            raise ValueError(f"Node '{node_id}' has no input data loaded")

        model = _create_model(cfg.model_class, cfg.model_params)
        if hasattr(model, 'catchment_area_km2') and node.area_km2 > 0:
            model.catchment_area_km2 = node.area_km2

        upstream_ids = self.network.upstream_ids(node_id)
        is_junction = len(upstream_ids) > 0

        link_params_out: Dict[Tuple[str, str], Dict] = {}
        link_flows_out: Dict[Tuple[str, str], np.ndarray] = {}

        if not is_junction:
            report, sim = self._calibrate_headwater(node, model, cfg)
            return (node_id, report, sim, link_params_out, link_flows_out)

        upstream_flow_series: Dict[str, np.ndarray] = {}
        for uid in upstream_ids:
            if uid in upstream_sims:
                upstream_flow_series[uid] = upstream_sims[uid]
            else:
                logger.warning(
                    "Upstream %s simulation not available for junction %s",
                    uid, node_id,
                )

        if not upstream_flow_series:
            report, sim = self._calibrate_headwater(node, model, cfg)
            return (node_id, report, sim, link_params_out, link_flows_out)

        routers: Dict[str, Any] = {}
        if self.link_routing:
            for uid in upstream_ids:
                link = self.network.get_link(uid, node_id)
                if link is not None:
                    router = _create_router(link)
                    if router is not None:
                        routers[uid] = (router, link)

        if (self.strategy == 'incremental' and self.link_routing
                and routers and node.is_gauged):
            report, sim, lrp, lrf = self._calibrate_junction_with_routing(
                node, model, cfg, upstream_flow_series, routers,
            )
            link_params_out.update(lrp)
            link_flows_out.update(lrf)
            return (node_id, report, sim, link_params_out, link_flows_out)

        routed_upstream = self._aggregate_upstream(
            node_id, upstream_flow_series, routers, existing_link_params,
        )
        for uid, (router, link) in routers.items():
            if uid in upstream_flow_series:
                params = existing_link_params.get((uid, node_id), {})
                if params:
                    router.set_parameters(params)
                routed = router.route(upstream_flow_series[uid], dt=1.0)
                link_flows_out[(uid, node_id)] = routed
                link_params_out[(uid, node_id)] = router.get_parameters()

        if node.is_gauged and node.observed is not None:
            report, sim = self._calibrate_junction_simple(
                node, model, cfg, routed_upstream,
            )
        else:
            model.reset()
            result_df = model.run(node.inputs)
            local_flow = result_df['flow'].values
            sim = local_flow + routed_upstream[:len(local_flow)]
            fake_result = CalibrationResult(
                best_parameters=model.parameters if hasattr(model, 'parameters') else {},
                best_objective=float('nan'),
                all_samples=pd.DataFrame(),
                method='none',
                objective_name='none',
                success=True,
                message='Ungauged node -- no calibration performed',
            )
            report = CalibrationReport(
                result=fake_result,
                observed=np.array([]),
                simulated=sim,
                dates=node.inputs.index,
            )

        return (node_id, report, sim, link_params_out, link_flows_out)

    # ------------------------------------------------------------------
    # Headwater calibration
    # ------------------------------------------------------------------
    def _calibrate_headwater(
        self, node: CatchmentNode, model: Any, cfg: ResolvedNodeConfig,
    ) -> Tuple[CalibrationReport, np.ndarray]:
        """Calibrate a headwater node (no upstream contributions)."""
        if not node.is_gauged or node.observed is None:
            model.reset()
            result_df = model.run(node.inputs)
            sim = result_df['flow'].values
            fake_result = CalibrationResult(
                best_parameters=model.parameters if hasattr(model, 'parameters') else {},
                best_objective=float('nan'),
                all_samples=pd.DataFrame(),
                method='none',
                objective_name='none',
                success=True,
                message='Ungauged headwater -- no calibration',
            )
            report = CalibrationReport(
                result=fake_result,
                observed=np.array([]),
                simulated=sim,
                dates=node.inputs.index,
            )
            return report, sim

        objective = copy.deepcopy(cfg.objective)
        if cfg.flow_transformation is not None and hasattr(objective, 'transform'):
            objective.transform = cfg.flow_transformation

        cal_runner = CalibrationRunner(
            model=model,
            inputs=node.inputs,
            observed=node.observed,
            objective=objective,
            parameter_bounds=cfg.parameter_bounds,
            warmup_period=cfg.warmup_period,
        )

        alg_kwargs = dict(cfg.algorithm)
        method = alg_kwargs.pop('method', 'sceua_direct')

        run_fn = getattr(cal_runner, f'run_{method}', None)
        if run_fn is None:
            if method in ('differential_evolution', 'dual_annealing', 'basin_hopping'):
                run_fn = lambda **kw: cal_runner.run_scipy(method=method, **kw)
            else:
                raise ValueError(f"Unknown calibration method: '{method}'")

        result = run_fn(**alg_kwargs)
        report = cal_runner.create_report(result)

        model.reset()
        model.parameters = result.best_parameters
        result_df = model.run(node.inputs)
        sim = result_df['flow'].values

        return report, sim

    # ------------------------------------------------------------------
    # Junction calibration (simple -- no routing calibration)
    # ------------------------------------------------------------------
    def _calibrate_junction_simple(
        self,
        node: CatchmentNode,
        model: Any,
        cfg: ResolvedNodeConfig,
        routed_upstream: np.ndarray,
    ) -> Tuple[CalibrationReport, np.ndarray]:
        """Calibrate local RR model at a junction with fixed upstream routing.

        Uses a JunctionModelWrapper that adds routed upstream to the local
        model's output, so CalibrationRunner sees a standard model interface.
        """
        assert node.observed is not None

        n_obs = len(node.observed)
        n_inputs = len(node.inputs)
        n_routed = len(routed_upstream)
        n = min(n_obs, n_inputs, n_routed)

        observed_trimmed = node.observed[:n]
        inputs_trimmed = node.inputs.iloc[:n]
        routed_trimmed = routed_upstream[:n]

        wrapper = _JunctionModelWrapper(model, routed_trimmed)

        objective = copy.deepcopy(cfg.objective)
        if cfg.flow_transformation is not None and hasattr(objective, 'transform'):
            objective.transform = cfg.flow_transformation

        warmup = min(cfg.warmup_period, n - 1)

        cal_runner = CalibrationRunner(
            model=wrapper,
            inputs=inputs_trimmed,
            observed=observed_trimmed,
            objective=objective,
            parameter_bounds=cfg.parameter_bounds,
            warmup_period=warmup,
        )

        alg_kwargs = dict(cfg.algorithm)
        method = alg_kwargs.pop('method', 'sceua_direct')
        run_fn = getattr(cal_runner, f'run_{method}', None)
        if run_fn is None:
            if method in ('differential_evolution', 'dual_annealing', 'basin_hopping'):
                run_fn = lambda **kw: cal_runner.run_scipy(method=method, **kw)
            else:
                raise ValueError(f"Unknown calibration method: '{method}'")

        result = run_fn(**alg_kwargs)

        model.reset()
        model.parameters = result.best_parameters
        result_df = model.run(inputs_trimmed)
        local_flow = result_df['flow'].values
        total_sim = local_flow + routed_trimmed[:len(local_flow)]

        report = CalibrationReport(
            result=result,
            observed=observed_trimmed[warmup:],
            simulated=total_sim[warmup:],
            dates=inputs_trimmed.index[warmup:],
            inputs=inputs_trimmed,
            warmup_days=warmup,
        )

        return report, total_sim

    # ------------------------------------------------------------------
    # Junction calibration WITH routing parameter calibration
    # ------------------------------------------------------------------
    def _calibrate_junction_with_routing(
        self,
        node: CatchmentNode,
        model: Any,
        cfg: ResolvedNodeConfig,
        upstream_sims: Dict[str, np.ndarray],
        routers: Dict[str, Tuple[Any, NetworkLink]],
    ) -> Tuple[CalibrationReport, np.ndarray, Dict[Tuple[str, str], Dict], Dict[Tuple[str, str], np.ndarray]]:
        """Calibrate local RR + link routing params jointly at a junction.

        The parameter vector is: [local_RR_params] + [link1_routing_params] + [link2_routing_params] + ...
        Link routing params use prefix: upstream_id__routing_K, etc.
        """
        assert node.observed is not None

        rr_bounds = cfg.parameter_bounds or model.get_parameter_bounds()
        rr_names = list(rr_bounds.keys())

        link_param_layout: List[Tuple[str, str, str]] = []
        all_bounds: Dict[str, Tuple[float, float]] = dict(rr_bounds)

        for uid, (router, link) in routers.items():
            if not link.calibrate_routing:
                continue
            router_bounds = router.get_parameter_bounds()
            custom_bounds = link.routing_bounds or {}
            for pname, (lo, hi) in router_bounds.items():
                prefixed = f"{uid}__{pname}"
                if pname in custom_bounds:
                    lo, hi = custom_bounds[pname]
                elif prefixed in custom_bounds:
                    lo, hi = custom_bounds[prefixed]
                all_bounds[prefixed] = (lo, hi)
                link_param_layout.append((uid, pname, prefixed))

        all_names = list(all_bounds.keys())
        n_rr = len(rr_names)

        n_obs = len(node.observed)
        n_inputs = len(node.inputs)
        n = min(n_obs, n_inputs)
        for uid, sim in upstream_sims.items():
            n = min(n, len(sim))

        observed_trimmed = node.observed[:n]
        inputs_trimmed = node.inputs.iloc[:n]
        warmup = min(cfg.warmup_period, n - 1)

        objective = copy.deepcopy(cfg.objective)
        if cfg.flow_transformation is not None and hasattr(objective, 'transform'):
            objective.transform = cfg.flow_transformation

        def _evaluate(param_array: np.ndarray) -> float:
            rr_params = dict(zip(rr_names, param_array[:n_rr]))

            for uid_name, pname, prefixed in link_param_layout:
                idx = all_names.index(prefixed)
                router_obj, _ = routers[uid_name]
                router_obj.set_parameters({pname: param_array[idx]})

            routed_sum = np.zeros(n)
            for uid, sim in upstream_sims.items():
                if uid in routers:
                    router_obj, _ = routers[uid]
                    router_obj.reset()
                    routed = router_obj.route(sim[:n], dt=1.0)
                    routed_sum[:len(routed)] += routed
                else:
                    routed_sum[:min(n, len(sim))] += sim[:min(n, len(sim))]

            model.reset()
            model.parameters = rr_params
            result_df = model.run(inputs_trimmed)
            local_flow = result_df['flow'].values
            total_flow = local_flow + routed_sum[:len(local_flow)]

            sim_eval = total_flow[warmup:]
            obs_eval = observed_trimmed[warmup:]
            n_eval = min(len(sim_eval), len(obs_eval))

            from pyrrm.calibration.objective_functions import get_calibration_value
            return get_calibration_value(objective, sim_eval[:n_eval], obs_eval[:n_eval])

        from pyrrm.calibration._sceua import minimize as sceua_minimize
        from pyrrm.calibration.objective_functions import is_new_interface

        bounds_list = [all_bounds[name] for name in all_names]

        should_maximize = True
        if hasattr(objective, 'direction'):
            should_maximize = objective.direction == 'maximize'
        elif is_new_interface(objective):
            should_maximize = True

        def _minimization_func(x):
            val = _evaluate(x)
            return -val if should_maximize else val

        alg_kwargs = dict(cfg.algorithm)
        max_evals = alg_kwargs.get('max_evals', 20000)
        seed = alg_kwargs.get('seed', None)

        sceua_result = sceua_minimize(
            func=_minimization_func,
            bounds=bounds_list,
            max_evals=max_evals,
            seed=seed,
        )

        best_params_arr = sceua_result.x
        best_obj = -sceua_result.fun if should_maximize else sceua_result.fun
        best_rr = dict(zip(rr_names, best_params_arr[:n_rr]))

        link_params_out: Dict[Tuple[str, str], Dict] = {}
        link_flows_out: Dict[Tuple[str, str], np.ndarray] = {}

        for uid_name, pname, prefixed in link_param_layout:
            idx = all_names.index(prefixed)
            key = (uid_name, node.id)
            link_params_out.setdefault(key, {})[pname] = best_params_arr[idx]
            router_obj, _ = routers[uid_name]
            router_obj.set_parameters({pname: best_params_arr[idx]})

        routed_sum = np.zeros(n)
        for uid, sim in upstream_sims.items():
            if uid in routers:
                router_obj, _ = routers[uid]
                router_obj.reset()
                routed = router_obj.route(sim[:n], dt=1.0)
                routed_sum[:len(routed)] += routed
                link_flows_out[(uid, node.id)] = routed
            else:
                routed_sum[:min(n, len(sim))] += sim[:min(n, len(sim))]

        model.reset()
        model.parameters = best_rr
        result_df = model.run(inputs_trimmed)
        local_flow = result_df['flow'].values
        total_sim = local_flow + routed_sum[:len(local_flow)]

        cal_result = CalibrationResult(
            best_parameters={**best_rr, **{
                p: float(best_params_arr[all_names.index(p)])
                for _, _, p in link_param_layout
            }},
            best_objective=best_obj,
            all_samples=pd.DataFrame(),
            runtime_seconds=0.0,
            method='sceua_direct',
            objective_name=getattr(objective, 'name', str(objective)),
            success=sceua_result.success,
            message=f"Joint RR+routing calibration at junction {node.id}",
        )

        report = CalibrationReport(
            result=cal_result,
            observed=observed_trimmed[warmup:],
            simulated=total_sim[warmup:],
            dates=inputs_trimmed.index[warmup:],
            inputs=inputs_trimmed,
            warmup_days=warmup,
        )

        return report, total_sim, link_params_out, link_flows_out

    # ------------------------------------------------------------------
    # Flow aggregation helper
    # ------------------------------------------------------------------
    def _aggregate_upstream(
        self,
        node_id: str,
        upstream_sims: Dict[str, np.ndarray],
        routers: Dict[str, Tuple[Any, NetworkLink]],
        existing_params: Dict[Tuple[str, str], Dict],
    ) -> np.ndarray:
        """Route and sum upstream outflows at a junction.

        For the 'residual' strategy, routing parameters are fixed.
        """
        max_len = max((len(s) for s in upstream_sims.values()), default=0)
        routed_sum = np.zeros(max_len)

        for uid, sim in upstream_sims.items():
            if uid in routers:
                router_obj, link = routers[uid]
                params = existing_params.get((uid, node_id), {})
                if params:
                    router_obj.set_parameters(params)
                router_obj.reset()
                routed = router_obj.route(sim, dt=1.0)
                n = min(len(routed_sum), len(routed))
                routed_sum[:n] += routed[:n]
            else:
                n = min(len(routed_sum), len(sim))
                routed_sum[:n] += sim[:n]

        return routed_sum


__all__ = [
    'ResolvedNodeConfig',
    'NetworkCalibrationResult',
    'CatchmentNetworkRunner',
]
