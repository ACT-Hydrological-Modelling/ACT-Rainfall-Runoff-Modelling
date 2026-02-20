"""
Batch experiment runner for single-catchment grid experiments.

Automates running multiple calibrations across combinations of
rainfall-runoff models, objective functions, optimization algorithms,
and flow transformations.

Example (programmatic):
    >>> from pyrrm.calibration.batch import ExperimentGrid, BatchExperimentRunner
    >>> from pyrrm.models import GR4J, GR5J
    >>> from pyrrm.objectives import NSE, KGE
    >>>
    >>> grid = ExperimentGrid(
    ...     models={'GR4J': GR4J(), 'GR5J': GR5J()},
    ...     objectives={'nse': NSE(), 'kge': KGE()},
    ...     algorithms={'sceua': {'method': 'sceua_direct', 'max_evals': 20000}},
    ... )
    >>> runner = BatchExperimentRunner(inputs, observed, grid, output_dir='./results')
    >>> batch_result = runner.run()

Example (YAML-driven):
    >>> runner = BatchExperimentRunner.from_config('experiment.yaml', inputs, observed)
    >>> batch_result = runner.run()
"""

from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import (
    Dict, List, Tuple, Optional, Any, Callable, Type, TYPE_CHECKING,
)
import copy
import json
import logging
import pickle
import time
import traceback
import warnings

import numpy as np
import pandas as pd

from pyrrm.calibration.runner import CalibrationRunner, CalibrationResult
from pyrrm.calibration.report import CalibrationReport
from pyrrm.parallel import create_backend, ParallelBackend

if TYPE_CHECKING:
    from pyrrm.models.base import BaseRainfallRunoffModel
    from pyrrm.calibration.objective_functions import ObjectiveFunction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# YAML / JSON optional dependency
# ---------------------------------------------------------------------------
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# ---------------------------------------------------------------------------
# Model registry -- maps string names to model classes
# ---------------------------------------------------------------------------
_MODEL_REGISTRY: Dict[str, Type] = {}


def _ensure_model_registry():
    """Lazy-populate the model registry on first use."""
    if _MODEL_REGISTRY:
        return
    from pyrrm.models import Sacramento, GR4J, GR5J, GR6J
    _MODEL_REGISTRY.update({
        'sacramento': Sacramento,
        'gr4j': GR4J,
        'gr5j': GR5J,
        'gr6j': GR6J,
    })


def get_model_class(name: str) -> Type:
    """Resolve a model name to its class.

    Args:
        name: Model name (case-insensitive), e.g. 'GR4J', 'Sacramento'.

    Returns:
        The model class.

    Raises:
        ValueError: If the model name is not recognised.
    """
    _ensure_model_registry()
    key = name.lower()
    if key not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: '{name}'. "
            f"Choose from: {list(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[key]


# ---------------------------------------------------------------------------
# Objective registry -- maps string names to objective factory functions
# ---------------------------------------------------------------------------
def _build_objective(spec: Dict[str, Any]):
    """Build an ObjectiveFunction from a spec dict.

    Spec format:
        {'type': 'NSE'}
        {'type': 'KGE', 'variant': '2012'}
        {'type': 'NSE', 'transform': {'type': 'sqrt'}}
    """
    from pyrrm.calibration.objective_functions import (
        NSE, KGE, RMSE, MAE, PBIAS, LogNSE,
    )
    try:
        from pyrrm.objectives import (
            FlowTransformation, SDEB, KGENonParametric,
        )
        _new_objs = True
    except ImportError:
        _new_objs = False

    obj_type = spec.get('type', 'NSE').upper()
    transform_spec = spec.get('transform', None)
    transform = None
    if transform_spec and _new_objs:
        transform = FlowTransformation(transform_spec.get('type', 'none'))

    obj_map = {
        'NSE': NSE,
        'KGE': KGE,
        'RMSE': RMSE,
        'MAE': MAE,
        'PBIAS': PBIAS,
        'LOGNSE': LogNSE,
    }
    if _new_objs:
        obj_map['SDEB'] = SDEB
        obj_map['KGENP'] = KGENonParametric
        obj_map['KGENONPARAMETRIC'] = KGENonParametric

    cls = obj_map.get(obj_type)
    if cls is None:
        raise ValueError(
            f"Unknown objective type: '{obj_type}'. "
            f"Choose from: {list(obj_map.keys())}"
        )

    kwargs = {k: v for k, v in spec.items() if k not in ('type', 'transform')}
    obj = cls(**kwargs)

    if transform is not None and hasattr(obj, 'transform'):
        obj.transform = transform
    return obj


# ---------------------------------------------------------------------------
# ExperimentSpec -- single experiment definition
# ---------------------------------------------------------------------------
@dataclass
class ExperimentSpec:
    """A single experiment configuration -- one (model, objective, algorithm, transformation) tuple.

    Attributes:
        key: Unique string identifier, e.g. 'GR4J__nse__sceua' or 'GR4J__nse__sceua__sqrt'.
        model_name: Name of the model.
        model: Model instance (deep-copied for isolation).
        objective_name: Name of the objective function.
        objective: Objective function instance.
        algorithm_name: Name of the optimisation algorithm.
        algorithm_kwargs: Keyword arguments for the calibration method.
        transformation_name: Optional flow transformation name.
        transformation: Optional FlowTransformation instance.
    """
    key: str
    model_name: str
    model: Any
    objective_name: str
    objective: Any
    algorithm_name: str
    algorithm_kwargs: Dict[str, Any]
    transformation_name: Optional[str] = None
    transformation: Optional[Any] = None


# ---------------------------------------------------------------------------
# ExperimentGrid -- defines the combinatorial axes
# ---------------------------------------------------------------------------
@dataclass
class ExperimentGrid:
    """Defines the combinatorial axes of a batch experiment.

    Each axis is a dict mapping a human-readable name to an instance.
    Only specified axes are crossed; optional axes default to a single
    ``None`` entry so they don't multiply the grid.

    Args:
        models: ``{'GR4J': GR4J(), 'Sacramento': Sacramento()}``.
        objectives: ``{'nse': NSE(), 'kge': KGE()}``.
        algorithms: ``{'sceua': {'method': 'sceua_direct', 'max_evals': 20000}}``.
        transformations: Optional axis for flow transformations.

    Example:
        >>> grid = ExperimentGrid(
        ...     models={'GR4J': GR4J()},
        ...     objectives={'nse': NSE(), 'kge': KGE()},
        ...     algorithms={'sceua': {'method': 'sceua_direct', 'max_evals': 20000}},
        ... )
        >>> specs = grid.combinations()
        >>> len(specs)
        2
    """
    models: Dict[str, Any]
    objectives: Dict[str, Any]
    algorithms: Dict[str, Dict[str, Any]]
    transformations: Optional[Dict[str, Any]] = None

    def combinations(self) -> List[ExperimentSpec]:
        """Generate all combinations as ExperimentSpec objects."""
        trans_items = (
            list(self.transformations.items())
            if self.transformations
            else [(None, None)]
        )
        specs = []
        for (m_name, model), (o_name, obj), (a_name, alg_kw), (t_name, trans) in product(
            self.models.items(),
            self.objectives.items(),
            self.algorithms.items(),
            trans_items,
        ):
            parts = [m_name, o_name, a_name]
            if t_name is not None:
                parts.append(t_name)
            key = '__'.join(parts)
            specs.append(ExperimentSpec(
                key=key,
                model_name=m_name,
                model=copy.deepcopy(model),
                objective_name=o_name,
                objective=copy.deepcopy(obj),
                algorithm_name=a_name,
                algorithm_kwargs=dict(alg_kw),
                transformation_name=t_name,
                transformation=copy.deepcopy(trans) if trans else None,
            ))
        return specs

    def __len__(self) -> int:
        n = len(self.models) * len(self.objectives) * len(self.algorithms)
        if self.transformations:
            n *= len(self.transformations)
        return n


# ---------------------------------------------------------------------------
# BatchResult -- aggregated results
# ---------------------------------------------------------------------------
@dataclass
class BatchResult:
    """Aggregated results from a batch experiment run.

    Attributes:
        results: Mapping of experiment key to CalibrationReport.
        failures: Mapping of experiment key to error message.
        runtime_seconds: Total wall-clock time.
    """
    results: Dict[str, CalibrationReport] = field(default_factory=dict)
    failures: Dict[str, str] = field(default_factory=dict)
    runtime_seconds: float = 0.0

    def to_dataframe(self) -> pd.DataFrame:
        """Summary table: one row per experiment.

        Columns include model, objective, algorithm, transformation,
        best_objective, runtime_seconds, n_parameters, and success.
        """
        rows = []
        for key, report in self.results.items():
            parts = key.split('__')
            row = {
                'key': key,
                'model': parts[0] if len(parts) > 0 else '',
                'objective': parts[1] if len(parts) > 1 else '',
                'algorithm': parts[2] if len(parts) > 2 else '',
                'transformation': parts[3] if len(parts) > 3 else 'none',
                'best_objective': report.result.best_objective,
                'runtime_seconds': report.result.runtime_seconds,
                'n_parameters': len(report.result.best_parameters),
                'method': report.result.method,
                'success': report.result.success,
            }
            for pname, pval in report.result.best_parameters.items():
                row[f'param_{pname}'] = pval
            rows.append(row)

        for key, err in self.failures.items():
            parts = key.split('__')
            rows.append({
                'key': key,
                'model': parts[0] if len(parts) > 0 else '',
                'objective': parts[1] if len(parts) > 1 else '',
                'algorithm': parts[2] if len(parts) > 2 else '',
                'transformation': parts[3] if len(parts) > 3 else 'none',
                'best_objective': np.nan,
                'runtime_seconds': np.nan,
                'n_parameters': np.nan,
                'method': '',
                'success': False,
            })

        return pd.DataFrame(rows)

    def best_by_objective(self) -> Dict[str, Tuple[str, float]]:
        """For each unique objective, return the best (key, value) pair."""
        grouped: Dict[str, List[Tuple[str, float]]] = {}
        for key, report in self.results.items():
            parts = key.split('__')
            obj_name = parts[1] if len(parts) > 1 else 'unknown'
            value = report.result.best_objective
            grouped.setdefault(obj_name, []).append((key, value))

        best = {}
        for obj_name, items in grouped.items():
            best_key, best_val = max(items, key=lambda x: x[1])
            best[obj_name] = (best_key, best_val)
        return best

    def save(self, path: str) -> None:
        """Save the entire BatchResult to a pickle file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("BatchResult saved to %s", path)

    @classmethod
    def load(cls, path: str) -> 'BatchResult':
        """Load a BatchResult from a pickle file."""
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected BatchResult, got {type(obj).__name__}")
        return obj

    def __repr__(self) -> str:
        n_ok = len(self.results)
        n_fail = len(self.failures)
        return (
            f"BatchResult({n_ok} completed, {n_fail} failed, "
            f"{self.runtime_seconds:.1f}s)"
        )


# ---------------------------------------------------------------------------
# BatchExperimentRunner -- the orchestrator
# ---------------------------------------------------------------------------
class BatchExperimentRunner:
    """Orchestrates batch calibration experiments across a combinatorial grid.

    Features:
        - Pluggable parallelism via ParallelBackend (sequential / multiprocessing / ray)
        - Progress reporting via callbacks and optional tqdm
        - Failure isolation -- failed experiments logged and skipped
        - Result persistence -- each experiment saves immediately on completion
        - Resume capability -- scans output_dir for completed results and skips them
        - Configurable via YAML/JSON or programmatic Python API

    Args:
        inputs: Input DataFrame with DatetimeIndex, 'precipitation', 'pet' columns.
        observed: Observed flow array.
        grid: ExperimentGrid defining the combinatorial axes.
        output_dir: Directory for per-experiment result files.
        warmup_period: Warmup timesteps excluded from the objective.
        catchment_info: Optional metadata (name, gauge_id, area_km2).
        backend: Parallelisation backend name ('sequential', 'multiprocessing', 'ray').
        max_workers: Number of workers for multiprocessing/ray.
        on_complete: Callback(key, report) invoked after each success.
        on_error: Callback(key, exception) invoked after each failure.

    Example:
        >>> runner = BatchExperimentRunner(inputs, observed, grid, './results')
        >>> result = runner.run(resume=True)
        >>> print(result.to_dataframe())
    """

    def __init__(
        self,
        inputs: pd.DataFrame,
        observed: np.ndarray,
        grid: ExperimentGrid,
        output_dir: str = './batch_results',
        warmup_period: int = 365,
        catchment_info: Optional[Dict[str, Any]] = None,
        backend: str = 'sequential',
        max_workers: Optional[int] = None,
        on_complete: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
    ):
        self.inputs = inputs
        self.observed = np.asarray(observed).flatten()
        self.grid = grid
        self.output_dir = Path(output_dir)
        self.warmup_period = warmup_period
        self.catchment_info = catchment_info or {}
        self.backend_name = backend
        self.max_workers = max_workers
        self._on_complete = on_complete
        self._on_error = on_error

    # ------------------------------------------------------------------
    # Run single experiment
    # ------------------------------------------------------------------
    def run_single(self, spec: ExperimentSpec) -> CalibrationReport:
        """Run a single experiment and return its CalibrationReport.

        Args:
            spec: The experiment specification to execute.

        Returns:
            CalibrationReport for this experiment.
        """
        logger.info("Starting experiment: %s", spec.key)

        model = copy.deepcopy(spec.model)
        objective = copy.deepcopy(spec.objective)

        if spec.transformation is not None and hasattr(objective, 'transform'):
            objective.transform = spec.transformation

        cal_runner = CalibrationRunner(
            model=model,
            inputs=self.inputs.copy(),
            observed=self.observed.copy(),
            objective=objective,
            warmup_period=self.warmup_period,
        )

        alg_kwargs = dict(spec.algorithm_kwargs)
        method = alg_kwargs.pop('method', 'sceua_direct')

        dispatch = {
            'sceua_direct': cal_runner.run_sceua_direct,
            'dream': cal_runner.run_dream,
            'differential_evolution': lambda **kw: cal_runner.run_scipy(method='differential_evolution', **kw),
            'dual_annealing': lambda **kw: cal_runner.run_scipy(method='dual_annealing', **kw),
            'basin_hopping': lambda **kw: cal_runner.run_scipy(method='basin_hopping', **kw),
        }

        if hasattr(cal_runner, f'run_{method}'):
            run_fn = getattr(cal_runner, f'run_{method}')
        elif method in dispatch:
            run_fn = dispatch[method]
        else:
            raise ValueError(
                f"Unknown calibration method: '{method}'. "
                f"Available: {list(dispatch.keys())}"
            )

        result = run_fn(**alg_kwargs)

        report = cal_runner.create_report(
            result, catchment_info=self.catchment_info,
        )
        return report

    # ------------------------------------------------------------------
    # Run all experiments
    # ------------------------------------------------------------------
    def run(self, resume: bool = True) -> BatchResult:
        """Run all experiments in the grid.

        Args:
            resume: If True, skip experiments whose results already exist
                in ``output_dir``.

        Returns:
            BatchResult aggregating all completed experiments.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        specs = self.grid.combinations()
        total = len(specs)

        completed: Dict[str, CalibrationReport] = {}
        failures: Dict[str, str] = {}

        if resume:
            for spec in specs:
                pkl_path = self.output_dir / f"{spec.key}.pkl"
                if pkl_path.exists():
                    try:
                        report = CalibrationReport.load(str(pkl_path))
                        completed[spec.key] = report
                        logger.info("Resumed: %s", spec.key)
                    except Exception as e:
                        logger.warning("Failed to load %s: %s", spec.key, e)

        remaining = [s for s in specs if s.key not in completed]
        logger.info(
            "Batch experiment: %d total, %d completed, %d remaining",
            total, len(completed), len(remaining),
        )

        if not remaining:
            return BatchResult(
                results=completed,
                failures=failures,
                runtime_seconds=0.0,
            )

        t0 = time.time()

        def _on_complete(key: str, report: CalibrationReport):
            pkl_path = self.output_dir / f"{key}.pkl"
            try:
                report.save(str(pkl_path))
            except Exception as e:
                logger.warning("Failed to save %s: %s", key, e)
            if self._on_complete:
                self._on_complete(key, report)

        def _on_error(key: str, exc: Exception):
            failures[key] = f"{type(exc).__name__}: {exc}"
            logger.error("Experiment %s failed: %s", key, exc)
            if self._on_error:
                self._on_error(key, exc)

        def _run_task(spec: ExperimentSpec) -> CalibrationReport:
            return self.run_single(spec)

        backend = create_backend(self.backend_name, self.max_workers)
        try:
            task_ids = [s.key for s in remaining]
            results = backend.map(
                _run_task,
                remaining,
                task_ids=task_ids,
                on_complete=_on_complete,
                on_error=_on_error,
            )
            completed.update(results)
        finally:
            backend.shutdown()

        elapsed = time.time() - t0
        return BatchResult(
            results=completed,
            failures=failures,
            runtime_seconds=elapsed,
        )

    # ------------------------------------------------------------------
    # YAML / JSON factory
    # ------------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        config_path: str,
        inputs: pd.DataFrame,
        observed: np.ndarray,
    ) -> 'BatchExperimentRunner':
        """Create a BatchExperimentRunner from a YAML or JSON config file.

        Args:
            config_path: Path to YAML or JSON experiment configuration.
            inputs: Input DataFrame with DatetimeIndex.
            observed: Observed flow array.

        Returns:
            Configured BatchExperimentRunner instance.

        YAML format example::

            catchment:
              name: "Queanbeyan River"
              gauge_id: "410734"
              area_km2: 490

            warmup_days: 365

            models:
              GR4J: {}
              GR5J: {}

            objectives:
              nse: {type: NSE}
              kge: {type: KGE, variant: "2012"}

            algorithms:
              sceua:
                method: sceua_direct
                max_evals: 20000
                seed: 42

            output_dir: "./results/batch_410734"
            backend: sequential
            max_workers: 4
        """
        config_path = Path(config_path)
        suffix = config_path.suffix.lower()

        if suffix in ('.yaml', '.yml'):
            if not YAML_AVAILABLE:
                raise ImportError(
                    "PyYAML is required to load YAML configs. "
                    "Install with: pip install pyyaml"
                )
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
        elif suffix == '.json':
            with open(config_path) as f:
                cfg = json.load(f)
        else:
            raise ValueError(
                f"Unsupported config format: '{suffix}'. Use .yaml or .json"
            )

        models = {}
        for name, model_cfg in cfg.get('models', {}).items():
            model_cls = get_model_class(name)
            models[name] = model_cls(**(model_cfg or {}))

        objectives = {}
        for name, obj_spec in cfg.get('objectives', {}).items():
            if obj_spec is None:
                obj_spec = {'type': name}
            objectives[name] = _build_objective(obj_spec)

        algorithms = {}
        for name, alg_cfg in cfg.get('algorithms', {}).items():
            algorithms[name] = dict(alg_cfg or {})

        transformations = None
        if 'transformations' in cfg:
            try:
                from pyrrm.objectives import FlowTransformation
                transformations = {}
                for name, t_spec in cfg['transformations'].items():
                    if isinstance(t_spec, str):
                        transformations[name] = FlowTransformation(t_spec)
                    else:
                        transformations[name] = FlowTransformation(
                            t_spec.get('type', name)
                        )
            except ImportError:
                warnings.warn(
                    "FlowTransformation not available; "
                    "skipping transformations axis"
                )

        grid = ExperimentGrid(
            models=models,
            objectives=objectives,
            algorithms=algorithms,
            transformations=transformations,
        )

        catchment_info = cfg.get('catchment', {})

        return cls(
            inputs=inputs,
            observed=observed,
            grid=grid,
            output_dir=cfg.get('output_dir', './batch_results'),
            warmup_period=cfg.get('warmup_days', 365),
            catchment_info=catchment_info,
            backend=cfg.get('backend', 'sequential'),
            max_workers=cfg.get('max_workers', None),
        )


__all__ = [
    'ExperimentSpec',
    'ExperimentGrid',
    'BatchResult',
    'BatchExperimentRunner',
    'get_model_class',
]
