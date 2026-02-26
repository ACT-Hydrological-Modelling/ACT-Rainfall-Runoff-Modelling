"""
Batch experiment runner for single-catchment experiments.

Automates running multiple calibrations across combinations of
rainfall-runoff models, objective functions, optimization algorithms,
and flow transformations.  Supports two configuration modes:

1. **Combinatorial grid** -- ``ExperimentGrid`` crosses models x objectives
   x algorithms x transformations (Cartesian product).
2. **Explicit experiment list** -- ``ExperimentList`` runs a user-curated
   sequence of ``ExperimentSpec`` objects with no combinatorial expansion.

Each call to ``run()`` creates a unique, timestamped run folder under the
user-supplied ``output_dir`` so successive runs never overwrite each other.

Example (combinatorial grid):
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

Example (explicit experiment list):
    >>> from pyrrm.calibration.batch import ExperimentList, ExperimentSpec
    >>> specs = [
    ...     ExperimentSpec(key='gr4j_nse', model_name='GR4J', model=GR4J(),
    ...                    objective_name='nse', objective=NSE(),
    ...                    algorithm_name='sceua',
    ...                    algorithm_kwargs={'method': 'sceua_direct', 'max_evals': 5000}),
    ... ]
    >>> exp_list = ExperimentList(specs)
    >>> runner = BatchExperimentRunner(inputs, observed, exp_list, output_dir='./results')
    >>> batch_result = runner.run(run_name='my_experiment')

Example (YAML-driven):
    >>> runner = BatchExperimentRunner.from_config('experiment.yaml', inputs, observed)
    >>> batch_result = runner.run()
"""

from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import (
    Dict, List, Tuple, Optional, Any, Callable, Type, Union, TYPE_CHECKING,
)
import copy
import json
import logging
import pickle
import time
import traceback
import uuid
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
# Experiment Naming Convention
# ---------------------------------------------------------------------------
#
# Canonical key format:
#   {catchment}_{model}_{objective}_{algorithm}[_{transformation}[-{tag}...]]
#
# Rules:
#   - All fields lowercased, non-alphanumeric chars stripped.
#   - Exactly 4 underscore-separated fields (5 with transformation).
#   - catchment is always present (use DEFAULT_CATCHMENT when unknown).
#   - APEX objectives append dash-separated parameter tags after the
#     transformation so the full configuration is readable from the key.
# ---------------------------------------------------------------------------

DEFAULT_CATCHMENT = 'catchment'

_SANITISE_KEEP = set('abcdefghijklmnopqrstuvwxyz0123456789')


def _sanitise(value: str) -> str:
    """Lowercase and strip non-alphanumeric characters."""
    return ''.join(c for c in value.strip().lower() if c in _SANITISE_KEEP)


def make_experiment_key(
    model: str,
    objective: str,
    algorithm: str,
    catchment: str = DEFAULT_CATCHMENT,
    transformation: Optional[str] = None,
    extra_tags: Optional[List[str]] = None,
) -> str:
    """Build a canonical experiment key.

    Format::

        {catchment}_{model}_{objective}[_{transformation}[-{tag}...]]_{algorithm}

    The algorithm is always the **last** underscore-separated field.
    When no transformation is present the key has 4 fields; with a
    transformation it has 5.  APEX-specific parameter tags are
    dash-separated *within* the transformation field.

    Args:
        model: Model name (e.g. ``'GR4J'``, ``'Sacramento'``).
        objective: Objective function name (e.g. ``'nse'``, ``'kge'``,
            ``'apex'``).
        algorithm: Algorithm name (e.g. ``'sceua'``, ``'dream'``).
        catchment: Catchment identifier.  Defaults to
            ``DEFAULT_CATCHMENT`` (``'catchment'``) when no real ID is
            available.  Use descriptive labels such as ``'synthetic'``,
            ``'demo'``, or the gauge ID.
        transformation: Optional flow transformation (e.g. ``'sqrt'``,
            ``'log'``, ``'inverse'``).
        extra_tags: Optional dash-separated tags appended after the
            transformation.  Used for APEX parameter encoding.

    Returns:
        Canonical key string.

    Example:
        >>> make_experiment_key('GR4J', 'nse', 'sceua', catchment='410734')
        '410734_gr4j_nse_sceua'
        >>> make_experiment_key('Sacramento', 'apex', 'sceua',
        ...     catchment='410734', transformation='sqrt',
        ...     extra_tags=['k05', 'uniform'])
        '410734_sacramento_apex_sqrt-k05-uniform_sceua'
    """
    parts = [
        _sanitise(catchment),
        _sanitise(model),
        _sanitise(objective),
    ]
    if transformation or extra_tags:
        suffix = _sanitise(transformation or 'none')
        if extra_tags:
            suffix += '-' + '-'.join(_sanitise(t) for t in extra_tags)
        parts.append(suffix)
    parts.append(_sanitise(algorithm))
    return '_'.join(parts)


def make_apex_tags(
    dynamics_strength: float = 0.5,
    regime_emphasis: str = 'uniform',
) -> List[str]:
    """Build APEX parameter tags.  Always includes all params.

    Args:
        dynamics_strength: Kappa value (dynamics multiplier strength).
        regime_emphasis: Regime emphasis mode (``'uniform'``,
            ``'low_flow'``, ``'balanced'``).

    Returns:
        List of short tag strings, e.g. ``['k05', 'uniform']``.

    Example:
        >>> make_apex_tags(dynamics_strength=0.3, regime_emphasis='low_flow')
        ['k03', 'lowflow']
    """
    return [
        f'k{int(dynamics_strength * 10):02d}',
        regime_emphasis.replace('_', ''),
    ]


def parse_experiment_key(key: str) -> Dict[str, Any]:
    """Parse a canonical experiment key into its components.

    The algorithm is always the **last** underscore-separated field.
    When 4 fields are present: catchment, model, objective, algorithm.
    When 5 fields: catchment, model, objective, transformation, algorithm.

    Args:
        key: Canonical key string.

    Returns:
        Dict with keys ``catchment``, ``model``, ``objective``,
        ``algorithm``, and optionally ``transformation`` and
        ``apex_tags``.

    Example:
        >>> parse_experiment_key('410734_gr4j_nse_sceua')
        {'catchment': '410734', 'model': 'gr4j', 'objective': 'nse',
         'algorithm': 'sceua'}
        >>> parse_experiment_key('410734_gr4j_nse_sqrt_sceua')
        {'catchment': '410734', 'model': 'gr4j', 'objective': 'nse',
         'algorithm': 'sceua', 'transformation': 'sqrt'}
        >>> parse_experiment_key('410734_sacramento_apex_sqrt-k05-uniform_sceua')
        {'catchment': '410734', 'model': 'sacramento', 'objective': 'apex',
         'algorithm': 'sceua', 'transformation': 'sqrt',
         'apex_tags': ['k05', 'uniform']}
    """
    parts = key.split('_')
    result: Dict[str, Any] = {}
    if len(parts) >= 4:
        result['catchment'] = parts[0]
        result['model'] = parts[1]
        result['objective'] = parts[2]
        result['algorithm'] = parts[-1]
    if len(parts) >= 5:
        middle = parts[3]
        if '-' in middle:
            segments = middle.split('-')
            result['transformation'] = segments[0]
            result['apex_tags'] = segments[1:]
        else:
            result['transformation'] = middle
    return result

try:
    from tqdm.auto import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


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
        catchment: Catchment identifier used in generated keys.
            Defaults to ``DEFAULT_CATCHMENT``.

    Example:
        >>> grid = ExperimentGrid(
        ...     models={'GR4J': GR4J()},
        ...     objectives={'nse': NSE(), 'kge': KGE()},
        ...     algorithms={'sceua': {'method': 'sceua_direct', 'max_evals': 20000}},
        ...     catchment='410734',
        ... )
        >>> specs = grid.combinations()
        >>> len(specs)
        2
    """
    models: Dict[str, Any]
    objectives: Dict[str, Any]
    algorithms: Dict[str, Dict[str, Any]]
    transformations: Optional[Dict[str, Any]] = None
    catchment: str = DEFAULT_CATCHMENT

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
            key = make_experiment_key(
                model=m_name,
                objective=o_name,
                algorithm=a_name,
                catchment=self.catchment,
                transformation=t_name,
            )
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
# ExperimentList -- user-defined list of specific experiments
# ---------------------------------------------------------------------------
@dataclass
class ExperimentList:
    """User-defined list of specific experiments (no combinatorial expansion).

    Use this instead of ``ExperimentGrid`` when you want to run a curated
    set of experiments rather than the full Cartesian product of axes.

    Args:
        specs: List of ``ExperimentSpec`` objects to run.

    Example:
        >>> from pyrrm.calibration.batch import ExperimentList, ExperimentSpec
        >>> specs = [
        ...     ExperimentSpec(
        ...         key='gr4j_nse_sceua', model_name='GR4J', model=GR4J(),
        ...         objective_name='nse', objective=NSE(),
        ...         algorithm_name='sceua',
        ...         algorithm_kwargs={'method': 'sceua_direct', 'max_evals': 5000},
        ...     ),
        ...     ExperimentSpec(
        ...         key='sac_kge_dream', model_name='Sacramento', model=Sacramento(),
        ...         objective_name='kge', objective=KGE(),
        ...         algorithm_name='dream',
        ...         algorithm_kwargs={'method': 'dream', 'nsamples': 10000},
        ...     ),
        ... ]
        >>> exp_list = ExperimentList(specs)
        >>> len(exp_list)
        2
    """
    specs: List[ExperimentSpec]

    def combinations(self) -> List[ExperimentSpec]:
        """Return the experiment specs as-is (no combinatorial expansion)."""
        return list(self.specs)

    def __len__(self) -> int:
        return len(self.specs)

    @classmethod
    def from_dicts(
        cls,
        experiment_dicts: List[Dict[str, Any]],
        catchment: str = DEFAULT_CATCHMENT,
    ) -> 'ExperimentList':
        """Build an ExperimentList from a list of plain dictionaries.

        Each dict should contain:
            - ``key`` (str, optional): Unique experiment identifier.
              Auto-generated via ``make_experiment_key`` when omitted.
            - ``model`` (str): Model name (resolved via model registry).
            - ``model_params`` (dict, optional): Model constructor kwargs.
            - ``objective`` (dict): Objective spec, e.g. ``{'type': 'NSE'}``.
            - ``algorithm`` (dict): Algorithm kwargs including ``method``.
            - ``transformation`` (str or dict, optional): Flow transformation.

        Args:
            experiment_dicts: List of experiment configuration dicts.
            catchment: Default catchment identifier used when
                auto-generating keys.

        Returns:
            ExperimentList with resolved ExperimentSpec objects.

        Example:
            >>> dicts = [
            ...     {'model': 'GR4J',
            ...      'objective': {'type': 'NSE'},
            ...      'algorithm': {'method': 'sceua_direct', 'max_evals': 5000}},
            ... ]
            >>> exp_list = ExperimentList.from_dicts(dicts, catchment='410734')
        """
        specs = []
        for d in experiment_dicts:
            model_name = d['model']
            model_cls = get_model_class(model_name)
            model_params = d.get('model_params', {}) or {}
            model = model_cls(**model_params)

            obj_spec = d.get('objective', {'type': 'NSE'})
            if isinstance(obj_spec, str):
                obj_spec = {'type': obj_spec}
            objective = _build_objective(obj_spec)
            objective_name = obj_spec.get('type', 'NSE').lower()

            alg_kwargs = dict(d.get('algorithm', {'method': 'sceua_direct'}))
            algorithm_name = alg_kwargs.get('method', 'sceua_direct')

            transformation = None
            transformation_name = None
            t_spec = d.get('transformation', None)
            if t_spec is not None:
                try:
                    from pyrrm.objectives import FlowTransformation
                    if isinstance(t_spec, str):
                        transformation = FlowTransformation(t_spec)
                        transformation_name = t_spec
                    elif isinstance(t_spec, dict):
                        t_type = t_spec.get('type', 'none')
                        transformation = FlowTransformation(t_type)
                        transformation_name = t_type
                except ImportError:
                    warnings.warn(
                        "FlowTransformation not available; "
                        f"skipping transformation for experiment '{d.get('key', model_name)}'"
                    )

            key = d.get('key') or make_experiment_key(
                model=model_name,
                objective=objective_name,
                algorithm=algorithm_name,
                catchment=catchment,
                transformation=transformation_name,
            )

            specs.append(ExperimentSpec(
                key=key,
                model_name=model_name,
                model=model,
                objective_name=objective_name,
                objective=objective,
                algorithm_name=algorithm_name,
                algorithm_kwargs=alg_kwargs,
                transformation_name=transformation_name,
                transformation=transformation,
            ))
        return cls(specs)


# Type alias for experiment sources accepted by the runner
ExperimentSource = Union[ExperimentGrid, ExperimentList]


# ---------------------------------------------------------------------------
# BatchLogger -- structured logging for batch runs
# ---------------------------------------------------------------------------
class _BatchOnlyFilter(logging.Filter):
    """Pass only records emitted by the batch orchestrator logger.

    The batch.log file should contain concise start/complete/summary lines
    only.  Iteration-level progress from SCE-UA, PyDREAM, etc. is captured
    by the per-experiment log handlers instead.
    """

    _BATCH_LOGGER_NAME = __name__  # 'pyrrm.calibration.batch'

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name == self._BATCH_LOGGER_NAME


class _BatchLogger:
    """Manages file-based and console logging for a batch run.

    Creates:
      - ``run_dir/batch.log``: Concise batch-level messages only (start,
        complete, summary) from the ``pyrrm.calibration.batch`` logger.
      - ``run_dir/logs/<key>.log``: Full per-experiment log files including
        iteration-level progress from child loggers.

    All handlers are attached to the ``pyrrm.calibration`` **parent** logger
    so that log messages emitted by child loggers (``runner``,
    ``sceua_adapter``, ``pydream_adapter``, ``_sceua.sceua``) are captured
    by the per-experiment handlers.  The batch.log handler has a filter that
    restricts it to batch orchestrator messages only.
    """

    _PARENT_LOGGER_NAME = 'pyrrm.calibration'

    def __init__(
        self,
        run_dir: Path,
        log_level: str = 'INFO',
        progress_bar: bool = True,
    ):
        self.run_dir = run_dir
        self.logs_dir = run_dir / 'logs'
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self._parent_logger = logging.getLogger(self._PARENT_LOGGER_NAME)
        self._handlers: List[logging.Handler] = []
        self._experiment_handlers: Dict[str, logging.FileHandler] = {}

        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )

        # batch.log — only batch-orchestrator messages (start, complete, summary)
        batch_log_path = run_dir / 'batch.log'
        fh = logging.FileHandler(str(batch_log_path), mode='a')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        fh.addFilter(_BatchOnlyFilter())
        self._parent_logger.addHandler(fh)
        self._handlers.append(fh)

        # Console — also only batch-level to keep output readable
        if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
                   for h in self._parent_logger.handlers):
            ch = logging.StreamHandler()
            ch.setLevel(numeric_level)
            ch.setFormatter(formatter)
            ch.addFilter(_BatchOnlyFilter())
            self._parent_logger.addHandler(ch)
            self._handlers.append(ch)

        self._parent_logger.setLevel(logging.DEBUG)

        self._use_progress_bar = progress_bar and TQDM_AVAILABLE
        self._pbar: Any = None

    def start_progress(self, total: int, completed: int) -> None:
        """Initialise the progress bar."""
        if self._use_progress_bar:
            self._pbar = tqdm(
                total=total, initial=completed,
                desc='Batch experiments', unit='exp',
                dynamic_ncols=True,
            )

    def update_progress(self, key: str, success: bool) -> None:
        """Advance the progress bar by one experiment."""
        if self._pbar is not None:
            status = 'ok' if success else 'FAIL'
            self._pbar.set_postfix_str(f'{key} [{status}]', refresh=True)
            self._pbar.update(1)

    def close_progress(self) -> None:
        """Close the progress bar."""
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None

    def start_experiment_log(self, key: str) -> None:
        """Open a per-experiment log file on the parent calibration logger."""
        log_path = self.logs_dir / f'{key}.log'
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        fh = logging.FileHandler(str(log_path), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        self._parent_logger.addHandler(fh)
        self._experiment_handlers[key] = fh

    def end_experiment_log(self, key: str) -> None:
        """Close the per-experiment log file handler."""
        fh = self._experiment_handlers.pop(key, None)
        if fh is not None:
            self._parent_logger.removeHandler(fh)
            fh.close()

    def close(self) -> None:
        """Remove all handlers added by this logger."""
        self.close_progress()
        for fh in self._experiment_handlers.values():
            self._parent_logger.removeHandler(fh)
            fh.close()
        self._experiment_handlers.clear()
        for h in self._handlers:
            self._parent_logger.removeHandler(h)
            h.close()
        self._handlers.clear()


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
        run_dir: Path to the timestamped run folder (set after ``run()``).
    """
    results: Dict[str, CalibrationReport] = field(default_factory=dict)
    failures: Dict[str, str] = field(default_factory=dict)
    runtime_seconds: float = 0.0
    run_dir: Optional[str] = None

    def to_dataframe(self) -> pd.DataFrame:
        """Summary table: one row per experiment.

        Columns include model, objective, algorithm, transformation,
        best_objective, runtime_seconds, n_parameters, and success.
        """
        rows = []
        for key, report in self.results.items():
            parsed = parse_experiment_key(key)
            row = {
                'key': key,
                'catchment': parsed.get('catchment', ''),
                'model': parsed.get('model', ''),
                'objective': parsed.get('objective', ''),
                'algorithm': parsed.get('algorithm', ''),
                'transformation': parsed.get('transformation', 'none'),
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
            parsed = parse_experiment_key(key)
            rows.append({
                'key': key,
                'catchment': parsed.get('catchment', ''),
                'model': parsed.get('model', ''),
                'objective': parsed.get('objective', ''),
                'algorithm': parsed.get('algorithm', ''),
                'transformation': parsed.get('transformation', 'none'),
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
            parsed = parse_experiment_key(key)
            obj_name = parsed.get('objective', 'unknown')
            value = report.result.best_objective
            grouped.setdefault(obj_name, []).append((key, value))

        best = {}
        for obj_name, items in grouped.items():
            best_key, best_val = max(items, key=lambda x: x[1])
            best[obj_name] = (best_key, best_val)
        return best

    def save(self, path: Optional[str] = None) -> None:
        """Save the entire BatchResult to a pickle file.

        Args:
            path: File path.  Defaults to ``run_dir/batch_result.pkl``
                if ``run_dir`` is set.
        """
        if path is None:
            if self.run_dir is None:
                raise ValueError(
                    "No path given and run_dir is not set. "
                    "Provide an explicit path or run the batch first."
                )
            path = str(Path(self.run_dir) / 'batch_result.pkl')
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

    # ------------------------------------------------------------------
    # Export (Excel / CSV)
    # ------------------------------------------------------------------

    def export(
        self,
        output_dir: Union[str, Path],
        format: str = 'excel',
        exceedance_pct_resolution: float = 1.0,
        skip_failures: bool = True,
    ) -> Dict[str, List[str]]:
        """Export every CalibrationReport to Excel and/or CSV.

        Delegates to :func:`pyrrm.calibration.export.export_batch`.

        Args:
            output_dir: Directory for exported files.  Created if needed.
            format: ``'excel'``, ``'csv'``, or ``'both'``.
            exceedance_pct_resolution: FDC grid step in percent.
            skip_failures: If *True*, skip experiments that fail to export.

        Returns:
            Dict mapping experiment key -> list of created file paths.

        Example:
            >>> batch = BatchResult.load('results/batch_result.pkl')
            >>> batch.export('exports/', format='excel')
        """
        from pyrrm.calibration.export import export_batch
        return export_batch(
            self,
            output_dir,
            format=format,
            exceedance_pct_resolution=exceedance_pct_resolution,
            skip_failures=skip_failures,
        )

    def __repr__(self) -> str:
        n_ok = len(self.results)
        n_fail = len(self.failures)
        parts = [
            f"BatchResult({n_ok} completed, {n_fail} failed, "
            f"{self.runtime_seconds:.1f}s)"
        ]
        if self.run_dir:
            parts.append(f"  run_dir: {self.run_dir}")
        return '\n'.join(parts)


# ---------------------------------------------------------------------------
# Run-folder helpers
# ---------------------------------------------------------------------------

def _make_run_dir_name(run_name: Optional[str] = None) -> str:
    """Generate a unique timestamped folder name."""
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    short_id = uuid.uuid4().hex[:4]
    name = f'{ts}_{short_id}'
    if run_name:
        safe_name = "".join(
            c if (c.isalnum() or c in '_-') else '_'
            for c in run_name
        )
        name = f'{name}_{safe_name}'
    return name


def _find_latest_run_dir(output_dir: Path) -> Optional[Path]:
    """Find the most recent run folder under *output_dir* by name sort."""
    candidates = sorted(
        [d for d in output_dir.iterdir()
         if d.is_dir() and (d / 'results').is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _write_summary(
    run_dir: Path,
    batch_result: 'BatchResult',
    specs: List[ExperimentSpec],
    elapsed: float,
) -> None:
    """Write batch_summary.json and batch_summary.csv to the run folder."""
    best = batch_result.best_by_objective()
    summary = {
        'run_dir': str(run_dir),
        'timestamp': datetime.now().isoformat(),
        'total_experiments': len(specs),
        'completed': len(batch_result.results),
        'failed': len(batch_result.failures),
        'runtime_seconds': round(elapsed, 2),
        'best_by_objective': {
            obj: {'key': k, 'value': round(v, 6)}
            for obj, (k, v) in best.items()
        },
        'failures': dict(batch_result.failures),
    }
    json_path = run_dir / 'batch_summary.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    df = batch_result.to_dataframe()
    csv_path = run_dir / 'batch_summary.csv'
    df.to_csv(csv_path, index=False)

    logger.info("Summary written to %s", run_dir)


def _write_config_snapshot(
    run_dir: Path,
    grid: ExperimentSource,
    catchment_info: Dict[str, Any],
    warmup_period: int,
    backend: str,
) -> None:
    """Write a config.yaml (or .json) snapshot of the experiment setup."""
    config: Dict[str, Any] = {
        'catchment': catchment_info,
        'warmup_days': warmup_period,
        'backend': backend,
    }

    if isinstance(grid, ExperimentGrid):
        config['mode'] = 'grid'
        config['models'] = list(grid.models.keys())
        config['objectives'] = list(grid.objectives.keys())
        config['algorithms'] = {
            name: dict(kw) for name, kw in grid.algorithms.items()
        }
        if grid.transformations:
            config['transformations'] = list(grid.transformations.keys())
    elif isinstance(grid, ExperimentList):
        config['mode'] = 'list'
        config['experiments'] = [
            {
                'key': s.key,
                'model': s.model_name,
                'objective': s.objective_name,
                'algorithm': s.algorithm_name,
                'algorithm_kwargs': s.algorithm_kwargs,
            }
            for s in grid.specs
        ]

    if YAML_AVAILABLE:
        path = run_dir / 'config.yaml'
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    else:
        path = run_dir / 'config.json'
        with open(path, 'w') as f:
            json.dump(config, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# BatchExperimentRunner -- the orchestrator
# ---------------------------------------------------------------------------
class BatchExperimentRunner:
    """Orchestrates batch calibration experiments.

    Accepts either an ``ExperimentGrid`` (combinatorial) or an
    ``ExperimentList`` (user-curated).  Each call to ``run()`` creates a
    unique timestamped run folder under ``output_dir``.

    Features:
        - Pluggable parallelism via ParallelBackend (sequential / multiprocessing / ray)
        - Progress reporting via tqdm and structured file logging
        - Failure isolation -- failed experiments logged and skipped
        - Result persistence -- each experiment saves immediately on completion
        - Resume capability -- scans for completed results and skips them
        - Organised output -- timestamped run folders with results/, logs/,
          batch.log, batch_summary.json/csv, and config snapshot
        - Configurable via YAML/JSON or programmatic Python API

    Args:
        inputs: Input DataFrame with DatetimeIndex, 'precipitation', 'pet' columns.
        observed: Observed flow array.
        grid: ``ExperimentGrid`` or ``ExperimentList`` defining the experiments.
        output_dir: Parent directory for timestamped run folders.
        warmup_period: Warmup timesteps excluded from the objective.
        catchment_info: Optional metadata (name, gauge_id, area_km2).
        backend: Parallelisation backend name ('sequential', 'multiprocessing', 'ray').
        max_workers: Number of workers for multiprocessing/ray.
        on_complete: Callback(key, report) invoked after each success.
        on_error: Callback(key, exception) invoked after each failure.
        log_level: Logging level for console output ('DEBUG', 'INFO', 'WARNING').
        progress_bar: Enable tqdm progress bar (requires tqdm).
        catchment: Catchment identifier forwarded to ``ExperimentGrid``
            if the grid does not already set one.

    Example:
        >>> runner = BatchExperimentRunner(inputs, observed, grid, './results')
        >>> result = runner.run(run_name='calibration_v1')
        >>> print(result.run_dir)
        >>> print(result.to_dataframe())
    """

    def __init__(
        self,
        inputs: pd.DataFrame,
        observed: np.ndarray,
        grid: ExperimentSource,
        output_dir: str = './batch_results',
        warmup_period: int = 365,
        catchment_info: Optional[Dict[str, Any]] = None,
        backend: str = 'sequential',
        max_workers: Optional[int] = None,
        on_complete: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
        log_level: str = 'INFO',
        progress_bar: bool = True,
        catchment: Optional[str] = None,
    ):
        self.inputs = inputs
        self.observed = np.asarray(observed).flatten()
        self.grid = grid
        if catchment and isinstance(grid, ExperimentGrid) and grid.catchment == DEFAULT_CATCHMENT:
            grid.catchment = catchment
        self.output_dir = Path(output_dir)
        self.warmup_period = warmup_period
        self.catchment_info = catchment_info or {}
        self.backend_name = backend
        self.max_workers = max_workers
        self._on_complete = on_complete
        self._on_error = on_error
        self.log_level = log_level
        self.progress_bar = progress_bar
        self._run_dir: Optional[Path] = None

    @property
    def run_dir(self) -> Optional[Path]:
        """Path to the run folder created by the most recent ``run()`` call."""
        return self._run_dir

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

        # Enable verbose logging and disable tqdm progress bars inside
        # individual experiments so that iteration-level progress is
        # captured in the per-experiment log file.
        alg_kwargs.setdefault('verbose', True)
        alg_kwargs.setdefault('progress_bar', False)

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

        if spec.transformation_name:
            result.objective_name = (
                f"{result.objective_name}({spec.transformation_name})"
            )

        report = cal_runner.create_report(
            result,
            catchment_info=self.catchment_info,
            experiment_name=spec.key,
        )
        return report

    # ------------------------------------------------------------------
    # Resume helpers
    # ------------------------------------------------------------------

    def _load_completed(
        self,
        results_dir: Path,
        specs: List[ExperimentSpec],
    ) -> Dict[str, CalibrationReport]:
        """Scan a results directory for previously completed experiments."""
        completed: Dict[str, CalibrationReport] = {}
        for spec in specs:
            pkl_path = results_dir / f"{spec.key}.pkl"
            if pkl_path.exists():
                try:
                    report = CalibrationReport.load(str(pkl_path))
                    completed[spec.key] = report
                    logger.info("Resumed: %s", spec.key)
                except Exception as e:
                    logger.warning("Failed to load %s: %s", spec.key, e)
        return completed

    # ------------------------------------------------------------------
    # Run all experiments
    # ------------------------------------------------------------------
    def run(
        self,
        resume: bool = True,
        resume_from: Optional[str] = None,
        run_name: Optional[str] = None,
    ) -> BatchResult:
        """Run all experiments.

        Args:
            resume: If True, try to resume from a previous run folder.
            resume_from: Explicit path to a run folder to resume from.
                When given, new results are written into that same folder.
            run_name: Human-readable label appended to the timestamped
                folder name (e.g. ``'calibration_v1'``).

        Returns:
            BatchResult aggregating all completed experiments.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        specs = self.grid.combinations()
        total = len(specs)

        completed: Dict[str, CalibrationReport] = {}
        failures: Dict[str, str] = {}

        # --- Determine run_dir and load completed results -----------------
        if resume_from is not None:
            run_dir = Path(resume_from)
            if not run_dir.exists():
                raise FileNotFoundError(
                    f"Resume folder not found: {resume_from}"
                )
            results_dir = run_dir / 'results'
            results_dir.mkdir(parents=True, exist_ok=True)
            completed.update(self._load_completed(results_dir, specs))
        elif resume:
            # Try the new folder layout first
            latest = _find_latest_run_dir(self.output_dir)
            if latest is not None:
                run_dir = latest
                results_dir = run_dir / 'results'
                completed.update(self._load_completed(results_dir, specs))
            else:
                # Backward compat: legacy flat .pkl files in output_dir
                legacy_completed = self._load_completed(self.output_dir, specs)
                if legacy_completed:
                    logger.info(
                        "Found %d legacy results in %s",
                        len(legacy_completed), self.output_dir,
                    )
                    completed.update(legacy_completed)
                run_dir = self.output_dir / _make_run_dir_name(run_name)
            if latest is not None:
                run_dir = latest
            else:
                run_dir = self.output_dir / _make_run_dir_name(run_name)
        else:
            run_dir = self.output_dir / _make_run_dir_name(run_name)

        self._run_dir = run_dir
        results_dir = run_dir / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)

        # --- Set up logging -----------------------------------------------
        batch_logger = _BatchLogger(
            run_dir, log_level=self.log_level,
            progress_bar=self.progress_bar,
        )

        _write_config_snapshot(
            run_dir, self.grid, self.catchment_info,
            self.warmup_period, self.backend_name,
        )

        remaining = [s for s in specs if s.key not in completed]
        logger.info(
            "Batch run: %d total, %d completed, %d remaining | run_dir=%s",
            total, len(completed), len(remaining), run_dir,
        )

        if not remaining:
            batch_result = BatchResult(
                results=completed,
                failures=failures,
                runtime_seconds=0.0,
                run_dir=str(run_dir),
            )
            _write_summary(run_dir, batch_result, specs, 0.0)
            batch_logger.close()
            return batch_result

        batch_logger.start_progress(total, len(completed))
        t0 = time.time()

        def _on_complete(key: str, report: CalibrationReport):
            pkl_path = results_dir / f"{key}.pkl"
            try:
                report.save(str(pkl_path))
            except Exception as e:
                logger.warning("Failed to save %s: %s", key, e)
            batch_logger.update_progress(key, success=True)
            batch_logger.end_experiment_log(key)
            if self._on_complete:
                self._on_complete(key, report)

        def _on_error(key: str, exc: Exception):
            failures[key] = f"{type(exc).__name__}: {exc}"
            logger.error("Experiment %s failed: %s", key, exc)
            batch_logger.update_progress(key, success=False)
            batch_logger.end_experiment_log(key)
            if self._on_error:
                self._on_error(key, exc)

        def _run_task(spec: ExperimentSpec) -> CalibrationReport:
            batch_logger.start_experiment_log(spec.key)
            t_start = time.time()
            logger.info(
                "Experiment %s started at %s",
                spec.key, datetime.now().strftime('%H:%M:%S'),
            )
            report = self.run_single(spec)
            elapsed_exp = time.time() - t_start
            logger.info(
                "Experiment %s completed in %.1fs | %s = %.6f",
                spec.key, elapsed_exp,
                report.result.objective_name, report.result.best_objective,
            )
            return report

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
        batch_logger.close_progress()

        # --- Completion summary -------------------------------------------
        n_ok = len(completed)
        n_fail = len(failures)
        logger.info("=" * 60)
        logger.info("BATCH RUN COMPLETE")
        logger.info("=" * 60)
        logger.info("  Total experiments : %d", total)
        logger.info("  Completed         : %d", n_ok)
        logger.info("  Failed            : %d", n_fail)
        logger.info("  Total runtime     : %.1fs", elapsed)
        best = {}
        grouped: Dict[str, List[Tuple[str, float]]] = {}
        for key, report in completed.items():
            parsed = parse_experiment_key(key)
            obj_name = parsed.get('objective', 'unknown')
            grouped.setdefault(obj_name, []).append(
                (key, report.result.best_objective)
            )
        for obj_name, items in grouped.items():
            bk, bv = max(items, key=lambda x: x[1])
            best[obj_name] = (bk, bv)
            logger.info("  Best %-12s: %s (%.6f)", obj_name, bk, bv)
        if failures:
            logger.info("  Failures:")
            for fkey, ferr in failures.items():
                logger.info("    %s: %s", fkey, ferr)
        logger.info("  Run folder: %s", run_dir)
        logger.info("=" * 60)

        batch_result = BatchResult(
            results=completed,
            failures=failures,
            runtime_seconds=elapsed,
            run_dir=str(run_dir),
        )

        _write_summary(run_dir, batch_result, specs, elapsed)
        batch_result.save()
        batch_logger.close()

        return batch_result

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

        Supports two modes (mutually exclusive):

        **Mode A -- Combinatorial grid** (existing)::

            models:
              GR4J: {}
              GR5J: {}
            objectives:
              nse: {type: NSE}
              kge: {type: KGE}
            algorithms:
              sceua:
                method: sceua_direct
                max_evals: 20000

        **Mode B -- Explicit experiment list** (new)::

            experiments:
              - key: "gr4j_nse_sceua"
                model: GR4J
                objective: {type: NSE}
                algorithm: {method: sceua_direct, max_evals: 5000}
              - key: "sacramento_kge_dream"
                model: Sacramento
                objective: {type: KGE}
                algorithm: {method: dream, nsamples: 10000}

        Common fields (both modes)::

            catchment:
              name: "Queanbeyan River"
              gauge_id: "410734"
            warmup_days: 365
            output_dir: "./results"
            backend: sequential
            max_workers: 4
            log_level: INFO
            progress_bar: true

        Args:
            config_path: Path to YAML or JSON experiment configuration.
            inputs: Input DataFrame with DatetimeIndex.
            observed: Observed flow array.

        Returns:
            Configured BatchExperimentRunner instance.

        Raises:
            ValueError: If both ``experiments`` and ``models`` are specified.
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

        has_experiments = 'experiments' in cfg
        has_grid = any(k in cfg for k in ('models', 'objectives', 'algorithms'))

        if has_experiments and has_grid:
            raise ValueError(
                "Config file must use either 'experiments' (explicit list) "
                "OR 'models'/'objectives'/'algorithms' (combinatorial grid), "
                "not both."
            )

        if has_experiments:
            grid: ExperimentSource = ExperimentList.from_dicts(cfg['experiments'])
        else:
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
        catchment_id = catchment_info.get(
            'gauge_id', catchment_info.get('name', None)
        )

        return cls(
            inputs=inputs,
            observed=observed,
            grid=grid,
            output_dir=cfg.get('output_dir', './batch_results'),
            warmup_period=cfg.get('warmup_days', 365),
            catchment_info=catchment_info,
            backend=cfg.get('backend', 'sequential'),
            max_workers=cfg.get('max_workers', None),
            log_level=cfg.get('log_level', 'INFO'),
            progress_bar=cfg.get('progress_bar', True),
            catchment=catchment_id,
        )


__all__ = [
    'DEFAULT_CATCHMENT',
    'make_experiment_key',
    'make_apex_tags',
    'parse_experiment_key',
    'ExperimentSpec',
    'ExperimentGrid',
    'ExperimentList',
    'ExperimentSource',
    'BatchResult',
    'BatchExperimentRunner',
    'get_model_class',
]
