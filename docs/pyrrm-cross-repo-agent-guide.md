# pyrrm Library Reference — Cross-Repository Agent Guide

This document enables a Cursor AI agent working in *this* repository to locate and use the **pyrrm** (Python Rainfall-Runoff Models) library, which lives in a separate repository on this machine.

## Repository Location

```
/Users/jcastilla/Desktop/ACTGOV/ACT-Rainfall-Runoff-Modelling
```

The pyrrm package source is at:

```
/Users/jcastilla/Desktop/ACTGOV/ACT-Rainfall-Runoff-Modelling/pyrrm/
```

## Conda Environment

All Python execution involving pyrrm **must** use the `pyrrm` conda environment:

```bash
conda activate pyrrm
```

pyrrm is installed in editable mode inside this environment. You do not need to `pip install` it again — just activate the environment.

## How to Import pyrrm

Once the `pyrrm` conda environment is active, pyrrm is importable from anywhere:

```python
import pyrrm
from pyrrm.models import Sacramento, GR4J, GR5J, GR6J
from pyrrm.calibration import CalibrationRunner
from pyrrm.objectives import NSE, KGE, PBIAS, RMSE
```

If you need to add pyrrm to `sys.path` manually (e.g., in a script that doesn't use the conda env):

```python
import sys
sys.path.insert(0, "/Users/jcastilla/Desktop/ACTGOV/ACT-Rainfall-Runoff-Modelling")
import pyrrm
```

## Library Architecture

### Package Structure

```
/Users/jcastilla/Desktop/ACTGOV/ACT-Rainfall-Runoff-Modelling/pyrrm/
├── __init__.py              # Top-level exports (lazy imports)
├── parallel.py              # Parallel backends (Sequential, Multiprocessing, Ray)
├── models/                  # Rainfall-runoff model implementations
│   ├── base.py              # BaseRainfallRunoffModel abstract class
│   ├── sacramento.py        # Sacramento model (22 parameters)
│   ├── gr4j.py              # GR4J model (4 parameters)
│   ├── gr5j.py              # GR5J model (5 parameters)
│   ├── gr6j.py              # GR6J model (6 parameters)
│   ├── numba_kernels.py     # Numba JIT-compiled kernels for all models
│   ├── sacramento_jax.py    # Sacramento JAX implementation
│   ├── gr4j_jax.py          # GR4J JAX implementation
│   └── utils/               # Unit hydrographs, S-curves
├── routing/                 # Channel routing
│   ├── base.py              # BaseRouter abstract class
│   ├── muskingum.py         # NonlinearMuskingumRouter
│   └── routed_model.py      # RoutedModel wrapper (combines RR model + routing)
├── objectives/              # Comprehensive objective functions library
│   ├── core/                # Base classes, MetricResult, utilities
│   ├── metrics/             # NSE, KGE, RMSE, MAE, PBIAS, SDEB, APEX
│   ├── transformations/     # Flow transformations (sqrt, log, inverse)
│   ├── fdc/                 # Flow duration curve metrics
│   ├── signatures/          # Hydrological signatures
│   └── composite/           # WeightedObjective, factory functions
├── calibration/             # Calibration framework
│   ├── runner.py            # CalibrationRunner unified interface
│   ├── report.py            # CalibrationReport for saving/loading
│   ├── export.py            # Export to Excel / CSV
│   ├── batch.py             # BatchExperimentRunner and ExperimentGrid
│   ├── checkpoint.py        # CheckpointManager for resumable calibration
│   ├── pydream_adapter.py   # PyDREAM MT-DREAM(ZS)
│   ├── numpyro_adapter.py   # NumPyro NUTS
│   ├── scipy_adapter.py     # SciPy optimization adapter
│   ├── sceua_adapter.py     # Vendored SCE-UA (no external deps)
│   └── objective_functions.py # Legacy compatibility layer
├── network/                 # Multi-catchment network calibration
│   ├── topology.py          # CatchmentNetwork DAG
│   ├── data.py              # NetworkDataLoader
│   ├── runner.py            # CatchmentNetworkRunner
│   └── visualization.py     # Network plots and Mermaid diagrams
├── analysis/                # Post-calibration analysis
│   ├── diagnostics.py       # 48-metric canonical diagnostic suite
│   ├── sensitivity.py       # Sobol global sensitivity analysis
│   └── mcmc_diagnostics.py  # MCMC convergence diagnostics (ArviZ)
├── visualization/           # Plotting (Matplotlib and Plotly)
│   ├── model_plots.py       # Hydrographs, FDC, scatter plots
│   ├── calibration_plots.py # Parameter traces, posteriors
│   ├── report_plots.py      # Report cards (Matplotlib & Plotly)
│   ├── sensitivity_plots.py # Sobol indices visualization
│   └── mcmc_plots.py        # MCMC traces, rank plots, uncertainty bands
├── data/                    # Data handling
│   ├── input_handler.py     # COLUMN_ALIASES, load_catchment_data()
│   └── parameter_bounds.py  # Parameter bounds definitions
├── bma/                     # Bayesian Model Averaging
│   ├── config.py            # BMAConfig, presets
│   ├── pipeline.py          # BMARunner
│   ├── data_prep.py         # Data preparation
│   ├── pre_screen.py        # Model pre-screening
│   ├── level1_equal.py      # Equal weights
│   ├── level2_grc.py        # Constrained regression weights
│   ├── level3_stacking.py   # Bayesian stacking
│   ├── level4_bma.py        # Full BMA (PyMC)
│   ├── level5_regime_bma.py # Regime-specific BMA
│   ├── evaluation.py        # BMA evaluation utilities
│   ├── prediction.py        # Prediction utilities
│   └── visualization.py     # BMA visualizations
└── examples/                # Example scripts
    └── quick_start.py       # Basic usage examples
```

### Key Reference Files

To understand specific components, read these files:

| Component | File |
|---|---|
| Model interface (ABC) | `/Users/jcastilla/Desktop/ACTGOV/ACT-Rainfall-Runoff-Modelling/pyrrm/models/base.py` |
| GR4J implementation | `/Users/jcastilla/Desktop/ACTGOV/ACT-Rainfall-Runoff-Modelling/pyrrm/models/gr4j.py` |
| Sacramento implementation | `/Users/jcastilla/Desktop/ACTGOV/ACT-Rainfall-Runoff-Modelling/pyrrm/models/sacramento.py` |
| CalibrationRunner | `/Users/jcastilla/Desktop/ACTGOV/ACT-Rainfall-Runoff-Modelling/pyrrm/calibration/runner.py` |
| Objective functions | `/Users/jcastilla/Desktop/ACTGOV/ACT-Rainfall-Runoff-Modelling/pyrrm/objectives/__init__.py` |
| Data loading | `/Users/jcastilla/Desktop/ACTGOV/ACT-Rainfall-Runoff-Modelling/pyrrm/data/input_handler.py` |
| Routing | `/Users/jcastilla/Desktop/ACTGOV/ACT-Rainfall-Runoff-Modelling/pyrrm/routing/__init__.py` |
| Network calibration | `/Users/jcastilla/Desktop/ACTGOV/ACT-Rainfall-Runoff-Modelling/pyrrm/network/__init__.py` |
| BMA | `/Users/jcastilla/Desktop/ACTGOV/ACT-Rainfall-Runoff-Modelling/pyrrm/bma/__init__.py` |
| Quick start examples | `/Users/jcastilla/Desktop/ACTGOV/ACT-Rainfall-Runoff-Modelling/pyrrm/examples/quick_start.py` |
| Tutorial notebooks (.py) | `/Users/jcastilla/Desktop/ACTGOV/ACT-Rainfall-Runoff-Modelling/notebooks/` |

## Common Usage Patterns

### 1. Loading Catchment Data

```python
from pyrrm.data import load_catchment_data

inputs, observed = load_catchment_data(
    precipitation_file="rain.csv",
    pet_file="pet.csv",
    observed_file="flow.csv",
    start_date="2000-01-01",
    end_date="2024-12-31",
)
# inputs: DataFrame with DatetimeIndex and columns ['precipitation', 'pet']
# observed: 1-D numpy array of observed flow values
```

Column aliases are flexible — pyrrm auto-detects columns named `precipitation`, `rainfall`, `precip`, `rain`, `P`, etc. Same for PET and observed flow. See `COLUMN_ALIASES` in `pyrrm/data/input_handler.py`.

### 2. Running a Model

```python
from pyrrm.models import GR4J

model = GR4J({'X1': 350, 'X2': 0, 'X3': 90, 'X4': 1.7})
results = model.run(inputs)  # Returns DataFrame with 'flow' column (mm/day)

# With catchment area → outputs in ML/day
model.catchment_area_km2 = 150.0
results = model.run(inputs)  # 'flow' now in ML/day
```

### 3. Calibrating a Model

```python
from pyrrm.models import GR4J
from pyrrm.calibration import CalibrationRunner
from pyrrm.objectives import NSE

model = GR4J()
runner = CalibrationRunner(model, inputs, observed, objective=NSE())

# SCE-UA (recommended — no external dependencies)
n_params = len(model.get_parameter_bounds())
result = runner.run_sceua_direct(
    max_evals=10000,
    n_complexes=2 * n_params + 1,  # always use 2n+1
    seed=42,
)
print(result.summary())
print(result.best_parameters)
```

### 4. Model with Channel Routing

```python
from pyrrm.models import Sacramento
from pyrrm.routing import NonlinearMuskingumRouter, RoutedModel

rr_model = Sacramento()
router = NonlinearMuskingumRouter(K=5.0, m=0.8, n_subreaches=3)
model = RoutedModel(rr_model, router)

# Routing parameters are automatically included in calibration bounds
runner = CalibrationRunner(model, inputs, observed, objective=NSE())
result = runner.run_sceua_direct(max_evals=15000, n_complexes=51, seed=42)
```

### 5. Diagnostics and Visualization

```python
from pyrrm.analysis import compute_diagnostics, print_diagnostics
from pyrrm.visualization import (
    plot_hydrograph_with_precipitation,
    plot_flow_duration_curve,
    create_calibration_dashboard,
)

# 48-metric diagnostic suite
diag = compute_diagnostics(observed, simulated)
print_diagnostics(diag)

# Plots
plot_hydrograph_with_precipitation(inputs, results, observed)
plot_flow_duration_curve(observed, simulated)
```

### 6. Batch Experiments

```python
from pyrrm.calibration import BatchExperimentRunner, ExperimentGrid

grid = ExperimentGrid(
    models=["GR4J", "Sacramento"],
    objectives=["NSE", "KGE"],
    seeds=[42, 123],
)
batch = BatchExperimentRunner(grid, inputs, observed)
results = batch.run_all()
```

### 7. Multi-Catchment Network

```python
from pyrrm.network import CatchmentNetwork, CatchmentNetworkRunner

network = CatchmentNetwork.from_csv(
    'nodes.csv', 'links.csv', data_dir='./data'
)
runner = CatchmentNetworkRunner(network, default_model_class='GR4J')
result = runner.run()
```

### 8. Bayesian Model Averaging

```python
from pyrrm.bma import BMAConfig, BMARunner

config = BMAConfig.from_preset("standard",
    model_predictions_path="predictions.csv",
    observed_flow_path="observed.csv",
)
runner = BMARunner(config)
runner.load_data()
runner.pre_screen()
cv_df = runner.run_cross_validation()
runner.fit_final()
runner.save_results()
```

## Available Models

| Model | Parameters | Key Params | Import |
|---|---|---|---|
| Sacramento | 22 | uztwm, uzfwm, lztwm, lzfwm, uzk, lzpk | `from pyrrm.models import Sacramento` |
| GR4J | 4 | X1 (prod store), X2 (exchange), X3 (routing store), X4 (UH time) | `from pyrrm.models import GR4J` |
| GR5J | 5 | X1–X4 + X5 (intercatchment exchange threshold) | `from pyrrm.models import GR5J` |
| GR6J | 6 | X1–X5 + X6 (exponential store capacity) | `from pyrrm.models import GR6J` |

## Available Objective Functions

| Metric | Class | Import |
|---|---|---|
| Nash-Sutcliffe Efficiency | `NSE` | `from pyrrm.objectives import NSE` |
| Kling-Gupta Efficiency | `KGE` (supports 2009/2012/2021 variants) | `from pyrrm.objectives import KGE` |
| Root Mean Square Error | `RMSE` | `from pyrrm.objectives import RMSE` |
| Mean Absolute Error | `MAE` | `from pyrrm.objectives import MAE` |
| Percent Bias | `PBIAS` | `from pyrrm.objectives import PBIAS` |
| SDEB (Sum Daily / Exceedance / Bias) | `SDEB` | `from pyrrm.objectives import SDEB` |
| APEX (enhanced SDEB) | `APEX` | `from pyrrm.objectives import APEX` |
| Non-parametric KGE | `KGENonParametric` | `from pyrrm.objectives import KGENonParametric` |
| Flow Duration Curve | `FDCMetric` | `from pyrrm.objectives import FDCMetric` |
| Hydrological Signatures | `SignatureMetric` | `from pyrrm.objectives import SignatureMetric` |

Composite factories: `kge_hilo`, `fdc_multisegment`, `comprehensive_objective`, `apex_objective`.

Flow transformations (for low/high flow emphasis): `FlowTransformation('sqrt')`, `FlowTransformation('log')`, `FlowTransformation('inverse')`.

## Available Calibration Methods

| Method | Function | Dependencies | Notes |
|---|---|---|---|
| SCE-UA | `runner.run_sceua_direct()` | None (vendored) | Recommended default. Set `n_complexes = 2n+1`. |
| Differential Evolution | `runner.run_differential_evolution()` | scipy | SciPy's DE implementation |
| Dual Annealing | `runner.run_dual_annealing()` | scipy | Global + local search |
| PyDREAM (MT-DREAM(ZS)) | `runner.run_dream()` | pydream | Bayesian MCMC with multi-try |
| NumPyro NUTS | `runner.run_nuts()` | numpyro, jax | Hamiltonian Monte Carlo |

## Input Data Format

pyrrm models expect a `pandas.DataFrame` with:
- **Index**: `DatetimeIndex` (daily frequency)
- **Required columns**: `precipitation` (mm/day), `pet` (mm/day)
- **Optional column**: `observed_flow` (for calibration)

Column names are flexible thanks to built-in aliases (see `COLUMN_ALIASES` in `pyrrm/data/input_handler.py`).

## Model Interface Contract

All models inherit from `BaseRainfallRunoffModel` and implement:

```python
model.run(inputs: pd.DataFrame) -> pd.DataFrame      # Full simulation
model.run_timestep(precipitation, pet) -> dict         # Single timestep
model.reset() -> None                                  # Reset internal states
model.get_parameters() -> Dict[str, float]             # Current parameters
model.set_parameters(params: Dict[str, float]) -> None # Update parameters
model.get_parameter_bounds() -> Dict[str, Tuple]       # (min, max) bounds
model.get_state() -> ModelState                        # Save internal state
model.set_state(state: ModelState) -> None             # Restore internal state
```

Output DataFrames always contain a `flow` column (mm/day by default, ML/day if `catchment_area_km2` is set).

## Dependencies

**Core** (always required): numpy, pandas, scipy, matplotlib

**Optional** (gracefully degraded if absent):
- `numba` — 30-70x faster model runs via JIT compilation
- `pydream` — MT-DREAM(ZS) Bayesian calibration
- `numpyro`, `jax` — NumPyro NUTS calibration
- `pymc`, `arviz` — BMA Levels 4-5
- `scikit-learn` — BMA stacking
- `SALib` — Sobol sensitivity analysis
- `seaborn` — Enhanced visualization
- `openpyxl` — Excel export

## Tutorial Notebooks

Jupytext-paired Python scripts (the `.py` file is the source of truth):

```
/Users/jcastilla/Desktop/ACTGOV/ACT-Rainfall-Runoff-Modelling/notebooks/
├── 01_sacramento_verification.py
├── 02_calibration_quickstart.py
├── 03_routing_quickstart.py
├── 04_objective_functions.py
├── 05_apex_complete_guide.py
├── 06_algorithm_comparison.py
├── 07_model_comparison.py
├── 08_calibration_monitor.py
├── 09_calibration_reports.py
├── 10_batch_runners.py
├── 11_network_runners.py
├── 12_nuts_calibration.py
└── 13_tvp_gr4j.py
```

## Cursor Rules for pyrrm Development

The pyrrm repository has its own `.cursor/rules/` with detailed agent instructions. Key rules:

| Rule File | Purpose |
|---|---|
| `pyrrm-agent-instructions.mdc` | High-level agent guidance |
| `pyrrm-development.mdc` | Development guidelines and patterns |
| `conda-environment.mdc` | Environment activation requirements |
| `jupytext-notebooks.mdc` | Notebook workflow (never edit .ipynb directly) |
| `commit-messages.mdc` | Conventional commit format |
| `sceua-calibration.mdc` | SCE-UA configuration (n_complexes = 2n+1) |
| `changelog-maintenance.mdc` | CHANGELOG.md maintenance |

These are located at:
```
/Users/jcastilla/Desktop/ACTGOV/ACT-Rainfall-Runoff-Modelling/.cursor/rules/
```

## Tips for the Agent

1. **Read before using**: Before importing or building on any pyrrm component, read the relevant source file to understand the current API — the library is under active development.
2. **Use `load_catchment_data()`** for data loading — it handles column aliases, missing value sentinels, and date parsing automatically.
3. **SCE-UA is the recommended calibration method** — it's vendored (zero external deps) and robust. Always set `n_complexes = 2 * n_params + 1`.
4. **Models output mm/day by default**. Set `model.catchment_area_km2` to get ML/day.
5. **Numba acceleration is automatic** — if numba is installed, models use JIT-compiled kernels transparently.
6. **The `pyrrm` conda env must be active** for all Python execution involving pyrrm.
