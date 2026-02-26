<p align="center">
  <h1 align="center">pyrrm</h1>
  <p align="center">
    <strong>Python Rainfall-Runoff Models</strong>
  </p>
  <p align="center">
    A modern, flexible Python library for conceptual rainfall-runoff modeling,<br>
    calibration, uncertainty quantification, and sensitivity analysis.
  </p>
  <p align="center">
    <em>Developed by the Water Information Services team at the<br>
    <a href="https://www.environment.act.gov.au/">ACT Government Office of Water</a></em>
  </p>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#available-models">Models</a> •
  <a href="#calibration">Calibration</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#web-gui">Web GUI</a>
</p>

---

## Overview

**pyrrm** (Python Rainfall-Runoff Models) is a comprehensive hydrological modeling library designed for water resources engineers, researchers, and operational hydrologists. It provides robust implementations of industry-standard conceptual models with a unified, Pythonic API.

Whether you're forecasting floods, managing water supply systems, assessing climate change impacts, or conducting academic research, pyrrm offers the tools you need to transform rainfall data into actionable streamflow predictions.

### Why pyrrm?

| Challenge | pyrrm Solution |
|-----------|----------------|
| **Fragmented tools** | Unified API across all models and calibration methods |
| **Black-box calibration** | Full Bayesian uncertainty quantification with MCMC (PyDREAM + NumPyro NUTS) |
| **Slow model runs** | Numba JIT kernels (30–70x speedup); optional JAX acceleration for GPU/TPU |
| **Parameter sensitivity** | Built-in Sobol global sensitivity analysis |
| **Multi-catchment systems** | Network runner with DAG topology for upstream-to-downstream calibration |
| **Batch experiments** | Grid-based experiment design with checkpointing and parallel backends |
| **Sharing results** | Export calibration reports to Excel / CSV with full diagnostics |
| **Reproducibility** | Clean, tested code with 48-metric canonical diagnostic suite |
| **Integration** | Works seamlessly with pandas, numpy, and the Python ecosystem |

---

## Features

### Hydrological Models

- **Sacramento** — Complex multi-zone soil moisture accounting model (US NWS, 22 parameters)
- **GR4J** — Parsimonious 4-parameter daily model (INRAE, France)
- **GR5J** — Extended GR4J with improved groundwater exchange
- **GR6J** — Further extended with exponential store for low-flow simulation
- **Numba JIT Acceleration** — Compiled kernels for all four models, 30–70x faster than pure Python
- **JAX Acceleration** — GPU/TPU-accelerated Sacramento and GR4J for gradient-based calibration

### Calibration Framework

- **Bayesian MCMC** — MT-DREAM(ZS) via PyDREAM with multi-try sampling and snooker updates
- **NumPyro NUTS** — No-U-Turn Sampler for gradient-based Bayesian inference via JAX
- **Global Optimization** — SCE-UA (vendored), Differential Evolution, Dual Annealing
- **Time-Varying Parameters** — TVP framework with Gaussian Random Walk priors for non-stationary calibration
- **Multiple Objectives** — NSE, KGE (2009/2012/2021), RMSE, MAE, PBIAS, FDC metrics, APEX
- **Flow Transformations** — sqrt, log, inverse, squared, power, Box-Cox for low-flow emphasis
- **Composite Objectives** — Weighted combinations and factory functions for multi-objective calibration

### Channel Routing

- **Nonlinear Muskingum** — Storage-discharge routing with configurable sub-reaches
- **RoutedModel Wrapper** — Seamlessly combines RR models with routing for calibration
- **Automatic Parameter Integration** — Routing parameters included in calibration bounds

### Batch & Network Calibration

- **Batch Experiment Runner** — Grid-based experiment design with `ExperimentGrid` and `BatchExperimentRunner`
- **Catchment Network Runner** — Upstream-to-downstream multi-catchment calibration with DAG topology
- **Checkpoint Manager** — Resumable calibration with automatic checkpointing and recovery
- **Parallel Backends** — Sequential, Multiprocessing (ProcessPoolExecutor), and Ray task-graph scheduling

### Objective Functions Module (`pyrrm.objectives`)

- **Traditional Metrics** — NSE, RMSE, MAE, PBIAS, SDEB
- **KGE Variants** — 2009, 2012, 2021 formulations; non-parametric (Spearman) option
- **APEX** — Adaptive process-explicit objective extending SDEB with dynamics and lag multipliers
- **Flow Transformations** — sqrt, log, inverse, squared, power, Box-Cox for flow-regime emphasis
- **Composite Objectives** — Weighted combinations, factory functions (`kge_hilo`, `comprehensive_objective`, etc.)
- **FDC Metrics** — Flow duration curve segment-based evaluation
- **Hydrological Signatures** — BFI, flashiness, Q5/Q95, water balance indices

### Diagnostics & Export

- **Canonical Diagnostic Suite** — 48-metric grouped evaluation via `compute_diagnostics()` covering NSE, KGE (4 variants x 4 transforms), bias, correlation, signatures
- **Baseflow Separation** — Lyne-Hollick digital filter for baseflow/quickflow partitioning
- **CalibrationReport Export** — Save results to multi-sheet Excel or CSV with time series, diagnostics, and parameters
- **Batch Export** — Bulk export all experiments from a `BatchResult` in one call

### Analysis & Visualization

- **Sobol Sensitivity Analysis** — Identify influential parameters with confidence intervals
- **Publication-Ready Plots** — Hydrographs, scatter plots, flow duration curves (Matplotlib)
- **Interactive Plotly Plots** — Report cards, hydrographs, FDC, scatter with hover and zoom
- **MCMC Diagnostics** — Trace plots, posterior distributions, convergence metrics (R-hat), ArviZ integration
- **NUTS-Specific Visualization** — Rank plots, posterior pairs, hydrographs with uncertainty bands
- **PyDREAM Diagnostics** — R-hat bar charts, forest grids, parameter identifiability heatmaps
- **Calibration Dashboards** — Comprehensive multi-panel summaries (Matplotlib and Plotly)

### Data Utilities

- **Column Alias Resolution** — `COLUMN_ALIASES` and `resolve_column()` for robust, case-insensitive column matching
- **Convenience Loader** — `load_catchment_data()` replaces ~30 lines of CSV-loading boilerplate
- **Data Preparation Guide** — Standalone guide for hydrologists ([docs/data_preparation.md](docs/data_preparation.md))

---

## Installation

### Using pip (Recommended)

```bash
# Clone the repository
git clone https://github.com/ACTGovernment/ACT-Rainfall-Runoff-Modelling.git
cd ACT-Rainfall-Runoff-Modelling

# Install with core dependencies
pip install -e .

# Or install with all optional dependencies (recommended)
pip install -e ".[full]"
```

### Using Conda

```bash
# Create a dedicated environment
conda create -n pyrrm python=3.11 -y
conda activate pyrrm

# Install core + common optional dependencies
pip install numpy pandas scipy matplotlib seaborn plotly
pip install numba            # JIT acceleration (recommended)
pip install pydream SALib    # Calibration & sensitivity
pip install jupytext ipykernel  # Notebook support

# Install pyrrm in editable mode
pip install -e .

# Register Jupyter kernel (if using notebooks)
python -m ipykernel install --user --name pyrrm --display-name "Python (pyrrm)"
```

### Optional Dependency Groups

Install only what you need:

```bash
pip install -e ".[numba]"        # Numba JIT kernels (30-70x speedup)
pip install -e ".[calibration]"  # PyDREAM MCMC
pip install -e ".[sensitivity]"  # SALib Sobol analysis
pip install -e ".[jax]"          # JAX + NumPyro NUTS + ArviZ
pip install -e ".[export]"       # Excel export (openpyxl)
pip install -e ".[full]"         # Everything above
pip install -e ".[dev]"          # Development tools (pytest, ruff, mypy)
```

### Dependencies

| Category | Packages | Purpose |
|----------|----------|---------|
| **Core** | numpy, pandas, scipy, matplotlib | Essential functionality |
| **Acceleration** | numba | JIT-compiled model kernels (30–70x faster) |
| **Visualization** | seaborn, plotly | Enhanced and interactive plotting |
| **Calibration** | pydream | MT-DREAM(ZS) advanced MCMC |
| **JAX/Bayesian** | jax, jaxlib, numpyro, arviz | GPU acceleration, NUTS sampler, MCMC diagnostics |
| **Sensitivity** | SALib | Sobol analysis |
| **Export** | openpyxl | CalibrationReport export to Excel |
| **Notebooks** | jupytext | Paired notebook workflow |

> **Note**: pyrrm works with only core dependencies installed. Numba is strongly recommended for production use — it accelerates all model kernels with no API changes. Other optional packages enable advanced calibration, analysis, and export features.

---

## Quick Start

### Running a Model

```python
import pandas as pd
from pyrrm.models import GR4J

model = GR4J({
    'X1': 350,   # Production store capacity (mm)
    'X2': 0,     # Groundwater exchange coefficient (mm/d)
    'X3': 90,    # Routing store capacity (mm)
    'X4': 1.7    # Unit hydrograph time base (days)
})

# Load input data (must have 'precipitation' and 'pet' columns)
inputs = pd.read_csv('catchment_data.csv', index_col='date', parse_dates=True)

results = model.run(inputs, warmup_period=365)
print(f"Mean flow: {results['flow'].mean():.2f} mm/d")
```

### Loading Data with Convenience Function

```python
from pyrrm.data import load_catchment_data

inputs, observed = load_catchment_data(
    'catchment_data.csv',
    start_date='2000-01-01',
    end_date='2020-12-31'
)
```

### Calibrating with DREAM

```python
from pyrrm.models import GR4J
from pyrrm.calibration import CalibrationRunner
from pyrrm.objectives import KGE

model = GR4J()
runner = CalibrationRunner(
    model=model,
    inputs=inputs,
    observed=observed,
    objective=KGE(),
    warmup_period=365
)

result = runner.run_dream(n_iterations=10000, n_chains=5)
print(f"Best KGE: {result.best_objective:.4f}")
print(f"Best parameters: {result.best_parameters}")
```

### Evaluating with the Diagnostic Suite

```python
from pyrrm.analysis import compute_diagnostics, print_diagnostics

diagnostics = compute_diagnostics(simulated, observed)
print_diagnostics(diagnostics)
# Grouped table of 48 metrics: NSE, KGE variants, bias, signatures, etc.
```

### Exporting Results

```python
# Single report to Excel
result.create_report().export('results/my_calibration.xlsx', fmt='excel')

# Batch export all experiments
batch_result.export('results/', fmt='csv')
```

### Adding Channel Routing

```python
from pyrrm.models import Sacramento
from pyrrm.routing import NonlinearMuskingumRouter, RoutedModel
from pyrrm.calibration import CalibrationRunner
from pyrrm.objectives import NSE

rr_model = Sacramento()
router = NonlinearMuskingumRouter(K=5.0, m=0.8, n_subreaches=3)
model = RoutedModel(rr_model, router)

results = model.run(inputs)

runner = CalibrationRunner(model, inputs, observed, objective=NSE())
result = runner.run_differential_evolution()
```

### Bayesian Calibration with NumPyro NUTS

```python
from pyrrm.models import GR4J
from pyrrm.calibration import CalibrationRunner
from pyrrm.objectives import KGE

model = GR4J()
runner = CalibrationRunner(model, inputs, observed, objective=KGE(), warmup_period=365)

result = runner.run_nuts(num_warmup=500, num_samples=1000, num_chains=4)
print(f"Best KGE: {result.best_objective:.4f}")
```

### Sensitivity Analysis

```python
from pyrrm.analysis import SobolSensitivityAnalysis
from pyrrm.objectives import NSE

analysis = SobolSensitivityAnalysis(
    model=model,
    inputs=inputs,
    observed=observed,
    objective=NSE()
)

sensitivity = analysis.run(n_samples=1024)
print(sensitivity.summary())
```

### Batch Experiments

```python
from pyrrm import BatchExperimentRunner, ExperimentGrid

grid = ExperimentGrid(
    models=['GR4J', 'Sacramento'],
    objectives=['KGE', 'NSE'],
    methods=['sceua_direct', 'differential_evolution']
)

batch = BatchExperimentRunner(inputs=inputs, observed=observed, warmup_period=365)
results = batch.run(grid)
results.summary()
```

### Visualization

```python
from pyrrm.visualization import (
    plot_hydrograph_with_precipitation,
    plot_flow_duration_curve,
    plot_scatter_with_metrics,
    plot_report_card_plotly
)

fig = plot_hydrograph_with_precipitation(
    dates=inputs.index,
    observed=observed,
    simulated=results['flow'].values,
    precipitation=inputs['precipitation'].values,
    title='Calibrated GR4J Model'
)
fig.savefig('hydrograph.png', dpi=300, bbox_inches='tight')
```

---

## Available Models

### Sacramento Soil Moisture Accounting Model

The **Sacramento** model is a complex, process-based conceptual model developed by the U.S. National Weather Service. It represents the catchment as a series of interconnected soil moisture zones, each with distinct hydrological behavior.

**Key Characteristics:**
- 22 calibratable parameters (can be reduced with fixed parameters)
- Upper and lower soil zones with tension and free water components
- Explicit representation of impervious area runoff
- Separate fast (interflow) and slow (baseflow) response pathways

**Best suited for:**
- Operational flood forecasting
- Catchments with complex soil moisture dynamics
- Applications requiring detailed internal state tracking

```python
from pyrrm.models import Sacramento

model = Sacramento({
    'uztwm': 50,    # Upper zone tension water capacity (mm)
    'uzfwm': 40,    # Upper zone free water capacity (mm)
    'lztwm': 130,   # Lower zone tension water capacity (mm)
    'lzfpm': 60,    # Lower zone primary free water capacity (mm)
    'uzk': 0.3,     # Upper zone drainage rate (1/day)
    # ... additional parameters
})
```

<details>
<summary><strong>Sacramento Model Structure Diagram</strong></summary>

```
┌─────────────────────────────────────────────────────────────┐
│                         RAINFALL                             │
└─────────────────────────────────┬───────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    UPPER ZONE                                │
│  ┌──────────────────┐    ┌──────────────────┐               │
│  │   Tension Water  │    │    Free Water    │               │
│  │      (UZTW)      │◄──►│      (UZFW)      │──► Interflow  │
│  └────────┬─────────┘    └────────┬─────────┘               │
│           │                       │                          │
│           ▼                       ▼                          │
│      Evaporation             Percolation                     │
└─────────────────────────────────┬───────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────┐
│                    LOWER ZONE                                │
│  ┌──────────────────┐    ┌──────────────────┐               │
│  │   Tension Water  │    │ Primary Free Water│──► Baseflow  │
│  │      (LZTW)      │◄──►│      (LZFP)      │               │
│  └────────┬─────────┘    └──────────────────┘               │
│           │              ┌──────────────────┐               │
│           │              │Supplemental Free │──► Baseflow   │
│           │              │     (LZFS)       │               │
│           ▼              └──────────────────┘               │
│      Evaporation                                             │
└─────────────────────────────────────────────────────────────┘
```

</details>

### GR4J (Génie Rural à 4 paramètres Journalier)

The **GR4J** model is an elegant, parsimonious daily rainfall-runoff model developed by INRAE (formerly Irstea/Cemagref) in France. Despite having only 4 parameters, it achieves excellent performance across diverse catchments worldwide.

**Key Characteristics:**
- Only 4 parameters — minimal overfitting risk
- Production store (soil moisture accounting)
- Routing store with unit hydrograph convolution
- Groundwater exchange term for gains/losses

**Best suited for:**
- Regional studies with limited data
- Climate change impact assessments
- Rapid prototyping and initial catchment assessment

```python
from pyrrm.models import GR4J

model = GR4J({
    'X1': 350,   # Production store capacity (mm)
    'X2': 0,     # Groundwater exchange coefficient (mm/d)
    'X3': 90,    # Routing store capacity (mm)
    'X4': 1.7    # Unit hydrograph time base (days)
})
```

### GR5J and GR6J

Extended versions of GR4J that address specific limitations:

| Model | Parameters | Enhancement |
|-------|------------|-------------|
| **GR5J** | 5 | Additional parameter for inter-catchment groundwater exchange threshold |
| **GR6J** | 6 | Exponential store for improved simulation of low flows and recession behavior |

```python
from pyrrm.models import GR5J, GR6J

# GR5J with groundwater exchange threshold
model_5j = GR5J({'X1': 350, 'X2': 0, 'X3': 90, 'X4': 1.7, 'X5': 0.1})

# GR6J with exponential store
model_6j = GR6J({'X1': 350, 'X2': 0, 'X3': 90, 'X4': 1.7, 'X5': 0.1, 'X6': 5.0})
```

---

## Calibration

### Calibration Methods

pyrrm provides a unified `CalibrationRunner` interface that abstracts away the complexity of different optimization libraries.

| Method | Algorithm | Library | Strengths |
|--------|-----------|---------|-----------|
| `run_dream()` | MT-DREAM(ZS) | PyDREAM | Full posterior distributions, multi-try proposals, snooker updates |
| `run_nuts()` | NUTS | NumPyro/JAX | Gradient-based Bayesian inference, efficient high-dimensional sampling |
| `run_sceua_direct()` | SCE-UA | vendored | Fast global convergence, no external deps, PCA recovery |
| `run_scipy()` | Differential Evolution | SciPy | Robust, no external dependencies |
| `run_scipy(method='dual_annealing')` | Dual Annealing | SciPy | Escapes local minima, rugged landscapes |

### Objective Functions

pyrrm includes a comprehensive `objectives` module with traditional metrics, KGE variants, flow transformations, and composite objectives.

**Traditional Metrics:**

| Function | Description | Range | Optimal |
|----------|-------------|-------|---------|
| `NSE` | Nash-Sutcliffe Efficiency | (-∞, 1] | 1 |
| `RMSE` | Root Mean Square Error | [0, ∞) | 0 |
| `MAE` | Mean Absolute Error | [0, ∞) | 0 |
| `PBIAS` | Percent Bias | (-∞, ∞) | 0 |
| `SDEB` | Sum of Daily Flows, Daily Exceedance Curve and Bias | varies | varies |

**KGE Family:**

| Function | Description | Variants |
|----------|-------------|----------|
| `KGE` | Kling-Gupta Efficiency | 2009, 2012, 2021 |
| `KGENonParametric` | Non-parametric KGE (Spearman) | - |

**APEX (Adaptive Process-Explicit):**

A novel objective function that extends SDEB with dynamics and lag penalty multipliers:

```python
from pyrrm.objectives import APEX

apex = APEX(
    alpha=0.1,              # Weight for chronological term
    dynamics_strength=0.5,  # κ: strength of dynamics penalty
    regime_emphasis='uniform'
)

runner = CalibrationRunner(model, inputs, observed, objective=apex)
```

APEX formula: `APEX = [α × E_chron + (1-α) × E_ranked] × BiasMultiplier × DynamicsMultiplier × [LagMultiplier]`

**Flow Transformations:**

Apply transformations to shift emphasis between high and low flows:

```python
from pyrrm.objectives import KGE, FlowTransformation

kge = KGE()                                         # Standard (high-flow emphasis)
kge_log = KGE(transform=FlowTransformation('log'))   # Low-flow emphasis
kge_inv = KGE(transform=FlowTransformation('inverse'))  # Strong low-flow emphasis
```

Available transformations: `none`, `sqrt`, `log`, `inverse`, `squared`, `power`, `boxcox`

### Multi-Objective Calibration

```python
from pyrrm.objectives import NSE, KGE, FlowTransformation, WeightedObjective

combined = WeightedObjective([
    (KGE(), 0.5),
    (KGE(transform=FlowTransformation('inverse')), 0.5),
])

# Or use factory functions for common combinations
from pyrrm.objectives import kge_hilo, comprehensive_objective

objective = kge_hilo(kge_weight=0.5)
objective = comprehensive_objective()

runner = CalibrationRunner(model, inputs, observed, objective=objective)
result = runner.run_dream(n_iterations=15000)
```

---

## Applications

pyrrm is designed to support a wide range of water resources applications:

### Flood Forecasting
Use calibrated models for real-time streamflow prediction during storm events. The Sacramento model's detailed soil moisture accounting makes it particularly suitable for operational forecasting systems.

### Water Supply Planning
Simulate reservoir inflows under various climate scenarios to support long-term water supply planning and drought contingency development.

### Climate Change Impact Assessment
Evaluate how changes in precipitation and temperature patterns may affect future streamflow regimes, using the GR4J family's parsimonious structure to minimize overfitting.

### Environmental Flows
Model natural flow regimes to establish environmental flow requirements and assess the impacts of water extraction on aquatic ecosystems.

### Multi-Catchment Network Calibration
Calibrate interconnected catchment systems in topological order, using upstream simulated flows as downstream inputs — essential for nested basin studies and water supply networks.

### Research & Education
The clean API and comprehensive documentation make pyrrm an excellent platform for teaching rainfall-runoff concepts and conducting academic research.

---

## Project Structure

```
pyrrm/
├── __init__.py                    # Package entry point (lazy imports)
├── parallel.py                    # Parallel backends (Sequential, Multiprocessing, Ray)
├── models/                        # Rainfall-runoff model implementations
│   ├── base.py                    # BaseRainfallRunoffModel abstract class
│   ├── sacramento.py              # Sacramento Soil Moisture Accounting
│   ├── gr4j.py                    # GR4J model
│   ├── gr5j.py                    # GR5J model
│   ├── gr6j.py                    # GR6J model
│   ├── numba_kernels.py           # Numba JIT-compiled kernels (all models)
│   ├── sacramento_jax.py          # Sacramento JAX implementation (GPU)
│   ├── gr4j_jax.py                # GR4J JAX implementation (GPU)
│   └── utils/                     # Shared utilities
│       ├── unit_hydrograph.py     # Unit hydrograph convolution
│       ├── s_curves.py            # S-curve interpolation
│       └── s_curves_jax.py        # S-curves JAX implementation
├── routing/                       # Channel routing methods
│   ├── base.py                    # BaseRouter abstract class
│   ├── muskingum.py               # NonlinearMuskingumRouter
│   └── routed_model.py            # RoutedModel wrapper for RR + routing
├── objectives/                    # Comprehensive objective functions library
│   ├── core/                      # Base classes, MetricResult, utilities
│   ├── metrics/                   # NSE, KGE, RMSE, MAE, PBIAS, SDEB, APEX
│   ├── transformations/           # sqrt, log, inverse, power, boxcox
│   ├── fdc/                       # Flow duration curve metrics
│   ├── signatures/                # Hydrological signatures (BFI, flashiness, etc.)
│   └── composite/                 # WeightedObjective, factory functions
├── calibration/                   # Calibration framework
│   ├── runner.py                  # Unified CalibrationRunner interface
│   ├── report.py                  # CalibrationReport for saving/loading results
│   ├── export.py                  # Export to Excel / CSV
│   ├── batch.py                   # BatchExperimentRunner and ExperimentGrid
│   ├── checkpoint.py              # CheckpointManager for resumable calibration
│   ├── pydream_adapter.py         # PyDREAM MT-DREAM(ZS) adapter
│   ├── numpyro_adapter.py         # NumPyro NUTS sampler adapter
│   ├── scipy_adapter.py           # SciPy optimization adapter
│   ├── sceua_adapter.py           # Direct SCE-UA implementation
│   ├── tvp_priors.py              # Time-varying parameter priors
│   ├── likelihoods_jax.py         # JAX-based likelihood functions
│   └── objective_functions.py     # Legacy compatibility layer
├── network/                       # Multi-catchment network calibration
│   ├── topology.py                # CatchmentNetwork DAG representation
│   ├── data.py                    # NetworkDataLoader for multi-catchment data
│   ├── runner.py                  # CatchmentNetworkRunner
│   └── visualization.py           # Network visualization (Mermaid, plots)
├── analysis/                      # Post-calibration analysis
│   ├── diagnostics.py             # Canonical 48-metric diagnostic suite
│   ├── sensitivity.py             # Sobol global sensitivity analysis
│   └── mcmc_diagnostics.py        # MCMC convergence diagnostics (ArviZ)
├── visualization/                 # Plotting functions
│   ├── model_plots.py             # Hydrographs, FDC, scatter plots
│   ├── calibration_plots.py       # Parameter traces, posteriors
│   ├── report_plots.py            # Report cards (Matplotlib & Plotly)
│   ├── sensitivity_plots.py       # Sobol indices visualization
│   └── mcmc_plots.py              # MCMC traces, rank plots, uncertainty bands
├── data/                          # Data handling utilities
│   ├── input_handler.py           # COLUMN_ALIASES, load_catchment_data()
│   └── parameter_bounds.py        # Parameter bounds definitions
└── examples/                      # Example scripts
    ├── quick_start.py
    └── mpi_calibration_example.py
```

### Additional Directories

```
notebooks/           # 13 Jupytext-paired educational notebooks (see Documentation)
notebooks_ACT/       # ACT Government catchment calibration notebooks
benchmark/           # Sacramento verification against C# reference implementation
test_data/           # Test data and saved calibration reports
docs/                # Supplementary documentation (data preparation guide, specs)
pyrrm-gui/           # Web-based GUI application (React + FastAPI + Celery)
diagrams/            # Architecture and model diagrams
```

---

## Web GUI

pyrrm includes a web-based graphical user interface for running and analyzing calibrations without writing code. Built with React (frontend), FastAPI (backend), and Celery (background workers).

**Features:**
- Step-by-step experiment configuration wizard
- Asynchronous calibration with real-time WebSocket progress
- Interactive results visualization (hydrographs, FDC, scatter)
- Experiment comparison dashboard

```bash
cd pyrrm-gui
docker-compose up --build
# Frontend: http://localhost:3000
# API Docs: http://localhost:8000/api/docs
```

See [pyrrm-gui/README.md](pyrrm-gui/README.md) for full setup instructions.

---

## Verification & Testing

The Sacramento implementation has been rigorously verified against reference C# implementations to ensure numerical accuracy.

```bash
# Run verification suite
python benchmark/run_benchmark.py --verify

# Generate comprehensive verification report
python benchmark/run_benchmark.py --all
```

Key verification metrics:
- **Correlation**: R² > 0.9999 against reference implementation
- **Mean Absolute Error**: < 0.001 mm/day
- **Mass Balance**: Preserved within numerical precision

Numba kernels are tested for exact equivalence (96 tests) across all models, edge cases, and class interfaces.

See the `benchmark/` directory for verification scripts and detailed results.

---

## Documentation

### Tutorial Notebooks

The `notebooks/` directory contains 13 educational notebooks (Jupytext-paired `.py` scripts) that guide you through pyrrm's capabilities:

| # | Notebook | Description | Best For |
|---|----------|-------------|----------|
| 01 | [sacramento_verification](notebooks/01_sacramento_verification.py) | Verification against C# and SOURCE benchmarks | Implementation correctness |
| 02 | [calibration_quickstart](notebooks/02_calibration_quickstart.py) | **Start here!** Loading data and calibrating models | **New users** |
| 03 | [routing_quickstart](notebooks/03_routing_quickstart.py) | Channel routing with Nonlinear Muskingum | Adding routing to models |
| 04 | [objective_functions](notebooks/04_objective_functions.py) | Deep dive into objective functions | Understanding what to optimize |
| 05 | [apex_complete_guide](notebooks/05_apex_complete_guide.py) | APEX objective function research & evaluation | Advanced calibration |
| 06 | [algorithm_comparison](notebooks/06_algorithm_comparison.py) | PyDREAM, SCE-UA, SciPy comparison with ArviZ diagnostics | Choosing algorithms |
| 07 | [model_comparison](notebooks/07_model_comparison.py) | GR4J, GR5J, GR6J vs Sacramento (13 objectives) | Choosing models |
| 08 | [calibration_monitor](notebooks/08_calibration_monitor.py) | Real-time monitoring of MCMC calibrations | Long-running jobs |
| 09 | [calibration_reports](notebooks/09_calibration_reports.py) | Working with saved CalibrationReport objects | Post-processing |
| 10 | [batch_runners](notebooks/10_batch_runners.py) | Batch experiment runner tutorial | Production workflows |
| 11 | [network_runners](notebooks/11_network_runners.py) | Catchment network calibration | Multi-catchment systems |
| 12 | [nuts_calibration](notebooks/12_nuts_calibration.py) | Bayesian MCMC with NumPyro NUTS | Bayesian inference |
| 13 | [tvp_gr4j](notebooks/13_tvp_gr4j.py) | Time-varying parameter calibration with GR4J | Non-stationary catchments |

#### Learning Paths

**Quick Start (1–2 hours):**
1. [02_calibration_quickstart](notebooks/02_calibration_quickstart.py) — Get a working calibration

**Complete Understanding (6–8 hours):**
1. [02_calibration_quickstart](notebooks/02_calibration_quickstart.py) — Fundamentals
2. [04_objective_functions](notebooks/04_objective_functions.py) — What to optimize
3. [06_algorithm_comparison](notebooks/06_algorithm_comparison.py) — How to optimize
4. [07_model_comparison](notebooks/07_model_comparison.py) — Which model to use
5. [03_routing_quickstart](notebooks/03_routing_quickstart.py) — Adding channel routing
6. [08_calibration_monitor](notebooks/08_calibration_monitor.py) — Monitor long runs

**Bayesian Inference (4–6 hours):**
1. [02_calibration_quickstart](notebooks/02_calibration_quickstart.py) — Calibration fundamentals
2. [06_algorithm_comparison](notebooks/06_algorithm_comparison.py) — Compare MCMC vs optimization
3. [12_nuts_calibration](notebooks/12_nuts_calibration.py) — NumPyro NUTS sampler
4. [13_tvp_gr4j](notebooks/13_tvp_gr4j.py) — Time-varying parameters

**Production Workflows (3–4 hours):**
1. [02_calibration_quickstart](notebooks/02_calibration_quickstart.py) — Single catchment calibration
2. [10_batch_runners](notebooks/10_batch_runners.py) — Grid-based batch experiments
3. [11_network_runners](notebooks/11_network_runners.py) — Multi-catchment network calibration

**Model Selection:**
- [07_model_comparison](notebooks/07_model_comparison.py) — Compare GR4J, GR5J, GR6J, Sacramento across 13 objectives

**Advanced Calibration:**
- [05_apex_complete_guide](notebooks/05_apex_complete_guide.py) — Research-focused APEX evaluation with 6 research questions

**Verification/Validation:**
- [01_sacramento_verification](notebooks/01_sacramento_verification.py) — Implementation correctness

### Supplementary Documentation

| Document | Description |
|----------|-------------|
| [Data Preparation Guide](docs/data_preparation.md) | How to prepare input data for single-catchment, batch, and network workflows |
| [APEX Guide](docs/APEX_GUIDE.md) | Detailed APEX objective function documentation |
| [Benchmark Scripts](benchmark/) | Sacramento verification against C# reference implementation |
| [Changelog](CHANGELOG.md) | All notable changes to pyrrm |
| [Lessons Learnt](LESSONS_LEARNT.md) | Development insights and pitfalls |

### API Reference

Detailed API documentation is available in the source code docstrings, following Google-style conventions.

```python
from pyrrm.models import GR4J
help(GR4J)
help(GR4J.run)
```

---

## References

### Models

**Sacramento Model:**
> Burnash, R.J.C., Ferral, R.L., & McGuire, R.A. (1973). *A generalized streamflow simulation system: Conceptual modeling for digital computers*. U.S. National Weather Service and California Department of Water Resources, Sacramento, CA.

**GR4J Model:**
> Perrin, C., Michel, C., & Andréassian, V. (2003). Improvement of a parsimonious model for streamflow simulation. *Journal of Hydrology*, 279(1-4), 275-289. https://doi.org/10.1016/S0022-1694(03)00225-7

**GR5J/GR6J Models:**
> Le Moine, N. (2008). *Le bassin versant de surface vu par le souterrain: une voie d'amélioration des performances et du réalisme des modèles pluie-débit?* PhD Thesis, Université Pierre et Marie Curie, Paris.

### Calibration Algorithms

**SCE-UA:**
> Duan, Q., Sorooshian, S., & Gupta, V. (1992). Effective and efficient global optimization for conceptual rainfall-runoff models. *Water Resources Research*, 28(4), 1015-1031.

**DREAM:**
> Vrugt, J.A. (2016). Markov chain Monte Carlo simulation using the DREAM software package. *Environmental Modelling & Software*, 75, 273-316.

### Software

**SALib:**
> Herman, J., & Usher, W. (2017). SALib: An open-source Python library for sensitivity analysis. *Journal of Open Source Software*, 2(9), 97. https://doi.org/10.21105/joss.00097

**NumPyro:**
> Phan, D., Pradhan, N., & Jankowiak, M. (2019). Composable effects for flexible and accelerated probabilistic programming in NumPyro. *arXiv preprint arXiv:1912.11554*.

**JAX:**
> Bradbury, J., Frostig, R., Hawkins, P., et al. (2018). JAX: composable transformations of Python+NumPy programs. http://github.com/google/jax

---

## Contributing

Contributions are welcome! Whether you're fixing bugs, adding new models, improving documentation, or suggesting features, we appreciate your help.

### Development Setup

```bash
git clone https://github.com/ACTGovernment/ACT-Rainfall-Runoff-Modelling.git
cd ACT-Rainfall-Runoff-Modelling
pip install -e ".[dev]"

# Run tests
pytest tests/
```

### Guidelines

- Follow the existing code style (type hints, Google-style docstrings)
- Add tests for new functionality
- Update documentation as needed
- See the `.cursor/rules/` directory for detailed development guidelines

---

## License

This project is released under the **MIT License**.

---

## Acknowledgments

Developed by the **Water Information Services** team at the **ACT Government Office of Water** for water resources management and research applications in the Australian Capital Territory and beyond.

---

<p align="center">
  <sub>Built with Python and a passion for hydrology</sub>
</p>
