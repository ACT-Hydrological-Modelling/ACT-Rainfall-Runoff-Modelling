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
  <a href="#documentation">Documentation</a>
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
| **Slow calibration** | JAX-accelerated models for GPU/TPU speedup |
| **Parameter sensitivity** | Built-in Sobol global sensitivity analysis |
| **Multi-catchment systems** | Network runner with DAG topology for upstream-to-downstream calibration |
| **Reproducibility** | Clean, tested code with comprehensive documentation |
| **Integration** | Works seamlessly with pandas, numpy, JAX, and the Python ecosystem |

---

## Features

### Hydrological Models

- **Sacramento** — Complex multi-zone soil moisture accounting model (US NWS)
- **GR4J** — Parsimonious 4-parameter daily model (INRAE, France)
- **GR5J** — Extended GR4J with improved groundwater exchange
- **GR6J** — Further extended with exponential store for low-flow simulation
- **JAX Acceleration** — GPU/TPU-accelerated implementations of Sacramento and GR4J for fast calibration

### Calibration Framework

- **Bayesian MCMC** — MT-DREAM(ZS) via PyDREAM with multi-try sampling and snooker updates
- **NumPyro NUTS** — No-U-Turn Sampler for gradient-based Bayesian inference via JAX
- **Global Optimization** — SCE-UA (vendored), Differential Evolution, Dual Annealing
- **Time-Varying Parameters** — TVP framework with Gaussian Random Walk priors for non-stationary calibration
- **Multiple Objectives** — NSE, KGE (2009/2012/2021), RMSE, MAE, PBIAS, FDC metrics, APEX
- **Flow Transformations** — sqrt, log, inverse transforms for low-flow emphasis
- **Composite Objectives** — Weighted combinations for multi-objective calibration

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
- **KGE Variants** — 2009, 2012, 2021 formulations; non-parametric options
- **APEX** — Novel adaptive process-explicit objective extending SDEB with dynamics and lag multipliers
- **Flow Transformations** — sqrt, log, inverse for low-flow emphasis
- **Composite Objectives** — Weighted combinations, factory functions
- **FDC Metrics** — Flow duration curve segment-based evaluation
- **Hydrological Signatures** — Flow indices, dynamics, water balance

### Analysis & Visualization

- **Sobol Sensitivity Analysis** — Identify influential parameters with confidence intervals
- **Publication-Ready Plots** — Hydrographs, scatter plots, flow duration curves
- **Interactive Plotly Plots** — Report cards, hydrographs, FDC, scatter with hover and zoom
- **MCMC Diagnostics** — Trace plots, posterior distributions, convergence metrics (R-hat), ArviZ integration
- **NUTS-Specific Visualization** — Rank plots, posterior pairs, hydrographs with uncertainty bands
- **Calibration Dashboards** — Comprehensive multi-panel summaries (Matplotlib and Plotly)

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

# Install dependencies
pip install numpy pandas scipy matplotlib seaborn plotly
pip install pydream SALib  # Optional: calibration libraries
pip install jupytext ipykernel    # Optional: notebook support

# Install pyrrm
pip install -e .

# Register Jupyter kernel (if using notebooks)
python -m ipykernel install --user --name pyrrm --display-name "Python (pyrrm)"
```

### Dependencies

| Category | Packages | Purpose |
|----------|----------|---------|
| **Core** | numpy, pandas, scipy, matplotlib | Essential functionality |
| **Visualization** | seaborn, plotly | Enhanced and interactive plotting |
| **Calibration** | pydream | MT-DREAM(ZS) advanced MCMC |
| **JAX/Bayesian** | jax, jaxlib, numpyro, arviz | GPU acceleration, NUTS sampler, MCMC diagnostics |
| **Sensitivity** | SALib | Sobol analysis |
| **Notebooks** | jupytext | Paired notebook workflow |

> **Note**: pyrrm works with only core dependencies installed. Optional packages enable advanced calibration and analysis features. Install with `pip install -e ".[full]"` for all features.

---

## Quick Start

### Running a Model

```python
import pandas as pd
from pyrrm.models import GR4J

# Create a GR4J model with parameters
model = GR4J({
    'X1': 350,   # Production store capacity (mm)
    'X2': 0,     # Groundwater exchange coefficient (mm/d)
    'X3': 90,    # Routing store capacity (mm)
    'X4': 1.7    # Unit hydrograph time base (days)
})

# Load input data (must have 'precipitation' and 'pet' columns)
inputs = pd.read_csv('catchment_data.csv', index_col='date', parse_dates=True)

# Run simulation
results = model.run(inputs, warmup_period=365)

# Access outputs
streamflow = results['flow']
print(f"Mean flow: {streamflow.mean():.2f} mm/d")
```

### Calibrating with DREAM

```python
from pyrrm.models import GR4J
from pyrrm.calibration import CalibrationRunner
from pyrrm.objectives import KGE  # New objectives module

# Prepare data
inputs = pd.read_csv('catchment_data.csv', index_col='date', parse_dates=True)
observed = inputs['observed_flow'].values

# Set up calibration
model = GR4J()
runner = CalibrationRunner(
    model=model,
    inputs=inputs,
    observed=observed,
    objective=KGE(),  # Use KGE objective
    warmup_period=365
)

# Run Bayesian calibration
result = runner.run_dream(n_iterations=10000, n_chains=5)

# Examine results
print(f"Best KGE: {result.best_objective:.4f}")
print(f"Best parameters: {result.best_parameters}")
```

### Sensitivity Analysis

```python
from pyrrm.analysis import SobolSensitivityAnalysis
from pyrrm.objectives import NSE

# Configure analysis
analysis = SobolSensitivityAnalysis(
    model=model,
    inputs=inputs,
    observed=observed,
    objective=NSE()
)

# Run Sobol method (requires SALib)
sensitivity = analysis.run(n_samples=1024)

# View first-order and total-effect indices
print(sensitivity.summary())
```

### Adding Channel Routing

```python
from pyrrm.models import Sacramento
from pyrrm.routing import NonlinearMuskingumRouter, RoutedModel
from pyrrm.calibration import CalibrationRunner
from pyrrm.objectives import NSE

# Create rainfall-runoff model
rr_model = Sacramento()

# Create router with initial parameters
router = NonlinearMuskingumRouter(K=5.0, m=0.8, n_subreaches=3)

# Combine into routed model
model = RoutedModel(rr_model, router)

# Run simulation (routing applied automatically)
results = model.run(inputs)

# Calibrate with routing - parameters are automatically combined
runner = CalibrationRunner(model, inputs, observed, objective=NSE())
result = runner.run_differential_evolution()  # Calibrates both RR and routing params
```

### Bayesian Calibration with NumPyro NUTS

```python
from pyrrm.models import GR4J
from pyrrm.calibration import CalibrationRunner
from pyrrm.objectives import KGE

model = GR4J()
runner = CalibrationRunner(model, inputs, observed, objective=KGE(), warmup_period=365)

# Run NUTS sampler (requires JAX + NumPyro)
result = runner.run_nuts(num_warmup=500, num_samples=1000, num_chains=4)

print(f"Best KGE: {result.best_objective:.4f}")
print(f"Best parameters: {result.best_parameters}")
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
    plot_scatter_with_metrics
)

# Create publication-ready hydrograph
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
└─────────────────────────────┬───────────────────────────────┘
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

# APEX with dynamics multiplier (penalizes gradient mismatch)
apex = APEX(
    alpha=0.1,              # Weight for chronological term
    dynamics_strength=0.5,  # κ: strength of dynamics penalty
    regime_emphasis='uniform'  # Flow regime weighting
)

# Use in calibration
runner = CalibrationRunner(model, inputs, observed, objective=apex)
```

APEX formula: `APEX = [α × E_chron + (1-α) × E_ranked] × BiasMultiplier × DynamicsMultiplier × [LagMultiplier]`

Novel contributions:
- **Dynamics Multiplier**: Penalizes mismatch in gradient/rate-of-change patterns
- **Lag Multiplier** (optional): Penalizes systematic timing offsets
- **Regime-weighted ranked term**: Continuous flow-regime weighting

**Flow Transformations:**

Apply transformations to shift emphasis between high and low flows:

```python
from pyrrm.objectives import KGE, FlowTransformation

# Standard KGE (high-flow emphasis)
kge = KGE()

# Log-transformed KGE (low-flow emphasis)
kge_log = KGE(transform=FlowTransformation('log'))

# Inverse-transformed KGE (strong low-flow emphasis)
kge_inv = KGE(transform=FlowTransformation('inverse'))
```

Available transformations: `none`, `sqrt`, `log`, `inverse`, `squared`, `power`, `boxcox`

### Multi-Objective Calibration

```python
from pyrrm.objectives import NSE, KGE, FlowTransformation, WeightedObjective

# Manual weighted combination
combined = WeightedObjective([
    (KGE(), 0.5),  # High flows
    (KGE(transform=FlowTransformation('inverse')), 0.5),  # Low flows
])

# Or use factory functions for common combinations
from pyrrm.objectives import kge_hilo, comprehensive_objective

# Balanced high/low flow objective
objective = kge_hilo(kge_weight=0.5)

# Comprehensive multi-metric objective
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

### Research & Education
The clean API and comprehensive documentation make pyrrm an excellent platform for teaching rainfall-runoff concepts and conducting academic research.

---

## Project Structure

```
pyrrm/
├── parallel.py                # Parallel backends (Sequential, Multiprocessing, Ray)
├── models/                    # Rainfall-runoff model implementations
│   ├── base.py                # BaseRainfallRunoffModel abstract class
│   ├── sacramento.py          # Sacramento Soil Moisture Accounting
│   ├── sacramento_jax.py      # Sacramento JAX implementation (GPU-accelerated)
│   ├── gr4j.py                # GR4J model
│   ├── gr4j_jax.py            # GR4J JAX implementation (GPU-accelerated)
│   ├── gr5j.py                # GR5J model
│   ├── gr6j.py                # GR6J model
│   └── utils/                 # Shared utilities
│       ├── unit_hydrograph.py # Unit hydrograph convolution
│       ├── s_curves.py        # S-curve interpolation
│       └── s_curves_jax.py    # S-curves JAX implementation
├── routing/                   # Channel routing methods
│   ├── base.py                # BaseRouter abstract class
│   ├── muskingum.py           # NonlinearMuskingumRouter
│   └── routed_model.py        # RoutedModel wrapper for RR + routing
├── objectives/                # Comprehensive objective functions library
│   ├── core/                  # Base classes and utilities
│   │   ├── base.py            # ObjectiveFunction abstract class
│   │   ├── result.py          # MetricResult container
│   │   └── utils.py           # evaluate_all, print_evaluation_report
│   ├── metrics/               # Traditional and KGE metrics
│   │   ├── traditional.py     # NSE, RMSE, MAE, PBIAS, SDEB
│   │   ├── kge.py             # KGE (2009/2012/2021), KGENonParametric
│   │   ├── apex.py            # APEX adaptive process-explicit objective
│   │   └── correlation.py     # Pearson, Spearman correlation
│   ├── transformations/       # Flow transformations
│   │   └── flow_transforms.py # sqrt, log, inverse, power, boxcox
│   ├── fdc/                   # Flow Duration Curve metrics
│   │   ├── curves.py          # compute_fdc
│   │   └── metrics.py         # FDCMetric
│   ├── signatures/            # Hydrological signatures
│   │   ├── flow_indices.py    # SignatureMetric
│   │   ├── dynamics.py        # Dynamic flow characteristics
│   │   └── water_balance.py   # Water balance signatures
│   └── composite/             # Multi-objective functions
│       ├── weighted.py        # WeightedObjective
│       ├── factories.py       # kge_hilo, comprehensive_objective
│       └── adaptive.py        # Adaptive composite objectives
├── calibration/               # Calibration framework
│   ├── runner.py              # Unified CalibrationRunner interface
│   ├── report.py              # CalibrationReport for saving/loading results
│   ├── pydream_adapter.py     # PyDREAM MT-DREAM(ZS) adapter
│   ├── numpyro_adapter.py     # NumPyro NUTS sampler adapter
│   ├── scipy_adapter.py       # SciPy optimization adapter
│   ├── sceua_adapter.py       # Direct SCE-UA implementation
│   ├── batch.py               # BatchExperimentRunner and ExperimentGrid
│   ├── checkpoint.py          # CheckpointManager for resumable calibration
│   ├── tvp_priors.py          # Time-varying parameter priors (GaussianRandomWalk)
│   ├── likelihoods_jax.py     # JAX-based likelihood functions
│   └── objective_functions.py # Legacy compatibility layer
├── network/                   # Multi-catchment network calibration
│   ├── topology.py            # CatchmentNetwork DAG representation
│   ├── data.py                # NetworkDataLoader for multi-catchment data
│   ├── runner.py              # CatchmentNetworkRunner (upstream-to-downstream)
│   └── visualization.py       # Network visualization (Mermaid, plots)
├── analysis/                  # Post-calibration analysis
│   ├── sensitivity.py         # Sobol global sensitivity analysis
│   ├── diagnostics.py         # Model performance diagnostics
│   └── mcmc_diagnostics.py    # MCMC convergence diagnostics (ArviZ)
├── visualization/             # Plotting functions
│   ├── model_plots.py         # Hydrographs, FDC, scatter plots
│   ├── calibration_plots.py   # Parameter traces, posteriors
│   ├── report_plots.py        # Report cards (Matplotlib & Plotly)
│   ├── sensitivity_plots.py   # Sobol indices visualization
│   └── mcmc_plots.py          # NUTS-specific traces, rank plots, uncertainty bands
├── data/                      # Data handling utilities
│   ├── input_handler.py       # Input data loading and validation
│   └── parameter_bounds.py    # Parameter bounds definitions
└── examples/                  # Example scripts
    ├── quick_start.py         # Getting started example
    └── mpi_calibration_example.py  # MPI parallelization example
```

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

See [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md) for detailed verification results and methodology.

---

## Documentation

### Tutorial Notebooks

The `notebooks/` directory contains a comprehensive series of educational notebooks that guide you through pyrrm's capabilities:

| # | Notebook | Description | Best For |
|---|----------|-------------|----------|
| 01 | [sacramento_verification](notebooks/01_sacramento_verification.py) | Verification against C# and SOURCE benchmarks | Implementation correctness |
| 02 | [calibration_quickstart](notebooks/02_calibration_quickstart.py) | **Start here!** Loading data and calibrating models | **New users** |
| 03 | [routing_quickstart](notebooks/03_routing_quickstart.py) | Channel routing with Nonlinear Muskingum | Adding routing to models |
| 04 | [objective_functions](notebooks/04_objective_functions.py) | Deep dive into objective functions (WIP) | Understanding what to optimize |
| 05 | [apex_complete_guide](notebooks/05_apex_complete_guide.py) | APEX objective function research & evaluation | Advanced calibration |
| 06 | [algorithm_comparison](notebooks/06_algorithm_comparison.py) | PyDREAM, SCE-UA, SciPy comparison | Choosing algorithms |
| 07 | [model_comparison](notebooks/07_model_comparison.py) | GR4J, GR5J, GR6J vs Sacramento (13 objectives) | Choosing models |
| 08 | [calibration_monitor](notebooks/08_calibration_monitor.py) | Real-time monitoring of MCMC calibrations | Long-running jobs |
| 09 | [calibration_reports](notebooks/09_calibration_reports.py) | Working with saved CalibrationReport objects | Post-processing |
| 10 | [executive_summary](notebooks/10_executive_summary.py) | Q&A guide from 65+ calibration experiments | Experimental insights |
| 11 | [batch_runners](notebooks/11_batch_runners.py) | Batch experiment runner tutorial | Production workflows |
| 12 | [network_runners](notebooks/12_network_runners.py) | Catchment network calibration | Multi-catchment systems |
| 13 | [nuts_calibration](notebooks/13_nuts_calibration.py) | Bayesian MCMC with NumPyro NUTS | Bayesian inference |
| 14 | [tvp_gr4j](notebooks/14_tvp_gr4j.py) | Time-varying parameter calibration with GR4J | Non-stationary catchments |

#### Learning Paths

**Quick Start (1-2 hours):**
1. [02_calibration_quickstart](notebooks/02_calibration_quickstart.py) - Get a working calibration

**Complete Understanding (6-8 hours):**
1. [02_calibration_quickstart](notebooks/02_calibration_quickstart.py) - Fundamentals
2. [04_objective_functions](notebooks/04_objective_functions.py) - What to optimize
3. [06_algorithm_comparison](notebooks/06_algorithm_comparison.py) - How to optimize
4. [07_model_comparison](notebooks/07_model_comparison.py) - Which model to use
5. [03_routing_quickstart](notebooks/03_routing_quickstart.py) - Adding channel routing
6. [08_calibration_monitor](notebooks/08_calibration_monitor.py) - Monitor long runs

**Bayesian Inference (4-6 hours):**
1. [02_calibration_quickstart](notebooks/02_calibration_quickstart.py) - Calibration fundamentals
2. [06_algorithm_comparison](notebooks/06_algorithm_comparison.py) - Compare MCMC vs optimization
3. [13_nuts_calibration](notebooks/13_nuts_calibration.py) - NumPyro NUTS sampler
4. [14_tvp_gr4j](notebooks/14_tvp_gr4j.py) - Time-varying parameters

**Production Workflows (3-4 hours):**
1. [02_calibration_quickstart](notebooks/02_calibration_quickstart.py) - Single catchment calibration
2. [11_batch_runners](notebooks/11_batch_runners.py) - Grid-based batch experiments
3. [12_network_runners](notebooks/12_network_runners.py) - Multi-catchment network calibration

**Model Selection:**
- [07_model_comparison](notebooks/07_model_comparison.py) - Compare GR4J, GR5J, GR6J, Sacramento across 13 objectives

**Advanced Calibration:**
- [05_apex_complete_guide](notebooks/05_apex_complete_guide.py) - Research-focused APEX evaluation with 6 research questions

**Experimental Analysis:**
- [10_executive_summary](notebooks/10_executive_summary.py) - Insights from 65+ calibration experiments on Queanbeyan River

**Verification/Validation:**
- [01_sacramento_verification](notebooks/01_sacramento_verification.py) - Implementation correctness

### API Reference

Detailed API documentation is available in the source code docstrings, following Google-style conventions.

```python
# Access built-in help
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
# Clone and install in development mode
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
- See [AGENTS.md](AGENTS.md) for detailed development guidelines

---

## License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

Developed by the **Water Information Services** team at the **ACT Government Office of Water** for water resources management and research applications in the Australian Capital Territory and beyond.

---

<p align="center">
  <sub>Built with Python and a passion for hydrology</sub>
</p>
