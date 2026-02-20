# APEX: Adaptive Process-Explicit Objective Function

**Complete Guide to Understanding and Using APEX**

---

## Table of Contents

1. [What is APEX?](#what-is-apex)
2. [Why APEX?](#why-apex)
3. [How APEX Works](#how-apex-works)
4. [Core Metric Flexibility](#core-metric-flexibility)
5. [Practical Usage](#practical-usage)
6. [Configuration Guide](#configuration-guide)
7. [Technical Reference](#technical-reference)
8. [FAQ](#faq)

---

## What is APEX?

**APEX** (Adaptive Process-Explicit) is a state-of-the-art composite objective function designed for hydrological model calibration. It combines multiple performance metrics to achieve comprehensive, robust calibration that captures:

- ✅ **Overall model fit** (high and low flows)
- ✅ **Flow distribution** (via multi-segment FDC)
- ✅ **Hydrological realism** (process signatures)
- ✅ **Volume conservation** (bias control)
- ✅ **Timing accuracy** (temporal correlation)

### Quick Summary

```
APEX = 40% Overall Fit + 30% FDC Matching + 20% Process Signatures + 10% Bias/Timing
```

**Key Innovation**: APEX is the first objective function to combine statistical performance metrics (KGE/NSE) with explicit flow distribution matching (FDC segments) and process-based hydrological signatures in a single, balanced framework.

---

## Why APEX?

### The Problem with Traditional Approaches

Traditional calibration often uses single metrics like NSE or KGE, which have limitations:

| Metric | Strength | Weakness |
|--------|----------|----------|
| **NSE** | Overall fit | Biased to high flows, ignores timing |
| **KGE** | Balanced components | Still emphasizes peaks, no process realism |
| **log-NSE** | Low flow focus | Poor high flow performance |
| **SDEB** | FDC + chronology | Single FDC metric, complex chronological term |

### APEX's Solution

APEX addresses these limitations by:

1. **Multi-scale evaluation**: Captures performance across all flow regimes
2. **Explicit FDC segmentation**: Targets specific flow percentiles (high, mid, low)
3. **Process constraints**: Ensures hydrologically realistic solutions
4. **Modular design**: Enables diagnostic analysis of calibration performance
5. **Flexible configuration**: Adaptable to different catchments and objectives

### Research Context

APEX was developed to potentially outperform **SDEB** (Sum of Daily Flows, Daily Exceedance Curve and Bias; Lerat et al., 2013), which is considered state-of-the-art in Australian operational hydrology. Initial testing on catchment 410734 shows promising results.

---

## How APEX Works

### The Formula

APEX combines 9 objective components with normalized weights:

$$\text{APEX} = \sum_{i=1}^{9} w_i \cdot f_i(Q_{obs}, Q_{sim})$$

where $w_i$ are weights that sum to 1.0, and $f_i$ are normalized objective functions (0 = worst, 1 = best).

### The 9 Components

APEX evaluates model performance across four categories:

#### 1. Core Performance (40% total weight)

Measures overall model fit using two complementary metrics:

| Component | Default Weight | Purpose |
|-----------|----------------|---------|
| **Core Metric 1** | 25% | Primary performance measure (high flow emphasis) |
| **Core Metric 2** | 15% | Transformed performance (balanced flows) |

**Default**: KGE (standard) + KGE(√Q)
**Flexible**: Can use KGE or NSE with various transformations (see [Flexibility](#core-metric-flexibility))

#### 2. Flow Duration Curve (30% total weight)

Explicitly evaluates flow distribution across three segments:

| Component | Default Weight | FDC Range | Target Flows |
|-----------|----------------|-----------|--------------|
| **FDC High** | 10% | 2nd-20th percentile | Floods and high flows |
| **FDC Mid** | 10% | 20th-70th percentile | Typical conditions |
| **FDC Low** | 10% | 70th-95th percentile | Baseflow (log-transformed) |

**Why segments?** Ensures the model matches the observed flow distribution at peaks, mid-range, and low flows independently.

#### 3. Hydrological Signatures (20% total weight)

Enforces process-based hydrological realism:

| Component | Default Weight | Calculation | Process |
|-----------|----------------|-------------|---------|
| **Baseflow Index** | 10% | $\frac{\text{Baseflow}}{\text{Total Flow}}$ | Groundwater contribution |
| **Flashiness Index** | 10% | $\frac{\sum \|Q_{t+1} - Q_t\|}{\sum Q_t}$ | Response dynamics |

**Why signatures?** Prevents equifinality—multiple parameter sets giving similar fit but different hydrological behavior.

#### 4. Bias and Timing (10% total weight)

Controls volume errors and temporal synchronization:

| Component | Default Weight | Calculation | Purpose |
|-----------|----------------|-------------|---------|
| **PBIAS** | 5% | Percent bias | Volume conservation |
| **Pearson r** | 5% | Linear correlation | Timing accuracy |

### Visual Breakdown

```
APEX Composition
┌─────────────────────────────────────────────────────────┐
│ Core Performance (40%)                                   │
│ ├─ Core Metric 1 [25%]  ███████████████████████         │
│ └─ Core Metric 2 [15%]  █████████████                   │
├─────────────────────────────────────────────────────────┤
│ Flow Duration Curve (30%)                               │
│ ├─ FDC High [10%]       ████████                        │
│ ├─ FDC Mid [10%]        ████████                        │
│ └─ FDC Low [10%]        ████████                        │
├─────────────────────────────────────────────────────────┤
│ Process Signatures (20%)                                │
│ ├─ Baseflow Index [10%] ████████                        │
│ └─ Flashiness [10%]     ████████                        │
├─────────────────────────────────────────────────────────┤
│ Bias & Timing (10%)                                     │
│ ├─ PBIAS [5%]          ████                             │
│ └─ Pearson r [5%]      ████                             │
└─────────────────────────────────────────────────────────┘
```

---

## Core Metric Flexibility

**NEW FEATURE**: APEX's core performance component (40% weight) is fully configurable.

### Understanding Core Metrics

APEX uses **two core metrics** to capture overall performance:

1. **Core Metric 1** (25%): Primary performance measure
2. **Core Metric 2** (15%): Same metric with transformation for flow regime balance

**Key Insight**: Using the same metric with different transformations ensures performance across flow regimes without over-constraining the calibration.

### Metric Type: KGE vs NSE

You can choose between two base metrics:

#### Option A: KGE (Kling-Gupta Efficiency) — DEFAULT

$$\text{KGE} = 1 - \sqrt{(r-1)^2 + (\beta-1)^2 + (\gamma-1)^2}$$

where:
- $r$ = correlation (timing)
- $\beta$ = bias ratio (mean error)
- $\gamma$ = variability ratio (std dev error)

**Advantages**:
- ✅ Decomposes performance into interpretable components
- ✅ Modern standard (Pool et al., 2018; Knoben et al., 2019)
- ✅ Less sensitive to extreme values than NSE
- ✅ Better balance of bias, variability, and correlation

**When to use**: General-purpose calibration, modern applications

#### Option B: NSE (Nash-Sutcliffe Efficiency)

$$\text{NSE} = 1 - \frac{\sum(Q_{obs} - Q_{sim})^2}{\sum(Q_{obs} - \bar{Q}_{obs})^2}$$

**Advantages**:
- ✅ Traditional standard (Nash & Sutcliffe, 1970)
- ✅ Directly comparable to extensive literature
- ✅ Simple interpretation (0 = mean model, 1 = perfect)

**When to use**: Replicating traditional studies, comparison with historical calibrations

### Flow Transformations

Apply transformations to emphasize different flow regimes:

| Transform | Formula | Effect | Visual Weight | Use Case |
|-----------|---------|--------|---------------|----------|
| **None** | $Q$ | High flow emphasis | 🌊🌊🌊 | Flood forecasting |
| **sqrt** | $\sqrt{Q}$ | Balanced | 🌊🌊💧 | General purpose |
| **log** | $\log(Q + \epsilon)$ | Low flow emphasis | 🌊💧💧 | Water supply |
| **inverse** | $\frac{1}{Q + \epsilon}$ | Strong low flow | 💧💧💧 | Environmental flows |

**Intuition**: 
- High flows (100 mm/day) vs low flows (1 mm/day) = 100:1 ratio
- With sqrt: (10 vs 1) = 10:1 ratio → more balanced
- With log: (2 vs 0) → equal attention
- With inverse: (0.01 vs 1) = 1:100 ratio → low flows dominate

### Configuration Examples

#### Default Configuration (Recommended)

```python
from pyrrm.objectives import apex_objective

apex = apex_objective()
# Equivalent to:
# apex = apex_objective(
#     core_metric_type='kge',
#     core_metric_1_transform=None,     # KGE on original flows
#     core_metric_2_transform='sqrt'    # KGE(√Q) for balance
# )
```

**Use for**: General-purpose calibration, no specific flow regime emphasis

#### Traditional NSE Configuration

```python
apex_nse = apex_objective(
    core_metric_type='nse',
    core_metric_1_transform=None,      # Standard NSE
    core_metric_2_transform='log'      # log-NSE (common in literature)
)
```

**Use for**: 
- Replicating traditional calibration studies
- Comparison with NSE-based literature
- Organizational standards requiring NSE

#### Low Flow Focus

```python
apex_lowflow = apex_objective(
    core_metric_type='kge',
    core_metric_1_transform='sqrt',     # Balanced KGE
    core_metric_2_transform='inverse',  # Strong low flow emphasis
    
    # Optionally adjust weights
    core_metric_1_weight=0.20,         # Reduce slightly
    core_metric_2_weight=0.20,         # Increase for low flows
    fdc_low_weight=0.15,               # Emphasize low FDC
    baseflow_index_weight=0.15,        # Emphasize baseflow
    flashiness_weight=0.05             # De-emphasize flashiness
)
```

**Use for**:
- Water supply modeling
- Environmental flow requirements
- Low flow ecology studies
- Drought management

#### High Flow / Flood Focus

```python
apex_flood = apex_objective(
    core_metric_type='kge',
    core_metric_1_transform=None,      # Standard KGE (high flow bias)
    core_metric_2_transform='sqrt',    # Balanced KGE
    
    # Adjust weights for flood focus
    fdc_high_weight=0.15,              # Increase high FDC
    flashiness_weight=0.15,            # Increase response dynamics
    baseflow_index_weight=0.05         # De-emphasize baseflow
)
```

**Use for**:
- Flood forecasting systems
- Peak flow analysis
- Flashy catchments
- Dam safety assessments

---

## Practical Usage

### Installation

Ensure you have pyrrm installed with calibration dependencies:

```bash
conda activate pyrrm
pip install -e /path/to/ACT-Rainfall-Runoff-Modelling
```

### Basic Calibration Workflow

```python
import numpy as np
import pandas as pd
from pyrrm.models import Sacramento
from pyrrm.calibration import CalibrationRunner
from pyrrm.objectives import apex_objective

# 1. Load your data
inputs = pd.read_csv('inputs.csv', parse_dates=['Date'], index_col='Date')
# Requires columns: 'precipitation', 'pet', 'observed_flow'

# 2. Create model
model = Sacramento(catchment_area_km2=490.0)

# 3. Create APEX objective (default configuration)
apex = apex_objective()

# 4. Set up calibration
runner = CalibrationRunner(
    model=model,
    inputs=inputs,
    observed_flow=inputs['observed_flow'].values,
    objective_function=apex,
    warmup_period=365
)

# 5. Run calibration (SCE-UA algorithm)
n_params = len(model.get_parameter_bounds())
ngs = 2 * n_params + 1  # Duan et al., 1994 recommendation

result = runner.run_sceua_direct(
    max_iterations=10000,
    ngs=ngs,
    kstop=5,
    pcento=0.01
)

# 6. Get best parameters
best_params = result['best_parameters']
print(f"Best APEX value: {result['best_objective']:.4f}")

# 7. Run model with calibrated parameters
model.set_parameters(best_params)
output = model.run(inputs)

# 8. Evaluate components
components = apex.evaluate_individual(
    inputs['observed_flow'].values[365:],
    output['flow'].values[365:]
)

print("\nAPEX Component Breakdown:")
for name, value in zip(components['names'], components['raw_values']):
    print(f"  {name:25s}: {value:7.4f}")
```

### Adaptive Weighting

APEX can automatically adapt weights based on flow regime characteristics:

```python
from pyrrm.objectives import apex_adaptive

# Analyze observed flow and create adapted APEX
obs_flow = inputs['observed_flow'].values
apex_adapted = apex_adaptive(
    Q_obs=obs_flow,
    verbose=True  # Print adaptation details
)

# Use in calibration as normal
runner = CalibrationRunner(
    model=model,
    inputs=inputs,
    observed_flow=obs_flow,
    objective_function=apex_adapted,
    warmup_period=365
)

result = runner.run_sceua_direct(max_iterations=10000, ngs=ngs)
```

**Adaptive weighting**:
- Analyzes observed flow statistics (CV, flashiness, baseflow index)
- Automatically adjusts component weights
- Useful for diverse catchment applications

### Complete Example with Custom Configuration

```python
from pyrrm.models import Sacramento
from pyrrm.calibration import CalibrationRunner
from pyrrm.objectives import apex_objective
import pandas as pd

# Load data
data_dir = 'data/410734'
inputs = pd.read_csv(f'{data_dir}/inputs.csv', 
                     parse_dates=['Date'], index_col='Date')

# Create model
model = Sacramento(catchment_area_km2=490.0)

# Load custom parameter bounds (optional)
custom_bounds = {}
with open(f'{data_dir}/sacramento_bounds_custom.txt', 'r') as f:
    for line in f:
        if line.strip() and not line.startswith('#'):
            parts = line.split()
            if len(parts) == 3:
                custom_bounds[parts[0]] = (float(parts[1]), float(parts[2]))

model.parameter_bounds = custom_bounds

# Create APEX with low flow focus
apex = apex_objective(
    core_metric_type='kge',
    core_metric_1_transform='sqrt',
    core_metric_2_transform='inverse',
    core_metric_2_weight=0.20,
    fdc_low_weight=0.15,
    baseflow_index_weight=0.15
)

# Calibrate
obs_flow = inputs['observed_flow'].values
runner = CalibrationRunner(
    model=model,
    inputs=inputs,
    observed_flow=obs_flow,
    objective_function=apex,
    warmup_period=365
)

# SCE-UA with appropriate ngs
n_params = len(model.get_parameter_bounds())
ngs = 2 * n_params + 1

print(f"Calibrating with {n_params} parameters, ngs={ngs}")
result = runner.run_sceua_direct(
    max_iterations=10000,
    ngs=ngs,
    kstop=5,
    pcento=0.01
)

# Analyze results
print(f"\nBest APEX: {result['best_objective']:.4f}")
print(f"Runtime: {result['runtime_seconds']:.1f} seconds")

# Component breakdown
model.set_parameters(result['best_parameters'])
output = model.run(inputs)

components = apex.evaluate_individual(
    obs_flow[365:],
    output['flow'].values[365:]
)

print("\nComponent Analysis:")
for i, (name, value, weight) in enumerate(zip(
    components['names'], 
    components['raw_values'],
    components['weights']
)):
    weighted = value * weight
    print(f"{i+1:2d}. {name:25s}: {value:6.4f} × {weight:5.3f} = {weighted:6.4f}")

print(f"\nTotal APEX: {components['composite']:.4f}")
```

---

## Configuration Guide

### Decision Framework

Use this flowchart to choose your APEX configuration:

```
START: What is your primary objective?
│
├─ General-purpose calibration?
│  └─> Use default: apex_objective()
│      (KGE + KGE(sqrt))
│
├─ Low flow emphasis?
│  ├─ Moderate emphasis?
│  │  └─> apex_objective(
│  │       core_metric_1_transform='sqrt',
│  │       core_metric_2_transform='log')
│  │
│  └─ Strong emphasis?
│     └─> apex_objective(
│          core_metric_1_transform='sqrt',
│          core_metric_2_transform='inverse',
│          core_metric_2_weight=0.20,
│          fdc_low_weight=0.15)
│
├─ High flow / flood emphasis?
│  └─> apex_objective(
│       core_metric_1_transform=None,
│       core_metric_2_transform='sqrt',
│       fdc_high_weight=0.15,
│       flashiness_weight=0.15)
│
├─ Traditional NSE approach?
│  └─> apex_objective(
│       core_metric_type='nse',
│       core_metric_1_transform=None,
│       core_metric_2_transform='log')
│
└─ Multiple catchments with different regimes?
   └─> Use apex_adaptive(Q_obs)
       (automatic adaptation)
```

### Hyperparameter Tuning

For optimal performance on your specific catchment, consider tuning:

#### Level 1: Core Metric Configuration (Start Here)

```python
# Test different metric types and transformations
configs = {
    'KGE_default': apex_objective(core_metric_type='kge'),
    'NSE_log': apex_objective(core_metric_type='nse', 
                              core_metric_2_transform='log'),
    'KGE_inverse': apex_objective(core_metric_type='kge',
                                  core_metric_2_transform='inverse')
}

# Calibrate with each and compare validation performance
```

#### Level 2: Component Weights

```python
# Adjust weights based on calibration diagnostics
apex_tuned = apex_objective(
    # If low flows are poor, increase:
    fdc_low_weight=0.15,          # from 0.10
    baseflow_index_weight=0.15,   # from 0.10
    
    # Compensate by reducing:
    fdc_high_weight=0.08,         # from 0.10
    flashiness_weight=0.07        # from 0.10
)
```

#### Level 3: Sensitivity Analysis

```python
from SALib.sample import saltelli
from SALib.analyze import sobol

# Define parameter space for APEX weights
problem = {
    'num_vars': 9,
    'names': ['core_1', 'core_2', 'fdc_high', 'fdc_mid', 'fdc_low',
              'baseflow', 'flashiness', 'pbias', 'timing'],
    'bounds': [
        [0.15, 0.35],  # core_1
        [0.10, 0.25],  # core_2
        [0.05, 0.15],  # fdc_high
        # ... etc
    ]
}

# Generate samples (ensure weights sum to 1.0 with post-processing)
param_values = saltelli.sample(problem, 512)

# Run calibrations and analyze sensitivity
# See notebooks/09_apex_sensitivity_analysis.py for full example
```

---

## Technical Reference

### API Reference

#### `apex_objective()`

Create an APEX objective function with configurable weights and metrics.

**Signature**:
```python
def apex_objective(
    # Core metrics (40% total)
    core_metric_1_weight: float = 0.25,
    core_metric_2_weight: float = 0.15,
    
    # FDC segments (30% total)
    fdc_high_weight: float = 0.10,
    fdc_mid_weight: float = 0.10,
    fdc_low_weight: float = 0.10,
    
    # Hydrological signatures (20% total)
    baseflow_index_weight: float = 0.10,
    flashiness_weight: float = 0.10,
    
    # Bias and timing (10% total)
    pbias_weight: float = 0.05,
    timing_correlation_weight: float = 0.05,
    
    # Configuration
    core_metric_type: str = 'kge',
    core_metric_1_transform: Optional[str] = None,
    core_metric_2_transform: str = 'sqrt',
    kge_variant: str = '2012'
) -> WeightedObjective:
```

**Parameters**:
- `core_metric_1_weight`: Weight for primary core metric (default: 0.25)
- `core_metric_2_weight`: Weight for transformed core metric (default: 0.15)
- `fdc_high_weight`: Weight for high flow FDC segment (default: 0.10)
- `fdc_mid_weight`: Weight for mid flow FDC segment (default: 0.10)
- `fdc_low_weight`: Weight for low flow FDC segment (default: 0.10)
- `baseflow_index_weight`: Weight for baseflow index signature (default: 0.10)
- `flashiness_weight`: Weight for flashiness index signature (default: 0.10)
- `pbias_weight`: Weight for percent bias (default: 0.05)
- `timing_correlation_weight`: Weight for Pearson correlation (default: 0.05)
- `core_metric_type`: Base metric type, 'kge' or 'nse' (default: 'kge')
- `core_metric_1_transform`: Transform for core metric 1 (default: None)
  - Options: None, 'sqrt', 'log', 'inverse', 'power', 'boxcox'
- `core_metric_2_transform`: Transform for core metric 2 (default: 'sqrt')
  - Options: None, 'sqrt', 'log', 'inverse', 'power', 'boxcox'
- `kge_variant`: KGE variant when using KGE (default: '2012')
  - Options: '2009', '2012', 'np' (non-parametric)

**Returns**: 
- `WeightedObjective`: Callable objective function

**Note**: Weights are automatically normalized to sum to 1.0

#### `apex_adaptive()`

Create an APEX objective with weights adapted to flow regime characteristics.

**Signature**:
```python
def apex_adaptive(
    Q_obs: np.ndarray,
    kge_variant: str = '2012',
    window: int = 5,
    verbose: bool = True,
    core_metric_type: str = 'kge',
    core_metric_1_transform: Optional[str] = None,
    core_metric_2_transform: str = 'sqrt'
) -> WeightedObjective:
```

**Parameters**:
- `Q_obs`: Observed flow time series (numpy array)
- `kge_variant`: KGE variant for core metrics (default: '2012')
- `window`: Rolling window for flashiness calculation (default: 5)
- `verbose`: Print adaptation details (default: True)
- `core_metric_type`: Base metric type (default: 'kge')
- `core_metric_1_transform`: Transform for core metric 1 (default: None)
- `core_metric_2_transform`: Transform for core metric 2 (default: 'sqrt')

**Returns**:
- `WeightedObjective`: APEX with adapted weights

**Adaptation Rules**:
- High CV (>1.5): Increase FDC high weight, reduce FDC low
- Low CV (<0.8): Increase FDC low weight, reduce FDC high
- High flashiness (>0.8): Increase flashiness weight
- High baseflow index (>0.7): Increase baseflow weight

### File Locations

```
pyrrm/objectives/
├── composite/
│   ├── factories.py           ⭐ apex_objective() implementation
│   ├── adaptive.py            ⭐ apex_adaptive() implementation
│   └── weighted.py            WeightedObjective class
├── metrics/
│   ├── traditional.py         NSE, KGE, RMSE, MAE, PBIAS
│   ├── fdc.py                 FDCMetric, FDCSegmentMetric
│   └── signatures.py          SignatureMetric (baseflow, flashiness)
└── tests/
    └── test_composite.py      APEX unit tests
```

### Performance Considerations

**Computational Cost**:
- APEX evaluation: ~2-3× slower than single NSE
- Dominated by FDC sorting operations
- Negligible compared to model runtime

**Typical Calibration Times** (Sacramento model, 10,000 iterations):
- Single core: 15-30 minutes
- Speedup primarily from model, not objective

**Recommendations**:
- Use warmup period (≥365 days for Sacramento)
- Ensure adequate iteration budget (10,000+ for Sacramento)
- Set ngs = 2×n_params + 1 for SCE-UA

---

## FAQ

### General Questions

**Q: When should I use APEX instead of NSE or KGE?**

A: Use APEX when you need:
- Comprehensive calibration across all flow regimes
- Explicit FDC matching (important for Australian hydrology)
- Hydrologically realistic parameter sets
- Diagnostic information on calibration performance

Use single metrics (NSE, KGE) when:
- Rapid prototyping or testing
- Specific single-criterion optimization
- Computational resources are very limited

---

**Q: How does APEX compare to SDEB?**

A: APEX improves on SDEB by:
- ✅ Explicit multi-segment FDC (vs. single ranked error)
- ✅ Process-based signatures (ensures hydrological realism)
- ✅ Separate timing metric (clearer than chronological term)
- ✅ Flexible core metrics (KGE or NSE with transforms)
- ✅ Modular design (component diagnostics)

Initial testing shows competitive or better performance on catchment 410734.

---

**Q: Is APEX only for the Sacramento model?**

A: No! APEX works with **any** rainfall-runoff model:
- Sacramento, GR4J, GR5J, GR6J (built into pyrrm)
- Any model implementing `BaseRainfallRunoffModel`
- Any external model (pass simulated flows to APEX)

---

**Q: Can I use APEX for validation?**

A: Yes! APEX provides comprehensive evaluation:
```python
# Calibration period
apex_cal = apex_objective()
# ... calibrate model ...

# Validation period
val_score = apex_cal(val_obs, val_sim)
components = apex_cal.evaluate_individual(val_obs, val_sim)

# Analyze which components perform well/poorly in validation
```

---

### Configuration Questions

**Q: Should I use KGE or NSE?**

A: **Default to KGE** unless:
- You need direct comparison with NSE-based literature
- Your organization requires NSE
- You're replicating a traditional study

KGE is generally preferred because:
- Better decomposition (bias, variability, correlation)
- Less sensitive to extremes
- Modern standard in hydrology

---

**Q: How do I choose transformations?**

A: Match to your application priority:

| Application | Core 1 Transform | Core 2 Transform |
|-------------|------------------|------------------|
| **General purpose** | None | sqrt |
| **Water supply** | sqrt | log or inverse |
| **Flood forecasting** | None | sqrt or None |
| **Environmental flows** | sqrt | inverse |
| **Traditional NSE** | None | log |

---

**Q: Can I use different metric types for each core metric?**

A: No, both must be the same type (both KGE or both NSE). This ensures:
- Consistent performance measurement
- Interpretable component contributions
- Numerical stability

You can use different **transformations** for each core metric.

---

**Q: How do I know if my configuration is working?**

A: Check:
1. **Convergence**: SCE-UA should converge (pcento criterion)
2. **Component balance**: No single component dominates
3. **Validation**: Test on holdout period
4. **Hydrographs**: Visual inspection of fit
5. **FDC plots**: Check distribution matching

---

### Technical Questions

**Q: Why 9 components? Can I add more?**

A: Nine components balance:
- Comprehensive evaluation
- Interpretability
- Computational efficiency

You can add components by modifying the `apex_objective()` factory function, but consider:
- More components = more tuning complexity
- Diminishing returns beyond key aspects
- Component correlation can reduce benefit

---

**Q: How were the default weights chosen?**

A: Based on hydrological priorities:
- 40% overall fit (most important)
- 30% FDC (critical for Australian hydrology)
- 20% process signatures (ensure realism)
- 10% bias/timing (important but captured by other components)

Validated through sensitivity analysis (see `notebooks/09_apex_sensitivity_analysis.py`).

---

**Q: Can APEX be used with multi-objective optimization?**

A: APEX is already multi-objective! It combines 9 objectives.

For traditional multi-objective (Pareto front):
```python
from pyrrm.objectives import NSE, FlowTransformation

# Define conflicting objectives
high_flow_obj = NSE()
low_flow_obj = NSE(transform=FlowTransformation('inverse'))

# Use multi-objective algorithm (e.g., NSGA-II)
# APEX can be one objective in the suite
```

---

**Q: How do I reproduce results with different random seeds?**

A: SCE-UA is stochastic. For reproducibility:
```python
import numpy as np

# Set seed before calibration
np.random.seed(42)

result = runner.run_sceua_direct(
    max_iterations=10000,
    ngs=ngs,
    random_seed=42  # If supported by implementation
)
```

For production: run multiple seeds and ensemble.

---

**Q: What if APEX components are in conflict?**

A: This is expected and desired! Components represent trade-offs:
- High vs. low flow performance
- Bias vs. variability
- Fit vs. process realism

APEX finds balanced solutions via weighted aggregation. If conflicts are severe:
1. Check for data issues
2. Consider if model structure is appropriate
3. Adjust weights to prioritize your application

---

**Q: Can I use APEX with Bayesian calibration?**

A: Yes! APEX can be the likelihood function:
```python
from pyrrm.objectives import apex_objective
from pydream.core import run_dream

apex = apex_objective()

def log_likelihood(params):
    model.set_parameters(params)
    output = model.run(inputs)
    apex_value = apex(obs_flow, output['flow'].values)
    # Convert to log-likelihood (assume normal errors)
    return -0.5 * len(obs_flow) * np.log(1 - apex_value)

# Run DREAM
# ... (see pyrrm DREAM examples)
```

---

### Troubleshooting

**Q: APEX returns NaN or very low values**

**A: Check**:
1. **Missing data**: APEX requires continuous data
2. **Units**: Ensure obs and sim have same units
3. **Warmup**: Use adequate warmup period (≥365 days)
4. **Model output**: Verify model runs successfully
5. **Transform issues**: log/inverse need epsilon for zero flows

```python
# Debug APEX
components = apex.evaluate_individual(obs, sim)
for name, value in zip(components['names'], components['raw_values']):
    if np.isnan(value):
        print(f"NaN in component: {name}")
```

---

**Q: Calibration is very slow**

**A**: 
1. **Reduce iterations**: Start with 1000-2000 for testing
2. **Check ngs**: Should be 2×n_params + 1
3. **Profile model**: APEX is rarely the bottleneck
4. **Use warmup strategically**: Balance spin-up vs. evaluation period

---

**Q: Results differ from SDEB significantly**

**A**: This can be expected because:
- APEX emphasizes different aspects (explicit FDC segments, signatures)
- Different applications may favor different objectives
- Check validation performance to determine which is better for your case

If results are **much worse**:
- Verify APEX configuration is appropriate
- Check for implementation errors
- Consider weight tuning
- Ensure fair comparison (same bounds, iterations, etc.)

---

## Example Notebook

### Complete APEX Guide

**`notebooks/01_apex_complete_guide.py`** (Jupytext format)

A comprehensive, all-in-one guide to APEX covering:

**Part 1: Introduction & Core Concepts**
- What APEX is and why it's better
- Understanding the 9 components
- Core metric flexibility (KGE vs NSE, transformations)
- Quick configuration demonstration

**Part 2: Comprehensive Calibration Study**
- Full calibration comparison (NSE vs SDEB vs APEX)
- Performance evaluation across 15+ metrics
- Component diagnostic analysis
- Visual comparisons (hydrographs, FDC, scatter plots)
- Parameter interpretation

**Part 3: Adaptive Weighting**
- Automatic weight adaptation based on flow regime
- Flow regime characterization
- Adaptive calibration example

**Part 4: Hyperparameter Optimization** (Optional, computationally intensive)
- Sobol sensitivity analysis on APEX weights
- Identifying most influential components
- Focused grid search optimization
- Weight tuning strategies

**Perfect for**:
- Beginners learning APEX
- Researchers conducting comprehensive studies
- Practitioners optimizing configurations
- Anyone wanting a complete APEX reference

**Run with**:
```bash
conda activate pyrrm
cd notebooks
jupyter notebook 01_apex_complete_guide.ipynb
```

**Sections are modular**: You can run Parts 1-2 for basic usage, skip Part 3 if not needed, and only run Part 4 if you need hyperparameter optimization.

---

## Citation

If you use APEX in your research, please cite:

```
APEX (Adaptive Process-Explicit) Objective Function
Developed as part of ACT Rainfall-Runoff Modelling project
Australian Capital Territory Government, 2026
```

And reference the foundational work:

- **KGE**: Gupta et al. (2009), Kling et al. (2012)
- **FDC-based calibration**: Westerberg et al. (2011), Pfannerstill et al. (2014)
- **Hydrological signatures**: McMillan et al. (2017), Pool et al. (2017)
- **SDEB**: Lerat et al. (2013)

---

## Summary

**APEX** is a comprehensive, flexible objective function that:

✅ Combines statistical fit, FDC matching, and process signatures  
✅ Supports KGE or NSE with flow transformations  
✅ Enables diagnostic component analysis  
✅ Adapts to different catchment characteristics  
✅ Provides robust, hydrologically realistic calibrations  

**Getting Started**:
1. Start with default: `apex = apex_objective()`
2. Calibrate and evaluate
3. Analyze component breakdown
4. Customize if needed for your application

**Key Files**:
- Implementation: `pyrrm/objectives/composite/factories.py`
- Example notebook: `notebooks/08_apex_development.py`
- Tests: `pyrrm/objectives/tests/test_composite.py`

**For Help**:
- Read this guide
- Run example notebook Section 1
- Check FAQ above
- Review test cases for validation

---

*Last updated: February 2026*  
*Version: 1.0*  
*Part of pyrrm library - Python Rainfall-Runoff Models*
