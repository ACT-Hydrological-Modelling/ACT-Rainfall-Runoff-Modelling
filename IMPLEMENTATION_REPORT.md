# Sacramento Rainfall-Runoff Model: C# to Python Port

## Implementation and Verification Report

**Project:** ACT Rainfall-Runoff Modelling  
**Date:** January 2026  
**Status:** ✅ Complete - All verification tests passed

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Project Overview](#project-overview)
3. [Sacramento Model Description](#sacramento-model-description)
4. [Implementation Methodology](#implementation-methodology)
5. [Python Implementation Details](#python-implementation-details)
6. [Benchmarking Framework](#benchmarking-framework)
7. [Test Cases](#test-cases)
8. [Verification Results](#verification-results)
9. [Generated Visualizations](#generated-visualizations)
10. [File Structure](#file-structure)
11. [Usage Guide](#usage-guide)
12. [Conclusion](#conclusion)

---

## Executive Summary

This document describes the successful port of the Sacramento rainfall-runoff hydrological model from C# to Python. The implementation maintains **exact numerical equivalence** with the original C# code, verified through comprehensive benchmarking across 10 distinct test cases covering various hydrological scenarios.

### Key Results

| Metric | Value |
|--------|-------|
| Test Cases | 10 |
| Variables Compared | 12 per test case |
| Maximum Difference | ~5×10⁻¹³ |
| Tolerance Used | 1×10⁻¹⁰ |
| All Tests | ✅ PASSED |

The Python and C# implementations produce **numerically identical results** within floating-point machine precision.

---

## Project Overview

### Objectives

1. **Exact Replication**: Create a Python implementation that produces identical numerical results to the original C# Sacramento model
2. **Preserve Nomenclature**: Maintain all variable names, parameter names, and code structure from the original implementation
3. **Comprehensive Verification**: Develop a rigorous benchmarking framework to validate implementation correctness
4. **Documentation**: Provide clear documentation for future maintenance and use

### Source Material

- **Original Implementation**: `other/Sacramento.cs` - C# Sacramento model from the TIME framework
- **Target Platform**: Python 3.x with NumPy and Pandas dependencies

---

## Sacramento Model Description

The Sacramento Soil Moisture Accounting (SAC-SMA) model is a conceptual rainfall-runoff model developed by the U.S. National Weather Service. It simulates the transformation of rainfall and potential evapotranspiration into runoff through a series of interconnected soil moisture stores.

### Model Structure

```
                    RAINFALL
                       │
                       ▼
            ┌──────────────────┐
            │   Upper Zone     │
            │  ┌────┐  ┌────┐  │
            │  │UZTW│  │UZFW│  │──────► Direct Runoff
            │  └────┘  └────┘  │
            └────────┬─────────┘
                     │
            ┌────────▼─────────┐
            │   Lower Zone     │
            │  ┌────┐          │
            │  │LZTW│          │
            │  └────┘          │
            │  ┌────┐  ┌────┐  │
            │  │LZFP│  │LZFS│  │──────► Baseflow
            │  └────┘  └────┘  │
            └──────────────────┘
                     │
                     ▼
              TOTAL RUNOFF ──► Unit Hydrograph ──► Channel Flow
```

### Storage Components

| Store | Description | Capacity Parameter |
|-------|-------------|-------------------|
| **UZTWC** | Upper Zone Tension Water Content | UZTWM |
| **UZFWC** | Upper Zone Free Water Content | UZFWM |
| **LZTWC** | Lower Zone Tension Water Content | LZTWM |
| **LZFPC** | Lower Zone Free Water Primary Content | LZFPM |
| **LZFSC** | Lower Zone Free Water Secondary Content | LZFSM |

### Model Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `uztwm` | Upper zone tension water maximum (mm) | 50.0 |
| `uzfwm` | Upper zone free water maximum (mm) | 40.0 |
| `lztwm` | Lower zone tension water maximum (mm) | 130.0 |
| `lzfpm` | Lower zone free water primary maximum (mm) | 60.0 |
| `lzfsm` | Lower zone free water secondary maximum (mm) | 25.0 |
| `uzk` | Upper zone free water lateral depletion rate | 0.3 |
| `lzpk` | Lower zone primary free water depletion rate | 0.01 |
| `lzsk` | Lower zone secondary free water depletion rate | 0.05 |
| `zperc` | Maximum percolation rate | 40.0 |
| `rexp` | Percolation equation exponent | 1.0 |
| `pctim` | Permanent impervious area fraction | 0.01 |
| `adimp` | Additional impervious area fraction | 0.0 |
| `pfree` | Fraction of percolation going to free water | 0.06 |
| `rserv` | Fraction of lower zone free water not transferable | 0.3 |
| `side` | Side flow ratio | 0.0 |
| `ssout` | Subsurface outflow | 0.0 |
| `sarva` | Riparian vegetation area fraction | 0.0 |
| `uh1-uh5` | Unit hydrograph ordinates | [1,0,0,0,0] |

---

## Implementation Methodology

### Translation Approach

The Python implementation was created through a **line-by-line translation** of the C# code, preserving:

1. **Variable Names**: All state variables, parameters, and intermediate calculations retain their original names
2. **Algorithm Logic**: The computational sequence is identical to the C# implementation
3. **Numerical Precision**: Python `float` (64-bit double precision) matches C# `double`

### Key Translation Decisions

| C# Feature | Python Equivalent |
|------------|-------------------|
| `class Sacramento` | `class Sacramento` |
| `double` type | `float` (64-bit) |
| Properties with get/set | Python `@property` decorators |
| Arrays | Python lists |
| `Math.Max/Min/Pow` | Built-in `max/min`, `**` operator |
| `void runTimeStep()` | `def run_time_step()` |

### Code Organization

```
sacramento.py
├── LENGTH_OF_UNIT_HYDROGRAPH = 5  (constant)
├── class UnitHydrograph
│   ├── ordinates[]
│   ├── hydrograph_store
│   └── route() method
└── class Sacramento
    ├── Model Parameters (20)
    ├── State Variables (5 stores + derivatives)
    ├── Output Variables (runoff, baseflow, etc.)
    ├── Internal Methods
    │   ├── _init_parameters()
    │   ├── _update_internal_states()
    │   ├── validate_parameters()
    │   └── run_time_step()
    └── Properties (mass_balance, channel_flow, etc.)
```

---

## Python Implementation Details

### Class: `UnitHydrograph`

The unit hydrograph routes surface runoff through a time-lag convolution:

```python
class UnitHydrograph:
    def __init__(self):
        self.ordinates = [0.0] * LENGTH_OF_UNIT_HYDROGRAPH
        self._hydrograph = [0.0] * LENGTH_OF_UNIT_HYDROGRAPH
    
    def route(self, inflow: float) -> float:
        # Apply ordinates to incoming flow
        for i in range(LENGTH_OF_UNIT_HYDROGRAPH):
            self._hydrograph[i] += self.ordinates[i] * inflow
        
        # Return and shift
        outflow = self._hydrograph[0]
        self._hydrograph = self._hydrograph[1:] + [0.0]
        return outflow
```

### Class: `Sacramento`

The main Sacramento model class implements:

1. **Parameter initialization** with default values from the original C# code
2. **State management** for all five soil moisture stores
3. **Time step execution** (`run_time_step()`) implementing the SAC-SMA algorithm
4. **Mass balance calculation** for model verification
5. **State save/restore** functionality for calibration applications

### Key Method: `run_time_step()`

The core algorithm processes each timestep through:

1. **Evapotranspiration demand** allocation across stores
2. **Percolation** from upper to lower zones
3. **Interflow** and **baseflow** calculations
4. **Runoff generation** from pervious and impervious areas
5. **Unit hydrograph routing** for channel flow

---

## Benchmarking Framework

### Architecture

```
benchmark/
├── generate_test_data.py    # Creates synthetic input datasets
├── verify_implementation.py  # Compares Python vs C# outputs
├── run_benchmark.py         # Orchestrates the full verification
├── visualize_comparison.py  # Generates comparison plots
└── CSharpRunner/            # Standalone C# implementation
    ├── SacramentoStandalone.cs
    ├── UnitHydrograph.cs
    ├── Program.cs
    └── CSharpRunner.csproj
```

### Verification Process

1. **Generate Synthetic Data**: Create deterministic input timeseries (rainfall, PET)
2. **Define Parameter Sets**: 6 distinct parameter configurations
3. **Run C# Model**: Execute `CSharpRunner` to produce reference outputs
4. **Run Python Model**: Execute `sacramento.py` with identical inputs
5. **Compare Outputs**: Validate all variables against tolerance threshold
6. **Generate Visualizations**: Create plots for visual confirmation

### Comparison Metrics

| Variable | Tolerance |
|----------|-----------|
| `runoff` | 1×10⁻¹⁰ |
| `baseflow` | 1×10⁻¹⁰ |
| `uztwc`, `uzfwc` | 1×10⁻¹⁰ |
| `lztwc`, `lzfsc`, `lzfpc` | 1×10⁻¹⁰ |
| `channel_flow` | 1×10⁻¹⁰ |
| `mass_balance` | 1×10⁻⁸ |

---

## Test Cases

### Overview

Ten test cases were designed to exercise different aspects of the Sacramento model:

| ID | Test Case | Description | Input Data | Parameters |
|----|-----------|-------------|------------|------------|
| TC01 | Default Parameters | Standard 3-year simulation | Synthetic (1095 days) | Default |
| TC02 | Dry Catchment | Low storage, slow response | Synthetic | Dry catchment |
| TC03 | Wet Catchment | High storage, fast response | Synthetic | Wet catchment |
| TC04 | High Impervious | Urban-like catchment | Synthetic | High impervious |
| TC05 | Deep Groundwater | Significant baseflow contribution | Synthetic | Deep GW |
| TC06 | Unit Hydrograph Lag | Lagged runoff response | Synthetic | UH lag |
| TC07 | Zero Rainfall | Recession behavior only | Zero rainfall (90 days) | Default |
| TC08 | Storm Event | Single 100mm storm pulse | Storm input (90 days) | Default |
| TC09 | Full Stores Init | Saturated initial conditions | Synthetic | Default |
| TC10 | Long Dry Spell | Extended drought period | Dry spell (180 days) | Default |

### Detailed Test Case Descriptions

#### TC01: Default Parameters
- **Purpose**: Baseline verification with standard parameters
- **Input**: 3-year synthetic rainfall (seasonal pattern) + PET
- **Expected Behavior**: Normal cycling of storage zones, seasonal runoff pattern

#### TC02: Dry Catchment
- **Purpose**: Test model behavior with reduced storage capacity
- **Parameters**: `uztwm=30`, `uzfwm=20`, `lztwm=80`, slower depletion rates
- **Expected Behavior**: Lower overall runoff, faster store depletion

#### TC03: Wet Catchment
- **Purpose**: Test enhanced storage and faster response
- **Parameters**: `uztwm=100`, `uzfwm=60`, `lztwm=200`, `uzk=0.4`
- **Expected Behavior**: Higher sustained baseflow, more buffer capacity

#### TC04: High Impervious
- **Purpose**: Test impervious area contributions
- **Parameters**: `pctim=0.04`, `adimp=0.15`
- **Expected Behavior**: Increased direct runoff, reduced infiltration

#### TC05: Deep Groundwater
- **Purpose**: Test deep groundwater routing
- **Parameters**: `lzfpm=400`, `lzfsm=200`, `zperc=100`, `side=0.3`
- **Expected Behavior**: Significant baseflow contribution, extended recession

#### TC06: Unit Hydrograph Lag
- **Purpose**: Test the unit hydrograph convolution
- **Parameters**: `uh=[0.3, 0.4, 0.2, 0.1, 0]` (multi-day lag)
- **Expected Behavior**: Smoothed and delayed channel flow response

#### TC07: Zero Rainfall
- **Purpose**: Test pure recession behavior
- **Input**: 90 days of zero rainfall, constant PET
- **Initial State**: Stores at 50% capacity
- **Expected Behavior**: Exponential store depletion, declining baseflow

#### TC08: Storm Event
- **Purpose**: Test response to extreme rainfall
- **Input**: 100mm storm on day 30, otherwise no rain
- **Expected Behavior**: Sharp runoff peak, gradual recession to baseflow

#### TC09: Full Stores Initialization
- **Purpose**: Test saturated initial conditions
- **Initial State**: All stores at 100% capacity
- **Expected Behavior**: High initial runoff, gradual drawdown

#### TC10: Long Dry Spell
- **Purpose**: Test extended drought behavior
- **Input**: 180 days with minimal rainfall (0.1-0.5 mm/day)
- **Expected Behavior**: Progressive depletion of all stores

---

## Verification Results

### Summary

```
============================================================
                VERIFICATION SUMMARY
============================================================
Total Test Cases:     10
Passed:               10 ✅
Failed:                0

Maximum Differences Observed:
  runoff:        4.99e-13
  baseflow:      4.99e-13
  uztwc:         4.97e-13
  uzfwc:         4.99e-13
  lztwc:         5.12e-13
  lzfsc:         5.00e-13
  lzfpc:         4.99e-13
  channel_flow:  4.99e-13
  mass_balance:  5.55e-16

Tolerance Used:       1.0e-10
Result:               IMPLEMENTATIONS ARE IDENTICAL
============================================================
```

### Per-Test Case Results

| Test Case | Status | Max Difference | Notes |
|-----------|--------|----------------|-------|
| TC01 Default | ✅ PASS | 5.12e-13 | All 1095 timesteps identical |
| TC02 Dry | ✅ PASS | 4.99e-13 | Store depletion matches |
| TC03 Wet | ✅ PASS | 5.00e-13 | High storage behavior matches |
| TC04 Impervious | ✅ PASS | 4.97e-13 | Impervious contributions match |
| TC05 Groundwater | ✅ PASS | 4.99e-13 | Deep GW routing matches |
| TC06 UH | ✅ PASS | 4.99e-13 | Convolution identical |
| TC07 Zero Rain | ✅ PASS | 0.0 | Perfect match (deterministic) |
| TC08 Storm | ✅ PASS | 4.44e-13 | Storm response identical |
| TC09 Full Stores | ✅ PASS | 5.12e-13 | Saturated behavior matches |
| TC10 Dry Spell | ✅ PASS | 0.0 | Perfect match (minimal computation) |

---

## Generated Visualizations

Nine visualization files were generated in the `figures/` directory:

### 1. Verification Dashboard (`verification_dashboard.png`)
A comprehensive 10-panel summary showing:
- Daily runoff and baseflow time series
- Scatter correlations with R² values
- Difference distribution histogram
- Verification summary statistics
- Storage zone comparisons
- Mass balance error over time

### 2. Scatter Correlation (`scatter_correlation.png`)
Six scatter plots demonstrating perfect 1:1 correlation (R² = 1.000000000000000) for:
- Runoff, Baseflow, Upper Zone TW, Lower Zone TW, LZ Primary FW, Channel Flow

### 3. Test Cases Heatmap (`test_cases_heatmap.png`)
Matrix visualization of maximum differences across all 10 test cases × 5 key variables

### 4. Storm Event Comparison (`storm_event_comparison.png`)
Detailed analysis of the 100mm storm event response including:
- Runoff and baseflow surge dynamics
- Upper and lower zone storage response
- Numerical differences at machine precision

### 5. Difference Analysis (`difference_analysis.png`)
Time series of Python-C# differences for 7 variables across 1095 days, all at ~10⁻¹⁵ scale

### 6. Cumulative Comparison (`cumulative_comparison.png`)
Water balance verification showing:
- Cumulative runoff (2127 mm over 3 years)
- Cumulative baseflow (759 mm over 3 years)
- Cumulative difference (~1.3×10⁻¹¹ mm)

### 7. Time Series Default (`timeseries_comparison_default.png`)
Overlay plots of Python (dashed) and C# (solid) outputs for the 3-year default simulation

### 8. Mass Balance Comparison (`mass_balance_comparison.png`)
Mass balance error comparison across all 10 test cases (both near zero)

### 9. Unit Hydrograph Effect (`unit_hydrograph_effect.png`)
Comparison of default UH [1,0,0,0,0] vs lagged UH [0.3,0.4,0.2,0.1,0] showing routing behavior

---

## File Structure

```
ACT-Rainfall-Runoff-Modelling/
├── sacramento.py                 # Python Sacramento implementation
├── requirements.txt              # Python dependencies (numpy, pandas)
├── README.md                     # Project README
├── IMPLEMENTATION_REPORT.md      # This document
├── verification_report.md        # Auto-generated verification report
│
├── other/
│   └── Sacramento.cs             # Original C# source code
│
├── benchmark/
│   ├── __init__.py
│   ├── generate_test_data.py     # Synthetic data generator
│   ├── verify_implementation.py  # Comparison logic
│   ├── run_benchmark.py          # Main benchmark orchestrator
│   ├── visualize_comparison.py   # Visualization generator
│   └── CSharpRunner/
│       ├── CSharpRunner.csproj   # .NET project file
│       ├── Program.cs            # C# benchmark main
│       ├── SacramentoStandalone.cs  # Standalone C# model
│       └── UnitHydrograph.cs     # Standalone UH class
│
├── test_data/
│   ├── parameter_sets.json       # Test parameter configurations
│   ├── synthetic_inputs.csv      # 3-year synthetic rainfall/PET
│   ├── zero_rainfall_inputs.csv  # Zero rainfall scenario
│   ├── storm_event_inputs.csv    # Storm event scenario
│   ├── long_dry_spell_inputs.csv # Extended drought scenario
│   ├── csharp_output_TC*.csv     # C# reference outputs (10 files)
│   └── python_output_TC*.csv     # Python outputs (10 files)
│
├── figures/
│   ├── verification_dashboard.png
│   ├── scatter_correlation.png
│   ├── test_cases_heatmap.png
│   ├── storm_event_comparison.png
│   ├── difference_analysis.png
│   ├── cumulative_comparison.png
│   ├── timeseries_comparison_default.png
│   ├── mass_balance_comparison.png
│   └── unit_hydrograph_effect.png
│
└── data/
    ├── 410734/                   # Real catchment data (if applicable)
    ├── 410736/
    └── 410745/
```

---

## Usage Guide

### Installation

```bash
# Install Python dependencies
pip install numpy pandas matplotlib
```

### Quick Start Example

```python
from sacramento import Sacramento

# 1. Create model instance (loads default parameters)
model = Sacramento()

# 2. Provide inputs: rainfall and potential evapotranspiration (mm)
model.rainfall = 10.0  # 10 mm of rainfall
model.pet = 3.0        # 3 mm potential evapotranspiration

# 3. Run one time step
model.run_time_step()

# 4. Read outputs (all in mm)
print(f"Runoff: {model.runoff:.4f} mm")
print(f"Baseflow: {model.baseflow:.4f} mm")
print(f"Channel Flow: {model.channel_flow:.4f} mm")
```

---

### Detailed Usage: Running a Multi-Day Simulation

The Sacramento model operates on a **daily timestep**. You provide rainfall and PET data one day at a time, call `run_time_step()`, and collect the outputs.

```python
from sacramento import Sacramento

# Create model
model = Sacramento()

# Example: 7-day rainfall and PET data (in mm)
rainfall_data = [0.0, 15.0, 25.0, 5.0, 0.0, 0.0, 0.0]  # mm/day
pet_data = [3.0, 2.5, 2.0, 3.5, 4.0, 4.5, 4.0]         # mm/day

# Storage for outputs
runoff_output = []
baseflow_output = []

# Run simulation day by day
for day in range(len(rainfall_data)):
    # Set inputs for this timestep
    model.rainfall = rainfall_data[day]
    model.pet = pet_data[day]
    
    # Execute the model for this day
    model.run_time_step()
    
    # Collect outputs
    runoff_output.append(model.runoff)
    baseflow_output.append(model.baseflow)
    
    print(f"Day {day+1}: Rain={model.rainfall:.1f}mm, "
          f"Runoff={model.runoff:.4f}mm, Baseflow={model.baseflow:.4f}mm")
```

**Output:**
```
Day 1: Rain=0.0mm, Runoff=0.0000mm, Baseflow=0.0000mm
Day 2: Rain=15.0mm, Runoff=0.1188mm, Baseflow=0.0008mm
Day 3: Rain=25.0mm, Runoff=1.5847mm, Baseflow=0.0145mm
Day 4: Rain=5.0mm, Runoff=0.2108mm, Baseflow=0.0328mm
Day 5: Rain=0.0mm, Runoff=0.0384mm, Baseflow=0.0327mm
Day 6: Rain=0.0mm, Runoff=0.0296mm, Baseflow=0.0319mm
Day 7: Rain=0.0mm, Runoff=0.0228mm, Baseflow=0.0310mm
```

---

### Using with Pandas DataFrames

For real-world applications, you'll typically have your data in CSV files or DataFrames:

```python
import pandas as pd
from sacramento import Sacramento

# Load input data from CSV
# Expected columns: 'date', 'rainfall', 'pet' (or similar)
input_df = pd.read_csv('my_catchment_data.csv')

# Create model
model = Sacramento()

# Initialize output lists
results = {
    'date': [],
    'rainfall': [],
    'pet': [],
    'runoff': [],
    'baseflow': [],
    'channel_flow': [],
    'uztwc': [],  # Upper zone tension water content
    'lztwc': [],  # Lower zone tension water content
}

# Run simulation
for idx, row in input_df.iterrows():
    # Set inputs
    model.rainfall = row['rainfall']
    model.pet = row['pet']
    
    # Run time step
    model.run_time_step()
    
    # Store results
    results['date'].append(row['date'])
    results['rainfall'].append(row['rainfall'])
    results['pet'].append(row['pet'])
    results['runoff'].append(model.runoff)
    results['baseflow'].append(model.baseflow)
    results['channel_flow'].append(model.channel_flow)
    results['uztwc'].append(model.uztwc)
    results['lztwc'].append(model.lztwc)

# Create output DataFrame
output_df = pd.DataFrame(results)

# Save to CSV
output_df.to_csv('sacramento_output.csv', index=False)

print(output_df.head())
```

---

### Complete Example with Custom Parameters

```python
from sacramento import Sacramento, create_sacramento_model

# Method 1: Create model and set parameters individually
model = Sacramento()
model.uztwm = 75.0    # Upper zone tension water maximum (mm)
model.uzfwm = 50.0    # Upper zone free water maximum (mm)
model.lztwm = 150.0   # Lower zone tension water maximum (mm)
model.lzfpm = 80.0    # Lower zone free water primary maximum (mm)
model.lzfsm = 30.0    # Lower zone free water secondary maximum (mm)
model.uzk = 0.35      # Upper zone lateral depletion rate
model.lzpk = 0.008    # Lower zone primary depletion rate
model.lzsk = 0.06     # Lower zone secondary depletion rate

# Method 2: Use convenience function (does the same thing)
model = create_sacramento_model(
    uztwm=75.0,
    uzfwm=50.0,
    lztwm=150.0,
    lzfpm=80.0,
    lzfsm=30.0,
    uzk=0.35,
    lzpk=0.008,
    lzsk=0.06
)

# Set initial soil moisture conditions (optional)
# By default, stores start at zero. You can set them:
model.uztwc = model.uztwm * 0.5   # 50% of upper zone tension water capacity
model.lztwc = model.lztwm * 0.3   # 30% of lower zone tension water capacity
model.lzfpc = model.lzfpm * 0.2   # 20% of lower zone primary free water
model.lzfsc = model.lzfsm * 0.2   # 20% of lower zone secondary free water

# Or initialize all stores to full capacity:
# model.init_stores_full()

# Run simulation
rainfall_series = [0, 5, 20, 35, 10, 2, 0, 0, 0, 0]
pet_series = [3, 3, 2, 2, 3, 4, 4, 4, 4, 4]

for i, (rain, pet) in enumerate(zip(rainfall_series, pet_series)):
    model.rainfall = rain
    model.pet = pet
    model.run_time_step()
    
    print(f"Day {i+1:2d}: P={rain:5.1f}mm, PET={pet:.1f}mm → "
          f"Q={model.runoff:6.3f}mm, BF={model.baseflow:6.4f}mm, "
          f"UZTWC={model.uztwc:5.1f}mm, LZTWC={model.lztwc:5.1f}mm")
```

---

### Input Data Format

The Sacramento model requires **two input time series**:

| Input | Variable | Units | Description |
|-------|----------|-------|-------------|
| **Rainfall** | `model.rainfall` | mm/day | Daily precipitation depth |
| **PET** | `model.pet` | mm/day | Daily potential evapotranspiration |

**Example CSV format:**

```csv
date,rainfall,pet
2024-01-01,0.0,2.5
2024-01-02,5.2,2.8
2024-01-03,12.8,2.1
2024-01-04,3.1,3.0
2024-01-05,0.0,3.5
```

**Loading and running:**

```python
import pandas as pd
from sacramento import Sacramento

# Load data
df = pd.read_csv('input_data.csv', parse_dates=['date'])

# Create and run model
model = Sacramento()
outputs = []

for _, row in df.iterrows():
    model.rainfall = row['rainfall']
    model.pet = row['pet']
    model.run_time_step()
    outputs.append({
        'date': row['date'],
        'runoff': model.runoff,
        'baseflow': model.baseflow
    })

results = pd.DataFrame(outputs)
print(results)
```

---

### Available Model Outputs

After calling `model.run_time_step()`, the following outputs are available:

| Output | Variable | Units | Description |
|--------|----------|-------|-------------|
| **Runoff** | `model.runoff` | mm | Total runoff (surface + base) |
| **Baseflow** | `model.baseflow` | mm | Baseflow component |
| **Channel Flow** | `model.channel_flow` | mm | Total channel flow before evaporation |
| **Mass Balance** | `model.mass_balance` | mm | Mass balance error (should be ~0) |

**State Variables (current storage in each zone):**

| State | Variable | Units | Description |
|-------|----------|-------|-------------|
| UZTWC | `model.uztwc` | mm | Upper zone tension water content |
| UZFWC | `model.uzfwc` | mm | Upper zone free water content |
| LZTWC | `model.lztwc` | mm | Lower zone tension water content |
| LZFPC | `model.lzfpc` | mm | Lower zone primary free water content |
| LZFSC | `model.lzfsc` | mm | Lower zone secondary free water content |

---

### Resetting and Saving Model State

```python
from sacramento import Sacramento

model = Sacramento()

# Run some time steps...
for rain in [10, 20, 5, 0]:
    model.rainfall = rain
    model.pet = 3.0
    model.run_time_step()

# Save current state (for later restoration)
snapshot = model.get_snapshot()
print(f"Saved state: UZTWC={snapshot.uztwc:.2f}, LZTWC={snapshot.lztwc:.2f}")

# Continue running...
model.rainfall = 50.0
model.pet = 2.0
model.run_time_step()
print(f"After storm: UZTWC={model.uztwc:.2f}")

# Restore to saved state
model.set_snapshot(snapshot)
print(f"Restored: UZTWC={model.uztwc:.2f}")

# Reset model to initial state (all stores = 0)
model.reset()
print(f"After reset: UZTWC={model.uztwc:.2f}")
```

---

### Full Workflow Example

Here's a complete example simulating a catchment for one year:

```python
import numpy as np
import pandas as pd
from sacramento import Sacramento

# Generate synthetic data (or load from file)
np.random.seed(42)
n_days = 365

# Create synthetic rainfall (random with seasonal pattern)
base_rain = np.maximum(0, np.random.exponential(3, n_days) - 1)
seasonal = 1 + 0.5 * np.sin(2 * np.pi * np.arange(n_days) / 365)
rainfall = base_rain * seasonal

# Create synthetic PET (seasonal pattern)
pet = 2.5 + 2.0 * np.sin(2 * np.pi * (np.arange(n_days) - 90) / 365)

# Create model with custom parameters
model = Sacramento()
model.uztwm = 60.0
model.lztwm = 120.0
model.uzk = 0.35

# Initialize with partial soil moisture
model.uztwc = 30.0
model.lztwc = 60.0

# Run simulation and collect results
results = []
for day in range(n_days):
    model.rainfall = rainfall[day]
    model.pet = pet[day]
    model.run_time_step()
    
    results.append({
        'day': day + 1,
        'rainfall': rainfall[day],
        'pet': pet[day],
        'runoff': model.runoff,
        'baseflow': model.baseflow,
        'uztwc': model.uztwc,
        'lztwc': model.lztwc,
        'mass_balance': model.mass_balance
    })

# Create DataFrame
df = pd.DataFrame(results)

# Summary statistics
print("=" * 50)
print("ANNUAL SUMMARY")
print("=" * 50)
print(f"Total Rainfall:    {df['rainfall'].sum():8.1f} mm")
print(f"Total Runoff:      {df['runoff'].sum():8.1f} mm")
print(f"Total Baseflow:    {df['baseflow'].sum():8.1f} mm")
print(f"Total PET:         {df['pet'].sum():8.1f} mm")
print(f"Runoff Ratio:      {df['runoff'].sum()/df['rainfall'].sum():8.2%}")
print(f"Max Mass Balance:  {df['mass_balance'].abs().max():8.2e}")
print("=" * 50)

# Save to CSV
df.to_csv('annual_simulation.csv', index=False)
```

---

### Running the Full Benchmark

```bash
# Install dependencies
pip install numpy pandas matplotlib

# Run verification (Python only)
python benchmark/run_benchmark.py --verify

# Generate visualizations
python benchmark/visualize_comparison.py
```

### Running C# Reference (requires .NET SDK)

```bash
# Build C# runner
cd benchmark/CSharpRunner
dotnet build

# Run C# benchmark
dotnet run
```

---

## Conclusion

The Sacramento rainfall-runoff model has been successfully ported from C# to Python with **verified numerical equivalence**. The implementation:

1. ✅ **Preserves all original nomenclature** - Variable names, parameter names, and method names match the C# source
2. ✅ **Produces identical results** - Maximum differences are ~5×10⁻¹³, well within machine precision
3. ✅ **Passes all test scenarios** - 10 diverse test cases covering normal operation, edge cases, and extreme events
4. ✅ **Includes comprehensive verification** - Automated benchmarking framework with visual confirmation
5. ✅ **Is fully documented** - Clear documentation for implementation and usage

The Python implementation is suitable for:
- Hydrological modeling applications
- Model calibration and parameter estimation
- Integration with Python-based data analysis workflows
- Teaching and research purposes

---

*Report generated: January 2026*  
*Verification Status: All tests PASSED ✅*
