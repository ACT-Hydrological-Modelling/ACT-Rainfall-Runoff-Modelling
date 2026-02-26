# Data Preparation Guide for pyrrm

This guide explains how to prepare input data for the **pyrrm** rainfall-runoff
modelling library. It covers single-catchment, batch, and multi-catchment
(network) workflows.

---

## What pyrrm Needs

Every pyrrm simulation requires three daily time series aligned on a common
`DatetimeIndex`:

| Variable | Canonical column name | Units | Description |
|---|---|---|---|
| Precipitation | `precipitation` | mm/day | Gridded or station rainfall averaged over the catchment |
| Potential evapotranspiration | `pet` | mm/day | PET estimate (e.g., Morton's wet-environment) |
| Observed streamflow | `observed_flow` | ML/day *or* mm/day | Recorded gauging station flow for calibration |

> **Note on flow units:** Sacramento accepts ML/day (set `catchment_area_km2`
> on the model to auto-convert). GR4J/GR5J/GR6J operate in mm/day — convert
> using `pyrrm.data.ml_per_day_to_mm()` before calibration.

---

## Canonical Column Names and Aliases

pyrrm uses a single source of truth for column-name lookups
(`pyrrm.data.COLUMN_ALIASES`). You are free to use **any** of the accepted
aliases — the library resolves them automatically.

| Canonical name | Accepted aliases |
|---|---|
| `precipitation` | `precipitation`, `rainfall`, `precip`, `rain`, `Precipitation`, `Rainfall`, `P` |
| `pet` | `pet`, `evapotranspiration`, `evap`, `PET`, `ET`, `Evapotranspiration` |
| `observed_flow` | `observed_flow`, `flow`, `runoff`, `discharge`, `streamflow`, `Q`, `observed` |
| `date` | `date`, `Date`, `datetime`, `Datetime`, `time`, `timestamp` |

The **recommended** column names are `precipitation`, `pet`, and
`observed_flow`. Using these avoids any ambiguity.

---

## CSV File Format

Each CSV file should have:

1. A **date column** (one of the date aliases above).
2. One or more **value columns**.

Example — `rainfall.csv`:

```
Date,rainfall
1985-01-01,0.0
1985-01-02,5.4
1985-01-03,12.1
...
```

### Supported date formats

`pd.to_datetime()` is used internally, so most unambiguous date strings work:

- `YYYY-MM-DD` (recommended)
- `DD/MM/YYYY`
- `MM/DD/YYYY` (use with caution — ambiguous)
- `YYYY-MM-DD HH:MM:SS`

### Missing values

The `load_catchment_data()` function replaces sentinel values (default
`-9999`) with `NaN` and drops any rows that still contain `NaN` in the
observed flow column. Precipitation and PET `NaN` rows are removed via the
inner join.

---

## Single-Catchment Workflow

### Recommended: `load_catchment_data()`

This convenience function replaces the 30+ lines of boilerplate previously
needed to load, merge, clean, and align three CSV files.

```python
from pathlib import Path
from pyrrm.data import load_catchment_data

DATA_DIR = Path('data/410734')

inputs, observed = load_catchment_data(
    precipitation_file=DATA_DIR / 'Rain.csv',
    pet_file=DATA_DIR / 'PET.csv',
    observed_file=DATA_DIR / 'Flow.csv',
    start_date='2000-01-01',   # optional — trim to period
    end_date='2024-12-31',     # optional
)
```

**Returns:**

| Name | Type | Description |
|---|---|---|
| `inputs` | `pd.DataFrame` | DatetimeIndex, columns `'precipitation'` and `'pet'` |
| `observed` | `np.ndarray` | 1-D array of observed flow values |

### If your observed flow column has a non-standard name

Pass it explicitly with `observed_value_column`:

```python
inputs, observed = load_catchment_data(
    precipitation_file=DATA_DIR / 'Rain.csv',
    pet_file=DATA_DIR / 'PET.csv',
    observed_file=DATA_DIR / 'Flow.csv',
    observed_value_column='Gauge: 410734: Recorded Gauging Station Flow (ML.day^-1)',
)
```

### Passing data to a CalibrationRunner

```python
from pyrrm.models import Sacramento
from pyrrm.calibration import CalibrationRunner, ObjectiveFunction

runner = CalibrationRunner(
    model=Sacramento(catchment_area_km2=516.63),
    inputs=inputs,
    observed=observed,
    objective=ObjectiveFunction('nse'),
    warmup_period=365,
)
result = runner.run_sceua(max_iterations=10_000)
```

### Alternative: `InputDataHandler`

If you already have a single pre-merged DataFrame:

```python
from pyrrm.data import InputDataHandler
from pyrrm.models import GR4J

model = GR4J({'X1': 350, 'X2': 0, 'X3': 90, 'X4': 1.7})
handler = InputDataHandler(model, merged_df)
precip, pet = handler.to_arrays()
```

`InputDataHandler` auto-detects and standardises column names using the same
alias table.

---

## Batch Workflow (ExperimentGrid / ExperimentList)

In a batch workflow the **data is fixed** — you load it once and let the
experiment definition vary models, objectives, and algorithms.

```python
from pyrrm.data import load_catchment_data
from pyrrm.calibration import (
    CalibrationRunner, ObjectiveFunction,
    ExperimentGrid, BatchExperimentRunner,
)
from pyrrm.models import Sacramento, GR4J

# 1. Load data (same as single-catchment)
inputs, observed = load_catchment_data(
    precipitation_file='data/Rain.csv',
    pet_file='data/PET.csv',
    observed_file='data/Flow.csv',
)

# 2. Define experiment grid
grid = ExperimentGrid(
    models=[
        Sacramento(catchment_area_km2=516.63),
        GR4J(catchment_area_km2=516.63),
    ],
    objectives=[
        ObjectiveFunction('nse'),
        ObjectiveFunction('kge'),
    ],
    algorithms=['sceua_direct'],
)

# 3. Run all combinations
batch_runner = BatchExperimentRunner(
    inputs=inputs,
    observed=observed,
    warmup_period=365,
)
batch_result = batch_runner.run_grid(grid)

# 4. Export results
batch_result.export('exports/', format='excel')
```

For an `ExperimentList` the pattern is identical — just replace `ExperimentGrid`
with a list of explicit `Experiment` objects.

---

## Network (Multi-Catchment) Workflow

Network calibration requires a **topology CSV** that describes how catchments
are connected, and per-catchment data files.

### Topology CSV

```
catchment_id,downstream_id,model,area_km2,precip_file,pet_file,observed_file
Upper,Junction,GR4J,120.0,upper_rain.csv,upper_pet.csv,upper_flow.csv
Lower,Junction,Sacramento,250.0,lower_rain.csv,lower_pet.csv,lower_flow.csv
Junction,,Sacramento,370.0,junction_rain.csv,junction_pet.csv,junction_flow.csv
```

Column names in each per-catchment CSV are resolved using the same alias table.

### Loading

```python
from pyrrm.network.data import NetworkDataLoader

loader = NetworkDataLoader('topology.csv', data_dir='./data/')
network_data = loader.load()
report = loader.validate()
print(report)
```

See the Network Runners notebook (Notebook 12) for a complete worked example.

---

## Unit Conversions

pyrrm provides four conversion functions in `pyrrm.data`:

| Function | Conversion |
|---|---|
| `mm_to_ml_per_day(depth_mm, area_km2)` | Depth (mm/day) → Volume (ML/day) |
| `ml_per_day_to_mm(flow_ml, area_km2)` | Volume (ML/day) → Depth (mm/day) |
| `cumecs_to_ml_per_day(flow_cumecs)` | m³/s → ML/day (factor: 86.4) |
| `ml_per_day_to_cumecs(flow_ml)` | ML/day → m³/s |

**Example — converting observed flow from ML/day to mm/day:**

```python
from pyrrm.data import load_catchment_data, ml_per_day_to_mm

inputs, observed_ml = load_catchment_data(...)
observed_mm = ml_per_day_to_mm(observed_ml, catchment_area_km2=516.63)
```

---

## Troubleshooting

### "Could not find a 'precipitation' column"

Your CSV does not contain any of the accepted precipitation aliases.
Rename the value column to `precipitation` or one of the aliases listed
above.

### "No date column found"

The CSV must have a column named `Date`, `date`, `datetime`, `time`, or
`timestamp`.

### Length mismatch between inputs and observed

`load_catchment_data()` uses an **inner join** on the date index, so all
three files must overlap in time. Check that your precipitation, PET, and
flow CSVs cover the same period.

### NaN values remaining after loading

Sentinel values other than `-9999` are not replaced by default. Pass them
explicitly:

```python
inputs, observed = load_catchment_data(
    ...,
    missing_values=[-9999, -99.99, 0],
)
```

### GR4J / GR5J / GR6J expects mm/day but I have ML/day

Convert observed flow before passing to the calibration runner:

```python
from pyrrm.data import ml_per_day_to_mm

observed_mm = ml_per_day_to_mm(observed, catchment_area_km2=516.63)
runner = CalibrationRunner(model=GR4J(), inputs=inputs,
                           observed=observed_mm, ...)
```

---

## Quick Reference

```python
from pyrrm.data import (
    load_catchment_data,     # Load 3 CSVs → (inputs, observed)
    COLUMN_ALIASES,          # Dict of accepted column names
    resolve_column,          # Find a column by canonical name
    InputDataHandler,        # OOP wrapper for a merged DataFrame
    mm_to_ml_per_day,        # mm/day → ML/day
    ml_per_day_to_mm,        # ML/day → mm/day
    cumecs_to_ml_per_day,    # m³/s  → ML/day
    ml_per_day_to_cumecs,    # ML/day → m³/s
)
```
