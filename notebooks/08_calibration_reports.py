# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python (pyrrm)
#     language: python
#     name: pyrrm
# ---

# %% [markdown]
# # Tutorial 08: Working with Calibration Reports
#
# This notebook demonstrates how to work with saved `CalibrationReport` objects.
# These reports contain everything needed for visualization, analysis, and re-simulation
# of calibrated rainfall-runoff models.
#
# ## Learning Objectives
#
# By the end of this tutorial, you will be able to:
#
# 1. Load and inspect saved calibration reports
# 2. Generate comprehensive report cards (Matplotlib and Plotly)
# 3. Extract metrics and parameters from saved calibrations
# 4. Compare multiple calibration reports
# 5. Re-run simulations using saved parameters

# %% [markdown]
# ---
# ## Setup

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# pyrrm imports
from pyrrm.calibration import CalibrationReport
from pyrrm.models import Sacramento

# Optional: for interactive plots
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

print("Setup complete!")

# %% [markdown]
# ---
# ## 1. Overview of Saved Calibration Reports
#
# Calibration reports from notebook 02 have been saved to `test_data/reports/`.
#
# ### Summary of Available Calibrations
#
# **NSE-based Objectives (Default Bounds):**
#
# | Report File | Objective | Flow Regime Focus |
# |-------------|-----------|-------------------|
# | `410734_nse.pkl` | NSE | High flows |
# | `410734_lognse.pkl` | LogNSE | Low flows |
# | `410734_invnse.pkl` | InvNSE | Very low flows |
# | `410734_sqrtnse.pkl` | SqrtNSE | Balanced |
# | `410734_sdeb.pkl` | SDEB | Flow duration curve |
#
# **KGE-based Objectives (Default Bounds):**
#
# | Report File | Objective | Flow Regime Focus |
# |-------------|-----------|-------------------|
# | `410734_kge.pkl` | KGE | High flows |
# | `410734_kge_inv.pkl` | KGE (1/Q) | Very low flows |
# | `410734_kge_sqrt.pkl` | KGE (√Q) | Balanced |
# | `410734_kge_log.pkl` | KGE (log Q) | Low flows |
# | `410734_kge_np.pkl` | KGE_np | Non-parametric |
#
# **Custom Bounds Calibrations:**
#
# | Report File | Objective | Notes |
# |-------------|-----------|-------|
# | `410734_sdeb_custom.pkl` | SDEB | Extended lower bounds |

# %% [markdown]
# ---
# ## 2. Loading and Listing Saved Reports

# %%
# List all saved reports
report_dir = Path('../test_data/reports')
report_files = sorted(report_dir.glob('410734_*.pkl'))

print("=" * 70)
print("SAVED CALIBRATION REPORTS")
print("=" * 70)
print(f"\nFound {len(report_files)} saved reports in {report_dir}:\n")

for report_file in report_files:
    try:
        report = CalibrationReport.load(report_file)
        print(f"  {report_file.name}: {report}")
    except Exception as e:
        print(f"  {report_file.name}: Error loading - {e}")

# %% [markdown]
# ---
# ## 3. Loading a Specific Report

# %%
# Load a specific report for detailed analysis
loaded_report = CalibrationReport.load('../test_data/reports/410734_sdeb.pkl')
print(f"Loaded report: {loaded_report}")

# %%
# Inspect report contents
print("\n" + "=" * 70)
print("REPORT CONTENTS")
print("=" * 70)

print(f"\nCatchment Info:")
for key, value in loaded_report.catchment_info.items():
    print(f"  {key}: {value}")

print(f"\nCalibration Period: {loaded_report.calibration_period[0]} to {loaded_report.calibration_period[1]}")

print(f"\nCalibration Result:")
print(f"  Method: {loaded_report.result.method}")
print(f"  Objective: {loaded_report.result.objective_name}")
print(f"  Best Objective Value: {loaded_report.result.best_objective:.4f}")

print(f"\nBest Parameters:")
for param, value in loaded_report.result.best_parameters.items():
    print(f"  {param}: {value:.4f}")

# %% [markdown]
# ---
# ## 4. Calculating Metrics from a Loaded Report

# %%
# Calculate metrics from loaded report
metrics = loaded_report.calculate_metrics()

print("=" * 70)
print("PERFORMANCE METRICS")
print("=" * 70)
print(f"\nMetrics calculated from saved observed/simulated data:\n")

for name, value in metrics.items():
    print(f"  {name}: {value:.4f}")

# %% [markdown]
# ---
# ## 5. Generating Report Cards
#
# ### 5.1 Matplotlib Report Card (Static Figure)
#
# The Matplotlib report card provides a comprehensive static figure suitable for
# publications and presentations.

# %%
# Generate a matplotlib report card (comprehensive figure)
fig = loaded_report.plot_report_card(figsize=(20, 24))
fig.savefig('../test_data/reports/410734_sdeb_report_card.png', dpi=150, bbox_inches='tight')
print("Report card saved to: ../test_data/reports/410734_sdeb_report_card.png")
plt.show()

# %% [markdown]
# ### 5.2 Plotly Report Card (Interactive HTML)
#
# The Plotly report card provides an interactive HTML dashboard for exploratory analysis.

# %%
# Generate an interactive Plotly report card (HTML)
if PLOTLY_AVAILABLE:
    fig_plotly = loaded_report.plot_report_card_interactive(height=1000)
    fig_plotly.write_html('../test_data/reports/410734_sdeb_report_card.html')
    print("Interactive report saved to: ../test_data/reports/410734_sdeb_report_card.html")
    fig_plotly.show()
else:
    print("Plotly not available - skipping interactive report")

# %% [markdown]
# ---
# ## 6. Comparing Multiple Calibration Reports
#
# Load multiple reports and compare their performance.

# %%
# Load multiple reports for comparison
reports_to_compare = [
    ('NSE', '410734_nse.pkl'),
    ('LogNSE', '410734_lognse.pkl'),
    ('SqrtNSE', '410734_sqrtnse.pkl'),
    ('SDEB', '410734_sdeb.pkl'),
    ('KGE', '410734_kge.pkl'),
    ('KGE(√Q)', '410734_kge_sqrt.pkl'),
]

loaded_reports = {}
for name, filename in reports_to_compare:
    try:
        loaded_reports[name] = CalibrationReport.load(report_dir / filename)
    except Exception as e:
        print(f"Could not load {filename}: {e}")

print(f"Loaded {len(loaded_reports)} reports for comparison")

# %%
# Compare key metrics across reports
print("=" * 100)
print("METRIC COMPARISON ACROSS CALIBRATIONS")
print("=" * 100)

comparison_data = []
for name, report in loaded_reports.items():
    metrics = report.calculate_metrics()
    comparison_data.append({
        'Calibration': name,
        'NSE': metrics.get('NSE', np.nan),
        'LogNSE': metrics.get('LogNSE', np.nan),
        'KGE': metrics.get('KGE', np.nan),
        'PBIAS (%)': metrics.get('PBIAS', np.nan),
        'RMSE': metrics.get('RMSE', np.nan),
    })

comparison_df = pd.DataFrame(comparison_data).set_index('Calibration')
print("\n")
print(comparison_df.round(4).to_string())

# %% [markdown]
# ---
# ## 7. Re-running Simulations with Saved Parameters
#
# You can use the saved parameters to re-run simulations on new data.

# %%
# Extract parameters from a saved report
saved_params = loaded_report.result.best_parameters

print("=" * 70)
print("EXTRACTING PARAMETERS FOR RE-USE")
print("=" * 70)

print("\nSaved parameters (can be used with Sacramento model):")
print("-" * 50)
for param, value in saved_params.items():
    print(f"  '{param}': {value:.6f},")

# %%
# Example: Create a new model with saved parameters
# (In practice, you would load new input data here)

print("\nTo re-use these parameters:")
print("-" * 50)
print("""
# Create model with saved parameters
from pyrrm.models import Sacramento

model = Sacramento(catchment_area_km2=516.63)
model.set_parameters(saved_params)
model.reset()

# Run on new data
output = model.run(new_input_data)
simulated_flow = output['runoff'].values
""")

# %% [markdown]
# ---
# ## 8. Exporting Report Data
#
# Export observed and simulated data from a report for external analysis.

# %%
# Export to CSV
export_df = pd.DataFrame({
    'date': loaded_report.dates,
    'observed': loaded_report.observed,
    'simulated': loaded_report.simulated
})
export_df.set_index('date', inplace=True)

# Save to CSV
export_path = '../test_data/reports/410734_sdeb_timeseries.csv'
export_df.to_csv(export_path)
print(f"Time series exported to: {export_path}")
print(f"\nShape: {export_df.shape}")
print(export_df.head(10))

# %% [markdown]
# ---
# ## Summary
#
# This notebook demonstrated how to:
#
# 1. **Load saved calibration reports** using `CalibrationReport.load()`
# 2. **Inspect report contents** - catchment info, parameters, calibration settings
# 3. **Calculate metrics** from saved observed/simulated data
# 4. **Generate report cards** in Matplotlib (static) and Plotly (interactive)
# 5. **Compare multiple reports** to evaluate different objective functions
# 6. **Extract parameters** for re-use in new simulations
# 7. **Export data** to CSV for external analysis
#
# ### Key Functions
#
# | Function | Description |
# |----------|-------------|
# | `CalibrationReport.load(path)` | Load a saved report |
# | `report.calculate_metrics()` | Calculate performance metrics |
# | `report.plot_report_card()` | Generate Matplotlib figure |
# | `report.plot_report_card_interactive()` | Generate Plotly HTML |

# %%
print("=" * 70)
print("TUTORIAL COMPLETE!")
print("=" * 70)
