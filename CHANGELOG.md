# Changelog

All notable changes to **pyrrm** (Python Rainfall-Runoff Models) are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added

- Add `dream_result_to_inference_data()` converter to build ArviZ `InferenceData` from PyDREAM `CalibrationResult` chain data (`visualization/mcmc_plots.py`)
- Add `plot_mcmc_forest()` wrapper around `az.plot_forest` for credible-interval forest plots (`visualization/mcmc_plots.py`)
- Numba JIT-compiled kernels for Sacramento, GR4J, GR5J, and GR6J with 30-70x speedup over pure Python (`models/numba_kernels.py`)
- Numba optional dependency group in `pyproject.toml` (`pip install pyrrm[numba]`)
- Comprehensive Numba equivalence tests: 96 tests across all models, edge cases, and class interface (`models/tests/test_numba_equivalence.py`)
- Numba performance benchmarks with calibration loop simulations (`models/tests/test_numba_benchmark.py`)
- `NUMBA_AVAILABLE` flag exported from `pyrrm.models` for runtime backend detection
- Time-varying parameter (TVP) framework with `GaussianRandomWalk` priors for non-stationary calibration (`calibration/tvp_priors.py`)
- TVP support in NumPyro NUTS adapter and `CalibrationRunner` (`calibration/numpyro_adapter.py`, `calibration/runner.py`)
- Time-varying parameter support in GR4J JAX implementation (`models/gr4j_jax.py`)
- NumPyro NUTS sampler adapter and JAX-based likelihood functions (`calibration/numpyro_adapter.py`, `calibration/likelihoods_jax.py`)
- JAX-accelerated model implementations for Sacramento and GR4J (`models/sacramento_jax.py`, `models/gr4j_jax.py`, `models/utils/s_curves_jax.py`)
- MCMC diagnostics module with ArviZ-based convergence checking (`analysis/mcmc_diagnostics.py`)
- MCMC-specific visualization: rank plots, posterior pairs, uncertainty bands (`visualization/mcmc_plots.py`)
- Batch experiment runner with `ExperimentGrid` and `BatchExperimentRunner` (`calibration/batch.py`)
- Catchment network runner with DAG topology for upstream-to-downstream calibration (`network/`)
- Network data loader for multi-catchment data handling (`network/data.py`)
- Network visualization with Mermaid diagrams and styled plots (`network/visualization.py`)
- Parallel backends: Sequential, Multiprocessing, Ray (`parallel.py`)
- Checkpoint manager for resumable calibration (`calibration/checkpoint.py`)
- Unified experiment naming convention for batch experiments (`calibration/batch.py`)
- JAX/NumPyro optional dependency group in `pyproject.toml`
- JAX/MCMC component exports from package `__init__` modules
- Notebook 10: Executive Summary -- Q&A from 65+ calibration experiments on Queanbeyan River
- Notebook 11: Batch Experiment Runners tutorial
- Notebook 12: Catchment Network Runners tutorial
- Notebook 13: Bayesian Calibration with NumPyro NUTS
- Notebook 14: Time-Varying Parameter (TVP) Calibration with GR4J
- Cursor rule enforcing standard notebook intro and step structure (`notebook-structure.mdc`)
- Canonical 46-metric diagnostic suite: `compute_diagnostics()`, `DIAGNOSTIC_GROUPS`, `print_diagnostics()` (`analysis/diagnostics.py`)
- Add 4 hydrological signature metrics to canonical suite: Sig_BFI (Lyne-Hollick, alpha=0.925), Sig_Flash (Richards-Baker), Sig_Q95, Sig_Q5 — reported as % error between simulated and observed signatures (`analysis/diagnostics.py`)
- KGE non-parametric (Pool et al. 2018) with 4 transformations and 3 components each (16 metrics) (`analysis/diagnostics.py`)
- Export CalibrationReport to multi-sheet Excel and/or CSV for sharing with colleagues (`calibration/export.py`)
- `CalibrationReport.export()` convenience method for single-report export (`calibration/report.py`)
- `export_batch()` for bulk export of all experiments in a `BatchResult` (`calibration/export.py`)
- `BatchResult.export()` convenience method delegating to `export_batch` (`calibration/batch.py`)
- Optional `export` dependency group in `pyproject.toml` for openpyxl (`pip install pyrrm[export]`)
- Export demo section at end of Notebook 02 showing single-report and batch export workflows (`notebooks/02_calibration_quickstart.py`)
- `COLUMN_ALIASES` single source of truth for column-name lookups across the library (`data/input_handler.py`)
- `resolve_column()` helper to find DataFrame columns by canonical name with case-insensitive fallback (`data/input_handler.py`)
- `load_catchment_data()` convenience function replacing ~30 lines of CSV-loading boilerplate per notebook (`data/input_handler.py`)
- Standalone data preparation guide for hydrologists (`docs/data_preparation.md`)
- Tests for `resolve_column`, `COLUMN_ALIASES`, and `load_catchment_data` (`data/tests/test_input_handler.py`)
- Add raw BFI_obs and BFI_sim (Lyne-Hollick baseflow index) to canonical 48-metric diagnostic suite (`analysis/diagnostics.py`)
- Add public `lyne_hollick_baseflow()` function for baseflow/quickflow separation (`analysis/diagnostics.py`)
- Add sim_baseflow and sim_quickflow columns (Lyne-Hollick separation) to TimeSeries Excel/CSV export (`calibration/export.py`)
- Add `plot_rhat_summary()` horizontal bar chart for per-parameter convergence at a glance (`visualization/mcmc_plots.py`)
- Add `param_bounds` argument to `plot_dream_traces()` so KDE x-axis spans full feasible parameter range (`visualization/mcmc_plots.py`)
- Add parameter identifiability heatmap (normalised HDI width / parameter range, 22 params x 4 transforms) in NB06 (`notebooks/06_algorithm_comparison.py`)
- Add SCE-UA point-in-posterior heatmap (position within PyDREAM 94% HDI, 22 params x 13 objectives) in NB06 (`notebooks/06_algorithm_comparison.py`)
- Add compact convergence summary table (max/mean R-hat, min ESS) for 4 PyDREAM transforms in NB06 (`notebooks/06_algorithm_comparison.py`)

### Changed

- Replace rudimentary histogram posterior plots in NB06 with professional ArviZ MCMC diagnostics (trace, pair, forest, rank plots) matching NUTS notebooks 13/14 (`notebooks/06_algorithm_comparison.py`)
- Show posteriors for all Sacramento parameters (not just a subset) across all 13 objective functions in NB06 (`notebooks/06_algorithm_comparison.py`)
- Replace uninformative rank plots with R-hat convergence bar charts in NB06 (`notebooks/06_algorithm_comparison.py`)
- Remove single-scale forest plot (mixed parameter units) in favour of per-parameter cross-objective forest plots in NB06 (`notebooks/06_algorithm_comparison.py`)
- Switch pair plots from hexbin to KDE contours with HDI levels for smoother publication-quality output (`visualization/mcmc_plots.py`)
- Expand all notebook heatmaps (NB05, NB07, NB14) to show full 22 headline diagnostic metrics with correct closer-to-zero colour scaling for bias and signature columns (`notebooks/`)
- Replace legacy `calculate_metrics` with canonical `compute_diagnostics` in NB10 (`notebooks/10_executive_summary.py`)
- Standardise diagnostic metrics across all notebooks (NB02-NB14) to use `compute_diagnostics` with canonical naming (`analysis/diagnostics.py`)
- Rename NSE variant keys to match KGE naming pattern: `LogNSE`→`NSE_log`, `SqrtNSE`→`NSE_sqrt`, `InvNSE`→`NSE_inv` (`analysis/diagnostics.py`, all notebooks, GUI)
- Update `CalibrationReport.calculate_comprehensive_metrics()` to delegate to `compute_diagnostics` (`calibration/report.py`)
- Update `CalibrationReport.summary()` to display the full 46-metric grouped table (`calibration/report.py`)
- Update `ModelDiagnostics.get_metrics()` to use `compute_diagnostics` instead of deprecated `calculate_metrics` (`analysis/diagnostics.py`)
- Standardise experiment/calibration file naming across all notebooks to canonical format `{catchment}_{model}_{objective}_{algorithm}[_{transformation}]` (`notebooks/`)
- Rename NB02 save paths from short names (e.g. `410734_nse`) to canonical keys (e.g. `410734_sacramento_nse_sceua`) (`notebooks/02_calibration_quickstart.py`)
- Fix NB03 to use `_inverse` (not `_inv`) and `kgenp` (not `kge_np`) in file name dictionaries (`notebooks/03_routing_quickstart.py`)
- Fix NB08 progress file path construction to use canonical `inverse` and `kgenp` components (`notebooks/08_calibration_monitor.py`)
- Fix NB09 documentation table to use `kgenp_sceua` (not `kge_np_sceua`) (`notebooks/09_calibration_reports.py`)
- Fix NB14 prior-result loading to use canonical suffixes matching NB02/06/07 output files (`notebooks/14_tvp_gr4j.py`)
- Rename 14 existing Sacramento SCE-UA `.pkl` report files to canonical naming convention (`test_data/reports/`)
- Refactor all model `run()` methods (Sacramento, GR4J, GR5J, GR6J) to use `resolve_column` instead of inline if/elif chains (`models/`)
- Refactor `CalibrationRunner.create_report()` to use `resolve_column` for precipitation/PET extraction (`calibration/runner.py`)
- Replace local `_PRECIP_COLUMNS` list in export module with `COLUMN_ALIASES` import (`calibration/export.py`)
- Replace local `PRECIP_SYNONYMS`/`PET_SYNONYMS`/`FLOW_SYNONYMS` in network data loader with `COLUMN_ALIASES` import (`network/data.py`)
- Update `InputDataHandler` class attributes to reference `COLUMN_ALIASES` instead of maintaining separate lists (`data/input_handler.py`)
- Align all real-data notebooks (02, 03, 04, 05, 06, 07, 13, 14) to use `load_catchment_data()` with canonical column names (`notebooks/`)
- Consolidate NB06 performance/metrics/dashboard into one compact comparison table and runtime bar chart (`notebooks/06_algorithm_comparison.py`)
- Replace cluttered cross-transform forest grids, posterior-vs-SCE-UA dropdown, and duplicated convergence diagnostics with focused heatmaps and summary table in NB06 (`notebooks/06_algorithm_comparison.py`)

### Deprecated

- `calculate_metrics()` in `calibration/objective_functions.py` — use `compute_diagnostics()` from `analysis.diagnostics` instead

### Changed

- Standardized all notebook intro sections with Purpose, What You'll Learn, Prerequisites, Steps table
- Renumbered notebooks 11-14 for logical progression (batch, network, NUTS, TVP)
- Aligned notebooks 13 and 14 with standard intro and step structure
- Relaxed SCE-UA and composite objective test tolerances for stochastic stability

### Removed

- SPOTPY dependency -- replaced with vendored SCE-UA implementation and direct adapters
- Vendored SPOTPY library files
- Combined batch-and-network notebook (split into separate notebooks 11 and 12)

---

## [0.1.0] - 2025-01-01

Initial release of pyrrm.

### Added

- **Models**: Sacramento Soil Moisture Accounting model (22 parameters, verified against C# reference)
- **Models**: GR4J -- 4-parameter daily rainfall-runoff model (Perrin et al., 2003)
- **Models**: GR5J -- 5-parameter extension with groundwater exchange threshold (Le Moine, 2008)
- **Models**: GR6J -- 6-parameter extension with exponential store for low flows
- **Models**: `BaseRainfallRunoffModel` abstract class and shared utilities (unit hydrographs, S-curves)
- **Calibration**: `CalibrationRunner` unified interface for all calibration methods
- **Calibration**: PyDREAM adapter for MT-DREAM(ZS) Bayesian MCMC (`calibration/pydream_adapter.py`)
- **Calibration**: SCE-UA vendored implementation with PCA recovery (`calibration/sceua_adapter.py`)
- **Calibration**: SciPy adapter for Differential Evolution, Dual Annealing, Basin Hopping (`calibration/scipy_adapter.py`)
- **Calibration**: `CalibrationReport` for saving and loading calibration results (`calibration/report.py`)
- **Objectives**: Comprehensive `pyrrm.objectives` module with pluggable architecture
- **Objectives**: Traditional metrics -- NSE, RMSE, MAE, PBIAS, SDEB
- **Objectives**: KGE family -- 2009, 2012, 2021 formulations; non-parametric KGE (Spearman)
- **Objectives**: APEX adaptive process-explicit objective with dynamics and lag multipliers
- **Objectives**: Flow transformations -- sqrt, log, inverse, squared, power, Box-Cox
- **Objectives**: FDC-based metrics and multi-segment evaluation
- **Objectives**: Hydrological signature metrics -- flow indices, dynamics, water balance
- **Objectives**: Composite objectives with `WeightedObjective` and factory functions (`kge_hilo`, `comprehensive_objective`)
- **Objectives**: Correlation metrics -- Pearson and Spearman
- **Routing**: Nonlinear Muskingum channel routing (`routing/muskingum.py`)
- **Routing**: `RoutedModel` wrapper combining RR models with routing for joint calibration
- **Routing**: `create_router` factory function
- **Analysis**: Sobol global sensitivity analysis with SALib (`analysis/sensitivity.py`)
- **Analysis**: Model performance diagnostics (`analysis/diagnostics.py`)
- **Visualization**: Hydrographs, flow duration curves, scatter plots with metrics (`visualization/model_plots.py`)
- **Visualization**: MCMC trace plots, posterior distributions, dotty plots (`visualization/calibration_plots.py`)
- **Visualization**: Sobol indices bar charts and interaction heatmaps (`visualization/sensitivity_plots.py`)
- **Visualization**: Report card plots in Matplotlib and Plotly (`visualization/report_plots.py`)
- **Data**: `InputDataHandler` for data loading and validation (`data/input_handler.py`)
- **Data**: Parameter bounds loading and saving utilities (`data/parameter_bounds.py`)
- Sacramento verification against C# reference implementation (R² > 0.9999)
- Notebook 01: Sacramento Verification against C# and SOURCE benchmarks
- Notebook 02: Calibration Quickstart -- first model calibration tutorial
- Notebook 03: Channel Routing with Nonlinear Muskingum
- Notebook 04: Objective Functions deep dive (WIP)
- Notebook 05: APEX Complete Guide -- research evaluation with 6 research questions
- Notebook 06: Algorithm Comparison -- PyDREAM vs SCE-UA vs SciPy
- Notebook 07: Model Comparison -- GR4J, GR5J, GR6J vs Sacramento across 13 objectives
- Notebook 08: Calibration Monitor -- real-time MCMC monitoring
- Notebook 09: Calibration Reports -- working with saved results
- Jupytext paired notebook workflow (`.py` percent format as canonical source)
- Comprehensive README with quick start, model documentation, and learning paths
- `pyproject.toml` with optional dependency groups (`full`, `calibration`, `sensitivity`, `jax`, `dev`)

---

[Unreleased]: https://github.com/ACTGovernment/ACT-Rainfall-Runoff-Modelling/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ACTGovernment/ACT-Rainfall-Runoff-Modelling/releases/tag/v0.1.0
