# Lessons Learnt

Development insights, pitfalls, and best practices discovered while building **pyrrm**. This is a living document -- entries are added as the codebase evolves.

---

## Numerical Precision

### Sacramento Mass Balance Requires Careful Tolerance

**Context**: Verifying the Python Sacramento implementation against the C# reference.

**Lesson**: Floating-point differences between languages accumulate over long time series (25+ years of daily steps). Direct equality checks fail; use relative tolerances (`rtol=1e-4`) combined with absolute tolerances (`atol=1e-8`) when comparing JAX and NumPy implementations. For cross-language verification, R² > 0.9999 and MAE < 0.001 mm/day are practical thresholds.

**Evidence**: `test_sacramento_jax.py` uses `rtol=1e-4, atol=1e-8`; the GR4J JAX tests achieve tighter `rtol=1e-8` because the model is simpler with fewer accumulated operations.

### JAX and NumPy Can Diverge on Edge Cases

**Context**: JAX JIT-compiled model runs vs pure NumPy implementations.

**Lesson**: JAX uses 32-bit floats by default (`jax.config.update("jax_enable_x64", True)` is needed for 64-bit). Even with 64-bit enabled, operation ordering differences in JAX's XLA compiler can produce subtly different results from NumPy. Always test JAX implementations against NumPy baselines with appropriate tolerances, not exact equality.

---

## Calibration

### Stochastic Calibration Tests Need Generous Tolerances

**Context**: Unit tests for SCE-UA and composite objective calibration were intermittently failing.

**Lesson**: Stochastic optimization algorithms (SCE-UA, Differential Evolution) do not produce identical results across runs. Test tolerances of `atol=0.1` to `atol=0.5` are appropriate for parameter recovery tests. The commit `e8bb7af` explicitly relaxed these after repeated CI failures. Prefer testing that the objective improves significantly rather than that exact parameter values are recovered.

**Evidence**: `test_sceua_direct.py` uses `atol=0.2` to `atol=0.5` for parameter assertions; `test_composite.py` uses `value > 0.85` rather than exact checks.

### NumPyro NUTS Is Sensitive to Prior Specification

**Context**: Adding the NumPyro NUTS adapter for Bayesian inference.

**Lesson**: The NUTS sampler can fail silently (producing divergent transitions) if priors are poorly specified. Uniform priors over the full parameter range work but are inefficient. Weakly informative priors centered on physically reasonable values dramatically improve sampling efficiency. The warmup phase (adaptation of step size and mass matrix) needs at least 500 samples for models with 4+ parameters; 200 is often insufficient.

### SCE-UA Vendoring Eliminates a Fragile Dependency

**Context**: The original calibration framework depended on SPOTPY, which had installation issues and API instability.

**Lesson**: For critical algorithms, vendoring a clean implementation is preferable to depending on an external library with uncertain maintenance. The vendored SCE-UA in `calibration/_sceua/sceua.py` includes PCA recovery for lost dimensions and ThreadPoolExecutor parallelization -- features that would be difficult to add via monkey-patching an external library. The removal of SPOTPY (commits `bea8b4c`, `b531982`) simplified the dependency tree significantly.

### PyDREAM Multi-Try Sampling Needs Chain Thinning

**Context**: Large MCMC runs with PyDREAM producing excessive memory usage.

**Lesson**: MT-DREAM(ZS) with snooker updates generates many candidate samples per iteration. For long runs (50,000+ iterations), storing all samples can exhaust memory. Thin chains during post-processing (keep every Nth sample) rather than trying to store everything. The `CalibrationReport` serialization also benefits from thinned chains -- a 50,000-iteration, 5-chain run can produce multi-GB pickle files without thinning.

---

## Model Implementation

### Optional Dependency Pattern Keeps Core Lightweight

**Context**: Supporting JAX, NumPyro, PyDREAM, SALib, ArviZ, and Plotly without requiring all of them.

**Lesson**: Use the `try/except ImportError` pattern with module-level availability flags (e.g., `PYDREAM_AVAILABLE`, `NUMPYRO_AVAILABLE`, `JAX_AVAILABLE`, `ARVIZ_AVAILABLE`). This lets the core library work with only numpy/pandas/scipy/matplotlib while enabling advanced features when optional packages are installed. Error messages should be actionable: tell the user exactly which package to install and how.

**Evidence**: `calibration/__init__.py` guards PyDREAM, NumPyro, and new objectives imports behind `try/except` blocks with corresponding `*_AVAILABLE` flags.

### JAX JIT Compilation Requires Static Shapes

**Context**: Porting Sacramento and GR4J to JAX for GPU acceleration.

**Lesson**: JAX's `jit` requires array shapes to be known at compile time. Dynamic-length time series must be padded to a fixed length or use `jax.lax.scan` with fixed iteration counts. The S-curve functions needed separate JAX implementations (`s_curves_jax.py`) because the original NumPy versions used dynamic indexing incompatible with JIT. Use `jax.lax.cond` instead of Python `if/else` for branching inside JIT-compiled functions.

### Parameter Bounds Must Match Reference Literature

**Context**: Sacramento model calibration producing physically unrealistic results.

**Lesson**: Parameter bounds that are too wide waste calibration effort exploring infeasible regions; bounds that are too narrow can exclude optimal solutions. Always verify bounds against the original model documentation. The Sacramento UZTWM bounds were initially set too tight at (0, 50) but the literature supports (1, 300). Commits `a8ce1a4` and `6562751` corrected this after calibration failures.

---

## Routing

### Nonlinear Muskingum Requires Sub-Reach Discretization

**Context**: Implementing the nonlinear Muskingum routing method (S = K*Q^m).

**Lesson**: A single reach with large K values produces numerical instability (negative flows, oscillations). Dividing the reach into `n_subreaches` (typically 3-10) and routing sequentially through each sub-reach maintains stability. The Courant condition (dt/K < 1 for each sub-reach) provides a practical guideline for choosing the number of sub-reaches.

---

## Testing

### Test Tolerances Should Reflect the Algorithm, Not Just the Math

**Context**: Establishing test suites for objectives, calibration adapters, and model implementations.

**Lesson**: Different categories of code need different tolerance strategies:
- **Deterministic math** (objective functions, model equations): tight tolerances (`atol=1e-10`)
- **Cross-implementation** (JAX vs NumPy): moderate tolerances (`rtol=1e-4` to `rtol=1e-8`)
- **Stochastic algorithms** (SCE-UA, DREAM, NUTS): generous tolerances (`atol=0.1` to `atol=0.5`) or behavioral assertions ("objective improved by at least X")
- **Cross-language** (Python vs C#): practical tolerances (R² > 0.9999, MAE < 0.001)

---

## Notebooks & Documentation

### Jupytext Percent Format Is the Canonical Source

**Context**: Managing 14 tutorial notebooks with version control.

**Lesson**: Never edit `.ipynb` files directly. The `.py` files in percent format (`# %%` delimiters) are the source of truth. Jupytext syncs them to `.ipynb` for execution. This produces clean git diffs, enables full IDE support (linting, autocomplete), and avoids notebook metadata noise in code review. The `.ipynb` files were removed from version control (commit `52df7d4`) after initially being tracked (commit `cc34033`).

### Notebook Renumbering Requires Updating All Cross-References

**Context**: Splitting the combined batch/network notebook into separate notebooks 11 and 12, then renumbering NUTS and TVP.

**Lesson**: When renumbering notebooks, every cross-reference must be updated: README notebook table, learning paths, Prerequisites sections in other notebooks, and any import examples that reference notebook numbers. A systematic search-and-replace across all `.py` notebook files and `README.md` is essential. The renumbering sequence (commits `24138f1` through `b1a7e8c`) required four separate commits to complete cleanly.

### Standardized Notebook Structure Improves Consistency

**Context**: Notebooks had inconsistent introductions making it hard for readers to know prerequisites and time commitment.

**Lesson**: Enforcing a standard structure (Purpose, What You'll Learn, Prerequisites, Estimated Time, Steps table, Key Insight, Setup) via a Cursor rule (`notebook-structure.mdc`) ensures every notebook is self-documenting. The structure was retrofitted to notebooks 13 and 14 (commits `2ff183a`, `838a6cd`) and should be applied from the start for new notebooks.

---

## Architecture

### Batch Experiments Need Consistent Naming Conventions

**Context**: Running 65+ calibration experiments on the Queanbeyan River catchment.

**Lesson**: Without a naming convention, experiment results become impossible to compare or reproduce. The unified naming convention (commit `9ddd24f`) encodes model, objective, method, and timestamp into each experiment name. This enabled the executive summary notebook (NB10) to systematically analyze and rank all experiments.

### Network Runner DAG Must Be Validated Before Execution

**Context**: Building the upstream-to-downstream multi-catchment calibration runner.

**Lesson**: The `CatchmentNetwork` topology must be a valid DAG (directed acyclic graph). Cycles cause infinite loops; disconnected nodes produce missing upstream contributions. Validate the DAG at construction time with topological sort and check that all referenced upstream catchments have data loaded in `NetworkDataLoader`. Fail early with clear error messages rather than producing silently incorrect results.

### Adapter Pattern Scales Well for Calibration Methods

**Context**: Supporting PyDREAM, SCE-UA, SciPy, and NumPyro from a single `CalibrationRunner`.

**Lesson**: Each calibration library has its own API conventions (parameter formats, objective function signatures, result containers). The adapter pattern (`pydream_adapter.py`, `sceua_adapter.py`, `scipy_adapter.py`, `numpyro_adapter.py`) translates between pyrrm's unified interface and each library's expectations. Adding a new method means writing one adapter file and one `run_*()` method on `CalibrationRunner` -- the rest of the framework (reporting, visualization, objectives) works unchanged.

---

## Performance

### Numba JIT Gives 30-70x Speedup for Conceptual Model Loops

**Context**: Pure-Python timestep loops in Sacramento and GR4J are the bottleneck during calibration (10,000+ model evaluations). NumPy vectorization is not possible because each timestep depends on the previous state.

**Lesson**: Numba `@njit(cache=True)` is the ideal acceleration strategy for sequential hydrological model loops. Key requirements for Numba compatibility: (1) extract class method logic into standalone functions taking only scalars and NumPy arrays, (2) replace `np.tanh()`/`np.ceil()` with `math.tanh()`/`math.ceil()` for scalars, (3) replace `math.pow()` with `**` operator, (4) avoid Python dicts, classes, string operations, and exception handling inside the JIT-compiled function, (5) use `cache=True` to persist compiled code across sessions. The Sacramento model required the most refactoring because `_run_time_step` accessed ~50 `self.*` attributes that had to be packed into explicit function arguments. GR4J/GR5J/GR6J already had standalone `_core` functions and needed minimal changes.

**Evidence**: Measured speedups on Apple M-series (ARM64): GR4J 28-34x, GR5J 30-34x, GR6J 58-69x, Sacramento 40-43x. A 1000-evaluation calibration loop on 10 years of daily data completes in 0.3s (GR4J) and 0.5s (Sacramento) with Numba, versus 9.2s and 18s with pure Python. See `models/tests/test_numba_benchmark.py`.

### Process Full Time Series Inside JIT for Maximum Speedup

**Context**: Sacramento's original `run()` method called `_run_time_step()` once per day in a Python loop, incurring Python-to-Numba transition overhead on every timestep.

**Lesson**: Moving the entire time-series loop inside the `@njit` function (so the Numba code processes all N timesteps in a single call) eliminates N Python-to-native transitions and allows Numba's LLVM backend to optimize the entire loop. For a 25-year simulation (9,125 timesteps), this is the difference between 9,125 function-call overheads and one. The trade-off is that the JIT function must return all state variables needed by the caller, and the class `run()` method must pack/unpack state before/after the call.

**Evidence**: Sacramento Numba kernel processes 25 years in ~1ms versus ~45ms for the per-timestep Python loop. The speedup would be much smaller if we called the Numba kernel once per timestep.

### Graceful Fallback Pattern for Optional Accelerators

**Context**: Numba is an optional dependency -- the library must work without it.

**Lesson**: Use a no-op decorator fallback so the same `@njit` decorators work with or without Numba installed. The pattern `try: from numba import njit; NUMBA_AVAILABLE = True except ImportError: ...` with a fallback `njit` that returns the function unchanged lets the same source file define both the accelerated and fallback paths. Model classes then dispatch with `core_fn = _core_numba if NUMBA_AVAILABLE else _core_python`. This mirrors the existing JAX pattern (`JAX_AVAILABLE`) already used in the codebase.

**Evidence**: `models/numba_kernels.py` implements this pattern; all four model classes dispatch transparently.

---

## Dependencies

### Pin Minimum Versions, Not Exact Versions

**Context**: Specifying dependencies in `pyproject.toml`.

**Lesson**: Use `>=` minimum version constraints (e.g., `numpy>=1.20`) rather than exact pins. This avoids conflicts when pyrrm is installed alongside other packages in a user's environment. Only pin exact versions if a known incompatibility exists. The `[full]` optional dependency group bundles all optional packages for convenience without forcing users to install everything.

### Single Source of Truth for Diagnostic Metrics

**Context**: Diagnostic metrics (NSE variants, KGE variants, FDC biases, etc.) were defined independently in multiple notebooks with inconsistent naming (`"NSE (log Q)"` vs `"LogNSE"`, `"FDC-high"` vs `"FHV"`) and incomplete metric sets (some notebooks computed 6 metrics, others 22, others 27).

**Lesson**: Promote the canonical metric suite into the library (`pyrrm.analysis.diagnostics.compute_diagnostics`) as the single source of truth. Notebooks import and alias as needed rather than defining their own computation functions. This ensures (a) consistent metric naming across all outputs, (b) no accidental formula differences between notebooks, and (c) a single place to fix if a metric definition changes. When a notebook needs additional domain-specific metrics beyond the canonical 42, it wraps `compute_diagnostics` and appends extra keys rather than reimplementing the base suite. All metric families must follow the same `{metric}[_{transformation}][_{component}]` pattern (e.g., `NSE_log`, `KGE_sqrt_alpha`, `KGE_np_inv_r`). The earlier `LogNSE`/`SqrtNSE`/`InvNSE` naming broke this pattern and was renamed to `NSE_log`/`NSE_sqrt`/`NSE_inv`.

**Evidence**: Before standardisation, NB02 used `"InvNSE (1/Q)"`, NB05 used `"FDC-high"`, and NB07 used `"FDC_High"` for the same conceptual metric. After standardisation, all notebooks use the canonical key set from `DIAGNOSTIC_GROUPS`. The suite now includes KGE non-parametric (Pool et al. 2018) across all 4 transformations with components, plus 4 hydrological signature errors (Sig_BFI via Lyne-Hollick, Sig_Flash via Richards-Baker, Sig_Q95, Sig_Q5), totalling 48 metrics in 13 groups (including raw BFI_obs/BFI_sim values alongside the percentage-error signatures). The 24 headline (non-component) metrics are used in all comparison heatmaps.

### Canonical Experiment Key Format Must Be Enforced in Notebooks

**Context**: NB02 (Calibration Quickstart) saved report files with short names like `410734_nse.pkl` (missing model and algorithm), while all downstream notebooks (NB03-NB14) expected the canonical format `410734_sacramento_nse_sceua.pkl`. Additionally, NB03 used `_inv` as an abbreviation while NB05-07 used the full `_inverse`, and `kge_np` vs `kgenp` varied across notebooks.

**Lesson**: The canonical experiment key format `{catchment}_{model}_{objective}_{algorithm}[_{transformation}]` defined in `pyrrm/calibration/batch.py::make_experiment_key()` must be the sole authority for file naming. Producer notebooks (NB02, NB07) must use canonical keys when saving, and consumer notebooks (NB03, NB05, NB06, NB09, NB10, NB14) must use the same keys when loading. Key rules: (a) underscores are field separators so `kge_np` becomes `kgenp` in the key, (b) transformations use the full word `inverse` (not `inv`), and (c) every key must contain all four required fields (catchment, model, objective, algorithm). When a new notebook is created, check that its file paths conform to `make_experiment_key` output.

**Evidence**: NB02 originally saved 14 files as `410734_nse.pkl`, `410734_lognse.pkl`, etc. NB05/06/07 expected `410734_sacramento_nse_sceua.pkl`, `410734_sacramento_nse_sceua_log.pkl`, causing silent load failures unless NB02 was re-run with canonical names. NB14's `OBJECTIVES_13` list constructed impossible filenames like `410734_sacramento_rmse_sceua.pkl` that no notebook produces.

### ArviZ Is Essential for MCMC Diagnostics but Heavy

**Context**: Adding convergence diagnostics for NumPyro NUTS.

**Lesson**: ArviZ provides R-hat, ESS, rank plots, and pair plots that are critical for validating MCMC results. However, it pulls in a large dependency tree (xarray, netcdf4, etc.). Guard it behind `ARVIZ_AVAILABLE` so users who only need SCE-UA or SciPy calibration are not burdened. The `mcmc_diagnostics.py` module degrades gracefully when ArviZ is absent.

### Duck-Type Batch Export to Work with Any Dict-of-Reports Container

**Context**: Adding `export_batch()` to export all experiments from an `ExperimentGrid` or `ExperimentList` workflow.

**Lesson**: `export_batch` accepts any object with a `.results` dict (mapping `str -> CalibrationReport`) rather than requiring a `BatchResult` type. This lets users build a pseudo-batch from on-disk pickle files (using `SimpleNamespace(results={...})`) and run the same bulk export without having re-run a full `BatchExperimentRunner`. Keep batch-level methods as thin wrappers around the per-report primitives; avoid coupling them tightly to the `BatchResult` class.

**Evidence**: In notebook 02, users load 14 previously-saved `.pkl` files into a dict and pass a `SimpleNamespace` to `export_batch` to create 14 Excel files in a single call, without needing the original `BatchResult` object.

### Single Source of Truth for Column-Name Aliases Prevents Silent Data Loss

**Context**: The library had three independent column-name lookup lists (in `InputDataHandler`, `CalibrationRunner.create_report()`, `export.py`, and `network/data.py`). When notebooks used `'rainfall'` but `create_report()` didn't include it in its list, `report.precipitation` was silently set to `None`, causing missing data in exports.

**Lesson**: Column-name resolution must be defined exactly once. `COLUMN_ALIASES` in `pyrrm/data/input_handler.py` is now the single source of truth, and every lookup site (models, runner, export, network) imports and uses `resolve_column()` from `pyrrm.data`. This eliminates the class of bugs where a new alias is added in one place but forgotten in another. The `resolve_column` function performs a two-pass lookup (exact match first, case-insensitive fallback) which covers the full range of column naming conventions encountered in Australian hydrological data.

**Evidence**: The precipitation-missing-from-exports bug was caused by `create_report()` not listing `'rainfall'` as a precipitation alias, even though `InputDataHandler` and `export.py` both did. After centralisation, all four lookup sites resolve `'rainfall'` → `'precipitation'` consistently.

## Dependencies

### PyDREAM multitry=2 Bug in Serial Mode

**Context**: GR5J and GR6J PyDREAM calibrations failed with `ValueError: not enough values to unpack (expected 2, got 1)` when configured with `multitry=2` and `parallel=False`.

**Lesson**: PyDREAM's `mt_evaluate_logps` has a bug in the `multitry == 2` code path (serial mode). Instead of iterating over proposal points individually (like the `multitry > 2` branch), it calls `pfunc(np.squeeze(proposed_pts))` with the full 2D array and wraps the result in `np.array([...])`, producing shape `(1, ...)` which cannot be unpacked into `(log_priors, log_likes)`. This fails regardless of what the likelihood function returns. **Workaround: use `multitry=1` or `multitry >= 3`, never `multitry=2` in serial mode.** Additionally, PyDREAM always expects the likelihood function to return a tuple `(log_prior, log_likelihood)`, not a single float — even with `multitry=1`.

**Evidence**: `pydream/Dream.py` line 867-868 vs 870-872 (version installed in `pyrrm` env). Fixed in `pyrrm/calibration/pydream_adapter.py` (`PyDREAMLikelihood.__call__` now returns tuple) and `notebooks/07_model_comparison.py` (changed `multitry=2` to `multitry=3`).

## Visualisation

### Heatmaps Scale Better Than Per-Parameter Forest Plot Grids

**Context**: Notebook 06 displayed 22 subplots (one per Sacramento parameter) with forest-plot intervals for 4 PyDREAM transforms and 13 SCE-UA objectives. The resulting Plotly grid was repeatedly deemed "cluttered" and "unusable" despite adjusting row heights, column counts, and legend placement.

**Lesson**: When comparing many parameters across many conditions, a single heatmap with a normalised metric (e.g., HDI width / parameter range) conveys the same information in far less visual space. The identifiability heatmap (22 rows × 4 columns) replaced an 8×3 grid of forest plots. Similarly, a point-in-posterior heatmap (22 rows × 13 columns, cell = percentage position within HDI) replaced a 22-figure dropdown. Heatmaps allow the reader to spot patterns (equifinality, transform sensitivity) at a glance — something impossible with per-subplot layouts. Reserve forest plots for detailed inspection of 1-3 parameters at a time.

**Evidence**: `notebooks/06_algorithm_comparison.py` — Sections A and B replaced the `plot_forest_grid_plotly` and `plot_forest_interactive` calls and the 22-parameter dropdown Plotly figure.
