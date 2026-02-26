# Testing Guide

This document describes the testing strategy, test inventory, and instructions for running the **pyrrm** test suite. It is intended for developers, reviewers (e.g. JOSS, Environmental Modelling & Software), and users who want to verify the correctness of the library.

---

## Quick Start

```bash
# Activate the environment
conda activate pyrrm

# Run all fast tests (excludes benchmarks)
pytest pyrrm/ -m "not slow"

# Run all tests including benchmarks
pytest pyrrm/

# Run with coverage report
pytest pyrrm/ -m "not slow" --cov=pyrrm --cov-report=term-missing

# Run a single module
pytest pyrrm/models/tests/test_numba_equivalence.py -v

# Run a single test class
pytest pyrrm/objectives/tests/test_apex.py::TestAPEXBasic -v
```

---

## Test Suite Summary

| Metric | Value |
|--------|-------|
| Total test functions | 547 |
| Test modules | 20 |
| Test directories | 4 (`models`, `calibration`, `objectives`, `data`) |
| Framework | pytest 7+ |
| Markers | `slow` (benchmarks and long-running MCMC) |

---

## Test Strategy

### Co-located Tests

Tests live alongside the source code they verify, inside `tests/` subdirectories within each package:

```
pyrrm/
├── models/tests/          5 test files, ~250 tests
├── calibration/tests/     7 test files, ~170 tests
├── objectives/tests/      7 test files (+ fixtures), ~100 tests
└── data/tests/            1 test file, ~25 tests
```

### Synthetic and Real-Data Fixtures

Most tests use **reproducible synthetic data** generated with fixed random seeds (`np.random.RandomState(42)`). This ensures:

- Tests are deterministic across platforms
- No dependency on external data files for the core test suite
- Edge cases (zero precipitation, storm bursts, alternating wet/dry) are explicitly tested

A subset of integration tests uses **real gauged data** (Gauge 410734, Queanbeyan, ~135 years of daily records). These tests are skipped automatically if the data files are not present.

### Numerical Tolerance Philosophy

Different backends have different numerical characteristics, and tolerances are set accordingly:

| Comparison | Relative tolerance | Rationale |
|---|---|---|
| Numba vs Python | `rtol=1e-12` | Both use IEEE 754 float64; differences arise only from expression ordering |
| Numba-fastmath vs Numba | `rtol=1e-6` | Fastmath permits fused multiply-add and reassociation |
| JAX vs NumPy | `rtol=1e-4` to `1e-8` | JAX uses XLA compilation with different floating-point semantics; GR4J achieves `1e-8`, Sacramento `1e-4` |
| MLX vs Numba | `rtol=1e-10` | Apple Metal backend, similar to standard float64 |
| C# reference vs Python | `rtol=1e-4` | Cross-language comparison with different math libraries |

### Test Categories

1. **Correctness tests** -- verify model outputs against reference implementations and across backends
2. **Physical validity tests** -- non-negative flows, no NaN/Inf, water balance, mass conservation
3. **Interface tests** -- class API (`run`, `reset`, `get_state`, `set_parameters`), serialization, export
4. **Edge case tests** -- zero precipitation, empty arrays, mismatched lengths, missing data, invalid parameters
5. **Performance benchmarks** (marked `@pytest.mark.slow`) -- speedup verification, calibration loop timing

---

## Test Inventory

### 1. Models (`pyrrm/models/tests/`)

#### `test_numba_equivalence.py` -- Backend Correctness

The cornerstone test file. Verifies that Numba-compiled kernels produce identical results to pure-Python implementations for all four models.

| Test Class | Tests | What It Validates |
|---|---|---|
| `TestSCurves` | 64 | S-curve functions match at `abs=1e-15` across 8 t-values x 4 X4 values x 2 curve types |
| `TestUnitHydrograph` | 1 | Unit hydrograph step function matches at `abs=1e-14` |
| `TestGR4JNumbaEquivalence` | 8 | Raw kernel output for 5 parameter sets, 10yr stability, zero-precip, non-negative flows |
| `TestGR5JNumbaEquivalence` | 3 | Raw kernel output for 3 parameter sets |
| `TestGR6JNumbaEquivalence` | 3 | Raw kernel output for 3 parameter sets including exponential store |
| `TestSacramentoNumbaEquivalence` | 8 | Raw kernel for 3 parameter sets, 10yr stability, zero-precip, full-stores init, C# reference, non-negative, no-NaN |
| `TestClassInterface` | 8 | Class-level `run()`, `run_timestep()`, `reset()`, `get_state()`, catchment area scaling for all models |
| `TestSyntheticPythonVsNumbaGR4J` | 12 | Class-level dispatch with monkey-patching for 5 param sets x 6 scenarios (synthetic, 10yr, storm, alternating, zero, state) |
| `TestSyntheticPythonVsNumbaGR5J` | 8 | Same pattern for GR5J across 3 param sets x multiple scenarios |
| `TestSyntheticPythonVsNumbaGR6J` | 8 | Same pattern for GR6J across 3 param sets x multiple scenarios |
| `TestSyntheticPythonVsNumbaSacramento` | 8 | Same pattern for Sacramento across 3 param sets x multiple scenarios |
| `TestRealData410734Equivalence` | 5 | Full 135-year record, final state, non-negative/finite, 2000-2023 subset, cumulative mass balance drift |

#### `test_gr4j_jax.py` -- JAX GR4J Equivalence

| Test Class | Tests | What It Validates |
|---|---|---|
| `TestNumpyJaxEquivalence` | 3 | JAX output matches NumPy at `rtol=1e-8` for standard, multiple, and multi-year runs |
| `TestGradients` | 2 | Finite, non-zero gradients for all 4 parameters; no NaN at 3 test points |
| `TestJIT` | 1 | `jax.jit` compilation succeeds and produces correct shape |
| `TestPhysicalValidity` | 3 | Non-negative flows, water balance (Q < P), zero-precip recession |
| `TestTVPForward` | 11 | Time-varying parameters: constant TVP matches scalar, varying TVP changes output, multiple TVP combinations, gradients through TVP, JIT with TVP |
| `TestUnitHydrograph` | 11 | UH1 and UH2 ordinates sum to 1.0 for 5 X4 values; non-negative ordinates |

#### `test_sacramento_jax.py` -- JAX Sacramento Equivalence

| Test Class | Tests | What It Validates |
|---|---|---|
| `TestNumpyJaxEquivalence` | 2 | Default params and multi-year stability at `rtol=1e-4` |
| `TestGradients` | 1 | Finite gradients for `uztwm`, `uzfwm`, `uzk` |
| `TestJIT` | 1 | JAX JIT compilation succeeds |
| `TestPhysicalValidity` | 2 | Non-negative flows, approximate water balance |

#### `test_numba_benchmark.py` -- Performance Benchmarks (`@pytest.mark.slow`)

| Test Class | Tests | What It Validates |
|---|---|---|
| `TestGR4JBenchmark` | 3 | Numba speedup > 1x at 1yr, 10yr, 25yr |
| `TestGR5JBenchmark` | 3 | Same for GR5J |
| `TestGR6JBenchmark` | 3 | Same for GR6J |
| `TestSacramentoBenchmark` | 3 | Same for Sacramento |
| `TestCalibrationLoopBenchmark` | 2 | 1000-eval GR4J loop speedup > 2x; 1000-eval Sacramento < 2 min |
| `TestCompilationOverhead` | 1 | First-call vs cached-call JIT timing |
| `TestSummaryTable` | 1 | Consolidated performance table for all models |

#### `test_sacramento_perf_comparison.py` -- Cross-Backend Comparison (`@pytest.mark.slow`)

| Test Class | Tests | What It Validates |
|---|---|---|
| `TestSingleRunBenchmarks` | 3 | Python vs Numba vs Numba-fast vs JAX vs MLX at 1yr/10yr/25yr |
| `TestFastmathEquivalence` | 3 | Fastmath matches standard Numba at `rtol=1e-6`; non-negative and finite |
| `TestMLXEquivalence` | 1 | MLX matches Numba at `rtol=1e-10` |
| `TestBatchCalibrationBenchmarks` | 3 | Sequential vs parallel batch at 100/500/1000 evaluations |
| `TestBatchEquivalence` | 1 | Parallel batch bit-identical to sequential |
| `TestRealDataBenchmark` | 2 | All backends on 135-year real data; fastmath cumulative drift |
| `TestSummaryTable` | 1 | Consolidated performance table |

---

### 2. Calibration (`pyrrm/calibration/tests/`)

#### `test_sceua_direct.py` -- SCE-UA Optimization

| Test Class | Tests | What It Validates |
|---|---|---|
| `TestSCEUAMinimize` | 6 | Quadratic and Rosenbrock convergence, bounds, seed reproducibility, initial point, message |
| `TestObjectiveDirection` | 4 | Maximize/minimize detection via `direction` and legacy `maximize` attributes |
| `TestSCEUAModelWrapper` | 4 | Bounds extraction, param vector/dict conversion, objective evaluation, negation for maximize |
| `TestRunSCEUADirect` | 5 | Basic calibration, DataFrame format, minimize objective, custom bounds, initial point dict |
| `TestCalibrateSCEUA` | 1 | Convenience function returns `SCEUACalibrationResult` dataclass |
| `TestIntegrationWithPyrrm` | 1 | Integration with actual Sacramento model |

#### `test_numpyro_adapter.py` -- NumPyro NUTS Bayesian Calibration (`@pytest.mark.slow`)

| Test Class | Tests | What It Validates |
|---|---|---|
| `TestRunNuts` | 2 | NUTS runs without error; parameter recovery within 50% of true values |
| `TestCalibrationRunnerNuts` | 2 | `CalibrationRunner.run_nuts()` returns correct method string and result type; sqrt transform |

#### `test_batch.py` -- Batch Experiment Runner

| Test Class | Tests | What It Validates |
|---|---|---|
| `TestExperimentList` | 7 | Construction, `combinations()`, `from_dicts()` factory, unknown model error |
| `TestRunDirNaming` | 4 | Timestamped directory naming, label sanitization, latest-dir discovery |
| `TestBatchRunnerWithExperimentList` | 1 | Runner accepts `ExperimentList`, produces results |
| `TestOutputStructure` | 7 | Timestamped folder, results/ and logs/ subdirectories, batch.log, JSON/CSV summaries, config snapshot, batch_result.pkl |
| `TestResume` | 3 | Skip completed experiments, resume from explicit path, nonexistent path error |
| `TestBackwardCompatResume` | 1 | Legacy flat .pkl directory layout |
| `TestConfigParsing` | 3 | JSON grid config, experiment-list config, ambiguous config error |
| `TestBatchResult` | 4 | Save/load, default path, explicit path, no-path error, repr |
| `TestMakeExperimentKey` | 6 | Four-field key, default catchment, transformation, extra tags, sanitization |
| `TestMakeApexTags` | 4 | Default and custom kappa/regime tags |
| `TestParseExperimentKey` | 5 | Four-field, five-field, APEX tags, roundtrip, short-key fallback |
| `TestExperimentGridCatchment` | 3 | Default/custom catchment, runner forwarding |
| `TestExperimentListAutoKey` | 2 | Auto-generated and explicit keys |

#### `test_export.py` -- Calibration Report Export

| Test Class | Tests | What It Validates |
|---|---|---|
| `TestExportCSV` | 6 | 4 CSV files created with correct columns (TimeSeries, Best_Calibration, Diagnostics, FDC) |
| `TestExportExcel` | 6 | 1 Excel file with 4 sheets and correct columns (requires openpyxl) |
| `TestExportBothAndPath` | 2 | `format='both'` produces 5 files; directory path uses experiment_name |
| `TestReportExportMethod` | 2 | `CalibrationReport.export()` delegates correctly; invalid format error |
| `TestExportErrors` | 2 | Invalid format, missing openpyxl errors |
| `TestExportEdgeCases` | 2 | Missing P/PET exports NaN; fallback to inputs DataFrame 'rainfall' column |
| `TestExportBatch` | 4 | Batch CSV/Excel subdirectories, empty batch, skip failures |

#### `test_persistence.py` -- Save/Load and Checkpointing

| Test Class | Tests | What It Validates |
|---|---|---|
| `TestCalibrationResultSerialization` | 6 | `to_dict()`/`from_dict()` roundtrip, samples inclusion, NumPy type JSON serialization |
| `TestCalibrationResultSaveLoad` | 8 | File creation (meta.json + parquet/csv), roundtrip, chain data (.npz), parent dirs, not-found error, suffix handling, `can_resume()` |
| `TestCheckpointManager` | 12 | Directory creation, save/load checkpoint, latest/best checkpoint, cleanup, list, clear, interval/improvement triggers, iteration tracking |
| `TestPersistenceIntegration` | 2 | Realistic save/load workflow, checkpoint-then-resume |

#### `test_tvp_priors.py` -- Time-Varying Parameter Priors

| Test Class | Tests | What It Validates |
|---|---|---|
| `TestGaussianRandomWalkInit` | 3 | Default/custom fields, TVPPrior subclass |
| `TestHyperparameterNames` | 2 | Naming convention (`{name}_intercept`, `{name}_sigma_delta`, `{name}_delta`) |
| `TestSampleNumpyro` | 12 | Output shapes at resolution 1 and 5, prefix_zero behavior, reproducibility, intercept bounds, sigma positivity, random walk identity, resolution block constancy |

#### `test_likelihoods_jax.py` -- JAX Likelihood Functions

| Test Class | Tests | What It Validates |
|---|---|---|
| `TestTransformEquivalence` | 5 | JAX transforms match NumPy for none/sqrt/log/inverse/boxcox |
| `TestGaussianLikelihood` | 3 | Integrated form matches NumPy, explicit sigma is negative, larger sigma lowers likelihood |
| `TestTransformedGaussianLikelihood` | 3 | Integrated form matches NumPy for sqrt/log/boxcox |
| `TestDifferentiability` | 3 | Finite gradients w.r.t. sigma (Gaussian/transformed) and phi (AR1) |
| `TestAR1` | 2 | phi=0 matches Gaussian, JIT compilation |

---

### 3. Objective Functions (`pyrrm/objectives/tests/`)

#### `test_traditional.py` -- NSE, RMSE, MAE, PBIAS, SDEB

| Test Class | Tests | What It Validates |
|---|---|---|
| `TestNSE` | 7 | Perfect match = 1.0, imperfect < 1.0, NaN handling, transform support, mean-simulation = 0, empty/mismatched errors |
| `TestRMSE` | 3 | Perfect match = 0.0, direction attribute, NRMSE normalization by mean |
| `TestMAE` | 2 | Perfect match = 0.0, direction attribute |
| `TestPBIAS` | 3 | Perfect match = 0.0, positive for overestimate, negative for underestimate |
| `TestSDEB` | 4 | Perfect match = 0.0, direction, component decomposition, custom/invalid alpha |

#### `test_kge.py` -- Kling-Gupta Efficiency Family

| Test Class | Tests | What It Validates |
|---|---|---|
| `TestKGE` | 10 | Perfect match = 1.0, components (r, alpha, beta), bias detection, 3 variants, invalid variant, custom/invalid weights, log transform warning, direction, transform support |
| `TestKGENonParametric` | 4 | Perfect match, Spearman components, direction, robustness to outliers vs standard KGE |

#### `test_transforms.py` -- Flow Transformations

| Test Class | Tests | What It Validates |
|---|---|---|
| `TestFlowTransformation` | 17 | All 8 transform types, flow emphasis, no-transform identity, sqrt correctness, 3 epsilon methods with values, power/boxcox, invalid type/method/value errors, equality, repr, hashability |

#### `test_fdc.py` -- Flow Duration Curve Metrics

| Test Class | Tests | What It Validates |
|---|---|---|
| `TestFDCCurves` | 4 | Sorted descending, Weibull plotting, NaN handling, interpolation, segment extraction |
| `TestFDCMetric` | 10 | All predefined segments, bounds, all metric types, custom bounds, invalid bounds, log transform, perfect match, direction, get_components |

#### `test_signatures.py` -- Hydrological Signatures

| Test Class | Tests | What It Validates |
|---|---|---|
| `TestSignatureMetric` | 5 | All 15 signatures, perfect match = 0% error, percent error formula, get_components, direction, invalid name |
| `TestDynamicsSignatures` | 3 | Flashiness index (constant=0, formula verification), rising/falling limb density |
| `TestWaterBalanceSignatures` | 3 | Baseflow index range [0,1], constant flow BFI=1 |

#### `test_apex.py` -- APEX Objective Function

| Test Class | Tests | What It Validates |
|---|---|---|
| `TestAPEXBasic` | 6 | Perfect = 0, imperfect > 0, direction, NaN handling, empty/mismatched errors, `for_calibration` negation |
| `TestAPEXTransformations` | 6 | Power, sqrt, log, inverse, none, custom FlowTransformation |
| `TestAPEXDynamicsMultiplier` | 4 | Perfect = 1.0, penalizes mismatch, disabled at strength=0, strength effect |
| `TestAPEXLagMultiplier` | 4 | Disabled by default, perfect timing = 1.0, detects shift, increases value |
| `TestAPEXRegimeEmphasis` | 6 | All 5 regime types (uniform, low_flow, high_flow, balanced, extremes), invalid error |
| `TestAPEXBiasMultiplier` | 4 | No-bias = 1.0, with-bias > 1.0, strength/power effects |
| `TestAPEXComponents` | 2 | Component keys completeness, formula consistency |
| `TestAPEXSDEBEquivalence` | 2 | Matches SDEB when dynamics_strength=0, dynamics adds penalty |
| `TestAPEXParameterValidation` | 10 | Invalid alpha, transform_param, bias_strength, bias_power, dynamics_strength, lag_strength, lag_reference, transform |
| `TestAPEXRepr` | 2 | Basic repr includes key params, lag_penalty status |

#### `test_composite.py` -- Composite Objective Functions

| Test Class | Tests | What It Validates |
|---|---|---|
| `TestWeightedObjective` | 9 | weighted_sum/product/min aggregation, weight normalization, get_components, direction, minmax normalization, empty/negative-weight errors, evaluate_individual |
| `TestFactoryFunctions` | 10 | `kge_hilo`, `fdc_multisegment`, `comprehensive_objective`, `nse_multiscale`, `apex_objective` with perfect match, custom weights, KGE variants, evaluate_individual |

---

### 4. Data Handling (`pyrrm/data/tests/`)

#### `test_input_handler.py` -- Column Resolution and Data Loading

| Test Class | Tests | What It Validates |
|---|---|---|
| `TestResolveColumn` | 21 | Exact alias match, canonical name, case-insensitive fallback, missing returns None, raise_on_missing, invalid canonical KeyError, all 15+ alias families (parametrized) |
| `TestColumnAliases` | 3 | Required keys present, canonical name is first alias, no empty alias lists |
| `TestLoadCatchmentData` | 6 | Basic load with column renaming, date slicing, -9999 sentinel replacement, explicit observed column, no-date error, empty-after-merge error |

---

## Test Data Sources

### Synthetic Data Generators

Shared fixtures in `pyrrm/objectives/tests/fixtures.py`:

| Generator | Purpose |
|---|---|
| `generate_test_data(n, seed)` | Lognormal observed + multiplicative-noise simulated |
| `generate_perfect_data(n, seed)` | Identical obs and sim for testing optimal values |
| `generate_with_nan(n, nan_fraction)` | Random NaN insertion for missing-data handling |
| `generate_biased_data(n, bias_factor)` | Constant multiplicative bias |
| `generate_timing_error_data(n, shift)` | Phase-shifted simulation |
| `generate_variable_data(n, var_ratio)` | Different variability ratio |

Model tests use per-file fixtures with fixed seeds:

| Fixture | Description |
|---|---|
| `synthetic_1yr` | 365 days, exponential precip (scale=5), sinusoidal PET, seed=42 |
| `synthetic_10yr` | 3650 days, same distributions |
| `zero_precip` | 365 days zero rainfall, constant PET=3 |
| `storm_burst` | 30 days at 50mm/d then 335 days drought |
| `alternating_wet_dry` | 30-day wet/dry cycles |

### Reference Data

| Source | Location | Description |
|---|---|---|
| C# Sacramento output | `test_data/csharp_output_TC01_default.csv` | Reference implementation comparison |
| Gauge 410734 (Queanbeyan) | `data/410734/` | ~135 years daily rainfall and PET with calibrated Sacramento parameters |

---

## Continuous Integration

Automated tests run via GitHub Actions on every push to `main` and on pull requests. The CI workflow:

- Uses Python 3.11 with core, dev, numba, and jax dependencies
- Runs `pytest pyrrm/ -m "not slow"` (excludes benchmark tests)
- Generates coverage reports via `pytest-cov`

The workflow configuration is at `.github/workflows/tests.yml`.

### Running Benchmarks Locally

Benchmark tests are excluded from CI because they require Numba JIT warm-up and consistent hardware. Run them locally:

```bash
pytest pyrrm/ -m slow -s -v
```

The `-s` flag is important to see the printed performance tables.

---

## Coverage

To generate a coverage report:

```bash
# Terminal report with missing lines
pytest pyrrm/ -m "not slow" --cov=pyrrm --cov-report=term-missing

# HTML report
pytest pyrrm/ -m "not slow" --cov=pyrrm --cov-report=html
open htmlcov/index.html

# XML report (for CI upload)
pytest pyrrm/ -m "not slow" --cov=pyrrm --cov-report=xml
```

---

## Adding New Tests

When adding tests to pyrrm:

1. Place tests in the appropriate `tests/` subdirectory alongside the source module
2. Follow the naming convention `test_<module>.py` with classes `Test<Feature>`
3. Use `@pytest.mark.slow` for tests that take more than a few seconds
4. Use fixed random seeds for reproducibility
5. For numerical comparisons, choose tolerances based on the backend comparison table above
6. Add edge cases: zero input, NaN, empty arrays, invalid parameters
7. For new models, add both kernel-level and class-level equivalence tests

---

## Glossary

| Term | Meaning |
|---|---|
| Backend | Execution engine: pure Python, Numba JIT, JAX XLA, MLX Metal |
| Equivalence test | Verifies two backends produce identical (within tolerance) results |
| Physical validity | Non-negative flows, finite values, mass conservation |
| Parametrized test | Single test function run with multiple parameter sets via `@pytest.mark.parametrize` |
| Monkey-patching | Temporarily replacing `NUMBA_AVAILABLE` to force a specific backend through the class interface |
