# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: pyrrm
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Bayesian Model Averaging for LBG Headwater Catchments
#
# ## Purpose
#
# This notebook applies **Bayesian Model Averaging (BMA)** to the 180
# calibrated model runs from the LBG headwater calibrations (5 gauges,
# 36 experiments each).  It systematically evaluates all five BMA
# combination levels -- from simple equal weighting to flow
# regime-specific Bayesian inference -- and determines which approach
# best captures predictive uncertainty across ACT headwater catchments.
#
# ## What You'll Learn
#
# - How to load `BatchResult` objects and ingest them into the BMA pipeline
# - How the three-step pre-screening reduces 36 models to a manageable ensemble
# - How BMA Levels 1-3 (equal, GRC, stacking) compare without requiring PyMC
# - How to run block temporal cross-validation with water-year boundaries
# - How to fit a full Bayesian mixture model (Level 4) and diagnose MCMC convergence
# - How to generate prediction intervals and evaluate probabilistic calibration (CRPS, PIT)
# - How regime-specific BMA (Level 5) adapts weights across high/medium/low flows
# - How BMA performance varies across the 5 LBG headwater catchments
# - How different CV strategies affect method rankings
#
# ## Prerequisites
#
# - **Headwater calibrations notebook** (`headwater_calibrations.ipynb`) has been run
#   and results saved to `notebooks_ACT/LBG/results/`
# - `pyrrm[bma]` installed: `pip install pyrrm[bma]` (adds PyMC, ArviZ, NumPyro, JAX,
#   scikit-learn).  Levels 1-3 work without these optional dependencies.
#
# ## Estimated Time
#
# - **Levels 1-3 only**: ~5 minutes (no MCMC sampling)
# - **Full pipeline (Levels 1-5, single gauge)**: ~30-60 minutes
# - **Multi-gauge comparison (Step 9)**: ~2-3 hours
#
# ## Steps in This Notebook
#
# | Step | Topic | Description |
# |------|-------|-------------|
# | 1 | Load batch results | Discover and load 5 gauge batch results; summary table |
# | 2 | Single-gauge deep dive | Load prediction matrix for gauge 410734; ensemble spaghetti plot |
# | 3 | Pre-screening | Hard thresholds, residual clustering, regime specialists |
# | 4 | Levels 1-3 weights | Equal weights, GRC, stacking; weight comparison and metrics |
# | 5 | Cross-validation | Block temporal CV across all 5 BMA levels |
# | 6 | Final BMA fit | Full-record Level 4 fit; MCMC diagnostics; posterior weights |
# | 7 | Prediction and uncertainty | Prediction bands, CRPS, PIT histogram, coverage |
# | 8 | Regime-specific BMA | Level 5 fit; sigmoid blending; regime weight heatmap; FDC |
# | 9 | Multi-gauge comparison | BMA pipeline on all 5 gauges; summary heatmap |
# | 10 | CV sensitivity analysis | Compare standard, flood-focused, operational presets |
#
# ## Key Insight
#
# > BMA combines the strengths of many individually-calibrated models into a
# > single probabilistic prediction.  Even simple weighting schemes (Levels 1-2)
# > often outperform the single best model, and full Bayesian inference (Level 4)
# > provides honest uncertainty bands that account for both model and parameter
# > uncertainty.

# %% [markdown]
# ---
# ## Setup

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path
import warnings
import os
import logging

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['figure.dpi'] = 100

from pyrrm.calibration.batch import BatchResult
from pyrrm.bma import BMAConfig, BMARunner, ACT_CV_PRESETS
from pyrrm.bma.data_prep import (
    apply_transform, back_transform, create_cv_splits,
    classify_regime, regime_thresholds,
)
from pyrrm.bma.pre_screen import pre_screen
from pyrrm.bma.level1_equal import equal_weight_predict, equal_weights
from pyrrm.bma.level2_grc import grc_fit, grc_predict
from pyrrm.bma.level3_stacking import stacking_fit, stacking_predict
from pyrrm.bma.evaluation import (
    evaluate_deterministic, evaluate_probabilistic,
    evaluate_by_regime, fdc_error, pit_values, crps_ensemble,
)
from pyrrm.bma.visualization import (
    plot_weight_comparison, plot_prediction_bands,
    plot_pit_histogram, plot_method_comparison,
)

try:
    from pyrrm.bma.level4_bma import (
        build_bma_model, sample_bma, check_convergence, extract_weights,
        PYMC_AVAILABLE,
    )
    from pyrrm.bma.prediction import generate_bma_predictions, back_transform_predictions
    from pyrrm.bma.level5_regime_bma import (
        build_regime_bma, compute_regime_blend_weights,
        regime_blend_predict,
    )
    from pyrrm.bma.visualization import plot_posterior_weights, plot_regime_weights
except ImportError:
    PYMC_AVAILABLE = False

try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
except NameError:
    cwd = Path.cwd()
    if (cwd / 'notebooks_ACT' / 'LBG' / 'results').exists():
        PROJECT_ROOT = cwd
    elif (cwd.parent / 'notebooks_ACT' / 'LBG' / 'results').exists():
        PROJECT_ROOT = cwd.parent
    elif (cwd.parent.parent / 'notebooks_ACT' / 'LBG' / 'results').exists():
        PROJECT_ROOT = cwd.parent.parent
    else:
        PROJECT_ROOT = cwd
RESULTS_DIR = PROJECT_ROOT / 'notebooks_ACT' / 'LBG' / 'results'
REFERENCE_GAUGE = '410734'

logging.basicConfig(level=logging.WARNING)

print("=" * 70)
print("BAYESIAN MODEL AVERAGING — LBG HEADWATER CATCHMENTS")
print("=" * 70)
print(f"\nResults directory: {RESULTS_DIR}")
print(f"PyMC available: {PYMC_AVAILABLE}")
if PYMC_AVAILABLE:
    import pymc as pm
    import arviz as az
    print(f"  PyMC {pm.__version__}, ArviZ {az.__version__}")

# %% [markdown]
# ---
# ## Step 1: Load Batch Results
#
# Each headwater gauge was calibrated with 36 experiments: 2 models (Sacramento,
# GR4J) crossed with 14 SCE-UA objectives and 4 DREAM likelihoods.  Results
# are saved as `BatchResult` pickle files.  We auto-discover them by walking
# the results directory.

# %%
def discover_batch_results(results_dir: Path) -> dict:
    """Walk the results directory and load every batch_result.pkl found."""
    loaded = {}
    for gauge_dir in sorted(results_dir.iterdir()):
        if not gauge_dir.is_dir() or gauge_dir.name.startswith('.'):
            continue
        if gauge_dir.name == 'exports':
            continue
        for run_dir in gauge_dir.iterdir():
            pkl = run_dir / 'batch_result.pkl'
            if pkl.exists():
                br = BatchResult.load(str(pkl))
                loaded[gauge_dir.name] = br
                break
    return loaded

batch_results = discover_batch_results(RESULTS_DIR)

rows = []
for gauge_id, br in batch_results.items():
    first_key = next(iter(br.results))
    ref = br.results[first_key]
    rows.append({
        'Gauge': gauge_id,
        'Experiments': len(br.results),
        'Failures': len(br.failures),
        'Start': ref.dates[0].strftime('%Y-%m-%d'),
        'End': ref.dates[-1].strftime('%Y-%m-%d'),
        'Days': len(ref.dates),
    })

summary_df = pd.DataFrame(rows)
print(f"Loaded {len(batch_results)} gauges:\n")
print(summary_df.to_string(index=False))

# %% [markdown]
# Each gauge has 36 successful experiments with no failures.  Record lengths
# range from ~28 years (gauge 410790) to ~94 years (gauge 410705).  All
# experiments within a gauge share the same date index and observed flow,
# which is exactly what the BMA pipeline requires.

# %% [markdown]
# ---
# ## Step 2: Single-Gauge Deep Dive (Gauge 410734)
#
# We use gauge **410734** (Queanbeyan River at Tinderry) as the worked example
# for Steps 2-8.  With ~57 years of daily data and 36 calibrated models, it
# provides a rich ensemble for BMA evaluation.

# %%
br_ref = batch_results[REFERENCE_GAUGE]
config = BMAConfig.from_preset('standard', transform='none', random_seed=42)

runner = BMARunner(config)
runner.load_data(batch_result=br_ref)

print(f"Gauge {REFERENCE_GAUGE}")
print(f"  Prediction matrix: {runner.F_raw.shape}  (timesteps, models)")
print(f"  Date range: {runner.dates[0].date()} to {runner.dates[-1].date()}")
print(f"\nModel names:")
for i, name in enumerate(runner.F_raw.columns):
    print(f"  {i+1:2d}. {name}")

# %% [markdown]
# The prediction matrix has 36 columns — one per calibrated experiment.  Let's
# visualise the raw ensemble spread against the observed flow.  The width of
# the spaghetti bundle shows where models agree and where they diverge.

# %%
fig, ax = plt.subplots(figsize=(14, 5))

F_arr = runner.F_raw.values
dates = runner.dates
y_obs = runner.y_obs.values

for k in range(F_arr.shape[1]):
    ax.plot(dates, F_arr[:, k], linewidth=0.3, alpha=0.4)

ax.plot(dates, y_obs, 'k-', linewidth=0.6, alpha=0.9, label='Observed')
ax.set_ylabel('Flow (mm/d)')
ax.set_xlabel('Date')
ax.set_title(f'Gauge {REFERENCE_GAUGE} — 36-Model Ensemble vs Observed')
ax.legend(fontsize=9, loc='upper right')
fig.tight_layout()
plt.show()

# %% [markdown]
# The ensemble fans out during high-flow events — exactly where model
# uncertainty is greatest and BMA is most valuable.

# %% [markdown]
# ---
# ## Step 3: Pre-Screening
#
# Before fitting BMA weights, we reduce the 36 models to a smaller set using
# three steps:
#
# | Step | Method | Purpose |
# |------|--------|---------|
# | 1 | Hard thresholds | Remove models with NSE < 0, KGE < -0.41, or |PBIAS| > 25% |
# | 2 | Residual clustering | Collapse redundant models (correlated residuals) |
# | 3 | Regime specialists | Add back models that excel for a specific flow regime |

# %%
runner.pre_screen()

print(f"\nPre-screening: {runner.F_raw.shape[1]} → {len(runner.kept_models)} models")
print(f"\nSurviving models:")
for i, name in enumerate(runner.kept_models):
    print(f"  {i+1:2d}. {name}")

# %% [markdown]
# ### 3.1 Screening Results
#
# The hard thresholds remove any model that performs worse than the mean of
# observations (NSE < 0).  Residual clustering then collapses models whose
# errors are highly correlated — for example, Sacramento calibrated with
# KGE and KGE-sqrt will produce nearly identical residual patterns.

# %% [markdown]
# ### 3.2 Residual Correlation
#
# The heatmap below shows pairwise Pearson correlation of residuals among the
# surviving models.  High values indicate redundant information — the BMA
# weighting will naturally down-weight these.

# %%
if runner._corr_matrix is not None and len(runner.kept_models) > 1:
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(runner._corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    n = len(runner.kept_models)
    short_names = [m.split('_', 1)[-1] if '_' in m else m for m in runner.kept_models]
    ax.set_xticks(range(n))
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(short_names, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8, label='Residual correlation')
    ax.set_title(f'Gauge {REFERENCE_GAUGE} — Residual Correlation Matrix (Post-Screening)')
    fig.tight_layout()
    plt.show()

# %% [markdown]
# Models calibrated with similar objectives (e.g., KGE variants) cluster
# together with high residual correlations.  The pre-screening step collapsed
# these into representative members, leaving a more diverse ensemble for BMA.

# %% [markdown]
# ---
# ## Step 4: Levels 1-3 (Core Dependencies)
#
# These three methods require only NumPy and SciPy — no PyMC needed.  They
# provide quick baselines against which the full Bayesian methods are compared.

# %%
F_s = runner.F_screened
y_s = runner.y_obs.values
model_names = runner.kept_models
K = F_s.shape[1]

# %% [markdown]
# ### 4.1 Equal Weights (Level 1)
#
# The simplest combination: every surviving model gets weight `1/K`.  This is
# the baseline that every more complex method must beat.

# %%
w_equal = equal_weights(K)
pred_equal = equal_weight_predict(F_s)
metrics_equal = evaluate_deterministic(y_s, pred_equal)

print("Level 1 — Equal Weights")
print(f"  Weights: 1/{K} = {1/K:.4f} for each model")
for metric, val in metrics_equal.items():
    if isinstance(val, float):
        print(f"  {metric}: {val:.4f}")

# %% [markdown]
# ### 4.2 GRC (Level 2)
#
# Granger-Ramanathan Combination fits constrained weights (non-negative,
# sum to 1) by minimising mean squared error.  This is a convex optimisation
# problem with a unique global solution.

# %%
w_grc = grc_fit(F_s, y_s)
pred_grc = grc_predict(F_s, w_grc)
metrics_grc = evaluate_deterministic(y_s, pred_grc)

print("Level 2 — GRC (Constrained Regression)")
print(f"  Non-zero weights: {(w_grc > 0.01).sum()} / {K}")
for metric, val in metrics_grc.items():
    if isinstance(val, float):
        print(f"  {metric}: {val:.4f}")

# %% [markdown]
# ### 4.3 Bayesian Stacking (Level 3)
#
# Stacking maximises the cross-validated log predictive density.  Unlike
# GRC which fits to the full dataset, stacking uses out-of-sample performance
# to set weights — reducing overfitting to the calibration period.

# %%
cv_splits = create_cv_splits(runner.dates, config, y_s)
w_stack = stacking_fit(F_s, y_s, cv_splits)
pred_stack = stacking_predict(F_s, w_stack)
metrics_stack = evaluate_deterministic(y_s, pred_stack)

print(f"Level 3 — Bayesian Stacking ({len(cv_splits)}-fold CV)")
print(f"  Non-zero weights: {(w_stack > 0.01).sum()} / {K}")
for metric, val in metrics_stack.items():
    if isinstance(val, float):
        print(f"  {metric}: {val:.4f}")

# %% [markdown]
# The weight comparison plot shows how the three methods distribute weight
# across models.  GRC and stacking tend to concentrate weight on fewer models
# than equal weighting.

# %%
weights_dict = {
    'L1: Equal': w_equal,
    'L2: GRC': w_grc,
    'L3: Stacking': w_stack,
}
fig = plot_weight_comparison(model_names, weights_dict)
fig.suptitle(f'Gauge {REFERENCE_GAUGE} — Weight Comparison (Levels 1-3)', y=1.02)
plt.show()

# %% [markdown]
# Summary of Levels 1-3 deterministic metrics:

# %%
l13_df = pd.DataFrame({
    'L1: Equal': metrics_equal,
    'L2: GRC': metrics_grc,
    'L3: Stacking': metrics_stack,
}).T
numeric_cols = l13_df.select_dtypes(include=[np.number]).columns[:6]
print(l13_df[numeric_cols].to_string(float_format='{:.4f}'.format))

# %% [markdown]
# ---
# ## Step 5: Cross-Validation (Levels 1-3)
#
# Cross-validation answers one question: **which BMA weighting method should
# we trust for out-of-sample prediction?**  When we fit weights on the full
# record every method looks good because it sees all the data.  CV holds out
# contiguous time blocks, fits weights on the rest, then scores the held-out
# period — revealing whether learned weights actually generalise.
#
# We restrict CV to **Levels 1-3** (equal, GRC, stacking) because:
#
# - They run in seconds — no MCMC sampling required.
# - Levels 4-5 require a full MCMC run *per fold*, which takes hours and
#   often fails to converge on fold-sized subsets (the Dirichlet mixture
#   posterior becomes multimodal when data is limited).
# - The BMA literature typically cross-validates the simpler methods and
#   validates Levels 4-5 via MCMC diagnostics and probabilistic calibration
#   (CRPS, PIT) on the full-record fit (Steps 6-8).
#
# The `"standard"` preset uses:
#
# | Parameter | Value |
# |-----------|-------|
# | Strategy | Block temporal |
# | Year boundary | Water year (July) |
# | Block size | 2 years |
# | Buffer | 60 days |

# %%
config_cv = BMAConfig.from_preset(
    'standard',
    transform='none',
    random_seed=42,
)

runner_cv = BMARunner(config_cv)
runner_cv.load_data(batch_result=br_ref)
runner_cv.pre_screen()

F_cv = runner_cv.F_screened
y_cv = runner_cv.y_obs.values
cv_splits = create_cv_splits(runner_cv.dates, config_cv, y_cv)

print(f"Cross-validation for gauge {REFERENCE_GAUGE}")
print(f"  Models after screening: {F_cv.shape[1]}")
print(f"  CV folds: {len(cv_splits)}")

from pyrrm.bma.data_prep import apply_transform, back_transform

F_t_cv, y_t_cv, tp_cv = apply_transform(F_cv, y_cv, config_cv)

cv_all = {'L1: Equal': [], 'L2: GRC': [], 'L3: Stacking': []}

for fold_i, (train_idx, val_idx) in enumerate(cv_splits):
    F_tr, F_va = F_t_cv[train_idx], F_t_cv[val_idx]
    y_tr, y_va = y_t_cv[train_idx], y_t_cv[val_idx]
    y_va_orig = back_transform(y_va, tp_cv)

    pred_eq = back_transform(equal_weight_predict(F_va), tp_cv)
    cv_all['L1: Equal'].append(evaluate_deterministic(y_va_orig, pred_eq))

    w_g = grc_fit(F_tr, y_tr)
    pred_g = back_transform(grc_predict(F_va, w_g), tp_cv)
    cv_all['L2: GRC'].append(evaluate_deterministic(y_va_orig, pred_g))

    try:
        inner_splits = create_cv_splits(runner_cv.dates[train_idx], config_cv)
        w_s = stacking_fit(F_tr, y_tr, inner_splits)
        pred_s = back_transform(stacking_predict(F_va, w_s), tp_cv)
        cv_all['L3: Stacking'].append(evaluate_deterministic(y_va_orig, pred_s))
    except ValueError:
        cv_all['L3: Stacking'].append(cv_all['L1: Equal'][-1])

    print(f"  Fold {fold_i+1}/{len(cv_splits)} done")

cv_rows = []
for method, fold_results in cv_all.items():
    keys = fold_results[0].keys()
    avg = {}
    for k in keys:
        vals = [r[k] for r in fold_results if k in r and isinstance(r.get(k), (int, float)) and not np.isnan(r.get(k, np.nan))]
        avg[k] = float(np.mean(vals)) if vals else np.nan
    cv_rows.append({'Method': method, **avg})

cv_results = pd.DataFrame(cv_rows).set_index('Method')

# %%
print(f"\nCross-Validation Results — Gauge {REFERENCE_GAUGE} (Levels 1-3)\n")
numeric_cv = cv_results.select_dtypes(include=[np.number])
print(numeric_cv.to_string(float_format='{:.4f}'.format))

# %% [markdown]
# The method comparison heatmap visualises all metrics side by side.  Green
# cells are better (higher NSE/KGE, lower RMSE/CRPS).  Note that Levels 4-5
# are not included here — their quality is assessed via MCMC diagnostics and
# probabilistic calibration in Steps 6-8.

# %%
fig = plot_method_comparison(cv_results)
fig.suptitle(f'Gauge {REFERENCE_GAUGE} — Cross-Validated Method Comparison (L1-L3)', y=1.02)
plt.show()

# %% [markdown]
# ---
# ## Step 6: Final BMA Fit (Level 4) and Diagnostics
#
# We now fit the full Bayesian mixture model on the entire record to obtain
# production weights.  This uses the NUTS sampler to explore the posterior
# distribution of Dirichlet weights, bias corrections, and residual variances.

# %%
if PYMC_AVAILABLE:
    config_final = BMAConfig.from_preset(
        'standard',
        transform='none',
        random_seed=42,
        draws=2000,
        tune=3000,
        chains=4,
        target_accept=0.95,
        nuts_sampler='numpyro',
        init='jitter+adapt_diag',
        dirichlet_alpha='uniform',
        use_manual_loglik=False,
    )

    runner_final = BMARunner(config_final)
    runner_final.load_data(batch_result=br_ref)
    runner_final.pre_screen()

    print("Fitting final BMA on the full record...")
    print(f"  draws={config_final.draws}, tune={config_final.tune}, chains={config_final.chains}")
    runner_final.fit_final()
    print("Done.")
else:
    print("PyMC not available — skipping Level 4 final fit.")
    print("Install with: pip install 'pyrrm[bma]'")

# %% [markdown]
# ### 6.1 Convergence Diagnostics
#
# Good MCMC convergence requires R-hat < 1.01, bulk ESS > 400, and zero
# divergences.  Issues here indicate the sampler needs more tuning iterations
# or a reparameterisation.

# %%
if PYMC_AVAILABLE and runner_final.final_idata is not None:
    issues = check_convergence(runner_final.final_idata)
    if issues:
        print("Convergence issues detected:")
        for key, val in issues.items():
            print(f"  {key}: {val}")
    else:
        print("All convergence diagnostics passed (R-hat < 1.01, ESS > 400, 0 divergences).")

# %% [markdown]
# ### 6.2 Posterior Weights
#
# The violin plot shows the full posterior distribution of BMA weights.
# Narrow violins indicate well-identified weights; wide ones indicate
# the data cannot clearly distinguish between models.

# %%
if PYMC_AVAILABLE and runner_final.final_idata is not None:
    fig = plot_posterior_weights(runner_final.final_idata, runner_final.kept_models)
    fig.suptitle(f'Gauge {REFERENCE_GAUGE} — BMA Posterior Weight Distributions', y=1.02)
    plt.show()

    weights_df = extract_weights(runner_final.final_idata, runner_final.kept_models)
    print("\nPosterior Weight Summary:\n")
    print(weights_df.to_string(float_format='{:.4f}'.format))

# %% [markdown]
# ---
# ## Step 7: Prediction and Uncertainty Bands
#
# Posterior predictive samples are drawn from the fitted Dirichlet-weighted
# Gaussian mixture.  Each sample picks a model component according to the
# posterior weights and adds the estimated residual noise, producing an
# honest uncertainty envelope.

# %%
if PYMC_AVAILABLE and runner_final.final_idata is not None:
    F_t, y_t, tparams = apply_transform(
        runner_final.F_screened, runner_final.y_obs.values, config_final,
    )
    preds = generate_bma_predictions(
        runner_final.final_idata, F_t, config_final, n_samples=4000,
    )
    preds_orig = back_transform_predictions(preds, tparams)

    mask_window = (runner_final.dates >= '2015-01-01') & (runner_final.dates <= '2020-12-31')
    idx_window = np.where(mask_window)[0]

    if len(idx_window) > 0:
        pred_window = {
            'mean': preds_orig['mean'][idx_window],
            'median': preds_orig['median'][idx_window],
            'intervals': {
                level: (lo[idx_window], hi[idx_window])
                for level, (lo, hi) in preds_orig['intervals'].items()
            },
        }
        fig = plot_prediction_bands(
            runner_final.dates[idx_window],
            runner_final.y_obs.values[idx_window],
            pred_window,
            title=f'Gauge {REFERENCE_GAUGE} — BMA Predictions (2015-2020)',
        )
        plt.show()

# %% [markdown]
# ### 7.1 Probabilistic Metrics
#
# - **CRPS** (Continuous Ranked Probability Score): lower is better; measures
#   both calibration and sharpness in a single score.
# - **PIT** (Probability Integral Transform): should be uniform if the
#   predictive distribution is well-calibrated.
# - **Coverage**: fraction of observations within prediction intervals —
#   should match the nominal level (e.g., 90% PI should cover ~90%).

# %%
if PYMC_AVAILABLE and runner_final.final_idata is not None:
    prob_metrics = evaluate_probabilistic(
        runner_final.y_obs.values, preds_orig,
    )
    print("Probabilistic Metrics:\n")
    for key, val in prob_metrics.items():
        print(f"  {key}: {val:.4f}")

    pit = pit_values(runner_final.y_obs.values, preds_orig['samples'])
    fig = plot_pit_histogram(pit)
    fig.suptitle(f'Gauge {REFERENCE_GAUGE} — PIT Histogram', y=1.02)
    plt.show()

# %% [markdown]
# A uniform PIT histogram indicates well-calibrated predictive uncertainty.
# U-shaped histograms suggest underdispersion (intervals too narrow); dome
# shapes suggest overdispersion (intervals too wide).

# %% [markdown]
# ---
# ## Step 8: Regime-Specific BMA (Level 5)
#
# Flow regimes — high, medium, low — exhibit different error structures.
# A model that excels during floods may perform poorly during baseflow.
# Level 5 fits separate BMA models per regime and blends them with smooth
# sigmoid transitions so the combined prediction has no discontinuities.

# %%
if PYMC_AVAILABLE and runner_final.final_idata is not None:
    F_t, y_t, tparams = apply_transform(
        runner_final.F_screened, runner_final.y_obs.values, config_final,
    )

    regime_masks = classify_regime(y_t, config_final)
    for regime, mask in regime_masks.items():
        print(f"  {regime:>8s}: {mask.sum():,d} timesteps ({mask.mean()*100:.1f}%)")

    q_high, q_low = regime_thresholds(y_t, config_final)
    print(f"\n  Thresholds: q_high={q_high:.3f}, q_low={q_low:.3f}")

# %% [markdown]
# Now we fit separate BMA models for each flow regime.  Regimes with fewer
# than 50 timesteps are skipped (they fall back to the global BMA).

# %%
if PYMC_AVAILABLE and runner_final.final_idata is not None:
    config_regime = BMAConfig.from_preset(
        'standard',
        transform='none',
        random_seed=42,
        draws=2000,
        tune=3000,
        chains=4,
        target_accept=0.95,
        init='jitter+adapt_diag',
        nuts_sampler='numpyro',
        dirichlet_alpha='uniform',
        use_manual_loglik=False,
    )

    print("Fitting regime-specific BMA models...")
    regime_results = build_regime_bma(F_t, y_t, regime_masks, config_regime)
    print(f"  Fitted regimes: {list(regime_results.keys())}")

# %% [markdown]
# The regime weight heatmap shows how model preferences shift across flow
# conditions.  For example, models calibrated with log-transformed objectives
# often dominate in the low-flow regime.

# %%
if PYMC_AVAILABLE and runner_final.final_idata is not None and regime_results:
    fig = plot_regime_weights(runner_final.kept_models, regime_results)
    fig.suptitle(f'Gauge {REFERENCE_GAUGE} — BMA Weights by Flow Regime', y=1.02)
    plt.show()

# %% [markdown]
# ### 8.1 Regime Metrics Comparison
#
# We compare the global BMA (Level 4) against regime-specific BMA (Level 5)
# for each flow condition separately.

# %%
if PYMC_AVAILABLE and runner_final.final_idata is not None and regime_results:
    regime_masks_orig = classify_regime(runner_final.y_obs.values, config_final)

    l4_by_regime = evaluate_by_regime(
        runner_final.y_obs.values, preds_orig['mean'], preds_orig['samples'],
        regime_masks_orig,
    )

    blend_w_range = config_final.regime_blend_width * (q_high - q_low)
    flow_proxy = F_t.mean(axis=1)
    blend_weights = compute_regime_blend_weights(flow_proxy, q_high, q_low, blend_w_range)
    regime_preds = regime_blend_predict(
        F_t, regime_results, blend_weights, config_regime,
    )
    regime_preds_orig = back_transform_predictions(regime_preds, tparams)

    l5_by_regime = evaluate_by_regime(
        runner_final.y_obs.values, regime_preds_orig['mean'],
        regime_preds_orig['samples'], regime_masks_orig,
    )

    print("Regime-Specific Metrics — Level 4 (Global) vs Level 5 (Regime)\n")
    for regime in ['high', 'medium', 'low']:
        if regime in l4_by_regime and regime in l5_by_regime:
            print(f"  {regime.upper()} flow:")
            for metric in ['NSE', 'KGE', 'RMSE']:
                v4 = l4_by_regime[regime].get(metric, float('nan'))
                v5 = l5_by_regime[regime].get(metric, float('nan'))
                better = '←' if abs(v4) >= abs(v5) and metric != 'RMSE' else '→'
                if metric == 'RMSE':
                    better = '←' if v4 <= v5 else '→'
                print(f"    {metric:>6s}: L4={v4:8.4f}  L5={v5:8.4f}  {better}")
            print()

# %% [markdown]
# ### 8.2 FDC Error Comparison
#
# Flow Duration Curve errors by exceedance-probability segment reveal where
# each method struggles.  Lower RMSE values are better.

# %%
if PYMC_AVAILABLE and runner_final.final_idata is not None:
    fdc_l4 = fdc_error(runner_final.y_obs.values, preds_orig['mean'])
    fdc_l5 = fdc_error(runner_final.y_obs.values, regime_preds_orig['mean']) if regime_results else {}

    if fdc_l4:
        fdc_df = pd.DataFrame({'L4 Global': fdc_l4, 'L5 Regime': fdc_l5}).T if fdc_l5 else pd.DataFrame({'L4 Global': fdc_l4}).T
        print("FDC RMSE by Exceedance Segment:\n")
        print(fdc_df.to_string(float_format='{:.3f}'.format))

# %% [markdown]
# ---
# ## Step 9: Multi-Gauge Comparison
#
# We now scale up the BMA pipeline across all 5 gauges.  For computational
# tractability we focus on Levels 1-3 (no MCMC) and use the `"standard"`
# CV preset.

# %%
config_multi = BMAConfig.from_preset(
    'standard',
    transform='none',
    random_seed=42,
    buffer_days=60,
    min_train_years=3.0,
    min_val_years=1.0,
)

multi_results = {}
for gauge_id, br in batch_results.items():
    print(f"\n{'='*50}")
    print(f"Gauge {gauge_id}")
    print(f"{'='*50}")

    r = BMARunner(config_multi)
    r.load_data(batch_result=br)
    r.pre_screen()

    F_s = r.F_screened
    y_s = r.y_obs.values
    K_g = F_s.shape[1]

    w_eq = equal_weights(K_g)
    pred_eq = equal_weight_predict(F_s)
    m_eq = evaluate_deterministic(y_s, pred_eq)

    w_grc_g = grc_fit(F_s, y_s)
    pred_grc_g = grc_predict(F_s, w_grc_g)
    m_grc = evaluate_deterministic(y_s, pred_grc_g)

    try:
        splits_g = create_cv_splits(r.dates, config_multi, y_s)
        w_st_g = stacking_fit(F_s, y_s, splits_g)
        pred_st_g = stacking_predict(F_s, w_st_g)
        m_stack = evaluate_deterministic(y_s, pred_st_g)
    except ValueError as e:
        print(f"  Stacking CV failed: {e}")
        m_stack = m_eq.copy()

    multi_results[gauge_id] = {
        'L1: Equal': m_eq,
        'L2: GRC': m_grc,
        'L3: Stacking': m_stack,
        'n_models': K_g,
    }
    print(f"  Models after screening: {K_g}")
    for method in ['L1: Equal', 'L2: GRC', 'L3: Stacking']:
        kge = multi_results[gauge_id][method].get('KGE', float('nan'))
        print(f"  {method}: KGE={kge:.4f}")

# %% [markdown]
# The summary heatmap compares KGE values across gauges and methods.

# %%
kge_rows = []
for gauge_id, methods in multi_results.items():
    row = {'Gauge': gauge_id}
    for method in ['L1: Equal', 'L2: GRC', 'L3: Stacking']:
        row[method] = methods[method].get('KGE', np.nan)
    kge_rows.append(row)

kge_df = pd.DataFrame(kge_rows).set_index('Gauge')
print("KGE Summary — All Gauges × Methods:\n")
print(kge_df.to_string(float_format='{:.4f}'.format))

# %%
fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(kge_df.values, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
ax.set_xticks(range(len(kge_df.columns)))
ax.set_xticklabels(kge_df.columns, fontsize=9)
ax.set_yticks(range(len(kge_df.index)))
ax.set_yticklabels(kge_df.index, fontsize=9)

for i in range(len(kge_df.index)):
    for j in range(len(kge_df.columns)):
        val = kge_df.iloc[i, j]
        if not np.isnan(val):
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=9)

fig.colorbar(im, ax=ax, shrink=0.8, label='KGE')
ax.set_title('Multi-Gauge BMA Comparison — KGE (Levels 1-3)')
fig.tight_layout()
plt.show()

# %% [markdown]
# The best method per gauge:

# %%
for gauge_id in kge_df.index:
    best_method = kge_df.loc[gauge_id].idxmax()
    best_val = kge_df.loc[gauge_id].max()
    print(f"  {gauge_id}: {best_method} (KGE={best_val:.4f})")

# %% [markdown]
# GRC and Stacking typically outperform equal weighting, confirming that
# learned weights add value.  The relative ranking of GRC vs Stacking can
# vary by catchment — stacking's cross-validated approach helps most when
# the ensemble contains models with very different out-of-sample skill.

# %% [markdown]
# ---
# ## Step 10: CV Sensitivity Analysis
#
# The choice of cross-validation strategy affects fold structure and can
# change method rankings.  We compare three ACT presets on gauge 410734:
#
# | Preset | Year start | Block size | Buffer | Strategy |
# |--------|------------|-----------|--------|----------|
# | `standard` | July (water year) | 2 years | 60 days | Block |
# | `flood_focused` | April | 1 year | 45 days | Block |
# | `operational` | July (water year) | 1 year | 60 days | Expanding window |

# %%
cv_sensitivity = {}
br_sens = batch_results[REFERENCE_GAUGE]

for preset_name in ['standard', 'flood_focused', 'operational']:
    print(f"\n--- Preset: {preset_name} ---")
    cfg = BMAConfig.from_preset(
        preset_name,
        transform='none',
        random_seed=42,
        min_train_years=2.0,
        min_val_years=0.5,
    )

    r = BMARunner(cfg)
    r.load_data(batch_result=br_sens)
    r.pre_screen()

    F_s = r.F_screened
    y_s = r.y_obs.values
    K_p = F_s.shape[1]

    m_eq = evaluate_deterministic(y_s, equal_weight_predict(F_s))

    w_grc_p = grc_fit(F_s, y_s)
    m_grc = evaluate_deterministic(y_s, grc_predict(F_s, w_grc_p))

    try:
        splits_p = create_cv_splits(r.dates, cfg, y_s)
        w_st_p = stacking_fit(F_s, y_s, splits_p)
        m_stack = evaluate_deterministic(y_s, stacking_predict(F_s, w_st_p))
        n_folds = len(splits_p)
    except ValueError as e:
        print(f"  CV failed: {e}")
        m_stack = m_eq.copy()
        n_folds = 0

    cv_sensitivity[preset_name] = {
        'n_folds': n_folds,
        'L1: Equal': m_eq,
        'L2: GRC': m_grc,
        'L3: Stacking': m_stack,
    }
    print(f"  Folds: {n_folds}, Models: {K_p}")
    for method in ['L1: Equal', 'L2: GRC', 'L3: Stacking']:
        kge_v = cv_sensitivity[preset_name][method].get('KGE', float('nan'))
        print(f"  {method}: KGE={kge_v:.4f}")

# %%
sens_rows = []
for preset, methods in cv_sensitivity.items():
    for method in ['L1: Equal', 'L2: GRC', 'L3: Stacking']:
        sens_rows.append({
            'Preset': preset,
            'Method': method,
            'Folds': methods['n_folds'],
            'NSE': methods[method].get('NSE', np.nan),
            'KGE': methods[method].get('KGE', np.nan),
            'RMSE': methods[method].get('RMSE', np.nan),
        })

sens_df = pd.DataFrame(sens_rows)
print(f"\nCV Sensitivity — Gauge {REFERENCE_GAUGE}\n")
print(sens_df.to_string(index=False, float_format='{:.4f}'.format))

# %% [markdown]
# The `standard` preset (water-year blocks) is recommended for ACT water
# resource planning since it aligns with the reporting cycle.  The
# `flood_focused` preset with April boundaries captures the full
# autumn-winter flood season within validation blocks, while `operational`
# (expanding window) mimics real-time forecasting where we never validate
# on data earlier than the training set.

# %% [markdown]
# ---
# ## Summary
#
# This notebook applied BMA to 180 calibrated model runs across 5 LBG
# headwater catchments:
#
# | Level | Method | Requires PyMC? | Typical Use Case |
# |-------|--------|----------------|------------------|
# | 1 | Equal weights | No | Quick baseline |
# | 2 | GRC (constrained regression) | No | Operational point prediction |
# | 3 | Bayesian stacking | No | Cross-validated point prediction |
# | 4 | Global BMA (Dirichlet mixture) | Yes | Full probabilistic prediction |
# | 5 | Regime-specific BMA | Yes | Heterogeneous catchment behaviour |
#
# ### Key Takeaways
#
# 1. **Pre-screening** reduced 36 experiments to a compact, diverse ensemble
#    by removing poor models and collapsing redundant ones.
# 2. **GRC and stacking** (Levels 2-3) consistently outperform equal weighting,
#    confirming that learned weights add value at negligible computational cost.
# 3. **Global BMA** (Level 4) provides calibrated uncertainty bands with CRPS
#    and PIT diagnostics — essential for probabilistic flood and drought
#    assessment.
# 4. **Regime-specific BMA** (Level 5) can improve predictions in specific
#    flow regimes but requires more data and may not generalise to short records.
# 5. **CV strategy matters**: water-year boundaries are recommended for ACT
#    catchments; expanding windows are best for operational forecasting.
# 6. **Multi-gauge consistency**: Levels 2-3 rankings are stable across the
#    5 LBG headwater catchments, suggesting the approach generalises well.
# 7. **Next steps**: try pre-screening sensitivity, add GR5J/GR6J models,
#    and run full MCMC (Level 4) across all gauges for probabilistic comparison.
