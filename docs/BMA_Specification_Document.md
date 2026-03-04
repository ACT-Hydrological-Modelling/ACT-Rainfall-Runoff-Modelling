# Bayesian Model Averaging for Multi-Model Hydrological Ensembles
## Theory, Hierarchy of Methods, and PyMC Implementation Specification

---

## 1. Plain-Language Guide: What Each Method Actually Does

Before diving into mathematics and code, here is an intuitive explanation of the five combination approaches, ordered from simplest to most sophisticated. Think of your 36 calibrated models as 36 different weather forecasters, each with their own strengths and weaknesses.

### Level 1: Equal Weights — "Everyone gets an equal vote"

This is the simplest possible combination. You take the prediction from each of your 36 models at every timestep and just average them. Every model contributes equally, whether it's brilliant or terrible.

**Why it works surprisingly well:** When models make independent errors, those errors tend to cancel out in the average. If model A overshoots a peak by 20 m³/s and model B undershoots it by 20 m³/s, their average is dead on. This cancellation effect is remarkably powerful and is why equal weights is a hard baseline to beat — a phenomenon known as the "forecast combination puzzle."

**Where it fails:** If you have 30 mediocre models and 6 excellent ones, the mediocre models drown out the good ones. And in your case, with a factorial design producing many similar models (e.g., several GR4J variants that all make similar errors), you are implicitly giving more total weight to GR4J than to Sacramento, not because GR4J is better but because you have more versions of it.

**What you get:** A single point prediction (the average). No uncertainty estimate.

### Level 2: GRC — "Let the data pick who matters, via regression"

GRC stands for "constrained Granger-Ramanathan combination." The idea is simple: treat the observed flow as the thing you want to predict, treat the 36 model predictions as 36 predictor variables, and find the best weighted combination using regression. The constraints are that all weights must be non-negative and must sum to 1.

In practice this means solving: "what set of weights, when applied to my 36 model predictions, produces a combined prediction that is closest to the observed flow (in a least-squares sense)?"

**The key insight vs. equal weights:** GRC lets the data tell you which models deserve more influence. A model that consistently gets the timing and magnitude right will receive a high weight. A model that's always off will get a weight near zero. Unlike equal weights, GRC can effectively ignore bad models.

**Where it fails:** GRC uses the same fixed weights for every timestep. A model that's great at predicting flood peaks but terrible at low flows gets one compromise weight that's suboptimal for both situations. It also can't tell you anything about uncertainty — you get a single combined prediction with no error bars.

**What you get:** A single point prediction with optimized weights, plus a weight vector telling you which models matter.

### Level 3: Bayesian Stacking — "GRC, but smarter about overfitting"

Stacking is conceptually similar to GRC but uses cross-validation to prevent overfitting. Instead of fitting weights on the same data you evaluate them on, stacking leaves out a chunk of data, fits the models on the rest, then measures how well each model predicts the left-out chunk. It repeats this for every chunk, then finds weights that maximize the combined prediction quality across all the left-out chunks.

**Why this matters:** With 36 models and limited data, GRC can overfit — it might find a weight combination that works perfectly on the training data but falls apart on new data. Stacking protects against this by always testing on data the weights haven't seen.

**The deeper difference:** Stacking is also theoretically better suited to your situation. Classical BMA (below) assumes that one of your 36 models is the "true" model and tries to figure out which one. Stacking makes no such assumption — it just finds the combination that predicts best, period. Since none of your conceptual models is actually the true rainfall-runoff process, stacking's assumption-free approach is often more appropriate. This is what statisticians call the "M-open" setting (Yao et al. 2018).

**What you get:** A single point prediction with cross-validated weights. Still no built-in uncertainty estimate, though you can bootstrap it.

### Level 4: BMA — "Full probabilistic combination with uncertainty"

BMA (Bayesian Model Averaging) is fundamentally different from the three methods above because it doesn't just combine point predictions — it combines entire probability distributions. Each of your 36 models doesn't just say "I predict 50 m³/s." Instead, it says "I predict the flow is normally distributed around 50 m³/s with a standard deviation of 8 m³/s." BMA then combines these 36 bell curves into one mixture distribution, weighted by how reliable each model has been historically.

The combined prediction is not a single number; it's a probability distribution for every timestep. From this distribution you can extract:

- The most likely flow value (the mean or median)
- A 90% prediction interval ("I'm 90% confident the flow will be between 30 and 75 m³/s")
- A full PDF that might be multi-modal if some models predict very different values
- A decomposition showing how much uncertainty comes from models disagreeing vs. each model being individually noisy

**How weights are determined:** BMA uses Bayesian inference. You start with a prior belief about the weights (typically: "I don't know, so maybe they're all equal") and update that belief based on the data. The update uses Bayes' theorem: models that assign higher probability to what actually happened get rewarded with more weight. This is done via MCMC sampling in PyMC, which gives you not just point estimates of the weights but full posterior distributions — so you know not only that model A's weight is about 0.15, but that it's somewhere between 0.10 and 0.20 with 94% probability.

**Where it shines:** BMA is the only method here that gives you calibrated uncertainty. If your BMA 90% prediction interval actually contains 90% of observations, you have a trustworthy probabilistic forecast — invaluable for flood risk analysis, reservoir operations, or any decision that depends on "how bad could it get?"

**Where it can struggle:** BMA is computationally expensive (MCMC sampling for 15+ weights takes time), can be sensitive to the choice of likelihood (Gaussian vs. Gamma vs. others), and classical BMA has a theoretical tendency to collapse onto a single model in large datasets rather than maintaining a diverse blend.

**What you get:** A full predictive distribution at every timestep, including means, medians, arbitrary quantiles, prediction intervals, and posterior weight distributions with uncertainty.

### Level 5: Flow Regime-Specific BMA — "Different experts for different situations"

This is the most sophisticated approach, and the one most likely to produce the biggest improvement for hydrology specifically. The core idea: instead of finding one set of weights that applies everywhere, allow the weights to change depending on the flow conditions.

Think about it this way. Looking at your clustermap, the Sacramento model calibrated on NSE of log-flows is probably excellent at reproducing low flows (because log-transforming amplifies the importance of small values), but it might underestimate peaks. Meanwhile, GR4J calibrated on raw NSE probably nails the big peaks but gets baseflow wrong. A single set of BMA weights must compromise between these two situations. Regime-specific BMA avoids this compromise entirely.

**How it works:** You divide the flow record into regimes — typically high flows (above the 90th percentile), medium flows (10th to 90th), and low flows (below the 10th). You fit a separate BMA model for each regime. During prediction, you identify which regime you're in and use the corresponding weights. At regime boundaries, you blend smoothly between the weight sets using a sigmoid function so the combined prediction doesn't have discontinuities.

**Why this is the biggest lever:** Your 36 models are specifically constructed to span different objectives (NSE vs. KGE vs. likelihood) and different transformations (raw vs. log vs. sqrt vs. inverse). These combinations were designed to emphasize different parts of the hydrograph. Regime-specific BMA is the only method that lets each combination shine where it was designed to shine.

**What you get:** Everything BMA gives you (full probabilistic prediction), but with weights that adapt to flow conditions. The result is typically sharper prediction intervals and better performance across the full flow range.

### Summary Comparison Table

| Method | Weights | Output | Handles Redundancy? | Uncertainty? | Complexity |
|---|---|---|---|---|---|
| Equal weights | 1/K for all | Point prediction | No (overcounts similar models) | No | Trivial |
| GRC | Optimized via regression | Point prediction | Partially (zero weights possible) | No | Low |
| Stacking | Cross-validated optimization | Point prediction | Yes (naturally zeros out duplicates) | No (bootstrappable) | Medium |
| BMA (global) | Bayesian posterior | Full distribution | Somewhat (correlated posteriors) | Yes — calibrated intervals | High |
| Regime-specific BMA | Bayesian posterior per regime | Full distribution per regime | Yes | Yes — flow-adaptive intervals | Highest |

### A cooking analogy

Imagine 36 chefs each cooked their version of a soup.

- **Equal weights** = pour them all into one pot and stir.
- **GRC** = taste each soup, then pour mostly from the best chefs into the pot, guided by the overall taste.
- **Stacking** = same as GRC, but you test the blend on a group of diners who haven't tasted it before, to make sure it's not just you who likes it.
- **BMA** = don't just blend into one pot. Instead, for any given spoonful, you get a little from each chef, but you also know how likely it is that the spoonful came from chef A vs. chef B, and you can say "there's a 90% chance this spoonful is between 'good' and 'great'."
- **Regime-specific BMA** = use different chefs for the appetizer, main course, and dessert, because some chefs are great at starters but terrible at desserts.

---

## 2. PyMC vs. MODELAVG vs. Huang's Code: Do You Need All Three?

This is an important architectural question. The short answer: **use PyMC as your primary framework, and use MODELAVG and Huang's code as reference implementations for validation and for borrowing ideas — not as production dependencies.**

### What each codebase does

**MODELAVG** (Vrugt's toolbox) is a standalone research code. It implements the EM algorithm and its own MCMC sampler (DREAM(ZS)) for BMA weight estimation, supports 8 different conditional PDFs (Normal, Gamma, Lognormal, GEV, Weibull, truncated Normal, generalized Normal, generalized Pareto), and includes methods like EWA, BGA, BMA, and MMA. It's a self-contained BMA toolkit written in a research-lab style. Its Python version is relatively recent and doesn't have a large user community, package management ecosystem, or the extensive documentation that PyMC offers.

**Huang & Merwade's code** implements both EM and Metropolis-Hastings MCMC specifically for flood ensemble BMA, with Normal and Gamma conditional distributions. It's focused and well-documented for its specific use case.

**PyMC** is a general-purpose probabilistic programming framework. It gives you the same MCMC inference (via NUTS, which is a more modern and efficient sampler than Metropolis-Hastings or DREAM(ZS) for continuous parameters), the same model-specification flexibility, plus a massive ecosystem: ArviZ for diagnostics, NUMPyro/JAX for GPU acceleration, and a community of thousands of users with extensive documentation.

### Why PyMC is the right primary choice

Everything MODELAVG does for BMA, you can express in PyMC — and more flexibly. The key advantages:

- **NUTS sampler** is the state-of-the-art for continuous parameters. It requires no hand-tuning of step sizes or proposal distributions, unlike Metropolis-Hastings or DREAM(ZS). For a Dirichlet-weighted mixture with 15+ components, this matters enormously.
- **ArviZ integration** gives you publication-quality diagnostics (R-hat, ESS, trace plots, pair plots, PIT histograms) with one-line function calls.
- **NUMPyro backend** lets you run on GPU via JAX, providing 3–10x speedup over CPU — crucial when running BMA on 5+ CV folds × 3 regimes.
- **Community and longevity**: PyMC has been under continuous development since 2003 and has a large user base. If you hit a problem, there's a Stack Overflow answer or Discourse thread for it.

### Where MODELAVG and Huang's code are valuable

**Validation**: Run both MODELAVG and your PyMC implementation on the same toy dataset and verify you get the same weights and variances. If MODELAVG gives weights [0.35, 0.25, 0.40] and your PyMC model gives [0.34, 0.26, 0.40], you know your PyMC specification is correct.

**Non-Gaussian likelihoods**: MODELAVG supports Gamma and GEV conditional distributions out of the box. When you want to move beyond Gaussian likelihoods in PyMC (which you should, since streamflow is non-negative and right-skewed), MODELAVG's implementation of the Gamma BMA is a useful reference for how to parameterize and initialize those components.

**Algorithm ideas**: Huang's code implements the multi-chain MCMC approach with detailed convergence diagnostics and EM-vs-MCMC comparisons. Their test cases are directly relevant.

### Practical workflow

1. Download MODELAVG and Huang's code into a `references/` directory in your project.
2. Build all production code in PyMC.
3. Create a `validate_vs_modelavg.py` script that runs one catchment through both MODELAVG and your PyMC pipeline, comparing weights, variances, and CRPS.
4. Once validated, all production runs go through your PyMC pipeline. You never need to call MODELAVG at runtime.

---

## 3. Mathematical Foundations

### The BMA mixture distribution

The BMA predictive distribution for observed flow y at timestep t, given training data D and K models, is:

```
p(y_t | D) = Σ_k  w_k · g_k(y_t | f_k,t, D)
```

where f_k,t is model k's prediction at time t, w_k ≥ 0 with Σ w_k = 1 are posterior model weights, and g_k is a conditional PDF (typically Gaussian with bias correction):

```
g_k(y_t | f_k,t) = Normal(y_t | a_k + b_k · f_k,t, σ_k²)
```

The parameters a_k, b_k handle linear bias correction (since your models are already calibrated, typically a_k ≈ 0 and b_k ≈ 1, so these are small corrections). The parameter σ_k² is model k's residual variance.

From this mixture you get:

**BMA mean:** E[y_t] = Σ_k w_k · (a_k + b_k · f_k,t)

**BMA variance:**

```
Var[y_t] = [within-model: Σ_k w_k · σ_k²] + [between-model: Σ_k w_k · (μ_k,t - μ̄_t)²]
```

The within-model term captures each model's inherent noisiness. The between-model term captures how much the models disagree. This decomposition is the fundamental value proposition of BMA over point-prediction methods.

### Weight estimation: EM vs. MCMC

**EM (Expectation-Maximization):** A fast iterative algorithm. In the E-step, you compute how "responsible" each model is for each observation: ẑ_kt = w_k · g_k(y_t | f_k,t) / Σ_l w_l · g_l(y_t | f_l,t). In the M-step, you update weights and variances: w_k = (1/T) · Σ_t ẑ_kt and σ_k² = Σ_t ẑ_kt · (y_t − f_k,t)² / Σ_t ẑ_kt. You iterate until convergence. EM gives point estimates only and can get stuck in local optima.

**MCMC (what PyMC does):** You place priors on all parameters — a Dirichlet prior on weights, Normal priors on bias terms, HalfNormal priors on sigma — and use MCMC sampling (NUTS algorithm) to draw thousands of samples from the joint posterior distribution. This gives you full posterior distributions for every parameter, including correlations between weights. It's slower than EM but gives richer output and is more robust to multimodality.

### The Dirichlet prior for weights

The Dirichlet distribution is the natural prior for a vector of weights that must be non-negative and sum to 1. Its behavior is controlled by a concentration parameter α:

- **α = 1** (uniform over the simplex): no preference — all weight combinations are equally likely a priori.
- **α < 1 (e.g., α = 1/K)**: **sparse** — favors solutions where most weights are near zero and only a few models dominate. This is the right choice when you have many models and expect most to be redundant.
- **α > 1**: pulls weights toward equal values, acting as regularization.

For your K ≈ 15 (post-screening) setup, **α = 1/K ≈ 0.067 is recommended.** This encourages the posterior to identify the few models that genuinely contribute, while allowing the data to override the prior if more models are needed.

---

## 4. Pre-Screening: From 36 to ~10–15 Models

### Why pre-screening is necessary

With 36 models, the Dirichlet has 35 free dimensions. MCMC in 35+ dimensions is slow and weight estimates become noisy. Pre-screening reduces dimensionality while preserving diversity.

### Step 1: Hard threshold removal

Remove models that clearly failed calibration. Based on your clustermap, apply:

- **NSE < 0** (worse than predicting the mean)
- **KGE < −0.41** (worse than climatology, per Knoben et al. 2019)
- **|PBIAS| > 25%** (gross water balance error)

Looking at your clustermap, several Sacramento KGE variants with values near 0.00 across many metrics would be removed here.

### Step 2: Residual correlation clustering

Many of your 36 models will make nearly identical predictions (and errors) because they share a conceptual structure. For example, `gr4j_nse_log_sceua` and `gr4j_nse_sqrt_sceua` likely produce very similar hydrographs.

Compute the Pearson correlation of model **residuals** (predicted − observed), then cluster models with correlation > 0.90 and keep one representative per cluster (the one with the best mean KGE). This typically reduces 36 to 8–15 distinct models.

### Step 3: Preserve regime specialists

Before discarding any model, check if it's the best model for any flow regime (high, medium, or low). A model with mediocre overall NSE but excellent low-flow performance (e.g., high NSE on log-transformed flows) should be kept for the regime-specific BMA, even if it would otherwise be clustered away.

---

## 5. The Hierarchy of Methods: Implementation Specification

The following is the complete specification for implementing all five levels in Python. This is designed to be handed directly to a Cursor AI agent.

### 5.1 Project Structure

```
bma_hydro/
├── config.py              # Configuration dataclass
├── data_prep.py           # Data loading, transformation, train/val splitting
├── pre_screen.py          # Model pre-screening and clustering
├── level1_equal.py        # Equal weights
├── level2_grc.py          # Granger-Ramanathan constrained
├── level3_stacking.py     # Bayesian stacking
├── level4_bma.py          # Global BMA in PyMC
├── level5_regime_bma.py   # Flow regime-specific BMA in PyMC
├── prediction.py          # Posterior predictive generation
├── evaluation.py          # All validation metrics
├── visualization.py       # All plots
├── run_pipeline.py        # Main orchestration
├── validate_vs_modelavg.py  # Compare against MODELAVG reference
└── references/
    ├── MODELAVG/          # Vrugt's toolbox (downloaded from GitHub)
    └── huang_bma/         # Huang & Merwade code (downloaded from GitHub)
```

### 5.2 Configuration (`config.py`)

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

@dataclass
class BMAConfig:
    # --- Paths ---
    model_predictions_path: Path     # CSV: rows=timesteps, cols=model_names
    observed_flow_path: Path         # CSV: datetime + Q column
    output_dir: Path = Path("output")

    # --- Pre-screening ---
    nse_threshold: float = 0.0
    kge_threshold: float = -0.41
    pbias_threshold: float = 25.0
    residual_corr_threshold: float = 0.90
    cluster_method: str = "ward"
    preserve_regime_specialists: bool = True

    # --- Flow transformation (applied before BMA fitting) ---
    transform: str = "none"          # "none", "log", "sqrt", "boxcox"
    log_epsilon: float = 0.01        # added before log to handle zeros
    boxcox_lambda: Optional[float] = None

    # --- BMA model ---
    dirichlet_alpha: str = "sparse"  # "sparse" (1/K), "uniform" (1.0), float
    bias_correction: str = "additive"  # "additive", "linear", "none"
    bias_prior_sigma: float = 0.5
    sigma_prior_sigma: float = 2.0
    heteroscedastic: bool = True     # per-model sigma vs shared sigma
    likelihood: str = "normal"       # "normal", "lognormal", "gamma"
    use_manual_loglik: bool = False  # pm.Potential vs pm.Mixture

    # --- Sampling ---
    draws: int = 2000
    tune: int = 3000
    chains: int = 4
    target_accept: float = 0.95
    nuts_sampler: str = "numpyro"    # "numpyro" (recommended) or "pymc"
    init: str = "advi"
    random_seed: int = 42

    # --- Cross-validation ---
    cv_strategy: str = "block"       # "block" or "expanding_window"
    n_cv_folds: int = 5
    buffer_days: int = 60

    # --- Flow regimes ---
    regime_quantiles: list = field(default_factory=lambda: [0.10, 0.70])
    # High: Q > Q10, Medium: Q10 >= Q >= Q70, Low: Q < Q70
    regime_blend_width: float = 0.05  # sigmoid blending width (frac of range)

    # --- Evaluation ---
    prediction_intervals: list = field(
        default_factory=lambda: [0.50, 0.80, 0.90, 0.95]
    )
```

### 5.3 Data Preparation (`data_prep.py`)

**Responsibilities:**

1. Load predictions CSV (shape T × K) and observed flow (shape T). Align on datetime index. Drop NaN rows.
2. Apply optional flow transformation to both F and y_obs. Store transformation parameters for back-transformation.
3. Generate block temporal CV splits. For n_cv_folds=5, divide the record into 5 contiguous blocks of roughly equal length. For each fold, the validation set is one block; the training set is the remaining blocks with buffer_days removed at each boundary.
4. Classify timesteps into flow regimes based on training-set quantiles (to avoid leakage).

**Key function signatures:**

```python
def load_and_align(config: BMAConfig) -> tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    """Returns (F, y_obs, dates). F has model names as columns."""

def apply_transform(F, y_obs, config) -> tuple[np.ndarray, np.ndarray, dict]:
    """Returns (F_transformed, y_transformed, transform_params).
    transform_params has keys 'method', 'lambda', 'epsilon' for back-transform."""

def back_transform(values, transform_params) -> np.ndarray:
    """Inverse transform predictions back to original flow units."""

def create_cv_splits(dates, config) -> list[tuple[np.ndarray, np.ndarray]]:
    """Returns list of (train_indices, val_indices) with buffer gaps."""

def classify_regime(y_train, config) -> dict[str, np.ndarray]:
    """Returns {'high': bool_mask, 'medium': bool_mask, 'low': bool_mask}
    based on quantiles computed from y_train only."""
```

### 5.4 Level 1: Equal Weights (`level1_equal.py`)

```python
def equal_weight_predict(F: np.ndarray) -> np.ndarray:
    """Simple arithmetic mean across models.
    F: shape (T, K)
    Returns: shape (T,) — point prediction per timestep.
    """
    return F.mean(axis=1)
```

That's it. The entire method is one line. But it's the baseline everything else must beat.

### 5.5 Level 2: GRC (`level2_grc.py`)

```python
from scipy.optimize import minimize
import numpy as np

def grc_fit(F_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    """Fit constrained Granger-Ramanathan weights.
    Minimizes ||y_train - F_train @ w||^2
    subject to: w >= 0, sum(w) = 1

    Returns: weight vector of shape (K,).
    """
    K = F_train.shape[1]
    w0 = np.ones(K) / K  # start at equal weights

    def objective(w):
        residuals = y_train - F_train @ w
        return np.sum(residuals ** 2)

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    bounds = [(0, None)] * K

    result = minimize(objective, w0, method='SLSQP',
                      bounds=bounds, constraints=constraints)
    return result.x


def gra_fit(F_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit unconstrained Granger-Ramanathan with intercept (GRA variant).
    y = a + F @ w  (no constraints on w, allows negative weights and intercept)

    Returns: (weights, intercept).
    """
    K = F_train.shape[1]
    # Add intercept column
    X = np.column_stack([np.ones(len(y_train)), F_train])
    # OLS solution
    coeffs = np.linalg.lstsq(X, y_train, rcond=None)[0]
    intercept = coeffs[0]
    weights = coeffs[1:]
    return weights, intercept


def grc_predict(F: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Apply GRC weights to make predictions."""
    return F @ weights


def gra_predict(F: np.ndarray, weights: np.ndarray, intercept: float) -> np.ndarray:
    """Apply GRA weights + intercept to make predictions."""
    return intercept + F @ weights
```

### 5.6 Level 3: Bayesian Stacking (`level3_stacking.py`)

```python
from scipy.optimize import minimize
import numpy as np

def stacking_fit(F: np.ndarray, y_obs: np.ndarray,
                 cv_splits: list) -> np.ndarray:
    """Fit stacking weights using block temporal CV.

    For each CV fold:
      - Compute the log predictive density of each model on held-out data
    Then find weights maximizing the pooled log mixture density.

    Args:
        F: (T, K) model predictions
        y_obs: (T,) observations
        cv_splits: list of (train_idx, val_idx) tuples

    Returns: stacking weights of shape (K,)
    """
    K = F.shape[1]

    # Collect log predictive densities across all CV folds
    # lpd[i, k] = log p(y_i | model k, trained on data excluding fold containing i)
    lpd = np.full((len(y_obs), K), np.nan)

    for train_idx, val_idx in cv_splits:
        for k in range(K):
            # For pre-calibrated models, the "training" is already done.
            # We compute log-normal density of held-out obs given model k's prediction.
            residuals = y_obs[val_idx] - F[val_idx, k]
            # Use sigma estimated from training fold only
            sigma_k = np.std(y_obs[train_idx] - F[train_idx, k])
            if sigma_k < 1e-10:
                sigma_k = 1e-10  # guard against zero variance
            lpd[val_idx, k] = (
                -0.5 * np.log(2 * np.pi)
                - np.log(sigma_k)
                - 0.5 * (residuals / sigma_k) ** 2
            )

    # Remove any rows still NaN
    valid = ~np.any(np.isnan(lpd), axis=1)
    lpd_valid = lpd[valid]

    # Optimize stacking weights (constrained: w >= 0, sum = 1)
    def neg_log_score(w):
        # Clamp to avoid log(0)
        w_safe = np.clip(w, 1e-15, None)
        w_safe = w_safe / w_safe.sum()
        mix_lpd = np.log(np.sum(w_safe[None, :] * np.exp(lpd_valid), axis=1))
        return -np.sum(mix_lpd)

    w0 = np.ones(K) / K
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    bounds = [(0, None)] * K

    result = minimize(neg_log_score, w0, method='SLSQP',
                      bounds=bounds, constraints=constraints)

    weights = np.clip(result.x, 0, None)
    weights = weights / weights.sum()
    return weights


def stacking_predict(F: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Point prediction using stacking weights."""
    return F @ weights
```

### 5.7 Level 4: Global BMA in PyMC (`level4_bma.py`)

This is the core of the project.

```python
import pymc as pm
import numpy as np
import pytensor.tensor as pt
import arviz as az
import pandas as pd


def build_bma_model(F_train: np.ndarray, y_train: np.ndarray,
                    config: 'BMAConfig') -> pm.Model:
    """
    Build a PyMC BMA model as a Dirichlet-weighted Gaussian mixture.

    Args:
        F_train: (T, K) array of model predictions (transformed if applicable)
        y_train: (T,) array of observations (transformed if applicable)
        config: BMAConfig instance

    Returns:
        pm.Model ready for sampling
    """
    K = F_train.shape[1]

    # Determine Dirichlet concentration
    if config.dirichlet_alpha == "sparse":
        alpha_val = 1.0 / K
    elif config.dirichlet_alpha == "uniform":
        alpha_val = 1.0
    else:
        alpha_val = float(config.dirichlet_alpha)

    with pm.Model() as model:
        # ---- Weights (on the K-simplex) ----
        w = pm.Dirichlet("w", a=np.ones(K) * alpha_val)

        # ---- Bias correction ----
        if config.bias_correction == "additive":
            # y ≈ f_k + a_k   (shift only)
            bias = pm.Normal("bias", mu=0, sigma=config.bias_prior_sigma, shape=K)
            mu = F_train + bias[None, :]  # (T, K)
        elif config.bias_correction == "linear":
            # y ≈ a_k + b_k * f_k   (shift + scale)
            a = pm.Normal("a", mu=0, sigma=config.bias_prior_sigma, shape=K)
            b = pm.Normal("b", mu=1, sigma=0.2, shape=K)  # prior centered on 1
            mu = a[None, :] + b[None, :] * F_train  # (T, K)
        else:  # "none"
            mu = F_train  # no bias correction

        # ---- Residual standard deviation ----
        if config.heteroscedastic:
            sigma = pm.HalfNormal("sigma", sigma=config.sigma_prior_sigma, shape=K)
        else:
            sigma = pm.HalfNormal("sigma", sigma=config.sigma_prior_sigma)

        # ---- Likelihood ----
        if config.use_manual_loglik:
            # Manual logsumexp — maximum control, avoids pm.Mixture shape issues
            if config.heteroscedastic:
                sig = sigma[None, :]  # (1, K) → broadcasts to (T, K)
            else:
                sig = sigma
            log_comp = (
                -0.5 * pt.log(2 * np.pi)
                - pt.log(sig)
                - 0.5 * ((y_train[:, None] - mu) / sig) ** 2
            )
            log_w = pt.log(w)[None, :]  # (1, K)
            log_lik = pm.math.logsumexp(log_w + log_comp, axis=1)  # (T,)
            pm.Potential("loglik", log_lik.sum())
        else:
            # pm.Mixture — cleaner, but can have shape issues with large K
            if config.heteroscedastic:
                comp_dists = pm.Normal.dist(
                    mu=mu,
                    sigma=sigma[None, :] * pt.ones_like(mu)
                )
            else:
                comp_dists = pm.Normal.dist(mu=mu, sigma=sigma)
            pm.Mixture("obs", w=w, comp_dists=comp_dists, observed=y_train)

    return model


def sample_bma(model: pm.Model, config: 'BMAConfig') -> az.InferenceData:
    """Sample the BMA model and return InferenceData."""
    with model:
        idata = pm.sample(
            draws=config.draws,
            tune=config.tune,
            chains=config.chains,
            target_accept=config.target_accept,
            init=config.init,
            nuts_sampler=config.nuts_sampler,
            random_seed=config.random_seed,
        )
    return idata


def check_convergence(idata: az.InferenceData) -> dict:
    """
    Check R-hat, ESS, divergences.
    Return dict of any issues found. Empty dict = all good.
    """
    issues = {}

    # R-hat (must be < 1.01 for all parameters)
    rhat = az.rhat(idata)
    for var in rhat:
        max_rhat = float(rhat[var].max())
        if max_rhat > 1.01:
            issues[f"rhat_{var}"] = max_rhat

    # Effective sample size (bulk and tail, target > 400)
    for kind in ["bulk", "tail"]:
        ess = az.ess(idata, kind=kind)
        for var in ess:
            min_ess = float(ess[var].min())
            if min_ess < 400:
                issues[f"ess_{kind}_{var}"] = min_ess

    # Divergences
    n_div = int(idata.sample_stats["diverging"].sum())
    if n_div > 0:
        issues["divergences"] = n_div

    return issues


def extract_weights(idata: az.InferenceData, model_names: list) -> pd.DataFrame:
    """Extract posterior weight summary with model names."""
    summary = az.summary(idata, var_names=["w"], hdi_prob=0.94, kind="stats")
    summary.index = model_names
    return summary
```

**Key sampling notes for Cursor agent:**

- **Use the NUMPyro backend.** Set `nuts_sampler="numpyro"`. The default PyTensor backend can be extremely slow for Dirichlet-based models. NUMPyro's JAX-based autodiff is 3–10x faster. Install with `pip install numpyro jax jaxlib`.
- **High target_accept is critical.** The simplex boundary (where some weights approach zero) causes geometric difficulties for NUTS. Setting `target_accept=0.95` forces smaller step sizes, reducing divergent transitions.
- **ADVI initialization helps.** Starting MCMC chains from variational inference estimates (`init="advi"`) places them in a reasonable region of the simplex, dramatically improving warmup efficiency.
- **Label switching is NOT a problem.** Unlike unsupervised mixture models, BMA components are identified — component k is always model k's predictions. No post-processing needed.

### 5.8 Level 5: Flow Regime-Specific BMA (`level5_regime_bma.py`)

```python
import numpy as np
from level4_bma import build_bma_model, sample_bma, check_convergence


def build_regime_bma(F_train: np.ndarray, y_train: np.ndarray,
                     regime_masks: dict, config: 'BMAConfig') -> dict:
    """
    Build and sample separate BMA models for each flow regime.

    Args:
        F_train: (T, K) predictions
        y_train: (T,) observations
        regime_masks: dict with keys 'high', 'medium', 'low',
                      values are boolean arrays of shape (T,)
        config: BMAConfig

    Returns:
        dict mapping regime name to (model, idata) tuples
    """
    results = {}
    for regime_name, mask in regime_masks.items():
        n_points = mask.sum()
        if n_points < 50:
            print(f"Warning: regime '{regime_name}' has only {n_points} points, "
                  f"falling back to global BMA for this regime")
            continue

        F_regime = F_train[mask]
        y_regime = y_train[mask]

        print(f"Fitting BMA for regime '{regime_name}' ({n_points} timesteps)...")
        model = build_bma_model(F_regime, y_regime, config)
        idata = sample_bma(model, config)

        issues = check_convergence(idata)
        if issues:
            print(f"  Convergence issues in '{regime_name}': {issues}")

        results[regime_name] = (model, idata)

    return results


def compute_regime_blend_weights(flow_proxy: np.ndarray,
                                  q_high: float, q_low: float,
                                  blend_width: float) -> dict:
    """
    Compute smooth sigmoid blending weights for regime transitions.

    At each timestep, returns the probability of belonging to each regime
    using sigmoid functions for smooth transitions.

    Args:
        flow_proxy: (T,) estimated flow level (e.g., mean of model predictions)
        q_high: flow value separating high from medium regime
        q_low: flow value separating medium from low regime
        blend_width: width of sigmoid transition zone (in flow units)

    Returns:
        dict with keys 'high', 'medium', 'low', values are (T,) arrays
        that sum to 1.0 at each timestep.
    """
    def sigmoid(x, center, width):
        z = (x - center) / max(width, 1e-10)
        return 1.0 / (1.0 + np.exp(-z))

    p_high = sigmoid(flow_proxy, q_high, blend_width)
    p_low = 1 - sigmoid(flow_proxy, q_low, blend_width)
    p_medium = np.clip(1 - p_high - p_low, 0, 1)

    # Normalize to sum to 1
    total = p_high + p_medium + p_low
    return {
        'high': p_high / total,
        'medium': p_medium / total,
        'low': p_low / total,
    }


def regime_blend_predict(F_val: np.ndarray,
                          regime_results: dict,
                          regime_blend_weights: dict,
                          config: 'BMAConfig',
                          n_samples: int = 4000) -> dict:
    """
    Generate blended predictions across regimes using sigmoid transitions.

    For each posterior draw:
        1. Randomly select a regime proportional to blend weights
        2. Use that regime's BMA to generate the prediction

    Returns:
        dict with 'samples', 'mean', 'median', 'intervals'
    """
    T = F_val.shape[0]
    all_samples = np.zeros((n_samples, T))

    for s in range(n_samples):
        for t in range(T):
            # Pick regime based on blend weights
            regime_probs = np.array([
                regime_blend_weights['high'][t],
                regime_blend_weights['medium'][t],
                regime_blend_weights['low'][t],
            ])
            regime_names = ['high', 'medium', 'low']
            chosen = np.random.choice(3, p=regime_probs)
            regime_name = regime_names[chosen]

            if regime_name not in regime_results:
                # Fallback: use whichever regime is available
                regime_name = list(regime_results.keys())[0]

            _, idata = regime_results[regime_name]

            # Draw random posterior sample from this regime's BMA
            w_all = idata.posterior["w"].values.reshape(-1, idata.posterior["w"].shape[-1])
            s_idx = np.random.randint(len(w_all))
            w_s = w_all[s_idx]

            # Get bias and sigma similarly
            if "bias" in idata.posterior:
                bias_all = idata.posterior["bias"].values.reshape(-1, w_all.shape[-1])
                bias_s = bias_all[s_idx]
            else:
                bias_s = np.zeros(len(w_s))

            sigma_all = idata.posterior["sigma"].values.reshape(-1, w_all.shape[-1])
            sigma_s = sigma_all[s_idx]

            # Sample component
            k = np.random.choice(len(w_s), p=w_s)
            mu_k = F_val[t, k] + bias_s[k]
            all_samples[s, t] = np.random.normal(mu_k, sigma_s[k])

    # Compute summaries
    result = {
        'samples': all_samples,
        'mean': all_samples.mean(axis=0),
        'median': np.median(all_samples, axis=0),
        'intervals': {}
    }
    for level in config.prediction_intervals:
        lo = (1 - level) / 2
        hi = 1 - lo
        result['intervals'][level] = (
            np.quantile(all_samples, lo, axis=0),
            np.quantile(all_samples, hi, axis=0)
        )
    return result
```

### 5.9 Posterior Predictive Generation (`prediction.py`)

```python
import numpy as np


def generate_bma_predictions(idata, F_val, config, n_samples=4000):
    """
    Generate posterior predictive samples from a fitted global BMA model.

    For each posterior draw s = 1..S:
        1. Get weights w^(s), bias^(s), sigma^(s) from posterior
        2. For each timestep t:
            a. Compute mu_kt = F_val[t,k] + bias_k^(s) for all k
            b. Sample component k ~ Categorical(w^(s))
            c. Sample y_t^(s) ~ Normal(mu_kt, sigma_k^(s))

    Returns:
        dict with keys:
            'samples': (n_samples, T_val) posterior predictive samples
            'mean': (T_val,) posterior predictive mean
            'median': (T_val,) posterior predictive median
            'intervals': dict mapping level to (lower, upper) arrays
    """
    # Extract posterior samples and reshape to (total_draws, K)
    w_samples = idata.posterior["w"].values
    w_flat = w_samples.reshape(-1, w_samples.shape[-1])

    has_bias = "bias" in idata.posterior
    if has_bias:
        bias_samples = idata.posterior["bias"].values
        bias_flat = bias_samples.reshape(-1, bias_samples.shape[-1])

    sigma_samples = idata.posterior["sigma"].values
    sigma_flat = sigma_samples.reshape(-1, sigma_samples.shape[-1])

    T_val = F_val.shape[0]
    K = F_val.shape[1]
    S = min(n_samples, len(w_flat))
    idx = np.random.choice(len(w_flat), S, replace=False)

    y_pred = np.zeros((S, T_val))

    for s_i, s in enumerate(idx):
        w_s = w_flat[s]
        bias_s = bias_flat[s] if has_bias else np.zeros(K)
        sigma_s = sigma_flat[s]

        mu = F_val + bias_s[None, :]  # (T_val, K)

        # Vectorized component sampling
        components = np.array([np.random.choice(K, p=w_s) for _ in range(T_val)])

        for t in range(T_val):
            k = components[t]
            sig_k = sigma_s[k] if sigma_s.ndim > 0 else float(sigma_s)
            y_pred[s_i, t] = np.random.normal(mu[t, k], sig_k)

    # Summaries
    result = {
        'samples': y_pred,
        'mean': y_pred.mean(axis=0),
        'median': np.median(y_pred, axis=0),
        'intervals': {}
    }
    for level in config.prediction_intervals:
        lo = (1 - level) / 2
        hi = 1 - lo
        result['intervals'][level] = (
            np.quantile(y_pred, lo, axis=0),
            np.quantile(y_pred, hi, axis=0)
        )
    return result


def back_transform_predictions(pred_dict, transform_params):
    """Apply inverse transform to all prediction arrays."""
    from data_prep import back_transform

    result = {
        'samples': back_transform(pred_dict['samples'], transform_params),
        'mean': back_transform(pred_dict['mean'], transform_params),
        'median': back_transform(pred_dict['median'], transform_params),
        'intervals': {}
    }
    for level, (lo, hi) in pred_dict['intervals'].items():
        result['intervals'][level] = (
            back_transform(lo, transform_params),
            back_transform(hi, transform_params)
        )
    return result
```

### 5.10 Evaluation (`evaluation.py`)

```python
import numpy as np
from scipy import stats


# ========================================
# DETERMINISTIC METRICS
# ========================================

def nse(y_obs, y_pred):
    """Nash-Sutcliffe Efficiency."""
    return 1 - np.sum((y_obs - y_pred)**2) / np.sum((y_obs - y_obs.mean())**2)


def kge(y_obs, y_pred):
    """Kling-Gupta Efficiency (2009 formulation)."""
    r = np.corrcoef(y_obs, y_pred)[0, 1]
    alpha = np.std(y_pred) / np.std(y_obs)
    beta = np.mean(y_pred) / np.mean(y_obs)
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)


def pbias(y_obs, y_pred):
    """Percent bias."""
    return 100 * np.sum(y_pred - y_obs) / np.sum(y_obs)


def rmse(y_obs, y_pred):
    """Root Mean Square Error."""
    return np.sqrt(np.mean((y_obs - y_pred)**2))


def evaluate_deterministic(y_obs, y_pred):
    """Compute all deterministic metrics. Returns dict."""
    return {
        'nse': nse(y_obs, y_pred),
        'kge': kge(y_obs, y_pred),
        'pbias': pbias(y_obs, y_pred),
        'rmse': rmse(y_obs, y_pred),
    }


# ========================================
# PROBABILISTIC METRICS (Levels 4-5 only)
# ========================================

def crps_ensemble(y_obs, y_samples):
    """CRPS for ensemble predictions.
    y_obs: (T,), y_samples: (S, T).
    Uses the ensemble CRPS formula: E|Y-x| - 0.5*E|Y-Y'|
    """
    T = len(y_obs)
    crps_values = np.zeros(T)
    for t in range(T):
        ens = y_samples[:, t]
        crps_values[t] = (
            np.mean(np.abs(ens - y_obs[t]))
            - 0.5 * np.mean(np.abs(ens[:, None] - ens[None, :]))
        )
    return crps_values.mean()


def pit_values(y_obs, y_samples):
    """Probability Integral Transform values.
    For each obs, compute the fraction of samples below it.
    Should be Uniform(0,1) if the predictive distribution is well-calibrated.
    """
    T = len(y_obs)
    pit = np.zeros(T)
    for t in range(T):
        pit[t] = np.mean(y_samples[:, t] <= y_obs[t])
    return pit


def coverage(y_obs, lower, upper):
    """Fraction of observations within the prediction interval."""
    return np.mean((y_obs >= lower) & (y_obs <= upper))


def interval_width(lower, upper):
    """Mean prediction interval width (sharpness measure)."""
    return np.mean(upper - lower)


def evaluate_probabilistic(y_obs, pred_dict):
    """Compute all probabilistic metrics from a prediction dict."""
    results = {
        'crps': crps_ensemble(y_obs, pred_dict['samples']),
    }
    for level, (lo, hi) in pred_dict['intervals'].items():
        results[f'coverage_{int(level*100)}'] = coverage(y_obs, lo, hi)
        results[f'width_{int(level*100)}'] = interval_width(lo, hi)
    return results


# ========================================
# REGIME-SPECIFIC EVALUATION
# ========================================

def evaluate_by_regime(y_obs, y_pred_mean, y_samples, regime_masks):
    """Compute all metrics separately per flow regime."""
    results = {}
    for regime, mask in regime_masks.items():
        if mask.sum() == 0:
            continue
        results[regime] = {
            **evaluate_deterministic(y_obs[mask], y_pred_mean[mask]),
            'crps': crps_ensemble(y_obs[mask], y_samples[:, mask]),
        }
    return results


# ========================================
# FLOW DURATION CURVE ERRORS
# ========================================

def fdc_error(y_obs, y_pred, segments=None):
    """Flow Duration Curve error by exceedance probability segment."""
    if segments is None:
        segments = [(0, 5), (5, 20), (20, 70), (70, 95), (95, 100)]

    obs_sorted = np.sort(y_obs)[::-1]
    pred_sorted = np.sort(y_pred)[::-1]
    n = len(obs_sorted)
    exc_prob = np.arange(1, n + 1) / n * 100

    errors = {}
    for lo, hi in segments:
        mask = (exc_prob >= lo) & (exc_prob < hi)
        if mask.sum() > 0:
            errors[f"FDC_{lo}_{hi}"] = rmse(obs_sorted[mask], pred_sorted[mask])
    return errors


# ========================================
# HYDROLOGIC SIGNATURES
# ========================================

def compute_signatures(y):
    """Compute key hydrologic signatures from a flow time series."""
    q5 = np.percentile(y, 95)   # high flow (5% exceedance)
    q95 = np.percentile(y, 5)   # low flow (95% exceedance)
    return {
        'Q5': q5,
        'Q95': q95,
        'Q5_Q95_ratio': q5 / max(q95, 1e-10),
        'runoff_ratio': np.mean(y),  # normalize by precip externally
        'baseflow_index': None,       # implement Lyne-Hollick filter separately
    }
```

### 5.11 Main Pipeline (`run_pipeline.py`)

```python
from pathlib import Path
from config import BMAConfig
from data_prep import (load_and_align, apply_transform, back_transform,
                       create_cv_splits, classify_regime)
from pre_screen import pre_screen
from level1_equal import equal_weight_predict
from level2_grc import grc_fit, grc_predict
from level3_stacking import stacking_fit, stacking_predict
from level4_bma import build_bma_model, sample_bma, check_convergence, extract_weights
from level5_regime_bma import (build_regime_bma, compute_regime_blend_weights,
                                regime_blend_predict)
from prediction import generate_bma_predictions, back_transform_predictions
from evaluation import (evaluate_deterministic, evaluate_probabilistic,
                         evaluate_by_regime, fdc_error)


def main():
    config = BMAConfig(
        model_predictions_path=Path("data/predictions.csv"),
        observed_flow_path=Path("data/observed.csv"),
        output_dir=Path("output"),
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # ========================
    # 1. LOAD AND PREPARE DATA
    # ========================
    print("Loading data...")
    F_raw, y_obs, dates = load_and_align(config)
    F, y, tparams = apply_transform(F_raw.values, y_obs.values, config)
    cv_splits = create_cv_splits(dates, config)
    print(f"Loaded {F_raw.shape[1]} models, {len(dates)} timesteps, {len(cv_splits)} CV folds")

    # ========================
    # 2. PRE-SCREEN MODELS
    # ========================
    print("Pre-screening models...")
    F_screened, kept_models, corr_matrix = pre_screen(F, y, F_raw.columns.tolist(), config)
    K = F_screened.shape[1]
    print(f"Kept {K} models from {F_raw.shape[1]} original: {kept_models}")

    # ========================
    # 3. CROSS-VALIDATED COMPARISON OF ALL 5 LEVELS
    # ========================
    all_results = {
        'equal': [], 'grc': [], 'stacking': [],
        'bma_global': [], 'bma_regime': []
    }

    for fold, (train_idx, val_idx) in enumerate(cv_splits):
        print(f"\n--- CV Fold {fold + 1}/{len(cv_splits)} ---")
        F_train, F_val = F_screened[train_idx], F_screened[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        y_val_orig = back_transform(y_val, tparams)

        # --- Level 1: Equal weights ---
        pred_eq = back_transform(equal_weight_predict(F_val), tparams)
        all_results['equal'].append(evaluate_deterministic(y_val_orig, pred_eq))
        print(f"  Equal weights KGE: {all_results['equal'][-1]['kge']:.3f}")

        # --- Level 2: GRC ---
        w_grc = grc_fit(F_train, y_train)
        pred_grc = back_transform(grc_predict(F_val, w_grc), tparams)
        all_results['grc'].append(evaluate_deterministic(y_val_orig, pred_grc))
        print(f"  GRC KGE: {all_results['grc'][-1]['kge']:.3f}")

        # --- Level 3: Stacking ---
        w_stack = stacking_fit(F_screened, y, cv_splits)
        pred_stack = back_transform(stacking_predict(F_val, w_stack), tparams)
        all_results['stacking'].append(evaluate_deterministic(y_val_orig, pred_stack))
        print(f"  Stacking KGE: {all_results['stacking'][-1]['kge']:.3f}")

        # --- Level 4: Global BMA ---
        print("  Fitting global BMA...")
        bma_model = build_bma_model(F_train, y_train, config)
        bma_idata = sample_bma(bma_model, config)
        issues = check_convergence(bma_idata)
        if issues:
            print(f"  BMA convergence issues: {issues}")
        bma_preds = generate_bma_predictions(bma_idata, F_val, config)
        bma_preds_orig = back_transform_predictions(bma_preds, tparams)
        bma_det = evaluate_deterministic(y_val_orig, bma_preds_orig['mean'])
        bma_prob = evaluate_probabilistic(y_val_orig, bma_preds_orig)
        all_results['bma_global'].append({**bma_det, **bma_prob})
        print(f"  BMA KGE: {bma_det['kge']:.3f}, CRPS: {bma_prob['crps']:.3f}")

        # --- Level 5: Regime-specific BMA ---
        print("  Fitting regime-specific BMA...")
        regime_masks_train = classify_regime(y_train, config)

        # Compute regime quantile thresholds from training data
        q_high = np.percentile(y_train, 100 * (1 - config.regime_quantiles[0]))
        q_low = np.percentile(y_train, 100 * (1 - config.regime_quantiles[1]))
        blend_w_range = config.regime_blend_width * (q_high - q_low)

        regime_results = build_regime_bma(F_train, y_train, regime_masks_train, config)

        # Compute blend weights for validation timesteps
        flow_proxy = F_val.mean(axis=1)
        blend_weights = compute_regime_blend_weights(flow_proxy, q_high, q_low, blend_w_range)

        regime_preds = regime_blend_predict(F_val, regime_results, blend_weights, config)
        regime_preds_orig = back_transform_predictions(regime_preds, tparams)
        regime_det = evaluate_deterministic(y_val_orig, regime_preds_orig['mean'])
        regime_prob = evaluate_probabilistic(y_val_orig, regime_preds_orig)
        all_results['bma_regime'].append({**regime_det, **regime_prob})
        print(f"  Regime BMA KGE: {regime_det['kge']:.3f}, CRPS: {regime_prob['crps']:.3f}")

    # ========================
    # 4. AGGREGATE AND REPORT
    # ========================
    print("\n" + "=" * 60)
    print("CROSS-VALIDATED RESULTS (mean across folds)")
    print("=" * 60)
    for method_name, fold_results in all_results.items():
        metrics = {}
        for key in fold_results[0]:
            values = [r[key] for r in fold_results if r[key] is not None]
            if values:
                metrics[key] = np.mean(values)
        print(f"\n{method_name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    # ========================
    # 5. FINAL FIT ON FULL DATA
    # ========================
    print("\nFitting final BMA on full dataset...")
    final_model = build_bma_model(F_screened, y, config)
    final_idata = sample_bma(final_model, config)
    final_weights = extract_weights(final_idata, kept_models)
    print("\nFinal BMA weights:")
    print(final_weights)

    # ========================
    # 6. SAVE
    # ========================
    final_weights.to_csv(config.output_dir / "posterior_weights.csv")
    final_idata.to_netcdf(config.output_dir / "final_bma_model.nc")
    print(f"\nResults saved to {config.output_dir}")


if __name__ == "__main__":
    main()
```

---

## 6. Temporal Cross-Validation Details

**Never use random k-fold CV for hydrological time series.** Temporal autocorrelation means random folds leak future information into training data.

**Block temporal CV:** Divide the record into 5–10 contiguous blocks of 1–3 water years each. For each fold, one block is validation and the rest are training, with a 30–90 day buffer at each boundary removed from both sets.

**Expanding window** (alternative): Train on years 1–N, validate on year N+1. Then train on years 1–(N+1), validate on year N+2. This mimics operational forecasting where you only use past data. More realistic but yields fewer effective folds.

For BMA weight estimation, you need a minimum of **2–5 years** of daily data in the training set, covering the full range of flow conditions (wet years, dry years, seasonal transitions).

---

## 7. Evaluation Framework

All five levels should be compared using the same metrics on the same held-out validation data from each CV fold.

**Deterministic** (all levels): KGE, NSE, PBIAS, RMSE — computed on the point prediction (mean for BMA, weighted average for others).

**Probabilistic** (Levels 4–5 only): CRPS, PIT histogram uniformity, coverage at 50/80/90/95%, mean interval width. These capture BMA's unique value.

**Regime-specific** (all levels): Repeat all metrics for high flows, medium flows, and low flows separately. This directly shows whether regime-specific BMA (Level 5) justifies its added complexity.

**FDC errors**: Compare flow duration curves across all 5 segments of exceedance probability (0–5%, 5–20%, 20–70%, 70–95%, 95–100%).

**Hydrologic signatures**: Q5/Q95 ratio, baseflow index, runoff ratio, recession constant — computed from predictions vs. observed.

---

## 8. Reference Codes and Validation Strategy

**MODELAVG** (github.com/jaspervrugt/MODELAVG): Download into `references/MODELAVG/`. This implements BMA via EM and DREAM(ZS) MCMC, supports 8 conditional PDFs, and includes hydrological examples. Use it to validate your PyMC BMA weights on a single catchment.

**Huang & Merwade BMA** (github.com/huan1441/Bayesian-Model-Averaging-for-Ensemble-Flood-Modeling): Download into `references/huang_bma/`. Implements both EM and Metropolis-Hastings for BMA. Good reference for Gamma conditional distributions.

**Validation protocol:** Pick one catchment with your 36 models. Run MODELAVG's EM algorithm. Run your PyMC BMA. Compare posterior mean weights — they should agree to within ~0.02. Compare CRPS on a held-out year. If they diverge significantly, debug the likelihood specification (most common source of disagreement is how bias correction is handled).

---

## 9. Dependencies

```
# Core
pymc>=5.10
arviz>=0.17
numpyro>=0.13
jax>=0.4
numpy
pandas
scipy
xarray

# Clustering and ML utilities
scikit-learn

# Evaluation
properscoring          # CRPS computation (alternative to manual implementation)

# Optimization (for GRC, stacking)
cvxpy                  # constrained optimization (optional, scipy SLSQP works too)

# Visualization
matplotlib
seaborn

# Optional
tqdm                   # progress bars for CV loops
```

---

## 10. Key Takeaways

The five levels form a natural progression. **Always start with equal weights** — if your more complex methods can't beat the simple average, something is wrong with either the method or the ensemble. **GRC is the minimum viable optimization** — it's fast, easy to debug, and often performs within a few percent of BMA for point predictions. **Stacking is theoretically preferable to classical BMA** for your setup because none of your 36 models is the true data-generating process. **BMA's unique value is probabilistic prediction** — calibrated uncertainty intervals and the between-model vs. within-model variance decomposition. **Regime-specific BMA is where the biggest hydrological gains live** because your 36 models were explicitly designed to excel at different flow regimes via their different objective functions and transformations.

Use PyMC as your production framework. Use MODELAVG and Huang's code as validation references, not dependencies. Pre-screen aggressively to get from 36 to ~10–15 models before running BMA. And always evaluate with regime-specific metrics and probabilistic scores — aggregate NSE alone will hide the most important differences between these methods.
