"""
BMA configuration dataclass.

Centralises all settings for the multi-model ensemble workflow:
data loading, pre-screening, cross-validation, BMA model specification,
sampling, evaluation, and output.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── Preset CV configurations for ACT catchments ─────────────────────────
ACT_CV_PRESETS: Dict[str, Dict] = {
    "standard": dict(
        cv_strategy="block",
        cv_year_start="water_year",
        cv_block_years=2,
        buffer_days=60,
    ),
    "flood_focused": dict(
        cv_strategy="block",
        cv_year_start="custom",
        cv_year_start_month=4,
        cv_block_years=1,
        buffer_days=45,
    ),
    "long_term_drought": dict(
        cv_strategy="block",
        cv_year_start="water_year",
        cv_block_years=5,
        buffer_days=90,
    ),
    "operational": dict(
        cv_strategy="expanding_window",
        cv_year_start="water_year",
        cv_block_years=1,
        buffer_days=60,
    ),
}


@dataclass
class BMAConfig:
    """Configuration for the Bayesian Model Averaging pipeline.

    Attributes are grouped into logical sections.  Only paths and the
    optional ``batch_result`` source are truly required; everything else
    has sensible defaults for ACT headwater catchments.

    Example::

        config = BMAConfig(
            model_predictions_path=Path("data/predictions.csv"),
            observed_flow_path=Path("data/observed.csv"),
        )
        # Or apply an ACT preset:
        config = BMAConfig.from_preset("flood_focused",
            model_predictions_path=..., observed_flow_path=...
        )
    """

    # ── Paths ────────────────────────────────────────────────────────────
    model_predictions_path: Optional[Path] = None
    observed_flow_path: Optional[Path] = None
    output_dir: Path = Path("output")

    # ── Pre-screening ────────────────────────────────────────────────────
    nse_threshold: float = 0.0
    kge_threshold: float = -0.41
    pbias_threshold: float = 25.0
    residual_corr_threshold: float = 0.90
    cluster_method: str = "ward"
    preserve_regime_specialists: bool = True

    # ── Flow transformation (applied before BMA fitting) ─────────────────
    transform: str = "none"
    log_epsilon: float = 0.01
    boxcox_lambda: Optional[float] = None

    # ── BMA model specification ──────────────────────────────────────────
    dirichlet_alpha: str = "sparse"
    bias_correction: str = "additive"
    bias_prior_sigma: float = 0.5
    sigma_prior_sigma: float = 2.0
    heteroscedastic: bool = True
    likelihood: str = "normal"
    use_manual_loglik: bool = True

    # ── Sampling ─────────────────────────────────────────────────────────
    draws: int = 2000
    tune: int = 3000
    chains: int = 4
    target_accept: float = 0.95
    nuts_sampler: str = "numpyro"
    init: str = "advi"
    random_seed: int = 42

    # ── Cross-validation ─────────────────────────────────────────────────
    cv_strategy: str = "block"
    cv_year_start: str = "water_year"
    cv_year_start_month: int = 7
    cv_block_years: int = 1
    n_cv_folds: Optional[int] = None
    buffer_days: int = 60
    min_train_years: float = 3.0
    min_val_years: float = 1.0
    require_seasonal_coverage: bool = True

    # ── Flow regimes ─────────────────────────────────────────────────────
    regime_quantiles: List[float] = field(
        default_factory=lambda: [0.10, 0.70]
    )
    regime_blend_width: float = 0.05

    # ── Evaluation ───────────────────────────────────────────────────────
    prediction_intervals: List[float] = field(
        default_factory=lambda: [0.50, 0.80, 0.90, 0.95]
    )

    # ── Derived ──────────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        if self.model_predictions_path is not None:
            self.model_predictions_path = Path(self.model_predictions_path)
        if self.observed_flow_path is not None:
            self.observed_flow_path = Path(self.observed_flow_path)
        self.output_dir = Path(self.output_dir)

    @property
    def resolved_start_month(self) -> int:
        """Return the numeric start month after resolving named presets."""
        if self.cv_year_start == "water_year":
            return 7
        if self.cv_year_start == "calendar_year":
            return 1
        return self.cv_year_start_month

    @classmethod
    def from_preset(cls, preset_name: str, **overrides) -> "BMAConfig":
        """Create a config from an ACT CV preset with optional overrides.

        Available presets: ``standard``, ``flood_focused``,
        ``long_term_drought``, ``operational``.
        """
        if preset_name not in ACT_CV_PRESETS:
            raise ValueError(
                f"Unknown preset '{preset_name}'. "
                f"Choose from: {list(ACT_CV_PRESETS)}"
            )
        kw = {**ACT_CV_PRESETS[preset_name], **overrides}
        return cls(**kw)
