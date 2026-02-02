# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python (pyrrm)
#     language: python
#     name: pyrrm
# ---

# %% [markdown]
# # Calibration Progress Monitor
#
# ## Purpose
#
# This notebook provides tools for monitoring ongoing calibration runs in real-time
# by reading the CSV output files written during MCMC and optimization algorithms.
# Use it to track progress, diagnose convergence issues, and export best parameters.
#
# ## What You'll Learn
#
# - When and why to monitor calibrations
# - How to interpret calibration progress diagnostics
# - Understanding MCMC convergence (Gelman-Rubin R-hat)
# - How to extract best parameters from ongoing runs
#
# ## Prerequisites
#
# - A running calibration (from Notebook 04 or your own script)
# - Understanding of calibration concepts (Notebook 02)
#
# ## Estimated Time
#
# - Setup: ~2 minutes
# - Monitoring: As long as your calibration runs

# %% [markdown]
# ---
# ## When and Why to Monitor Calibrations
#
# ### When to Use This Notebook
#
# | Scenario | Use Monitor? | Why |
# |----------|--------------|-----|
# | **Long MCMC runs** (10K+ iterations) | ✓ Yes | Track convergence, detect problems early |
# | **Production calibrations** | ✓ Yes | Ensure resources aren't wasted on stuck runs |
# | **Algorithm debugging** | ✓ Yes | Understand algorithm behavior |
# | **Quick test runs** (<1000 iterations) | ✗ Optional | Usually finishes before monitoring helps |
# | **Automated pipelines** | ✗ No | Use programmatic diagnostics instead |
#
# ### What to Watch For
#
# **Good Signs (calibration is working):**
# - Running best objective improves over time
# - Parameter traces show mixing (oscillating around values)
# - R-hat statistics approaching 1.0
# - Posterior distributions narrowing as samples accumulate
#
# **Warning Signs (something may be wrong):**
# - Objective function stuck at same value for many iterations
# - Parameter traces show trends instead of oscillations
# - R-hat values > 1.2 after many iterations
# - Very wide or multi-modal posteriors
# - One chain much worse than others
#
# ### Monitoring Strategy
#
# ```
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                        CALIBRATION MONITORING WORKFLOW                      │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │                                                                             │
# │   1. START CALIBRATION (in another terminal/script)                         │
# │      └─► Creates CSV output file as calibration runs                        │
# │                                                                             │
# │   2. EARLY CHECK (after ~10% complete)                                      │
# │      └─► Is objective improving?                                            │
# │      └─► Are parameters moving?                                             │
# │      └─► If stuck → consider restarting with different settings             │
# │                                                                             │
# │   3. MIDPOINT CHECK (after ~50% complete)                                   │
# │      └─► Check R-hat convergence (should be improving)                      │
# │      └─► Look at parameter traces for mixing                                │
# │      └─► Adjust burn-in estimate                                            │
# │                                                                             │
# │   4. FINAL CHECK (near completion)                                          │
# │      └─► Confirm R-hat < 1.1 for all parameters                             │
# │      └─► Review posterior distributions                                     │
# │      └─► Export best parameters                                             │
# │                                                                             │
# └─────────────────────────────────────────────────────────────────────────────┘
# ```

# %% [markdown]
# ---
# ## CSV File Formats by Calibration Method
#
# Each calibration method writes slightly different CSV formats. This section
# documents what to expect from each.
#
# ### SpotPy DREAM / SCE-UA
#
# **File location:** Same directory as your script, named via `dbname` parameter
#
# **Columns:**
# ```
# like1,paruztwm,paruzfwm,parlztwm,parlzfpm,...
# -0.234,45.23,32.11,125.67,55.89,...
# -0.198,48.67,35.22,130.45,58.12,...
# ```
#
# | Column | Description |
# |--------|-------------|
# | `like1` | Objective function value (likelihood/NSE) |
# | `parXXX` | Parameter value (prefixed with "par") |
# | `simulation_0`, etc. | Simulated values (if saved) |
#
# ### PyDREAM
#
# **File location:** Specified via `dbname` parameter in `run_pydream()`
#
# **Columns:**
# ```
# like1,paruztwm,paruzfwm,parlztwm,...
# ```
#
# Same format as SpotPy, with `like1` for likelihood and `parXXX` for parameters.
#
# ### SciPy Differential Evolution
#
# **Note:** SciPy DE does not write progress to CSV by default. Use the `callback`
# parameter if you need progress logging:
#
# ```python
# def progress_callback(xk, convergence):
#     with open('scipy_progress.csv', 'a') as f:
#         f.write(','.join(map(str, xk)) + ',' + str(convergence) + '\n')
#     return False  # Continue optimization
#
# result = runner.run_differential_evolution(callback=progress_callback)
# ```

# %% [markdown]
# ---
# ## Setup and Configuration

# %%
# Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from datetime import datetime
import warnings
import time
import os

# Interactive widgets for auto-refresh
try:
    from IPython.display import display, clear_output
    from ipywidgets import interact, interactive, Button, Output, IntSlider, FloatSlider
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False
    print("ipywidgets not available - auto-refresh disabled")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Plot styling
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

print("=" * 70)
print("CALIBRATION PROGRESS MONITOR")
print("=" * 70)
print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Calibration Monitor loaded successfully!")

# %% [markdown]
# ---
# ## Configuration
#
# Set the path to the calibration CSV output file you want to monitor.
# Update this section based on which calibration method you're running.

# %%
# =============================================================================
# CONFIGURATION - EDIT THIS SECTION
# =============================================================================

# Path to calibration CSV output file
# Choose the file based on which calibration method you're running:

# SpotPy DREAM calibration
CSV_FILE = Path('spotpy_dream_calib.csv')

# PyDREAM calibration (requires dbname='pydream_calib' in run_pydream)
# CSV_FILE = Path('pydream_calib.csv')

# SpotPy SCE-UA calibration
# CSV_FILE = Path('sceua_calib.csv')

# Algorithm comparison outputs from Notebook 04:
# CSV_FILE = Path('algo_spotpy_dream.csv')
# CSV_FILE = Path('algo_pydream.csv')
# CSV_FILE = Path('algo_sceua.csv')

# Display configuration
MAX_PARAMS_TO_SHOW = 18  # Maximum parameters to display in plots
DOTTY_SAMPLE_SIZE = 5000  # Downsample for dotty plots if >this many samples
DARK_THEME = True  # Use dark theme for plots

# Burn-in fraction for posterior distributions (0.0 to 0.9)
# Start with 0 to see all samples, increase as calibration progresses
DEFAULT_BURNIN = 0.0

print(f"Monitoring file: {CSV_FILE.absolute()}")
print(f"File exists: {CSV_FILE.exists()}")

if not CSV_FILE.exists():
    print("\n⚠ File not found. Make sure:")
    print("  1. Calibration is running or has completed")
    print("  2. CSV_FILE path is correct")
    print("  3. dbname parameter was set when running calibration")

# %% [markdown]
# ---
# ## Data Loading Functions

# %%
def load_calibration_data(csv_path: Path, verbose: bool = True) -> pd.DataFrame:
    """
    Load SpotPy/PyDREAM calibration CSV file.
    
    Handles:
    - Incomplete files (still being written)
    - Missing values
    - Different column naming conventions
    
    Args:
        csv_path: Path to CSV file
        verbose: Print loading info
        
    Returns:
        DataFrame with columns: iteration, likelihood, and parameter columns
    """
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Get file stats
    file_size = csv_path.stat().st_size / 1024  # KB
    file_mtime = datetime.fromtimestamp(csv_path.stat().st_mtime)
    
    if verbose:
        print(f"Loading: {csv_path.name}")
        print(f"  Size: {file_size:.1f} KB")
        print(f"  Last modified: {file_mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Read CSV - handle incomplete last line
    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip')
    except Exception as e:
        # Fallback: read line by line
        with open(csv_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 2:
            raise ValueError(f"CSV file too short: {len(lines)} lines")
        
        # Parse header
        header = lines[0].strip().split(',')
        
        # Parse data, skipping bad lines
        data = []
        for line in lines[1:]:
            try:
                values = line.strip().split(',')
                if len(values) == len(header):
                    data.append([float(v) for v in values])
            except (ValueError, IndexError):
                continue
        
        df = pd.DataFrame(data, columns=header)
    
    if verbose:
        print(f"  Loaded {len(df)} samples")
    
    return df


def extract_parameters(df: pd.DataFrame) -> tuple:
    """
    Extract parameter names and objective column from SpotPy DataFrame.
    
    Returns:
        Tuple of (param_names, objective_col, param_df)
    """
    # Find objective/likelihood column
    obj_col = None
    for col in ['like1', 'like', 'likelihood', 'objectivefunction']:
        if col in df.columns:
            obj_col = col
            break
    
    if obj_col is None:
        raise ValueError(f"No objective column found. Columns: {list(df.columns)}")
    
    # Find parameter columns (start with 'par')
    param_cols = [col for col in df.columns if col.startswith('par')]
    
    # Clean parameter names (remove 'par' prefix)
    param_names = [col[3:] if col.startswith('par') else col for col in param_cols]
    
    # Create clean DataFrame
    param_df = df[param_cols].copy()
    param_df.columns = param_names
    param_df['likelihood'] = df[obj_col]
    param_df['iteration'] = np.arange(len(df))
    
    return param_names, obj_col, param_df


def get_calibration_stats(df: pd.DataFrame, param_names: list) -> dict:
    """Calculate summary statistics from calibration data."""
    stats = {
        'n_samples': len(df),
        'best_likelihood': df['likelihood'].max(),
        'best_idx': df['likelihood'].idxmax(),
        'mean_likelihood': df['likelihood'].mean(),
        'std_likelihood': df['likelihood'].std(),
    }
    
    # Best parameters
    stats['best_params'] = {}
    for param in param_names:
        if param in df.columns:
            stats['best_params'][param] = df.loc[stats['best_idx'], param]
    
    return stats

# %% [markdown]
# ---
# ## Load Current Data

# %%
# Load the calibration data
try:
    raw_df = load_calibration_data(CSV_FILE)
    param_names, obj_col, calib_df = extract_parameters(raw_df)
    stats = get_calibration_stats(calib_df, param_names)
    
    print(f"\n{'='*60}")
    print("CALIBRATION PROGRESS SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples:      {stats['n_samples']:,}")
    print(f"Best likelihood:    {stats['best_likelihood']:.6f}")
    print(f"Mean likelihood:    {stats['mean_likelihood']:.6f}")
    print(f"Std likelihood:     {stats['std_likelihood']:.6f}")
    print(f"\nNumber of parameters: {len(param_names)}")
    print(f"Parameters: {', '.join(param_names[:10])}{'...' if len(param_names) > 10 else ''}")
    
    print(f"\n{'='*60}")
    print("BEST PARAMETERS")
    print(f"{'='*60}")
    for param, value in stats['best_params'].items():
        print(f"  {param:15s}: {value:.6f}")
    
except FileNotFoundError as e:
    print(f"ERROR: {e}")
    print("\nPlease update CSV_FILE in the Configuration section above.")
    raw_df = None
    calib_df = None

# %% [markdown]
# ---
# ## Visualization Styling

# %%
# Dark theme color palette
if DARK_THEME:
    COLORS = {
        'background': '#1a1a2e',
        'panel': '#16213e',
        'text': '#eaeaea',
        'accent': '#e94560',
        'trace': '#4ecdc4',
        'hist': '#45b7d1',
        'scatter': '#f7dc6f',
        'best': '#ff6b6b',
        'kde': '#4ecdc4',
        'grid': '#0f3460',
    }
else:
    COLORS = {
        'background': 'white',
        'panel': '#f8f9fa',
        'text': '#333333',
        'accent': '#e94560',
        'trace': '#2ecc71',
        'hist': '#3498db',
        'scatter': '#f39c12',
        'best': '#e74c3c',
        'kde': '#2ecc71',
        'grid': '#ecf0f1',
    }


def apply_style(fig, axes):
    """Apply consistent styling to figures."""
    if DARK_THEME:
        fig.patch.set_facecolor(COLORS['background'])
        if isinstance(axes, np.ndarray):
            for ax in axes.flatten():
                ax.set_facecolor(COLORS['panel'])
                ax.tick_params(colors=COLORS['text'])
                ax.xaxis.label.set_color(COLORS['text'])
                ax.yaxis.label.set_color(COLORS['text'])
                ax.title.set_color(COLORS['text'])
        else:
            axes.set_facecolor(COLORS['panel'])
            axes.tick_params(colors=COLORS['text'])

# %% [markdown]
# ---
# ## 1. Objective Function Progress
#
# Track how the objective function evolves over iterations.
# The running best should improve over time.

# %%
def plot_objective_progress(df: pd.DataFrame, figsize=(14, 5)):
    """Plot objective function progress over iterations."""
    if df is None or len(df) == 0:
        print("No data to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    apply_style(fig, np.array([ax1, ax2]))
    
    iterations = df['iteration'].values
    objectives = df['likelihood'].values
    
    # Left: All samples scatter plot
    ax1.scatter(iterations, objectives, alpha=0.4, s=3, c=COLORS['scatter'])
    
    # Mark best
    best_idx = np.argmax(objectives)
    best_val = objectives[best_idx]
    ax1.axhline(y=best_val, color=COLORS['best'], linestyle='--', 
                linewidth=2, label=f'Best: {best_val:.4f}')
    ax1.scatter([best_idx], [best_val], color=COLORS['best'], s=100, 
                zorder=5, marker='*', edgecolor='white')
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Likelihood/NSE')
    ax1.set_title('All Objective Function Values')
    ax1.legend(loc='lower right', facecolor=COLORS['panel'], edgecolor=COLORS['accent'])
    
    # Right: Running best
    running_best = np.maximum.accumulate(objectives)
    ax2.plot(iterations, running_best, color=COLORS['trace'], linewidth=2)
    ax2.fill_between(iterations, running_best.min(), running_best, 
                     alpha=0.3, color=COLORS['trace'])
    ax2.axhline(y=best_val, color=COLORS['best'], linestyle='--', linewidth=1.5)
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Best Likelihood/NSE')
    ax2.set_title('Running Best Objective')
    
    # Add progress annotation
    current_iter = len(df)
    ax2.annotate(f'Iteration: {current_iter:,}\nBest: {best_val:.4f}',
                xy=(0.98, 0.05), xycoords='axes fraction',
                ha='right', va='bottom',
                fontsize=11, color=COLORS['text'],
                bbox=dict(boxstyle='round', facecolor=COLORS['panel'], 
                         edgecolor=COLORS['accent'], alpha=0.9))
    
    plt.suptitle('Objective Function Progress', fontsize=14, fontweight='bold',
                 color=COLORS['text'], y=1.02)
    plt.tight_layout()
    plt.show()
    
    return fig


# Plot if data is loaded
if calib_df is not None:
    plot_objective_progress(calib_df)

# %% [markdown]
# ---
# ## 2. Dotty Plots (Parameter Sensitivity)
#
# Show the relationship between each parameter value and the objective.
# - **Clear peaks**: Parameter is well-identified
# - **Flat scatter**: Parameter is insensitive
# - **Multiple peaks**: Equifinality issues

# %%
def plot_dotty_plots(df: pd.DataFrame, params: list = None, 
                     max_params: int = MAX_PARAMS_TO_SHOW,
                     sample_size: int = DOTTY_SAMPLE_SIZE,
                     figsize: tuple = None):
    """Create dotty plots showing parameter vs objective function."""
    if df is None or len(df) == 0:
        print("No data to plot")
        return
    
    if params is None:
        params = [col for col in df.columns 
                  if col not in ['iteration', 'likelihood']]
    
    params = params[:max_params]
    n_params = len(params)
    
    n_cols = min(4, n_params)
    n_rows = int(np.ceil(n_params / n_cols))
    
    if figsize is None:
        figsize = (3.5 * n_cols, 2.5 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()
    apply_style(fig, axes)
    
    # Downsample if needed
    if len(df) > sample_size:
        plot_df = df.sample(n=sample_size, random_state=42)
    else:
        plot_df = df
    
    objectives = plot_df['likelihood'].values
    best_idx = df['likelihood'].idxmax()
    obj_norm = (objectives - objectives.min()) / (objectives.max() - objectives.min() + 1e-10)
    
    for i, (ax, param) in enumerate(zip(axes[:n_params], params)):
        if param in plot_df.columns:
            values = plot_df[param].values
            scatter = ax.scatter(values, objectives, c=obj_norm, 
                                cmap='viridis', alpha=0.4, s=8, edgecolor='none')
            best_val = df.loc[best_idx, param]
            ax.axvline(x=best_val, color=COLORS['best'], linestyle='--', 
                      linewidth=2, alpha=0.8)
            ax.set_xlabel(param, fontsize=9)
            if i % n_cols == 0:
                ax.set_ylabel('Likelihood', fontsize=9)
            ax.tick_params(labelsize=8)
    
    for ax in axes[n_params:]:
        ax.set_visible(False)
    
    plt.suptitle('Dotty Plots - Parameter Sensitivity', fontsize=14, 
                 fontweight='bold', color=COLORS['text'], y=1.02)
    plt.tight_layout()
    plt.show()
    return fig


if calib_df is not None:
    plot_dotty_plots(calib_df)

# %% [markdown]
# ---
# ## 3. Parameter Traces (MCMC Chain Evolution)
#
# Good mixing shows oscillation around stable values.
# Trends or stuck chains indicate convergence problems.

# %%
def plot_parameter_traces(df: pd.DataFrame, params: list = None,
                          max_params: int = 8, figsize: tuple = None):
    """Plot parameter traces over iterations."""
    if df is None or len(df) == 0:
        print("No data to plot")
        return
    
    if params is None:
        params = [col for col in df.columns 
                  if col not in ['iteration', 'likelihood']]
    
    params = params[:max_params]
    n_params = len(params)
    
    if figsize is None:
        figsize = (14, 2 * n_params)
    
    fig, axes = plt.subplots(n_params, 1, figsize=figsize, sharex=True)
    if n_params == 1:
        axes = [axes]
    
    apply_style(fig, np.array(axes))
    
    iterations = df['iteration'].values
    best_idx = df['likelihood'].idxmax()
    
    for ax, param in zip(axes, params):
        if param in df.columns:
            values = df[param].values
            ax.plot(iterations, values, color=COLORS['trace'], 
                   alpha=0.6, linewidth=0.5)
            
            window = max(len(values) // 50, 10)
            if len(values) > window:
                ma = pd.Series(values).rolling(window=window, center=True).mean()
                ax.plot(iterations, ma, color=COLORS['accent'], 
                       linewidth=2, label='Moving avg')
            
            best_val = df.loc[best_idx, param]
            ax.axhline(y=best_val, color=COLORS['best'], linestyle='--',
                      linewidth=1.5, label=f'Best: {best_val:.3f}')
            
            ax.set_ylabel(param, fontsize=10)
            ax.legend(loc='upper right', fontsize=8, 
                     facecolor=COLORS['panel'], edgecolor=COLORS['accent'])
    
    axes[-1].set_xlabel('Iteration', fontsize=11)
    
    plt.suptitle('Parameter Traces (MCMC Chain Evolution)', fontsize=14,
                 fontweight='bold', color=COLORS['text'], y=1.01)
    plt.tight_layout()
    plt.show()
    return fig


if calib_df is not None:
    params_to_show = param_names[:8] if len(param_names) > 8 else param_names
    plot_parameter_traces(calib_df, params=params_to_show)

# %% [markdown]
# ---
# ## 4. Posterior Distributions
#
# Use burn-in to exclude early (unconverged) samples.
# - **Narrow peaks**: Well-identified
# - **Wide/flat**: Poorly identified
# - **Multiple modes**: Equifinality

# %%
def plot_posterior_distributions(df: pd.DataFrame, params: list = None,
                                  burnin: float = DEFAULT_BURNIN,
                                  max_params: int = MAX_PARAMS_TO_SHOW,
                                  figsize: tuple = None):
    """Plot posterior parameter distributions."""
    if df is None or len(df) == 0:
        print("No data to plot")
        return
    
    n_burnin = int(len(df) * burnin)
    plot_df = df.iloc[n_burnin:].copy()
    
    if len(plot_df) < 10:
        print(f"Too few samples after burn-in ({len(plot_df)}). Reduce burn-in fraction.")
        return
    
    if params is None:
        params = [col for col in df.columns 
                  if col not in ['iteration', 'likelihood']]
    
    params = params[:max_params]
    n_params = len(params)
    
    n_cols = min(4, n_params)
    n_rows = int(np.ceil(n_params / n_cols))
    
    if figsize is None:
        figsize = (3.5 * n_cols, 2.5 * n_rows)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()
    apply_style(fig, axes)
    
    best_idx = df['likelihood'].idxmax()
    
    for i, (ax, param) in enumerate(zip(axes[:n_params], params)):
        if param in plot_df.columns:
            values = plot_df[param].dropna().values
            
            if len(values) > 0:
                ax.hist(values, bins=50, density=True, alpha=0.7,
                       color=COLORS['hist'], edgecolor='white', linewidth=0.3)
                
                try:
                    from scipy import stats
                    kde = stats.gaussian_kde(values)
                    x_range = np.linspace(values.min(), values.max(), 200)
                    ax.plot(x_range, kde(x_range), color=COLORS['kde'], linewidth=2)
                except Exception:
                    pass
                
                best_val = df.loc[best_idx, param]
                ax.axvline(x=best_val, color=COLORS['best'], linestyle='--',
                          linewidth=2, label='Best')
                
                mean_val = np.mean(values)
                std_val = np.std(values)
                ax.set_title(f'{param}\n{mean_val:.3g} ± {std_val:.3g}', 
                            fontsize=9, color=COLORS['text'])
        
        ax.tick_params(labelsize=8)
    
    for ax in axes[n_params:]:
        ax.set_visible(False)
    
    title = f'Posterior Distributions (burn-in: {burnin*100:.0f}%, n={len(plot_df):,})'
    plt.suptitle(title, fontsize=14, fontweight='bold', 
                 color=COLORS['text'], y=1.02)
    plt.tight_layout()
    plt.show()
    return fig


if calib_df is not None:
    plot_posterior_distributions(calib_df, burnin=0.3)

# %% [markdown]
# ---
# ## 5. Interactive Posterior with Burn-in Control

# %%
if WIDGETS_AVAILABLE and calib_df is not None:
    @interact(burnin=FloatSlider(min=0.0, max=0.9, step=0.05, value=0.3,
                                  description='Burn-in:'))
    def interactive_posterior(burnin):
        plot_posterior_distributions(calib_df, burnin=burnin)
else:
    print("Interactive widgets not available or no data loaded.")
    print("Run plot_posterior_distributions(calib_df, burnin=X) manually")

# %% [markdown]
# ---
# ## 6. Convergence Diagnostics (Gelman-Rubin R-hat)
#
# ### Understanding R-hat
#
# The Gelman-Rubin statistic compares variance within chains to variance between chains:
# - **R-hat ≈ 1.0**: Chains have converged (exploring same distribution)
# - **R-hat > 1.1**: Chains haven't converged, need more iterations
# - **R-hat > 1.2**: Serious convergence problems
#
# ### Interpretation
#
# | R-hat | Status | Action |
# |-------|--------|--------|
# | < 1.05 | ✓ Converged | Safe to use posteriors |
# | 1.05 - 1.1 | OK | Consider more iterations |
# | 1.1 - 1.2 | ⚠ Borderline | Definitely need more iterations |
# | > 1.2 | ✗ Not converged | Check for problems, restart if needed |

# %%
def estimate_gelman_rubin(df: pd.DataFrame, param: str, n_chains: int = 5) -> float:
    """Estimate Gelman-Rubin R-hat statistic for a parameter."""
    if param not in df.columns:
        return np.nan
    
    values = df[param].values
    n_samples = len(values)
    samples_per_chain = n_samples // n_chains
    
    if samples_per_chain < 10:
        return np.nan
    
    chains = []
    for chain_idx in range(n_chains):
        chain_samples = values[chain_idx::n_chains][:samples_per_chain]
        half_point = len(chain_samples) // 2
        chains.append(chain_samples[half_point:])
    
    chains = np.array(chains)
    n = chains.shape[1]
    m = chains.shape[0]
    
    W = np.mean(np.var(chains, axis=1, ddof=1))
    chain_means = np.mean(chains, axis=1)
    B = n * np.var(chain_means, ddof=1)
    var_estimate = ((n - 1) / n) * W + (1 / n) * B
    
    if W > 0:
        r_hat = np.sqrt(var_estimate / W)
    else:
        r_hat = np.nan
    
    return r_hat


def print_convergence_diagnostics(df: pd.DataFrame, params: list = None, 
                                   n_chains: int = 5):
    """Print Gelman-Rubin convergence diagnostics."""
    if df is None:
        print("No data available")
        return
    
    if params is None:
        params = [col for col in df.columns 
                  if col not in ['iteration', 'likelihood']]
    
    print(f"\n{'='*60}")
    print("GELMAN-RUBIN CONVERGENCE DIAGNOSTICS")
    print(f"{'='*60}")
    print(f"Samples: {len(df):,} | Assumed chains: {n_chains}")
    print(f"\nR-hat < 1.1 indicates convergence")
    print(f"R-hat < 1.05 indicates good convergence")
    print(f"{'-'*60}")
    print(f"{'Parameter':20s} {'R-hat':>10s} {'Status':>15s}")
    print(f"{'-'*60}")
    
    converged = 0
    for param in params[:MAX_PARAMS_TO_SHOW]:
        r_hat = estimate_gelman_rubin(df, param, n_chains)
        
        if np.isnan(r_hat):
            status = "N/A"
        elif r_hat < 1.05:
            status = "✓ Converged"
            converged += 1
        elif r_hat < 1.1:
            status = "OK"
            converged += 1
        elif r_hat < 1.2:
            status = "⚠ Borderline"
        else:
            status = "✗ Not converged"
        
        r_hat_str = f"{r_hat:.4f}" if not np.isnan(r_hat) else "N/A"
        print(f"{param:20s} {r_hat_str:>10s} {status:>15s}")
    
    print(f"{'-'*60}")
    print(f"Converged parameters: {converged}/{len(params[:MAX_PARAMS_TO_SHOW])}")


if calib_df is not None:
    print_convergence_diagnostics(calib_df, param_names, n_chains=5)

# %% [markdown]
# ---
# ## 7. Refresh Data
#
# Reload the CSV file and update all plots.

# %%
def refresh_and_plot():
    """Reload data and regenerate all plots."""
    global raw_df, calib_df, param_names, stats
    
    print(f"\n{'='*60}")
    print(f"REFRESHING DATA - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    try:
        raw_df = load_calibration_data(CSV_FILE)
        param_names_new, obj_col, calib_df = extract_parameters(raw_df)
        param_names = param_names_new
        stats = get_calibration_stats(calib_df, param_names)
        
        print(f"\nProgress: {stats['n_samples']:,} samples | Best: {stats['best_likelihood']:.6f}")
        
        print("\n1. Objective Progress")
        plot_objective_progress(calib_df)
        
        print("\n2. Dotty Plots")
        plot_dotty_plots(calib_df)
        
        print("\n3. Parameter Traces (first 8)")
        params_to_show = param_names[:8] if len(param_names) > 8 else param_names
        plot_parameter_traces(calib_df, params=params_to_show)
        
        print("\n4. Posterior Distributions")
        plot_posterior_distributions(calib_df, burnin=0.3)
        
        print("\n5. Convergence Diagnostics")
        print_convergence_diagnostics(calib_df, param_names, n_chains=5)
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


# Uncomment to refresh:
# refresh_and_plot()

# %% [markdown]
# ---
# ## 8. Export Best Parameters

# %%
def export_best_parameters():
    """Print best parameters in formats ready for use."""
    if calib_df is None:
        print("No data loaded")
        return
    
    stats = get_calibration_stats(calib_df, param_names)
    
    print(f"\n{'='*60}")
    print("EXPORT BEST PARAMETERS")
    print(f"{'='*60}")
    print(f"Best likelihood: {stats['best_likelihood']:.6f}")
    print(f"Sample #{stats['best_idx']:,}")
    
    print(f"\n{'='*60}")
    print("Python Dictionary Format:")
    print(f"{'='*60}")
    print("best_params = {")
    for param, value in stats['best_params'].items():
        print(f"    '{param}': {value:.8f},")
    print("}")
    
    print(f"\n{'='*60}")
    print("For pyrrm model.set_parameters():")
    print(f"{'='*60}")
    params_str = ", ".join([f"'{k}': {v:.6f}" for k, v in stats['best_params'].items()])
    print(f"model.set_parameters({{{params_str}}})")


if calib_df is not None:
    export_best_parameters()

# %% [markdown]
# ---
# ## Troubleshooting Guide
#
# ### Common Issues and Solutions
#
# #### "File not found" error
# - Check that calibration is actually running
# - Verify the `CSV_FILE` path is correct
# - Ensure `dbname` and `dbformat='csv'` were set in calibration call
#
# #### No samples loading
# - Calibration may not have written any samples yet
# - Wait a few seconds and try again
# - Check the file size is growing
#
# #### R-hat values are NaN
# - Not enough samples yet (need at least 50-100 per chain)
# - Too few chains assumed (try adjusting `n_chains` parameter)
#
# #### Objective function not improving
# - Normal early in calibration - give it more iterations
# - If stuck after many iterations, check:
#   - Are parameter bounds reasonable?
#   - Is the objective function working correctly?
#   - Try different algorithm settings
#
# #### Wide/flat posteriors
# - Parameter may be insensitive (not affecting objective much)
# - May need more data or tighter bounds
# - Consider fixing or removing insensitive parameters
#
# #### Multiple peaks in posterior
# - **Equifinality**: Multiple parameter sets give similar performance
# - Common in hydrological models
# - Consider constraining with additional objectives or physical bounds

# %% [markdown]
# ---
# ## Summary
#
# This notebook provides real-time monitoring of MCMC calibrations.
#
# **Key functions:**
# - `load_calibration_data(csv_path)` - Load calibration CSV output
# - `plot_objective_progress(df)` - Objective function evolution
# - `plot_dotty_plots(df)` - Parameter sensitivity
# - `plot_parameter_traces(df)` - MCMC chain evolution
# - `plot_posterior_distributions(df, burnin)` - Parameter distributions
# - `print_convergence_diagnostics(df, params, n_chains)` - Gelman-Rubin R-hat
# - `refresh_and_plot()` - Reload data and update all plots
# - `export_best_parameters()` - Export optimal parameter values
#
# **Tips:**
# - Start with `burnin=0.0` early in calibration, increase to `0.3-0.5` later
# - R-hat < 1.1 indicates convergence, < 1.05 is ideal
# - Wide posteriors suggest poorly identified parameters
# - Multiple peaks in posteriors indicate equifinality

# %%
print("=" * 70)
print("CALIBRATION MONITOR READY")
print("=" * 70)
print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nTo monitor an ongoing calibration:")
print("  1. Update CSV_FILE to point to your calibration output")
print("  2. Run refresh_and_plot() periodically to update plots")
