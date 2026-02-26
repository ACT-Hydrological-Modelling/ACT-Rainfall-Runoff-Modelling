# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: pyrrm
#     language: python
#     name: python3
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
# ### PyDREAM
#
# **File location:** Specified via `dbname` parameter in `run_pydream()`
#
# **Columns:**
# ```
# like1,paruztwm,paruzfwm,parlztwm,...
# -0.234,45.23,32.11,125.67,55.89,...
# -0.198,48.67,35.22,130.45,58.12,...
# ```
#
# | Column | Description |
# |--------|-------------|
# | `like1` | Objective function value (likelihood/NSE) |
# | `parXXX` | Parameter value (prefixed with "par") |
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
from pathlib import Path
from datetime import datetime
import warnings
import time
import os

# Plotly for interactive visualizations
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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

OUTPUT_DIR = Path('../test_data/08_calibration_monitor')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("CALIBRATION PROGRESS MONITOR")
print("=" * 70)
print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("Calibration Monitor loaded successfully!")
print("Using Plotly for interactive visualizations")

# %% [markdown]
# ---
# ## Configuration
#
# Set the path to the calibration CSV output file you want to monitor.
# This notebook supports PyDREAM calibrations.
#
# ### Current Target: PyDREAM from Notebook 06
#
# Notebook 06 (Algorithm Comparison) runs PyDREAM calibrations for 13 objective 
# functions. The progress files are saved to:
# ```
# test_data/reports/pydream/progress_410734_sacramento_{objective}_dream[_{transformation}].csv
# ```
#
# Available objective functions to monitor:
# - `nse`, `lognse`, `invnse`, `sqrtnse`
# - `sdeb`
# - `kge`, `kge_inv`, `kge_sqrt`, `kge_log`
# - `kge_np`, `kge_np_inv`, `kge_np_sqrt`, `kge_np_log`

# %%
# =============================================================================
# CONFIGURATION - EDIT THIS SECTION
# =============================================================================

# ------------------------------------------------------------------------------
# QUICK CONFIG: Choose what to monitor
# ------------------------------------------------------------------------------
# Options:
#   'pydream_synthetic'   - Synthetic hydrograph test from Notebook 06
#   'pydream_realdata'    - Real data calibrations from Notebook 06 (13 objectives)
#   'custom'              - Custom path (edit CSV_FILE below)

CALIBRATION_SOURCE = 'pydream_synthetic'  # <-- CHANGE THIS

# For real data PyDREAM calibrations, specify the objective function
# Options: 'nse', 'lognse', 'invnse', 'sqrtnse', 'sdeb',
#          'kge', 'kge_inv', 'kge_sqrt', 'kge_log',
#          'kge_np', 'kge_np_inv', 'kge_np_sqrt', 'kge_np_log'
PYDREAM_OBJECTIVE = 'nse'

# ------------------------------------------------------------------------------
# Path Configuration (automatic based on CALIBRATION_SOURCE)
# ------------------------------------------------------------------------------
NB06_OUTPUT_DIR = Path('../test_data/06_algorithm_comparison')
PYDREAM_DIR = NB06_OUTPUT_DIR / 'reports'

if CALIBRATION_SOURCE == 'pydream_synthetic':
    NB06_FIGURES_DIR = NB06_OUTPUT_DIR / 'figures'
    CSV_FILE = NB06_FIGURES_DIR / 'pydream_synthetic_progress.csv'
    
elif CALIBRATION_SOURCE == 'pydream_realdata':
    # Real data calibrations from Notebook 06 (13 objective functions)
    obj_lower = PYDREAM_OBJECTIVE.lower()
    transform_suffixes = {
        'lognse': ('nse', 'log'), 'invnse': ('nse', 'inverse'), 'sqrtnse': ('nse', 'sqrt'),
        'kge_inv': ('kge', 'inverse'), 'kge_sqrt': ('kge', 'sqrt'), 'kge_log': ('kge', 'log'),
        'kge_np': ('kgenp', None), 'kge_np_inv': ('kgenp', 'inverse'),
        'kge_np_sqrt': ('kgenp', 'sqrt'), 'kge_np_log': ('kgenp', 'log'),
    }
    if obj_lower in transform_suffixes:
        base_obj, transform = transform_suffixes[obj_lower]
        if transform:
            CSV_FILE = PYDREAM_DIR / f'progress_410734_sacramento_{base_obj}_dream_{transform}.csv'
        else:
            CSV_FILE = PYDREAM_DIR / f'progress_410734_sacramento_{base_obj}_dream.csv'
    else:
        CSV_FILE = PYDREAM_DIR / f'progress_410734_sacramento_{obj_lower}_dream.csv'
    
elif CALIBRATION_SOURCE == 'custom':
    # Custom path - edit directly
    CSV_FILE = Path('your_calibration_file.csv')
else:
    raise ValueError(f"Unknown CALIBRATION_SOURCE: {CALIBRATION_SOURCE}")

# ------------------------------------------------------------------------------
# Display Options
# ------------------------------------------------------------------------------
MAX_PARAMS_TO_SHOW = None  # Show all parameters (set to int to limit, e.g., 18)
DOTTY_SAMPLE_SIZE = 5000  # Downsample for dotty plots if >this many samples
DARK_THEME = True  # Use dark theme for plots

# Burn-in fraction for posterior distributions (0.0 to 0.9)
# Start with 0.0 to see all samples, increase to 0.3-0.5 as calibration progresses
DEFAULT_BURNIN = 0.3

# Auto-refresh interval (seconds) for monitoring running calibrations
AUTO_REFRESH_INTERVAL = 30

# ------------------------------------------------------------------------------
# Show Available PyDREAM Progress Files
# ------------------------------------------------------------------------------
print("=" * 70)
print("CALIBRATION MONITOR CONFIGURATION")
print("=" * 70)

if CALIBRATION_SOURCE == 'pydream_synthetic':
    print(f"\nSource: SYNTHETIC HYDROGRAPH TEST (Notebook 06)")
    print(f"This monitors the PyDREAM validation run on synthetic data")
    
elif CALIBRATION_SOURCE == 'pydream_realdata':
    print(f"\nSource: REAL DATA PyDREAM (Notebook 06)")
    print(f"Objective: {PYDREAM_OBJECTIVE.upper()}")
    
    # List available progress files
    if PYDREAM_DIR.exists():
        progress_files = sorted(PYDREAM_DIR.glob('progress_410734_sacramento_*_dream*.csv'))
        if progress_files:
            print(f"\nAvailable PyDREAM progress files:")
            for f in progress_files:
                size_kb = f.stat().st_size / 1024
                mtime = datetime.fromtimestamp(f.stat().st_mtime)
                n_lines = sum(1 for _ in open(f)) - 1  # Subtract header
                parts = f.stem.replace('progress_410734_sacramento_', '').replace('_dream', '')
                obj_name = parts.upper() if parts else f.stem.upper()
                status = "✓ CURRENT" if f.name == CSV_FILE.name else ""
                print(f"  {obj_name:<12}: {n_lines:>8,} samples, {size_kb:>8.1f} KB, "
                      f"updated {mtime.strftime('%H:%M:%S')} {status}")
        else:
            print(f"\n⚠ No progress files found in {PYDREAM_DIR}")
            print("  Run Notebook 06 to start PyDREAM calibrations")
    else:
        print(f"\n⚠ PyDREAM directory not found: {PYDREAM_DIR}")

print(f"\n{'-'*70}")
print(f"Monitoring file: {CSV_FILE}")
print(f"File exists: {CSV_FILE.exists()}")

if CSV_FILE.exists():
    file_size = CSV_FILE.stat().st_size / 1024
    file_mtime = datetime.fromtimestamp(CSV_FILE.stat().st_mtime)
    print(f"File size: {file_size:.1f} KB")
    print(f"Last modified: {file_mtime.strftime('%Y-%m-%d %H:%M:%S')}")
else:
    print("\n⚠ File not found. Make sure:")
    print("  1. Calibration is running or has completed")
    print("  2. PYDREAM_OBJECTIVE matches an active calibration")
    print("  3. Run Notebook 06 to start calibrations")

# %% [markdown]
# ---
# ## Helper: Switch Calibration Target
#
# Use this function to easily switch to monitoring a different PyDREAM calibration.

# %%
def monitor_pydream(objective: str) -> None:
    """
    Switch to monitoring a different PyDREAM calibration from Notebook 06.
    
    Args:
        objective: Name of the objective function to monitor.
                   Options: 'nse', 'lognse', 'invnse', 'sqrtnse', 'sdeb',
                           'kge', 'kge_inv', 'kge_sqrt', 'kge_log',
                           'kge_np', 'kge_np_inv', 'kge_np_sqrt', 'kge_np_log'
    
    Example:
        >>> monitor_pydream('kge')  # Switch to monitoring KGE calibration
        >>> refresh_and_plot()      # Update plots with new data
    """
    global CSV_FILE, PYDREAM_OBJECTIVE
    
    PYDREAM_OBJECTIVE = objective.lower()
    transform_suffixes = {
        'lognse': ('nse', 'log'), 'invnse': ('nse', 'inverse'), 'sqrtnse': ('nse', 'sqrt'),
        'kge_inv': ('kge', 'inverse'), 'kge_sqrt': ('kge', 'sqrt'), 'kge_log': ('kge', 'log'),
        'kge_np': ('kgenp', None), 'kge_np_inv': ('kgenp', 'inverse'),
        'kge_np_sqrt': ('kgenp', 'sqrt'), 'kge_np_log': ('kgenp', 'log'),
    }
    if PYDREAM_OBJECTIVE in transform_suffixes:
        base_obj, transform = transform_suffixes[PYDREAM_OBJECTIVE]
        if transform:
            CSV_FILE = PYDREAM_DIR / f'progress_410734_sacramento_{base_obj}_dream_{transform}.csv'
        else:
            CSV_FILE = PYDREAM_DIR / f'progress_410734_sacramento_{base_obj}_dream.csv'
    else:
        CSV_FILE = PYDREAM_DIR / f'progress_410734_sacramento_{PYDREAM_OBJECTIVE}_dream.csv'
    
    print(f"\n{'='*60}")
    print(f"SWITCHED TO: {objective.upper()} CALIBRATION")
    print(f"{'='*60}")
    
    if CSV_FILE.exists():
        file_size = CSV_FILE.stat().st_size / 1024
        file_mtime = datetime.fromtimestamp(CSV_FILE.stat().st_mtime)
        n_lines = sum(1 for _ in open(CSV_FILE)) - 1
        print(f"File: {CSV_FILE.name}")
        print(f"Samples: {n_lines:,}")
        print(f"Size: {file_size:.1f} KB")
        print(f"Last modified: {file_mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nRun refresh_and_plot() to load data and generate plots")
    else:
        print(f"⚠ File not found: {CSV_FILE}")
        print("  This calibration may not have started yet")


def list_available_calibrations() -> None:
    """List all available PyDREAM progress files from Notebook 06."""
    print(f"\n{'='*60}")
    print("AVAILABLE PYDREAM CALIBRATIONS (Notebook 06)")
    print(f"{'='*60}\n")
    
    if not PYDREAM_DIR.exists():
        print(f"⚠ Directory not found: {PYDREAM_DIR}")
        return
    
    progress_files = sorted(PYDREAM_DIR.glob('progress_410734_sacramento_*_dream*.csv'))
    
    if not progress_files:
        print("No progress files found. Run Notebook 06 to start calibrations.")
        return
    
    print(f"{'Objective':<15} {'Samples':>10} {'Size (KB)':>10} {'Last Update':>12} {'Status':<10}")
    print("-" * 60)
    
    for f in progress_files:
        parts = f.stem.replace('progress_410734_sacramento_', '').replace('_dream', '')
        obj_name = parts.upper() if parts else f.stem.upper()
        size_kb = f.stat().st_size / 1024
        mtime = datetime.fromtimestamp(f.stat().st_mtime)
        n_lines = sum(1 for _ in open(f)) - 1
        
        # Check if file is being actively written (modified in last 2 minutes)
        age_seconds = (datetime.now() - mtime).total_seconds()
        if age_seconds < 120:
            status = "🔄 Active"
        elif age_seconds < 3600:
            status = "⏸ Paused?"
        else:
            status = "✓ Complete"
        
        current = " ← CURRENT" if f.name == CSV_FILE.name else ""
        print(f"{obj_name:<15} {n_lines:>10,} {size_kb:>10.1f} {mtime.strftime('%H:%M:%S'):>12} {status}{current}")
    
    print(f"\n{'='*60}")
    print("To switch calibration: monitor_pydream('objective_name')")
    print("Example: monitor_pydream('kge')")


# Show available calibrations on startup
if CALIBRATION_SOURCE == 'pydream_realdata' and PYDREAM_DIR.exists():
    list_available_calibrations()

# %% [markdown]
# ---
# ## Data Loading Functions

# %%
def load_calibration_data(csv_path: Path, verbose: bool = True) -> pd.DataFrame:
    """
    Load PyDREAM calibration CSV file.
    
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
    Extract parameter names and objective column from calibration DataFrame.
    
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
#
# Automatically load the calibration data from the configured source.

# %%
# Load the calibration data
try:
    raw_df = load_calibration_data(CSV_FILE)
    param_names, obj_col, calib_df = extract_parameters(raw_df)
    stats = get_calibration_stats(calib_df, param_names)
    
    # Check file activity status
    file_mtime = datetime.fromtimestamp(CSV_FILE.stat().st_mtime)
    age_seconds = (datetime.now() - file_mtime).total_seconds()
    if age_seconds < 120:
        status_msg = "🔄 ACTIVELY RUNNING"
    elif age_seconds < 3600:
        status_msg = "⏸ POSSIBLY PAUSED"
    else:
        status_msg = "✓ LIKELY COMPLETE"
    
    print(f"\n{'='*60}")
    print("CALIBRATION PROGRESS SUMMARY")
    print(f"{'='*60}")
    
    if CALIBRATION_SOURCE == 'pydream_synthetic':
        print(f"Source: PyDREAM SYNTHETIC TEST (Notebook 06)")
    elif CALIBRATION_SOURCE == 'pydream_realdata':
        print(f"Source: PyDREAM REAL DATA (Notebook 06)")
        print(f"Objective: {PYDREAM_OBJECTIVE.upper()}")
    
    print(f"Status: {status_msg}")
    print(f"\nTotal samples:      {stats['n_samples']:,}")
    
    # Count valid samples (not -inf)
    valid_mask = calib_df['likelihood'] > -np.inf
    valid_count = valid_mask.sum()
    print(f"Valid samples:      {valid_count:,} ({100*valid_count/len(calib_df):.1f}%)")
    
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
    print("\n" + "="*60)
    print("FILE NOT FOUND - Possible reasons:")
    print("="*60)
    if CALIBRATION_SOURCE == 'pydream_synthetic':
        print("\n1. The synthetic hydrograph calibration hasn't started yet")
        print("   → Run the PyDREAM synthetic test in Notebook 06")
        print("\n2. The calibration completed and the file was moved/deleted")
        print(f"   → Check if file exists: {CSV_FILE}")
    elif CALIBRATION_SOURCE == 'pydream_realdata':
        print(f"\n1. The '{PYDREAM_OBJECTIVE.upper()}' calibration hasn't started yet")
        print("   → Run Notebook 06 and wait for calibration to begin")
        print(f"\n2. The objective name is incorrect")
        print("   → Run list_available_calibrations() to see available files")
        print(f"\n3. The PyDREAM directory doesn't exist")
        print(f"   → Check that {PYDREAM_DIR} exists")
    else:
        print("\n1. Check that the calibration has started")
        print("2. Verify the CSV_FILE path is correct")
        print("3. Ensure dbname parameter was set when running calibration")
    
    raw_df = None
    calib_df = None

# %% [markdown]
# ---
# ## Visualization Styling

# %%
# Plotly color palette and theme
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
        'running_best_fill': 'rgba(78, 205, 196, 0.3)',
    }
    PLOTLY_TEMPLATE = 'plotly_dark'
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
        'running_best_fill': 'rgba(46, 204, 113, 0.3)',
    }
    PLOTLY_TEMPLATE = 'plotly_white'


def get_plotly_layout(title: str = None, height: int = None, width: int = None) -> dict:
    """Get consistent Plotly layout settings."""
    layout = {
        'template': PLOTLY_TEMPLATE,
        'paper_bgcolor': COLORS['background'],
        'plot_bgcolor': COLORS['panel'],
        'font': {'color': COLORS['text'], 'family': 'Inter, sans-serif'},
        'title': {'font': {'size': 16, 'color': COLORS['text']}, 'x': 0.5},
        'hovermode': 'closest',
        'margin': {'t': 60, 'b': 50, 'l': 60, 'r': 40},
    }
    if title:
        layout['title']['text'] = title
    if height:
        layout['height'] = height
    if width:
        layout['width'] = width
    return layout

# %% [markdown]
# ---
# ## 1. Objective Function Progress
#
# Track how the objective function evolves over iterations.
# The running best should improve over time.

# %%
def plot_objective_progress(df: pd.DataFrame, height: int = 450):
    """
    Plot objective function progress over iterations using Plotly.
    
    Creates an interactive two-panel figure:
    - Left: All objective function samples as scatter plot
    - Right: Running best objective over iterations
    
    Args:
        df: DataFrame with 'iteration' and 'likelihood' columns
        height: Figure height in pixels
    
    Returns:
        Plotly figure object
    """
    if df is None or len(df) == 0:
        print("No data to plot")
        return None
    
    iterations = df['iteration'].values
    objectives = df['likelihood'].values
    
    # Calculate running best and best point
    best_idx = np.argmax(objectives)
    best_val = objectives[best_idx]
    running_best = np.maximum.accumulate(objectives)
    
    # Create subplot figure
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('All Objective Function Values', 'Running Best Objective'),
        horizontal_spacing=0.08
    )
    
    # Left panel: All samples scatter
    fig.add_trace(
        go.Scattergl(
            x=iterations, y=objectives,
            mode='markers',
            marker=dict(
                size=4,
                color=COLORS['scatter'],
                opacity=0.5
            ),
            name='Samples',
            hovertemplate='Iteration: %{x}<br>Objective: %{y:.4f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Best value horizontal line (left panel)
    fig.add_hline(
        y=best_val, line_dash='dash', line_color=COLORS['best'],
        line_width=2, annotation_text=f'Best: {best_val:.4f}',
        annotation_position='bottom right',
        row=1, col=1
    )
    
    # Best point marker (left panel)
    fig.add_trace(
        go.Scatter(
            x=[iterations[best_idx]], y=[best_val],
            mode='markers',
            marker=dict(
                size=14,
                color=COLORS['best'],
                symbol='star',
                line=dict(color='white', width=1)
            ),
            name='Best',
            hovertemplate=f'Best at iteration {best_idx}<br>Value: {best_val:.4f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Right panel: Running best with fill
    fig.add_trace(
        go.Scatter(
            x=iterations, y=running_best,
            mode='lines',
            fill='tozeroy',
            fillcolor=COLORS['running_best_fill'],
            line=dict(color=COLORS['trace'], width=2),
            name='Running Best',
            hovertemplate='Iteration: %{x}<br>Best so far: %{y:.4f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Best value horizontal line (right panel)
    fig.add_hline(
        y=best_val, line_dash='dash', line_color=COLORS['best'],
        line_width=1.5, row=1, col=2
    )
    
    # Add annotation with progress info
    current_iter = len(df)
    fig.add_annotation(
        x=0.98, y=0.05,
        xref='x2 domain', yref='y2 domain',
        text=f'<b>Iteration:</b> {current_iter:,}<br><b>Best:</b> {best_val:.4f}',
        showarrow=False,
        font=dict(size=12, color=COLORS['text']),
        bgcolor=COLORS['panel'],
        bordercolor=COLORS['accent'],
        borderwidth=1,
        borderpad=6,
        xanchor='right', yanchor='bottom'
    )
    
    # Update layout
    fig.update_layout(
        **get_plotly_layout('Objective Function Progress', height=height),
        showlegend=False
    )
    
    # Update axes
    fig.update_xaxes(title_text='Iteration', gridcolor=COLORS['grid'], row=1, col=1)
    fig.update_xaxes(title_text='Iteration', gridcolor=COLORS['grid'], row=1, col=2)
    fig.update_yaxes(title_text='Likelihood/NSE', gridcolor=COLORS['grid'], row=1, col=1)
    fig.update_yaxes(title_text='Best Likelihood/NSE', gridcolor=COLORS['grid'], row=1, col=2)
    
    fig.show()
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
                     max_params: int = None,
                     sample_size: int = DOTTY_SAMPLE_SIZE,
                     height_per_row: int = 220):
    """
    Create interactive dotty plots showing parameter vs objective function using Plotly.
    
    Dotty plots reveal parameter sensitivity:
    - Clear peaks: Parameter is well-identified
    - Flat scatter: Parameter is insensitive
    - Multiple peaks: Equifinality issues
    
    Args:
        df: DataFrame with parameter columns and 'likelihood' column
        params: List of parameter names to plot (default: all)
        max_params: Maximum number of parameters to show
        sample_size: Downsample to this many points if df is larger
        height_per_row: Height in pixels per row of plots
    
    Returns:
        Plotly figure object
    """
    if df is None or len(df) == 0:
        print("No data to plot")
        return None
    
    if params is None:
        params = [col for col in df.columns 
                  if col not in ['iteration', 'likelihood']]
    
    if max_params is not None:
        params = params[:max_params]
    n_params = len(params)
    
    if n_params == 0:
        print("No parameters to plot")
        return None
    
    n_cols = min(4, n_params)
    n_rows = int(np.ceil(n_params / n_cols))
    
    # Downsample if needed for performance
    if len(df) > sample_size:
        plot_df = df.sample(n=sample_size, random_state=42)
    else:
        plot_df = df
    
    objectives = plot_df['likelihood'].values
    best_idx = df['likelihood'].idxmax()
    
    # Normalize objectives for color scale
    obj_min, obj_max = objectives.min(), objectives.max()
    obj_range = obj_max - obj_min + 1e-10
    
    # Create subplot figure
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=params,
        horizontal_spacing=0.06,
        vertical_spacing=0.12
    )
    
    # Add dotty plots for each parameter
    for i, param in enumerate(params):
        row = i // n_cols + 1
        col = i % n_cols + 1
        
        if param in plot_df.columns:
            values = plot_df[param].values
            best_val = df.loc[best_idx, param]
            
            # Scatter plot with color scale based on objective
            fig.add_trace(
                go.Scattergl(
                    x=values, y=objectives,
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=objectives,
                        colorscale='Viridis',
                        opacity=0.5,
                        showscale=(i == 0),  # Only show colorbar for first plot
                        colorbar=dict(
                            title='Likelihood',
                            len=0.5,
                            y=0.75
                        ) if i == 0 else None
                    ),
                    name=param,
                    hovertemplate=f'{param}: %{{x:.4f}}<br>Likelihood: %{{y:.4f}}<extra></extra>',
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # Best value vertical line
            fig.add_vline(
                x=best_val, line_dash='dash', line_color=COLORS['best'],
                line_width=2, opacity=0.8,
                row=row, col=col
            )
            
            # Update axes labels
            fig.update_xaxes(
                title_text=param if row == n_rows else None,
                gridcolor=COLORS['grid'],
                row=row, col=col
            )
            fig.update_yaxes(
                title_text='Likelihood' if col == 1 else None,
                gridcolor=COLORS['grid'],
                row=row, col=col
            )
    
    # Update layout
    fig.update_layout(
        **get_plotly_layout(
            'Dotty Plots - Parameter Sensitivity',
            height=height_per_row * n_rows + 80
        ),
        showlegend=False
    )
    
    # Style subplot titles
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=11, color=COLORS['text'])
    
    fig.show()
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
                          max_params: int = None, height_per_param: int = 120,
                          n_chains: int = 1, show_individual_chains: bool = False):
    """
    Plot parameter traces over iterations using Plotly.
    
    Shows MCMC chain evolution with:
    - Raw parameter values (semi-transparent) or individual chains
    - Moving average for trend visualization (when not showing chains)
    - Best value reference line
    
    Good mixing shows oscillation around stable values.
    Trends or stuck chains indicate convergence problems.
    
    Args:
        df: DataFrame with parameter columns and 'iteration' column
        params: List of parameter names to plot (default: all)
        max_params: Maximum number of parameters to show
        height_per_param: Height in pixels per parameter subplot
        n_chains: Number of parallel MCMC chains (for PyDREAM DREAM)
        show_individual_chains: If True and n_chains > 1, plot each chain 
            with a different color to assess chain mixing and convergence.
    
    Returns:
        Plotly figure object
        
    Example:
        # Plot all chains separately (good for convergence diagnosis):
        >>> plot_parameter_traces(calib_df, n_chains=5, show_individual_chains=True)
        
        # Plot combined trace with moving average (default):
        >>> plot_parameter_traces(calib_df)
    """
    if df is None or len(df) == 0:
        print("No data to plot")
        return None
    
    if params is None:
        params = [col for col in df.columns 
                  if col not in ['iteration', 'likelihood']]
    
    if max_params is not None:
        params = params[:max_params]
    n_params = len(params)
    
    if n_params == 0:
        print("No parameters to plot")
        return None
    
    iterations = df['iteration'].values
    best_idx = df['likelihood'].idxmax()
    
    # Chain colors - distinct colors for up to 10 chains
    chain_colors = [
        '#4ecdc4',  # teal
        '#ff6b6b',  # coral
        '#ffd93d',  # yellow
        '#6bcb77',  # green
        '#4d96ff',  # blue
        '#ff9f43',  # orange
        '#a55eea',  # purple
        '#26de81',  # mint
        '#fd79a8',  # pink
        '#74b9ff',  # light blue
    ]
    
    # Create subplot figure with shared x-axis
    fig = make_subplots(
        rows=n_params, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=None  # We'll add y-axis titles instead
    )
    
    for i, param in enumerate(params):
        row = i + 1
        
        if param in df.columns:
            values = df[param].values
            best_val = df.loc[best_idx, param]
            
            if show_individual_chains and n_chains > 1:
                # Plot each chain separately with different colors
                n_samples = len(values)
                samples_per_chain = n_samples // n_chains
                
                for chain_idx in range(n_chains):
                    # Extract samples for this chain (interleaved format)
                    chain_values = values[chain_idx::n_chains][:samples_per_chain]
                    chain_iterations = np.arange(len(chain_values))
                    
                    color = chain_colors[chain_idx % len(chain_colors)]
                    
                    # Line trace for overall trend (semi-transparent)
                    fig.add_trace(
                        go.Scattergl(
                            x=chain_iterations, y=chain_values,
                            mode='lines',
                            line=dict(color=color, width=1),
                            opacity=0.3,
                            name=f'Chain {chain_idx + 1}',
                            hoverinfo='skip',  # Skip hover for lines
                            showlegend=(i == 0),  # Only show legend for first param
                            legendgroup=f'chain{chain_idx}'  # Group chains across params
                        ),
                        row=row, col=1
                    )
                    
                    # Scatter points for individual samples (more visible)
                    fig.add_trace(
                        go.Scattergl(
                            x=chain_iterations, y=chain_values,
                            mode='markers',
                            marker=dict(
                                color=color, 
                                size=4,
                                opacity=0.7
                            ),
                            name=f'Chain {chain_idx + 1}',
                            hovertemplate=f'{param} (Chain {chain_idx + 1}): %{{y:.4f}}<br>Sample: %{{x}}<extra></extra>',
                            showlegend=False,  # Don't duplicate legend
                            legendgroup=f'chain{chain_idx}'  # Group with line trace
                        ),
                        row=row, col=1
                    )
            else:
                # Original behavior: single combined trace with moving average
                # Calculate moving average
                window = max(len(values) // 50, 10)
                if len(values) > window:
                    ma = pd.Series(values).rolling(window=window, center=True).mean().values
                else:
                    ma = None
                
                # Raw trace (semi-transparent)
                fig.add_trace(
                    go.Scattergl(
                        x=iterations, y=values,
                        mode='lines',
                        line=dict(color=COLORS['trace'], width=0.8),
                        opacity=0.5,
                        name=f'{param}',
                        hovertemplate=f'{param}: %{{y:.4f}}<br>Iteration: %{{x}}<extra></extra>',
                        showlegend=False
                    ),
                    row=row, col=1
                )
                
                # Moving average
                if ma is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=iterations, y=ma,
                            mode='lines',
                            line=dict(color=COLORS['accent'], width=2),
                            name='Moving avg',
                            hovertemplate=f'{param} (MA): %{{y:.4f}}<extra></extra>',
                            showlegend=(i == 0)  # Only show in legend for first param
                        ),
                        row=row, col=1
                    )
            
            # Best value horizontal line
            fig.add_hline(
                y=best_val, line_dash='dash', line_color=COLORS['best'],
                line_width=1.5,
                annotation_text=f'Best: {best_val:.3f}',
                annotation_position='right',
                annotation_font_size=10,
                row=row, col=1
            )
            
            # Update y-axis with parameter name
            fig.update_yaxes(
                title_text=param,
                title_font_size=10,
                gridcolor=COLORS['grid'],
                row=row, col=1
            )
    
    # Update x-axis for bottom plot only
    x_label = 'Sample (per chain)' if (show_individual_chains and n_chains > 1) else 'Iteration'
    fig.update_xaxes(
        title_text=x_label,
        gridcolor=COLORS['grid'],
        row=n_params, col=1
    )
    
    # Update layout
    title = 'Parameter Traces (MCMC Chain Evolution)'
    if show_individual_chains and n_chains > 1:
        title = f'Parameter Traces by Chain ({n_chains} chains)'
    
    fig.update_layout(
        **get_plotly_layout(
            title,
            height=height_per_param * n_params + 80
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    
    fig.show()
    return fig


if calib_df is not None:
    # Show all parameters in trace plots
    plot_parameter_traces(calib_df, params=param_names)

# %% [markdown]
# ---
# ## 3b. Individual Chain Traces
#
# When using MCMC methods (PyDREAM), multiple chains run in parallel.
# Visualizing individual chains helps assess:
# - **Chain mixing**: Are all chains exploring the same regions?
# - **Convergence**: Have all chains settled to similar values?
# - **Stuck chains**: Is any chain trapped in a local optimum?
#
# Good chains should:
# - Overlap substantially after burn-in
# - Not show persistent trends
# - Not be stuck at constant values
#
# **Note**: PyDREAM typically uses 3-5 chains. The default assumption
# is interleaved samples (chain1-sample1, chain2-sample1, ..., chainN-sample1, etc.)

# %%
# Number of MCMC chains (adjust based on your calibration settings)
# PyDREAM default: typically 3-5 chains
N_CHAINS = 3

if calib_df is not None:
    # Show all parameters in individual chain traces
    print(f"Showing individual chain traces (assuming {N_CHAINS} chains)")
    print("Each color represents a different parallel MCMC chain")
    print("Good mixing: chains overlap and explore same regions")
    print("Poor mixing: chains stuck at different values or showing trends\n")
    plot_parameter_traces(
        calib_df, 
        params=param_names,
        n_chains=N_CHAINS,
        show_individual_chains=True
    )

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
                                  max_params: int = None,
                                  height_per_row: int = 220):
    """
    Plot posterior parameter distributions using Plotly.
    
    Shows histograms with KDE overlay for each parameter after burn-in removal.
    
    Interpretation:
    - Narrow peaks: Well-identified parameters
    - Wide/flat: Poorly identified parameters
    - Multiple modes: Equifinality issues
    
    Args:
        df: DataFrame with parameter columns
        params: List of parameter names to plot (default: all)
        burnin: Fraction of samples to discard as burn-in (0.0-0.9)
        max_params: Maximum number of parameters to show
        height_per_row: Height in pixels per row of plots
    
    Returns:
        Plotly figure object
    """
    if df is None or len(df) == 0:
        print("No data to plot")
        return None
    
    n_burnin = int(len(df) * burnin)
    plot_df = df.iloc[n_burnin:].copy()
    
    if len(plot_df) < 10:
        print(f"Too few samples after burn-in ({len(plot_df)}). Reduce burn-in fraction.")
        return None
    
    if params is None:
        params = [col for col in df.columns 
                  if col not in ['iteration', 'likelihood']]
    
    if max_params is not None:
        params = params[:max_params]
    n_params = len(params)
    
    if n_params == 0:
        print("No parameters to plot")
        return None
    
    n_cols = min(4, n_params)
    n_rows = int(np.ceil(n_params / n_cols))
    
    best_idx = df['likelihood'].idxmax()
    
    # Create subplot titles with stats
    subplot_titles = []
    for param in params:
        if param in plot_df.columns:
            values = plot_df[param].dropna().values
            if len(values) > 0:
                mean_val = np.mean(values)
                std_val = np.std(values)
                subplot_titles.append(f'{param}<br><sup>{mean_val:.3g} ± {std_val:.3g}</sup>')
            else:
                subplot_titles.append(param)
        else:
            subplot_titles.append(param)
    
    # Create subplot figure
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.06,
        vertical_spacing=0.15
    )
    
    for i, param in enumerate(params):
        row = i // n_cols + 1
        col = i % n_cols + 1
        
        if param in plot_df.columns:
            values = plot_df[param].dropna().values
            
            if len(values) > 0:
                best_val = df.loc[best_idx, param]
                
                # Histogram
                fig.add_trace(
                    go.Histogram(
                        x=values,
                        nbinsx=50,
                        histnorm='probability density',
                        marker=dict(
                            color=COLORS['hist'],
                            line=dict(color='white', width=0.5)
                        ),
                        opacity=0.7,
                        name=param,
                        hovertemplate=f'{param}<br>Value: %{{x:.4f}}<br>Density: %{{y:.4f}}<extra></extra>',
                        showlegend=False
                    ),
                    row=row, col=col
                )
                
                # KDE overlay
                try:
                    from scipy import stats
                    kde = stats.gaussian_kde(values)
                    x_range = np.linspace(values.min(), values.max(), 200)
                    kde_values = kde(x_range)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x_range, y=kde_values,
                            mode='lines',
                            line=dict(color=COLORS['kde'], width=2),
                            name='KDE',
                            hovertemplate=f'Density: %{{y:.4f}}<extra></extra>',
                            showlegend=(i == 0)  # Only show in legend for first plot
                        ),
                        row=row, col=col
                    )
                except Exception:
                    pass
                
                # Best value vertical line
                fig.add_vline(
                    x=best_val, line_dash='dash', line_color=COLORS['best'],
                    line_width=2,
                    row=row, col=col
                )
                
                # Update axes
                fig.update_xaxes(gridcolor=COLORS['grid'], row=row, col=col)
                fig.update_yaxes(gridcolor=COLORS['grid'], row=row, col=col)
    
    # Update layout
    title = f'Posterior Distributions (burn-in: {burnin*100:.0f}%, n={len(plot_df):,})'
    fig.update_layout(
        **get_plotly_layout(title, height=height_per_row * n_rows + 80),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    
    # Style subplot titles
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=11, color=COLORS['text'])
    
    fig.show()
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
        """Interactive posterior distribution viewer with burn-in slider."""
        plot_posterior_distributions(calib_df, burnin=burnin)
else:
    print("Interactive widgets not available or no data loaded.")
    print("Run plot_posterior_distributions(calib_df, burnin=X) manually to explore")

# %% [markdown]
# ---
# ## 6. Convergence Diagnostics (Gelman-Rubin R-hat)
#
# ### Understanding R-hat
#
# The Gelman-Rubin statistic (R-hat) compares variance **within** chains to variance 
# **between** chains. It measures whether multiple parallel chains have converged to
# the same posterior distribution.
#
# **Implementation**: Uses PyDREAM's `Gelman_Rubin` function when available, which:
# - Automatically removes 50% burn-in from each chain
# - Computes: `R-hat = sqrt(var_estimate / W)` where:
#   - `W` = mean within-chain variance
#   - `B` = between-chain variance  
#   - `var_estimate = W*(1 - 1/n) + B`
#
# **Chain Detection**: 
# - If your CSV has a `chain` column (PyDREAM output), chains are detected automatically
# - Otherwise, assumes interleaved format
#
# ### Interpretation
#
# | R-hat | Status | Action |
# |-------|--------|--------|
# | < 1.01 | ✓✓ Excellent | Chains well-converged, posteriors reliable |
# | < 1.05 | ✓ Good | Safe to use posteriors |
# | < 1.1 | ~ Acceptable | Usable, but more iterations recommended |
# | 1.1 - 1.2 | ⚠ Borderline | Definitely need more iterations |
# | > 1.2 | ✗ Not converged | Check for problems, extend run |
#
# ### Sample Size Requirements
#
# R-hat requires sufficient samples **per chain** to be reliable:
#
# | Samples/Chain (after burn-in) | Reliability |
# |------------------------------|-------------|
# | < 20 | Insufficient - cannot compute |
# | 20-50 | Unreliable - results not trustworthy |
# | 50-100 | Marginal - may be noisy |
# | > 100 | Reliable - estimates stable |
#
# **Note**: PyDREAM uses 50% burn-in, so you need 2× these numbers in total samples per chain.

# %%
# =============================================================================
# GELMAN-RUBIN CONVERGENCE DIAGNOSTICS
# =============================================================================
# Uses PyDREAM's implementation when available, with proper chain handling.
# PyDREAM stores chain information in CSV output, making this much more reliable.

# Try to import PyDREAM's Gelman-Rubin implementation
try:
    from pydream.convergence import Gelman_Rubin as pydream_gelman_rubin
    PYDREAM_GR_AVAILABLE = True
except ImportError:
    PYDREAM_GR_AVAILABLE = False

# Minimum samples per chain for reliable R-hat (after burn-in removal)
# PyDREAM uses 50% burn-in internally, so these are post-burn-in samples
MIN_SAMPLES_PER_CHAIN_RELIABLE = 100
MIN_SAMPLES_PER_CHAIN_MARGINAL = 50
MIN_SAMPLES_PER_CHAIN_MINIMUM = 20


def _extract_chains_from_dataframe(df: pd.DataFrame, param_names: list) -> tuple:
    """
    Extract chain data from DataFrame in PyDREAM's expected format.
    
    PyDREAM expects: List[np.ndarray] where each array is (nsamples, nparams)
    
    Args:
        df: DataFrame with parameter columns and optionally 'chain' column
        param_names: List of parameter names
    
    Returns:
        Tuple of (chains_list, n_chains, samples_per_chain)
        where chains_list is [chain0_array, chain1_array, ...]
        and each chain array has shape (samples_per_chain, n_params)
    """
    # Check if we have explicit chain information (from PyDREAM adapter)
    if 'chain' in df.columns:
        # Group by chain
        chain_groups = df.groupby('chain')
        n_chains = len(chain_groups)
        
        chains = []
        for chain_id in sorted(df['chain'].unique()):
            chain_df = df[df['chain'] == chain_id]
            # Extract parameter values as 2D array (samples x params)
            chain_data = chain_df[param_names].values
            chains.append(chain_data)
        
        # Find minimum chain length (chains may have slightly different lengths)
        min_len = min(len(c) for c in chains)
        chains = [c[:min_len] for c in chains]
        samples_per_chain = min_len
        
    else:
        # No chain column - assume interleaved format
        # DREAM writes samples interleaved: c1s1, c2s1, c3s1, c1s2, c2s2, c3s2, ...
        # Default assumption: try to detect or use 3 chains
        
        # Try to infer number of chains from data patterns
        # For now, use a reasonable default
        n_chains = 3  # Reasonable DREAM default
        
        values = df[param_names].values
        n_total = len(values)
        samples_per_chain = n_total // n_chains
        
        # Reshape assuming interleaved format
        chains = []
        for chain_idx in range(n_chains):
            chain_data = values[chain_idx::n_chains][:samples_per_chain]
            chains.append(chain_data)
    
    return chains, n_chains, samples_per_chain


def compute_gelman_rubin(df: pd.DataFrame, param_names: list) -> dict:
    """
    Compute Gelman-Rubin R-hat using PyDREAM's implementation.
    
    This function properly extracts chain information from the DataFrame
    (using the 'chain' column if available from PyDREAM output) and
    computes R-hat using PyDREAM's standard formula:
    
    R-hat = sqrt(var_estimate / W)
    
    where:
    - W = mean within-chain variance
    - B = between-chain variance  
    - var_estimate = W*(1 - 1/n) + B
    - Burn-in: 50% of each chain is discarded
    
    Args:
        df: DataFrame with parameter columns
        param_names: List of parameter names
        
    Returns:
        Dictionary with:
        - 'r_hat': dict of {param: r_hat_value}
        - 'n_chains': number of chains detected
        - 'samples_per_chain': samples per chain (before burn-in)
        - 'effective_samples': samples per chain after 50% burn-in
        - 'reliability': 'reliable', 'marginal', 'unreliable', or 'insufficient'
        - 'method': 'pydream' or 'fallback'
    """
    # Extract chains in PyDREAM's expected format
    chains, n_chains, samples_per_chain = _extract_chains_from_dataframe(df, param_names)
    
    # PyDREAM uses 50% burn-in internally
    effective_samples = samples_per_chain // 2
    
    # Determine reliability
    if effective_samples < MIN_SAMPLES_PER_CHAIN_MINIMUM:
        reliability = 'insufficient'
    elif effective_samples < MIN_SAMPLES_PER_CHAIN_MARGINAL:
        reliability = 'unreliable'
    elif effective_samples < MIN_SAMPLES_PER_CHAIN_RELIABLE:
        reliability = 'marginal'
    else:
        reliability = 'reliable'
    
    # Compute R-hat
    r_hat_values = {}
    
    if PYDREAM_GR_AVAILABLE and reliability != 'insufficient':
        try:
            # Use PyDREAM's implementation directly
            # It expects list of arrays: [chain0, chain1, ...] where each is (nsamples, nparams)
            gr_array = pydream_gelman_rubin(chains)
            
            for i, param in enumerate(param_names):
                r_hat_values[param] = float(gr_array[i])
            
            method = 'pydream'
            
        except Exception as e:
            # Fallback if PyDREAM fails
            warnings.warn(f"PyDREAM Gelman-Rubin failed: {e}, using fallback")
            method = 'fallback'
            r_hat_values = _compute_gelman_rubin_fallback(chains, param_names)
    elif reliability == 'insufficient':
        # Not enough samples
        for param in param_names:
            r_hat_values[param] = np.nan
        method = 'insufficient_samples'
    else:
        # PyDREAM not available, use fallback
        method = 'fallback'
        r_hat_values = _compute_gelman_rubin_fallback(chains, param_names)
    
    return {
        'r_hat': r_hat_values,
        'n_chains': n_chains,
        'samples_per_chain': samples_per_chain,
        'effective_samples': effective_samples,
        'reliability': reliability,
        'method': method,
    }


def _compute_gelman_rubin_fallback(chains: list, param_names: list) -> dict:
    """
    Fallback Gelman-Rubin implementation matching PyDREAM's formula.
    
    Used when PyDREAM is not available. Implements the same algorithm:
    - 50% burn-in removal
    - Standard Gelman-Rubin formula
    
    Reference:
        Gelman, A., & Rubin, D. B. (1992). Inference from iterative simulation 
        using multiple sequences. Statistical Science, 7(4), 457-472.
    """
    nchains = len(chains)
    nsamples = len(chains[0])
    nburnin = nsamples // 2
    
    r_hat_values = {}
    
    for i, param in enumerate(param_names):
        # Extract parameter values for all chains, after burn-in
        param_chains = [chain[nburnin:, i] for chain in chains]
        
        # Within-chain variance
        chain_vars = [np.var(c, ddof=1) for c in param_chains]
        W = np.mean(chain_vars)
        
        # Between-chain variance
        chain_means = [np.mean(c) for c in param_chains]
        B = np.var(chain_means, ddof=1)
        
        # Variance estimate
        n_post_burnin = nsamples - nburnin
        var_est = W * (1 - 1./n_post_burnin) + B
        
        # R-hat
        if W > 1e-10:
            r_hat = np.sqrt(var_est / W)
        else:
            r_hat = np.nan
        
        r_hat_values[param] = float(r_hat)
    
    return r_hat_values


def print_convergence_diagnostics(df: pd.DataFrame, params: list = None, 
                                   n_chains: int = None):
    """
    Print Gelman-Rubin convergence diagnostics using PyDREAM's implementation.
    
    This function automatically detects chain information from the DataFrame
    (if 'chain' column is present from PyDREAM output) and computes R-hat
    using the standard Gelman-Rubin formula with 50% burn-in.
    
    Args:
        df: DataFrame with parameter columns
        params: List of parameter names (default: auto-detect)
        n_chains: Ignored if 'chain' column present in df (auto-detected).
                  Otherwise used for interleaved format.
    """
    if df is None:
        print("No data available")
        return
    
    if params is None:
        params = [col for col in df.columns 
                  if col not in ['iteration', 'likelihood', 'chain', 'simulation']]
    
    # Check for chain column
    has_chain_col = 'chain' in df.columns
    
    # Compute R-hat using PyDREAM's method
    gr_results = compute_gelman_rubin(df, params)
    
    n_chains_detected = gr_results['n_chains']
    samples_per_chain = gr_results['samples_per_chain']
    effective_samples = gr_results['effective_samples']
    reliability = gr_results['reliability']
    method = gr_results['method']
    r_hat_dict = gr_results['r_hat']
    
    print(f"\n{'='*70}")
    print("GELMAN-RUBIN CONVERGENCE DIAGNOSTICS")
    print(f"{'='*70}")
    print(f"Method: {'PyDREAM' if method == 'pydream' else 'Fallback implementation'}")
    print(f"Chain detection: {'Explicit (chain column)' if has_chain_col else 'Inferred (interleaved)'}")
    print(f"Chains detected: {n_chains_detected}")
    print(f"Samples per chain: {samples_per_chain:,} (total: {len(df):,})")
    print(f"After 50% burn-in: {effective_samples:,} samples/chain")
    
    # Reliability warning
    if reliability == 'insufficient':
        print(f"\n⚠️  WARNING: Insufficient samples ({effective_samples} < {MIN_SAMPLES_PER_CHAIN_MINIMUM})")
        print(f"   R-hat cannot be reliably computed. Need more iterations.")
        print(f"   Minimum required: ~{MIN_SAMPLES_PER_CHAIN_MINIMUM * 2 * n_chains_detected:,} total samples")
    elif reliability == 'unreliable':
        print(f"\n⚠️  WARNING: Very few samples ({effective_samples} < {MIN_SAMPLES_PER_CHAIN_MARGINAL})")
        print(f"   R-hat estimates are UNRELIABLE. Results should not be trusted.")
    elif reliability == 'marginal':
        print(f"\n⚠️  CAUTION: Marginal sample size ({effective_samples} < {MIN_SAMPLES_PER_CHAIN_RELIABLE})")
        print(f"   R-hat estimates may be noisy. Consider more iterations.")
    else:
        print(f"\n✓ Sample size adequate for R-hat estimation")
    
    print(f"\nR-hat interpretation:")
    print(f"  < 1.01 = Excellent convergence")
    print(f"  < 1.05 = Good convergence") 
    print(f"  < 1.1  = Acceptable")
    print(f"  < 1.2  = Borderline (more iterations recommended)")
    print(f"  > 1.2  = Not converged")
    print(f"{'─'*70}")
    print(f"{'Parameter':20s} {'R-hat':>10s} {'Status':>20s}")
    print(f"{'─'*70}")
    
    converged = 0
    good_convergence = 0
    excellent_convergence = 0
    
    # Show all parameters in convergence diagnostics
    for param in params:
        r_hat = r_hat_dict.get(param, np.nan)
        
        # Determine status based on R-hat value
        if np.isnan(r_hat) or np.isinf(r_hat):
            status = "N/A"
        elif reliability in ['unreliable', 'insufficient']:
            # Don't claim convergence with unreliable estimates
            status = f"? ({r_hat:.3f})"
        elif r_hat < 1.01:
            status = "✓✓ Excellent"
            converged += 1
            good_convergence += 1
            excellent_convergence += 1
        elif r_hat < 1.05:
            status = "✓  Good"
            converged += 1
            good_convergence += 1
        elif r_hat < 1.1:
            status = "~  Acceptable"
            converged += 1
        elif r_hat < 1.2:
            status = "⚠  Borderline"
        else:
            status = "✗  Not converged"
        
        r_hat_str = f"{r_hat:.4f}" if not (np.isnan(r_hat) or np.isinf(r_hat)) else "N/A"
        print(f"{param:20s} {r_hat_str:>10s} {status:>20s}")
    
    print(f"{'─'*70}")
    
    n_params = len(params)
    
    if reliability in ['reliable', 'marginal']:
        print(f"\nSummary:")
        print(f"  Excellent (R-hat < 1.01): {excellent_convergence}/{n_params}")
        print(f"  Good (R-hat < 1.05):      {good_convergence}/{n_params}")
        print(f"  Converged (R-hat < 1.1):  {converged}/{n_params}")
        
        if converged < n_params:
            not_converged = n_params - converged
            print(f"\n  ⚠️  {not_converged} parameter(s) not converged - consider more iterations")
        else:
            print(f"\n  ✓ All parameters converged!")
    else:
        print(f"\n⚠️  Convergence assessment not reliable with current sample size")
        print(f"   Minimum recommended: {MIN_SAMPLES_PER_CHAIN_RELIABLE} samples/chain after burn-in")
        print(f"   Current: {effective_samples} samples/chain")
        needed = MIN_SAMPLES_PER_CHAIN_RELIABLE * 2 * n_chains_detected  # *2 for burn-in
        print(f"   Need approximately {needed:,} total samples for reliable diagnostics")


if calib_df is not None:
    print_convergence_diagnostics(calib_df, param_names, n_chains=5)

# %% [markdown]
# ---
# ## 7. Refresh Data
#
# Reload the CSV file and update all plots.
#
# ### Functions Available:
# - `refresh_and_plot()` - Full refresh with all plots
# - `quick_status()` - Quick status check without plots
# - `auto_monitor(interval=30)` - Continuous monitoring with auto-refresh
# - `monitor_pydream('objective')` - Switch to different PyDREAM calibration

# %%
def quick_status() -> dict:
    """
    Quick status check of the current calibration without generating plots.
    
    Returns:
        dict: Summary statistics
    """
    print(f"\n{'='*60}")
    print(f"QUICK STATUS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    if not CSV_FILE.exists():
        print(f"⚠ File not found: {CSV_FILE}")
        return {}
    
    # File info
    file_size = CSV_FILE.stat().st_size / 1024
    file_mtime = datetime.fromtimestamp(CSV_FILE.stat().st_mtime)
    age_seconds = (datetime.now() - file_mtime).total_seconds()
    
    # Load data
    try:
        df = load_calibration_data(CSV_FILE, verbose=False)
        param_names_local, obj_col, calib_df_local = extract_parameters(df)
        stats_local = get_calibration_stats(calib_df_local, param_names_local)
    except Exception as e:
        print(f"⚠ Error loading data: {e}")
        return {}
    
    # Status display
    if CALIBRATION_SOURCE == 'pydream_synthetic':
        print(f"Source: SYNTHETIC HYDROGRAPH TEST")
    elif CALIBRATION_SOURCE == 'pydream_realdata':
        print(f"Objective: {PYDREAM_OBJECTIVE.upper()}")
    print(f"File: {CSV_FILE.name}")
    print(f"Size: {file_size:.1f} KB")
    print(f"Last modified: {file_mtime.strftime('%Y-%m-%d %H:%M:%S')} ({age_seconds:.0f}s ago)")
    
    if age_seconds < 120:
        print(f"Status: 🔄 ACTIVELY RUNNING")
    elif age_seconds < 3600:
        print(f"Status: ⏸ POSSIBLY PAUSED")
    else:
        print(f"Status: ✓ LIKELY COMPLETE")
    
    print(f"\n{'-'*60}")
    print(f"Samples: {stats_local['n_samples']:,}")
    print(f"Best likelihood: {stats_local['best_likelihood']:.6f}")
    print(f"Mean likelihood: {stats_local['mean_likelihood']:.6f}")
    print(f"Parameters: {len(param_names_local)}")
    
    # Check for valid samples (not -inf)
    valid_samples = calib_df_local[calib_df_local['likelihood'] > -np.inf]
    print(f"Valid samples (not -inf): {len(valid_samples):,} ({100*len(valid_samples)/len(calib_df_local):.1f}%)")
    
    # Show best parameters
    print(f"\n{'-'*60}")
    print("Best Parameters:")
    for param, value in list(stats_local['best_params'].items())[:10]:
        print(f"  {param:15s}: {value:.6f}")
    if len(stats_local['best_params']) > 10:
        print(f"  ... and {len(stats_local['best_params']) - 10} more")
    
    return stats_local


def refresh_and_plot(show_all: bool = True):
    """
    Reload data and regenerate plots.
    
    Args:
        show_all: If True, show all diagnostic plots. If False, show only key plots.
    """
    global raw_df, calib_df, param_names, stats
    
    print(f"\n{'='*70}")
    print(f"REFRESHING DATA - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    
    if CALIBRATION_SOURCE == 'pydream_synthetic':
        print(f"Source: PyDREAM SYNTHETIC HYDROGRAPH TEST (Notebook 06)")
    elif CALIBRATION_SOURCE == 'pydream_realdata':
        print(f"Source: PyDREAM REAL DATA | Objective: {PYDREAM_OBJECTIVE.upper()}")
    
    print(f"File: {CSV_FILE}\n")
    
    try:
        raw_df = load_calibration_data(CSV_FILE)
        param_names_new, obj_col, calib_df = extract_parameters(raw_df)
        param_names = param_names_new
        stats = get_calibration_stats(calib_df, param_names)
        
        # Check file activity
        file_mtime = datetime.fromtimestamp(CSV_FILE.stat().st_mtime)
        age_seconds = (datetime.now() - file_mtime).total_seconds()
        if age_seconds < 120:
            status_msg = "🔄 ACTIVELY RUNNING"
        elif age_seconds < 3600:
            status_msg = "⏸ POSSIBLY PAUSED"
        else:
            status_msg = "✓ LIKELY COMPLETE"
        
        print(f"Status: {status_msg}")
        print(f"Progress: {stats['n_samples']:,} samples | Best: {stats['best_likelihood']:.6f}")
        
        # Count valid samples
        valid_samples = calib_df[calib_df['likelihood'] > -np.inf]
        print(f"Valid samples: {len(valid_samples):,} ({100*len(valid_samples)/len(calib_df):.1f}%)")
        
        print("\n" + "="*70)
        print("1. OBJECTIVE PROGRESS")
        print("="*70)
        plot_objective_progress(calib_df)
        
        if show_all:
            print("\n" + "="*70)
            print("2. DOTTY PLOTS (Parameter Sensitivity)")
            print("="*70)
            plot_dotty_plots(calib_df)
        
        print("\n" + "="*70)
        print("3. PARAMETER TRACES")
        print("="*70)
        # Show all parameters in trace plots
        plot_parameter_traces(calib_df, params=param_names)
        
        print("\n" + "="*70)
        print("4. POSTERIOR DISTRIBUTIONS")
        print("="*70)
        plot_posterior_distributions(calib_df, burnin=DEFAULT_BURNIN)
        
        print("\n" + "="*70)
        print("5. CONVERGENCE DIAGNOSTICS")
        print("="*70)
        print_convergence_diagnostics(calib_df, param_names, n_chains=5)
        
        # Show best parameters
        print(f"\n{'='*70}")
        print("BEST PARAMETERS FOUND")
        print(f"{'='*70}")
        for param, value in stats['best_params'].items():
            print(f"  {param:15s}: {value:.6f}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


def auto_monitor(interval: int = None, max_iterations: int = 100):
    """
    Continuously monitor the calibration with auto-refresh.
    
    Args:
        interval: Refresh interval in seconds. Defaults to AUTO_REFRESH_INTERVAL.
        max_iterations: Maximum number of refresh cycles.
    
    Note:
        Press Ctrl+C (interrupt kernel) to stop monitoring.
    """
    if interval is None:
        interval = AUTO_REFRESH_INTERVAL
    
    print(f"\n{'='*70}")
    print("AUTO-MONITOR MODE")
    print(f"{'='*70}")
    print(f"Refresh interval: {interval} seconds")
    print(f"Press Ctrl+C (interrupt kernel) to stop")
    print(f"{'='*70}\n")
    
    try:
        for i in range(max_iterations):
            if WIDGETS_AVAILABLE:
                clear_output(wait=True)
            
            print(f"Iteration {i+1}/{max_iterations}")
            quick_status()
            
            # Only show key plot (objective progress)
            try:
                raw_df_local = load_calibration_data(CSV_FILE, verbose=False)
                _, _, calib_df_local = extract_parameters(raw_df_local)
                plot_objective_progress(calib_df_local)
            except Exception as e:
                print(f"Plot error: {e}")
            
            print(f"\nNext refresh in {interval} seconds...")
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        print("Run refresh_and_plot() for full analysis.")


# Quick status check on load
print("\n" + "-"*70)
print("Quick functions available:")
print("  quick_status()          - Quick status check")
print("  refresh_and_plot()      - Full refresh with all plots")
print("  auto_monitor(interval)  - Continuous monitoring")
print("  monitor_pydream('obj')  - Switch to different calibration")
print("  list_available_calibrations()  - Show all available PyDREAM files")
print("-"*70)

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
# This notebook provides real-time monitoring of MCMC calibrations, including
# PyDREAM calibrations from Notebook 06 (Algorithm Comparison).
#
# ### Key Functions
#
# **Monitoring:**
# - `quick_status()` - Quick status check without plots
# - `refresh_and_plot()` - Full refresh with all diagnostic plots
# - `auto_monitor(interval)` - Continuous monitoring with auto-refresh
#
# **Switching Calibrations:**
# - `monitor_pydream('objective')` - Switch to different PyDREAM calibration
# - `list_available_calibrations()` - Show all available PyDREAM files
#
# **Visualization:**
# - `plot_objective_progress(df)` - Objective function evolution
# - `plot_dotty_plots(df)` - Parameter sensitivity
# - `plot_parameter_traces(df)` - MCMC chain evolution
# - `plot_posterior_distributions(df, burnin)` - Parameter distributions
# - `print_convergence_diagnostics(df)` - Gelman-Rubin R-hat
#
# **Export:**
# - `export_best_parameters()` - Export optimal parameter values
#
# ### Tips
#
# - Start with `burnin=0.0` early in calibration, increase to `0.3-0.5` later
# - R-hat < 1.1 indicates convergence, < 1.05 is ideal
# - Wide posteriors suggest poorly identified parameters
# - Multiple peaks in posteriors indicate equifinality
# - Use `auto_monitor(30)` for hands-free progress tracking

# %%
print("=" * 70)
print("CALIBRATION MONITOR READY")
print("=" * 70)
print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if CALIBRATION_SOURCE == 'pydream_synthetic':
    print(f"\nCurrently monitoring: PyDREAM SYNTHETIC HYDROGRAPH TEST")
    print(f"File: {CSV_FILE.name}")
elif CALIBRATION_SOURCE == 'pydream_realdata':
    print(f"\nCurrently monitoring: PyDREAM {PYDREAM_OBJECTIVE.upper()} calibration (real data)")
    print(f"File: {CSV_FILE.name}")

print("""
Quick Start Commands:
  quick_status()               - Check progress without plots
  refresh_and_plot()           - Full analysis with all plots
  auto_monitor(30)             - Auto-refresh every 30 seconds
  
Switch Calibrations (PyDREAM from Notebook 06):
  list_available_calibrations()    - Show all real-data calibration files
  monitor_pydream('kge')           - Switch to KGE real-data calibration
  monitor_pydream('nse')           - Switch to NSE calibration
  monitor_pydream('sdeb')          - Switch to SDEB calibration
""")
