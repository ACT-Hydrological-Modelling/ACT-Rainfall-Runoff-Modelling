# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
#   kernelspec:
#     display_name: Python (pyrrm)
#     language: python
#     name: pyrrm
# ---

# %% [markdown]
# # Executive Summary: pyrrm Calibration Experiments
#
# **A question-and-answer guide to everything we learned from 65+ calibration
# experiments on the Queanbeyan River catchment.**
#
# ---
#
# **Who is this for?**
# A fellow hydrological modeller who is comfortable with eWater Source and
# conceptual rainfall-runoff models, but wants a plain-English walkthrough
# of what we tested, what we found, and what it means for practice.
#
# **How to read it:**
# Each section poses a question you might naturally ask, then answers it
# with narrative, figures, and tables drawn from saved results.  No
# calibrations are re-run here -- everything loads from pre-computed reports.

# %% [markdown]
# ---
# # Part 1 — Setting the Scene

# %% [markdown]
# ## Q1: What are we trying to do here?
#
# At its simplest, **rainfall-runoff modelling** is about building a
# mathematical recipe that turns rainfall and evaporation data into
# streamflow.  The recipe has adjustable settings -- called **parameters** --
# and finding the best settings so the recipe's output matches what we
# actually measured at the gauge is called **calibration**.
#
# ### The Explorer Analogy
#
# Throughout this document we will use a single unifying picture:
#
# > **Calibration is an explorer navigating a landscape.**
#
# | Component | What it represents | Plain-language meaning |
# |---|---|---|
# | **The Explorer** | The calibration process | The whole exercise of searching for the best parameters |
# | **The Vehicle** | The model (GR4J, Sacramento …) | The mathematical recipe -- some vehicles are simple sedans, others are heavy trucks |
# | **The Eyes** | Objective function **+** flow transformation | What the explorer looks for — the *metric* (NSE, KGE …) decides *how* to score, while the *lens* (√Q, log Q, 1/Q) decides *which flows to magnify*. Together they determine what the explorer sees. |
# | **The Brain** | The optimisation algorithm (SCE-UA, PyDREAM) | How the explorer searches -- methodical hill-climber vs. full cartographer |
# | **The Terrain** | The parameter space | The landscape of all possible parameter combinations |
# | **The Scorecard** | Diagnostic metrics | Independent judges who score the trip after it is over |
#
# ```mermaid
# flowchart LR
#   subgraph explorer ["The Calibration Explorer"]
#     brain["Brain<br>(Algorithm)"]
#     subgraph eyes ["Eyes: metric + lens"]
#       metric["Metric<br>(NSE, KGE …)"]
#       lens["Lens<br>(√Q, log Q, 1/Q)"]
#     end
#     vehicle["Vehicle<br>(Model)"]
#   end
#   terrain["Terrain<br>(Parameter Space)"] --> explorer
#   explorer --> scorecard["Scorecard<br>(Diagnostic Metrics)"]
# ```
#
# The rest of this document explores each component in turn, then puts them
# all together.

# %% [markdown]
# ## Q2: Why not just use Source?
#
# eWater Source is the right tool for operational modelling in Australia.
# But its built-in calibration options are limited: typically a single
# objective function and one search algorithm.
#
# **pyrrm** is a research workbench that lets us systematically test
# *many* combinations of models, objectives, and algorithms in a
# reproducible way.  The insights flow back into Source work:
#
# - Which objective function should I choose in Source for my application?
# - Does it matter which model I pick for a given catchment?
# - How much should I trust a single "best" parameter set?
#
# This is **complementary, not a replacement** for Source.

# %% [markdown]
# ## Q3: What is the Queanbeyan River catchment?
#
# All experiments in this tranche use **gauge 410734 — Queanbeyan River at
# Queanbeyan**, a ~490 km² catchment in the ACT region.
#
# | Property | Value |
# |---|---|
# | Gauge ID | 410734 |
# | Area | ~490 km² |
# | Rainfall data | SILO gridded (from 1890) |
# | PET data | Morton's wet-environment areal |
# | Observed flow | From 1985 |
#
# It is a good test case because it has long, quality-controlled records
# and is representative of ACT conditions.  Let's have a quick look at the
# data.

# %%
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

REPORT_DIR = Path("../test_data/reports")
FIGURE_DIR = Path("../figures")
FIGURE_DIR.mkdir(exist_ok=True)

from pyrrm.calibration import CalibrationReport

rep = CalibrationReport.load(REPORT_DIR / "410734_gr4j_nse.pkl")

rainfall = rep.inputs["rainfall"].values[-len(rep.dates):] if rep.inputs is not None else None

fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True,
                          gridspec_kw={"height_ratios": [1, 2]})

if rainfall is not None:
    axes[0].bar(rep.dates, rainfall, color="steelblue", width=1, alpha=0.7)
axes[0].invert_yaxis()
axes[0].set_ylabel("Rainfall (mm/d)")
axes[0].set_title("Queanbeyan River at Queanbeyan (410734) — Calibration Period")

axes[1].plot(rep.dates, rep.observed, color="black", lw=0.8, label="Observed")
axes[1].set_ylabel("Flow (ML/d)")
axes[1].set_xlabel("Date")
axes[1].legend()

plt.tight_layout()
fig.savefig(FIGURE_DIR / "catchment_overview.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Q3b: How do all 9 notebooks connect? — The Master Journey
#
# We ran experiments across 9 notebooks, each building on the last.
# Think of them as waypoints on an expedition:
#
# ```mermaid
# flowchart TD
#   subgraph basecamp ["Part 1 — Base Camp"]
#     NB01["01 Sacramento Verification<br>(Trust the vehicle)"]
#     NB02["02 Calibration Quickstart<br>(First expedition + flow transformations)"]
#     NB03["03 Routing Quickstart<br>(River travel time)"]
#   end
#
#   subgraph eyes ["Part 2 — Training the Eyes"]
#     NB04["04 Objective Functions<br>(Choosing what to see)"]
#     NB05["05 APEX Guide<br>(Building better eyes)"]
#   end
#
#   subgraph brain ["Part 3 — Testing the Brain"]
#     NB06["06 Algorithm Comparison<br>(SCE-UA vs PyDREAM)"]
#     NB08["08 Calibration Monitor<br>(Watching the search)"]
#   end
#
#   subgraph summit ["Part 4 — The Summit"]
#     NB07["07 Model Comparison<br>(Which vehicle wins?)"]
#     NB09["09 Calibration Reports<br>(Trip report cards)"]
#   end
#
#   NB01 --> NB02
#   NB02 --> NB03
#   NB02 --> NB04
#   NB04 --> NB05
#   NB04 --> NB06
#   NB06 --> NB08
#   NB02 --> NB07
#   NB04 --> NB07
#   NB06 --> NB07
#   NB07 --> NB09
# ```
#
# **Base Camp** — We verified our Sacramento implementation against the
# C# reference (Notebook 01), ran our first calibration **and explored
# flow transformations** — discovering that *what the model sees* depends
# on both the objective function and the mathematical lens applied to the
# flow data (02) — and learned about channel routing (03).
#
# **Training the Eyes** — We explored 13 objective functions and saw how
# each one changes what the model learns (04), then built the APEX
# composite objective to get "multi-focal vision" (05).
#
# **Testing the Brain** — We compared two fundamentally different search
# strategies: SCE-UA optimisation vs. PyDREAM Bayesian inference (06), and
# learned how to monitor long-running calibrations (08).
#
# **The Summit** — We brought everything together: 4 models × 13 objectives
# = 52 calibrations, judged by 25 diagnostic metrics (07), with full
# report-card tooling (09).
#
# ![Master Journey Diagram](../figures/00_master_journey.png)

# %% [markdown]
# ---
# # Part 2 — The Models (The Vehicle Fleet)

# %% [markdown]
# ## Q4: What models did we test, and how do they differ?
#
# We tested four conceptual rainfall-runoff models.  Think of them as a
# **vehicle fleet** -- each vehicle has a different number of gears
# (parameters) and a different design philosophy:
#
# | Model | Parameters | Origin | Analogy |
# |---|---|---|---|
# | **GR4J** | 4 | INRAE, France | Compact sedan — nimble, easy to handle |
# | **GR5J** | 5 | INRAE, France | SUV — adds a groundwater exchange threshold |
# | **GR6J** | 6 | INRAE, France | Pickup — adds an exponential store for low flows |
# | **Sacramento** | 22 | US NWS | Articulated truck — maximum capability, hard to park |
#
# ```mermaid
# flowchart LR
#   GR4J["GR4J<br>4 params<br>Production + Routing"] --> GR5J["GR5J<br>5 params<br>+ Exchange threshold"]
#   GR5J --> GR6J["GR6J<br>6 params<br>+ Exponential store"]
#   Sacramento["Sacramento<br>22 params<br>Multi-zone soil moisture"]
# ```
#
# The **GR family** shares the same core structure: a production store that
# decides how much rain becomes runoff, and a routing store that smooths
# the output.  Each successive model adds one component:
#
# - **GR5J** adds a threshold on groundwater exchange, giving more control
#   over when water enters or leaves the system.
# - **GR6J** adds an exponential store, specifically designed to improve
#   the simulation of very low flows during dry spells.
#
# **Sacramento** takes a completely different approach: it divides the soil
# into upper and lower zones, each with tension and free water stores,
# governed by 22 interacting parameters.  More detail, but also a much
# harder landscape for the explorer to navigate.
#
# ![Vehicle Fleet](../figures/02_vehicle_fleet.png)

# %% [markdown]
# ## Q5: Does a more complex model always perform better?
#
# **Short answer: No.**
#
# This was one of the most striking findings.  Let's look at the numbers.

# %%
comp_df = pd.read_csv("../test_data/model_comparison_all_13_objectives.csv")

model_order = ["GR4J", "GR5J", "GR6J", "Sacramento"]
comp_df["model"] = pd.Categorical(comp_df["model"], categories=model_order, ordered=True)

summary = comp_df.groupby("model", observed=True).agg(
    n_params=("n_params", "first"),
    mean_runtime=("runtime_s", "mean"),
).reset_index()

print("Model Summary (across 13 objective functions)")
print("=" * 55)
for _, row in summary.iterrows():
    print(f"  {row['model']:<12} {int(row['n_params']):>2} params   "
          f"mean runtime: {row['mean_runtime']:>6.0f}s")

# %%
metrics_df = pd.read_csv("../test_data/model_comparison_comprehensive_metrics.csv")

higher_better = ["NSE", "KGE", "KGE_np", "LogNSE", "InvNSE", "SqrtNSE",
                  "KGE_inv", "KGE_sqrt", "KGE_log", "Pearson_r", "Spearman_r"]
lower_better = ["RMSE", "MAE", "SDEB"]
lower_abs_better = ["PBIAS", "FDC_Peak", "FDC_High", "FDC_Mid", "FDC_Low",
                     "FDC_VeryLow", "Q95_error", "Q50_error", "Q5_error",
                     "Flashiness_error", "BFI_error"]

models = ["GR4J", "GR5J", "GR6J", "Sacramento"]
objectives = metrics_df["objective"].unique()
win_counts = {m: 0 for m in models}

for obj in objectives:
    sub = metrics_df[metrics_df["objective"] == obj]
    for metric in higher_better:
        if metric in sub.columns:
            best = sub.loc[sub[metric].idxmax(), "model"]
            win_counts[best] += 1
    for metric in lower_better:
        if metric in sub.columns:
            best = sub.loc[sub[metric].idxmin(), "model"]
            win_counts[best] += 1
    for metric in lower_abs_better:
        if metric in sub.columns:
            best = sub.loc[sub[metric].abs().idxmin(), "model"]
            win_counts[best] += 1

total = sum(win_counts.values())
print("\nWin Counts — Which model scores best most often?")
print("=" * 55)
for m in models:
    bar = "█" * int(win_counts[m] / total * 50)
    print(f"  {m:<12} {win_counts[m]:>4} wins ({win_counts[m]/total*100:5.1f}%)  {bar}")

# %%
fig, ax = plt.subplots(figsize=(8, 4))
colors = {"GR4J": "#2196F3", "GR5J": "#4CAF50", "GR6J": "#FF9800", "Sacramento": "#9C27B0"}
bars = ax.barh(models, [win_counts[m] for m in models],
               color=[colors[m] for m in models], edgecolor="white", linewidth=0.5)
ax.set_xlabel("Number of Metric-Objective Wins")
ax.set_title("Which model wins most often across all diagnostics and objectives?")
for bar, m in zip(bars, models):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
            f"{win_counts[m]} ({win_counts[m]/total*100:.0f}%)",
            va="center", fontsize=10)
ax.set_xlim(0, max(win_counts.values()) * 1.2)
plt.tight_layout()
fig.savefig(FIGURE_DIR / "model_win_counts.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Key Finding
#
# **No single model dominates all diagnostics.**  The margins between models
# are often small, and *which* model "wins" depends heavily on *which*
# diagnostic you look at and *which* objective function was used to
# calibrate.
#
# In fact, one of the strongest conclusions from the entire experiment
# tranche is:
#
# > **Objective function choice often matters more than model choice.**
#
# A simpler model (GR4J, 4 parameters) with a well-chosen objective can
# outperform a complex model (Sacramento, 22 parameters) with a poorly
# chosen one.  Simpler models also have a practical advantage for
# ungauged catchments: fewer parameters means less data needed to
# constrain the calibration.

# %% [markdown]
# ---
# # Part 3 — Objective Functions (The Explorer's Eyes)

# %% [markdown]
# ## Q6: What is an objective function, and why does it matter?
#
# An objective function is **the scorecard you hand the explorer before the
# trip**.  It defines what "good" means.
#
# Think of it this way: if you ask a student to study for an exam that
# tests only maths, they will focus on maths and might neglect reading.
# Similarly, if you calibrate a model using an objective that only cares
# about flood peaks, the model will learn to get peaks right — and may
# completely ignore baseflow.
#
# ### The "Eyes" Analogy
#
# The explorer's "eyes" are really **two things combined**:
#
# 1. **The metric** (NSE, KGE, SDEB …) — *how* the comparison is scored.
# 2. **The flow transformation / lens** (raw Q, √Q, log Q, 1/Q) — *which
#    flows are magnified* before scoring.
#
# We first encountered flow transformations in Notebook 02 (Calibration
# Quickstart), where we saw how applying √Q, log Q, or 1/Q compresses
# high flows and expands low flows — fundamentally changing the
# information the model receives during calibration.
#
# | Objective type | Lens | What the "eyes" see well | What they miss |
# |---|---|---|---|
# | **NSE** | raw Q | Sharp distance vision — flood peaks | Near-sighted for low flows |
# | **Log-NSE** | log Q | Reading glasses — recession, baseflow | Dazzled by bright floods |
# | **Inv-NSE** | 1/Q | Microscope — very low flows | Cannot see peaks at all |
# | **Sqrt-NSE** | √Q | Bifocals — balanced view | Jack of all trades, master of none |
# | **KGE** | raw Q | Trifocals — correlation + spread + bias | Still peak-weighted in raw form |
# | **SDEB** | mixed | Panoramic — daily errors + FDC + bias | Complex to interpret |
#
# **The eyes you choose — metric *and* lens — determine what the explorer
# finds.**

# %% [markdown]
# ## Q7: What objective functions did we test?
#
# We tested **13 objective functions** organised into four families.  Each
# family applies the same base metric with four different **flow
# transformations** that shift emphasis from peaks to low flows:
#
# ```mermaid
# flowchart TD
#   root["13 Objective Functions"] --> NSEfam["NSE Family"]
#   root --> KGEfam["KGE Family"]
#   root --> KGEnpfam["KGE-np Family"]
#   root --> mSDEB["SDEB<br>(Composite)"]
#
#   NSEfam --> NSE["NSE<br>(peaks)"]
#   NSEfam --> SqrtNSE["Sqrt-NSE<br>(balanced)"]
#   NSEfam --> LogNSE["Log-NSE<br>(low flows)"]
#   NSEfam --> InvNSE["Inv-NSE<br>(very low)"]
#
#   KGEfam --> KGE["KGE"]
#   KGEfam --> KGE_sqrt["KGE-sqrt"]
#   KGEfam --> KGE_log["KGE-log"]
#   KGEfam --> KGE_inv["KGE-inv"]
#
#   KGEnpfam --> KGEnp["KGE-np"]
#   KGEnpfam --> KGEnp_sqrt["KGE-np-sqrt"]
#   KGEnpfam --> KGEnp_log["KGE-np-log"]
#   KGEnpfam --> KGEnp_inv["KGE-np-inv"]
# ```
#
# | Family | Base Metric | Key Idea |
# |---|---|---|
# | **NSE** | Nash-Sutcliffe Efficiency | Squared-error skill score; 1 = perfect |
# | **KGE** | Kling-Gupta Efficiency | Decomposes into correlation, variability ratio, and bias ratio |
# | **KGE-np** | Non-parametric KGE | Uses Spearman rank correlation; more robust to outliers |
# | **SDEB** | Sum of Daily Errors + Bias | Composite: daily time-series error + flow-duration-curve shape + volume bias |
#
# The **flow transformations** act like different lenses:
#
# | Transform | Formula | What it emphasises |
# |---|---|---|
# | None (raw Q) | Q | High flows / flood peaks |
# | Square root (√Q) | √Q | Balanced — mid-range flows |
# | Logarithm (log Q) | ln(Q+ε) | Low flows / baseflow |
# | Inverse (1/Q) | 1/(Q+ε) | Very low flows / drought |

# %% [markdown]
# ## Q8: What happens when you change the objective function?
#
# The effect is dramatic.  Below we load GR4J calibrated with three very
# different objectives and overlay a zoom window so you can see the
# difference in peak and baseflow reproduction.

# %%
obj_labels = {
    "nse": "NSE (peak-focused)",
    "sqrtnse": "Sqrt-NSE (balanced)",
    "invnse": "Inv-NSE (low-flow focused)",
}

fig, axes = plt.subplots(len(obj_labels), 1, figsize=(14, 3 * len(obj_labels)),
                          sharex=True)

for ax, (obj_key, label) in zip(axes, obj_labels.items()):
    r = CalibrationReport.load(REPORT_DIR / f"410734_gr4j_{obj_key}.pkl")
    ax.plot(r.dates, r.observed, "k-", lw=0.7, alpha=0.6, label="Observed")
    ax.plot(r.dates, r.simulated, lw=0.9, label=f"GR4J — {label}")
    ax.set_ylabel("Flow (ML/d)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(bottom=0)

axes[0].set_title("Same model, different eyes — how objective function changes the fit")
axes[-1].set_xlabel("Date")
plt.tight_layout()
fig.savefig(FIGURE_DIR / "objective_comparison_hydrographs.png", dpi=150,
            bbox_inches="tight")
plt.show()

# %% [markdown]
# **What to notice:**
#
# - **NSE** (top): peaks are tracked closely, but baseflow is over- or
#   under-estimated in places.
# - **Sqrt-NSE** (middle): a compromise — peaks are slightly less sharp
#   but baseflow is much better.
# - **Inv-NSE** (bottom): baseflow and recessions are tightly matched,
#   but peak flows are systematically under-predicted.
#
# This is the fundamental trade-off: **no single pair of eyes can see
# everything equally well.**

# %% [markdown]
# ---
# ## Q8b: How do we judge performance after calibration? — Diagnostic Metrics
#
# After the explorer returns from the trip, we bring in a **panel of 25
# independent judges**.  Each judge focuses on a different part of the
# hydrograph.  We group them by which **flow regime** they are most
# sensitive to:
#
# ```mermaid
# flowchart LR
#   subgraph vhigh ["Very High Flows<br>(0-2% exceedance)"]
#     FDC_Peak["FDC Peak"]
#     mNSE["NSE"]
#     Pearson["Pearson r"]
#     Q5["Q5 error"]
#   end
#
#   subgraph high ["High Flows<br>(2-10%)"]
#     FDC_High["FDC High"]
#     mKGE["KGE"]
#     KGE_np["KGE-np"]
#     Flash["Flashiness"]
#   end
#
#   subgraph mid ["Mid Flows<br>(20-70%)"]
#     FDC_Mid["FDC Mid"]
#     SqrtNSE["Sqrt-NSE"]
#     KGE_sqrt["KGE-sqrt"]
#     Q50["Q50 error"]
#     mSDEB["SDEB"]
#     Spearman["Spearman rho"]
#   end
#
#   subgraph low ["Low Flows<br>(70-90%)"]
#     FDC_Low["FDC Low"]
#     LogNSE["Log-NSE"]
#     KGE_log["KGE-log"]
#     Q95["Q95 error"]
#     BFI["BFI error"]
#   end
#
#   subgraph vlow ["Very Low Flows<br>(90-100%)"]
#     FDC_VL["FDC VeryLow"]
#     InvNSE["Inv-NSE"]
#     KGE_inv["KGE-inv"]
#   end
# ```
#
# ### The Full Panel of Judges
#
# | Metric | Full Name | What It Measures | Flow Regime | Direction |
# |---|---|---|---|---|
# | **NSE** | Nash-Sutcliffe Efficiency | Overall skill, peak-dominated | Very High | Higher = better |
# | **Pearson r** | Pearson Correlation | Linear agreement (peak-sensitive) | Very High | Higher = better |
# | **Q5 error** | 5th percentile flow error | Accuracy of highest flows | Very High | Closer to 0 = better |
# | **FDC Peak** | FDC Peak segment bias | Volume bias in top 0-2% of flows | Very High | Closer to 0 = better |
# | **KGE** | Kling-Gupta Efficiency | Correlation + variability + bias | High | Higher = better |
# | **KGE-np** | Non-parametric KGE | Robust KGE (rank-based) | High | Higher = better |
# | **Flashiness** | Flashiness index error | How "spiky" the hydrograph is | High | Closer to 0 = better |
# | **FDC High** | FDC High segment bias | Volume bias in 2-10% band | High | Closer to 0 = better |
# | **Sqrt-NSE** | NSE on √Q | Balanced skill score | Mid | Higher = better |
# | **KGE-sqrt** | KGE on √Q | Balanced KGE | Mid | Higher = better |
# | **Spearman ρ** | Spearman Rank Correlation | Shape agreement (rank-based) | Mid | Higher = better |
# | **Q50 error** | Median flow error | Accuracy of typical flows | Mid | Closer to 0 = better |
# | **FDC Mid** | FDC Mid segment bias | Volume bias in 20-70% band | Mid | Closer to 0 = better |
# | **SDEB** | Sum Daily Error + Bias | Composite (daily + FDC + bias) | Mid | Lower = better |
# | **Log-NSE** | NSE on log(Q) | Low-flow skill | Low | Higher = better |
# | **KGE-log** | KGE on log(Q) | Low-flow KGE | Low | Higher = better |
# | **Q95 error** | 95th percentile flow error | Accuracy of low flows | Low | Closer to 0 = better |
# | **BFI error** | Baseflow index error | Baseflow proportion accuracy | Low | Closer to 0 = better |
# | **FDC Low** | FDC Low segment bias | Volume bias in 70-90% band | Low | Closer to 0 = better |
# | **Inv-NSE** | NSE on 1/Q | Very-low-flow skill | Very Low | Higher = better |
# | **KGE-inv** | KGE on 1/Q | Very-low-flow KGE | Very Low | Higher = better |
# | **FDC VeryLow** | FDC Very-Low segment bias | Volume bias in 90-100% band | Very Low | Closer to 0 = better |
# | **RMSE** | Root Mean Squared Error | Overall error magnitude | All | Lower = better |
# | **MAE** | Mean Absolute Error | Average error magnitude | All | Lower = better |
# | **PBIAS** | Percent Bias | Overall water balance error | All | Closer to 0 = better |
#
# **Rule of thumb:** if your application is about floods, focus on the
# blue-zone (Very High / High) metrics.  If it's about environmental
# flows or drought, focus on the amber-zone (Low / Very Low) metrics.
#
# ![Diagnostic Spectrum](../figures/03_diagnostic_spectrum.png)

# %%
regime_colors = {
    "Very High": "#1565C0",
    "High": "#42A5F5",
    "Mid": "#66BB6A",
    "Low": "#FFA726",
    "Very Low": "#EF5350",
    "All": "#78909C",
}

regime_metrics = {
    "Very High": ["NSE", "Pearson_r", "FDC_Peak", "Q5_error"],
    "High": ["KGE", "KGE_np", "FDC_High", "Flashiness_error"],
    "Mid": ["SqrtNSE", "KGE_sqrt", "Spearman_r", "Q50_error", "FDC_Mid", "SDEB"],
    "Low": ["LogNSE", "KGE_log", "Q95_error", "BFI_error", "FDC_Low"],
    "Very Low": ["InvNSE", "KGE_inv", "FDC_VeryLow"],
    "All": ["RMSE", "MAE", "PBIAS"],
}

fig, ax = plt.subplots(figsize=(12, 4))
x_pos = 0
for regime, mets in regime_metrics.items():
    for m in mets:
        ax.barh(x_pos, 1, color=regime_colors[regime], edgecolor="white", linewidth=0.5)
        ax.text(0.5, x_pos, m.replace("_", " "), ha="center", va="center",
                fontsize=8, fontweight="bold", color="white")
        x_pos += 1
    x_pos += 0.5

ax.set_xlim(0, 1)
ax.set_ylim(-0.5, x_pos)
ax.set_yticks([])
ax.set_xticks([])
ax.set_title("Diagnostic Metrics Grouped by Flow Regime Sensitivity", fontsize=12)

legend_patches = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in regime_colors.values()]
ax.legend(legend_patches, regime_colors.keys(), loc="upper right",
          fontsize=9, title="Flow Regime")
ax.invert_yaxis()
plt.tight_layout()
fig.savefig(FIGURE_DIR / "diagnostic_regime_spectrum.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
# # Part 4 — APEX: Building Better Eyes

# %% [markdown]
# ## Q9: What is APEX, and why did we develop it?
#
# Traditional objectives suffer from **tunnel vision**: each pair of "eyes"
# is good at one thing but blind to others.  NSE tracks peaks but ignores
# baseflow.  Log-NSE tracks baseflow but misses peaks.
#
# **APEX** (Adaptive Process-Explicit objective) is our attempt to build
# **multi-focal vision** — a single scorecard that simultaneously penalises:
#
# 1. **Chronological errors** — how well does the daily time series match?
# 2. **Ranked (FDC) errors** — does the flow-duration curve have the right shape?
# 3. **Volume bias** — is the overall water balance correct?
# 4. **Dynamics mismatch** — does the model get the rate of change right (timing)?
#
# APEX extends the SDEB framework (Lerat et al., 2013) by adding a
# **dynamics multiplier** that penalises gradient/timing mismatches.
#
# ```mermaid
# flowchart LR
#   subgraph apex ["APEX Score"]
#     chronological["Chronological Term<br>(daily time-series fit)"]
#     ranked["Ranked Term<br>(FDC shape fit)"]
#     bias["Bias Multiplier<br>(volume penalty)"]
#     dynamics["Dynamics Multiplier<br>(timing/gradient penalty)"]
#   end
#
#   alpha["alpha weight"] --> chronological
#   alpha --> ranked
#   chronological --> combined["Weighted Sum"]
#   ranked --> combined
#   combined --> penalised["Final APEX"]
#   bias --> penalised
#   dynamics --> penalised
# ```
#
# Different **flow transformations** (none, √Q, log Q, 1/Q) can be
# applied within APEX to shift the emphasis — just like swapping lenses
# on a pair of multi-focal glasses.

# %% [markdown]
# ## Q10: Does APEX actually improve things?
#
# We tested 8 APEX configurations against the standard objectives.  Let's
# compare the APEX-sqrt result (our recommended default) against NSE and
# SDEB using the GR4J model.

# %%
apex_configs = {
    "APEX sqrt": "410734_APEX_sqrt.pkl",
    "APEX none": "410734_APEX_none.pkl",
    "APEX log": "410734_APEX_log.pkl",
    "APEX inverse": "410734_APEX_inverse.pkl",
}

comparison_metrics = ["NSE", "KGE", "LogNSE", "RMSE", "PBIAS", "MAE"]

from pyrrm.calibration.objective_functions import calculate_metrics

print("APEX Configuration Comparison (Sacramento model)")
print("=" * 80)
header = f"{'Config':<20}" + "".join(f"{m:>10}" for m in comparison_metrics)
print(header)
print("-" * 80)

for label, fname in apex_configs.items():
    fpath = REPORT_DIR / fname
    if fpath.exists():
        r = CalibrationReport.load(fpath)
        mets = calculate_metrics(r.simulated, r.observed)
        row = f"{label:<20}"
        for m in comparison_metrics:
            val = mets.get(m, np.nan)
            row += f"{val:>10.4f}" if np.isfinite(val) else f"{'N/A':>10}"
        print(row)

print("\nStandard Objectives (Sacramento, for reference)")
print("-" * 80)
for obj_key, obj_label in [("nse", "NSE"), ("sqrtnse", "SqrtNSE"), ("sdeb", "SDEB")]:
    fpath = REPORT_DIR / f"410734_{obj_key}.pkl"
    if fpath.exists():
        r = CalibrationReport.load(fpath)
        mets = calculate_metrics(r.simulated, r.observed)
        row = f"{obj_label:<20}"
        for m in comparison_metrics:
            val = mets.get(m, np.nan)
            row += f"{val:>10.4f}" if np.isfinite(val) else f"{'N/A':>10}"
        print(row)

# %% [markdown]
# ### APEX Configuration Guide
#
# | Application | Transform | Dynamics (κ) | Regime | Rationale |
# |---|---|---|---|---|
# | **Flood forecasting** | none | 0.5 | uniform | Peak timing is critical |
# | **General operations** | sqrt | 0.5 | uniform | Balanced performance |
# | **Environmental flows** | log | 0.5 | low_flow | Low-flow accuracy critical |
# | **Drought planning** | inverse | 0.5 | low_flow | Extreme low-flow focus |
#
# **Key takeaway:** APEX adds value through its *structural innovations*
# (dynamics multiplier, ranked term) beyond what flow transformation alone
# can achieve.  The dynamics multiplier in particular helps with timing
# errors that traditional objectives miss entirely.

# %% [markdown]
# ---
# # Part 5 — Calibration Algorithms (The Explorer's Brain)

# %% [markdown]
# ## Q11: What is the difference between optimisation and Bayesian calibration?
#
# This is about **how the explorer searches** the landscape of possible
# parameters.
#
# ### SCE-UA — The Hill-Climber
#
# **Shuffled Complex Evolution** (SCE-UA) is an optimisation algorithm.
# Think of it as a focused, methodical hill-climber:
#
# - It sends out multiple search parties (complexes) across the landscape
# - The parties periodically shuffle members and share information
# - It converges on the single highest peak it can find
# - **Output:** one "best" parameter set — GPS coordinates of the summit
#
# ### PyDREAM — The Cartographer
#
# **MT-DREAM(ZS)** is a Bayesian MCMC algorithm.  Think of it as a
# cartographer who maps the entire mountain range:
#
# - It runs multiple chains that wander the landscape, spending more time
#   in "good" regions and less time in "bad" ones
# - Over time, the wandering pattern builds up a picture of *all*
#   plausible parameter combinations, not just the best one
# - **Output:** a full topographic map (posterior distribution) showing
#   ridges, valleys, and plateaus
#
# ```mermaid
# flowchart TD
#   start["Choose Algorithm"] --> q1{"Need uncertainty<br>estimates?"}
#   q1 -->|Yes| pydream["PyDREAM<br>(full posterior)"]
#   q1 -->|No| q2{"Computational<br>budget?"}
#   q2 -->|Limited| sceua["SCE-UA<br>(fast point estimate)"]
#   q2 -->|Generous| pydream
# ```

# %% [markdown]
# ## Q12: Does the algorithm choice matter for the final answer?
#
# We ran both algorithms across all 13 objective functions on the Sacramento
# model.  Let's compare.

# %%
pydream_dir = REPORT_DIR / "pydream"

algo_comparison = []

for obj_key in ["nse", "lognse", "invnse", "sqrtnse", "kge", "kge_inv",
                "kge_sqrt", "kge_log", "kge_np", "kge_np_inv",
                "kge_np_sqrt", "kge_np_log", "sdeb"]:
    sceua_path = REPORT_DIR / f"410734_{obj_key}.pkl"
    dream_path = pydream_dir / f"410734_pydream_{obj_key}.pkl"

    if sceua_path.exists() and dream_path.exists():
        r_sceua = CalibrationReport.load(sceua_path)
        r_dream = CalibrationReport.load(dream_path)

        m_sceua = calculate_metrics(r_sceua.simulated, r_sceua.observed)
        m_dream = calculate_metrics(r_dream.simulated, r_dream.observed)

        algo_comparison.append({
            "objective": obj_key,
            "SCE-UA NSE": m_sceua.get("NSE", np.nan),
            "PyDREAM NSE": m_dream.get("NSE", np.nan),
            "SCE-UA KGE": m_sceua.get("KGE", np.nan),
            "PyDREAM KGE": m_dream.get("KGE", np.nan),
        })

if algo_comparison:
    algo_df = pd.DataFrame(algo_comparison)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric in zip(axes, ["NSE", "KGE"]):
        sceua_col = f"SCE-UA {metric}"
        dream_col = f"PyDREAM {metric}"
        valid = algo_df[[sceua_col, dream_col]].dropna()
        ax.scatter(valid[sceua_col], valid[dream_col], s=60, c="#1976D2",
                   edgecolors="white", zorder=3)
        lims = [min(valid[sceua_col].min(), valid[dream_col].min()) - 0.05,
                max(valid[sceua_col].max(), valid[dream_col].max()) + 0.05]
        ax.plot(lims, lims, "k--", lw=0.8, alpha=0.5, label="1:1 line")
        ax.set_xlabel(f"SCE-UA {metric}")
        ax.set_ylabel(f"PyDREAM {metric}")
        ax.set_title(f"{metric}: SCE-UA vs PyDREAM")
        ax.legend(fontsize=9)
        ax.set_aspect("equal")

    plt.tight_layout()
    fig.savefig(FIGURE_DIR / "algorithm_comparison_scatter.png", dpi=150,
                bbox_inches="tight")
    plt.show()

    print("\nAlgorithm Comparison Summary (Sacramento)")
    print("=" * 75)
    print(f"{'Objective':<15} {'SCE-UA NSE':>12} {'PyDREAM NSE':>12} "
          f"{'SCE-UA KGE':>12} {'PyDREAM KGE':>12}")
    print("-" * 75)
    for _, row in algo_df.iterrows():
        print(f"{row['objective']:<15} {row['SCE-UA NSE']:>12.4f} "
              f"{row['PyDREAM NSE']:>12.4f} {row['SCE-UA KGE']:>12.4f} "
              f"{row['PyDREAM KGE']:>12.4f}")

# %% [markdown]
# ### Key Finding
#
# **Both algorithms produce very similar "best" fits when convergence is
# achieved.**  Points cluster close to the 1:1 line, meaning SCE-UA and
# PyDREAM arrive at comparable performance.
#
# The crucial difference is what you get *beyond* the best fit:
#
# | Feature | SCE-UA | PyDREAM |
# |---|---|---|
# | Best parameter set | Yes | Yes |
# | Uncertainty estimates | No | Yes (posterior distributions) |
# | Parameter identifiability | No | Yes (from posterior width) |
# | Multi-modality detection | No | Yes (from posterior shape) |
# | Typical runtime | ~2-5 min | ~30-120 min |
#
# **When to choose which:**
# - **SCE-UA:** Quick screening, operational calibration, limited compute
# - **PyDREAM:** Research, uncertainty analysis, decision support under
#   uncertainty

# %% [markdown]
# ---
# # Part 6 — Putting It All Together

# %% [markdown]
# ## Q13: What model and objective should I pick for my application?
#
# The answer depends on what you are modelling.  Here are data-driven
# recommendations from our experiments:
#
# ```mermaid
# flowchart TD
#   app["What is your<br>application?"] --> flood["Flood forecasting"]
#   app --> supply["Water supply"]
#   app --> envflow["Environmental flows"]
#   app --> drought["Drought planning"]
#
#   flood --> floodRec["NSE or KGE<br>+ any model"]
#   supply --> supplyRec["Sqrt-NSE or KGE-sqrt<br>+ GR5J or GR6J"]
#   envflow --> envRec["Log-NSE or KGE-log<br>+ GR6J"]
#   drought --> droughtRec["Inv-NSE or KGE-inv<br>+ model with low-flow store"]
# ```

# %%
RANKING_DIAG = {
    "High-flow focus": [("NSE", True), ("KGE", True), ("KGE_np", True),
                        ("Pearson_r", True), ("Spearman_r", True)],
    "Balanced (sqrt-Q)": [("SqrtNSE", True), ("KGE_sqrt", True), ("SDEB", False)],
    "Low-flow (log-Q)": [("LogNSE", True), ("KGE_log", True)],
    "Very-low-flow (1/Q)": [("InvNSE", True), ("KGE_inv", True)],
    "Volume accuracy": [("PBIAS", False), ("RMSE", False), ("MAE", False)],
}

print("Application-Based Model Recommendations")
print("=" * 70)

for grp_name, diag_list in RANKING_DIAG.items():
    model_scores = {}
    for model_name in models:
        vals = []
        model_data = metrics_df[metrics_df["model"] == model_name]
        for dkey, higher_better_flag in diag_list:
            if dkey not in model_data.columns:
                continue
            raw = model_data[dkey].values
            if not higher_better_flag:
                raw = -np.abs(raw)
            vals.extend(raw)
        model_scores[model_name] = np.nanmean(vals) if vals else np.nan

    sorted_models_grp = sorted(model_scores, key=lambda m: model_scores[m], reverse=True)
    best = sorted_models_grp[0]
    runner_up = sorted_models_grp[1]

    if grp_name == "High-flow focus":
        interp = "Best for flood forecasting and peak flow estimation"
    elif grp_name == "Balanced (sqrt-Q)":
        interp = "Best for general-purpose water supply modelling"
    elif grp_name == "Low-flow (log-Q)":
        interp = "Best for baseflow and dry-season flow estimation"
    elif grp_name == "Very-low-flow (1/Q)":
        interp = "Best for drought analysis and environmental flow"
    else:
        interp = "Best for overall volume accuracy"

    print(f"\n  {grp_name}")
    print(f"    Winner: {best} (score: {model_scores[best]:+.4f})")
    print(f"    Runner-up: {runner_up} (score: {model_scores[runner_up]:+.4f})")
    print(f"    → {interp}")

# %% [markdown]
# ## Q14: What are the main takeaways for our Source modelling work?
#
# Here are the five most important lessons from this entire tranche of
# experiments:
#
# 1. **Objective function choice often matters more than model choice.**
#    For the Queanbeyan catchment, switching from NSE to Sqrt-NSE on the
#    same model produced bigger performance changes than switching from
#    GR4J to Sacramento with the same objective.
#
# 2. **Always evaluate across multiple diagnostics, not just the one you
#    calibrated to.**  A model that scores NSE = 0.85 might have PBIAS =
#    30% (massively wrong water balance) or InvNSE = -2 (useless for low
#    flows).  The diagnostic panel catches what a single metric hides.
#
# 3. **Consider APEX or SDEB for operational calibration** where balanced
#    performance matters.  These composite objectives avoid the tunnel
#    vision of single-metric calibration.
#
# 4. **Uncertainty quantification matters for decision-making.**  PyDREAM
#    shows us not just the "best" answer but the range of plausible
#    answers.  When a decision depends on low-flow accuracy, knowing that
#    the parameter uncertainty translates to ±20% in Q95 is valuable.
#
# 5. **These findings need validation on more ACT catchments.**  The
#    Queanbeyan is one catchment.  Before generalising to ACT-wide
#    guidance, we should repeat on contrasting catchments.

# %% [markdown]
# ## Q15: How should I read all those diagnostic numbers?
#
# Refer back to the diagnostic panel in Q8b.  Here is a practical
# reading guide:
#
# **Step 1: Identify your flow regime of interest.**
# - Floods? → focus on Very High / High metrics (NSE, KGE, FDC Peak)
# - Water supply? → focus on Mid metrics (Sqrt-NSE, KGE-sqrt, SDEB)
# - Environmental flows? → focus on Low metrics (Log-NSE, KGE-log, Q95)
# - Drought? → focus on Very Low metrics (Inv-NSE, KGE-inv, FDC VeryLow)
#
# **Step 2: Check the regime-agnostic metrics.**
# - PBIAS tells you if the model is systematically too wet or too dry.
# - RMSE/MAE give you error in real units (ML/d).
#
# **Step 3: Look at FDC segment errors for a spatial view.**
# - These tell you *where* in the flow distribution the model succeeds
#   or fails.  A model might nail the mid-range but be 50% off on peaks.
#
# **Step 4: Check hydrologic signatures.**
# - Q95, Q50, BFI, Flashiness — these are regime-characterising numbers
#   that summarise the hydrograph's "personality."  Large errors here mean
#   the model is misrepresenting the catchment's fundamental behaviour.

# %%
print("Quick Reference: Diagnostic Metrics by Flow Regime")
print("=" * 75)
for regime, mets in regime_metrics.items():
    color_label = {"Very High": "BLUE", "High": "BLUE", "Mid": "GREEN",
                   "Low": "AMBER", "Very Low": "RED", "All": "GREY"}
    print(f"\n  [{color_label[regime]}] {regime} Flows")
    for m in mets:
        direction = "↑ higher = better" if m in higher_better else "↓ lower/|closer to 0| = better"
        print(f"    • {m:<20} {direction}")

# %% [markdown]
# ## Q16: What should we do next?
#
# The experiments so far give us a strong foundation.  The logical next
# steps are:
#
# 1. **Multi-catchment validation** — extend beyond the Queanbeyan to
#    contrasting ACT catchments (different sizes, climates, land use).
#    Do the same patterns hold?
#
# 2. **Split-sample testing** — calibrate on one period, validate on
#    another.  This tests whether the parameters generalise or are
#    over-fitted to the calibration period.
#
# 3. **Integrate channel routing** — Notebook 03 introduced Muskingum
#    routing; we should include routed models in the comparison.
#
# 4. **Ensemble predictions** — use PyDREAM posterior distributions to
#    generate flow ensembles that capture parameter uncertainty.
#
# 5. **Feed insights into Source** — translate the objective function and
#    model recommendations into Source configuration guidance for the team.

# %% [markdown]
# ---
# # Part 7 — Appendix

# %% [markdown]
# ## Glossary
#
# | Term | Plain-English Definition |
# |---|---|
# | **NSE** | Nash-Sutcliffe Efficiency — a score from -∞ to 1 measuring how much better the model is than just using the mean flow.  1 = perfect. |
# | **KGE** | Kling-Gupta Efficiency — like NSE but broken into three parts: correlation, variability, and bias.  1 = perfect. |
# | **KGE-np** | Non-parametric KGE — uses rank-based correlation instead of Pearson, making it more robust to outliers. |
# | **SDEB** | Sum of Daily Errors and Bias — a composite score that penalises time-series errors, FDC shape errors, and volume bias. |
# | **APEX** | Adaptive Process-Explicit objective — extends SDEB with a dynamics multiplier that penalises timing errors. |
# | **FDC** | Flow Duration Curve — a plot showing how often each flow level is exceeded.  The backbone of flow regime analysis. |
# | **MCMC** | Markov Chain Monte Carlo — a family of algorithms for sampling from probability distributions.  PyDREAM uses this. |
# | **SCE-UA** | Shuffled Complex Evolution — a global optimisation algorithm widely used in hydrology. |
# | **BFI** | Baseflow Index — the proportion of total flow that comes from baseflow (slow, sustained flow). |
# | **PBIAS** | Percent Bias — the percentage by which the model over- or under-predicts total volume.  0% = perfect balance. |
# | **RMSE** | Root Mean Squared Error — the average error in the same units as flow.  Lower = better. |
# | **MAE** | Mean Absolute Error — like RMSE but without squaring, so less sensitive to outliers. |
# | **Equifinality** | The problem where many different parameter sets produce equally good (or bad) model outputs. |
# | **Posterior distribution** | The probability distribution of parameters after calibration — tells you not just the best value but the range of plausible values. |
# | **Warmup period** | The initial period of a simulation discarded so the model's internal stores can fill to realistic levels. |

# %% [markdown]
# ## Full Results Table
#
# The complete metrics for all 4 models × 13 objectives (52 calibrations)
# are loaded below from the saved CSV.

# %%
full_df = pd.read_csv("../test_data/model_comparison_comprehensive_metrics.csv")

display_cols = ["model", "objective", "NSE", "KGE", "SqrtNSE", "LogNSE",
                "InvNSE", "RMSE", "PBIAS", "SDEB"]
available = [c for c in display_cols if c in full_df.columns]

print("Comprehensive Results Summary (selected metrics)")
print("=" * 100)
print(full_df[available].to_string(index=False, float_format="{:.4f}".format))

# %% [markdown]
# ## Notebook Cross-References
#
# | Notebook | Title | Key Output |
# |---|---|---|
# | 01 | Sacramento Verification | Confirmed Python ≡ C# (tolerance 1e-10) |
# | 02 | Calibration Quickstart | First Sacramento calibration + flow transformations + CalibrationReport workflow |
# | 03 | Routing Quickstart | Muskingum routing demonstration |
# | 04 | Objective Functions (WIP) | 13-objective catalogue + transformation guide |
# | 05 | APEX Complete Guide | APEX development + 8-configuration comparison |
# | 06 | Algorithm Comparison | PyDREAM vs SCE-UA across 13 objectives |
# | 07 | Model Comparison | 4 models × 13 objectives = 52 calibrations |
# | 08 | Calibration Monitor | Real-time MCMC convergence monitoring |
# | 09 | Calibration Reports | Report card generation + multi-calibration comparison |

# %% [markdown]
# ## References
#
# - **GR4J:** Perrin, C., Michel, C., & Andréassian, V. (2003).
#   Improvement of a parsimonious model for streamflow simulation.
#   *Journal of Hydrology*, 279(1-4), 275-289.
#
# - **GR5J:** Le Moine, N. (2008). *Le bassin versant vu par le
#   souterrain: une voie d'amélioration des performances et du réalisme
#   des modèles pluie-débit?* PhD thesis, Université Pierre et Marie Curie.
#
# - **GR6J:** Pushpalatha, R., Perrin, C., Le Moine, N., & Andréassian,
#   V. (2011). A downward structural sensitivity analysis of hydrological
#   models to improve low-flow simulation. *Journal of Hydrology*, 411(1-2),
#   66-76.
#
# - **KGE:** Gupta, H. V., Kling, H., Yilmaz, K. K., & Martinez, G. F.
#   (2009). Decomposition of the mean squared error and NSE performance
#   criteria: Implications for improving hydrological modelling. *Journal
#   of Hydrology*, 377(1-2), 80-91.
#
# - **SDEB:** Lerat, J., Thyer, M., McInerney, D., Kavetski, D., & Kuczera,
#   G. (2013). A robust approach for calibrating a daily rainfall-runoff
#   model to monthly streamflow data. *Journal of Hydrology*, 492, 163-174.
#
# - **SCE-UA:** Duan, Q., Sorooshian, S., & Gupta, V. K. (1994). Optimal
#   use of the SCE-UA global optimization method for calibrating watershed
#   models. *Journal of Hydrology*, 158(3-4), 265-284.
#
# - **DREAM:** Vrugt, J. A. (2016). Markov chain Monte Carlo simulation
#   using the DREAM software package: Theory, concepts, and MATLAB
#   implementation. *Environmental Modelling & Software*, 75, 273-316.
