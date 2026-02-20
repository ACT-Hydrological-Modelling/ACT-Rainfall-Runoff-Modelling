# Executive Summary: Master Slide Deck — Specification and Outline

This document provides **specifications** and a **slide-by-slide outline** with background content so the deck can be built in PowerPoint, Google Slides, Keynote, or another tool. It does not generate slides directly.

---

## 1. Deck Specifications

### 1.1 Purpose and Use

- **Primary use:** Present the pyrrm calibration experiment tranche to a fellow hydrological modeller (eWater Source, GUI-oriented, limited scripting).
- **Secondary use:** Standalone leave-behind or handout companion to the notebook `notebooks/10_executive_summary.py`.
- **Duration:** Plan for **20–35 minutes** speaking time (flexible by collapsing or expanding sections).
- **Takeaway:** “What we did, what we found, and what it means for how we choose models and objectives.”

### 1.2 Audience

- **Who:** Hydrological modeller, comfortable with conceptual rainfall-runoff models and eWater Source; not deeply familiar with calibration algorithms or objective-function theory.
- **Tone:** Collegial, first-principles, plain English. Avoid algorithm jargon; use the **Explorer analogy** (brain = algorithm, eyes = objective, vehicle = model) as the through-line.
- **No prerequisite:** No need to have seen the notebooks or code.

### 1.3 Design and Format

| Aspect | Specification |
|--------|----------------|
| **Aspect ratio** | 16:9 (widescreen). |
| **Template** | Single, consistent template; clear title area; space for one main visual or bullet block per slide. |
| **Typography** | Sans-serif for titles and body; minimum ~18 pt body so it’s readable when projected. |
| **Colour** | Use the same regime coding as the notebook/Excalidraw: e.g. blue = high flow, green = mid, amber = low, red = very low; grey/neutral for regime-agnostic. Optional: accent colour for “key message” or “takeaway” slides. |
| **Visuals** | Prefer one major visual per slide (diagram, chart, or short bullet list). Avoid dense tables; use summary tables or key numbers only. |
| **Slide density** | 1 main idea per slide; 3–7 bullets max where bullets are used. |
| **Branding** | Optional: logo, project name, “ACT Rainfall-Runoff Modelling” or “pyrrm” in footer or title master. |
| **Builds / animation** | Optional: simple builds (e.g. reveal bullets or diagram regions in order). Not required. |
| **Backup / appendix** | Last section can be “Appendix” with glossary, references, and notebook links for Q&A or follow-up. |

### 1.4 Asset Locations (for embedding in slides)

All paths relative to project root `ACT-Rainfall-Runoff-Modelling/`:

| Asset | Path | Use in deck |
|-------|------|-------------|
| Master journey (Excalidraw) | `diagrams/00_master_journey.excalidraw` | Export to PNG; use on “Journey” slide. |
| Explorer anatomy | `diagrams/01_explorer_anatomy.excalidraw` | Export to PNG; use when introducing the analogy. |
| Vehicle fleet | `diagrams/02_vehicle_fleet.excalidraw` | Export to PNG; use in Models section. |
| Diagnostic spectrum | `diagrams/03_diagnostic_spectrum.excalidraw` | Export to PNG; use in Diagnostics section. |
| Catchment overview | `figures/catchment_overview.png` | Queanbeyan data context. |
| Model win counts | `figures/model_win_counts.png` | “No single model wins” finding. |
| Objective comparison hydrographs | `figures/objective_comparison_hydrographs.png` | Same model, different objectives → different fit. |
| Diagnostic regime spectrum (matplotlib) | `figures/diagnostic_regime_spectrum.png` | Metrics by flow regime. |
| Algorithm comparison scatter | `figures/algorithm_comparison_scatter.png` | SCE-UA vs PyDREAM agreement. |
| Executive summary notebook | `notebooks/10_executive_summary.py` | Source of narrative and Mermaid; reference for “full detail”. |

**Note:** If Excalidraw PNGs are not yet exported, open each `.excalidraw` in Excalidraw and export as PNG, or use the matplotlib-generated figures where specified above.

### 1.5 Naming and Versioning

- **Suggested deck title:** “pyrrm Calibration Experiments: Executive Summary” or “Rainfall-Runoff Calibration Experiments — What We Did and What It Means.”
- **File naming:** e.g. `Executive_Summary_pyrrm_Calibration_Experiments.pptx` (or equivalent).
- **Versioning:** Date or version in footer or filename (e.g. `v1.0_2025-02`).

---

## 2. Master Outline (Slide-by-Slide)

Each slide below has: **Slide number**, **Title**, **Type** (title, content, visual, takeaway, section divider, appendix), **Content/script notes**, and **Assets** to embed.

---

### Section 0: Opening (Slides 1–2)

**Slide 1 — Title**  
- **Type:** Title  
- **Title:** Executive Summary: pyrrm Calibration Experiments  
- **Subtitle (optional):** What we did, what we found, and what it means for model and objective choice  
- **Content:** No body. Optional: short line such as “Queanbeyan River catchment · 65+ experiments · 9 notebooks.”  
- **Assets:** None required.

**Slide 2 — Who this is for / How we’ll tell the story**  
- **Type:** Content  
- **Title:** Who This Is For (and the Story in One Sentence)  
- **Content:**  
  - For a fellow modeller who uses Source and conceptual models but isn’t deep on calibration internals.  
  - One sentence: “We ran dozens of calibrations across four models and 13 objective functions on the Queanbeyan, and we’ll walk through what that means using a single analogy: **calibration as an explorer** with a **brain** (how it searches), **eyes** (what it looks for), and a **vehicle** (the model).”  
- **Assets:** None.

---

### Section 1: Setting the Scene (Slides 3–7)

**Slide 3 — Section divider**  
- **Type:** Section divider  
- **Title:** Part 1 — Setting the Scene  
- **Content:** Optional one line: “Why we did this, where we did it, and how the experiments connect.”  
- **Assets:** None.

**Slide 4 — What we’re doing (rainfall-runoff and calibration)**  
- **Type:** Content  
- **Title:** What Are We Trying to Do?  
- **Content:**  
  - Rainfall-runoff modelling: a **recipe** (model) turns rainfall and evaporation into streamflow.  
  - **Calibration** = finding the best **parameter settings** so the recipe’s output matches observed flow.  
  - Today we use one analogy for the whole talk: **the calibration explorer**.  
- **Assets:** None.

**Slide 5 — The calibration explorer (analogy)**  
- **Type:** Visual + short bullets  
- **Title:** The Calibration Explorer: One Analogy for the Whole Talk  
- **Content (bullets):**  
  - **Explorer** = the calibration process.  
  - **Brain** = how it searches → optimisation algorithm (SCE-UA vs PyDREAM).  
  - **Eyes** = what it looks for → objective function (NSE, KGE, etc.).  
  - **Vehicle** = the model (GR4J, Sacramento, etc.).  
  - **Terrain** = parameter space.  
  - **Scorecard** = diagnostic metrics we use to judge the result.  
- **Assets:** `diagrams/01_explorer_anatomy.excalidraw` (export as PNG) — **embed as main visual**.

**Slide 6 — Why not just use Source?**  
- **Type:** Content  
- **Title:** Why Not Just Use Source?  
- **Content:**  
  - Source is the right tool for operational modelling; its calibration options are limited.  
  - pyrrm is a **research workbench**: we systematically test many model × objective × algorithm combinations.  
  - Insights from pyrrm **feed back into** how we use Source (e.g. which objective to choose).  
  - Complementary, not a replacement.  
- **Assets:** None.

**Slide 7 — Where: Queanbeyan River**  
- **Type:** Visual + short bullets  
- **Title:** Where We Did It: Queanbeyan River at Queanbeyan (Gauge 410734)  
- **Content (bullets):**  
  - ~490 km²; ACT region.  
  - SILO rainfall, Morton PET, observed flow from 1985.  
  - Long, quality-controlled record; representative of ACT conditions.  
- **Assets:** `figures/catchment_overview.png` — rainfall and flow time series.

---

### Section 2: The Journey (Slides 8–9)

**Slide 8 — Section divider**  
- **Type:** Section divider  
- **Title:** How Do All the Experiments Connect?  
- **Content:** One line: “Nine notebooks as waypoints on an expedition.”  
- **Assets:** None.

**Slide 9 — Master journey diagram**  
- **Type:** Visual  
- **Title:** The Master Journey: 9 Notebooks as an Expedition  
- **Content (script notes):**  
  - **Base Camp (Part 1):** 01 Sacramento verification, 02 Calibration quickstart, 03 Routing quickstart.  
  - **Training the Eyes (Part 2):** 04 Objective functions, 05 APEX guide.  
  - **Testing the Brain (Part 3):** 06 Algorithm comparison (SCE-UA vs PyDREAM), 08 Calibration monitor.  
  - **The Summit (Part 4):** 07 Model comparison (4 models × 13 objectives), 09 Calibration reports.  
  - Arrows show dependencies and flow of results (e.g. reports, CSVs).  
- **Assets:** `diagrams/00_master_journey.excalidraw` (export as PNG) — **full-slide or large visual**.

---

### Section 3: The Models — Vehicle Fleet (Slides 10–13)

**Slide 10 — Section divider**  
- **Type:** Section divider  
- **Title:** Part 2 — The Models (The Vehicle Fleet)  
- **Content:** One line: “Four models, from sedan to truck.”  
- **Assets:** None.

**Slide 11 — What models we tested**  
- **Type:** Visual + short table or bullets  
- **Title:** What Models Did We Test?  
- **Content:**  
  - GR4J (4 params) — compact sedan.  
  - GR5J (5 params) — SUV; adds groundwater exchange threshold.  
  - GR6J (6 params) — pickup; adds exponential store for low flows.  
  - Sacramento (22 params) — articulated truck; multi-zone soil moisture.  
  - GR family: production store + routing store; each step adds one mechanism.  
- **Assets:** `diagrams/02_vehicle_fleet.excalidraw` (export as PNG).

**Slide 12 — Does a more complex model always win?**  
- **Type:** Visual + takeaway  
- **Title:** Does a More Complex Model Always Perform Better?  
- **Content (script notes):**  
  - **No.** We counted “wins” across all metric–objective combinations.  
  - No single model dominates. Margins are often small.  
  - **Objective function choice often matters more than model choice.**  
- **Assets:** `figures/model_win_counts.png` (horizontal bar chart of wins by model).

**Slide 13 — Key finding (models)**  
- **Type:** Takeaway  
- **Title:** Key Finding: Models  
- **Content (bullets):**  
  - No single model wins everywhere.  
  - Simpler models (e.g. GR4J) with a good objective can match or beat a complex model with a poor objective.  
  - For ungauged catchments, fewer parameters can be an advantage.  
- **Assets:** None.

---

### Section 4: Objective Functions — The Eyes (Slides 14–18)

**Slide 14 — Section divider**  
- **Type:** Section divider  
- **Title:** Part 3 — Objective Functions (The Explorer’s Eyes)  
- **Content:** One line: “What we optimise determines what the model learns to do well.”  
- **Assets:** None.

**Slide 15 — What is an objective function?**  
- **Type:** Content  
- **Title:** What Is an Objective Function, and Why Does It Matter?  
- **Content:**  
  - The **scorecard we hand the explorer before the trip**: it defines “good.”  
  - **Eyes analogy:** NSE “eyes” see peaks well but are weak on baseflow; log-NSE “eyes” see low flows but are dazzled by floods; sqrt-NSE is more balanced.  
  - The eyes we choose determine what the explorer finds.  
- **Assets:** None.

**Slide 16 — The 13 objectives we tested**  
- **Type:** Content (optional simple diagram or list)  
- **Title:** What Objective Functions Did We Test?  
- **Content:**  
  - **13 objectives** in four families: NSE (4 transforms), KGE (4), KGE-np (4), SDEB (1).  
  - Transforms: none (peaks), sqrt (balanced), log (low flows), inverse (very low).  
  - Same idea: different “lenses” shift emphasis from floods to drought.  
- **Assets:** Optional: simple tree or table (NSE / KGE / KGE-np / SDEB with transform variants).

**Slide 17 — What happens when we change the objective?**  
- **Type:** Visual  
- **Title:** Same Model, Different Eyes → Different Fit  
- **Content (script notes):**  
  - One model (e.g. GR4J), three calibrations: NSE, Sqrt-NSE, Inv-NSE.  
  - NSE: good peaks, weaker baseflow. Sqrt-NSE: compromise. Inv-NSE: good low flows, peaks under-predicted.  
- **Assets:** `figures/objective_comparison_hydrographs.png` (three panels: observed vs simulated for nse, sqrtnse, invnse).

**Slide 18 — Key finding (objectives)**  
- **Type:** Takeaway  
- **Title:** Key Finding: Objectives  
- **Content:**  
  - No single pair of “eyes” sees everything equally well.  
  - Choose the objective to match the application (floods vs water supply vs environmental flows vs drought).  
- **Assets:** None.

---

### Section 5: Diagnostic Metrics — The Scorecard (Slides 19–21)

**Slide 19 — Section divider**  
- **Type:** Section divider  
- **Title:** How Do We Judge Performance? The Panel of Judges  
- **Content:** One line: “25 diagnostic metrics, each focused on a different part of the hydrograph.”  
- **Assets:** None.

**Slide 20 — Diagnostic metrics by flow regime**  
- **Type:** Visual  
- **Title:** Diagnostic Metrics: Grouped by Flow Regime  
- **Content (script notes):**  
  - **Very high (0–2%):** FDC Peak, NSE, Pearson r, Q5 error.  
  - **High (2–10%):** FDC High, KGE, KGE-np, Flashiness.  
  - **Mid (20–70%):** FDC Mid, Sqrt-NSE, KGE-sqrt, Q50, SDEB, Spearman.  
  - **Low (70–90%):** FDC Low, Log-NSE, KGE-log, Q95, BFI.  
  - **Very low (90–100%):** FDC VeryLow, Inv-NSE, KGE-inv.  
  - **All:** RMSE, MAE, PBIAS.  
  - Rule of thumb: floods → blue zone; environmental/drought → amber/red zone.  
- **Assets:** `diagrams/03_diagnostic_spectrum.excalidraw` or `figures/diagnostic_regime_spectrum.png`.

**Slide 21 — How to read the diagnostics**  
- **Type:** Content  
- **Title:** How to Read the Diagnostic Numbers  
- **Content (bullets):**  
  - Identify your flow regime of interest; focus on the metrics in that band.  
  - Always check PBIAS (volume balance) and at least one error in real units (e.g. RMSE).  
  - FDC segment errors show *where* in the flow distribution the model succeeds or fails.  
  - Signatures (Q95, Q50, BFI, Flashiness) summarise regime “personality.”  
- **Assets:** None.

---

### Section 6: APEX (Slides 22–24)

**Slide 22 — Section divider**  
- **Type:** Section divider  
- **Title:** Part 4 — APEX: Building Better Eyes  
- **Content:** One line: “A composite objective that avoids single-metric tunnel vision.”  
- **Assets:** None.

**Slide 23 — What is APEX?**  
- **Type:** Content + optional simple diagram  
- **Title:** What Is APEX, and Why Did We Develop It?  
- **Content:**  
  - Traditional objectives have tunnel vision (e.g. NSE → peaks only).  
  - **APEX** = Adaptive Process-Explicit: penalises (1) chronological errors, (2) FDC shape (ranked term), (3) volume bias, (4) dynamics/timing (multiplier).  
  - Built on SDEB (Lerat et al.) with a dynamics multiplier.  
  - Different transforms (none, sqrt, log, inverse) shift emphasis within APEX.  
- **Assets:** Optional: simple APEX component diagram (chronological + ranked + bias + dynamics → final score).

**Slide 24 — Does APEX improve things?**  
- **Type:** Content / takeaway  
- **Title:** Does APEX Actually Improve Things?  
- **Content:**  
  - We tested 8 APEX configurations. APEX adds value through **structural** innovations (dynamics, ranked term), not just transformation.  
  - Configuration guide: flood → none; general → sqrt; environmental → log; drought → inverse.  
- **Assets:** None (or one small table of recommended configs by application).

---

### Section 7: Algorithms — The Brain (Slides 25–28)

**Slide 25 — Section divider**  
- **Type:** Section divider  
- **Title:** Part 5 — Calibration Algorithms (The Explorer’s Brain)  
- **Content:** One line: “How we search: hill-climber vs cartographer.”  
- **Assets:** None.

**Slide 26 — SCE-UA vs PyDREAM**  
- **Type:** Content  
- **Title:** What’s the Difference Between Optimisation and Bayesian Calibration?  
- **Content:**  
  - **SCE-UA (hill-climber):** finds one “best” parameter set; fast; no uncertainty.  
  - **PyDREAM (cartographer):** maps the full posterior (all plausible parameter sets); slower; gives uncertainty.  
  - Same analogy: one gives GPS of the summit, the other a map of the whole range.  
- **Assets:** None.

**Slide 27 — Does algorithm choice change the answer?**  
- **Type:** Visual  
- **Title:** Does the Algorithm Choice Matter for the Final Answer?  
- **Content (script notes):**  
  - We ran both on Sacramento across 13 objectives.  
  - **Finding:** When converged, both give very similar “best” fits (points near 1:1 line).  
  - Difference: PyDREAM also gives uncertainty (posterior); SCE-UA is faster.  
- **Assets:** `figures/algorithm_comparison_scatter.png` (NSE and KGE: SCE-UA vs PyDREAM, with 1:1 line).

**Slide 28 — When to use which algorithm**  
- **Type:** Takeaway  
- **Title:** When to Use Which Algorithm  
- **Content:**  
  - **SCE-UA:** quick screening, operational calibration, limited compute.  
  - **PyDREAM:** research, uncertainty analysis, decision support.  
- **Assets:** None.

---

### Section 8: Putting It Together (Slides 29–33)

**Slide 29 — Section divider**  
- **Type:** Section divider  
- **Title:** Part 6 — Putting It All Together  
- **Content:** One line: “Recommendations and next steps.”  
- **Assets:** None.

**Slide 30 — What to pick by application**  
- **Type:** Content (optional decision tree or table)  
- **Title:** What Model and Objective Should I Pick for My Application?  
- **Content:**  
  - **Flood forecasting:** NSE or KGE + any model.  
  - **General water supply:** Sqrt-NSE or KGE-sqrt + GR5J or GR6J.  
  - **Environmental flows:** Log-NSE or KGE-log + GR6J.  
  - **Drought planning:** Inv-NSE or KGE-inv + model with low-flow store.  
- **Assets:** Optional: simple decision tree (application → objective + model).

**Slide 31 — Five main takeaways**  
- **Type:** Takeaway  
- **Title:** Five Main Takeaways for Our Source Modelling Work  
- **Content (numbered):**  
  1. Objective function choice often matters more than model choice.  
  2. Always evaluate with multiple diagnostics, not just the one you calibrated to.  
  3. Consider APEX or SDEB where balanced performance matters.  
  4. PyDREAM-style uncertainty is valuable when decisions depend on risk.  
  5. Validate these findings on more ACT catchments before generalising.  
- **Assets:** None.

**Slide 32 — What we should do next**  
- **Type:** Content  
- **Title:** What Should We Do Next?  
- **Content (bullets):**  
  - Multi-catchment validation (beyond Queanbeyan).  
  - Split-sample testing (calibrate vs validate periods).  
  - Include routing in the comparison.  
  - Use PyDREAM posteriors for ensemble predictions.  
  - Feed insights into Source configuration guidance.  
- **Assets:** None.

**Slide 33 — Summary / closing**  
- **Type:** Takeaway  
- **Title:** Summary  
- **Content:**  
  - We ran 65+ calibrations (4 models × 13 objectives, 2 algorithms, APEX variants) on the Queanbeyan.  
  - **Explorer analogy:** brain (algorithm), eyes (objective), vehicle (model), scorecard (diagnostics).  
  - **Main message:** Choose eyes and vehicle to match your application; judge with the full panel of diagnostics.  
- **Assets:** None.

---

### Section 9: Appendix (Slides 34–37)

**Slide 34 — Section divider**  
- **Type:** Section divider  
- **Title:** Appendix  
- **Content:** Optional: “Glossary, references, and where to find more.”  
- **Assets:** None.

**Slide 35 — Glossary (abridged)**  
- **Type:** Content (two columns or scroll)  
- **Title:** Glossary  
- **Content:** Short definitions: NSE, KGE, KGE-np, SDEB, APEX, FDC, MCMC, SCE-UA, BFI, PBIAS, RMSE, MAE, posterior, warmup. (Full list in notebook.)  
- **Assets:** None.

**Slide 36 — References**  
- **Type:** Content  
- **Title:** References  
- **Content:**  
  - GR4J: Perrin et al. (2003). GR5J: Le Moine (2008). GR6J: Pushpalatha et al. (2011).  
  - KGE: Gupta et al. (2009). SDEB: Lerat et al. (2013). SCE-UA: Duan et al. (1994). DREAM: Vrugt (2016).  
- **Assets:** None.

**Slide 37 — Where to find more**  
- **Type:** Content  
- **Title:** Where to Find More  
- **Content:**  
  - **Full Q&A and all details:** `notebooks/10_executive_summary.py` (run or read as script).  
  - **Diagrams:** `diagrams/00–03_*.excalidraw`; **figures:** `figures/*.png`.  
  - **Data and results:** `test_data/model_comparison_*.csv`, `test_data/reports/*.pkl`.  
  - **Individual experiments:** Notebooks 01–09 in `notebooks/`.  
- **Assets:** None.

---

## 3. Summary: Slide Count and Section Lengths

| Section | Slides | Suggested time (approx.) |
|---------|--------|---------------------------|
| Opening | 2 | 1–2 min |
| Setting the scene | 5 | 4–5 min |
| The journey | 2 | 2–3 min |
| Models (vehicle fleet) | 4 | 4–5 min |
| Objectives (eyes) | 5 | 4–5 min |
| Diagnostics (scorecard) | 3 | 2–3 min |
| APEX | 3 | 2–3 min |
| Algorithms (brain) | 4 | 3–4 min |
| Putting it together | 5 | 4–5 min |
| Appendix | 4 | As needed / Q&A |
| **Total** | **37** | **~25–35 min** |

---

## 4. Background Reference: Key Numbers and Facts

Use these when building slide text or speaker notes:

- **Catchment:** Queanbeyan River at Queanbeyan; gauge 410734; ~490 km²; SILO rainfall, Morton PET; observed flow from 1985.
- **Models:** GR4J (4), GR5J (5), GR6J (6), Sacramento (22 parameters).
- **Objective functions:** 13 (NSE × 4, KGE × 4, KGE-np × 4, SDEB × 1).
- **Algorithms:** SCE-UA (optimisation), PyDREAM (Bayesian MCMC).
- **APEX:** 8 configurations tested.
- **Total calibrations:** 52 for model comparison (4 × 13); 65+ including APEX and PyDREAM runs.
- **Diagnostics:** 25 metrics in 5 flow-regime bands + volume/error (RMSE, MAE, PBIAS).
- **Main finding (models):** No single model dominates; GR4J and Sacramento lead on different metrics; objective choice often matters more than model choice.
- **Main finding (algorithms):** SCE-UA and PyDREAM give similar best fits when converged; PyDREAM adds uncertainty.

---

## 5. Checklist for Building the Deck in Another Tool

- [ ] Apply 16:9 template; set title and body font sizes (e.g. title 24–32 pt, body ≥18 pt).
- [ ] Export Excalidraw diagrams to PNG and place on the correct slides (journey, explorer, vehicles, diagnostics).
- [ ] Add figures from `figures/` (catchment, win counts, hydrograph comparison, algorithm scatter, diagnostic spectrum).
- [ ] Use section-divider slides for Parts 1–6 and Appendix.
- [ ] Add “Key finding” or “Takeaway” emphasis (e.g. one accent colour or icon) for slides 13, 18, 24, 28, 31, 33.
- [ ] Optional: add speaker notes from the “Content (script notes)” and “Content” fields above.
- [ ] Add footer with project name and date/version.
- [ ] Run a final pass for consistency (terminology: Explorer, Brain, Eyes, Vehicle, Scorecard; flow-regime colours).

---

*End of specification and outline. Use this document as the single source of structure and content when creating the slide deck in your chosen presentation tool.*
