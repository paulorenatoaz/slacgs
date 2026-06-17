# CoInfoSim Sprint 1 Report Upgrade — Tasks

## Purpose

This document defines the tasks for upgrading the **CoInfoSim Sprint 1 report** after the first functional implementation of Synthetic Scenario 1.

The first Sprint 1 implementation successfully created the new simulation core and a basic HTML report. This document focuses on improving the report so that it becomes more readable, more mathematically expressive, and closer to the legacy SLACGS visual style.

The upgrade must preserve the core Sprint 1 decisions:

- empirical test loss only;
- no empirical train loss;
- no theoretical loss;
- no Bayes error or Bayes reference error;
- fixed test set;
- balanced incremental training samples;
- all non-empty channel subsets of `X = (X1, X2, X3)`;
- three evaluated classifiers: Linear SVM, Logistic Regression, Gaussian Naive Bayes.

The report upgrade should not redesign the Monte Carlo core unless strictly necessary.

---

## Source documents and code

Use these references:

1. `tasks.md` — original Sprint 1 implementation task list.
2. `coinfosim_sprint1_implementation_guidelines.pdf` / `.tex` — Sprint 1 design specification.
3. Current implementation in the repository, especially:
   - `src/coinfosim/reports/sprint1.py`;
   - `src/coinfosim/simulation/monte_carlo.py`;
   - `src/coinfosim/results/analysis.py`;
   - `src/coinfosim/results/summary.py`;
   - `src/coinfosim/models/gaussian.py`;
   - `src/coinfosim/samplers/gaussian.py`.
4. Legacy visualization code for inspiration only, especially:
   - `src/coinfosim/core/model.py`, functions related to 1D, 2D, and 3D data plots.

Do not copy the legacy implementation mechanically. Adapt the visual ideas to the new CoInfoSim architecture.

---

## Definition of done

This report upgrade is complete when the repository can generate an improved Synthetic Scenario 1 report with:

- clearer model-parameter section;
- improved mathematical notation in plots and tables;
- cleaner loss-curve section without repetitive titles;
- redesigned rankings that clearly show which subset is winning;
- interpolated cooperative advantage threshold estimates `N*`;
- simplified Monte Carlo uncertainty summary;
- SLACGS-inspired geometric diagnostics:
  - 1D feature grid;
  - 2D pairwise grid with ellipses and linear separator;
  - 3D triple-wise grid with ellipsoids and linear separator when feasible;
- a sample-growth GIF or animation artifact showing training samples and curves evolving with `n_per_class`;
- tests for the new analysis and visualization functions;
- an extended smoke/test run up to `n_per_class = 32`.

---

## Explicit non-goals

Do **not** implement the following in this report upgrade:

- new synthetic scenarios beyond Synthetic Scenario 1;
- real-data reports;
- cost-aware reports;
- Bayes error;
- empirical train loss;
- theoretical loss;
- full interactive dashboards;
- dependency-heavy visualization frameworks unless absolutely necessary;
- a full rewrite of the simulation core.

The goal is to improve the existing report and visual diagnostics, not to replace the entire system.

---

## Important design decisions

### Loss notation

Use mathematical notation in plots.

- Use `n` or `n_{per class}` on the x-axis.
- Use `L` or `\hat{L}` on the y-axis.
- Explain once in the report that `L` denotes empirical misclassification rate on the fixed test set.
- Do not repeat "Empirical test loss" in every chart title.

### Rankings

The current ranking table is too repetitive and hard to interpret. Replace it with clearer views:

1. **Best subset by sample size**:
   - rows: `n_per_class`;
   - columns: classifiers;
   - cell: winning subset.

2. **Final ranking at largest n**:
   - one table per classifier or one compact grouped table;
   - rank all subsets by mean loss at the largest evaluated `n_per_class`.

3. **Cooperative advantage summary**:
   - baseline subset;
   - cooperative subset;
   - grid threshold;
   - interpolated threshold;
   - classifier.

### Interpolated N-star

The report should no longer use only a discrete `N*`.

For a baseline subset `A`, cooperative subset `B`, and classifier `f`, define:

```text
Delta(n) = L_A(n) - L_B(n)
```

The cooperative subset `B` beats `A` when:

```text
Delta(n) > 0
```

If two consecutive evaluated sample sizes `n_left < n_right` satisfy:

```text
Delta(n_left) <= 0
Delta(n_right) > 0
```

then estimate the interpolated threshold by linear interpolation:

```text
N_star_interp = n_left + (0 - Delta(n_left)) * (n_right - n_left) / (Delta(n_right) - Delta(n_left))
```

Also keep the discrete grid threshold:

```text
N_star_grid = n_right
```

In the report, prefer showing both:

- `N*_grid`;
- `N*_interp`.

If no crossing is observed, both values should be reported as missing / not observed.

### Monte Carlo uncertainty

Do not show one huge table with every subset/classifier/n cell unless placed in an appendix or collapsible section.

The main report should show:

1. Compact stopping table by `n_per_class`:
   - `n`;
   - replications;
   - stopping reason;
   - max CI half-width.

2. Optional hardest-cell table:
   - for each `n`, show the subset/classifier cell with largest CI half-width.

### Visual diagnostics

Use one **linear classifier** for geometric separators. Prefer Linear SVM.

The geometric diagnostics are illustrative. They do not need to visualize every classifier. The simulation still evaluates all three classifiers.

Visual diagnostics should include all feature combinations by dimensionality:

- 1D grid: all single channels;
- 2D grid: all pairwise channel combinations;
- 3D grid: all triple channel combinations.

For future scenarios with more than 3 channels, the same logic should generalize:

- all 1D combinations;
- all 2D combinations;
- all 3D combinations.

---

## Checkpoint 0 — Inspect current report and visualization code

Before editing, inspect the current implementation.

Tasks:

- [ ] Open the current `src/coinfosim/reports/sprint1.py`.
- [ ] Identify how current loss curves are generated.
- [ ] Identify how current scatter plots are generated.
- [ ] Inspect `src/coinfosim/results/analysis.py` for existing `N*` logic.
- [ ] Inspect `src/coinfosim/results/summary.py` for summary-table generation.
- [ ] Inspect legacy `src/coinfosim/core/model.py` visualization functions for inspiration:
  - 1D plot;
  - 2D plot with ellipses and SVM line;
  - 3D plot with ellipsoids and SVM hyperplane;
  - combined exported image.
- [ ] Propose the exact files to edit or add.

Do not make large changes in this checkpoint.

Expected report in chat:

```text
Checkpoint 0 report:
- Current report structure
- Current visualization limitations
- Legacy visualization ideas worth preserving
- Proposed files/modules
- Risks / open questions
- Request permission to proceed to Checkpoint 1
```

---

## Checkpoint 1 — Improve report layout and mathematical notation

Goal: make the existing report more readable before adding complex visuals.

Tasks:

- [ ] Improve the model-parameter section using cards, boxes, or more readable HTML/CSS.
- [ ] Improve display of `mu0`, `mu1`, `Sigma0`, and `Sigma1`.
- [ ] Add a concise explanation that `L` denotes empirical misclassification rate on the fixed test set.
- [ ] Update loss-curve titles:
  - remove repetitive "Empirical test loss" phrase;
  - use classifier names as chart titles.
- [ ] Update plot labels with math notation:
  - x-axis: `$n$` or `$n_{per\ class}$`;
  - y-axis: `$L$` or `$\hat{L}$`.
- [ ] Improve legend readability.
- [ ] Keep the explicit notice excluding empirical train loss, theoretical loss, and Bayes error.

Tests/checks:

- [ ] Generate a smoke report.
- [ ] Confirm the HTML report opens.
- [ ] Confirm model parameters are easier to read.
- [ ] Confirm chart titles and axis labels changed correctly.
- [ ] Confirm no Bayes/train/theoretical loss appears.

Stop and report before continuing.

---

## Checkpoint 2 — Redesign rankings and Monte Carlo uncertainty tables

Goal: replace unclear tables with interpretable result summaries.

Tasks:

- [ ] Replace or supplement the current ranking table with **Best subset by sample size**:
  - rows: `n_per_class`;
  - columns: classifiers;
  - values: best subset label.
- [ ] Add **Final ranking at largest n**:
  - rank all subsets by mean loss at the largest `n_per_class`;
  - present separately by classifier or in a clear grouped table.
- [ ] Simplify the Monte Carlo uncertainty section:
  - keep stopping table by `n_per_class`;
  - add optional hardest-cell table by `n_per_class`.
- [ ] Move any very detailed per-cell summary to a secondary section if still needed.
- [ ] Ensure tables explain what they are ranking.

Suggested implementation:

- Add or update functions in `src/coinfosim/results/analysis.py` and/or `src/coinfosim/reports/sprint1.py`.
- Prefer reusable DataFrame functions over one-off HTML logic.

Tests:

- [ ] Unit test best subset by sample size.
- [ ] Unit test final ranking at largest n.
- [ ] Unit test hardest-cell summary if implemented.
- [ ] Generate smoke report and inspect tables.

Stop and report before continuing.

---

## Checkpoint 3 — Implement interpolated N-star

Goal: support both discrete grid threshold and interpolated threshold.

Tasks:

- [ ] Extend the cooperative-threshold analysis to compute:
  - `N*_grid`;
  - `N*_interp`.
- [ ] Use the interpolation rule described above.
- [ ] Preserve no-threshold behavior when no crossing is observed.
- [ ] Make sure the crossing is based on:

```text
Delta(n) = L_A(n) - L_B(n)
```

- [ ] `B` is cooperative if `Delta(n) > 0`.
- [ ] If `Delta(n_left) <= 0` and `Delta(n_right) > 0`, interpolate.
- [ ] If the first evaluated point already has `Delta(n) > 0`, report the grid threshold as the first evaluated `n`; interpolation may be unavailable or equal to the first `n` depending on implementation. Document the chosen behavior.
- [ ] Update the `N*` table in the report to show both grid and interpolated estimates.
- [ ] Use mathematical notation in table headings where practical.
- [ ] Explain that the interpolated value is an estimate obtained between consecutive sample sizes.

Tests:

- [ ] Test crossing between two sample sizes with known interpolated value.
- [ ] Test crossing exactly at an observed sample size.
- [ ] Test first-point already positive case.
- [ ] Test no-crossing case.
- [ ] Test multiple comparisons across classifiers.

Stop and report before continuing.

---

## Checkpoint 4 — Add SLACGS-inspired geometric visualization functions

Goal: implement reusable geometry plots before integrating them into the report.

Create a new module if appropriate, for example:

```text
src/coinfosim/reports/visualizations.py
```

or:

```text
src/coinfosim/visualization/geometry.py
```

Tasks:

### 1D grid

- [ ] Implement a function that plots all single-channel views.
- [ ] For each channel:
  - show class samples along the x-axis;
  - show approximate class density curves when feasible;
  - show mean markers;
  - show one-sigma dispersion markers;
  - show linear separator threshold from Linear SVM trained on the selected training data.

### 2D grid

- [ ] Implement a function that plots all pairwise channel combinations.
- [ ] For each pair:
  - show class scatter;
  - show class centers;
  - show Gaussian ellipses for each class;
  - show Linear SVM separating line.
- [ ] Use model means/covariances, not only empirical sample estimates, for the ellipses.
- [ ] Use adaptive axis limits based on data and ellipses, not fixed `[-10, 10]` unless appropriate.

### 3D grid

- [ ] Implement a function that plots all triple channel combinations.
- [ ] For each triple:
  - show class scatter;
  - show class centers;
  - show Gaussian ellipsoids if feasible;
  - show Linear SVM separating hyperplane if feasible.
- [ ] If 3D ellipsoids are too heavy, implement a clear first version and document limitations.

General requirements:

- [ ] Use Linear SVM only for geometric separators.
- [ ] Do not confuse geometric diagnostic classifier with the three evaluated classifiers.
- [ ] Keep plotting functions deterministic.
- [ ] Return Matplotlib figures or encoded image URIs in a reusable way.
- [ ] Avoid bloating HTML excessively if images become large.

Tests/checks:

- [ ] Test that 1D grid function returns a figure.
- [ ] Test that 2D grid function returns a figure.
- [ ] Test that 3D grid function returns a figure or gracefully degrades.
- [ ] Test ellipse helper on a known covariance matrix.
- [ ] Test that functions work for `d=3`.
- [ ] If feasible, test `d=4` combination counts without requiring a full simulation.

Stop and report before continuing.

---

## Checkpoint 5 — Integrate geometric diagnostics into the HTML report

Goal: replace the simple scatter diagnostics with richer geometric diagnostics.

Tasks:

- [ ] Replace or supplement current `Data diagnostics` section.
- [ ] Rename it to something clearer, for example:

```text
Synthetic data geometry
```

or:

```text
Model geometry and sample growth diagnostics
```

- [ ] Add 1D single-channel grid.
- [ ] Add 2D pairwise grid with ellipses and linear separator.
- [ ] Add 3D triple-wise grid with ellipsoids/hyperplane if feasible.
- [ ] Add explanatory text:
  - geometric separators use Linear SVM only;
  - simulation metrics still use all three classifiers;
  - ellipses/ellipsoids represent the Gaussian model geometry.
- [ ] Keep the report readable and not overloaded.

Tests/checks:

- [ ] Generate report in smoke mode.
- [ ] Confirm all three geometry sections appear.
- [ ] Confirm figures render in HTML.
- [ ] Confirm report size remains reasonable.
- [ ] Confirm no Bayes/train/theoretical loss appears.

Stop and report before continuing.

---

## Checkpoint 6 — Implement sample-growth GIF with curves and N-star markers

Goal: add an animated diagnostic inspired by SLACGS.

The GIF should show the simulation building over `n_per_class` values.

Recommended frame structure:

- top or left: geometric sample-growth view for a representative training replication, preferably `replication_id = 0`;
- bottom or right: loss curves for the three evaluated classifiers;
- curves should be drawn progressively up to the current `n`;
- interpolated or grid `N*` should be marked with a red dot or red vertical marker once it is available.

Tasks:

- [ ] Implement a GIF-generation function.
- [ ] Use a representative training replication, e.g. `replication_id = 0`.
- [ ] Use the configured sample sizes as animation frames.
- [ ] Show training data growing with `n_per_class`.
- [ ] Include at least one geometric panel. Prefer 2D pairwise grid if full 1D/2D/3D composite is too heavy.
- [ ] Include three curve panels, one per classifier.
- [ ] Each classifier panel should contain curves for all channel subsets.
- [ ] Add red markers for `N*` when available.
- [ ] Save the GIF as a separate artifact, e.g.:

```text
output/reports/synthetic_scenario_1_growth.gif
```

- [ ] Embed the GIF in the HTML report if size is acceptable.
- [ ] If embedding makes the report too large, link to the GIF artifact instead.

Tests/checks:

- [ ] Generate GIF in smoke/extended-smoke mode.
- [ ] Confirm file exists.
- [ ] Confirm it has the expected number of frames.
- [ ] Confirm it can be opened by a standard image viewer or browser.
- [ ] Confirm report references or embeds it.

Stop and report before continuing.

---

## Checkpoint 7 — Extend smoke/test configuration to n_per_class up to 32

Goal: make the test report more informative than the initial `[2, 4, 8]` run.

Tasks:

- [ ] Add or adjust a configuration suitable for report validation with:

```text
sample_sizes = [2, 4, 8, 16, 32]
```

- [ ] This may be a new mode, for example:

```text
report_smoke
```

or an optional CLI flag, for example:

```text
--max-n-per-class 32
```

- [ ] Do not make the default smoke mode too slow unless acceptable.
- [ ] Run an extended test simulation up to `n_per_class = 32`.
- [ ] Generate the improved report and GIF using the extended sample sizes.

Tests/checks:

- [ ] CLI or script accepts the extended configuration.
- [ ] Simulation runs successfully up to `n_per_class = 32`.
- [ ] Loss curves have all expected sample-size points.
- [ ] GIF includes all expected frames.
- [ ] `N*` interpolation uses the extended sample grid.

Stop and report before continuing.

---

## Checkpoint 8 — Final validation and cleanup

Goal: ensure the report upgrade is stable and documented.

Tasks:

- [ ] Run the full test suite.
- [ ] Run `python -m compileall src`.
- [ ] Run the report generation command in smoke or extended-smoke mode.
- [ ] Confirm output paths.
- [ ] Update minimal documentation if needed.
- [ ] Confirm the old Sprint 1 functionality still works.
- [ ] Confirm legacy simulator was not broken.

Final manual checklist:

- [ ] Model parameter section is visually clearer.
- [ ] Loss curves use mathematical notation.
- [ ] Ranking section clearly explains what is ranked.
- [ ] Final ranking at largest `n` appears.
- [ ] Best subset by sample size appears.
- [ ] `N*` table includes interpolated estimate.
- [ ] Monte Carlo uncertainty summary is compact and interpretable.
- [ ] 1D grid appears.
- [ ] 2D pairwise grid appears with ellipses and separator.
- [ ] 3D grid appears or degrades gracefully.
- [ ] GIF exists and is referenced or embedded.
- [ ] No empirical train loss appears.
- [ ] No theoretical loss appears.
- [ ] No Bayes error appears.

Final chat report must include:

- summary of changes;
- files created/modified;
- tests run;
- report path;
- GIF path;
- screenshots or textual description if possible;
- known limitations;
- recommendations for the next report iteration.

---

## Suggested implementation order summary

1. Inspect current report and legacy visualizations.
2. Improve layout and notation.
3. Redesign ranking and uncertainty tables.
4. Implement interpolated `N*`.
5. Implement geometry visualization helpers.
6. Integrate geometry into report.
7. Implement GIF.
8. Extend test run to `n_per_class = 32`.
9. Validate and clean up.

---

## Notes for implementation agent

Work checkpoint by checkpoint. After each checkpoint, stop and report in chat. Do not continue until the user approves.

Prefer simple, testable improvements over a large unreviewable rewrite. The current simulation core appears acceptable; focus on report quality, analysis clarity, visual diagnostics, and interpolated cooperative thresholds.
