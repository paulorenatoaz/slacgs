# CoInfoSim Sprint 1 — Implementation Tasks

## Purpose

This document lists the implementation tasks for **CoInfoSim Sprint 1**.

Sprint 1 implements the first functional CoInfoSim experiment:

> **Synthetic Scenario 1 — Simple Complementary Channel**

The goal is to build a new CoInfoSim simulation core, execute the first synthetic scenario, and generate a clean report showing empirical test-loss curves, channel-subset rankings, cooperative advantage thresholds, Monte Carlo replication information, and visual diagnostics.

This document is intended to be referenced by an implementation prompt. It is operational: tasks are written so an agent can execute them in order.

---

## Source documents

Use these documents as conceptual references:

1. `coinfosim_research_proposal_v4.pdf`
2. `coinfosim_research_proposal_v4.tex`
3. `coinfosim_sprint1_implementation_guidelines.pdf`
4. `coinfosim_sprint1_implementation_guidelines.tex`

The Sprint 1 implementation guidelines are the main source of truth for this task list.

---

## Definition of done

Sprint 1 is complete when the repository can:

- define the Sprint 1 Gaussian simulation model explicitly through class means and covariance matrices;
- generate balanced, incremental training samples with `n_per_class` samples per class;
- generate one fixed large test set per simulation model;
- evaluate all non-empty channel subsets of `X = (X1, X2, X3)`;
- evaluate three classifiers: Linear SVM, Logistic Regression, and Gaussian Naive Bayes;
- estimate empirical test loss only;
- stop Monte Carlo repetitions using a standard-error stopping rule;
- support execution modes `smoke`, `fast`, and `full`;
- compute best subsets and cooperative advantage thresholds `N*`;
- produce a readable report for Synthetic Scenario 1;
- preserve the legacy simulator without breaking existing imports or tests.

---

## Explicit non-goals for Sprint 1

Do **not** implement the following in Sprint 1:

- real-data experiments;
- dataset-anchored scenarios;
- cost-aware channel selection;
- channel costs or label/reference costs;
- scenario grids;
- multiple synthetic scenarios beyond Scenario 1;
- empirical train loss;
- theoretical loss as a loss type;
- Bayes error, Bayes reference error, or Bayes-risk diagnostics;
- a full replacement of the legacy SLACGS-compatible simulator;
- a full publication-grade reporting system for every future scenario.

The new simulator should be built in parallel to the legacy implementation.

---

## Task group 0 — Repository preparation

- [ ] Inspect the current repository structure before editing.
- [ ] Identify the current legacy implementation modules, especially `core.Model`, `core.Simulator`, and current reporting code.
- [ ] Do not remove or rewrite the legacy simulator.
- [ ] Add short comments or documentation where useful to clarify that the current `core` implementation is legacy/SLACGS-compatible.
- [ ] Decide where the new Sprint 1 implementation will live. Recommended structure:

```text
src/coinfosim/
  models/
  samplers/
  classifiers/
  simulation/
  scenarios/
  results/
  reports/
```

- [ ] Keep public package imports stable unless a change is necessary.
- [ ] Ensure the new modules do not depend on legacy `LossType` or legacy sigma/rho parameter vectors.

---

## Task group 1 — Gaussian simulation model

Create a new model class for the CoInfoSim simulation model.

Recommended file:

```text
src/coinfosim/models/gaussian.py
```

Recommended class:

```python
GaussianSimulationModel
```

Requirements:

- [ ] The model must be initialized explicitly from class means and covariance matrices:

```python
GaussianSimulationModel(
    means={0: mu0, 1: mu1},
    covariances={0: Sigma0, 1: Sigma1},
)
```

- [ ] Do not use a sigma/rho parameter vector.
- [ ] Do not include `channel_names` in Sprint 1.
- [ ] Infer `d` from the length of the mean vectors.
- [ ] Infer class labels from the keys of `means` and `covariances`.
- [ ] Validate that every class has both a mean vector and a covariance matrix.
- [ ] Validate that all mean vectors have length `d`.
- [ ] Validate that all covariance matrices have shape `(d, d)`.
- [ ] Validate that all covariance matrices are symmetric.
- [ ] Validate that all covariance matrices are positive definite.
- [ ] Store means and covariances as NumPy arrays internally.
- [ ] Provide a property or method returning the number of classes `K`.
- [ ] Provide a property or method returning the number of channels `d`.
- [ ] Provide a method to extract a restricted model or restricted parameters for a channel subset.
- [ ] Add unit tests for valid and invalid model definitions.

Sprint 1 model parameters:

```python
mu0 = [-0.70, -0.55, -0.30]
mu1 = [ 0.70,  0.55,  0.30]

Sigma = [
    [1.00, 0.35, 0.05],
    [0.35, 1.00, 0.05],
    [0.05, 0.05, 1.00],
]
```

For Sprint 1:

```python
means = {0: mu0, 1: mu1}
covariances = {0: Sigma, 1: Sigma}
```

---

## Task group 2 — Dataset containers

Create minimal data containers for train/test datasets.

Recommended file:

```text
src/coinfosim/samplers/dataset.py
```

Recommended class:

```python
Dataset
```

Requirements:

- [ ] Store feature matrix `X`.
- [ ] Store target vector `y`.
- [ ] Use shape `(n_samples, d)` for `X`.
- [ ] Use shape `(n_samples,)` for `y`.
- [ ] Validate matching number of rows between `X` and `y`.
- [ ] Provide a method to restrict the dataset to a channel subset:

```python
dataset.select_channels(subset)
```

- [ ] Add unit tests for shape validation and channel selection.

---

## Task group 3 — Gaussian class-conditional sampler

Create a sampler that generates training and test datasets from `GaussianSimulationModel`.

Recommended file:

```text
src/coinfosim/samplers/gaussian.py
```

Recommended class:

```python
GaussianClassConditionalSampler
```

Requirements:

- [ ] The sampler must receive a `GaussianSimulationModel`.
- [ ] The sampler must receive a base random seed.
- [ ] The sampler must generate balanced training data.
- [ ] In Sprint 1, `n_per_class` means samples per class.
- [ ] Do not use the legacy `half_n = int(n / 2)` semantics.
- [ ] Implement:

```python
sample_train(n_per_class: int, replication_id: int) -> Dataset
```

- [ ] Implement:

```python
sample_test() -> Dataset
```

- [ ] The test set must be fixed for the simulation model.
- [ ] The fixed test set must be generated once and reused across sample sizes, replications, subsets, and classifiers.
- [ ] The sampler must support `test_samples_per_class`.
- [ ] Training generation must be deterministic by class and replication.
- [ ] For the same class and replication, samples must be prefix-nested:

```text
sample_train(n_per_class=16, replication_id=r)
```

must be contained as the first samples of:

```text
sample_train(n_per_class=64, replication_id=r)
```

for the same `r`.

- [ ] Add tests confirming balanced training output.
- [ ] Add tests confirming deterministic generation.
- [ ] Add tests confirming nested-prefix behavior.
- [ ] Add tests confirming fixed test-set behavior.

Implementation note:

Use deterministic seed derivation such as:

```python
seed = derive_seed(base_seed, class_label, replication_id, split="train")
```

and for the fixed test set:

```python
seed = derive_seed(base_seed, class_label, split="test")
```

---

## Task group 4 — Channel subset generation

Create utilities to enumerate non-empty channel subsets.

Recommended file:

```text
src/coinfosim/simulation/subsets.py
```

Requirements:

- [ ] Implement:

```python
all_nonempty_subsets(d: int) -> list[tuple[int, ...]]
```

- [ ] For `d = 3`, return seven subsets:

```text
(0,), (1,), (2,), (0,1), (0,2), (1,2), (0,1,2)
```

- [ ] Provide display labels using one-based channel names:

```text
X1, X2, X3, X1+X2, X1+X3, X2+X3, X1+X2+X3
```

- [ ] Add unit tests for subset generation and labels.

---

## Task group 5 — Classifier registry

Create a small classifier registry for Sprint 1.

Recommended file:

```text
src/coinfosim/classifiers/registry.py
```

Required classifiers:

- Linear SVM;
- Logistic Regression;
- Gaussian Naive Bayes.

Requirements:

- [ ] Implement factory functions returning fresh unfitted classifiers.
- [ ] Use deterministic classifier configuration where applicable.
- [ ] Recommended classifier keys:

```text
linear_svm
logistic_regression
gaussian_nb
```

- [ ] Recommended display labels:

```text
Linear SVM
Logistic Regression
Gaussian Naive Bayes
```

- [ ] Do not include classifier hyperparameter search in Sprint 1.
- [ ] Do not include RBF SVM, kNN, Random Forest, gradient boosting, or neural networks in Sprint 1.
- [ ] Add unit tests confirming that each registry entry creates an unfitted estimator with `fit` and `predict` methods.

Recommended defaults:

```python
LinearSVC or SVC(kernel="linear")
LogisticRegression(max_iter=1000)
GaussianNB()
```

Prefer robust scikit-learn defaults. If using `LinearSVC`, ensure probability output is not required.

---

## Task group 6 — Empirical test-loss evaluation

Create a metric utility for empirical classification loss.

Recommended file:

```text
src/coinfosim/simulation/metrics.py
```

Requirements:

- [ ] Implement:

```python
empirical_test_loss(estimator, test_dataset: Dataset) -> float
```

- [ ] The metric must compute misclassification rate:

```text
mean(predicted_y != true_y)
```

- [ ] The metric must return a float in `[0, 1]`.
- [ ] Do not compute empirical train loss in Sprint 1.
- [ ] Do not compute theoretical loss in Sprint 1.
- [ ] Do not compute Bayes error in Sprint 1.
- [ ] Add unit tests with a dummy classifier or small known example.

---

## Task group 7 — Monte Carlo accumulator and result objects

Create result containers for storing losses and computing summaries.

Recommended files:

```text
src/coinfosim/results/accumulator.py
src/coinfosim/results/summary.py
```

Requirements:

- [ ] Store individual replication losses indexed by:

```text
n_per_class, subset, classifier_name, replication_id
```

- [ ] Provide methods to compute mean test loss:

```python
mean_loss(n_per_class, subset, classifier_name)
```

- [ ] Provide methods to compute standard deviation and standard error:

```python
std_loss(n_per_class, subset, classifier_name)
standard_error(n_per_class, subset, classifier_name)
```

- [ ] Provide method to compute number of replications completed for each `n_per_class`.
- [ ] For Sprint 1, all subsets/classifiers should share the same number of replications for a given `n_per_class`.
- [ ] Provide a table-like export as pandas DataFrame or CSV.
- [ ] Add tests for accumulation and summary statistics.

---

## Task group 8 — Standard-error stopping rule

Implement the Monte Carlo stopping rule based on standard error.

Recommended file:

```text
src/coinfosim/simulation/stopping.py
```

Recommended class:

```python
StandardErrorStoppingRule
```

Requirements:

- [ ] The rule must wait until `min_replications` is reached.
- [ ] The rule must evaluate only at replication batch boundaries.
- [ ] For each `n_per_class`, compute the standard error for every subset/classifier pair.
- [ ] Use an approximate 95% confidence half-width:

```text
ci_half_width_observed = 1.96 * SE
```

- [ ] Stop when:

```text
max_observed_ci_half_width <= ci_half_width_target
```

across all subset/classifier pairs.

- [ ] Stop if `max_replications` is reached, even if the CI target is not met.
- [ ] Record whether each `n_per_class` stopped by convergence or by maximum budget.
- [ ] Add tests for stopping before and after `min_replications`.
- [ ] Add tests for stopping at the CI target.
- [ ] Add tests for maximum-budget termination.

Do not use the legacy mean-difference stopping rule as the default for Sprint 1.

---

## Task group 9 — Monte Carlo budget and execution modes

Create configuration presets for `smoke`, `fast`, and `full` modes.

Recommended file:

```text
src/coinfosim/simulation/config.py
```

Recommended classes or functions:

```python
MonteCarloConfig
get_mode_config(mode: str) -> MonteCarloConfig
```

Requirements:

- [ ] Implement mode names:

```text
smoke
fast
full
```

- [ ] Each mode must define:

```text
sample_sizes
min_replications
max_replications
replication_batch_size
test_samples_per_class
ci_half_width_target
base_seed
```

- [ ] `smoke` must run quickly and mainly verify correctness.
- [ ] `fast` must support development and visual inspection.
- [ ] `full` must be suitable for a more stable report.
- [ ] Add validation for invalid mode names.
- [ ] Add tests that mode configs are internally consistent.

Suggested initial presets may be adjusted after runtime testing:

```text
smoke:
  sample_sizes: [2, 4, 8]
  min_replications: 5
  max_replications: 20
  replication_batch_size: 5
  test_samples_per_class: 200
  ci_half_width_target: 0.05

fast:
  sample_sizes: [2, 4, 8, 16, 32, 64]
  min_replications: 30
  max_replications: 300
  replication_batch_size: 10
  test_samples_per_class: 1000
  ci_half_width_target: 0.01

full:
  sample_sizes: [2, 4, 8, 16, 32, 64, 128, 256, 512]
  min_replications: 100
  max_replications: 2000
  replication_batch_size: 20
  test_samples_per_class: 5000
  ci_half_width_target: 0.005
```

If these values are too expensive, tune them while preserving the conceptual structure.

---

## Task group 10 — New Monte Carlo simulator

Create the new CoInfoSim Monte Carlo simulator.

Recommended file:

```text
src/coinfosim/simulation/monte_carlo.py
```

Recommended class:

```python
CooperativeMonteCarloSimulator
```

Requirements:

- [ ] The simulator must receive:

```text
GaussianSimulationModel
GaussianClassConditionalSampler
classifier registry or classifier list
channel subsets
MonteCarloConfig
StandardErrorStoppingRule
```

- [ ] The simulator must execute the loop:

```text
n_per_class -> replication -> subset -> classifier
```

- [ ] For every `n_per_class`, evaluate all subset/classifier pairs.
- [ ] For every replication, generate one balanced training dataset.
- [ ] Reuse the fixed test set.
- [ ] Fit each classifier on the selected training channels.
- [ ] Evaluate empirical test loss on the selected test channels.
- [ ] Accumulate results.
- [ ] Check stopping rule at batch boundaries.
- [ ] Store final summaries.
- [ ] Record runtime metadata.
- [ ] Record mode and configuration used.
- [ ] Return a structured result object.
- [ ] Add a smoke/integration test that runs the full loop on a tiny configuration.

The simulator must not compute empirical train loss, theoretical loss, or Bayes error.

---

## Task group 11 — Scenario 1 definition

Create a scenario definition for Synthetic Scenario 1.

Recommended file:

```text
src/coinfosim/scenarios/synthetic.py
```

Recommended function:

```python
make_synthetic_scenario_1() -> GaussianSimulationModel
```

Requirements:

- [ ] Name the scenario:

```text
Synthetic Scenario 1 — Simple Complementary Channel
```

- [ ] Store or expose the scientific question:

```text
When does an individually weaker channel improve classification by adding complementary information?
```

- [ ] Use the following model:

```python
mu0 = [-0.70, -0.55, -0.30]
mu1 = [ 0.70,  0.55,  0.30]
Sigma = [
    [1.00, 0.35, 0.05],
    [0.35, 1.00, 0.05],
    [0.05, 0.05, 1.00],
]
```

- [ ] Use `means = {0: mu0, 1: mu1}`.
- [ ] Use `covariances = {0: Sigma, 1: Sigma}`.
- [ ] Ensure the scenario exposes `d = 3` and all seven non-empty subsets.
- [ ] Add tests confirming the scenario can be constructed.

---

## Task group 12 — Best-subset and cooperative-threshold summaries

Implement post-processing summaries.

Recommended file:

```text
src/coinfosim/results/analysis.py
```

Requirements:

- [ ] Compute the best subset for each classifier and sample size:

```text
A*_f(n) = argmin_A Lbar_{A,f}(n)
```

- [ ] Compute cooperative advantage thresholds:

```text
N*(A, B; f) = min { n : Lbar_{B,f}(n) < Lbar_{A,f}(n) }
```

- [ ] For Sprint 1, compute at least the following comparisons:

```text
best pair vs best single
full subset vs best pair
X1+X3 vs X1
X1+X2+X3 vs X1+X2
```

- [ ] Handle cases where no threshold is observed.
- [ ] Export rankings and thresholds as tables.
- [ ] Add tests using synthetic loss tables with known expected thresholds.

---

## Task group 13 — Sprint 1 report generation

Create a first report generator for Scenario 1.

Recommended file:

```text
src/coinfosim/reports/sprint1.py
```

Recommended output:

```text
output/reports/synthetic_scenario_1_report.html
```

A Markdown report is acceptable as an intermediate step if HTML is not yet ready, but the target for Sprint 1 is an HTML report.

Minimum report content:

- [ ] Report title.
- [ ] Scenario name.
- [ ] Scientific question.
- [ ] Model parameters: `mu0`, `mu1`, `Sigma0`, `Sigma1`.
- [ ] Execution mode: `smoke`, `fast`, or `full`.
- [ ] Sample sizes used.
- [ ] Fixed test-set size per class.
- [ ] Monte Carlo stopping rule and target CI half-width.
- [ ] Final number of replications per `n_per_class`.
- [ ] Classifiers evaluated.
- [ ] Channel subsets evaluated.
- [ ] Empirical test-loss curves by classifier and subset.
- [ ] Best-subset rankings.
- [ ] Cooperative advantage thresholds `N*`.
- [ ] Monte Carlo uncertainty summary.
- [ ] 2D scatter plots for `(X1, X2)`, `(X1, X3)`, and `(X2, X3)`.
- [ ] Optional 3D scatter plot.
- [ ] Optional GIF or image sequence showing sample-size growth.

Report quality requirements:

- [ ] The report should be visually clean.
- [ ] Avoid overloading a single plot with too many curves when separate plots by classifier are clearer.
- [ ] Use consistent subset labels.
- [ ] Make it obvious that only empirical test loss is reported.
- [ ] Make it obvious that Bayes error is intentionally excluded from Sprint 1.

---

## Task group 14 — CLI or script entry point

Provide a simple way to run Synthetic Scenario 1.

Recommended options:

1. CLI command, preferred:

```bash
coinfosim run-scenario synthetic-1 --mode smoke
coinfosim run-scenario synthetic-1 --mode fast
coinfosim run-scenario synthetic-1 --mode full
```

2. Script fallback:

```bash
python scripts/run_synthetic_scenario_1.py --mode smoke
```

Requirements:

- [ ] Support `--mode smoke|fast|full`.
- [ ] Support `--output-dir` if easy.
- [ ] Print output report path after successful execution.
- [ ] Fail clearly on invalid mode.
- [ ] Add a smoke test or documented manual test for the entry point.

Do not remove legacy CLI commands unless necessary.

---

## Task group 15 — Tests

Add tests for the new Sprint 1 implementation.

Minimum tests:

- [ ] `GaussianSimulationModel` validation.
- [ ] Gaussian sampler balanced output.
- [ ] Gaussian sampler deterministic output.
- [ ] Gaussian sampler nested-prefix behavior.
- [ ] Fixed test-set behavior.
- [ ] Channel subset generation.
- [ ] Classifier registry.
- [ ] Empirical test loss.
- [ ] Accumulator summaries.
- [ ] Standard-error stopping rule.
- [ ] Mode configuration.
- [ ] Scenario 1 construction.
- [ ] Cooperative threshold computation.
- [ ] End-to-end smoke simulation.

Recommended command:

```bash
pytest
```

Also run:

```bash
python -m compileall src
```

---

## Task group 16 — Documentation updates

Update documentation only as needed for Sprint 1.

Requirements:

- [ ] Add a short README section or docs page explaining how to run Sprint 1.
- [ ] State that the new Sprint 1 simulator is separate from the legacy simulator.
- [ ] State that Sprint 1 reports empirical test loss only.
- [ ] State that Bayes error is intentionally excluded from the new Sprint 1 implementation.
- [ ] State that `n_per_class` is the number of training samples per class.
- [ ] Add a link or pointer to the Sprint 1 implementation guidelines PDF/LaTeX if those files are committed.

---

## Task group 17 — Manual validation checklist

After implementation, run the following checks manually:

- [ ] `python -m compileall src`
- [ ] `pytest`
- [ ] `pip install -e .`
- [ ] `coinfosim --help`
- [ ] `coinfosim run-scenario synthetic-1 --mode smoke`
- [ ] Confirm report file is produced.
- [ ] Open the report and inspect visual layout.
- [ ] Confirm all seven subsets appear.
- [ ] Confirm all three classifiers appear.
- [ ] Confirm empirical test loss curves appear.
- [ ] Confirm no empirical train loss appears.
- [ ] Confirm no theoretical loss appears.
- [ ] Confirm no Bayes error appears.
- [ ] Confirm Monte Carlo replication counts are shown.
- [ ] Confirm stopping status is shown for each `n_per_class`.
- [ ] Confirm `N*` table appears.

---

## Task group 18 — Implementation notes and constraints

- [ ] Keep the implementation simple and explicit.
- [ ] Avoid premature abstraction for future phases.
- [ ] Do not implement all planned synthetic scenarios now.
- [ ] Do not implement dataset-anchored simulation now.
- [ ] Do not implement cost-aware optimization now.
- [ ] Do not introduce `channel_names` into the Sprint 1 model.
- [ ] Use labels `X1`, `X2`, `X3` for display.
- [ ] Use zero-based indices internally for subsets.
- [ ] Use one-based labels in reports.
- [ ] Keep random seed handling explicit and reproducible.
- [ ] Store enough metadata to reproduce a run.
- [ ] Prefer readable code over aggressive optimization in Sprint 1.

---

## Expected final deliverables

At the end of Sprint 1, the repository should include:

- [ ] New CoInfoSim simulation model class.
- [ ] New Gaussian sampler.
- [ ] New classifier registry.
- [ ] New Monte Carlo simulator.
- [ ] New standard-error stopping rule.
- [ ] New scenario definition for Synthetic Scenario 1.
- [ ] New result summaries for losses, rankings, and `N*`.
- [ ] New report generator for Synthetic Scenario 1.
- [ ] CLI command or script to run the scenario.
- [ ] Tests for the new components.
- [ ] Documentation explaining how to run Sprint 1.
- [ ] A generated report for Synthetic Scenario 1.

---

## Suggested implementation order

1. Add new module folders.
2. Implement `GaussianSimulationModel`.
3. Implement `Dataset` container.
4. Implement `GaussianClassConditionalSampler`.
5. Implement subset utilities.
6. Implement classifier registry.
7. Implement empirical test-loss metric.
8. Implement accumulator and summaries.
9. Implement standard-error stopping rule.
10. Implement mode configs.
11. Implement `CooperativeMonteCarloSimulator`.
12. Implement Synthetic Scenario 1.
13. Implement ranking and `N*` analysis.
14. Implement initial report generator.
15. Add CLI command or script entry point.
16. Add tests.
17. Run smoke mode.
18. Run fast mode.
19. Improve report layout.
20. Run full mode if computationally feasible.

