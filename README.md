# CoInfoSim: A Simulator for Cooperative Classification from Multiple Information Channels

CoInfoSim is a research simulator for evaluating **cooperative advantage among information channels** in supervised classification tasks. It studies when a *subset* of channels provides a measurable advantage over isolated channels, redundant pairs, or simpler subsets, and how many labeled samples are needed before that advantage appears.

CoInfoSim is a conceptual evolution of the earlier **SLACGS** and **CoSenSim** lines of work. It preserves their incremental Monte Carlo protocol, reproducible sample generation, adaptive repetition logic, and scenario-based reporting structure, while reformulating the scientific object from sensor-network dimensionality to multi-channel classification.

> **Status:** Early-stage research project under active development. This repository currently describes the CoInfoSim research direction and prepares the codebase for it. The new samplers, Monte Carlo loop, dataset-anchored pipeline, cost-aware optimization, and automated report engine described below are **planned architecture**, not yet implemented. The functional code inherited from SLACGS/CoSenSim remains available under the `coinfosim` package while the reformulation proceeds.

## Core research question

> When does cooperation among information channels improve supervised classification?

Rather than asking only whether a single channel is informative, CoInfoSim evaluates when combining channels yields lower classification loss, when an apparently useful channel is redundant, and when a weaker channel is valuable because it complements stronger ones.

## Modeling framework

Let

$$
X = (X_1, \ldots, X_d)
$$

be a **standardized** input vector, where each component $X_j$ is an **information channel**, and let $Y \in \{1, \ldots, K\}$ be the class label. An information channel may be a sensor reading, a sensor-derived variable, a laboratory or contextual measurement, an engineered feature, or any other standardized observable variable available to a classifier. Multi-channel sensing systems are an important *motivating* application, but the formulation is intentionally broader.

In the parametric Gaussian mode, each class is described by its own class-conditional distribution:

$$
X_c \sim \mathcal{N}(\mu_c, \Sigma_c), \qquad c = 1, \ldots, K.
$$

The simulator receives the class centers and covariance matrices directly, as

$$
\{(\mu_c, \Sigma_c)\}_{c=1}^{K}.
$$

Unlike the previous restricted formulation, CoInfoSim does **not** require covariance matrices to be equal across classes; it allows

$$
\Sigma_0 \neq \Sigma_1,
$$

so differences in location, dispersion, correlation, or distributional shape may all affect classification.

A **channel subset** is denoted $A \subseteq \{1, \ldots, d\}$, and the classifier observes only the restricted vector $X_A$. For $d = 3$, the simulator evaluates all $2^3 - 1 = 7$ non-empty subsets — three isolated channels, three pairs, and the full three-channel set.

### Balanced sampling and standardization

The initial protocol uses **class-balanced** sampling: $n$ always denotes the number of labeled training samples *per class*. Synthetic simulations generate $n$ samples per class; real-data simulations draw balanced subsets from the training reservoir. When sampling without replacement from real data, the feasible maximum $n$ is limited by the smallest class in the training reservoir. CoInfoSim operates on **standardized** channels by default, which matters for scale-sensitive classifiers such as Linear SVM and Logistic Regression. In dataset-anchored mode, the estimated parameters $\hat{\mu}_c$ and $\hat{\Sigma}_c$ are computed from standardized data.

## Core empirical object

The central quantity is the **Monte Carlo average classification loss**

$$
\overline{L}_{A,f}(n),
$$

the average loss for channel subset $A$, classifier $f$, and $n$ labeled training samples per class. For replication $r$,

$$
\widehat{L}_{A,f,r}(n) = \frac{1}{m} \sum_{i=1}^{m} \mathbf{1}\{\widehat{Y}^{(A,f)}_{i,r}(n) \neq Y_i\},
\qquad
\overline{L}_{A,f}(n) = \frac{1}{R_n} \sum_{r=1}^{R_n} \widehat{L}_{A,f,r}(n),
$$

where the number of repetitions $R_n$ may depend on $n$ (more repetitions for smaller samples, fewer once estimates stabilize).

Two derived quantities are central. The **best subset** for a classifier is

$$
A_f^\star(n) = \arg\min_A \overline{L}_{A,f}(n),
$$

and the **cooperative advantage threshold** between subsets $A$ and $B$ is

$$
N^*(A, B; f) = \min\{n : \overline{L}_{B,f}(n) < \overline{L}_{A,f}(n)\},
$$

the smallest sample size per class at which subset $B$ first achieves lower average loss than subset $A$.

## Initial classifiers

The initial classifier set is intentionally small and interpretable:

- **Linear SVM** — preserves continuity with the previous framework and provides a margin-based linear baseline.
- **Logistic Regression** — a lightweight probabilistic linear baseline.
- **Gaussian Naive Bayes** — a probabilistic baseline whose conditional-independence assumption helps distinguish cooperative gains that arise from marginal evidence accumulation from gains that depend on multivariate dependence.

Future classifier sets may include RBF SVM, kNN, Random Forest, gradient boosting, or neural networks, added once the Monte Carlo and reporting protocols are stable.

## Research plan

CoInfoSim is organized into three phases.

### Phase 1 — Idealized Synthetic Multi-Channel Scenarios

Controlled experiments using manually specified Gaussian simulation models, each defined by class centers and covariance matrices. The initial focus is binary classification with three standardized channels. This phase studies additive gains from weak but complementary channels, redundancy among strong channels, channel-subset ranking as $n$ grows, cooperative advantage thresholds $N^*$, and differences among the initial classifiers.

### Phase 2 — Dataset-Anchored Multi-Channel Simulation

Real datasets are used to build dataset-anchored scenarios. For each dataset the simulator produces a real-data simulation report (balanced sampling from the standardized reservoir) and a Gaussian-anchored synthetic simulation report (parameters estimated on the standardized data). The dataset-anchored scenario report compares

$$
\overline{L}^{\,real}_{A,f}(n) \quad \text{versus} \quad \overline{L}^{\,synth}_{A,f}(n),
$$

assessing whether the Gaussian anchored model preserves the cooperative patterns of the real dataset.

### Phase 3 — Cost-Aware Channel Selection

A generic **channel cost** $C_X(A)$ and a **label/reference cost** $C_Y(n)$ are introduced, with total

$$
C(A, n) = C_X(A) + C_Y(n).
$$

Channel cost is treated generically and may include acquisition, deployment, calibration, maintenance, computation, latency, energy, or logistical burden. This supports constrained decisions such as minimizing $C(A,n)$ subject to $\overline{L}_{A,f}(n) \le L_{\max}$, or penalized objectives $\overline{L}_{A,f}(n) + \lambda C_X(A) + \gamma C_Y(n)$.

## Key concepts and vocabulary

| Term | Meaning |
|---|---|
| Information channel | A component $X_j$ of the input vector $X$ |
| Channel subset $A$ | A subset $A \subseteq \{1,\ldots,d\}$ observed by the classifier |
| Simulation model | The object holding the class-conditional parameters $\{(\mu_c, \Sigma_c)\}$ |
| Scenario | One or more simulation models grouped around an experimental question |
| Scenario grid | A scenario whose models come from a parameter grid |
| Simulation report | Report for one simulation model |
| Scenario report | Aggregated report for one scenario |
| Dataset report | Report describing a real dataset used in dataset-anchored experiments |

## Planned reporting structure

The reporting system is **planned architecture**. It is designed to remain generic: automated reports display the parameters explicitly defined in each simulation model and scenario grid together with generic derived metrics, while scenario-specific interpretive quantities live in accompanying analyses rather than as required report fields. The layered structure is:

1. **Project index** — published entry point listing scenario, dataset, and simulation reports, result files, generation date, and software/commit version.
2. **Dataset report** — dataset name and source, target $Y$, class distribution, candidate and selected channels, preprocessing and standardization, training reservoir/test set, and visual diagnostics of standardized data.
3. **Simulation report** — model id and source, class centers and covariance matrices, evaluated subsets, classifiers, values of $n$, repetitions/convergence, loss curves $\overline{L}_{A,f}(n)$, subset rankings, thresholds $N^*$, and data-geometry visualizations.
4. **Scenario report** — the scenario question, a table of simulation models, links to simulation reports, aggregate loss curves, threshold summaries, best-subset maps, scenario-grid heatmaps, and animations of geometry across the scenario.
5. **Dataset-anchored scenario report** — links to the dataset report and both simulation reports, real-versus-synthetic loss curves and rankings, threshold comparisons, and real–synthetic discrepancies.

Planned visual elements include synthetic dataset visualizations, real-data visual diagnostics, GIF animations of sample-size growth, panels/animations of scenario-parameter variation, loss curves $\overline{L}_{A,f}(n)$, channel-subset rankings, cooperative advantage thresholds $N^*$, best-subset maps, and scenario-grid heatmaps, with links connecting scenario reports to their simulation and dataset reports.

## Real-world motivating cases

The dataset-anchored phase is motivated by small, interpretable real-data studies with a few selected channels:

- **Occupancy detection** — channels such as CO₂, light, and temperature.
- **Water potability** — channels such as pH, conductivity, and turbidity.
- **Air quality** — field sensor responses compared with certified reference measurements.
- **Hydraulic systems / condition monitoring** — multiple physical or operational measurements for fault classification (future work).

These cases motivate channel-subset evaluation and the separation between channel cost $C_X(A)$ and label/reference cost $C_Y(n)$. Dataset pipelines are **not** implemented in this repository yet.

## Relationship to SLACGS and CoSenSim

CoInfoSim evolves from SLACGS, which studied cooperative gains in synthetic Gaussian settings and was organized primarily around dimensionality comparisons (lower- versus higher-dimensional models). CoInfoSim generalizes that idea: the object of comparison is no longer dimension $d$ but the channel subset $A \subseteq \{1,\ldots,d\}$, and the framing moves from sensor networks to multi-channel classification. The intermediate CoSenSim stage carried this toward real-data-anchored sensor studies; CoInfoSim completes the generalization to information channels. The reproducible Monte Carlo machinery, nested sample growth, adaptive repetition, and scenario-level reporting from SLACGS are retained.

## Repository structure

```
docs/          # Documentation, design notes, and planned-architecture descriptions
data/          # Dataset READMEs and pointers (no large data committed)
experiments/   # Experiment manifests and scripts (planned)
src/coinfosim/ # Main package (inherited functional code; reformulation in progress)
tests/         # Smoke and unit tests
```

The package namespace and CLI are now `coinfosim`. Renaming the GitHub repository itself from `cosensim` to `coinfosim` is an external operation (see *Follow-up* below).

## Installation (developer quick-start)

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Basic usage (placeholder)

The CLI entry point is `coinfosim`:

```bash
coinfosim --help
```

The simulator API and command set are being reformulated for the CoInfoSim research direction; current commands reflect the inherited implementation and will evolve as the phases above are implemented.

## Citation

If you use CoInfoSim in published research, please cite this repository. See [CITATION.cff](CITATION.cff).

## License

CoInfoSim is distributed under the GNU General Public License v3.0 (GPL-3.0), inherited from the SLACGS project from which it is derived. See the [LICENSE](LICENSE) file for the full text.

## Contributing / development notes

- This is an academic research project under active development.
- Keep pull requests small and focused; propose larger refactors in an issue first.
- Preserve reproducibility: experiments should record random seeds and environment details.

## Follow-up

- The GitHub repository is still named `cosensim`; renaming it to `coinfosim` is an external GitHub operation. After renaming, update the `origin` remote URL locally.
- The new simulator logic (samplers, dataset-anchored pipeline, Gaussian-anchored simulation, reformulated Monte Carlo loop, cost-aware optimization, automated HTML reports, and scenario-grid execution) is planned and not implemented in this task.


