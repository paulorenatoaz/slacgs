# CoSenSim: A Simulator for Evaluating Cooperative Advantage in Sensor Networks

CoSenSim is a research simulator for evaluating cooperative advantage in sensor and measurement-channel systems. It is an academic evolution of the SLACGS project and extends SLACGS's experimental framing toward real-data‑anchored studies that ask when combinations of signals become more useful than isolated signals or smaller subsets.

## Short description

CoSenSim simulates and evaluates classification performance for different subsets of sensor or measurement channels. The simulator estimates class-conditional structures from real datasets, generates synthetic datasets anchored in those estimates, and compares cooperative patterns across isolated channels, pairs, and larger subsets under controlled sample-size regimes.

## Relationship to SLACGS

CoSenSim is a derivative and conceptual evolution of the SLACGS repository. It preserves SLACGS's core experimental perspective (margin-based and simple linear baselines) while prioritizing real-data anchoring and a systematic two-phase roadmap focused on operational costs and decision-making.

## Research motivation

Many practical sensing problems expose decisions about which signals or instruments to acquire and combine. Some measurements are inexpensive and abundant, while others are costly or slow. Our central question is: when does cooperation among sensor or measurement channels become advantageous? CoSenSim provides a controlled simulation environment to study this question using real-data anchors and synthetic experiments.

## Core empirical object

We evaluate Monte Carlo average classification loss for subsets of channels. Formally, for a subset of channels $A$, classifier $f$, and sample size per class $n$, the central empirical object is

$$
\\bar{L}_{A,f}(n)
$$

where $\\bar{L}_{A,f}(n)$ denotes the Monte Carlo average classification loss (e.g., average misclassification rate or other loss) when training classifier $f$ on $n$ labeled samples per class using channels in $A$.

The simulator compares $\\bar{L}_{A,f}(n)$ across isolated channels, pairs, and larger channel subsets to quantify cooperative gains.

## Methodological overview

1. Estimate class-conditional parameters from a chosen real dataset $D_{real}$.
2. Generate synthetic datasets $D_{synth}$ anchored to those estimates.
3. For each non-empty subset $A$ of selected channels and each classifier $f$, estimate $\\bar{L}_{A,f}(n)$ via Monte Carlo sampling across a range of $n$.
4. Compare cooperative patterns between $D_{real}$ and $D_{synth}$ and evaluate operational trade-offs when acquisition costs are introduced.

In early experiments we will typically restrict the signal space to $d=3$ selected channels so that there are $2^{3}-1 = 7$ non-empty subsets to compare (three isolated channels, three pairs, and the full 3-channel set).

## Two-phase roadmap

- Phase 1 — Real-data-anchored cooperative simulation
  - Focus on small, interpretable real datasets from the start.
  - Estimate class-conditional structures from $D_{real}$, generate $D_{synth}$ anchored to these estimates, then compare cooperative patterns.
  - Early experiments will often use $d=3$ selected channels for interpretability.

- Phase 2 — Operational decision with costs
  - Introduce acquisition and operational costs per channel.
  - Ask operational questions: which channel subset gives acceptable loss at lower cost? Can low-cost channels partially replace expensive ones? How many labeled samples are needed before cooperation becomes useful?

## Initial classifier plan

Initial classifier set (concise):

- Linear SVM — a margin-based linear baseline, preserving continuity with SLACGS.
- Logistic Regression — a lightweight probabilistic linear baseline.
- Gaussian Naive Bayes — a simple, interpretable baseline that assumes conditional independence across channels; useful for diagnosing whether cooperative gains arise from interaction structure beyond marginal evidence accumulation.

Expansion classifiers (short list): RBF SVM, k‑nearest neighbors, Random Forest. Future work may explore neural networks or gradient boosting methods.

## Initial real-data directions (candidate datasets)

Early empirical work will focus on small, interpretable studies with $d=3$ selected channels. Candidate directions include:

- Occupancy Detection: CO2, light, temperature.
- Water Potability: pH, conductivity, turbidity.
- Air Quality: field gas sensor responses vs. certified reference measurements.
- Hydraulic Systems Condition Monitoring: fault detection and predictive maintenance (future work).


## Installation (developer quick-start)

These are minimal local setup instructions for development. They assume Python 3.10+ and a virtual environment.

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install minimal dependencies:

```bash
pip install -r requirements.txt
```

3. Install the package in editable mode for development:

```bash
pip install -e .
```

## Basic usage (placeholder)

This repository is an early-stage scaffold. After Sprint 1, typical workflows will include:

- `src/cosensim` code to estimate class-conditional parameters from real data
- `experiments/` scripts to generate anchored synthetic datasets and compute $\\bar{L}_{A,f}(n)$
- Jupyter notebooks in `notebooks/` for exploration and figures

## Citation

Please cite this repository once experiments and results are published. A preliminary CITATION.cff is included as a placeholder.

## License

This repository currently includes a `LICENSE` placeholder. If you derived this repository from SLACGS, preserve and carry forward the original license; otherwise choose an appropriate open-source license (e.g., MIT, BSD, Apache-2.0).

## Contributing / development notes

- This is an academic research project under active development.
- Keep pull requests small and focused; refactorings should be proposed and discussed in issue threads before large changes.
- Preserve reproducibility: experiments should have manifests listing random seeds and environment details.


