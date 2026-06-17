"""Tests for the Sprint 1 HTML report generator (Checkpoint 6)."""

from coinfosim.scenarios.synthetic import make_synthetic_scenario_1
from coinfosim.simulation.config import MonteCarloConfig
from coinfosim.simulation.monte_carlo import CooperativeMonteCarloSimulator
from coinfosim.reports.sprint1 import generate_sprint1_report


def _tiny_config():
    return MonteCarloConfig(
        mode="smoke",
        sample_sizes=(2, 4),
        min_replications=3,
        max_replications=6,
        replication_batch_size=3,
        test_samples_per_class=50,
        ci_half_width_target=0.05,
        base_seed=0,
    )


def _run():
    scenario = make_synthetic_scenario_1()
    sim = CooperativeMonteCarloSimulator(scenario.model, _tiny_config())
    return scenario, sim.run(), sim.sampler


def test_report_file_created(tmp_path):
    scenario, result, sampler = _run()
    out = generate_sprint1_report(
        scenario, result, tmp_path, sampler=sampler, make_gif=False
    )
    assert out.exists()
    assert out.name == "synthetic_scenario_1_report.html"


def test_report_contains_required_sections(tmp_path):
    scenario, result, sampler = _run()
    out = generate_sprint1_report(
        scenario, result, tmp_path, sampler=sampler, make_gif=False
    )
    text = out.read_text(encoding="utf-8")

    # Title / scenario / question.
    assert "Synthetic Scenario 1" in text
    assert "Simple Complementary Channel" in text
    assert "complementary information" in text

    # Required sections.
    for section in (
        "Model parameters",
        "Run configuration",
        "Monte Carlo stopping",
        "Empirical test-loss curves",
        "Best subset by sample size",
        "Final ranking at largest n",
        "Cooperative advantage thresholds",
        "Monte Carlo uncertainty",
        "Synthetic data geometry",
        "1-D single-channel views",
        "2-D pairwise views",
        "3-D triple-wise views",
    ):
        assert section in text, f"missing section: {section}"

    # All seven subset labels appear.
    for label in ("X1", "X2", "X3", "X1+X2", "X1+X3", "X2+X3", "X1+X2+X3"):
        assert label in text

    # All three classifiers appear.
    for clf in ("Linear SVM", "Logistic Regression", "Gaussian Naive Bayes"):
        assert clf in text

    # Figures embedded as base64 PNGs.
    assert "data:image/png;base64," in text


def test_report_excludes_forbidden_metrics(tmp_path):
    scenario, result, sampler = _run()
    out = generate_sprint1_report(
        scenario, result, tmp_path, sampler=sampler, make_gif=False
    )
    text = out.read_text(encoding="utf-8").lower()

    # The exclusion notice must be present.
    assert "empirical train loss, theoretical loss, and bayes error are" in text

    # Forbidden metric phrases may only appear inside the two exclusion
    # notices (top and bottom). They must never appear as reported results.
    # "bayes" alone is allowed because the "Gaussian Naive Bayes" classifier
    # legitimately contains the word.
    assert text.count("bayes error") == 2  # only in the two exclusion notices
    assert text.count("theoretical loss") == 2  # only in the two exclusion notices
    assert text.count("train loss") == 2  # only in the two exclusion notices
    assert "bayes risk" not in text
    assert "empirical train" in text  # part of the exclusion statement


def test_report_includes_n_star_table(tmp_path):
    scenario, result, sampler = _run()
    out = generate_sprint1_report(
        scenario, result, tmp_path, sampler=sampler, make_gif=False
    )
    text = out.read_text(encoding="utf-8")
    assert "N*" in text
    assert "X1+X3 vs X1" in text
    assert "X1+X2+X3 vs X1+X2" in text
