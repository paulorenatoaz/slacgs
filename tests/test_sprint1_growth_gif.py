"""Tests for the Sprint 1 sample-growth GIF (report upgrade CP6)."""

from PIL import Image

from coinfosim.reports.sprint1 import generate_sprint1_report
from coinfosim.reports.visualizations import generate_growth_gif
from coinfosim.scenarios.synthetic import make_synthetic_scenario_1
from coinfosim.simulation.config import MonteCarloConfig
from coinfosim.simulation.monte_carlo import CooperativeMonteCarloSimulator


def _tiny_config():
    return MonteCarloConfig(
        mode="smoke",
        sample_sizes=(2, 4, 8),
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


def test_growth_gif_created_with_expected_frames(tmp_path):
    scenario, result, sampler = _run()
    out = tmp_path / "growth.gif"
    path = generate_growth_gif(scenario.model, sampler, result, out)

    assert path.exists()
    img = Image.open(path)
    # One frame per configured sample size.
    assert img.n_frames == len(result.sample_sizes)
    # The image is a valid, non-trivial GIF.
    assert img.format == "GIF"
    assert img.size[0] > 0 and img.size[1] > 0


def test_growth_gif_is_deterministic(tmp_path):
    scenario, result, sampler = _run()
    a = generate_growth_gif(scenario.model, sampler, result, tmp_path / "a.gif")
    b = generate_growth_gif(scenario.model, sampler, result, tmp_path / "b.gif")
    assert a.read_bytes() == b.read_bytes()


def test_report_embeds_gif(tmp_path):
    scenario, result, sampler = _run()
    out = generate_sprint1_report(
        scenario, result, tmp_path, sampler=sampler, make_gif=True
    )
    text = out.read_text(encoding="utf-8")

    # The GIF artifact is written next to the report.
    assert (tmp_path / "synthetic_scenario_1_growth.gif").exists()
    # The animation section and an embedded (or linked) GIF are present.
    assert "Sample-growth animation" in text
    assert "data:image/gif;base64," in text


def test_report_can_skip_gif(tmp_path):
    scenario, result, sampler = _run()
    out = generate_sprint1_report(
        scenario, result, tmp_path, sampler=sampler, make_gif=False
    )
    assert not (tmp_path / "synthetic_scenario_1_growth.gif").exists()
    text = out.read_text(encoding="utf-8")
    assert "data:image/gif;base64," not in text
