"""Smoke tests for the CoSenSim package identity and CLI."""

import subprocess
import sys


def test_import_cosensim():
    import cosensim

    assert cosensim is not None


def test_version_present():
    import cosensim

    assert isinstance(cosensim.__version__, str)
    assert cosensim.__version__


def test_public_api_symbols():
    import cosensim

    for symbol in ("Model", "Simulator", "Report", "load_config"):
        assert hasattr(cosensim, symbol), f"missing public symbol: {symbol}"


def test_cli_help_runs():
    """`python -m cosensim --help` should exit cleanly."""
    result = subprocess.run(
        [sys.executable, "-m", "cosensim", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "cosensim" in (result.stdout + result.stderr).lower()
