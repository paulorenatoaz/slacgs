"""Smoke tests for the CoInfoSim package identity and CLI."""

import subprocess
import sys


def test_import_coinfosim():
    import coinfosim

    assert coinfosim is not None


def test_version_present():
    import coinfosim

    assert isinstance(coinfosim.__version__, str)
    assert coinfosim.__version__


def test_public_api_symbols():
    import coinfosim

    for symbol in ("Model", "Simulator", "Report", "load_config"):
        assert hasattr(coinfosim, symbol), f"missing public symbol: {symbol}"


def test_cli_help_runs():
    """`python -m coinfosim --help` should exit cleanly."""
    result = subprocess.run(
        [sys.executable, "-m", "coinfosim", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "coinfosim" in (result.stdout + result.stderr).lower()
