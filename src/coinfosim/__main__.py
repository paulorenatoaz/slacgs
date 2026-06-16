"""
Enable running CoInfoSim as a module: python -m coinfosim

This allows the package to be executed directly:
    python -m coinfosim --help
    python -m coinfosim run-simulation --params "[1,4,0.6]"
"""

from coinfosim.cli import main

if __name__ == "__main__":
    main()
