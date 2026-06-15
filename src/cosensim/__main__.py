"""
Enable running CoSenSim as a module: python -m cosensim

This allows the package to be executed directly:
    python -m cosensim --help
    python -m cosensim run-simulation --params "[1,4,0.6]"
"""

from cosensim.cli import main

if __name__ == "__main__":
    main()
