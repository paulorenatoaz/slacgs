"""
Enable running SLACGS as a module: python -m slacgs

This allows the package to be executed directly:
    python -m slacgs --help
    python -m slacgs run-simulation --params "[1,4,0.6]"
"""

from slacgs.cli import main

if __name__ == "__main__":
    main()
