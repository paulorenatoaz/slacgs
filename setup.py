from setuptools import setup, find_packages

# Read the contents of the README file
with open('README.md', 'r', encoding='utf-8') as file:
    long_description = file.read()


setup(
    name='slacgs',
    version='0.2.0',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    python_requires='>=3.8,<4.0',
    install_requires=[
        # Core scientific stack
        'numpy>=1.19.5',
        'matplotlib>=3.3.3',
        'plotly>=5.0.0',
        'shapely>=2.0.1',
        'ipython>=7.34.0',
        'scikit-learn>=1.2.2',
        'scipy>=1.10.1',
        'Pillow>=8.1.0',
        'pandas>=1.5.3',
        'tabulate>=0.9.0',
        'reportlab>=4.0.4',
        
        # CLI and configuration
        'tomli>=2.0.0;python_version<"3.11"',  # TOML support (built-in in 3.11+)
        'platformdirs>=3.0.0',                  # Cross-platform config paths
        'typer[all]>=0.9.0',                   # CLI framework with rich
    ],
    extras_require={
        'legacy': [
            # Deprecated Google Drive/Sheets integration
            'pygsheets>=2.0.6',
            'google-api-python-client>=2.83.0',
            'cryptography>=41.0.2',
        ],
        'dev': [
            # Development and testing tools
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=23.0.0',
            'ruff>=0.1.0',
            'mypy>=1.0.0',
        ],
    },
    author='Paulo Azevedo',
    author_email='paulorenatoaz@dcc.ufrj.br',
    description='A Simulator for Loss Analysis of Classifiers on Gaussian Samples.',
    long_description=long_description,
    long_description_content_type='text/markdown',  # Use 'text/x-rst' for reStructuredText
    url='https://github.com/paulorenatoaz/slacgs',
    entry_points={
        'console_scripts': [
            'slacgs=slacgs.cli:main',
        ],
    },
)
