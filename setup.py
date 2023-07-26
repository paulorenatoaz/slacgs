from setuptools import setup, find_packages

setup(
    name='slacgs',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # List any dependencies your package requires
        'numpy>=1.19.5',
        'matplotlib>=3.3.3',
        'shapely>=2.0.1',
        'ipython>=7.34.0',
        'scikit-learn>=1.2.2',
        'scipy>=1.10.1',
        'pygsheets>=2.0.6',
        'google-api-python-client>=2.83.0',
        'cryptography>=41.0.2',
        'Pillow>=8.1.0',
        'pandas>=1.5.3',
        'setuptools>=51.3.3'

    ],
    author='Paulo Azevedo',
    author_email='paulorenatoaz@dcc.ufrj.br',
    description='A Simulator for Loss Analysis of Linear Classifiers on Gaussian Samples.',
    url='https://github.com/paulorenatoaz/slacgs',
)
