from setuptools import setup, find_packages

setup(
    name='slacgs',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # List any dependencies your package requires
        'numpy~=1.24.2',
        'matplotlib~=3.7.1',
        'shapely~=2.0.1',
        'ipython==8.13.0',
        'scikit-learn==1.3.0',
        'scipy~=1.10.1',
        'pygsheets~=2.0.6',
    ],
    author='Paulo Azevedo',
    author_email='paulorenatoaz@dcc.ufrj.br',
    description='A Simulator for Loss Analysis of Linear Classifiers on Gaussian Samples.',
    url='https://github.com/paulorenatoaz/slacgs',
)
