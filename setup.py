from setuptools import setup, find_packages

# Read the contents of the README file
with open('README.md', 'r', encoding='utf-8') as file:
    long_description = file.read()


setup(
    name='slacgs',
    version='0.1.8',
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
        'setuptools>=51.3.3',
        'tabulate>=0.9.0',
        'reportlab>=4.0.4'

    ],
    author='Paulo Azevedo',
    author_email='paulorenatoaz@dcc.ufrj.br',
    description='A Simulator for Loss Analysis of Classifiers on Gaussian Samples.',
    long_description=long_description,
    long_description_content_type='text/markdown',  # Use 'text/x-rst' for reStructuredText
    url='https://github.com/paulorenatoaz/slacgs',
)
