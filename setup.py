from setuptools import setup, find_packages

__version__ = '0.0.1'
URL = None
install_requires = [
    "prettytable",
    "gensim",
    "tensorboard",
    "scikit-learn",
    "tabulate",
    "matplotlib",
    "jupyter",
    'jupyterlab',
]

setup(
    name='dhgas',
    version=__version__,
    description='dhgas',
    author='mnlab',
    url=URL,
    python_requires='>=3.6',
    install_requires=install_requires,
    packages=find_packages(),
    include_package_data=True,
)

# Require
# python == 3.6
# pytorch == 1.8.2+cu102
# torch-geometric == 2.0.3+cu102

# Also tested ok for:
# python == 3.8
# conda install pytorch==1.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
# pip install torch-scatter torch-sparse torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu113.html