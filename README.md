# CHERAB-Inversion

[![PyPI](https://img.shields.io/pypi/v/cherab-inversion?label=PyPI&logo=PyPI)](https://pypi.org/project/cherab-inversion/)
[![Conda](https://img.shields.io/conda/v/conda-forge/cherab-inversion?logo=anaconda)](https://anaconda.org/conda-forge/cherab-inversion)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cherab-inversion?logo=Python)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10118752.svg)](https://doi.org/10.5281/zenodo.10118752)
[![GitHub](https://img.shields.io/github/license/munechika-koyo/cherab_inversion)](https://opensource.org/licenses/BSD-3-Clause)

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/munechika-koyo/cherab_inversion/main.svg)](https://results.pre-commit.ci/latest/github/munechika-koyo/cherab_inversion/main)
[![Documentation Status](https://readthedocs.org/projects/cherab-inversion/badge/?version=latest)](https://cherab-inversion.readthedocs.io/en/latest/?badge=latest)
[![PyPI Publish](https://github.com/munechika-koyo/cherab_inversion/actions/workflows/deploy-pypi.yml/badge.svg)](https://github.com/munechika-koyo/cherab_inversion/actions/workflows/deploy-pypi.yml)


CHERAB for Inversion, which is a package for the inversion technique of SVD, MFR, etc.
For more information, see the [documentation pages](https://cherab-inversion.readthedocs.io/).

Quick Start
-----------
You can quickly try the inversion technique with `Pixi` tool:
```bash
git clone https://github.com/munechika-koyo/cherab_inversion
cd cherab_inversion
pixi run lab
```
Then, JupyterLab will be launched and you can try the inversion technique with the example notebook.

![JupyterLab window](/docs/source/_static/images/quickstart_jupyterlab.webp)

Installation
------------
You can install the package from conda-forge:
```bash
mamba install -c conda-forge cherab-inversion
```

The rest of the installation methods are described in the [documentation](https://cherab-inversion.readthedocs.io/en/latest/installation.html).
