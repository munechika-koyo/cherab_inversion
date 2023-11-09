# CHERAB-Inversion

[![PyPI](https://img.shields.io/pypi/v/cherab-inversion?label=PyPI&logo=PyPI)](https://pypi.org/project/cherab-inversion/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cherab-inversion?logo=Python)

[![DOI](https://zenodo.org/badge/239309930.svg)](https://zenodo.org/badge/latestdoi/239309930)
[![GitHub](https://img.shields.io/github/license/munechika-koyo/cherab_phix)](https://opensource.org/licenses/BSD-3-Clause)

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/munechika-koyo/cherab_phix/master.svg)](https://results.pre-commit.ci/latest/github/munechika-koyo/cherab_phix/master)
[![Documentation Status](https://readthedocs.org/projects/cherab-inversion/badge/?version=latest)](https://cherab-inversion.readthedocs.io/en/latest/?badge=latest)
[![PyPI Publish](https://github.com/munechika-koyo/cherab_phix/actions/workflows/python-publish.yml/badge.svg)](https://github.com/munechika-koyo/cherab_phix/actions/workflows/python-publish.yml)

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Docstring formatter: docformatter](https://img.shields.io/badge/%20formatter-docformatter-fedcba.svg)](https://github.com/PyCQA/docformatter)
[![Docstring style: numpy](https://img.shields.io/badge/%20style-numpy-459db9.svg)](https://numpydoc.readthedocs.io/en/latest/format.html)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)


CHERAB for Inversion, which is a package for the inversion technique of SVD, MFR, etc.
For more information, see the [documentation pages](https://cherab-inversion.readthedocs.io/).

Quick installation
-------------------
`pip` command enables us to install `cherab-inversion` from [PyPI](https://pypi.org/project/cherab-inversion/) repository.

```Shell
python -m pip install cherab-inversion
```

For Developpers
---
If you would like to develop `cherab-inversion`, it is much easier to create a conda environment after cloning repository.
```Shell
conda env create -f environment.yaml
conda activate cherab-inv-dev
python dev.py build
python dev.py install
```
Please follow the [development procedure](https://cherab-inversion.readthedocs.io/en/development/user/contribution.html).
