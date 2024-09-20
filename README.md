# CHERAB-Inversion

| | |
| ------- | ------- |
| CI/CD   | [![pre-commit.ci status][pre-commit-ci-badge]][pre-commit-ci] [![PyPI Publish][PyPI-publish-badge]][PyPi-publish]|
| Docs    | [![Documentation Status][Docs-badge]][Docs] |
| Package | [![PyPI - Version][PyPI-badge]][PyPI] [![Conda][Conda-badge]][Conda] [![PyPI - Python Version][Python-badge]][PyPI] |
| Meta    | [![DOI][DOI-badge]][DOI] [![License - BSD3][License-badge]][License] [![Pixi Badge][pixi-badge]][pixi-url] |

[pre-commit-ci-badge]: https://results.pre-commit.ci/badge/github/munechika-koyo/cherab_inversion/main.svg
[pre-commit-ci]: https://results.pre-commit.ci/latest/github/munechika-koyo/cherab_inversion/main
[PyPI-publish-badge]: https://img.shields.io/github/actions/workflow/status/munechika-koyo/cherab_inversion/deploy-pypi.yml?style=flat-square&label=PyPI%20Publish&logo=github
[PyPI-publish]: https://github.com/munechika-koyo/cherab_inversion/actions/workflows/deploy-pypi.yml
[Docs-badge]: https://readthedocs.org/projects/cherab-inversion/badge/?version=latest&style=flat-square
[Docs]: https://cherab-inversion.readthedocs.io/en/latest/?badge=latest
[PyPI-badge]: https://img.shields.io/pypi/v/cherab-inversion?label=PyPI&logo=pypi&logoColor=gold&style=flat-square
[PyPI]: https://pypi.org/project/cherab-inversion/
[Conda-badge]: https://img.shields.io/conda/vn/conda-forge/cherab-inversion?logo=conda-forge&style=flat-square
[Conda]: https://prefix.dev/channels/conda-forge/packages/cherab-inversion
[Python-badge]: https://img.shields.io/pypi/pyversions/cherab-inversion?logo=Python&logoColor=gold&style=flat-square
[DOI-badge]: https://zenodo.org/badge/DOI/10.5281/zenodo.10118752.svg
[DOI]: https://doi.org/10.5281/zenodo.10118752
[License-badge]: https://img.shields.io/github/license/munechika-koyo/cherab_inversion?style=flat-square
[License]: https://opensource.org/licenses/BSD-3-Clause
[pixi-badge]:https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json&style=flat-square
[pixi-url]: https://pixi.sh

----

CHERAB for Inversion, which is a package for the inversion technique of SVD, MFR, etc.
For more information, see the [documentation pages](https://cherab-inversion.readthedocs.io/).

## Quick Start

You can quickly try the inversion technique with [`Pixi`][pixi-url] tool:
```bash
git clone https://github.com/munechika-koyo/cherab_inversion
cd cherab_inversion
pixi run lab
```
Then, JupyterLab will be launched and you can try the inversion technique with the example notebook.

![JupyterLab window](/docs/source/_static/images/quickstart_jupyterlab.webp)

## Installation

You can install the package from conda-forge:
```bash
mamba install -c conda-forge cherab-inversion
```

The rest of the installation methods are described in the [documentation](https://cherab-inversion.readthedocs.io/en/stable/user/installation.html).
