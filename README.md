# CHERAB-Inversion

<!-- BEGIN-HEADER -->

|         |                                                                                                                       |
| ------- | --------------------------------------------------------------------------------------------------------------------- |
| CI/CD   | [![CI][ci-badge]][ci] [![PyPI Publish][PyPI-publish-badge]][PyPi-publish] [![codecov][codecov-badge]][codecov]        |
| Docs    | [![Read the Docs (version)][Docs-dev-badge]][Docs-dev] [![Read the Docs (version)][Docs-release-badge]][Docs-release] |
| Package | [![PyPI - Version][pypi-badge]][pypi] [![Conda][conda-badge]][conda] [![PyPI - Python Version][python-badge]][pypi]   |
| Meta    | [![DOI][DOI-badge]][DOI] [![License - BSD-3-Clause][license-badge]][license] [![Pixi Badge][pixi-badge]][pixi-url]    |

[ci]: https://github.com/munechika-koyo/cherab_inversion/actions/workflows/ci.yaml
[ci-badge]: https://img.shields.io/github/actions/workflow/status/munechika-koyo/cherab_inversion/ci.yaml?style=flat-square&logo=GitHub&label=CI
[codecov]: https://codecov.io/github/munechika-koyo/cherab_inversion
[codecov-badge]: https://img.shields.io/codecov/c/github/munechika-koyo/cherab_inversion?token=05LZGWUUXA&style=flat-square&logo=codecov
[conda]: https://prefix.dev/channels/conda-forge/packages/cherab-inversion
[conda-badge]: https://img.shields.io/conda/vn/conda-forge/cherab-inversion?logo=conda-forge&style=flat-square
[Docs-dev-badge]: https://img.shields.io/readthedocs/cherab-inversion/latest?style=flat-square&logo=readthedocs&label=dev
[Docs-dev]: https://cherab-inversion.readthedocs.io/en/latest/?badge=latest
[Docs-release-badge]: https://img.shields.io/readthedocs/cherab-inversion/stable?style=flat-square&logo=readthedocs&label=release
[Docs-release]: https://cherab-inversion.readthedocs.io/en/stable/?badge=stable
[DOI-badge]: https://zenodo.org/badge/DOI/10.5281/zenodo.10118752.svg
[DOI]: https://doi.org/10.5281/zenodo.10118752
[license]: https://opensource.org/licenses/BSD-3-Clause
[license-badge]: https://img.shields.io/github/license/munechika-koyo/cherab_inversion?style=flat-square
[pixi-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json&style=flat-square
[pixi-url]: https://pixi.sh
[pypi]: https://pypi.org/project/cherab-inversion/
[pypi-badge]: https://img.shields.io/pypi/v/cherab-inversion?label=PyPI&logo=pypi&logoColor=gold&style=flat-square
[pypi-publish]: https://github.com/munechika-koyo/cherab_inversion/actions/workflows/deploy-pypi.yml
[pypi-publish-badge]: https://img.shields.io/github/actions/workflow/status/munechika-koyo/cherab_inversion/deploy-pypi.yml?event=release&style=flat-square&logo=github&label=PyPI%20Publish
[python-badge]: https://img.shields.io/pypi/pyversions/cherab-inversion?logo=Python&logoColor=gold&style=flat-square

---

[`cherab`](https://github.com/cherab/core) add-on module that provides inversion techniques including SVD, MFR, and more.

<!-- END-HEADER -->

## 🚀 Quick Start

Get started with the inversion technique using the [`pixi`][pixi-url] tool:

```bash
git clone https://github.com/munechika-koyo/cherab_inversion
cd cherab_inversion
pixi run lab
```

JupyterLab will then launch, allowing you to explore the example notebooks.

![JupyterLab window](/docs/source/_static/images/quickstart_jupyterlab.webp)

## 🌐 Install

**With `pip`:**

```
pip install cherab-inversion
```

**With `conda`:**

```
conda install -c conda-forge cherab-inversion
```

## 📝 Documentation

See the [official documentation](https://cherab-inversion.readthedocs.io/) to learn more.
