(intro/get-started)=

# 🚀 Get Started

This page gives a quick overview of how to get started with `cherab-inversion`, including installation instructions and a simple example script.

## Installation

[![PyPI - Version][pypi-badge]][pypi]
[![Conda][conda-badge]][conda]

[pypi]: https://pypi.org/project/cherab-inversion/
[pypi-badge]: https://img.shields.io/pypi/v/cherab-inversion?label=PyPI&logo=pypi&logoColor=gold&style=flat-square
[conda]: https://prefix.dev/channels/conda-forge/packages/cherab-inversion
[conda-badge]: https://img.shields.io/conda/vn/conda-forge/cherab-inversion?logo=conda-forge&style=flat-square

`cherab-inversion` can be installed by many package managers.
Explore the various methods below to install `cherab-inversion` using your preferred package manager.

::::{md-tab-set}

:::{md-tab-item} conda

```bash
conda install -c conda-forge cherab-inversion
```

:::
:::{md-tab-item} pixi

```bash
pixi add cherab-inversion
```

:::
:::{md-tab-item} uv

It is necessary to install the [SuiteSparse](https://people.engr.tamu.edu/davis/suitesparse.html) library manually.
On MacOS:

```bash
brew install suite-sparse
```

On Debian/Ubuntu systems:

```bash
sudo apt-get install libsuitesparse-dev
```

On Arch Linux:

```bash
sudo pacman -S suitesparse
```

Then, you can install `cherab-inversion` using the following command:

```bash
uv add cherab-inversion
```

:::
:::{md-tab-item} pip

It is necessary to install the [SuiteSparse](https://people.engr.tamu.edu/davis/suitesparse.html) library manually.
On MacOS:

```bash
brew install suite-sparse
```

On Debian/Ubuntu systems:

```bash
sudo apt-get install libsuitesparse-dev
```

On Arch Linux:

```bash
sudo pacman -S suitesparse
```

Then, you can install `cherab-inversion` using the following command:

```bash
pip install cherab-inversion
```

:::
::::

## Quick Start

Here's a minimal example to get you started:

```python
from cherab.inversion import SVD, Lcurve

# Your example code here
```

For more detailed examples, check the [Examples](examples) section.

## Notebooks

We also have a collection of Jupyter notebooks that demonstrate various features of `cherab-inversion`, which are same as the ones in the [Examples](examples) section. You can explore these notebooks to see how to use `cherab-inversion` in practice.
To get started with the notebooks locally, you can clone the repository and launch JupyterLab.
[`pixi`](https://pixi.sh) makes it easy to set up the environment and launch JupyterLab with all dependencies installed.

```bash
git clone https://github.com/munechika-koyo/cherab_inversion
cd cherab_inversion
pixi run lab
```

## Citations

If you use `cherab-inversion` in your research, please cite the <doi:10.5281/zenodo.10118752>.
