:orphan:

.. _installation:

============
Installation
============


Installing via Conda
====================
`conda` (or `mamba`, a faster `conda` alternative) is the most recommended way to install
CHERAB-Inversion.

.. prompt:: bash

    conda install -c conda-forge cherab-inversion



Installing with Pip
===================
If you want install ``pip``, you need to install `suitesparse` library for `scikit-sparse` package
firstly.

.. prompt:: bash

    # Linux (Debian/Ubuntu)
    sudo apt-get install libsuitesparse-dev

    # macOS
    brew install suite-sparse

Then, you can install CHERAB-Inversion by ``pip``:

.. prompt:: bash

    python -m pip install cherab-inversion



Installing for Developper
==========================
If you plan to make any modifications to do any development work on CHERAB-Inversion,
and want to be able to edit the source code without having to run the setup script again
to have your changes take effect, you can install CHERAB-Inversion on editable mode.

Manually downloading source
---------------------------
Before install the package, it is required to download the source code from github repository.
The source codes can be cloned from the GitHub reporepository with the command:

.. prompt:: bash

    git clone https://github.com/munechika-koyo/cherab_inversion

The repository will be cloned inside a new subdirectory named as ``cherab_inversion``.

Building and Installing
-----------------------
Firstly, you need to install dependencies.
The easiest way is to create a conda development environment:

.. prompt:: bash

    conda env create -f environment.yaml
    conda activate cherab-inv-dev

you need to build this package using the ``dev.py`` CLI:

.. prompt:: bash

    python dev.py build

This command enables us to compile cython codes with meson build-tool and put built shared object
(``.so``) files into the source tree.
This interface has some options, allowing you to perform all regular development-related tasks
(building, building docs, formatting codes, etc.).
Here we document a few of the most commonly used options; run ``python dev.py --help`` or ``--help``
on each of the subcommands for more details.

Additionally, to make a path to this package and register it as a `cherab` namespace package,
run the following command:

.. prompt:: bash

    python dev.py install

In this CLI, the ``setuptools`` shall install it into the ``**/site-packages/`` directory
as a namespace package with the develop (editable) mode.

Alternatively, you can use the ``meson-python``'s editable mode to install this package:

.. prompt:: bash

    python -m pip install --no-build-isolation --editable .

This editable install enable us to make the editted codes effective without re-installation.
Please see `meson-python documentation <https://meson-python.readthedocs.io/en/latest/how-to-guides/editable-installs.html>`_
for more details.
