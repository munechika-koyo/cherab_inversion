:orphan:

.. _installation:

============
Installation
============


via Conda
=========
`conda` (or `mamba`, a faster `conda` alternative) is the most recommended way to install
`cherab-inversion`::

    mamba install -c conda-forge cherab-inversion


via Pip
=======
If you want use ``pip``, you need to install `suitesparse` library for `scikit-sparse` package
firstly.

.. tab-set::

    .. tab-item:: Linux

        For Debian/Ubuntu users, you can do::

            sudo apt install libsuitesparse-dev
            python -m pip install cherab-inversion

    .. tab-item:: macOS

        The easiest way to install `suitesparse` on macOS is to use `Homebrew <https://brew.sh/>`__
        ::

            brew install suite-sparse
            python -m pip install cherab-inversion


For Developers
==============
If you want to install from source in order to work on `cherab-inversion` itself, first clone the
`cherab-inversion` repository::

    git clone https://github.com/munechika-koyo/cherab_inversion
    cd cherab_inversion

Then run `postinstall-e` command in `dev` environment by `Pixi`_::

    pixi run -e dev postinstall-e

This will install the package in editable mode with all the necessary dependencies.
If you want to run the tests, you can do so with::

    pixi run -e dev test

The rest of useful commands for development are shown in the :ref:`contribution` section.
