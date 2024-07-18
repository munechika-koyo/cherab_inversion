:orphan:

.. _contribution:

============
Contribution
============

Contributions from the community are welcome.
Interested collaborators can make contact with Koyo Munechika (Core Developer) from the
`Source Repository`_.

.. include:: ../../../AUTHORS.md


For Developers
--------------
If you would like to develop this package, please fork the `GitHub Repository`_ at first, and follow
the :ref:`installation procedure<installation>`.

`Pixi`_ is required for several development tasks, such as building the documentation and running
the tests.
Please install it by following the `Pixi Installation Guide<https://pixi.sh/latest#installation>`
in advance.

.. tab-set::

    .. tab-item:: test

        To run the tests, you can do so with::

            pixi run -e dev test

    .. tab-item:: docs

        To build the documentation, you can do so with::

            pixi run -e dev docs

        The documentation will be built in the ``docs/build/html`` directory.

        If you want to clean the documentation, you can do so with::

            pixi run -e dev clean-docs

    .. tab-item:: format

        The easiest way to lint/format the code is to use the `pre-commit<https://pre-commit.com>`
        framework.
        If you have never run the `pre-commit` before, you should install hooks by::

            pixi run -e dev pre-commit install

        Then, you can run the `pre-commit` by::

            pixi run -e dev pre-commit run --all-files

        After installing the `pre-commit`, it is automatically run before each commit.

    .. tab-item:: ipython

        To run the IPython shell, you can do so with::

            pixi run -e dev ipython

        The IPython shell will be started with the `cherab-inversion` package installed.


.. note::

    All commands above are performed in the `dev` environment.
    If you are annoyed by typing `pixi -e dev` every time, you can activate the `dev` environment
    by::

        pixi shell -e dev

    Then, you can execute the commands like `ipython`, `python`, etc. installed in the `dev`
    environment.
