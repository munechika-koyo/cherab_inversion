:orphan:

.. _quickstart:

===========
Quick Start
===========

Only 3 steps to get started with `cherab-inversion` with `Pixi`_.
Please make sure you have `Pixi`_ installed on your system.
If not, please follow the `Pixi installation guide <https://pixi.sh/latest#installation>`__.

1. Download the repository::

    git clone https://github.com/munechika-koyo/cherab_inversion

2. Move to the repository::

    cd cherab_inversion

3. Run the ``lab`` command::

    pixi run lab

Then, you can see the JupyterLab is automatically launched and enjoy some notebooks in the
`docs/notebooks/` directory.
If you don't see browser window opened, please find the URL like
(http://127.0.0.1:8888/lab?token=...) in the terminal output and open it in your browser.

.. figure:: ../_static/images/quickstart_jupyterlab.webp
   :align: center
   :alt: JupyterLab Window

   JupyterLab launched by `pixi run lab` command. You can execute notebooks in
   :ref:`Example Gallery<example-gallery>`.


If you want to use the `cherab-inversion` as a library, please follow the
:ref:`installation guide<installation>`.
