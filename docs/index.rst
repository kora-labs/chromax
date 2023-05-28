.. ChromaX documentation master file, created by
   sphinx-quickstart on Fri May 26 20:32:24 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ChromaX: a fast and scalable breeding program simulator
========================================================

ChromaX is a Python library that enables the simulation of genetic recombination, genomic estimated
breeding value calculations, and selection processes.
The library is based on `JAX <https://jax.readthedocs.io>`_ to exploit parallelization.


Installation
===================================

You can install the library via Python Package manager `pip`:

.. code-block:: bash

   pip install chromax

This will install all thre requirements, like JAX, NumPy and pandas. 

Citing
===================================

.. note::

  The sample data used in the example is taken from `Wang, Shichen et al. "Characterization of polyploid wheat genomic diversity using a high-density 90 000 single nucleotide polymorphism array". Plant Biotechnology Journal 12. 6(2014): 787-796.`

To cite ChromaX in publications:

.. code-block:: bibtex

  @misc{younis2023chromax,
    title={ChromaX: a fast and scalable breeding program simulator},
    author={Younis, Omar G. and Turchetta, Matteo and Ariza Suarez, Daniel and Yates, Steven and Studer, Bruno and Athanasiadis, Ioannis and Buhmann, Joachim M. and Krause, Andreas and Corinzia, Luca },
    year={2023}
  }

.. toctree::
  :maxdepth: 3
  :caption: Modules:

  modules/simulator
  modules/functional

.. toctree::
  :maxdepth: 3
  :caption: Tutorials:

  tutorials/data_format

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
