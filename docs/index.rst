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

This will install all the requirements, like JAX, NumPy and Pandas. 

Citing
===================================

.. note::

  The sample data used in the examples is taken from `Wang, Shichen et al. "Characterization of polyploid wheat genomic diversity using a high-density 90 000 single nucleotide polymorphism array". Plant Biotechnology Journal 12. 6(2014): 787-796.`

To cite ChromaX in publications:

.. code-block:: bibtex

  @article{Younis2023.05.29.542709,
    abstract = {ChromaX is a Python library that enables the simulation of genetic recombination, genomic estimated breeding value calculations, and selection processes. By utilizing GPU processing, it can perform these simulations up to two orders of magnitude faster than existing tools with standard hardware. This offers breeders and scientists new opportunities to simulate genetic gain and optimize breeding schemes.},
    author = {Omar G. Younis and Matteo Turchetta and Daniel Ariza Suarez and Steven Yates and Bruno Studer and Ioannis N. Athanasiadis and Andreas Krause and Joachim M. Buhmann and Luca Corinzia},
    doi = {10.1101/2023.05.29.542709},
    elocation-id = {2023.05.29.542709},
    eprint = {https://www.biorxiv.org/content/early/2023/05/31/2023.05.29.542709.1.full.pdf},
    journal = {bioRxiv},
    publisher = {Cold Spring Harbor Laboratory},
    title = {ChromaX: a fast and scalable breeding program simulator},
    url = {https://www.biorxiv.org/content/early/2023/05/31/2023.05.29.542709.1},
    year = {2023},
    bdsk-url-1 = {https://www.biorxiv.org/content/early/2023/05/31/2023.05.29.542709.1},
    bdsk-url-2 = {https://doi.org/10.1101/2023.05.29.542709}
  }

.. toctree::
  :maxdepth: 3
  :caption: Modules:

  modules/simulator
  modules/functional
  modules/index_functions

.. toctree::
  :maxdepth: 3
  :caption: Tutorials:

  tutorials/data_format
  tutorials/wheat_bp
  tutorials/distributed_computation

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
