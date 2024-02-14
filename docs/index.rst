.. ChromaX documentation master file, created by
   sphinx-quickstart on Fri May 26 20:32:24 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ChromaX: a fast and scalable breeding program simulator
========================================================

ChromaX is a Python library that enables the simulation of genetic recombination, genomic estimated
breeding value calculations, and selection processes.
The library is based on `JAX <https://jax.readthedocs.io>`_ to exploit parallelization.
It can smoothly operate on CPU, GPU (NVIDIA, Apple, AMD, and Intel), or TPU.

Installation
===================================
.. note::

  To exploit parallelization capabilities of your hardware, it is recommended to install jax manually. 
  You can find the instruction for your hardware in `google/jax <https://github.com/google/jax?tab=readme-ov-file#installation>`_.

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

  @article{10.1093/bioinformatics/btad691,
      author = {Younis, Omar G and Turchetta, Matteo and Ariza Suarez, Daniel and Yates, Steven and Studer, Bruno and Athanasiadis, Ioannis N and Krause, Andreas and Buhmann, Joachim M and Corinzia, Luca},
      title = "{ChromaX: a fast and scalable breeding program simulator}",
      journal = {Bioinformatics},
      volume = {39},
      number = {12},
      pages = {btad691},
      year = {2023},
      month = {11},
      abstract = "{ChromaX is a Python library that enables the simulation of genetic recombination, genomic estimated breeding value calculations, and selection processes. By utilizing GPU processing, it can perform these simulations up to two orders of magnitude faster than existing tools with standard hardware. This offers breeders and scientists new opportunities to simulate genetic gain and optimize breeding schemes.The documentation is available at https://chromax.readthedocs.io. The code is available at https://github.com/kora-labs/chromax.}",
      issn = {1367-4811},
      doi = {10.1093/bioinformatics/btad691},
      url = {https://doi.org/10.1093/bioinformatics/btad691},
      eprint = {https://academic.oup.com/bioinformatics/article-pdf/39/12/btad691/54143193/btad691.pdf},
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
