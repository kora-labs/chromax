# ChromaX

ChromaX is a breeding simulator for large-scale experiments. It is based on [JAX](https://github.com/google/jax) and can run smoothly on multiple devices. We designed this library for researchers and breeders with the scope of guiding the design choices of breeding programs.

## Installation

You can install ChromaX via pip:

```batch
pip install chromax
```

## Quickstart

```python
from chromax import Simulator
from chromax.sample_data import genetic_map, genome

simulator = Simulator(genetic_map=genetic_map)
f1 = simulator.load_population(genome)
f2, parent_ids = simulator.random_crosses(f1, n_crosses=10, n_offspring=20)
```

## Citing ChromaX

```bibtex

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
```
