# ChromaX

ChromaX is a breeding simulator for large-scale experiments. It is based on JAX and can run smoothly on multiple devices. We designed this library for researchers and breeders with the scope of guiding the design choices of breeding programs.

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
f2 = simulator.random_crosses(f1, n_crosses=10, n_offspring=20)
```

## Citing ChromaX

```bibtex

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
```