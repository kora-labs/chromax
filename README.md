## ChromaX

ChromaX is a breeding simulator for large-scale experiments. It is based on JAX and can run smoothly on multiple devices. We designed this library for researchers and breeders with the scope of guiding the design choices of breeding programs.

### Installation

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