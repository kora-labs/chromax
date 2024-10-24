Distributed computation
==========================

We present here how to perform computation on multiple devices.

Consider a scenario where you have access to four GPUs and aim to distribute the workload effectively among them.
To achieve this, we employ the `JAX pmap <https://jax.readthedocs.io/en/latest/_autosummary/jax.pmap.html>`_ function, which allows seamless distribution of functions across multiple accelerators.

.. code-block:: python

   from chromax import Simulator
   from chromax.sample_data import genome, genetic_map
   import jax

   simulator = Simulator(genetic_map=genetic_map)
   # load 200 individuals
   population = simulator.load_population(genome)[:200]
   # divide them in 4 groups
   population = population.reshape(4, -1, *population.shape[1:])

   # prepare a parallelized function over groups
   pmap_dh = jax.pmap(
       simulator.double_haploid,
       in_axes=(0, None),
       static_broadcasted_argnums=1
   )
   # perform distributed computation
   dh_pop = pmap_dh(population, 10)
   # reshape to an ungrouped population
   dh_pop = dh_pop.reshape(-1, *dh_pop.shape[2:])


If you want to perform random crosses or full diallel, grouping the population will change the semantics (the random crosses or the full diallel will be performed by group independently).
In this case, you should use the function ``cross`` after generating the proper array of parents.
For example, to perform random crosses:

.. code-block:: python

    from chromax import Simulator
    from chromax.sample_data import genome, genetic_map
    import numpy as np
    import jax
    simulator = Simulator(genetic_map=genetic_map)
    population = simulator.load_population(genome)
    
    random_indices = np.random.random_integers(0, len(population) - 1, size=(200, 2))
    parents = population[random_indices]
    parents = parents.reshape(4, -1, *parents.shape[1:])
    pmap_cross = jax.pmap(simulator.cross,)
    new_pop = pmap_cross(parents)
    new_pop = new_pop.reshape(-1, *new_pop.shape[2:])

