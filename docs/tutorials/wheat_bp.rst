Simulate a wheat breeding program
===================================

We describe here a sample breeding program to develop inbred cultivars.
Specifically, we follow the conventional breeding scheme for wheat described in `Gaynor, R., et al. 
"A Two-Part Strategy for Using Genomic Selection to Develop Inbred Lines."``

First of all, let's import the simulator and the module containing the index functions we will use to select the plants:

.. code-block:: python

    from chromax import Simulator
    from chromax import index_functions

and then we initialize the simulator and load the population. For the example, we can use the sample data in the package:

.. code-block:: python

    from chromax.sample_data import genome, genetic_map
    simulator = Simulator(
        genetic_map=genetic_map,
        trait_names=["Yield"],
        h2=[0.4]
    )
    f0 = simulator.load_population(genome)

We are interested in selecting based on `Yield` only and we set an heritability of ``0.4``.
We start the breeding program by performing some random crosses that produce a new population. Then, we obtain a line of plants from each individual by performing double haploid induction:

.. code-block:: python

    f1 = simulator.random_crosses(f0, 100)
    dh_lines = simulator.double_haploid(f1, n_offspring=100)

In this way we obtain 100 lines, each containing 100 plants.
We then start a typical bottleneck selection process, were we start with low accuracy methodologies to reduce the number of plants and we iterativeley increase the accuracy (and cost) of selection method.
In particular, we start with a visual selection on the rows and then we test the plants on an increasing number of locations. 
The code will be like this:

.. code-block:: python

    headrows = simulator.select(
        dh_lines,
        k=5,
        f_index=index_functions.visual_selection(simulator, seed=7)
    )
    headrows = headrows.reshape(len(dh_lines) * 5, *dh_lines.shape[2:])

    envs = simulator.create_environments(num_environments=16)
    pyt = simulator.select(
        headrows,
        k=50,
        f_index=index_functions.phenotype_index(simulator, envs[0])
    )
    ayt = simulator.select(
        pyt,
        k=10,
        f_index=index_functions.phenotype_index(simulator, envs[:4])
    )

    released_variety = simulator.select(
        ayt,
        k=1,
        f_index=index_functions.phenotype_index(simulator, envs)
    )

In this way we simulate the developing of a cultivar after a breeding cycle. If we want to continue with multiple cycle, we also need to compose the founder population of the next cycle.  
For example:

.. code-block:: python

    hdrw_next_cycle = simulator.select(
        dh_lines.reshape(dh_lines.shape[0] * dh_lines.shape[1], *dh_lines.shape[2:]),
        k=20,
        f_index=index_functions.visual_selection(simulator, seed=7)
    )
    pyt_next_cycle = simulator.select(
        headrows,
        k=20,
        f_index=index_functions.phenotype_index(simulator, envs[0])
    )
    next_cycle_f0 = np.concatenate(
        (pyt_next_cycle, ayt, hdrw_next_cycle),
        axis=0
    )

And then repeating the breeding scheme using the `next_cycle_f0` as founder population.
The code for the breeding scheme can be found `here <https://github.com/kora-labs/chromax/blob/master/examples/wheat_bp.py>`_.