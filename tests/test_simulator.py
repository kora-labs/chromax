import warnings

import jax
import numpy as np
import pandas as pd
import pytest

from chromax import Simulator
from chromax.sample_data import genetic_map, genome
from mock_simulator import MockSimulator


@pytest.mark.parametrize("idx", [0, 1])
def test_cross_r(idx):
    n_markers = 1000
    recombination_vec = np.zeros(n_markers, dtype="bool")
    recombination_vec[0] = idx
    simulator = MockSimulator(recombination_vec=recombination_vec)

    ploidy = 4
    size = (1, 2, simulator.n_markers, ploidy)
    parents = np.random.choice(a=[False, True], size=size, p=[0.5, 0.5])

    new_pop = simulator.cross(parents)

    assert new_pop.shape == (1, simulator.n_markers, ploidy)

    ind = new_pop[0]
    for i in range(ploidy):
        pair_chr_idx = i % 2
        assert np.all(ind[:, i] == parents[0, pair_chr_idx, :, i - pair_chr_idx + idx])


def test_equal_parents():
    simulator = Simulator(genetic_map=genetic_map)

    ploidy = 4
    parents = np.zeros((1, 2, simulator.n_markers, ploidy), dtype="bool")
    child = simulator.cross(parents)
    assert np.all(child == 0)

    parents = np.ones((1, 2, simulator.n_markers, ploidy), dtype="bool")
    child = simulator.cross(parents)
    assert np.all(child == 1)


def test_ad_hoc_cross():
    rec_vec = np.array(
        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.int8
    )

    simulator = MockSimulator(recombination_vec=rec_vec)
    population = simulator.load_population(2, ploidy=4)
    parents = population[np.array([[0, 1]])]
    child = simulator.cross(parents)

    assert child.shape == (1, *population[0].shape)

    chr_idx = 0
    for mrk_idx, rec_prob in enumerate(rec_vec):
        if rec_prob == 1:
            chr_idx = 1 - chr_idx
        assert child[1, mrk_idx, 0] == population[0, mrk_idx, chr_idx]
        assert child[1, mrk_idx, 1] == population[1, mrk_idx, chr_idx]
        assert child[1, mrk_idx, 2] == population[0, mrk_idx, 2 + chr_idx]
        assert child[1, mrk_idx, 3] == population[1, mrk_idx, 2 + chr_idx]


def test_cross_two_times():
    n_markers = 100_000
    n_ind = 2
    simulator = MockSimulator(n_markers=n_markers)
    population = simulator.load_population(n_ind, ploidy=4)

    parents = population[np.array([[0, 1], [0, 1]])]
    children = simulator.cross(parents)

    assert np.any(children[0] != children[1])


def test_double_haploid():
    n_markers = 1000
    n_ind = 100
    n_offspring = 10
    ploidy = 4

    simulator = MockSimulator(n_markers=n_markers)
    population = simulator.load_population(n_ind, ploidy=ploidy)

    new_pop = simulator.double_haploid(population, n_offspring=n_offspring)
    assert new_pop.shape == (len(population), n_offspring, *population.shape[1:])
    assert np.all(new_pop[..., 0] == new_pop[..., 1])
    assert np.all(new_pop[..., 2] == new_pop[..., 3])

    new_pop = simulator.double_haploid(population)
    assert new_pop.shape == population.shape
    assert np.all(new_pop[..., 0] == new_pop[..., 1])
    assert np.all(new_pop[..., 2] == new_pop[..., 3])


def test_diallel():
    n_markers = 1000
    n_ind = 100
    ploidy = 4
    simulator = MockSimulator(n_markers=n_markers)
    population = simulator.load_population(n_ind, ploidy=ploidy)

    diallel_indices = simulator._diallel_indices(np.arange(10))
    assert len(np.unique(diallel_indices, axis=0)) == 45

    new_pop = simulator.diallel(population)
    assert new_pop.shape == (n_ind * (n_ind - 1) // 2, n_markers, ploidy)

    new_pop = simulator.diallel(population, n_offspring=10)
    assert new_pop.shape == (n_ind * (n_ind - 1) // 2, 10, n_markers, ploidy)


def test_select():
    n_markers = 1000
    n_ind = 100
    ploidy = 4
    simulator = MockSimulator(n_markers=n_markers)
    population = simulator.load_population(n_ind, ploidy=ploidy)
    pop_GEBV = simulator.GEBV(population)

    selected_pop = simulator.select(population, k=10)
    selected_GEBV = simulator.GEBV(selected_pop)
    assert np.all(selected_GEBV.mean() > pop_GEBV.mean())
    assert np.all(selected_GEBV.max() == pop_GEBV.max())
    assert np.all(selected_GEBV.min() > pop_GEBV.min())

    dh = simulator.double_haploid(population, n_offspring=100)
    selected_dh = simulator.select(dh, k=5)
    assert selected_dh.shape == (n_ind, 5, n_markers, ploidy)
    for i in range(n_ind):
        dh_GEBV = simulator.GEBV(dh[i])
        selected_GEBV = simulator.GEBV(selected_dh[i])
        assert np.all(selected_GEBV.mean() > dh_GEBV.mean())
        assert np.all(selected_GEBV.max() == dh_GEBV.max())
        assert np.all(selected_GEBV.min() > dh_GEBV.min())


def test_random_crosses():
    n_markers = 1000
    n_ind = 100
    ploidy = 4
    simulator = MockSimulator(n_markers=n_markers)
    population = simulator.load_population(n_ind, ploidy=ploidy)

    n_crosses = 300
    new_pop, _ = simulator.random_crosses(population, n_crosses=n_crosses)
    assert new_pop.shape == (n_crosses, n_markers, ploidy)

    n_offspring = 10
    new_pop, _ = simulator.random_crosses(
        population=population, n_crosses=n_crosses, n_offspring=n_offspring
    )
    assert new_pop.shape == (n_crosses, n_offspring, n_markers, ploidy)


def test_multi_trait():
    trait_names = [
        "Heading Date",
        "Protein Content",
        "Plant Height",
        "Thousand Kernel Weight",
        "Yield",
        "Fusarium Head Blight",
        "Spike Emergence Period",
    ]
    simulator = Simulator(genetic_map=genetic_map, trait_names=trait_names)
    population = simulator.load_population(genome)

    gebv_shape = len(population), len(trait_names)
    assert simulator.GEBV(population).shape == gebv_shape


def test_device():
    local_devices = jax.local_devices()
    if len(local_devices) == 1:
        warnings.warn("Device testing skipped because there is only one device.")
        return

    device = local_devices[1]
    simulator = Simulator(genetic_map=genetic_map, device=device)

    population = simulator.load_population(genome)
    assert population.device_buffer.device() == device

    GEBV = simulator.GEBV_model(population)
    assert GEBV.device_buffer.device() == device

    selected_pop = simulator.select(population, k=10)
    assert selected_pop.device_buffer.device() == device

    diallel = simulator.diallel(selected_pop, n_offspring=10)
    assert diallel.device_buffer.device() == device

    dh_pop = simulator.double_haploid(diallel)
    assert dh_pop.device_buffer.device() == device

    cross_indices = np.array([[1, 5], [3, 10], [100, 2], [7, 93], [28, 41]])
    new_pop = simulator.cross(dh_pop[cross_indices])
    assert new_pop.device_buffer.device() == device


def test_seed_deterministic():
    n_ind = 100
    ploidy = 4
    simulator1 = Simulator(genetic_map=genetic_map, seed=7)
    simulator2 = Simulator(genetic_map=genetic_map, seed=7)
    mock_simulator = MockSimulator(n_markers=simulator1.n_markers)
    population = mock_simulator.load_population(n_ind, ploidy=ploidy)

    new_pop1, _ = simulator1.random_crosses(population, n_crosses=10)
    new_pop2, _ = simulator2.random_crosses(population, n_crosses=10)

    assert np.all(new_pop1 == new_pop2)


def test_gebv():
    n_markers, n_ind = 100, 10
    ploidy = 4
    simulator = MockSimulator(n_markers=n_markers)
    population = simulator.load_population(n_ind, ploidy=ploidy)

    gebv_pandas = simulator.GEBV(population)
    assert len(gebv_pandas) == n_ind
    assert isinstance(gebv_pandas, pd.DataFrame)

    gebv_array = simulator.GEBV(population, raw_array=True)
    assert len(gebv_array) == n_ind
    assert np.all(gebv_pandas.values == gebv_array)


def test_phenotyping():
    n_markers, n_ind = 100, 10
    ploidy = 4
    simulator = MockSimulator(n_markers=n_markers)
    population = simulator.load_population(n_ind, ploidy=ploidy)

    phenotype = simulator.phenotype(population, num_environments=4)
    assert len(phenotype) == n_ind

    environments = np.random.uniform(-1, 1, size=(8,))
    _ = simulator.phenotype(population, environments=environments)
    assert len(phenotype) == n_ind

    with pytest.raises(ValueError):
        simulator.phenotype(population, num_environments=8, environments=environments)
