import jax
import numpy as np
import pytest

from chromax import functional
from chromax.index_functions import conventional_index
from chromax.trait_model import TraitModel


@pytest.mark.parametrize("idx", [0, 1])
def test_cross(idx):
    n_markers, ploidy = 1000, 4
    n_crosses = 50
    parents_shape = (n_crosses, 2, n_markers, ploidy)
    parents = np.random.choice([False, True], size=parents_shape)
    rec_vec = np.zeros(n_markers)
    rec_vec[0] = idx
    random_key = jax.random.key(42)
    new_pop = functional.cross(parents, rec_vec, random_key)

    for i in range(ploidy):
        assert np.all(new_pop[..., i] == parents[:, i % 2, :, i - i % 2 + idx])


def test_double_haploid():
    n_chr, chr_len, ploidy = 10, 100, 2
    n_offspring = 10
    pop_shape = (50, n_chr * chr_len, ploidy)
    f1 = np.random.choice([False, True], size=pop_shape)
    rec_vec = np.full((n_chr * chr_len,), 1.5 / chr_len)
    random_key = jax.random.key(42)
    dh = functional.double_haploid(f1, n_offspring, rec_vec, random_key)
    assert dh.shape == (len(f1), n_offspring, n_chr * chr_len, ploidy)

    for i in range(ploidy // 2):
        assert np.all(dh[..., i * 2] == dh[..., i * 2 + 1])


def test_select():
    n_markers, ploidy = 1000, 4
    k = 10
    pop_shape = (50, n_markers, ploidy)
    f1 = np.random.choice([False, True], size=pop_shape)
    marker_effects = np.random.randn(n_markers)
    gebv_model = TraitModel(marker_effects[:, None])
    f_index = conventional_index(gebv_model)
    f2, best_indices = functional.select(f1, k=k, f_index=f_index)
    assert f2.shape == (k, *f1.shape[1:])
    assert best_indices.shape == (k,)

    f1_gebv = gebv_model(f1)
    f2_gebv = gebv_model(f2)
    assert np.max(f2_gebv) == np.max(f1_gebv)
    assert np.mean(f2_gebv) > np.mean(f1_gebv)
    assert np.min(f2_gebv) > np.min(f1_gebv)


def test_cross_mutation():
    n_markers, ploidy = 1000, 4
    zeros_pop = np.zeros((50, 2, n_markers, ploidy))
    ones_pop = np.ones((50, 2, n_markers, ploidy))
    rec_vec = np.full((n_markers,), 1.5e-2)
    cross = functional.cross

    random_key = jax.random.key(42)
    assert np.all(cross(zeros_pop, rec_vec, random_key) == 0)
    assert np.all(cross(zeros_pop, rec_vec, random_key, 1) == 1)
    mutated_pop = cross(zeros_pop, rec_vec, random_key, 0.5)
    assert np.count_nonzero(mutated_pop) > 0
    assert np.count_nonzero(1 - mutated_pop) > 0

    assert np.all(cross(ones_pop, rec_vec, random_key) == 1)
    assert np.all(cross(ones_pop, rec_vec, random_key, 1) == 0)
    mutated_pop = cross(ones_pop, rec_vec, random_key, 0.5)
    assert np.count_nonzero(mutated_pop) > 0
    assert np.count_nonzero(1 - mutated_pop) > 0


def test_dh_mutation():
    n_markers, ploidy = 1000, 4
    zeros_pop = np.zeros((50, n_markers, ploidy))
    ones_pop = np.ones((50, n_markers, ploidy))
    rec_vec = np.full((n_markers,), 1.5e-2)
    dh = functional.double_haploid

    random_key = jax.random.key(42)
    assert np.all(dh(zeros_pop, 10, rec_vec, random_key) == 0)
    assert np.all(dh(zeros_pop, 10, rec_vec, random_key, 1) == 1)
    mutated_pop = dh(zeros_pop, 10, rec_vec, random_key, 0.5)
    assert np.count_nonzero(mutated_pop) > 0
    assert np.count_nonzero(1 - mutated_pop) > 0

    assert np.all(dh(ones_pop, 10, rec_vec, random_key) == 1)
    assert np.all(dh(ones_pop, 10, rec_vec, random_key, 1) == 0)
    mutated_pop = dh(ones_pop, 10, rec_vec, random_key, 0.5)
    assert np.count_nonzero(mutated_pop) > 0
    assert np.count_nonzero(1 - mutated_pop) > 0
