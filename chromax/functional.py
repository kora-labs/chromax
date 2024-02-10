"""Functional module."""
from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
from einops import reduce

from .typing import N_MARKERS, Haploid, Individual, Parents, Population


@jax.jit
def cross(
    parents: Parents["n"],
    recombination_vec: Float[Array, N_MARKERS],
    random_key: jax.Array,
    mutation_probability: float = 0.0,
) -> Population["n"]:
    """Main function that computes crosses from a list of parents.

    :param parents: parents to compute the cross. The shape of
        the parents is (n, 2, m, d), where n is the number of parents,
        m is the number of markers, and d is the ploidy.
    :type parents: ndarray
    :param recombination_vec: array of m probabilities.
        The i-th value represent the probability to recombine before the marker i.
    :type recombination_vec:
    :param random_key: JAX random key, for reproducibility purpose.
    :type random_key: jax.Array
    :param mutation_probability: The probability of having a mutation in a marker.
    :type mutation_probability: float
    :return: offspring population of shape (n, m, d).
    :rtype: ndarray

    :Example:
        >>> from chromax import functional
        >>> import numpy as np
        >>> import jax
        >>> n_chr, chr_len, ploidy = 10, 100, 2
        >>> n_crosses = 50
        >>> parents_shape = (n_crosses, 2, n_chr * chr_len, ploidy)
        >>> parents = np.random.choice([False, True], size=parents_shape)
        >>> rec_vec = np.full((n_chr, chr_len), 1.5 / chr_len)
        >>> rec_vec[:, 0] = 0.5  # equal probability on starting haploid
        >>> rec_vec = rec_vec.flatten()
        >>> random_key = jax.random.key(42)
        >>> f2 = functional.cross(parents, rec_vec, random_key)
        >>> f2.shape
        (50, 1000, 2)
    """
    parents = parents.reshape(*parents.shape[:3], -1, 2)
    random_keys = jax.random.split(
        random_key, num=2 * len(parents) * 2 * parents.shape[3]
    )
    random_keys = random_keys.reshape(2, len(parents), 2, parents.shape[3])
    cross_random_key, mutate_random_key = random_keys

    offsprings = _cross(
        parents,
        recombination_vec,
        cross_random_key,
        mutate_random_key,
        mutation_probability,
    )
    return offsprings.reshape(*offsprings.shape[:-2], -1)


@jax.jit
@partial(jax.vmap, in_axes=(0, None, 0, 0, None))  # parallelize across individuals
@partial(jax.vmap, in_axes=(0, None, 0, 0, None), out_axes=2)  # parallelize parents
def _cross(
    parent: Individual,
    recombination_vec: Float[Array, N_MARKERS],
    cross_random_key: jax.Array,
    mutate_random_key: jax.Array,
    mutation_probability: float,
) -> Haploid:
    return _meiosis(
        parent,
        recombination_vec,
        cross_random_key,
        mutate_random_key,
        mutation_probability,
    )


def double_haploid(
    population: Population["n"],
    n_offspring: int,
    recombination_vec: Float[Array, N_MARKERS],
    random_key: jax.Array,
    mutation_probability: float = 0.0,
) -> Population["n n_offspring"]:
    """Computes the double haploid of the input population.

    :param population: input population of shape (n, m, d).
    :type population: ndarray
    :param n_offspring: number of offspring per plant.
    :type n_offspring: int
    :param recombination_vec: array of m probabilities.
        The i-th value represent the probability to recombine before the marker i.
    :type recombination_vec: ndarray
    :param random_key: JAX random key, for reproducibility purpose.
    :type random_key: jax.Array
    :param mutation_probability: The probability of having a mutation in a marker.
    :type mutation_probability: float
    :return: output population of shape (n, n_offspring, m, d).
        This population will be homozygote.
    :rtype: ndarray

    :Example:
        >>> from chromax import functional
        >>> import numpy as np
        >>> import jax
        >>> n_chr, chr_len, ploidy = 10, 100, 2
        >>> pop_shape = (50, n_chr * chr_len, ploidy)
        >>> f1 = np.random.choice([False, True], size=pop_shape)
        >>> rec_vec = np.full((n_chr, chr_len), 1.5 / chr_len)
        >>> rec_vec[:, 0] = 0.5  # equal probability on starting haploid
        >>> rec_vec = rec_vec.flatten()
        >>> random_key = jax.random.key(42)
        >>> dh = functional.double_haploid(f1, 10, rec_vec, random_key)
        >>> dh.shape
        (50, 10, 1000, 2)
    """
    population = population.reshape(*population.shape[:2], -1, 2)
    keys = jax.random.split(
        random_key, num=2 * len(population) * n_offspring * population.shape[2]
    ).reshape(2, len(population), n_offspring, population.shape[2])
    cross_random_key, mutate_random_key = keys
    haploids = _double_haploid(
        population,
        recombination_vec,
        cross_random_key,
        mutate_random_key,
        mutation_probability,
    )
    dh_pop = jnp.broadcast_to(haploids[..., None], shape=(*haploids.shape, 2))
    return dh_pop.reshape(*dh_pop.shape[:-2], -1)


@jax.jit
@partial(jax.vmap, in_axes=(0, None, 0, 0, None))  # parallelize across individuals
@partial(jax.vmap, in_axes=(None, None, 0, 0, None))  # parallelize across offsprings
def _double_haploid(
    individual: Individual,
    recombination_vec: Float[Array, N_MARKERS],
    cross_random_key: jax.Array,
    mutate_random_key: jax.Array,
    mutation_probability: float,
) -> Haploid:
    return _meiosis(
        individual,
        recombination_vec,
        cross_random_key,
        mutate_random_key,
        mutation_probability,
    )


@jax.jit
@partial(
    jax.vmap, in_axes=(1, None, 0, 0, None), out_axes=1
)  # parallelize pair of chromosomes
def _meiosis(
    individual: Individual,
    recombination_vec: Float[Array, N_MARKERS],
    cross_random_key: jax.Array,
    mutate_random_key: jax.Array,
    mutation_probability: float,
) -> Haploid:
    samples = jax.random.uniform(cross_random_key, shape=recombination_vec.shape)
    rec_sites = samples < recombination_vec
    crossover_mask = jax.lax.associative_scan(jnp.logical_xor, rec_sites)

    crossover_mask = crossover_mask.astype(jnp.int8)
    haploid = jnp.take_along_axis(individual, crossover_mask[:, None], axis=-1)

    mutation_samples = jax.random.uniform(mutate_random_key, shape=haploid.shape)
    mutation_sites = mutation_samples < mutation_probability
    haploid = jnp.where(mutation_sites, 1 - haploid, haploid)

    return haploid.squeeze()


def select(
    population: Population["n"],
    k: int,
    f_index: Callable[[Population["n"]], Float[Array, "n"]],
    weighting: Float[Array, "n traits"] | None = None,
) -> Tuple[Population["k"], Int[Array, "k"]]:
    """Function to select individuals based on their score (index).

    :param population: input grouped population of shape (n, m, d)
    :type population: ndarray
    :param k: number of individual to select.
    :type k: int
    :param f_index: function that computes a score for each individual.
        The function accepts as input a population, i.e. and array of shape
        (n, m, 2) and returns an array of n float number.
    :type f_index: Callable
    :param weigting: array of t float number to weight the traits.
    :type weighting: ndarray

    :return: output population of shape (k, m, d), output indices of shape (k,)
    :rtype: tuple of two ndarrays

    :Example:
        >>> from chromax import functional
        >>> from chromax.trait_model import TraitModel
        >>> from chromax.index_functions import conventional_index
        >>> import numpy as np
        >>> n_chr, chr_len, ploidy = 10, 100, 2
        >>> pop_shape = (50, n_chr * chr_len, ploidy)
        >>> f1 = np.random.choice([False, True], size=pop_shape)
        >>> marker_effects = np.random.randn(n_chr * chr_len)
        >>> gebv_model = TraitModel(marker_effects[:, None])
        >>> f_index = conventional_index(gebv_model)
        >>> f2, selected_indices = functional.select(f1, k=10, f_index=f_index)
        >>> f2.shape
        (10, 1000, 2)
    """
    indices = f_index(population)
    if weighting is not None:
        assert weighting.shape[0] == indices.shape[1]
        indices = jnp.dot(indices, weighting)
        #indices = reduce(indicies * weighting, "n n_traits -> n", "sum")
    else:
        indices = indices[..., 0]
    _, best_pop = jax.lax.top_k(indices, k)
    return population[best_pop, :, :], best_pop
