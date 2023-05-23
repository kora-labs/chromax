from typing import Callable
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from .typing import N_MARKERS, Haploid, Individual, Population
from functools import partial


@jax.jit
@partial(jax.vmap, in_axes=(0, None, 0))  # parallelize across individuals
@partial(jax.vmap, in_axes=(0, None, 0), out_axes=1)  # parallelize parents
def cross(
    parent: Individual,
    recombination_vec: Float[Array, N_MARKERS],
    random_key: jax.random.PRNGKeyArray
) -> Haploid:
    """Main function that computes crosses from a list of parents.

    Args:
        - parents (array): parents to compute the cross. The shape of
        the parents is (n, 2, m, 2), where n is the number of parents
        and m is the number of markers.
        - recombination_vec (array): array of m probabilities.
        The i-th value represent the probability to recombine before the marker i.
        - random_key (array): array of n PRNGKey, one for each pair of parents.

    Returns:
        - population (array): offspring population of shape (n, m, 2).

    Example:
    >>> from chromax import functional
    >>> import numpy as np
    >>> import jax
    >>> n_chr, chr_len = 10, 100
    >>> n_crosses = 50
    >>> parents_shape = (n_crosses, 2, n_chr * chr_len, 2)
    >>> parents = np.random.choice([False, True], size=parents_shape)
    >>> rec_vec = np.full((n_chr, chr_len), 1.5 / chr_len)
    >>> rec_vec[:, 0] = 0.5  # equal probability on starting haploid
    >>> rec_vec = rec_vec.flatten()
    >>> random_key = jax.random.PRNGKey(42)
    >>> random_keys = jax.random.split(random_key, num=2*n_crosses)
    >>> random_keys = random_keys.reshape(n_crosses, 2, 2)
    >>> f2 = functional.cross(parents, rec_vec, random_keys)
    >>> f2.shape
    (50, 1000, 2)
    """
    return _cross_individual(
        parent,
        recombination_vec,
        random_key
    )


def double_haploid(
    population: Population["n"],
    n_offspring: int,
    recombination_vec: Float[Array, N_MARKERS],
    random_key: jax.random.PRNGKeyArray
) -> Population["n n_offspring"]:
    """Computes the double haploid of the input population.

    Args:
        - population (array): input population of shape (n, m, 2).
        - n_offspring (int): number of offspring per plant.
        - recombination_vec (array): array of m probabilities.
        The i-th value represent the probability to recombine before the marker i.
        - random_key (array): array of n PRNGKey, one for each individual.

    Returns:
        - population (array): output population of shape (n, n_offspring, m, 2).
        This population will be homozygote.
    
    Example:
    >>> from chromax import functional
    >>> import numpy as np
    >>> import jax
    >>> n_chr, chr_len = 10, 100
    >>> pop_shape = (50, n_chr * chr_len, 2)
    >>> f1 = np.random.choice([False, True], size=pop_shape)
    >>> rec_vec = np.full((n_chr, chr_len), 1.5 / chr_len)
    >>> rec_vec[:, 0] = 0.5  # equal probability on starting haploid
    >>> rec_vec = rec_vec.flatten()
    >>> random_key = jax.random.PRNGKey(42)
    >>> dh = functional.double_haploid(f1, 10, rec_vec, random_key)
    >>> dh.shape
    (50, 10, 1000, 2)
    """
    keys = jax.random.split(
        random_key, 
        num=len(population) * n_offspring
    ).reshape(len(population), n_offspring, 2)
    haploid = _double_haploid(population, recombination_vec, keys)
    return jnp.broadcast_to(haploid[..., None], shape=(*haploid.shape, 2))


@jax.jit
@partial(jax.vmap, in_axes=(0, None, 0))  # parallelize across individuals
@partial(jax.vmap, in_axes=(None, None, 0))  # parallelize across offsprings
def _double_haploid(
    individual: Individual,
    recombination_vec: Float[Array, N_MARKERS],
    random_key: jax.random.PRNGKeyArray
) -> Haploid:
    return _cross_individual(
        individual,
        recombination_vec,
        random_key
    )


@jax.jit
def _cross_individual(
    parent: Individual,
    recombination_vec: Float[Array, N_MARKERS],
    random_key: jax.random.PRNGKeyArray
) -> Haploid:
    samples = jax.random.uniform(random_key, shape=recombination_vec.shape)
    rec_sites = samples < recombination_vec
    crossover_mask = jax.lax.associative_scan(jnp.logical_xor, rec_sites)

    crossover_mask = crossover_mask.astype(jnp.int8)
    haploid = jnp.take_along_axis(
        parent,
        crossover_mask[:, None],
        axis=-1
    )

    return haploid.squeeze()


def select(
    population: Population["n"],
    k: int,
    f_index: Callable[[Population["n"]], Float[Array, "n"]]
) -> Population["k"]:
    """Function to select individuals based on their score (index).

    Args:
        - population (array): input grouped population of shape (n, m, 2)
        - k (int): number of individual to select.
        - f_index (function): function that computes a score for each individual.
        The function accepts as input a population, i.e. and array of shape
        (n, m, 2) and returns an arrray of n float number.

    Returns:
        - population (array): output population of (k, m, 2)

    Example:
    >>> from chromax import functional
    >>> from chromax.trait_model import TraitModel
    >>> from chromax.index_functions import conventional_index
    >>> import numpy as np
    >>> n_chr, chr_len = 10, 100
    >>> pop_shape = (50, n_chr * chr_len, 2)
    >>> f1 = np.random.choice([False, True], size=pop_shape)
    >>> marker_effects = np.random.randn(n_chr * chr_len)
    >>> gebv_model = TraitModel(marker_effects[:, None])
    >>> f_index = conventional_index(gebv_model)
    >>> f2 = functional.select(f1, k=10, f_index=f_index)
    >>> f2.shape
    (10, 1000, 2)
    """

    indices = f_index(population)
    _, best_pop = jax.lax.top_k(indices, k)
    return population[best_pop, :, :]