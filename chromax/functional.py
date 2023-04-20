import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from .typing import N_MARKERS, Haploid, Individual
from functools import partial


@jax.jit
@partial(jax.vmap, in_axes=(0, None, 0))  # parallelize across individuals
@partial(jax.vmap, in_axes=(0, None, 0), out_axes=1)  # parallelize parents
def cross(
    parent: Individual,
    recombination_vec: Float[Array, N_MARKERS],
    random_key: jax.random.PRNGKeyArray
) -> Haploid:
    return _cross_individual(
        parent,
        recombination_vec,
        random_key
    )


@jax.jit
@partial(jax.vmap, in_axes=(0, None, 0))  # parallelize across individuals
def double_haploid(
    individual: Individual,
    recombination_vec: Float[Array, N_MARKERS],
    random_key: jax.random.PRNGKeyArray
) -> Individual:
    haploid = _cross_individual(
        individual,
        recombination_vec,
        random_key
    )
    return jnp.broadcast_to(haploid[:, None], shape=(*haploid.shape, 2))


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
