"""Module containing the trait model."""
from typing import Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from .typing import Population


class TraitModel:
    """Breeding simulator class. It can perform the most common operation of a breeding program.

    :param marker_effects: linear regressor weights for the traits.
    :type marker_effects: Array of shape (m, t), where t is the number of traits.
    :param offset: linear regressor offsets for each trait.
    :type offset: 1-dimensional array of length t or a float number.
    :param device: the device on which compute the traits estimation.
        If not specified, the default device will be chosen.
    :type device: XLA Device
    """

    def __init__(
        self,
        marker_effects: Float[Array, "m traits"],
        offset: Float[Array, "#traits"] = 0,
        device=None,
    ) -> None:
        """Initialize a trait model."""
        self.device = device
        self.marker_effects = jax.device_put(marker_effects, device=self.device)
        self.offset = jax.device_put(offset, device=self.device)
        self.n_traits = marker_effects.shape[1]

        props = _effect_properties(self.marker_effects, offset)
        self.positive_mask, self.max, self.min, self.mean, self.var = props

    def __call__(self, population: Population["n"]) -> Float[Array, "n traits"]:
        """Estimate the traits for the given population."""
        return _call(population, self.marker_effects, self.offset)


@jax.jit
def _call(
    population: Population["n"],
    marker_effects: Float[Array, "m traits"],
    offset: Float[Array, "#traits"],
) -> Float[Array, "n traits"]:
    monoploidy = population.sum(axis=-1, dtype=jnp.int8)
    return jnp.dot(monoploidy, marker_effects) + offset


@jax.jit
def _effect_properties(
    marker_effects: Float[Array, "m traits"], offset: Float[Array, "#traits"]
) -> Tuple[Bool[Array, "m"], float, float, float, float]:
    positive_mask = marker_effects > 0

    max_gebv = 2 * jnp.sum(marker_effects, axis=0, where=positive_mask) + offset
    min_gebv = 2 * jnp.sum(marker_effects, axis=0, where=~positive_mask) + offset
    mean = jnp.sum(marker_effects, axis=0) + offset
    # using variance property for sum of independent variables
    var = jnp.sum(marker_effects**2, axis=0) / 2

    return positive_mask, max_gebv, min_gebv, mean, var
