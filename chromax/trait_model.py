from typing import Tuple
import jax.numpy as jnp
import jax
from jaxtyping import Array, Bool, Float
from .typing import Population


class TraitModel:

    def __init__(
        self,
        marker_effects: Float[Array, "m traits"],
        mean: Float[Array, "#traits"] = 0,
        device=None
    ) -> None:
        self.device = device
        self.marker_effects = jax.device_put(
            marker_effects,
            device=self.device
        )
        self.mean = jax.device_put(mean, device=self.device)
        self.n_traits = marker_effects.shape[1]

        props = _effect_properties(self.marker_effects)
        self.positive_mask, self.max, self.min, self.mean, self.var = props

    def __call__(
        self,
        population: Population["n"]
    ) -> Float[Array, "n traits"]:
        return _call(population, self.marker_effects, self.mean)


@jax.jit
def _call(
    population: Population["n"],
    marker_effects: Float[Array, "m traits"],
    mean: Float[Array, "#traits"]
) -> Float[Array, "n traits"]:
    monoploidy = population.sum(axis=-1, dtype=jnp.int8)
    return jnp.dot(monoploidy, marker_effects) + mean


@jax.jit
def _effect_properties(
    marker_effects: Float[Array, "m traits"]
) -> Tuple[Bool[Array, "m"], float, float, float, float]:
    positive_mask = marker_effects > 0

    max_gebv = 2 * jnp.sum(marker_effects, axis=0, where=positive_mask)
    min_gebv = 2 * jnp.sum(marker_effects, axis=0, where=~positive_mask)
    mean = jnp.sum(marker_effects, axis=0)
    # using variance property for sum of independent variables
    var = jnp.mean(marker_effects**2, axis=0) / 2

    return positive_mask, max_gebv, min_gebv, mean, var
