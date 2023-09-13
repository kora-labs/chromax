"""Utility module with common index functions."""
from math import ceil
from typing import Callable

import numpy as np
from jaxtyping import Array, Float

from chromax.trait_model import TraitModel
from chromax.typing import Population


def phenotype_index(simulator, environments: np.ndarray) -> Callable:
    """Function to select based on phenotyping with some environments.

    :param simulator: chromax simulator instance to use for phenotyping simulation.
    :type simulator: chromax.Simulator
    :param environments: environments created using `create_environments` method from the simulator.

    :return: the phenotyping index function to use for selection.
    :rtype: Callable[[Population["n"]], Float[Array, "n"]]
    """

    def phenotype_index_f(population: Population["n"]) -> Float[Array, "n"]:
        phenotype = simulator.phenotype(
            population, environments=environments, raw_array=True
        )
        return phenotype[..., 0]

    return phenotype_index_f


def visual_selection(simulator, noise_factor: int = 1, seed: int = None) -> Callable:
    """Function to select based on visual selection.

    Practically, this is similar to phenotyping but with more noise.

    :param simulator: chromax simulator instance to use for phenotyping simulation.
    :type simulator: chromax.Simulator
    :param noise_factor: variance ratio between the phenotype and artificial noise added
        to simulate visual selection inaccuracy.
    :type noise_factor: int
    :param seed: random seed for reproducibility.
    :type seed: int

    :return: the visual selection index function.
    :rtype: Callable[[Population["n"]], Float[Array, "n"]]
    """
    generator = np.random.default_rng(seed)

    def visual_selection_f(population: Population["n"]) -> Float[Array, "n"]:
        phenotype = simulator.phenotype(population, raw_array=True)[..., 0]
        noise_var = (simulator.GxE_model.var + simulator.GxE_model.var) * noise_factor
        noise = generator.normal(scale=np.sqrt(noise_var), size=phenotype.shape)
        return phenotype + noise

    return visual_selection_f


def conventional_index(GEBV_model: TraitModel):
    """Function to select based on Genomic Estimated Breeding Value (GEBV).

    :param GEBV_model: GEBV model to estimate the genomic breeding value.
        It must return a single value for an individual, i.e. estimate a single trait.
    :type GEBV_model: chromax.TraitModel

    :return: the conventional genomic selection index function.
    :rtype: Callable[[Population["n"]], Float[Array, "n"]]
    """

    def conventional_index_f(pop: Population["n"]) -> Float[Array, "n"]:
        gebv = GEBV_model(pop)
        assert gebv.shape[-1] == 1
        return gebv[..., 0]

    return conventional_index_f


def _poor_conventional_indices(GEBV_model, pop, F):
    n_poors = int(len(pop) * F)
    GEBVs = conventional_index(GEBV_model)(pop)
    sorted_indices = np.argsort(GEBVs)
    poor_indices = sorted_indices[:n_poors]
    return poor_indices


def optimal_haploid_value(GEBV_model, F=0, B=None, chr_lens=None):
    """Method implementing Optimal Haploid Value (OHV) index function."""

    def optimal_haploid_value_f(pop):
        OHP = optimal_haploid_pop(GEBV_model, pop, B, chr_lens)
        OHV = 2 * GEBV_model(OHP[..., None]).squeeze()
        OHV = np.array(OHV)

        if F > 0:
            remove_indices = _poor_conventional_indices(GEBV_model, pop, F)
            OHV[remove_indices] = -float("inf")

        return OHV

    return optimal_haploid_value_f


def optimal_haploid_pop(GEBV_model, population, B=None, chr_lens=None):
    """Function returning the optimal haploid of a population."""
    if B is None:
        return _optimal_haploid_pop(GEBV_model, population)
    else:
        assert chr_lens is not None, "chr_lens needed with B"
        return _optimal_haploid_pop_B(GEBV_model, population, B, chr_lens)


def _optimal_haploid_pop_B(GEBV_model, population, B, chr_lens):
    OHP = np.empty((population.shape[0], population.shape[1]), dtype="bool")

    start_idx = 0
    for chr_length in chr_lens:
        block_length = ceil(chr_length / B)
        end_chr = start_idx + chr_length
        for _ in range(B):
            end_block = min(start_idx + block_length, end_chr)
            pop_slice = population[:, start_idx:end_block]
            effect_slice = GEBV_model.marker_effects[start_idx:end_block]
            block_gebv = np.einsum("nml,me->nl", pop_slice, effect_slice)
            best_blocks = np.argmax(block_gebv, axis=-1)
            OHP[:, start_idx:end_block] = np.take_along_axis(
                pop_slice, best_blocks[:, None, None], axis=2
            ).squeeze(axis=-1)
            start_idx = end_block

        assert start_idx == end_chr

    return OHP


def _optimal_haploid_pop(GEBV_model, population):
    optimal_haploid_pop = np.empty(
        (population.shape[0], population.shape[1]), dtype="bool"
    )

    positive_mask = GEBV_model.positive_mask.squeeze()

    optimal_haploid_pop[:, positive_mask] = np.any(
        population[:, positive_mask], axis=-1
    )
    optimal_haploid_pop[:, ~positive_mask] = np.all(
        population[:, ~positive_mask], axis=-1
    )

    return optimal_haploid_pop


def optimal_population_value(GEBV_model, n, F=0, B=None, chr_lens=None):
    """Method implementing Optimal Population Value (OPV) index function."""

    def optimal_population_value_f(population):
        indices = np.arange(len(population))
        remove_indices = _poor_conventional_indices(GEBV_model, population, F)
        indices = np.delete(indices, remove_indices)

        output = np.zeros(len(population), dtype="bool")
        positive_mask = GEBV_model.marker_effects[:, 0] > 0
        current_set = ~positive_mask
        G = optimal_haploid_pop(GEBV_model, population[indices], B, chr_lens)

        for _ in range(n):
            dummy_pop = np.broadcast_to(current_set[None, :], G.shape)
            dummy_pop = np.stack((G, dummy_pop), axis=2)
            G = optimal_haploid_pop(GEBV_model, dummy_pop, B, chr_lens)
            best_ind = np.argmax(GEBV_model(G[:, :, None]))
            output[indices[best_ind]] = True
            current_set = G[best_ind]

            indices = np.delete(indices, best_ind)
            G = optimal_haploid_pop(GEBV_model, population[indices], B, chr_lens)

        assert np.count_nonzero(output) == n
        return output  # not OPV but a mask

    return optimal_population_value_f
