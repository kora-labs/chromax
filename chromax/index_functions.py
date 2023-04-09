from math import ceil
import numpy as np


def conventional_index(GEBV_model):
    def conventional_index_f(pop):
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

    def optimal_haploid_value_f(pop):
        OHP = optimal_haploid_pop(GEBV_model, pop, B, chr_lens)
        OHV = 2 * GEBV_model(OHP[..., None]).squeeze()
        OHV = np.array(OHV)

        if F > 0:
            remove_indices = _poor_conventional_indices(GEBV_model, pop, F)
            OHV[remove_indices] = -float('inf')

        return OHV

    return optimal_haploid_value_f


def optimal_haploid_pop(GEBV_model, population, B=None, chr_lens=None):
    if B is None:
        return _optimal_haploid_pop(GEBV_model, population)
    else:
        assert chr_lens is not None, "chr_lens needed with B"
        return _optimal_haploid_pop_B(GEBV_model, population, B, chr_lens)


def _optimal_haploid_pop_B(GEBV_model, population, B, chr_lens):
    OHP = np.empty((population.shape[0], population.shape[1]), dtype='bool')

    start_idx = 0
    for chr_length in chr_lens:
        block_length = ceil(chr_length / B)
        end_chr = start_idx + chr_length
        for _ in range(B):
            end_block = min(start_idx + block_length, end_chr)
            pop_slice = population[:, start_idx:end_block]
            effect_slice = GEBV_model.marker_effects[start_idx:end_block]
            block_gebv = np.einsum(
                "nmd,me->nd", pop_slice, effect_slice
            )
            best_blocks = np.argmax(block_gebv, axis=-1)
            OHP[:, start_idx:end_block] = np.take_along_axis(
                pop_slice, best_blocks[:, None, None], axis=2
            ).squeeze(axis=-1)
            start_idx = end_block

        assert start_idx == end_chr

    return OHP


def _optimal_haploid_pop(GEBV_model, population):
    optimal_haploid_pop = np.empty(
        (population.shape[0], population.shape[1]), dtype='bool'
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

    def optimal_population_value_f(population):
        indices = np.arange(len(population))
        remove_indices = _poor_conventional_indices(GEBV_model, population, F)
        indices = np.delete(indices, remove_indices)

        output = np.zeros(len(population), dtype='bool')
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
