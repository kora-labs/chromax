"""Implementing the conventional breeding program example in Gaynor, R., et al.
"A Two-Part Strategy for Using Genomic Selection to Develop Inbred Lines."
"""

from typing import Tuple
from chromax import Simulator
from chromax.index_functions import phenotype_index
from chromax.typing import Individual, Population
import numpy as np


def wheat_schema(
    simulator: Simulator,
    germplasm: Population["50"]
) -> Tuple[Population["50"], Individual]:
    f1 = simulator.random_crosses(germplasm, 100)
    dh_lines = simulator.double_haploid(f1, n_offspring=100)
    headrows = simulator.select(
        dh_lines,
        k=5,
        f_index=visual_selection(simulator, seed=7)
    ).reshape(len(dh_lines) * 5, *dh_lines.shape[2:])
    hdrw_next_year = simulator.select(
        dh_lines.reshape(dh_lines.shape[0] * dh_lines.shape[1], *dh_lines.shape[2:]),
        k=20,
        f_index=visual_selection(simulator, seed=7)
    )

    envs = simulator.create_environments(num_environments=16)
    pyt = simulator.select(
        headrows,
        k=50,
        f_index=phenotype_index(simulator, envs[0])
    )
    pyt_next_year = simulator.select(
        headrows,
        k=20,
        f_index=phenotype_index(simulator, envs[0])
    )
    ayt = simulator.select(
        pyt,
        k=10,
        f_index=phenotype_index(simulator, envs[:4])
    )

    released_variety = simulator.select(
        ayt,
        k=1,
        f_index=phenotype_index(simulator, envs)
    )

    next_year_germplasm = np.concatenate(
        (pyt_next_year, ayt, hdrw_next_year),
        axis=0
    )
    return next_year_germplasm, released_variety


def visual_selection(simulator, noise_factor=1, seed=None):
    generator = np.random.default_rng(seed)

    def visual_selection_f(population):
        phenotype = simulator.phenotype(population, raw_array=True)[..., 0]
        noise_var = (simulator.GxE_model.var + simulator.GxE_model.var) * noise_factor
        noise = generator.normal(scale=np.sqrt(noise_var), size=phenotype.shape)
        return phenotype + noise

    return visual_selection_f
