"""Implementing the conventional breeding program example in Gaynor, R., et al.
"A Two-Part Strategy for Using Genomic Selection to Develop Inbred Lines."
"""

from typing import Tuple

import numpy as np

from chromax import Simulator
from chromax.index_functions import phenotype_index, visual_selection
from chromax.typing import Individual, Population


def wheat_schema(
    simulator: Simulator, germplasm: Population["50"]
) -> Tuple[Population["50"], Individual]:
    f1, _ = simulator.random_crosses(germplasm, 100)
    dh_lines = simulator.double_haploid(f1, n_offspring=100)
    headrows, _ = simulator.select(
        dh_lines, k=5, f_index=visual_selection(simulator, seed=7)
    ).reshape(len(dh_lines) * 5, *dh_lines.shape[2:])
    hdrw_next_cycle, _ = simulator.select(
        dh_lines.reshape(dh_lines.shape[0] * dh_lines.shape[1], *dh_lines.shape[2:]),
        k=20,
        f_index=visual_selection(simulator, seed=7),
    )

    envs = simulator.create_environments(num_environments=16)
    pyt, _ = simulator.select(headrows, k=50, f_index=phenotype_index(simulator, envs[0]))
    pyt_next_cycle, _ = simulator.select(
        headrows, k=20, f_index=phenotype_index(simulator, envs[0])
    )
    ayt, _ = simulator.select(pyt, k=10, f_index=phenotype_index(simulator, envs[:4]))

    released_variety, _ = simulator.select(
        ayt, k=1, f_index=phenotype_index(simulator, envs)
    )

    next_cycle_germplasm = np.concatenate(
        (pyt_next_cycle, ayt, hdrw_next_cycle), axis=0
    )
    return next_cycle_germplasm, released_variety
