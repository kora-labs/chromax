from chromax import Simulator
import timeit
import pandas as pd
import numpy as np
from chromax.index_functions import phenotype_index, visual_selection


def wheat_schema(
    simulator: Simulator,
    germplasm,
    factor = 1,
):
    f1 = simulator.random_crosses(germplasm, 200 * factor)
    
    dh_lines = simulator.double_haploid(f1, n_offspring=100)
    # Use the following instead on M1 with 20x factor to avoid trashing
    # dh_lines1 = simulator.double_haploid(f1[:100*factor], n_offspring=100)
    # dh_lines2 = simulator.double_haploid(f1[100*factor:], n_offspring=100)
    # dh_lines = jax.numpy.concatenate((dh_lines1, dh_lines2))

    headrows = simulator.select(
        dh_lines,
        5,
        visual_selection(simulator, seed=7)
    )
    headrows = headrows.reshape(1000 * factor, -1, 2)

    envs = simulator.create_environments(num_environments=16)
    pyt = simulator.select(
        headrows,
        k=100 * factor,
        f_index=phenotype_index(simulator, envs[0])
    )
    ayt = simulator.select(
        pyt,
        k=10 * factor,
        f_index=phenotype_index(simulator, envs[:4])
    )

    released_variety = simulator.select(
        ayt,
        k=1,
        f_index=phenotype_index(simulator, envs)
    )

    return released_variety


if __name__ == "__main__":
    repeats = 100
    n_chr = 21
    chr_len = 100
    factor = 1
    times = np.empty(repeats)

    genetic_map = pd.DataFrame({
        "CHR.PHYS": np.arange(n_chr * chr_len, dtype=np.int32) // chr_len,
        "Yield": np.random.standard_normal(n_chr * chr_len).astype(np.float32),
        "RecombRate": np.full(n_chr * chr_len, 1.5 / 1000)
    })
    simulator = Simulator(genetic_map=genetic_map)
    
    for i in range(repeats):
        germplasm = np.random.choice(
            a=[False, True],
            size=(50 * factor, n_chr * chr_len, 2),
            p=[0.5, 0.5]
        )
        
        t = timeit.timeit(
            lambda: wheat_schema(simulator, germplasm, factor)[0].block_until_ready(),
            number=1
        )
        times[i] = t
        del germplasm

    print("Mean", np.mean(times[1:]))
    print("Std", np.std(times[1:]))