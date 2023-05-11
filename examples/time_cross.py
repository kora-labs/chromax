from chromax import Simulator
import timeit
import numpy as np
import pandas as pd
import os
import jax

if __name__ == "__main__":    
    repeats = 100
    n_chr = 10
    chr_len = 100
    n_crosses = 1_000
    
    genetic_map = pd.DataFrame({
        "CHR.PHYS": np.arange(n_chr * chr_len) // chr_len,
        "Yield": np.random.standard_normal(n_chr * chr_len),
        "RecombRate": np.full(n_chr * chr_len, 1.5 / chr_len)
    })
    simulator = Simulator(genetic_map=genetic_map)
    
    ts = np.empty(repeats)
    for i in range(repeats):
        germplasm = np.random.choice(
            a=np.array([False, True]),
            size=(1000, n_chr * chr_len, 2),
        )
        germplasm = jax.device_put(germplasm, device=simulator.device)
        plan = np.random.randint(0, 1000, size=(n_crosses, 2))
        plan = jax.device_put(plan, device=simulator.device)


        t = timeit.timeit(
            lambda: simulator.cross(germplasm[plan]).block_until_ready(),
            number=1
        )
        ts[i] = t * 1000
        
        del germplasm
        del plan

    print(ts)
    print("Mean", np.mean(ts[1:]))
    print("Std", np.std(ts[1:]))