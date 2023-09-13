import numpy as np
import pandas as pd

from chromax import Simulator


class MockSimulator(Simulator):
    def __init__(
        self,
        n_markers=None,
        marker_effects=None,
        recombination_vec=None,
        n_chr=10,
        **kwargs,
    ):
        self.n_markers = n_markers
        if self.n_markers is None:
            if marker_effects is not None:
                self.n_markers = len(marker_effects)
            elif recombination_vec is not None:
                self.n_markers = len(recombination_vec)
            else:
                raise Exception(
                    "You must specify at least one between ",
                    "n_markers, marker_effects and recombination_vec",
                )

        if marker_effects is None:
            marker_effects = np.random.randn(self.n_markers)
        if recombination_vec is None:
            recombination_vec = np.random.uniform(size=self.n_markers)
            recombination_vec /= self.n_markers / 20

        if (
            len(marker_effects) != self.n_markers
            or len(recombination_vec) != self.n_markers
        ):
            raise Exception(
                "Incompatible arguments. ",
                f"Length of marker_effects is {len(marker_effects)}.",
                f"Length of recombination_vec is {len(recombination_vec)}.",
            )

        chromosomes = np.arange(self.n_markers) // (self.n_markers // n_chr)

        data = np.vstack([chromosomes, recombination_vec, marker_effects]).T
        genetic_map = pd.DataFrame(data, columns=["CHR.PHYS", "RecombRate", "Yield"])

        super().__init__(genetic_map=genetic_map, **kwargs)
        self.recombination_vec = recombination_vec

    def load_population(self, n_individual=100, ploidy=2):
        return np.random.choice(
            a=[False, True], size=(n_individual, self.n_markers, ploidy), p=[0.5, 0.5]
        )
