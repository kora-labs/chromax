"""Utility module for typing."""
from jaxtyping import Array, Bool


N_MARKERS = "m"
DIPLOID_SHAPE = N_MARKERS + " d"

Haploid = Bool[Array, N_MARKERS]
Individual = Bool[Array, DIPLOID_SHAPE]


class _MetaPopulation(type):
    def __getitem__(cls, n: str):
        return Bool[Array, f"{n} {DIPLOID_SHAPE}"]


class Population(metaclass=_MetaPopulation):
    """Typing class representing a population of n individuals."""

    pass


class _MetaParents(type):
    def __getitem__(cls, n: str):
        return Population[f"{n} 2"]


class Parents(metaclass=_MetaParents):
    """Typing class representing a pair of individuals."""

    pass
