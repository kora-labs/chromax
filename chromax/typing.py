from jaxtyping import Array, Bool

N_MARKERS = "m"
DIPLOID_SHAPE = N_MARKERS + "d"

Haploid = Bool[Array, N_MARKERS]
Individual = Bool[Array, DIPLOID_SHAPE]


class _MetaPopulation(type):
    def __getitem__(cls, n: str):
        return Bool[Array, f"{n} {DIPLOID_SHAPE}"]


class Population(metaclass=_MetaPopulation):
    pass


class _MetaParents(type):
    def __getitem__(cls, n: str):
        return Population[f"{n} 2"]


class Parents(metaclass=_MetaParents):
    pass
