from antibody.antibody import Antibody

import numpy as np
from typing import Callable

class BinaryAntibody(Antibody):
    def __init__(
        self,
        genes: np.ndarray,
        distance_fn: Callable[[np.ndarray, np.ndarray], float]
    ):
        self.genes = genes
        self.distance_fn = distance_fn
    #

    def affinity(self, antigen: np.ndarray) -> float:
        "Compute the affinity between this antibody and a given antigen"
        n = len(self.genes)
        return 1.0 - self.distance(antigen) / n  # ∈ [0, 1]

    def clone(self) -> 'Antibody':
        "Create and return a copy (clone) of this antibody"
        return (
            BinaryAntibodyBuilder()
            .with_genes(self.genes.copy())
            .with_distance_fn(self.distance_fn)
            .build()
        )

    def mutation(self, rate: float) -> 'Antibody':
        "Apply mutation with a given rate and return a new mutated antibody"
        clone = self.clone()
        mask = np.random.rand(len(clone.genes)) < rate
        clone.genes ^= mask  # bitflip
        return clone

    def distance(self, other: np.ndarray) -> float:
        "Compute a distance (similarity metric) between this antibody and another"
        return self.distance_fn(self.genes, other)

    @staticmethod
    def builder() -> 'BinaryAntibodyBuilder':
        return BinaryAntibodyBuilder()
#

class BinaryAntibodyBuilder:
    def __init__(self):
        self._genes = None
        self._distance_fn: Callable[[np.ndarray, np.ndarray], float] = lambda a, b: np.sum(a != b)
    #

    def with_genes(self, genes: np.ndarray) -> 'BinaryAntibodyBuilder':
        self._genes = genes
        return self

    def with_distance_fn(self, distance_fn: Callable[[np.ndarray, np.ndarray], float]) -> 'BinaryAntibodyBuilder':
        self._distance_fn = distance_fn
        return self

    def build(self) -> 'BinaryAntibody':
        if self._genes is None:
            raise ValueError("Genes must be set before building BinaryAntibody")
        return BinaryAntibody(genes=self._genes, distance_fn=self._distance_fn)
#