from antibody import Antibody

import numpy as np
from typing import Callable

class RealAntibody(Antibody):
    def __init__(
        self,
        genes: np.ndarray,
        bounds: list[tuple],

        distance_fn: Callable[[np.ndarray, np.ndarray], float]
    ):
        self.genes = genes
        self.bounds = bounds

        self.distance_fn = distance_fn
    #

    def affinity(self, antigen: np.ndarray) -> float:
        "Compute the affinity between this antibody and a given antigen"
        return 1.0 / (1.0 + self.distance(antigen))

    def clone(self) -> 'Antibody':
        "Create and return a copy (clone) of this antibody"

        return (
            RealAntibodyBuilder()
            .with_genes(self.genes)
            .with_bounds(self.bounds)
            .with_distance_fn(self.distance_fn)
            .build()
        )

    def mutation(self, rate) -> 'Antibody':
        "Apply mutation with a given rate and return a new mutated antibody"

        clone = self.clone()
        mask = np.random.rand(len(clone.genes)) < rate
        clone.genes ^= mask  # bitflip
        return clone

    def distance(self, other: 'Antibody') -> float:
        "Compute a distance (similarity metric) between this antibody and another"
        return self.distance_fn(self.genes, other.genes)
#

class RealAntibodyBuilder():
    def __init__(self):
        self._genes = None
        self._bounds: list[tuple] = None

        self._distance_fn: Callable[[np.ndarray, np.ndarray], float] = lambda a, b: np.linalg.norm(a - b)
    #

    def with_genes(self, genes: np.ndarray) -> 'RealAntibodyBuilder':
        self._genes = genes
        return self

    def with_distance_fn(self, distance_fn: Callable[[np.ndarray, np.ndarray], float]) -> 'RealAntibodyBuilder':
        self._distance_fn = distance_fn
        return self

    def with_bounds(self, bounds: list[tuple]) -> 'RealAntibodyBuilder':
        self._bounds = bounds
        return self

    def build(self):
        if self._genes is None:
            raise ValueError("Genes must be set before building RealAntibody")
        if self._bounds is None:
            raise ValueError("Bounds must be set before building RealAntibody")

        for i, gene in enumerate(self._genes):
            if not(self._bounds[i][0] <= gene <= self._bounds[i][1]):
                raise ValueError(f"Gene at index {i} = {gene} is out of bounds {self._bounds[i]}")

        return RealAntibody(genes=self._genes, bounds=self._bounds, distance_fn=self._distance_fn)
#