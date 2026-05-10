from clonalg.antibody.antibody import Antibody

import numpy as np
from typing import Callable


class PermutationAntibody(Antibody):
    def __init__(
        self,
        genes: np.ndarray,
        distance_fn: Callable[[np.ndarray, np.ndarray], float],
        cost_fn: Callable[[np.ndarray], float] = None,
    ):
        self.genes = genes
        self.distance_fn = distance_fn
        self.cost_fn = cost_fn

    #

    def affinity(self, antigen: np.ndarray = None) -> float:
        "Compute the affinity. Uses cost_fn (optimization mode) if set, else distance to antigen."
        if self.cost_fn is not None:
            return 1.0 / (1.0 + self.cost_fn(self.genes))
        return 1.0 / (1.0 + self.distance(antigen))

    def clone(self) -> "Antibody":
        "Create and return a copy (clone) of this antibody"
        return (
            PermutationAntibodyBuilder()
            .with_genes(self.genes.copy())
            .with_distance_fn(self.distance_fn)
            .with_cost_fn(self.cost_fn)
            .build()
        )

    def mutation(self, rate: float) -> "Antibody":
        "Apply mutation with a given rate and return a new mutated antibody"
        clone = self.clone()
        n = len(clone.genes)
        n_ops = max(1, int(rate * n))
        for _ in range(n_ops):
            i, j = sorted(np.random.choice(n, 2, replace=False))
            clone.genes[i : j + 1] = clone.genes[i : j + 1][::-1]  # 2-opt reverse
        return clone

    def distance(self, other: np.ndarray) -> float:
        "Compute a distance (similarity metric) between this antibody and another"
        return self.distance_fn(self.genes, other)

    @staticmethod
    def builder() -> "PermutationAntibodyBuilder":
        return PermutationAntibodyBuilder()


#


class PermutationAntibodyBuilder:
    def __init__(self):
        self._genes = None
        self._distance_fn: Callable[[np.ndarray, np.ndarray], float] = lambda a, b: (
            np.sum(a != b)
        )
        self._cost_fn: Callable[[np.ndarray], float] = None

    #

    def with_genes(self, genes: np.ndarray) -> "PermutationAntibodyBuilder":
        self._genes = genes
        return self

    def with_distance_fn(
        self, distance_fn: Callable[[np.ndarray, np.ndarray], float]
    ) -> "PermutationAntibodyBuilder":
        self._distance_fn = distance_fn
        return self

    def with_cost_fn(
        self, cost_fn: Callable[[np.ndarray], float]
    ) -> "PermutationAntibodyBuilder":
        self._cost_fn = cost_fn
        return self

    def build(self):
        if self._genes is None:
            raise ValueError("Genes must be set before building PermutationAntibody")

        return PermutationAntibody(
            genes=self._genes,
            distance_fn=self._distance_fn,
            cost_fn=self._cost_fn,
        )


#
