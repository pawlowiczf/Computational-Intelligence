from antibody import Antibody

import numpy as np
from typing import Callable



class ClonALG:
    def __init__(
        self,
        fitness_fn: Callable[[np.ndarray], float],
        dimensions: int = 10,
        bounds: tuple[float, float] = (0.0, 1.0),
        population_size: int = 30,
        n_best = 8,
        memory_size: int = 5,
        swap_percentage: int = 10,
        tolerance: float = 1e-6,
        patience: int = 20
    ):
        self.fitness_fn = fitness_fn
        self.population_size = population_size
        self.n_best = n_best
        self.memory_size = memory_size
        self.swap_percentage = swap_percentage
        self.tolerance = tolerance
        self.patience = patience
        self.dimensions = dimensions
        self.bounds = bounds
        self.beta = 1.0
        self.P: list[Antibody] = [Antibody.random(dimensions, bounds) for _ in range(population_size)]
        self.M: list[Antibody] = []
    #

    def _evaluate(self, population: list[Antibody]) -> None:
        for ab in population:
            ab.affinity = self.fitness_fn(ab.genes)
    #

    def _hypermutation_rate(self, affinity: float) -> float:
        return np.exp(-self.beta * affinity)
    #

    def _mutate(self, ab: Antibody) -> Antibody:
        rate = self._hypermutation_rate(ab.affinity)
        noise = np.random.normal(0, rate, self.dimensions)
        mutant = Antibody(ab.genes + noise, self.bounds)
        return mutant
    #

    def _clone_count(self, rank: int) -> int:
        return max(1, round(self.beta * self.population_size / (rank + 1)))
    #