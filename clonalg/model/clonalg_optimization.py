import numpy as np

from clonalg.antibody.antibody import Antibody
from typing import Callable

class OptimizationClonalg:
    def __init__(
        self,
        population_size: int,   # N - whole population is memory (no separate Ab_m)
        clone_factor: float,    # beta - clone multiplier
        n_replace: int,         # d - how many weakest to replace with random
        n_generations: int,
        antibody_factory: Callable[[], Antibody],
        rho: float = 1.0,       # decay rate: alpha = exp(-rho * f)
    ):
        self.population_size = population_size
        self.clone_factor = clone_factor
        self.n_replace = n_replace
        self.n_generations = n_generations
        self.antibody_factory = antibody_factory
        self.rho = rho

        self.population: list[Antibody] = [
            antibody_factory() for _ in range(population_size)
        ]

    def _clone_and_mutate(self) -> list[list[Antibody]]:
        # N_c = round(beta * N) - same for all Ab's (no rank proportionality)
        n_clones = round(self.clone_factor * self.population_size)
        clone_groups = []
        for parent in self.population:
            matured = []
            for clone in [parent.clone() for _ in range(n_clones)]:
                affinity = clone.affinity(None)
                rate = np.exp(-self.rho * affinity)
                matured.append(clone.mutation(rate))
            clone_groups.append(matured)
        return clone_groups

    def _replace_weakest(self):
        # Population must be sorted descending before calling this
        self.population[-self.n_replace:] = [
            self.antibody_factory() for _ in range(self.n_replace)
        ]

    def run(self) -> list[Antibody]:
        for _ in range(self.n_generations):
            # Steps 4-5: clone all N Ab's with equal number of clones
            clone_groups = self._clone_and_mutate()

            # Step 7: n Ab's selected from clones - best clone replaces parent if better
            for i, clones in enumerate(clone_groups):
                best_clone = max(clones, key=lambda ab: ab.affinity(None))
                if best_clone.affinity(None) > self.population[i].affinity(None):
                    self.population[i] = best_clone

            # Step 8: replace d lowest affinity Ab's with new random ones
            self.population.sort(key=lambda ab: ab.affinity(None), reverse=True)
            self._replace_weakest()

        return self.population