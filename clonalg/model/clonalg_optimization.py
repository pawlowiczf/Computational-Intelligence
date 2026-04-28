from typing_extensions import deprecated

import numpy as np

from clonalg.antibody.antibody import Antibody
from typing import Callable

"""
In a optimization version of Clonalg algorithm, there are a few differences:
1. There is no explicit Ag population to be recognized, but an objective function to be optimized. Ab affinity (that is distance between Ab and Ag) discards Ag and corresponds to the evaluation of the function. Each Ab represents the input (element) of the input space.
2. The whole Ab population composes the memory set, it is no longer necessary to store a separate memory set.
"""

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
        """
        Clone each antibody and apply hypermutation inversely proportional to affinity.
        High affinity (good solution) -> small mutation step -> local exploitation.
        Low affinity  (poor solution) -> large mutation step-> global exploration.

        Note: number of clones is fixed for all antibodies (no rank proportionality).
        """
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

    @deprecated
    def _replace_weakest(self):
        # Population must be sorted descending before calling this
        self.population[-self.n_replace:] = [
            self.antibody_factory() for _ in range(self.n_replace)
        ]

    def _suppress_and_refill(self, sigma_s):
        """
        Apply suppression niching to maintain population diversity,
        then refill suppressed slots with new random antibodies.

        Antibodies are considered in descending order of affinity (best first).
        A candidate is accepted only if it lies at least `sigma_s` away from
        every already-accepted survivor, ensuring no two survivors occupy the
        same niche. Suppressed slots are replaced with random antibodies to
        encourage continued exploration of the search space.

        Args:
            sigma_s:
                Suppression threshold - minimum Euclidean distance
                required between any two survivors.
        """
        survivors = []
        for ab in sorted(self.population, key=lambda x: -x.affinity(None)):
            if all(np.linalg.norm(ab.genes - s.genes) >= sigma_s for s in survivors):
                survivors.append(ab)
        while len(survivors) < self.population_size:
            survivors.append(self.antibody_factory())
        self.population = survivors

    def run(self) -> list[Antibody]:
        """
        Run the optimization version of CLONALG for a fixed number of generations.
        Each generation applies cloning, hypermutation, selection, and suppression niching
        to drive the population toward multiple optima of the objective function.
        """
        for _ in range(self.n_generations):
            # Clone each Ab N_c times and apply hypermutation
            # (mutation rate inversely proportional to affinity: α = exp(-ρ · f))
            clone_groups = self._clone_and_mutate()

            # Select the best clone from each group and replace parent if improved
            # (each parent competes only with its own clones — no cross-group competition)
            for i, clones in enumerate(clone_groups):
                best_clone = max(clones, key=lambda ab: ab.affinity(None))
                if best_clone.affinity(None) > self.population[i].affinity(None):
                    self.population[i] = best_clone

            # Sort population by affinity descending (best first)
            # required before suppression, which accepts candidates greedily from the top
            self.population.sort(key=lambda ab: ab.affinity(None), reverse=True)

            # Apply suppression niching to remove redundant Ab's occupying
            # the same niche, then refill vacated slots with new random Ab's
            self._suppress_and_refill(0.1)

        return self.population