from typing_extensions import deprecated

import numpy as np
from tqdm import tqdm

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
        population_size: int,  # N - whole population is memory (no separate Ab_m)
        clone_factor: float,  # beta - clone multiplier
        n_generations: int,
        antibody_factory: Callable[[], Antibody],
        rho: float = 1.0,  # decay rate: alpha = exp(-rho * f)
        suppression_threshold: float = 0.1,  # sigma_s - min distance between survivors (in the metric defined by Antibody.distance)
        hypermutation_strategy: str = "rank",  # "rank" (scale-invariant, generic) or "affinity" (uses raw affinity, requires well-scaled cost)
    ):
        if hypermutation_strategy not in ("rank", "affinity"):
            raise ValueError(
                f"hypermutation_strategy must be 'rank' or 'affinity', got {hypermutation_strategy!r}"
            )

        self.population_size = population_size
        self.clone_factor = clone_factor
        self.n_generations = n_generations
        self.antibody_factory = antibody_factory
        self.rho = rho
        self.suppression_threshold = suppression_threshold
        self.hypermutation_strategy = hypermutation_strategy

        self.population: list[Antibody] = [
            antibody_factory() for _ in range(population_size)
        ]

    def _clone_and_mutate(self) -> list[list[Antibody]]:
        """
        Clone each antibody and apply hypermutation.

        The mutation rate assigned to each parent depends on the selected
        `hypermutation_strategy`:

        - "rank":
            rate = exp(-rho * (N - 1 - rank) / (N - 1))

            Rank-based scaling independent of absolute fitness values.
            Robust to arbitrary cost magnitudes (e.g. TSP, knapsack problems).
            Ignores absolute differences between fitness values and relies only on ordering.

        - "affinity":
            rate = exp(-rho * affinity)

            Uses raw affinity values to modulate mutation strength.
            Preserves continuous fitness information, allowing smoother adaptation
            as the population converges and affinities become more informative locally.

            Requires properly scaled affinity values (typically normalized to a bounded range).
            If affinity values are too large or poorly scaled, mutation rates may saturate
            (often collapsing toward a constant value), reducing selection pressure.

        Behavioral interpretation:
        - High affinity / low rank (good solutions) → low mutation rate → exploitation
        - Low affinity / high rank (poor solutions) → high mutation rate → exploration

        Note:
        The number of clones per antibody is constant.
        Rank affects only mutation intensity, not clone count distribution.
        """
        # N_c = round(beta * N) - same for all Ab's (no rank proportionality)
        n_clones = round(self.clone_factor * self.population_size)
        N = self.population_size

        affinities = [ab.affinity(None) for ab in self.population]

        if self.hypermutation_strategy == "rank":
            # rank 0 = best (highest affinity)
            order = sorted(range(N), key=lambda i: -affinities[i])
            rank = [0] * N
            for r, idx in enumerate(order):
                rank[idx] = r
            scores = [(N - 1 - rank[i]) / max(1, N - 1) for i in range(N)]
        else:  # "affinity"
            scores = affinities

        clone_groups = []
        for i, parent in enumerate(self.population):
            rate = np.exp(-self.rho * scores[i])
            matured = [parent.clone().mutation(rate) for _ in range(n_clones)]
            clone_groups.append(matured)
        return clone_groups

    # @deprecated("Use another method instead")
    # def _replace_weakest(self):
    #     # Population must be sorted descending before calling this
    #     self.population[-self.n_replace :] = [
    #         self.antibody_factory() for _ in range(self.n_replace)
    #     ]

    def _select_best_clones(self, clone_groups: list[list[Antibody]]) -> None:
        for i, clones in enumerate(clone_groups):
            best_clone = max(clones, key=lambda ab: ab.affinity(None))
            if best_clone.affinity(None) > self.population[i].affinity(None):
                self.population[i] = best_clone

    def _suppress_and_refill(self, sigma_s):
        """
        Apply suppression niching to maintain population diversity,
        then refill suppressed slots with new random antibodies.

        Antibodies are considered in descending order of affinity (best first).
        A candidate is accepted only if it lies at least `sigma_s` away from
        every already-accepted survivor, ensuring no two survivors occupy the
        same niche. Suppressed slots are replaced with random antibodies to
        encourage continued exploration of the search space.

        Distance is computed via `Antibody.distance`, so the metric is
        defined by the concrete antibody type (Euclidean for real-valued,
        Hamming for binary/permutation, etc.).

        Args:
            sigma_s:
                Suppression threshold - minimum distance (in the antibody's
                own metric) required between any two survivors.
        """
        survivors = []
        for ab in sorted(self.population, key=lambda x: -x.affinity(None)):
            if all(ab.distance(s.genes) >= sigma_s for s in survivors):
                survivors.append(ab)
        while len(survivors) < self.population_size:
            survivors.append(self.antibody_factory())
        self.population = survivors

    def run(self, verbose: bool = True) -> list[Antibody]:
        """
        Run the optimization version of CLONALG for a fixed number of generations.
        Each generation applies cloning, hypermutation, selection, and suppression niching
        to drive the population toward multiple optima of the objective function.
        """
        for _ in tqdm(range(self.n_generations), disable=not verbose):
            # Clone each Ab N_c times and apply hypermutation
            # (mutation rate inversely proportional to affinity: α = exp(-ρ · f))
            clone_groups = self._clone_and_mutate()

            # Select the best clone from each group and replace parent if improved
            # (each parent competes only with its own clones - no cross-group competition)
            self._select_best_clones(clone_groups)

            # Sort population by affinity descending (best first)
            # required before suppression, which accepts candidates greedily from the top
            self.population.sort(key=lambda ab: ab.affinity(None), reverse=True)

            # Apply suppression niching to remove redundant Ab's occupying
            # the same niche, then refill vacated slots with new random Ab's
            self._suppress_and_refill(self.suppression_threshold)

        return self.population
