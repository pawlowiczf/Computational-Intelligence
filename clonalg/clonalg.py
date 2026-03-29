import numpy as np
from antibody.antibody import Antibody
from typing import Callable

class CLONALG:
    def __init__(
        self,
        population_size: int,       # N - total population size (Ab_r + Ab_m)
        clone_factor: float,        # beta - clone multiplier
        n_select: int,              # n - how many best to clone each generation
        n_replace: int,             # d - how many weakest to replace with random
        n_generations: int,
        memory_size: int,           # |M| - number of memory cells
        antibody_factory: Callable[[], Antibody],
    ):
        self.population_size = population_size
        self.clone_factor = clone_factor
        self.n_select = n_select
        self.n_replace = n_replace
        self.n_generations = n_generations
        self.memory_size = memory_size
        self.antibody_factory = antibody_factory

        self.memory: list[Antibody] = []        # Ab_m - memory cells
        self.population: list[Antibody] = [     # Ab_r - remainder
            antibody_factory() for _ in range(population_size)
        ]

    def _clone_and_mutate(
        self,
        selected: list[Antibody],
        antigen: np.ndarray,
    ) -> list[list[Antibody]]:
        """Steps 4-5: clone and hypermutate, grouped by parent antibody.

        Returns a list of clone groups - one group per selected parent.
        Higher affinity parents get more clones, lower affinity = higher mutation rate.
        """
        clone_groups = []
        for i, parent in enumerate(selected):
            # Step 4: number of clones proportional to affinity rank
            n_clones = int(np.ceil(self.clone_factor * self.population_size / (i + 1)))

            # Step 5: hypermutate each clone - rate inversely proportional to affinity
            matured = []
            for clone in [parent.clone() for _ in range(n_clones)]:
                affinity = clone.affinity(antigen)
                rate = 1.0 - affinity  # higher affinity → lower mutation rate
                matured.append(clone.mutation(rate))

            clone_groups.append(matured)

        return clone_groups

    def _update_memory(
        self,
        clone_groups: list[list[Antibody]],
        antigen: np.ndarray,
    ):
        """Step 7: best clone from each group competes with worst memory Ab.

        If the best clone has higher affinity than the worst memory cell,
        it replaces it. If memory is not full yet, clone is added unconditionally.
        """
        for clones in clone_groups:
            if not clones:
                continue

            best_clone = max(clones, key=lambda ab: ab.affinity(antigen))
            best_affinity = best_clone.affinity(antigen)

            if len(self.memory) < self.memory_size:
                # Memory not full yet - just add
                self.memory.append(best_clone)
            else:
                # Find the worst memory Ab and compete with it
                worst_idx = min(
                    range(len(self.memory)),
                    key=lambda i: self.memory[i].affinity(antigen)
                )
                if best_affinity > self.memory[worst_idx].affinity(antigen):
                    self.memory[worst_idx] = best_clone

    def _replace_weakest(self, antigen: np.ndarray):
        """Step 8: replace d lowest affinity Ab's from Ab_r with new random ones."""
        # Population (Ab_r) must be sorted descending before calling this
        self.population[-self.n_replace:] = [
            self.antibody_factory() for _ in range(self.n_replace)
        ]

    def run(self, antigens: list[np.ndarray]) -> list[Antibody]:
        for gen in range(self.n_generations):
            antigen = antigens[np.random.randint(len(antigens))]
            combined = self.population + self.memory

            # Steps 2-3: compute affinities, select n best from combined Ab
            combined.sort(key=lambda ab: ab.affinity(antigen), reverse=True)
            selected = combined[:self.n_select]

            # Steps 4-5: clone and hypermutate, grouped by parent
            clone_groups = self._clone_and_mutate(selected, antigen)

            # Steps 6-7: best clone per group competes with respective memory Ab
            self._update_memory(clone_groups, antigen)

            # Step 8: replace d lowest affinity Ab's from Ab_r only
            self.population.sort(key=lambda ab: ab.affinity(antigen), reverse=True)
            self._replace_weakest(antigen)

        return self.memory if self.memory else self.population