import numpy as np
from clonalg.antibody.antibody import Antibody
from typing import Callable


class PatternClonalg:
    def __init__(
        self,
        population_size: int,  # N - total population size (Ab_r + Ab_m)
        clone_factor: float,  # beta - clone multiplier
        n_select: int,  # n - how many best to clone each generation
        n_replace: int,  # d - how many weakest to replace with random
        n_generations: int,
        memory_size: int,  # |M| - number of memory cells
        antibody_factory: Callable[[], Antibody],
        rho: float = 1.0,  # decay rate: alpha = exp(-rho * f)
    ):
        self.population_size = population_size
        self.clone_factor = clone_factor
        self.n_select = n_select
        self.n_replace = n_replace
        self.n_generations = n_generations
        self.memory_size = memory_size
        self.antibody_factory = antibody_factory
        self.rho = rho

        self.memory: list[Antibody] = []  # Ab_m - memory cells
        self.population: list[Antibody] = [  # Ab_r - remainder
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
                rate = np.exp(-self.rho * affinity)
                matured.append(clone.mutation(rate))

            clone_groups.append(matured)

        return clone_groups

    def _update_memory(
        self,
        clone_groups: list[list[Antibody]],
        memory_indices: list[int | None],
        antigen: np.ndarray,
    ):
        """Step 7: best clone from each group competes with its respective memory Ab.

        If the parent came from Ab_m, the best clone competes with that exact memory slot.
        If the parent came from Ab_r, the best clone competes with the worst memory cell.
        """
        for clones, mem_idx in zip(clone_groups, memory_indices):
            if not clones:
                continue

            best_clone = max(clones, key=lambda ab: ab.affinity(antigen))
            best_affinity = best_clone.affinity(antigen)

            if len(self.memory) < self.memory_size:
                self.memory.append(best_clone)
            elif mem_idx is not None:
                # Parent came from Ab_m - compete with the respective memory slot
                if best_affinity > self.memory[mem_idx].affinity(antigen):
                    self.memory[mem_idx] = best_clone
            else:
                # Parent came from Ab_r - compete with worst memory Ab
                worst_idx = min(
                    range(len(self.memory)),
                    key=lambda i: self.memory[i].affinity(antigen),
                )
                if best_affinity > self.memory[worst_idx].affinity(antigen):
                    self.memory[worst_idx] = best_clone

    def _replace_weakest(self, antigen: np.ndarray):
        """Step 8: replace d lowest affinity Ab's from Ab_r with new random ones."""
        # Population (Ab_r) must be sorted descending before calling this
        self.population[-self.n_replace :] = [
            self.antibody_factory() for _ in range(self.n_replace)
        ]

    def run(self, antigens: list[np.ndarray]) -> list[Antibody]:
        for gen in range(self.n_generations):
            antigen = antigens[np.random.randint(len(antigens))]

            # Steps 2-3: compute affinities, select n best from Ab = Ab_r ∪ Ab_m
            # Track memory index for each Ab so step 7 knows the respective slot
            pop_tagged = [(ab, None) for ab in self.population]
            mem_tagged = [(ab, i) for i, ab in enumerate(self.memory)]
            tagged = pop_tagged + mem_tagged
            tagged.sort(key=lambda t: t[0].affinity(antigen), reverse=True)
            selected_tagged = tagged[: self.n_select]
            selected = [ab for ab, _ in selected_tagged]
            memory_indices = [idx for _, idx in selected_tagged]

            # Steps 4-5: clone and hypermutate, grouped by parent
            clone_groups = self._clone_and_mutate(selected, antigen)

            # Steps 6-7: best clone per group competes with respective memory Ab
            self._update_memory(clone_groups, memory_indices, antigen)

            # Step 8: replace d lowest affinity Ab's from Ab_r only
            self.population.sort(key=lambda ab: ab.affinity(antigen), reverse=True)
            self._replace_weakest(antigen)

        return self.memory if self.memory else self.population
