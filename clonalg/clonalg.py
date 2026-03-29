import numpy as np
from antibody.antibody import Antibody
from typing import Callable

class CLONALG:
    def __init__(
        self,
        population_size: int,       # N - population size
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

        self.memory: list[Antibody] = []  # M - memory cells
        self.population: list[Antibody] = [
            antibody_factory() for _ in range(population_size)
        ]
    #

    def _select_and_clone(self) -> list[Antibody]:
        "Clone the n best individuals proportionally to affinity rank"
        clones = []
        for i, antibody in enumerate(self.population[:self.n_select]):
            n_clones = int(np.ceil(self.clone_factor * self.population_size / (i + 1)))
            clones.extend([antibody.clone() for _ in range(n_clones)])
        return clones

    def _hypermutate(self, clones: list[Antibody], antigen: np.ndarray) -> list[Antibody]:
        "Mutate clones - rate inversely proportional to affinity"
        mutated = []
        for clone in clones:
            affinity = clone.affinity(antigen)
            rate = 1.0 - affinity  # higher affinity → lower mutation rate
            mutated.append(clone.mutation(rate))
        return mutated

    def _update_memory(self, matured: list[Antibody], antigen: np.ndarray):
        "Update memory set with best matured clones"
        candidates = self.memory + matured
        candidates.sort(key=lambda ab: ab.affinity(antigen), reverse=True)
        self.memory = candidates[:self.memory_size]

    def _replace_weakest(self):
        "Replace the n_replace weakest antibodies with new random ones"
        self.population[-self.n_replace:] = [
            self.antibody_factory() for _ in range(self.n_replace)
        ]

    def run(self, antigens: list[np.ndarray]) -> list[Antibody]:
        for gen in range(self.n_generations):
            for antigen in antigens:
                # 1. P = Pr + M
                self.population = self.population + self.memory

                # 2. Select n best
                self.population.sort(key=lambda ab: ab.affinity(antigen), reverse=True)
                self.population = self.population[:self.population_size]

                # 3. Clone n best proportionally to rank
                clones = self._select_and_clone(antigen)

                # 4. Hypermutate clones
                matured = self._hypermutate(clones, antigen)

                # 5. Update memory with best matured clones
                self._update_memory(matured, antigen)

                # 6. Replace weakest with random antibodies
                self._replace_weakest()

        return self.memory if self.memory else self.population
    #
#