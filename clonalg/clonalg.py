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
        p: float = 5.0,             # p - exponent for mutation rate
        verbose: bool = False,
    ):
        self.population_size = population_size
        self.clone_factor = clone_factor
        self.n_select = n_select
        self.n_replace = n_replace
        self.n_generations = n_generations
        self.memory_size = memory_size
        self.antibody_factory = antibody_factory
        self.p = p
        self.verbose = verbose

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
        "Mutate clones - rate exponentially decreasing with affinity"
        if not clones: return []
        
        affs = np.array([c.affinity(antigen) for c in clones])
        f_min, f_max = affs.min(), affs.max()
        
        # Normalize affinities to [0, 1]
        if f_max == f_min:
            f_norm = np.ones_like(affs)
        else:
            f_norm = (affs - f_min) / (f_max - f_min)

        mutated = []
        for i, clone in enumerate(clones):
            rate = np.exp(-self.p * f_norm[i])
            mutated.append(clone.mutation(rate))
        return mutated

    def _update_memory(self, matured: list[Antibody], antigen: np.ndarray):
        "Update memory set with best matured clone vs respective memory cell"
        best_matured = max(matured, key=lambda ab: ab.affinity(antigen))
        
        if not self.memory:
            self.memory.append(best_matured)
            return

        distances = [best_matured.distance(m.genes) for m in self.memory]
        idx = np.argmin(distances) # respective Ab index in memory
        
        if best_matured.affinity(antigen) > self.memory[idx].affinity(antigen):
            self.memory[idx] = best_matured
        elif len(self.memory) < self.memory_size:
            self.memory.append(best_matured)

    def _replace_weakest(self, antigen: np.ndarray):
        "Replace the n_replace weakest antibodies with new random ones"
        self.population.sort(key=lambda ab: ab.affinity(antigen), reverse=True)
        
        self.population[-self.n_replace:] = [self.antibody_factory() for _ in range(self.n_replace)]

    def run(self, antigens: list[np.ndarray]) -> list[Antibody]:
        for gen in range(self.n_generations):
            for antigen in antigens:
                # 1. P = Pr + M
                self.population = self.population + self.memory

                # 2. Select n best
                self.population.sort(key=lambda ab: ab.affinity(antigen), reverse=True)
                self.population = self.population[:self.population_size]

                # 3. Clone n best proportionally to rank
                clones = self._select_and_clone()

                # 4. Hypermutate clones
                matured = self._hypermutate(clones, antigen)

                # 5. Update memory with best matured clones
                self._update_memory(matured, antigen)

                # 6. Replace weakest with random antibodies
                self._replace_weakest(antigen)

            if self.verbose:
                print(f"Generation {gen + 1}/{self.n_generations} completed. Memory best affinity: {max(ab.affinity(antigen) for ab in self.memory):.4f}")
        return self.memory if self.memory else self.population
    #
#