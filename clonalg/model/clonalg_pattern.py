import numpy as np
from clonalg.antibody.antibody import Antibody
from typing import Callable

"""
In the pattern recognition version of the CLONALG algorithm:
1. An explicit Ag (antigen) population is available for recognition.
2. The antibody repertoire (Ab) is explicitly decomposed into a memory set (Ab_m) and a remaining pool (Ab_r).
3. The goal is to perform learning and memory acquisition by producing individuals with increasing affinities (maturation of the immune response).
"""

class PatternClonalg:

    def __init__(
        self,
        population_size: int, # N - total available Ab repertoire size (Ab = Ab_r U Ab_m)
        clone_factor: float, # beta - multiplying factor for clone size calculation
        n_select: int, # n - number of highest affinity Ab's to select for cloning
        n_replace: int, # d - amount of lowest affinity Ab's from Ab_r to be replaced by newcomers
        n_generations: int, # N_gen - predefined maximum number of generations (stopping criterion)
        memory_size: int, # m - number of memory cells in Ab_m
        antibody_factory: Callable[[], Antibody],
        rho: float = 1.0, # rho - controls the decay of the mutation step size: alpha = exp(-rho * f)
    ):
        self.population_size = population_size
        self.clone_factor = clone_factor
        self.n_select = n_select
        self.n_replace = n_replace
        self.n_generations = n_generations
        self.memory_size = memory_size
        self.antibody_factory = antibody_factory
        self.rho = rho

        # Ab_m - Memory Ab repertoire
        self.memory: list[Antibody] = [
            antibody_factory() for _ in range(memory_size)
        ]
        # Ab_r - Remaining Ab repertoire
        self.population: list[Antibody] = [
            antibody_factory() for _ in range(population_size - memory_size)
        ]

    def _clone_and_mutate(
        self,
        selected: list[Antibody],
        antigen: np.ndarray,
    ) -> list[Antibody]:
        """
        Steps 4 and 5: Cloning and Affinity Maturation.

        Step 4: The selected antibodies are cloned independently and proportionally
        to their antigenic affinities. The clone size is determined by
        round(beta * N / i), where i is the rank of the selected Ab (1-indexed).

        Step 5: The resulting clone repertoire is submitted to an affinity maturation
        (hypermutation) process inversely proportional to the antigenic affinity.
        The mutation rate is alpha = exp(-rho * f).
        The antigenic affinity 'f' is normalized over the interval [0, 1].
        """
        if not selected:
            return []

        all_matured_clones = []
        affinities = np.array([ab.affinity(antigen) for ab in selected])
        f_min, f_max = affinities.min(), affinities.max()

        if f_max == f_min:
            f_norm = np.ones_like(affinities)
        else:
            f_norm = (affinities - f_min) / (f_max - f_min)

        for i, (parent, norm_aff) in enumerate(zip(selected, f_norm)):
            n_clones = int(round(self.clone_factor * self.population_size / (i + 1)))
            rate = np.exp(-self.rho * norm_aff)
            for _ in range(n_clones):
                all_matured_clones.append(parent.mutation(rate))

        return all_matured_clones

    def _update_memory(
        self,
        matured_clones: list[Antibody],
        antigen: np.ndarray,
    ):
        """
        Step 7: Memory Reselection.

        Selects the best clone from the matured set to be a candidate for the memory pool.
        If its affinity is higher than the respective memory Ab, it replaces the memory Ab.

        Implementation Note:
        The paper describes replacing the "respective memory Ab" associated explicitly
        with Ag_j. Since this code doesn't strictly track Ag_j indices, it dynamically
        finds the memory cell that currently has the highest affinity for the presented antigen
        and defines that as the respective memory cell to compete against.
        """
        if not matured_clones:
            return

        best_clone = max(matured_clones, key=lambda ab: ab.affinity(antigen))
        best_clone_affinity = best_clone.affinity(antigen)

        respective_memory_idx = max(
            range(len(self.memory)),
            key=lambda i: self.memory[i].affinity(antigen)
        )

        if best_clone_affinity > self.memory[respective_memory_idx].affinity(antigen):
            self.memory[respective_memory_idx] = best_clone

    def _replace_weakest(self):
        """
        Step 8: Receptor Editing / Diversity Maintenance.

        Replace the 'd' lowest affinity antibodies from the remaining pool (Ab_r)
        with new random individuals.
        """
        self.population[-self.n_replace:] = [
            self.antibody_factory() for _ in range(self.n_replace)
        ]

    def run(self, antigens: list[np.ndarray]) -> list[Antibody]:
        for gen in range(self.n_generations):
            perm = np.random.permutation(len(antigens))
            for idx in perm:
                # Randomly select an antigen Ag_j from the set of antigens
                antigen = antigens[idx]
                # Determine affinity of all Ab's (Ab_r + Ab_m) to the antigen and select the 'n' highest affinity Ab's
                tagged = self.population + self.memory
                tagged.sort(key=lambda ab: ab.affinity(antigen), reverse=True)
                selected = tagged[:self.n_select]

                # Clone selected Ab's proportionally to affinity, then hypermutate
                matured_clones = self._clone_and_mutate(selected, antigen)

                # Determine clone affinities and reselect the best candidate to enter memory
                self._update_memory(matured_clones, antigen)

                # Replace the 'd' lowest affinity Ab's in the remaining repertoire (Ab_r)
                self.population.sort(key=lambda ab: ab.affinity(antigen), reverse=True)
                self._replace_weakest()

        return self.memory