import numpy as np
from antibody.real_antibody import RealAntibody, RealAntibodyBuilder
from clonalg import CLONALG

from problems.problem import Rastrigin
from problems.visualization import *

rastrigin = Rastrigin()
n_dims = 2
bounds = [(-5.12, 5.12)] * n_dims

def factory() -> RealAntibody:
    genes = np.array([np.random.uniform(lo, hi) for lo, hi in bounds])
    return (
        RealAntibodyBuilder()
        .with_genes(genes)
        .with_bounds(bounds)
        .with_distance_fn(lambda a, b: rastrigin(a))  # b ignored, affinity = 1 / (1 + f(a))
        .build()
    )

clonalg = CLONALG(
    population_size=50,
    clone_factor=0.2,       # beta — n_clones = ceil(beta * N / rank)
    n_select=10,            # clone 10 best individuals each generation
    n_replace=5,            # replace 5 weakest each generation
    n_generations=100,
    memory_size=10,         # keep 10 best solutions in memory
    antibody_factory=factory,
)

memory = clonalg.run(antigens=[np.zeros(n_dims)])  # antigen ignored

# --- results ---
dummy = np.zeros(n_dims)
memory.sort(key=lambda ab: ab.affinity(dummy), reverse=True)
best = memory[0]

print(f"Best solution: x = {best.genes}")
print(f"f(x)         = {rastrigin(best.genes):.6f}")
print(f"affinity     = {best.affinity(dummy):.6f}")

plot_3d_surface(rastrigin)