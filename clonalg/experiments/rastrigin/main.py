import numpy as np

from clonalg.antibody.real_antibody import RealAntibody, RealAntibodyBuilder
from clonalg.model.clonalg_optimization import OptimizationClonalg
from clonalg.problems.problem import Rastrigin
from clonalg.problems.visualization import plot_3d_surface, plot_contour_and_paths

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

clonalg = OptimizationClonalg(
    population_size=20,
    clone_factor=0.2,       # beta — n_clones = ceil(beta * N / rank)
    n_replace=5,            # replace 5 weakest each generation
    n_generations=10000,
    antibody_factory=factory,
)

memory = clonalg.run()  # antigen ignored

memory.sort(key=lambda ab: ab.affinity(None), reverse=True)
best = memory[0]

paths = list(map(lambda x: np.array(x.genes).reshape(1, 2), memory))

print(f"Best solution: x = {best.genes}")
print(f"f(x)         = {rastrigin(best.genes):.6f}")
print(f"affinity     = {best.affinity(None):.6f}")

plot_3d_surface(rastrigin)
plot_contour_and_paths(rastrigin, paths)

input("Press Enter to exit...")
