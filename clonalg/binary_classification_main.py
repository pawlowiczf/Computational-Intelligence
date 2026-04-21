import numpy as np
from antibody.binary_antibody import BinaryAntibody, BinaryAntibodyBuilder
from clonalg.model.clonalg_pattern import CLONALG

# --- antigens: binary patterns to recognize ---
antigens = [
    np.array([1, 0, 1, 0, 1, 0, 1, 0]),  # alternating
    np.array([1, 1, 1, 1, 0, 0, 0, 0]),  # half-half
    np.array([0, 0, 0, 0, 0, 0, 0, 1]),  # single bit
]

n_bits = len(antigens[0])

# --- factory: random binary antibody ---
def factory() -> BinaryAntibody:
    genes = np.random.randint(0, 2, size=n_bits)
    return (
        BinaryAntibodyBuilder()
        .with_genes(genes)
        .with_distance_fn(lambda a, b: np.sum(a != b))
        .build()
    )

# --- run CLONALG ---
clonalg = CLONALG(
    population_size=30,
    clone_factor=0.3,
    n_replace=3,
    n_generations=1000,
    antibody_factory=factory,
)

population = clonalg.run(antigens=antigens)

# --- results: best antibody per antigen ---
print("Best match per antigen:")
for antigen in antigens:
    best = max(population, key=lambda ab: ab.affinity(antigen))
    print(f"  antigen : {antigen}")
    print(f"  antibody: {best.genes}")
    print(f"  affinity: {best.affinity(antigen):.3f}")
    print(f"  hamming : {int(np.sum(best.genes != antigen))}/{n_bits}")
    print()
