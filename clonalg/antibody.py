import numpy as np

class Antibody:
    def __init__(self, genes: np.ndarray, bounds: tuple[float, float] = (0.0, 1.0)):
        self.genes = genes
        self.bounds = bounds
        self.affinity: float = 0.0

    @classmethod
    def random(cls, dim: int, bounds=(0.0, 1.0)) -> 'Antibody':
        genes = np.random.uniform(bounds[0], bounds[1], dim)
        return cls(genes, bounds)

    def clone(self) -> 'Antibody':
        return Antibody(self.genes.copy(), self.bounds)
#