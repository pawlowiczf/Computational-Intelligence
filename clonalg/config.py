import numpy as np
from dataclasses import dataclass, field
from typing import Callable

@dataclass
class Config():
    fitness_fn: Callable[[np.ndarray], float]
    dimensions: int = 10
    bounds: tuple[float, float] = (0.0, 1.0)
    population_size: int = 30
    n_best = 8
    memory_size: int = 5
    swap_percentage: int = 10
    tolerance: float = 1e-6
    patience: int = 20

    def __post_init__(self):
        if not callable(self.fitness_fn):
            raise TypeError("fitness_fn must be callable")
        if self.dimensions <= 0:
            raise ValueError("dimensions must be positive")
        if not (0 <= self.swap_percentage <= 100):
            raise ValueError("swap_percentage must be between 0 and 100")
        if self.bounds[0] >= self.bounds[1]:
            raise ValueError("bounds must be (min, max) with min < max")
    #
#