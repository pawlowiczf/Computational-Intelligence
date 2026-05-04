from abc import ABC, abstractmethod
import numpy as np

class Antibody(ABC):

    @abstractmethod
    def affinity(self, antigen: np.ndarray = None) -> float:
        "Compute the affinity between this antibody and a given antigen (or via cost_fn in optimization mode)"
        ...

    @abstractmethod
    def clone(self) -> 'Antibody':
        "Create and return a copy (clone) of this antibody"
        ...

    @abstractmethod
    def mutation(self, rate) -> 'Antibody':
        "Apply mutation with a given rate and return a new mutated antibody"
        ...

    @abstractmethod
    def distance(self, other: np.ndarray) -> float:
        "Compute a distance (similarity metric) between this antibody and another"
        ...
#