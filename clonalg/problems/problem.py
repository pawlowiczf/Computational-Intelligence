import numpy as np
from abc import ABC, abstractmethod


class Problem(ABC):
    @abstractmethod
    def __call__(self, x: np.ndarray) -> float:
        "Compute the function value at point x."
        raise NotImplementedError

    @abstractmethod
    def grad(self, x: np.ndarray) -> np.ndarray:
        "Compute the gradient at point x."
        raise NotImplementedError


class Sphere(Problem):
    def __call__(self, x: np.ndarray) -> float:
        return np.sum(x**2)

    def grad(self, x: np.ndarray) -> np.ndarray:
        return 2 * x


class Rosenbrock(Problem):
    def __call__(self, x: np.ndarray) -> float:
        return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

    def grad(self, x: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(x)
        n = x.size
        grad[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
        for i in range(1, n - 1):
            grad[i] = (
                200 * (x[i] - x[i - 1] ** 2)
                - 400 * x[i] * (x[i + 1] - x[i] ** 2)
                - 2 * (1 - x[i])
            )
        grad[-1] = 200 * (x[-1] - x[-2] ** 2)
        return grad


class Rastrigin(Problem):
    def __call__(self, x: np.ndarray) -> float:
        A = 10
        n = x.size
        return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))

    def grad(self, x: np.ndarray) -> np.ndarray:
        A = 10
        return 2 * x + 2 * np.pi * A * np.sin(2 * np.pi * x)


#
