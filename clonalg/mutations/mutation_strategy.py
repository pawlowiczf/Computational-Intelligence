from abc import ABC, abstractmethod

class MutationStrategy(ABC):
    @abstractmethod
    def mutate(self) -> float:
        pass