from clonalg.clonalg import CLONALG
from antibody.antibody import Antibody
import numpy as np

# TODO

class CLONALGClassifier:
    def __init__(self, n_classes: int, **clonalg_kwargs):
        # One independent CLONALG instance per class
        self.models: dict[int, CLONALG] = {
            label: CLONALG(**clonalg_kwargs)
            for label in range(n_classes)
        }

    def fit(self, X: list[np.ndarray], y: list[int]):
        # Group antigens by class label
        antigens_per_class: dict[int, list[np.ndarray]] = {
            label: [] for label in self.models
        }
        for x, label in zip(X, y):
            antigens_per_class[label].append(x)

        # Train each CLONALG only on its own class antigens
        for label, model in self.models.items():
            model.run(antigens_per_class[label])

    def predict(self, x: np.ndarray) -> int:
        # Ask each population: how well do you recognize this?
        best_affinity = -1
        best_label = -1

        for label, model in self.models.items():
            population = model.memory if model.memory else model.population
            # Best affinity from this class's population
            affinities = [ab.affinity(x) for ab in population]
            class_affinity = max(affinities)

            if class_affinity > best_affinity:
                best_affinity = class_affinity
                best_label = label

        return best_label

    def predict_batch(self, X: list[np.ndarray]) -> list[int]:
        return [self.predict(x) for x in X]
#

classifier = CLONALGClassifier(
    n_classes=3,
    population_size=50,
    clone_factor=0.1,
    n_select=10,
    n_replace=5,
    n_generations=20,
    memory_size=10,
    antibody_factory=lambda: Antibody.random(n_features=4),
)

classifier.fit(X_train, y_train)
predictions = classifier.predict_batch(X_test)