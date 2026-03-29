import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from clonalg_classifier import CLONALGClassifier
from antibody.real_antibody import RealAntibody

data = load_wine()
X, y = data.data, data.target

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

n_features = 13
bounds = [(0.0, 1.0) for _ in range(n_features)]

def wine_antibody_factory():
    return RealAntibody.builder() \
        .with_genes(np.random.uniform(0, 1, size=n_features)) \
        .with_bounds(bounds) \
        .with_distance_fn(lambda a, b: np.linalg.norm(a - b)) \
        .build()

classifier = CLONALGClassifier(
    n_classes=3,
    population_size=30,
    clone_factor=0.2,
    n_select=5,
    n_replace=10,
    n_generations=20,
    memory_size=5,
    antibody_factory=wine_antibody_factory,
    p=10.0,
    verbose=True,
)

classifier.fit(X_train, y_train)

predictions = classifier.predict_batch(X_test)
print(classification_report(y_test, predictions, target_names=data.target_names))