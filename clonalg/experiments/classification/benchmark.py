import os
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from clonalg.model.clonalg_classification import ClassificationClonalg

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

def get_bench_datasets(samples_per_dataset=1200):
    datasets = {}
    
    X_mnist, y_mnist = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
    y_mnist = y_mnist.astype(int)
    _, X_m_sub, _, y_m_sub = train_test_split(
        X_mnist, y_mnist, test_size=samples_per_dataset, stratify=y_mnist, random_state=42
    )
    datasets["MNIST"] = (X_m_sub, y_m_sub)
    
    X_spam, y_spam = fetch_openml('spambase', version=1, return_X_y=True, as_frame=False, parser='auto')
    y_spam = y_spam.astype(int)
    _, X_s_sub, _, y_s_sub = train_test_split(
        X_spam, y_spam, test_size=samples_per_dataset, stratify=y_spam, random_state=42
    )
    datasets["UCI Spambase"] = (X_s_sub, y_s_sub)
    
    return datasets

def run_uci_benchmark():
    SEEDS = [42, 2026]
    N_FOLDS = 3
    SUB_SAMPLE_SIZE = 1200
    
    datasets = get_bench_datasets(samples_per_dataset=SUB_SAMPLE_SIZE)

    def get_classical_models(seed):
        return {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=seed),
            "k-NN (k=3)": KNeighborsClassifier(n_neighbors=3),
            "Linear SVM": SVC(kernel="linear", random_state=seed),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=seed)
        }

    raw_results = []

    for dataset_name, (X, y) in datasets.items():
        n_classes = len(np.unique(y))
        
        for seed in SEEDS:
            skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
            
            for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                dynamic_bounds = (float(X_train_scaled.min() - 1), float(X_train_scaled.max() + 1))
                
                clonalg = ClassificationClonalg(
                    n_classes=n_classes,
                    population_size=50,  
                    memory_size=12,
                    clone_factor=0.5,
                    n_select=12,
                    n_replace=0,
                    n_generations=150,
                    rho=5,
                    bounds=dynamic_bounds,
                    init="kmeans"
                )
                
                try:
                    clonalg.fit(X_train_scaled, y_train)
                    y_pred_clonalg = clonalg.predict_batch(X_test_scaled, k=1)
                    
                    raw_results.append({
                        "Dataset": dataset_name, "Seed": seed, "Fold": fold_idx,
                        "Model": "CLONALG (Your Algorithm)",
                        "Accuracy": accuracy_score(y_test, y_pred_clonalg),
                        "F1-Score (Macro)": f1_score(y_test, y_pred_clonalg, average="macro", zero_division=0)
                    })
                except Exception:
                    pass

                models = get_classical_models(seed)
                for model_name, model in models.items():
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    
                    raw_results.append({
                        "Dataset": dataset_name, "Seed": seed, "Fold": fold_idx,
                        "Model": model_name,
                        "Accuracy": accuracy_score(y_test, y_pred),
                        "F1-Score (Macro)": f1_score(y_test, y_pred, average="macro", zero_division=0)
                    })

    df_raw = pd.DataFrame(raw_results)
    df_raw.to_csv("uci_benchmark_raw_folds.csv", index=False, encoding="utf-8")
    
    df_summary = df_raw.groupby(["Dataset", "Model"]).agg({
        "Accuracy": ["mean", "std"],
        "F1-Score (Macro)": ["mean", "std"]
    }).reset_index()
    
    df_summary.columns = [
        "Dataset", "Model", 
        "Accuracy (Mean)", "Accuracy (Std)", 
        "F1-Score Mean", "F1-Score Std"
    ]
    
    df_summary.to_csv("uci_benchmark_summary.csv", index=False, encoding="utf-8")

if __name__ == "__main__":
    run_uci_benchmark()