import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple
from tqdm import tqdm
from sklearn.cluster import KMeans

def _train_clonalg_class(args: Tuple) -> Tuple[int, np.ndarray]:
    """
    Independent training function for a single class (cluster).
    Extracted outside the main class to allow serialization by ProcessPoolExecutor.
    Executes the 8 steps of the pattern recognition CLONALG algorithm.
    """
    (label, X_class, pop_size, mem_size, clone_factor, 
     n_select, n_replace, n_gen, rho, bounds, init) = args
    
    np.random.seed()

    dim = X_class.shape[1]
    pop_rem_size = pop_size - mem_size
    
    # INITIALIZATION
    # Memory pool (Ab_m): Initialized using Heuristic Initialization (KMeans centroids).
    if init == "kmeans":
        km = KMeans(n_clusters=mem_size, n_init=1, max_iter=50)
        km.fit(X_class)
        memory = km.cluster_centers_.copy()
    elif init == "random":
        memory = np.random.uniform(bounds[0], bounds[1], (mem_size, dim))
    elif init == "sample":
        sampled_indices = np.random.choice(X_class.shape[0], size=mem_size, replace=False)
        memory = X_class[sampled_indices].copy()
    
    # Remaining pool (Ab_r): Initialized randomly within defined bounds.
    population = np.random.uniform(bounds[0], bounds[1], (pop_rem_size, dim))
    
    # Pre-calculate the number of clones for each rank (1 to n_select).
    # Step 4 formula: N_c = round(beta * N / i), where i is the rank.
    ranks = np.arange(1, n_select + 1)
    n_clones_per_rank = np.round(clone_factor * pop_size / ranks).astype(int)

    feature_std = np.maximum(X_class.std(axis=0), (bounds[1] - bounds[0]) * 0.1)
    
    for gen in range(n_gen):
        perm = np.random.permutation(X_class.shape[0])
        
        for idx in perm:
            # Step 1: Randomly choose an antigen Ag_j (and present it to all Ab's).
            antigen = X_class[idx]
            
            # Step 2: Determine the affinity of Ag_j to all N Ab's in the repertoire.
            # Affinity is inversely proportional to Euclidean distance.
            combined = np.vstack((memory, population))
            dists = np.linalg.norm(combined - antigen, axis=1)
            affinities = 1.0 / (1 + dists)
            
            # Step 3: Select the n highest affinity Ab's to compose a new set Ab_n.
            best_idx = np.argsort(affinities)[::-1][:n_select]
            selected = combined[best_idx]
            selected_affs = affinities[best_idx]
            
            # Step 4: Clone selected Ab's proportionally to their antigenic affinities.
            # The higher the affinity (lower rank index), the higher the number of clones.
            clones = np.repeat(selected, n_clones_per_rank, axis=0)
            
            # Step 5: Submit the clone repertoire to an affinity maturation (hypermutation) process.
            # Mutation rate alpha is inversely proportional to affinity: alpha = exp(-rho * f).
            f_min, f_max = affinities.min(), affinities.max()
            if f_max > f_min:
                f_norm = (selected_affs - f_min) / (f_max - f_min)
            else:
                f_norm = np.ones_like(selected_affs)
                
            rates = np.exp(-rho * f_norm)
            clone_rates = np.repeat(rates, n_clones_per_rank)[:, np.newaxis]
            
            # Dynamic Sigma: Standard deviation of the Gaussian noise is scaled by the mutation rate 
            # and the total width of the allowed variable bounds.
            sigma = clone_rates * feature_std
                    
            # Apply Gaussian mutation
            noise = np.random.normal(0, 1, size=clones.shape)
            matured_clones = clones + (noise * sigma)
            
            # Ensure mutated genes do not violate the shape space boundaries
            matured_clones = np.clip(matured_clones, bounds[0], bounds[1])
            
            # Step 6: Determine the affinity of the matured clones C* in relation to Ag_j.
            matured_dists = np.linalg.norm(matured_clones - antigen, axis=1)
            matured_affs = 1.0 / (1 + matured_dists)
            best_clone_idx = np.argmax(matured_affs)
            
            # Evaluate current respective memory cell affinity
            mem_dists = np.linalg.norm(memory - antigen, axis=1)
            mem_affs = 1.0 / (1 + mem_dists)
            best_mem_idx = np.argmax(mem_affs)
            
            # Step 7: Reselect the best matured clone to enter the memory pool.
            # If its affinity is larger than the respective memory Ab, it replaces it.
            if matured_affs[best_clone_idx] > mem_affs[best_mem_idx]:
                memory[best_mem_idx] = matured_clones[best_clone_idx]
                
            # Step 8: Replace the d lowest affinity Ab's from Ab_r by new individuals.
            # This simulates receptor editing and introduces diversity.
            pop_dists = np.linalg.norm(population - antigen, axis=1)
            pop_affs = 1.0 / (1 + pop_dists)
            
            # Extract indices of the worst 'n_replace' (d) antibodies in the remaining pool
            worst_idx = np.argpartition(pop_affs, n_replace)[:n_replace]
            
            # Replace them with newly generated random individuals
            population[worst_idx] = np.random.uniform(bounds[0], bounds[1], (n_replace, dim))

    return label, memory


class ClassificationClonalg:
    '''
    init:
        - kmeans: Use KMeans centroids for heuristic initialization of the memory pool.
        - random: Randomly initialize the memory pool within the defined bounds.
        - sample: Randomly sample initial memory cells from the training data of the class.
    '''
    def __init__(
        self, 
        n_classes: int,
        population_size: int,       # N - total available Ab repertoire size (Ab = Ab_r U Ab_m)
        clone_factor: float,        # beta - multiplying factor for clone size calculation
        n_select: int,              # n - number of highest affinity Ab's to select for cloning
        n_replace: int,             # d - amount of low-affinity Ab's to be replaced
        n_generations: int,         # N_gen - predefined maximum number of generations
        memory_size: int,           # m - number of memory cells in Ab_m
        rho: float = 1.0,           # rho - controls the decay of the mutation step size
        bounds: Tuple[float, float] = (-5.0, 5.0), # Boundary constraints for the shape space
        init: str = "kmeans"   # Whether to use heuristic initialization (KMeans) for memory pool
    ):
    
        if n_classes <= 0:
            raise ValueError("n_classes must be greater than 0")
        
        if population_size < memory_size + n_replace:
            raise ValueError("population_size must be >= memory_size + n_replace")

        self.n_classes = n_classes
        self.pop_size = population_size
        self.clone_factor = clone_factor
        self.n_select = n_select
        self.n_replace = n_replace
        self.n_generations = n_generations
        self.memory_size = memory_size
        self.rho = rho
        self.bounds = bounds
        self.init = init
        
        # Dictionary storing the trained memory matrices (Ab_m) for each class
        self.models: Dict[int, np.ndarray] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClassificationClonalg":
        """
        Train the classification model by spawning a separate CLONALG process for each class.
        Each process is exposed exclusively to the antigens (samples) of its target class.
        """
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

        antigens_per_class: Dict[int, List[np.ndarray]] = {
            label: [] for label in range(self.n_classes)
        }

        for x_val, label in zip(X, y):
            if label not in antigens_per_class:
                raise ValueError(f"label {label} is out of range [0, {self.n_classes - 1}]")
            antigens_per_class[label].append(x_val)

        jobs = []
        for label, class_antigens in antigens_per_class.items():
            if class_antigens:
                X_matrix = np.array(class_antigens)
                jobs.append((
                    label, X_matrix, self.pop_size, self.memory_size, 
                    self.clone_factor, self.n_select, self.n_replace, 
                    self.n_generations, self.rho, self.bounds, self.init
                ))

        if len(jobs) <= 1:
            # Single-threaded execution
            for job in jobs:
                label, memory_matrix = _train_clonalg_class(job)
                self.models[label] = memory_matrix
            return self

        # Multi-process execution for parallel learning of distinct classes
        with ProcessPoolExecutor(max_workers=len(jobs)) as executor:
            futures = [executor.submit(_train_clonalg_class, job) for job in jobs]
            
            for future in tqdm(as_completed(futures), total=len(jobs)):
                label, memory_matrix = future.result()
                self.models[label] = memory_matrix

        return self

    def predict(self, x: np.ndarray, k: int = 1) -> int:
        if not self.models:
            raise RuntimeError("Model is not fitted yet.")

        all_affinities = []
        all_labels = []

        for label, memory_matrix in self.models.items():
            dists = np.linalg.norm(memory_matrix - x, axis=1)
            affs = 1.0 / (1.0 + dists) 
            all_affinities.extend(affs)
            all_labels.extend([label] * len(affs))

        all_affinities = np.array(all_affinities)
        all_labels = np.array(all_labels)

        if k < len(all_affinities):
            top_k_indices = np.argpartition(all_affinities, -k)[-k:]
            top_k_labels = all_labels[top_k_indices]
            top_k_affs = all_affinities[top_k_indices]
        else:
            top_k_labels = all_labels
            top_k_affs = all_affinities

        class_scores = {}
        for label, aff in zip(top_k_labels, top_k_affs):
            class_scores[label] = class_scores.get(label, 0) + aff

        best_label = max(class_scores, key=class_scores.get)
        return best_label

    def predict_batch(self, X: np.ndarray, k: int = 1) -> np.ndarray:
        return np.array([self.predict(x, k) for x in X])