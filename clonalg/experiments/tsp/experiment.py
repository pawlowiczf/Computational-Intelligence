"""Run CLONALG TSP experiment(s) and persist results to disk.

Hyperparameters live in the EXPERIMENTS dict below; CLI only chooses where
to save and which problems / how many workers to use.

Examples:
    python experiment.py --output-dir ./results
    python experiment.py --output-dir ./results --problem bayg29 berlin52
    python experiment.py --output-dir ./results --n-workers 8
"""

import argparse
import itertools
import json
import sys
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend - safe under multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import tsplib95

SCRIPT_DIR = Path(__file__).resolve().parent


def _find_project_root(start: Path) -> Path:
    "Walk upward from `start` to find the directory containing the `clonalg` package."
    for d in [start, *start.parents]:
        if (d / "clonalg" / "__init__.py").is_file():
            return d
    raise RuntimeError(
        f"Could not locate the 'clonalg' package starting from {start}. "
        "Make sure the package directory is somewhere on the path from the script."
    )


PROJECT_ROOT = _find_project_root(SCRIPT_DIR)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clonalg.antibody.permutation_antibody import (
    PermutationAntibody,
    PermutationAntibodyBuilder,
)
from clonalg.model.clonalg_optimization import OptimizationClonalg

DATA_DIR = PROJECT_ROOT / "clonalg" / "experiments" / "tsp" / "data"


# ---------------------------------------------------------------------------
# Hyperparameter grid per TSP instance. Each value is a list of candidates;
# the runner takes the Cartesian product (itertools.product) — one run per
# combination, one subfolder per run.
#
# n_runs (per combo) is realised via the `seed` list — N seeds = N repetitions.
# ---------------------------------------------------------------------------
EXPERIMENTS: dict[str, dict[str, list]] = {
    "bayg29": {
        "population_size": [20, 40, 80],
        "clone_factor": [0.15, 0.2, 0.3],
        "rho": [3.0, 5.0],
        "suppression_threshold": [2.0, 4.0],
        "n_generations": [1000],
        "hypermutation_strategy": ["rank"],
        "seed": [0, 1, 2],
    },
    "bays29": {
        "population_size": [20, 40, 80],
        "clone_factor": [0.15, 0.2, 0.25],
        "rho": [3.0, 4.0, 5.0],
        "suppression_threshold": [2.0, 3.0, 4.0],
        "n_generations": [1000],
        "hypermutation_strategy": ["rank"],
        "seed": [0, 1, 2],
    },
    "berlin52": {
        "population_size": [40, 80, 120],
        "clone_factor": [0.15, 0.2, 0.3],
        "rho": [3.0, 4.0, 5.0],
        "suppression_threshold": [2.0, 3.0, 4.0],
        "n_generations": [2000],
        "hypermutation_strategy": ["rank"],
        "seed": [0, 1, 2],
    },
    "pr124": {
        "population_size": [80, 120, 160],
        "clone_factor": [0.1, 0.15],
        "rho": [5.0, 6.0],
        "suppression_threshold": [4.0, 5.0, 6.0],
        "n_generations": [2000],
        "hypermutation_strategy": ["rank"],
        "seed": [0, 1],
    },
}


def expand_grid(grid: dict[str, list]) -> list[dict]:
    "Cartesian product of a {param: [values]} grid -> list of {param: value} dicts."
    keys = list(grid.keys())
    return [dict(zip(keys, values)) for values in itertools.product(*grid.values())]


def load_solutions(filepath: Path) -> dict[str, int]:
    solutions: dict[str, int] = {}
    for line in filepath.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, rest = line.partition(":")
        solutions[name.strip().lower()] = int(rest.strip())
    return solutions


def tour_cost(tour: np.ndarray, problem: tsplib95.models.StandardProblem) -> int:
    n = len(tour)
    return sum(
        problem.get_weight(int(tour[i]), int(tour[(i + 1) % n])) for i in range(n)
    )


def tour_edges(genes: np.ndarray) -> set:
    tour = np.concatenate(([1], genes))
    n = len(tour)
    return {frozenset((int(tour[i]), int(tour[(i + 1) % n]))) for i in range(n)}


def edge_distance(a: np.ndarray, b: np.ndarray) -> int:
    return len(tour_edges(a) - tour_edges(b))


def gap(found: float, optimal: int) -> float:
    return (found - optimal) / optimal * 100


def make_factory(problem: tsplib95.models.StandardProblem):
    n_cities = problem.dimension

    def factory() -> PermutationAntibody:
        genes = np.random.permutation(np.arange(2, n_cities + 1))
        return (
            PermutationAntibodyBuilder()
            .with_genes(genes)
            .with_cost_fn(lambda g: tour_cost(np.concatenate(([1], g)), problem))
            .with_distance_fn(edge_distance)
            .build()
        )

    return factory


def plot_tsp(coords: np.ndarray, tour: np.ndarray, title: str, save_path: Path) -> None:
    full_tour = np.concatenate(([0], tour, [0]))
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(len(full_tour) - 1):
        a, b = full_tour[i], full_tour[i + 1]
        ax.plot(
            [coords[a, 0], coords[b, 0]],
            [coords[a, 1], coords[b, 1]],
            "b-",
            linewidth=0.8,
            alpha=0.6,
        )
    ax.scatter(coords[:, 0], coords[:, 1], c="red", s=20, zorder=3)
    for i, (x, y) in enumerate(coords):
        ax.annotate(str(i), (x, y), fontsize=7, ha="right")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def auto_run_name(params: dict) -> str:
    return (
        f"pop{params['population_size']}"
        f"_beta{params['clone_factor']}"
        f"_rho{params['rho']}"
        f"_sigma{params['suppression_threshold']}"
        f"_gen{params['n_generations']}"
        f"_{params['hypermutation_strategy']}"
        f"_seed{params['seed']}"
    )


def run_experiment(
    problem_name: str,
    output_dir: Path,
    run_name: str,
    population_size: int,
    clone_factor: float,
    rho: float,
    suppression_threshold: float,
    n_generations: int,
    hypermutation_strategy: str = "rank",
    seed: int | None = None,
    verbose: bool = False,
) -> dict:
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if seed is not None:
        np.random.seed(seed)

    problem = tsplib95.load(str(DATA_DIR / f"{problem_name}.tsp"))
    solutions = load_solutions(DATA_DIR / "solutions")
    optimal = solutions.get(problem_name.lower())

    factory = make_factory(problem)

    clonalg = OptimizationClonalg(
        population_size=population_size,
        clone_factor=clone_factor,
        rho=rho,
        suppression_threshold=suppression_threshold,
        antibody_factory=factory,
        n_generations=n_generations,
        hypermutation_strategy=hypermutation_strategy,
    )

    config = {
        "problem_name": problem_name,
        "population_size": population_size,
        "clone_factor": clone_factor,
        "rho": rho,
        "suppression_threshold": suppression_threshold,
        "n_generations": n_generations,
        "hypermutation_strategy": hypermutation_strategy,
        "seed": seed,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    start = time.perf_counter()
    memory = clonalg.run(verbose=verbose)
    elapsed = time.perf_counter() - start

    memory.sort(key=lambda ab: ab.affinity(None), reverse=True)
    best = memory[0]
    best_tour = np.concatenate(([1], best.genes))
    cost = int(tour_cost(best_tour, problem))

    results = {
        "problem_name": problem_name,
        "run_name": run_name,
        "tour_cost": cost,
        "optimal_cost": optimal,
        "gap_percent": gap(cost, optimal) if optimal is not None else None,
        "affinity": float(best.affinity(None)),
        "runtime_seconds": elapsed,
        "n_cities": problem.dimension,
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    np.savetxt(run_dir / "best_tour.txt", best_tour.astype(int), fmt="%d")

    coords_dict = problem.node_coords or problem.display_data
    if coords_dict:
        coords = np.array(
            [coords_dict[i + 1] for i in range(problem.dimension)], dtype=float
        )
        title = f"{problem_name} - cost {cost}"
        if optimal is not None:
            title += f" (optimal {optimal}, gap {results['gap_percent']:.2f}%)"
        plot_tsp(coords, best.genes - 1, title, run_dir / "plot.png")

    return results


def _worker(task: dict) -> dict:
    try:
        return run_experiment(**task)
    except Exception as exc:
        return {
            "problem_name": task.get("problem_name"),
            "run_name": task.get("run_name"),
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }


def _build_tasks(problems: list[str], output_dir: Path) -> list[dict]:
    tasks: list[dict] = []
    for prob in problems:
        if prob not in EXPERIMENTS:
            raise SystemExit(
                f"No EXPERIMENTS entry for {prob!r}. Known: {list(EXPERIMENTS)}"
            )
        for i, cfg in enumerate(expand_grid(EXPERIMENTS[prob])):
            task = {
                **cfg,
                "verbose": False,
                "problem_name": prob,
                "output_dir": output_dir / prob,
            }
            task["run_name"] = auto_run_name(task) or f"run_{i:04d}"
            tasks.append(task)
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="CLONALG TSP experiment runner.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root folder for the whole experiment.",
    )
    parser.add_argument(
        "--problem",
        nargs="+",
        default=None,
        help="Problem(s) from EXPERIMENTS to run (default: all).",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Worker processes (default: cpu_count).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    problems = args.problem or list(EXPERIMENTS)
    tasks = _build_tasks(problems, args.output_dir)

    import multiprocessing as mp

    print(f"Launching {len(tasks)} runs across {args.n_workers or 'cpu_count'} workers")
    summary_files: dict[str, "object"] = {}
    try:
        with mp.Pool(processes=args.n_workers) as pool:
            for i, res in enumerate(pool.imap_unordered(_worker, tasks), start=1):
                prob = res.get("problem_name", "_unknown")
                if prob not in summary_files:
                    (args.output_dir / prob).mkdir(parents=True, exist_ok=True)
                    summary_files[prob] = (
                        args.output_dir / prob / "summary.jsonl"
                    ).open("w")
                summary_files[prob].write(json.dumps(res) + "\n")
                summary_files[prob].flush()

                tag = f"{prob}/{res.get('run_name')}"
                if "error" in res:
                    print(f"[{i}/{len(tasks)}] FAIL {tag}: {res['error']}")
                else:
                    print(
                        f"[{i}/{len(tasks)}] {tag}: cost={res['tour_cost']} "
                        f"gap={res.get('gap_percent')} time={res['runtime_seconds']:.1f}s"
                    )
    finally:
        for f in summary_files.values():
            f.close()
    for prob in summary_files:
        print(f"Summary: {args.output_dir / prob / 'summary.jsonl'}")


if __name__ == "__main__":
    main()
