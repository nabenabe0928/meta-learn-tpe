import json
import os
from enum import Enum
from typing import Dict, List, Optional, Type

from fast_pareto import nondominated_rank

import numpy as np

from optimizers import RandomOptimizer

from targets.base_tabularbench_api import BaseTabularBenchAPI


def save_observations(file_path: str, observations: Dict[str, np.ndarray], include: Optional[List[str]] = None) -> None:
    subdirs = "/".join(file_path.split("/")[:-1])
    os.makedirs(subdirs, exist_ok=True)

    with open(file_path, mode="w") as f:
        json.dump({k: v.tolist() for k, v in observations.items() if include is None or k in include}, f, indent=4)


def get_result_file_path(dataset_name: str, opt_name: str, seed: int) -> str:
    return f"results/{dataset_name}/{opt_name}/{seed:0>2}.json"


def collect_metadata(
    benchmark: Type[BaseTabularBenchAPI],
    dataset_choices: Enum,
    max_evals: int,
    seed: int,
    exclude: str,
) -> Dict[str, np.ndarray]:

    metadata: Dict[str, Dict[str, np.ndarray]] = {}
    for dataset in dataset_choices:
        if dataset.name == exclude:
            continue
        file_path = get_result_file_path(dataset.name, opt_name="random", seed=seed)
        if os.path.exists(file_path):
            data = {k: np.asarray(v) for k, v in json.load(open(file_path)).items()}
        else:
            bm = benchmark(dataset=dataset)
            opt = RandomOptimizer(
                obj_func=bm.obj_func,
                config_space=bm.config_space,
                objective_names=bm.obj_names,
                max_evals=max_evals,
            )
            opt.optimize()
            data = opt.fetch_observations()
            save_observations(file_path=file_path, observations=data, include=None)

        metadata[dataset.name] = data

    return metadata


def select_warm_start_configs(
    metadata: Dict[str, Dict[str, np.ndarray]],
    n_configs: int,
    hp_names: List[str],
    obj_names: List[str],
    seed: Optional[int] = None,
    larger_is_better_objectives: Optional[List[int]] = None,
) -> Dict[str, np.ndarray]:
    n_tasks = len(metadata)
    n_top = (n_configs + n_tasks - 1) // n_tasks
    _configs: Dict[str, List[np.ndarray]] = {name: [] for name in hp_names + obj_names}
    for data in metadata.values():
        costs = np.asarray([data[obj_name] for obj_name in obj_names]).T
        top_indices = np.argsort(
            nondominated_rank(costs=costs, tie_break=True, larger_is_better_objectives=larger_is_better_objectives)
        )[:n_top]
        for name in hp_names + obj_names:
            _configs[name].append(data[name][top_indices])
    else:
        rng = np.random.RandomState(seed)
        indices = rng.choice(np.arange(n_top * n_tasks), n_configs, replace=False)
        configs: Dict[str, np.ndarray] = {name: np.hstack(v)[indices] for name, v in _configs.items()}

    return configs
