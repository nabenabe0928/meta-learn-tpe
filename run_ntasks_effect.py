import os
import json
from typing import Dict, List, Tuple

import numpy as np

from optimizers.tpe.optimizer import RandomOptimizer, TPEOptimizer

from targets.ellipsoid.func import Ellipsoid


DIM = 10
MAX_EVALS = 200
N_INDEPENDENT_RUNS = 20
N_METADATA = 100
N_INIT = MAX_EVALS * 5 // 100
N_WARMSTART = N_METADATA * 5 // 100
QUANTILE = 0.1
LOSS_NAME = "loss"
TARGET_BENCH = Ellipsoid(center=0.0, dim=DIM)

TPE_PARAMS = dict(
    obj_func=TARGET_BENCH.objective_func,
    config_space=TARGET_BENCH.config_space,
    max_evals=MAX_EVALS,
    n_init=N_INIT,
    quantile=QUANTILE,
)


def get_metadata(task_ids: List[int], seed: int, center: float) -> Dict[str, Dict[str, np.ndarray]]:
    metadata = {}
    for task_id in task_ids:
        bench = Ellipsoid(center=center, dim=DIM)
        opt = RandomOptimizer(
            obj_func=bench.objective_func,
            config_space=bench.config_space,
            max_evals=N_METADATA,
            seed=seed,
        )
        opt.optimize()
        data = opt.fetch_observations()
        data.pop(opt._runtime_name)
        metadata[f"center={task_id:0>2}"] = data

    return metadata


def select_warmstart_configs(
    task_ids: List[int],
    metadata: Dict[str, Dict[str, np.ndarray]],
    seed: int,
) -> Dict[str, np.ndarray]:
    warmstart_configs: Dict[str, np.ndarray] = {k: [] for k in next(iter(metadata.values())).keys()}

    for task_id in task_ids:
        data = metadata[f"center={task_id:0>2}"]
        order = np.argsort(data[LOSS_NAME])[:N_WARMSTART]
        for k, v in data.items():
            warmstart_configs[k].append(v[order])
    else:
        rng = np.random.RandomState(seed)
        warmstart_configs = {
            k: rng.choice(np.hstack(v), replace=False, size=N_WARMSTART)
            for k, v in warmstart_configs.items()
        }

    return warmstart_configs


def optimize_by_tpe(seed) -> Tuple[np.ndarray, np.ndarray]:
    opt = TPEOptimizer(seed=seed, **TPE_PARAMS)
    opt.optimize()
    return opt.fetch_observations()[LOSS_NAME]


def optimize_by_metalearn_tpe(
    seed: int,
    metadata: Dict[str, Dict[str, np.ndarray]],
    warmstart_configs: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    opt = TPEOptimizer(
        seed=seed,
        dim_reduction_factor=2.5,
        metadata=metadata,
        warmstart_configs=warmstart_configs,
        **TPE_PARAMS,
    )
    opt.optimize()
    return opt.fetch_observations()[LOSS_NAME]


def collect_data(seed, center):
    results = {0: optimize_by_tpe(seed)}
    metadata = get_metadata(task_ids=np.arange(8), seed=seed, center=center)
    for task_ids in [np.arange(1), np.arange(2), np.arange(4), np.arange(8)]:
        print(seed, task_ids)
        warmstart_configs = select_warmstart_configs(task_ids, metadata, seed)
        loss = optimize_by_metalearn_tpe(
            seed=seed, metadata={f"center={i:0>2}": metadata[f"center={i:0>2}"] for i in range(task_ids.size)},
            warmstart_configs=warmstart_configs
        )
        results[task_ids.size] = loss

    return results


if __name__ == "__main__":
    dir_name = "extra-results"
    os.makedirs(dir_name, exist_ok=True)
    for center in [0.0, 1.0, 2.0, 3.0]:
        print(f"Center: {center}")
        for seed in range(N_INDEPENDENT_RUNS):
            print(f"Seed: {seed}")
            results = collect_data(seed, center)
            path = os.path.join(dir_name, f"center={int(center)}_seed{seed:0>2}.json")
            with open(path, mode="w") as f:
                json.dump({str(k): v.tolist() for k, v in results.items()}, f, indent=4)
