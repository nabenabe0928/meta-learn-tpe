import json
from typing import List, Tuple

from viz.constants import (
    N_OBJ,
    N_RUNS,
    N_SAMPLES,
)

import numpy as np


def get_costs(obj_names: List[str], dataset_name: str, opt_name: str) -> None:
    n_samples = N_SAMPLES
    costs = np.empty((N_RUNS, n_samples, N_OBJ))
    for i in range(N_RUNS):
        data = json.load(open(f"results/{dataset_name}/{opt_name}/{i:0>2}.json"))
        costs[i, :, 0] = data[obj_names[0]][:n_samples]
        costs[i, :, 1] = data[obj_names[1]][:n_samples]

    return costs


def get_true_pareto_front_and_ref_point(
    obj_names: List[str],
    bench_name: str,
    dataset_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    _bench_name = "nmt_bench" if bench_name == "nmt" else bench_name
    data = json.load(open(f"targets/{_bench_name}/pareto-fronts/{dataset_name}.json"))
    ref_point = np.asarray([
        json.load(open(f"targets/{_bench_name}/reference-point/{dataset_name}.json"))[name]
        for name in obj_names
    ])
    true_pf = np.empty((len(data[obj_names[0]]), N_OBJ))
    true_pf[:, 0], true_pf[:, 1] = data[obj_names[0]], data[obj_names[1]]
    return true_pf, ref_point
