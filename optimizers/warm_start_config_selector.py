from typing import Dict, List, Optional

from fast_pareto import nondominated_rank

import numpy as np


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
