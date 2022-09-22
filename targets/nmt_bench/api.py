from enum import Enum
import json
import os
from typing import Any, Dict, Optional

from fast_pareto import is_pareto_front

import numpy as np

from targets.nmt_bench.hyperparameters import Hyperparameters, KEY_ORDER
from targets.base_tabularbench_api import BaseTabularBenchAPI


MODULE_PATH = "targets/nmt_bench"
DATA_DIR = f'{os.environ["HOME"]}/tabular_benchmarks/nmt-bench'
N_CONFIGS = 648
PERF_KEY = "bleu"
RUNTIME_KEY = "decoding_time"


class DatasetChoices(Enum):
    so_en = "so-en.json"
    sw_en = "sw-en.json"
    tl_en = "tl-en.json"


class NMTBench(BaseTabularBenchAPI):
    """
    Check our the dataset here:
        https://github.com/Este1le/hpo_nmt
    """

    def __init__(
        self,
        path: str = DATA_DIR,
        dataset: DatasetChoices = DatasetChoices.so_en,
        seed: Optional[int] = None,
    ):
        super().__init__(
            hp_module_path=MODULE_PATH,
            dataset_name=dataset.name,
            seed=seed,
            obj_names=[PERF_KEY, RUNTIME_KEY],
        )
        self._hp_names = list(self._search_space.keys())
        self._path = path

        config2id_key = "config_id_to_idx"
        json_data = json.load(open(os.path.join(path, dataset.value)))
        self._data = {k: np.asarray(v) for k, v in json_data.items() if k != config2id_key}
        self._dataset = dataset
        self._config2id = json_data[config2id_key]

    def _encode_config(self, config: Dict[str, Any]) -> str:
        config_id = ""
        for hp_name in KEY_ORDER:
            idx = self._search_space[hp_name].index(config[hp_name])
            config_id += str(idx)

        return config_id

    def _compute_reference_point(self) -> Dict[str, float]:
        # smaller is better ==> larger is worse
        _sign = {PERF_KEY: -1, RUNTIME_KEY: 1}
        return {
            obj_name: _sign[obj_name] * np.max(_sign[obj_name] * self._data[obj_name]) for obj_name in self._obj_names
        }

    def _compute_pareto_front(self) -> Dict[str, np.ndarray]:
        costs = np.asarray([self._data[PERF_KEY], self._data[RUNTIME_KEY]]).T
        pareto_mask = is_pareto_front(costs, larger_is_better_objectives=[0])
        front_sols = costs[pareto_mask]
        pareto_front = {PERF_KEY: front_sols[:, 0].tolist(), RUNTIME_KEY: front_sols[:, 1].tolist()}
        return pareto_front

    def objective_func(self, config: Dict[str, Any]) -> Dict[str, float]:
        config = Hyperparameters(**config).__dict__
        idx = self._config2id[self._encode_config(config)]
        results = {obj_name: self._data[obj_name][idx] for obj_name in self._obj_names}
        return results

    @property
    def data(self) -> Dict[str, np.ndarray]:
        return self._data

    @property
    def dataset(self) -> DatasetChoices:
        return self._dataset

    @property
    def minimize(self) -> Dict[str, bool]:
        return {PERF_KEY: False, RUNTIME_KEY: True}
