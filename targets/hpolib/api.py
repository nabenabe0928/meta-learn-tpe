from enum import Enum
import json
import os
import pickle
from typing import Any, Dict, List, Optional, TypedDict

from fast_pareto import is_pareto_front

import numpy as np

import h5py

from targets.hpolib.hyperparameters import BudgetConfig, Hyperparameters
from targets.base_tabularbench_api import BaseTabularBenchAPI


DATA_DIR = f'{os.environ["HOME"]}/tabular_benchmarks/hpolib'
MODULE_PATH = "targets/hpolib"
N_CONFIGS = 62208
AVAIL_SEEDS = [0, 1, 2, 3]
LOSS_KEY = "valid_mse"
RUNTIME_KEY = "runtime"


class TabularDataRowType(TypedDict):
    """
    The row data type of the tabular dataset.
    Each row is specified by a string that can be
    casted to dict and this dict is the hyperparameter
    configuration of this row data.

    Attributes:
        final_test_error (List[float]):
            The final test error over 4 seeds
        n_params (List[float]):
            The number of parameters of the model over 4 seeds
        runtime (List[float]):
            The runtime of the model over 4 seeds
        train_loss (List[List[float]]):
            The training loss of the model over 100 epochs
            with 4 different seeds
        train_mse (List[List[float]]):
            The training mse of the model over 100 epochs
            with 4 different seeds
        valid_loss (List[List[float]]):
            The validation loss of the model over 100 epochs
            with 4 different seeds
        valid_mse (List[List[float]]):
            The validation mse of the model over 100 epochs
            with 4 different seeds
    """

    final_test_error: List[float]
    n_params: List[float]
    runtime: List[float]
    train_loss: List[List[float]]
    train_mse: List[List[float]]
    valid_loss: List[List[float]]
    valid_mse: List[List[float]]


class DatasetChoices(Enum):
    slice_localization = "fcnet_slice_localization_data.hdf5"
    protein_structure = "fcnet_protein_structure_data.hdf5"
    naval_propulsion = "fcnet_naval_propulsion_data.hdf5"
    parkinsons_telemonitoring = "fcnet_parkinsons_telemonitoring_data.hdf5"


class HPOLib(BaseTabularBenchAPI):
    """
    Download the datasets via:
        $ wget http://ml4aad.org/wp-content/uploads/2019/01/fcnet_tabular_benchmarks.tar.gz
        $ tar xf fcnet_tabular_benchmarks.tar.gz
    """

    def __init__(
        self,
        path: str = DATA_DIR,
        dataset: DatasetChoices = DatasetChoices.protein_structure,
        seed: Optional[int] = None,
    ):
        super().__init__(
            hp_module_path=MODULE_PATH,
            dataset_name=dataset.name,
            seed=seed,
            obj_names=[LOSS_KEY, RUNTIME_KEY],
        )
        self._path = path
        self._data = h5py.File(os.path.join(path, dataset.value), "r")
        self._dataset = dataset

    def _fetch_metric_vals(self) -> Dict[str, np.ndarray]:
        if os.path.exists(self.metric_pkl_path):
            with open(self.metric_pkl_path, "rb") as f:
                data = pickle.load(f)
        else:
            data = self._create_pickle_and_return_results()

        return data

    def _compute_reference_point(self) -> Dict[str, float]:
        data = self._fetch_metric_vals()
        return {obj_name: np.max(data[obj_name]) for obj_name in self._obj_names}

    def _compute_pareto_front(self) -> Dict[str, np.ndarray]:
        data = self._fetch_metric_vals()
        costs = np.asarray([data[LOSS_KEY], data[RUNTIME_KEY]]).T
        pareto_mask = is_pareto_front(costs)
        front_sols = costs[pareto_mask]
        pareto_front = {LOSS_KEY: front_sols[:, 0].tolist(), RUNTIME_KEY: front_sols[:, 1].tolist()}
        return pareto_front

    def _collect_dataset_info(self) -> Dict[str, np.ndarray]:
        results = {obj_name: np.empty(N_CONFIGS * len(AVAIL_SEEDS)) for obj_name in [LOSS_KEY, RUNTIME_KEY]}
        epochs, cnt = 99, 0
        for key in self._data.keys():
            info = self._data[key]
            runtime_vals = info[RUNTIME_KEY]
            loss_vals = info[LOSS_KEY][:, epochs]

            for loss, runtime in zip(loss_vals, runtime_vals):
                results[LOSS_KEY][cnt] = loss
                results[RUNTIME_KEY][cnt] = runtime
                cnt += 1

        return {k: v[:cnt] for k, v in results.items()}

    def _create_pickle_and_return_results(self) -> Dict[str, np.ndarray]:
        data = self._collect_dataset_info()
        with open(self.metric_pkl_path, "wb") as f:
            pickle.dump(data, f)

        return data

    def objective_func(self, config: Dict[str, Any], budget: Dict[str, Any] = {}) -> Dict[str, float]:
        _budget = BudgetConfig(**budget)
        type_dict = {
            np.int32: int,
            np.int64: int,
            int: int,
            np.float32: float,
            np.float64: float,
            float: float,
            np.str_: str,
            str: str,
        }
        config = {k: type_dict[type(v)](v) for k, v in config.items()}
        config = Hyperparameters(**config).__dict__

        idx = self.rng.randint(4)
        key = json.dumps(config, sort_keys=True)
        results: Dict[str, float] = {}
        for obj_name in self._obj_names:
            data = self.data[key][obj_name][idx]
            if isinstance(data, np.ndarray):
                results[obj_name] = data[_budget.epochs - 1]
            else:
                results[obj_name] = float(data)

        return results

    @property
    def metric_pkl_path(self) -> str:
        dataset_name = self._dataset.name
        dir_name = "metric_vals"
        file_name = os.path.join(MODULE_PATH, dir_name, f"{dataset_name}.pkl")
        os.makedirs(os.path.join(MODULE_PATH, dir_name), exist_ok=True)
        return file_name

    @property
    def data(self) -> Any:
        return self._data

    @property
    def dataset(self) -> DatasetChoices:
        return self._dataset

    @property
    def minimize(self) -> Dict[str, bool]:
        return {LOSS_KEY: True, RUNTIME_KEY: True}
