from enum import Enum
import os
import pickle
from typing import Any, Dict, Optional, TypedDict, Union

from fast_pareto import is_pareto_front

import numpy as np

import pyarrow.parquet as pq  # type: ignore

from targets.hpobench.hyperparameters import Hyperparameters
from targets.base_tabularbench_api import BaseTabularBenchAPI


DATA_DIR = f'{os.environ["HOME"]}/tabular_benchmarks/hpo-bench'
MODULE_PATH = "targets/hpobench"
PREFIX, SUFFIX = "nn_", "_data.parquet.gzip"
N_CONFIGS = 30000
AVAIL_SEEDS = [665, 1319, 7222, 7541, 8916]
PERF_KEY = "bal_acc"
PREC_KEY = "precision"


class TabularDataRowType(TypedDict):
    """
    The row data type of val_scores in the tabular dataset.
    Each row is specified by a string that can be
    casted to dict and this dict is the hyperparameter
    configuration of this row data.

    Attributes:
        acc (float):
            The accuracy in [0, 1].
        bal_acc (float):
            The balanced accuracy in [0, 1].
            The difference from acc is whether
            it considers the data imbalance or not.
        f1 (float):
            F1 score is known as:
                F1 := 2 / (recall^-1 + precision^-1)
        precision (float):
            precision := (# of true positive) / (# of positive)
            positive just means the classifier says a sample is positive.
    """

    acc: float
    bal_acc: float
    f1: float
    precision: float


class DatasetChoices(Enum):
    credit_g = f"31/{PREFIX}31{SUFFIX}"
    vehicle = f"53/{PREFIX}53{SUFFIX}"
    kc1 = f"3917/{PREFIX}3917{SUFFIX}"
    phoneme = f"9952/{PREFIX}9952{SUFFIX}"
    blood_transfusion = f"10101/{PREFIX}10101{SUFFIX}"
    australian = f"146818/{PREFIX}146818{SUFFIX}"
    car = f"146821/{PREFIX}146821{SUFFIX}"
    segment = f"146822/{PREFIX}146822{SUFFIX}"


def _validate_query(
    query: Dict[str, Any], config: Dict[str, Union[int, float]]
) -> None:
    if len(query["__index_level_0__"]) != 1:
        raise ValueError(
            f"There must be only one row for config={config}, but got query={query}"
        )

    queried_config = {k: query[k][0] for k in config.keys()}
    if not all(np.isclose(queried_config[k], v, rtol=1e-3) for k, v in config.items()):
        raise ValueError(
            f"The query must have the identical config as {config}, but got {queried_config}"
        )


class HPOBench(BaseTabularBenchAPI):
    """
    For the download of datasets and more details see
        https://github.com/nabenabe0928/easy-hpo-bench
    """

    def __init__(
        self,
        path: str = DATA_DIR,
        dataset: DatasetChoices = DatasetChoices.credit_g,
        seed: Optional[int] = None,
    ):
        super().__init__(
            hp_module_path=MODULE_PATH,
            dataset_name=dataset.name,
            seed=seed,
            obj_names=[PERF_KEY, PREC_KEY],
        )
        self._path = path
        db = pq.read_table(os.path.join(path, dataset.value), filters=[("iter", "==", 243)])
        self._data = db.drop(["iter", "subsample"])
        self._dataset = dataset

    def _fetch_metric_vals(self) -> Dict[str, np.ndarray]:
        if os.path.exists(self.metric_pkl_path):
            with open(self.metric_pkl_path, "rb") as f:
                data = pickle.load(f)
        else:
            data = self._create_pickle_and_return_results()

        return data

    def _compute_reference_point(self) -> Dict[str, float]:
        # both metrics are better when they are larger
        data = self._fetch_metric_vals()
        return {obj_name: np.min(data[obj_name]) for obj_name in self._obj_names}

    def _compute_pareto_front(self) -> Dict[str, np.ndarray]:
        data = self._fetch_metric_vals()
        costs = np.asarray([data[PERF_KEY], data[PREC_KEY]]).T
        pareto_mask = is_pareto_front(costs, larger_is_better_objectives=[0, 1])
        front_sols = costs[pareto_mask]
        pareto_front = {PERF_KEY: front_sols[:, 0].tolist(), PREC_KEY: front_sols[:, 1].tolist()}
        return pareto_front

    def _collect_dataset_info(self) -> Dict[str, np.ndarray]:
        df = self._data.column("result").to_pandas()
        dataset_info: Dict[str, np.ndarray] = {
            obj_name: df.apply(lambda row: row["info"]["val_scores"][obj_name]).to_numpy()
            for obj_name in [PERF_KEY, PREC_KEY]
        }
        return dataset_info

    def _create_pickle_and_return_results(self) -> Dict[str, np.ndarray]:
        data = self._collect_dataset_info()
        with open(self.metric_pkl_path, "wb") as f:
            pickle.dump(data, f)

        return data

    def objective_func(self, config: Dict[str, Any]) -> Dict[str, float]:
        config = Hyperparameters(**config).__dict__
        config["seed"] = AVAIL_SEEDS[self.rng.randint(5)]

        idx = 0
        for k, v in config.items():
            idx = self._data[k].index(v, start=idx).as_py()

        query = self._data.take([idx]).to_pydict()["result"][0]["info"]["val_scores"]
        _validate_query(query, config)
        results: Dict[str, float] = {
            PERF_KEY: query[PERF_KEY][0],
            PREC_KEY: query[PREC_KEY][0],
        }
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
        return {PERF_KEY: False, PREC_KEY: False}
