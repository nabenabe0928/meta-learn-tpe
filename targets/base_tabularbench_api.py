import os
from abc import abstractmethod, ABCMeta
from typing import Any, Dict, List, Optional

import numpy as np

import json

import ConfigSpace as CS

from targets.utils import get_config_space, ParameterSettings


class BaseTabularBenchAPI(metaclass=ABCMeta):
    def __init__(
        self,
        hp_module_path: str,
        dataset_name: str,
        obj_names: List[str],
        seed: Optional[int] = None,
    ):
        self._module_path = hp_module_path
        self._search_space = {
            k: v["sequence"] if "sequence" in v else v["choices"]
            for k, v in json.load(open(f"{self._module_path}/params.json")).items()
            if not k.startswith("_")
        }
        self._rng = np.random.RandomState(seed)
        self._obj_names = obj_names[:]
        js = open(f"{hp_module_path}/params.json")
        search_space: Dict[str, ParameterSettings] = json.load(js)
        self._config_space = get_config_space(search_space, hp_module_path=".".join(hp_module_path.split("/")))
        self._hp_names = [hp_name for hp_name in self._config_space]

    @abstractmethod
    def _compute_reference_point(self) -> Dict[str, float]:
        """The worst values for each objective"""
        raise NotImplementedError

    @abstractmethod
    def _compute_pareto_front(self) -> Dict[str, np.ndarray]:
        """
        Return the Pareto front solutions.

        Returns:
            pareto_sols (Dict[str, np.ndarray]):
                Dict[obj_name, obj array].
        """
        raise NotImplementedError

    def find_reference_point(self) -> Dict[str, float]:
        """The worst values for each objective"""
        dir_name = "reference-point"
        file_name = os.path.join(self._module_path, dir_name, f"{self._dataset.name}.json")
        if not os.path.exists(file_name):
            os.makedirs(os.path.join(self._module_path, dir_name), exist_ok=True)
            ref_point = self._compute_reference_point()
            with open(file_name, mode="w") as f:
                json.dump(ref_point, f, indent=4)

        return json.load(open(file_name))

    def find_pareto_front(self) -> Dict[str, np.ndarray]:
        """
        Return the Pareto front solutions.

        Returns:
            pareto_sols (Dict[str, np.ndarray]):
                Dict[obj_name, obj array].
        """
        dir_name = "pareto-fronts"
        file_name = os.path.join(self._module_path, dir_name, f"{self._dataset.name}.json")
        if not os.path.exists(file_name):
            os.makedirs(os.path.join(self._module_path, dir_name), exist_ok=True)
            pareto_front = self._compute_pareto_front()
            with open(file_name, mode="w") as f:
                json.dump(pareto_front, f, indent=4)

        return {k: np.asarray(v) for k, v in json.load(open(file_name)).items()}

    @abstractmethod
    def objective_func(self, config: Dict[str, Any], budget: Dict[str, Any] = {}) -> Dict[str, float]:
        """
        Args:
            config (Dict[str, Any]):
                The dict of the configuration and the corresponding value
            budget (Dict[str, Any]):
                The budget information

        Returns:
            results (Dict[str, float]):
                A pair of loss or constraint value and its name.
        """
        raise NotImplementedError

    @property
    def rng(self) -> np.random.RandomState:
        return self._rng

    @property
    def config_space(self) -> CS.ConfigurationSpace:
        """The config space of the child tabular benchmark"""
        return self._config_space

    @property
    def obj_names(self) -> List[str]:
        return self._obj_names[:]

    @property
    def hp_names(self) -> List[str]:
        return self._hp_names[:]

    @property
    @abstractmethod
    def data(self) -> Any:
        """API for the target dataset"""
        raise NotImplementedError

    @property
    @abstractmethod
    def minimize(self) -> Dict[str, bool]:
        """Whether to minimize each objective"""
        raise NotImplementedError
