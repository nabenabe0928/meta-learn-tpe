from abc import ABCMeta, abstractmethod
from typing import Callable, Dict, List, Optional, Tuple, Union

from optimizers.meta_learn_bo.utils import (
    HyperParameterType,
    NumericType,
    validate_bounds,
    validate_categorical_info,
)

import numpy as np


class BaseSampler(metaclass=ABCMeta):
    def __init__(
        self,
        bounds: Dict[str, Tuple[NumericType, NumericType]],
        hp_info: Dict[str, HyperParameterType],
        minimize: Dict[str, bool],
        max_evals: int,
        obj_func: Callable,
        verbose: bool,
        categories: Optional[Dict[str, List[str]]],
        seed: Optional[int],
    ):
        """The base class samplers.

        Args:
            bounds (Dict[str, Tuple[NumericType, NumericType]]):
                The lower and upper bounds for each hyperparameter.
                If the parameter is categorical, it must be [0, the number of categories - 1].
                Dict[hp_name, Tuple[lower bound, upper bound]].
            hp_info (Dict[str, HyperParameterType]):
                The type information of each hyperparameter.
                Dict[hp_name, HyperParameterType].
            minimize (Dict[str, bool]):
                The direction of the optimization for each objective.
                Dict[obj_name, whether to minimize or not].
            obj_func (Callable):
                The objective function that takes `eval_config` and returns `results`.
            max_evals (int):
                How many hyperparameter configurations to evaluate during the optimization.
            categories (Optional[Dict[str, List[str]]]):
                Categories for each categorical parameter.
                Dict[categorical hp name, List[each category name]].
            verbose (bool):
                Whether to print the results at each iteration.
            seed (Optional[int]):
                The random seed.
        """
        self._bounds = bounds
        self._hp_info = hp_info
        self._minimize = minimize
        self._obj_func = obj_func
        self._obj_names = list(minimize.keys())
        self._hp_names = list(hp_info.keys())
        # cat_dim specifies which dimensions are categorical parameters
        self._cat_dims = [idx for idx, hp_name in enumerate(self._hp_names) if hp_info[hp_name] == str]
        self._max_evals = max_evals
        self._categories: Dict[str, List[str]] = categories if categories is not None else {}
        self._rng = np.random.RandomState(seed)
        self._observations: Dict[str, np.ndarray] = {name: np.array([]) for name in self._hp_names + self._obj_names}
        self._verbose = verbose

        self._validate_input()

    @property
    @abstractmethod
    def observations(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    def optimize(self) -> None:
        for t in range(self._max_evals):
            eval_config = self.sample()
            results = self._obj_func(eval_config.copy())

            if self._verbose:
                print(f"Iteration {t + 1}: eval_config={eval_config}, results={results}")

            self.update(eval_config=eval_config, results=results)

    def _validate_input(self) -> None:
        validate_bounds(hp_names=self._hp_names, bounds=self._bounds)
        validate_categorical_info(
            categories=self._categories, cat_dims=self._cat_dims, bounds=self._bounds, hp_names=self._hp_names
        )

    @abstractmethod
    def sample(self) -> Dict[str, Union[str, NumericType]]:
        """
        Sample the next configuration according to the child sampler.

        Returns:
            eval_config (Dict[str, Union[str, NumericType]]):
                The hyperparameter configuration that were evaluated.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, eval_config: Dict[str, Union[str, NumericType]], results: Dict[str, float]) -> None:
        """
        Update required information.

        Args:
            eval_config (Dict[str, Union[str, NumericType]]):
                The hyperparameter configuration that were evaluated.
            results (Dict[str, float]):
                The results obtained from the evaluation of eval_config.
        """
        raise NotImplementedError
