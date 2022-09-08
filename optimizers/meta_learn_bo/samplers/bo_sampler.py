from typing import Callable, Dict, List, Optional, Tuple, Union

from optimizers.meta_learn_bo.models.base_weighted_gp import BaseWeightedGP
from optimizers.meta_learn_bo.samplers.base_sampler import BaseSampler
from optimizers.meta_learn_bo.utils import HyperParameterType, NumericType

import numpy as np


class MetaLearnGPSampler(BaseSampler):
    def __init__(
        self,
        bounds: Dict[str, Tuple[NumericType, NumericType]],
        hp_info: Dict[str, HyperParameterType],
        minimize: Dict[str, bool],
        max_evals: int,
        obj_func: Callable,
        model: BaseWeightedGP,
        verbose: bool = True,
        categories: Optional[Dict[str, List[str]]] = None,
        seed: Optional[int] = None,
    ):
        """Meta-learn GP sampler.

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
            model (BaseWeightedGP):
                The meta learn GP model.
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
        super().__init__(
            bounds=bounds,
            hp_info=hp_info,
            minimize=minimize,
            max_evals=max_evals,
            obj_func=obj_func,
            verbose=verbose,
            categories=categories,
            seed=seed,
        )
        self._model = model

    @property
    def observations(self) -> Dict[str, np.ndarray]:
        return self._model.observations

    def sample(self) -> Dict[str, Union[str, NumericType]]:
        """
        Sample the next configuration according to the provided gp sampler.

        Returns:
            eval_config (Dict[str, Union[str, NumericType]]):
                The hyperparameter configuration that were evaluated.
        """
        return self._model.optimize_acq_fn()

    def update(self, eval_config: Dict[str, Union[str, NumericType]], results: Dict[str, float]) -> None:
        """
        Update the target observations, (a) Gaussian process model(s),
        and its/their acquisition function(s).
        If the acq_fn_type is ParEGO, we need to re-train each Gaussian process models
        and the corresponding acquisition functions.

        Args:
            eval_config (Dict[str, Union[str, NumericType]]):
                The hyperparameter configuration that were evaluated.
            results (Dict[str, float]):
                The results obtained from the evaluation of eval_config.
        """
        self._model.update(eval_config, results)
