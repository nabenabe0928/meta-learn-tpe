from typing import Dict, List, Optional

import ConfigSpace as CS

from fast_pareto import is_pareto_front, nondominated_rank

import numpy as np

from optimizers.tpe.optimizer.models import BaseTPE

from optimizers.tpe.utils.constants import TIE_BREAK_METHOD


class MultiObjectiveTPE(BaseTPE):
    def __init__(
        self,
        config_space: CS.ConfigurationSpace,
        n_ei_candidates: int,
        objective_names: List[str],
        runtime_name: str,
        seed: Optional[int],
        min_bandwidth_factor: float,
        top: float,
        minimize: Optional[Dict[str, bool]],
        # The control parameters for experiments
        quantile: float,
    ):
        super().__init__(
            config_space=config_space,
            n_ei_candidates=n_ei_candidates,
            objective_names=objective_names,
            runtime_name=runtime_name,
            seed=seed,
            min_bandwidth_factor=min_bandwidth_factor,
            top=top,
            minimize=minimize,
        )
        self._n_fronts: int
        self._quantile = quantile
        self._nondominated_ranks: np.ndarray

    def _percentile_func(self) -> int:
        n_observations = self._observations[self._objective_names[0]].size
        return max(self._n_fronts, int(np.ceil(self._quantile * n_observations)))

    def _calculate_order(self, results: Optional[Dict[str, float]] = None) -> np.ndarray:
        with_new_result = results is not None

        n_observations = self._observations[self._objective_names[0]].size
        n_objectives = len(self._objective_names)
        costs = np.zeros((n_observations + with_new_result, n_objectives))
        for idx, objective_name in enumerate(self._objective_names):
            if not with_new_result:
                costs[:, idx] = self._observations[objective_name]
                continue
            else:
                costs[:-1, idx] = self._observations[objective_name]

            assert results is not None  # mypy redefinition
            new_loss = results.get(objective_name, None)
            if new_loss is None:
                raise ValueError(f"The evaluation must return {objective_name}.")

            costs[-1, idx] = new_loss

        larger_is_better_objectives = [
            idx for idx, obj_name in enumerate(self._objective_names) if not self._minimize[obj_name]
        ]
        self._nondominated_ranks = nondominated_rank(
            costs,
            tie_break=TIE_BREAK_METHOD,
            larger_is_better_objectives=larger_is_better_objectives
        )
        self._n_fronts = np.sum(is_pareto_front(costs))
        self._order = np.argsort(self._nondominated_ranks)
        return self._order
