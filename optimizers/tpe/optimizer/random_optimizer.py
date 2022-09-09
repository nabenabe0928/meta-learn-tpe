from typing import Any, Dict, List, Optional

import numpy as np

import ConfigSpace as CS

from optimizers.tpe.optimizer.base_optimizer import BaseOptimizer, ObjectiveFunc


class RandomOptimizer(BaseOptimizer):
    def __init__(
        self,
        obj_func: ObjectiveFunc,
        config_space: CS.ConfigurationSpace,
        objective_names: List[str],
        resultfile: str = "temp",
        n_init: int = 10,
        max_evals: int = 100,
        runtime_name: str = "iter_time",
        only_requirements: bool = True,
        constraints: Dict[str, float] = {},
        seed: Optional[int] = None,
    ):

        super().__init__(
            obj_func=obj_func,
            config_space=config_space,
            constraints=constraints,
            resultfile=resultfile,
            objective_names=objective_names,
            n_init=n_init,
            max_evals=max_evals,
            runtime_name=runtime_name,
            seed=seed,
            only_requirements=only_requirements,
        )

        self._observations: Dict[str, np.ndarray] = {runtime_name: np.array([])}
        self._observations.update({hp_name: np.array([]) for hp_name in self._hp_names})
        self._observations.update({obj_name: np.array([]) for obj_name in self._objective_names})

    def update(self, eval_config: Dict[str, Any], results: Dict[str, float], runtime: float) -> None:
        for hp_name, val in eval_config.items():
            self._observations[hp_name] = np.append(self._observations[hp_name], val)

        for obj_name, val in results.keys():
            self._observations[obj_name] = np.append(self._observations[obj_name], val)

        self._observations[self._runtime_name] = np.append(self._observations, runtime)

    def fetch_observations(self) -> Dict[str, np.ndarray]:
        return {k: v.copy() for k, v in self._observations.items()}

    def sample(self) -> Dict[str, Any]:
        return self.initial_sample()
