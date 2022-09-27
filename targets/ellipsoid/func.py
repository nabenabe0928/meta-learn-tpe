from typing import Dict

import ConfigSpace as CS

import numpy as np


LB, UB = -5.0, 5.0


class Ellipsoid:
    def __init__(self, center: float, dim: int = 2):
        self._config_space = CS.ConfigurationSpace()
        self._config_space.add_hyperparameters([CS.UniformFloatHyperparameter(f"x{d}", LB, UB) for d in range(dim)])
        self._center = center
        self._weights = np.array([float(5 ** d) for d in range(dim)])
        self._dim = dim
        self._lb, self._ub = LB, UB

    def __call__(self, eval_config: Dict[str, float]) -> Dict[str, float]:
        return self.obj_func(eval_config)

    def objective_func(self, eval_config: Dict[str, float]) -> Dict[str, float]:
        x_squared = (np.array([float(eval_config[f"x{d}"]) for d in range(self._dim)]) - self._center) ** 2
        return {"loss": self._weights @ x_squared}

    @property
    def config_space(self) -> CS.ConfigurationSpace:
        return self._config_space

    @property
    def lb(self) -> float:
        return self._lb

    @property
    def ub(self) -> float:
        return self._ub

    @property
    def center(self) -> float:
        return self._center
