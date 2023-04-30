from typing import Dict, Union

import ConfigSpace as CS

import numpy as np


LB, UB = -5.0, 5.0


class Ellipsoid:
    def __init__(
        self,
        center: Union[float, np.ndarray],
        dim: int = 2
    ):
        self._config_space = CS.ConfigurationSpace()
        self._config_space.add_hyperparameters([CS.UniformFloatHyperparameter(f"x{d}", LB, UB) for d in range(dim)])

        if isinstance(center, float):
            self._centers = np.full(dim, center)
        else:
            assert isinstance(center, np.ndarray)
            assert center.size == dim
            self._centers = center.copy()

        self._weights = np.array([float(5 ** d) for d in range(dim)])
        self._dim = dim
        self._lb, self._ub = LB, UB

    def __call__(self, eval_config: Dict[str, float]) -> Dict[str, float]:
        return self.obj_func(eval_config)

    def objective_func(self, eval_config: Dict[str, float]) -> Dict[str, float]:
        x_squared = (np.array([float(eval_config[f"x{d}"]) for d in range(self._dim)]) - self._centers) ** 2
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
