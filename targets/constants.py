from typing import Callable, Union

import ConfigSpace.hyperparameters as CSH

import numpy as np


EPS = 1.0e-300
NumericType = Union[float, int]
SQR2, SQR2PI = np.sqrt(2), np.sqrt(2 * np.pi)

CategoricalHPType = CSH.CategoricalHyperparameter
NumericalHPType = Union[CSH.UniformIntegerHyperparameter, CSH.UniformFloatHyperparameter, CSH.OrdinalHyperparameter]

config2type = {"UniformFloatHyperparameter": float, "UniformIntegerHyperparameter": int, "OrdinalHyperparameter": float}

type2config = {
    float: "UniformFloatHyperparameter",
    int: "UniformIntegerHyperparameter",
    bool: "CategoricalHyperparameter",
    str: "CategoricalHyperparameter",
}

DOMAIN_SIZE_CHOICES = list(range(10, 110, 10))


def default_percentile_maker() -> Callable[[np.ndarray], int]:
    def _imp(vals: np.ndarray, min_num: int = 1) -> int:
        size = vals.size
        return max(int(np.ceil(0.25 * np.sqrt(size))), min_num)

    return _imp


def default_threshold_maker(upper_bound: float) -> Callable[[np.ndarray], int]:
    def _imp(vals: np.ndarray, min_num: int = 1) -> int:
        n_lower = max(int(np.searchsorted(vals, upper_bound, side="right")), min_num)
        return min(n_lower, vals.size)

    return _imp


def default_weights(size: int) -> np.ndarray:
    weights = np.ones(size)
    return weights / weights.sum()
