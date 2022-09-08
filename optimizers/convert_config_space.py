from typing import Dict, List, Tuple, Union

import ConfigSpace as CS

from optimizers.meta_learn_bo import HyperParameterType


NumericType = Union[int, float]


def convert(config_space: CS.ConfigurationSpace) -> None:
    hp_info: Dict[str, HyperParameterType] = {}
    bounds: Dict[str, Tuple[NumericType, NumericType]] = {}
    categories: Dict[str, List[str]] = {}
    for hp_name in config_space:
        config = config_space.get_hyperparameter(hp_name)
        if isinstance(config, CS.UniformFloatHyperparameter):
            bounds[hp_name] = (config.lower, config.upper)
            hp_info[hp_name] = HyperParameterType.Continuous
        elif isinstance(config, CS.UniformIntegerHyperparameter):
            bounds[hp_name] = (config.lower, config.upper)
            hp_info[hp_name] = HyperParameterType.Integer
        elif isinstance(config, CS.CategoricalHyperparameter):
            hp_info[hp_name] = HyperParameterType.Categorical
            n_choices = len(config.choices)
            bounds[hp_name] = (0, n_choices - 1)
            categories[hp_name] = list(config.choices)
        elif isinstance(config, CS.OrdinalHyperparameter):
            hp_info[hp_name] = HyperParameterType.Integer
            n_choices = len(config.sequence)
            bounds[hp_name] = (0, n_choices - 1)
        else:
            raise ValueError(f"Got an unknown config type {type(config)}")

    kwargs = dict(
        hp_info=hp_info,
        bounds=bounds,
        categories=categories,
    )
    return kwargs
