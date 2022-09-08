from typing import Dict, List, Optional, Union

import ConfigSpace as CS

import numpy as np

from optimizers.tpe.optimizer.models import AbstractTPE, MultiObjectiveTPE, TPE


TPESamplerType = Union[TPE, MultiObjectiveTPE]


def _copy_observations(observations: Dict[str, np.ndarray], param_names: List[str]) -> Dict[str, np.ndarray]:
    return {param_name: observations[param_name].copy() for param_name in param_names}


class ConstraintTPE(AbstractTPE):
    def __init__(
        self,
        config_space: CS.ConfigurationSpace,
        n_ei_candidates: int,
        objective_names: List[str],
        runtime_name: str,
        seed: Optional[int],
        min_bandwidth_factor: float,
        top: float,
        constraints: Dict[str, float],
    ):
        raise NotImplementedError
