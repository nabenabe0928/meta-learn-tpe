from __future__ import annotations

import time

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

import numpy as np

from tpe.optimizer import TPEOptimizer


def sphere(eval_config: dict[str, float]) -> tuple[dict[str, float], float]:
    start = time.time()
    vals = np.array(list(eval_config.values()))
    vals *= vals
    return {"loss": np.sum(vals)}, time.time() - start


if __name__ == "__main__":
    dim = 10
    cs = CS.ConfigurationSpace()
    for d in range(dim):
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(f"x{d}", lower=-5, upper=5))

    opt = TPEOptimizer(obj_func=sphere, config_space=cs, min_bandwidth_factor=1e-2, resultfile="sphere")
    opt.optimize(logger_name="sphere")
