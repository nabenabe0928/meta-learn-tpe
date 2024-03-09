from __future__ import annotations

import time

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

import numpy as np

from optimizers import TPEOptimizer


def sphere(eval_config: dict[str, float], shift: float = 0) -> tuple[dict[str, float], float]:
    start = time.time()
    vals = np.array(list(eval_config.values()))
    return {"loss": np.sum((vals - shift) ** 2)}, time.time() - start


if __name__ == "__main__":
    dim = 10
    cs = CS.ConfigurationSpace()
    for d in range(dim):
        cs.add_hyperparameter(CSH.UniformFloatHyperparameter(f"x{d}", lower=-5, upper=5))

    meta_learner = TPEOptimizer(
        obj_func=lambda eval_config: sphere(eval_config, shift=1),
        config_space=cs,
        n_init=50,
        max_evals=50,
    )
    meta_learner.optimize()
    metadata = {"shift=1": meta_learner.fetch_observations()}
    opt = TPEOptimizer(
        obj_func=sphere, config_space=cs, min_bandwidth_factor=1e-2, metadata=metadata, resultfile="sphere"
    )
    opt.optimize(logger_name="sphere")
