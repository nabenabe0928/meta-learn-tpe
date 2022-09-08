import warnings

from optimizers import MetaLearnGPSampler, RankingWeightedGaussianProcessEnsemble
from optimizers.meta_learn_bo import (
    HyperParameterType,
    get_random_samples,
)


warnings.filterwarnings("ignore")


def func(eval_config, shift=0):
    assert eval_config["i"] in [-2, -1, 0, 1, 2]
    assert eval_config["c"] in ["A", "B", "C"]

    x, y = eval_config["x"], eval_config["y"]
    f1 = (x + shift) ** 2 + (y + shift) ** 2
    f2 = (x - 2 + shift) ** 2 + (y - 2 + shift) ** 2
    return {"f1": f1, "f2": f2}


def run():
    # Define the search space!
    # 1. bounds: The bounds for each hyperparameter.
    # NOTE: Categorical parameters must specify the bounds such that (0, n_cats - 1).
    bounds = {"x": (-5, 5), "y": (-5, 5), "i": (-2, 2), "c": (0, 2)}
    # 2. hp_info: The parameter type of each hyperparameter.
    hp_info = {
        "x": HyperParameterType.Continuous,
        "y": HyperParameterType.Continuous,
        "i": HyperParameterType.Integer,
        "c": HyperParameterType.Categorical,
    }
    # 3. minimize: The direction to optimize for each objective
    minimize = {"f1": True, "f2": True}
    # 4. categories: The choices for each categorical parameters
    categories = {"c": ["A", "B", "C"]}
    kwargs = dict(minimize=minimize, bounds=bounds, hp_info=hp_info, categories=categories)

    # Collect metadata on meta tasks.
    metadata = {
        "src": get_random_samples(n_samples=30, obj_func=lambda eval_config: func(eval_config, shift=2), **kwargs)
    }
    # Collect the initial samples on the target task
    init_data = get_random_samples(n_samples=10, obj_func=func, **kwargs)

    # Define the meta-learn GP model based on the init_data and the metadata
    rgpe = RankingWeightedGaussianProcessEnsemble(init_data=init_data, metadata=metadata, **kwargs)

    # Define the sampler
    sampler = MetaLearnGPSampler(max_evals=20, obj_func=func, model=rgpe, **kwargs)

    # Run the optimization.
    sampler.optimize()
    # Output the observations during the whole optimization (random sample + BO sample)
    print(sampler.observations)


if __name__ == "__main__":
    run()
