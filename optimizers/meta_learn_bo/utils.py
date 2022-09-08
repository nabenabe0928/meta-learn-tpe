import itertools
from collections import OrderedDict
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

from botorch import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.acquisition.multi_objective import ExpectedHypervolumeImprovement
from botorch.models import SingleTaskGP
from botorch.models.gp_regression_mixed import MixedSingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.optim.optimize import optimize_acqf, optimize_acqf_mixed
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning

from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

import numpy as np

import torch


PAREGO, EHVI = "parego", "ehvi"
AcqFuncType = Literal["parego", "ehvi"]
NumericType = Union[int, float]
SingleTaskGPType = Union[SingleTaskGP, MixedSingleTaskGP]
ModelType = Union[SingleTaskGP, MixedSingleTaskGP, ModelListGP]


class HyperParameterType(Enum):
    Categorical: Type = str
    Integer: Type = int
    Continuous: Type = float

    def __eq__(self, type_: Type) -> bool:  # type: ignore
        return self.value == type_


def validate_bounds(hp_names: List[str], bounds: Dict[str, Tuple[NumericType, NumericType]]) -> None:
    if not all(hp_name in hp_names for hp_name in bounds.keys()):
        raise ValueError(
            "bounds must have the bounds for all hyperparameters. "
            f"Expected {hp_names}, but got {list(bounds.keys())}"
        )


def validate_data(data: Dict[str, np.ndarray], hp_names: List[str], obj_names: List[str]) -> None:
    if not all(name in data for name in hp_names + obj_names):
        raise ValueError(
            "data must have the data for all hyperparameters and objectives. "
            f"Expected {hp_names + obj_names}, but got {list(data.keys())}"
        )


def validate_categorical_info(
    categories: Dict[str, List[str]],
    cat_dims: List[int],
    hp_names: List[str],
    bounds: Dict[str, Tuple[NumericType, NumericType]],
) -> None:
    if len(cat_dims) == 0:
        return

    cat_hp_names = [hp_names[d] for d in cat_dims]
    if categories is None or not all(hp_names[d] in categories for d in cat_dims):
        raise ValueError(f"categories must include the categories for {cat_hp_names}, but got {categories}")
    for hp_name in cat_hp_names:
        n_cats = len(categories[hp_name])
        if not all(isinstance(cat, str) for cat in categories[hp_name]):
            raise ValueError(f"Categories must be str, but got {categories[hp_name]} for the hyperparameter {hp_name}")
        if bounds[hp_name] != (0, n_cats - 1):
            raise ValueError(
                f"The categorical parameter `{hp_name}` has {n_cats} categories and expects "
                f"the bound to be (0, n_cats - 1)=(0, {n_cats - 1}), but got {bounds[hp_name]}"
            )


def validate_config_and_results(
    eval_config: Dict[str, Union[str, NumericType]],
    results: Dict[str, float],
    hp_names: List[str],
    obj_names: List[str],
    hp_info: Dict[str, HyperParameterType],
    bounds: Dict[str, Tuple[NumericType, NumericType]],
    categories: Dict[str, List[str]],
) -> None:
    if not all(obj_name in results for obj_name in obj_names):
        raise KeyError(f"results must have keys {obj_names}, but got {results}")
    if not all(hp_name in eval_config for hp_name in hp_names):
        raise KeyError(f"eval_config must have keys {hp_names}, but got {eval_config}")

    EPS = 1e-12
    for hp_name, val in eval_config.items():
        if not isinstance(val, hp_info[hp_name].value):
            raise TypeError(
                f"`{hp_name}` in eval_config must have the type {hp_info[hp_name].value}, but got {type(val)}"
            )
        lb, ub = bounds[hp_name]
        if hp_info[hp_name] == str:
            if val not in categories[hp_name]:
                raise ValueError(f"`{hp_name}` in eval_config must be in {categories[hp_name]}, but got {val}")
        elif val < lb - EPS or ub + EPS < val:
            raise ValueError(f"`{hp_name}` in eval_config must be in [{lb}, {ub}], but got {val}")


def validate_weights(weights: torch.Tensor) -> None:
    if not torch.isclose(weights.sum(), torch.tensor(1.0)):
        raise ValueError(f"The sum of the weights must be 1, but got {weights}")


def update_observations(
    observations: Dict[str, np.ndarray],
    eval_config: Dict[str, Union[str, NumericType]],
    results: Dict[str, float],
    hp_info: Dict[str, HyperParameterType],
    categories: Dict[str, List[str]],
) -> None:
    for hp_name, val in eval_config.items():
        new_val: NumericType
        if hp_info[hp_name] == str:
            assert isinstance(val, str)  # mypy redefinition
            new_val = categories[hp_name].index(val)
        else:
            assert isinstance(val, (float, int))
            new_val = val

        np_type = np.int32 if isinstance(new_val, int) else np.float64
        observations[hp_name] = np.append(observations[hp_name], new_val).astype(np_type)

    for obj_name, val in results.items():
        observations[obj_name] = np.append(observations[obj_name], val)


def normalize(
    observations: Dict[str, np.ndarray],
    bounds: Dict[str, Tuple[NumericType, NumericType]],
    hp_names: List[str],
) -> torch.Tensor:
    """
    Normalize the feature tensor so that each dimension is in [0, 1].

    Args:
        observations (Dict[str, np.ndarray]):
            The observations.
            Dict[hp_name/obj_name, the array of the corresponding param].
        bounds (Dict[str, Tuple[NumericType, NumericType]]):
            The lower and upper bounds for each hyperparameter.
            Dict[hp_name, Tuple[lower bound, upper bound]].
        hp_names (List[str]):
            The list of hyperparameter names.
            List[hp_name].

    Returns:
        X (torch.Tensor):
            The transformed feature tensor with the shape (dim, n_samples).
    """
    return torch.as_tensor(
        np.asarray(
            [
                (observations[hp_name] - bounds[hp_name][0]) / (bounds[hp_name][1] - bounds[hp_name][0])
                for hp_name in hp_names
            ]
        )
    )


def denormalize(
    X: torch.Tensor,
    bounds: Dict[str, Tuple[NumericType, NumericType]],
    hp_names: List[str],
) -> Dict[str, float]:
    """
    De-normalize the feature tensor from the range of [0, 1].

    Args:
        X (torch.Tensor):
            The transformed feature tensor with the shape (dim, n_samples).
        bounds (Dict[str, Tuple[NumericType, NumericType]]):
            The lower and upper bounds for each hyperparameter.
            Dict[hp_name, Tuple[lower bound, upper bound]].
        hp_names (List[str]):
            The list of hyperparameter names.
            List[hp_name].

    Returns:
        config (Dict[str, float]):
            The config reverted from X.
            Dict[hp_name/obj_name, the corresponding param value].
    """
    shape = (len(hp_names),)
    if X.shape != (len(hp_names),):
        raise ValueError(f"The shape of X must be {shape}, but got {X.shape}")

    return {
        hp_name: float(X[idx]) * (bounds[hp_name][1] - bounds[hp_name][0]) + bounds[hp_name][0]
        for idx, hp_name in enumerate(hp_names)
    }


def sample(model: ModelType, X: torch.Tensor) -> torch.Tensor:
    """
    Sample from the posterior based on the model given X.

    Args:
        model (ModelType):
            The Gaussian process model trained on the provided dataset.
        X (torch.Tensor):
            The feature tensor with the shape of (n_samples, dim) that takes as a condition.
            Basically, we sample from y ~ model(f|X).

    Returns:
        preds (torch.Tensor):
            The array with the shape of (batch size, n_samples, n_objectives).
    """
    with torch.no_grad():
        return model.posterior(X).sample()


def scalarize(
    Y_train: torch.Tensor,
    weights: torch.Tensor,
    rho: float = 0.05,
) -> torch.Tensor:
    """
    Compute the linear combination used for ParEGO.

    Args:
        Y_train (torch.Tensor):
            The preprocessed objective tensor.
            Note that this tensor is preprocessed so that
            `larger is better` for the BoTorch internal implementation.
            Y_train.shape = (n_obj, n_evals).
        weights (torch.Tensor):
            The weights for each objective with the shape of (n_obj, ).
        rho (float):
            The hyperparameter used in ParEGO.

    Returns:
        Y_train (torch.Tensor):
            The linear combined version of the objective tensor.
            The shape is (n_evals, ).
    """
    validate_weights(weights)
    # Y_train.shape = (n_obj, n_samples), Y_weighted.shape = (n_obj, n_samples)
    Y_weighted = Y_train * weights[:, None]
    # NOTE: since Y is "Larger is better", so we take min of Y_weighted
    return torch.amin(Y_weighted, dim=0) + rho * torch.sum(Y_weighted, dim=0)


def get_train_data(
    observations: Dict[str, np.ndarray],
    bounds: Dict[str, Tuple[NumericType, NumericType]],
    hp_names: List[str],
    minimize: Dict[str, bool],
    weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preprocess the observations so that BoTorch can train
    Gaussian process using this data.

    Args:
        observations (Dict[str, np.ndarray]):
            The observations.
            Dict[hp_name/obj_name, the array of the corresponding param].
        bounds (Dict[str, Tuple[NumericType, NumericType]]):
            The lower and upper bounds for each hyperparameter.
            Dict[hp_name, Tuple[lower bound, upper bound]].
        hp_names (List[str]):
            The list of hyperparameter names.
            List[hp_name].
        minimize (Dict[str, bool]):
            The direction of the optimization for each objective.
            Dict[obj_name, whether to minimize or not].
        weights (Optional[torch.Tensor]):
                The weights used in the scalarization of ParEGO.

    Returns:
        X_train (torch.Tensor):
            The preprocessed hyperparameter configurations tensor.
            X_train.shape = (n_evals, dim).
        Y_train (torch.Tensor):
            The preprocessed objective tensor.
            Note that this tensor is preprocessed so that
            `larger is better` for the BoTorch internal implementation.
            Y_train.shape = (n_obj, n_evals).
    """
    # NOTE: Y_train will be transformed so that larger is better for botorch
    # X_train.shape = (n_samples, dim)
    X_train = normalize(observations=observations, bounds=bounds, hp_names=hp_names).T
    # Y_train.shape = (n_obj, n_samples)
    Y_train = torch.as_tensor(
        np.asarray([(1 - 2 * do_min) * observations[obj_name] for obj_name, do_min in minimize.items()])
    )
    if weights is None:
        Y_mean = torch.mean(Y_train, dim=-1)
        Y_std = torch.std(Y_train, dim=-1)
        return X_train, (Y_train - Y_mean[:, None]) / Y_std[:, None]
    else:  # scalarization
        Y_train = scalarize(Y_train=Y_train, weights=weights)
        Y_mean = torch.mean(Y_train)
        Y_std = torch.std(Y_train)
        return X_train, (Y_train - Y_mean) / Y_std


def fit_model(
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    cat_dims: List[int],
    scalarize: bool = False,
    state_dict: Optional[OrderedDict] = None,
) -> ModelType:
    """
    Fit Gaussian process model on the provided data.

    Args:
        X_train (torch.Tensor):
            The preprocessed hyperparameter configurations tensor.
            X_train.shape = (n_evals, dim).
        Y_train (torch.Tensor):
            The preprocessed objective tensor.
            Note that this tensor is preprocessed so that
            `larger is better` for the BoTorch internal implementation.
            Y_train.shape = (n_obj, n_evals).
        cat_dims (List[int]):
            The indices of the categorical parameters.
        scalarize (bool):
            Whether to use the scalarization or not.
        state_dict (Optional[OrderedDict]):
            The state dict to reduce the training time in BoTorch.
            This is used for leave-one-out cross validation.

    Returns:
        model (ModelType):
            The Gaussian process model trained on the provided dataset.
    """
    gp_cls = SingleTaskGP if len(cat_dims) == 0 else MixedSingleTaskGP
    kwargs = dict() if len(cat_dims) == 0 else dict(cat_dims=cat_dims)
    if scalarize:  # ParEGO
        model = gp_cls(train_X=X_train, train_Y=Y_train.squeeze()[:, None], **kwargs)
        mll_cls = ExactMarginalLogLikelihood
    else:  # EHVI
        models: List[SingleTaskGPType] = []
        for Y in Y_train:
            _model = gp_cls(train_X=X_train, train_Y=Y[:, None], **kwargs)
            models.append(_model)

        model = ModelListGP(*models)
        mll_cls = SumMarginalLogLikelihood

    if state_dict is None:
        mll = mll_cls(model.likelihood, model)
        fit_gpytorch_model(mll)
    else:
        model.load_state_dict(state_dict)

    return model


def get_model_and_train_data(
    observations: Dict[str, np.ndarray],
    bounds: Dict[str, Tuple[NumericType, NumericType]],
    hp_names: List[str],
    minimize: Dict[str, bool],
    cat_dims: List[int],
    weights: Optional[torch.Tensor] = None,
) -> Tuple[ModelType, torch.Tensor, torch.Tensor]:
    scalarize = weights is not None
    if weights is not None:
        validate_weights(weights)

    X_train, Y_train = get_train_data(
        observations=observations,
        bounds=bounds,
        hp_names=hp_names,
        minimize=minimize,
        weights=weights,
    )
    model = fit_model(X_train=X_train, Y_train=Y_train, cat_dims=cat_dims, scalarize=scalarize)

    return model, X_train, Y_train


def get_parego(model: SingleTaskGPType, X_train: torch.Tensor, Y_train: torch.Tensor) -> ExpectedImprovement:
    """
    Get the ParEGO acquisition funciton.

    Args:
        model (SingleTaskGPType):
            The Gaussian process model trained on the provided dataset.
        X_train (torch.Tensor):
            The preprocessed hyperparameter configurations tensor.
            X_train.shape = (n_evals, dim).
        Y_train (torch.Tensor):
            The preprocessed objective tensor.
            Note that this tensor is preprocessed so that
            `larger is better` for the BoTorch internal implementation.
            Y_train.shape = (n_obj, n_evals).

    Returns:
        acq_fn (ExpectedImprovement):
            The acquisition function obtained based on the provided dataset and the model.
    """
    acq_fn = ExpectedImprovement(model=model, best_f=Y_train.amax())
    return acq_fn


def get_ehvi(model: ModelListGP, X_train: torch.Tensor, Y_train: torch.Tensor) -> ExpectedHypervolumeImprovement:
    """
    Get the Expected hypervolume improvement acquisition funciton.

    Args:
        model (ModelListGP):
            The Gaussian process model trained on the provided dataset.
        X_train (torch.Tensor):
            The preprocessed hyperparameter configurations tensor.
            X_train.shape = (n_evals, dim).
        Y_train (torch.Tensor):
            The preprocessed objective tensor.
            Note that this tensor is preprocessed so that
            `larger is better` for the BoTorch internal implementation.
            Y_train.shape = (n_obj, n_evals).

    Returns:
        acq_fn (ExpectedHypervolumeImprovement):
            The acquisition function obtained based on the provided dataset and the model.
    """
    with torch.no_grad():
        pred = model.posterior(X_train).mean

    # NOTE: botorch maximizes all objectives and notice that Y.min() is alywas negative
    ref_point = torch.as_tensor([Y.min() * 1.1 for Y in Y_train])
    partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=pred)
    acq_fn = ExpectedHypervolumeImprovement(
        model=model,
        ref_point=ref_point,
        partitioning=partitioning,
    )
    return acq_fn


def get_acq_fn(
    model: ModelType, X_train: torch.Tensor, Y_train: torch.Tensor, acq_fn_type: AcqFuncType
) -> Union[ExpectedImprovement, ExpectedHypervolumeImprovement]:
    """
    Get the specified acquisition funciton.

    Args:
        model (ModelType):
            The Gaussian process model trained on the provided dataset.
        X_train (torch.Tensor):
            The preprocessed hyperparameter configurations tensor.
            X_train.shape = (n_evals, dim).
        Y_train (torch.Tensor):
            The preprocessed objective tensor.
            Note that this tensor is preprocessed so that
            `larger is better` for the BoTorch internal implementation.
            Y_train.shape = (n_obj, n_evals).

    Returns:
        acq_fn (Union[ExpectedImprovement, ExpectedHypervolumeImprovement]):
            The acquisition function obtained based on the provided dataset and the model.
    """
    supported_acq_fn_types = {"parego": get_parego, "ehvi": get_ehvi}
    for acq_fn_name, func in supported_acq_fn_types.items():
        if acq_fn_name == acq_fn_type:
            return func(model=model, X_train=X_train, Y_train=Y_train)
    else:
        raise ValueError(f"acq_fn_type must be in {supported_acq_fn_types}, but got {acq_fn_type}")


def optimize_acq_fn(
    acq_fn: ExpectedHypervolumeImprovement,
    bounds: Dict[str, Tuple[NumericType, NumericType]],
    hp_names: List[str],
    fixed_features_list: Optional[List[Dict[int, float]]],
) -> Dict[str, float]:
    """
    Optimize the given acquisition function and obtain the next configuration to evaluate.

    Args:
        acq_fn (Union[ExpectedImprovement, ExpectedHypervolumeImprovement]):
            The acquisition function obtained based on the provided dataset and the model.
        bounds (Dict[str, Tuple[NumericType, NumericType]]):
            The lower and upper bounds for each hyperparameter.
            Dict[hp_name, Tuple[lower bound, upper bound]].
        hp_names (List[str]):
            The list of hyperparameter names.
            List[hp_name].
        fixed_features_list (Optional[List[Dict[int, float]]]):
            A list of maps `{feature_index: value}`.
            The i-th item represents the fixed_feature for the i-th optimization.
            Basically, we would like to perform len(fixed_features_list) times of
            optimizations and we use each `fixed_features` in each optimization.

    Returns:
        eval_config (Dict[str, float]):
            The config to evaluate.
            Dict[hp_name/obj_name, the corresponding param value].
    """
    kwargs = dict(q=1, num_restarts=10, raw_samples=1 << 8, return_best_only=True)
    standard_bounds = torch.zeros((2, len(hp_names)))
    standard_bounds[1] = 1
    if fixed_features_list is None:
        X, _ = optimize_acqf(acq_function=acq_fn, bounds=standard_bounds, **kwargs)
    else:
        X, _ = optimize_acqf_mixed(
            acq_function=acq_fn, bounds=standard_bounds, fixed_features_list=fixed_features_list, **kwargs
        )

    eval_config = denormalize(X=X.squeeze(), bounds=bounds, hp_names=hp_names)
    return eval_config


def get_fixed_features_list(
    hp_names: List[str], cat_dims: List[int], categories: Dict[str, List[str]]
) -> Optional[List[Dict[int, float]]]:
    """
    Returns:
        fixed_features_list (Optional[List[Dict[int, float]]]):
            A list of maps `{feature_index: value}`.
            The i-th item represents the fixed_feature for the i-th optimization.
            Basically, we would like to perform len(fixed_features_list) times of
            optimizations and we use each `fixed_features` in each optimization.

    NOTE:
        Due to the normalization, we need to define each parameter to be in [0, 1].
        For this reason, when we have K categories, the possible choices will be
        [0, 1/(K-1), 2/(K-1), ..., (K-1)/(K-1)].
    """
    if len(cat_dims) == 0:
        return None

    fixed_features_list: List[Dict[int, float]] = []
    for feats in itertools.product(*(np.linspace(0, 1, len(categories[hp_names[d]])) for d in cat_dims)):
        fixed_features_list.append({d: val for val, d in zip(feats, cat_dims)})

    return fixed_features_list


def convert_categories_into_index(
    data: Dict[str, np.ndarray], categories: Optional[Dict[str, List[str]]]
) -> Dict[str, np.ndarray]:
    """
    Convert data so that categorical arrays will be the corresponding index arrays.

    Args:
        data (Dict[str, np.ndarray]):
            The data of observations.
            Dict[hp_name/obj_name, the array of the corresponding param].
        categories (Optional[Dict[str, List[str]]]):
            Categories for each categorical parameter.
            Dict[categorical hp name, List[each category name]].

    Returns:
        converted_data (Dict[str, np.ndarray]):
            This data has only numerical arrays as categorical arrays
            are converted into the corresponding index arrays.
    """
    if categories is None:
        return data

    for hp_name, cats in categories.items():
        n_cats = len(cats)
        data[hp_name] = np.asarray([cats.index(v) if isinstance(v, str) else int(v) for v in data[hp_name]])

        if not np.all((0 <= data[hp_name]) & (data[hp_name] < n_cats)):
            raise ValueError(
                f"Provided data in the categorical hyperparameter `{hp_name}` must be in [0, {n_cats - 1}], "
                f"but got {data[hp_name]}"
            )
    return data
