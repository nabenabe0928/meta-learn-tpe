from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from fast_pareto import nondominated_rank

from optimizers.meta_learn_bo.models.base_weighted_gp import BaseWeightedGP
from optimizers.meta_learn_bo.utils import AcqFuncType, HyperParameterType, NumericType, PAREGO, fit_model, sample

import numpy as np

import torch


def drop_ranking_loss(
    ranking_loss: torch.Tensor,
    n_evals: int,
    max_evals: int,
    rng: np.random.RandomState,
) -> torch.Tensor:
    """
    Drop some weights in meta tasks to prevent weight dilution
    as described in Section 4.3 of the original paper.
    The dropout rate is computed based on Eq. (9) in the original paper.

    Args:
        ranking_loss (torch.Tensor):
            The ranking loss described in Eqs. (3),(4) of the original paper.
            ranking_loss[-1] is for the target task
            and ranking_loss[:-1] is for the meta tasks.
            The shape is (n_tasks, n_bootstraps).
        n_evals (int):
            The number of config evaluations up to now.
        max_evals (int):
            The maximum number of evaluations during the optimization.
            In other words, the budget of this optimization.
        rng (np.random.RandomState):
            The random number generator.

    Returns:
        ranking_loss (torch.Tensor):
            The ranking loss after applied the weight dilution.
            Basically, it drops the weights for some meta tasks
            with the probability of p_drop := 1 - p_keep.
    """
    (n_tasks, n_bootstraps) = ranking_loss.shape
    better_than_target = torch.sum(ranking_loss[:-1] < ranking_loss[-1], dim=-1)
    p_keep = (better_than_target / n_bootstraps) * (1 - n_evals / max_evals)
    p_keep = torch.hstack([p_keep, torch.tensor(1.0)])  # the target task will not be dropped.

    rnd = torch.as_tensor(rng.random(n_tasks))
    # if rand > p_keep --> drop
    ranking_loss[rnd > p_keep] = torch.max(ranking_loss) * 2 + 1
    return ranking_loss


def leave_one_out_ranks(
    X: torch.Tensor, Y: torch.Tensor, cat_dims: List[int], scalarize: bool, state_dict: OrderedDict
) -> torch.Tensor:
    """
    Compute the ranking of each x in X using leave-one-out cross validation.
    The computation is based on Eq. (4) in the original paper.
    Since we assume multi-objective optimization settings, we use nondominated sort
    to obtain the ranking of each configuration.

    Args:
        X (torch.Tensor):
            The training data used for the task weights.
            In principle, this is the observations in the target task.
            X_train.shape = (n_evals, dim).
        Y (torch.Tensor):
            The training data used for the task weights.
            In principle, this is the observations in the target task.
            Y_train.shape = (n_obj, n_evals).
        cat_dims (List[int]):
            The indices of the categorical parameters.

    Returns:
        ranks (torch.Tensor):
            The ranking of each configuration using the nondominated sort.
            The shape is (n_evals, ).

    NOTE:
        The nondominated sort follows the paper:
            Techniques for Highly Multiobjective Optimisation: Some Nondominated Points are Better than Others

        Furthermore, although we mostly followed the implementation in:
        https://github.com/automl/transfer-hpo-framework/blob/main/rgpe/methods/rgpe.py
        We also referred to the implementation in:
        https://botorch.org/tutorials/meta_learning_with_rgpe
        to check how we train Gaussian models, which is super expensive we always train them from scracth.
        We use `load_state_dict` to speed up as in the BoTorch website.
    """
    (n_obj, n_evals) = Y.shape
    masks = torch.eye(n_evals, dtype=torch.bool)
    loo_preds = np.zeros((n_evals, n_obj))
    for idx, mask in enumerate(masks):
        X_train, Y_train, x_test = X[~mask], Y[:, ~mask], X[mask]
        loo_model = fit_model(
            X_train=X_train, Y_train=Y_train, cat_dims=cat_dims, scalarize=scalarize, state_dict=state_dict
        )
        # predict returns the array with the shape of (batch, n_evals, n_objectives)
        loo_preds[idx] = sample(loo_model, x_test)[0][0].numpy()

    return torch.tensor(nondominated_rank(costs=loo_preds, tie_break=True))


def compute_rank_weights(ranking_loss: torch.Tensor) -> torch.Tensor:
    """
    Compute the rank weights based on Eq. (5).

    Args:
        ranking_loss (torch.Tensor):
            The ranking loss described in Eqs. (3),(4) of the original paper.
            ranking_loss[-1] is for the target task
            and ranking_loss[:-1] is for the meta tasks.
            The shape is (n_tasks, n_bootstraps).

    Returns:
        task_weights (torch.Tensor):
            The task weights with the shape (n_tasks, ).
    """
    (n_tasks, n_bootstraps) = ranking_loss.shape
    sample_wise_min = torch.amin(ranking_loss, dim=0)  # shape = (n_bootstraps, )
    best_counts = torch.zeros(n_tasks)
    best_task_masks = (ranking_loss == sample_wise_min).T  # shape = (n_bootstraps, n_tasks)
    counts_of_best_in_sample = torch.sum(best_task_masks, dim=-1)  # shape = (n_bootstraps, )
    for best_task_mask, count in zip(best_task_masks, counts_of_best_in_sample):
        best_counts[best_task_mask] += 1.0 / count

    return best_counts / n_bootstraps


def compute_ranking_loss(rank_preds: torch.Tensor, rank_targets: torch.Tensor, bs_indices: np.ndarray) -> torch.Tensor:
    """
    Compute the rank weights based on Eqs. (3),(4).

    Args:
        rank_preds (torch.Tensor):
            The ranking predictions on meta tasks and the target task given X.
            rank_preds[-1] is the predictions on the target task using
            leave-one-out cross validation.
            The shape is (n_tasks, n_evals).
        rank_targets (torch.Tensor):
            The ranking on the target task (observations).
            The shape is (n_evals, ).
        bs_indices (np.ndarray):
            The indices for the bootstrapping.
            The shape is (n_bootstraps, n_evals).

    Returns:
        ranking_loss (torch.Tensor):
            The ranking loss described in Eqs. (3),(4) of the original paper.
            ranking_loss[-1] is for the target task
            and ranking_loss[:-1] is for the meta tasks.
            The shape is (n_tasks, n_bootstraps).
    """
    n_tasks = rank_preds.shape[0]
    (n_bootstraps, n_evals) = bs_indices.shape
    bs_preds = torch.stack([r[bs_indices] for r in rank_preds])  # (n_tasks, n_bootstraps, n_evals)
    bs_targets = torch.as_tensor(rank_targets[bs_indices]).reshape((n_bootstraps, n_evals))

    ranking_loss = torch.zeros((n_tasks, n_bootstraps))
    ranking_loss[:-1] += torch.sum(
        (bs_preds[:-1, :, :, None] < bs_preds[:-1, :, None, :]) ^ (bs_targets[:, :, None] < bs_targets[:, None, :]),
        dim=(2, 3),
    )
    ranking_loss[-1] += torch.sum(
        (bs_preds[-1, :, :, None] < bs_targets[:, None, :]) ^ (bs_targets[:, :, None] < bs_targets[:, None, :]),
        dim=(1, 2),
    )
    return ranking_loss


class RankingWeightedGaussianProcessEnsemble(BaseWeightedGP):
    def __init__(
        self,
        init_data: Dict[str, np.ndarray],
        metadata: Dict[str, Dict[str, np.ndarray]],
        bounds: Dict[str, Tuple[NumericType, NumericType]],
        hp_info: Dict[str, HyperParameterType],
        minimize: Dict[str, bool],
        n_bootstraps: int = 1000,
        acq_fn_type: AcqFuncType = "ehvi",
        target_task_name: str = "target_task",
        max_evals: int = 100,
        categories: Optional[Dict[str, List[str]]] = None,
        seed: Optional[int] = None,
    ):
        """The default setting of the paper:
        "Practical transfer learning for Bayesian optimization".
        https://arxiv.org/abs/1802.02219 (Accessed on 18 Aug 2022).

        We followed the implementations provided in:
            * https://github.com/automl/transfer-hpo-framework/blob/main/rgpe/methods/rgpe.py
            ==> The details of ranking loss computations follow this implementation.
            * https://botorch.org/tutorials/meta_learning_with_rgpe
            ==> The details of GP training follows this implementation.

        Args:
            init_data (Dict[str, np.ndarray]):
                The observations of the target task
                sampled from the random sampling.
                Dict[hp_name/obj_name, the array of the corresponding param].
            metadata (Dict[str, Dict[str, np.ndarray]]):
                The observations of the tasks to transfer.
                Dict[task_name, Dict[hp_name/obj_name, the array of the corresponding param]].
            n_bootstraps (int):
                The number of bootstraps.
                For more details, see Algorithm 1 and Section 4.1 in the original paper.
            bounds (Dict[str, Tuple[NumericType, NumericType]]):
                The lower and upper bounds for each hyperparameter.
                Dict[hp_name, Tuple[lower bound, upper bound]].
            hp_info (Dict[str, HyperParameterType]):
                The type information of each hyperparameter.
                Dict[hp_name, HyperParameterType].
            minimize (Dict[str, bool]):
                The direction of the optimization for each objective.
                Dict[obj_name, whether to minimize or not].
            acq_fn_type (Literal[PAREGO, EHVI]):
                The acquisition function type.
            target_task_name (str):
                The name of the target task.
            max_evals (int):
                How many hyperparameter configurations to evaluate during the optimization.
            categories (Optional[Dict[str, List[str]]]):
                Categories for each categorical parameter.
                Dict[categorical hp name, List[each category name]].
            seed (Optional[int]):
                The random seed.

        NOTE:
            This implementation is exclusively for multi-objective optimization settings.
        """
        self._n_bootstraps = n_bootstraps
        super().__init__(
            init_data=init_data,
            metadata=metadata,
            bounds=bounds,
            hp_info=hp_info,
            minimize=minimize,
            acq_fn_type=acq_fn_type,
            target_task_name=target_task_name,
            max_evals=max_evals,
            categories=categories,
            seed=seed,
        )

    def _bootstrap(self, ranks: torch.Tensor, X_train: torch.Tensor, Y_train: torch.Tensor) -> torch.Tensor:
        """
        Perform the bootstrapping described in Section 4.1 of the original paper.

        Args:
            ranks (torch.Tensor):
            X_train (torch.Tensor):
                The training data used for the task weights.
                In principle, this is the observations in the target task.
                X_train.shape = (n_evals, dim).
            Y_train (torch.Tensor):
                The training data used for the task weights.
                In principle, this is the observations in the target task.
                Y_train.shape = (n_obj, n_evals).

        Returns:
            ranking_loss (torch.Tensor):
                The ranking loss described in Eqs. (3),(4) of the original paper.
                ranking_loss[-1] is for the target task
                and ranking_loss[:-1] is for the meta tasks.
                The shape is (n_tasks, n_bootstraps).
        """
        target_state_dict = self._base_models[self._target_task_name].state_dict()
        loo_ranks = leave_one_out_ranks(
            X=X_train,
            Y=Y_train,
            cat_dims=self._cat_dims,
            scalarize=self._acq_fn_type == PAREGO,
            state_dict=target_state_dict,
        )
        rank_preds = torch.vstack([ranks, loo_ranks])
        rank_targets = nondominated_rank(Y_train.T.numpy(), tie_break=True)
        (n_tasks, n_evals) = ranks.shape
        bs_indices = self._rng.choice(n_evals, size=(self._n_bootstraps, n_evals), replace=True)

        return compute_ranking_loss(rank_preds=rank_preds, rank_targets=rank_targets, bs_indices=bs_indices)

    def _compute_rank_weights(self, X_train: torch.Tensor, Y_train: torch.Tensor) -> torch.Tensor:
        """
        Compute the rank weights based on Eq. (5) in Section 4.1. of the original paper.

        Args:
            X_train (torch.Tensor):
                The training data used for the task weights.
                In principle, this is the observations in the target task.
                X_train.shape = (n_evals, dim).
            Y_train (torch.Tensor):
                The training data used for the task weights.
                In principle, this is the observations in the target task.
                Y_train.shape = (n_obj, n_evals).

        Returns:
            torch.Tensor:
                The task weights.
                The sum of the weights must be 1.
                The shape is (n_tasks, ).
        """
        if self._n_tasks == 1 or X_train.shape[0] < 3:  # Not sufficient data points
            return torch.ones(self._n_tasks) / self._n_tasks

        (n_obj, n_evals) = Y_train.shape
        ranks = torch.zeros((self._n_tasks - 1, n_evals), dtype=torch.int32)
        for idx, task_name in enumerate(self._task_names[:-1]):
            model = self._base_models[task_name]
            # flip the sign because larger is better in base models
            rank = nondominated_rank(-sample(model, X_train)[0].numpy(), tie_break=True)
            ranks[idx] = torch.as_tensor(rank)

        ranking_loss = self._bootstrap(ranks=ranks, X_train=X_train, Y_train=Y_train)
        ranking_loss = drop_ranking_loss(
            ranking_loss=ranking_loss,
            n_evals=n_evals,
            max_evals=self._max_evals,
            rng=self._rng,
        )
        rank_weights = compute_rank_weights(ranking_loss=ranking_loss)
        return rank_weights
