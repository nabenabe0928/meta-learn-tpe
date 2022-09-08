from typing import Dict, List, Optional, Tuple

from fast_pareto import nondominated_rank

from optimizers.meta_learn_bo.models.base_weighted_gp import BaseWeightedGP
from optimizers.meta_learn_bo.utils import AcqFuncType, HyperParameterType, NumericType, sample

import numpy as np

import torch


def compute_ranking_loss(rank_preds: np.ndarray, rank_targets: np.ndarray, bandwidth: float) -> torch.Tensor:
    """
    Compute the rank weights based on the original paper.

    Args:
        rank_preds (np.ndarray):
            The ranking predictions on meta tasks given X.
            The shape is (n_tasks - 1, n_evals).
        rank_targets (np.ndarray):
            The ranking on the target task (observations).
            The shape is (n_evals, ).
        bandwidth (float):
            rho in the original paper.

    Returns:
        ranking_loss (torch.Tensor):
            The ranking loss described in the original paper.
            It does not include the ranking loss for the target task unlike RGPE.
            The shape is (n_tasks - 1, n_evals, n_evals).
    """
    n_evals = rank_targets.shape[-1]
    n_pairs = n_evals * (n_evals - 1)
    discordant_info = (rank_preds[:, :, np.newaxis] < rank_preds[:, np.newaxis, :]) ^ (
        rank_targets[:, np.newaxis] < rank_targets
    )
    return torch.as_tensor(np.sum(discordant_info, axis=(1, 2)) / (n_pairs * bandwidth))


class TwoStageTransferWithRanking(BaseWeightedGP):
    def __init__(
        self,
        init_data: Dict[str, np.ndarray],
        metadata: Dict[str, Dict[str, np.ndarray]],
        bounds: Dict[str, Tuple[NumericType, NumericType]],
        hp_info: Dict[str, HyperParameterType],
        minimize: Dict[str, bool],
        acq_fn_type: AcqFuncType = "ehvi",
        target_task_name: str = "target_task",
        max_evals: int = 100,
        bandwidth: float = 0.1,
        categories: Optional[Dict[str, List[str]]] = None,
        seed: Optional[int] = None,
    ):
        """Two-stage transfer surrogate with ranking from the paper:
        "Scalable Gaussian process-based transfer surrogates for hyperparameter optimization"
        https://link.springer.com/article/10.1007/s10994-017-5684-y

        Args:
            init_data (Dict[str, np.ndarray]):
                The observations of the target task
                sampled from the random sampling.
                Dict[hp_name/obj_name, the array of the corresponding param].
            metadata (Dict[str, Dict[str, np.ndarray]]):
                The observations of the tasks to transfer.
                Dict[task_name, Dict[hp_name/obj_name, the array of the corresponding param]].
            bounds (Dict[str, Tuple[NumericType, NumericType]]):
                The lower and upper bounds for each hyperparameter.
                Dict[hp_name, Tuple[lower bound, upper bound]].
            hp_info (Dict[str, HyperParameterType]):
                The type information of each hyperparameter.
                Dict[hp_name, HyperParameterType].
            bandwidth (float):
                rho in the original paper.
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
        self._bandwidth = bandwidth
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

    def _compute_rank_weights(self, X_train: torch.Tensor, Y_train: torch.Tensor) -> torch.Tensor:
        """
        Compute the rank weights based on Section 6.2 in the original paper.

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
        n_evals = X_train.shape[0]
        if self._n_tasks == 1 or n_evals < 2:  # Not sufficient data points
            return torch.ones(self._n_tasks) / self._n_tasks

        # ranks.shape = (n_tasks - 1, n_evals)
        rank_preds = np.asarray(
            [
                # flip the sign because larger is better in base models
                nondominated_rank(-sample(self._base_models[task_name], X_train)[0].numpy(), tie_break=True)
                for task_name in self._task_names[:-1]
            ]
        )
        rank_targets = nondominated_rank(Y_train.T.numpy(), tie_break=True)
        ranking_loss = compute_ranking_loss(rank_preds=rank_preds, rank_targets=rank_targets, bandwidth=self._bandwidth)
        ts = torch.minimum(ranking_loss, torch.tensor(1.0))

        weights = torch.ones(self._n_tasks) * 0.75
        weights[:-1] *= 1 - ts**2
        return weights / torch.sum(weights)  # normalize and return
