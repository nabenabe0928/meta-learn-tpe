from typing import List, Union

from botorch.acquisition import ExpectedImprovement
from botorch.acquisition.multi_objective import ExpectedHypervolumeImprovement
from botorch.acquisition.multi_objective.analytic import MultiObjectiveAnalyticAcquisitionFunction

from optimizers.meta_learn_bo.utils import validate_weights

import torch


EPS = 1e-8


class TransferAcquisitionFunction(MultiObjectiveAnalyticAcquisitionFunction):
    def __init__(
        self, acq_fn_list: List[Union[ExpectedHypervolumeImprovement, ExpectedImprovement]], weights: torch.Tensor
    ):
        """Transfer acquisition function proposed in the paper:
        "Scalable Gaussian process-based transfer surrogates for hyperparameter optimization"
        https://link.springer.com/article/10.1007/s10994-017-5684-y

        Args:
            acq_fn_list (List[Union[ExpectedHypervolumeImprovement, ExpectedImprovement]]):
                The list of acquisition functions to take the linear combination.
            weights (torch.Tensor):
                The task weights for each acquisition funciton.
        """
        super().__init__(model=None)
        self._acq_fn_list = acq_fn_list
        self._weights = weights
        self._validate_input()

    def _validate_input(self) -> None:
        validate_weights(self._weights)
        acq_fn_cls = type(self._acq_fn_list[0])
        if not all(isinstance(acq_fn, acq_fn_cls) for acq_fn in self._acq_fn_list):
            raise TypeError(f"All acquisition function must be the class {acq_fn_cls}")

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute the transfer acquisition function values.

        Args:
            X (torch.Tensor):
                The input tensor X with the shape of (batch_size, q, dim).
                `q` is the number of candidates that must be considered jointly.

        Returns:
            out (torch.Tensor):
                The transfer acquisition function values.
                The shape is (batch size).
        """
        batch_size = X.shape[0]
        out = torch.zeros((batch_size,), dtype=torch.float64)
        for acq_fn, weight in zip(self._acq_fn_list, self._weights):
            if weight > EPS:  # basically, if weight is non-zero, we compute
                out += weight * acq_fn(X)

        return out
