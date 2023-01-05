# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union

import torch
from torch import Tensor
from torch.nn import ModuleList

from torchmetrics.collections import MetricCollection
from torchmetrics.metric import Metric
from torchmetrics.utilities.prints import rank_zero_warn
from torchmetrics.wrappers import MultioutputWrapper


class MetricTracker(ModuleList):
    """A wrapper class that can help keeping track of a metric or metric collection over time and implement useful
    methods. The wrapper implements the standard ``.update()``, ``.compute()``, ``.reset()`` methods that just
    calls corresponding method of the currently tracked metric. However, the following additional methods are
    provided:

        -``MetricTracker.n_steps``: number of metrics being tracked
        -``MetricTracker.increment()``: initialize a new metric for being tracked
        -``MetricTracker.compute_all()``: get the metric value for all steps
        -``MetricTracker.best_metric()``: returns the best value

    Args:
        metric: instance of a ``torchmetrics.Metric`` or ``torchmetrics.MetricCollection``
            to keep track of at each timestep.
        maximize: either single bool or list of bool indicating if higher metric values are
            better (``True``) or lower is better (``False``).

    Example (single metric):
        >>> from torchmetrics import MetricTracker
        >>> from torchmetrics.classification import MulticlassAccuracy
        >>> _ = torch.manual_seed(42)
        >>> tracker = MetricTracker(MulticlassAccuracy(num_classes=10, average='micro'))
        >>> for epoch in range(5):
        ...     tracker.increment()
        ...     for batch_idx in range(5):
        ...         preds, target = torch.randint(10, (100,)), torch.randint(10, (100,))
        ...         tracker.update(preds, target)
        ...     print(f"current acc={tracker.compute()}")
        current acc=0.1120000034570694
        current acc=0.08799999952316284
        current acc=0.12600000202655792
        current acc=0.07999999821186066
        current acc=0.10199999809265137
        >>> best_acc, which_epoch = tracker.best_metric(return_step=True)
        >>> best_acc  # doctest: +ELLIPSIS
        0.1260...
        >>> which_epoch
        2
        >>> tracker.compute_all()
        tensor([0.1120, 0.0880, 0.1260, 0.0800, 0.1020])

    Example (multiple metrics using MetricCollection):
        >>> from torchmetrics import MetricTracker, MetricCollection, MeanSquaredError, ExplainedVariance
        >>> _ = torch.manual_seed(42)
        >>> tracker = MetricTracker(MetricCollection([MeanSquaredError(), ExplainedVariance()]), maximize=[False, True])
        >>> for epoch in range(5):
        ...     tracker.increment()
        ...     for batch_idx in range(5):
        ...         preds, target = torch.randn(100), torch.randn(100)
        ...         tracker.update(preds, target)
        ...     print(f"current stats={tracker.compute()}")  # doctest: +NORMALIZE_WHITESPACE
        current stats={'MeanSquaredError': tensor(1.8218), 'ExplainedVariance': tensor(-0.8969)}
        current stats={'MeanSquaredError': tensor(2.0268), 'ExplainedVariance': tensor(-1.0206)}
        current stats={'MeanSquaredError': tensor(1.9491), 'ExplainedVariance': tensor(-0.8298)}
        current stats={'MeanSquaredError': tensor(1.9800), 'ExplainedVariance': tensor(-0.9199)}
        current stats={'MeanSquaredError': tensor(2.2481), 'ExplainedVariance': tensor(-1.1622)}
        >>> from pprint import pprint
        >>> best_res, which_epoch = tracker.best_metric(return_step=True)
        >>> pprint(best_res)  # doctest: +ELLIPSIS
        {'ExplainedVariance': -0.829...,
         'MeanSquaredError': 1.821...}
        >>> which_epoch
        {'MeanSquaredError': 0, 'ExplainedVariance': 2}
        >>> pprint(tracker.compute_all())
        {'ExplainedVariance': tensor([-0.8969, -1.0206, -0.8298, -0.9199, -1.1622]),
         'MeanSquaredError': tensor([1.8218, 2.0268, 1.9491, 1.9800, 2.2481])}
    """

    def __init__(self, metric: Union[Metric, MetricCollection], maximize: Union[bool, List[bool]] = True) -> None:
        super().__init__()
        if not isinstance(metric, (Metric, MetricCollection)):
            raise TypeError(
                "Metric arg need to be an instance of a torchmetrics"
                f" `Metric` or `MetricCollection` but got {metric}"
            )
        self._base_metric = metric
        if not isinstance(maximize, (bool, list)):
            raise ValueError("Argument `maximize` should either be a single bool or list of bool")
        if isinstance(maximize, list) and isinstance(metric, MetricCollection) and len(maximize) != len(metric):
            raise ValueError("The len of argument `maximize` should match the length of the metric collection")
        if isinstance(metric, Metric) and not isinstance(maximize, bool):
            raise ValueError("Argument `maximize` should be a single bool when `metric` is a single Metric")
        self.maximize = maximize

        self._increment_called = False

    @property
    def n_steps(self) -> int:
        """Returns the number of times the tracker has been incremented."""
        return len(self) - 1  # subtract the base metric

    def increment(self) -> None:
        """Creates a new instance of the input metric that will be updated next."""
        self._increment_called = True
        self.append(deepcopy(self._base_metric))

    def forward(self, *args, **kwargs) -> None:  # type: ignore
        """Calls forward of the current metric being tracked."""
        self._check_for_increment("forward")
        return self[-1](*args, **kwargs)

    def update(self, *args, **kwargs) -> None:  # type: ignore
        """Updates the current metric being tracked."""
        self._check_for_increment("update")
        self[-1].update(*args, **kwargs)

    def compute(self) -> Any:
        """Call compute of the current metric being tracked."""
        self._check_for_increment("compute")
        return self[-1].compute()

    def compute_all(self) -> Union[Tensor, Dict[str, Tensor], List[Tensor]]:
        """Compute the metric value for all tracked metrics."""
        self._check_for_increment("compute_all")
        # The i!=0 accounts for the self._base_metric should be ignored
        res = [metric.compute() for i, metric in enumerate(self) if i != 0]
        if isinstance(self._base_metric, MetricCollection):
            keys = res[0].keys()
            d = {}
            for k in keys:
                if isinstance(res[0][k], Tensor):
                    d[k] = torch.stack([r[k] for r in res], dim=0)
                elif isinstance(res[0][k], list):
                    d[k] = self._stack_multioutput(multioutput_results=[r[k] for r in res])
                else:
                    raise ValueError(
                        f"Expected each metric's `compute` method to return a Tensor or a list of"
                        f" Tensors but got {type(res[0][k])}"
                    )
            return d
        if isinstance(self._base_metric, MultioutputWrapper):
            return self._stack_multioutput(multioutput_results=res)
        return torch.stack(res, dim=0)

    def reset(self) -> None:
        """Resets the current metric being tracked."""
        self[-1].reset()

    def reset_all(self) -> None:
        """Resets all metrics being tracked."""
        for metric in self:
            metric.reset()

    def best_metric(
        self, return_step: bool = False
    ) -> Union[
        None,
        float,
        Tuple[float, int],
        Tuple[None, None],
        List[float],
        Tuple[List[float], List[int]],
        Dict[str, Union[float, None]],
        Tuple[Dict[str, Union[float, None]], Dict[str, Union[int, None]]],
        Dict[str, Union[List[float], None]],
        Tuple[Dict[str, Union[List[float], None]], Dict[str, Union[List[int], None]]],
    ]:
        """Returns the highest metric out of all tracked.

        Args:
            return_step: If ``True`` will also return the step with the highest metric value.

        Returns:
            The best metric value, and optionally the time-step.
        """
        res = self.compute_all()
        if isinstance(self._base_metric, MetricCollection):
            maximize = self.maximize if isinstance(self.maximize, list) else len(res) * [self.maximize]
            value, idx = {}, {}
            for i, (k, v) in enumerate(res.items()):
                fn = torch.max if maximize[i] else torch.min
                try:
                    # MultioutputWrapper
                    if isinstance(v, list):
                        out = [fn(v_, 0) for v_ in v]
                        value[k], idx[k] = [o[0].item() for o in out], [o[1].item() for o in out]
                    # Simple Metric
                    else:
                        out = fn(v, 0)
                        value[k], idx[k] = out[0].item(), out[1].item()
                except ValueError as error:
                    rank_zero_warn(
                        f"Encountered the following error when trying to get the best metric for metric {k}:"
                        f"{error} this is probably due to the 'best' not being defined for this metric."
                        "Returning `None` instead.",
                        UserWarning,
                    )
                    value[k], idx[k] = None, None

        elif isinstance(self._base_metric, MultioutputWrapper):
            fn = torch.max if self.maximize else torch.min
            try:
                out = [fn(v, 0) for v in res]
                value, idx = [o[0].item() for o in out], [o[1].item() for o in out]
            except ValueError as error:
                rank_zero_warn(
                    f"Encountered the following error when trying to get the best metric: {error}"
                    "this is probably due to the 'best' not being defined for this metric."
                    "Returning `None` instead.",
                    UserWarning,
                )
                value, idx = None, None

        # Simple Metric
        else:
            fn = torch.max if self.maximize else torch.min
            try:
                out = fn(res, 0)
                value, idx = out[0].item(), out[1].item()
            except ValueError as error:
                rank_zero_warn(
                    f"Encountered the following error when trying to get the best metric: {error}"
                    "this is probably due to the 'best' not being defined for this metric."
                    "Returning `None` instead.",
                    UserWarning,
                )
                value, idx = None, None

        if return_step:
            return value, idx
        return value

    def _check_for_increment(self, method: str) -> None:
        if not self._increment_called:
            raise ValueError(f"`{method}` cannot be called before `.increment()` has been called")

    @staticmethod
    def _stack_multioutput(multioutput_results: List[List[Tensor]]):
        stacked_results = []
        for i in range(len(multioutput_results[0])):
            stacked_results.append(
                torch.stack([multioutput_results[j][i] for j in range(len(multioutput_results))], dim=0)
            )
        return stacked_results
