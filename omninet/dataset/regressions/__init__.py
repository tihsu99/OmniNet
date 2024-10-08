from typing import List, Dict, Type, Callable
from torch import Tensor

from omninet.dataset.regressions.base_regression import Regression
from omninet.dataset.types import Statistics
from omninet.dataset.regressions.gaussian_regression import GaussianRegression
from omninet.dataset.regressions.laplacian_regression import LaplacianRegression
from omninet.dataset.regressions.log_gaussian_regression import LogGaussianRegression


all_regressions: List[Type[Regression]] = [
    GaussianRegression,
    LaplacianRegression,
    LogGaussianRegression
]

regression_mapping: Dict[str, Type[Regression]] = {
    regression.name(): regression
    for regression in all_regressions
}


def regression_class(name: str) -> Type[Regression]:
    return regression_mapping[name]


def regression_loss(name: str) -> Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]:
    return regression_mapping[name].loss


def regression_statistics(name: str) -> Callable[[Tensor], Statistics]:
    return regression_mapping[name].statistics

