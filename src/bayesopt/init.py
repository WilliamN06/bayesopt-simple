"""BayesOpt Made Simple - Bayesian optimisation for ML practitioners."""

from bayesopt.core.optimizer import BayesOpt, OptimisationResult
from bayesopt.core.space import ParameterSpace, ParameterType
from bayesopt.core.surrogate import GPSurrogate
from bayesopt.core.acquisition import ExpectedImprovement

__version__ = "1.0.0"
__all__ = [
    "BayesOpt",
    "OptimisationResult",
    "ParameterSpace",
    "ParameterType",
    "GPSurrogate",
    "ExpectedImprovement",
]