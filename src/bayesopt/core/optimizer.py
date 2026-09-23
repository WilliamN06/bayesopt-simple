"""Main BayesOpt optimizer class."""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import time
from dataclasses import dataclass, field

from bayesopt.core.space import ParameterSpace
from bayesopt.core.surrogate import GPSurrogate
from bayesopt.core.acquisition import ExpectedImprovement
from bayesopt.core.state import OptimizerState
from bayesopt.explanation.engine import ExplanationEngine
from bayesopt.utils.logging import ConsoleLogger


@dataclass
class OptimisationResult:
    """Container for optimisation results."""
    x: Dict
    y: float
    history: List[Dict] = field(default_factory=list)
    x_iter: List[Dict] = field(default_factory=list)
    y_iter: List[float] = field(default_factory=list)
    n_calls: int = 0
    time_taken: float = 0.0
    
    def print_summary(self) -> None:
        """Pretty print results."""
        print("=" * 60)
        print("OPTIMISATION COMPLETE")
        print("=" * 60)
        print("Best parameters:")
        for key, value in self.x.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
        print(f"\nBest value: {self.y:.6f}")
        print(f"Total evaluations: {self.n_calls}")
        print(f"Time taken: {self.time_taken:.2f}s")
        print("=" * 60)
    
    def to_dataframe(self):
        """Export history as pandas DataFrame."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for to_dataframe()")
        
        rows = []
        for entry in self.history:
            row = dict(entry['params'])
            row['iteration'] = entry.get('iteration', len(rows) + 1)
            row['value'] = entry['value']
            row['timestamp'] = entry.get('timestamp', 0.0)
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def get_best_index(self) -> int:
        """Get the iteration index of the best value."""
        if not self.y_iter:
            return -1
        return int(np.argmin(self.y_iter))
    
    def improvement_over_initial(self) -> float:
        """Calculate improvement over initial best."""
        if len(self.y_iter) < 2:
            return 0.0
        return self.y_iter[0] - self.y