"""Main BayesOpt optimizer class."""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import time
from dataclasses import dataclass, field

from bayesopt.core.space import ParameterSpace
from bayesopt.core.surrogate import GPSurrogate
from bayesopt.core.acquisition import ExpectedImprovement
from bayesopt.core.state import OptimizerState


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
        print(f"Best parameters:")
        for key, value in self.x.items():
            print(f"  {key}: {value}")
        print(f"\nBest value: {self.y:.6f}")
        print(f"Total evaluations: {self.n_calls}")
        print(f"Time taken: {self.time_taken:.2f}s")
        print("=" * 60)
    
    def to_dataframe(self):
        """Export history as pandas DataFrame."""
        try:
            import pandas as pd
            return pd.DataFrame(self.history)
        except ImportError:
            raise ImportError("pandas is required for to_dataframe()")


class BayesOpt:
    """Bayesian Optimisation for ML practitioners."""
    
    def __init__(
        self,
        f: Callable[[Dict[str, Union[float, int, str]]], float],
        bounds: Dict[str, Tuple],
        random_state: int = 42,
        verbose: int = 1,
        explain: bool = True,
        n_initial_points: int = 5
    ):
        self.f = f
        self.bounds = bounds
        self.random_state = random_state
        self.verbose = verbose
        self.explain = explain
        self.n_initial_points = n_initial_points
        
        np.random.seed(random_state)
        
        self.space = ParameterSpace(bounds)
        self.gp = GPSurrogate(random_state=random_state)
        self.ei = ExpectedImprovement(Xi=0.01)
        self.state = OptimizerState()
        
        self._initialised = False
        self._n_calls = 0
    
    def _log(self, message: str, level: int = 1) -> None:
        """Log message based on verbosity."""
        if self.verbose >= level:
            print(message)
    
    def _initial_exploration(self) -> None:
        """Run initial random exploration."""
        self._log(f"Starting initial random exploration ({self.n_initial_points} points)", 1)
        
        for i in range(self.n_initial_points):
            params = self.space.sample_random()
            value = self._evaluate(params)
            self.state.update(params, value)
            
            self._log(f"  Point {i+1}/{self.n_initial_points}: value = {value:.6f}", 2)
        
        self._log(f"Initial exploration complete. Best value: {self.state.best_y:.6f}", 1)
    
    def _evaluate(self, params: Dict) -> float:
        """Evaluate objective function with error handling."""
        try:
            return float(self.f(params))
        except Exception as e:
            self._log(f"Warning: Objective function error: {e}", 1)
            return float('inf')
    
    def _get_next_point(self) -> Dict:
        """Get next point to evaluate using acquisition optimisation."""
        if not self.gp.is_fitted():
            return self.space.sample_random()
        
        X, y = self.state.get_X_y()
        if len(X) == 0:
            return self.space.sample_random()
        
        self.gp.fit(X, y)
        
        y_best = self.state.best_y
        
        x_opt, ei_val = self.ei.optimise(
            self.gp, self.space, y_best,
            n_restarts=10, random_state=self.random_state
        )
        
        params = self.space.from_array(x_opt)
        
        if self.explain and self.verbose >= 2:
            explanation = self.ei.get_explanation_data(
                x_opt.reshape(1, -1), self.gp, y_best
            )
            self._log_explanation(explanation, self.state.iteration + 1)
        
        return params
    
    def _log_explanation(self, explanation: Dict, iteration: int) -> None:
        """Log explanation of acquisition decision."""
        classification = explanation.get('classification', 'unknown')
        mean = explanation.get('mean', 0.0)
        std = explanation.get('std', 0.0)
        ei = explanation.get('ei', 0.0)
        
        if classification == 'exploration':
            msg = f"Exploring uncertain region (std={std:.4f})"
        elif classification == 'exploitation':
            msg = f"Exploiting promising region (mean={mean:.4f})"
        elif classification == 'balanced':
            msg = f"Balancing exploration/exploitation (ei={ei:.4f})"
        else:
            msg = f"Random exploration"
        
        self._log(f"Iteration {iteration}: {msg}", 2)
    
    def suggest(self, n_points: int = 1) -> List[Dict]:
        """Get next point(s) to evaluate."""
        if not self._initialised:
            self._initial_exploration()
            self._initialised = True
        
        if n_points == 1:
            return [self._get_next_point()]
        else:
            return [self._get_next_point() for _ in range(n_points)]
    
    def tell(self, x: Dict, y: float) -> None:
        """Tell the optimizer about a previous evaluation."""
        self.state.update(x, y)
        self._n_calls += 1
        
        if self.verbose >= 1:
            self._log(f"  Evaluation {self._n_calls}: value = {y:.6f}", 1)
            if y < self.state.best_y:
                self._log(f"  New best! {y:.6f}", 1)
    
    def run(self, n_calls: int = 50) -> OptimisationResult:
        """Run optimisation for n_calls evaluations."""
        self.state.start()
        self._n_calls = 0
        
        self._log(f"Starting optimisation for {n_calls} evaluations", 1)
        self._log(f"Parameter space: {self.space.param_names}", 2)
        
        self._initial_exploration()
        self._initialised = True
        
        remaining = n_calls - self.n_initial_points
        if remaining <= 0:
            self._log("Warning: n_calls <= n_initial_points, skipping BO loop", 1)
        else:
            self._log(f"Starting Bayesian optimisation loop ({remaining} iterations)", 1)
            
            for i in range(remaining):
                params = self._get_next_point()
                value = self._evaluate(params)
                self.tell(params, value)
        
        elapsed = time.time() - self.state.start_time if self.state.start_time else 0.0
        
        best_x, best_y = self.state.get_best()
        history = self.state.get_history()
        
        x_iter = [entry['params'] for entry in history]
        y_iter = [entry['value'] for entry in history]
        
        return OptimisationResult(
            x=best_x,
            y=best_y,
            history=history,
            x_iter=x_iter,
            y_iter=y_iter,
            n_calls=len(history),
            time_taken=elapsed
        )
    
    def plot(self, plot_type: str = 'convergence') -> None:
        """Generate visualisation of the optimisation."""
        raise NotImplementedError("Visualisation will be implemented in Milestone 4")