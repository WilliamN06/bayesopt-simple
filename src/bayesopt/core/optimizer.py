"""Main BayesOpt optimizer class."""

from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import time
from dataclasses import dataclass, field

from bayesopt.core.space import ParameterSpace
from bayesopt.core.surrogate import GPSurrogate
from bayesopt.core.acquisition import ExpectedImprovement
from bayesopt.core.state import OptimizerState
from bayesopt.explanation.engine import ExplanationEngine, ExplanationFormatter
from bayesopt.explanation.translators import GPStateTranslator, DecisionTranslator
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
        
        self.logger = ConsoleLogger(verbose=verbose)
        self.explanation_engine = ExplanationEngine(verbose=verbose)
        self.formatter = ExplanationFormatter()
        
        self._initialised = False
        self._n_calls = 0
    
    def _log(self, message: str, level: int = 1) -> None:
        """Log message based on verbosity."""
        self.logger.info(message, level)
    
    def _log_explanation(self, explanation: str, level: int = 1) -> None:
        """Log explanation text."""
        self.logger.log_explanation(explanation, level)
    
    def _initial_exploration(self) -> None:
        """Run initial random exploration."""
        self.logger.section("Initial Exploration", 1)
        self._log(f"Sampling {self.n_initial_points} random points", 1)
        
        for i in range(self.n_initial_points):
            params = self.space.sample_random()
            value = self._evaluate(params)
            self.state.update(params, value)
            
            if self.explain and self.verbose >= 2:
                explanation = self.explanation_engine.explain_iteration(
                    i + 1, params, value, self.state.best_y,
                    {'classification': 'random'}
                )
                self._log_explanation(explanation, 2)
            else:
                self._log(f"  Point {i+1}: value = {value:.6f}", 2)
        
        self.logger.success(f"Best so far: {self.state.best_y:.6f}", 1)
    
    def _evaluate(self, params: Dict) -> float:
        """Evaluate objective function with error handling."""
        try:
            return float(self.f(params))
        except Exception as e:
            self.logger.error(f"Objective function error: {e}", 1)
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
            explanation_data = self.ei.get_explanation_data(
                x_opt.reshape(1, -1), self.gp, y_best
            )
            explanation = self.explanation_engine.explain_iteration(
                self.state.iteration + 1, params, 0.0, y_best, explanation_data
            )
            self._log_explanation(explanation, 2)
        
        return params
    
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
        was_best = y < self.state.best_y
        self.state.update(x, y)
        self._n_calls += 1
        
        if self.verbose >= 1:
            if was_best and self._n_calls > self.n_initial_points:
                improvement = self.state.best_y - y
                self.logger.success(f"New best! {y:.6f} (improvement: {improvement:.6f})", 1)
            else:
                self._log(f"  Evaluation {self._n_calls}: value = {y:.6f}", 2)
        
        if self.explain and self.verbose >= 1 and self._n_calls % 5 == 0:
            progress = self.explanation_engine.explain_progress(
                self._n_calls, self.state.iteration + 1,
                self.state.best_y, self.state.history
            )
            self._log_explanation(progress, 1)
    
    def run(self, n_calls: int = 50) -> OptimisationResult:
        """Run optimisation for n_calls evaluations."""
        self.state.start()
        self._n_calls = 0
        
        self.logger.section("Bayesian Optimisation", 1)
        self._log(f"Target: {n_calls} evaluations", 1)
        self._log(f"Parameters: {', '.join(self.space.param_names)}", 2)
        
        self._initial_exploration()
        self._initialised = True
        
        remaining = n_calls - self.n_initial_points
        if remaining <= 0:
            self.logger.warning("n_calls <= n_initial_points, skipping BO loop", 1)
        else:
            self._log(f"Starting BO loop ({remaining} iterations)", 1)
            
            for i in range(remaining):
                params = self._get_next_point()
                value = self._evaluate(params)
                self.tell(params, value)
        
        elapsed = time.time() - self.state.start_time if self.state.start_time else 0.0
        
        best_x, best_y = self.state.get_best()
        history = self.state.get_history()
        
        x_iter = [entry['params'] for entry in history]
        y_iter = [entry['value'] for entry in history]
        
        if self.explain and self.verbose >= 1:
            gp_stats = self.gp.get_stats()
            final_explanation = self.explanation_engine.explain_final_result(
                best_x, best_y, len(history), elapsed, gp_stats
            )
            self._log_explanation(final_explanation, 1)
        
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