"""Acquisition functions for Bayesian optimisation."""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from typing import Dict, Optional, Tuple, Union

from bayesopt.core.surrogate import GPSurrogate
from bayesopt.core.space import ParameterSpace


class ExpectedImprovement:
    """Expected Improvement acquisition function."""
    
    def __init__(self, Xi: float = 0.01):
        if Xi < 0:
            raise ValueError(f"Xi must be >= 0, got {Xi}")
        self.Xi = Xi
    
    def evaluate(
        self, 
        X: np.ndarray, 
        gp: GPSurrogate, 
        y_best: float
    ) -> np.ndarray:
        if not gp.is_fitted():
            return np.ones(len(X))
        
        mean, std = gp.predict(X)
        std = np.maximum(std, 1e-10)
        
        improvement = mean - y_best - self.Xi
        Z = improvement / std
        ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
        ei = np.maximum(ei, 0.0)
        
        if np.any(np.isnan(ei)) or np.any(np.isinf(ei)):
            return std.flatten()
        
        return ei.flatten()
    
    def optimise(
        self,
        gp: GPSurrogate,
        space: ParameterSpace,
        y_best: float,
        n_restarts: int = 10,
        n_initial_points: int = 100,
        random_state: Optional[int] = None
    ) -> Tuple[np.ndarray, float]:
        if random_state is not None:
            np.random.seed(random_state)
        
        bounds = space.bounds_array()
        n_dims = space.get_dimensions()
        
        if not gp.is_fitted():
            x_opt = np.random.uniform(bounds[:, 0], bounds[:, 1])
            return x_opt, 0.0
        
        initial_points = np.random.uniform(
            bounds[:, 0], 
            bounds[:, 1], 
            size=(n_initial_points, n_dims)
        )
        
        ei_initial = self.evaluate(initial_points, gp, y_best)
        best_idx = np.argmax(ei_initial)
        best_x = initial_points[best_idx]
        best_ei = ei_initial[best_idx]
        
        best_results = []
        starts = [best_x]
        for _ in range(n_restarts - 1):
            starts.append(np.random.uniform(bounds[:, 0], bounds[:, 1]))
        
        for start in starts:
            result = minimize(
                fun=lambda x: -self.evaluate(x.reshape(1, -1), gp, y_best)[0],
                x0=start,
                method='L-BFGS-B',
                bounds=bounds,
                options={'ftol': 1e-6, 'maxiter': 100}
            )
            
            if result.success:
                x_opt = result.x
                x_opt = np.clip(x_opt, bounds[:, 0], bounds[:, 1])
                ei_val = self.evaluate(x_opt.reshape(1, -1), gp, y_best)[0]
                best_results.append((x_opt, ei_val))
        
        if not best_results:
            return best_x, best_ei
        
        best_results.sort(key=lambda x: x[1], reverse=True)
        return best_results[0]
    
    def get_explanation_data(
        self,
        X: np.ndarray,
        gp: GPSurrogate,
        y_best: float
    ) -> Dict[str, Union[np.ndarray, float, str]]:
        if not gp.is_fitted():
            return {
                'mean': 0.0,
                'std': 1.0,
                'ei': 0.0,
                'classification': 'random',
                'exploration_contribution': 1.0,
                'exploitation_contribution': 0.0
            }
        
        mean, std = gp.predict(X)
        mean = mean[0]
        std = std[0]
        
        improvement = mean - y_best - self.Xi
        Z = improvement / max(std, 1e-10)
        ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
        ei = max(ei, 0.0)
        
        std_normalized = std / (abs(y_best) + 1e-5)
        std_normalized = min(std_normalized, 10.0)
        
        if std_normalized > 2.0 and ei > 0:
            classification = 'exploration'
        elif mean > y_best - 0.1 * abs(y_best) and std_normalized < 1.0:
            classification = 'exploitation'
        elif ei > 0:
            classification = 'balanced'
        else:
            classification = 'random'
        
        exp_contrib = min(std_normalized / 5.0, 1.0)
        expl_contrib = 1.0 - exp_contrib
        
        return {
            'mean': mean,
            'std': std,
            'ei': ei,
            'classification': classification,
            'exploration_contribution': exp_contrib,
            'exploitation_contribution': expl_contrib,
            'y_best': y_best,
            'improvement': improvement
        }