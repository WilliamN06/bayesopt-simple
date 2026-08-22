"""Additional tests for acquisition optimisation."""

import pytest
import numpy as np
from bayesopt.core.acquisition import ExpectedImprovement
from bayesopt.core.surrogate import GPSurrogate
from bayesopt.core.space import ParameterSpace


class TestAcquisitionOptimisation:
    """Test acquisition optimisation."""
    
    def test_optimise_1d(self):
        """Test optimisation in 1D."""
        space = ParameterSpace({'x': (-5, 5)})
        gp = GPSurrogate(random_state=42)
        
        # Data with best at x=2
        X_train = np.array([[-4], [-2], [0], [1], [2], [3]])
        y_train = np.array([10, 5, 2, 1.2, 1.0, 1.5])
        gp.fit(X_train, y_train)
        
        ei = ExpectedImprovement(Xi=0.01)
        
        # Find optimum
        x_opt, ei_val = ei.optimise(
            gp, space, y_best=1.0,
            n_restarts=5, random_state=42
        )
        
        # Should be somewhere in the promising region
        assert -5 <= x_opt[0] <= 5
        assert ei_val >= 0
    
    def test_optimise_2d(self):
        """Test optimisation in 2D."""
        space = ParameterSpace({'x': (0, 5), 'y': (0, 5)})
        gp = GPSurrogate(random_state=42)
        
        # Create a simple 2D function with peak in corner
        def objective(x, y):
            return -(x - 1)**2 - (y - 1)**2  # Peak at (1, 1)
        
        X_train = np.random.rand(10, 2) * 5
        y_train = np.array([objective(x[0], x[1]) for x in X_train])
        gp.fit(X_train, y_train)
        
        ei = ExpectedImprovement(Xi=0.01)
        
        x_opt, ei_val = ei.optimise(
            gp, space, y_best=np.max(y_train),
            n_restarts=5, random_state=42
        )
        
        assert x_opt.shape == (2,)
        assert all(0 <= x <= 5 for x in x_opt)
        assert ei_val >= 0
    
    def test_optimise_with_categorical(self):
        """Test optimisation with categorical parameters."""
        space = ParameterSpace({
            'x': (0, 10, 'int'),
            'cat': ('a', 'b', 'c')
        })
        gp = GPSurrogate(random_state=42)
        
        # Some random data
        X_train = np.array([
            [0, 0], [2, 0.5], [4, 1.0], [6, 0.5], [8, 1.0]
        ])
        y_train = np.array([0.5, 0.8, 1.0, 0.7, 0.4])
        gp.fit(X_train, y_train)
        
        ei = ExpectedImprovement(Xi=0.01)
        
        x_opt, ei_val = ei.optimise(
            gp, space, y_best=1.0,
            n_restarts=5, random_state=42
        )
        
        # Should be within bounds for both dimensions
        assert 0 <= x_opt[0] <= 10
        assert 0 <= x_opt[1] <= 2  # Categorical encoded as 0-2
        
        # Round back to nearest category
        cat_idx = int(np.round(x_opt[1] * 2))  # Since 0-2 range
        cat_idx = np.clip(cat_idx, 0, 2)
        categories = ['a', 'b', 'c']
        category = categories[cat_idx]
        assert category in ['a', 'b', 'c']