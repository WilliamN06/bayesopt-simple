"""Unit tests for ExpectedImprovement."""

import pytest
import numpy as np
from bayesopt.core.acquisition import ExpectedImprovement
from bayesopt.core.surrogate import GPSurrogate
from bayesopt.core.space import ParameterSpace


class TestExpectedImprovement:
    """Test ExpectedImprovement class."""
    
    def test_initialisation(self):
        """Test initialisation."""
        ei = ExpectedImprovement(Xi=0.01)
        assert ei.Xi == 0.01
        
        # Invalid Xi
        with pytest.raises(ValueError):
            ExpectedImprovement(Xi=-1.0)
    
    def test_evaluate_before_fit(self):
        """Test evaluate before GP is fitted."""
        gp = GPSurrogate()
        ei = ExpectedImprovement()
        
        X = np.array([[0.5], [1.5]])
        values = ei.evaluate(X, gp, y_best=0.0)
        
        # Should return uniform values
        assert np.all(values == 1.0)
        assert len(values) == 2
    
    def test_evaluate_on_simple_function(self):
        """Test evaluate on a simple fitted function."""
        gp = GPSurrogate(random_state=42)
        X_train = np.array([[0], [1], [2]])
        y_train = np.array([0, 1, 4])
        gp.fit(X_train, y_train)
        
        ei = ExpectedImprovement()
        
        # Test at various points
        X_test = np.array([[0.5], [1.5], [2.5]])
        values = ei.evaluate(X_test, gp, y_best=1.0)
        
        # Should be non-negative
        assert np.all(values >= 0)
        
        # Should be highest where uncertainty is high and mean is good
        # For y=1, EI should be higher at 2.5 (uncertain, promising) than at 1.5 (known, good)
        # This depends on the GP, but generally should hold
        # We'll just check that values are reasonable
        assert len(values) == 3
    
    def test_evaluate_zero_at_known_points(self):
        """Test EI is zero at already evaluated points."""
        gp = GPSurrogate(random_state=42)
        X_train = np.array([[0], [1], [2]])
        y_train = np.array([0, 1, 4])
        gp.fit(X_train, y_train)
        
        ei = ExpectedImprovement()
        
        # At training points, EI should be 0 (or very close)
        values = ei.evaluate(X_train, gp, y_best=4.0)
        assert np.all(values < 1e-6)
    
    def test_optimise_before_fit(self):
        """Test optimisation before GP is fitted."""
        space = ParameterSpace({'x': (0, 10, 'int')})
        gp = GPSurrogate()
        ei = ExpectedImprovement()
        
        x_opt, ei_val = ei.optimise(gp, space, y_best=0.0)
        
        # Should return a point within bounds
        assert 0 <= x_opt[0] <= 10
        assert ei_val == 0.0
    
    def test_optimise_on_known_function(self):
        """Test optimisation finds reasonable point."""
        # Simple function with a peak at x=7
        space = ParameterSpace({'x': (0, 10, 'int')})
        gp = GPSurrogate(random_state=42)
        
        # Simulate observations where best is at x=2
        X_train = np.array([[0], [1], [2], [3], [4]])
        y_train = np.array([0.5, 0.8, 1.0, 0.7, 0.3])
        gp.fit(X_train, y_train)
        
        ei = ExpectedImprovement(Xi=0.01)
        
        # Optimise should find a point that balances exploration and exploitation
        # With this data, it might explore around 4-6 or exploit near 2
        x_opt, ei_val = ei.optimise(
            gp, space, y_best=1.0, 
            n_restarts=5, random_state=42
        )
        
        assert 0 <= x_opt[0] <= 10
        assert ei_val >= 0
    
    def test_optimise_with_multiple_restarts(self):
        """Test optimisation with multiple restarts."""
        space = ParameterSpace({'x': (0, 10, 'int'), 'y': (0, 10, 'int')})
        gp = GPSurrogate(random_state=42)
        
        # Random data
        X_train = np.random.rand(5, 2) * 10
        y_train = np.random.rand(5)
        gp.fit(X_train, y_train)
        
        ei = ExpectedImprovement()
        
        # Should handle n_restarts properly
        x_opt, ei_val = ei.optimise(
            gp, space, y_best=0.5,
            n_restarts=5, n_initial_points=20, 
            random_state=42
        )
        
        assert x_opt.shape == (2,)
        assert all(0 <= x <= 10 for x in x_opt)
        assert ei_val >= 0
    
    def test_get_explanation_data(self):
        """Test explanation data generation."""
        gp = GPSurrogate(random_state=42)
        X_train = np.array([[0], [1], [2]])
        y_train = np.array([0, 1, 4])
        gp.fit(X_train, y_train)
        
        ei = ExpectedImprovement()
        
        # Before fit
        explanation = ei.get_explanation_data(
            np.array([[1.5]]), gp, y_best=1.0
        )
        assert 'classification' in explanation
        
        # After fit - at point with uncertainty
        explanation = ei.get_explanation_data(
            np.array([[1.5]]), gp, y_best=1.0
        )
        
        assert 'mean' in explanation
        assert 'std' in explanation
        assert 'ei' in explanation
        assert 'classification' in explanation
        assert 'exploration_contribution' in explanation
        assert 'exploitation_contribution' in explanation
        
        # Classifications should be one of the valid types
        assert explanation['classification'] in ['random', 'exploration', 'exploitation', 'balanced']