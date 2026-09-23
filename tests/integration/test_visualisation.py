"""Integration tests for visualisation components."""

import pytest
import numpy as np
from bayesopt.core.optimizer import BayesOpt
from bayesopt.core.space import ParameterSpace
from bayesopt.core.surrogate import GPSurrogate
from bayesopt.core.acquisition import ExpectedImprovement


class TestVisualisationIntegration:
    """Test visualisation integration with other components."""
    
    def test_convergence_with_optimizer(self):
        def objective(x):
            return x['x']**2
        
        bounds = {'x': (-5, 5)}
        opt = BayesOpt(objective, bounds, random_state=42, verbose=0)
        result = opt.run(n_calls=10)
        
        # Should not raise
        opt.plot('convergence', save_path=None)
        opt.plot('improvement', save_path=None)
    
    def test_importance_with_optimizer(self):
        def objective(x):
            return x['a']**2 + x['b']**2
        
        bounds = {'a': (-3, 3), 'b': (-3, 3)}
        opt = BayesOpt(objective, bounds, random_state=42, verbose=0)
        opt.run(n_calls=10)
        
        opt.plot('importance', save_path=None)
    
    def test_acquisition_1d_with_optimizer(self):
        def objective(x):
            return x['x']**2
        
        bounds = {'x': (-5, 5)}
        opt = BayesOpt(objective, bounds, random_state=42, verbose=0)
        opt.run(n_calls=10)
        
        opt.plot('acquisition', save_path=None)
    
    def test_acquisition_2d_with_optimizer(self):
        def objective(x):
            return x['a']**2 + x['b']**2
        
        bounds = {'a': (-3, 3), 'b': (-3, 3)}
        opt = BayesOpt(objective, bounds, random_state=42, verbose=0)
        opt.run(n_calls=10)
        
        opt.plot('acquisition', save_path=None)
    
    def test_acquisition_3d_raises_error(self):
        def objective(x):
            return x['a']**2 + x['b']**2 + x['c']**2
        
        bounds = {'a': (-3, 3), 'b': (-3, 3), 'c': (-3, 3)}
        opt = BayesOpt(objective, bounds, random_state=42, verbose=0)
        opt.run(n_calls=10)
        
        with pytest.raises(ValueError):
            opt.plot('acquisition')
    
    def test_plot_all(self):
        def objective(x):
            return x['a']**2 + x['b']**2
        
        bounds = {'a': (-3, 3), 'b': (-3, 3)}
        opt = BayesOpt(objective, bounds, random_state=42, verbose=0)
        opt.run(n_calls=10)
        
        opt.plot('all', save_path=None)
    
    def test_plot_unknown_type(self):
        def objective(x):
            return x['x']**2
        
        bounds = {'x': (-5, 5)}
        opt = BayesOpt(objective, bounds, random_state=42, verbose=0)
        opt.run(n_calls=5)
        
        with pytest.raises(ValueError):
            opt.plot('unknown')
    
    def test_plot_without_running(self):
        def objective(x):
            return x['x']**2
        
        bounds = {'x': (-5, 5)}
        opt = BayesOpt(objective, bounds, random_state=42, verbose=0)
        
        with pytest.raises(RuntimeError):
            opt.plot('importance')
        
        with pytest.raises(RuntimeError):
            opt.plot('acquisition')