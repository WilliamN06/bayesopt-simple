"""Integration tests for suggest/tell API."""

import numpy as np
from bayesopt.core.optimizer import BayesOpt


class TestSuggestTell:
    """Test suggest/tell API."""
    
    def test_suggest_tell_loop(self):
        def objective(x):
            return x['x']**2
        
        bounds = {'x': (-5, 5)}
        opt = BayesOpt(objective, bounds, random_state=42, verbose=0)
        
        for i in range(10):
            if i < 5:
                # Initial random exploration
                params = opt.suggest()[0]
            else:
                params = opt.suggest()[0]
            
            value = objective(params)
            opt.tell(params, value)
        
        assert opt.state.iteration == 10
        assert opt.state.best_y < 1.0
    
    def test_mixed_usage(self):
        def objective(x):
            return x['a']**2 + x['b']**2
        
        bounds = {'a': (-3, 3), 'b': (-3, 3)}
        opt = BayesOpt(objective, bounds, random_state=42, verbose=0)
        
        # Run some iterations
        opt.run(n_calls=5)
        initial_best = opt.state.best_y
        
        # Then use suggest/tell
        params = opt.suggest()[0]
        value = objective(params)
        opt.tell(params, value)
        
        assert opt.state.iteration == 6
        assert opt.state.best_y <= initial_best
    
    def test_multiple_suggestions(self):
        def objective(x):
            return x['x']**2
        
        bounds = {'x': (-5, 5)}
        opt = BayesOpt(objective, bounds, random_state=42, verbose=0)
        
        points = opt.suggest(n_points=5)
        assert len(points) == 5
        
        for params in points:
            value = objective(params)
            opt.tell(params, value)
        
        assert opt.state.iteration == 5