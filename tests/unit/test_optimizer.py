"""Unit tests for BayesOpt optimizer."""

import pytest
import numpy as np
from bayesopt.core.optimizer import BayesOpt, OptimisationResult


class TestOptimizer:
    """Test BayesOpt class."""
    
    def test_initialisation(self):
        def objective(x):
            return x['a']**2
        
        bounds = {'a': (-5, 5)}
        opt = BayesOpt(objective, bounds, random_state=42, verbose=1, explain=True)
        
        assert opt.space.n_params == 1
        assert opt.space.param_names == ['a']
        assert opt.verbose == 1
        assert opt.explain is True
    
    def test_initial_exploration(self):
        def objective(x):
            return x['a']**2
        
        bounds = {'a': (-5, 5)}
        opt = BayesOpt(objective, bounds, random_state=42, n_initial_points=3)
        
        opt._initial_exploration()
        assert opt.state.iteration == 3
        assert len(opt.state.history) == 3
        assert opt.state.best_y < 25
    
    def test_evaluate(self):
        def objective(x):
            return x['a']**2
        
        bounds = {'a': (-5, 5)}
        opt = BayesOpt(objective, bounds)
        
        result = opt._evaluate({'a': 3.0})
        assert result == 9.0
        
        def bad_objective(x):
            raise ValueError("Test error")
        
        opt.f = bad_objective
        result = opt._evaluate({'a': 1.0})
        assert result == float('inf')
    
    def test_tell(self):
        def objective(x):
            return x['a']**2
        
        bounds = {'a': (-5, 5)}
        opt = BayesOpt(objective, bounds, verbose=0)
        opt.state.start()
        
        opt.tell({'a': 1.0}, 1.0)
        assert opt.state.iteration == 1
        assert opt.state.best_y == 1.0
        assert opt._n_calls == 1
        
        opt.tell({'a': 0.5}, 0.25)
        assert opt.state.iteration == 2
        assert opt.state.best_y == 0.25
        assert opt._n_calls == 2
    
    def test_suggest(self):
        def objective(x):
            return x['a']**2 + x['b']**2
        
        bounds = {'a': (-5, 5), 'b': (-3, 3)}
        opt = BayesOpt(objective, bounds, random_state=42, verbose=0)
        
        points = opt.suggest(n_points=3)
        assert len(points) == 3
        for point in points:
            assert 'a' in point
            assert 'b' in point
            assert -5 <= point['a'] <= 5
            assert -3 <= point['b'] <= 3
    
    def test_run_basic(self):
        def objective(x):
            return x['x']**2
        
        bounds = {'x': (-5, 5)}
        opt = BayesOpt(objective, bounds, random_state=42, verbose=0)
        
        result = opt.run(n_calls=10)
        
        assert isinstance(result, OptimisationResult)
        assert result.n_calls == 10
        assert result.y <= 1.0  # Should find near minimum
        assert abs(result.x['x']) < 1.0
        assert len(result.history) == 10
        assert len(result.x_iter) == 10
        assert len(result.y_iter) == 10
    
    def test_run_2d(self):
        def objective(x):
            return (x['a'] - 1)**2 + (x['b'] + 2)**2
        
        bounds = {'a': (-5, 5), 'b': (-5, 5)}
        opt = BayesOpt(objective, bounds, random_state=42, verbose=0, n_initial_points=5)
        
        result = opt.run(n_calls=20)
        
        assert result.n_calls == 20
        assert result.y < 5.0
        assert abs(result.x['a'] - 1.0) < 2.0
        assert abs(result.x['b'] + 2.0) < 2.0
    
    def test_run_no_bo_loop(self):
        def objective(x):
            return x['x']**2
        
        bounds = {'x': (-5, 5)}
        opt = BayesOpt(objective, bounds, random_state=42, verbose=0, n_initial_points=10)
        
        result = opt.run(n_calls=5)
        
        assert result.n_calls == 5
        assert len(result.history) == 5
    
    def test_print_summary(self):
        def objective(x):
            return x['x']**2
        
        bounds = {'x': (-5, 5)}
        opt = BayesOpt(objective, bounds, random_state=42, verbose=0)
        
        result = opt.run(n_calls=5)
        
        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output
        result.print_summary()
        sys.stdout = sys.__stdout__
        
        output = captured_output.getvalue()
        assert "OPTIMISATION COMPLETE" in output
        assert "Best parameters:" in output
        assert "Best value:" in output
    
    def test_to_dataframe(self):
        def objective(x):
            return x['x']**2
        
        bounds = {'x': (-5, 5)}
        opt = BayesOpt(objective, bounds, random_state=42, verbose=0)
        
        result = opt.run(n_calls=5)
        
        try:
            import pandas as pd
            df = result.to_dataframe()
            assert len(df) == 5
            assert 'params' in df.columns
            assert 'value' in df.columns
        except ImportError:
            with pytest.raises(ImportError):
                result.to_dataframe()