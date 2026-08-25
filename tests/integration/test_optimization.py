"""Integration tests for full optimisation pipeline."""

import pytest
import numpy as np
from bayesopt.core.optimizer import BayesOpt


class TestOptimizationPipeline:
    """Test full optimisation pipeline."""
    
    def test_1d_quadratic(self):
        def objective(x):
            return x['x']**2 + 1
        
        bounds = {'x': (-5, 5)}
        opt = BayesOpt(objective, bounds, random_state=42, verbose=0)
        
        result = opt.run(n_calls=15)
        
        assert result.n_calls == 15
        assert abs(result.x['x']) < 0.5
        assert 1.0 <= result.y <= 1.5
    
    def test_2d_branin(self):
        def branin(params):
            x = params['x']
            y = params['y']
            a = 1
            b = 5.1 / (4 * np.pi**2)
            c = 5 / np.pi
            r = 6
            s = 10
            t = 1 / (8 * np.pi)
            return a * (y - b * x**2 + c * x - r)**2 + s * (1 - t) * np.cos(x) + s
        
        bounds = {'x': (-5, 10), 'y': (0, 15)}
        opt = BayesOpt(branin, bounds, random_state=42, verbose=0, n_initial_points=8)
        
        result = opt.run(n_calls=30)
        
        global_min = 0.397887
        assert result.y < 5.0
        assert 0.3 <= result.y <= 10.0
    
    def test_log_scale_parameter(self):
        def objective(x):
            lr = x['lr']
            return (lr - 0.01)**2
        
        bounds = {'lr': (1e-4, 1e-1, 'log')}
        opt = BayesOpt(objective, bounds, random_state=42, verbose=0)
        
        result = opt.run(n_calls=15)
        
        assert 1e-4 <= result.x['lr'] <= 1e-1
        assert result.y < 1e-4
    
    def test_integer_parameter(self):
        def objective(x):
            n = x['n']
            return (n - 10)**2
        
        bounds = {'n': (0, 20, 'int')}
        opt = BayesOpt(objective, bounds, random_state=42, verbose=0)
        
        result = opt.run(n_calls=15)
        
        assert 0 <= result.x['n'] <= 20
        assert result.y < 5.0
        assert result.x['n'] == 10 or abs(result.x['n'] - 10) <= 3
    
    def test_mixed_parameters(self):
        def objective(x):
            return (x['a'] - 2)**2 + (x['b'] - 5)**2 + (0 if x['c'] == 'good' else 10)
        
        bounds = {
            'a': (0, 5),
            'b': (0, 10, 'int'),
            'c': ('good', 'bad')
        }
        opt = BayesOpt(objective, bounds, random_state=42, verbose=0, n_initial_points=10)
        
        result = opt.run(n_calls=25)
        
        assert abs(result.x['a'] - 2) < 1.5
        assert abs(result.x['b'] - 5) <= 2
        assert result.x['c'] == 'good'
    
    def test_with_verbose_output(self):
        def objective(x):
            return x['x']**2
        
        bounds = {'x': (-5, 5)}
        opt = BayesOpt(objective, bounds, random_state=42, verbose=2, explain=True)
        
        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output
        result = opt.run(n_calls=8)
        sys.stdout = sys.__stdout__
        
        output = captured_output.getvalue()
        assert "Starting optimisation" in output
        assert "Initial random exploration" in output
        assert "Bayesian optimisation loop" in output
        assert "OPTIMISATION COMPLETE" not in output  # Not printed by run
    
    def test_result_attributes(self):
        def objective(x):
            return x['x']**2
        
        bounds = {'x': (-5, 5)}
        opt = BayesOpt(objective, bounds, random_state=42, verbose=0)
        
        result = opt.run(n_calls=10)
        
        assert hasattr(result, 'x')
        assert hasattr(result, 'y')
        assert hasattr(result, 'history')
        assert hasattr(result, 'x_iter')
        assert hasattr(result, 'y_iter')
        assert hasattr(result, 'n_calls')
        assert hasattr(result, 'time_taken')
        
        assert len(result.history) == 10
        assert len(result.x_iter) == 10
        assert len(result.y_iter) == 10
        assert result.n_calls == 10
    
    def test_convergence_tracking(self):
        def objective(x):
            return x['x']**2 + 0.1 * np.random.randn()
        
        bounds = {'x': (-5, 5)}
        opt = BayesOpt(objective, bounds, random_state=42, verbose=0)
        
        result = opt.run(n_calls=20)
        
        best_values = []
        current_best = float('inf')
        for y in result.y_iter:
            if y < current_best:
                current_best = y
            best_values.append(current_best)
        
        assert best_values[-1] <= best_values[0]
        assert best_values[-1] < 1.0