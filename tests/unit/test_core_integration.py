"""Integration tests for core components working together."""

import pytest
import numpy as np
from bayesopt.core.space import ParameterSpace
from bayesopt.core.surrogate import GPSurrogate
from bayesopt.core.acquisition import ExpectedImprovement


class TestCoreIntegration:
    """Test integration of core components."""
    
    def test_full_workflow_1d(self):
        bounds = {'x': (-5, 5)}
        space = ParameterSpace(bounds)
        
        X_init = np.array([[-4], [-2], [0], [2]])
        y_init = np.array([16, 4, 0, 4])
        
        gp = GPSurrogate(random_state=42)
        gp.fit(X_init, y_init)
        
        y_best = np.min(y_init)
        
        ei = ExpectedImprovement(Xi=0.01)
        
        x_next, ei_val = ei.optimise(
            gp, space, y_best=y_best,
            n_restarts=5, random_state=42
        )
        
        assert len(x_next) == 1
        assert -5 <= x_next[0] <= 5
        assert ei_val >= 0
        
        y_next = x_next[0] ** 2
        
        X_new = np.vstack([X_init, x_next.reshape(1, -1)])
        y_new = np.hstack([y_init, [y_next]])
        
        gp.fit(X_new, y_new)
        
        y_new_best = np.min(y_new)
        assert y_new_best <= 0
        
        mean, std = gp.predict(np.array([[1.5]]))
        assert np.isfinite(mean[0])
        assert np.isfinite(std[0])
    
    def test_full_workflow_2d(self):
        def objective(x, y):
            return (x - 1)**2 + (y + 2)**2
        
        bounds = {'x': (-3, 3), 'y': (-4, 2)}
        space = ParameterSpace(bounds)
        
        X_init = []
        y_init = []
        for _ in range(5):
            params = space.sample_random()
            x = params['x']
            y = params['y']
            X_init.append([x, y])
            y_init.append(objective(x, y))
        
        X_init = np.array(X_init)
        y_init = np.array(y_init)
        
        gp = GPSurrogate(random_state=42)
        gp.fit(X_init, y_init)
        
        y_best = np.min(y_init)
        
        ei = ExpectedImprovement(Xi=0.01)
        
        x_next, ei_val = ei.optimise(
            gp, space, y_best=y_best,
            n_restarts=5, random_state=42
        )
        
        assert len(x_next) == 2
        assert -3 <= x_next[0] <= 3
        assert -4 <= x_next[1] <= 2
        assert ei_val >= 0
        
        y_next = objective(x_next[0], x_next[1])
        
        X_new = np.vstack([X_init, x_next.reshape(1, -1)])
        y_new = np.hstack([y_init, [y_next]])
        
        gp.fit(X_new, y_new)
        
        y_new_best = np.min(y_new)
        assert y_new_best <= y_best + 1e-6
    
    def test_multiple_iterations(self):
        def objective(x):
            return x**2 + 1
        
        bounds = {'x': (-5, 5)}
        space = ParameterSpace(bounds)
        
        gp = GPSurrogate(random_state=42)
        ei = ExpectedImprovement(Xi=0.01)
        
        X_hist = []
        y_hist = []
        
        for _ in range(5):
            params = space.sample_random()
            x = params['x']
            y = objective(x)
            X_hist.append(x)
            y_hist.append(y)
        
        X_hist = np.array(X_hist).reshape(-1, 1)
        y_hist = np.array(y_hist)
        
        for _ in range(20):
            gp.fit(X_hist, y_hist)
            y_best = np.min(y_hist)
            x_next, _ = ei.optimise(
                gp, space, y_best=y_best,
                n_restarts=5, random_state=None
            )
            y_next = objective(x_next[0])
            X_hist = np.vstack([X_hist, x_next.reshape(1, -1)])
            y_hist = np.hstack([y_hist, [y_next]])
        
        best_idx = np.argmin(y_hist)
        best_x = X_hist[best_idx][0]
        best_y = y_hist[best_idx]
        
        assert abs(best_x) < 0.5
        assert 1.0 <= best_y <= 2.0
    
    def test_acquisition_optimisation_speed(self):
        space = ParameterSpace({'x': (0, 1), 'y': (0, 1), 'z': (0, 1)})
        
        X_train = np.random.rand(10, 3)
        y_train = np.random.rand(10)
        
        gp = GPSurrogate(random_state=42)
        gp.fit(X_train, y_train)
        
        ei = ExpectedImprovement()
        
        import time
        start = time.time()
        
        x_opt, ei_val = ei.optimise(
            gp, space, y_best=np.min(y_train),
            n_restarts=5, n_initial_points=50,
            random_state=42
        )
        
        elapsed = time.time() - start
        
        assert elapsed < 2.0
        assert x_opt.shape == (3,)
        assert ei_val >= 0
    
    def test_categorical_handling(self):
        bounds = {
            'x': (0, 1, 'int'),
            'cat': ('a', 'b', 'c', 'd')
        }
        space = ParameterSpace(bounds)
        
        X_train = np.array([[0, 0.0], [1, 0.33], [2, 0.66]])
        y_train = np.array([0.5, 0.3, 0.8])
        
        gp = GPSurrogate(random_state=42)
        gp.fit(X_train, y_train)
        
        ei = ExpectedImprovement()
        
        x_opt, ei_val = ei.optimise(
            gp, space, y_best=0.3,
            n_restarts=3, random_state=42
        )
        
        assert 0 <= x_opt[0] <= 1
        assert 0 <= x_opt[1] <= 1
        assert ei_val >= 0