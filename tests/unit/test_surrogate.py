"""Unit tests for GPSurrogate."""

import pytest
import numpy as np
from bayesopt.core.surrogate import GPSurrogate


class TestGPSurrogate:
    """Test GPSurrogate class."""
    
    def test_initialisation(self):
        gp = GPSurrogate()
        assert not gp.is_fitted()
        assert gp.kernel_name == 'matern52'
        
        gp2 = GPSurrogate(kernel='rbf', n_restarts=5)
        assert gp2.kernel_name == 'rbf'
        assert gp2.n_restarts == 5
    
    def test_fit_linear_function(self):
        gp = GPSurrogate(random_state=42)
        
        X = np.linspace(0, 5, 10).reshape(-1, 1)
        y = 2 * X.flatten() + 1
        
        gp.fit(X, y)
        assert gp.is_fitted()
        
        X_test = np.array([[2.5], [3.5]])
        mean, std = gp.predict(X_test)
        
        assert mean[0] == pytest.approx(6.0, abs=0.5)
        assert mean[1] == pytest.approx(8.0, abs=0.5)
    
    def test_fit_quadratic_function(self):
        gp = GPSurrogate(random_state=42)
        
        X = np.linspace(-3, 3, 15).reshape(-1, 1)
        y = X.flatten() ** 2
        
        gp.fit(X, y)
        
        X_test = np.array([[0], [2], [-2]])
        mean, std = gp.predict(X_test)
        
        assert mean[0] == pytest.approx(0.0, abs=0.5)
        assert mean[1] == pytest.approx(4.0, abs=0.5)
        assert mean[2] == pytest.approx(4.0, abs=0.5)
    
    def test_uncertainty_decreases_with_more_data(self):
        gp = GPSurrogate(random_state=42)
        
        X_small = np.array([[0], [1], [2]])
        y_small = np.array([0, 1, 4])
        
        gp.fit(X_small, y_small)
        _, std_small = gp.predict(np.array([[1.5]]))
        
        X_large = np.linspace(0, 2, 10).reshape(-1, 1)
        y_large = X_large.flatten() ** 2
        gp.fit(X_large, y_large)
        _, std_large = gp.predict(np.array([[1.5]]))
        
        assert std_large[0] < std_small[0]
    
    def test_predict_prior_when_no_data(self):
        gp = GPSurrogate()
        
        with pytest.raises(RuntimeError):
            gp.predict(np.array([[0]]))
    
    def test_predict_with_covariance(self):
        gp = GPSurrogate(random_state=42)
        
        X = np.array([[0], [1], [2]])
        y = np.array([0, 1, 4])
        gp.fit(X, y)
        
        X_test = np.array([[1.5]])
        mean, std, cov = gp.predict(X_test, return_std=True, return_cov=True)
        
        assert mean.shape == (1,)
        assert std.shape == (1,)
        assert cov.shape == (1, 1)
    
    def test_get_stats(self):
        gp = GPSurrogate(random_state=42)
        
        stats = gp.get_stats()
        assert not stats['is_fitted']
        assert stats['n_samples'] == 0
        
        X = np.array([[0], [1], [2]])
        y = np.array([0, 1, 4])
        gp.fit(X, y)
        
        stats = gp.get_stats()
        assert stats['is_fitted']
        assert stats['n_samples'] == 3
        assert 'lengthscales' in stats
        assert 'noise' in stats
        assert 'log_likelihood' in stats
        assert 'kernel_parameters' in stats
    
    def test_reset(self):
        gp = GPSurrogate(random_state=42)
        
        X = np.array([[0], [1], [2]])
        y = np.array([0, 1, 4])
        gp.fit(X, y)
        assert gp.is_fitted()
        
        gp.reset()
        assert not gp.is_fitted()
        assert gp._X_train is None
        assert gp._y_train is None
    
    def test_get_training_data(self):
        gp = GPSurrogate(random_state=42)
        
        X = np.array([[0], [1], [2]])
        y = np.array([0, 1, 4])
        gp.fit(X, y)
        
        X_train, y_train = gp.get_training_data()
        assert np.array_equal(X_train, X)
        assert np.array_equal(y_train, y)
    
    def test_duplicate_points(self):
        gp = GPSurrogate(random_state=42)
        
        X = np.array([[0], [0], [1], [2]])
        y = np.array([0, 0.1, 1, 4])
        
        gp.fit(X, y)
        assert gp.is_fitted()
        
        mean, _ = gp.predict(np.array([[0.5]]))
        assert 0.0 <= mean[0] <= 2.0