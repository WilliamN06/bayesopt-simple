"""Gaussian Process surrogate model wrapper."""

from typing import Dict, Optional, Tuple, Union
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel, ConstantKernel
from sklearn.gaussian_process.kernels import Kernel
import warnings


class GPSurrogate:
    """
    Wrapper for scikit-learn's Gaussian Process Regressor.
    
    Uses Matern 5/2 kernel with automatic hyperparameter optimisation.

    """
    
    def __init__(
        self,
        kernel: Union[str, Kernel] = 'matern52',
        n_restarts: int = 10,
        noise_alpha: float = 1e-6,
        normalize_y: bool = True,
        random_state: int = 42
    ):
        self.kernel_name = kernel
        self.n_restarts = n_restarts
        self.noise_alpha = noise_alpha
        self.normalize_y = normalize_y
        self.random_state = random_state
        self._kernel = self._create_kernel(kernel)
        
        self.gp = GaussianProcessRegressor(
            kernel=self._kernel,
            n_restarts_optimizer=n_restarts,
            alpha=noise_alpha,
            normalize_y=normalize_y,
            random_state=random_state,
            copy_X_train=True
        )
        
        self._is_fitted = False
        self._X_train = None
        self._y_train = None
        
    def _create_kernel(self, kernel_spec: Union[str, Kernel]) -> Kernel:
        """Create the kernel based on specification."""
        if isinstance(kernel_spec, Kernel):
            return kernel_spec
        
        if kernel_spec == 'matern52':
            # Matern 5/2 kernel (standard choice for BO)
            return Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=2.5)
        elif kernel_spec == 'rbf':
            # RBF kernel (infinite smoothness)
            return RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3))
        else:
            raise ValueError(f"Unknown kernel: {kernel_spec}. Choose 'matern52' or 'rbf'.")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the GP to observations.

        """
        if len(X) == 0:
            raise ValueError("Cannot fit GP with empty X")
        
        if len(X) != len(y):
            raise ValueError(f"X length ({len(X)}) doesn't match y length ({len(y)})")
        
        # Convert to 2D if needed
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        # Check for duplicates - warn but continue
        # sklearn handles duplicates with noise alpha
        
        # Fit the GP
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.gp.fit(X, y)
        
        self._is_fitted = True
        self._X_train = X.copy()
        self._y_train = y.copy()
    
    def predict(
        self, 
        X: np.ndarray,
        return_std: bool = True,
        return_cov: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Make predictions at new points.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Points to predict at.
        return_std : bool, default=True
            Whether to return standard deviations.
        return_cov : bool, default=False
            Whether to return full covariance matrix.
            
    
        """
        if not self._is_fitted:
            raise RuntimeError("GP not fitted yet. Call fit() first.")
        
        # Convert to 2D if needed
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        if return_cov:
            mean, cov = self.gp.predict(X, return_cov=True)
            if return_std:
                std = np.sqrt(np.diag(cov))
                return mean, std, cov
            return mean, cov
        else:
            mean, std = self.gp.predict(X, return_std=return_std)
            return mean, std
    
    def get_stats(self) -> Dict[str, Union[float, np.ndarray]]:
        """
        Extract statistics from the fitted GP.
        
    
        """
        if not self._is_fitted:
            return {
                'lengthscales': None,
                'noise': None,
                'log_likelihood': None,
                'n_samples': 0,
                'kernel_parameters': {},
                'is_fitted': False
            }
        
        stats = {
            'is_fitted': True,
            'n_samples': len(self._X_train) if self._X_train is not None else 0
        }
        
        # Get kernel parameters
        kernel = self.gp.kernel_
        kernel_params = kernel.get_params()
        
        # Try to extract lengthscales
        lengthscales = None
        if hasattr(kernel, 'length_scale'):
            lengthscales = kernel.length_scale
        elif hasattr(kernel, 'k1') and hasattr(kernel.k1, 'length_scale'):
            # Matern kernel inside ConstantKernel * Matern
            lengthscales = kernel.k1.length_scale
        elif hasattr(kernel, 'k2') and hasattr(kernel.k2, 'length_scale'):
            # ConstantKernel * Matern or RBF
            lengthscales = kernel.k2.length_scale
        
        if lengthscales is not None:
            stats['lengthscales'] = np.array(lengthscales).flatten()
        
        # Get noise (alpha)
        stats['noise'] = float(self.gp.alpha) if isinstance(self.gp.alpha, (int, float)) else None
        
        # Get log marginal likelihood
        try:
            stats['log_likelihood'] = float(self.gp.log_marginal_likelihood_value_)
        except (AttributeError, ValueError):
            stats['log_likelihood'] = None
        
        # Full kernel parameters
        stats['kernel_parameters'] = kernel_params
        
        return stats
    
    def reset(self) -> None:
        """Reset the GP state (for a new optimisation)."""
        # Re-initialise the GP
        self.gp = GaussianProcessRegressor(
            kernel=self._create_kernel(self.kernel_name),
            n_restarts_optimizer=self.n_restarts,
            alpha=self.noise_alpha,
            normalize_y=self.normalize_y,
            random_state=self.random_state
        )
        self._is_fitted = False
        self._X_train = None
        self._y_train = None
    
    def is_fitted(self) -> bool:
        """Check if the GP has been fitted."""
        return self._is_fitted
    
    def get_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the training data.
        
        Returns
        -------
        X : np.ndarray
            Training inputs.
        y : np.ndarray
            Training targets.
        """
        if not self._is_fitted:
            return None, None
        return self._X_train, self._y_train