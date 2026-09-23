"""Acquisition function visualisation."""

from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

from bayesopt.core.space import ParameterSpace
from bayesopt.core.surrogate import GPSurrogate
from bayesopt.core.acquisition import ExpectedImprovement
from bayesopt.visualisation.base import VisualisationBase


class AcquisitionPlotter(VisualisationBase):
    """Plot acquisition function surface."""
    
    def plot_surface(
        self,
        gp: GPSurrogate,
        space: ParameterSpace,
        y_best: float,
        n_points: int = 50,
        figsize: Optional[Tuple[int, int]] = None,
        title: str = "Acquisition Function Surface",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot 2D acquisition function surface.
        
        Parameters
        ----------
        gp : GPSurrogate
            Fitted Gaussian Process.
        space : ParameterSpace
            Parameter space definition.
        y_best : float
            Best observed value so far.
        n_points : int
            Grid resolution.
        figsize : tuple, optional
            Figure size (width, height) in inches.
        title : str
            Plot title.
        save_path : str, optional
            Path to save figure.
        """
        self._check_matplotlib()
        
        if space.get_dimensions() != 2:
            raise ValueError(
                f"Acquisition surface plot requires 2 dimensions, "
                f"got {space.get_dimensions()}"
            )
        
        if not gp.is_fitted():
            raise ValueError("GP not fitted yet")
        
        if figsize:
            self.figsize = figsize
        
        param_names = space.get_names()
        bounds = space.bounds_array()
        
        x_vals = np.linspace(bounds[0, 0], bounds[0, 1], n_points)
        y_vals = np.linspace(bounds[1, 0], bounds[1, 1], n_points)
        X, Y = np.meshgrid(x_vals, y_vals)
        
        Z = np.zeros_like(X)
        ei = ExpectedImprovement()
        
        for i in range(n_points):
            for j in range(n_points):
                point = np.array([[X[i, j], Y[i, j]]])
                Z[i, j] = ei.evaluate(point, gp, y_best)[0]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figsize[0] * 1.8, self.figsize[1]))
        colours = self._get_colours()
        
        contour = ax1.contourf(X, Y, Z, levels=20, cmap='viridis')
        ax1.set_xlabel(param_names[0])
        ax1.set_ylabel(param_names[1])
        ax1.set_title(f"{title}\nHigher = More Promising")
        
        X_train, y_train = gp.get_training_data()
        if X_train is not None:
            scatter = ax1.scatter(
                X_train[:, 0], X_train[:, 1],
                c=y_train, cmap='coolwarm',
                s=50, edgecolor='black', linewidth=0.5
            )
        
        best_idx = np.argmin(y_train) if y_train is not None else 0
        if X_train is not None and len(X_train) > 0:
            ax1.scatter(
                X_train[best_idx, 0], X_train[best_idx, 1],
                s=100, marker='*', color='red',
                label='Current best'
            )
        
        cbar = fig.colorbar(contour, ax=ax1)
        cbar.set_label('Expected Improvement')
        
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(Z.max(axis=0))
        ax2.set_xlabel(param_names[0])
        ax2.set_ylabel('Max Expected Improvement')
        ax2.set_title('Acquisition Profile')
        ax2.grid(True, alpha=0.3)
        
        self.save_figure(fig, save_path)
    
    def plot_1d_acquisition(
        self,
        gp: GPSurrogate,
        space: ParameterSpace,
        y_best: float,
        n_points: int = 100,
        figsize: Optional[Tuple[int, int]] = None,
        title: str = "Acquisition Function",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot 1D acquisition function.
        
        Parameters
        ----------
        gp : GPSurrogate
            Fitted Gaussian Process.
        space : ParameterSpace
            Parameter space definition.
        y_best : float
            Best observed value so far.
        n_points : int
            Number of points to evaluate.
        figsize : tuple, optional
            Figure size (width, height) in inches.
        title : str
            Plot title.
        save_path : str, optional
            Path to save figure.
        """
        self._check_matplotlib()
        
        if space.get_dimensions() != 1:
            raise ValueError(
                f"1D acquisition plot requires 1 dimension, "
                f"got {space.get_dimensions()}"
            )
        
        if not gp.is_fitted():
            raise ValueError("GP not fitted yet")
        
        if figsize:
            self.figsize = figsize
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figsize[0] * 1.8, self.figsize[1]))
        colours = self._get_colours()
        
        bounds = space.bounds_array()
        X = np.linspace(bounds[0, 0], bounds[0, 1], n_points).reshape(-1, 1)
        
        mean, std = gp.predict(X)
        ei = ExpectedImprovement()
        ei_vals = ei.evaluate(X, gp, y_best)
        
        ax1.plot(X, mean, color=colours['exploitation'], linewidth=2, label='Mean')
        ax1.fill_between(
            X.flatten(),
            mean - std,
            mean + std,
            color=colours['exploration'],
            alpha=0.2,
            label='Uncertainty'
        )
        
        X_train, y_train = gp.get_training_data()
        if X_train is not None:
            ax1.scatter(X_train.flatten(), y_train, s=50, 
                        color=colours['grid'], edgecolor='black', label='Observations')
        
        ax1.axhline(y=y_best, color=colours['best'], linestyle='--', alpha=0.5, label='Best so far')
        ax1.set_xlabel(space.get_names()[0])
        ax1.set_ylabel('Predicted Value')
        ax1.set_title('Gaussian Process Prediction')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(X, ei_vals, color=colours['acquisition'], linewidth=2)
        ax2.fill_between(X.flatten(), 0, ei_vals, color=colours['acquisition'], alpha=0.2)
        
        if X_train is not None:
            ax2.scatter(X_train.flatten(), [0] * len(X_train), s=30,
                        color='black', alpha=0.3)
        
        max_idx = np.argmax(ei_vals)
        if ei_vals[max_idx] > 0:
            ax2.scatter(X[max_idx], ei_vals[max_idx], s=100, 
                        color='red', marker='*', label='Suggested point')
        
        ax2.set_xlabel(space.get_names()[0])
        ax2.set_ylabel('Expected Improvement')
        ax2.set_title(title)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        self.save_figure(fig, save_path)