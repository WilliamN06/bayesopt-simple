"""Convergence visualisation."""

from typing import List, Dict, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

from bayesopt.visualisation.base import VisualisationBase


class ConvergencePlotter(VisualisationBase):
    """Plot convergence of optimisation."""
    
    def plot(
        self,
        history: List[Dict],
        figsize: Optional[Tuple[int, int]] = None,
        title: str = "Optimisation Convergence",
        show_random_baseline: bool = True,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot convergence of best value over iterations.
        
        Parameters
        ----------
        history : List[Dict]
            Optimisation history with 'iteration' and 'value' keys.
        figsize : tuple, optional
            Figure size (width, height) in inches.
        title : str
            Plot title.
        show_random_baseline : bool
            Show random search baseline for comparison.
        save_path : str, optional
            Path to save figure.
        """
        self._check_matplotlib()
        
        if not history:
            raise ValueError("History is empty")
        
        if figsize:
            self.figsize = figsize
        
        fig, ax = plt.subplots(figsize=self.figsize)
        colours = self._get_colours()
        
        iterations = [h.get('iteration', i+1) for i, h in enumerate(history)]
        values = [h['value'] for h in history]
        
        best_values = []
        current_best = float('inf')
        for v in values:
            if v < current_best:
                current_best = v
            best_values.append(current_best)
        
        ax.scatter(iterations, values, alpha=0.5, s=30, 
                   label='Evaluated points', color=colours['grid'])
        ax.plot(iterations, best_values, linewidth=2.5, 
                label='Best so far', color=colours['best'])
        
        if show_random_baseline and len(values) > 3:
            random_best = []
            rb = float('inf')
            for v in np.random.choice(values, len(values), replace=False):
                if v < rb:
                    rb = v
                random_best.append(rb)
            ax.plot(iterations, random_best, '--', linewidth=1.5,
                    label='Random search baseline', color=colours['exploration'], alpha=0.6)
        
        improvement = best_values[0] - best_values[-1]
        if improvement > 0:
            best_idx = np.argmin(best_values)
            ax.annotate(
                f'Best: {best_values[-1]:.4f}',
                xy=(iterations[best_idx], best_values[best_idx]),
                xytext=(iterations[best_idx], best_values[best_idx] * 1.1),
                arrowprops=dict(arrowstyle='->', color=colours['best'], lw=1.5),
                fontsize=10, color=colours['text']
            )
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective Value')
        ax.set_title(title)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        self.save_figure(fig, save_path)
    
    def plot_improvement(
        self,
        history: List[Dict],
        figsize: Optional[Tuple[int, int]] = None,
        title: str = "Improvement Over Time",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot improvement at each iteration.
        
        Parameters
        ----------
        history : List[Dict]
            Optimisation history with 'iteration' and 'value' keys.
        figsize : tuple, optional
            Figure size (width, height) in inches.
        title : str
            Plot title.
        save_path : str, optional
            Path to save figure.
        """
        self._check_matplotlib()
        
        if not history:
            raise ValueError("History is empty")
        
        if figsize:
            self.figsize = figsize
        
        fig, ax = plt.subplots(figsize=self.figsize)
        colours = self._get_colours()
        
        iterations = [h.get('iteration', i+1) for i, h in enumerate(history)]
        values = [h['value'] for h in history]
        
        improvements = [0.0]
        for i in range(1, len(values)):
            if values[i] < values[i-1]:
                improvements.append(values[i-1] - values[i])
            else:
                improvements.append(0.0)
        
        ax.bar(iterations, improvements, color=colours['best'], alpha=0.7)
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Improvement')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        self.save_figure(fig, save_path)