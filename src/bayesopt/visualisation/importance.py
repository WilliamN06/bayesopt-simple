"""Parameter importance visualisation."""

from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt

from bayesopt.visualisation.base import VisualisationBase


class ImportancePlotter(VisualisationBase):
    """Plot parameter importance."""
    
    def plot_importance(
        self,
        gp_stats: Dict,
        param_names: List[str],
        figsize: Optional[Tuple[int, int]] = None,
        title: str = "Parameter Importance",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot parameter importance based on inverse lengthscales.
        
        Parameters
        ----------
        gp_stats : Dict
            GP statistics from GPSurrogate.get_stats()
        param_names : List[str]
            Parameter names in order.
        figsize : tuple, optional
            Figure size (width, height) in inches.
        title : str
            Plot title.
        save_path : str, optional
            Path to save figure.
        """
        self._check_matplotlib()
        
        lengthscales = gp_stats.get('lengthscales')
        if lengthscales is None or len(lengthscales) == 0:
            raise ValueError("No lengthscales available in GP stats")
        
        if len(lengthscales) != len(param_names):
            raise ValueError(
                f"Lengthscales length {len(lengthscales)} does not match "
                f"param_names length {len(param_names)}"
            )
        
        if figsize:
            self.figsize = figsize
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figsize[0] * 1.8, self.figsize[1]))
        colours = self._get_colours()
        
        importance = 1.0 / (lengthscales + 1e-6)
        importance = importance / np.max(importance)
        
        sorted_idx = np.argsort(importance)[::-1]
        sorted_names = [param_names[i] for i in sorted_idx]
        sorted_importance = importance[sorted_idx]
        sorted_lengthscales = lengthscales[sorted_idx]
        
        bars = ax1.barh(sorted_names, sorted_importance, color=colours['best'])
        ax1.set_xlabel('Relative Importance')
        ax1.set_title(title)
        ax1.grid(True, alpha=0.3, axis='x')
        
        for i, (bar, imp, length) in enumerate(zip(bars, sorted_importance, sorted_lengthscales)):
            width = bar.get_width()
            ax1.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                    f'{imp:.3f}', va='center', fontsize=9)
        
        ax2.barh(sorted_names, sorted_lengthscales, color=colours['exploration'])
        ax2.set_xlabel('Lengthscale')
        ax2.set_title('Lengthscale Values (Lower = More Important)')
        ax2.grid(True, alpha=0.3, axis='x')
        
        if gp_stats.get('is_fitted', False):
            fig.suptitle(
                f"Based on {gp_stats.get('n_samples', 0)} evaluations",
                fontsize=10, color=colours['text']
            )
        
        self.save_figure(fig, save_path)