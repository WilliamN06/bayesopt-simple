"""Base visualisation utilities."""

from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


class VisualisationBase:
    """Base class for visualisation utilities."""
    
    def __init__(self, figsize: Tuple[int, int] = (10, 6), dpi: int = 100):
        self.figsize = figsize
        self.dpi = dpi
        self._setup_style()
    
    def _setup_style(self) -> None:
        """Set up consistent plotting style."""
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except OSError:
            try:
                plt.style.use('seaborn-darkgrid')
            except OSError:
                pass
        
        mpl.rcParams['figure.figsize'] = self.figsize
        mpl.rcParams['figure.dpi'] = self.dpi
        mpl.rcParams['font.size'] = 11
        mpl.rcParams['axes.labelsize'] = 12
        mpl.rcParams['axes.titlesize'] = 14
        mpl.rcParams['legend.fontsize'] = 10
        mpl.rcParams['xtick.labelsize'] = 10
        mpl.rcParams['ytick.labelsize'] = 10
    
    def _check_matplotlib(self) -> None:
        """Check if matplotlib is available."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for visualisation. "
                "Install with: pip install matplotlib seaborn"
            )
    
    def _get_colours(self) -> dict:
        """Get consistent colour scheme."""
        return {
            'best': '#2ecc71',
            'exploration': '#3498db',
            'exploitation': '#e74c3c',
            'acquisition': '#9b59b6',
            'uncertainty': '#f1c40f',
            'grid': '#ecf0f1',
            'text': '#2c3e50'
        }
    
    def save_figure(self, fig: plt.Figure, filename: Optional[str] = None) -> None:
        """Save figure to file if filename provided."""
        if filename:
            fig.savefig(filename, dpi=self.dpi, bbox_inches='tight')
            plt.close(fig)
        else:
            plt.show()