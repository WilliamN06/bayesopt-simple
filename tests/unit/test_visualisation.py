"""Unit tests for visualisation components."""

import pytest
import numpy as np
from bayesopt.visualisation.base import VisualisationBase
from bayesopt.visualisation.convergence import ConvergencePlotter
from bayesopt.visualisation.importance import ImportancePlotter
from bayesopt.core.space import ParameterSpace
from bayesopt.core.surrogate import GPSurrogate


class TestVisualisationBase:
    """Test VisualisationBase class."""
    
    def test_initialisation(self):
        base = VisualisationBase(figsize=(8, 5), dpi=120)
        assert base.figsize == (8, 5)
        assert base.dpi == 120
    
    def test_check_matplotlib(self):
        base = VisualisationBase()
        # Should not raise
        base._check_matplotlib()
    
    def test_get_colours(self):
        base = VisualisationBase()
        colours = base._get_colours()
        assert 'best' in colours
        assert 'exploration' in colours
        assert 'exploitation' in colours


class TestConvergencePlotter:
    """Test ConvergencePlotter class."""
    
    def test_initialisation(self):
        plotter = ConvergencePlotter()
        assert plotter.figsize == (10, 6)
    
    def test_plot_empty_history(self):
        plotter = ConvergencePlotter()
        with pytest.raises(ValueError):
            plotter.plot([])
    
    def test_plot_single_point(self):
        plotter = ConvergencePlotter()
        history = [{'value': 1.0}]
        # Should not raise
        plotter.plot(history, save_path=None)
    
    def test_plot_multiple_points(self):
        plotter = ConvergencePlotter()
        history = [
            {'value': 1.0},
            {'value': 0.8},
            {'value': 0.7},
            {'value': 0.6}
        ]
        plotter.plot(history, save_path=None)
    
    def test_plot_improvement(self):
        plotter = ConvergencePlotter()
        history = [
            {'value': 1.0},
            {'value': 0.8},
            {'value': 0.7},
            {'value': 0.6}
        ]
        plotter.plot_improvement(history, save_path=None)


class TestImportancePlotter:
    """Test ImportancePlotter class."""
    
    def test_plot_importance(self):
        plotter = ImportancePlotter()
        gp_stats = {
            'lengthscales': np.array([0.5, 1.2, 0.8]),
            'is_fitted': True,
            'n_samples': 10
        }
        param_names = ['a', 'b', 'c']
        plotter.plot_importance(gp_stats, param_names, save_path=None)
    
    def test_plot_importance_no_lengthscales(self):
        plotter = ImportancePlotter()
        gp_stats = {'is_fitted': True}
        param_names = ['a', 'b']
        with pytest.raises(ValueError):
            plotter.plot_importance(gp_stats, param_names)
    
    def test_plot_importance_mismatch(self):
        plotter = ImportancePlotter()
        gp_stats = {'lengthscales': np.array([0.5, 1.2])}
        param_names = ['a', 'b', 'c']
        with pytest.raises(ValueError):
            plotter.plot_importance(gp_stats, param_names)