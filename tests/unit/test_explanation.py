"""Unit tests for explanation engine."""

import pytest
import numpy as np
from bayesopt.explanation.engine import ExplanationEngine, ExplanationFormatter
from bayesopt.explanation.translators import GPStateTranslator, DecisionTranslator
from bayesopt.utils.logging import ConsoleLogger


class TestExplanationEngine:
    """Test ExplanationEngine class."""
    
    def test_initialisation(self):
        engine = ExplanationEngine(verbose=1)
        assert engine.level == 1
    
    def test_explain_iteration_random(self):
        engine = ExplanationEngine(verbose=2)
        explanation = engine.explain_iteration(
            1, {'x': 0.5}, 0.25, 1.0,
            {'classification': 'random'}
        )
        assert "Iteration" in explanation
        assert "Randomly" in explanation or "exploring" in explanation.lower()
    
    def test_explain_iteration_exploration(self):
        engine = ExplanationEngine(verbose=2)
        explanation = engine.explain_iteration(
            1, {'x': 0.5}, 0.25, 1.0,
            {'classification': 'exploration', 'std': 0.5}
        )
        assert "uncertain" in explanation.lower()
    
    def test_explain_iteration_exploitation(self):
        engine = ExplanationEngine(verbose=2)
        explanation = engine.explain_iteration(
            1, {'x': 0.5}, 0.25, 1.0,
            {'classification': 'exploitation', 'mean': 0.8}
        )
        assert "promising" in explanation.lower() or "refining" in explanation.lower()
    
    def test_explain_iteration_improvement(self):
        engine = ExplanationEngine(verbose=2)
        explanation = engine.explain_iteration(
            1, {'x': 0.5}, 0.25, 0.5,
            {'classification': 'exploration'}
        )
        assert "improvement" in explanation.lower() or "better" in explanation.lower()
    
    def test_explain_gp_state_not_fitted(self):
        engine = ExplanationEngine(verbose=1)
        explanation = engine.explain_gp_state({'is_fitted': False})
        assert explanation == ""
    
    def test_explain_gp_state_fitted(self):
        engine = ExplanationEngine(verbose=1)
        explanation = engine.explain_gp_state({
            'is_fitted': True,
            'lengthscales': np.array([0.5, 1.2]),
            'noise': 0.01,
            'n_samples': 10,
            'log_likelihood': -5.0
        })
        assert "Gaussian Process" in explanation
        assert "samples" in explanation
    
    def test_explain_final_result(self):
        engine = ExplanationEngine(verbose=1)
        explanation = engine.explain_final_result(
            {'x': 0.5, 'y': 2.0}, 0.25, 10, 5.0,
            {'is_fitted': True, 'lengthscales': np.array([0.5, 0.8])}
        )
        assert "Optimisation complete" in explanation
        assert "Best configuration" in explanation
    
    def test_explain_progress(self):
        engine = ExplanationEngine(verbose=1)
        history = [{'value': 1.0}, {'value': 0.8}, {'value': 0.7}]
        explanation = engine.explain_progress(5, 20, 0.7, history)
        assert "Progress:" in explanation
        assert "evaluations" in explanation
    
    def test_format_section(self):
        formatter = ExplanationFormatter()
        section = formatter.format_section("Test", "Content")
        assert "Test" in section
        assert "Content" in section
    
    def test_format_best_result(self):
        formatter = ExplanationFormatter()
        result = formatter.format_best_result({'x': 0.5}, 0.25, 0.1)
        assert "NEW BEST" in result
        assert "x: 0.500000" in result


class TestGPStateTranslator:
    """Test GPStateTranslator class."""
    
    def test_translate_lengthscales(self):
        translator = GPStateTranslator()
        result = translator.translate_lengthscales(
            np.array([0.5, 1.2, 0.8]),
            ['a', 'b', 'c']
        )
        assert 'most_important' in result
        assert 'least_important' in result
        assert 'a' in result['most_important']
    
    def test_translate_noise(self):
        translator = GPStateTranslator()
        result = translator.translate_noise(0.001)
        assert "Low" in result['summary']
        
        result = translator.translate_noise(0.05)
        assert "Moderate" in result['summary']
        
        result = translator.translate_noise(0.5)
        assert "High" in result['summary']
    
    def test_translate_convergence(self):
        translator = GPStateTranslator()
        result = translator.translate_convergence(10, [0.1, 0.05, 0.02])
        assert "Converged" in result['summary'] or "approaching" in result['summary']


class TestDecisionTranslator:
    """Test DecisionTranslator class."""
    
    def test_translate_exploration(self):
        translator = DecisionTranslator()
        result = translator.translate_exploration(0.5, 0.8, 1.0)
        assert "Exploring" in result['summary']
    
    def test_translate_exploitation(self):
        translator = DecisionTranslator()
        result = translator.translate_exploitation(0.9, 0.1, 1.0)
        assert "Exploiting" in result['summary']


class TestConsoleLogger:
    """Test ConsoleLogger class."""
    
    def test_initialisation(self):
        logger = ConsoleLogger(verbose=2)
        assert logger.verbose == 2
    
    def test_info(self):
        logger = ConsoleLogger(verbose=1)
        # Should not raise
        logger.info("Test message")
    
    def test_success(self):
        logger = ConsoleLogger(verbose=1)
        logger.success("Success message")
    
    def test_warning(self):
        logger = ConsoleLogger(verbose=1)
        logger.warning("Warning message")
    
    def test_error(self):
        logger = ConsoleLogger(verbose=1)
        logger.error("Error message")
    
    def test_section(self):
        logger = ConsoleLogger(verbose=1)
        logger.section("Test Section")
    
    def test_progress(self):
        logger = ConsoleLogger(verbose=1)
        logger.progress(5, 10, "Testing")
    
    def test_log_explanation(self):
        logger = ConsoleLogger(verbose=1)
        logger.log_explanation("Line 1\n  * Detail 1\n  Detail 2")
    
    def test_log_result(self):
        logger = ConsoleLogger(verbose=1)
        logger.log_result({'x': 0.5, 'y': 2}, 0.25)