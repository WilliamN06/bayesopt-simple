"""Explanation engine for plain-language descriptions."""

from typing import Dict, Any, Optional
import numpy as np


class ExplanationEngine:
    """Generates plain-language explanations of optimisation decisions."""
    
    def __init__(self, verbose: int = 1):
        self.level = verbose
        self._templates = self._init_templates()
    
    def _init_templates(self) -> Dict:
        """Initialise explanation templates."""
        return {
            'exploration': [
                "Exploring an uncertain region",
                "Checking an unknown area",
                "Investigating a poorly understood part of the space"
            ],
            'exploitation': [
                "Refining a promising area",
                "Concentrating on a known good region",
                "Fine-tuning around the current best"
            ],
            'balanced': [
                "Balancing exploration and exploitation",
                "Seeking a compromise between known and unknown",
                "Managing the exploration-exploitation trade-off"
            ],
            'random': [
                "Randomly sampling initial points",
                "Exploring without prior information",
                "Building initial understanding"
            ],
            'improvement': [
                "New best value found",
                "Improved result discovered",
                "Better configuration identified"
            ],
            'convergence': [
                "Optimisation appears to be converging",
                "Improvements are becoming smaller",
                "Approaching a likely optimum"
            ]
        }
    
    def explain_iteration(
        self,
        iteration: int,
        params: Dict[str, Any],
        value: float,
        best_value: float,
        explanation_data: Dict[str, Any]
    ) -> str:
        """Generate explanation for a single iteration."""
        if self.level < 1:
            return ""
        
        lines = []
        
        if self.level >= 2:
            lines.append(f"Iteration {iteration}:")
        
        classification = explanation_data.get('classification', 'random')
        template = np.random.choice(self._templates.get(classification, self._templates['random']))
        
        if classification == 'exploration':
            std = explanation_data.get('std', 0.0)
            lines.append(f"  {template} (uncertainty = {std:.4f})")
        elif classification == 'exploitation':
            mean = explanation_data.get('mean', 0.0)
            lines.append(f"  {template} (predicted value = {mean:.4f})")
        elif classification == 'balanced':
            ei = explanation_data.get('ei', 0.0)
            lines.append(f"  {template} (expected improvement = {ei:.4f})")
        else:
            lines.append(f"  {template}")
        
        if value < best_value:
            improvement = best_value - value
            template = np.random.choice(self._templates['improvement'])
            lines.append(f"  * {template}! improvement of {improvement:.6f}")
        else:
            diff = value - best_value
            if diff < 0.1 * abs(best_value) + 0.1:
                lines.append(f"  * Close to current best (difference = {diff:.6f})")
            else:
                lines.append(f"  * Still exploring (difference = {diff:.6f})")
        
        if self.level >= 2:
            lines.append(f"  * Best so far: {best_value:.6f}")
        
        return "\n".join(lines)
    
    def explain_gp_state(self, gp_stats: Dict[str, Any]) -> str:
        """Explain what the Gaussian Process has learned."""
        if self.level < 1 or not gp_stats.get('is_fitted', False):
            return ""
        
        lines = []
        lines.append("Gaussian Process state:")
        
        lengthscales = gp_stats.get('lengthscales')
        if lengthscales is not None:
            if len(lengthscales) == 1:
                lines.append(f"  * Lengthscale: {lengthscales[0]:.4f}")
            else:
                lines.append(f"  * Lengthscales: {', '.join([f'{x:.4f}' for x in lengthscales])}")
        
        noise = gp_stats.get('noise')
        if noise is not None:
            if noise < 0.01:
                lines.append("  * Objective appears smooth (low noise)")
            elif noise < 0.1:
                lines.append("  * Moderate noise detected")
            else:
                lines.append("  * Objective appears noisy")
        
        n_samples = gp_stats.get('n_samples', 0)
        lines.append(f"  * Learned from {n_samples} evaluations")
        
        log_likelihood = gp_stats.get('log_likelihood')
        if log_likelihood is not None:
            lines.append(f"  * Model fit quality: {log_likelihood:.2f} (higher is better)")
        
        return "\n".join(lines)
    
    def explain_final_result(
        self,
        best_params: Dict[str, Any],
        best_value: float,
        n_calls: int,
        time_taken: float,
        gp_stats: Dict[str, Any]
    ) -> str:
        """Generate final summary explanation."""
        if self.level < 1:
            return ""
        
        lines = []
        lines.append("Optimisation complete!")
        lines.append("")
        lines.append("Best configuration found:")
        
        for key, value in best_params.items():
            if isinstance(value, float):
                lines.append(f"  * {key}: {value:.6f}")
            else:
                lines.append(f"  * {key}: {value}")
        
        lines.append("")
        lines.append(f"Best objective value: {best_value:.6f}")
        lines.append(f"Total evaluations: {n_calls}")
        lines.append(f"Time taken: {time_taken:.2f} seconds")
        
        if gp_stats.get('is_fitted', False):
            lengthscales = gp_stats.get('lengthscales')
            if lengthscales is not None and len(lengthscales) > 1:
                param_names = list(best_params.keys())
                if len(param_names) == len(lengthscales):
                    important = sorted(
                        zip(param_names, lengthscales),
                        key=lambda x: x[1]
                    )
                    lines.append("")
                    lines.append("Parameter importance (lower lengthscale = more important):")
                    for name, length in important:
                        lines.append(f"  * {name}: {length:.4f}")
        
        return "\n".join(lines)
    
    def explain_progress(
        self,
        iteration: int,
        total_calls: int,
        best_value: float,
        history: list
    ) -> str:
        """Summarise optimisation progress."""
        if self.level < 1:
            return ""
        
        progress = iteration / total_calls
        lines = []
        
        if progress < 0.25:
            lines.append("Early stage: building understanding")
        elif progress < 0.5:
            lines.append("Mid stage: refining search")
        elif progress < 0.75:
            lines.append("Late stage: converging on solution")
        else:
            lines.append("Final stage: fine-tuning")
        
        lines.append(f"Progress: {iteration}/{total_calls} evaluations")
        lines.append(f"Current best: {best_value:.6f}")
        
        if len(history) > 10:
            recent = history[-10:]
            improvements = sum(1 for i in range(1, len(recent)) 
                             if recent[i]['value'] < recent[i-1]['value'])
            if improvements == 0:
                lines.append("No recent improvements - may have converged")
            elif improvements < 3:
                lines.append("Sparse recent improvements - approaching convergence")
            else:
                lines.append("Steady progress being made")
        
        return "\n".join(lines)


class ExplanationFormatter:
    """Formats explanations for console output."""
    
    @staticmethod
    def format_section(title: str, content: str, width: int = 60) -> str:
        """Format a section with a title and content."""
        lines = []
        lines.append("=" * width)
        lines.append(f" {title} ".center(width))
        lines.append("=" * width)
        lines.append(content)
        lines.append("=" * width)
        return "\n".join(lines)
    
    @staticmethod
    def format_best_result(params: Dict, value: float, improvement: Optional[float] = None) -> str:
        """Format a new best result message."""
        lines = []
        lines.append("=" * 60)
        lines.append(" NEW BEST RESULT ".center(60))
        lines.append("=" * 60)
        
        for key, val in params.items():
            if isinstance(val, float):
                lines.append(f"{key}: {val:.6f}")
            else:
                lines.append(f"{key}: {val}")
        
        lines.append(f"Value: {value:.6f}")
        
        if improvement is not None:
            lines.append(f"Improvement: {improvement:.6f}")
        
        lines.append("=" * 60)
        return "\n".join(lines)