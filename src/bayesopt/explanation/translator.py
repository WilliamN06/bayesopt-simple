"""Translators for converting GP statistics to plain language."""

from typing import Dict, Any, List, Optional
import numpy as np


class GPStateTranslator:
    """Translates GP state to plain-language descriptions."""
    
    @staticmethod
    def translate_lengthscales(lengthscales: np.ndarray, param_names: List[str]) -> Dict[str, str]:
        """Translate lengthscales to parameter importance descriptions."""
        if lengthscales is None or len(lengthscales) == 0:
            return {}
        
        if len(lengthscales) == 1:
            return {
                'summary': f"The model has learned a lengthscale of {lengthscales[0]:.4f}",
                'detail': "This indicates how quickly the function changes across the parameter"
            }
        
        result = {}
        descriptions = []
        
        sorted_indices = np.argsort(lengthscales)
        most_important = param_names[sorted_indices[0]] if len(param_names) > 0 else ""
        least_important = param_names[sorted_indices[-1]] if len(param_names) > 0 else ""
        
        result['most_important'] = f"{most_important} (lengthscale: {lengthscales[sorted_indices[0]]:.4f})"
        result['least_important'] = f"{least_important} (lengthscale: {lengthscales[sorted_indices[-1]]:.4f})"
        
        for i, name in enumerate(param_names):
            desc = f"{name}: {lengthscales[i]:.4f}"
            if i == sorted_indices[0]:
                desc += " (most important)"
            elif i == sorted_indices[-1]:
                desc += " (least important)"
            descriptions.append(desc)
        
        result['detail'] = "\n  ".join(descriptions)
        result['summary'] = f"Parameter {most_important} appears most important"
        
        return result
    
    @staticmethod
    def translate_noise(noise: Optional[float]) -> Dict[str, str]:
        """Translate noise level to description."""
        if noise is None:
            return {'summary': "No noise information available"}
        
        if noise < 0.001:
            return {
                'summary': "Very low noise detected",
                'detail': "The objective function appears deterministic"
            }
        elif noise < 0.01:
            return {
                'summary': "Low noise detected",
                'detail': "The objective function is fairly consistent"
            }
        elif noise < 0.1:
            return {
                'summary': "Moderate noise detected",
                'detail': "There is some variability in the objective"
            }
        else:
            return {
                'summary': "High noise detected",
                'detail': "The objective function has significant variability"
            }
    
    @staticmethod
    def translate_convergence(n_iterations: int, best_improvements: List[float]) -> Dict[str, str]:
        """Translate convergence status to description."""
        if len(best_improvements) < 3:
            return {'summary': "Too early to assess convergence"}
        
        recent = best_improvements[-5:] if len(best_improvements) >= 5 else best_improvements
        recent_improvements = [imp for imp in recent if imp > 0.01]
        
        if len(recent_improvements) == 0:
            return {
                'summary': "Converged",
                'detail': "No significant recent improvements"
            }
        elif len(recent_improvements) < 2:
            return {
                'summary': "Approaching convergence",
                'detail': "Improvements are becoming rare"
            }
        else:
            return {
                'summary': "Still improving",
                'detail': f"Found {len(recent_improvements)} improvements in recent iterations"
            }
    
    @staticmethod
    def translate_overall_state(gp_stats: Dict[str, Any], param_names: List[str]) -> Dict[str, str]:
        """Generate overall state description."""
        if not gp_stats.get('is_fitted', False):
            return {'summary': "Gaussian Process not yet fitted"}
        
        result = {}
        
        lengthscales = gp_stats.get('lengthscales')
        if lengthscales is not None and len(lengthscales) > 0:
            importance = GPStateTranslator.translate_lengthscales(lengthscales, param_names)
            result.update(importance)
        
        noise = gp_stats.get('noise')
        if noise is not None:
            noise_desc = GPStateTranslator.translate_noise(noise)
            result['noise'] = noise_desc['summary']
        
        n_samples = gp_stats.get('n_samples', 0)
        result['samples'] = f"Based on {n_samples} evaluations"
        
        return result


class DecisionTranslator:
    """Translates acquisition decisions to plain language."""
    
    @staticmethod
    def translate_exploration(std: float, mean: float, best_value: float) -> Dict[str, str]:
        """Translate exploration decision."""
        std_relative = std / (abs(best_value) + 1e-6)
        
        if std_relative > 2.0:
            confidence = "very uncertain"
        elif std_relative > 1.0:
            confidence = "somewhat uncertain"
        else:
            confidence = "moderately uncertain"
        
        return {
            'summary': f"Exploring due to {confidence} predictions",
            'detail': f"The model is {confidence} about this region (std={std:.4f})"
        }
    
    @staticmethod
    def translate_exploitation(mean: float, std: float, best_value: float) -> Dict[str, str]:
        """Translate exploitation decision."""
        diff = abs(mean - best_value)
        relative_diff = diff / (abs(best_value) + 1e-6)
        
        if relative_diff < 0.05:
            closeness = "very close"
        elif relative_diff < 0.1:
            closeness = "close"
        else:
            closeness = "reasonably close"
        
        return {
            'summary': f"Exploiting because predicted value is {closeness} to best",
            'detail': f"Predicted value {mean:.4f}, best {best_value:.4f} (diff={diff:.4f})"
        }
    
    @staticmethod
    def translate_balanced(mean: float, std: float, ei: float) -> Dict[str, str]:
        """Translate balanced decision."""
        return {
            'summary': "Balancing exploration and exploitation",
            'detail': f"Expected improvement of {ei:.4f} from predicted {mean:.4f} (uncertainty={std:.4f})"
        }