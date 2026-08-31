"""Explanation templates for plain-language descriptions."""

from typing import Dict, List


class ExplanationTemplates:
    """Container for explanation templates."""
    
    def __init__(self):
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, List[str]]:
        """Load all explanation templates."""
        return {
            'exploration': [
                "We are exploring this point because the model is uncertain here",
                "The model has high uncertainty in this region, so we are investigating",
                "We are sampling in an area we have not explored much before",
                "This region is poorly understood, so we are gathering data"
            ],
            'exploitation': [
                "We are focusing on this area because it appears promising",
                "The model predicts good performance here, so we are refining",
                "We are concentrating on a region that has produced good results",
                "This area seems likely to contain the optimum"
            ],
            'balanced': [
                "This point balances exploring new areas with exploiting known good regions",
                "We are considering both uncertainty and predicted performance here",
                "The model sees this as a good compromise between exploration and exploitation"
            ],
            'random': [
                "Randomly selecting initial exploration points",
                "Building initial understanding with random samples",
                "No prior information yet - exploring randomly"
            ],
            'improvement': [
                "We have found a better configuration",
                "Improved result discovered",
                "New best value achieved",
                "Significant improvement found"
            ],
            'convergence': [
                "The optimisation appears to be converging",
                "Improvements are becoming smaller",
                "We are approaching a likely optimum",
                "The search is narrowing in on a solution"
            ],
            'noisy': [
                "The objective function appears noisy - we are being cautious",
                "Noise detected in the objective, using more conservative estimates",
                "The model is accounting for noise in the evaluations"
            ],
            'uncertain': [
                "The model is uncertain about this region",
                "This area has not been explored enough",
                "We need more data to be confident about this region"
            ],
            'promising': [
                "This region looks promising based on current data",
                "The model predicts good performance here",
                "We have seen good results in similar areas"
            ]
        }
    
    def get(self, key: str) -> List[str]:
        """Get templates for a specific category."""
        return self.templates.get(key, self.templates['random'])
    
    def random_template(self, key: str) -> str:
        """Get a random template from a category."""
        import random
        templates = self.get(key)
        return random.choice(templates)