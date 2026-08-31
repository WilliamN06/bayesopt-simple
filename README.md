# BayesOpt Made Simple

**Bayesian optimisation for ML practitioners — no stats PhD required**


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Why This Exists

You know hyperparameters matter. You know grid search is wasting your time. 
You've heard Bayesian optimisation is more efficient. 

But when you try to use BoTorch or Optuna, you hit a wall of mathematical 
notation and statistical jargon. 

**This library cuts through that complexity.**

BayesOpt Made Simple gives you:
-  **Sensible defaults** that work 80% of the time
-  **Plain-language explanations** of what the optimiser is doing
-  **Clear visualisations** that show why decisions are made
-  **Drop-in simplicity** — 3 lines of code to start optimising

No GP theory required. No acquisition function decisions. Just efficient 
hyperparameter tuning that tells you what it's doing.

## Quick Start

```python
from bayesopt import BayesOpt

def train_model(params):
    # Your model training code here
    return validation_loss

# Define your search space
bounds = {
    'learning_rate': (1e-4, 1e-1, 'log'),
    'n_estimators': (50, 500, 'int'),
    'max_depth': (3, 15, 'int')
}

# Optimise!
optimizer = BayesOpt(train_model, bounds, explain=True)
result = optimizer.run(n_calls=50)

print(f"Best params: {result.x}")
print(f"Best value: {result.y}")