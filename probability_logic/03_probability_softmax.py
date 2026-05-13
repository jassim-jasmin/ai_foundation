import numpy as np

# 1. Raw Confidence Scores from a Model
# Tetra, SAE, Snail
logits = np.array([3.0, 1.0, 0.2])

# The 'Stretch' (Exponentiation)
# This ensure all values are positive and amplifies differences
stretched_scores = np.exp(logits)

# The 'Probability' (Normalization)
# Make them all sum to 1.0 (100%)
probabilities = stretched_scores / np.sum(stretched_scores)

print(f"Stretched Scores: {stretched_scores}")
print(f"Probabilities: {probabilities}")