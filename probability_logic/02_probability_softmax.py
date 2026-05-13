import numpy as np


# Imagine these are 'importance scores' for your fish
logits = np.array([2.0, 1.0, 0.1]) # Tetra, SAE, Snail

# The Softmax-like step: Exponentiate and Normalize
exp_logits = np.exp(logits) # Exponentiate the logits
probabilities = exp_logits / np.sum(exp_logits) # Normalize to get probabilities

print("AI-style Probabilities (Softmax):")

for label, prob in zip(['Tetra', 'SAE', 'Snail'], probabilities):
    print(f"{label}: {prob:.4f}")