import numpy as np

# 1. The Ground Truth (The actual fish is a Snail at index 2)
# [Tetra, SAE, Snail]
truth = np.array([0, 0, 1])

# 2. The AI's Guess (It thought it was a Tetra!)
# [Tetra, SAE, Snail]
prediction = np.array([0.8, 0.1, 0.1])

# 3. Cross-Entropy Loss Calculation
# Forumula: -sum(truth * log(prediction))
# We add a tiny number (1e-9) to avoid log(0) errors
loss = -np.sum(truth * np.log(prediction + 1e-9))

print(f"AI Prediction: {prediction}")
print(f"The 'Penaity' (Loss): {loss:.4f}")