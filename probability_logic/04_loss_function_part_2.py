import numpy as np

# 1. The Ground Truth (The actual fish is a Snail at index 2)
# [Tetra, SAE, Snail]
truth = np.array([0, 0, 1])

# Scenario A: The AI is confidently WRONG (thinks it's a Tetra)
bad_prediction = np.array([0.8, 0.1, 0.1])

# Scenario B: The AI is confidently RIGHT (thinks it's a Snail)
good_prediction = np.array([0.05, 0.05, 0.98])

def calculate_loss(true_labels, predicted_probs):
    # -sum(truth * log(prediction))
    # We add 1e-9 to prevent log(0) math errors
    return -np.sum(true_labels * np.log(predicted_probs + 1e-9))

loss_a = calculate_loss(truth, bad_prediction)
loss_b = calculate_loss(truth, good_prediction)

print(f"Scenario A (Confidently Wrong) Penality: {loss_a:.4f}")
print(f"Scenario B (Confidently Right) Penality: {loss_b:.4f}")