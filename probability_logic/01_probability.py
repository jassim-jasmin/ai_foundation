import numpy as np


# Counts of aquarium inhabitants
counts = np.array([10, 2, 1])
labels = ["Tetra", "SAE", "Snail"]

# Probability  = Individual Count / Total Count
probabilities = counts / np.sum(counts)

for label, prob in zip(labels, probabilities):
    print(f"Probability of {label}: {prob:.2f}")