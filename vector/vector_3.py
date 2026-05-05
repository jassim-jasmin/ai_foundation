import numpy as np

# Vector [Speed, Cleaning, Size]
aquarium_matrix = np.array([
    [9,1,5],
    [4, 9, 6],
    [1, 10, 3]
])

# Weight priority: Cleaning
weights = np.array([0.1, 0.8, 0.1])

# Applay weights (Dot product)

scores = np.dot(aquarium_matrix, weights)

print(f"Tetra: {scores[0]}")
print(f"SAE: {scores[1]}")
print(f"Snail: {scores[2]}")