import numpy as np

# Vector [Speed, Cleaning, Size]
aquarium_matrix = np.array([
    [9,1,5],
    [4, 9, 6],
    [1, 10, 3]
])

# Min-max normalization: (Value - Min) / (Max - Min
min_values = aquarium_matrix.min(axis=0)
max_values = aquarium_matrix.max(axis=0)
normalized_matrix = (aquarium_matrix - min_values) / (max_values - min_values)

print(f"Normalized Matrix (0-1): \n{normalized_matrix}")
print(f"Min Values: {min_values}")
print(f"Max Values: {max_values}")