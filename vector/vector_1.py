import numpy as np

# Your 3D Fish Vectors [Speed, Cleaning, Size]
tetra = np.array([9, 1, 5]) 
sae = np.array([4, 9, 6])

# Calculating distance is now just one line of code:
distance = np.linalg.norm(tetra - sae)

print(f"The 'AI Distance' between your fish is: {distance:.2f}")