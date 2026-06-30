import numpy as np


class ArtificialNeuron:
    def __init__(self, input_size):
        # Initialize random weights for each input feature, and a bias knob
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()

    def sigmoid(self, x):
        # Sigmoid activation function to squash output between 0 and 1
        return 1 / (1 + np.exp(-x))
    
    def forward(self, x_vector):
        # Calculate the weighted sum of inputs and bias
        z = np.dot(self.weights, x_vector) + self.bias
        # Pass through sigmoid to get the final output
        return self.sigmoid(z)

# 1. Create a stable neuron with fixed weights and bias for testing
neuron = ArtificialNeuron(input_size=3)
neuron.weights = np.array([0.8, -0.5, 0.2]) # Cares about speed, dislikes size
neuron.bias = -0.1                           # A slight negative baseline bias
# 2. Define two different fish profiles [Speed, Size, Cleaning]
fast_small_fish = np.array([0.9, 0.1, 0.2])  # Fits what the weights like
slow_large_fish = np.array([0.1, 0.9, 0.2])  # Opposite of what weights like
# 3. Run them through the neuron
prediction_a = neuron.forward(fast_small_fish)
prediction_b = neuron.forward(slow_large_fish)
print(f"Fast, Small Fish Confidence: {prediction_a:.4f}")
print(f"Slow, Large Fish Confidence: {prediction_b:.4f}")