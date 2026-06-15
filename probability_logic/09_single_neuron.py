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
    

# Let's test your neuron on a single fish vector!
#Input features: [Normalized Speed, Normalized Size, Normalized Cleaning]
tetra_features = np.array([0.9, 0.5, 0.1])

# Create a neuron designed to take 3 input features
neuron = ArtificialNeuron(input_size=3)

# Run the data forward through the neuron
prediction = neuron.forward(tetra_features)

print(f"Neuron Weights: {neuron.weights}")
print(f"Neuron Bias: {neuron.bias}")
print("-" * 50)
print(f"Prediction: {prediction}")