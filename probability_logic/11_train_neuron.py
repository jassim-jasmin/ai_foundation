import numpy as np
import time
class TrainableNeuron:
   def __init__(self):
       # Start with a single weight for 1 input feature (Speed) and a bias
       self.weight = 2.0
       self.bias = -1.0
   def sigmoid(self, z):
       return 1 / (1 + np.exp(-z))
   def forward(self, x):
       return self.sigmoid(x * self.weight + self.bias)
   
# 1. Setup Data: Input is Speed, Target is the correct answer
# We want the neuron to output 1.0 when speed is high (0.9)
input_data = 0.9
target_output = 1.0
neuron = TrainableNeuron()
learning_rate = 0.5
h = 0.0001
print("Training the neuron to recognize a fast fish...")
print("-" * 55)

# 2. The Training Loop
for epoch in range(1, 31):
   # Step A: Forward Pass & Current Loss
   prediction = neuron.forward(input_data)
   current_loss = (prediction - target_output) ** 2
   # Step B: Calculate Derivative for the Weight (Tweak Weight)
   neuron.weight += h
   loss_tweak_w = (neuron.forward(input_data) - target_output) ** 2
   deriv_w = (loss_tweak_w - current_loss) / h
   neuron.weight -= h # Reset weight
   # Step C: Calculate Derivative for the Bias (Tweak Bias)
   neuron.bias += h
   loss_tweak_b = (neuron.forward(input_data) - target_output) ** 2
   deriv_b = (loss_tweak_b - current_loss) / h
   neuron.bias -= h # Reset bias
   # Step D: Gradient Descent Update
   neuron.weight -= learning_rate * deriv_w
   neuron.bias -= learning_rate * deriv_b
   print(f"Epoch {epoch:02d} | Prediction: {prediction:.4f} | Loss: {current_loss:.4f}")
   time.sleep(0.05)
print("-" * 55)
print(f"Training Complete! Final Prediction: {neuron.forward(input_data):.4f}")