import numpy as np
import time

# Loss = Weight squared
def calculate_loss(weight):
    return weight ** 2

# 1. Initialize our setup
weight = 3.0
learning_rate = 0.1 # Our small step size (Alpha)
h = 0.0001 # The tiny tweak to calculate the derivative

print("String trainig loop...")
print("-" * 45)

# 2. The training loop (Epochs)
for epoch in range(1, 31):
    current_loss = calculate_loss(weight)

    # Calculate the derivative numerically (just like last time)
    tweaked_loss = calculate_loss(weight + h)
    derivative = (tweaked_loss - current_loss) / h

    # THE GRADIENT DESCENT STEP: New Weight = Old Weight - (Learning Rate * Derivative)
    weight = weight - (learning_rate * derivative)

    print(f"Epoch {epoch:02d} | Weight: {weight:.4f} | Loss: {current_loss:.4f}")

print("-" * 45)
print(f"Training Complete! Final Weight: {weight:.4f} | Final Loss: {calculate_loss(weight):.4f}")