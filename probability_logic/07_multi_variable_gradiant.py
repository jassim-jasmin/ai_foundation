import numpy as np
import time

# Loss function with FOUR variables: Loss = (W1^2) + (W2^2) + (W3^2) + (W4^2)
# Perfect score (Loss = 0) happens when ALL weights are 0
def calculate_loss(W1, W2, W3, W4):
    return (W1 ** 2) + (W2 ** 2) + (W3 ** 2) + (W4 ** 2)

# 1. Initialize four separate random weights
w1 = 3.0
w2 = -2.0
w3 = 1.5
w4 = -5.0
learning_rate = 0.1
h = 0.0001 # Our tiny tweak value

print("Starting Multi-Variable Training...")
print("-"*55)

# 2. Run a short 15-epoch loop to watch them coordinate
for each in range(1, 16):
    current_loss = calculate_loss(w1, w2, w3, w4)

    # --- Calculate Derivative for W1 (Freeze W2, W3, W4) --
    loss_tweak_w1 = calculate_loss(w1 + h, w2, w3, w4)  # Tweak W1 only
    derivative_w1 = (loss_tweak_w1 - current_loss) / h

    # --- Calculate Derivative for W2 (Freeze W1, W3, W4) --
    loss_tweak_w2 = calculate_loss(w1, w2 + h, w3, w4)  # Tweak W2 only
    derivative_w2 = (loss_tweak_w2 - current_loss) / h

    # --- Calculate Derivative for W3 (Freeze W1, W2, W4) --
    loss_tweak_w3 = calculate_loss(w1, w2, w3 + h, w4)  # Tweak W3 only
    derivative_w3 = (loss_tweak_w3 - current_loss) / h

    # --- Calculate Derivative for W4 (Freeze W1, W2, W3) --
    loss_tweak_w4 = calculate_loss(w1, w2, w3, w4 + h)  # Tweak W4 only
    derivative_w4 = (loss_tweak_w4 - current_loss) / h

    # The Gradient is just the collection of all derivatives: [derivative_w1, derivative_w2, derivative_w3, derivative_w4]
    # --- Update Weights ---
    w1 -= learning_rate * derivative_w1
    w2 -= learning_rate * derivative_w2
    w3 -= learning_rate * derivative_w3
    w4 -= learning_rate * derivative_w4

    print(f"Epoch {each:02d} | W1 = {w1:.3f}, W2 = {w2:.3f}, W3 = {w3:.3f}, W4 = {w4:.3f}, Loss = {current_loss:.4f}")
    time.sleep(0.1)

print("-"*55)
print(f"Final Weights -> W1: {w1:.3f}, W2: {w2:.3f}, W3: {w3:.3f}, W4: {w4:.3f}, Final Loss: {calculate_loss(w1, w2, w3, w4):.4f}")