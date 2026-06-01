import numpy as np

# Let's create a simple function : Loss = Weight squared (W^2)
# Theideal loss is 0, which happens when Weight is 0.

def loss_function(weight):
    return weight ** 2

# 1. Start with a random weight
current_weight = 3.0
inital_loss = loss_function(current_weight)

# 2. Tweak the weight by a tiny fraction (h)
h = 0.0001

twaked_weight = current_weight + h
new_loss = loss_function(twaked_weight)

# 3. Calculate the Derivative: (Change in Loss) / (Change in Weight)
derivative = (new_loss - inital_loss) / h

print(f"Original Weight: {current_weight:.4f} | Initial Loss: {inital_loss:.4f}")
print(f"Tweaked Weight: {twaked_weight:.4f} | New Loss: {new_loss:.4f}")
print(f"Estimated Derivative: {derivative:.4f}")
