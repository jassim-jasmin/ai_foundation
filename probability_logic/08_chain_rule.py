import numpy as np
# Step 1: Intermediate layer (Node B) depends on our Weight (Knob A)
# Let's say: B = A * 3
def calculate_intermediate(weight_a):
   return weight_a * 3

# Step 2: Final Loss depends entirely on the Intermediate layer (Node B)
# Let's say: Loss = B^2
def calculate_loss(intermediate_b):
   return intermediate_b ** 2

# Our starting point
current_weight_a = 2.0
h = 0.0001 # Our tiny tweak value

# --- STEP 1: How does a tweak in A change B? (Local Derivative 1) ---
base_b = calculate_intermediate(current_weight_a)
tweaked_b = calculate_intermediate(current_weight_a + h)
deriv_b_respect_to_a = (tweaked_b - base_b) / h

# --- STEP 2: How does a tweak in B change the Loss? (Local Derivative 2) ---
base_loss = calculate_loss(base_b)
tweaked_loss = calculate_loss(base_b + h)
deriv_loss_respect_to_b = (tweaked_loss - base_loss) / h

# --- THE CHAIN RULE ---
# Total Impact = (Change in B / Change in A) * (Change in Loss / Change in B)
total_derivative = deriv_b_respect_to_a * deriv_loss_respect_to_b
print(f"Local Derivative 1 (How A affects B): {deriv_b_respect_to_a:.4f}")
print(f"Local Derivative 2 (How B affects Loss): {deriv_loss_respect_to_b:.4f}")
print("-" * 50)
print(f"The Chain Rule Gradient (Total Impact of A on Loss): {total_derivative:.4f}")
