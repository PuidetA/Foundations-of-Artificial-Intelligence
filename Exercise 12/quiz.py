"""import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Given values
x1, x2 = 0.4, 0.1
b1, b2 = 0.35, 0.4
w1, w2, w3, w4 = 0.2, 0.3, 0.4, 0.1
w5, w6, w7, w8 = 0.6, 0.6, 0.2, 0.1
y1, y2 = 0, 1  # Target outputs
alpha = 0.5  # Learning rate

# Forward pass for hidden neurons
z1 = w1 * x1 + w2 * x2 + b1
z2 = w3 * x1 + w4 * x2 + b2
h1 = sigmoid(z1)
h2 = sigmoid(z2)

# Forward pass for output neurons (no activation function at output)
o1 = w5 * h1 + w6 * h2
o2 = w7 * h1 + w8 * h2

# Loss computation (Mean Squared Error)
loss = 0.5 * ((y1 - o1) ** 2 + (y2 - o2) ** 2)

# Backpropagation (Gradients calculation)
dL_do1 = o1 - y1  # Gradient of loss w.r.t. o1
dL_do2 = o2 - y2  # Gradient of loss w.r.t. o2

# Gradients for weights w5, w6, w7, w8 (Output layer)
dL_dw5 = dL_do1 * h1
dL_dw6 = dL_do1 * h2
dL_dw7 = dL_do2 * h1
dL_dw8 = dL_do2 * h2

# Updating weights for output layer
w5_new = w5 - alpha * dL_dw5
w6_new = w6 - alpha * dL_dw6
w7_new = w7 - alpha * dL_dw7
w8_new = w8 - alpha * dL_dw8

# Derivatives for hidden layer gradients using the chain rule
do1_dh1, do1_dh2 = w5, w6
do2_dh1, do2_dh2 = w7, w8

# Gradients of the loss w.r.t. h1 and h2
dL_dh1 = dL_do1 * do1_dh1 + dL_do2 * do2_dh1
dL_dh2 = dL_do1 * do1_dh2 + dL_do2 * do2_dh2

# Gradients for hidden layer pre-activations z1 and z2
dL_dz1 = dL_dh1 * sigmoid_derivative(z1)
dL_dz2 = dL_dh2 * sigmoid_derivative(z2)

# Gradients for input weights w1, w2, w3, w4
dL_dw1 = dL_dz1 * x1
dL_dw2 = dL_dz1 * x2
dL_dw3 = dL_dz2 * x1
dL_dw4 = dL_dz2 * x2

# Updating the weights for input layer
w1_new = w1 - alpha * dL_dw1
w2_new = w2 - alpha * dL_dw2
w3_new = w3 - alpha * dL_dw3
w4_new = w4 - alpha * dL_dw4

# Print the updated weights
print(w1_new, w2_new, w3_new, w4_new, w5_new, w6_new, w7_new, w8_new)

print(round(w1_new, 3), round(w2_new, 3), round(w3_new, 3), round(w4_new, 3),
      round(w5_new, 3), round(w6_new, 3), round(w7_new, 3), round(w8_new, 3))
"""
import numpy as np

# Given values
x1, x2 = 0.4, 0.1
b1, b2 = 0.35, 0.4
w1, w2, w3, w4, w5, w6, w7, w8 = 0.2, 0.3, 0.4, 0.1, 0.6, 0.6, 0.2, 0.1
y1, y2 = 0, 1  # Target outputs
alpha = 0.5  # Learning rate

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Sigmoid derivative
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Step 1: Forward pass for hidden neurons
z1 = w1 * x1 + w2 * x2 + b1
z2 = w3 * x1 + w4 * x2 + b2
h1 = sigmoid(z1)
h2 = sigmoid(z2)

# Step 2: Forward pass for output neurons (linear outputs)
o1 = w5 * h1 + w6 * h2
o2 = w7 * h1 + w8 * h2

# Step 3: Loss (Mean Squared Error)
loss = 0.5 * ((y1 - o1)**2 + (y2 - o2)**2)

# Step 4: Gradients for output layer (o1 and o2)
dL_do1 = o1 - y1
dL_do2 = o2 - y2

# Gradients for weights w5, w6, w7, w8
dL_dw5 = dL_do1 * h1
dL_dw6 = dL_do1 * h2
dL_dw7 = dL_do2 * h1
dL_dw8 = dL_do2 * h2

# Update weights for output layer
w5_new = w5 - alpha * dL_dw5
w6_new = w6 - alpha * dL_dw6
w7_new = w7 - alpha * dL_dw7
w8_new = w8 - alpha * dL_dw8

# Gradients for hidden layer
do1_dh1, do1_dh2 = w5, w6
do2_dh1, do2_dh2 = w7, w8

dL_dh1 = dL_do1 * do1_dh1 + dL_do2 * do2_dh1
dL_dh2 = dL_do1 * do1_dh2 + dL_do2 * do2_dh2

dL_dz1 = dL_dh1 * sigmoid_derivative(z1)
dL_dz2 = dL_dh2 * sigmoid_derivative(z2)

# Gradients for input weights w1, w2, w3, w4
dL_dw1 = dL_dz1 * x1
dL_dw2 = dL_dz1 * x2
dL_dw3 = dL_dz2 * x1
dL_dw4 = dL_dz2 * x2

# Update weights for input layer
w1_new = w1 - alpha * dL_dw1
w2_new = w2 - alpha * dL_dw2
w3_new = w3 - alpha * dL_dw3
w4_new = w4 - alpha * dL_dw4

# Output results
w1_new, w2_new, w3_new, w4_new, w5_new, w6_new, w7_new, w8_new, loss
print(w1_new, w2_new, w3_new, w4_new, w5_new, w6_new, w7_new, w8_new)

print(round(w1_new, 3), round(w2_new, 3), round(w3_new, 3), round(w4_new, 3),
      round(w5_new, 3), round(w6_new, 3), round(w7_new, 3), round(w8_new, 3))
