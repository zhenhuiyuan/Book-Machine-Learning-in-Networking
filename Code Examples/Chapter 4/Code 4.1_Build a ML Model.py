#Copyright@Zhenhui Yuan, 2025
# This is a "Hello World" version of your first machine learning
# Implement the four components of building a ML: data, model, cost function, and optimization.
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Dataset: Generate synthetic data (Y = 3X + noise)
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # 100 samples, range [0,10]
Y = 3 * X + np.random.randn(100, 1) * 2  # Add Gaussian noise

# convert NumPy arrays to PyTorch tensors  because PyTorch operations (like model training, backpropagation, and optimization) require tensors
X_train = torch.tensor(X, dtype=torch.float32)
Y_train = torch.tensor(Y, dtype=torch.float32)


# 2. Model: Define a simple linear regression model
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # One input, one output

    def forward(self, x):
        return self.linear(x)


model = LinearRegression()

# 3. Cost Function: Mean Squared Error (MSE)
criterion = nn.MSELoss()

# 4. Optimization: Gradient Descent (Adam optimizer)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
epochs = 500
for epoch in range(epochs):
    model.train()

    # Forward pass: Predict
    Y_pred = model(X_train)

    # Compute loss
    loss = criterion(Y_pred, Y_train)

    # Backpropagation: Compute gradients
    optimizer.zero_grad()
    loss.backward()

    # Update model parameters
    optimizer.step()

    # Print loss every 50 epochs
    if (epoch + 1) % 50 == 0:
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')

# Extract learned parameters
w, b = model.linear.weight.item(), model.linear.bias.item()
print(f"\nLearned parameters: w = {w:.4f}, b = {b:.4f}")

# Visualize results
with torch.no_grad():
    Y_pred = model(X_train).numpy()

plt.scatter(X, Y, label="Actual Data")
plt.plot(X, Y_pred, color='red', label="Learned Regression Line")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

