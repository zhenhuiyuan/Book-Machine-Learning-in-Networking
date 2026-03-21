#Copyright@Zhenhui Yuan, 2025
import numpy as np
import matplotlib.pyplot as plt
import sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#Ensures reproducibility
np.random.seed(42)

# return a 2-dimensional space, X-axis: Feature 1 (Signal Strength), Y-axis: Feature 2 (Distance from Router).
#n_samples: 100 samples
#n_informative: both features influence the class labels
#n_redundant: All features are unique and contribute to the classification problem.
#n_clusters_per_class: each class forms one cluster
#n_classes: create a binary calssification problem
# IMPORTANT: y is the label (0,1) that randomly assigned to X.
X, y = make_classification(n_samples=100, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1, n_classes=2, random_state=42)


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Initialize k-NN classifier with k=5 (you can try different k values)
k = 30 # change k=5
knn = KNeighborsClassifier(n_neighbors=k)

# Train the model
knn.fit(X_train, y_train)

# Make predictions on the test data
y_pred = knn.predict(X_test)
# 1. Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# 2. Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)


# 3. Classification Report (Precision, Recall, F1-Score)
class_report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(class_report)

# Create a mesh grid for decision boundary visualization
# Range for Feature 1 (Signal Strength). X[:, 0]: Refers to Feature 1 values
#find min and max values to ensure the grid fully spans the dataset.
#Add/minus 1 extends the range slightly to avoid cutting off the decision boundary at the edges.
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# Range for Feature 2 (Distance). X[:, 1]: Refers to Feature 2 values
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# Create a mesh grid that defines a grid of coordinates over the feature space.
#a sequence of values from x_min to x_max with a step size of 0.01. A smaller step size ensures a smoother visualization but increases computation.
#xx: Contains the x-coordinates (Feature 1 values) of the grid.
#yy: Contains the y-coordinates (Feature 2 values) of the grid.
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Different k values
k_values = [1, 15, 30]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
cmap_bold = ['#FF0000', '#0000FF']

for i, k in enumerate(k_values):
    ax = axes[i]
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)

    # Predict class labels for each point on the grid
    #Z is a 1D array of predictions, where each entry corresponds to the predicted class (e.g., 0 or 1) for a grid point.
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    #Reshape the predictions to match the grid shape
    Z = Z.reshape(xx.shape)

    # Plot decision boundary
    ax.contourf(xx, yy, Z, alpha=0.8, cmap=cmap_light)

    # Plot data points
    for idx, color in enumerate(cmap_bold):
        ax.scatter(X[y == idx][:, 0], X[y == idx][:, 1], c=color, edgecolor='k', label=f'Class {idx + 1}')

    ax.set_title(f'k = {k}')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.legend()
    ax.set_xlabel('Feature 1: Signal Strength')
    ax.set_ylabel('Feature 2: Distance from Router')

plt.tight_layout()
plt.show()

#*****Plot Accuracy for Different k Values******
# Define range of k values
k_values = range(1, 31)  # Trying k from 1 to 30
accuracies = []

# Loop through each k value and calculate the accuracy
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)  # Train the model on the training set
    y_pred = knn.predict(X_test)  # Predict on the test set
    accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
    accuracies.append(accuracy)

# Plot the accuracy curve
plt.figure(figsize=(8, 6))
plt.plot(k_values, accuracies, marker='o', color='b', linestyle='-', linewidth=2, markersize=6)
plt.title("Accuracy vs. k (Number of Neighbors)", fontsize=14)
plt.xlabel("k (Number of Neighbors)", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.grid(True)
plt.xticks(range(1, 31, 2))  # X-axis ticks at every 2 steps for clarity
plt.show()
#************************************************


# Example data point (Feature 1, Feature 2)
new_data_point = np.array([[2.5, 3.5]])

# Predict the class for the new data point
predicted_class = knn.predict(new_data_point)

print(f"Predicted Class for the new data point {new_data_point}: {predicted_class}")
