#Copyright@Zhenhui Yuan, 2025
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# Loading dataset
dataset_path = "../../Dataset/Berlin-V2X/sidelink_dataframe.parquet"  # Replace with the correct file path
data = pd.read_parquet(dataset_path)
print(list(data.columns))

# Select only numeric columns
numeric_data = data.select_dtypes(include=[np.number])
x = data['distance']
y = data['SNR']

# Handle missing values
x = x.dropna()
y = y.dropna()

# Align indices of x and y after dropping NaN
# Ensure 'speed_kmh' and 'datarate' have aligned indices and no duplicates
x = data['distance'].dropna().reset_index(drop=True)
y = data['SNR'].dropna().reset_index(drop=True)

# Combine into a single DataFrame and drop any remaining NaN
data_cleaned = pd.DataFrame({'distance': x, 'SNR': y}).dropna()

# Extract the cleaned features
x = data_cleaned['distance'].values.reshape(-1, 1)
y = data_cleaned['SNR'].values

plt.scatter(x, y, alpha=0.5)
plt.xlabel('distance')
plt.ylabel('SNR')
plt.title('Scatter Plot of distance vs SNR')
plt.show()

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# Function to plot polynomial regression results
def plot_polynomial_regression(degree):
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree)
    x_train_poly = poly.fit_transform(x_train)
    x_test_poly = poly.transform(x_test)

    # Train the model
    model = LinearRegression()
    model.fit(x_train_poly, y_train)

    # Predictions
    y_train_pred = model.predict(x_train_poly)
    y_test_pred = model.predict(x_test_poly)

    # Plot results with different colors based on x values (speed)
    plt.scatter(x, y, c=x, cmap='viridis', s=10, label='Data', alpha=0.7)

    # Create a regression curve for visualization
    x_curve = np.linspace(x.min(), x.max(), 500).reshape(-1, 1)
    y_curve = model.predict(poly.transform(x_curve))

    # Plot the regression curve
    plt.plot(x_curve, y_curve, label=f'Degree {degree}', linewidth=2, color='red')

    # Calculate errors
    train_error = mean_squared_error(y_train, y_train_pred)
    test_error = mean_squared_error(y_test, y_test_pred)
    print(f"Degree {degree} - Train Error: {train_error:.2f}, Test Error: {test_error:.2f}")

    # Labels and title
    plt.title(f'Polynomial Regression (Degree {degree})')
    plt.xlabel('distance')
    plt.ylabel('SNR')
    plt.legend()
    plt.colorbar(label='distance')  # Add colorbar to show speed values
    plt.show()


# Plot polynomial regression for degrees 1, 3, and 10
for degree in [1, 3, 10]:
    plot_polynomial_regression(degree)
