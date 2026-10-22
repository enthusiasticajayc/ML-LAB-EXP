
import numpy as np
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot

# Radial kernel function
def radial_kernel(x0, X, tau):
    return np.exp(-((X - x0) ** 2) / (2 * tau ** 2))

# Locally Weighted Regression
def local_regression(x0, X, Y, tau):
    # Add bias term
    x0_bias = np.array([1.0, x0])
    X_bias = np.c_[np.ones(len(X)), X]

    # Calculate kernel weights
    weights = radial_kernel(x0, X, tau)

    # Weighted regression using normal equations
    Xw = X_bias.T * weights
    beta = np.linalg.pinv(Xw @ X_bias) @ Xw @ Y

    # Predict value
    return x0_bias @ beta

# Generate dataset
n = 1000
X = np.linspace(-3, 3, num=n)

print("The Data Set (10 Samples) X:\n", X[1:10])

Y = np.log(np.abs(X ** 2 - 1) + 0.5)

print("The Fitting Curve Data Set (10 Samples) Y:\n", Y[1:10])

# Add noise to X
np.random.seed(42)
X += np.random.normal(scale=0.1, size=n)

print("Noisy (10 Samples) X:\n", X[1:10])

# Prediction domain
domain = np.linspace(-3, 3, num=300)

print("X0 Domain Space (10 Samples):\n", domain[1:10])

# Plot LWR for different tau values
def plot_lwr(tau):
    prediction = [
        local_regression(x0, X, Y, tau)
        for x0 in domain
    ]

    plot = figure(
        width=400,
        height=400,
        title=f"tau = {tau}",
        x_axis_label="X",
        y_axis_label="Y"
    )

    plot.scatter(X, Y, alpha=0.3, size=4)
    plot.line(
        domain,
        prediction,
        line_width=2,
        color="red"
    )

    return plot

# Display four plots
show(gridplot([
    [plot_lwr(10.0), plot_lwr(1.0)],
    [plot_lwr(0.1), plot_lwr(0.01)]
]))
