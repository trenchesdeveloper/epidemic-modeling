import numpy as np
from models.sde_sir import simulate_sde_sir
from sklearn.linear_model import LinearRegression

def generate_ml_training_data(num_samples=50):
    """
    Generate synthetic training data by varying beta, gamma, and sigma,
    then record the outbreak peak time from the SDE simulation.
    """
    X = []
    y = []
    for _ in range(num_samples):
        beta = np.random.uniform(0.2, 1.0)
        gamma = np.random.uniform(0.05, 0.5)
        sigma = np.random.uniform(0.0, 0.3)
        S, I, R, t = simulate_sde_sir(beta, gamma, sigma, 990, 10, 0, 50, 0.5)
        peak_time = t[np.argmax(I)]
        X.append([beta, gamma, sigma])
        y.append(peak_time)
        print(f"Generated ML sample: beta={beta:.2f}, gamma={gamma:.2f}, sigma={sigma:.2f}, peak_time={peak_time:.2f}")
    return np.array(X), np.array(y)

def train_ml_model(X, y):
    """
    Train a linear regression model to predict outbreak peak time from simulation parameters.
    """
    model = LinearRegression()
    model.fit(X, y)
    print("ML model trained. Coefficients:", model.coef_, "Intercept:", model.intercept_)
    return model
