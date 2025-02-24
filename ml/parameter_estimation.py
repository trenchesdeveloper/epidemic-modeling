import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import curve_fit


def sir_model(t, beta, gamma, S0, I0, R0):
    """
    Solve the SIR ODE system and return the cumulative infected (I + R)
    as a function of time t for the given parameters and initial conditions.
    """

    def deriv(y, t, beta, gamma, N):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    N = S0 + I0 + R0
    y0 = (S0, I0, R0)
    ret = odeint(deriv, y0, t, args=(beta, gamma, N))
    S, I, R = ret.T
    return I + R  # Cumulative cases


def estimate_parameters(t_data, y_data, S0, I0, R0, bounds=([0, 0], [10, 10])):
    """
    Estimate the parameters beta and gamma for the SIR model using curve fitting.

    Parameters:
        t_data: 1D numpy array of time points (in days)
        y_data: 1D numpy array of cumulative confirmed cases
        S0, I0, R0: initial conditions for S, I, R
        bounds: lower and upper bounds for beta and gamma during fitting

    Returns:
        popt: Fitted parameters [beta, gamma]
        pcov: Covariance of the parameters
    """
    # Define a lambda that fixes the initial conditions
    fit_func = lambda t, beta, gamma: sir_model(t, beta, gamma, S0, I0, R0)
    popt, pcov = curve_fit(fit_func, t_data, y_data, p0=[0.5, 0.1], bounds=bounds, maxfev=10000)
    return popt, pcov


def plot_parameter_estimation(t_data, y_data, popt, S0, I0, R0):
    """
    Plot the real data and the SIR model output with the fitted parameters.
    """
    fitted = sir_model(t_data, popt[0], popt[1], S0, I0, R0)
    plt.figure(figsize=(8, 5))
    plt.scatter(t_data, y_data, label="Data", color="red")
    plt.plot(t_data, fitted, label=f"Fitted SIR (β={popt[0]:.3f}, γ={popt[1]:.3f})", color="blue")
    plt.xlabel("Time (days)")
    plt.ylabel("Cumulative Confirmed Cases")
    plt.title("Parameter Estimation for SIR Model")
    plt.legend()
    plt.show()
