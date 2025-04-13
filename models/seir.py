import numpy as np
from scipy.integrate import odeint


def simulate_seir(beta, gamma, sigma, S0, E0, I0, R0, T, dt):
    """
    Simulate the SEIR model using ODEs.

    Equations:
        dS/dt = -beta * S * I / N
        dE/dt = beta * S * I / N - sigma * E
        dI/dt = sigma * E - gamma * I
        dR/dt = gamma * I
    """

    def deriv(y, t, beta, gamma, sigma, N):
        S, E, I, R = y
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt

    N = S0 + E0 + I0 + R0
    y0 = (S0, E0, I0, R0)
    t_arr = np.linspace(0, T, int(T / dt))
    ret = odeint(deriv, y0, t_arr, args=(beta, gamma, sigma, N))
    S, E, I, R = ret.T
    return S, E, I, R, t_arr
