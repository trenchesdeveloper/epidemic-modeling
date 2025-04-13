import numpy as np
from scipy.integrate import odeint


def simulate_sird(beta, gamma, mu, S0, I0, R0, D0, T, dt):
    """
    Simulate the SIRD model using ODEs.

    Equations:
        dS/dt = -beta * S * I / N
        dI/dt = beta * S * I / N - gamma * I - mu * I
        dR/dt = gamma * I
        dD/dt = mu * I
    """

    def deriv(y, t, beta, gamma, mu, N):
        S, I, R, D = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I - mu * I
        dRdt = gamma * I
        dDdt = mu * I
        return dSdt, dIdt, dRdt, dDdt

    N = S0 + I0 + R0 + D0
    y0 = (S0, I0, R0, D0)
    t_arr = np.linspace(0, T, int(T / dt))
    ret = odeint(deriv, y0, t_arr, args=(beta, gamma, mu, N))
    S, I, R, D = ret.T
    return S, I, R, D, t_arr
