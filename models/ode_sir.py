import numpy as np
from scipy.integrate import odeint

def simulate_ode_sir(beta, gamma, S0, I0, R0, T, dt):
    """
    Simulate the SIR model using deterministic ODEs.
    """
    def deriv(y, t, beta, gamma, N):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    t = np.linspace(0, T, int(T / dt))
    N = S0 + I0 + R0
    y0 = (S0, I0, R0)
    ret = odeint(deriv, y0, t, args=(beta, gamma, N))
    S, I, R = ret.T
    print("ODE SIR simulation completed. Final values: S={:.2f}, I={:.2f}, R={:.2f}".format(S[-1], I[-1], R[-1]))
    return S, I, R, t
