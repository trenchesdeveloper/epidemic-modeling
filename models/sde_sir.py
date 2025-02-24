import numpy as np

def simulate_sde_sir(beta, gamma, sigma, S0, I0, R0, T, dt):
    """
    Simulate the SIR model using a stochastic differential equation approach (Euler–Maruyama).
    """
    n_steps = int(T / dt)
    S = np.zeros(n_steps)
    I = np.zeros(n_steps)
    R = np.zeros(n_steps)
    t = np.linspace(0, T, n_steps)
    S[0], I[0], R[0] = S0, I0, R0
    N = S0 + I0 + R0

    for i in range(1, n_steps):
        # Noise term proportional to sqrt(dt)
        dW = np.random.normal(0, np.sqrt(dt))
        dS = -beta * S[i - 1] * I[i - 1] / N * dt + sigma * S[i - 1] * dW
        dI = (beta * S[i - 1] * I[i - 1] / N - gamma * I[i - 1]) * dt + sigma * I[i - 1] * dW
        dR = gamma * I[i - 1] * dt
        S[i] = S[i - 1] + dS
        I[i] = I[i - 1] + dI
        R[i] = R[i - 1] + dR

    print("SDE SIR simulation completed. Final values: S={:.2f}, I={:.2f}, R={:.2f}".format(S[-1], I[-1], R[-1]))
    return S, I, R, t
