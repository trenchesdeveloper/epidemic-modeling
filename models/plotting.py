import matplotlib.pyplot as plt

def plot_simulation(t, S, I, R, title="Epidemic Simulation"):
    """
    Plot simulation results for S, I, R over time.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.title(title)
    plt.legend()
    plt.show()
