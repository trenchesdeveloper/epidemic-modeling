import numpy as np


def simulate_agent_based_positions(num_agents=200, steps=100, initial_infected=5, infection_radius=0.05):
    """
    Run an agent-based simulation that tracks positions and states for each agent at each time step.

    Each agent is represented as a dictionary with keys: 'id', 'x', 'y', 'state'.
    state: 0 = susceptible, 1 = infected, 2 = recovered (for now, we only use 0 and 1).

    Returns:
        history: A list (length=steps) of snapshots. Each snapshot is a list of agent dictionaries.
    """
    agents = []
    for i in range(num_agents):
        state = 1 if i < initial_infected else 0
        agent = {'id': i, 'x': np.random.rand(), 'y': np.random.rand(), 'state': state}
        agents.append(agent)

    history = [[agent.copy() for agent in agents]]  # initial snapshot

    for step in range(1, steps):
        for agent in agents:
            # Each agent performs a random walk
            agent['x'] += np.random.uniform(-0.01, 0.01)
            agent['y'] += np.random.uniform(-0.01, 0.01)
            # Keep positions within [0, 1]
            agent['x'] = np.clip(agent['x'], 0, 1)
            agent['y'] = np.clip(agent['y'], 0, 1)
        # Infection dynamics: infected agents infect nearby susceptible ones
        for agent in agents:
            if agent['state'] == 1:
                for other in agents:
                    if other['state'] == 0:
                        dist = np.sqrt((agent['x'] - other['x']) ** 2 + (agent['y'] - other['y']) ** 2)
                        if dist < infection_radius:
                            other['state'] = 1
        history.append([agent.copy() for agent in agents])

    return history
