import numpy as np
import networkx as nx

def simulate_network_dynamics(num_patches=5, T=50, dt=0.1, beta=0.5, gamma=0.1):
    """
    Simulate epidemic spread across interconnected patches using a network model.
    Each patch runs a local SIR model and connected patches exchange a fraction of infected individuals.
    """
    G = nx.erdos_renyi_graph(num_patches, 0.6, seed=42)
    patches = {node: {'S': 990, 'I': 10, 'R': 0} for node in G.nodes()}
    history = {node: [] for node in G.nodes()}

    for t in np.arange(0, T, dt):
        for node in G.nodes():
            patch = patches[node]
            N = patch['S'] + patch['I'] + patch['R']
            new_infections = beta * patch['S'] * patch['I'] / N * dt
            new_recoveries = gamma * patch['I'] * dt
            patch['S'] -= new_infections
            patch['I'] += new_infections - new_recoveries
            patch['R'] += new_recoveries
            history[node].append((t, patch['S'], patch['I'], patch['R']))
        # Simple travel between connected patches
        for edge in G.edges():
            transfer = 0.01
            diff = patches[edge[0]]['I'] - patches[edge[1]]['I']
            transfer_amount = transfer * diff
            patches[edge[0]]['I'] -= transfer_amount
            patches[edge[1]]['I'] += transfer_amount

    print("Network dynamics simulation completed.")
    return history, G
