from models.sde_sir import simulate_sde_sir
from models.ode_sir import simulate_ode_sir
from models.plotting import plot_simulation
from utils.export import export_results_to_csv, export_results_to_json
from simulations.agent_based import simulate_agent_based
from simulations.network_dynamics import simulate_network_dynamics
from ml.ml_model import generate_ml_training_data, train_ml_model
from utils.data_loader import download_and_load_covid_confirmed_data

def main():
    # Load COVID-19 confirmed cases time-series data
    print("\nDownloading and loading COVID-19 confirmed cases data...")
    df_confirmed = download_and_load_covid_confirmed_data()
    print("Sample of the loaded data:")
    print(df_confirmed.head())

    # Common simulation parameters
    beta = 0.5
    gamma = 0.1
    sigma = 0.05
    S0, I0, R0 = 990, 10, 0
    T = 50
    dt = 0.5

    # 1. Run SDE (stochastic) simulation
    print("\nStarting SDE SIR simulation...")
    S_sde, I_sde, R_sde, t_sde = simulate_sde_sir(beta, gamma, sigma, S0, I0, R0, T, dt)
    plot_simulation(t_sde, S_sde, I_sde, R_sde, "Stochastic SIR Model (SDE)")

    # 2. Run ODE simulation
    print("\nStarting ODE SIR simulation...")
    S_ode, I_ode, R_ode, t_ode = simulate_ode_sir(beta, gamma, S0, I0, R0, T, dt)
    plot_simulation(t_ode, S_ode, I_ode, R_ode, "Deterministic SIR Model (ODE)")

    # 3. Export simulation results (using SDE simulation data)
    data = {
        "time": t_sde.tolist(),
        "Susceptible": S_sde.tolist(),
        "Infected": I_sde.tolist(),
        "Recovered": R_sde.tolist()
    }
    export_results_to_csv(data, "sde_sir_results.csv")
    export_results_to_json(data, "sde_sir_results.json")

    # 4. Run agent-based simulation
    print("\nRunning agent-based simulation...")
    agent_history = simulate_agent_based()
    print("Final agent-based simulation state:", agent_history[-1])

    # 5. Run network dynamics simulation
    print("\nSimulating network dynamics...")
    network_history, G = simulate_network_dynamics()
    final_patch = list(network_history.keys())[0]
    print(f"Final state for patch {final_patch}:", network_history[final_patch][-1])

    # 6. (Optional) Machine Learning: Generate synthetic training data and train ML model
    # You can also integrate parameter estimation from real data here.
    print("\nGenerating ML training data...")
    X, y = generate_ml_training_data(num_samples=10)
    ml_model = train_ml_model(X, y)
    sample_params = [[0.6, 0.15, 0.1]]
    predicted_peak = ml_model.predict(sample_params)
    print(f"Predicted outbreak peak time for parameters {sample_params[0]}: {predicted_peak[0]:.2f}")

    print("\nProject simulation completed.")

if __name__ == '__main__':
    main()
