import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import networkx as nx

# Import modules for our other features
from utils.data_loader import download_and_load_covid_confirmed_data
from ml.parameter_estimation import estimate_parameters, sir_model
from simulations.agent_based import simulate_agent_based_positions
from models.sde_sir import simulate_sde_sir
from models.ode_sir import simulate_ode_sir

# Load COVID‑19 data globally (for parameter estimation tab)
df_confirmed = download_and_load_covid_confirmed_data()
countries = sorted(df_confirmed["Country/Region"].unique())

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label="SIR Simulation", children=[
            html.Div([
                html.H2("Stochastic SIR Model Simulation"),
                html.Label("Beta"),
                dcc.Slider(id='beta-slider', min=0.1, max=1.0, step=0.1, value=0.5),
                html.Label("Gamma"),
                dcc.Slider(id='gamma-slider', min=0.1, max=1.0, step=0.1, value=0.1),
                html.Label("Sigma"),
                dcc.Slider(id='sigma-slider', min=0.0, max=0.5, step=0.05, value=0.05),
                dcc.Graph(id='sir-graph')
            ], style={'padding': 20})
        ]),
        dcc.Tab(label="Parameter Estimation", children=[
            html.Div([
                html.H2("Parameter Estimation for SIR Model"),
                html.Label("Select Country"),
                dcc.Dropdown(id='country-dropdown',
                             options=[{'label': c, 'value': c} for c in countries],
                             value='Italy'),
                dcc.Graph(id='param-estimation-graph'),
                html.Div(id='estimation-output', style={'marginTop': '20px', 'fontSize': '18px'})
            ], style={'padding': 20})
        ]),
        dcc.Tab(label="Agent-Based Simulation", children=[
            html.Div([
                html.H2("Agent-Based Simulation Visualization"),
                dcc.Graph(id='agent-based-graph')
            ], style={'padding': 20})
        ]),
        dcc.Tab(label="Model Comparison", children=[
            html.Div([
                html.H2("Compare Deterministic ODE vs. Stochastic SDE Models"),
                html.Label("Beta"),
                dcc.Slider(id='comp-beta-slider', min=0.1, max=1.0, step=0.1, value=0.5),
                html.Label("Gamma"),
                dcc.Slider(id='comp-gamma-slider', min=0.1, max=1.0, step=0.1, value=0.1),
                html.Label("Sigma (SDE noise)"),
                dcc.Slider(id='comp-sigma-slider', min=0.0, max=0.5, step=0.05, value=0.05),
                dcc.Graph(id='model-comp-graph')
            ], style={'padding': 20})
        ]),
        dcc.Tab(label="Sensitivity Analysis", children=[
            html.Div([
                html.H2("Interactive Sensitivity Analysis"),
                html.Label("Select Parameter to Vary"),
                dcc.Dropdown(
                    id='sensitivity-param-dropdown',
                    options=[
                        {'label': 'Beta (Transmission Rate)', 'value': 'beta'},
                        {'label': 'Gamma (Recovery Rate)', 'value': 'gamma'},
                        {'label': 'Sigma (Noise Level)', 'value': 'sigma'},
                    ],
                    value='beta'
                ),
                html.Br(),
                html.Label("Sensitivity Range Lower Bound"),
                dcc.Slider(id='sensitivity-lower-slider', min=0.1, max=1.0, step=0.05, value=0.1),
                html.Label("Sensitivity Range Upper Bound"),
                dcc.Slider(id='sensitivity-upper-slider', min=0.1, max=1.0, step=0.05, value=1.0),
                html.Br(),
                html.H4("Baseline Parameter Values"),
                html.Label("Beta"),
                dcc.Slider(id='baseline-beta', min=0.1, max=1.0, step=0.1, value=0.5),
                html.Label("Gamma"),
                dcc.Slider(id='baseline-gamma', min=0.05, max=0.5, step=0.05, value=0.1),
                html.Label("Sigma"),
                dcc.Slider(id='baseline-sigma', min=0.0, max=0.5, step=0.05, value=0.05),
                dcc.Graph(id='sensitivity-graph')
            ], style={'padding': 20})
        ]),
        dcc.Tab(label="Network Dynamics", children=[
            html.Div([
                html.H2("Multi-Patch/Network Dynamics Visualization"),
                html.Label("Number of Patches"),
                dcc.Slider(id="num-patches-slider", min=3, max=20, step=1, value=5),
                html.Label("Connectivity Probability"),
                dcc.Slider(id="connectivity-slider", min=0.1, max=1.0, step=0.1, value=0.6),
                html.Label("Travel Rate"),
                dcc.Slider(id="travel-rate-slider", min=0.001, max=0.05, step=0.001, value=0.01),
                dcc.Graph(id="network-dynamics-graph")
            ], style={'padding': 20})
        ])
    ])
])


# ----------------------------
# Existing Callbacks (SIR, Parameter Estimation, Agent-Based, Model Comparison, Sensitivity Analysis)
# (Omitted here for brevity – see previous integrations)
# ----------------------------
# Callback for the SIR Simulation tab (using a placeholder simulation)
@app.callback(
    Output('sir-graph', 'figure'),
    [Input('beta-slider', 'value'),
     Input('gamma-slider', 'value'),
     Input('sigma-slider', 'value')]
)
def update_sir_simulation(beta, gamma, sigma):
    import numpy as np
    t = np.linspace(0, 50, 100)
    # Dummy dynamics for illustration. Replace with your SDE simulation as needed.
    S = 1000 * np.exp(-beta * t)
    I = 1000 * (1 - np.exp(-gamma * t)) * np.exp(-sigma * t)
    R = 1000 - S - I
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=S, mode='lines', name='Susceptible'))
    fig.add_trace(go.Scatter(x=t, y=I, mode='lines', name='Infected'))
    fig.add_trace(go.Scatter(x=t, y=R, mode='lines', name='Recovered'))
    fig.update_layout(title="Stochastic SIR Simulation", xaxis_title="Time (days)", yaxis_title="Population")
    return fig


# Callback for the Parameter Estimation tab
@app.callback(
    [Output('param-estimation-graph', 'figure'),
     Output('estimation-output', 'children')],
    Input('country-dropdown', 'value')
)
def update_param_estimation(selected_country):
    # Filter and aggregate data for the selected country.
    df_country = df_confirmed[df_confirmed["Country/Region"] == selected_country]
    df_country = df_country.groupby("Date")["Confirmed"].sum().reset_index()
    df_country = df_country.sort_values("Date")

    # Create time axis (in days)
    df_country["Days"] = (pd.to_datetime(df_country["Date"]) - pd.to_datetime(df_country["Date"].min())).dt.days
    t_data = df_country["Days"].values.astype(float)
    y_data = df_country["Confirmed"].values.astype(float)

    # Define initial conditions (adjust as needed)
    I0 = y_data[0] if y_data[0] > 0 else 1
    R0 = 0
    S0 = 1_000_000 - I0  # example susceptible population

    try:
        popt, _ = estimate_parameters(t_data, y_data, S0, I0, R0)
        beta_est, gamma_est = popt
    except Exception as e:
        return go.Figure(), f"Parameter estimation failed: {str(e)}"

    fitted = sir_model(t_data, beta_est, gamma_est, S0, I0, R0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_data, y=y_data, mode='markers', name='Observed Data'))
    fig.add_trace(go.Scatter(x=t_data, y=fitted, mode='lines',
                             name=f'Fitted SIR (β={beta_est:.3f}, γ={gamma_est:.3f})'))
    fig.update_layout(title=f"Parameter Estimation for {selected_country}",
                      xaxis_title="Days Since First Case", yaxis_title="Cumulative Confirmed Cases")

    output_text = f"Estimated β: {beta_est:.3f}, Estimated γ: {gamma_est:.3f}"
    return fig, output_text


# Callback for the Agent-Based Simulation tab
@app.callback(
    Output('agent-based-graph', 'figure'),
    Input('agent-based-graph', 'id')  # dummy input to trigger once
)
def update_agent_based_simulation(dummy):
    # Run the agent-based simulation to obtain agent snapshots over time.
    history = simulate_agent_based_positions(num_agents=200, steps=100, initial_infected=5, infection_radius=0.05)

    frames = []
    for i, snapshot in enumerate(history):
        x_vals = [agent['x'] for agent in snapshot]
        y_vals = [agent['y'] for agent in snapshot]
        # Color-code: Blue for susceptible (0), Red for infected (1), Green for recovered (2)
        colors = ["blue" if agent['state'] == 0 else "red" if agent['state'] == 1 else "green" for agent in snapshot]
        frames.append(dict(data=[go.Scatter(x=x_vals, y=y_vals, mode='markers',
                                            marker=dict(color=colors, size=8))],
                           name=str(i)))

    # Initial frame (first snapshot)
    initial_snapshot = history[0]
    x_vals = [agent['x'] for agent in initial_snapshot]
    y_vals = [agent['y'] for agent in initial_snapshot]
    colors = ["blue" if agent['state'] == 0 else "red" if agent['state'] == 1 else "green" for agent in
              initial_snapshot]
    scatter = go.Scatter(x=x_vals, y=y_vals, mode='markers', marker=dict(color=colors, size=8))

    fig = go.Figure(data=[scatter], frames=frames)

    # Add play button
    fig.update_layout(
        title="Agent-Based Simulation: Agent Movements Over Time",
        xaxis=dict(range=[0, 1], title="X Position"),
        yaxis=dict(range=[0, 1], title="Y Position"),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}])]
        )]
    )

    # Configure slider for frames
    sliders = [dict(
        steps=[dict(method="animate",
                    args=[[str(k)], {"frame": {"duration": 100, "redraw": True}, "mode": "immediate"}],
                    label=str(k)) for k in range(len(history))],
        transition={"duration": 0},
        x=0,
        y=0,
        currentvalue={"font": {"size": 12}, "prefix": "Step: ", "visible": True, "xanchor": "center"},
        len=1.0
    )]

    fig.update_layout(sliders=sliders)
    return fig


@app.callback(
    Output('model-comp-graph', 'figure'),
    [Input('comp-beta-slider', 'value'),
     Input('comp-gamma-slider', 'value'),
     Input('comp-sigma-slider', 'value')]
)
def update_model_comparison(beta, gamma, sigma):
    # Set common simulation parameters
    S0, I0, R0 = 990, 10, 0
    T = 50
    dt = 0.5

    # Run the stochastic (SDE) simulation
    S_sde, I_sde, R_sde, t_sde = simulate_sde_sir(beta, gamma, sigma, S0, I0, R0, T, dt)
    # Run the deterministic (ODE) simulation
    S_ode, I_ode, R_ode, t_ode = simulate_ode_sir(beta, gamma, S0, I0, R0, T, dt)

    fig = go.Figure()

    # Plot SDE results (solid lines)
    fig.add_trace(go.Scatter(x=t_sde, y=S_sde, mode='lines', name='S (SDE)', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=t_sde, y=I_sde, mode='lines', name='I (SDE)', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=t_sde, y=R_sde, mode='lines', name='R (SDE)', line=dict(color='green')))

    # Plot ODE results (dashed lines)
    fig.add_trace(go.Scatter(x=t_ode, y=S_ode, mode='lines', name='S (ODE)', line=dict(color='blue', dash='dash')))
    fig.add_trace(go.Scatter(x=t_ode, y=I_ode, mode='lines', name='I (ODE)', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=t_ode, y=R_ode, mode='lines', name='R (ODE)', line=dict(color='green', dash='dash')))

    fig.update_layout(title="Comparison: Deterministic ODE vs. Stochastic SDE Models",
                      xaxis_title="Time (days)", yaxis_title="Population")
    return fig


# Callback for the Sensitivity Analysis tab
@app.callback(
    Output('sensitivity-graph', 'figure'),
    [Input('sensitivity-param-dropdown', 'value'),
     Input('sensitivity-lower-slider', 'value'),
     Input('sensitivity-upper-slider', 'value'),
     Input('baseline-beta', 'value'),
     Input('baseline-gamma', 'value'),
     Input('baseline-sigma', 'value')]
)
def update_sensitivity_graph(varied_param, lower_bound, upper_bound, baseline_beta, baseline_gamma, baseline_sigma):
    # Use deterministic ODE simulation for sensitivity analysis
    S0, I0, R0 = 990, 10, 0
    T, dt = 50, 0.5

    # Generate a range of values for the chosen parameter
    param_values = np.linspace(lower_bound, upper_bound, num=20)

    peak_times = []
    peak_infected = []
    final_infections = []

    for val in param_values:
        if varied_param == 'beta':
            beta = val
            gamma = baseline_gamma
            sigma = baseline_sigma  # sigma not used in ODE simulation
        elif varied_param == 'gamma':
            beta = baseline_beta
            gamma = val
            sigma = baseline_sigma
        elif varied_param == 'sigma':
            beta = baseline_beta
            gamma = baseline_gamma
            sigma = val
        else:
            beta, gamma, sigma = baseline_beta, baseline_gamma, baseline_sigma

        S, I, R, t = simulate_ode_sir(beta, gamma, S0, I0, R0, T, dt)
        peak_time = t[np.argmax(I)]
        peak_inf = max(I)
        final_inf = R[-1]

        peak_times.append(peak_time)
        peak_infected.append(peak_inf)
        final_infections.append(final_inf)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=param_values, y=peak_times, mode='lines+markers', name='Peak Time (days)'))
    fig.add_trace(go.Scatter(x=param_values, y=peak_infected, mode='lines+markers', name='Peak Infected'))
    fig.add_trace(go.Scatter(x=param_values, y=final_infections, mode='lines+markers', name='Final Infections'))

    fig.update_layout(title="Sensitivity Analysis: Impact on Key Outcomes",
                      xaxis_title=f"{varied_param.capitalize()} Value",
                      yaxis_title="Outcome Value",
                      legend_title="Metrics")
    return fig


# Callback for the Network Dynamics tab
@app.callback(
    Output("network-dynamics-graph", "figure"),
    [Input("num-patches-slider", "value"),
     Input("connectivity-slider", "value"),
     Input("travel-rate-slider", "value")]
)
def update_network_dynamics(num_patches, connectivity_prob, travel_rate):
    # Define a helper simulation function that returns a network and final patch states.
    def simulate_network_dynamics_interactive(num_patches, connectivity_prob, travel_rate, T=50, dt=0.1, beta=0.5,
                                              gamma=0.1):
        # Generate a random network graph with specified connectivity probability.
        G = nx.erdos_renyi_graph(num_patches, connectivity_prob, seed=42)
        patches = {node: {'S': 990, 'I': 10, 'R': 0} for node in G.nodes()}
        for t in np.arange(0, T, dt):
            for node in G.nodes():
                patch = patches[node]
                N = patch['S'] + patch['I'] + patch['R']
                new_infections = beta * patch['S'] * patch['I'] / N * dt
                new_recoveries = gamma * patch['I'] * dt
                patch['S'] -= new_infections
                patch['I'] += new_infections - new_recoveries
                patch['R'] += new_recoveries
            # Simulate travel between connected patches.
            for edge in G.edges():
                transfer = travel_rate
                diff = patches[edge[0]]['I'] - patches[edge[1]]['I']
                transfer_amount = transfer * diff
                patches[edge[0]]['I'] -= transfer_amount
                patches[edge[1]]['I'] += transfer_amount
        return G, patches

    # Run the simulation with the user-defined parameters.
    T = 50;
    dt = 0.1;
    beta = 0.5;
    gamma = 0.1
    G, patches = simulate_network_dynamics_interactive(num_patches, connectivity_prob, travel_rate, T, dt, beta, gamma)

    # Get positions for nodes using a spring layout.
    pos = nx.spring_layout(G, seed=42)

    # Create edge traces.
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # Create node traces.
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        info = patches[node]
        text = f"Patch {node}<br>S: {info['S']:.0f}<br>I: {info['I']:.0f}<br>R: {info['R']:.0f}"
        node_text.append(text)
        node_color.append(info['I'])  # color nodes by the number of infected individuals.
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlOrRd',
            reversescale=False,
            color=node_color,
            size=20,
            colorbar=dict(
                thickness=15,
                title=dict(text='Infected', side='right'),
                xanchor='left'
            ),
            line_width=2
        ),
        text=node_text
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=dict(
                            text="Network Dynamics: Multi-Patch Epidemic Spread",
                            font=dict(size=16)  # Set font size here
                        ),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
