import pandas as pd
import json

def export_results_to_csv(data, filename):
    """
    Export simulation results to CSV.
    """
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Results exported to {filename} (CSV).")

def export_results_to_json(data, filename):
    """
    Export simulation results to JSON.
    """
    with open(filename, 'w') as f:
        json.dump(data, f)
    print(f"Results exported to {filename} (JSON).")
