'''
python3 run_experiments.py

BEFORE executing this py file to create the results/logs/results.csv file that will be used for plotting
'''

import pandas as pd
import matplotlib.pyplot as plt
import os

# ----------------------------
# Load CSV
# ----------------------------
csv_path = "results/logs/results.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"{csv_path} not found!")

df = pd.read_csv(csv_path)

# ----------------------------
# Add a column for network type based on filename
# ----------------------------
def detect_network_type(model_name):
    name = model_name.lower()
    if "small" in name:
        return "small"
    elif "tiny" in name:
        return "tiny"
    elif "wide" in name:
        return "wide"
    else:
        return "unknown"

df['network'] = df['model'].apply(detect_network_type)

# ----------------------------
# Plot: SAT/UNSAT vs eps for each network
# ----------------------------
plt.figure(figsize=(8,6))

for net_type, group in df.groupby('network'):
    # Compute fraction SAT (robustness violated) for each eps
    sat_rate = group.groupby('eps')['result'].apply(lambda x: (x=="sat").mean())
    plt.plot(sat_rate.index, sat_rate.values, marker='o', label=net_type)

plt.xlabel("Epsilon (eps)")
plt.ylabel("Fraction SAT (robustness violation)")
plt.title("Robustness violation vs eps by network")
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------
# Plot: Solver time vs eps for each network
# ----------------------------
plt.figure(figsize=(8,6))

for net_type, group in df.groupby('network'):
    mean_time = group.groupby('eps')['solve_time_s'].mean()
    plt.plot(mean_time.index, mean_time.values, marker='o', label=net_type)

plt.xlabel("Epsilon (eps)")
plt.ylabel("Average solver time (s)")
plt.title("Solver time vs eps by network")
plt.legend()
plt.grid(True)
plt.show()

# ----------------------------
# Optional: Scatter plot of counterexamples for SAT cases
# ----------------------------
import ast

plt.figure(figsize=(8,6))
colors = {'small':'r','tiny':'g','wide':'b'}

for net_type, group in df[df['result']=='sat'].groupby('network'):
    for ce_str in group['counterexample'].dropna():
        ce = ast.literal_eval(ce_str)  # convert string to list
        plt.scatter(ce[0], ce[1], color=colors[net_type], label=net_type, alpha=0.6)

plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.title("Counterexamples found by network")
plt.text(-1,1, "small:red, tiny:green, wide:blue")
plt.grid(True)
plt.show()
