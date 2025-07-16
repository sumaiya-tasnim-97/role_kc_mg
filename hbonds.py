"""
H-Bond Distance Summary and Comparison Plot Script

Description:
    This script reads multiple `.csv` or `.dat` files containing hydrogen bond distance
    data across molecular dynamics simulation frames, calculates the average and standard
    deviation of each H-bond across time, and generates a grouped bar plot comparing
    these values across different simulation runs.

Workflow:
    - Automatically scans the working directory for `.csv` and `.dat` files.
    - Reads each file, assuming a column format of H-bond labels with optional 'frame' column.
    - Computes the mean and standard deviation for each H-bond.
    - Groups the results by simulation and generates a side-by-side bar plot with error bars.

Requirements:
    - Python with `pandas`, `numpy`, `matplotlib`, and `os` modules installed.
    - Input files should:
        • Be in `.csv` or whitespace-delimited `.dat` format.
        • Contain H-bond distance values in columns (one column per H-bond).
        • Optionally include a `frame` column, which will be ignored.
    - Each file represents one simulation or condition and should have a descriptive name.

Output:
    - A bar plot image saved as `hbond_comparison_barplot.png`, showing the average H-bond
      distances and their standard deviations across simulations.
    - Each group of bars corresponds to one H-bond, with separate bars for each simulation.

Use Case:
    This script is ideal for summarizing and comparing structural stability (e.g., H-bond lengths)
    in different simulation conditions—such as ionic environments, base mutations, or loop conformations.

"""



import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === SETTINGS ===
data_dir = "./"
csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv") or f.endswith(".dat")])

summary_data = {}
labels = None

# === READ AND PROCESS EACH FILE ===
for f in csv_files:
    df = pd.read_csv(os.path.join(data_dir, f), delim_whitespace=f.endswith(".dat"))
    
    # Drop 'frame' column if present
    if 'frame' in df.columns:
        df = df.drop(columns=['frame'])
    
    if labels is None:
        labels = df.columns.tolist()
    
    # Compute column-wise mean and std
    avg = df.mean()
    std = df.std()

    sim_name = os.path.splitext(f)[0]
    summary_data[sim_name] = {"mean": avg, "std": std}

# === PLOT ===
x = np.arange(len(labels))  # positions on x-axis
bar_width = 0.15
offsets = np.linspace(-bar_width * len(summary_data) / 2, bar_width * len(summary_data) / 2, len(summary_data))

plt.figure(figsize=(12, 6))

for i, (sim_name, stats) in enumerate(summary_data.items()):
    plt.bar(x + offsets[i], stats["mean"], yerr=stats["std"], width=bar_width,
            capsize=4, label=sim_name)

# === LABELS AND TICKS ===
plt.xticks(ticks=x, labels=labels, rotation=45, ha='right')
plt.ylabel("Average H-Bond Distance (Å)")
plt.title("Comparison of H-Bond Distances Across Simulations")
plt.legend()
plt.tight_layout()
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.savefig("hbond_comparison_barplot.png", dpi=300)
plt.show()

