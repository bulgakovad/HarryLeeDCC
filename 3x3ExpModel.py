#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import os

# Q2 values for the 3x3 grid of experimental plots
q2_grid = [2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093,8.294, 9.699]  # last one is where model is overlaid

# Function to load experimental data from file
def load_exp_data(q2):
    W_exp = []
    sigma_exp = []
    error_exp = []
    file_name = f"exp_data/InclusiveExpValera_Q2={q2}.dat"
    if not os.path.exists(file_name):
        print(f"Warning: {file_name} not found.")
        return [], [], []
    with open(file_name, "r") as f:
        for line in f:
            tokens = line.strip().split()
            try:
                W = float(tokens[0])
                if W > 2.0:
                    continue
                sigma = float(tokens[2]) * 1e-3
                err = float(tokens[3]) * 1e-3
                sys_err = float(tokens[4]) * 1e-3
                total_err = np.sqrt(err**2 + sys_err**2)
                W_exp.append(W)
                sigma_exp.append(sigma)
                error_exp.append(total_err)
            except (ValueError, IndexError):
                continue
    return W_exp, sigma_exp, error_exp

# Function to load model data only for Q2 = 2.75
def load_model_data():
    W_model = []
    sigma_model = []
    file_path = "output/cc_for_Q2=2.75_GeV.txt"
    if not os.path.exists(file_path):
        print("Warning: model file not found.")
        return [], []
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("E_le") or not line.strip():
                continue
            tokens = line.strip().split()
            try:
                W = float(tokens[1])
                sigma = float(tokens[3])
                W_model.append(W)
                sigma_model.append(sigma)
            except (ValueError, IndexError):
                continue
    return W_model, sigma_model

# Setup 3x3 subplots
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle("Inclusive Cross Sections VS W: Experimental Data (overlayed with Model at Q$^2$=2.75)", fontsize=16)

for idx, q2 in enumerate(q2_grid):
    row, col = divmod(idx, 3)
    ax = axes[row][col]
    W_exp, sigma_exp, error_exp = load_exp_data(q2)
    if W_exp:
        ax.errorbar(W_exp, sigma_exp, yerr=error_exp, fmt='o', markersize=3, capsize=3, label=f"Exp: Q$^2$ = {q2} GeV$^2$", color='blue')
    if q2 == 2.774:
        W_model, sigma_model = load_model_data()
        if W_model:
            ax.plot(W_model, sigma_model, linestyle='-', color='black', label="Model: $Q^2 = 2.75$ GeV")
    ax.set_xlim(1.1, 2.05)
    ax.set_ylim(bottom=0)
    ax.set_xticks(np.arange(1.1, 2.1, 0.1))
    ax.grid(True)
    ax.legend(fontsize=8)
    ax.set_xlabel("W, GeV")
    ax.set_ylabel("d$σ$/dWdQ$^2$ [$μb/GeV^3$]")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("compare_3x3_exp_model_plots.png")
plt.show()
