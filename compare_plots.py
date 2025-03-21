#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# Q2 values to compare for the model data
Q2_values = [2.75]

# Create a new figure for the combined plot
plt.figure()

# Loop over each Q2 value and plot its model data
for Q2 in Q2_values:
    # For Q2==2.99, label it as 3.0 for display purposes
    Q2_label = 3.0 if Q2 == 2.99 else Q2
    file_path = f"output/cc_for_Q2={Q2}_GeV.txt"
    
    # Lists to store data for this Q2 value
    W_values = []
    lp_cross = []
    
    # Open and parse the model data file
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines or header lines starting with "E_le"
            if not line or line.startswith("E_le"):
                continue
            tokens = line.split()
            if len(tokens) < 7:
                continue
            try:
                # tokens[1] is W and tokens[3] is the inclusive cross section (l p -> l- X)
                W = float(tokens[1])
                lp_val = float(tokens[3])
            except ValueError:
                continue
            W_values.append(W)
            lp_cross.append(lp_val)
    
    # Plot the model data for this Q2 value
    plt.plot(W_values, lp_cross, linestyle='-', label=f"Model: Q$^2$ = {Q2_label} GeV$^2$")

# Function to plot experimental data
def plot_exp_data(file_name, label, color):
    W_exp = []
    sigma_exp = []
    error_exp = []

    with open(file_name, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            # Skip header lines (if any)
            try:
                W_val = float(tokens[0])
            except ValueError:
                continue
            # Only consider experimental data for W up to 2 GeV
            if W_val > 2.0:
                continue
            # Multiply sigma and errors by 10^(-3)
            sigma_val = float(tokens[2]) * 1e-3  
            err = float(tokens[3]) * 1e-3
            sys_err = float(tokens[4]) * 1e-3
            total_err = np.sqrt(err**2 + sys_err**2)  
            W_exp.append(W_val)
            sigma_exp.append(sigma_val)
            error_exp.append(total_err)

    # Plot the experimental data with error bars
    plt.errorbar(W_exp, sigma_exp, yerr=error_exp, fmt='o', capsize=4, label=label, color=color,markersize=4)

# --- Add the experimental data ---
plot_exp_data("exp_data/InclusiveExpValera_Q2=2.774.dat", "Exp. Data: Q$^2$ = 2.774 GeV$^2$", color="red")
#plot_exp_data("InclusiveExpValera_Q2=3.25.dat", "Exp. Data: Q$^2$ = 3.244 GeV$^2$", color="blue")

# --- Customize the combined plot ---
plt.xlabel("W, GeV")
plt.xticks(np.arange(1.1, 2.0 + 0.1, 0.1))  # Set x-ticks with a step of 0.1
plt.ylabel("d$\sigma$/dWdQ$^2$ [$\mu$b/GeV$^3$]")
plt.ylim(bottom=0)  # Always include 0 on the y-axis
plt.title("Comparison of Inclusive Cross Sections vs W")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save and display the combined plot in the current directory
plt.savefig("compare_cc_plots.png")
plt.show()
