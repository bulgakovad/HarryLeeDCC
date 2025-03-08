#!/usr/bin/env python3
import matplotlib.pyplot as plt

# Lists to store data
W_values = []
lp_cross = []      # l p -> l- X values (4th column)
lp_pi_cross = []   # l p -> l- piN values (5th column)

# Open and parse the output file
with open("output.txt", "r") as f:
    for line in f:
        line = line.strip()
        # Skip header lines or empty lines
        if not line or line.startswith("E_le"):
            continue
        tokens = line.split()
        if len(tokens) < 7:
            continue
        try:
            # tokens[1] is W, tokens[3] is l p -> l- X, tokens[4] is l p -> l- piN
            W = float(tokens[1])
            lp_val = float(tokens[3])
            lp_pi_val = float(tokens[4])
        except ValueError:
            continue
        W_values.append(W)
        lp_cross.append(lp_val)
        lp_pi_cross.append(lp_pi_val)

# Create the plot
plt.figure()
plt.plot(W_values, lp_cross, marker='o', linestyle='-', label="l p -> l- X")
plt.plot(W_values, lp_pi_cross, marker='s', linestyle='--', label="l p -> l- piN")
plt.xlabel("W (GeV)")
plt.ylabel("Cross Section")
plt.title("Cross Sections vs  for Q^2 = 2.75 GeV^2")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the plot as a PNG file
plt.savefig("cc_VS_W_for_Q2=2.75_GeV_RGA.png")

# Display the plot
plt.show()
