#!/usr/bin/env python3
import matplotlib.pyplot as plt

# Lists to store data
W_values = []
lp_cross = []      # l p -> l- X values (4th column)
lp_pi_cross = []   # l p -> l- piN values (5th column)

Q2 = 2.75  # GeV^2

# Open and parse the output file (e.g., "cc_for_0.5.txt")
with open(f"cc_for_Q2={Q2}_GeV.txt", "r") as f:
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
plt.plot(W_values, lp_cross, marker='o', linestyle='-', label="Inclusive: ep -> e X")
plt.plot(W_values, lp_pi_cross, marker='s', linestyle='--', label="Single pion: ep -> e pi N")
plt.xlabel("W (GeV)")
plt.ylabel("Cross Section")
plt.title(f"Cross Sections vs W for Q^2 = {Q2} GeV^2")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save the plot as a PNG file (e.g., "cc_VS_W_for_Q2=0.5_GeV_RGA.png")
plt.savefig(f"cc_VS_W_for_Q2={Q2}_GeV_RGA.png")

# Display the plot
plt.show()
