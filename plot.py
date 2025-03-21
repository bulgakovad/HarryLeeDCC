#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# Lists to store data
W_values = []
lp_cross = []      # l p -> l- X values (4th column)
lp_pi_cross = []   # l p -> l- piN values (5th column)

Q2_available = [2.99, 2.75, 2.5, 2.25, 2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

'''
Well, actually the code is capable to produce cross sections for these Q2 values: 
[ 0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.0125, 0.015, 0.0175, 0.02, 0.0225, 0.025, 0.0275, 0.03, 0.035, 0.04,
0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.25,
2.5, 2.75, 3.0]
'''

for Q2 in Q2_available:   
    # Open and parse the output file (e.g., "cc_for_0.5.txt")
    with open(f"output/cc_for_Q2={Q2}_GeV.txt", "r") as f:
        if Q2 == 2.99:
            Q2 = 3.0
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
            #lp_pi_cross.append(lp_pi_val)

    # Create the plot
    plt.figure()
    plt.plot(W_values, lp_cross,  linestyle='-', label="Inclusive: ep -> e X") # delete  marker='o' for smooth line
    #plt.plot(W_values, lp_pi_cross, marker='s', linestyle='--', label="Single pion: ep -> e pi N")
    plt.xlabel("W, GeV")
    plt.xticks(np.arange(min(W_values), max(W_values) + 0.1, 0.1)) # Set x-ticks with a step of 0.1
    plt.ylabel("d$\sigma$/dWdQ$^{2}, $ $\mu$b/GeV$^{3}$")
    plt.ylim(bottom=0) # Set the lower limit of the y-axis t0 0
    plt.title(f"Cross Sections vs W for Q$^2$ = {Q2} GeV$^2$")
    plt.grid(True)
    #plt.legend()  # don't need legend if only inclusive cross section is plotted
    plt.tight_layout()
    
    plt.savefig(f"plots/cc_VS_W_for_Q2={Q2}_GeV_RGA.png") # Save the plot as a PNG file in the plots folder

    plt.show() # Display the plot

    W_values.clear()  # clear the lists for the next iteration
    lp_cross.clear()
    #lp_pi_cross.clear()
