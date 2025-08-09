#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Load data
data = np.loadtxt("Output/F2_fixQ2_cj15.txt")
Q2_vals, W_vals = data[:,0], data[:,1]
F2naked = data[:,2]      # Uncorrected F2
F2brady = data[:,5]  # just TMC-corrected F2 (from your file columns)
F2bradyht = data[:,6]    # TMC+HT-corrected F2 (from your file columns)

# Q2 values of interest
Q2_targets = [1.025, 2.025, 3.025, 4.025]

fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=False,
                        gridspec_kw={'hspace': 0.25, 'wspace': 0.15})
axs = axs.flatten()
ymax = {1.025: 0.4, 2.025:0.3, 3.025: 0.2, 4.025: 0.12}

for i, Q2 in enumerate(Q2_targets):
    ax = axs[i]
    mask = (np.isclose(Q2_vals, Q2))
    W = W_vals[mask]
    F2_uncorr = F2naked[mask]
    F2_tmc_ht = F2bradyht[mask]
    F2_tmc = F2brady[mask] 
    

    range_mask = (W >= 1.0) & (W <= 1.8)
    W = W[range_mask]
    F2_uncorr = F2_uncorr[range_mask]
    F2_tmc_ht = F2_tmc_ht[range_mask]
    F2_tmc = F2_tmc[range_mask]

    ax.plot(W, F2_uncorr, label='CJ15 LT', color='red', linestyle='dotted')
    ax.plot(W, F2_tmc_ht, label='CJ15 LT + TMC + HT ', color='red')
    ax.plot(W, F2_tmc, label='CJ15 LT + TMC only', color='red', linestyle='dashed')
    ax.set_title(f'Q² = {Q2} GeV²')
    ax.set_xlabel('W [GeV]')
    ax.set_ylabel('F2')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.minorticks_on()
    ax.legend()

    #ax.yaxis.set_tick_params(labelleft=True)
    #ax.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))

    ax.set_ylim(bottom=0, top=ymax[Q2])

plt.suptitle('F2 vs W for different Q² (CJ15nlo) with and without TMC correction', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.94])
plt.savefig("F2_vs_W_2x2_TMC_vs_uncorr.png", dpi=300)
plt.show()
