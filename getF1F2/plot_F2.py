#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ------------ File paths ------------ #
model_file = "Output/F2_fixQ2_cj15.txt"
exp_file   = "exp_data_F2_strfun/F2_vs_W.dat"  # columns: Q2  W  Quantity  Uncertainty

# ------------ Load model data ------------ #
data = np.loadtxt(model_file)
Q2_vals, W_vals = data[:, 0], data[:, 1]
F2naked   = data[:, 2]  # CJ15 LT (uncorrected)
F2_tmc    = data[:, 5]  # CJ15 LT + TMC only
F2_tmc_ht = data[:, 6]  # CJ15 LT + TMC + HT

# ------------ Load experimental data ------------ #
# The first line is a header; skip it.
# The file is whitespace-delimited after the header.
exp = np.loadtxt(exp_file, skiprows=1)
Q2_exp, W_exp, F2_exp, dF2_exp = exp[:, 0], exp[:, 1], exp[:, 2], exp[:, 3]

# Q² panels to plot
Q2_targets = [1.025, 2.025, 3.025, 4.025]
ymax = {1.025: 0.4, 2.025: 0.3, 3.025: 0.2, 4.025: 0.12}

fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharex=False, gridspec_kw={'hspace': 0.25, 'wspace': 0.15})
axs = axs.flatten()

for i, Q2 in enumerate(Q2_targets):
    ax = axs[i]

    # -------- model selection for this Q² --------
    msk = np.isclose(Q2_vals, Q2)
    W = W_vals[msk]
    F2_uncorr = F2naked[msk]
    F2_tmc_only = F2_tmc[msk]
    F2_tmc_ht_sel = F2_tmc_ht[msk]

    # W-window
    range_mask = (W >= 1.0) & (W <= 1.8)
    W = W[range_mask]
    F2_uncorr = F2_uncorr[range_mask]
    F2_tmc_only = F2_tmc_only[range_mask]
    F2_tmc_ht_sel = F2_tmc_ht_sel[range_mask]

    # -------- exp selection for this Q² --------
    emsk = np.isclose(Q2_exp, Q2)
    W_e = W_exp[emsk]
    F2_e = F2_exp[emsk]
    dF2_e = dF2_exp[emsk]

    # Keep the same W-window for exp points
    erange = (W_e >= 1.0) & (W_e <= 1.8)
    W_e = W_e[erange]
    F2_e = F2_e[erange]
    dF2_e = dF2_e[erange]

    # Sort by W for prettier plotting (optional but nice)
    order = np.argsort(W)
    W, F2_uncorr, F2_tmc_ht_sel, F2_tmc_only = (
        W[order], F2_uncorr[order], F2_tmc_ht_sel[order], F2_tmc_only[order]
    )
    eorder = np.argsort(W_e)
    W_e, F2_e, dF2_e = W_e[eorder], F2_e[eorder], dF2_e[eorder]

    # -------- draw --------
    ax.plot(W, F2_uncorr,   label='CJ15 LT',               color='red', linestyle='dotted')
    ax.plot(W, F2_tmc_ht_sel, label='CJ15 LT + TMC + HT',    color='red', linewidth=1.8)
    ax.plot(W, F2_tmc_only, label='CJ15 LT + TMC only',    color='red', linestyle='dashed')

    # Experimental points with uncertainties
    ax.errorbar(
        W_e, F2_e, yerr=dF2_e,
        fmt='o', markersize=3, capsize=2, linewidth=1.0,
        label='CLAS + World data', color='black'
    )

    ax.set_title(f'Q² = {Q2} GeV²')
    ax.set_xlabel('W [GeV]')
    ax.set_ylabel('F2')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.minorticks_on()
    ax.set_ylim(bottom=0, top=ymax[Q2])
    ax.legend()

plt.suptitle('F2 vs W for different Q² (CJ15nlo) with and without TMC correction', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.94])
plt.savefig("F2_vs_W_2x2_TMC_vs_uncorr_with_exp.png", dpi=300)
plt.show()
