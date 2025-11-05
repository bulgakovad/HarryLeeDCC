import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_F2_columns_vs_W(Q2_list):
    """
    Reads a txt file with columns:
      Q2, W, F2full, F2_LO, F2_Q1, F2_G1
    and for each Q2 in Q2_list, plots all F2 variants vs W on one figure.

    Parameters:
        file_path : str   - path to the .txt data file
        Q2_list   : list  - list of Q² values to plot
        out_folder: str   - folder to save output plots (default: 'plots_F2_vs_W')
    """

    file_path = "Output/F2_NLO_terms.txt"

    # Read file (tab or whitespace separated)
    df = pd.read_csv(file_path, sep=r"\s+", header=None,
                     names=["Q2", "W", "F2full", "F2_LO", "F2_Q1", "F2_G1"])

    # Read external LO (CJ15LO) file
    df_lo_cj15 = pd.read_csv("Output_naked/ALL_Q2_broad_W_F2_LO.txt",
                             sep=r"\s+", comment="#", header=None,
                             names=["Q2", "W", "x", "F2_LO_CJ15"])

    # Loop over requested Q² values
    for Q2_val in Q2_list:
        # tolerance for floating comparison
        tol = 1e-3
        sub = df[np.isclose(df["Q2"], Q2_val, atol=tol)]
        if sub.empty:
            print(f"[WARN] No data found for Q² = {Q2_val}")
            continue

        # Sort by W to ensure smooth curves
        sub = sub.sort_values("W")

        # Compute sum of LO, Q1, G1
        sub["F2_sum"] = sub["F2_LO"] + sub["F2_Q1"] + sub["F2_G1"]
        sub["F2_NLO_only"] = sub["F2_Q1"] + sub["F2_G1"]

        # Get LO from CJ15LO
        sub_lo = df_lo_cj15[np.isclose(df_lo_cj15["Q2"], Q2_val, atol=tol)].sort_values("W")

        # Plot all F2 components
        plt.figure(figsize=(8,6))
        plt.plot(sub["W"], sub["F2full"], label=r"$F_2^{full:LO+NLO}$", lw=2)
        plt.plot(sub["W"], sub["F2_LO"],   label=r"$F_2^{LO}$", lw=2)
        plt.plot(sub["W"], sub["F2_NLO_only"],   label=r"$F_2^{NLO:Q_1+G_1}$",  lw=2)
        plt.plot(sub["W"], sub["F2_Q1"],   label=r"$F_2^{Q_1}$", lw=2, ls="dashed")
        plt.plot(sub["W"], sub["F2_G1"],   label=r"$F_2^{G_1}$", lw=2, ls="dashdot")

        # New CJ15 LO curve
        if not sub_lo.empty:
            plt.plot(sub_lo["W"], sub_lo["F2_LO_CJ15"], label=r"$F_2^{LO\; CJ15lo} $", lw=2, color="black", ls=":")

        plt.xlabel(r"$W$ [GeV]", fontsize=13)
        plt.ylabel(r"$F_2(W, Q^2)$", fontsize=13)
        plt.title(rf"$Q^2 = {Q2_val:.3f}\,\mathrm{{GeV}}^2$")
        plt.legend()
        plt.grid(True, ls=":")
        plt.tight_layout()

        # Save to file
        out_file = os.path.join(f"NLO_components_plots/F2_vs_W_Q2={Q2_val:.3f}.png")
        plt.savefig(out_file, dpi=200)
        plt.close()

        print(f"Saved plot: {out_file}")

plot_F2_columns_vs_W([1.025,2.025,2.774,3.244,3.793,4.435,5.187,6.065,7.093,8.294,9.699,15.0,20.0])
