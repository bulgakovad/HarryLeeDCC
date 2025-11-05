#!/usr/bin/env python3
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import lhapdf

# --------------------------
# Constants
# --------------------------
M_PROTON = 0.93891897
E2 = {1: 1.0/9.0, 2: 4.0/9.0, 3: 1.0/9.0, 4: 4.0/9.0, 5: 1.0/9.0}

# --------------------------
# Helper: x from W and Q²
# --------------------------
def x_from_WQ2(W, Q2, M=M_PROTON):
    denom = W*W - M*M + Q2
    return Q2 / denom if denom > 0 else np.nan

# --------------------------
# LHAPDF F₂ calculation (CJ15lo)
# --------------------------
def f2_lo_lhapdf(pdf, x, Q2):
    """Pure LO, massless F₂ using LHAPDF x*f(x)."""
    if not (0.0 < x < 1.0):
        return np.nan
    f2 = 0.0
    # active flavors based on thresholds
    mc = pdf.quarkThreshold(4)
    mb = pdf.quarkThreshold(5)
    nf = 3
    if Q2 > mc*mc: nf += 1
    if Q2 > mb*mb: nf += 1
    for pid in range(1, nf+1):
        f2 += E2[pid] * (pdf.xfxQ2(+pid, x, Q2) + pdf.xfxQ2(-pid, x, Q2))
    return f2

# --------------------------
# Manual CJ-table version (central only)
# --------------------------
def get_pdf_interpolator_manual(fixed_Q2, central_iset=400):
    """Return interpolator for F₂(W) from your precomputed CJ tables."""
    Mp = 0.9385
    Mc, Mb = 1.3, 4.2
    q2_str = str(fixed_Q2).rstrip("0").rstrip(".")
    folder = f"../get_PDF/output/Q2={q2_str}"
    filename = f"{folder}/tst_CJpdf_ISET={central_iset}_Q2={q2_str}.dat"
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Cannot find {filename}")

    df = pd.read_csv(filename, sep=r"\s+")
    x = df["x"].values
    nu, nub = df["u"], df["ub"]
    nd, ndb = df["d"], df["db"]
    ns, nsb = df["s"], df["sb"]
    nc, ncb = df["c"], df["cb"]
    nb, nbb = df["b"], df["bb"]

    if fixed_Q2 < Mc**2:
        F2 = (4/9)*(nu+nub) + (1/9)*(nd+ndb+ns+nsb)
    elif Mc**2 <= fixed_Q2 < Mb**2:
        F2 = (4/9)*(nu+nub+nc+ncb) + (1/9)*(nd+ndb+ns+nsb)
    else:
        F2 = (4/9)*(nu+nub+nc+ncb) + (1/9)*(nd+ndb+ns+nsb+nb+nbb)

    W2 = Mp**2 + fixed_Q2*(1 - x)/x
    W = np.sqrt(W2)
    order = np.argsort(W)
    return interp1d(W[order], F2.values[order],
                    kind="cubic", bounds_error=False, fill_value="extrapolate")

# --------------------------
# Main comparison plotter
# --------------------------
def compare_F2_manual_vs_LHAPDF(Q2_list, Wmax=30.0, Wstep=0.02,
                                pdfname="CJ15lo", outdir="compare_F2_LO"):
    """Plot F₂(W,Q²) from CJ-table manual version vs LHAPDF (CJ15lo)."""
    os.makedirs(outdir, exist_ok=True)
    pdf = lhapdf.mkPDF(pdfname, 0)

    for Q2 in Q2_list:
        # get manual interpolator
        try:
            F2_manual_interp = get_pdf_interpolator_manual(Q2)
        except FileNotFoundError as e:
            print(e)
            continue

        # build W grid
        W_vals = np.arange(1.07, Wmax + Wstep/2, Wstep)

        # compute from both sources
        F2_manual = F2_manual_interp(W_vals)
        F2_lhapdf = []
        for W in W_vals:
            x = x_from_WQ2(W, Q2)
            F2_lhapdf.append(f2_lo_lhapdf(pdf, x, Q2))
        F2_lhapdf = np.array(F2_lhapdf)

        # --- Plot ---
        plt.figure(figsize=(8,6))
        plt.plot(W_vals, F2_manual, label="Manual CJ-table", lw=2)
        plt.plot(W_vals, F2_lhapdf, "--", label="LHAPDF CJ15lo", lw=2)
        plt.xlabel(r"$W$ [GeV]", fontsize=13)
        plt.ylabel(r"$F_2(W,Q^2)$", fontsize=13)
        plt.title(rf"$Q^2 = {Q2:.3f}\,\mathrm{{GeV}}^2$")
        plt.grid(True, ls=":")
        plt.legend()
        plt.tight_layout()

        outfile = os.path.join(outdir, f"compare_F2_LO_Q2={Q2:.3f}.png")
        plt.savefig(outfile, dpi=200)
        plt.close()
        print(f"Saved {outfile}")

# --------------------------
# Run for your Q² list
# --------------------------
if __name__ == "__main__":
    Q2_list = [1.2, 1.3, 1.4, 1.5, 1.69, 1.8]
    compare_F2_manual_vs_LHAPDF(Q2_list)





