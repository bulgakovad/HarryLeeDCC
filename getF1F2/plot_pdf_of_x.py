#!/usr/bin/env python3
import os
import math
import lhapdf
import numpy as np
import matplotlib.pyplot as plt

# ------------------ Helpers ------------------
def q_separate(pdf, pid, x, Q2, times_x=True):
    """Return q or x*q for |pid| (1=d,2=u,3=s,4=c)."""
    xf_q    = pdf.xfxQ2(pid, x, Q2)
    return xf_q if times_x else (xf_q / x if x > 0 else 0.0)

def q_sum(pdf, pid_abs, x, Q2, times_x=True):
    """Return (q + qbar) or x*(q + qbar) for |pid| (1=d,2=u,3=s,4=c). Use negative PID for antiquark"""
    xf_q    = pdf.xfxQ2(+pid_abs, x, Q2)
    xf_qbar = pdf.xfxQ2(-pid_abs, x, Q2)
    xf_sum  = xf_q + xf_qbar
    return xf_sum if times_x else (xf_sum / x if x > 0 else 0.0)

def make_x_grid(xmin, xmax, npts, logx=True):
    if logx:
        return np.logspace(math.log10(xmin), math.log10(xmax), npts)
    return np.linspace(xmin, xmax, npts)

def plot_pdf_of_x(pdf_set,sum_or_sep, Q2_list):

#-----------------------Params------------------------------------------#

    pdf_set   = pdf_set         # e.g. "CT18LO", "CJ15lo"
    member    = 0                 # PDF member index
    Q2_list   = Q2_list # GeV^2 values to plot
    sum_or_sep = sum_or_sep    # "sum" or "separate" for (q+qbar) or separate q and qbar

    # x grid
    xmin      = 1e-4
    xmax      = 0.9
    npts      = 300
    linear_x  = False              # False -> log x axis; True -> linear

    # What to plot
    times_x   = True              # True: plot x*(q+qbar); False: (q+qbar)

    # Output
    outdir    = f"Output/PDF_curves_{pdf_set}_{sum_or_sep}"
    os.makedirs(outdir, exist_ok=True)

#-----------------------Params------------------------------------------#

    pdf = lhapdf.mkPDF(pdf_set, member)

    # Validity ranges (note: xMin/xMax are methods; q2Min/q2Max are properties)
    xmin_valid, xmax_valid = pdf.xMin, pdf.xMax
    q2min, q2max = pdf.q2Min, pdf.q2Max
    print(f"[{pdf_set} member {member}] x in [{xmin_valid}, {xmax_valid}],  Q^2 in [{q2min}, {q2max}] GeV^2")

    xgrid = make_x_grid(xmin, xmax, npts, logx=(not linear_x))
    if sum_or_sep == "sum":
        label_suffix = "x·(q+q̄)" if times_x else "(q+q̄)"
    elif sum_or_sep == "separate":
        label_suffix = "x·q and x·q̄" if times_x else "q and q̄"

    for Q2 in Q2_list:
        if not (q2min <= Q2 <= q2max):
            print(f"WARNING: Q^2={Q2} is outside nominal range; values will freeze.")
            
        # --- NEW: gluon curve ---
        if times_x:
            g_arr = np.array([pdf.xfxQ2(21, x, Q2) for x in xgrid])      # x*g(x,Q2)
        else:
            g_arr = np.array([pdf.xfxQ2(21, x, Q2)/x if x>0 else 0.0 for x in xgrid])  # g(x,Q2)

        if sum_or_sep == "sum":
            # Compute arrays
            u_sum = np.array([q_sum(pdf, 2, x, Q2, times_x=times_x) for x in xgrid])
            d_sum = np.array([q_sum(pdf, 1, x, Q2, times_x=times_x) for x in xgrid])
            s_sum = np.array([q_sum(pdf, 3, x, Q2, times_x=times_x) for x in xgrid])
            c_sum = np.array([q_sum(pdf, 4, x, Q2, times_x=times_x) for x in xgrid])
        elif sum_or_sep == "separate":
            # Compute arrays
            u = np.array([q_separate(pdf, +2, x, Q2, times_x=times_x) for x in xgrid])
            d = np.array([q_separate(pdf, +1, x, Q2, times_x=times_x) for x in xgrid])
            s = np.array([q_separate(pdf, +3, x, Q2, times_x=times_x) for x in xgrid])
            c = np.array([q_separate(pdf, +4, x, Q2, times_x=times_x) for x in xgrid])
            u_bar = np.array([q_separate(pdf, -2, x, Q2, times_x=times_x) for x in xgrid])
            d_bar = np.array([q_separate(pdf, -1, x, Q2, times_x=times_x) for x in xgrid])
            s_bar = np.array([q_separate(pdf, -3, x, Q2, times_x=times_x) for x in xgrid])
            c_bar = np.array([q_separate(pdf, -4, x, Q2, times_x=times_x) for x in xgrid])

        fig, ax = plt.subplots(figsize=(8.0, 5.2))
        if sum_or_sep == "sum":
            # New figure per Q2
            ax.plot(xgrid, u_sum, label="u+ū")
            ax.plot(xgrid, d_sum, label="d+d̄")
            ax.plot(xgrid, s_sum, label="s+s̄")
            ax.plot(xgrid, c_sum, label="c+c̄")
            #ax.plot(xgrid, g_arr, linestyle="--", label="g")   # <-- NEW
        elif sum_or_sep == "separate":
            # New figure per Q2
            ax.plot(xgrid, u, label="u")
            ax.plot(xgrid, d, label="d")
            ax.plot(xgrid, s, label="s")
            ax.plot(xgrid, c, label="c")
            ax.plot(xgrid, u_bar, linestyle="--", label="ū")
            ax.plot(xgrid, d_bar, linestyle="--", label="d̄")
            ax.plot(xgrid, s_bar, linestyle="--", label="s̄")
            ax.plot(xgrid, c_bar, linestyle="--", label="c̄")
            #ax.plot(xgrid, g_arr, linestyle=":", label="g")   # <-- NEW

        ax.set_xscale("linear" if linear_x else "log")
        ax.set_xlabel("x")
        ax.set_ylabel(label_suffix)
        ax.set_title(f"{pdf_set}: {label_suffix} vs x for Q² = {Q2:g} GeV²")
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.legend(ncol=1, fontsize=9)
        fig.tight_layout()

        save_name = f"{pdf_set}_q_{sum_or_sep}_{'x' if times_x else 'noX'}_Q2={Q2:g}.png"
        save_path = os.path.join(outdir, save_name)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {save_path}")


plot_pdf_of_x("CT18NLO", "sum", [2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699])
plot_pdf_of_x("CJ15nlo", "sum", [2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699])
plot_pdf_of_x("CT18LO" , "sum", [2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699])
plot_pdf_of_x("CJ15lo" , "sum", [2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699])

plot_pdf_of_x("CT18NLO", "separate", [2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699])
plot_pdf_of_x("CJ15nlo", "separate", [2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699])
plot_pdf_of_x("CT18LO" , "separate", [2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699])
plot_pdf_of_x("CJ15lo" , "separate", [2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699])