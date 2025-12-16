import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions_pdf import  get_nlo_pdf_interpolators

from pathlib import Path

import os



def F2_from_xsect(Q2_value, make_plots = True, show_lines = False, vs_what = "w", pdf_set_nlo="CJ15nlo"):
    def x_of_W(W,Q2): return Q2 / (W*W - M*M + Q2)
    E_beam = 10.6
    M = 0.9382720813
    alpha = 1/137.035999084
    four_pi2_alpha = 4.0*np.pi**2*alpha

    # --- read file (headerless) and coerce numerics ---
    in_path = f"exp_data/InclusiveExpValera_Q2={Q2_value}.dat"
    df = pd.read_csv(in_path, sep=r"\s+", header=None,
                     names=["W", "eps", "sigma_dW_dQ2", "err_stat", "err_sys"])
    df = df.apply(pd.to_numeric, errors="coerce").dropna()

    # arrays
    W      = df["W"].to_numpy()
    eps    = df["eps"].to_numpy()
    s_dWQ  = df["sigma_dW_dQ2"].to_numpy()
    e_stat = df["err_stat"].to_numpy()
    e_sys  = df["err_sys"].to_numpy()
    e_full = np.sqrt(e_stat**2 + e_sys**2)
    
    # R_LT
    r_path = f"R_LT_tables_from_Yannick/Wdist_Q2_{Q2_value}_GLOBAL_LT.dat"

    r_df = pd.read_csv(r_path, sep=r"\s+", comment="#", header=None)

    # columns (0-based): 0=W, 5=R_LT, 6=dR_LT
    W_R  = r_df.iloc[:, 0].to_numpy()
    R_LT = r_df.iloc[:, 5].to_numpy()
    dR_LT   = r_df.iloc[:, 6].to_numpy()

    # kinematics
    nu   = (W**2 + Q2_value - M**2) / (2.0*M)
    E_p  = E_beam - nu
    K    = (W**2 - M**2) / (2.0*M)
    x    = x_of_W(W, Q2_value)
    rho2 = 1.0 + (4.0*M**2*x**2) / Q2_value

    # flux & Jacobian
    Gamma_v = (alpha/(2.0*np.pi**2)) * (E_p/E_beam) * (K/Q2_value) * (1.0/(1.0 - eps))
    J = (W * np.pi) / (M * E_beam * E_p)

    # σ_U and units
    sigma_U = s_dWQ / (Gamma_v * J)
    sigma_U_err = e_full / (Gamma_v * J)
    NB_TO_GEV2 = 1.0 / 389_379.365
    sigma_U *= NB_TO_GEV2
    sigma_U_err *= NB_TO_GEV2
    
    pref = (K*M) / four_pi2_alpha  # KM/(4π²α)

    F2 = pref * (2.0*x/rho2) * ((1.0 + R_LT)/(1.0 + eps*R_LT)) * sigma_U
    
    A = pref * (2.0*x/rho2)
    f = (1.0 + R_LT) / (1.0 + eps*R_LT) # Auxillary funcrion f(R) to calculate uncertainty

    df_dR = (1.0 - eps) / (1.0 + eps*R_LT)**2
    f_err = np.abs(df_dR) * dR_LT

    F2_err_from_sigma = A * f * sigma_U_err
    F2_err_from_R     = A * sigma_U * f_err

    F2_err = np.sqrt(F2_err_from_sigma**2 + F2_err_from_R**2)
    
    if make_plots == False:
        return pd.DataFrame({
            "W": W,
            "x": x,
            "F2": F2,
            "F2_err": F2_err
        })
    
    #------------------------------------PDF predictions----------------------------------------
    have_nlo = False
    try:
          _, _, _, _, F2_NLO, F2_NLO_TMC, F2_NLO_TMC_HT, _, _, _, W_nlo_rng = get_nlo_pdf_interpolators(Q2_value,pdf_set=pdf_set_nlo)
          W_nlo_min = float(np.min(W_nlo_rng))
          have_nlo = True
    except Exception:
          pass
      
    # W grid
    wmins = []
    if have_nlo: wmins.append(W_nlo_min)
    W_min_global = max(1.0, min(wmins)) if wmins else 1.0
    W_max_global = float(np.max(W))+0.03
    W_vals = np.linspace(W_min_global, W_max_global, 400)
    
     # Evaluate NLO (LT, TMC only, TMC+HT, HT-only)
    F2_NLO_vals = np.full_like(W_vals, np.nan, dtype=float)
    F2_NLO_TMC_vals = np.full_like(W_vals, np.nan, dtype=float)
    F2_NLO_TMC_HT_vals = np.full_like(W_vals, np.nan, dtype=float)
    if have_nlo:
        m = (W_vals >= W_nlo_min) & (W_vals <= W_max_global)
        try: F2_NLO_vals[m]  = F2_NLO(W_vals[m])
        except Exception: pass
        try: F2_NLO_TMC_vals[m] = F2_NLO_TMC(W_vals[m])
        except Exception: pass
        try: F2_NLO_TMC_HT_vals[m] = F2_NLO_TMC_HT(W_vals[m])
        except Exception: pass
    
    # ---------------------ranges for future integration----------------------------------------
    
    W_min = 1.15 # now corresponds to data range
    Wmax1 = 1.35 # end of 1st resonance region
    Wmin2 = 1.45 # start of 2nd resonance region
    Wmax2 = 1.6 # end of 2nd resonance region
    Wmin3 = 1.61 # CRUTCH for visibility
    Wmax3 = 1.85 # end of 3rd resonance region
    W_max = 2.5 
    if Q2_value == 9.699:
      W_max = 2.25

    xmax = x_of_W(W_min, Q2_value)
 
    x1 = x_of_W(Wmax1, Q2_value) # W = 1.35 GeV
    xmin2 = x_of_W(Wmin2, Q2_value) # W = 1.45 GeV
    x2 = x_of_W(Wmax2, Q2_value) # W = 1.6 GeV
    xmin3 = x_of_W(Wmin3, Q2_value) # W = 1.62 GeV CRUTCH for visibility
    x3 = x_of_W(Wmax3, Q2_value) # W = 1.85 GeV
    
    xmin = x_of_W(W_max, Q2_value) # W = 2.5 GeV (2.25 GeV at highest Q2)

    
        # -----------------------------
    # Plot (exp points with errors and PDF-based predictions)
    # -----------------------------
    if make_plots:
        plt.figure(figsize=(7, 5))
        vs = vs_what.lower().strip()
        if vs in ["w", "W"]:
            order = np.argsort(W)
            x_axis = W[order]
            xlab = r"$W$ [GeV]"
            tag = "W"
        elif vs in ["x", "X"]:
            order = np.argsort(x)  # increasing x
            x_axis = x[order]
            xlab = r"$x_{B}$"
            tag = "x"
        else:
            raise ValueError(f"vs_what must be 'w' or 'x' (got '{vs_what}')")

        F2_plot = F2[order]
        F2e_plot = F2_err[order]

        plt.errorbar(x_axis, F2_plot, yerr=F2e_plot, label="RGA data (V.Klimenko)", color="black", fmt="o", linestyle="none", markersize=2.5, capsize=2)

        if tag == "W" and have_nlo:
            if np.isfinite(F2_NLO_vals).any():
                good = np.isfinite(F2_NLO_vals)
                h_naked, = plt.plot(W_vals[good], F2_NLO_vals[good], label=f"{pdf_set_nlo}: NLO + LT", color="green", ls="dashed", lw=1.3)

            if np.isfinite(F2_NLO_TMC_HT_vals).any():
                good = np.isfinite(F2_NLO_TMC_HT_vals)
                h_bht, = plt.plot(W_vals[good], F2_NLO_TMC_HT_vals[good], label=f"{pdf_set_nlo}: NLO + LT + TMC (OPE) + HT", color="orange", ls="solid", lw=1.3)
         # --- PDF curves on x-axis ---
        if tag == "x" and have_nlo:
            x_pdf = x_of_W(W_vals, Q2_value)

            # NLO + LT
            good = np.isfinite(F2_NLO_vals) & np.isfinite(x_pdf)
            if good.any():
                p = np.argsort(x_pdf[good])  # increasing x
                plt.plot(x_pdf[good][p], F2_NLO_vals[good][p], label=f"{pdf_set_nlo}: NLO + LT", color="green", ls="dashed", lw=1.3)

            # NLO + LT + TMC + HT
            good = np.isfinite(F2_NLO_TMC_HT_vals) & np.isfinite(x_pdf)
            if good.any():
                p = np.argsort(x_pdf[good])
                plt.plot(x_pdf[good][p], F2_NLO_TMC_HT_vals[good][p], label=f"{pdf_set_nlo}: NLO + LT + TMC (OPE) + HT",color="orange", ls="solid", lw=1.3)
        if show_lines:
            if tag == "W":
                ax.axvline(W_min, linestyle="--", linewidth=1, color = "red")
                ax.axvline(Wmax1, linestyle="--", linewidth=1, color = "red")

                ax.axvline(Wmin2, linestyle="--", linewidth=1, color = "green")
                ax.axvline(Wmax2, linestyle="--", linewidth=1, color = "green")

                ax.axvline(Wmin3, linestyle="--", linewidth=1, color = "blue")
                ax.axvline(Wmax3, linestyle="--", linewidth=1, color = "blue")

            if tag == "x":
                ax.axvline(xmax, linestyle="--", linewidth=1, color = "red")
                ax.axvline(x1, linestyle="--", linewidth=1, color = "red")

                ax.axvline(xmin2, linestyle="--", linewidth=1, color = "green")
                ax.axvline(x2, linestyle="--", linewidth=1, color = "green")

                ax.axvline(xmin3, linestyle="--", linewidth=1, color = "blue")
                ax.axvline(x3, linestyle="--", linewidth=1, color = "blue")

        ax = plt.gca()
        plt.xlabel(xlab)
        plt.ylabel(r"$F_2$")
        plt.title(rf"$F_2$ structure function; $Q^2 = {Q2_value}$ GeV$^2$")
        plt.grid(True)
        plt.legend(frameon=False, fontsize=10, loc="best")
        plt.tight_layout()

        out_dir = "F2_from_data_plots"
        os.makedirs(out_dir, exist_ok=True)
        out_png = os.path.join(out_dir, f"F2_Q2={Q2_value}_vs_{tag}.png")
        plt.savefig(out_png, dpi=200)
        plt.close()


#for Q2 in [2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699]: 
#    F2_from_xsect(Q2, make_plots=True, vs_what="w")
#    F2_from_xsect(Q2, make_plots=True, vs_what="x")

 


def calc_trunc_moment_data(Q2_value, region, n=2):
    """
    Truncated Cornwall–Norton moment from data:
        M_n(Q2; region) = ∫_{x_lo}^{x_hi} x^{n-2} F2(x,Q2) dx
    Region is defined via W-bounds, converted to x-bounds at fixed Q2.

    Returns DataFrame with: Q2, region, n, x_lo, x_hi, moment, error
    """

    M = 0.9382720813

    # --- get data (x, F2, dF2) ---
    df = F2_from_xsect(Q2_value, make_plots=False, vs_what="x")
    x  = df["x"].to_numpy(dtype=float)
    F2 = df["F2"].to_numpy(dtype=float)
    dF = df["F2_err"].to_numpy(dtype=float)

    mask = np.isfinite(x) & np.isfinite(F2) & np.isfinite(dF)
    x, F2, dF = x[mask], F2[mask], dF[mask]
    if x.size < 2:
        return pd.DataFrame([{
            "Q2": Q2_value, "region": region, "n": n,
            "x_lo": np.nan, "x_hi": np.nan,
            "moment": 0.0, "error": 0.0
        }])

    # sort by increasing x
    o = np.argsort(x)
    x, F2, dF = x[o], F2[o], dF[o]

    # --- region -> W bounds (edit here if you want different) ---
    W_min_data = 1.15
    Wmax1 = 1.35
    Wmin2 = 1.45
    Wmax2 = 1.60
    Wmin3 = Wmax2
    Wmax3 = 1.85
    W_max = 2.25 if np.isclose(Q2_value, 9.699, atol=1e-3) else 2.50

    reg = str(region).lower().strip()
    region_map = {
        "1": (W_min_data, Wmax1), "r1": (W_min_data, Wmax1), "first": (W_min_data, Wmax1), "1st": (W_min_data, Wmax1),
        "2": (Wmin2, Wmax2),      "r2": (Wmin2, Wmax2),      "second": (Wmin2, Wmax2),      "2nd": (Wmin2, Wmax2),
        "3": (Wmin3, Wmax3),      "r3": (Wmin3, Wmax3),      "third": (Wmin3, Wmax3),       "3rd": (Wmin3, Wmax3),
        "res": (W_min_data, Wmax3), "resonance": (W_min_data, Wmax3),
        "tail": (Wmax3, W_max),
        "all": (W_min_data, W_max), "total": (W_min_data, W_max)
    }
    if reg not in region_map:
        raise ValueError(f"Unknown region='{region}'. Use one of: {sorted(region_map.keys())}")

    W_lo, W_hi = region_map[reg]

    # --- W -> x bounds at this Q2 ---
    def x_of_W(W):
        return Q2_value / (W*W - M*M + Q2_value)

    x1 = x_of_W(W_lo)
    x2 = x_of_W(W_hi)
    x_lo_bound = min(x1, x2)
    x_hi_bound = max(x1, x2)

    # --- intersect with available data x-range ---
    lo = max(x_lo_bound, x[0])
    hi = min(x_hi_bound, x[-1])
    if lo >= hi:
        return pd.DataFrame([{
            "Q2": Q2_value, "region": region, "n": n,
            "x_lo": lo, "x_hi": hi,
            "moment": 0.0, "error": 0.0
        }])

    # --- build segment including interpolated endpoints ---
    F2_lo = np.interp(lo, x, F2); F2_hi = np.interp(hi, x, F2)
    dF_lo = np.interp(lo, x, dF); dF_hi = np.interp(hi, x, dF)

    mid = (x > lo) & (x < hi)
    x_seg  = np.concatenate(([lo], x[mid], [hi]))
    F2_seg = np.concatenate(([F2_lo], F2[mid], [F2_hi]))
    dF_seg = np.concatenate(([dF_lo], dF[mid], [dF_hi]))

    # --- moment integrand and trapezoid integral ---
    w = x_seg**(n - 2)              # CN weight
    y = w * F2_seg
    dy = w * dF_seg

    moment = float(np.trapz(y, x_seg))

    # --- uncorrelated trapezoid error propagation ---
    dx = np.diff(x_seg)
    err2 = np.sum((0.5*dx)**2 * (dy[:-1]**2 + dy[1:]**2))
    error = float(np.sqrt(err2))

    return pd.DataFrame([{
        "Q2": Q2_value, "region": region, "n": n,
        "x_lo": lo, "x_hi": hi,
        "moment": moment, "error": error
    }])








def plot_epsilon_vs_W_grid(
    q2_values=(2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699),
    data_dir="exp_data",
    out_png="epsilon_vs_W_Q2grid.png",
    w_min=1.07,
    w_max=2.50,
):
    """
    Make a 3x3 panel plot of ε (virtual-photon polarization) vs W
    from Klimenko files: {data_dir}/InclusiveExpValera_Q2={Q2}.dat

    Each file is parsed as whitespace-separated with columns:
        W, eps, sigma_dW_dQ2, err_stat, err_sys

    Args:
        q2_values: iterable of Q^2 values (GeV^2) to plot (len should be 9 for a 3x3 grid).
        data_dir: directory where the Klimenko files live.
        out_png : filename for the saved figure.
        w_min, w_max: x-range to display (GeV).

    Returns:
        dict mapping Q2 -> DataFrame with columns ["W","eps"] actually plotted.
    """
    q2_values = list(q2_values)
    if len(q2_values) != 9:
        raise ValueError("Please provide exactly 9 Q^2 values for a 3x3 panel plot.")

    results = {}

    fig, axes = plt.subplots(3, 3, figsize=(11, 9), sharex=True, sharey=True, constrained_layout=True)

    for ax, Q2 in zip(axes.ravel(), q2_values):
        fpath = Path(data_dir) / f"InclusiveExpValera_Q2={Q2}.dat"

        if not fpath.exists():
            ax.text(0.5, 0.5, "missing file", ha="center", va="center", transform=ax.transAxes, color="crimson")
            ax.set_title(f"Q²={Q2:g} GeV²")
            ax.grid(alpha=0.25)
            continue

        # read and coerce numerics
        df = pd.read_csv(
            fpath,
            sep=r"\s+",
            header=None,
            names=["W", "eps", "sigma_dW_dQ2", "err_stat", "err_sys"],
            engine="python",
        )
        df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=["W", "eps"])

        # restrict W-range for display
        df = df[(df["W"] >= w_min) & (df["W"] <= w_max)].copy()
        if df.empty:
            ax.text(0.5, 0.5, "no points in range", ha="center", va="center", transform=ax.transAxes, color="gray")
            ax.set_title(f"Q²={Q2:g} GeV²")
            ax.grid(alpha=0.25)
            continue

        df = df.sort_values("W")
        results[Q2] = df[["W", "eps"]].reset_index(drop=True)

        ax.plot(df["W"].to_numpy(), df["eps"].to_numpy(), marker="o", ms=3, lw=1)
        ax.set_title(f"Q²={Q2:g} GeV²")
        ax.grid(alpha=0.25)

    # common labels and limits
    for ax in axes[-1]:
        ax.set_xlabel("W  [GeV]")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\varepsilon$")

    axes[0, 0].set_ylim(0.0, 1.05)
    axes[0, 0].set_xlim(w_min, w_max)

    fig.suptitle(r"$\varepsilon(W)$ from Klimenko data", y=1.02, fontsize=14)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    return results



