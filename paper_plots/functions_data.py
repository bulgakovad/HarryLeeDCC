import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path


def F1F2_from_d2sigma(Q2_value, vs_what="w"):
    """
    Reads exp_data/InclusiveExpValera_Q2={Q2}.dat (W, eps, d2σ/dWdQ2, err, sys),
    converts to σ_U using Hand flux + Jacobian (W*pi)/(M*E*E'),
    computes F1, F2, plots vs W or x (unchanged behavior), and RETURNS ONLY x,F1,F2.
    Also overlays experimental F2(x) with errors from:
        strfun_F1F2_data/F2_vs_x_Q2={Q2}.dat
    when vs_what == "x".
    """
    # constants
    E_beam = 10.6
    RLT = 0.18
    M = 0.9382720813
    alpha = 1/137.035999084

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

    # kinematics
    nu   = (W**2 + Q2_value - M**2) / (2.0*M)
    E_p  = E_beam - nu
    K    = (W**2 - M**2) / (2.0*M)
    x    = Q2_value / (W**2 - M**2 + Q2_value)
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

    # F1, F2 (+ errs)
    four_pi2_alpha = 4.0*np.pi**2*alpha
    denom = (1.0 + eps*RLT)
    sigma_T = sigma_U / denom

    F1 = (K*M / four_pi2_alpha) * sigma_T
    F2 = (K*M / four_pi2_alpha) * (2.0*x/rho2) * ((1.0+RLT)/denom) * sigma_U

    F1_err = (K*M / four_pi2_alpha) * (sigma_U_err/denom)
    F2_err = (K*M / four_pi2_alpha) * (2.0*x/rho2) * ((1.0+RLT)/denom) * sigma_U_err

    # plotting axis choice
    vw = (vs_what or "w").lower()
    if vw == "x":
        x_axis = x; xlabel = "x_Bjorken"; png_suffix = "x"
    else:
        x_axis = W; xlabel = "W  [GeV]"; png_suffix = "W"

    order = np.argsort(x_axis)
    x_plot, F1_plot, F1e_plot = x_axis[order], F1[order], F1_err[order]
    F2_plot, F2e_plot = F2[order], F2_err[order]

    def x_of_W(W_): return Q2_value / (W_*W_ - M*M + Q2_value)
    Wmin, Wmax = 1.07, 1.75
    x_lo_bound = x_of_W(Wmax)  # smaller x
    x_hi_bound = x_of_W(Wmin)  # larger x

    out_png = f"F1F2_vs_{png_suffix}_Q2={Q2_value}.png"
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

    # F1 panel
    ax[0].errorbar(x_plot, F1_plot, yerr=F1e_plot, color="black", fmt="o", ms=3, lw=1)
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel("F1 (dimensionless)")
    ax[0].set_title(f"F1 vs {xlabel.split()[0]}  (Q²={Q2_value} GeV²)")

    # F2 panel (Klimenko/your calc) — always plotted
    ax[1].errorbar(x_plot, F2_plot, yerr=F2e_plot, fmt="o", ms=3, lw=1,
                   color="black", label="RGA data (V.Klimenko)")

    # --- overlay strfun F2(x) if present; otherwise silently skip ---
    if vw == "x":
        from pathlib import Path
        exp_path = Path(f"strfun_F1F2_data/F2_vs_x_Q2={Q2_value}.dat")
        exp = None
        if exp_path.exists():
            try:
                exp = pd.read_csv(exp_path, sep=r"\s+", header=0, comment="#", engine="python")
                exp.columns = [c.strip().lower() for c in exp.columns]
                if "quantity" in exp.columns:
                    exp = exp.rename(columns={"quantity": "f2", "uncertainty": "f2_err"})
                else:
                    exp = exp.rename(columns={exp.columns[1]: "f2", exp.columns[2]: "f2_err"})
            except Exception:
                try:
                    exp = pd.read_csv(exp_path, sep=r"\s+", header=None, comment="#", engine="python",
                                      names=["x", "f2", "f2_err"])
                except Exception:
                    exp = None

        if exp is not None:
            for c in ["x", "f2", "f2_err"]:
                exp[c] = pd.to_numeric(exp[c], errors="coerce")
            exp = exp.dropna(subset=["x", "f2", "f2_err"]).sort_values("x")
            if not exp.empty:
                ax[1].errorbar(exp["x"].to_numpy(),
                               exp["f2"].to_numpy(),
                               yerr=exp["f2_err"].to_numpy(),
                               fmt="s", ms=2, lw=1, alpha=0.9, color="green",
                               label="CLAS+World data interpolation")

        # vertical bounds for x-plots only
        for a in ax:
            a.axvline(x_lo_bound, linestyle="--", color="red", linewidth=1)
            a.axvline(x_hi_bound, linestyle="--", color="red", linewidth=1)

    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel("F2 (dimensionless)")
    ax[1].set_title(f"F2 vs {xlabel.split()[0]}  (Q²={Q2_value} GeV²)")
    ax[1].legend(loc="best", frameon=False)

    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    # return minimal table (with F2_err for moment error propagation)
    out = pd.DataFrame({"x": x, "F1": F1, "F2": F2, "F2_err": F2_err})
    return out.sort_values("x").reset_index(drop=True)


def calc_trunc_moment_data(Q2_value, Wmin=1.07, Wmax=1.75):
    """
    Integrate F2 over x corresponding to W in [Wmin, Wmax] at fixed Q^2.
    Calls F1F2_from_d2sigma(Q2_value, vs_what="x") (which also makes the x-plot).
    Returns a one-row DataFrame: Q2, moment, error, where
      moment = ∫_{x(Wmax)}^{x(Wmin)} F2(x,Q^2) dx
      error  = 0.5 * [ ∫ (F2+σ) dx  -  ∫ (F2-σ) dx ]
    """
    # Get x, F2, and its pointwise uncertainty
    df = F1F2_from_d2sigma(Q2_value, vs_what="x")

    # Convert W-bounds -> x-bounds at this Q^2
    M = 0.9382720813
    def x_of_W(W):
        return Q2_value / (W*W - M*M + Q2_value)

    x_lo_bound = x_of_W(Wmax)   # smaller x (since x decreases with W)
    x_hi_bound = x_of_W(Wmin)   # larger x

    # Extract and sort data
    x  = df["x"].to_numpy()
    F2 = df["F2"].to_numpy()
    dF = df["F2_err"].to_numpy()

    mask = np.isfinite(x) & np.isfinite(F2) & np.isfinite(dF)
    x, F2, dF = x[mask], F2[mask], dF[mask]
    if x.size < 2:
        return pd.DataFrame([{"Q2": Q2_value, "moment": 0.0, "error": 0.0}])

    o = np.argsort(x)
    x, F2, dF = x[o], F2[o], dF[o]

    # Intersect requested [x_lo_bound, x_hi_bound] with available data range
    lo = max(x_lo_bound, x[0])
    hi = min(x_hi_bound, x[-1])
    if lo >= hi:
        return pd.DataFrame([{"Q2": Q2_value, "moment": 0.0, "error": 0.0}])

    # Interpolate endpoints and gather interior points
    F2_lo = np.interp(lo, x, F2); F2_hi = np.interp(hi, x, F2)
    dF_lo = np.interp(lo, x, dF); dF_hi = np.interp(hi, x, dF)
    mid = (x > lo) & (x < hi)

    x_seg  = np.concatenate(([lo], x[mid], [hi]))
    F2_seg = np.concatenate(([F2_lo], F2[mid], [F2_hi]))
    dF_seg = np.concatenate(([dF_lo], dF[mid], [dF_hi]))

    # Central value and symmetric error from (F2±dF) envelopes
    m_c  = float(np.trapz(F2_seg, x_seg))
    m_hi = float(np.trapz(F2_seg + dF_seg, x_seg))
    m_lo = float(np.trapz(F2_seg - dF_seg, x_seg))
    err  = 0.5 * (m_hi - m_lo)

    return pd.DataFrame([{"Q2": Q2_value, "moment": m_c, "error": err}])


def calc_trunc_moment_scan(out_png="trunc_moment_vs_Q2.png",
                           theory_path="../getF1F2/Output/F2_trunc_cj15.txt"):
    """
    For Q² in [2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699],
    compute the truncated moment over x corresponding to W∈[1.07,1.75] using
    calc_trunc_moment_data(Q2_value). Plot Moment±error vs Q² and overlay CJ15
    theory points (5th col = TMC+HT, 9th col = TMC-only). Save PNG.

    Returns a DataFrame with columns: Q2, moment, error, CJ15_TMC_HT, CJ15_TMC
    (sorted by Q2).
    """
    q2_values = [2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699]
    rows = []

    for q2 in q2_values:
        try:
            # one-row DataFrame with Q2, moment, error
            res = calc_trunc_moment_data(q2)
            if isinstance(res, pd.DataFrame) and not res.empty:
                rows.append(res.iloc[0])
        except FileNotFoundError:
            print(f"Warning: missing input file for Q²={q2}; skipping.")
        except Exception as e:
            print(f"Warning: failed at Q²={q2}: {e}")

    if not rows:
        return pd.DataFrame(columns=["Q2", "moment", "error", "CJ15_TMC_HT", "CJ15_TMC"])

    out = pd.DataFrame(rows)[["Q2", "moment", "error"]].sort_values("Q2").reset_index(drop=True)

    # ---- Read CJ15 theory (first col = Q2, 5th = TMC+HT, 9th = TMC-only) ----
    CJ15_TMC_HT = None
    CJ15_TMC = None
    try:
        th = pd.read_csv(theory_path, sep=r"\s+", header=None, comment="#", engine="python")
        th = th.apply(pd.to_numeric, errors="coerce").dropna(how="any")
        # select columns by 0-based index: 0 (Q2), 4 (5th), 8 (9th)
        th = th[[0, 4, 8]].copy()
        th.columns = ["Q2", "CJ15_TMC_HT", "CJ15_TMC"]

        # robust join by rounded Q2 to avoid float mismatches
        out["Q2_round"] = out["Q2"].round(3)
        th["Q2_round"] = th["Q2"].round(3)
        out = out.merge(th[["Q2_round", "CJ15_TMC_HT", "CJ15_TMC"]],
                        on="Q2_round", how="left").drop(columns=["Q2_round"])
    except FileNotFoundError:
        print(f"Warning: theory file not found at {theory_path}; plotting data only.")
    except Exception as e:
        print(f"Warning: failed to parse theory file: {e}")

    # ---- Plot Moment vs Q² with error bars + theory overlays ----
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    ax.errorbar(out["Q2"].to_numpy(),
                out["moment"].to_numpy(),
                yerr=out["error"].to_numpy(),
                fmt="o", ms=2, lw=1, label="RGA data (V.Klimenko)", color="black")

    # overlay if available
    if "CJ15_TMC_HT" in out.columns and out["CJ15_TMC_HT"].notna().any():
        ax.plot(out["Q2"].to_numpy(),
                out["CJ15_TMC_HT"].to_numpy(),
                marker="s", linestyle="-", ms=2, lw=1, color="red", label="NLO CJ15 (TMC+HT)")
    if "CJ15_TMC" in out.columns and out["CJ15_TMC"].notna().any():
        ax.plot(out["Q2"].to_numpy(),
                out["CJ15_TMC"].to_numpy(),
                marker="^", linestyle="dotted", ms=2, lw=1, color="green", label="NLO CJ15 (TMC only)")

    ax.set_xlabel("Q²  [GeV²]")
    ax.set_ylabel(r"$M_2$ =   $\int_{x(W\in[1.07,1.75])} F_2(x,Q^2)\,dx$")
    ax.set_title("Truncated moment vs Q² in full resonance region")
    ax.legend(frameon=False)

    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    return out

def F1F2_compare_RLT(Q2_value, vs_what="w", rlt_list=(0.007, 0.18, 0.7 )):
    """
    Compare F1, F2 computed with different R_LT values on one plot.
    Keeps the strfun overlay (if present) and saves:
        F1F2_vs_{W|x}_RLTscan_Q2={Q2}.png

    Returns: dict mapping RLT -> DataFrame with columns ["x","F1","F2","F1_err","F2_err"]
             (each sorted by x). Also returns the axis variable used.
    """
    # ===== constants =====
    E_beam = 10.6
    M = 0.9382720813
    alpha = 1/137.035999084
    four_pi2_alpha = 4.0*np.pi**2*alpha

    # ===== read Klimenko d2σ/(dW dQ2) input =====
    in_path = f"exp_data/InclusiveExpValera_Q2={Q2_value}.dat"
    df = pd.read_csv(in_path, sep=r"\s+", header=None,
                     names=["W", "eps", "sigma_dW_dQ2", "err_stat", "err_sys"])
    df = df.apply(pd.to_numeric, errors="coerce").dropna()

    # arrays
    W      = df["W"].to_numpy()
    eps    = df["eps"].to_numpy()
    s_dWQ  = df["sigma_dW_dQ2"].to_numpy()
    e_full = np.sqrt(df["err_stat"].to_numpy()**2 + df["err_sys"].to_numpy()**2)

    # ===== kinematics =====
    nu   = (W**2 + Q2_value - M**2) / (2.0*M)
    E_p  = E_beam - nu
    K    = (W**2 - M**2) / (2.0*M)
    x    = Q2_value / (W**2 - M**2 + Q2_value)
    rho2 = 1.0 + (4.0*M**2*x**2) / Q2_value

    # ===== Hand flux & Jacobian =====
    Gamma_v = (alpha/(2.0*np.pi**2)) * (E_p/E_beam) * (K/Q2_value) * (1.0/(1.0 - eps))
    J = (W * np.pi) / (M * E_beam * E_p)

    # ===== σ_U and units (assumes nb input) =====
    sigma_U = s_dWQ / (Gamma_v * J)
    sigma_U_err = e_full / (Gamma_v * J)
    NB_TO_GEV2 = 1.0 / 389_379.365
    sigma_U *= NB_TO_GEV2
    sigma_U_err *= NB_TO_GEV2

    # ===== choose plotting axis =====
    vw = (vs_what or "w").lower()
    if vw == "x":
        axis = x
        xlabel = "x_Bjorken"
        png_suffix = "x"
        # bounds for W ∈ [1.07, 1.75]
        def x_of_W(Wv): return Q2_value / (Wv*Wv - M*M + Q2_value)
        x_lo_bound = x_of_W(1.75)
        x_hi_bound = x_of_W(1.07)
    else:
        axis = W
        xlabel = "W  [GeV]"
        png_suffix = "W"
        x_lo_bound = x_hi_bound = None  # not used

    order = np.argsort(axis)
    axis_sorted = axis[order]

    # ===== compute F1, F2 for each RLT =====
    results = {}
    markers = ["o", "^", "s"]  # cycle markers for clarity
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)

    for i, RLT in enumerate(rlt_list):
        denom = (1.0 + eps*RLT)
        sigma_T = sigma_U / denom

        F1 = (K*M / four_pi2_alpha) * sigma_T
        F2 = (K*M / four_pi2_alpha) * (2.0*x/rho2) * ((1.0+RLT)/denom) * sigma_U

        F1_err = (K*M / four_pi2_alpha) * (sigma_U_err/denom)
        F2_err = (K*M / four_pi2_alpha) * (2.0*x/rho2) * ((1.0+RLT)/denom) * sigma_U_err

        # sort by chosen axis for plotting
        F1p, F1ep = F1[order], F1_err[order]
        F2p, F2ep = F2[order], F2_err[order]

        # store a tidy table sorted by x for return
        results[RLT] = pd.DataFrame({
            "x": x, "F1": F1, "F2": F2, "F1_err": F1_err, "F2_err": F2_err
        }).sort_values("x").reset_index(drop=True)

        lab = f"RGA (V. Klimenko) data with R_LT={RLT:.3f}"
        mk = markers[i % len(markers)]

        ax[0].errorbar(axis_sorted, F1p, yerr=F1ep, fmt=mk, ms=1, lw=1, label=lab)
        ax[1].errorbar(axis_sorted, F2p, yerr=F2ep, fmt=mk, ms=1, lw=1, label=lab)

    # ===== overlay strfun F2 points if available =====
    if vw == "x":
        exp_path = Path(f"strfun_F1F2_data/F2_vs_x_Q2={Q2_value}.dat")
        if exp_path.exists():
            try:
                exp = pd.read_csv(exp_path, sep=r"\s+", header=0, comment="#", engine="python")
                exp.columns = [c.strip().lower() for c in exp.columns]
                if "quantity" in exp.columns:
                    exp = exp.rename(columns={"quantity": "f2", "uncertainty": "f2_err"})
                else:
                    exp = exp.rename(columns={exp.columns[1]: "f2", exp.columns[2]: "f2_err"})
            except Exception:
                try:
                    exp = pd.read_csv(exp_path, sep=r"\s+", header=None, comment="#", engine="python",
                                      names=["x", "f2", "f2_err"])
                except Exception:
                    exp = None

            if exp is not None:
                for c in ["x", "f2", "f2_err"]:
                    exp[c] = pd.to_numeric(exp[c], errors="coerce")
                exp = exp.dropna(subset=["x", "f2", "f2_err"]).sort_values("x")
                if not exp.empty:
                    ax[1].errorbar(exp["x"].to_numpy(),
                                   exp["f2"].to_numpy(),
                                   yerr=exp["f2_err"].to_numpy(),
                                   fmt="D", ms=1, lw=1, label="strfun F2 (exp)")

    else:
        # convert exp x -> W for W plots
        exp_path = Path(f"strfun_F1F2_data/F2_vs_x_Q2={Q2_value}.dat")
        if exp_path.exists():
            try:
                exp = pd.read_csv(exp_path, sep=r"\s+", header=0, comment="#", engine="python")
                exp.columns = [c.strip().lower() for c in exp.columns]
                if "quantity" in exp.columns:
                    exp = exp.rename(columns={"quantity": "f2", "uncertainty": "f2_err"})
                else:
                    exp = exp.rename(columns={exp.columns[1]: "f2", exp.columns[2]: "f2_err"})
                for c in ["x", "f2", "f2_err"]:
                    exp[c] = pd.to_numeric(exp[c], errors="coerce")
                exp = exp.dropna(subset=["x", "f2", "f2_err"]).sort_values("x")
                if not exp.empty:
                    W_exp = np.sqrt(M*M + Q2_value*(1.0/exp["x"].to_numpy() - 1.0))
                    ax[1].errorbar(W_exp,
                                   exp["f2"].to_numpy(),
                                   yerr=exp["f2_err"].to_numpy(),
                                   fmt="D", ms=1, lw=1, label="CLAS+World data interpolation")
            except Exception:
                pass

    # ===== cosmetics, bounds, save =====
    ax[0].set_xlabel(xlabel); ax[1].set_xlabel(xlabel)
    ax[0].set_ylabel("F1 (dimensionless)")
    ax[1].set_ylabel("F2 (dimensionless)")
    ax[0].set_title(f"F1 vs {xlabel.split()[0]}  (Q²={Q2_value} GeV²)")
    ax[1].set_title(f"F2 vs {xlabel.split()[0]}  (Q²={Q2_value} GeV²)")

    if vw == "x":
        for a in ax:
            a.axvline(x_lo_bound, linestyle="--", linewidth=1)
            a.axvline(x_hi_bound, linestyle="--", linewidth=1)

    ax[0].legend(frameon=False, fontsize=9)
    ax[1].legend(frameon=False, fontsize=9)

    out_png = f"F1F2_vs_{png_suffix}_RLTscan_Q2={Q2_value}.png"
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    return results, (axis_sorted, xlabel)

for q2 in [2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699]:
    F1F2_compare_RLT(q2, vs_what="x")
 

