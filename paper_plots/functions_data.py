import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def F1F2_from_d2sigma(Q2_value, vs_what="w"):
    """
    Reads exp_data/InclusiveExpValera_Q2={Q2}.dat (W, eps, d2σ/dWdQ2, err, sys),
    converts to σ_U using Hand flux + Jacobian (W*pi)/(M*E*E'),
    computes F1, F2, plots vs W or x (unchanged behavior), and RETURNS ONLY x,F1,F2.
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

    # columns -> arrays
    W      = df["W"].to_numpy()
    eps    = df["eps"].to_numpy()
    s_dWQ  = df["sigma_dW_dQ2"].to_numpy()
    e_stat = df["err_stat"].to_numpy()
    e_sys  = df["err_sys"].to_numpy()
    e_full = np.sqrt(e_stat**2 + e_sys**2)

    # --- kinematics ---
    nu   = (W**2 + Q2_value - M**2) / (2.0*M)
    E_p  = E_beam - nu                         # E'
    K    = (W**2 - M**2) / (2.0*M)
    x    = Q2_value / (W**2 - M**2 + Q2_value)  # Bjorken x
    rho2 = 1.0 + (4.0*M**2*x**2) / Q2_value

    # --- Hand flux & Jacobian ---
    Gamma_v = (alpha/(2.0*np.pi**2)) * (E_p/E_beam) * (K/Q2_value) * (1.0/(1.0 - eps))
    J = (W * np.pi) / (M * E_beam * E_p)

    # --- σ_U from measured d^2σ/(dW dQ^2) ---
    sigma_U = s_dWQ / (Gamma_v * J)
    sigma_U_err = e_full / (Gamma_v * J)

    # units: nb -> GeV^-2   (change if your input units differ)
    NB_TO_GEV2 = 1.0 / 389_379.365
    sigma_U *= NB_TO_GEV2
    sigma_U_err *= NB_TO_GEV2

    # --- F1, F2 (and their errors for plotting) ---
    four_pi2_alpha = 4.0*np.pi**2*alpha
    denom = (1.0 + eps*RLT)
    sigma_T = sigma_U / denom

    F1 = (K*M / four_pi2_alpha) * sigma_T
    F2 = (K*M / four_pi2_alpha) * (2.0*x/rho2) * ((1.0+RLT)/denom) * sigma_U

    F1_err = (K*M / four_pi2_alpha) * (sigma_U_err/denom)
    F2_err = (K*M / four_pi2_alpha) * (2.0*x/rho2) * ((1.0+RLT)/denom) * sigma_U_err

    # ---- Plot vs W or x (unchanged style) ----
    vw = (vs_what or "w").lower()
    if vw == "x":
        x_axis = x
        xlabel = "x_Bjorken"
        png_suffix = "x"
    else:
        x_axis = W
        xlabel = "W  [GeV]"
        png_suffix = "W"

    order = np.argsort(x_axis)
    x_plot   = x_axis[order]
    F1_plot  = F1[order]
    F1e_plot = F1_err[order]
    F2_plot  = F2[order]
    F2e_plot = F2_err[order]
    
    def x_of_W(W):
        return Q2_value / (W*W - M*M + Q2_value)
    Wmin, Wmax = 1.07, 1.75  # GeV
    
    x_lo_bound = x_of_W(Wmax)   # smaller x (since x decreases with W)
    x_hi_bound = x_of_W(Wmin) 

    out_png = f"F1F2_vs_{png_suffix}_Q2={Q2_value}.png"
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    ax[0].errorbar(x_plot, F1_plot, yerr=F1e_plot, fmt="o", ms=3, lw=1)
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel("F1 (dimensionless)")
    ax[0].set_title(f"F1 vs {xlabel.split()[0]}  (Q²={Q2_value} GeV²)")
    ax[1].errorbar(x_plot, F2_plot, yerr=F2e_plot, fmt="o", ms=3, lw=1)
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel("F2 (dimensionless)")
    ax[1].set_title(f"F2 vs {xlabel.split()[0]}  (Q²={Q2_value} GeV²)")
    for a in ax:
        a.axvline(x_lo_bound, linestyle="--", color="red", linewidth=1)
        a.axvline(x_hi_bound, linestyle="--", color="red", linewidth=1)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    # ---- Return minimal table you need, including F2_err so we can propagate moment error ----
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


def calc_trunc_moment_scan(out_png="trunc_moment_vs_Q2.png"):
    """
    For Q² in [2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699],
    compute the truncated moment over the available x-range using
    calc_trunc_moment_data(Q2_value), then plot Moment±error vs Q² and save PNG.

    Returns a DataFrame with columns: Q2, moment, error (sorted by Q2).
    """
    q2_values = [2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699]
    rows = []

    for q2 in q2_values:
        try:
            res = calc_trunc_moment_data(q2)  # one-row DataFrame with Q2, moment, error
            if isinstance(res, pd.DataFrame) and not res.empty:
                rows.append(res.iloc[0])
        except FileNotFoundError:
            print(f"Warning: missing input file for Q²={q2}; skipping.")
        except Exception as e:
            print(f"Warning: failed at Q²={q2}: {e}")

    if not rows:
        return pd.DataFrame(columns=["Q2", "moment", "error"])

    out = pd.DataFrame(rows)[["Q2", "moment", "error"]].sort_values("Q2").reset_index(drop=True)

    # Plot Moment vs Q² with error bars
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    ax.errorbar(out["Q2"].to_numpy(),
                out["moment"].to_numpy(),
                yerr=out["error"].to_numpy(),
                fmt="o", ms=4, lw=1)
    ax.set_xlabel("Q²  [GeV²]")
    ax.set_ylabel(r"Truncated moment  $\int F_2(x,Q^2)\,dx$  (available x-range)")
    ax.set_title("Truncated moment vs Q²")
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    return out

calc_trunc_moment_scan()
