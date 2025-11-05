import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path



def F1F2_from_d2sigma(Q2_value, vs_what="w"):
    """
    Reads exp_data/InclusiveExpValera_Q2={Q2}.dat (W, eps, d2σ/dWdQ2, err, sys),
    converts to σ_U using Hand flux + Jacobian (W*pi)/(M*E*E'),
    computes F1, F2, and plots ONLY F2 vs W or x.

    Uses Osipenko parametrization R(W,Q^2) in the resonance region (W≤2.5)
    instead of a constant R_LT.

    Overlays PDF predictions (LO LT, NLO LT, NLO TMC, NLO TMC+HT) but, for the
    PDF curves only, restricts their W-domain to:
        W ∈ [1.07, 2.5] GeV   (or [1.07, 2.25] if Q²≈9.699)
    When plotting vs x, the PDF curves are shown for x corresponding to that W-range,
    i.e. up to x(W=1.07).

    Returns a DataFrame with columns: x, F1, F2, F2_err (sorted by x).
    """
    # --- imports / constants ---
    from for_RLT_calc import R_osipenko

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

    # -------- Use Osipenko R(W,Q^2) instead of a constant R_LT --------
    R_vec = R_osipenko(W, Q2_value)  # vector over W; NaN outside its validity
    valid = np.isfinite(R_vec)       # typically W <= 2.5
    
    R_mid = 0.2

    # allocate NaNs then fill on valid points
    denom = np.full_like(W, np.nan, dtype=float)
    denom[valid] = 1.0 + eps[valid] * R_vec[valid]
    #denom[valid] = 1.0 + eps[valid] * R_mid

    sigma_T = np.full_like(W, np.nan, dtype=float)
    sigma_T[valid] = sigma_U[valid] / denom[valid]

    F1 = np.full_like(W, np.nan, dtype=float)
    F2 = np.full_like(W, np.nan, dtype=float)
    F1_err = np.full_like(W, np.nan, dtype=float)
    F2_err = np.full_like(W, np.nan, dtype=float)

    F1[valid]     = (K[valid]*M / four_pi2_alpha) * sigma_T[valid]
    F2[valid]     = (K[valid]*M / four_pi2_alpha) * (2.0*x[valid]/rho2[valid]) * ((1.0 + R_vec[valid])/denom[valid]) * sigma_U[valid]
    F1_err[valid] = (K[valid]*M / four_pi2_alpha) * (sigma_U_err[valid]/denom[valid])
    F2_err[valid] = (K[valid]*M / four_pi2_alpha) * (2.0*x[valid]/rho2[valid]) * ((1.0 + R_vec[valid])/denom[valid]) * sigma_U_err[valid]
    # -------------------------------------------------------------------

    # ---- plotting cutoff in W (PLOTTING ONLY) ----
    W_MIN = 1.07
    W_MAX = 2.25 if np.isclose(Q2_value, 9.699) else 2.5
    mask_W = (W >= W_MIN) & (W <= W_MAX) & np.isfinite(F2) & np.isfinite(F2_err)

    # masked copies for plotting only
    W_p   = W[mask_W]
    x_p   = x[mask_W]
    F2_p  = F2[mask_W]
    F2e_p = F2_err[mask_W]

    # plotting axis choice (use masked arrays)
    vw = (vs_what or "w").lower()
    if vw == "x":
        x_axis = x_p; xlabel = "x_Bjorken"; png_suffix = "x"
    else:
        x_axis = W_p; xlabel = "W  [GeV]"; png_suffix = "W"

    order = np.argsort(x_axis)
    x_plot   = x_axis[order]
    F2_plot  = F2_p[order]
    F2e_plot = F2e_p[order]

    # bounds (for optional vertical lines on x-plot)
    def x_of_W(W_): return Q2_value / (W_*W_ - M*M + Q2_value)
    Wmin_bounds, Wmax_bounds = 1.15, 1.75
    x_lo_bound = x_of_W(Wmax_bounds)  # smaller x
    x_hi_bound = x_of_W(Wmin_bounds)  # larger x

    out_png = f"F2_vs_{png_suffix}_Q2={Q2_value}.png"
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)

    # F2 panel (Klimenko/your calc)
    ax.errorbar(x_plot, F2_plot, yerr=F2e_plot, fmt="o", ms=3, lw=1,
                color="black", label="RGA data (V.Klimenko)")

    # ---------- PDF-based predictions overlay (respect W ∈ [1.07, W_MAX]) ----------
    try:
        from functions_pdf import get_pdf_interpolators_with_error, get_nlo_pdf_interpolators

        F1_LO_W, F2_LO_W, _, _, W_lo = get_pdf_interpolators_with_error(Q2_value, central_iset=400)
        (F1_NLO_LT, F1_NLO_TMC, F1_NLO_TMC_alt, F1_NLO_TMC_HT, F1_NLO_HT,
         F2_NLO_LT, F2_NLO_TMC, F2_NLO_TMC_HT, F2_NLO_HT, W_nlo) = get_nlo_pdf_interpolators(Q2_value)

        W_union = np.unique(np.concatenate([W_lo, W_nlo, W]))

        if vw == "x":
            W_mask = (W_union >= W_MIN) & (W_union <= W_MAX)
            W_eval = W_union[W_mask]

            denomW = (W_eval**2 - M**2 + Q2_value)
            x_pdf = Q2_value / denomW
            mask = np.isfinite(x_pdf) & (x_pdf > 0)
            x_pdf = x_pdf[mask]; W_eval = W_eval[mask]
            o = np.argsort(x_pdf)
            x_pdf = x_pdf[o]; W_eval = W_eval[o]

            ax.plot(x_pdf, F2_LO_W(W_eval),             ':',  color="blue",   lw=1.2, label="LO LT (CJ15)")
            ax.plot(x_pdf, F2_NLO_LT(W_eval),           '-.', color="purple", lw=1.2, label="NLO LT (CJ15)")
            ax.plot(x_pdf, F2_NLO_HT(W_eval),       '-',  color="green", lw=1.2, label="NLO LT + HT (CJ15)")
            #ax.plot(x_pdf, F2_NLO_TMC(W_eval),          '--', color="green",  lw=1.2, label="PDF NLO TMC(OPE) only")
            ax.plot(x_pdf, F2_NLO_TMC_HT(W_eval),       '-',  color="orange", lw=1.2, label="NLO LT + HT + TMC (CJ15)")
            
        else:
            W_eval = np.sort(W_union[(W_union >= W_MIN) & (W_union <= W_MAX)])
            ax.plot(W_eval, F2_LO_W(W_eval),             ':',  color="blue",   lw=1.2, label="LO LT (CJ15)")
            ax.plot(W_eval, F2_NLO_LT(W_eval),           '-.', color="purple", lw=1.2, label="NLO LT (CJ15)")
            ax.plot(W_eval, F2_NLO_HT(W_eval),       '-',  color="green", lw=1.2, label="NLO LT + HT (CJ15)")
            #ax.plot(W_eval, F2_NLO_TMC(W_eval),          '--', color="green",  lw=1.2, label="PDF NLO TMC(OPE) only")
            ax.plot(W_eval, F2_NLO_TMC_HT(W_eval),       '-',  color="orange", lw=1.2, label="NLO LT + HT + TMC (CJ15)")
            
    except Exception:
        pass
    # ----------------------------------------------------------------

    # cosmetics
    ax.set_xlabel(xlabel)
    ax.set_ylabel("F2 (dimensionless)")
    ax.set_title(f"F2 vs {xlabel.split()[0]}  (Q²={Q2_value} GeV²)")
    if vw == "x":
        ax.axvline(x_lo_bound, linestyle="--", color="red", linewidth=1)
        ax.axvline(x_hi_bound, linestyle="--", color="red", linewidth=1)
    ax.legend(loc="best", frameon=False)

    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    # return minimal table
    out = pd.DataFrame({"x": x, "F1": F1, "F2": F2, "F2_err": F2_err})
    return out.sort_values("x").reset_index(drop=True)



def F1F2_compare_RLT(Q2_value, vs_what="w", rlt_list=(0.05, 0.20, 0.35 )):
    """
    Compare F1, F2 computed with different R_LT values on one plot.
    Keeps the strfun overlay (if present) and saves:
        F1F2_vs_{W|x}_RLTscan_Q2={Q2}.png

    Returns: dict mapping key -> DataFrame with columns ["x","F1","F2","F1_err","F2_err"]
             where keys are each numeric RLT in rlt_list and "RLT_osip" for the
             Osipenko-based R(W,Q2). Also returns (axis_sorted, xlabel).
    """
    from pathlib import Path
    from for_RLT_calc import R_osipenko  # NEW

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

    # ===== compute F1, F2 for each RLT in the scan =====
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

        lab = f"RGA (V. Klimenko) data w/ R_LT={RLT:.3f}"
        mk = markers[i % len(markers)]

        ax[0].errorbar(axis_sorted, F1p, yerr=F1ep, fmt=mk, ms=1, lw=1, label=lab)
        ax[1].errorbar(axis_sorted, F2p, yerr=F2ep, fmt=mk, ms=1, lw=1, label=lab)

    # ===== Osipenko-style R_LT(W,Q2) as DOTS with ERROR BARS =====
    try:
        R_osip = R_osipenko(W, Q2_value)             # vectorized over W
        valid = np.isfinite(R_osip)                   # (R defined only for W<=2.5 etc.)

        denom_os = 1.0 + eps[valid] * R_osip[valid]
        sigma_T_os = sigma_U[valid] / denom_os

        F1_os  = (K[valid]*M / four_pi2_alpha) * sigma_T_os
        F2_os  = (K[valid]*M / four_pi2_alpha) * (2.0*x[valid]/rho2[valid]) * ((1.0 + R_osip[valid])/denom_os) * sigma_U[valid]
        F1e_os = (K[valid]*M / four_pi2_alpha) * (sigma_U_err[valid]/denom_os)
        F2e_os = (K[valid]*M / four_pi2_alpha) * (2.0*x[valid]/rho2[valid]) * ((1.0 + R_osip[valid])/denom_os) * sigma_U_err[valid]

        # order along the chosen axis but only for valid points
        order_valid = np.argsort(axis[valid])

        ax[0].errorbar(axis[valid][order_valid], F1_os[order_valid],
                       yerr=F1e_os[order_valid], fmt="D", ms=1, lw=1, color="k",
                       label="RGA (V.Klimenko) w/ R_LT parametrisation")
        ax[1].errorbar(axis[valid][order_valid], F2_os[order_valid],
                       yerr=F2e_os[order_valid], fmt="D", ms=1, lw=1, color="k",
                       label="RGA (V.Klimenko) w/ R_LT parametrisation")

        # store table under a clear key
        # fill full-length arrays with NaN where invalid so downstream shapes match
        F1_full  = np.full_like(W, np.nan, dtype=float); F1_full[valid]  = F1_os
        F2_full  = np.full_like(W, np.nan, dtype=float); F2_full[valid]  = F2_os
        F1e_full = np.full_like(W, np.nan, dtype=float); F1e_full[valid] = F1e_os
        F2e_full = np.full_like(W, np.nan, dtype=float); F2e_full[valid] = F2e_os

        results["RLT_osip"] = pd.DataFrame({
            "x": x, "F1": F1_full, "F2": F2_full, "F1_err": F1e_full, "F2_err": F2e_full
        }).sort_values("x").reset_index(drop=True)

    except Exception as e:
        print(f"[WARN] Osipenko R_LT evaluation failed: {e}")

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
                #if not exp.empty:
                    #ax[1].errorbar(exp["x"].to_numpy(), exp["f2"].to_numpy(), yerr=exp["f2_err"].to_numpy(), fmt="D", ms=1, lw=1, label="CLAS+World data interpolation")

    else:
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
                #if not exp.empty:
                    #W_exp = np.sqrt(M*M + Q2_value*(1.0/exp["x"].to_numpy() - 1.0))
                    #ax[1].errorbar(W_exp, exp["f2"].to_numpy(), yerr=exp["f2_err"].to_numpy(), fmt="D", ms=1, lw=1, label="CLAS+World data interpolation")
            except Exception:
                pass

    # ===== cosmetics, bounds, save =====
    ax[0].set_xlabel(xlabel); ax[1].set_xlabel(xlabel)
    ax[0].set_ylabel("F1 (dimensionless)")
    ax[1].set_ylabel("F2 (dimensionless)")
    ax[0].set_title(f"F1 vs {xlabel.split()[0]}  (Q²={Q2_value} GeV²)")
    ax[1].set_title(f"F2 vs {xlabel.split()[0]}  (Q²={Q2_value} GeV²)")


    ax[0].legend(frameon=False, fontsize=9)
    ax[1].legend(frameon=False, fontsize=9)

    out_png = f"F1F2_vs_{png_suffix}_RLTscan_Q2={Q2_value}.png"
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    return results, (axis_sorted, xlabel)






def calc_trunc_moment_data(Q2_value, res_region):
    
    
    """
    Integrate F2 over x corresponding to W in [Wmin, Wmax] at fixed Q^2.
    Calls F1F2_from_d2sigma(Q2_value, vs_what="x") (which also makes the x-plot).
    Returns a one-row DataFrame: Q2, moment, error, where
      moment = ∫_{x(Wmax)}^{x(Wmin)} F2(x,Q^2) dx
      error  = 0.5 * [ ∫ (F2+σ) dx  -  ∫ (F2-σ) dx ]
      TUNE W RANGE BELOW!!!!
    """
    
    if res_region == "full":
        Wmin, Wmax = 1.15, 2.5
        if np.isclose(Q2_value, 9.699, rtol=0, atol=1e-3):
            Wmax = 2.25
    elif res_region == "part":
        Wmin, Wmax = 1.15, 1.75
        
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
        print(f"[WARN] No x-overlap at Q²={Q2_value:.3f}: "
              f"requested [{x_lo_bound:.4f},{x_hi_bound:.4f}], "
              f"data [{x[0]:.4f},{x[-1]:.4f}]")
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


def calc_trunc_moment_scan(res_region="full"):
    """
    For Q² in [2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699],
    compute the truncated moment over x corresponding to W∈[1.15,1.75] or W∈[1.15,W_max_data] using
    calc_trunc_moment_data(Q2_value). Plot Moment±error vs Q² and overlay CJ15
    theory points (5th col = TMC+HT, 9th col = TMC-only). Save PNG.

    Returns a DataFrame with columns: Q2, moment, error, CJ15_TMC_HT, CJ15_TMC
    (sorted by Q2).
    """
    out_png=f"trunc_moment_vs_Q2_{res_region}.png"
    theory_path=f"../getF1F2/Output/F2_trunc_cj15_{res_region}.txt"
    
    q2_values = [2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699]
    rows = []

    for q2 in q2_values:
        try:
            # one-row DataFrame with Q2, moment, error
            res = calc_trunc_moment_data(q2, res_region)
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
        # columns: 0=Q2, 4=F2bradyhtall (TMC+HT “all”), 8=F2bradyall (TMC-only “all”)
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
                marker="s", linestyle="-", ms=2, lw=1, color="red", label="NLO LT + HT + TMC (CJ15)")
    if "CJ15_TMC" in out.columns and out["CJ15_TMC"].notna().any():
        ax.plot(out["Q2"].to_numpy(),
                out["CJ15_TMC"].to_numpy(),
                marker="^", linestyle="dotted", ms=2, lw=1, color="green", label="NLO LT + HT (CJ15)")

    ax.set_xlabel("Q²  [GeV²]")
    if res_region == "full":
        ax.set_ylabel(r"$M_2$ =   $\int_{x(W\in[1.15,2.5])} F_2(x,Q^2)\,dx$")
        ax.set_title(f"Truncated moment vs Q² in full resonance region")
    elif res_region == "part":
        ax.set_ylabel(r"$M_2$ =   $\int_{x(W\in[1.15,1.75])} F_2(x,Q^2)\,dx$")
        ax.set_title(f"Truncated moment vs Q² in partial resonance region")
    ax.legend(frameon=False)

    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    return out







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



#for q2 in [2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699]:
#    F1F2_compare_RLT(q2, vs_what="x")
#    F1F2_compare_RLT(q2, vs_what="w")
for q2 in [2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699]:
    F1F2_from_d2sigma(q2, vs_what="x") 
    F1F2_from_d2sigma(q2, vs_what="w")

calc_trunc_moment_scan("full")
calc_trunc_moment_scan("part")

#xsec_from_pdf_F2_HT(2.774, 10.6)

#plot_epsilon_vs_W_grid()