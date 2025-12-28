import numpy as np
import math
from scipy.interpolate import  interp1d, CubicSpline
from scipy.integrate import quad
import os
import pandas as pd


"""
Helper functions for PDF-based structure function calculations
"""


def get_lo_pdf_interpolators(fixed_Q2, pdf_set, q2_tol=1e-4):
    """
    Load LO PDF structure function data from getF1F2 output files,
    filter by fixed Q², and create interpolators over W.

    Args:
        fixed_Q2 (float): Q² value in GeV²
        pdf_set (str): PDF set name (e.g., "CT18NLO")
        q2_tol (float): tolerance for matching Q² values
    """
    folder  = f"../getF1F2/Output/Output_{pdf_set}_LO"
    f1_file = f"{folder}/F1_LO.txt"
    f2_file = f"{folder}/F2_LO.txt"
    
    df1 = pd.read_csv(f1_file, sep=r"\s+", header=None,
                      names=["Q2","W","F1_LO"])
    df2 = pd.read_csv(f2_file, sep=r"\s+", header=None,
                      names=["Q2","W","F2_LO"])
    
    # Helper: safe interpolator with sorting + cubic→linear fallback
    def _make_interp(W, Y, kind="cubic"):
        W = np.asarray(W, dtype=float)
        Y = np.asarray(Y, dtype=float)
        # sort & unique by W (keep first occurrence)
        order = np.argsort(W)
        Ws, Ys = W[order], Y[order]
        # drop duplicate W
        if np.any(np.diff(Ws) == 0):
            uniq_idx = np.concatenate(([0], np.where(np.diff(Ws) != 0)[0] + 1))
            Ws, Ys = Ws[uniq_idx], Ys[uniq_idx]
        # fallback if too short for cubic
        use_kind = kind if Ws.size >= 4 else "linear"
        return interp1d(Ws, Ys, kind=use_kind, bounds_error=False, fill_value="extrapolate")

    # Single NaN-producing callable
    def _nan_i(W):
        W = np.asarray(W, dtype=float)
        return np.full_like(W, np.nan, dtype=float)
    
    # Masks (more forgiving to float noise)
    mask1 = np.isclose(df1["Q2"].to_numpy(), fixed_Q2, atol=q2_tol, rtol=0.0)
    mask2 = np.isclose(df2["Q2"].to_numpy(), fixed_Q2, atol=q2_tol, rtol=0.0)
    
    # F2 (required)
    if not mask2.any():
        raise ValueError(f"Q²={fixed_Q2} not found in F2 file.")
    W2         = df2.loc[mask2, "W"].to_numpy()
    F2_LO   = df2.loc[mask2, "F2_LO"].to_numpy()
    F2_LO_i   = _make_interp(W2, F2_LO)


    # F1 (optional)
    if mask1.any():
        W1           = df1.loc[mask1, "W"].to_numpy()
        F1_LO     = df1.loc[mask1, "F1_LO"].to_numpy()   
        F1_LO_i      = _make_interp(W1, F1_LO)

    else:
        W1 = None
        F1_LO_i = _nan_i
        
    # W grid to report (keep your original behavior: F1∩F2 if F1 exists; else F2)
    #W_sorted = np.sort(np.intersect1d(W1, W2)) if W1 is not None else np.sort(W2)
    
    W_sorted = np.sort(W2)  # base the range on F2 only

    return (F1_LO_i, F2_LO_i, W_sorted)

    
    
    

def get_nlo_pdf_interpolators(fixed_Q2, pdf_set, q2_tol=1e-4):
    

    folder  = f"../getF1F2/Output/Output_{pdf_set}"
    f1_file = f"{folder}/F1.txt"
    f2_file = f"{folder}/F2.txt"
    fl_file = f"{folder}/FL.txt"

    df1 = pd.read_csv(f1_file, sep=r"\s+", header=None,
                      names=["Q2","W","F1_naked","F1_brady","F1_brady_alt","F1_bradyHT"])
    df2 = pd.read_csv(f2_file, sep=r"\s+", header=None,
                      names=["Q2","W","F2_naked","F2_moffat","F2_brady0","F2_brady","F2_bradyHT"])
    df3 = pd.read_csv(fl_file, sep=r"\s+", header=None,
                      names=["Q2","W","FL_naked","FL_moffat","FL_brady0","FL_brady","FL_bradyHT"])

    # Helper: safe interpolator with sorting + cubic→linear fallback
    def _make_interp(W, Y, kind="cubic"):
        W = np.asarray(W, dtype=float)
        Y = np.asarray(Y, dtype=float)
        # sort & unique by W (keep first occurrence)
        order = np.argsort(W)
        Ws, Ys = W[order], Y[order]
        # drop duplicate W
        if np.any(np.diff(Ws) == 0):
            uniq_idx = np.concatenate(([0], np.where(np.diff(Ws) != 0)[0] + 1))
            Ws, Ys = Ws[uniq_idx], Ys[uniq_idx]
        # fallback if too short for cubic
        use_kind = kind if Ws.size >= 4 else "linear"
        return interp1d(Ws, Ys, kind=use_kind, bounds_error=False, fill_value="extrapolate")

    # Single NaN-producing callable
    def _nan_i(W):
        W = np.asarray(W, dtype=float)
        return np.full_like(W, np.nan, dtype=float)

    # Masks (more forgiving to float noise)
    mask1 = np.isclose(df1["Q2"].to_numpy(), fixed_Q2, atol=q2_tol, rtol=0.0)
    mask2 = np.isclose(df2["Q2"].to_numpy(), fixed_Q2, atol=q2_tol, rtol=0.0)
    maskl = np.isclose(df3["Q2"].to_numpy(), fixed_Q2, atol=q2_tol, rtol=0.0)

    # F2 (required)
    if not mask2.any():
        raise ValueError(f"Q²={fixed_Q2} not found in F2 file.")
    W2         = df2.loc[mask2, "W"].to_numpy()
    F2_naked   = df2.loc[mask2, "F2_naked"].to_numpy()
    F2_brady   = df2.loc[mask2, "F2_brady"].to_numpy()
    F2_bradyHT = df2.loc[mask2, "F2_bradyHT"].to_numpy()
    F2_naked_i   = _make_interp(W2, F2_naked)
    F2_brady_i   = _make_interp(W2, F2_brady)
    F2_bradyHT_i = _make_interp(W2, F2_bradyHT)

    # F1 (optional)
    if mask1.any():
        W1           = df1.loc[mask1, "W"].to_numpy()
        F1_naked     = df1.loc[mask1, "F1_naked"].to_numpy()
        F1_brady     = df1.loc[mask1, "F1_brady"].to_numpy()
        F1_brady_alt = df1.loc[mask1, "F1_brady_alt"].to_numpy()
        F1_bradyHT   = df1.loc[mask1, "F1_bradyHT"].to_numpy()
        F1_naked_i      = _make_interp(W1, F1_naked)
        F1_brady_i      = _make_interp(W1, F1_brady)
        F1_brady_alt_i  = _make_interp(W1, F1_brady_alt)
        F1_bradyHT_i    = _make_interp(W1, F1_bradyHT)
    else:
        W1 = None
        F1_naked_i = F1_brady_i = F1_brady_alt_i = F1_bradyHT_i = _nan_i

    # FL (optional)
    if maskl.any():
        Wl         = df3.loc[maskl, "W"].to_numpy()
        FL_naked   = df3.loc[maskl, "FL_naked"].to_numpy()
        FL_brady   = df3.loc[maskl, "FL_brady"].to_numpy()
        FL_bradyHT = df3.loc[maskl, "FL_bradyHT"].to_numpy()
        FL_naked_i   = _make_interp(Wl, FL_naked)
        FL_brady_i   = _make_interp(Wl, FL_brady)
        FL_bradyHT_i = _make_interp(Wl, FL_bradyHT)
    else:
        Wl = None
        FL_naked_i = FL_moffat_i = FL_brady0_i = FL_brady_i = FL_bradyHT_i = _nan_i

    # W grid to report (keep your original behavior: F1∩F2 if F1 exists; else F2)
    #W_sorted = np.sort(np.intersect1d(W1, W2)) if W1 is not None else np.sort(W2)
    
    W_sorted = np.sort(W2)  # base the range on F2 only

    return (F1_naked_i, F1_brady_i, F1_brady_alt_i, F1_bradyHT_i,
            F2_naked_i, F2_brady_i, F2_bradyHT_i,
            FL_naked_i, FL_brady_i, FL_bradyHT_i,
            W_sorted)


    
    
def compute_pdf_cross_sections(W, Q2, beam_energy, F1_interp, F2_interp): ## Renamed function name!
    """
    Computes the differential cross section using PDF-based
    structure function interpolators.

    Parameters:
        W          : hadronic invariant mass (GeV)
        Q2         : squared momentum transfer (GeV²)
        beam_energy: incident lepton energy (GeV)
        F1_interp  : interpolator for chosen F1 (e.g., F1_brady_interp)
        F2_interp  : interpolator for chosen F2 (e.g., F2_bradyHT_interp)

    Returns:
        dσ (float): differential cross section
    """

    alpha = 1 / 137.04
    Mp = 0.9385
    pi = math.pi

    wtot = math.sqrt(2 * Mp * beam_energy + Mp**2)
    if W > wtot:
        raise ValueError("W is greater than lab energy (w_tot).")

    elepi = beam_energy
    omeg = (W**2 + Q2 - Mp**2) / (2 * Mp)
    elepf = elepi - omeg
    if elepf <= 0:
        raise ValueError("Final lepton energy is non-positive.")

    plepi = elepi
    plepf = elepf
    clep = (-Q2 + 2 * elepi * elepf) / (2 * plepi * plepf)

    fac3 = pi * W / (Mp * elepi * elepf)
    fcrs3 = 4 * (alpha / Q2)**2 * (0.197327**2) * 1e4 * (elepf**2)

    ss2 = (1 - clep) / 2
    cc2 = (1 + clep) / 2

    F1 = float(F1_interp(W))
    F2 = float(F2_interp(W))
    W1 = F1 / Mp
    W2 = F2 / omeg

    xxx = 2 * ss2 * W1 + cc2 * W2
    dcrs = fcrs3 * fac3 * xxx

    return dcrs

def compute_pdf_cross_sections_from_F2_FL(W, Q2, beam_energy, F2_func, FL_func): # E0 = beam energy for consistency, name changed for consistency
    
        E0 = beam_energy
        # constants (keep local so the function is self-contained)
        alpha = 1/137.035999084
        GEV2_TO_UB = 389.379      # 1 GeV^-2 = 389.379 μb
        M = 0.9385

        W2 = W*W
        denom = W2 - M*M + Q2
        if denom <= 0.0:
            return np.nan

        # DIS vars
        x = Q2 / denom
        if x <= 0.0:
            return np.nan
        K = (W2 - M*M) / (2.0*M)           # Hand K
        rho2 = 1.0 + 4.0*M*M*x*x/Q2        # ρ²

        # electron kinematics
        nu = denom / (2.0*M)               # energy transfer (lab)
        Ep = E0 - nu                       # scattered e- energy
        den_ang = 4.0*E0*Ep - Q2
        if Ep <= 0.0 or den_ang <= 0.0 or K <= 0.0:
            return np.nan
        tan2 = Q2 / den_ang
        eps  = 1.0 / (1.0 + 2.0*(1.0 + (nu*nu)/Q2)*tan2)  # <-- your ε(ν)

        # structure functions
        F2 = F2_func(W)
        FL = FL_func(W)
        if not (np.isfinite(F2) and np.isfinite(FL)):
            return np.nan

        # Identity: W(W^2−M^2)/(2 x K) = (W M)/x
        pref = (alpha*alpha*math.pi) * (W / (E0*E0 * M*M * (1.0 - eps) * Q2 * x))
        val = pref * (F2*rho2 + FL*(eps - 1.0))
        return val * GEV2_TO_UB
    




def get_R_from_F1F2(W, Q2, E_beam, F1_interp, F2_interp):
    """
    Compute R = σ_L / σ_T and d²σ/dW dQ² from F1 and F2 interpolators
    at given (W, Q², E_beam).

    Args:
        W        (float): hadronic invariant mass (GeV)
        Q2       (float): Q^2 (GeV^2)
        E_beam   (float): beam energy E (GeV)
        F1_interp: callable F1(W) interpolator
        F2_interp: callable F2(W) interpolator

    Returns:
        R, sigma_L, sigma_T, d2sigma_dWdQ2

        σ_L, σ_T in GeV^{-2}, d²σ/dW dQ² in μb / GeV^3
    """

    # constants
    alpha = 1.0 / 137.035999084
    M = 0.9385
    GEV2_TO_UB = 389.379  # 1 GeV^{-2} = 389.379 μb
    
    nan6 = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    W2 = W * W
    numer = W2 - M * M + Q2
    if numer <= 0.0 or Q2 <= 0.0:
        return nan6

    # Hand K
    K = (W2 - M * M) / (2.0 * M)
    if K <= 0.0:
        return nan6

    # energy transfer (lab)
    nu = numer / (2.0 * M)
    if nu <= 0.0:
        return nan6

    # scattered electron energy
    E = float(E_beam)
    E_prime = E - nu
    if E_prime <= 0.0:
        return nan6

    # structure functions at this W
    F1 = float(F1_interp(W))
    F2 = float(F2_interp(W))
    if not (np.isfinite(F1) and np.isfinite(F2)):
        return nan6

    # W1, W2 from F1, F2
    W1 = F1 / M
    W2 = F2 / nu

    # prefactor from σ_T, σ_L equations
    pref = 4.0 * math.pi**2 * alpha / K

    # σ_T, σ_L in GeV^{-2}
    sigma_T = pref * W1
    sigma_L = pref * ((1.0 + nu**2 / Q2) * W2 - W1)
    

    # ratio R
    R = sigma_L / sigma_T if sigma_T != 0.0 else np.nan
    
    #Cross check: back to W1, W2 from σ_T, R
    W1_pdf = sigma_T/pref
    W2_pdf = (R + 1.0) * W1_pdf / (1.0 + nu**2 / Q2)

    # virtual photon polarization ε
    # sin^2(theta/2) from Q^2 = 4 E E' sin^2(theta/2)
    sin2_half_theta = Q2 / (4.0 * E * E_prime)
    if sin2_half_theta <= 0.0 or sin2_half_theta >= 1.0:
        return R, sigma_L, sigma_T, np.nan, W1_pdf, W2_pdf

    tan2_half_theta = sin2_half_theta / (1.0 - sin2_half_theta)
    eps = 1.0 / (1.0 + 2.0 * (1.0 + nu**2 / Q2) * tan2_half_theta)

    # Hand flux Γ_Hand
    if eps >= 1.0:
        return R, sigma_L, sigma_T, np.nan, W1_pdf, W2_pdf

    Gamma_hand = (
        alpha / (2.0 * math.pi**2) *
        (E_prime / E) *
        (K / Q2) *
        1.0 / (1.0 - eps)
    )

    # d^2σ / dW dQ^2 (μb / GeV^3)
    d2sigma_dWdQ2 = (
        GEV2_TO_UB *
        Gamma_hand *
        (sigma_T + eps * sigma_L) *
        (math.pi * W / (M * E * E_prime))
    )

    return R, sigma_L, sigma_T, d2sigma_dWdQ2, W1_pdf, W2_pdf






def get_pdf_xsecs_table(fixed_Q2, beam_energy,
                             pdf_set_nlo, pdf_set_lo,
                             out_dir="PDF_based_xsecs_tables",
                             use_F1_alt=False,
                             W_vals=None,
                             filename=None):
    """
    Create a .dat table with columns:
        Q2  W  TMC_xsection  TMC_HT_xsection
    using NLO Brady tables and your get_nlo_* helpers.

    Args:
        fixed_Q2 (float): Q² value in GeV²
        beam_energy (float): beam energy in GeV
        out_dir (str): output directory (created if missing)
        use_F1_alt (bool): if True, use F1_brady_alt instead of F1_brady
        W_vals (array-like or None): custom W grid. If None, uses the common W grid from interpolators
        filename (str or None): custom filename. If None, auto-named.

    Returns:
        str: path to the written .dat file
    """

    # Build NLO interpolators
    F1_naked, F1_brady, F1_brady_alt, F1_bradyHT, F2_naked, F2_brady, F2_bradyHT,_,_,_, W_common = get_nlo_pdf_interpolators(fixed_Q2, pdf_set=pdf_set_nlo)
    
    # Build LO interpolators 
    F1_LO, F2_LO, W_common_lo = get_lo_pdf_interpolators(fixed_Q2, pdf_set=pdf_set_lo)
    

    if W_vals is not None:
        W_grid = np.asarray(W_vals, dtype=float)
    else:
    # safer: intersect to avoid extrapolation surprises
        W_grid = np.sort(np.intersect1d(np.asarray(W_common, dtype=float),np.asarray(W_common_lo, dtype=float)))

        


    # Compute cross sections
    lo_xsec = []
    nlo_xsec = []
    nlo_tmc_xsec = []
    nlo_tmc_ht_xsec = []
    
    
    # LO xsecs
    for W in W_grid:
        #LO
        try:
            lo_xsec.append(compute_pdf_cross_sections(W, fixed_Q2, beam_energy,
                                                       F1_interp=F1_LO, F2_interp=F2_LO))
        except Exception:
            lo_xsec.append(np.nan)
        #NLO
        try:
            nlo_xsec.append(compute_pdf_cross_sections(W, fixed_Q2, beam_energy,
                                                       F1_interp=F1_naked, F2_interp=F2_naked))
        except Exception:
            nlo_xsec.append(np.nan)
        #NLO TMC
        try:
            nlo_tmc_xsec.append(compute_pdf_cross_sections(W, fixed_Q2, beam_energy,
                                                          F1_interp=F1_brady, F2_interp=F2_brady))
        except Exception:
            nlo_tmc_xsec.append(np.nan)
        #NLO TMC HT
        try:
            nlo_tmc_ht_xsec.append(compute_pdf_cross_sections(W, fixed_Q2, beam_energy,
                                                          F1_interp=F1_bradyHT, F2_interp=F2_bradyHT))
        except Exception:
            nlo_tmc_ht_xsec.append(np.nan)    
        

    lo_xsec = np.asarray(lo_xsec, dtype=float)
    nlo_xsec = np.asarray(nlo_xsec, dtype=float)
    nlo_tmc_xsec = np.asarray(nlo_tmc_xsec, dtype=float)
    nlo_tmc_ht_xsec = np.asarray(nlo_tmc_ht_xsec, dtype=float)

    # Assemble table: Q2, W, TMC_HT_xsection
    Q2_col = np.full_like(W_grid, float(fixed_Q2), dtype=float)
    table = np.column_stack([Q2_col, W_grid, lo_xsec, nlo_xsec, nlo_tmc_xsec, nlo_tmc_ht_xsec])

    # Save
    os.makedirs(out_dir, exist_ok=True)
    if filename is None:
        q2_str = str(fixed_Q2).rstrip("0").rstrip(".")
        filename = f"PDF_based_xsecs_Q2={q2_str}_E={beam_energy}_nlo_from_{pdf_set_nlo}_lo_from_{pdf_set_lo}.dat"
    out_path = os.path.join(out_dir, filename)

    header = "Q2\tW\tLO_xsection(mub/GeV)\tNLO_xsection(mub/GeV)\tNLO_TMC_xsection(mub/GeV)\tNLO_TMC_HT_xsection(mub/GeV)"
    np.savetxt(out_path, table, fmt="%.6e", delimiter="\t", header=header, comments="")

    return out_path


def calculate_moment_LO_pdf(Q2_value, region, pdf_set="CJ15lo", n=2,
                            q2_tol=1e-4,
                            epsabs=1e-8, epsrel=1e-6, limit=200,
                            n_W_dense=800):
    """
    Truncated Cornwall–Norton moment from LO PDF:
        M_n(Q2; region) = ∫_{x_lo}^{x_hi} x^{n-2} F2_LO(x,Q2) dx

    Region is defined via W-bounds (same mapping as AO/data),
    then converted to x-bounds at fixed Q2.

    Implementation:
      1) get F2_LO(W) interpolator from get_lo_pdf_interpolators()
      2) sample F2_LO on W in [W_lo, W_hi]
      3) convert sampled W -> x, build CubicSpline F2_LO(x)
      4) integrate with quad in x

    Returns DataFrame with:
      Q2, region, n, x_lo, x_hi, moment, error(=0)
    """


    M = 0.9382720813
    Q2 = float(Q2_value)

    # --- get LO interpolator over W and its native W coverage ---
    F1_LO_i, F2_LO_i, W_sorted = get_lo_pdf_interpolators(Q2, pdf_set, q2_tol=q2_tol)
    W_sorted = np.asarray(W_sorted, dtype=float)
    Wmin_cov = float(np.nanmin(W_sorted))
    Wmax_cov = float(np.nanmax(W_sorted))

    # --- region -> W bounds (same as AO/data) ---
    W_min_data = 1.15
    Wmax1 = 1.35
    Wmin2 = Wmax1
    Wmax2 = 1.60
    Wmin3 = Wmax2
    Wmax3 = 2.0
    W_max = 2.25 if np.isclose(Q2, 9.699, atol=1e-3) else 2.50

    reg = str(region).lower().strip()
    region_map = {
        "1": (W_min_data, Wmax1), "r1": (W_min_data, Wmax1), "first": (W_min_data, Wmax1), "1st": (W_min_data, Wmax1),
        "2": (Wmin2, Wmax2),      "r2": (Wmin2, Wmax2),      "second": (Wmin2, Wmax2),      "2nd": (Wmin2, Wmax2),
        "3": (Wmin3, Wmax3),      "r3": (Wmin3, Wmax3),      "third": (Wmin3, Wmax3),       "3rd": (Wmin3, Wmax3),
        "partial": (W_min_data, Wmax3), "part": (W_min_data, Wmax3),
        "tail": (Wmax3, W_max),
        "full": (W_min_data, W_max), "all": (W_min_data, W_max),
    }
    if reg not in region_map:
        raise ValueError(f"Unknown region='{region}'. Use one of: {sorted(region_map.keys())}")

    W_lo, W_hi = region_map[reg]

    # --- intersect region with LO PDF coverage to avoid extrapolation artifacts ---
    W_lo_use = max(W_lo, Wmin_cov)
    W_hi_use = min(W_hi, Wmax_cov)
    if W_lo_use >= W_hi_use:
        # still report consistent x bounds for attempted region
        def x_of_W(Wv):
            return Q2 / (Wv * Wv - M * M + Q2)

        x1 = x_of_W(W_lo_use)
        x2 = x_of_W(W_hi_use)
        return pd.DataFrame([{
            "Q2": Q2_value, "region": region, "n": n,
            "x_lo": min(x1, x2), "x_hi": max(x1, x2),
            "moment": 0.0, "error": 0.0
        }])

    # --- build a W grid inside the region: native points + dense grid + endpoints ---
    W_native = W_sorted[(W_sorted >= W_lo_use) & (W_sorted <= W_hi_use)]
    W_dense  = np.linspace(W_lo_use, W_hi_use, int(n_W_dense))
    W_grid   = np.unique(np.concatenate(([W_lo_use], W_native, W_dense, [W_hi_use])))

    # --- evaluate F2_LO(W), convert to x, clean ---
    F2_grid = np.asarray(F2_LO_i(W_grid), dtype=float)
    x_grid  = Q2 / (W_grid * W_grid - M * M + Q2)

    m = np.isfinite(x_grid) & np.isfinite(F2_grid)
    x_grid = x_grid[m]
    F2_grid = F2_grid[m]

    if x_grid.size < 2:
        return pd.DataFrame([{
            "Q2": Q2_value, "region": region, "n": n,
            "x_lo": np.nan, "x_hi": np.nan,
            "moment": 0.0, "error": 0.0
        }])

    # --- sort by increasing x, ensure unique x for spline ---
    o = np.argsort(x_grid)
    x_grid = x_grid[o]
    F2_grid = F2_grid[o]

    x_u, idx = np.unique(x_grid, return_index=True)
    F2_u = F2_grid[idx]

    if x_u.size < 2:
        return pd.DataFrame([{
            "Q2": Q2_value, "region": region, "n": n,
            "x_lo": np.nan, "x_hi": np.nan,
            "moment": 0.0, "error": 0.0
        }])

    # --- x bounds from W bounds (region definition) ---
    def x_of_W(Wv):
        return Q2 / (Wv * Wv - M * M + Q2)

    xb1 = x_of_W(W_lo_use)
    xb2 = x_of_W(W_hi_use)
    x_lo_bound = min(xb1, xb2)
    x_hi_bound = max(xb1, xb2)

    # --- intersect with available x coverage from the sampled grid ---
    lo = max(x_lo_bound, float(x_u[0]))
    hi = min(x_hi_bound, float(x_u[-1]))
    if lo >= hi:
        return pd.DataFrame([{
            "Q2": Q2_value, "region": region, "n": n,
            "x_lo": lo, "x_hi": hi,
            "moment": 0.0, "error": 0.0
        }])

    # --- cubic spline F2(x), no extrapolation ---
    F2_spline_x = CubicSpline(x_u, F2_u, bc_type="natural", extrapolate=False)

    def integrand(xv):
        f2 = F2_spline_x(xv)
        if not np.isfinite(f2):
            return 0.0
        return (xv ** (n - 2)) * float(f2)

    moment, _ = quad(integrand, lo, hi, epsabs=epsabs, epsrel=epsrel, limit=limit)

    return pd.DataFrame([{
        "Q2": Q2_value, "region": region, "n": n,
        "x_lo": lo, "x_hi": hi,
        "moment": float(moment), "error": 0.0
    }])







# Get PDF-based xsec tables for various Q2 and beam energies

#get_pdf_xsecs_table(fixed_Q2=2.774, beam_energy=10.6, pdf_set_nlo="CT18NLO", pdf_set_lo="CT18LO", W_vals=np.arange(1.07, 2.51, 0.01))
#get_pdf_xsecs_table(fixed_Q2=3.244, beam_energy=10.6, pdf_set_nlo="CT18NLO", pdf_set_lo="CT18LO", W_vals=np.arange(1.07, 2.51, 0.01))
#get_pdf_xsecs_table(fixed_Q2=3.793, beam_energy=10.6, pdf_set_nlo="CT18NLO", pdf_set_lo="CT18LO", W_vals=np.arange(1.07, 2.51, 0.01))
#get_pdf_xsecs_table(fixed_Q2=4.435, beam_energy=10.6, pdf_set_nlo="CT18NLO", pdf_set_lo="CT18LO", W_vals=np.arange(1.07, 2.51, 0.01))
#get_pdf_xsecs_table(fixed_Q2=5.187, beam_energy=10.6, pdf_set_nlo="CT18NLO", pdf_set_lo="CT18LO", W_vals=np.arange(1.07, 2.51, 0.01))
#get_pdf_xsecs_table(fixed_Q2=6.065, beam_energy=10.6, pdf_set_nlo="CT18NLO", pdf_set_lo="CT18LO", W_vals=np.arange(1.07, 2.51, 0.01))
#get_pdf_xsecs_table(fixed_Q2=7.093, beam_energy=10.6, pdf_set_nlo="CT18NLO", pdf_set_lo="CT18LO", W_vals=np.arange(1.07, 2.51, 0.01))
#get_pdf_xsecs_table(fixed_Q2=8.294, beam_energy=10.6, pdf_set_nlo="CT18NLO", pdf_set_lo="CT18LO", W_vals=np.arange(1.07, 2.51, 0.01))
#get_pdf_xsecs_table(fixed_Q2=9.699, beam_energy=10.6, pdf_set_nlo="CT18NLO", pdf_set_lo="CT18LO", W_vals=np.arange(1.07, 2.26, 0.01))
#get_pdf_xsecs_table(fixed_Q2=12.0,beam_energy=15.0, pdf_set_nlo="CJ15nlo", pdf_set_lo="CJ15lo", W_vals=np.arange(1.07, 2.51, 0.01))
#get_pdf_xsecs_table(fixed_Q2=14.0,beam_energy=15.0, pdf_set_nlo="CJ15nlo", pdf_set_lo="CJ15lo", W_vals=np.arange(1.07, 2.51, 0.01))
#get_pdf_xsecs_table(fixed_Q2=16.0,beam_energy=22.0, pdf_set_nlo="CJ15nlo", pdf_set_lo="CJ15lo", W_vals=np.arange(1.07, 2.51, 0.01))
#get_pdf_xsecs_table(fixed_Q2=18.0,beam_energy=22.0, pdf_set_nlo="CJ15nlo", pdf_set_lo="CJ15lo", W_vals=np.arange(1.07, 2.51, 0.01))
#get_pdf_xsecs_table(fixed_Q2=20.0,beam_energy=22.0, pdf_set_nlo="CJ15nlo", pdf_set_lo="CJ15lo", W_vals=np.arange(1.07, 2.51, 0.01))
