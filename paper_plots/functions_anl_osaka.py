import numpy as np
import math
from scipy.interpolate import RectBivariateSpline, CubicSpline
from scipy.integrate import quad
import matplotlib.pyplot as plt
import os
import pandas as pd
from matplotlib.ticker import MultipleLocator, FormatStrFormatter



"""Functions for interpolating structure functions and computing cross sections for ANL-Osaka model.
"""
    # --- constants (GeV units) ---
ALPHA_EM = 1/137.035999084
M_PROTON = 0.9382720813
GEV2_TO_UB = 389.379  # 1 GeV^{-2} = 389.379 microbarn

def interpolate_structure_functions(file_path, target_W, target_Q2):
    """
    Interpolates the structure functions W1 and W2 for given W and Q2 values
    using bicubic (cubic spline) interpolation.

    If the target values are outside the data range, a ValueError is raised.

    Parameters:
        file_path (str): Path to the input .dat file containing the data.
        target_W (float): The W (invariant mass) value at which to interpolate.
        target_Q2 (float): The Q2 (virtuality) value at which to interpolate.

    Returns:
        tuple: Interpolated (W1, W2) values.
    """
    # Load the data (assumed columns: W, Q2, W1, W2)
    data = np.loadtxt(file_path)
    
    # Extract columns
    W = data[:, 0]
    Q2 = data[:, 1]
    W1 = data[:, 2]
    W2 = data[:, 3]
    
    # Get unique grid points (assumes a regular grid)
    W_unique = np.unique(W)
    Q2_unique = np.unique(Q2)
    
    # Check if the target values are within the available data range
    W_min, W_max = W_unique[0], W_unique[-1]
    Q2_min, Q2_max = Q2_unique[0], Q2_unique[-1]
    
    if target_W < W_min or target_W > W_max:
        raise ValueError(f"Error: Target W = {target_W} is outside the available range: {W_min} to {W_max}")
    if target_Q2 < Q2_min or target_Q2 > Q2_max:
        raise ValueError(f"Error: Target Q2 = {target_Q2} is outside the available range: {Q2_min} to {Q2_max}")
    
    # Determine grid dimensions
    nW = len(W_unique)
    nQ2 = len(Q2_unique)
    
    # Reshape structure functions into 2D grids.
    W1_grid = W1.reshape(nW, nQ2)
    W2_grid = W2.reshape(nW, nQ2)
    
    # Build bicubic spline interpolators
    interp_W1 = RectBivariateSpline(W_unique, Q2_unique, W1_grid, kx=3, ky=3)
    interp_W2 = RectBivariateSpline(W_unique, Q2_unique, W2_grid, kx=3, ky=3)
    
    # Evaluate the interpolators at the target values.
    W1_interp = interp_W1(target_W, target_Q2)[0, 0]
    W2_interp = interp_W2(target_W, target_Q2)[0, 0]
    
    return W1_interp, W2_interp


def interpolate_structure_functions_1pi(file_path, target_W, target_Q2):
    """
    Bicubic interpolation of single-pion structure functions (W1, W2)
    on a regular (W, Q²) grid, using RectBivariateSpline.

    Expected columns in *file_path*:
        0 : W   (GeV)
        1 : Q²  (GeV²)
        2 : W1
        3 : W2
        (extra columns are ignored)
    """
    # ---------- load ----------
    data = np.loadtxt(file_path)
    if data.shape[1] < 4:
        raise ValueError(
            f"{file_path} must have ≥ 4 columns (W, Q², W1, W2); "
            f"found {data.shape[1]}"
        )

    W_all, Q2_all = data[:, 0], data[:, 1]
    W1_all, W2_all = data[:, 2], data[:, 3]

    # ---------- unique axes ----------
    W_uni   = np.unique(W_all)
    Q2_uni  = np.unique(Q2_all)
    nW, nQ2 = len(W_uni), len(Q2_uni)

    # range check
    if not (W_uni[0] <= target_W <= W_uni[-1]):
        raise ValueError(f"W = {target_W} GeV outside table range {W_uni[0]}–{W_uni[-1]}")
    if not (Q2_uni[0] <= target_Q2 <= Q2_uni[-1]):
        raise ValueError(f"Q² = {target_Q2} GeV² outside table range {Q2_uni[0]}–{Q2_uni[-1]}")

    # ---------- reshape to 2-D grid ----------
    try:
        W1_grid = W1_all.reshape(nW, nQ2)
        W2_grid = W2_all.reshape(nW, nQ2)
    except ValueError:
        raise ValueError(
            "1π table is not a complete rectangular W–Q² grid. "
            "Fill in the missing points or use a scattered-data interpolator."
        )

    # ---------- bicubic splines ----------
    spl_W1 = RectBivariateSpline(W_uni, Q2_uni, W1_grid, kx=3, ky=3)
    spl_W2 = RectBivariateSpline(W_uni, Q2_uni, W2_grid, kx=3, ky=3)

    W1_interp = spl_W1(target_W, target_Q2)[0, 0]
    W2_interp = spl_W2(target_W, target_Q2)[0, 0]

    return W1_interp, W2_interp
def compute_cross_section_model(W, Q2, beam_energy, file_path="input_data/wempx.dat", verbose=True):
    """
    Computes the differential cross section dσ/dW/dQ² for an electromagnetic (EM)
    reaction using interpolated structure functions.

    The reaction is fixed to N(e,e')X with massless leptons.

    Parameters:
        W          : Invariant mass of the final hadron system (GeV)
        Q2         : Photon virtuality (GeV²)
        beam_energy: Beam (lepton) energy in the lab (GeV)
        file_path  : Path to the structure function file (default "input_data/wempx.dat")
        verbose    : If True, prints the interpolated structure functions.

    Returns:
        dcrs       : Differential cross section in units of 10^(-30) cm²/GeV³
    """
    # Define physical constants (in GeV units)
    fnuc = 0.9385         # Nucleon mass m_N
    pi = 3.1415926
    alpha = 1 / 137.04    # Fine-structure constant

    # For EM reaction, both initial and final lepton masses are zero.
    flepi = 0.0
    flepf = 0.0

    # Step 1: Interpolate structure functions
    W1, W2 = interpolate_structure_functions(file_path, W, Q2)
    if verbose:
        print(f"Interpolated structure functions at (W={W:.3f}, Q²={Q2:.3f}):")
        print(f"    W1 = {W1:.5e}")
        print(f"    W2 = {W2:.5e}")
    
    # Step 2: Kinematics
    # Total available energy: w_tot = sqrt(2*m_N*E + m_N²)
    wtot = math.sqrt(2 * fnuc * beam_energy + fnuc**2)
    if W > wtot:
        raise ValueError("W is greater than the available lab energy (w_tot).")
    
    # For massless leptons, energy equals momentum.
    elepi = beam_energy  # initial lepton energy
    plepi = elepi        # momentum of initial lepton
    
    # Energy transfer: ω = (W² + Q² - m_N²) / (2*m_N)
    omeg = (W**2 + Q2 - fnuc**2) / (2 * fnuc)
    elepf = elepi - omeg
    if elepf <= 0:
        raise ValueError("Final lepton energy is non-positive.")
    plepf = elepf        # momentum of final lepton
    
    # Cosine of the lepton scattering angle:
    # clep = (-Q² + 2*elepi*elepf) / (2*plepi*plepf)
    clep = (-Q2 + 2 * elepi * elepf) / (2 * plepi * plepf)
    
    # Step 3: Cross Section Calculation
    # Common kinematic factor: fac3 = π * W / (m_N * elepi * elepf)
    fac3 = pi * W / (fnuc * elepi * elepf)
    
    # Reaction-dependent factor for EM:
    # fcrs3 = 4 * ((alpha)/Q²)² * (0.197327²) * 1e4 * (elepf²)
    fcrs3 = 4 * (alpha / Q2)**2 * (0.197327**2) * 1e4 * (elepf**2)
    
    # Angular factors:
    ss2 = (1 - clep) / 2
    cc2 = (1 + clep) / 2
    
    # Combine structure functions: xxx = 2*ss2*W1 + cc2*W2
    xxx = 2 * ss2 * W1 + cc2 * W2
    
    # Differential cross section: dσ/dW/dQ² = fcrs3 * fac3 * xxx
    dcrs = fcrs3 * fac3 * xxx
    return dcrs


def compute_1pi_cross_section_model(W, Q2, beam_energy, file_path="input_data/wemp-pi.dat", verbose=True):
    """
    Computes the differential cross section dσ/dW/dQ² for the single-pion production
    channel (1π) in electromagnetic scattering N(e,e'π)X using interpolated structure functions.

    The reaction is fixed to EM interaction with massless leptons.

    Parameters:
        W          : Invariant mass of the final hadron system (GeV)
        Q2         : Photon virtuality (GeV²)
        beam_energy: Beam (lepton) energy in the lab frame (GeV)
        file_path  : Path to 1π structure function file (default: "input_data/wemp-pi.dat")
        verbose    : If True, prints the interpolated structure functions.

    Returns:
        dcrs       : Differential cross section in units of 10^(-30) cm²/GeV³
    """
    # Physical constants
    fnuc = 0.9385         # Nucleon mass in GeV
    pi = 3.1415926
    alpha = 1 / 137.04    # Fine-structure constant

    # Massless lepton assumption
    flepi = 0.0
    flepf = 0.0

    # Step 1: Interpolate structure functions W1 and W2 for 1pi production
    W1, W2 = interpolate_structure_functions_1pi(file_path, W, Q2)
    if verbose:
        print(f"[1π] Interpolated structure functions at (W={W:.3f}, Q²={Q2:.3f}):")
        print(f"    W1 = {W1:.5e}")
        print(f"    W2 = {W2:.5e}")

    # Step 2: Kinematics
    wtot = math.sqrt(2 * fnuc * beam_energy + fnuc**2)
    if W > wtot:
        raise ValueError("W is greater than the available lab energy (w_tot).")

    elepi = beam_energy
    plepi = elepi

    omeg = (W**2 + Q2 - fnuc**2) / (2 * fnuc)
    elepf = elepi - omeg
    if elepf <= 0:
        raise ValueError("Final lepton energy is non-positive.")
    plepf = elepf

    clep = (-Q2 + 2 * elepi * elepf) / (2 * plepi * plepf)

    # Step 3: Cross section calculation
    fac3 = pi * W / (fnuc * elepi * elepf)
    fcrs3 = 4 * (alpha / Q2)**2 * (0.197327**2) * 1e4 * (elepf**2)

    ss2 = (1 - clep) / 2
    cc2 = (1 + clep) / 2

    xxx = 2 * ss2 * W1 + cc2 * W2
    dcrs = fcrs3 * fac3 * xxx

    return dcrs

def compute_2pi_cross_section_model(
    W,
    Q2,
    beam_energy,
    full_file_path="input_data/wempx.dat",
    onepi_file_path="input_data/wemp-pi.dat",
    verbose=False,
    clamp_nonneg=False,
):
    """
    Returns the difference between the full ANL-Osaka cross section and the 1π contribution:
        dσ_2π ≡ dσ_full - dσ_1π

    Parameters
    ----------
    W : float
        Invariant mass of the hadronic system (GeV).
    Q2 : float
        Photon virtuality (GeV²).
    beam_energy : float
        Incident lepton energy in the lab (GeV).
    full_file_path : str
        Path to the (W,Q²)->(W1,W2) table for the full ANL-Osaka model.
        Default: "input_data/wempx.dat"
    onepi_file_path : str
        Path to the (W,Q²)->(W1,W2) table for the single-pion (1π) channel.
        Default: "input_data/wemp-pi.dat"
    verbose : bool
        If True, prints the component cross sections and the difference.
    clamp_nonneg : bool
        If True, returns max(dσ_full - dσ_1π, 0.0).

    Returns
    -------
    float
        Differential cross section dσ/dW/dQ² for (full − 1π),
        in the same units as compute_cross_section (10^(-30) cm²/GeV³).
    """
    # Full ANL-Osaka (all channels included by the table)
    dcs_full = compute_cross_section_model(
        W=W, Q2=Q2, beam_energy=beam_energy,
        file_path=full_file_path, verbose=False
    )

    # Single-pion exclusive contribution
    dcs_1pi = compute_1pi_cross_section_model(
        W=W, Q2=Q2, beam_energy=beam_energy,
        file_path=onepi_file_path, verbose=False
    )

    dcs_diff = dcs_full - dcs_1pi
    if clamp_nonneg and dcs_diff < 0.0:
        dcs_diff = 0.0

    if verbose:
        print(f"[2π proxy] At (W={W:.3f} GeV, Q²={Q2:.3f} GeV², E={beam_energy:.3f} GeV):")
        print(f"    dσ_full = {dcs_full:.6e}   (10^(-30) cm²/GeV³)")
        print(f"    dσ_1π   = {dcs_1pi:.6e}   (10^(-30) cm²/GeV³)")
        print(f"    dσ_full - dσ_1π = {dcs_diff:.6e}   (10^(-30) cm²/GeV³)"
              + ("   [clamped ≥ 0]" if clamp_nonneg else ""))

    return dcs_diff





def make_sigma_LT_table(Q2):
    """
    Compute σ_L(W) and σ_T(W) at fixed Q² using your existing
    interpolate_structure_functions(file_path, W, Q2) -> (W1, W2),
    then write a table and save a plot.

    Output:
      - sigma_LT_tables/sigma_LT_Q2={Q2:.3f}.dat    (columns: W, sigma_L, sigma_T)
      - sigma_LT_tables/sigma_LT_Q2={Q2:.3f}.png    (plot of sigma_L and sigma_T vs W)
    """
    assert Q2 > 0.0, "Q2 must be > 0"



    # Hand flux helpers
    def _nu(W):  # ν = (W^2 + Q^2 - M^2) / (2M)
        return (W**2 + Q2 - M_PROTON**2) / (2.0 * M_PROTON)

    def _K(W):   # Hand's equivalent photon energy: K = (W^2 - M^2) / (2M)
        return (W**2 - M_PROTON**2) / (2.0 * M_PROTON)

    # Read native W grid from your ANL-Osaka table
    file_path = "input_data/wempx.dat"
    data = np.loadtxt(file_path)
    W = np.unique(data[:, 0]).astype(float)

    # Interpolate W1, W2 on that grid
    W1 = np.empty_like(W)
    W2 = np.empty_like(W)
    for i, w in enumerate(W):
        W1[i], W2[i] = interpolate_structure_functions(file_path, w, Q2)

    # Convert to sigma_T and sigma_L
    K = _K(W)
    nu = _nu(W)

    # Avoid division by zero or negative K (physically W should be > M)
    mask = K > 0.0
    W_use  = W[mask]
    W1_use = W1[mask]
    W2_use = W2[mask]
    K_use  = K[mask]
    nu_use = nu[mask]

    pref = (4.0 * np.pi**2 * ALPHA_EM) / K_use
    sigma_T = pref * W1_use
    sigma_L = pref * ((1.0 + (nu_use**2)/Q2) * W2_use - W1_use)

    # Write table
    os.makedirs("sigma_LT_tables", exist_ok=True)
    dat_path = f"sigma_LT_tables/sigma_LT_Q2={Q2:.3f}.dat"
    with open(dat_path, "w") as f:
        f.write("#W\tsigma_L\tsigma_T\n")
        for w, sL, sT in zip(W_use, sigma_L, sigma_T):
            f.write(f"{w:.6f}\t{sL:.8e}\t{sT:.8e}\n")

    # Make and save plot (both curves on one canvas)
    plt.figure()
    plt.plot(W_use, sigma_L, label=r"$\sigma_L$")
    plt.plot(W_use, sigma_T, label=r"$\sigma_T$")
    plt.xlabel("W [GeV]")
    plt.ylabel(r"Cross section $\sigma$ (model units)")
    plt.title(fr"$\sigma_L,\ \sigma_T$ vs $W$ at $Q^2={Q2:.3f}\ \mathrm{{GeV}}^2$")
    plt.legend(loc="upper right")
    plt.grid(True)
    png_path = f"sigma_LT_tables/sigma_LT_Q2={Q2:.3f}.png"
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()
    
    


def sigma_LT_to_F2_AO_model(fixed_Q2,
                            W_out=None,
                            E_beam=10.6,
                            in_dir="tables_from_Yannick/fine_binning/AO",
                            convert_ub_to_GeV2=True,
                            divide_by_Gamma=True):
    """
    Reads:
      tables_from_Yannick/fine_binning/AO/Wdist_Q2_{fixed_Q2}_GLOBAL_LT.dat
    Skips header: 1 line

    Takes columns (0-based):
      W                 = col 0
      (Gamma*sigma_T)   = col 2
      (Gamma*sigma_L)   = col 4   # sigma_L(no-epsilon)

    If divide_by_Gamma=True:
      sigma_T = (Gamma*sigma_T)/Gamma
      sigma_L = (Gamma*sigma_L)/Gamma
    where:
      epsilon = [1 + 2(1 + nu^2/Q2) tan^2(theta/2)]^{-1}
      Gamma   = alpha*E'*(W^2 - M^2)/(4*pi^2*Q2*M*E*(1-epsilon))

    Then computes:
      F2 = (K*M/(4*pi^2*alpha)) * (2x/rho^2) * (sigma_T + sigma_L)
      K = (W^2 - M^2)/(2M)
      x = Q2/(W^2 - M^2 + Q2)
      rho^2 = 1 + 4*M^2*x^2/Q2

    Returns:
      (W, F2) on native grid, or (W_out, F2_interp) if W_out provided.
    """

    Q2 = float(fixed_Q2)
    q2_tag = str(fixed_Q2)
    in_path = os.path.join(in_dir, f"Wdist_Q2_{q2_tag}_GLOBAL_LT.dat")
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"Cannot find input file: {in_path}")

    data = np.loadtxt(in_path, skiprows=1)

    W = data[:, 0]
    sigT_raw = data[:, 2]  # Gamma*sigma_T  
    sigL_raw = data[:, 4]  # Gamma*sigma_L  

    # constants
    M = 0.9382720813
    alpha = 1.0 / 137.035999084

    # kinematics needed for epsilon & Gamma
    nu = (W**2 + Q2 - M**2) / (2.0 * M)
    Eprime = E_beam - nu

    # protect against unphysical points (Eprime<=0 or sin^2>=1)
    F2 = np.full_like(W, np.nan, dtype=float)



    # theta from Q2 = 4 E E' sin^2(theta/2)
    sin2 = Q2 / (4.0 * E_beam * Eprime)


    tan2 = sin2 / (1.0 - sin2)

    eps = 1.0 / (1.0 + 2.0 * (1.0 + (nu**2)/Q2) * tan2)

    # virtual photon flux Gamma (your screenshot form)
    Gamma = (alpha * Eprime * (W**2 - M**2)) / (4.0 * np.pi**2 * Q2 * M * E_beam * (1.0 - eps))
    
    # Jacobian J 
    J = (W * np.pi) / (M * E_beam * Eprime)
    

    if divide_by_Gamma:
        sigT = sigT_raw / (Gamma*J)
        sigL = sigL_raw / (Gamma*J)
    else:
        sigT = sigT_raw
        sigL = sigL_raw

    # ub -> GeV^-2 if needed
    if convert_ub_to_GeV2:
        sigT = sigT / GEV2_TO_UB
        sigL = sigL / GEV2_TO_UB

    # F2 ingredients
    K = (W**2 - M**2) / (2.0 * M)
    x = Q2 / (W**2 - M**2 + Q2)
    rho2 = 1.0 + (4.0 * M**2 * x**2) / Q2
    pref = (K * M) / (4.0 * np.pi**2 * alpha)
    F2 = pref * (2.0 * x / rho2) * (sigT + sigL)

    if W_out is None:
        return W, F2

    W_out = np.asarray(W_out, dtype=float)

    # ---- NEW: cubic spline interpolation in W (no extrapolation) ----
    m = np.isfinite(W) & np.isfinite(F2)
    Wv = np.asarray(W[m], dtype=float)
    F2v = np.asarray(F2[m], dtype=float)

    if Wv.size < 2:
        return W_out, np.full_like(W_out, np.nan, dtype=float)

    # sort and enforce strictly increasing W
    o = np.argsort(Wv)
    Wv = Wv[o]
    F2v = F2v[o]

    Wv_u, idx = np.unique(Wv, return_index=True)
    F2v_u = F2v[idx]

    if Wv_u.size < 2:
        return W_out, np.full_like(W_out, np.nan, dtype=float)

    cs = CubicSpline(Wv_u, F2v_u, bc_type="natural", extrapolate=False)
    F2_out = cs(W_out)  # returns nan outside [min(W), max(W)] because extrapolate=False

    return W_out, F2_out



def calculate_moment_AO_model(Q2_value, region, n=2,
                              E_beam=10.6,
                              in_dir="tables_from_Yannick/fine_binning/AO",
                              convert_ub_to_GeV2=True,
                              divide_by_Gamma=True):
    """
    Truncated Cornwall–Norton moment from AO model:
        M_n(Q2; region) = ∫_{x_lo}^{x_hi} x^{n-2} F2(x,Q2) dx

    Region is defined via W-bounds (same as data), then converted to x-bounds at fixed Q2.

    Uses:
      - cubic spline interpolation in x (CubicSpline)
      - quad integration in x
    """
    

    M = 0.9382720813
    Q2 = float(Q2_value)

    # --- get AO model F2 on native W grid ---
    W, F2W = sigma_LT_to_F2_AO_model(
        fixed_Q2=Q2,
        W_out=None,
        E_beam=E_beam,
        in_dir=in_dir,
        convert_ub_to_GeV2=convert_ub_to_GeV2,
        divide_by_Gamma=divide_by_Gamma
    )

    W = np.asarray(W, dtype=float)
    F2W = np.asarray(F2W, dtype=float)

    # --- convert to x and clean ---
    x = Q2 / (W * W - M * M + Q2)
    mask = np.isfinite(x) & np.isfinite(F2W)
    x = x[mask]
    F2W = F2W[mask]

    if x.size < 2:
        return pd.DataFrame([{
            "Q2": Q2_value, "region": region, "n": n,
            "x_lo": np.nan, "x_hi": np.nan,
            "moment": 0.0, "error": 0.0
        }])

    # --- sort by increasing x and enforce strictly increasing x for CubicSpline ---
    o = np.argsort(x)
    x = x[o]
    F2W = F2W[o]

    x_u, idx = np.unique(x, return_index=True)
    F2_u = F2W[idx]

    if x_u.size < 2:
        return pd.DataFrame([{
            "Q2": Q2_value, "region": region, "n": n,
            "x_lo": np.nan, "x_hi": np.nan,
            "moment": 0.0, "error": 0.0
        }])

    # --- region -> W bounds (same as your original) ---
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

    # --- W -> x bounds at this Q2 ---
    def x_of_W(Wv):
        return Q2 / (Wv * Wv - M * M + Q2)

    xb1 = x_of_W(W_lo)
    xb2 = x_of_W(W_hi)
    x_lo_bound = min(xb1, xb2)
    x_hi_bound = max(xb1, xb2)

    # --- intersect with available AO x-range ---
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

    moment, _ = quad(integrand, lo, hi, epsabs=1e-8, epsrel=1e-6, limit=200)

    return pd.DataFrame([{
        "Q2": Q2_value, "region": region, "n": n,
        "x_lo": lo, "x_hi": hi,
        "moment": float(moment), "error": 0.0
    }])








#-----------------------------Trash probably ----------------------------

def make_sigma_LT_1pi_table(Q2):
    """
    Compute σ_L^{1π}(W) and σ_T^{1π}(W) at fixed Q² using your existing
    interpolate_structure_functions_1pi(file_path, W, Q2) -> (W1, W2),
    then write a table and save a plot.

    Output:
      - sigma_LT_tables/sigma_LT_Q2={Q2:.3f}_1pi.dat  (columns: W, sigma_L, sigma_T)
      - sigma_LT_tables/sigma_LT_Q2={Q2:.3f}_1pi.png  (plot of sigma_L and sigma_T vs W)
    """
    assert Q2 > 0.0, "Q2 must be > 0"


    # Hand flux helpers
    def _nu(W):  # ν = (W^2 + Q^2 - M^2) / (2M)
        return (W**2 + Q2 - M_PROTON**2) / (2.0 * M_PROTON)

    def _K(W):   # Hand's equivalent photon energy: K = (W^2 - M^2) / (2M)
        return (W**2 - M_PROTON**2) / (2.0 * M_PROTON)

    # Read native W grid from the 1π ANL-Osaka table
    file_path = "input_data/wemp-pi.dat"
    data = np.loadtxt(file_path)
    W = np.unique(data[:, 0]).astype(float)

    # Interpolate W1, W2 (1π) on that grid
    W1 = np.empty_like(W)
    W2 = np.empty_like(W)
    for i, w in enumerate(W):
        W1[i], W2[i] = interpolate_structure_functions_1pi(file_path, w, Q2)

    # Convert to sigma_T and sigma_L
    K = _K(W)
    nu = _nu(W)

    # Avoid division by zero or negative K (physically W should be > M)
    mask = K > 0.0
    W_use  = W[mask]
    W1_use = W1[mask]
    W2_use = W2[mask]
    K_use  = K[mask]
    nu_use = nu[mask]

    pref = (4.0 * np.pi**2 * ALPHA_EM) / K_use
    sigma_T = pref * W1_use
    sigma_L = pref * ((1.0 + (nu_use**2)/Q2) * W2_use - W1_use)

    # Write table
    os.makedirs("sigma_LT_tables", exist_ok=True)
    dat_path = f"sigma_LT_tables/sigma_LT_Q2={Q2:.3f}_1pi.dat"
    with open(dat_path, "w") as f:
        f.write("#W\tsigma_L\tsigma_T\n")
        for w, sL, sT in zip(W_use, sigma_L, sigma_T):
            f.write(f"{w:.6f}\t{sL:.8e}\t{sT:.8e}\n")

    # Make and save plot (both curves on one canvas)
    plt.figure()
    plt.plot(W_use, sigma_L, label=r"$\sigma_L^{1\pi}$")
    plt.plot(W_use, sigma_T, label=r"$\sigma_T^{1\pi}$")
    plt.xlabel("W [GeV]")
    plt.ylabel(r"Cross section $\sigma$ (model units)")
    plt.title(r"$\sigma_L^{1 \pi},\ \sigma_T^{1 \pi}$ vs $W$ at $Q^2={Q2:.3f}\ \mathrm{{GeV}}^2$")
    plt.legend(loc="upper right")
    plt.grid(True)
    png_path = f"sigma_LT_tables/sigma_LT_Q2={Q2:.3f}_1pi.png"
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    return dat_path, png_path


def make_dsigma_dWdQ2_full_vs_1pi_plot(Q2, E_beam):
    """
    Compute and plot on ONE canvas:
      - inclusive  d^2σ/dW dQ^2 (from interpolate_structure_functions on wempx.dat)
      - 1π         d^2σ/dW dQ^2 (from interpolate_structure_functions_1pi on wemp-pi.dat)
    using Hand flux and the W–Q² Jacobian. Y-axis is in microbarn/GeV^3.

    Output:
      sigma_LT_tables/dsigma_dWdQ2_Q2={Q2:.3f}_E={E_beam:.3f}_full_vs_1pi.png
    """
    assert Q2 > 0.0 and E_beam > 0.0, "Q2 and E_beam must be > 0"

    # --- constants (GeV units) ---
    ALPHA_EM = 1/137.035999084
    M_PROTON = 0.9382720813
    GEV2_TO_UB = 389.379  # 1 GeV^{-2} = 389.379 microbarn

    # Helpers
    def _nu(W):  # ν = (W^2 + Q^2 - M^2) / (2M)
        return (W**2 + Q2 - M_PROTON**2) / (2.0 * M_PROTON)

    def _K(W):   # Hand: K = (W^2 - M^2) / (2M)
        return (W**2 - M_PROTON**2) / (2.0 * M_PROTON)

    def _sigma_from_W1W2(W, W1, W2):
        """Return (sigma_T, sigma_L) in natural units GeV^{-2}."""
        K = _K(W); nu = _nu(W)
        pref = (4.0 * np.pi**2 * ALPHA_EM) / K
        sigma_T = pref * W1
        sigma_L = pref * ((1.0 + (nu**2)/Q2) * W2 - W1)
        return sigma_T, sigma_L, K, nu

    def _build_curve(file_path, interp_fn):
        """
        Build (W, dsigma_dWdQ2_ub) for either inclusive or 1π:
        - read native W grid
        - interpolate W1,W2
        - compute sigma_T, sigma_L
        - compute ε, Γ_Hand, Jacobian
        - return μb/GeV^3
        """
        data = np.loadtxt(file_path)
        W_all = np.unique(data[:, 0]).astype(float)

        # Interpolate W1, W2 on native W grid
        W1 = np.empty_like(W_all)
        W2 = np.empty_like(W_all)
        for i, w in enumerate(W_all):
            W1[i], W2[i] = interp_fn(file_path, w, Q2)

        # σ_T, σ_L and kinematics
        sigma_T, sigma_L, K, nu = _sigma_from_W1W2(W_all, W1, W2)
        Eprime = E_beam - nu

        # Physical mask: K>0, E'>0, 0<sin^2(θ/2)<1
        mask = (K > 0.0) & (Eprime > 0.0)
        sin2 = np.empty_like(W_all)
        sin2[mask] = Q2 / (4.0 * E_beam * Eprime[mask])
        mask &= (sin2 > 0.0) & (sin2 < 1.0)
        if not np.any(mask):
            return np.array([]), np.array([])

        W = W_all[mask]
        sigma_T = sigma_T[mask]
        sigma_L = sigma_L[mask]
        K = K[mask]
        Eprime = Eprime[mask]
        sin2 = sin2[mask]
        nu_masked = nu[mask]

        # ε and Hand flux
        tan2 = sin2 / (1.0 - sin2)
        eps = 1.0 / (1.0 + 2.0 * (1.0 + (nu_masked**2)/Q2) * tan2)
        Gamma = (ALPHA_EM / (2.0 * np.pi**2)) * (Eprime / E_beam) * (K / Q2) * (1.0 / (1.0 - eps))

        # Jacobian dΩ dE' → dW dQ^2
        J = (np.pi * W) / (M_PROTON * E_beam * Eprime)

        # d^2σ/dW dQ^2 in μb/GeV^3
        dsigma = Gamma * (sigma_T + eps * sigma_L) * J
        dsigma_ub = dsigma * GEV2_TO_UB
        return W, dsigma_ub

    # Inclusive curve (wempx.dat)
    W_full, dsig_full = _build_curve("input_data/wempx.dat", interpolate_structure_functions)
    # 1π curve (wemp-pi.dat)
    W_1pi, dsig_1pi = _build_curve("input_data/wemp-pi.dat", interpolate_structure_functions_1pi)

    # Plot both on one canvas
    os.makedirs("sigma_LT_tables", exist_ok=True)
    png_path = f"sigma_LT_tables/dsigma_dWdQ2_Q2={Q2:.3f}_E={E_beam:.3f}_full_vs_1pi.png"

    plt.figure()
    if W_full.size:
        plt.plot(W_full, dsig_full, label=r"Inclusive $d^2\sigma/dW\,dQ^2$", color="black")
    if W_1pi.size:
        plt.plot(W_1pi, dsig_1pi, label=r"$1\pi$ $d^2\sigma/dW\,dQ^2$",color="black", linestyle="dashed")

    plt.xlabel("W [GeV]")
    plt.ylabel(r"$d^2\sigma/dW\,dQ^2$  [$\mu$b/GeV$^3$]")
    plt.title(fr"$Q^2={Q2:.3f}$ GeV$^2$, $E={E_beam:.3f}$ GeV (full vs 1$\pi$)")
    plt.grid(True); plt.legend(loc="best")

    # Optional visual settings matching your earlier requests:
    plt.xlim(1.0, 2.0)
    plt.ylim(0, 0.0035)
    #ax = plt.gca()
    #ax.yaxis.set_major_locator(MultipleLocator(1e-4))
    #ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))

    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    return png_path

# Testing git


