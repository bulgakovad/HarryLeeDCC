import numpy as np
import math
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
import os
import pandas as pd

"""Functions for interpolating structure functions and computing cross sections for ANL-Osaka model.
"""

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
def compute_cross_section(W, Q2, beam_energy, file_path="input_data/wempx.dat", verbose=True):
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


def calculate_1pi_cross_section(W, Q2, beam_energy, file_path="input_data/wemp-pi.dat", verbose=True):
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

def compute_2pi_cross_section(
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
    dcs_full = compute_cross_section(
        W=W, Q2=Q2, beam_energy=beam_energy,
        file_path=full_file_path, verbose=False
    )

    # Single-pion exclusive contribution
    dcs_1pi = calculate_1pi_cross_section(
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
