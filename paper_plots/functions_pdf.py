import numpy as np
import math
from scipy.interpolate import  interp1d
import os
import pandas as pd

"""
Helper functions for PDF-based structure function calculations
"""

def get_pdf_interpolators_with_error(fixed_Q2, central_iset=400):
    """
    Loads central and error PDF tables for a given fixed Q² and ISET value = 400
    computes F1(W) and F2(W) interpolators, and error bands using Hessian prescription.

    Returns:
        tuple: (F1_W_interp, F2_W_interp, F1_err_func, F2_err_func, W_range)
    """

    Mp = 0.9385
    q2_str = str(fixed_Q2).rstrip("0").rstrip(".")
    folder = f"../get_PDF/output/Q2={q2_str}"

    # Load central PDF
    def load_table(iset):
        filename = f"{folder}/tst_CJpdf_ISET={iset}_Q2={q2_str}.dat"
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"PDF file not found: {filename}")
        return pd.read_csv(filename, sep=r'\s+')

    df0 = load_table(central_iset)
    x = df0['x'].values
    nu0 = df0['u'].values
    nub0 = df0['ub'].values
    nd0 = df0['d'].values
    ndb0 = df0['db'].values

    F2_0 = (4/9)*(nu0 + nub0) + (1/9)*(nd0 + ndb0)
    F1_0 = F2_0 / (2 * x)

    W2 = Mp**2 + fixed_Q2 * (1 - x) / x
    W = np.sqrt(W2)

    sorted = np.argsort(W)
    W_sorted = W[sorted]
    F1_sorted = F1_0[sorted]
    F2_sorted = F2_0[sorted]

    F1_W_interp = interp1d(W_sorted, F1_sorted, kind='cubic', bounds_error=False, fill_value="extrapolate")
    F2_W_interp = interp1d(W_sorted, F2_sorted, kind='cubic', bounds_error=False, fill_value="extrapolate")

    # Load eigenvector variations
    iset_range = range(401, 449)
    

    F1_variations = []
    F2_variations = []

    for iset in iset_range:
        try:
            dfi = load_table(iset)
        except FileNotFoundError:
            continue
        nui = dfi['u'].values
        nubi = dfi['ub'].values
        ndi = dfi['d'].values
        ndbi = dfi['db'].values
        F2_i = (4/9)*(nui + nubi) + (1/9)*(ndi + ndbi)
        F1_i = F2_i / (2 * x)

        F1_variations.append(F1_i[sorted])
        F2_variations.append(F2_i[sorted])

    F1_variations = np.array(F1_variations)
    F2_variations = np.array(F2_variations)

    # Compute symmetric error bands (standard Hessian method)
    F1_err = np.sqrt(np.sum((F1_variations - F1_sorted) ** 2, axis=0))
    F2_err = np.sqrt(np.sum((F2_variations - F2_sorted) ** 2, axis=0))

    # Return error functions (interpolators)
    F1_err_func = interp1d(W_sorted, F1_err, kind='linear', bounds_error=False, fill_value="extrapolate")
    F2_err_func = interp1d(W_sorted, F2_err, kind='linear', bounds_error=False, fill_value="extrapolate")

    return F1_W_interp, F2_W_interp, F1_err_func, F2_err_func, W_sorted

def get_nlo_pdf_interpolators(fixed_Q2):
    """
    Loads Brady NLO tables (F1, F2) for a given fixed Q²
    and returns interpolators for F1_brady, F1_brady_alt,
    F2_brady, and F2_bradyHT.

    Returns:
        tuple: (F1_brady_interp, F1_brady_alt_interp,
                F2_brady_interp, F2_bradyHT_interp, W_sorted)
    """

    folder = "../getF1F2/Output"
    f1_file = f"{folder}/F1_fixQ2_cj15.txt"
    f2_file = f"{folder}/F2_fixQ2_cj15.txt"

    # Load F1 and F2 files
    df1 = pd.read_csv(f1_file, sep=r'\s+', header=None,
                      names=["Q2", "W", "F1_brady", "F1_brady_alt", "F1_bradyHT"])
    df2 = pd.read_csv(f2_file, sep=r'\s+', header=None,
                      names=["Q2", "W", "F2_naked", "F2_moffat", "F2_brady0", "F2_brady", "F2_bradyHT"])

    # Select rows with matching Q²
    mask1 = np.isclose(df1["Q2"].values, fixed_Q2, atol=1e-6)
    mask2 = np.isclose(df2["Q2"].values, fixed_Q2, atol=1e-6)

    if not (mask1.any() and mask2.any()):
        raise ValueError(f"Q²={fixed_Q2} not found in both F1 and F2 files.")

    # Extract W and structure functions
    W1 = df1.loc[mask1, "W"].values
    F1_brady = df1.loc[mask1, "F1_brady"].values
    F1_brady_alt = df1.loc[mask1, "F1_brady_alt"].values
    F1_bradyHT = df1.loc[mask1, "F1_bradyHT"].values

    W2 = df2.loc[mask2, "W"].values
    F2_brady = df2.loc[mask2, "F2_brady"].values
    F2_bradyHT = df2.loc[mask2, "F2_bradyHT"].values

    # Use intersection of W grids to stay consistent
    W_common = np.intersect1d(W1, W2)

    # Sort W grid
    W_sorted = np.sort(W_common)

    # Interpolators
    F1_brady_interp     = interp1d(W1, F1_brady, kind='cubic', bounds_error=False, fill_value="extrapolate")
    F1_brady_alt_interp = interp1d(W1, F1_brady_alt, kind='cubic', bounds_error=False, fill_value="extrapolate")
    F1_bradyHT_interp = interp1d(W1, F1_bradyHT, kind='cubic', bounds_error=False, fill_value="extrapolate")
    
    F2_brady_interp     = interp1d(W2, F2_brady, kind='cubic', bounds_error=False, fill_value="extrapolate")
    F2_bradyHT_interp   = interp1d(W2, F2_bradyHT, kind='cubic', bounds_error=False, fill_value="extrapolate")

    return F1_brady_interp, F1_brady_alt_interp, F1_bradyHT_interp, F2_brady_interp, F2_bradyHT_interp, W_sorted



def compute_cross_section_pdf_with_error(W, Q2, beam_energy,
                                         F1_W_interp, F2_W_interp,
                                         F1_err_func=None, F2_err_func=None):
    """
    Computes the differential cross section and optionally its uncertainty
    using interpolated PDF-based structure functions.

    Returns:
        dσ and (optional) dσ uncertainty (if error functions provided)
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

    F1 = F1_W_interp(W)
    F2 = F2_W_interp(W)
    W1 = F1 / Mp
    W2 = F2 / omeg

    xxx = 2 * ss2 * W1 + cc2 * W2
    dcrs = fcrs3 * fac3 * xxx

    if F1_err_func is not None and F2_err_func is not None:
        F1_err = F1_err_func(W)
        F2_err = F2_err_func(W)
        W1_err = F1_err / Mp
        W2_err = F2_err / omeg
        dcrs_err = fcrs3 * fac3 * np.sqrt((2 * ss2 * W1_err) ** 2 + (cc2 * W2_err) ** 2)
        return dcrs, dcrs_err
    else:
        return dcrs
    
    
def get_nlo_pdf_cross_sections(W, Q2, beam_energy,
                               F1_interp, F2_interp):
    """
    Computes the differential cross section using NLO PDF-based
    structure function interpolators (Brady tables).

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
