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
    ns0  = df0['s'].values   
    nsb0 = df0['sb'].values  
    nc0  = df0['c'].values   
    ncb0 = df0['cb'].values  
    nb0  = df0['b'].values   
    nbb0 = df0['bb'].values  

    F2_0 = ((4/9)*(nu0 + nub0 + nc0 + ncb0) + (1/9)*(nd0 + ndb0 + ns0 + nsb0 + nb0 + nbb0))
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
        nui  = dfi['u'].values
        nubi = dfi['ub'].values
        ndi  = dfi['d'].values
        ndbi = dfi['db'].values
        nsi  = dfi['s'].values   
        nsbi = dfi['sb'].values  
        nci  = dfi['c'].values   
        ncbi = dfi['cb'].values  
        nbi  = dfi['b'].values   
        nbbi = dfi['bb'].values  

        F2_i = ((4/9)*(nui + nubi + nci + ncbi) + (1/9)*(ndi + ndbi + nsi + nsbi + nbi + nbbi))
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
    f1_file = f"{folder}/ALL_Q2_broad_W_F1_cj15.txt"
    f2_file = f"{folder}/ALL_Q2_broad_W_F2_cj15.txt"

    # Load F1 and F2 files
    df1 = pd.read_csv(f1_file, sep=r'\s+', header=None, names=["Q2", "W", "F1_brady", "F1_brady_alt", "F1_bradyHT"])
    df2 = pd.read_csv(f2_file, sep=r'\s+', header=None, names=["Q2", "W", "F2_naked", "F2_moffat", "F2_brady0", "F2_brady", "F2_bradyHT"])

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
    
    
def get_nlo_pdf_cross_sections(W, Q2, beam_energy, F1_interp, F2_interp):
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


def get_lo_pdf_xsecs_table(q2_list, beam_energy,
                           out_dir="lo_pdf_tables",
                           fmt="%.6e"):
    """
    For each Q² in q2_list, compute LO PDF cross sections (with LO error band)
    over the W grid returned by get_pdf_interpolators_with_error, and save a
    3-column table:
        #W    lo_pdf_xsect    error
    Files are saved under out_dir as: lo_pdf_xsecs_Q2={Q2}_E={E}.dat

    Args:
        q2_list (iterable): list/tuple of Q² values (GeV²)
        beam_energy (float): beam energy (GeV)
        out_dir (str): output directory
        fmt (str): numpy savetxt format for numbers (default scientific: '%.6e')

    Returns:
        list of str: paths to the written files
    """
    import os
    import numpy as np

    os.makedirs(out_dir, exist_ok=True)
    out_paths = []

    for q2 in q2_list:
        try:
            F1_W, F2_W, F1_err_f, F2_err_f, W_range = get_pdf_interpolators_with_error(q2, central_iset=400)
        except Exception as e:
            print(f"[WARN] Skipping Q²={q2}: failed to build LO interpolators ({e})")
            continue

        W_vals, sig_vals, err_vals = [], [], []

        for W in W_range:
            try:
                sigma, sigma_err = compute_cross_section_pdf_with_error(
                    W, q2, beam_energy, F1_W, F2_W, F1_err_f, F2_err_f
                )
                W_vals.append(W)
                sig_vals.append(sigma)
                err_vals.append(sigma_err)
            except Exception:
                # kinematically invalid point (e.g., W>wtot or E'<0) → skip row
                continue

        if len(W_vals) == 0:
            print(f"[WARN] No valid points for Q²={q2} at E={beam_energy}. Skipping file.")
            continue

        W_vals = np.asarray(W_vals, dtype=float)
        sig_vals = np.asarray(sig_vals, dtype=float)
        err_vals = np.asarray(err_vals, dtype=float)

        table = np.column_stack([W_vals, sig_vals, err_vals])

        q2_str = str(q2).rstrip("0").rstrip(".")
        fname = f"lo_pdf_xsecs_Q2={q2_str}_E={beam_energy}.dat"
        out_path = os.path.join(out_dir, fname)

        header = "#W\tlo_pdf_xsect\terror"
        np.savetxt(out_path, table, fmt=fmt, delimiter="\t", header=header, comments="")

        out_paths.append(out_path)
        print(f"Saved → {out_path}")

    return out_paths

def get_nlo_pdf_xsecs_table(fixed_Q2, beam_energy,
                             out_dir="pdf_tables",
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
    import os
    import numpy as np

    # Build NLO interpolators
    F1_brady, F1_brady_alt, F1_bradyHT, F2_brady, F2_HT, W_common = get_nlo_pdf_interpolators(fixed_Q2)

    # Choose F1
    F1_use = F1_brady_alt if use_F1_alt else F1_brady
    F1_use_ht = F1_bradyHT

    # Choose W grid
    if W_vals is None:
        W_vals = np.asarray(W_common, dtype=float)
    else:
        W_vals = np.asarray(W_vals, dtype=float)

    # Compute cross sections
    tmc_vals = []
    tmc_ht_vals = []
    for W in W_vals:
        try:
            tmc_vals.append(get_nlo_pdf_cross_sections(W, fixed_Q2, beam_energy,
                                                       F1_interp=F1_use, F2_interp=F2_brady))
        except Exception:
            tmc_vals.append(np.nan)
        try:
            tmc_ht_vals.append(get_nlo_pdf_cross_sections(W, fixed_Q2, beam_energy,
                                                          F1_interp=F1_use_ht, F2_interp=F2_HT))
        except Exception:
            tmc_ht_vals.append(np.nan)

    tmc_vals = np.asarray(tmc_vals, dtype=float)
    tmc_ht_vals = np.asarray(tmc_ht_vals, dtype=float)

    # Assemble table: Q2, W, TMC_xsection, TMC_HT_xsection
    Q2_col = np.full_like(W_vals, float(fixed_Q2), dtype=float)
    table = np.column_stack([Q2_col, W_vals, tmc_vals, tmc_ht_vals])

    # Save
    os.makedirs(out_dir, exist_ok=True)
    if filename is None:
        q2_str = str(fixed_Q2).rstrip("0").rstrip(".")
        filename = f"pdf_xsecs_Q2={q2_str}_E={beam_energy}.dat"
    out_path = os.path.join(out_dir, filename)

    header = "Q2\tW\tTMC_xsection\tTMC_HT_xsection"
    np.savetxt(out_path, table, fmt="%.6e", delimiter="\t", header=header, comments="")

    return out_path


def get_pdf_struct_func_table(Q2_list, vs_what="x"):
    """
    For each Q² in Q2_list, write:
      pdf_based_struct_func_LO_NLO/pdf_struct_func_Q2={Q2}.dat

    Columns (tab-separated):
      x (or W), F1_NLO_TMC_only, F1_NLO_TMC_only_alternative, F1_NLO_TMC_HT_prelim,
      F2_NLO_TMC_only, F2_NLO_TMC_HT, F1_LO, F2_LO

    Uses:
      - get_nlo_pdf_interpolators(fixed_Q2)  -> F1_brady, F1_brady_alt, F1_bradyHT, F2_brady, F2_bradyHT, W_nlo
      - get_pdf_interpolators_with_error(fixed_Q2) -> F1_LO, F2_LO, (errs...), W_lo

    If a given Q² is missing in either source, it is skipped.
    """
    import os
    import numpy as np

    out_dir = "pdf_based_struct_func_LO_NLO"
    os.makedirs(out_dir, exist_ok=True)

    Mp = 0.9385

    for Q2 in Q2_list:
        try:
            # NLO (Brady)
            F1_b, F1_b_alt, F1_b_HT, F2_b, F2_b_HT, W_nlo = get_nlo_pdf_interpolators(Q2)
            # LO
            F1_lo, F2_lo, _, _, W_lo = get_pdf_interpolators_with_error(Q2, central_iset=400)
        except Exception as e:
            print(f"[WARN] Q²={Q2}: skipping (failed to build interpolators) -> {e}")
            continue

        # Shared W grid (union). Keep it simple and let interpolators extrapolate if needed.
        W_grid = np.unique(np.concatenate([np.asarray(W_nlo, dtype=float),
                                           np.asarray(W_lo,  dtype=float)]))
        # Evaluate structure functions at W
        F1_nlo_tmc      = F1_b(W_grid)
        F1_nlo_tmc_alt  = F1_b_alt(W_grid)
        F1_nlo_tmc_ht   = F1_b_HT(W_grid)
        F2_nlo_tmc      = F2_b(W_grid)
        F2_nlo_tmc_ht   = F2_b_HT(W_grid)
        F1_lo_vals      = F1_lo(W_grid)
        F2_lo_vals      = F2_lo(W_grid)

        # Choose output abscissa
        if str(vs_what).lower() == "x":
            # x_Bj(Q2, W) = Q2 / (W^2 - M^2 + Q2)
            denom = (W_grid**2 - Mp**2 + Q2)
            x_vals = Q2 / denom
            # Keep only physically sane x>0 (avoid zeros/negatives)
            mask = x_vals > 0
            x_vals = x_vals[mask]
            # Apply the same mask to all columns
            cols = [
                x_vals,
                F1_nlo_tmc[mask], F1_nlo_tmc_alt[mask], F1_nlo_tmc_ht[mask],
                F2_nlo_tmc[mask], F2_nlo_tmc_ht[mask],
                F1_lo_vals[mask], F2_lo_vals[mask],
            ]
            # Sort by x ascending
            order = np.argsort(x_vals)
            cols = [c[order] for c in cols]
            header_first = "x"
        else:
            # vs W
            cols = [
                W_grid,
                F1_nlo_tmc, F1_nlo_tmc_alt, F1_nlo_tmc_ht,
                F2_nlo_tmc, F2_nlo_tmc_ht,
                F1_lo_vals, F2_lo_vals,
            ]
            header_first = "W"

        table = np.column_stack(cols)

        # Save
        q2_str = str(Q2).rstrip("0").rstrip(".")
        out_path = os.path.join(out_dir, f"pdf_struct_func_Q2={q2_str}.dat")
        header = (
            f"{header_first}\t"
            "F1_NLO_TMC_only\tF1_NLO_TMC_only_alternative\tF1_NLO_TMC_HT_prelim\t"
            "F2_NLO_TMC_only\tF2_NLO_TMC_HT\tF1_LO\tF2_LO"
        )
        np.savetxt(out_path, table, fmt="%.6e", delimiter="\t", header=header, comments="")
        print("Saved →", out_path)


#get_pdf_struct_func_table([0.5, 0.75, 1.0, 1.75, 2.0, 2.5, 2.774, 3.0], vs_what="x")