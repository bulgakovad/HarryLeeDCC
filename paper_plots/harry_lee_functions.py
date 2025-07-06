import numpy as np
import math
from scipy.interpolate import RectBivariateSpline, interp1d
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import os
import pandas as pd

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

def generate_table_struct_funcs(file_path, fixed_Q2, onepi_file="input_data/wemp-pi.dat"):
    """
    Generates a table of structure functions W1 and W2 for full and 1π channels,
    for a fixed Q². The format is:
        W   W1_full   W1_1pi   W2_full   W2_1pi

    Parameters:
        file_path (str): Path to the full-channel structure function file (W, Q², W1, W2)
        fixed_Q2 (float): Q² value for which to extract and interpolate structure functions
        onepi_file (str): Path to 1π-channel structure function file
    """
    # Load full-channel data
    data = np.loadtxt(file_path)
    W = data[:, 0]
    Q2 = data[:, 1]
    W1 = data[:, 2]
    W2 = data[:, 3]

    # Identify the closest grid Q² value
    Q2_unique = np.unique(Q2)
    idx = np.argmin(np.abs(Q2_unique - fixed_Q2))
    grid_Q2 = Q2_unique[idx]
    tol = 1e-6
    rows = data[np.abs(Q2 - grid_Q2) < tol]

    output_rows = []
    for row in rows:
        W_val = row[0]
        W1_full = row[2]
        W2_full = row[3]

        # Try to interpolate W1 and W2 for 1π at this W, Q²
        try:
            W1_1pi, W2_1pi = interpolate_structure_functions_1pi(onepi_file, W_val, fixed_Q2)
        except Exception:
            W1_1pi, W2_1pi = float('nan'), float('nan')

        output_rows.append([W_val, W1_full, W1_1pi, W2_full, W2_1pi])

    # Save to file
    header = "W\tW1_full\tW1_1pi\tW2_full\tW2_1pi"
    output_filename = f"tables_struct_funcs/struct_funcs_W12_for_Q2={fixed_Q2}.txt"
    np.savetxt(output_filename, np.array(output_rows), header=header, fmt="%.6e", delimiter="\t")
    print(f"Table saved as {output_filename}")

def generate_table_xsecs(file_path, fixed_Q2, beam_energy, onepi_file="input_data/wemp-pi.dat"):
    """
    Generates a table of differential cross sections dσ/dWdQ² for full and 1π channels,
    at a fixed Q². The format is:
        W   xsec_full   xsec_1pi

    Parameters:
        file_path (str): Path to the full-channel structure function file (W, Q², W1, W2)
        fixed_Q2 (float): Q² value at which cross sections are evaluated
        beam_energy (float): Beam energy (E) in GeV
        onepi_file (str): Path to 1π-channel structure function file
    """
    data = np.loadtxt(file_path)
    W = data[:, 0]
    Q2 = data[:, 1]

    Q2_unique = np.unique(Q2)
    idx = np.argmin(np.abs(Q2_unique - fixed_Q2))
    grid_Q2 = Q2_unique[idx]
    tol = 1e-6
    rows = data[np.abs(Q2 - grid_Q2) < tol]

    output_rows = []
    for row in rows:
        W_val = row[0]

        try:
            xsec_full = compute_cross_section(W_val, fixed_Q2, beam_energy, file_path, verbose=False)
        except Exception:
            xsec_full = float('nan')

        try:
            xsec_1pi = calculate_1pi_cross_section(W_val, fixed_Q2, beam_energy, onepi_file, verbose=False)
        except Exception:
            xsec_1pi = float('nan')

        output_rows.append([W_val, xsec_full, xsec_1pi])

    header = "W\txsec_full\txsec_1pi"
    output_filename = f"tables_xsecs/xsecs_for_Q2={fixed_Q2}.txt"
    os.makedirs("tables_xsecs", exist_ok=True)
    np.savetxt(output_filename, np.array(output_rows), header=header, fmt="%.6e", delimiter="\t")
    print(f"Table saved as {output_filename}")

    

def compare_strfun(fixed_Q2, beam_energy,
                   interp_file="input_data/wempx.dat",
                   onepi_file="input_data/wemp-pi.dat",
                   num_points=200):
    """
    Compare ANL-Osaka, 1π, PDF (LO and NLO), and data (strfun + RGA) cross sections up to W=2 GeV.
    """

    from scipy.interpolate import interp1d
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    W_cutoff = 2.0
    data = np.loadtxt(interp_file)
    W_grid = np.unique(data[:, 0])
    W_vals = np.linspace(W_grid.min(), min(W_grid.max(), W_cutoff), num_points)

    anl_xs, onepi_xs = [], []
    pdf_lo_xs, pdf_nlo_xs = [], []
    pdf_lo_err, pdf_nlo_err = [], []

    # Get PDF interpolators with error bands
    F1_lo, F2_lo, F1_lo_err, F2_lo_err, _ = get_pdf_interpolators_with_error(fixed_Q2, central_iset=400)
    F1_nlo, F2_nlo, F1_nlo_err, F2_nlo_err, _ = get_pdf_interpolators_with_error(fixed_Q2, central_iset=500)

    for w in W_vals:
        anl_xs.append(compute_cross_section(w, fixed_Q2, beam_energy, file_path=interp_file, verbose=False))

        try:
            onepi_xs.append(calculate_1pi_cross_section(w, fixed_Q2, beam_energy, file_path=onepi_file, verbose=False))
        except Exception:
            onepi_xs.append(np.nan)

        try:
            pdf_val, pdf_err = compute_cross_section_pdf_with_error(w, fixed_Q2, beam_energy,
                                                                    F1_lo, F2_lo,
                                                                    F1_lo_err, F2_lo_err)
            pdf_lo_xs.append(pdf_val)
            pdf_lo_err.append(pdf_err)
        except Exception:
            pdf_lo_xs.append(np.nan)
            pdf_lo_err.append(np.nan)

        try:
            pdf_val, pdf_err = compute_cross_section_pdf_with_error(w, fixed_Q2, beam_energy,
                                                                    F1_nlo, F2_nlo,
                                                                    F1_nlo_err, F2_nlo_err)
            pdf_nlo_xs.append(pdf_val)
            pdf_nlo_err.append(pdf_err)
        except Exception:
            pdf_nlo_xs.append(np.nan)
            pdf_nlo_err.append(np.nan)

    anl_xs      = np.asarray(anl_xs)
    onepi_xs    = np.asarray(onepi_xs)
    pdf_lo_xs   = np.asarray(pdf_lo_xs)
    pdf_nlo_xs  = np.asarray(pdf_nlo_xs)
    pdf_lo_err  = np.asarray(pdf_lo_err)
    pdf_nlo_err = np.asarray(pdf_nlo_err)

    # Load strfun data
    meas_file = f"strfun_data/cs_Q2={fixed_Q2}_E={beam_energy}.dat"
    if not os.path.isfile(meas_file):
        raise FileNotFoundError(meas_file + " not found")
    mdat = np.genfromtxt(meas_file, names=["W", "Quantity", "Uncertainty"], delimiter="\t", skip_header=1)
    mask_meas = mdat["W"] <= W_cutoff
    W_meas = mdat["W"][mask_meas]
    cs_meas = mdat["Quantity"][mask_meas]
    err_meas = mdat["Uncertainty"][mask_meas]

    # Sort for interpolation
    sort_idx = np.argsort(W_meas)
    W_sorted = W_meas[sort_idx]
    cs_sorted = cs_meas[sort_idx]
    err_sorted = err_meas[sort_idx]

    cs_interp = interp1d(W_sorted, cs_sorted, kind='cubic', bounds_error=False, fill_value="extrapolate")
    err_interp = interp1d(W_sorted, err_sorted, kind='cubic', bounds_error=False, fill_value="extrapolate")

    # Do NOT extrapolate smoothed curve beyond the data
    W_data_min = W_sorted[0]
    W_data_max = W_sorted[-1]
    mask_within_data = (W_vals >= W_data_min) & (W_vals <= W_data_max)
    W_vals_data = W_vals[mask_within_data]

    cs_vals = cs_interp(W_vals_data)
    err_vals = err_interp(W_vals_data)

    # Load Klimenko data if Q² ~ 2.774
    plot_rga = abs(fixed_Q2 - 2.774) < 1e-3
    if plot_rga:
        rga_file = "exp_data/InclusiveExpValera_Q2=2.774.dat"
        if not os.path.isfile(rga_file):
            raise FileNotFoundError(f"Expected RGA data {rga_file} not found")
        rga_data = np.genfromtxt(rga_file,
                                 names=["W", "eps", "sigma", "error", "sys_error"],
                                 delimiter="\t", skip_header=1)
        mask_rga = rga_data["W"] <= W_cutoff
        W_rga = rga_data["W"][mask_rga]
        sigma_rga = rga_data["sigma"][mask_rga] * 1e-3
        err_rga = np.sqrt(rga_data["error"][mask_rga]**2 + rga_data["sys_error"][mask_rga]**2) * 1e-3

    # ----- Plot -----
    plt.figure(figsize=(8, 6))

    h_anl, = plt.plot(W_vals, anl_xs, label="ANL-Osaka model: full cross section", color="black", lw=2)
    good = ~np.isnan(onepi_xs)
    h_1pi, = plt.plot(W_vals[good], onepi_xs[good], label="ANL-Osaka model: 1π contribution", color="black", ls="--", lw=2)
    #h_strfun_line, = plt.plot(W_vals_data, cs_vals, color="grey", lw=2, label="CLAS+Wodld data smoothed")
   # plt.fill_between(W_vals_data, cs_vals - err_vals, cs_vals + err_vals, color="grey", alpha=0.3)
    #h_strfun_pts = plt.errorbar(W_meas, cs_meas, yerr=err_meas, fmt="o", color="green", capsize=1, ms=2, label="strfun (points)")

    # PDF LO with error band
    good_lo = ~np.isnan(pdf_lo_xs)
   # h_pdf_lo, = plt.plot(W_vals[good_lo], pdf_lo_xs[good_lo], label="LO PDF (CJ15)", color="blue", ls="dotted", lw=1)
   # plt.fill_between(W_vals[good_lo],
               #      pdf_lo_xs[good_lo] - pdf_lo_err[good_lo],
                 #    pdf_lo_xs[good_lo] + pdf_lo_err[good_lo],
                 #    color="blue", alpha=0.3)

    # PDF NLO with error band
    #good_nlo = ~np.isnan(pdf_nlo_xs)
   # h_pdf_nlo, = plt.plot(W_vals[good_nlo], pdf_nlo_xs[good_nlo], label="NLO PDF (CJ15)", color="blue", ls="dotted", lw=1)
   # plt.fill_between(W_vals[good_nlo], pdf_nlo_xs[good_nlo] - pdf_nlo_err[good_nlo], pdf_nlo_xs[good_nlo] + pdf_nlo_err[good_nlo], color="blue", alpha=0.3)

    if plot_rga and len(W_rga) > 0:
        h_rga = plt.errorbar(
            W_rga, sigma_rga, yerr=err_rga,
            fmt="s", color="magenta", capsize=1, ms=2,
            label="RGA data (V. Klimenko)"
        )

    handles = [plt.Line2D([], [], color='white', label=f"Q² = {fixed_Q2:.3f} GeV², E = {beam_energy} GeV"),
           h_anl,
            h_1pi,
           #h_pdf_lo,
           #h_pdf_nlo,
           #h_strfun_pts,
           #h_strfun_line
               ]
    if plot_rga and len(W_rga) > 0:
        handles.append(h_rga)

    plt.xlabel("W (GeV)")
    plt.ylabel(r"$d \sigma / dW dQ^2$ ($\mathrm{\mu bn/GeV^3}$)")
    plt.grid(True)
    plt.legend(handles=handles, loc="upper left", fontsize="small")

    os.makedirs("compare_strfun", exist_ok=True)
    fname = f"compare_strfun/compare_strfun_Q2={fixed_Q2}_E={beam_energy}.pdf"
    plt.savefig(fname, dpi=300)
    plt.close()
    print("Saved →", fname)


    
def get_pdf_interpolators_with_error(fixed_Q2, central_iset=400):
    """
    Loads central and error PDF tables for a given fixed Q² and ISET value (400 or 500),
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
    if central_iset == 400:
        iset_range = range(401, 449)
    elif central_iset == 500:
        iset_range = range(501, 549)
    else:
        raise ValueError("Expected central_iset to be 400 (LO) or 500 (NLO).")

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
    
    
def compare_exp_model_pdf_bjorken_x(fixed_Q2, beam_energy,
                                    interp_file="input_data/wempx.dat",
                                    onepi_file="input_data/wemp-pi.dat",
                                    num_points=200):


    Mp = 0.9385
    

    valid_Q2_list = [3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699]
    plot_rga = abs(fixed_Q2 - 2.774) < 1e-3
    only_pdf_and_klim = (fixed_Q2 > 3.0) and any(abs(fixed_Q2 - q2) < 1e-3 for q2 in valid_Q2_list)

    # W → x
    W_cutoff = 2.5 if only_pdf_and_klim else 2.0
    W_vals = np.linspace(1.1, W_cutoff, num_points)
    x_vals = fixed_Q2 / (W_vals**2 - Mp**2 + fixed_Q2)
    mask = x_vals > 0
    W_vals = W_vals[mask]
    x_vals = x_vals[mask]

    # Load PDFs
    F1_lo, F2_lo, F1_lo_err, F2_lo_err, _ = get_pdf_interpolators_with_error(fixed_Q2, central_iset=400)
    pdf_lo_xs, pdf_lo_err = [], []

    for w in W_vals:
        try:
            lo, lo_err = compute_cross_section_pdf_with_error(w, fixed_Q2, beam_energy, F1_lo, F2_lo, F1_lo_err, F2_lo_err)
        except Exception:
            lo, lo_err = np.nan, np.nan
        pdf_lo_xs.append(lo)
        pdf_lo_err.append(lo_err)

    jacobian = fixed_Q2 / (2.0 * W_vals * x_vals**2)
    pdf_lo_xs = np.array(pdf_lo_xs) * jacobian
    pdf_lo_err = np.array(pdf_lo_err) * jacobian

    anl_xs, onepi_xs = [], []
    if not only_pdf_and_klim:
        # Load strfun data
        strfun_file = f"strfun_data/cs_Q2={fixed_Q2}_E={beam_energy}.dat"
        if not os.path.isfile(strfun_file):
            raise FileNotFoundError(f"Strfun data not found: {strfun_file}")
        strfun_data = np.genfromtxt(strfun_file, delimiter="\t", skip_header=1)
        W_exp = strfun_data[:, 0]
        sigma_exp = strfun_data[:, 1]
        sigma_err = strfun_data[:, 2]

        x_exp = fixed_Q2 / (W_exp**2 - Mp**2 + fixed_Q2)
        dWdx_exp = fixed_Q2 / (2.0 * W_exp * x_exp**2)
        sigma_dx = sigma_exp * dWdx_exp
        sigma_dx_err = sigma_err * dWdx_exp

        # Interpolation of strfun data
        sort_idx = np.argsort(x_exp)
        x_sorted = x_exp[sort_idx]
        sigma_sorted = sigma_dx[sort_idx]
        err_sorted = sigma_dx_err[sort_idx]

        cs_interp = interp1d(x_sorted, sigma_sorted, kind='cubic', bounds_error=False, fill_value=np.nan)
        err_interp = interp1d(x_sorted, err_sorted, kind='cubic', bounds_error=False, fill_value=np.nan)

        cs_vals = cs_interp(x_vals)
        err_vals = err_interp(x_vals)

        # Load ANL + 1π
        for w in W_vals:
            try:
                anl = compute_cross_section(w, fixed_Q2, beam_energy, file_path=interp_file, verbose=False)
            except Exception:
                anl = np.nan
            try:
                onepi = calculate_1pi_cross_section(w, fixed_Q2, beam_energy, file_path=onepi_file, verbose=False)
            except Exception:
                onepi = np.nan
            anl_xs.append(anl)
            onepi_xs.append(onepi)
        anl_xs = np.array(anl_xs) * jacobian
        onepi_xs = np.array(onepi_xs) * jacobian

    # Klimenko data if needed
    if plot_rga or only_pdf_and_klim:
        klim_file = f"exp_data/InclusiveExpValera_Q2={fixed_Q2}.dat"
        if not os.path.isfile(klim_file):
            raise FileNotFoundError("Klimenko RGA data not found.")
        klim = np.genfromtxt(klim_file, delimiter="\t", skip_header=1)
        W_klim = klim[:, 0]
        sigma_klim = klim[:, 2] * 1e-3
        err_stat = klim[:, 3] * 1e-3
        err_sys  = klim[:, 4] * 1e-3
        err_total = np.sqrt(err_stat**2 + err_sys**2)
        x_klim = fixed_Q2 / (W_klim**2 - Mp**2 + fixed_Q2)
        dWdx_klim = fixed_Q2 / (2.0 * W_klim * x_klim**2)
        sigma_klim_dx = sigma_klim * dWdx_klim
        err_klim_dx = err_total * dWdx_klim

    # ----------- PLOT ------------
    plt.figure(figsize=(8, 6))
    legend_text = f"$Q^2$ = {fixed_Q2:.3f} GeV$^2$, $E_{{beam}}$ = {beam_energy:.2f} GeV"
    plt.plot([], [], ' ', label=legend_text)

    if not only_pdf_and_klim:
        plt.plot(x_vals, anl_xs, label="ANL-Osaka model: full cross section", color="black")
        plt.plot(x_vals, onepi_xs, "--", label="ANL-Osaka model: $1\\pi$ contribution", color="black", ls="--")
        plt.plot(x_vals, cs_vals, color="grey", lw=1, label="CLAS+World data (smoothed)")
        plt.fill_between(x_vals, cs_vals - err_vals, cs_vals + err_vals, color="grey", alpha=0.3)

    plt.plot(x_vals, pdf_lo_xs, label="LO PDF (CJ15)", color="blue", ls="dotted", lw=1)
    plt.fill_between(x_vals, pdf_lo_xs - pdf_lo_err, pdf_lo_xs + pdf_lo_err, color="blue", alpha=0.3)

    if plot_rga or only_pdf_and_klim:
        plt.errorbar(x_klim, sigma_klim_dx, yerr=err_klim_dx, fmt="s", ms=3, capsize=2, color="magenta", label="RGA data (V. Klimenko)")

    plt.xlabel("Bjorken x")
    plt.ylabel(r"$d\sigma/dQ^2dx$ [$\mu b/GeV^2$]")
    plt.grid(True)
    plt.legend(fontsize="small")

    # Axis limits
    xlim_min = np.min(x_vals) * 0.95
    xlim_max = np.max(x_vals) * 1.05
    ylim_max = 1.05 * np.nanmax([
        np.nanmax(pdf_lo_xs + pdf_lo_err),
        np.nanmax(sigma_klim_dx) if (plot_rga or only_pdf_and_klim) else 0,
        np.nanmax(cs_vals + err_vals) if not only_pdf_and_klim else 0,
        np.nanmax(anl_xs) if not only_pdf_and_klim else 0])
    plt.xlim(xlim_min, xlim_max)
    plt.ylim(0, ylim_max)
    plt.tight_layout()

    # Save plot
    os.makedirs("compare_strfun_x_xi", exist_ok=True)
    fname = f"compare_strfun_x_xi/compare_strfun_vs_x_Q2={fixed_Q2}_E={beam_energy}.pdf"
    plt.savefig(fname, dpi=300)
    plt.close()
    print("Saved →", fname)



def compare_exp_model_pdf_nachtmann_xi(fixed_Q2, beam_energy,
                                       interp_file="input_data/wempx.dat",
                                       onepi_file="input_data/wemp-pi.dat",
                                       num_points=200):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d

    Mp = 0.9385
    

    def xi_from_x(x, Q2):
        t = 4 * x**2 * Mp**2 / Q2
        return 2 * x / (1 + np.sqrt(1 + t))

    def dxdxi(x, Q2):
        t = 4 * x**2 * Mp**2 / Q2
        return 0.5 * (1 + t + np.sqrt(1 + t))

    def dWdx(W, x):
        return (W**2 + fixed_Q2 - Mp**2) / (2.0 * W * x)

    valid_Q2_list = [3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699]
    plot_rga = abs(fixed_Q2 - 2.774) < 1e-3
    only_pdf_and_klim = (fixed_Q2 > 3.0) and any(abs(fixed_Q2 - q2) < 1e-3 for q2 in valid_Q2_list)

    # W → x → ξ
    W_cutoff = 2.5 if only_pdf_and_klim else 2.0
    W_vals = np.linspace(1.1, W_cutoff, num_points)
    x_vals = fixed_Q2 / (W_vals**2 - Mp**2 + fixed_Q2)
    mask = x_vals > 0
    W_vals = W_vals[mask]
    x_vals = x_vals[mask]
    xi_vals = xi_from_x(x_vals, fixed_Q2)

    F1_lo, F2_lo, F1_lo_err, F2_lo_err, _ = get_pdf_interpolators_with_error(fixed_Q2, central_iset=400)

    anl_xs, onepi_xs = [], []
    pdf_lo_xs, pdf_lo_err = [], []

    for w in W_vals:
        try:
            lo, lo_err = compute_cross_section_pdf_with_error(w, fixed_Q2, beam_energy, F1_lo, F2_lo, F1_lo_err, F2_lo_err)
        except Exception:
            lo, lo_err = np.nan, np.nan
        pdf_lo_xs.append(lo)
        pdf_lo_err.append(lo_err)

        if not only_pdf_and_klim:
            try:
                anl = compute_cross_section(w, fixed_Q2, beam_energy, file_path=interp_file, verbose=False)
            except Exception:
                anl = np.nan
            try:
                onepi = calculate_1pi_cross_section(w, fixed_Q2, beam_energy, file_path=onepi_file, verbose=False)
            except Exception:
                onepi = np.nan
            anl_xs.append(anl)
            onepi_xs.append(onepi)

    dWdx_vals = dWdx(W_vals, x_vals)
    dxdxi_vals = dxdxi(x_vals, fixed_Q2)
    dWdXi_vals = dWdx_vals * dxdxi_vals
    pdf_lo_xs = np.array(pdf_lo_xs) * dWdXi_vals
    pdf_lo_err = np.array(pdf_lo_err) * dWdXi_vals

    if not only_pdf_and_klim:
        anl_xs = np.array(anl_xs) * dWdXi_vals
        onepi_xs = np.array(onepi_xs) * dWdXi_vals

        strfun_file = f"strfun_data/cs_Q2={fixed_Q2}_E={beam_energy}.dat"
        if not os.path.isfile(strfun_file):
            raise FileNotFoundError(f"Strfun data not found: {strfun_file}")
        strfun_data = np.genfromtxt(strfun_file, delimiter="\t", skip_header=1)
        W_exp = strfun_data[:, 0]
        sigma_exp = strfun_data[:, 1]
        sigma_err = strfun_data[:, 2]

        x_exp = fixed_Q2 / (W_exp**2 - Mp**2 + fixed_Q2)
        xi_exp = xi_from_x(x_exp, fixed_Q2)
        dWdx_exp = dWdx(W_exp, x_exp)
        dxdxi_exp = dxdxi(x_exp, fixed_Q2)
        dWdXi_exp = dWdx_exp * dxdxi_exp
        sigma_dxi = sigma_exp * dWdXi_exp
        sigma_dxi_err = sigma_err * dWdXi_exp

        sort_idx = np.argsort(xi_exp)
        xi_sorted = xi_exp[sort_idx]
        sigma_sorted = sigma_dxi[sort_idx]
        err_sorted = sigma_dxi_err[sort_idx]

        cs_interp = interp1d(xi_sorted, sigma_sorted, kind='cubic', bounds_error=False, fill_value=np.nan)
        err_interp = interp1d(xi_sorted, err_sorted, kind='cubic', bounds_error=False, fill_value=np.nan)

        cs_vals = cs_interp(xi_vals)
        err_vals = err_interp(xi_vals)

    if plot_rga or only_pdf_and_klim:
        klim_file = f"exp_data/InclusiveExpValera_Q2={fixed_Q2}.dat"
        if not os.path.isfile(klim_file):
            raise FileNotFoundError("Klimenko RGA data not found.")
        klim = np.genfromtxt(klim_file, delimiter="\t", skip_header=1)
        W_klim = klim[:, 0]
        sigma_klim = klim[:, 2] * 1e-3
        err_stat = klim[:, 3] * 1e-3
        err_sys = klim[:, 4] * 1e-3
        err_total = np.sqrt(err_stat**2 + err_sys**2)

        x_klim = fixed_Q2 / (W_klim**2 - Mp**2 + fixed_Q2)
        xi_klim = xi_from_x(x_klim, fixed_Q2)
        dWdx_klim = dWdx(W_klim, x_klim)
        dxdxi_klim = dxdxi(x_klim, fixed_Q2)
        dWdXi_klim = dWdx_klim * dxdxi_klim
        sigma_klim_dxi = sigma_klim * dWdXi_klim
        err_klim_dxi = err_total * dWdXi_klim

    # Plot
    plt.figure(figsize=(8, 6))
    legend_text = f"$Q^2$ = {fixed_Q2:.3f} GeV$^2$, $E_{{beam}}$ = {beam_energy:.2f} GeV"
    plt.plot([], [], ' ', label=legend_text)

    if not only_pdf_and_klim:
        plt.plot(xi_vals, anl_xs, label="ANL-Osaka model: full cross section", color="black")
        plt.plot(xi_vals, onepi_xs, "--", label="ANL-Osaka model: $1\\pi$ contribution", color="black", ls="--")
        plt.plot(xi_vals, cs_vals, color="grey", lw=1, label="CLAS+World data (smoothed)")
        plt.fill_between(xi_vals, cs_vals - err_vals, cs_vals + err_vals, color="grey", alpha=0.3)

    plt.plot(xi_vals, pdf_lo_xs, label="LO PDF (CJ15)", color="blue", ls="dotted", lw=1)
    plt.fill_between(xi_vals, pdf_lo_xs - pdf_lo_err, pdf_lo_xs + pdf_lo_err, color="blue", alpha=0.3)

    if plot_rga or only_pdf_and_klim:
        plt.errorbar(xi_klim, sigma_klim_dxi, yerr=err_klim_dxi, fmt="s", ms=3, capsize=2, color="magenta", label="RGA data (V. Klimenko)")

    plt.xlabel("Nachtmann ξ")

    xlim_min = np.min(xi_vals) * 0.95
    xlim_max = np.max(xi_vals) * 1.05
    ylim_max = 1.05 * np.nanmax([
        np.nanmax(pdf_lo_xs + pdf_lo_err),
        np.nanmax(sigma_klim_dxi) if (plot_rga or only_pdf_and_klim) else 0,
        np.nanmax(cs_vals + err_vals) if not only_pdf_and_klim else 0,
        np.nanmax(anl_xs) if not only_pdf_and_klim else 0
    ])
    plt.xlim(xlim_min, xlim_max)
    plt.ylim(0, ylim_max)
    plt.ylabel(r"$d\sigma/dQ^2d\xi$ [$\mu b/GeV^2$]")
    plt.grid(True)
    plt.legend(fontsize="small")
    plt.tight_layout()

    os.makedirs("compare_strfun_x_xi", exist_ok=True)
    fname = f"compare_strfun_x_xi/compare_strfun_vs_xi_Q2={fixed_Q2}_E={beam_energy}.pdf"
    plt.savefig(fname, dpi=300)
    plt.close()
    print("Saved →", fname)
