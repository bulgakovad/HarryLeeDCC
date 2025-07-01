import numpy as np
import math
from scipy.interpolate import RectBivariateSpline, interp1d
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


def plot_cross_section_vs_W(Q2, beam_energy, file_path="input_data/wempx.dat", num_points=200):
    """
    Just ANL model. No comparison.
    Plots the differential cross section dσ/dW/dQ² as a function of W for a fixed Q² and beam energy.
    The plot is saved as a PNG file with a filename that includes the Q² and beam energy values.

    Parameters:
        Q2         : Fixed photon virtuality (GeV²)
        beam_energy: Beam (lepton) energy in the lab (GeV)
        file_path  : Path to the structure function file (default "input_data/wempx.dat")
        num_points : Number of points in W to compute (default: 200)
    """
    # Load file to get available W range from data
    data = np.loadtxt(file_path)
    W = data[:, 0]
    W_unique = np.unique(W)
    w_min, w_max = W_unique[0], W_unique[-1]
    
    # Generate W values in the available range
    W_vals = np.linspace(w_min, w_max, num_points)
    xsec_vals = []

    # Compute cross section for each W value (with verbose off)
    for w in W_vals:
        try:
            xsec = compute_cross_section(w, Q2, beam_energy, file_path, verbose=False)
        except Exception:
            xsec = np.nan
        xsec_vals.append(xsec)
    
    # Plot the cross section vs W
    plt.figure(figsize=(8, 6))
    plt.plot(W_vals, xsec_vals, label=f"Q² = {Q2} GeV², E = {beam_energy} GeV")
    plt.xlabel("W (GeV)")
    plt.ylabel("dσ/dW/dQ² (10⁻³⁰ cm²/GeV³)")
    plt.title("Differential Cross Section vs W")
    plt.xticks(np.arange(1.1, 2.1 + 0.1, 0.1))  # Set x-ticks with a step of 0.1
    plt.legend()
    plt.grid(True)
    
    # Build a filename that includes Q² and beam energy values
    filename = f"plots_ANL_model_only/cross_section_vs_W_Q2={Q2:.3f}_E={beam_energy:.3f}.png"
    
    # Save the plot as a PNG file
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Plot saved as {filename}")

def generate_table(file_path, fixed_Q2, beam_energy):
    """
    Generates a text table with columns: Q2, W, W1, W2, and CrossSection - from ANL model.
    The table is generated for a fixed Q2 value (the nearest grid Q2 is used)
    by extracting all rows from the input file that correspond to that grid Q2.
    For each row, the cross section is computed using the fixed Q2, the grid's W,
    and the structure functions from the input file.
    
    The table is saved as a tab-delimited text file.

    Parameters:
        file_path (str): Path to the input data file (W, Q2, W1, W2)
        fixed_Q2 (float): The Q2 value for which the table is generated.
        beam_energy (float): The beam (lepton) energy in GeV.
    """
    # Load the data
    data = np.loadtxt(file_path)
    # Columns: W, Q2, W1, W2
    W = data[:, 0]
    Q2 = data[:, 1]
    W1 = data[:, 2]
    W2 = data[:, 3]
    
    # Get unique Q2 values from the grid
    Q2_unique = np.unique(Q2)
    # Find the grid Q2 closest to fixed_Q2
    idx = np.argmin(np.abs(Q2_unique - fixed_Q2))
    grid_Q2 = Q2_unique[idx]
    
    # Define a tolerance to select rows corresponding to the chosen grid Q2
    tol = 1e-6
    # Select rows where Q2 is within tolerance of grid_Q2
    rows = data[np.abs(Q2 - grid_Q2) < tol]
    
    # Prepare the table rows: each row will be [Q2, W, W1, W2, CrossSection]
    output_rows = []
    for row in rows:
        W_val = row[0]
        Q2_val = row[1]  # should be grid_Q2
        W1_val = row[2]
        W2_val = row[3]
        # Compute cross section for these values
        cs = compute_cross_section(W_val, Q2_val, beam_energy, file_path, verbose=False)
        output_rows.append([Q2_val, W_val, W1_val, W2_val, cs])
    
    # Create header and save the table to a text file with tab delimiter
    header = "Q2\tW\tW1\tW2\tCrossSection"
    output_filename = f"tables_ANL_model/ANL_model_CS_Q2={fixed_Q2}.txt"
    np.savetxt(output_filename, np.array(output_rows), header=header, fmt="%.6e", delimiter="\t")
    print(f"Table saved as {output_filename}")
    
def compare_strfun(fixed_Q2, beam_energy,
                   interp_file="input_data/wempx.dat",
                   onepi_file="input_data/wemp-pi.dat",
                   num_points=200):
    """
    Compare ANL-Osaka, PDF, 1π, and data (strfun + RGA) cross sections up to W=2 GeV.
    """

    W_cutoff = 2.0

    # ---------- W grid ----------
    data = np.loadtxt(interp_file)
    W_grid = np.unique(data[:, 0])
    W_vals = np.linspace(W_grid.min(), min(W_grid.max(), W_cutoff), num_points)

    # ---------- model curves ----------
    anl_xs, pdf_xs, onepi_xs = [], [], []
    F1_interp, F2_interp, _ = get_pdf_interpolators(fixed_Q2)

    for w in W_vals:
        anl_xs.append(
            compute_cross_section(w, fixed_Q2, beam_energy,
                                  file_path=interp_file, verbose=False)
        )
        try:
            pdf_xs.append(
                compute_cross_section_pdf(w, fixed_Q2, beam_energy,
                                          F1_interp, F2_interp)
            )
        except Exception:
            pdf_xs.append(np.nan)

        try:
            onepi_xs.append(
                calculate_1pi_cross_section(w, fixed_Q2, beam_energy,
                                            file_path=onepi_file, verbose=False)
            )
        except Exception:
            onepi_xs.append(np.nan)

    anl_xs   = np.asarray(anl_xs)
    pdf_xs   = np.asarray(pdf_xs)
    onepi_xs = np.asarray(onepi_xs)

    # ---------- experimental strfun data ----------
    meas_file = f"strfun_data/cs_Q2={fixed_Q2}_E={beam_energy}.dat"
    if not os.path.isfile(meas_file):
        raise FileNotFoundError(meas_file + " not found")

    mdat = np.genfromtxt(
        meas_file,
        names=["W", "Quantity", "Uncertainty"],
        delimiter="\t", skip_header=1,
    )
    mask_meas = mdat["W"] <= W_cutoff

    # ---------- optional Klimenko RGA data ----------
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

    # ---------- plot ----------
    plt.figure(figsize=(8, 6))

    h_anl, = plt.plot(W_vals, anl_xs, label="ANL-Osaka model:full cross section", color="blue", lw=2)
    #h_pdf, = plt.plot(W_vals, pdf_xs, label="PDF model (outdated!)", color="orange", ls="--", lw=2)
    good = ~np.isnan(onepi_xs)
    h_1pi, = plt.plot(W_vals[good], onepi_xs[good], label="ANL-Osaka model: 1π contribution", color="purple", lw=2)

    h_data = plt.errorbar(
        mdat["W"][mask_meas],
        mdat["Quantity"][mask_meas],
        yerr=mdat["Uncertainty"][mask_meas],
        fmt="o", color="green", capsize=1, ms=2,
        label="strfun website: CLAS+world data"
    )

    if plot_rga and len(W_rga) > 0:
        h_rga = plt.errorbar(
            W_rga, sigma_rga, yerr=err_rga,
            fmt="s", color="red", capsize=1, ms=2,
            label="RGA data (V. Klimenko)"
        )

    # ---------- legend ----------
    handles = [plt.Line2D([], [], color='white', label=f"Q² = {fixed_Q2:.3f} GeV², E = {beam_energy} GeV"),
               h_anl,
               h_1pi,
               #h_pdf,
               h_data]
    if plot_rga and len(W_rga) > 0:
        handles.append(h_rga)

    labels = [h.get_label() for h in handles]

    plt.xlabel("W (GeV)")
    plt.ylabel(r"Cross Section ($\mathrm{\mu bn/GeV^3}$)")
    plt.grid(True)
    plt.legend(handles, labels, loc="upper left", fontsize="small")

    os.makedirs("compare_strfun", exist_ok=True)
    fname = f"compare_strfun/compare_strfun_Q2={fixed_Q2}_E={beam_energy}.pdf"
    plt.savefig(fname, dpi=300)
    plt.close()
    print("Saved →", fname)




# --- New functions for PDF-based comparison ---

def get_pdf_interpolators(fixed_Q2):
    """
    Loads the PDF table for a given fixed Q², computes F1 and F2 as functions of x,
    then computes W and returns cubic interpolators for F1(W) and F2(W), along with the minimum W.
    
    Parameters:
        fixed_Q2 (float): The fixed Q² value used in the PDF table filename.
    
    Returns:
        tuple: (F1_W_interp, F2_W_interp, W_min) where:
            - F1_W_interp is a cubic interpolator for F1 as a function of W.
            - F2_W_interp is a cubic interpolator for F2 as a function of W.
            - W_min is the minimum W value from the PDF table.
    """
    Mp = 0.9385  # Proton mass in GeV
    filename = f'PDF_tables/tst_CJpdf_ISET=400_Q2={fixed_Q2}.dat'
    pdf_table = pd.read_csv(filename, delim_whitespace=True)
    
    # Assuming columns: x, u, ub, d, db
    x = pdf_table['x'].values
    nu = pdf_table['u'].values
    nub = pdf_table['ub'].values
    nd = pdf_table['d'].values
    ndb = pdf_table['db'].values
    
    # Compute structure functions F2 and F1
    F2_x = (4/9)*(nu + nub) + (1/9)*(nd + ndb)
    F1_x = F2_x / (2 * x)
    
    # Compute W for each x (W² = Mp² + Q²*(1-x)/x)
    W2 = Mp**2 + fixed_Q2 * (1 - x) / x
    W = np.sqrt(W2)
    
    # Create cubic interpolators (sorting by W)
    sorted_indices = np.argsort(W)
    W_sorted = W[sorted_indices]
    F1_sorted = F1_x[sorted_indices]
    F2_sorted = F2_x[sorted_indices]
    
    F1_W_interp = interp1d(W_sorted, F1_sorted, kind='cubic', bounds_error=False, fill_value="extrapolate")
    F2_W_interp = interp1d(W_sorted, F2_sorted, kind='cubic', bounds_error=False, fill_value="extrapolate")
    
    return F1_W_interp, F2_W_interp, np.min(W_sorted)

def compute_cross_section_pdf(W, Q2, beam_energy, F1_W_interp, F2_W_interp):
    """
    Computes the differential cross section using PDF-based structure functions interpolated from a PDF table.

    Parameters:
        W           : Invariant mass (GeV)
        Q2          : Photon virtuality (GeV²) (should match fixed_Q2 used in the PDF table)
        beam_energy : Beam (lepton) energy (GeV)
        F1_W_interp : Interpolator for F1(W)
        F2_W_interp : Interpolator for F2(W)

    Returns:
        dcrs        : Differential cross section in units of 10^(-30) cm²/GeV³
    """
    alpha = 1 / 137.04
    Mp = 0.9385
    pi = math.pi
    wtot = math.sqrt(2 * Mp * beam_energy + Mp**2)
    if W > wtot:
        raise ValueError("W is greater than the available lab energy (w_tot).")
    
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
    return dcrs

def compare_exp_model_pdf(q2_list, beam_energy, num_points=200):
    """
    Compare the PDF-based theoretical cross section with the ANL model cross section and experimental data,
    as a function of W. Creates a shared cross-section canvas with subplots for each Q²,
    and per-Q² 4-panel plots of structure functions.
    """

    Mp = 0.9385
    num_q2 = len(q2_list)
    cols = math.ceil(math.sqrt(num_q2))
    rows = math.ceil(num_q2 / cols)
    fig_cs, axs_cs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharex=True)
    axs_cs = axs_cs.flatten()  # flatten 2D grid to 1D list for easy indexing

    for idx, fixed_Q2 in enumerate(q2_list):
        # Get PDF interpolators
        F1_W_interp, F2_W_interp, W_min = get_pdf_interpolators(fixed_Q2)
        W_vals = np.linspace(W_min, 2.6, num_points)

        # Storage arrays
        pdf_F1_vals, pdf_F2_vals = [], []
        pdf_W1_vals, pdf_W2_vals = [], []
        anl_W1_vals, anl_W2_vals = [], []
        anl_F1_vals, anl_F2_vals = [], []
        pdf_cross_sections, anl_cross_sections = [], []

        for W in W_vals:
            omega = (W**2 + fixed_Q2 - Mp**2) / (2.0 * Mp)

            # PDF model
            try:
                F1_pdf = F1_W_interp(W)
                F2_pdf = F2_W_interp(W)
            except Exception:
                F1_pdf, F2_pdf = np.nan, np.nan
            pdf_F1_vals.append(F1_pdf)
            pdf_F2_vals.append(F2_pdf)
            W1_pdf = F1_pdf / Mp if not np.isnan(F1_pdf) else np.nan
            W2_pdf = F2_pdf / omega if (omega > 0 and not np.isnan(F2_pdf)) else np.nan
            pdf_W1_vals.append(W1_pdf)
            pdf_W2_vals.append(W2_pdf)

            # ANL model
            try:
                anl_w1, anl_w2 = interpolate_structure_functions("input_data/wempx.dat", W, fixed_Q2)
            except Exception:
                anl_w1, anl_w2 = np.nan, np.nan
            anl_W1_vals.append(anl_w1)
            anl_W2_vals.append(anl_w2)
            anl_F1_vals.append(Mp * anl_w1)
            anl_F2_vals.append(omega * anl_w2 if omega > 0 else np.nan)

            # Cross sections
            try:
                cs_pdf = compute_cross_section_pdf(W, fixed_Q2, beam_energy, F1_W_interp, F2_W_interp)
            except Exception:
                cs_pdf = np.nan
            pdf_cross_sections.append(cs_pdf)

            try:
                cs_anl = compute_cross_section(W, fixed_Q2, beam_energy, file_path="input_data/wempx.dat", verbose=False)
            except Exception:
                cs_anl = np.nan
            anl_cross_sections.append(cs_anl)

        # --- Plot: cross sections subplot (one per Q²)
        ax_cs = axs_cs[idx]
        ax_cs.plot(W_vals, pdf_cross_sections, label="PDF model", color='green', linestyle='--')
        ax_cs.plot(W_vals, anl_cross_sections, label="ANL-Osaka model", color='blue', linestyle='-')

        exp_file = f"exp_data/InclusiveExpValera_Q2={fixed_Q2}.dat"
        if os.path.isfile(exp_file):
            exp_data = np.genfromtxt(exp_file, names=["W", "eps", "sigma", "error", "sys_error"], delimiter="\t", skip_header=1)
            ax_cs.errorbar(exp_data["W"], exp_data["sigma"] * 1e-3,
                           yerr=np.sqrt(exp_data["error"]**2 + exp_data["sys_error"]**2) * 1e-3,
                           fmt="o", color="red", markersize=3, label="Experimental data")
        else:
            print(f"Warning: Experimental file {exp_file} not found. Skipping data overlay.")
        
        ax_cs.set_ylabel("$dσ/dW dQ^2$ $\mu bn$/GeV^3)")
        ax_cs.set_title(f"Q² = {fixed_Q2:.3f} GeV²")
        ax_cs.grid(True)
        ax_cs.legend()

        # --- Save structure function 2x2 panel plot for this Q²
        fig_sf, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs[0, 0].plot(W_vals, pdf_W1_vals, label="PDF: W1 = F1/Mₚ", color="magenta", linestyle="-")
        axs[0, 0].plot(W_vals, anl_W1_vals, label="ANL-Osaka: W1", color="magenta", linestyle="--")
        axs[0, 0].set_xlabel("W (GeV)")
        axs[0, 0].set_ylabel("W1")
        axs[0, 0].set_title("W1 vs W")
        axs[0, 0].grid(True)
        axs[0, 0].legend()

        axs[0, 1].plot(W_vals, pdf_W2_vals, label="PDF: W2 = F2/ω", color="orange", linestyle="-")
        axs[0, 1].plot(W_vals, anl_W2_vals, label="ANL-Osaka: W2", color="orange", linestyle="--")
        axs[0, 1].set_xlabel("W (GeV)")
        axs[0, 1].set_ylabel("W2")
        axs[0, 1].set_title("W2 vs W")
        axs[0, 1].grid(True)
        axs[0, 1].legend()

        axs[1, 0].plot(W_vals, pdf_F1_vals, label="PDF: F1", color="blue", linestyle="-")
        axs[1, 0].plot(W_vals, anl_F1_vals, label="ANL-Osaka: F1", color="blue", linestyle="--")
        axs[1, 0].set_xlabel("W (GeV)")
        axs[1, 0].set_ylabel("F1")
        axs[1, 0].set_title("F1 vs W")
        axs[1, 0].grid(True)
        axs[1, 0].legend()

        axs[1, 1].plot(W_vals, pdf_F2_vals, label="PDF: F2", color="green", linestyle="-")
        axs[1, 1].plot(W_vals, anl_F2_vals, label="ANL-Osaka: F2", color="green", linestyle="--")
        axs[1, 1].set_xlabel("W (GeV)")
        axs[1, 1].set_ylabel("F2")
        axs[1, 1].set_title("F2 vs W")
        axs[1, 1].grid(True)
        axs[1, 1].legend()

        fig_sf.suptitle(f"Structure Functions vs W at Q² = {fixed_Q2:.3f} GeV², E = {beam_energy} GeV", fontsize=16)
        fig_sf.tight_layout(rect=[0, 0, 1, 0.95])
        filename_sf = f"structure_functions_4plots_vs_W_Q2={fixed_Q2}_Ebeam={beam_energy}.png"
        fig_sf.savefig(filename_sf, dpi=300)
        plt.close(fig_sf)
        print(f"4-panel structure function plot saved as {filename_sf}")

        # --- Save structure function table
        table_data = np.column_stack((np.full(W_vals.shape, fixed_Q2), W_vals, pdf_W1_vals, anl_W1_vals,
                                      pdf_W2_vals, anl_W2_vals, anl_F1_vals, anl_F2_vals,
                                      pdf_F1_vals, pdf_F2_vals, pdf_cross_sections))
        table_filename = f"structure_functions_table_vs_W_Q2={fixed_Q2}_Ebeam={beam_energy}.txt"
        header_str = "Q2\tW\tPDF_W1\tANL_W1\tPDF_W2\tANL_W2\tANL_F1\tANL_F2\tPDF_F1\tPDF_F2\tPDF_CrossSection"
        np.savetxt(table_filename, table_data, fmt="%.6e", delimiter="\t", header=header_str)
        print(f"Structure function table saved as {table_filename}")

    # Finalize and save shared cross section canvas
    axs_cs[-1].set_xlabel("W (GeV)")
    fig_cs.suptitle(f"Cross Section vs W for Q² values at E = {beam_energy} GeV", fontsize=16)
    fig_cs.tight_layout(rect=[0, 0, 1, 0.96])
    fig_cs.savefig(f"cross_section_multiQ2_vs_W_E={beam_energy}.png", dpi=300)
    plt.close(fig_cs)
    print(f"Combined cross section plot saved as cross_section_multiQ2_vs_W_E={beam_energy}.png")



def compare_exp_model_pdf_Bjorken_x(fixed_Q2, beam_energy, num_points=200):
    """
    Compares the PDF‐based theoretical cross section with the ANL model cross section and experimental data,
    as functions of Bjorken x. Also produces a 2×2 canvas with:
      Top row: Derived structure functions W1 = F1/Mp and W2 = F2/ω vs x (PDF: solid; ANL: dashed).
      Bottom row: Raw structure functions F1 and F2 vs x (PDF: solid; ANL: dashed).
    Generates a text table with columns: Q2, x, PDF_W1, PDF_W2, PDF_F1, PDF_F2, ANL_W1, ANL_W2, ANL_F1, ANL_F2.
    """

    Mp = 0.9385
    pdf_filename = f"PDF_tables/tst_CJpdf_ISET=400_Q2={fixed_Q2}.dat"
    if not os.path.isfile(pdf_filename):
        raise FileNotFoundError(f"PDF file not found: {pdf_filename}")
    pdf_table = pd.read_csv(pdf_filename, delim_whitespace=True)
    x_vals = np.linspace(np.min(pdf_table['x'].values), np.max(pdf_table['x'].values), num_points)
    
    # Get PDF interpolators.
    F1_W_interp, F2_W_interp, _ = get_pdf_interpolators(fixed_Q2)
    
    x_theory = []
    W_vals = []
    pdf_cross_sections_dx = []
    anl_cross_sections_dx = []
    PDF_F1_vals = []
    PDF_F2_vals = []
    PDF_W1_vals = []
    PDF_W2_vals = []
    ANL_F1_vals = []
    ANL_F2_vals = []
    ANL_W1_vals = []
    ANL_W2_vals = []
    
    def dWdx_abs(W, x, Q2):
        return Q2/(2.0 * W * x**2)
    
    for x in x_vals:
        try:
            W_val = math.sqrt(Mp**2 + fixed_Q2*(1 - x)/x)
        except Exception:
            W_val = np.nan
        W_vals.append(W_val)
        x_theory.append(x)
        try:
            cs_pdf_dw = compute_cross_section_pdf(W_val, fixed_Q2, beam_energy, F1_W_interp, F2_W_interp)
        except Exception:
            cs_pdf_dw = np.nan
        try:
            cs_anl_dw = compute_cross_section(W_val, fixed_Q2, beam_energy, file_path="input_data/wempx.dat", verbose=False)
        except Exception:
            cs_anl_dw = np.nan
        try:
            jacobian = dWdx_abs(W_val, x, fixed_Q2)
        except Exception:
            jacobian = np.nan
        pdf_cross_sections_dx.append(cs_pdf_dw * jacobian if not np.isnan(jacobian) else np.nan)
        anl_cross_sections_dx.append(cs_anl_dw * jacobian if not np.isnan(jacobian) else np.nan)
        
        try:
            F1_pdf = F1_W_interp(W_val)
            F2_pdf = F2_W_interp(W_val)
        except Exception:
            F1_pdf, F2_pdf = np.nan, np.nan
        PDF_F1_vals.append(F1_pdf)
        PDF_F2_vals.append(F2_pdf)
        W1_pdf = F1_pdf / Mp if not np.isnan(F1_pdf) else np.nan
        omega = (W_val**2 + fixed_Q2 - Mp**2)/(2.0*Mp)
        W2_pdf = F2_pdf/omega if (omega>0 and not np.isnan(F2_pdf)) else np.nan
        PDF_W1_vals.append(W1_pdf)
        PDF_W2_vals.append(W2_pdf)
        
        try:
            W1_anl, W2_anl = interpolate_structure_functions("input_data/wempx.dat", W_val, fixed_Q2)
        except Exception:
            W1_anl, W2_anl = np.nan, np.nan
        ANL_W1_vals.append(W1_anl)
        ANL_W2_vals.append(W2_anl)
        F1_anl = W1_anl * Mp if not np.isnan(W1_anl) else np.nan
        F2_anl = W2_anl * omega if (omega>0 and not np.isnan(W2_anl)) else np.nan
        ANL_F1_vals.append(F1_anl)
        ANL_F2_vals.append(F2_anl)
    
    # Experimental data conversion (W -> x).
    exp_file = f"exp_data/InclusiveExpValera_Q2={fixed_Q2}.dat"
    if not os.path.isfile(exp_file):
        raise FileNotFoundError(f"Experimental data file {exp_file} not found.")
    exp_data = pd.read_csv(exp_file, sep=r'\s+')
    W_exp = exp_data["W"].values
    sigma_exp_dw = exp_data["sigma"].values * 1e-3
    err_exp_dw = np.sqrt(exp_data["error"]**2+exp_data["sys_error"]**2)*1e-3
    x_exp = []
    sigma_exp_dx = []
    err_exp_dx = []
    for i, W_e in enumerate(W_exp):
        denom = (W_e**2 - Mp**2 + fixed_Q2)
        if denom != 0:
            x_e = fixed_Q2/denom
        else:
            x_e = np.nan
        x_exp.append(x_e)
        try:
            jac = dWdx_abs(W_e, x_e, fixed_Q2)
        except Exception:
            jac = np.nan
        sigma_exp_dx.append(sigma_exp_dw[i]*jac if not np.isnan(jac) else np.nan)
        err_exp_dx.append(err_exp_dw[i]*jac if not np.isnan(jac) else np.nan)
    
    plt.figure(figsize=(8,6))
    plt.plot(x_theory, pdf_cross_sections_dx, label="PDF Model", color="green", linestyle="--")
    plt.plot(x_theory, anl_cross_sections_dx, label="ANL-Osaka Model", color="blue")
    plt.errorbar(x_exp, sigma_exp_dx, yerr=err_exp_dx, fmt="o", color="red", markersize=3, label="Experimental data")
    plt.xlabel("Bjorken x")
    plt.ylabel("dσ/dQ²dx ($\mu bn/GeV^2$)")
    plt.title(f"dσ/dQ²dx vs x at Q²={fixed_Q2} GeV², E={beam_energy} GeV")
    plt.legend()
    plt.grid(True)
    x_plot_min = max(np.nanmin(x_exp), np.nanmin(x_theory))*0.95
    plt.xlim(x_plot_min, np.nanmax(x_theory))
    #plt.xlim(0.3,1)
    y_max = np.nanmax(sigma_exp_dx)
    plt.ylim(0, y_max*1.1)
    #plt.ylim(0, 0.02)
    plt.tight_layout()
    filename_cs = f"cross_section_vs_x_comparison_Q2={fixed_Q2}_Ebeam={beam_energy}.png"
    plt.savefig(filename_cs, dpi=300)
    plt.close()
    print(f"Cross section vs x plot saved as {filename_cs}")
    
    # 2x2 panel for structure functions vs x.
    fig, axs = plt.subplots(2,2, figsize=(12,10))
    axs[0,0].plot(x_theory, PDF_W1_vals, label="PDF W1 (F1/Mp)", color="magenta")
    axs[0,0].plot(x_theory, ANL_W1_vals, label="ANL-Osaka W1", color="magenta", linestyle="--")
    axs[0,0].set_xlabel("x (Bjorken)")
    axs[0,0].set_ylabel("W1")
    axs[0,0].set_title("W1 vs x")
    axs[0,0].grid(True)
    axs[0,0].legend()
    
    axs[0,1].plot(x_theory, PDF_W2_vals, label="PDF W2 (F2/ω)", color="orange")
    axs[0,1].plot(x_theory, ANL_W2_vals, label="ANL-Osaka W2", color="orange", linestyle="--")
    axs[0,1].set_xlabel("x (Bjorken)")
    axs[0,1].set_ylabel("W2")
    axs[0,1].set_title("W2 vs x")
    axs[0,1].grid(True)
    axs[0,1].legend()
    
    axs[1,0].plot(x_theory, PDF_F1_vals, label="PDF F1", color="blue")
    axs[1,0].plot(x_theory, ANL_F1_vals, label="ANL-Osaka F1", color="blue", linestyle="--")
    axs[1,0].set_xlabel("x (Bjorken)")
    axs[1,0].set_ylabel("F1")
    axs[1,0].set_title("F1 vs x")
    axs[1,0].grid(True)
    axs[1,0].legend()
    
    axs[1,1].plot(x_theory, PDF_F2_vals, label="PDF F2", color="green")
    axs[1,1].plot(x_theory, ANL_F2_vals, label="ANL-Osaka F2", color="green", linestyle="--")
    axs[1,1].set_xlabel("x (Bjorken)")
    axs[1,1].set_ylabel("F2")
    axs[1,1].set_title("F2 vs x")
    axs[1,1].grid(True)
    axs[1,1].legend()
    
    fig.suptitle(f"Structure Functions vs x at Q²={fixed_Q2} GeV²", fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.95])
    filename_sf = f"structure_functions_4plots_vs_x_Q2={fixed_Q2}_Ebeam={beam_energy}.png"
    #plt.savefig(filename_sf, dpi=300)
    plt.close()
    print(f"4-panel structure functions vs x plot saved as {filename_sf}")
    
    table_data = np.column_stack((np.full(len(x_theory), fixed_Q2), np.array(x_theory),
                                   np.array(W_vals), np.array(PDF_W1_vals), np.array(PDF_W2_vals),
                                   np.array(PDF_F1_vals), np.array(PDF_F2_vals),
                                   np.array(ANL_W1_vals), np.array(ANL_W2_vals),
                                   np.array(ANL_F1_vals), np.array(ANL_F2_vals)))
    table_filename = f"structure_functions_table_vs_x_Q2={fixed_Q2}_Ebeam={beam_energy}.txt"
    header_str = "Q2\tx\tW\tPDF_W1\tPDF_W2\tPDF_F1\tPDF_F2\tANL_W1\tANL_W2\tANL_F1\tANL_F2"
    #Do not save
    np.savetxt(table_filename, table_data, fmt="%.6e", delimiter="\t", header=header_str)
    print(f"Structure functions vs x table saved as {table_filename}")



def compare_exp_model_pdf_Nachtmann_xi(fixed_Q2, beam_energy, num_points=200):
    """
    Compares the PDF‐based theoretical cross section with the ANL model cross section and experimental data,
    as functions of the Nachtmann variable ξ. The conversion uses the chain rule:
       dσ/(dQ² dξ) = dσ/(dQ² dW) · (dW/dx) · (dx/dξ).
    Also produces a 2×2 canvas with:
      Top row: Derived structure functions W1 = F1/Mp and W2 = F2/ω vs ξ (PDF: solid; ANL: dashed).
      Bottom row: Raw structure functions F1 and F2 vs ξ (PDF: solid; ANL: dashed).
    Generates a text table with columns: Q2, ξ, W, PDF_W1, PDF_W2, PDF_F1, PDF_F2, ANL_W1, ANL_W2, ANL_F1, ANL_F2.
    """


    Mp = 0.9385

    def dWdx_abs(W, x, Q2):
        return Q2/(2.0 * W * x**2)

    def xi_of_x(x, Q2):
        t = 4.0 * x**2 * Mp**2 / Q2
        return 2.0 * x / (1.0 + np.sqrt(1.0+t))

    def dxi_dx(x, Q2):
        t = 4.0 * x**2 * Mp**2 / Q2
        sqrt_term = np.sqrt(1.0+t)
        f = 1.0 + sqrt_term
        return (2.0*f - (4.0*x*Mp**2)/(Q2*sqrt_term))/(f**2)

    pdf_filename = f"PDF_tables/tst_CJpdf_ISET=400_Q2={fixed_Q2}.dat"
    if not os.path.isfile(pdf_filename):
        raise FileNotFoundError(f"PDF file not found: {pdf_filename}")
    pdf_table = pd.read_csv(pdf_filename, delim_whitespace=True)
    x_vals = np.linspace(np.min(pdf_table['x'].values), np.max(pdf_table['x'].values), num_points)
    xi_vals = [xi_of_x(x, fixed_Q2) for x in x_vals]
    
    F1_W_interp, F2_W_interp, _ = get_pdf_interpolators(fixed_Q2)
    
    x_theory = []
    W_vals = []
    pdf_cross_sections_dxi = []
    anl_cross_sections_dxi = []
    PDF_F1_vals = []
    PDF_F2_vals = []
    PDF_W1_vals = []
    PDF_W2_vals = []
    ANL_F1_vals = []
    ANL_F2_vals = []
    ANL_W1_vals = []
    ANL_W2_vals = []
    
    for x in x_vals:
        try:
            W_val = math.sqrt(Mp**2 + fixed_Q2*(1 - x)/x)
        except Exception:
            W_val = np.nan
        W_vals.append(W_val)
        x_theory.append(x)
        try:
            cs_pdf_dw = compute_cross_section_pdf(W_val, fixed_Q2, beam_energy, F1_W_interp, F2_W_interp)
        except Exception:
            cs_pdf_dw = np.nan
        try:
            cs_anl_dw = compute_cross_section(W_val, fixed_Q2, beam_energy, file_path="input_data/wempx.dat", verbose=False)
        except Exception:
            cs_anl_dw = np.nan
        try:
            jacobian1 = dWdx_abs(W_val, x, fixed_Q2)
        except Exception:
            jacobian1 = np.nan
        cs_pdf_dx = cs_pdf_dw * jacobian1 if not np.isnan(jacobian1) else np.nan
        cs_anl_dx = cs_anl_dw * jacobian1 if not np.isnan(jacobian1) else np.nan
        try:
            dxi_dx_val = dxi_dx(x, fixed_Q2)
            dx_dxi = 1.0/dxi_dx_val if (dxi_dx_val != 0 and not np.isnan(dxi_dx_val)) else np.nan
        except Exception:
            dx_dxi = np.nan
        cs_pdf_dxi = cs_pdf_dx * dx_dxi if not np.isnan(dx_dxi) else np.nan
        cs_anl_dxi = cs_anl_dx * dx_dxi if not np.isnan(dx_dxi) else np.nan
        pdf_cross_sections_dxi.append(cs_pdf_dxi)
        anl_cross_sections_dxi.append(cs_anl_dxi)
        
        try:
            F1_pdf = F1_W_interp(W_val)
            F2_pdf = F2_W_interp(W_val)
        except Exception:
            F1_pdf, F2_pdf = np.nan, np.nan
        PDF_F1_vals.append(F1_pdf)
        PDF_F2_vals.append(F2_pdf)
        W1_pdf = F1_pdf / Mp if not np.isnan(F1_pdf) else np.nan
        omega = (W_val**2 + fixed_Q2 - Mp**2)/(2.0*Mp)
        W2_pdf = F2_pdf/omega if (omega>0 and not np.isnan(F2_pdf)) else np.nan
        PDF_W1_vals.append(W1_pdf)
        PDF_W2_vals.append(W2_pdf)
        
        try:
            W1_anl, W2_anl = interpolate_structure_functions("input_data/wempx.dat", W_val, fixed_Q2)
        except Exception:
            W1_anl, W2_anl = np.nan, np.nan
        ANL_W1_vals.append(W1_anl)
        ANL_W2_vals.append(W2_anl)
        F1_anl = W1_anl * Mp if not np.isnan(W1_anl) else np.nan
        F2_anl = W2_anl * omega if (omega>0 and not np.isnan(W2_anl)) else np.nan
        ANL_F1_vals.append(F1_anl)
        ANL_F2_vals.append(F2_anl)
    
    exp_file = f"exp_data/InclusiveExpValera_Q2={fixed_Q2}.dat"
    if not os.path.isfile(exp_file):
        raise FileNotFoundError(f"Experimental data file {exp_file} not found.")
    exp_data = pd.read_csv(exp_file, sep=r'\s+')
    W_exp = exp_data["W"].values
    sigma_exp_dw = exp_data["sigma"].values * 1e-3
    err_exp_dw = np.sqrt(exp_data["error"]**2+exp_data["sys_error"]**2)*1e-3
    xi_exp = []
    sigma_exp_dxi = []
    err_exp_dxi = []
    for i, W_e in enumerate(W_exp):
        denom = (W_e**2 - Mp**2 + fixed_Q2)
        if denom != 0:
            x_e = fixed_Q2/denom
        else:
            x_e = np.nan
        xi_e = xi_of_x(x_e, fixed_Q2)
        xi_exp.append(xi_e)
        try:
            jac1_e = dWdx_abs(W_e, x_e, fixed_Q2)
        except Exception:
            jac1_e = np.nan
        sigma_exp_dx = sigma_exp_dw[i]*jac1_e if not np.isnan(jac1_e) else np.nan
        try:
            dxi_dx_e = dxi_dx(x_e, fixed_Q2)
            dx_dxi_e = 1.0/dxi_dx_e if (dxi_dx_e != 0 and not np.isnan(dxi_dx_e)) else np.nan
        except Exception:
            dx_dxi_e = np.nan
        sigma_exp_val = sigma_exp_dx * dx_dxi_e if not np.isnan(dx_dxi_e) else np.nan
        err_exp_val = err_exp_dw[i]*jac1_e*dx_dxi_e if (not np.isnan(jac1_e) and not np.isnan(dx_dxi_e)) else np.nan
        sigma_exp_dxi.append(sigma_exp_val)
        err_exp_dxi.append(err_exp_val)
    
    plt.figure(figsize=(8,6))
    plt.plot(xi_vals, pdf_cross_sections_dxi, label="PDF Model", color="green", linestyle="--")
    plt.plot(xi_vals, anl_cross_sections_dxi, label="ANL-Osaka Model", color="blue")
    plt.errorbar(xi_exp, sigma_exp_dxi, yerr=err_exp_dxi, fmt="o", color="red", markersize=3, label="Experimental data")
    plt.xlabel("Nachtmann ξ")
    plt.ylabel("dσ/dQ²dξ ($\mu bn/GeV^2$)")
    plt.title(f"dσ/dQ²dξ vs ξ at Q²={fixed_Q2} GeV², E={beam_energy} GeV")
    plt.legend()
    plt.grid(True)
    xi_plot_min = max(np.nanmin(xi_exp), np.nanmin(xi_vals))*0.95
    xi_plot_max = np.nanmax(xi_vals)*1.05
    plt.xlim(xi_plot_min, xi_plot_max)
    #plt.xlim(0.3,0.9)
    y_max = np.nanmax(sigma_exp_dxi)
    plt.ylim(0, y_max*1.1)
    #plt.ylim(0, 0.0225)
    plt.tight_layout()
    filename_cs = f"cross_section_vs_xi_comparison_Q2={fixed_Q2}_Ebeam={beam_energy}.png"
    plt.savefig(filename_cs, dpi=300)
    plt.close()
    print(f"Cross section vs ξ plot saved as {filename_cs}")
    
    # 2x2 panel for structure functions vs ξ.
    fig, axs = plt.subplots(2,2, figsize=(12,10))
    axs[0,0].plot(xi_vals, PDF_W1_vals, label="PDF W1 (F1/Mp)", color="magenta")
    axs[0,0].plot(xi_vals, ANL_W1_vals, label="ANL-Osaka W1", color="magenta", linestyle="--")
    axs[0,0].set_xlabel("ξ (Nachtmann)")
    axs[0,0].set_ylabel("W1")
    axs[0,0].set_title("W1 vs ξ")
    axs[0,0].grid(True)
    axs[0,0].legend()
    
    axs[0,1].plot(xi_vals, PDF_W2_vals, label="PDF W2 (F2/ω)", color="orange")
    axs[0,1].plot(xi_vals, ANL_W2_vals, label="ANL-Osaka W2", color="orange", linestyle="--")
    axs[0,1].set_xlabel("ξ (Nachtmann)")
    axs[0,1].set_ylabel("W2")
    axs[0,1].set_title("W2 vs ξ")
    axs[0,1].grid(True)
    axs[0,1].legend()
    
    axs[1,0].plot(xi_vals, PDF_F1_vals, label="PDF F1", color="blue")
    axs[1,0].plot(xi_vals, ANL_F1_vals, label="ANL-Osaka F1", color="blue", linestyle="--")
    axs[1,0].set_xlabel("ξ (Nachtmann)")
    axs[1,0].set_ylabel("F1")
    axs[1,0].set_title("F1 vs ξ")
    axs[1,0].grid(True)
    axs[1,0].legend()
    
    axs[1,1].plot(xi_vals, PDF_F2_vals, label="PDF F2", color="green")
    axs[1,1].plot(xi_vals, ANL_F2_vals, label="ANL-Osaka F2", color="green", linestyle="--")
    axs[1,1].set_xlabel("ξ (Nachtmann)")
    axs[1,1].set_ylabel("F2")
    axs[1,1].set_title("F2 vs ξ")
    axs[1,1].grid(True)
    axs[1,1].legend()
    
    fig.suptitle(f"Structure Functions vs ξ at Q²={fixed_Q2} GeV²", fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.95])
    filename_sf = f"structure_functions_4plots_vs_xi_Q2={fixed_Q2}_Ebeam={beam_energy}.png"
    #plt.savefig(filename_sf, dpi=300)
    plt.close()
    print(f"4-panel structure functions vs ξ plot saved as {filename_sf}")
    
    table_data = np.column_stack((np.full(len(xi_vals), fixed_Q2), np.array(xi_vals),
                                    np.array(W_vals), np.array(PDF_W1_vals), np.array(PDF_W2_vals),
                                    np.array(PDF_F1_vals), np.array(PDF_F2_vals),
                                    np.array(ANL_W1_vals), np.array(ANL_W2_vals),
                                    np.array(ANL_F1_vals), np.array(ANL_F2_vals)))
    table_filename = f"structure_functions_table_vs_xi_Q2={fixed_Q2}_Ebeam={beam_energy}.txt"
    header_str = "Q2\tξ\tW\tPDF_W1\tPDF_W2\tPDF_F1\tPDF_F2\tANL_W1\tANL_W2\tANL_F1\tANL_F2"
    # Do not save
    #np.savetxt(table_filename, table_data, fmt="%.6e", delimiter="\t", header=header_str) 
    print(f"Structure functions vs ξ table saved as {table_filename}")
    
    
def compare_f12_strfun(q2_list, xaxis_choice, interp_file="input_data/wempx.dat"):
    """
    Compares F1 and F2 structure functions from ANL model, PDF model, and strfun experimental data.
    For each Q² in q2_list, it plots:
        - Left: F1 vs X
        - Right: F2 vs X
      where X is either W or x depending on user choice.

    Data files must exist in 'F1_F2_strfun/' as:
        - F{1,2}_VS_{W,x}_Q2={q2}.dat
    Format: Tab-delimited with header: X, Quantity, Uncertainty
    """

    from functions import get_pdf_interpolators  # Ensure this is available in your module
    Mp = 0.9385

    if xaxis_choice not in ("w", "x"):
        print("Invalid choice. Use 'W' or 'x'.")
        return

    for q2 in q2_list:
        suffix = f"VS_{xaxis_choice.upper()}_Q2={q2}"
        f1_file = f"F1_F2_strfun/F1_{suffix}.dat"
        f2_file = f"F1_F2_strfun/F2_{suffix}.dat"

        if not os.path.exists(f1_file) or not os.path.exists(f2_file):
            print(f"Missing data files for Q² = {q2}: {f1_file} or {f2_file}")
            continue

        exp_f1 = np.genfromtxt(f1_file, names=True, delimiter="\t")
        exp_f2 = np.genfromtxt(f2_file, names=True, delimiter="\t")

        F1_pdf_interp, F2_pdf_interp, W_min = get_pdf_interpolators(q2)

        if xaxis_choice == "w":
            X_vals = np.linspace(max(W_min, 1.1), 2.1, 250)
            F1_anl, F2_anl = [], []
            F1_pdf, F2_pdf = [], []
            for W in X_vals:
                try:
                    w1, w2 = interpolate_structure_functions(interp_file, W, q2)
                    omega = (W**2 + q2 - Mp**2) / (2 * Mp)
                    F1_anl.append(Mp * w1)
                    F2_anl.append(omega * w2)
                    F1_pdf.append(F1_pdf_interp(W))
                    F2_pdf.append(F2_pdf_interp(W))
                except Exception:
                    F1_anl.append(np.nan)
                    F2_anl.append(np.nan)
                    F1_pdf.append(np.nan)
                    F2_pdf.append(np.nan)
        else:
            X_vals = []
            F1_anl, F2_anl = [], []
            F1_pdf, F2_pdf = [], []
            for W in np.linspace(1.1, 2.1, 250):
                try:
                    omega = (W**2 + q2 - Mp**2) / (2 * Mp)
                    x = q2 / (2 * Mp * omega)
                    if not (0 < x < 1):
                        continue
                    w1, w2 = interpolate_structure_functions(interp_file, W, q2)
                    F1_anl.append(Mp * w1)
                    F2_anl.append(omega * w2)
                    F1_pdf.append(F1_pdf_interp(W))
                    F2_pdf.append(F2_pdf_interp(W))
                    X_vals.append(x)
                except Exception:
                    continue

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        
        if xaxis_choice == "x":
            x_axis_name = "x"
        else:
            x_axis_name = "W"
            
        if xaxis_choice == "x":
            x_axis_units = "x Bjorken"
        else:
            x_axis_units = "W,(GeV)"

        # F1 plot
        axs[0].plot(X_vals, F1_anl, label="ANL-Osaka model", color="blue")
        axs[0].plot(X_vals, F1_pdf, label="PDF model", color="green", linestyle="--")
        axs[0].errorbar(exp_f1[x_axis_name], exp_f1["Quantity"], yerr=exp_f1["Uncertainty"],
                        fmt="o", color="red", label="CLAS and World data", markersize=4, capsize=2)
        axs[0].set_title(f"F₁ vs {x_axis_name} at Q² = {q2} GeV²")
        axs[0].set_xlabel(x_axis_units)
        axs[0].set_ylabel("F₁")
        axs[0].grid(True)
        axs[0].legend()

        # F2 plot
        axs[1].plot(X_vals, F2_anl, label="ANL-Osaka model", color="blue")
        axs[1].plot(X_vals, F2_pdf, label="PDF model", color="green", linestyle="--")
        axs[1].errorbar(exp_f2[x_axis_name], exp_f2["Quantity"], yerr=exp_f2["Uncertainty"],
                        fmt="o", color="red", label="CLAS and World data", markersize=4, capsize=2)
        axs[1].set_title(f"F₂ vs {x_axis_name} at Q² = {q2} GeV²")
        axs[1].set_xlabel(x_axis_units)
        axs[1].set_ylabel("F₂")
        axs[1].grid(True)
        axs[1].legend()

        plt.tight_layout()
        filename = f"F1_F2_comparison_strfun/compare_F12_strfun_VS_{xaxis_choice.upper()}_Q2={q2}.pdf"
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Saved F₁/F₂ comparison plot for Q² = {q2} as {filename}")





def compare_exp_pdf_resonance(fixed_Q2, beam_energy):
    """
    Compares experimental RGA data with the sum of the PDF prediction and the resonance contribution.
    
    This function looks for a resonance prediction file in the folder "resonance_contributions"
    with the name "sum_res_Q2={fixed_Q2}.dat". This file is assumed to contain at least three columns:
      - The first column is W.
      - The second-to-last column contains the mean predicted cross section as a function of W.
      - The last column contains the standard deviation of the predicted cross section.
    The mean and standard deviation values are multiplied by 1e-3 to convert to microbarns.
    
    Then the function obtains the PDF-based differential cross section prediction at the same
    W values using compute_cross_section_pdf. The combined prediction is computed as the sum of
    the PDF prediction and the resonance mean.
    
    Finally, experimental data is loaded from "exp_data/InclusiveExpValera_Q2={fixed_Q2}.dat" and all three
    predictions (PDF-only, Resonance-only, and Combined) are plotted along with the experimental data.
    Additionally, an uncertainty band is drawn around both the resonance-only curve and the combined prediction.
   """

    # Build the file path for the resonance prediction file.
    res_file = f"resonance_contributions/sum_res_Q2={fixed_Q2}.dat"
    if not os.path.isfile(res_file):
        raise FileNotFoundError(f"Resonance file {res_file} not found.")
    
    # Try loading resonance data while skipping header, fall back if it fails.
    try:
        res_data = np.loadtxt(res_file, skiprows=1)
    except Exception:
        res_data = np.loadtxt(res_file)
    
    # Extract resonance data:
    # First column: W.
    # Second-to-last column: resonance mean.
    # Last column: resonance standard deviation.
    W_res = res_data[:, 0]
    resonance_mean = res_data[:, -2] * 1e-3   # convert to microbarns
    resonance_std  = res_data[:, -1] * 1e-3     # convert to microbarns

    # Get PDF interpolators (based on the given fixed_Q2).
    F1_W_interp, F2_W_interp, _ = get_pdf_interpolators(fixed_Q2)
    
    # For each W in the resonance file, compute the PDF prediction using compute_cross_section_pdf.
    pdf_pred = []
    for W in W_res:
        try:
            pred = compute_cross_section_pdf(W, fixed_Q2, beam_energy, F1_W_interp, F2_W_interp)
        except Exception:
            pred = np.nan
        pdf_pred.append(pred)
    pdf_pred = np.array(pdf_pred)
    
    # Compute the combined prediction: PDF prediction + resonance mean.
    combined_pred = pdf_pred + resonance_mean
    # Compute combined uncertainty band (assuming PDF is error-free).
    combined_lower = pdf_pred + (resonance_mean - resonance_std)
    combined_upper = pdf_pred + (resonance_mean + resonance_std)
    
    # Load experimental data from the RGA file.
    exp_file = f"exp_data/InclusiveExpValera_Q2={fixed_Q2}.dat"
    if not os.path.isfile(exp_file):
        raise FileNotFoundError(f"Experimental data file {exp_file} not found.")
    # Read using pandas (assuming the file has a header).
    exp_data = pd.read_csv(exp_file, sep=r'\s+')
    W_exp = exp_data["W"].values
    sigma_exp = exp_data["sigma"].values * 1e-3  # convert to microbarns
    sigma_err = np.sqrt(exp_data["error"]**2 + exp_data["sys_error"]**2) * 1e-3

    # Plot all curves.
    plt.figure(figsize=(8,6))
    # Plot PDF prediction.
    plt.plot(W_res, pdf_pred, label="PDF Prediction", color="green", linestyle=":")
    # Plot resonance prediction with its band.
    plt.plot(W_res, resonance_mean, label="Resonance Mean", color="blue", linestyle="--")
    plt.fill_between(W_res, resonance_mean - resonance_std, resonance_mean + resonance_std,
                     color="blue", alpha=0.3 )
    # Plot combined prediction with its uncertainty band.
    plt.plot(W_res, combined_pred, label="Combined (PDF + Resonance)", color="purple", linewidth=2)
    plt.fill_between(W_res, combined_lower, combined_upper, color="purple", alpha=0.3)
    # Plot experimental data.
    plt.errorbar(W_exp, sigma_exp, yerr=sigma_err, fmt="o", color="red", markersize=4,
                 label="Experimental RGA Data")
    plt.xlabel("W (GeV)")
    plt.ylabel("dσ/dW/dQ² (μb/GeV³)")
    plt.title(f"Combined PDF + Resonance vs Experimental Data at Q² = {fixed_Q2} GeV², E = {beam_energy} GeV")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = f"compare_resonance_vs_exp_Q2={fixed_Q2}_E={beam_energy}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Combined PDF + Resonance vs Experimental plot saved as {filename}")
    
    
def anl_struct_func_xsecs(fixed_Q2, beam_energy, num_points=200):
    """
    For a given fixed Q² and beam energy, this routine:
      1. Reads the existing “input_data/wempx.dat” grid of (W, Q², W1, W2).
      2. Builds a smooth W‐grid (num_points points from W_min to W_max).
      3. At each W:
         • Interpolates the ANL model’s W1, W2 via interpolate_structure_functions().
         • Computes the ANL cross section via compute_cross_section().
      4. Plots W1 and W2 vs W (both curves on the same figure).
      5. Writes out a text file with columns:
           Q2,    W,    W1,    W2,    xsec
    """

    # First, load “input_data/wempx.dat” to determine the available W‐range.
    if not os.path.isfile("input_data/wempx.dat"):
        raise FileNotFoundError("ANL interpolation file 'input_data/wempx.dat' not found.")
    data = np.loadtxt("input_data/wempx.dat")
    W_all = data[:, 0]
    Q2_all = data[:, 1]

    # Determine unique W values in the grid:
    W_unique = np.unique(W_all)
    W_min, W_max = W_unique[0], W_unique[-1]

    # Create a smooth W‐grid:
    W_vals = np.linspace(W_min, W_max, num_points)

    # Prepare storage lists:
    W1_vals = []
    W2_vals = []
    xsec_vals = []

    # Loop over the smooth W‐grid:
    for W in W_vals:
        try:
            # Interpolate W1, W2 at (W, fixed_Q2)
            anl_w1, anl_w2 = interpolate_structure_functions("input_data/wempx.dat", W, fixed_Q2)
        except Exception:
            anl_w1, anl_w2 = np.nan, np.nan

        W1_vals.append(anl_w1)
        W2_vals.append(anl_w2)

        # Compute the ANL cross section at (W, fixed_Q2)
        try:
            cs_anl = compute_cross_section(W, fixed_Q2, beam_energy,
                                           file_path="input_data/wempx.dat", verbose=False)
        except Exception:
            cs_anl = np.nan
        xsec_vals.append(cs_anl)

    W1_vals = np.array(W1_vals)
    W2_vals = np.array(W2_vals)
    xsec_vals = np.array(xsec_vals)

    # ---- 1) Make the “W1 vs W” & “W2 vs W” plot ----
    plt.figure(figsize=(8, 6))
    plt.plot(W_vals, W1_vals, label="ANL W1", color="C0", linewidth=2)
    plt.plot(W_vals, W2_vals, label="ANL W2", color="C1", linewidth=2)
    plt.xlabel("W (GeV)")
    plt.ylabel("Structure functions")
    plt.title(f"ANL Model: W1 & W2 vs W (at Q²={fixed_Q2:.3f} GeV², E={beam_energy:.1f} GeV)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    png_name = f"ANL_struct_func_xsecs_Q2={fixed_Q2:.3f}_E={beam_energy:.1f}.png"
    plt.savefig(png_name, dpi=300)
    plt.close()
    print(f"  W1/W2 plot saved as '{png_name}'")

    # ---- 2) Write out the text table: [Q2, W, W1, W2, xsec] ----
    # Prepare a 2D array of shape (num_points, 5):
    #   column 0: fixed_Q2  (repeated)
    #   column 1: W_vals
    #   column 2: W1_vals
    #   column 3: W2_vals
    #   column 4: xsec_vals
    table_data = np.column_stack((
        np.full(W_vals.shape, fixed_Q2),
        W_vals,
        W1_vals,
        W2_vals,
        xsec_vals
    ))

    txt_name = f"ANL_struct_func_xsecs_Q2={fixed_Q2:.3f}_E={beam_energy:.1f}.txt"
    header = "Q2\tW\tW1\tW2\tCrossSection"
    np.savetxt(txt_name, table_data, fmt="%.6e", delimiter="\t", header=header)
    print(f"  Table saved as '{txt_name}'\n")
    
    

def compute_dsigma_dOmega_dEprime(W, Q2, E_beam, theta_deg, file_path="input_data/wempx.dat"):
    """
    Compute dσ/dΩ/dE′ using W1 and W2 from the structure function table.
    θ (in deg) is the scattering angle of the outgoing electron.
    """
    from math import sin, cos, radians
    from scipy.interpolate import RectBivariateSpline

    Mp = 0.9385
    alpha = 1 / 137.04

    theta = radians(theta_deg)
    denom = 2 * E_beam * (1 - cos(theta))
    if denom == 0:
        return np.nan
    E_prime = Q2 / denom

    if E_prime <= 0 or E_prime >= E_beam:
        return np.nan

    data = np.loadtxt(file_path)
    W_all = data[:, 0]
    Q2_all = data[:, 1]
    W1_all = data[:, 2]
    W2_all = data[:, 3]

    W_unique = np.unique(W_all)
    Q2_unique = np.unique(Q2_all)
    nW = len(W_unique)
    nQ2 = len(Q2_unique)
    W1_grid = W1_all.reshape(nW, nQ2)
    W2_grid = W2_all.reshape(nW, nQ2)

    interp_W1 = RectBivariateSpline(W_unique, Q2_unique, W1_grid)
    interp_W2 = RectBivariateSpline(W_unique, Q2_unique, W2_grid)
    W1 = interp_W1(W, Q2)[0, 0]
    W2 = interp_W2(W, Q2)[0, 0]

    prefactor = 4 * alpha**2 * E_prime**2 / Q2**2 * (0.197327**2) * 1e4  # in μb
    angle_part = 2 * sin(theta / 2)**2 * W1 + cos(theta / 2)**2 * W2
    return prefactor * angle_part



def plot_xsect_omega_energy(q2_list, theta_list, E_beam, file_path="input_data/wempx.dat"):
    """
    For a list of Q² and theta values, compute and plot dσ/dΩ/dE′ vs W at fixed E_beam.
    Each theta corresponds to Q² in q2_list.
    """
    assert len(q2_list) == len(theta_list), "Q² list and θ list must be of equal length"

    data = np.loadtxt(file_path)
    W_all = data[:, 0]
    W_unique = np.unique(W_all)
    W_min, W_max = np.min(W_unique), np.max(W_unique)
    W_grid = np.linspace(W_min + 1e-4, W_max - 1e-4, 250)

    fig, axes = plt.subplots(1, len(q2_list), figsize=(5 * len(q2_list),6))
    if len(q2_list) == 1:
        axes = [axes]

    for idx, (Q2, theta_deg) in enumerate(zip(q2_list, theta_list)):
        Y_vals = [compute_dsigma_dOmega_dEprime(W, Q2, E_beam, theta_deg, file_path) for W in W_grid]
        ax = axes[idx]
        ax.plot(W_grid, Y_vals, label=rf"$Q^2$ = {Q2} GeV², $\theta$ = {theta_deg:.1f}°", lw=2)
        ax.set_xlabel("W (GeV)")
        ax.set_ylabel(r"$d\sigma/d\Omega dE'$ (μb/GeV·sr)")
        ax.grid(True)
        ax.legend()
        if idx == 1:
            lee2 = np.loadtxt("Lee_exp_data/Lee_exp_data_2.txt")
            W_lee2 = lee2[:, 5]
            W = W_lee2 + 0.2
            xsec_lee2 = lee2[:, -2]/1e3
            err_lee2 = lee2[:, -1]/1e3
            ax.errorbar(W, xsec_lee2, yerr=err_lee2, fmt='o', color='black', label='Lee exp Q² #2')

        if idx == 2:
            lee3 = np.loadtxt("Lee_exp_data/Lee_exp_data_3.txt")
            W_lee3 = lee3[:, 5]
            W = W_lee3 - 0.5
            xsec_lee3 = lee3[:, -2]/1e3
            err_lee3 = lee3[:, -1]/1e3
            ax.errorbar(W, xsec_lee3, yerr=err_lee3, fmt='o', color='black', label='Lee exp Q² #3')

    plt.tight_layout()
    plt.savefig("xsect_dOmega_dEprime_vs_W.png", dpi=200)
    plt.close()
    print("Saved plot to xsect_dOmega_dEprime_vs_W.png")

    
    #--------------------------------------Auxillary functions fit_exp_data-----------------------------------------# 


def AMAX1(left, right):
    return max(left, right)

# BODEK PARAMETRIZATION
# Describes resonanses
def bodek(what_to_show, WM, QSQ,
          a2, a1,
          bodek_0, bodek_1, bodek_2, bodek_3, bodek_4,
          bodek_5, bodek_6, bodek_7, bodek_8, bodek_9, bodek_10):
    """
    Combined background/resonance:
      what_to_show=0 : full B_bg + B_res
      what_to_show=1 : background only (quadratic)
      what_to_show=2 : resonance only (old RESSUM*(1-BRES))
    
    Background: B_bg = a2*(W-1.076)**2 + a1*(W-1.076)
    Resonance: unchanged from before.
    """
    import math

    if WM < 0.94:
        return 0.0

    # constants
    PMSQ = 0.880324
    PM2  = 1.876512
    PM   = 0.938256
    NBKG = 5
    # compute BRES (uses old B1/B2) and RESSUM exactly as before
    # — we only use RESSUM*(1−BRES) for the resonance part
    # first build C (unchanged)
    C = [0.,
         1.0741163,0.75531124,3.3506491,1.7447015,3.5102405,1.14391,
         1.2299128,0.114735,0.621974,1.49826,0.12269,0.514898,
         1.71184,0.1177,0.51329,1.94343,0.202702,
         -0.17498537,0.0096701919,-0.035256748,
         3.5185207,-0.599937,4.7615828,0.41167589]
    # overwrite only the resonance parameters (the bodek_* args)
    C[6]  = bodek_3
    C[9]  = bodek_4
    C[12] = bodek_5
    C[15] = bodek_6
    C[8]  = bodek_7
    C[11] = bodek_8
    C[14] = bodek_9
    C[17] = bodek_10
    # the mass shifts
    C[16] = bodek_2

    WSQ   = WM*WM
    # compute old B1/B2 for BRES
    B1 = 0.0
    if WM != C[1]:
        B1 = max(0., WM-C[1])/(WM-C[1])*C[2]
    EB1 = C[3]*(WM-C[1])
    if EB1 <= 25.:
        B1 *= (1. - math.exp(-EB1))
    B2 = 0.0
    if WM != C[4]:
        B2 = max(0., WM-C[4])/(WM-C[4])*(1.-C[2])
    EB2 = C[5]*(WSQ - C[4]**2)
    if EB2 <= 25.:
        B2 *= (1. - math.exp(-EB2))
    BBKG = B1 + B2
    BRES = C[2] + B2

    # resonance sum
    RESSUM = 0.0
    for I in range(1,5):
        INDEX = (I-1)*3 + 1 + NBKG
        RAM   = C[INDEX]
        if I == 1:
            RAM += C[18]*QSQ + C[19]*QSQ**2
        RMA   = C[INDEX+1]
        if I == 3:
            RMA *= (1. + C[20]/(1. + C[21]*QSQ))
        RWD   = C[INDEX+2]
        # kinematics for resonance width
        QSTARN = math.sqrt(max(0., ((WSQ+PMSQ-(C[1]-PM)**2)/(2*WM))**2 - PMSQ))
        QSTARO = math.sqrt(max(0., ((RMA**2 - PMSQ + (C[1]-PM)**2)/(2*RMA))**2 - (C[1]-PM)**2))
        if QSTARO > 1e-10:
            TERM   = 6.08974 * QSTARN
            TERMO  = 6.08974 * QSTARO
            J      = 2* [0,1,2,3,2][I]
            GAMRES = RWD*(TERM/TERMO)**(J+1)*(1.+TERMO**J)/(1.+TERM**J)/2.
            BRWIG  = GAMRES/((WM-RMA)**2 + GAMRES**2)/math.pi
            RESSUM += RAM*BRWIG/PM2

    # build our two pieces
    B_res = RESSUM * (1. - BRES)
    B_bg  = a2*(WM - 1.076)**2 + a1*(WM - 1.076)

    if what_to_show == 0:
        return B_bg + B_res
    elif what_to_show == 1:
        return B_bg
    elif what_to_show == 2:
        return B_res




def fact(g2,x):
    result=1.135
    if (x > 0.1 and x < 0.2 and g2 < 0.5):
        return result
    if (x > 0.1 and g2 < 0.5):
        return result
    return 1.


def gp_h(q0, q2, a1, a2):
    """
    Background structure-function contribution parameterized as
      a1*(W - 1.076) + a2*(W - 1.076)^2
    then converted to the usual gp_h form.
    """
    pm = 0.938279
    # Compute kinematic W from q0, q2
    W2 = pm**2 + 2.0 * pm * q0 - q2
    if W2 <= 0:
        return 0.0
    W = math.sqrt(W2)
    dW = W - 1.076
    # Now convert to structure-function form
    xx = q2 / (2.0 * pm * q0)
    gi = 2.0 * pm * q0
    ww = (gi + 1.642) / (q2 + 0.376)
    t  = (1.-1./ww)
    #wp = 0.24035*t**3+a1*t**4+a2*t**5
    wp = a1*t**3+a2*t**7    # THE SHAPE OF SCALING FUNC!
    
    return wp * ww * q2 / (2.0 * pm * q0) * fact(q2, xx)
   

def w2h(what_to_show, gp2, gp0,
        a1, a2,
        bodek_0, bodek_1, bodek_2, bodek_3, bodek_4,
        bodek_5, bodek_6, bodek_7, bodek_8, bodek_9, bodek_10):
    pm = 0.938279
    WSQ_term = pm**2 + 2.0 * pm * gp0 - gp2
    if WSQ_term <= pm**2:
        return 0.0
    wi = math.sqrt(WSQ_term)
    # background term via new gp_h
    bg_term = gp_h(gp0, gp2, a1, a2)
    # resonance term via unchanged bodek
    res_term = bodek(what_to_show, wi, gp2, a2, a1,
                     bodek_0, bodek_1, bodek_2, bodek_3, bodek_4,
                     bodek_5, bodek_6, bodek_7, bodek_8, bodek_9, bodek_10)
    return bg_term * res_term / gp0

# Mott cross section
def gmott(ei,ue):
    g1=(1./137)**2*math.cos(ue/2.)**2
    g2=4.*ei**2*math.sin(ue/2.)**4
    result=g1/g2*0.389385*1000.*1000.
    return result

# Inelastic scattering on hydrogen
def h_inel(what_to_show, ei, er, ue,
           a1, a2,
           bodek_0, bodek_1, bodek_2, bodek_3, bodek_4,
           bodek_5, bodek_6, bodek_7, bodek_8, bodek_9, bodek_10):
    pm = 0.938279
    r = 0.18
    # Mott
    gm = gmott(ei, ue)
    # Compute w2h and w1h from updated w2h
    w2h_full = w2h(what_to_show, 4.*ei*er*math.sin(ue/2.)**2, ei-er,
                   a1, a2,
                   bodek_0, bodek_1, bodek_2, bodek_3, bodek_4,
                   bodek_5, bodek_6, bodek_7, bodek_8, bodek_9, bodek_10)
    w1h = (1.0 + (ei-er)**2/(4.*ei*er*math.sin(ue/2.)**2)) / (1.0 + r) * w2h_full
    result = gm * (w2h_full + 2.0 * math.tan(ue/2.0)**2 * w1h)
    return result

# plot background function only
def PlotBack(q2_in, w_in, a1, a2):
    # convert (q2_in, w_in) to scattering kinematics
    # then call gp_h directly
    # here we only need gp_h(q0=q0, q2=q2_in, a1,a2)
    # find q0 from w_in, q2_in via inversion: energy transfer
    pm = 0.938279
    # placeholder beam = 10.604 GeV
    q0 = (w_in**2 + q2_in - pm**2) / (2.0 * pm)
    return gp_h(q0, q2_in, a1, a2)

# uet is theta in deg
# er momentum in GeV
# the function that returns cross section for w, Q2 bin. It takes bodek parametrization parameters

#pp0, pp1, pp2, pp3, pp4, pp5, pp6, pp7, pp8, pp9 ,pp10

def getXSEC_fitting(what_to_show, q2_in, w_in,
                    a1, a2,
                    bodek_0, bodek_1, bodek_2, bodek_3, bodek_4,
                    bodek_5, bodek_6, bodek_7, bodek_8, bodek_9, bodek_10):
    
    #initialize beam conditions
    ei_0   = 10.604
    ei_1   = 0./1000.
    ei     = ei_0  + ei_1
    lepin = 7
    pi=3.14159
    
    #convert W and Q2 to theta and P
    mass_neut = 0.939565420
    omega_lab = (w_in**2 + q2_in - mass_neut**2 ) /  (2 * mass_neut)
    er = ei_0 - omega_lab
    
    uet = 2 * math.asin(math.sqrt(q2_in / (4 * er * ei_0))) *180/pi
    
    #constants
    pm=0.938279
    em=0.000511
    alfa=1./137.
    bt=4./3.
    r=0.18

    # NANOBARN 
    v_measure = 1.
    v_m = v_measure

    #scattered angle range determination
    ue=uet*pi/180.
    
    # sub_type_spec = 11.
    ero = ei / (1.+2*ei/pm*math.sin(ue/2.)**2)
    if(ero < er):
        print('idk 1')
        return 0
    
    gp0 = ei-er
    gp2 = 4.*ei*er*math.sin(ue/2.)**2
    gpv = math.sqrt(gp0**2+gp2)
    w2  = pm**2+2.*pm*gp0-gp2
    if(w2 < 0):
        print('idk 2')
        return 0
    w   = math.sqrt(w2)
    x   = gp2/2./pm/gp0
    
    if (x > 2.0):
        print('idk 3')
        return 0
    
    q2 = 4.0*ei*er*math.sin(ue/2.0)**2
    epsilon=1./(1.+2.*(1.+(ei-er)**2/q2)*(math.tan(ue/2))**2)
    gamma_t=alfa*(w2-pm**2)*er/4./q2/pm/ei/(1-epsilon)/pi**2
    gamma_w=alfa*(w2-pm**2)*w/8./q2/pm**2/(1-epsilon)/ei**2/pi**2
    jacob=pi*w/ei/er/pm
    
    #Inelastic scattering
    cs = h_inel(what_to_show, ei, er, ue,
                a1, a2,
                bodek_0, bodek_1, bodek_2, bodek_3, bodek_4,
                bodek_5, bodek_6, bodek_7, bodek_8, bodek_9, bodek_10)
    return round(cs * jacob/1000,10)

# Wrapper for cross section function
# Just for easier use of getXSEC_fitting. The main purpose is plotting
def getCS(what_to_show, q2_in, w_in,
          isBackPass=False, isBodekPass=False,
          backParams=None, bodekParams=None):
    """
    Wrapper for the two‑parameter background + 11‑parameter resonance cross section.

    what_to_show: 0=full, 1=background only, 2=resonance only
    q2_in, w_in : kinematics
    backParams : [a1, a2]
    bodekParams: [pp0…pp10]
    """
    # default resonance params (iteration‑3)
    default_bodek = [1.5, 1.711,1.94343,1.14391,0.621974,0.514898,0.513290,0.114735,0.122690,0.117700,0.202702]
    # default background params
    default_back = [0.2367, -0.0375943454827298]

    if backParams is None:
        backParams = default_back.copy()
    if bodekParams is None:
        bodekParams = default_bodek.copy()

    if len(backParams) != 2:
        raise ValueError("backParams must contain exactly 2 elements: [a1, a2]")
    if len(bodekParams) != 11:
        raise ValueError("bodekParams must contain exactly 11 elements (pp0…pp10)")

    # Delegate to the core fitting routine
    return getXSEC_fitting(
        what_to_show,
        q2_in,
        w_in,
        # background a1, a2
        backParams[0], backParams[1],
        # resonance params
        *bodekParams
    )

#----------------------------------------------------------------------------------------------------------------------#
def fit_exp_data_individual(q2_list, beam_energy, exp_file="bodek_fitting/exp_data_all.dat", output_png="fit_exp_data.png"):
    """
    For each Q² in q2_list, read residuals from exp_file (cols:
      Q2, W, eps, exp_minus_pdf, stat_error, sys_error, ScaleType),
    plot the residual data and overlay:
      • Full fit (background + resonances)
      • Background only
      • Resonance only
      • (optionally PDF + resonance)
    on a dynamically‐sized grid so there are no empty slots.
    """

    # --- load parameter tables ---
    def load_param_file(path, expected_count):
        param_dict = {}
        with open(path) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                tokens = [float(x.strip()) for x in line.strip().split(",")]
                q2 = tokens[0]
                params = tokens[1:]
                if len(params) != expected_count:
                    raise ValueError(f"Expected {expected_count} parameters, got {len(params)} in {path}")
                param_dict[q2] = params
        return param_dict

    bg_params_dict = load_param_file("bodek_fitting/bg_res_params/bg_params.dat", 2)
    res_params_dict = load_param_file("bodek_fitting/bg_res_params/res_params.dat", 11)

    # --- load your residual table ---
    data = np.loadtxt(exp_file, delimiter=",", skiprows=1)
    Q2_vals, W_all, _, yexp_all, err_stat, err_sys, _ = data.T
    dy = np.sqrt(err_stat**2 + err_sys**2)

    # --- dynamic grid size ---
    n = len(q2_list)
    if n == 0:
        raise ValueError("Need at least one Q² to plot.")
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5*ncols, 4*nrows),
                             squeeze=False)
    plt.subplots_adjust(wspace=0.3, hspace=0.4)

    # --- loop panels ---
    for idx, q2 in enumerate(q2_list):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]

        # retrieve parameters
        if q2 not in bg_params_dict or q2 not in res_params_dict:
            print(f"Skipping Q² = {q2:.3f}: parameters not found.")
            continue
        backParams = bg_params_dict[q2]
        bodekParams = res_params_dict[q2]

        # select this Q²
        mask = np.isclose(Q2_vals, q2)
        Wd, Yd, dY = W_all[mask], yexp_all[mask], dy[mask]

        # data
        ax.errorbar(Wd, Yd, yerr=dY, fmt='D', ms=5, color='red', label='Exp data')

        # common W grid
        Wg = np.linspace(1.15, 2.2, 300)

        # fit components
        Yfull = [getCS(0, q2, W, True, True, backParams, bodekParams) for W in Wg]
        Ybkg  = [getCS(1, q2, W, True, True, backParams, bodekParams) for W in Wg]
        Yres  = [getCS(2, q2, W, True, True, backParams, bodekParams) for W in Wg]

        ax.plot(Wg, Yfull, label='Full fit', color='black', lw=2)
        ax.plot(Wg, Ybkg,  label='Background', linestyle='--', color='orange')
        ax.plot(Wg, Yres,  label='Resonance', linestyle=':', color='blue')

        # optional: PDF + resonance
        F1i, F2i, _ = get_pdf_interpolators(q2)
        Ypdf = [compute_cross_section_pdf(W, q2, beam_energy, F1i, F2i) for W in Wg]
        Ypr = np.array(Ypdf)  # or add Yres if needed
        ax.plot(Wg, Ypr, label='PDF ONLY', linestyle='-.', color='green')

        # formatting
        ax.set_title(f"Q² = {q2:.3f} GeV²")
        ax.set_xlabel("W (GeV)")
        ax.set_ylabel(r"$d\sigma/dW/dQ^2$ (μb/GeV³)")
        ax.set_xlim(1.1, 2.6)
        ymax = max(Yd.max(), np.nanmax(Yfull), np.nanmax(Ypr))
        ax.set_ylim(0, ymax*1.1)
        ax.grid(True)
        ax.legend(fontsize="small")

    # turn off unused panels
    for empty in range(n, nrows*ncols):
        fig.delaxes(axes.flat[empty])

    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)
    print(f"Fit figure saved as {output_png}")

def fit_exp_data_global(q2_list, beam_energy, exp_file="bodek_fitting_all_Q2/exp_data_all.dat", output_png="fit_exp_data_global.png"):
    """
    Plot global fit for multiple Q² bins using shared (global) background and resonance parameters.
    Reads global parameters from:
      - bg_params_global.dat (2 params)
      - res_params_global.dat (11 params)
    """

    # --- Load global parameters ---
    def load_global_params(path, expected_count):
        with open(path) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                params = [float(x.strip()) for x in line.strip().split(",")]
                if len(params) != expected_count:
                    raise ValueError(f"Expected {expected_count} parameters in {path}, got {len(params)}")
                return params
        raise RuntimeError(f"No valid parameter line found in {path}")

    backParams = load_global_params("bodek_fitting_all_Q2/bg_res_params/bg_params_global.dat", 2)
    bodekParams = load_global_params("bodek_fitting_all_Q2/bg_res_params/res_params_global.dat", 11)

    # --- Load experimental residuals ---
    data = np.loadtxt(exp_file, delimiter=",", skiprows=1)
    Q2_vals, W_all, _, yexp_all, err_stat, err_sys, _ = data.T
    dy = np.sqrt(err_stat**2 + err_sys**2)

    # --- Dynamic grid layout ---
    n = len(q2_list)
    if n == 0:
        raise ValueError("Need at least one Q² to plot.")
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    plt.subplots_adjust(wspace=0.3, hspace=0.4)

    # --- Plot each Q² panel ---
    for idx, q2 in enumerate(q2_list):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]

        mask = np.isclose(Q2_vals, q2)
        Wd, Yd, dY = W_all[mask], yexp_all[mask], dy[mask]

        ax.errorbar(Wd, Yd, yerr=dY, fmt='D', ms=5, color='red', label='Exp data')

        W_max = 2.0
        Wg = np.linspace(1.15, 2.0, 300)

        Yfull = [getCS(0, q2, W, True, True, backParams, bodekParams) for W in Wg]
        Ybkg  = [getCS(1, q2, W, True, True, backParams, bodekParams) for W in Wg]
        Yres  = [getCS(2, q2, W, True, True, backParams, bodekParams) for W in Wg]

        ax.plot(Wg, Yfull, label='Full fit', color='black', lw=2)
        ax.plot(Wg, Ybkg,  label='Background', linestyle='--', color='orange')
        ax.plot(Wg, Yres,  label='Resonance', linestyle=':', color='blue')

        # PDF + optional comparison
        F1i, F2i, _ = get_pdf_interpolators(q2)
        Ypdf = [compute_cross_section_pdf(W, q2, beam_energy, F1i, F2i) for W in Wg]
        ax.plot(Wg, Ypdf, label='PDF ONLY', linestyle='-.', color='green')

        ax.set_title(f"Q² = {q2:.3f} GeV²")
        ax.set_xlabel("W (GeV)")
        ax.set_ylabel(r"$d\sigma/dW/dQ^2$ (μb/GeV³)")
        ax.set_xlim(1.1, 2.6)
        ymax = max(Yd.max(), np.nanmax(Yfull), np.nanmax(Ypdf))
        ax.set_ylim(0, ymax * 1.1)
        ax.grid(True)
        ax.legend(fontsize="small")

    # Turn off empty subplots
    for empty in range(n, nrows * ncols):
        fig.delaxes(axes.flat[empty])

    fig.tight_layout()
    fig.savefig(output_png, dpi=150)
    plt.close(fig)
    print(f"Fit figure saved as {output_png}")
