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
    
def compare_strfun(fixed_Q2, beam_energy, interp_file="input_data/wempx.dat", num_points=200):
    """
    Compares the interpolated cross section from ANL model with 
    experimentally measured and interpolated cross sections from strfun website https://clas.sinp.msu.ru/strfun/.

    The measured data is expected in the folder "strfun_data" with a file
    named "cs_Q2=<fixed_Q2>.dat" (using the exact fixed_Q2 value as typed by the user).
    That file should have a header and three columns: W, Quantity, and Uncertainty. Taken from the website.

    If fixed_Q2 is near 2.75 (within 0.01), a third dataset is loaded from the 
    "exp_data" folder with the file "InclusiveExpValera_Q2=2.774.dat". This file is 
    assumed to have a header and columns: W, eps, sigma, error, sys_error. The total 
    error is computed as sqrt(error² + sys_error²).

    The function:
      - Generates a fine grid of W values (from the interp_file) over the available range.
      - Computes the cross section at each W (using compute_cross_section) for the given fixed Q² and beam energy.
      - Loads the measured data from the appropriate file.
      - Plots the interpolated cross section as a smooth blue line (labeled "ANL model"),
        overlays the measured data as red markers with error bars (labeled "strfun website"),
        and, if applicable, overlays the experiment data as purple markers (labeled "experiment RGA").
      - Saves the plot as a PNG file with a filename that includes the fixed Q² and beam energy.

    Parameters:
      fixed_Q2 (float): Fixed Q² value.
      beam_energy (float): Beam energy in GeV.
      interp_file (str): Path to the interpolation file (default "input_data/wempx.dat").
      num_points (int): Number of W points for interpolation (default 200).
    """
    # Load interpolation file to get the available W range.
    data = np.loadtxt(interp_file)
    W_all = data[:, 0]
    W_unique = np.unique(W_all)
    W_min = W_unique[0]
    W_max = W_unique[-1]
    
    # Generate fine grid of W values.
    W_vals = np.linspace(W_min, W_max, num_points)
    cross_sections = []
    for w in W_vals:
        cs = compute_cross_section(w, fixed_Q2, beam_energy, file_path=interp_file, verbose=False)
        cross_sections.append(cs)
    
    # Load measured data from "strfun_data" folder.
    measured_filename = f"strfun_data/cs_Q2={fixed_Q2}.dat"
    if not os.path.isfile(measured_filename):
        raise FileNotFoundError(f"Measured data file {measured_filename} not found.")
    # The file is assumed to be tab-delimited with columns: W, Quantity, Uncertainty.
    measured_data = np.genfromtxt(measured_filename, names=["W", "Quantity", "Uncertainty"], delimiter="\t", skip_header=1)
    W_meas = measured_data["W"]
    quantity_meas = measured_data["Quantity"]
    uncertainty_meas = measured_data["Uncertainty"]
    
    plt.figure(figsize=(8, 6))
    plt.plot(W_vals, cross_sections, label="ANL model", color="blue", linewidth=2)
    plt.errorbar(W_meas, quantity_meas, yerr=uncertainty_meas, fmt="o", color="green", capsize=1, markersize=2, label="strfun website:CLAS and world data")
    
    # If fixed_Q2 is near 2.75, load additional experiment dataset.
    if abs(fixed_Q2 - 2.774) < 0.001:
        exp_filename = "exp_data/InclusiveExpValera_Q2=2.774.dat"
        if not os.path.isfile(exp_filename):
            raise FileNotFoundError(f"Experiment data file {exp_filename} not found.")
        # File assumed to have a header and columns: W, eps, sigma, error, sys_error.
        exp_data = np.genfromtxt(exp_filename, names=["W", "eps", "sigma", "error", "sys_error"], delimiter="\t", skip_header=1)
        exp_W = exp_data["W"]
        exp_sigma = exp_data["sigma"]*1e-3
        exp_error = np.sqrt(exp_data["error"]**2 + exp_data["sys_error"]**2)*1e-3
        plt.errorbar(exp_W, exp_sigma, yerr=exp_error, fmt="o", color="red", capsize=1, markersize=2, label="experiment RGA")
    
    plt.xlabel("W (GeV)")
    plt.ylabel("Cross Section (10⁻³⁰ cm²/GeV³)")
    plt.title(f"Cross Section Comparison at Q² = {fixed_Q2} GeV², Beam Energy = {beam_energy} GeV")
    plt.legend()
    plt.grid(True)
    
    filename = f"compare_strfun/compare_strfun_Q2={fixed_Q2}_E={beam_energy}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Comparison plot saved as {filename}")

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

def compare_exp_model_pdf(fixed_Q2, beam_energy, num_points=200):
    """
    Compares the PDF-based theoretical cross section with the ANL model cross section and experimental data,
    as a function of W. Also produces a 2×2 canvas that shows, versus W:
      Top left: Derived structure function W1 = F1/Mₚ.
      Top right: Derived structure function W2 = F2/ω.
      Bottom left: Raw structure function F1.
      Bottom right: Raw structure function F2.
    For each subplot the PDF-based (solid line) and ANL model (dashed line) curves are overlaid.
    A text table is also generated.
    """


    Mp = 0.9385
    # Get PDF interpolators from the PDF table.
    F1_W_interp, F2_W_interp, W_min = get_pdf_interpolators(fixed_Q2)
    # Define W grid (we use the PDF's minimum and a chosen upper bound, e.g. 2.5 GeV).
    W_vals = np.linspace(W_min, 2.6, num_points)

    # Arrays to store PDF-based structure functions.
    pdf_F1_vals = []   # raw F1 from PDF (via interpolation)
    pdf_F2_vals = []
    pdf_W1_vals = []   # derived: W1 = F1/Mp
    pdf_W2_vals = []   # derived: W2 = F2/ω

    # Arrays to store ANL model structure functions.
    anl_F1_vals = []   # raw F1 from ANL model = Mₚ * (W1 from ANL)
    anl_F2_vals = [] = []  # raw F2 from ANL model = ω * (W2 from ANL)
    anl_W1_vals = []   # directly interpolated W1 from ANL model
    anl_W2_vals = []   # directly interpolated W2 from ANL model

    # Also compute cross sections (for the upper plot)
    pdf_cross_sections = []
    anl_cross_sections = []

    for W in W_vals:
        # --- PDF-based structure functions
        try:
            F1_pdf = F1_W_interp(W)
            F2_pdf = F2_W_interp(W)
        except Exception:
            F1_pdf, F2_pdf = np.nan, np.nan
        pdf_F1_vals.append(F1_pdf)
        pdf_F2_vals.append(F2_pdf)
        W1_pdf = F1_pdf / Mp if not np.isnan(F1_pdf) else np.nan
        omega = (W**2 + fixed_Q2 - Mp**2) / (2.0 * Mp)
        W2_pdf = F2_pdf / omega if (omega > 0 and not np.isnan(F2_pdf)) else np.nan
        pdf_W1_vals.append(W1_pdf)
        pdf_W2_vals.append(W2_pdf)

        # --- ANL model structure functions (from input file)
        try:
            anl_w1, anl_w2 = interpolate_structure_functions("input_data/wempx.dat", W, fixed_Q2)
        except Exception:
            anl_w1, anl_w2 = np.nan, np.nan
        anl_W1_vals.append(anl_w1)
        anl_W2_vals.append(anl_w2)
        # Convert to raw F1 and F2: F1 = Mₚ * W1, F2 = ω * W2.
        F1_anl = Mp * anl_w1
        F2_anl = omega * anl_w2 if omega > 0 else np.nan
        anl_F1_vals.append(F1_anl)
        anl_F2_vals.append(F2_anl)

        # --- Cross sections for comparison plots.
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

    # --- Plot cross sections vs W.
    plt.figure(figsize=(8,6))
    plt.plot(W_vals, pdf_cross_sections, label="PDF model", color='green', linestyle='--')
    plt.plot(W_vals, anl_cross_sections, label="ANL-Osaka model", color='blue', linestyle='-')
    # Load experimental data:
    exp_file = f"exp_data/wempx.dat"  # (Or your actual experimental filename, here using same as before)
    # Here we assume experimental file exists; adjust as needed.
    exp_file = f"exp_data/InclusiveExpValera_Q2={fixed_Q2}.dat"
    if not os.path.isfile(exp_file):
        raise FileNotFoundError(f"Experimental data file {exp_file} not found.")
    exp_data = np.genfromtxt(exp_file, names=["W", "eps", "sigma", "error", "sys_error"], delimiter="\t", skip_header=1)
    plt.errorbar(exp_data["W"], exp_data["sigma"]*1e-3,
                 yerr=np.sqrt(exp_data["error"]**2+exp_data["sys_error"]**2)*1e-3,
                 fmt="o", color="red", label="Experimental data")
    plt.xlabel("W (GeV)")
    plt.ylabel("dσ/dW/dQ² (10⁻³⁰ cm²/GeV³)")
    plt.title(f"Cross Section vs W at Q² = {fixed_Q2} GeV², E = {beam_energy} GeV")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename_cs = f"cross_section_vs_W_comparison_Q2={fixed_Q2}_Ebeam={beam_energy}.png"
    plt.savefig(filename_cs, dpi=300)
    plt.close()
    print(f"Cross section vs W plot saved as {filename_cs}")

    # --- 2×2 Panel: Structure Functions vs W (Both PDF and ANL).
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    # Top left: W1 vs W
    axs[0, 0].plot(W_vals, pdf_W1_vals, label="PDF: W1 = F1/Mₚ", color="magenta", linestyle="-")
    axs[0, 0].plot(W_vals, anl_W1_vals, label="ANL-Osaka: W1", color="magenta", linestyle="--")
    axs[0, 0].set_xlabel("W (GeV)")
    axs[0, 0].set_ylabel("W1")
    axs[0, 0].set_title("W1 vs W")
    axs[0, 0].grid(True)
    axs[0, 0].legend()
    # Top right: W2 vs W
    axs[0, 1].plot(W_vals, pdf_W2_vals, label="PDF: W2 = F2/ω", color="orange", linestyle="-")
    axs[0, 1].plot(W_vals, anl_W2_vals, label="ANL-Osaka: W2", color="orange", linestyle="--")
    axs[0, 1].set_xlabel("W (GeV)")
    axs[0, 1].set_ylabel("W2")
    axs[0, 1].set_title("W2 vs W")
    axs[0, 1].grid(True)
    axs[0, 1].legend()
    # Bottom left: F1 vs W
    axs[1, 0].plot(W_vals, pdf_F1_vals, label="PDF: F1", color="blue", linestyle="-")
    axs[1, 0].plot(W_vals, anl_F1_vals, label="ANL-Osaka: F1", color="blue", linestyle="--")
    axs[1, 0].set_xlabel("W (GeV)")
    axs[1, 0].set_ylabel("F1")
    axs[1, 0].set_title("F1 vs W")
    axs[1, 0].grid(True)
    axs[1, 0].legend()
    # Bottom right: F2 vs W
    axs[1, 1].plot(W_vals, pdf_F2_vals, label="PDF: F2", color="green", linestyle="-")
    axs[1, 1].plot(W_vals, anl_F2_vals, label="ANL-Osaka: F2", color="green", linestyle="--")
    axs[1, 1].set_xlabel("W (GeV)")
    axs[1, 1].set_ylabel("F2")
    axs[1, 1].set_title("F2 vs W")
    axs[1, 1].grid(True)
    axs[1, 1].legend()
    fig.suptitle(f"Structure Functions vs W at Q² = {fixed_Q2} GeV², E = {beam_energy} GeV", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filename_sf = f"structure_functions_4plots_vs_W_Q2={fixed_Q2}_Ebeam={beam_energy}.png"
    #plt.savefig(filename_sf, dpi=300)
    plt.close()
    print(f"4-panel structure functions vs W plot saved as {filename_sf}")

    # Write text table with columns: Q2, W, PDF_W1, ANL_W1, PDF_W2, ANL_W2.
    table_data = np.column_stack((np.full(W_vals.shape, fixed_Q2), W_vals, pdf_W1_vals, anl_W1_vals, pdf_W2_vals, anl_W2_vals, anl_F1_vals, anl_F2_vals, pdf_F1_vals, pdf_F2_vals, pdf_cross_sections))
    table_filename = f"structure_functions_table_vs_W_Q2={fixed_Q2}_Ebeam={beam_energy}.txt"
    header_str = "Q2\tW\tPDF_W1\tANL_W1\tPDF_W2\tANL_W2\tANL_F1\tANL_F2\tPDF_F1\tPDF_F2\tPDF_CrossSection"
    np.savetxt(table_filename, table_data, fmt="%.6e", delimiter="\t", header=header_str)
    print(f"Structure functions table vs W saved as {table_filename}")


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
    plt.ylabel("dσ/dQ²dx (10⁻³⁰ cm²)")
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
    plt.ylabel("dσ/dQ²dξ (10⁻³⁰ cm²)")
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

    
    #--------------------------------------Auxillary functions fit_exp_data-----------------------------------------# 


def AMAX1(left, right):
    return max(left, right)

# BODEK PARAMETRIZATION
# Describes resonanses
def bodek(what_to_show, WM,QSQ, bodek_0, bodek_1, bodek_2, bodek_3, bodek_4, bodek_5, bodek_6, bodek_7, bodek_8, bodek_9, bodek_10):

    if (WM < 0.94): 
        return 0.
    
    PMSQ = 0.880324
    PM2 = 1.876512
    PM = 0.938256
    NRES = 4
    NBKG = 5  
    LSPIN = [0.,1,2,3,2]
    # first element just to shift index by 1 (pythom -> fortran) there is no other meaning in C[0] and L[0]
    LSPIN = [0.,1,2,3,2]

    #after elas:
    # it can be PRC parameters or not, 
    # it does not matter bacause the code will always overwrite them
    # Consider them as a placeholder
    # If you decide to use them, check what it is

    # constants - do not change. 
    C = [0.,1.0741163,0.75531124,3.3506491,1.7447015,3.5102405,1.14391,
         1.2299128,0.114735,0.621974,1.49826,0.12269,0.514898,
         1.71184,0.1177,0.51329,1.94343,0.202702,-0.17498537,
         0.0096701919,-0.035256748,3.5185207,-0.599937,4.7615828,0.41167589]

    # mass not used
    if (False):
        #C[7]=bodek_0
        #remove peak shift
        C[10]=bodek_0
        C[13]=bodek_1
            
    C[16]=bodek_2
    #amplitude
        
    C[6]=bodek_3
    C[9]=bodek_4
    C[12]=bodek_5
    C[15]=bodek_6
    
    #width
    C[8]=bodek_7
    C[11]=bodek_8
    C[14]=bodek_9
    C[17]=bodek_10
    
    WSQ=WM*WM
    OMEGA=1.+(WSQ-PMSQ)/QSQ                                           
    X=1./OMEGA                                                        
    XPX=C[22]+C[23]*(X-C[24])**2                                      
    PIEMSQ=(C[1]-PM)**2                                               

    # added part (Misak's comment)
    B1 = 0.0
    B2 = 0.0
    # 0/0
    if (WM != C[1]):
        B1=AMAX1(0.,(WM-C[1]))/(WM-C[1])*C[2]
    EB1=C[3]*(WM-C[1])
    if (EB1 <= 25.):
        B1=B1*(1.0-math.exp(-EB1))
        
    if (WM != C[4]):
        #AMAX1
        B2=AMAX1(0.,(WM-C[4]))/(WM-C[4])*(1.-C[2])

    EB2=C[5]*(WSQ-C[4]**2)                                           
    if (EB2 <= 25.):
        B2=B2*(1.-math.exp(-EB2))
    BBKG=B1+B2
    BRES=C[2]+B2                                                      
    RESSUM=0.

    for I in range(1,5,1):
        INDEX=(I-1)*3+1+NBKG
        #amplitude
        RAM=C[INDEX]
        #print(INDEX, C[INDEX])
        if (I == 1):
            RAM=C[INDEX]+C[18]*QSQ+C[19]*QSQ**2 
        #IF(I.EQ.1)RAM=C(INDEX)+C(18)*QSQ+C(19)*QSQ**2
        # mass
        #print(INDEX+1, C[INDEX+1])
        RMA=C[INDEX+1]
        if (I == 3):
            RMA=RMA*(1.+C[20]/(1.+C[21]*QSQ))
        #IF(I.EQ.3)RMA=RMA*(1.+C(20)/(1.+C(21)*QSQ))                       A1506350
        
        RWD=C[INDEX+2]
        QSTARN=math.sqrt(AMAX1(0.,((WSQ+PMSQ-PIEMSQ)/(2.*WM))**2-PMSQ))
        QSTARO=math.sqrt(AMAX1(0.,((RMA**2-PMSQ+PIEMSQ)/(2.*RMA))**2-PIEMSQ))
        RES = 0
        #IF(QSTARO.LE.1.E-10)GO TO 40                                      A1506390
        if (QSTARO < 1e-10):
            RES = 0
        else:
            TERM=6.08974*QSTARN
            TERMO=6.08974*QSTARO
            J=2*LSPIN[I]
            K=J+1
            GAMRES=RWD*(TERM/TERMO)**K*(1.+TERMO**J)/(1.+TERM**J)
            GAMRES=GAMRES/2.
            BRWIG=GAMRES/((WM-RMA)**2+GAMRES**2)/3.1415926
            RES=RAM*BRWIG/PM2
            RESSUM=RESSUM+RES   
            
    if (what_to_show == 0):                                              
        B=BBKG*(1.+(1.-BBKG)*XPX)+RESSUM*(1.-BRES)
    elif(what_to_show == 1):
        B=BBKG*(1.+(1.-BBKG)*XPX)
    elif(what_to_show == 2):
        B=RESSUM*(1.-BRES)
        
    
    return B



def fact(g2,x):
    result=1.135
    if (x > 0.1 and x < 0.2 and g2 < 0.5):
        return result
    if (x > 0.1 and g2 < 0.5):
        return result
    return 1.


# background function,
def gp_h(q0,q2, backg_0, backg_1, backg_2, backg_3, backg_4):
    pm=0.938279
    pi=3.14159
    xx = q2/(2.*pm*q0)
    gi = 2.*pm*q0
    ww = (gi+1.642)/(q2+0.376)
    t  = (1.-1./ww)
  
    wp = 0.24035*t**3+backg_1*t**4+backg_2*t**5+backg_3*t**6+backg_4*t**7
    result=wp*ww*q2/(2.*pm*q0)*fact(q2,xx)
    
    return result
   

def w2h(what_to_show, gp2,gp0, backg_0, backg_1, backg_2, backg_3, backg_4, 
        bodek_0, bodek_1, bodek_2, bodek_3, bodek_4, bodek_5, bodek_6, bodek_7, bodek_8, bodek_9, bodek_10):
    
    pm=0.938279
    pi=3.14159
    fm2=pm**2+2.*pm*gp0-gp2
    w2h=0.
    if (fm2 < pm**2):
        return 0
    wi=math.sqrt(fm2)
    # gp_h and b
    result=gp_h(gp0, gp2, backg_0, backg_1, backg_2, backg_3, backg_4) * bodek(what_to_show, wi, gp2, bodek_0, bodek_1, bodek_2, bodek_3, bodek_4, bodek_5, bodek_6, bodek_7, bodek_8, bodek_9, bodek_10) / gp0
    return  result

# Mott cross section
def gmott(ei,ue):
    g1=(1./137)**2*math.cos(ue/2.)**2
    g2=4.*ei**2*math.sin(ue/2.)**4
    result=g1/g2*0.389385*1000.*1000.
    return result

# Inelastic scattering on hydrogen
def h_inel(what_to_show, ei,er,ue,
           # background
           backg_0, backg_1, backg_2, backg_3, backg_4,
           # bodek params
           bodek_0, bodek_1, bodek_2, bodek_3, bodek_4, bodek_5, bodek_6, 
           bodek_7, bodek_8, bodek_9, bodek_10
          ):
    
    pm=0.938279
    pi=3.14159
    r   = 0.18
    gp0 = ei-er
    gp2 = 4.*ei*er*math.sin(ue/2.)**2
    gpv = math.sqrt(gp2+gp0**2)
    # MOTT CS
    gm  = gmott(ei,ue)
    #W2H
    w1h = (1.+gp0**2/gp2)/(1.+r)*w2h(what_to_show, gp2,gp0, 
                                     # background:
                                     backg_0, backg_1, backg_2, backg_3, backg_4, 
                                     # bodek:
                                     bodek_0, bodek_1, bodek_2, bodek_3, bodek_4, 
                                     bodek_5, bodek_6, bodek_7, bodek_8, 
                                     bodek_9, bodek_10)
    w2  = pm**2+2.*pm*gp0-gp2
    w = 0.
    if (w2 < 0.):
        w = 0.
    else:
        w = math.sqrt(w2)
        
    result = (gm*(w2h(what_to_show, gp2, gp0, backg_0, backg_1, backg_2, backg_3, backg_4,
                      bodek_0, bodek_1, bodek_2, bodek_3, bodek_4, bodek_5, 
                      bodek_6, bodek_7, bodek_8, bodek_9, bodek_10)+2.*math.tan(ue/2.)**2*w1h))

    return result

# plot background function only
def PlotBack(q2_in, w_in, backg_0, backg_1, backg_2, backg_3, backg_4):
    
    #initialize beam conditions
    ei_0   = 10.604
    ei_1   = 0./1000.
    ei     = ei_0  + ei_1
    lepin = 7
    pi=3.14159
    
    #W and Q2 to theta and P
    mass_neut = 0.939565420
    omega_lab = (w_in**2 + q2_in - mass_neut**2 ) /  (2 * mass_neut)
    er = ei_0 - omega_lab
    
    uet = 2 * math.asin(math.sqrt(q2_in / (4 * er * ei_0))) *180/pi
    
    #BLOCK OF PARAMETERS
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
    
    
    
    pm=0.938279
    pi=3.14159
    r   = 0.18
    gp0 = ei-er
    gp2 = 4.*ei*er*math.sin(ue/2.)**2
    gpv = math.sqrt(gp2+gp0**2)
    # MOTT CS
    gm  = gmott(ei,ue)
    #W2H

    return gp_h(gp0, gp2, backg_0, backg_1, backg_2, backg_3, backg_4);

# uet is theta in deg
# er momentum in GeV
# the function that returns cross section for w, Q2 bin. It takes bodek parametrization parameters

#pp0, pp1, pp2, pp3, pp4, pp5, pp6, pp7, pp8, pp9 ,pp10

def getXSEC_fitting(what_to_show, q2_in, w_in, 
                    # background
                    backg_0, backg_1, backg_2, backg_3, backg_4,
                    # bodek params
                    bodek_0, bodek_1, bodek_2, bodek_3, bodek_4, bodek_5, bodek_6, 
                    bodek_7, bodek_8, bodek_9, bodek_10
                   ):
    
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
    cs_y_in = h_inel(what_to_show, ei,er,ue,                    
                     # background
                     backg_0, backg_1, backg_2, backg_3, backg_4,
                     # bodek params
                     bodek_0, bodek_1, bodek_2, bodek_3, bodek_4, bodek_5, bodek_6, 
                     bodek_7, bodek_8, bodek_9, bodek_10) / v_m
    
    # Jacobian (theta, P - > w, Q2) and to nb
    cs_y_in_w = round(cs_y_in*jacob/1000,10)
    return cs_y_in_w

# Wrapper for cross section function
# Just for easier use of getXSEC_fitting. The main purpose is plotting
def getCS(what_to_show, q2_in, w_in, isBackPass = False, isBodekPass = False, backParams = [], bodekParams = []):
    # Number of varied params:
    # it is hardcoded in other places (for example in input of getXSEC_fitting()), so do not change it
    totalParamsBodek = 11
    totalParamsBckgr = 4

    # Default params. (the one submitted to PRC)
    if (not isBackPass):
        # PRC:
        #print("PRC backgrd fun is used")
        #backParams = [0.2367, 2.178, 0.898, -6.726, 3.718]

        # Iterated by Valerii after PRC:
        backParams = [0.2367, 2.185651400350511, 0.6782367867507779, -6.735990002785817, 4.163236272156371]

        
    if (not isBodekPass):
        # PRC:
        bodekParams = [1.5,1.711,1.94343, 1.14391, 6.21974e-01,  5.14898e-01,
                       5.13290e-01 , 1.14735e-01, 1.22690e-01, 1.17700e-01, 2.02702e-01]
        

    if (len(backParams) < totalParamsBckgr):
        raise Exception("backParams is not defined, make sure that you passed backParams, if isBackPass = True ")

    if (len(bodekParams) < totalParamsBodek):
        raise Exception("backParams is not defined, make sure that you passed isBodekPass, if isBodekPass = True ")

    # Call XSEC function
    return getXSEC_fitting(what_to_show, q2_in, w_in, *backParams, *bodekParams)

#----------------------------------------------------------------------------------------------------------------------#
def fit_exp_data(q2_list, exp_file="exp_data_all.dat",beam_energy=10.6):
    """
    For each Q² in q2_list, read exp_minus_pdf.txt (columns:
      Q2, W, epsilon, exp_minus_pdf, stat_error, sys_error, ScaleType),
      • Full fit (background + resonance)
      • Background only
      • Resonance only
    Saves a 3×3 panel to 'fit_exp_data.png'.
    """

    # load the residual table
    data = np.loadtxt(exp_file, delimiter=",", skiprows=1)
    Q2_vals, W_data_all, eps_all, yexp_all, err_stat, err_sys, scale_type = data.T
    dyexp_all = np.sqrt(err_stat**2 + err_sys**2 )

    # the specific params you found for "exp minus pdf"
    backParams = [0.2367, 2.185651400350511, 0.6782367867507779, -6.735990002785817, 4.163236272156371]
    bodekParams = [1.5, 1.711, 1.9455061694828741, 1.1508931812039305, 0.622807570128967, 0.5140353260311096, 0.5197534788784222, 0.10792042803309153, 0.12666581366215277, 0.1179007082920964, 0.20173174661509075]

    # prepare 3×3 canvas
    fig, axes = plt.subplots(3, 3, figsize=(45, 25), squeeze=False)
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9,
                        wspace=0.25, hspace=0.55)

    # only plot up to 9 Q² values
    for idx, q2 in enumerate(q2_list[:9]):
        row, col = divmod(idx, 3)
        ax = axes[row][col]

        # select only points for this Q²
        mask = np.isclose(Q2_vals, q2)
        W_data = W_data_all[mask]
        Y_data = yexp_all[mask]
        dY_data = dyexp_all[mask]

        # plot residual data
        ax.errorbar(W_data, Y_data, yerr=dY_data, fmt='D', ms=5, color='red', label='exp data')

        # define W grid for model curves
        W_grid = np.linspace(1.1, 2.6, 300)

        # full fit: background + resonance
        Y_full = [getCS(0, q2, W, isBackPass=True, isBodekPass=True, backParams=backParams, bodekParams=bodekParams) for W in W_grid ]
        ax.plot(W_grid, Y_full, color='black', linewidth=2, label='Full fit')

        # background only (use PlotBack, convert to μb)
        Y_bkg = [getCS(1, q2, W, isBackPass=True, isBodekPass=True, backParams=backParams, bodekParams=bodekParams) for W in W_grid ]
        #ax.plot(W_grid, Y_bkg, color='orange', linestyle='--', linewidth=2, label='Background only')

        # resonance only (bodek, convert to μb)
        Y_res = [getCS(2, q2, W, isBackPass=True, isBodekPass=True, backParams=backParams, bodekParams=bodekParams) for W in W_grid ]
        #ax.plot(W_grid, Y_res, color='blue', linestyle=':', linewidth=2, label='Resonance only')

        F1i, F2i, _ = get_pdf_interpolators(q2)
        Y_pdf = [ compute_cross_section_pdf(W, q2, beam_energy, F1i, F2i) for W in W_grid]
        #ax.plot(W_grid, Y_pdf, color='green', linestyle='-.', linewidth=2, label='PDF prediction')
        
        Y_pdf_plus_res = [Y_pdf[i] + Y_res[i] for i in range(len(Y_pdf))]
        ax.plot(W_grid, Y_pdf_plus_res, color='purple', linestyle='--', linewidth=2, label='PDF + Resonance')
        
        # formatting
        ax.set_title(f"Q² = {q2:.3f} GeV²", fontsize=50, loc='right')
        ax.set_xlabel('W (GeV)', fontsize=40)
        ax.set_ylabel(r"$d\sigma/dWdQ^2$ (μb/GeV³)", fontsize=40)
        ax.grid(True)
        ax.legend(fontsize=20)
        ax.set_xlim([1.1, 2.6])

        # dynamic y‐limit based on all curves
        Y_pdf_plus_res = [Y_pdf[i] + Y_res[i] for i in range(len(Y_pdf))]
        ymax = max(Y_data.max(), np.nanmax(Y_full), np.nanmax(Y_pdf_plus_res))
        ax.set_ylim([0, ymax * 1.1])

    # turn off unused subplots if fewer than 9 Q²
    for idx in range(len(q2_list), 9):
        fig.delaxes(axes.flat[idx])

    fig.savefig('fit_exp_data_PDF_plus_RES.png', bbox_inches='tight', dpi=50)
    plt.close(fig)
    print("Fit figure saved as fit_exp_data.png")


def exp_data_minus_pdf_table(q2_list, beam_energy, output_filename="exp_minus_pdf.txt"):
    """
    For each Q² in q2_list, load the experimental RGA data,
    subtract the PDF-based cross section at each (W,Q2) point,
    and write a single .txt file with columns:
      Q2, W, epsilon, (exp − PDF) xsec, stat_error, sys_error
    """

    rows = []
    for q2 in q2_list:
        exp_path = f"exp_data/InclusiveExpValera_Q2={q2}.dat"
        if not os.path.isfile(exp_path):
            raise FileNotFoundError(f"Missing experimental file: {exp_path}")
        df = pd.read_csv(exp_path, sep=r"\s+")
        F1i, F2i, _ = get_pdf_interpolators(q2)

        for _, r in df.iterrows():
            W   = r["W"]
            eps = r["eps"]
            sigma_exp = r["sigma"]     * 1e-3  # to μb
            stat_err  = r["error"]      * 1e-3
            sys_err   = r["sys_error"]  * 1e-3

            # PDF-based cross section at this point
            cs_pdf = compute_cross_section_pdf(W, q2, beam_energy, F1i, F2i)

            rows.append([q2, W, eps, sigma_exp - cs_pdf, stat_err, sys_err, 0])

    data = np.array(rows)
    header = "Q2,W,epsilon,exp_minus_pdf,stat_error,sys_error,ScaleType"
    np.savetxt(output_filename, data, fmt="%.6e", delimiter=",", header=header)
    print(f"Residual table saved to {output_filename}")

