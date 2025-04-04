import numpy as np
import math
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
import os

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
    filename = f"plots_cs/cross_section_vs_W_Q2={Q2:.3f}_E={beam_energy:.3f}.png"
    
    # Save the plot as a PNG file
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Plot saved as {filename}")

def generate_table(file_path, fixed_Q2, beam_energy):
    """
    Generates a text table with columns: Q2, W, W1, W2, and CrossSection.
    The table is generated for a fixed Q2 value (the nearest grid Q2 is used)
    by extracting all rows from the input file that correspond to that grid Q2.
    For each row, the cross section is computed using the fixed Q2, the grid's W,
    and the structure functions from the input file.
    
    The table is saved as a tab-delimited text file.

    Parameters:
        file_path (str): Path to the input data file (W, Q2, W1, W2)
        fixed_Q2 (float): The Q2 value for which the table is generated.
        beam_energy (float): The beam (lepton) energy in GeV.
        output_filename (str): The name of the output text file.
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
    output_filename = f"tables/ANL_model_CS_Q2={fixed_Q2}.txt"
    np.savetxt(output_filename, np.array(output_rows), header=header, fmt="%.6e", delimiter="\t")
    print(f"Table saved as {output_filename}")
    
def compare_strfun(fixed_Q2, beam_energy, interp_file="input_data/wempx.dat", num_points=200):
    """
    Compares the interpolated cross section with measured structure function data.

    The measured data is expected in the folder "strfun_data" with a file
    named "cs_Q2=<fixed_Q2>.dat" (using the exact fixed_Q2 value as typed by the user).
    That file should have a header and three columns: W, Quantity, and Uncertainty.

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
    plt.errorbar(W_meas, quantity_meas, yerr=uncertainty_meas, fmt="o", color="green", capsize=1,markersize=2, label="strfun website:CLAS and world data")
    
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