import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import math
import sys
sys.path.append('.')  # Ensure current directory is in path
from functions import compute_cross_section as anl_cross_section

# Constants
Mp = 0.9385  # Proton mass in GeV
Q2_fixed = 2.774
beam_energy = 10.6  # Example beam energy in GeV 

# Load PDF table
filename = f'PDF_tables/tst_CJpdf_ISET=400_Q2={Q2_fixed}.dat'
pdf_table = pd.read_csv(filename, delim_whitespace=True)

# Assuming columns: x, u, ub, d, db
x = pdf_table['x'].values
nu = pdf_table['u'].values
nub = pdf_table['ub'].values
nd = pdf_table['d'].values
ndb = pdf_table['db'].values

# Compute F2 and F1 as functions of x
F2_x = (4/9)*(nu + nub) + (1/9)*(nd + ndb)
F1_x = F2_x / (2 * x)

# Compute W for each x (fixed Q2)
W2 = Mp**2 + Q2_fixed * (1 - x) / x
W = np.sqrt(W2)

# Interpolate F1 and F2 as functions of W
def interpolate_structure_function(W_array, F_array):
    sorted_indices = np.argsort(W_array)
    W_sorted = W_array[sorted_indices]
    F_sorted = F_array[sorted_indices]
    interp_func = interp1d(W_sorted, F_sorted, kind='cubic', bounds_error=False, fill_value="extrapolate")
    return interp_func

F1_W_interp = interpolate_structure_function(W, F1_x)
F2_W_interp = interpolate_structure_function(W, F2_x)

# Compute cross section using interpolated F1, F2
def compute_cross_section(W, Q2, beam_energy):
    alpha = 1 / 137.04
    pi = np.pi
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
    fcrs3 = 4 * (alpha / Q2)**2 * (0.197327**2) * 1e4 * elepf**2

    ss2 = (1 - clep) / 2
    cc2 = (1 + clep) / 2

    F1 = F1_W_interp(W)
    F2 = F2_W_interp(W)
    W1 = F1 / Mp
    W2 = F2 / omeg

    xxx = 2 * ss2 * W1 + cc2 * W2
    dcrs = fcrs3 * fac3 * xxx
    return dcrs

# Plot cross section vs W with experimental and ANL data
def plot_cross_section_vs_W(Q2, beam_energy, exp_data_file, anl_data_file, num_points=200):
    W_min, W_max = min(W), 2.5
    W_vals = np.linspace(W_min, W_max, num_points)
    cross_sections = []
    anl_cross_sections = []

    for Wi in W_vals:
        try:
            cs_theory = compute_cross_section(Wi, Q2, beam_energy)
        except Exception:
            cs_theory = np.nan
        cross_sections.append(cs_theory)

        try:
            cs_anl = anl_cross_section(Wi, Q2, beam_energy, anl_data_file, verbose=False)
        except Exception:
            cs_anl = np.nan
        anl_cross_sections.append(cs_anl)

    # Load experimental data correctly with sep='\s+'
    exp_data = pd.read_csv(exp_data_file, sep='\s+')
    W_exp = exp_data['W'].values
    sigma_exp = exp_data['sigma'].values*1e-3
    total_err = np.sqrt(exp_data['error']**2 + exp_data['sys_error']**2)*1e-3

    plt.figure(figsize=(8, 6))
    plt.plot(W_vals, cross_sections, label="Theory (from PDF)", color='green', linestyle='--')
    plt.plot(W_vals, anl_cross_sections, label="ANL Model", color='blue')
    plt.errorbar(W_exp, sigma_exp, yerr=total_err, fmt='o', color='red', label="Experimental data")
    plt.xlabel("W (GeV)")
    plt.ylabel("dσ/dW/dQ² (10⁻³⁰ cm²/GeV³)")
    plt.title("Differential Cross Section vs W")
    plt.grid(True)
    plt.legend()
    plt.xlim(W_min, 2.5)
    plt.tight_layout()

    # Save the figure as PNG
    plt.savefig(f"cross_section_vs_W_comparison_Q2={Q2_fixed}_Ebeam={beam_energy}.png", dpi=300)
    plt.close()

# Example call to plot function with experimental data and ANL model data files
exp_data_path = f'exp_data/InclusiveExpValera_Q2={Q2_fixed}.dat'
anl_data_path = 'input_data/wempx.dat'
plot_cross_section_vs_W(Q2_fixed, beam_energy, exp_data_path, anl_data_path)
