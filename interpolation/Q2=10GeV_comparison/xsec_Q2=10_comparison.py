import numpy as np
import math
import matplotlib.pyplot as plt
import os

# -------------------------
# Define kinematic constants
fnuc = 0.9385         # nucleon mass in GeV
pi_val = 3.1415926
alpha = 1/137.04      # fine-structure constant
conv_factor = 0.197327  # GeV*fm conversion factor (squared later)

# -------------------------
# User input: beam energy
try:
    beam_energy = float(input("Enter beam (lepton) energy in GeV: "))
except ValueError:
    print("Invalid beam energy.")
    exit(1)

# Fixed Q² for model (should be 10 GeV for this file)
fixed_Q2 = 10.0

# -------------------------
# Load model data from "w1w2-10.dat"
model_filename = "w1w2-10.dat"
if not os.path.isfile(model_filename):
    raise FileNotFoundError(f"Model data file {model_filename} not found.")

# Load data assuming the file has a header and five columns, but we use only first four:
# Columns: Q2, W, W1, W2
model_data = np.loadtxt(model_filename, skiprows=1, usecols=(0,1,2,3))
# Extract columns
Q2_model = model_data[:, 0]
W_model = model_data[:, 1]
W1_model = model_data[:, 2]
W2_model = model_data[:, 3]

# Check that Q² is essentially constant and near fixed_Q2:
if not np.allclose(Q2_model, fixed_Q2, atol=1e-3):
    raise ValueError(f"Model file Q² values differ from expected fixed Q² = {fixed_Q2}")

# Get unique W values (assuming the model file contains multiple rows with different W)
W_unique = np.unique(W_model)
W_unique.sort()

# -------------------------
# Define function to calculate differential cross section
def calc_xsec(W, Q2, E, W1, W2):
    """
    Calculates the differential cross section dσ/dW/dQ² for an electromagnetic reaction.
    
    Uses the following steps (with massless leptons):
      - Total lab energy: w_tot = sqrt(2*m_N*E + m_N^2)
      - Energy transfer: ω = (W^2 + Q2 - m_N^2) / (2*m_N)
      - Final lepton energy: elepf = E - ω
      - cosθ = (-Q2 + 2*E*elepf) / (2*E*elepf)
      - Common factor: fac3 = π * W / (m_N * E * elepf)
      - Reaction-dependent factor: fcrs3 = 4*(alpha/Q2)^2*(conv_factor^2)*1e4*(elepf^2)
      - Angular factors: ss2 = (1 - cosθ)/2, cc2 = (1 + cosθ)/2
      - Cross section: dσ/dW/dQ² = fcrs3 * fac3 * (2*ss2*W1 + cc2*W2)
      
    Returns:
        Cross section (float)
    """
    # Total available energy
    w_tot = math.sqrt(2 * fnuc * E + fnuc**2)
    if W > w_tot:
        raise ValueError("W exceeds available lab energy")
    
    # Energy transfer and final lepton energy
    omega = (W**2 + Q2 - fnuc**2) / (2 * fnuc)
    elepf = E - omega
    if elepf <= 0:
        raise ValueError("Final lepton energy is non-positive")
    
    # Cosine of lepton scattering angle
    cos_theta = (-Q2 + 2 * E * elepf) / (2 * E * elepf)
    
    fac3 = pi_val * W / (fnuc * E * elepf)
    fcrs3 = 4 * (alpha / Q2)**2 * (conv_factor**2) * 1e4 * (elepf**2)
    ss2 = (1 - cos_theta) / 2
    cc2 = (1 + cos_theta) / 2
    return fcrs3 * fac3 * (2 * ss2 * W1 + cc2 * W2)

# -------------------------
# Calculate model cross sections for each unique W
model_xsec = []
for w in W_unique:
    # Find the first row in model_data corresponding to this W
    mask = np.isclose(W_model, w, atol=1e-6)
    if not np.any(mask):
        model_xsec.append(np.nan)
        continue
    idx = np.argmax(mask)
    W1_val = W1_model[idx]
    W2_val = W2_model[idx]
    try:
        cs = calc_xsec(w, fixed_Q2, beam_energy, W1_val, W2_val)
    except Exception as err:
        cs = np.nan
    model_xsec.append(cs)
model_xsec = np.array(model_xsec)

# -------------------------
# Load experimental data from "exp_data/InclusiveExpValera_Q2=9.699.dat"
exp_filename = "../exp_data/InclusiveExpValera_Q2=9.699.dat"
if not os.path.isfile(exp_filename):
    raise FileNotFoundError(f"Experimental data file {exp_filename} not found.")

# Load experimental data (assumed whitespace or tab-delimited)
exp_data = np.genfromtxt(exp_filename, names=True)
# Expected columns: W, eps, sigma, error, sys_error.
W_exp = exp_data["W"]
sigma_exp = exp_data["sigma"] * 1e-3  # multiply by 1e-3
error_exp = exp_data["error"] * 1e-3
sys_error_exp = exp_data["sys_error"] * 1e-3
total_error = np.sqrt(error_exp**2 + sys_error_exp**2)

# -------------------------
# Plot cross section comparison
plt.figure(figsize=(8,6))
plt.plot(W_unique, model_xsec, label="ANL model", color="blue", linewidth=2)
plt.errorbar(W_exp, sigma_exp, yerr=total_error, fmt="o", color="red", capsize=5, label="experiment RGA")
plt.xlabel("W (GeV)")
plt.ylabel("Cross Section (10⁻³⁰ cm²/GeV³)")
plt.title(f"Cross Section Comparison at Q² = {fixed_Q2} GeV², Beam Energy = {beam_energy} GeV")
plt.legend()
plt.grid(True)

output_filename = "xsec_Q2=10_comparison.png"
plt.savefig(output_filename, dpi=300)
plt.close()
print(f"Comparison plot saved as {output_filename}")

# -------------------------
# Additionally, plot structure functions W1 and W2 as functions of W on one graph.
model_W1 = []
model_W2 = []
for w in W_unique:
    mask = np.isclose(W_model, w, atol=1e-6)
    if not np.any(mask):
        model_W1.append(np.nan)
        model_W2.append(np.nan)
    else:
        idx = np.argmax(mask)
        model_W1.append(W1_model[idx])
        model_W2.append(W2_model[idx])
model_W1 = np.array(model_W1)
model_W2 = np.array(model_W2)

plt.figure(figsize=(8,6))
plt.plot(W_unique, model_W1, label="W1", color="green", linewidth=2)
plt.plot(W_unique, model_W2, label="W2", color="purple", linewidth=2)
plt.xlabel("W (GeV)")
plt.ylabel("Structure Function")
plt.title(f"Structure Functions vs W at Q² = {fixed_Q2} GeV²")
plt.legend()
plt.grid(True)

output_filename_sf = "structure_functions_vs_W.png"
plt.savefig(output_filename_sf, dpi=300)
plt.close()
print(f"Structure functions plot saved as {output_filename_sf}")
