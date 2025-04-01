import numpy as np
import math

def get_structure_functions(W_input, Q2_input, filename):
    """
    Reads the structure-function table from the given file and selects the 
    nearest grid point to the provided (W_input, Q2_input). The file is assumed 
    to have 4 columns: W, Q2, W1, and W2.
    
    Returns:
        W_near, Q2_near, w1, w2
    """
    # Load the data; each row: [W, Q2, w1, w2]
    data = np.loadtxt(filename)
    
    # Get unique grid values (assuming data is on a regular grid)
    Ws = np.unique(data[:, 0])
    Q2s = np.unique(data[:, 1])
    
    # Find the nearest grid point in W and Q2 separately
    i_w = np.argmin(np.abs(Ws - W_input))
    i_q = np.argmin(np.abs(Q2s - Q2_input))
    W_near = Ws[i_w]
    Q2_near = Q2s[i_q]
    
    # Identify the row corresponding to the selected grid point.
    # A small tolerance is used to match the floating-point values.
    tol = 1e-6
    mask = (np.abs(data[:, 0] - W_near) < tol) & (np.abs(data[:, 1] - Q2_near) < tol)
    if not np.any(mask):
        raise ValueError("No matching grid point found in file.")
    row = data[mask][0]
    w1 = row[2]
    w2 = row[3]
    return W_near, Q2_near, w1, w2

def compute_cross_section(W, Q2, E, filename="wempx.dat"):
    """
    Computes the differential cross section dσ/dW/dQ^2 for an electromagnetic (EM)
    reaction (N(e,e')X) using structure functions from the file.

    Parameters:
        W      : Invariant mass of the final hadron system (GeV)
        Q2     : Photon virtuality (GeV^2)
        E      : Beam (lepton) energy in the lab (GeV)
        filename: File containing the table (assumed to be "wempx.dat")
    
    Returns:
        dcrs   : Differential cross section in units of 10^{-30} cm^2/GeV^3
    """
    # --- Define physical constants (in GeV units) ---
    fnuc = 0.9385         # nucleon mass, m_N
    fpio = 0.1385         # pion mass (not used here)
    pi = 3.1415926
    # For EM reaction, the lepton is massless:
    flepi = 0.0           # initial lepton mass
    flepf = 0.0           # final lepton mass

    # --- Get the structure functions from the table ---
    # The file is assumed to have four columns: W, Q2, W1, and W2.
    W_grid, Q2_grid, w1, w2 = get_structure_functions(W, Q2, filename)
    print(f"Using nearest grid point: W = {W_grid:.5f}, Q2 = {Q2_grid:.5f}")
    
    # --- Check that W and Q2 are within the allowed ranges ---
    if not (1.077 <= W_grid <= 2.0):
        raise ValueError("W out of range. Must be between 1.077 and 2 GeV.")
    if not (0 <= Q2_grid <= 3.0):
        raise ValueError("Q2 out of range. Must be between 0 and 3 GeV^2.")
    
    # --- Calculate total available energy in the lab (w_tot) ---
    # w_tot = sqrt(2*m_N*E + m_N^2 + m_lepi^2)
    # (For massless lepton, m_lepi = 0)
    wtot = math.sqrt(2 * fnuc * E + fnuc**2)
    if W_grid > wtot:
        raise ValueError("W is greater than the available energy (w_tot).")
    
    # --- Calculate center-of-mass (CM) momenta and energies ---
    # For massless particles: momentum = energy.
    # Initial lepton momentum (p_cmi) is given by:
    #   p_cmi = (w_tot^2 - m_N^2) / (2*w_tot)
    pcmi = (wtot**2 - fnuc**2) / (2 * wtot)
    # Final lepton momentum (p_cmf):
    #   p_cmf = (w_tot^2 - W^2) / (2*w_tot)
    pcmf = (wtot**2 - W_grid**2) / (2 * wtot)
    ecmi = pcmi
    ecmf = pcmf
    
    # --- Allowed Q^2 range ---
    # For massless leptons: Q2_min = 0, Q2_max = 4 * p_cmi * p_cmf
    Q2_max_allowed = 4 * pcmi * pcmf
    if Q2_grid < 0 or Q2_grid > Q2_max_allowed:
        raise ValueError(f"Q2 out of allowed range [0, {Q2_max_allowed:.5f}].")
    
    # --- Compute kinematics for the cross section calculation ---
    # Beam energy (initial lepton energy)
    elepi = E
    plepi = elepi  # p_lepi = E (massless)
    # Energy transfer:
    #   ω = (W^2 + Q^2 - m_N^2) / (2*m_N)
    omeg = (W_grid**2 + Q2_grid - fnuc**2) / (2 * fnuc)
    # Final lepton energy:
    elepf = elepi - omeg
    if elepf <= 0:
        raise ValueError("Final lepton energy is non-positive.")
    plepf = elepf  # massless
    
    # Cosine of the lepton scattering angle:
    #   clep = (-Q^2 + 2*elepi*elepf) / (2*plepi*plepf)
    clep = (-Q2_grid + 2 * elepi * elepf) / (2 * plepi * plepf)
    
    # Common kinematic factor:
    #   fac3 = π * W / (m_N * p_lepi * p_lepf)
    fac3 = pi * W_grid / (fnuc * plepi * plepf)
    
    # --- Reaction-dependent factor for electromagnetic (EM) cross section ---
    # For EM:
    #   fcrs3 = 4 * ((1/137.04)/Q^2)^2 * (0.197327^2) * 1e4 * (p_lepf/ p_lepi)*elepi*elepf
    # For massless leptons, (p_lepf/ p_lepi)*elepi*elepf simplifies to elepf^2.
    alpha = 1 / 137.04
    fcrs3 = 4 * (alpha / Q2_grid)**2 * (0.197327**2) * 1e4 * (elepf**2)
    
    # --- Angular factors ---
    # For EM:
    #   ss2 = (1 - clep) / 2
    #   cc2 = (1 + clep) / 2
    ss2 = (1 - clep) / 2
    cc2 = (1 + clep) / 2
    
    # --- Combine the structure functions ---
    # Here, only W1 and W2 are used:
    #   xxx = 2 * ss2 * W1 + cc2 * W2
    xxx = 2 * ss2 * w1 + cc2 * w2
    
    # --- Final differential cross section ---
    #   dσ/dW/dQ^2 = fcrs3 * fac3 * xxx
    dcrs = fcrs3 * fac3 * xxx
    return dcrs

def main():
    """
    Main routine:
      - Prompts the user for W, Q^2, and beam energy.
      - Computes the differential cross section.
      - Prints the result.
    """
    try:
        W_input = float(input("Enter W (GeV) [1.077 to 2]: "))
        Q2_input = float(input("Enter Q^2 (GeV^2) [0 to 3]: "))
        beam_energy = float(input("Enter beam energy (GeV): "))
    except ValueError:
        print("Invalid input. Please enter numerical values.")
        return
    
    try:
        # Compute the differential cross section
        xsec = compute_cross_section(W_input, Q2_input, beam_energy, filename="input_data/wempx.dat")
        print("\n=== Results ===")
        print(f"Beam energy: {beam_energy:.5f} GeV")
        print(f"W: {W_input:.5f} GeV, Q^2: {Q2_input:.5f} GeV^2")
        print(f"Differential cross section dσ/dW/dQ^2: {xsec:.5e} (10^(-30) cm^2/GeV^3)")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
