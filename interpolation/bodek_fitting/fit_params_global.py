import ROOT
import math
import numpy as np
import os
import sys
sys.path.append(os.path.abspath("../"))  # adjust path as needed
from functions import getXSEC_fitting


# Load full dataset
data = np.loadtxt("exp_data_all.dat", delimiter=",", skiprows=1)
Q2_vals, W_vals, eps_vals, yexp_vals, dy1_vals, dy2_vals, _ = data.T
dy_vals = np.sqrt(dy1_vals**2 + dy2_vals**2)

# Global χ² function with 13 parameters
def global_chi2(params):
    p_bkg1, p_bkg2 = params[0], params[1]
    bodek = params[2:]

    chi2 = 0.0
    for Q2, W, yexp, dy in zip(Q2_vals, W_vals, yexp_vals, dy_vals):
        yth = getXSEC_fitting(0, Q2, W, p_bkg1, p_bkg2, *bodek)
        chi2 += (yexp - yth)**2 / dy**2

    return chi2

# ROOT minimizer setup
minimizer = ROOT.Math.Factory.CreateMinimizer("Minuit", "Migrad")
minimizer.SetMaxFunctionCalls(100000)
minimizer.SetMaxIterations(10000)
minimizer.SetTolerance(1e-4)
minimizer.SetPrintLevel(1)

# Wrap the χ² in a ROOT Functor
fcn = ROOT.Math.Functor(global_chi2, 13)
minimizer.SetFunction(fcn)

# Initial guesses (can be from previous runs)
initial_params = [
    1.5, -0.9,  # background: p1, p2
    1.5, 1.711, 1.94343, 1.14391, 0.621974, 0.514898, 0.513290,
    0.114735, 0.122690, 0.117700, 0.202702  # resonance
]

# Define variables
for i, val in enumerate(initial_params):
    minimizer.SetVariable(i, f"p{i+1}", val, 0.0001)

# Perform minimization
minimizer.Minimize()

# Output results
final_params = [minimizer.X()[i] for i in range(13)]
for i, val in enumerate(final_params):
    print(f"p{i+1} = {val:.8f}")
