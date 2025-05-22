import ROOT
from functions import getXSEC_fitting
import math
import os

# List of Q² bins
q2_bins = [2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699]

# Max W
W_max = 2.2

# Load background parameters from file
bg_dict = {}
with open("bg_params.dat") as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        tokens = line.strip().split(",")
        q2_val = float(tokens[0])
        p1 = float(tokens[1])
        p2 = float(tokens[2])
        bg_dict[q2_val] = [p1, p2]
        
# Load previous resonance parameters as initial guess
res_init = {}
if os.path.exists("res_params.dat"):
    with open("res_params.dat") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            tokens = [float(x) for x in line.strip().split(",")]
            q2 = tokens[0]
            params = tokens[1:]
            res_init[q2] = params





# Output file to store fitted parameters
with open("bg_params.dat", "w") as fout:
    fout.write("# Q2, p1, p2\n")

    for q2_target in q2_bins:
        bodekParams = res_init[q2_target] 
        # Redefine chi² function for current Q²
        def mincalc(pp):
            chi2 = 0
            with open('exp_data_all.dat') as ff:
                ff.readline()
                for line in ff:
                    xx, yy, eps, yexp, dy1, dy2, _ = [float(vv) for vv in line.strip().split(',')]
                    if abs(xx - q2_target) > 1e-3 or yy > W_max:
                        continue
                    dyexp = math.sqrt(dy1**2 + dy2**2)
                    ytheory = getXSEC_fitting(0, xx, yy, pp[0], pp[1], *bodekParams)
                    chi2 += (yexp - ytheory) ** 2 / dyexp ** 2
            return chi2

        
        pp = bg_dict[q2_target]

        # Set up minimizer
        minimum = ROOT.Math.Factory.CreateMinimizer("Minuit", "Migrad")
        minimum.SetMaxFunctionCalls(10000)
        minimum.SetMaxIterations(10000)
        minimum.SetTolerance(0.0001)
        minimum.SetPrintLevel(1)

        fh = ROOT.Math.Functor(mincalc, 2)
        minimum.SetFunction(fh)

        #minimum.SetLimitedVariable(0, "p1", pp[0], 0.0001, 0, 3)
        #minimum.SetLimitedVariable(1, "p2", pp[1], 0.0001, -2, -0.3)
        minimum.SetVariable(0, "p1", pp[0], 0.00001)
        minimum.SetVariable(1, "p2", pp[1], 0.00001)

        minimum.Minimize()

        p1_fit, p2_fit = minimum.X()[0], minimum.X()[1]
        fout.write(f"{q2_target:.3f}, {p1_fit:.10f}, {p2_fit:.10f}\n")
        print(f"Done Q2={q2_target:.3f} => p1={p1_fit:.5f}, p2={p2_fit:.5f}")
