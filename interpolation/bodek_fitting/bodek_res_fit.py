import ROOT
import math
import os
import sys

sys.path.append(os.path.abspath("../"))  # adjust path as needed
from functions import getXSEC_fitting


# List of Q² bins to process
q2_bins = [2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699]

# Max W
W_max = 2.2

# Load background parameters from file
bg_dict = {}
with open("bg_res_params/bg_params.dat") as f:
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
if os.path.exists("bg_res_params/res_params.dat"):
    with open("bg_res_params/res_params.dat") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            tokens = [float(x) for x in line.strip().split(",")]
            q2 = tokens[0]
            params = tokens[1:]
            res_init[q2] = params

# Open output file
with open("bg_res_params/res_params.dat", "w") as fout:
    fout.write("# Q2, " + ", ".join([f"p{i+1}" for i in range(11)]) + "\n")

    for q2_target in q2_bins:
        if q2_target not in bg_dict:
            print(f"Skipping Q²={q2_target:.3f}: No background parameters found.")
            continue

        bg_params = bg_dict[q2_target]
        pp = res_init[q2_target]
        

        def mincalc(pp):
            chi2 = 0
            with open('exp_data_all.dat') as ff:
                ff.readline()
                for line in ff:
                    xx, yy, eps, yexp, dy1, dy2, dy3 = [float(vv) for vv in line.strip().split(',')]
                    if abs(xx - q2_target) > 1e-4 or yy > W_max:
                        continue
                    dyexp = math.sqrt(dy1**2 + dy2**2)
                    ytheory = getXSEC_fitting(0, xx, yy, *bg_params,  pp[0], pp[1], pp[2], pp[3], pp[4], pp[5], pp[6], pp[7], pp[8], pp[9], pp[10])
                    chi2 += (yexp - ytheory)**2/dyexp**2
            return chi2

                # start parameters
        

        minimum = ROOT.Math.Factory.CreateMinimizer("Minuit", "Migrad")

        minimum.SetMaxFunctionCalls(10000)
        minimum.SetMaxIterations(10000)
        minimum.SetTolerance(0.0001)
        minimum.SetPrintLevel(1)

        fh = ROOT.Math.Functor(mincalc, 11)
        minimum.SetFunction(fh)


        p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11 = pp[0], pp[1], pp[2], pp[3], pp[4], pp[5], pp[6], pp[7], pp[8], pp[9], pp[10]

        #minimum.SetVariable(0, "p1", p1, 0.0001)
        #minimum.SetVariable(1, "p2", p2, 0.0001)
        #minimum.SetVariable(2, "p3", p3, 0.0001)
        #minimum.SetVariable(3, "p4", p4, 0.0001)
        #minimum.SetVariable(4, "p5", p5, 0.0001)
        #minimum.SetVariable(5, "p6", p6, 0.0001)
        #minimum.SetVariable(6, "p7", p7, 0.0001)
        #minimum.SetVariable(7, "p8", p8, 0.0001)
        #minimum.SetVariable(8, "p9", p9, 0.0001)
        #minimum.SetVariable(9, "p10", p10, 0.0001)
        #minimum.SetVariable(10, "p11", p11, 0.0001)
        
        minimum.SetLimitedVariable(0, "p1", p1, 0.0001, 1, 2)
        minimum.SetLimitedVariable(1, "p2", p2, 0.0001, 1.65, 1.75)
        minimum.SetLimitedVariable(2, "p3", p3, 0.0001, 1.9, 2.2)
        minimum.SetLimitedVariable(3, "p4", p4, 0.0001, 1.1, 1.4)
        minimum.SetLimitedVariable(4, "p5", p5, 0.0001, 0.3, 0.8)
        minimum.SetLimitedVariable(5, "p6", p6, 0.0001, 0.1, 0.6)
        minimum.SetLimitedVariable(6, "p7", p7, 0.0001, 0, 3)
        minimum.SetLimitedVariable(7, "p8", p8, 0.0001, 0.1, 0.2)
        minimum.SetLimitedVariable(8, "p9", p9, 0.0001, 0,0.13 )
        minimum.SetLimitedVariable(9, "p10", p10, 0.0001, 0.04, 0.15)
        minimum.SetLimitedVariable(10, "p11", p11, 0.0001, 0, 0.2)
        
        

        minimum.Minimize()

        result = [minimum.X()[i] for i in range(11)]
        fout.write(f"{q2_target:.3f}, " + ", ".join(f"{v:.10f}" for v in result) + "\n")
        print(f"Done Q²={q2_target:.3f} => " + ", ".join(f"p{i+1}={v:.5f}" for i, v in enumerate(result)))
        
  
