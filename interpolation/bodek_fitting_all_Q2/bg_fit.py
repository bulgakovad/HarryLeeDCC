import ROOT
import sys
import math
import os

sys.path.append(os.path.abspath("../"))
from functions import getXSEC_fitting

W_max = 2.0

# Load fixed resonance parameters (initial/fixed for this fit)
with open("bg_res_params/res_params_global.dat") as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        bodekParams = [float(x) for x in line.strip().split(",")]

# Load initial background parameters from file
with open("bg_res_params/bg_params_global.dat") as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        pp = [float(x) for x in line.strip().split(",")]

def mincalc(pp):
    chi2 = 0
    with open('exp_data_all.dat') as ff:
        ff.readline()
        for line in ff:
            xx, yy, eps, yexp, dy1, dy2, _ = [float(vv) for vv in line.strip().split(',')]
            if yy > W_max:
                continue
            dyexp = math.sqrt(dy1**2 + dy2**2)
            ytheory = getXSEC_fitting(0, xx, yy, pp[0], pp[1], *bodekParams)
            chi2 += (yexp - ytheory) ** 2 / dyexp ** 2
    return chi2

minimum = ROOT.Math.Factory.CreateMinimizer("Minuit", "Migrad")
minimum.SetMaxFunctionCalls(10000)
minimum.SetMaxIterations(10000)
minimum.SetTolerance(0.0001)
minimum.SetPrintLevel(1)

fh = ROOT.Math.Functor(mincalc, 2)
minimum.SetFunction(fh)

minimum.SetVariable(0, "p1", pp[0], 0.00001)
minimum.SetVariable(1, "p2", pp[1], 0.00001)

minimum.Minimize()

p1_fit, p2_fit = minimum.X()[0], minimum.X()[1]

with open("bg_res_params/bg_params_global.dat", "w") as fout:
    fout.write("# p1, p2\n")
    fout.write(f"{p1_fit:.10f}, {p2_fit:.10f}\n")

print(f"Global BG Fit => p1={p1_fit:.5f}, p2={p2_fit:.5f}")
