import ROOT
import math
import os
import sys

sys.path.append(os.path.abspath("../"))
from functions import getXSEC_fitting

W_max = 2.0

# Load fixed background parameters
with open("bg_res_params/bg_params_global.dat") as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        bg_params = [float(x) for x in line.strip().split(",")]

# Load initial guess for resonance parameters
with open("bg_res_params/res_params_global.dat") as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        pp = [float(x) for x in line.strip().split(",")]

def mincalc(pp):
    chi2 = 0
    with open('exp_data_all.dat') as ff:
        ff.readline()
        for line in ff:
            xx, yy, eps, yexp, dy1, dy2, dy3 = [float(vv) for vv in line.strip().split(',')]
            if yy > W_max:
                continue
            dyexp = math.sqrt(dy1**2 + dy2**2)
            ytheory = getXSEC_fitting(0, xx, yy, *bg_params, pp[0], pp[1], pp[2], pp[3], pp[4], pp[5], pp[6], pp[7], pp[8], pp[9], pp[10])
            chi2 += (yexp - ytheory) ** 2 / dyexp ** 2
    return chi2

minimum = ROOT.Math.Factory.CreateMinimizer("Minuit", "Migrad")
minimum.SetMaxFunctionCalls(10000)
minimum.SetMaxIterations(10000)
minimum.SetTolerance(0.0001)
minimum.SetPrintLevel(1)
fh = ROOT.Math.Functor(mincalc, 11)
minimum.SetFunction(fh)

p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11 = pp[0], pp[1], pp[2], pp[3], pp[4], pp[5], pp[6], pp[7], pp[8], pp[9], pp[10]

minimum.SetVariable(0, "p1", p1, 0.0001)
minimum.SetVariable(1, "p2", p2, 0.0001)
minimum.SetVariable(2, "p3", p3, 0.0001)
minimum.SetVariable(3, "p4", p4, 0.0001)
minimum.SetVariable(4, "p5", p5, 0.0001)
minimum.SetVariable(5, "p6", p6, 0.0001)
minimum.SetVariable(6, "p7", p7, 0.0001)
minimum.SetVariable(7, "p8", p8, 0.0001)
minimum.SetVariable(8, "p9", p9, 0.0001)
minimum.SetVariable(9, "p10", p10, 0.0001)
minimum.SetVariable(10, "p11", p11, 0.0001)

#minimum.SetLimitedVariable(0, "p1", p1, 0.0001, 1, 2)
#minimum.SetLimitedVariable(1, "p2", p2, 0.0001, 1.65, 1.75)
#minimum.SetLimitedVariable(2, "p3", p3, 0.0001, 1.9, 2.2)
#minimum.SetLimitedVariable(3, "p4", p4, 0.0001, 1.1, 1.4)
#minimum.SetLimitedVariable(4, "p5", p5, 0.0001, 0.3, 0.8)
#minimum.SetLimitedVariable(5, "p6", p6, 0.0001, 0.1, 0.6)
#minimum.SetLimitedVariable(6, "p7", p7, 0.0001, 0, 3)
#minimum.SetLimitedVariable(7, "p8", p8, 0.0001, 0.1, 0.2)
#minimum.SetLimitedVariable(8, "p9", p9, 0.0001, 0,0.13 )
#minimum.SetLimitedVariable(9, "p10", p10, 0.0001, 0.04, 0.15)
#minimum.SetLimitedVariable(10, "p11", p11, 0.0001, 0, 0.2)

minimum.Minimize()

result = [minimum.X()[i] for i in range(11)]


with open("bg_res_params/res_params_global.dat", "w") as fout:
    fout.write("# " + ", ".join([f"p{i+1}" for i in range(11)]) + "\n")
    fout.write(", ".join(f"{v:.10f}" for v in result) + "\n")

print("Global resonance fit:")
for i, v in enumerate(result):
    print(f"p{i+1} = {v:.5f}")
