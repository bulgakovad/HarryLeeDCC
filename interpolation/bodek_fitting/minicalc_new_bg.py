# Fitting background:
# the procedure starts with fitting background:

import ROOT
from functions import getXSEC_fitting
import math

def mincalc(pp):
    chi2 = 0
    with open('exp_data_all.dat') as ff:
        ff.readline()
        for line in ff:
            xx,yy, eps, yexp, dyexp_1, dyexp_2, dyexp_3 = [float(vv) for vv in line.strip().split(',')]
            dyexp = math.sqrt(dyexp_1**2 + dyexp_2**2)

            # PRC:
            # We fit onnly back so, resonanse params stay the same
            bodekParams = [1.5, 1.711, 1.8480749999978037, 1.1439502360264717, 19.360431074996903, 0.5149462924759121, 0.5252228835731393, 0.11472502620982951, 0.9251355800507702, 0.11767418259186742, 0.2026970464482057] 
        
            ytheory = getXSEC_fitting(0, xx,yy, pp[0], pp[1], *bodekParams)
            chi2 += (yexp-ytheory)**2/dyexp**2
        return chi2

# start parameters
pp=[1.0000009536745438, -0.9952316284179688]
print(mincalc(pp))


minimum = ROOT.Math.Factory.CreateMinimizer("Minuit", "Migrad")

minimum.SetMaxFunctionCalls(10000)
minimum.SetMaxIterations(10000)
minimum.SetTolerance(0.0001)
minimum.SetPrintLevel(1)

fh = ROOT.Math.Functor(mincalc, 2)
minimum.SetFunction(fh)


p1, p2 = pp[0], pp[1]

minimum.SetVariable(0, "p1", p1, 0.00001)
minimum.SetVariable(1, "p2", p2, 0.00001)


minimum.Minimize()

for i in range(2):
 print(minimum.X()[i], end = ', ')



