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
            bodekParams = [1.5,1.711,1.94343, 1.14391, 6.21974e-01,  5.14898e-01,
                           5.13290e-01 , 1.14735e-01, 1.22690e-01, 1.17700e-01, 2.02702e-01]
        
            ytheory = getXSEC_fitting(0, xx,yy, pp[0], pp[1], pp[2], pp[3], pp[4], *bodekParams)
            chi2 += (yexp-ytheory)**2/dyexp**2
        return chi2

# start parameters
pp=[0.2367, 2.178, 0.898, -6.726, 3.718]
print(mincalc(pp))


minimum = ROOT.Math.Factory.CreateMinimizer("Minuit", "Migrad")

minimum.SetMaxFunctionCalls(10000)
minimum.SetMaxIterations(10000)
minimum.SetTolerance(0.0001)
minimum.SetPrintLevel(1)

fh = ROOT.Math.Functor(mincalc, 5)
minimum.SetFunction(fh)


p1, p2, p3, p4, p5 = [0.2367, 2.178, 0.898, -6.726, 3.718]

minimum.SetVariable(0, "p1", p1, 0.00001)
minimum.SetVariable(1, "p2", p2, 0.00001)
minimum.SetVariable(2, "p3", p3, 0.00001)
minimum.SetVariable(3, "p4", p4, 0.00001)
minimum.SetVariable(4, "p5", p5, 0.00001)

minimum.Minimize()

for i in range(5):
 print(minimum.X()[i], end = ', ')
