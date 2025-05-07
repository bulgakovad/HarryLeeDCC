# Fitting background:
# the procedure starts with fitting background:

import ROOT
from functions import getXSEC_fitting
import math

def mincalc(pp):
    chi2 = 0
    with open('exp_minus_pdf.txt') as ff:
        ff.readline()
        for line in ff:
            xx,yy, eps, yexp, dyexp_1, dyexp_2, dyexp_3 = [float(vv) for vv in line.strip().split(',')]
            dyexp = math.sqrt(dyexp_1**2 + dyexp_2**2 + dyexp_3**2)

            # PRC:
            # We fit onnly back so, resonanse params stay the same
            bg_params_exp_minus_pdf = [0.2367, -0.0375943454827298, 4.968682500440749, -4.2722561482930175, -12.8657473408563]
            ytheory = getXSEC_fitting(xx,yy, *bg_params_exp_minus_pdf, pp[0], pp[1], pp[2], pp[3], pp[4], pp[5], pp[6], pp[7], pp[8], pp[9], pp[10])
            

            chi2 += (yexp-ytheory)**2/dyexp**2
        return chi2

# start parameters
pp= [1.5,1.711,1.94343, 1.14391, 6.21974e-01,  5.14898e-01, 5.13290e-01 , 1.14735e-01, 1.22690e-01, 1.17700e-01, 2.02702e-01]
print(mincalc(pp))


minimum = ROOT.Math.Factory.CreateMinimizer("Minuit", "Migrad")

minimum.SetMaxFunctionCalls(10000)
minimum.SetMaxIterations(10000)
minimum.SetTolerance(0.0001)
minimum.SetPrintLevel(1)

fh = ROOT.Math.Functor(mincalc, 11)
minimum.SetFunction(fh)


p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11 = [1.5,1.711,1.94343, 1.14391, 6.21974e-01,  5.14898e-01,
                                                5.13290e-01 , 1.14735e-01, 1.22690e-01, 1.17700e-01, 2.02702e-01]

minimum.SetVariable(0, "p1", p1, 0.001)
minimum.SetVariable(1, "p2", p2, 0.001)
minimum.SetVariable(2, "p3", p3, 0.001)
minimum.SetVariable(3, "p4", p4, 0.0001)
minimum.SetVariable(4, "p5", p5, 0.0001)
minimum.SetVariable(5, "p6", p6, 0.0001)
minimum.SetVariable(6, "p7", p7, 0.0001)
minimum.SetVariable(7, "p8", p8, 0.0001)
minimum.SetVariable(8, "p9", p9, 0.0001)
minimum.SetVariable(9, "p10", p10, 0.0001)
minimum.SetVariable(10, "p11", p11, 0.0001)

minimum.Minimize()

for i in range(11):
 print(minimum.X()[i], end = ', ')