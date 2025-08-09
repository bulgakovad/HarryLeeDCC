#!/usr/bin/env python
import sys,os
import numpy as np
from scipy.integrate import quad,fixed_quad,dblquad
from theory import IDIS

#thy=IDIS('JAM19PDF_proton_nlo')
thy=IDIS('CJ15nlo')
M=thy.M
mpi=thy.mpi
h0p = -3.2874
h1p = 1.9274
h2p = -2.0701

#	Writes F2 fixed Q2 files
def mainF2F1():
  f2 = open("Output/F2_fixQ2_cj15.txt","w")
  f1 = open("Output/F1_fixQ2_cj15.txt","w")
  fl = open("Output/FL_fixQ2_cj15.txt","w")
  for j in range(1,5):
    Q2 = j+0.025
    for i in range(0,74):
      W = 1.07+0.01*i#M+mpi+i*0.1
      nu = (W**2 - M**2 + Q2)/(2*M)
      x = Q2/(2.0*M*nu)
      rho = (1.0 + 4.0*x**2*M**2/Q2)**0.5
      xN = 2.0*x/(1.+rho)
      CHT = h0p*x**h1p*(1.+h2p*x)
      h2integrand=lambda u:thy.get_F2(u,Q2,'p')/u**2
      h2=thy.integrator(h2integrand,xN,1.0)
      g2integrand=lambda u:thy.get_F2(u,Q2,'p')/u**2*(u-xN)
      g2=thy.integrator(g2integrand,xN,1.0)
      F2naked=thy.get_F2(x,Q2,'p')
      F2moffat=(1.0+rho)/(2.0*rho**2)*thy.get_F2(xN,Q2,'p')
      F2brady0=(1.0+rho)/(2.0*rho)*F2moffat
      F2brady=F2brady0+3.0*x*(rho**2-1.0)/(2.0*rho**4)*(h2+(rho**2-1.0)/(2.0*x*rho)*g2)
      F2bradyht=F2brady*(1.+CHT/Q2)

      FLnaked=thy.get_FL(x,Q2,'p')
      FLmoffat=(1.0+rho)/2.0*thy.get_FL(xN,Q2,'p')
      FLbrady0=(1.0+rho)**2/(4.0*rho)*thy.get_FL(xN,Q2,'p')
      FLbrady=FLbrady0+x*(rho**2-1.0)/rho**2*(h2+(rho**2-1.0)/(2.0*x*rho)*g2)
      FLbradyht=FLbrady*(1.+CHT/Q2)

      F1brady0=(1.0+rho)/(2.0*rho)*thy.get_F1(xN,Q2,'p')
      F1brady=F1brady0+(rho**2-1.0)/(4.0*rho**2)*(h2+(rho**2-1.0)/(2.0*x*rho)*g2)
      F1brady0alt=((1.0+4.0*thy.M**2/Q2*x**2)*F2brady0-FLbrady0)/(2.0*x)
      F1bradyalt=((1.0+4.0*thy.M**2/Q2*x**2)*F2brady-FLbrady)/(2.0*x)
      f2.write(str(Q2)+"\t"+str(W)+"\t"+str(F2naked)+"\t"+str(F2moffat)+"\t"+str(F2brady0)+"\t"+str(F2brady)+"\t"+str(F2bradyht)+"\n")
      fl.write(str(Q2)+"\t"+str(W)+"\t"+str(FLnaked)+"\t"+str(FLmoffat)+"\t"+str(FLbrady0)+"\t"+str(FLbrady)+"\t"+str(FLbradyht)+"\n")
      f1.write(str(Q2)+"\t"+str(W)+"\t"+str(F1brady)+"\t"+str(F1bradyalt)+"\n")

#	Writes FL fixed Q2 files
def mainFLQ2():
  FLQ2list = [0.75,1.75,2.5,3.75]
  FLWlist = [1.23,1.42,1.52,1.71]
  flq = open("Output/FL_fixQ2_cj15.txt","w")
  for Q2 in FLQ2list:
    for i in range(0,74):
      W = 1.07+0.01*i
      nu = (W**2 - M**2 + Q2)/(2*M)
      x= Q2/(2.0*M*nu)
      rho = (1.0 + 4.0*x**2*M**2/Q2)**0.5
      xN = 2.0*x/(1.+rho)
      CHT = h0p*x**h1p*(1.+h2p*x)
      h2integrand=lambda u:thy.get_F2(u,Q2,'p')/u**2
      h2=thy.integrator(h2integrand,xN,1.0)
      g2integrand=lambda u:thy.get_F2(u,Q2,'p')/u**2*(u-xN)
      g2=thy.integrator(g2integrand,xN,1.0)
      FLnaked=thy.get_FL(x,Q2,'p')
      FLmoffat=(1.0+rho)/2.0*thy.get_FL(xN,Q2,'p')
      FLbrady0=(1.0+rho)**2/(4.0*rho)*thy.get_FL(xN,Q2,'p')
      FLbrady=FLbrady0+x*(rho**2-1.0)/rho**2*(h2+(rho**2-1.0)/(2.0*x*rho)*g2)
      FLbradyht=FLbrady*(1.+CHT/Q2)
      flq.write(str(Q2)+"\t"+str(W)+"\t"+str(FLnaked)+"\t"+str(FLmoffat)+"\t"+str(FLbrady0)+"\t"+str(FLbrady)+"\t"+str(FLbradyht)+"\n")

#	Writes FL fixed W files
def mainFLW():
  FLWlist = [1.23,1.42,1.52,1.71]
  flw = open("Output/FL_fixW_cj15.txt","w")
  for W in FLWlist:
    for i in range(0,51):
      Q2 = 0.5+0.1*i
      nu = (W**2 - M**2 + Q2)/(2*M)
      x= Q2/(2.0*M*nu)
      rho = (1.0 + 4.0*x**2*M**2/Q2)**0.5
      xN = 2.0*x/(1.+rho)
      h2integrand=lambda u:thy.get_F2(u,Q2,'p')/u**2
      h2=thy.integrator(h2integrand,xN,1.0)
      g2integrand=lambda u:thy.get_F2(u,Q2,'p')/u**2*(u-xN)
      g2=thy.integrator(g2integrand,xN,1.0)
      FLbrady0=(1.0+rho)**2/(4.0*rho)*thy.get_FL(xN,Q2,'p')
      FLbrady=FLbrady0+x*(rho**2-1.0)/rho**2*(h2+(rho**2-1.0)/(2.0*x*rho)*g2)
      flw.write(str(Q2)+"\t"+str(W)+"\t"+str(FLbrady)+"\n")

def mainF2trunc():
  f2 = open("Output/F2_trunc_cj15.txt","w")
  f1 = open("Output/F1_trunc_cj15.txt","w")
  fl = open("Output/FL_trunc_cj15.txt","w")
  for j in range(100,401):
    Q2 = j/100.
    W0 = np.sqrt(1.125)
    Wmax = np.sqrt(1.9)
    Wmax2 = np.sqrt(2.5)
    Wmax3 = np.sqrt(3.1)
    nu0 = (W0**2 - M**2 + Q2)/(2*M)
    numax = (Wmax**2 - M**2 + Q2)/(2*M)
    numax2 = (Wmax2**2 - M**2 + Q2)/(2*M)
    numax3 = (Wmax3**2 - M**2 + Q2)/(2*M)
    xmax = 1.
    x0 = Q2/(2.0*M*numax)
    x02 = Q2/(2.0*M*numax2)
    x03 = Q2/(2.0*M*numax3)
    rho = lambda x: (1.0 + 4.0*x**2*M**2/Q2)**0.5
    F2nakedint=lambda x:thy.get_F2(x,Q2,'p')
    xN = lambda x: 2.0*x/(1.+rho(x))
    h2integrand=lambda u:thy.get_F2(u,Q2,'p')/u**2
    h2int=lambda x: thy.integrator(h2integrand,xN(x),1.0,n=10)
    g2integrand=lambda x,u:thy.get_F2(u,Q2,'p')/u**2*(u-xN(x))
    CHT = lambda x: h0p*x**h1p*(1.+h2p*x)
    F2brady0htint=lambda x:(1.0+rho(x))**2/(4.0*rho(x)**3)*thy.get_F2(xN(x),Q2,'p')*(1.+CHT(x)/Q2)
    F2brady0int=lambda x:(1.0+rho(x))**2/(4.0*rho(x)**3)*thy.get_F2(xN(x),Q2,'p')
    F2xhtint=lambda x: 3.0*x*(rho(x)**2-1.0)/(2.0*rho(x)**4)*h2int(x)*(1.+CHT(x)/Q2)
    F2xint=lambda x: 3.0*x*(rho(x)**2-1.0)/(2.0*rho(x)**4)*h2int(x)
    F2uxint=lambda x,u:3.0*x*(rho(x)**2-1.0)/(2.0*rho(x)**4)*(rho(x)**2-1.0)/(2.0*x*rho(x))*g2integrand(x,u)
    F2bradyuxht=lambda x: fixed_quad(lambda u: np.vectorize(F2uxint)(x,u),xN(x),1.0,n=10)[0]*(1.+CHT(x)/Q2)
    F2bradyux=lambda x: fixed_quad(lambda u: np.vectorize(F2uxint)(x,u),xN(x),1.0,n=10)[0]
    F2naked=thy.integrator(F2nakedint,x0,xmax,n=10)
    F2naked2=thy.integrator(F2nakedint,x02,x0,n=10)
    F2naked3=thy.integrator(F2nakedint,x03,x02,n=10)
    F2nakedall=thy.integrator(F2nakedint,x03,xmax,n=10)
    F2bradyht=thy.integrator(F2brady0htint,x0,xmax,n=10)+thy.integrator(F2xhtint,x0,xmax,n=10)+fixed_quad(np.vectorize(F2bradyuxht),x0,xmax,n=10)[0]
    F2bradyht2=thy.integrator(F2brady0htint,x02,x0,n=10)+thy.integrator(F2xhtint,x02,x0,n=10)+fixed_quad(np.vectorize(F2bradyuxht),x02,x0,n=10)[0]
    F2bradyht3=thy.integrator(F2brady0htint,x03,x02,n=10)+thy.integrator(F2xhtint,x03,x02,n=10)+fixed_quad(np.vectorize(F2bradyuxht),x03,x02,n=10)[0]
    F2bradyhtall=thy.integrator(F2brady0htint,x03,xmax,n=10)+thy.integrator(F2xhtint,x03,xmax,n=10)+fixed_quad(np.vectorize(F2bradyuxht),x03,xmax,n=10)[0]
    F2brady=thy.integrator(F2brady0int,x0,xmax,n=10)+thy.integrator(F2xint,x0,xmax,n=10)+fixed_quad(np.vectorize(F2bradyux),x0,xmax,n=10)[0]
    F2brady2=thy.integrator(F2brady0int,x02,x0,n=10)+thy.integrator(F2xint,x02,x0,n=10)+fixed_quad(np.vectorize(F2bradyux),x02,x0,n=10)[0]
    F2brady3=thy.integrator(F2brady0int,x03,x02,n=10)+thy.integrator(F2xint,x03,x02,n=10)+fixed_quad(np.vectorize(F2bradyux),x03,x02,n=10)[0]
    F2bradyall=thy.integrator(F2brady0int,x03,xmax,n=10)+thy.integrator(F2xint,x03,xmax,n=10)+fixed_quad(np.vectorize(F2bradyux),x03,xmax,n=10)[0]
    F1brady0int=lambda x:(1.0+rho(x))/(2.0*rho(x))*thy.get_F1(xN(x),Q2,'p')
    F1xint=lambda x: (rho(x)**2-1.0)/(4.0*rho(x)**2)*h2int(x)
    F1uxint=lambda x,u:(rho(x)**2-1.0)/(4.0*rho(x)**2)*(rho(x)**2-1.0)/(2.0*x*rho(x))*g2integrand(x,u)
    F1bradyux=lambda x: fixed_quad(lambda u: np.vectorize(F1uxint)(x,u),xN(x),1.0,n=10)[0]
    F1brady=thy.integrator(F1brady0int,x0,xmax,n=10)+thy.integrator(F1xint,x0,xmax,n=10)+fixed_quad(np.vectorize(F1bradyux),x0,xmax,n=10)[0]
    F1brady2=thy.integrator(F1brady0int,x02,x0,n=10)+thy.integrator(F1xint,x02,x0,n=10)+fixed_quad(np.vectorize(F1bradyux),x02,x0,n=10)[0]
    F1brady3=thy.integrator(F1brady0int,x03,x02,n=10)+thy.integrator(F1xint,x03,x02,n=10)+fixed_quad(np.vectorize(F1bradyux),x03,x02,n=10)[0]
    F1bradyall=thy.integrator(F1brady0int,x03,xmax,n=10)+thy.integrator(F1xint,x03,xmax,n=10)+fixed_quad(np.vectorize(F1bradyux),x03,xmax,n=10)[0]
    FLbrady0htint=lambda x:(1.0+rho(x))**2/(4.0*rho(x))*thy.get_FL(xN(x),Q2,'p')*(1.+CHT(x)/Q2)
    FLbrady0int=lambda x:(1.0+rho(x))**2/(4.0*rho(x))*thy.get_FL(xN(x),Q2,'p')
    FLxhtint=lambda x: x*(rho(x)**2-1.0)/rho(x)**2*h2int(x)*(1.+CHT(x)/Q2)
    FLxint=lambda x: x*(rho(x)**2-1.0)/rho(x)**2*h2int(x)
    FLuxint=lambda x,u:x*(rho(x)**2-1.0)/rho(x)**2*(rho(x)**2-1.0)/(2.0*x*rho(x))*g2integrand(x,u)
    FLbradyuxht=lambda x: fixed_quad(lambda u: np.vectorize(FLuxint)(x,u),xN(x),1.0,n=10)[0]*(1.+CHT(x)/Q2)
    FLbradyux=lambda x: fixed_quad(lambda u: np.vectorize(FLuxint)(x,u),xN(x),1.0,n=10)[0]
    FLbradyht=thy.integrator(FLbrady0htint,x0,xmax,n=10)+thy.integrator(FLxhtint,x0,xmax,n=10)+fixed_quad(np.vectorize(FLbradyuxht),x0,xmax,n=10)[0]
    FLbradyht2=thy.integrator(FLbrady0htint,x02,x0,n=10)+thy.integrator(FLxhtint,x02,x0,n=10)+fixed_quad(np.vectorize(FLbradyuxht),x02,x0,n=10)[0]
    FLbradyht3=thy.integrator(FLbrady0htint,x03,x02,n=10)+thy.integrator(FLxhtint,x03,x02,n=10)+fixed_quad(np.vectorize(FLbradyuxht),x03,x02,n=10)[0]
    FLbradyhtall=thy.integrator(FLbrady0htint,x03,xmax,n=10)+thy.integrator(FLxhtint,x03,xmax,n=10)+fixed_quad(np.vectorize(FLbradyuxht),x03,xmax,n=10)[0]
    FLbrady=thy.integrator(FLbrady0int,x0,xmax,n=10)+thy.integrator(FLxint,x0,xmax,n=10)+fixed_quad(np.vectorize(FLbradyux),x0,xmax,n=10)[0]
    FLbrady2=thy.integrator(FLbrady0int,x02,x0,n=10)+thy.integrator(FLxint,x02,x0,n=10)+fixed_quad(np.vectorize(FLbradyux),x02,x0,n=10)[0]
    FLbrady3=thy.integrator(FLbrady0int,x03,x02,n=10)+thy.integrator(FLxint,x03,x02,n=10)+fixed_quad(np.vectorize(FLbradyux),x03,x02,n=10)[0]
    FLbradyall=thy.integrator(FLbrady0int,x03,xmax,n=10)+thy.integrator(FLxint,x03,xmax,n=10)+fixed_quad(np.vectorize(FLbradyux),x03,xmax,n=10)[0]
    f2.write(str(Q2)+"\t"+str(F2bradyht)+"\t"+str(F2bradyht2)+"\t"+str(F2bradyht3)+"\t"+str(F2bradyhtall)+"\t"+str(F2brady)+"\t"+str(F2brady2)+"\t"+str(F2brady3)+"\t"+str(F2bradyall)+"\t"+str(F2naked)+"\t"+str(F2naked2)+"\t"+str(F2naked3)+"\t"+str(F2nakedall)+"\n")
    f1.write(str(Q2)+"\t"+str(F1brady)+"\t"+str(F1brady2)+"\t"+str(F1brady3)+"\t"+str(F1bradyall)+"\n")
    fl.write(str(Q2)+"\t"+str(FLbradyht)+"\t"+str(FLbradyht2)+"\t"+str(FLbradyht3)+"\t"+str(FLbradyhtall)+"\t"+str(FLbrady)+"\t"+str(FLbrady2)+"\t"+str(FLbrady3)+"\t"+str(FLbradyall)+"\n")




def mainTMC():
  f1a = open("Output/F1TMC_abs_cj15.txt","w")
  f1r0 = open("Output/F1TMC_rel_uncorr_cj15.txt","w")
  f1rOPE = open("Output/F1TMC_rel_OPE_cj15.txt","w")
  f2a = open("Output/F2TMC_abs_cj15.txt","w")
  f2r0 = open("Output/F2TMC_rel_uncorr_cj15.txt","w")
  f2rOPE = open("Output/F2TMC_rel_OPE_cj15.txt","w")
  fla = open("Output/FLTMC_abs_cj15.txt","w")
  flr0 = open("Output/FLTMC_rel_uncorr_cj15.txt","w")
  flrOPE = open("Output/FLTMC_rel_OPE_cj15.txt","w")
  for j in range(1,90):
    x = j/100.
    Q2=2.
    rho = (1.0 + 4.0*x**2*M**2/Q2)**0.5
    xN = 2.0*x/(1.+rho)
    h2integrand=lambda u:thy.get_F2(u,Q2,'p')/u**2
    h2=thy.integrator(h2integrand,xN,1.0,n=10)
    g2integrand=lambda u:thy.get_F2(u,Q2,'p')/u**2*(u-xN)
    g2=thy.integrator(g2integrand,xN,1.0,n=10)
    CHT = h0p*x**h1p*(1.+h2p*x)

    F1naked=thy.get_F1(x,Q2,'p')
    F1moffat=thy.get_F1(xN,Q2,'p')
    F1brady0=(1.0+rho)/(2.0*rho)*F1moffat
    F1brady=F1brady0+(rho**2-1.0)/(4.0*rho**2)*(h2+(rho**2-1.0)/(2.0*x*rho)*g2)

    F2naked=thy.get_F2(x,Q2,'p')
    F2moffat=(1.0+rho)/(2.0*rho**2)*thy.get_F2(xN,Q2,'p')
    F2brady0=(1.0+rho)/(2.0*rho)*F2moffat
    F2brady=F2brady0+3.0*x*(rho**2-1.0)/(2.0*rho**4)*(h2+(rho**2-1.0)/(2.0*x*rho)*g2)
    F2brady=F2brady*(1.+CHT/Q2)

    FLnaked=thy.get_FL(x,Q2,'p')
    FLmoffat=(1.0+rho)/2.0*thy.get_FL(xN,Q2,'p')
    FLbrady0=(1.0+rho)/(2.0*rho)*FLmoffat
    FLbrady=FLbrady0+x*(rho**2-1.0)/rho**2*(h2+(rho**2-1.0)/(2.0*x*rho)*g2)

    F1nakedalt=((1.0+4.0*thy.M**2/Q2*x**2)*F2naked-FLnaked)/(2.0*x)
    F1moffatalt=((1.0+4.0*thy.M**2/Q2*x**2)*F2moffat-FLmoffat)/(2.0*x)
    F1brady0alt=((1.0+4.0*thy.M**2/Q2*x**2)*F2brady0-FLbrady0)/(2.0*x)
    F1bradyalt=((1.0+4.0*thy.M**2/Q2*x**2)*F2brady-FLbrady)/(2.0*x)

    f1a.write(str(x)+"\t"+str(F1naked)+"\t"+str(F1moffat)+"\t"+str(F1brady0)+"\t"+str(F1brady)+"\t"+str(F1nakedalt)+"\t"+str(F1moffatalt)+"\t"+str(F1brady0alt)+"\t"+str(F1bradyalt)+"\n")
    f1r0.write(str(x)+"\t"+str(F1moffat/F1naked)+"\t"+str(F1brady/F1naked)+"\t"+str(F1brady0/F1naked)+"\t"+str(F1moffatalt/F1nakedalt)+"\t"+str(F1bradyalt/F1nakedalt)+"\t"+str(F1brady0alt/F1nakedalt)+"\n")
    f1rOPE.write(str(x)+"\t"+str(F1naked/F1brady)+"\t"+str(F1moffat/F1brady)+"\t"+str(F1brady0/F1brady)+"\t"+str(F1nakedalt/F1bradyalt)+"\t"+str(F1moffatalt/F1bradyalt)+"\t"+str(F1brady0alt/F1bradyalt)+"\n")
    f2a.write(str(x)+"\t"+str(F2naked)+"\t"+str(F2moffat)+"\t"+str(F2brady0)+"\t"+str(F2brady)+"\n")
    f2r0.write(str(x)+"\t"+str(F2moffat/F2naked)+"\t"+str(F2brady/F2naked)+"\t"+str(F2brady0/F2naked)+"\n")
    f2rOPE.write(str(x)+"\t"+str(F2naked/F2brady)+"\t"+str(F2moffat/F2brady)+"\t"+str(F2brady0/F2brady)+"\n")
    fla.write(str(x)+"\t"+str(FLnaked)+"\t"+str(FLmoffat)+"\t"+str(FLbrady0)+"\t"+str(FLbrady)+"\n")
    flr0.write(str(x)+"\t"+str(FLmoffat/FLnaked)+"\t"+str(FLbrady/FLnaked)+"\t"+str(FLbrady0/FLnaked)+"\n")
    flrOPE.write(str(x)+"\t"+str(FLnaked/FLbrady)+"\t"+str(FLmoffat/FLbrady)+"\t"+str(FLbrady0/FLbrady)+"\n")


if __name__== "__main__":
     mainTMC()
#    mainF2trunc()
    #mainF2F1()
    #mainFLQ2()
#    mainFLW()






















