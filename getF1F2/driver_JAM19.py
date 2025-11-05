#!/usr/bin/env python
import sys,os
import numpy as np
from scipy.integrate import quad,fixed_quad,dblquad
from theory import IDIS

thy=IDIS('JAM19PDF_proton_nlo')
M=thy.M
mpi=thy.mpi

#	Writes F2 fixed Q2 files
def mainF2F1():
  f2 = open("Output/F2_fixQ2.txt","w")
  f1 = open("Output/F1_fixQ2.txt","w")
  for j in range(1,5):
    Q2 = j+0.025
    for i in range(0,74):
      W = 1.07+0.01*i#M+mpi+i*0.1
      nu = (W**2 - M**2 + Q2)/(2*M)
      x = Q2/(2.0*M*nu)
      rho = (1.0 + 4.0*x**2*M**2/Q2)**0.5
      xN = 2.0*x/(1.+rho)
      h2integrand=lambda u:thy.get_F2(u,Q2,'p')/u**2
      h2=thy.integrator(h2integrand,xN,1.0)
      g2integrand=lambda u:thy.get_F2(u,Q2,'p')/u**2*(u-xN)
      g2=thy.integrator(g2integrand,xN,1.0)
      F2naked=thy.get_F2(x,Q2,'p')
      F2moffat=(1.0+rho)/(2.0*rho**2)*thy.get_F2(xN,Q2,'p')
      F2brady0=(1.0+rho)**2/(4.0*rho**3)*thy.get_F2(xN,Q2,'p')
      F2brady=F2brady0+3.0*x*(rho**2-1.0)/(2.0*rho**4)*(h2+(rho**2-1.0)/(2.0*x*rho)*g2)
      F1naked=thy.get_F1(x,Q2,'p')
      F1moffat=thy.get_F1(xN,Q2,'p')
      F1brady0=(1.0+rho)/(2.0*rho)*thy.get_F1(xN,Q2,'p')
      F1brady=F1brady0+(rho**2-1.0)/(4.0*rho**2)*(h2+(rho**2-1.0)/(2.0*x*rho)*g2)
      FLnaked=thy.get_FL(x,Q2,'p')
      FLmoffat=(1.0+rho)/2.0*thy.get_FL(xN,Q2,'p')
      FLbrady0=(1.0+rho)**2/(4.0*rho)*thy.get_FL(xN,Q2,'p')
      FLbrady=FLbrady0+x*(rho**2-1.0)/rho**2*(h2+(rho**2-1.0)/(2.0*x*rho)*g2)
      F1nakedalt=((1.0+4.0*thy.M**2/Q2*x**2)*F2naked-FLnaked)/(2.0*x)
      F1moffatalt=((1.0+4.0*thy.M**2/Q2*x**2)*F2moffat-FLmoffat)/(2.0*x)
      F1brady0alt=((1.0+4.0*thy.M**2/Q2*x**2)*F2brady0-FLbrady0)/(2.0*x)
      F1bradyalt=((1.0+4.0*thy.M**2/Q2*x**2)*F2brady-FLbrady)/(2.0*x)
      f2.write(str(Q2)+"\t"+str(W)+"\t"+str(F2naked)+"\t"+str(F2moffat)+"\t"+str(F2brady0)+"\t"+str(F2brady)+"\n")
      f1.write(str(Q2)+"\t"+str(W)+"\t"+str(F1naked)+"\t"+str(F1moffat)+"\t"+str(F1brady0)+"\t"+str(F1brady)+"\t"+str(F1nakedalt)+"\t"+str(F1moffatalt)+"\t"+str(F1brady0alt)+"\t"+str(F1bradyalt)+"\n")

#	Writes FL fixed Q2 files
def mainFLQ2():
  FLQ2list = [0.75,1.75,2.5,3.75]
  FLWlist = [1.23,1.42,1.52,1.71]
  flq = open("Output/FL_fixQ2.txt","w")
  for Q2 in FLQ2list:
    for i in range(0,74):
      W = 1.07+0.01*i
      nu = (W**2 - M**2 + Q2)/(2*M)
      x= Q2/(2.0*M*nu)
      rho = (1.0 + 4.0*x**2*M**2/Q2)**0.5
      xN = 2.0*x/(1.+rho)
      FLnaked=thy.get_FL(x,Q2,'p')
      FLmoffat=(1.0+rho)/2.0*thy.get_FL(xN,Q2,'p')
      FLbrady0=(1.0+rho)**2/(4.0*rho)*thy.get_FL(xN,Q2,'p')
      h2integrand=lambda u:thy.get_F2(u,Q2,'p')/u**2
      h2=thy.integrator(h2integrand,xN,1.0)
      g2integrand=lambda u:thy.get_F2(u,Q2,'p')/u**2*(u-xN)
      g2=thy.integrator(g2integrand,xN,1.0)
      FLbrady=FLbrady0+x*(rho**2-1.0)/rho**2*(h2+(rho**2-1.0)/(2.0*x*rho)*g2)
      flq.write(str(Q2)+"\t"+str(W)+"\t"+str(FLnaked)+"\t"+str(FLmoffat)+"\t"+str(FLbrady0)+"\t"+str(FLbrady)+"\n")

#	Writes FL fixed W files
def mainFLW():
  FLWlist = [1.23,1.42,1.52,1.71]
  flw = open("Output/FL_fixW.txt","w")
  for W in FLWlist:
    for i in range(0,51):
      Q2 = 0.5+0.1*i
      nu = (W**2 - M**2 + Q2)/(2*M)
      x= Q2/(2.0*M*nu)
      rho = (1.0 + 4.0*x**2*M**2/Q2)**0.5
      xN = 2.0*x/(1.+rho)
      FLnaked=thy.get_FL(x,Q2,'p')
      FLmoffat=(1.0+rho)/2.0*thy.get_FL(xN,Q2,'p')
      FLbrady0=(1.0+rho)**2/(4.0*rho)*thy.get_FL(xN,Q2,'p')
      h2integrand=lambda u:thy.get_F2(u,Q2,'p')/u**2
      h2=thy.integrator(h2integrand,xN,1.0)
      g2integrand=lambda u:thy.get_F2(u,Q2,'p')/u**2*(u-xN)
      g2=thy.integrator(g2integrand,xN,1.0)
      FLbrady=FLbrady0+x*(rho**2-1.0)/rho**2*(h2+(rho**2-1.0)/(2.0*x*rho)*g2)
      flw.write(str(Q2)+"\t"+str(W)+"\t"+str(FLnaked)+"\t"+str(FLmoffat)+"\t"+str(FLbrady0)+"\t"+str(FLbrady)+"\n")


def mainF2Simple():
  f = open("Output/F2_simple.txt","w")
  Q2 = 2.0
  for i in range(1,100):
    x = 0.01*i
    rho = (1.0 + 4.0*x**2*M**2/Q2)**0.5
    xN = 2.0*x/(1.+rho)
    F2naked = (1-x)**3
    F2moffat=(1.0+rho)/(2.0*rho**2)*(1-xN)**3
    F2brady0=(1.0+rho)**2/(4.0*rho**3)*(1-xN)**3
    h2integrand=lambda u:(1-u)**3/u**2
    h2=thy.integrator(h2integrand,xN,1.0)
    g2integrand=lambda u:(1-u**3)/u**2*(u-xN)
    g2=thy.integrator(g2integrand,xN,1.0)
    F2brady=F2brady0+3.0*x*(rho**2-1.0)/(2.0*rho**4)*(h2+(rho**2-1.0)/(2.0*x*rho)*g2)
    f.write(str(x)+"\t"+str(F2naked)+"\t"+str(F2moffat)+"\t"+str(F2brady0)+"\t"+str(F2brady)+"\n")

def mainF2trunc():
  f2 = open("Output/F2_trunc.txt","w")
  f1 = open("Output/F1_trunc.txt","w")
  fl = open("Output/FL_trunc.txt","w")
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
    xN = lambda x: 2.0*x/(1.+rho(x))
    h2integrand=lambda u:thy.get_F2(u,Q2,'p')/u**2
    h2int=lambda x: thy.integrator(h2integrand,xN(x),1.0,n=10)
    g2integrand=lambda x,u:thy.get_F2(u,Q2,'p')/u**2*(u-xN(x))
# Note that this is the 1st moment of F2, while in the following it's the 2nd moment of F1 and FL. This is just a choice, might change later.
    F2nakedint=lambda x:thy.get_F2(x,Q2,'p')
    F2naked=thy.integrator(F2nakedint,x0,xmax,n=10)
    F2naked2=thy.integrator(F2nakedint,x02,x0,n=10)
    F2naked3=thy.integrator(F2nakedint,x03,x02,n=10)
    F2nakedall=thy.integrator(F2nakedint,x03,xmax)
    F2moffatint=lambda x:(1.0+rho(x))/(2.0*rho(x)**2)*thy.get_F2(xN(x),Q2,'p')
    F2moffat=thy.integrator(F2moffatint,x0,xmax,n=10)
    F2moffat2=thy.integrator(F2moffatint,x02,x0,n=10)
    F2moffat3=thy.integrator(F2moffatint,x03,x02,n=10)
    F2moffatall=thy.integrator(F2moffatint,x03,xmax,n=10)
    F2brady0int=lambda x:(1.0+rho(x))**2/(4.0*rho(x)**3)*thy.get_F2(xN(x),Q2,'p')
    F2brady0=thy.integrator(F2brady0int,x0,xmax,n=10)
    F2brady02=thy.integrator(F2brady0int,x02,x0,n=10)
    F2brady03=thy.integrator(F2brady0int,x03,x02,n=10)
    F2brady0all=thy.integrator(F2brady0int,x03,xmax,n=10)
    F2xint=lambda x: 3.0*x*(rho(x)**2-1.0)/(2.0*rho(x)**4)*h2int(x)
    F2uxint=lambda x,u:3.0*x*(rho(x)**2-1.0)/(2.0*rho(x)**4)*(rho(x)**2-1.0)/(2.0*x*rho(x))*g2integrand(x,u)
    F2bradyux=lambda x: fixed_quad(lambda u: np.vectorize(F2uxint)(x,u),xN(x),1.0,n=10)[0]
    F2brady=F2brady0+thy.integrator(F2xint,x0,xmax,n=10)+fixed_quad(np.vectorize(F2bradyux),x0,xmax,n=10)[0]
    F2brady2=F2brady02+thy.integrator(F2xint,x02,x0,n=10)+fixed_quad(np.vectorize(F2bradyux),x02,x0,n=10)[0]
    F2brady3=F2brady03+thy.integrator(F2xint,x03,x02,n=10)+fixed_quad(np.vectorize(F2bradyux),x03,x02,n=10)[0]
    F2bradyall=F2brady0all+thy.integrator(F2xint,x03,xmax,n=10)+fixed_quad(np.vectorize(F2bradyux),x03,xmax,n=10)[0]
    F1nakedint=lambda x:thy.get_F1(x,Q2,'p')
    F1naked=thy.integrator(F1nakedint,x0,xmax,n=10)
    F1naked2=thy.integrator(F1nakedint,x02,x0,n=10)
    F1naked3=thy.integrator(F1nakedint,x03,x02,n=10)
    F1nakedall=thy.integrator(F1nakedint,x03,xmax)
    F1moffatint=lambda x:thy.get_F1(xN(x),Q2,'p')
    F1moffat=thy.integrator(F1moffatint,x0,xmax,n=10)
    F1moffat2=thy.integrator(F1moffatint,x02,x0,n=10)
    F1moffat3=thy.integrator(F1moffatint,x03,x02,n=10)
    F1moffatall=thy.integrator(F1moffatint,x03,xmax,n=10)
    F1brady0int=lambda x:(1.0+rho(x))/(2.0*rho(x))*thy.get_F1(xN(x),Q2,'p')
    F1brady0=thy.integrator(F1brady0int,x0,xmax,n=10)
    F1brady02=thy.integrator(F1brady0int,x02,x0,n=10)
    F1brady03=thy.integrator(F1brady0int,x03,x02,n=10)
    F1brady0all=thy.integrator(F1brady0int,x03,xmax,n=10)
    F1xint=lambda x: (rho(x)**2-1.0)/(4.0*rho(x)**2)*h2int(x)
    F1uxint=lambda x,u:(rho(x)**2-1.0)/(4.0*rho(x)**2)*(rho(x)**2-1.0)/(2.0*x*rho(x))*g2integrand(x,u)
    F1bradyux=lambda x: fixed_quad(lambda u: np.vectorize(F1uxint)(x,u),xN(x),1.0,n=10)[0]
    F1brady=F1brady0+thy.integrator(F1xint,x0,xmax,n=10)+fixed_quad(np.vectorize(F1bradyux),x0,xmax,n=10)[0]
    F1brady2=F1brady02+thy.integrator(F1xint,x02,x0,n=10)+fixed_quad(np.vectorize(F1bradyux),x02,x0,n=10)[0]
    F1brady3=F1brady03+thy.integrator(F1xint,x03,x02,n=10)+fixed_quad(np.vectorize(F1bradyux),x03,x02,n=10)[0]
    F1bradyall=F1brady0all+thy.integrator(F1xint,x03,xmax,n=10)+fixed_quad(np.vectorize(F1bradyux),x03,xmax,n=10)[0]
    FLnakedint=lambda x:thy.get_FL(x,Q2,'p')
    FLnaked=thy.integrator(FLnakedint,x0,xmax,n=10)
    FLnaked2=thy.integrator(FLnakedint,x02,x0,n=10)
    FLnaked3=thy.integrator(FLnakedint,x03,x02,n=10)
    FLnakedall=thy.integrator(FLnakedint,x03,xmax)
    FLmoffatint=lambda x:(1.0+rho(x))/2.0*thy.get_FL(xN(x),Q2,'p')
    FLmoffat=thy.integrator(FLmoffatint,x0,xmax,n=10)
    FLmoffat2=thy.integrator(FLmoffatint,x02,x0,n=10)
    FLmoffat3=thy.integrator(FLmoffatint,x03,x02,n=10)
    FLmoffatall=thy.integrator(FLmoffatint,x03,xmax,n=10)
    FLbrady0int=lambda x:(1.0+rho(x))**2/(4.0*rho(x))*thy.get_FL(xN(x),Q2,'p')
    FLbrady0=thy.integrator(FLbrady0int,x0,xmax,n=10)
    FLbrady02=thy.integrator(FLbrady0int,x02,x0,n=10)
    FLbrady03=thy.integrator(FLbrady0int,x03,x02,n=10)
    FLbrady0all=thy.integrator(FLbrady0int,x03,xmax,n=10)
    FLxint=lambda x: x*(rho(x)**2-1.0)/rho(x)**2*h2int(x)
    FLuxint=lambda x,u:x*(rho(x)**2-1.0)/rho(x)**2*(rho(x)**2-1.0)/(2.0*x*rho(x))*g2integrand(x,u)
    FLbradyux=lambda x: fixed_quad(lambda u: np.vectorize(FLuxint)(x,u),xN(x),1.0,n=10)[0]
    FLbrady=FLbrady0+thy.integrator(FLxint,x0,xmax,n=10)+fixed_quad(np.vectorize(FLbradyux),x0,xmax,n=10)[0]
    FLbrady2=FLbrady02+thy.integrator(FLxint,x02,x0,n=10)+fixed_quad(np.vectorize(FLbradyux),x02,x0,n=10)[0]
    FLbrady3=FLbrady03+thy.integrator(FLxint,x03,x02,n=10)+fixed_quad(np.vectorize(FLbradyux),x03,x02,n=10)[0]
    FLbradyall=FLbrady0all+thy.integrator(FLxint,x03,xmax,n=10)+fixed_quad(np.vectorize(FLbradyux),x03,xmax,n=10)[0]
    f2.write(str(Q2)+"\t"+str(F2naked)+"\t"+str(F2moffat)+"\t"+str(F2brady0)+"\t"+str(F2brady)+"\t"+str(F2naked2)+"\t"+str(F2moffat2)+"\t"+str(F2brady02)+"\t"+str(F2brady2)+"\t"+str(F2naked3)+"\t"+str(F2moffat3)+"\t"+str(F2brady03)+"\t"+str(F2brady3)+"\t"+str(F2nakedall)+"\t"+str(F2moffatall)+"\t"+str(F2brady0all)+"\t"+str(F2bradyall)+"\n")
    f1.write(str(Q2)+"\t"+str(F1naked)+"\t"+str(F1moffat)+"\t"+str(F1brady0)+"\t"+str(F1brady)+"\t"+str(F1naked2)+"\t"+str(F1moffat2)+"\t"+str(F1brady02)+"\t"+str(F1brady2)+"\t"+str(F1naked3)+"\t"+str(F1moffat3)+"\t"+str(F1brady03)+"\t"+str(F1brady3)+"\t"+str(F1nakedall)+"\t"+str(F1moffatall)+"\t"+str(F1brady0all)+"\t"+str(F1bradyall)+"\n")
    fl.write(str(Q2)+"\t"+str(FLnaked)+"\t"+str(FLmoffat)+"\t"+str(FLbrady0)+"\t"+str(FLbrady)+"\t"+str(FLnaked2)+"\t"+str(FLmoffat2)+"\t"+str(FLbrady02)+"\t"+str(FLbrady2)+"\t"+str(FLnaked3)+"\t"+str(FLmoffat3)+"\t"+str(FLbrady03)+"\t"+str(FLbrady3)+"\t"+str(FLnakedall)+"\t"+str(FLmoffatall)+"\t"+str(FLbrady0all)+"\t"+str(FLbradyall)+"\n")




def mainTMC():
  f1a = open("Output/F1TMC_abs.txt","w")
  f1r0 = open("Output/F1TMC_rel_uncorr.txt","w")
  f1rOPE = open("Output/F1TMC_rel_OPE.txt","w")
  f2a = open("Output/F2TMC_abs.txt","w")
  f2r0 = open("Output/F2TMC_rel_uncorr.txt","w")
  f2rOPE = open("Output/F2TMC_rel_OPE.txt","w")
  fla = open("Output/FLTMC_abs.txt","w")
  flr0 = open("Output/FLTMC_rel_uncorr.txt","w")
  flrOPE = open("Output/FLTMC_rel_OPE.txt","w")
  for j in range(1,90):
    x = j/100.
    Q2=2.
    rho = (1.0 + 4.0*x**2*M**2/Q2)**0.5
    xN = 2.0*x/(1.+rho)
    h2integrand=lambda u:thy.get_F2(u,Q2,'p')/u**2
    h2=thy.integrator(h2integrand,xN,1.0,n=10)
    g2integrand=lambda u:thy.get_F2(u,Q2,'p')/u**2*(u-xN)
    g2=thy.integrator(g2integrand,xN,1.0,n=10)

    F1naked=thy.get_F1(x,Q2,'p')
    F1moffat=thy.get_F1(xN,Q2,'p')
    F1brady0=(1.0+rho)/(2.0*rho)*F1moffat
    F1brady=F1brady0+(rho**2-1.0)/(4.0*rho**2)*(h2+(rho**2-1.0)/(2.0*x*rho)*g2)

    F2naked=thy.get_F2(x,Q2,'p')
    F2moffat=(1.0+rho)/(2.0*rho**2)*thy.get_F2(xN,Q2,'p')
    F2brady0=(1.0+rho)/(2.0*rho)*F2moffat
    F2brady=F2brady0+3.0*x*(rho**2-1.0)/(2.0*rho**4)*(h2+(rho**2-1.0)/(2.0*x*rho)*g2)

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
#    mainTMC()
    mainF2trunc()
#    mainF2F1()
#    mainF2Simple()
#    mainFLQ2()
#    mainFLW()






















