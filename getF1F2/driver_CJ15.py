#!/usr/bin/env python
import sys,os
import numpy as np
from scipy.integrate import quad,fixed_quad,dblquad
from theory import IDIS



h0p = -3.2874
h1p = 1.9274
h2p = -2.0701



#	Writes F2 fixed Q2 files
def mainF2F1(pdf_set):
  pdf_set=pdf_set
  thy=IDIS(pdf_set)
  M=thy.M
  mpi=thy.mpi
  def x_of_W(W,Q2): return Q2 / (W*W - M*M + Q2)
  
  
  out_dir = f"Output/Output_{pdf_set}"
  os.makedirs(out_dir, exist_ok=True)
  f2 = open(f"{out_dir}/F2_almost_all.txt","w")
  f1 = open(f"{out_dir}/F1_almost_all.txt","w")
  fl = open(f"{out_dir}/FL_almost_all.txt","w")
  for Q2 in [1.025, 2.025, 3.025, 4.025, 11.25, 12.0, 14.0, 16.0, 18.0, 20.0, 22.5, 25.0, 27.5, 30.0]:
    for i in range(0,1500):
      W = 1.07+0.02*i                         #M_prot+mpi+i*0.01
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
      
      #-----------------------------------------------------NOT PART OF THE ORIGINAL CODE. I added this-----------------------------------------------------------------------------------------
      F1bradyht=((1.0+4.0*thy.M**2/Q2*x**2)*F2bradyht-FLbradyht)/(2.0*x) # Can I do this? Analogous to F1bradyalt
      F1naked=((1.0+4.0*thy.M**2/Q2*x**2)*F2naked-FLnaked)/(2.0*x) # Can I do this? Analogous to F1bradyalt
      
      F2ht_only=F2naked*(1.+CHT/Q2) # Just to see the effect of CHT on F2naked. Dr. Joo asked me to do this
      F1ht_only=F1naked*(1.+CHT/Q2) # Just to see the effect of CHT on F1naked. Dr. Joo asked me to do this
      #----------------------------------------------------------------------------------------------------------------------------------------------
      
      #Write out -----------------------------------------------------------------------------------------------------------------------------------
      f2.write(str(Q2)+"\t"+str(W)+"\t"+str(F2naked)+"\t"+str(F2moffat)+"\t"+str(F2brady0)+"\t"+str(F2brady)+"\t"+str(F2bradyht)+"\n")
      fl.write(str(Q2)+"\t"+str(W)+"\t"+str(FLnaked)+"\t"+str(FLmoffat)+"\t"+str(FLbrady0)+"\t"+str(FLbrady)+"\t"+str(FLbradyht)+"\n")
      f1.write(str(Q2)+"\t"+str(W)+"\t"+str(F1naked)+"\t"+str(F1brady)+"\t"+str(F1bradyalt)+"\t"+str(F1bradyht)+"\n")

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

def mainF2trunc(pdf_set):
  print("Started mainF2trunc with pdf set:", pdf_set)
  my_pdf_set=pdf_set
  thy=IDIS(pdf_set)
  M=thy.M
  mpi=thy.mpi
  def x_of_W(W,Q2): return Q2 / (W*W - M*M + Q2)
  
  out_dir = "Output/truncated_moments"
  os.makedirs(out_dir, exist_ok=True)
  f2 = open(f"{out_dir}/M2_{my_pdf_set}.txt","w")
  
  for Q2 in [2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699]:
    print(f"Running for Q2 = {Q2}\n")
    W_min = 1.15 # now corresponds to data range
    Wmax1 = 1.35 # end of 1st resonance region
    Wmin2 = 1.45 # start of 2nd resonance region
    Wmax2 = 1.6 # end of 2nd resonance region
    Wmax3 = 1.85 # end of 3rd resonance region
    W_max = 2.5 
    if Q2 == 9.699:
      W_max = 2.25

    xmax = x_of_W(W_min, Q2)
 
    x1 = x_of_W(Wmax1, Q2) # W = 1.35 GeV
    xmin2 = x_of_W(Wmin2, Q2) # W = 1.45 GeV
    x2 = x_of_W(Wmax2, Q2) # W = 1.6 GeV
    x3 = x_of_W(Wmax3, Q2) # W = 1.85 GeV
    
    xmin = x_of_W(W_max, Q2) # W = 2.5 GeV (2.25 GeV at highest Q2)
  
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
    
    F2naked1=thy.integrator(F2nakedint,x1,xmax,n=10) # 1st resonance region W: 1.15 - 1.35
    F2naked2=thy.integrator(F2nakedint,x2,xmin2,n=10)   # 2nd resonance region W: 1.45 - 1.6
    F2naked3=thy.integrator(F2nakedint,x3,x2,n=10)   # 3rd resonance region W: 1.6 - 1.85
    F2naked_tail = thy.integrator(F2nakedint,xmin,x3,n=10)   # tail region W: 1.85 - 2.5 (2.25)
    F2nakedall=thy.integrator(F2nakedint,xmin,xmax,n=10)  # all range W: 1.15 - 2.5 (2.25)
    
    F2bradyht1=thy.integrator(F2brady0htint,x1,xmax,n=10)+thy.integrator(F2xhtint,x1,xmax,n=10)+fixed_quad(np.vectorize(F2bradyuxht),x1,xmax,n=10)[0]
    F2bradyht2=thy.integrator(F2brady0htint,x2,xmin2,n=10)+thy.integrator(F2xhtint,x2,xmin2,n=10)+fixed_quad(np.vectorize(F2bradyuxht),x2,xmin2,n=10)[0]
    F2bradyht3=thy.integrator(F2brady0htint,x3,x2,n=10)+thy.integrator(F2xhtint,x3,x2,n=10)+fixed_quad(np.vectorize(F2bradyuxht),x3,x2,n=10)[0]
    F2bradyht_tail=thy.integrator(F2brady0htint,xmin,x3,n=10)+thy.integrator(F2xhtint,xmin,x3,n=10)+fixed_quad(np.vectorize(F2bradyuxht),xmin,x3,n=10)[0]
    F2bradyhtall=thy.integrator(F2brady0htint,xmin,xmax,n=10)+thy.integrator(F2xhtint,xmin,xmax,n=10)+fixed_quad(np.vectorize(F2bradyuxht),xmin,xmax,n=10)[0]
    
    #F2brady1=thy.integrator(F2brady0int,x1,xmax,n=10)+thy.integrator(F2xint,x1,xmax,n=10)+fixed_quad(np.vectorize(F2bradyux),x1,xmax,n=10)[0]
    #F2brady2=thy.integrator(F2brady0int,x2,x1,n=10)+thy.integrator(F2xint,x2,x1,n=10)+fixed_quad(np.vectorize(F2bradyux),x2,x1,n=10)[0]
    #F2brady3=thy.integrator(F2brady0int,x3,x2,n=10)+thy.integrator(F2xint,x3,x2,n=10)+fixed_quad(np.vectorize(F2bradyux),x3,x2,n=10)[0]
    #F2bradyall=thy.integrator(F2brady0int,xmin,xmax,n=10)+thy.integrator(F2xint,xmin,xmax,n=10)+fixed_quad(np.vectorize(F2bradyux),xmin,xmax,n=10)[0]
    
    f2.write(str(Q2)+"\t"+str(F2bradyht1)+"\t"+str(F2bradyht2)+"\t"+str(F2bradyht3)+"\t"+str(F2bradyht_tail)+"\t"+str(F2bradyhtall)+"\t"+str(F2naked1)+"\t"+str(F2naked2)+"\t"+str(F2naked3)+"\t"+str(F2naked_tail)+"\t"+str(F2nakedall)+"\n")
    



def mainTMC():
  f1a = open("Output/F1TMC_abs_cj15.txt","w")
  #f1r0 = open("Output/F1TMC_rel_uncorr_cj15.txt","w")
  #f1rOPE = open("Output/F1TMC_rel_OPE_cj15.txt","w")
  f2a = open("Output/F2TMC_abs_cj15.txt","w")
  #f2r0 = open("Output/F2TMC_rel_uncorr_cj15.txt","w")
  #f2rOPE = open("Output/F2TMC_rel_OPE_cj15.txt","w")
  #fla = open("Output/FLTMC_abs_cj15.txt","w")
  #flr0 = open("Output/FLTMC_rel_uncorr_cj15.txt","w")
  #flrOPE = open("Output/FLTMC_rel_OPE_cj15.txt","w")
  for j in range(1,100):
    x = j/100.
    Q2=2.774
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
    #f1r0.write(str(x)+"\t"+str(F1moffat/F1naked)+"\t"+str(F1brady/F1naked)+"\t"+str(F1brady0/F1naked)+"\t"+str(F1moffatalt/F1nakedalt)+"\t"+str(F1bradyalt/F1nakedalt)+"\t"+str(F1brady0alt/F1nakedalt)+"\n")
    #f1rOPE.write(str(x)+"\t"+str(F1naked/F1brady)+"\t"+str(F1moffat/F1brady)+"\t"+str(F1brady0/F1brady)+"\t"+str(F1nakedalt/F1bradyalt)+"\t"+str(F1moffatalt/F1bradyalt)+"\t"+str(F1brady0alt/F1bradyalt)+"\n")
    f2a.write(str(x)+"\t"+str(F2naked)+"\t"+str(F2moffat)+"\t"+str(F2brady0)+"\t"+str(F2brady)+"\n")
    #f2r0.write(str(x)+"\t"+str(F2moffat/F2naked)+"\t"+str(F2brady/F2naked)+"\t"+str(F2brady0/F2naked)+"\n")
    #f2rOPE.write(str(x)+"\t"+str(F2naked/F2brady)+"\t"+str(F2moffat/F2brady)+"\t"+str(F2brady0/F2brady)+"\n")
    #fla.write(str(x)+"\t"+str(FLnaked)+"\t"+str(FLmoffat)+"\t"+str(FLbrady0)+"\t"+str(FLbrady)+"\n")
    #flr0.write(str(x)+"\t"+str(FLmoffat/FLnaked)+"\t"+str(FLbrady/FLnaked)+"\t"+str(FLbrady0/FLnaked)+"\n")
    #flrOPE.write(str(x)+"\t"+str(FLnaked/FLbrady)+"\t"+str(FLmoffat/FLbrady)+"\t"+str(FLbrady0/FLbrady)+"\n")


if __name__== "__main__":
     #mainTMC()
    mainF2trunc("CJ15nlo")
     #mainF2F1("CJ15nlo")
     #mainF2F1("CT18NLO")
    #mainFLQ2()
#    mainFLW()






















