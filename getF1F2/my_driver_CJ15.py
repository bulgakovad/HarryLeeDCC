#!/usr/bin/env python
import sys,os
import numpy as np
from scipy.integrate import quad,fixed_quad,dblquad
from my_theory import IDIS

#thy=IDIS('JAM19PDF_proton_nlo')
thy=IDIS('CJ15nlo')
M=thy.M
mpi=thy.mpi
h0p = -3.2874
h1p = 1.9274
h2p = -2.0701
def x_of_W(W,Q2): return Q2 / (W*W - M*M + Q2)

#	Writes F2 fixed Q2 files
def mainF2F1():
  f2 = open("Output/F2_NLO_terms_large_Q2_test.txt","w")
  #for j in [30,100]:
  for Q2 in [15,20]:
    for i in range(0,1500):
      W = 1.07+0.02*i                         #M_prot+mpi+i*0.01
      nu = (W**2 - M**2 + Q2)/(2*M)
      x = Q2/(2.0*M*nu)
      rho = (1.0 + 4.0*x**2*M**2/Q2)**0.5
      xN = 2.0*x/(1.+rho)

      
      F2full=thy.get_F2_full(x,Q2,'p')
      F2_LO = thy.get_F2_LO(x,Q2,'p')
      F2_Q1 = thy.get_F2_Q1(x,Q2,'p')
      F2_G1 = thy.get_F2_G1(x,Q2,'p')
      
      
      #Write out -----------------------------------------------------------------------------------------------------------------------------------
      f2.write(str(Q2)+"\t"+str(W)+"\t"+str(F2full)+"\t"+str(F2_LO)+"\t"+str(F2_Q1)+"\t"+str(F2_G1)+"\n")
      



if __name__== "__main__":
     #mainTMC()
     #mainF2trunc("full")
     #mainF2trunc("part")
     mainF2F1()
    #mainFLQ2()
#    mainFLW()






















