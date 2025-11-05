import sys,os
import numpy as np
from scipy.integrate import quad,fixed_quad
import lhapdf
from mpmath import fp

class IDIS:
  
    def __init__(self,fname):

        self.pdf=lhapdf.mkPDF(fname, 0)
        self.mc=self.pdf.quarkThreshold(4)
        self.mb=self.pdf.quarkThreshold(5)
        self.TR=0.5
        self.CF=4./3.
        self.alfa=1/137.036
        self.M=0.93891897
        self.mpi=0.139
        apU=4.0/9.0
        apD=1.0/9.0
        self.couplings={}
        self.couplings['p']={1:apD,2:apU,3:apD,4:apU,5:apD}
        self.couplings['n']={1:apU,2:apD,3:apD,4:apU,5:apD}
        self.fmap={}

        self.F2_full = {'p': {}, 'n': {}}
        self.F2_LO   = {'p': {}, 'n': {}}
        self.F2_Q1   = {'p': {}, 'n': {}}
        self.F2_G1   = {'p': {}, 'n': {}}

   
    def integrator(self,f,xmin,xmax,method='gauss',n=100):
        f=np.vectorize(f)
        if method=='quad':
            return quad(f,xmin,xmax)[0]
        elif method=='gauss':
            return fixed_quad(f,xmin,xmax,n=n)[0]
      
    def log_plus(self,z,f,x):
        return np.log(1-z)/(1-z)*(f(x/z)/z-f(x)) + 0.5*np.log(1-x)**2*f(x)/(1-x)
  
    def one_plus(self,z,f,x):
        return 1/(1-z)*(f(x/z)/z-f(x))+ np.log(1-x)*f(x)/(1-x)
  
    def C2q(self,z,f,x):
        return self.CF*(2*self.log_plus(z,f,x)-1.5*self.one_plus(z,f,x)\
          +(-(1+z)*np.log(1-z)-(1+z*z)/(1-z)*np.log(z)+3+2*z)*f(x/z)/z\
          -(np.pi**2/3+4.5)*f(x)/(1-x))
      
    def C2g(self,z,f,x):
        return 0.5*(((1-z)**2+z*z)*np.log((1-z)/z)-8*z*z+8*z-1)*f(x/z)/z
   
    def CLq(self,z,f,x):
        return 2*self.CF*z*f(x/z)/z #<--- note prefactor 2, instead of 4 used by MVV
      
    def CLg(self,z,f,x):
        return 4*z*(1-z)*f(x/z)/z
  
    def qplus(self,x,Q2):
        output=0
        for i in range(1,self.Nf+1):
            output+=self.couplings[self.tar][i]*(self.pdf.xfxQ2(i,x,Q2)/x+self.pdf.xfxQ2(-i,x,Q2)/x)
        return output

    def sumpdfquark(self,x,Q2,tar):
        self.tar=tar
        output=self.qplus(x,Q2)
        return output
  
    def glue(self,x,Q2):
        output=0
        for i in range(1,self.Nf+1):
            output+=2*self.couplings[self.tar][i]
        return output*self.pdf.xfxQ2(21,x,Q2)/x
        
    def integrand_F2(self,x,z,Q2):
        return self.C2q(z,lambda y:self.qplus(y,Q2),x) + self.C2g(z,lambda y:self.glue(y,Q2),x)
    
    def integrand_F2_Q1(self,x,z,Q2):
        return self.C2q(z,lambda y:self.qplus(y,Q2),x)
    
    def integrand_F2_G1(self,x,z,Q2):
        return self.C2g(z,lambda y:self.glue(y,Q2),x)
      
    def get_F2_full(self,x,Q2,tar):
        if (x,Q2) not in self.F2_full[tar]:
            self.tar=tar
            alphaS = self.pdf.alphasQ2(Q2)
            self.Nf=3
            if Q2>self.mc**2: self.Nf+=1
            if Q2>self.mb**2: self.Nf+=1
            LO=self.qplus(x,Q2)
            integrand=lambda z:self.integrand_F2(x,z,Q2)
            NLO=self.integrator(integrand,x,1)
            self.F2_full[tar][(x,Q2)]=x*(LO+alphaS/np.pi/2.0*NLO)
        return self.F2_full[tar][(x,Q2)]
    
    def get_F2_LO(self,x,Q2,tar):
        if (x,Q2) not in self.F2_LO[tar]:
            self.tar=tar
            alphaS = self.pdf.alphasQ2(Q2)
            self.Nf=3
            if Q2>self.mc**2: self.Nf+=1
            if Q2>self.mb**2: self.Nf+=1
            LO=self.qplus(x,Q2)
            self.F2_LO[tar][(x,Q2)]=x*LO
        return self.F2_LO[tar][(x,Q2)]
    
    def get_F2_Q1(self,x,Q2,tar):
        if (x,Q2) not in self.F2_Q1[tar]:
            self.tar=tar
            alphaS = self.pdf.alphasQ2(Q2)
            self.Nf=3
            if Q2>self.mc**2: self.Nf+=1
            if Q2>self.mb**2: self.Nf+=1
            integrand=lambda z:self.integrand_F2_Q1(x,z,Q2)
            NLO=self.integrator(integrand,x,1)
            self.F2_Q1[tar][(x,Q2)]=x*(alphaS/np.pi/2.0*NLO)
        return self.F2_Q1[tar][(x,Q2)]
  
    def get_F2_G1(self,x,Q2,tar):
        if (x,Q2) not in self.F2_G1[tar]:
            self.tar=tar
            alphaS = self.pdf.alphasQ2(Q2)
            self.Nf=3
            if Q2>self.mc**2: self.Nf+=1
            if Q2>self.mb**2: self.Nf+=1
            integrand=lambda z:self.integrand_F2_G1(x,z,Q2)
            NLO=self.integrator(integrand,x,1)
            self.F2_G1[tar][(x,Q2)]=x*(alphaS/np.pi/2.0*NLO)
        return self.F2_G1[tar][(x,Q2)]
    




