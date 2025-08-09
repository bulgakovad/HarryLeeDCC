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

        self.F2={'p':{},'n':{}}
        self.FL={'p':{},'n':{}}
   
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
      
    def integrand_FL(self,x,z,Q2):
        return self.CLq(z,lambda y:self.qplus(y,Q2),x) + self.CLg(z,lambda y:self.glue(y,Q2),x)
      
    def get_F2(self,x,Q2,tar):
        if (x,Q2) not in self.F2[tar]:
            self.tar=tar
            alphaS = self.pdf.alphasQ2(Q2)
            self.Nf=3
            if Q2>self.mc**2: self.Nf+=1
            if Q2>self.mb**2: self.Nf+=1
            LO=self.qplus(x,Q2)
            integrand=lambda z:self.integrand_F2(x,z,Q2)
            NLO=self.integrator(integrand,x,1)
            self.F2[tar][(x,Q2)]=x*(LO+alphaS/np.pi/2.0*NLO)
        return self.F2[tar][(x,Q2)]
  
    def get_FL(self,x,Q2,tar):
        if (x,Q2) not in self.FL[tar]:
            self.tar=tar
            alphaS = self.pdf.alphasQ2(Q2)
            self.Nf=3
            if Q2>self.mc**2: self.Nf+=1
            if Q2>self.mb**2: self.Nf+=1
            integrand=lambda z:self.integrand_FL(x,z,Q2)
            NLO=self.integrator(integrand,x,1)
            self.FL[tar][(x,Q2)]= x*alphaS/np.pi/2.0*NLO
        return self.FL[tar][(x,Q2)]
  
    def get_F1(self,x,Q2,tar):
        F2=self.get_F2(x,Q2,tar)
        FL=self.get_FL(x,Q2,tar)
        return ((1+4*self.M**2/Q2*x**2)*F2-FL)/(2*x)
   
    def get_dsigdxdQ2(self,x,y,Q2,target,precalc=False):
        if precalc==False: 
            return 4*np.pi*self.alfa**2/Q2**2/x*((1-y+y**2/2)*self.get_F2(x,Q2,target)-y**2/2*self.get_FL(x,Q2,target))
        else:
            return self.storage.retrieve([x,y,Q2,target])    
  




#--load PPDFs and calculate...
class PIDIS:
  
    def __init__(self,fname):

        self.pdf=lhapdf.mkPDFs(fname)
        self.size = len(self.pdf)
        self.mc=self.pdf[0].quarkThreshold(4)
        self.mb=self.pdf[0].quarkThreshold(5)
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

        self.g1={'p':{},'n':{}}
        self.g1['p']['mean'] = {}
        self.g1['p']['std']  = {}
        self.g1['n']['mean'] = {}
        self.g1['n']['std']  = {}

        self.g2={'p':{},'n':{}}
        self.g2['p']['mean'] = {}
        self.g2['p']['std']  = {}
        self.g2['n']['mean'] = {}
        self.g2['n']['std']  = {}
   
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
  
    def PC1q(self,z,f,x):
        zeta2 = fp.zeta(2)
        return self.CF*(4*self.log_plus(z,f,x)-3*self.one_plus(z,f,x)\
        +(-2*(1+z)*np.log(1-z)-2*(1+z*z)*np.log(z)/(1-z)+4+2*z)*f(x/z)/z\
        -(4*zeta2 + 9)*f(x)/(1-x))

    def PC1g(self,z,f,x):
        return 0.5*(4*(2*z-1)*(np.log(1-z)-np.log(z))+4*(3-4*z))*f(x/z)/z
 
    def qplus(self,rep,x,Q2):
        output=0
        for i in range(1,self.Nf+1):
            output+=self.couplings[self.tar][i]*(self.pdf[rep].xfxQ2(900+i,x,Q2)/x)#+self.pdf[rep].xfxQ2(-i,x,Q2)/x)
        return output

    def sumpdfquark(self,x,Q2,tar):
        self.tar=tar
        output=self.qplus(x,Q2)
        return output
  
    def glue(self,rep,x,Q2):
        output=0
        for i in range(1,self.Nf+1):
            output+=2*self.couplings[self.tar][i]
        return output*self.pdf[rep].xfxQ2(21,x,Q2)/x
        
    def integrand_g1(self,rep,x,z,Q2):
        return self.PC1q(z,lambda y:self.qplus(rep,y,Q2),x) + self.PC1g(z,lambda y:self.glue(rep,y,Q2),x)
      
    def get_g1(self,x,Q2,tar):
        if (x,Q2) not in self.g1[tar]['mean']:
            G1 = []
            self.tar=tar
            alphaS = self.pdf[0].alphasQ2(Q2)
            self.Nf=3
            if Q2>self.mc**2: self.Nf+=1
            if Q2>self.mb**2: self.Nf+=1
            for rep in range(self.size):
                LO=self.qplus(rep,x,Q2)
                integrand=lambda z:self.integrand_g1(rep,x,z,Q2)
                NLO=self.integrator(integrand,x,1,n=10)
                G1.append(0.5*(LO+alphaS/np.pi/4.0*NLO))
            G1 = np.array(G1)
            self.g1[tar]['mean'][(x,Q2)]=np.mean(G1)
            self.g1[tar]['std'] [(x,Q2)]=np.std (G1)
        return self.g1[tar]['mean'][(x,Q2)], self.g1[tar]['std'][(x,Q2)]
      
    def get_g1_rep(self,rep,x,Q2,tar):
        self.tar=tar
        alphaS = self.pdf[0].alphasQ2(Q2)
        self.Nf=3
        if Q2>self.mc**2: self.Nf+=1
        if Q2>self.mb**2: self.Nf+=1
        LO=self.qplus(rep,x,Q2)
        integrand=lambda z:self.integrand_g1(rep,x,z,Q2)
        NLO=self.integrator(integrand,x,1,n=10)
        res=0.5*(LO+alphaS/np.pi/4.0*NLO)
        return res

    def get_g2(self,x,Q2,tar):
        if (x,Q2) not in self.g2[tar]['mean']:
            G2 = []
            self.tar=tar
            alphaS = self.pdf[0].alphasQ2(Q2)
            self.Nf=3
            if Q2>self.mc**2: self.Nf+=1
            if Q2>self.mb**2: self.Nf+=1
            for rep in range(self.size):
                LO=-self.get_g1_rep(rep,x,Q2,tar)
                integrand=lambda z:self.get_g1_rep(rep,z,Q2,tar)/z
                NLO=self.integrator(integrand,x,1,n=10)
                G2.append(LO+NLO)
            G2 = np.array(G2)
            self.g2[tar]['mean'][(x,Q2)]=np.mean(G2)
            self.g2[tar]['std'] [(x,Q2)]=np.std (G2)
        return self.g2[tar]['mean'][(x,Q2)], self.g2[tar]['std'][(x,Q2)]



#--load PFFs and calculate...
class PSF:
  
    def __init__(self,fname):

        self.sf=lhapdf.mkPDFs(fname)
        self.size = len(self.sf)

        self.g1={'lt':{},'ht':{}}
        self.g1['lt']['mean'] = {}
        self.g1['lt']['std']  = {}
        self.g1['ht']['mean'] = {}
        self.g1['ht']['std']  = {}

        self.g2={'lt':{},'ht':{}}
        self.g2['lt']['mean'] = {}
        self.g2['lt']['std']  = {}
        self.g2['ht']['mean'] = {}
        self.g2['ht']['std']  = {}
  
    def g1rep(self,rep,x,Q2):
        return self.sf[rep].xfxQ2(950,x,Q2)

    def g2rep(self,rep,x,Q2):
        return self.sf[rep].xfxQ2(951,x,Q2)
  
    def g1ltrep(self,rep,x,Q2):
        return self.sf[rep].xfxQ2(967,x,Q2)

    def g2ltrep(self,rep,x,Q2):
        return self.sf[rep].xfxQ2(970,x,Q2)
      
    def get_g1(self,x,Q2,twis):
        if (x,Q2) not in self.g1[twis]['mean']:
            G1 = []
            self.twis=twis
            if twis=='ht':
	            for rep in range(self.size):
	                G1.append(self.g1rep(rep,x,Q2))
            if twis=='lt':
	            for rep in range(self.size):
	                G1.append(self.g1ltrep(rep,x,Q2))
            G1 = np.array(G1)
            self.g1[twis]['mean'][(x,Q2)]=np.mean(G1)
            self.g1[twis]['std'] [(x,Q2)]=np.std (G1)
        return self.g1[twis]['mean'][(x,Q2)], self.g1[twis]['std'][(x,Q2)]
      
    def get_g2(self,x,Q2,twis):
        if (x,Q2) not in self.g2[twis]['mean']:
            G2 = []
            self.twis=twis
            if twis=='ht':
	            for rep in range(self.size):
	                G2.append(self.g2rep(rep,x,Q2))
            if twis=='lt':
	            for rep in range(self.size):
	                G2.append(self.g2ltrep(rep,x,Q2))
            G2 = np.array(G2)
            self.g2[twis]['mean'][(x,Q2)]=np.mean(G2)
            self.g2[twis]['std'] [(x,Q2)]=np.std (G2)
        return self.g2[twis]['mean'][(x,Q2)], self.g2[twis]['std'][(x,Q2)]

