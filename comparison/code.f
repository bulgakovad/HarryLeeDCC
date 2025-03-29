c
c     Calculate dsigma/dW/dQ^2 using structure function of DCC model
c
c    model is valid for 1.077GeV < W < 2GeV and 0 < Q^2 < 3GeV^2
c     
c     ire = 1 electromagnetic      N(e,e')X
c           2 CC neutirno reaction N(nu,l)X
c           3 NC neutrino reaction N(nu,nu')X
c
c     ilep = 0 masless  2 electron   3 muon   4 tau
c           
c---------------------------------------------------------------------
      implicit real*8(a-h,o-z)
      parameter(maxwcm=150,maxq2=50)
      character cre(3)*2,ctg(2)*7,clm(0:3)*8
      common / const /fnuc,fpio,flepi,flepf,pi,gf,vud,flepm(0:3),fmw,fmz
      common / cstruc / wcmdat(maxwcm),q2dat(maxq2)
     &     ,walldat(5,2,maxwcm,maxq2),wpidat(5,2,maxwcm,maxq2)
     &     ,mxwcm,mxq2
      data flepm/0.d0,0.511d-2,0.10566d0,1.77684d0/
      data fnuc/0.9385d0/, fpio/0.1385d0/,gf/1.16637d-5/,vud/0.9740d0/
      data fmw/80.385d0/, fmz/91.1876d0/,pi/3.1415926d0/
      data cre/'EM','CC','NC'/
      data ctg/'proton ','neutron'/
      data clm/'massless','electo  ','muon    ','tau     '/

c----------------------------------------------------------      
      write(*,*)
      write(*,*)'**************************************************'
      write(*,*)'         dsigma/dW/dQ^2 DCC model'
      write(*,*)
      write(*,*)' dsigma/dw/dQ^2 for'
      write(*,*)'          proton    and neutron'
      write(*,*)'          inclusive and 1pi production'
      write(*,*)'          neutrino  and anti-neutrino'
      write(*,*)
      write(*,*)'     neutrino + p(n)->l^- + X(pi,eta,2pi,..),l^- + pi'
      write(*,*)'anti-neutrino + p(n)->l^+ + X(pi,eta,2pi,..),l^+ + pi'
      write(*,*)
      write(*,*)' input '
      write(*,*)'    reaction           : EM, CC, NC'
      write(*,*)'    Charged Lepton mass: e,mu,tau'
      write(*,*)'    CM energy          :W > m_pi + m_N(GeV)'
      write(*,*)'    4momentum transfer :Q ^2>0(GeV^2)'
      write(*,*)
      write(*,*)' Note the nearest W,Q2 point of table are chosen!'
      write(*,*)'        No interpolation'
      write(*,*)
      write(*,*)'**************************************************'
      write(*,*)
 
      write(*,*)" Select reaction EM, CC or NC"
      write(*,*)" 1 N(e,e')X 2 CC N(nu,l)X 3 NC N(nu,nu')X"
      write(*,*)' input 1(EM),2(CC) or 3(NC)'
      read(*,*)ire

      if(ire.eq.1) then
        ilep = 0               ! massless
        flepi = flepm(ilep)    
        flepf = flepi
      else if(ire.eq.2) then
         write(*,*)" 1 electron, 2 muon, 3 tau"
         write(*,*)" input 1(e), 2(mu) or 3(tau)"
         read(*,*)ilep
         flepi = 0
         flepf = flepm(ilep) 
      else if(ire.eq.3) then
         ilep  = 0
         flepi = 0
         flepf = 0
      else
         stop
      end if

c----------------------------------------------------------
c     
c     read data files of structure functions
c
      call readstruc(ire)   
c----------------------------------------------------------
      
      write(*,*)' input W(GeV), Q^2(GeV^2) '
      write(*,*)' 1.077 < W < 2 GeV, 0 < Q^2 < 3 GeV^2'
      read(*,*)wcmin,q2in

c     choose nearest W,Q2 from table
      
      call setiwiq(wcmin,q2in,iw,iq)
      wcm = wcmdat(iw)
      q2  = q2dat(iq)
      
      write(*,*)' Input Lab initial lepton energy(GeV)'
      read(*,*)elep

c     check  wcm < sqrt(2*m_N*E_l + m_N^2 + m_l^2) - m_{l'}
      
      wtot = sqrt(2.d0*fnuc*elep+fnuc**2 + flepi**2)
      if(wcm.gt.wtot-flepf) then
         write(*,*)' wcm of range '
         stop
      end if

c     check  Q2

      pcmi = sqrt(((wtot**2-fnuc**2+flepi**2)/(2.d0*wtot))**2-flepi**2)
      pcmf = sqrt(((wtot**2- wcm**2+flepf**2)/(2.d0*wtot))**2-flepf**2)
      ecmi = sqrt(pcmi**2+flepi**2)
      ecmf = sqrt(pcmf**2+flepf**2)
      qxmin = -flepi**2-flepf**2+2.d0*(ecmi*ecmf-pcmi*pcmf)
      qxmax = -flepi**2-flepf**2+2.d0*(ecmi*ecmf+pcmi*pcmf)
      if(q2.lt.qxmin.or.q2.gt.qxmax) then
         write(*,*) ' q2 out of range'
         stop
      end if
      
      
c---------------------------------------------------------------------
c
c  calculate cross section      
c
c---------------------------------------------------------------------
         write(*,3000)cre(ire),flepi,flepf
 3000    format(1h ,' reaction: ',a2,' lepton mass: initial ',f10.5
     &            ,'  final ',f10.5)
         write(*,*)
         
c     
      write(*,*)
      if(ire.ne.1) then
      write(*,2004)clm(ilep)
 2004 format(1h ,'     dsigma/dW/dQ^2 [ 10^{-38}cm^3/GeV^3] ',a8)
      else
      write(*,2014)clm(ilep)
 2014 format(1h ,'     dsigma/dW/dQ^2 [ 10^{-30}cm^3/GeV^3] ',a8)
      end if
      write(*,*)
      
c     inu = 1 neutrino,lep-,  inu=-1 anti-neutrino,lep+
      inmax =  1
      inmin = -1
      if(ire.eq.1) inmin = 1
      
      do inu = inmax,inmin,-2
      
      if(inu.eq.1) then
         if(ire.eq.1) then
            write(*,2003)
         else
            write(*,2001)
         end if
      else if(inu.eq.-1) then
         if(ire.ne.1) then
            write(*,2002)
         end if
      end if
      
 2001 format(1h ,'E_nu',6x,'W',9x,'Q2',8x,'nu p -> l- X',3x,
     &  'nu p -> l- piN','  nu n -> l- X ',' nu n -> l- piN')
 2003 format(1h ,'E_le',6x,'W',9x,'Q2',8x,'l  p -> l- X',3x,
     &  'l  p -> l- piN','  l  n -> l- X ',' l  n -> l- piN')
 2002 format(1h ,'E_nu',6x,'W',9x,'Q2',8x,'a-nu p -> l+ X',1x,
     &  'a-nu p -> l+ piN','  a-nu n -> l+ X ',' a-nu n -> l+ piN')

      if(inu.eq.1) then
c     proton
         
         ipn = 1
         w1  = wpidat(1,ipn,iw,iq)
         w2  = wpidat(2,ipn,iw,iq)
         w3  = wpidat(3,ipn,iw,iq)
         w4  = wpidat(4,ipn,iw,iq)
         w5  = wpidat(5,ipn,iw,iq)
         
      call cross(elep,wcm,q2,ire,inu,w1,w2,w3,w4,w5,xrsppi)

         w1  = walldat(1,ipn,iw,iq)
         w2  = walldat(2,ipn,iw,iq)
         w3  = walldat(3,ipn,iw,iq)
         w4  = walldat(4,ipn,iw,iq)
         w5  = walldat(5,ipn,iw,iq)
         
      call cross(elep,wcm,q2,ire,inu,w1,w2,w3,w4,w5,xrspx)

c     neutron
      
         ipn = 2
         w1  = wpidat(1,ipn,iw,iq)
         w2  = wpidat(2,ipn,iw,iq)
         w3  = wpidat(3,ipn,iw,iq)
         w4  = wpidat(4,ipn,iw,iq)
         w5  = wpidat(5,ipn,iw,iq)
         
      call cross(elep,wcm,q2,ire,inu,w1,w2,w3,w4,w5,xrsnpi)

         w1  = walldat(1,ipn,iw,iq)
         w2  = walldat(2,ipn,iw,iq)
         w3  = walldat(3,ipn,iw,iq)
         w4  = walldat(4,ipn,iw,iq)
         w5  = walldat(5,ipn,iw,iq)
         
      call cross(elep,wcm,q2,ire,inu,w1,w2,w3,w4,w5,xrsnx)

c     we assume iso-spin symmetry
c     for anti-neutrino, W(proton) <-> W(neutron)
c
      
      else if(inu.eq.-1) then

c     proton
         
         ipn = 2
         w1  = wpidat(1,ipn,iw,iq)
         w2  = wpidat(2,ipn,iw,iq)
         w3  = wpidat(3,ipn,iw,iq)
         w4  = wpidat(4,ipn,iw,iq)
         w5  = wpidat(5,ipn,iw,iq)
         
      call cross(elep,wcm,q2,ire,inu,w1,w2,w3,w4,w5,xrsppi)

         w1  = walldat(1,ipn,iw,iq)
         w2  = walldat(2,ipn,iw,iq)
         w3  = walldat(3,ipn,iw,iq)
         w4  = walldat(4,ipn,iw,iq)
         w5  = walldat(5,ipn,iw,iq)
         
      call cross(elep,wcm,q2,ire,inu,w1,w2,w3,w4,w5,xrspx)

c     neutron
      
         ipn = 1
         w1  = wpidat(1,ipn,iw,iq)
         w2  = wpidat(2,ipn,iw,iq)
         w3  = wpidat(3,ipn,iw,iq)
         w4  = wpidat(4,ipn,iw,iq)
         w5  = wpidat(5,ipn,iw,iq)
         
      call cross(elep,wcm,q2,ire,inu,w1,w2,w3,w4,w5,xrsnpi)

         w1  = walldat(1,ipn,iw,iq)
         w2  = walldat(2,ipn,iw,iq)
         w3  = walldat(3,ipn,iw,iq)
         w4  = walldat(4,ipn,iw,iq)
         w5  = walldat(5,ipn,iw,iq)
         
      call cross(elep,wcm,q2,ire,inu,w1,w2,w3,w4,w5,xrsnx)

      end if

      write(*,2000)elep,wcm,q2,xrspx,xrsppi,xrsnx,xrsnpi
 2000 format(1h ,3f10.5,4e15.5)
      end do

      stop
      end

c
      subroutine cross(elepi,wcm,q2,ire,inu,w1,w2,w3,w4,w5,dcrs)
      implicit real*8(a-h,o-y)
      implicit complex*16(z)      
      common / const /fnuc,fpio,flepi,flepf,pi,gf,vud,flepm(0:3),fmw,fmz
      plepi  = sqrt(elepi**2 - flepi**2)
      omeg   = (wcm**2 + q2 -fnuc**2)/(2.d0*fnuc)
      elepf  = elepi - omeg
      plepf  = sqrt(elepf**2 - flepf**2)
      clep   = (-q2 - flepi**2 - flepf**2 + 2.d0*elepi*elepf)
     &        /(2.d0*plepi*plepf)

      
      fac3 = pi*wcm/fnuc/plepi/plepf
      if(ire.eq.1) then        ! elemag  10^{-30}cm^2
      fcrs3 =  4.d0*(1.d0/137.04d0/q2)**2
     &        * 0.197327d0**2*1.d4
     &        * plepf/plepi*elepi*elepf
      else if(ire.eq.2) then   !cc      10^{-38}cm^2
      fcrs3 =  (gf*vud)**2
     &        * 0.197327d0**2*1.d12
     &        * plepf*elepf/2.d0/pi**2
     &        * (fmw**2/(fmw**2 + q2))**2
      else if(ire.eq.3) then   ! NC
      fcrs3 =  gf**2
     &       * 0.197327d0**2*1.d12
     &        * plepf*elepf/2.d0/pi**2
     &        * (fmz**2/(fmz**2 + q2))**2
      end if

      betai = plepi/elepi
      betaf = plepf/elepf
      if(ire.eq.1) then
         ss2 = (1.d0 - betai*betaf*clep)/2.d0-flepf**2/elepi/elepf
         cc2 = (1.d0 + betai*betaf*clep)/2.d0+flepf**2/elepi/elepf/2.d0
      else if(ire.eq.2) then
         ss2 = (1.d0 - betaf*clep)/2.d0
         cc2 = (1.d0 + betaf*clep)/2.d0
      else if(ire.eq.3) then
         ss2 = (1.d0 - clep)/2.d0
         cc2 = (1.d0 + clep)/2.d0
      end if
      
      w1x = w1
      w2x = w2
      w3x = w3
      w4x = w4
      w5x = w5

      xxx = 2.d0*ss2*w1x + cc2*w2x
c  20190306      
      if(ire.eq.2) then
      xxx = xxx 
     &  + dble(inu)*w3x*
     &    ((elepi+elepf)*ss2 - flepf**2/2.d0/elepf)/fnuc
     &  + flepf**2/fnuc**2*ss2*w4x
     &  - flepf**2/fnuc/elepf*w5x
      else if(ire.eq.3) then
      xxx = xxx  + dble(inu)*w3x*(elepi+elepf)*ss2/fnuc
      end if

      dcrs = fcrs3*fac3*xxx


c      write(*,1234)ire,inu,w1,w2,w3,w4,w5,xxx
c 1234 format(1h ,2i3,10e15.5)

      return
      end

c---------------------------------------------------------------
      subroutine readstruc(ire)
      implicit real*8(a-h,o-y)
      implicit complex*16(z)
      parameter(maxwcm=150,maxq2=50)
      common / cstruc / wcmdat(maxwcm),q2dat(maxq2)
     &     ,walldat(5,2,maxwcm,maxq2),wpidat(5,2,maxwcm,maxq2)
     &     ,mxwcm,mxq2
      common / cmxwq2/ wcminin,wcmaxin,q2minin,q2maxin

      
      if(ire.eq.1) then
         open(unit=11,file='wempx.dat')
         open(unit=12,file='wemnx.dat')
         open(unit=13,file='wemp-pi.dat')
         open(unit=14,file='wemn-pi.dat')

         
      else if(ire.eq.2) then
         open(unit=11,file='wccpx.dat')
         open(unit=12,file='wccnx.dat')
         open(unit=13,file='wccp-pi.dat')
         open(unit=14,file='wccn-pi.dat')


      else if(ire.eq.3) then
         open(unit=11,file='wncpx.dat')
         open(unit=12,file='wncnx.dat')
         open(unit=13,file='wncp-pi.dat')
         open(unit=14,file='wncn-pi.dat')

         
      end if
       

      call readincl(ire)
      call readpi(ire)

      close(unit=11)
      close(unit=12)
      close(unit=13)
      close(unit=14)

      wcminin = wcmdat(1)
      q2minin = q2dat(1)
      wcmaxin = wcmdat(mxwcm)
      q2maxin = q2dat(mxq2)
      
      return
      end
c-------------------------------------------------------------------
      subroutine readincl(ire)
      implicit real*8(a-h,o-y)
      implicit complex*16(z)
      parameter(maxwcm=150,maxq2=50)
      common / cstruc / wcmdat(maxwcm),q2dat(maxq2)
     &     ,walldat(5,2,maxwcm,maxq2),wpidat(5,2,maxwcm,maxq2)
     &     ,mxwcm,mxq2

      walldat = 0
         read(11,*)mxwcm,mxq2
         read(12,*)mxwcm,mxq2
         do iw = 1,mxwcm
         do iq = 1,mxq2
            if(ire.eq.1) then   
            read(11,*)wcm,q2,wp1,wp2
            read(12,*)wcm,q2,wn1,wn2
            walldat(1,1,iw,iq) = wp1
            walldat(1,2,iw,iq) = wn1
            walldat(2,1,iw,iq) = wp2
            walldat(2,2,iw,iq) = wn2
            else if(ire.eq.2) then
            read(11,*)wcm,q2,wp1,wp2,wp3,wp4,wp5
            read(12,*)wcm,q2,wn1,wn2,wn3,wn4,wn5
            walldat(1,1,iw,iq) = wp1
            walldat(1,2,iw,iq) = wn1
            walldat(2,1,iw,iq) = wp2
            walldat(2,2,iw,iq) = wn2
            walldat(3,1,iw,iq) = wp3
            walldat(3,2,iw,iq) = wn3
            walldat(4,1,iw,iq) = wp4
            walldat(4,2,iw,iq) = wn4
            walldat(5,1,iw,iq) = wp5
            walldat(5,2,iw,iq) = wn5
            else if(ire.eq.3) then
            read(11,*)wcm,q2,wp1,wp2,wp3
            read(12,*)wcm,q2,wn1,wn2,wn3
            walldat(1,1,iw,iq) = wp1
            walldat(1,2,iw,iq) = wn1
            walldat(2,1,iw,iq) = wp2
            walldat(2,2,iw,iq) = wn2
            walldat(3,1,iw,iq) = wp3
            walldat(3,2,iw,iq) = wn3
            end if
            wcmdat(iw) = wcm
            q2dat(iq)  = q2
         end do
      end do
      return
      end
c--------------------------------------------------------------------------
      subroutine readpi(ire)
      implicit real*8(a-h,o-y)
      implicit complex*16(z)
      parameter(maxwcm=150,maxq2=50)
      common / cstruc / wcmdat(maxwcm),q2dat(maxq2)
     &     ,walldat(5,2,maxwcm,maxq2),wpidat(5,2,maxwcm,maxq2)
     &     ,mxwcm,mxq2

      wpidat  = 0
         read(13,*)mxwcm,mxq2
         read(14,*)mxwcm,mxq2
         do iw = 1,mxwcm
         do iq = 1,mxq2
            if(ire.eq.1) then   
            read(13,*)wcm,q2,wp1p,wp2p
            read(14,*)wcm,q2,wn1p,wn2p
            wpidat(1,1,iw,iq) = wp1p
            wpidat(1,2,iw,iq) = wn1p
            wpidat(2,1,iw,iq) = wp2p
            wpidat(2,2,iw,iq) = wn2p
            else if(ire.eq.2) then
            read(13,*)wcm,q2,wp1p,wp2p,wp3p,wp4p,wp5p
            read(14,*)wcm,q2,wn1p,wn2p,wn3p,wn4p,wn5p
            wpidat(1,1,iw,iq) = wp1p
            wpidat(1,2,iw,iq) = wn1p
            wpidat(2,1,iw,iq) = wp2p
            wpidat(2,2,iw,iq) = wn2p
            wpidat(3,1,iw,iq) = wp3p
            wpidat(3,2,iw,iq) = wn3p
            wpidat(4,1,iw,iq) = wp4p
            wpidat(4,2,iw,iq) = wn4p
            wpidat(5,1,iw,iq) = wp5p
            wpidat(5,2,iw,iq) = wn5p
            else if(ire.eq.3) then
            read(13,*)wcm,q2,wp1p,wp2p,wp3p
            read(14,*)wcm,q2,wn1p,wn2p,wn3p
            wpidat(1,1,iw,iq) = wp1p
            wpidat(1,2,iw,iq) = wn1p
            wpidat(2,1,iw,iq) = wp2p
            wpidat(2,2,iw,iq) = wn2p
            wpidat(3,1,iw,iq) = wp3p
            wpidat(3,2,iw,iq) = wn3p
            end if
            wcmdat(iw) = wcm
            q2dat(iq)  = q2
            
      end do
      end do
      return
      end
c
c    find W,Q2 of table 
c     
      subroutine setiwiq(win,q2in,iw,iq)
      parameter(maxwcm=150,maxq2=50)
      implicit real*8(a-h,o-z)
      common / cstruc / wcmdat(maxwcm),q2dat(maxq2)
     &     ,walldat(5,2,maxwcm,maxq2),wpidat(5,2,maxwcm,maxq2)
     &     ,mxwcm,mxq2
c
      wmin  = 1.077d0
      wmax  = 2.d0
      q2min = 0.d0
      q2max = 3.d0

      if(win.lt.wmin.or.win.gt.wmax) then
         write(*,*)' W out of range : 1.077 < W < 2 '
         stop
      end if
      if(q2in.lt.q2min.or.q2in.gt.q2max) then
         write(*,*)' Q2 out of range : 0 < Q2 < 3 '
         stop
      end if
      
c
      iw1    = 1
      do iwx = 2,mxwcm
         if(win.lt.wcmdat(iwx)) then
            iw2 = iwx
            go to 100
         else
            iw1 = iwx
         end if
      end do
 100  continue
      
      xx = (win- wcmdat(iw1))/(wcmdat(iw2)-wcmdat(iw1))
      if(xx.lt.0.5d0) then
         iw = iw1
      else
         iw = iw2
      end if
c
      iq1    = 1
      do iqx = 2,mxq2
         if(q2in.lt.q2dat(iqx)) then
            iq2 = iqx
            go to 200
         else
            iq1 = iqx
         end if
      end do
 200  continue
      xx = (q2in- q2dat(iq1))/(q2dat(iq2)-q2dat(iq1))
      if(xx.lt.0.5d0) then
         iq = iq1
      else
         iq = iq2
      end if
      
      return
      end
      