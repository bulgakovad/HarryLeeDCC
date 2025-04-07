      Program tst_CJpdf
      implicit real*8 (a-h,o-z)
      integer iset, j, l, k
      real*8 x, Q2, Q, row(13)
      dimension pdf(-5:5)
      character*80 outfile
      character*20 iset_str, Q2_str
      integer nQ
      nQ = 1
      Q2 = 2.774

      print*,'ISET = '
      read*, iset
      call setCJ(iset)

      write(iset_str, '(I0)') iset
      write(Q2_str, '(F0.3)') Q2
      outfile = 'output/tst_CJpdf_ISET=' // trim(iset_str) // 
     1 '_Q2=' // trim(Q2_str) // '.dat'
      open(unit=1, file=outfile, status='unknown')

C     Write header including the Q**2 column as the first column
      write(1,4)
 4    format(5x,'Q**2',10x,'x',9x,'u',11x,'d',11x,'g',10x,'ub',10x,'db',/
     1 10x,'sb',10x,'s',10x,'cb',10x,'c',10x,'bb',10x,'b')

      do 10 j = 1, nQ
         Q = sqrt(Q2)
         do 20 l = 1, 19
            x = l * 0.05
            do 30 k = -5, 5
               pdf(k) = x * CJpdf(k, x, Q)
   30       continue
            row(1)  = Q2
            row(2)  = x
            row(3)  = pdf(1)
            row(4)  = pdf(2)
            row(5)  = pdf(0)
            row(6)  = pdf(-1)
            row(7)  = pdf(-2)
            row(8)  = pdf(-3)
            row(9)  = pdf(3)
            row(10) = pdf(-4)
            row(11) = pdf(4)
            row(12) = pdf(-5)
            row(13) = pdf(5)
            write(1,3) row
 3       format(f10.3, f10.3, 11(1x,e12.3))
   20    continue
         write(1,*)
   10 continue

      print*,'Output PDFs in ', outfile
      call exit
      end
