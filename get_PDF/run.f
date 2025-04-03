c  to run: gfortran -o executable run.f tst_CJpdf.f
      Program tst_CJpdf
      implicit real*8 (a-h,o-z)
      integer iset, j, l, k
      real*8 x, Q2, Q, row(13)
      dimension pdf(-5:5)
      character*80 outfile
      integer nQ
      nQ = 100

      print*,'ISET = '
      read*, iset
      call setCJ(iset)

      write(outfile, '(A,I3,A)') 'output/tst_CJpdf_ISET=', iset, '.out'
      open(unit=1, file=outfile, status='unknown')

C     Write header including the Q**2 column as the first column
      write(1,4)
 4    format(5x,'Q**2',10x,'x',9x,'u',11x,'d',11x,'g',10x,'ub',10x,'db',
     1 10x,'sb',10x,'s',10x,'cb',10x,'c',10x,'bb',10x,'b')

      do j = 1, nQ
         Q2 = j * 0.1
         Q = sqrt(Q2)
         do l = 1, 99
            x = l * 0.01
            do k = -5, 5
               pdf(k) = x * CJpdf(k, x, Q)
            enddo
C           Build output row:
C             row(1)  = Q**2,
C             row(2)  = x,
C             row(3)  = pdf(1),
C             row(4)  = pdf(2),
C             row(5)  = pdf(0),
C             row(6)  = pdf(-1),
C             row(7)  = pdf(-2),
C             row(8)  = pdf(-3),
C             row(9)  = pdf(3),
C             row(10) = pdf(-4),
C             row(11) = pdf(4),
C             row(12) = pdf(-5),
C             row(13) = pdf(5)
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
 3       format(f10.3, f10.3, 11e12.3)
         enddo
         write(1,*)
      enddo

      print*,'Output PDFs in ', outfile
      call exit
      end