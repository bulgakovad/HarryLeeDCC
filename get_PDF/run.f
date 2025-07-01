      Program tst_CJpdf
      implicit real*8 (a-h,o-z)
      integer iset, j, l, k
      real*8 x, Q2, Q, row(13)
      dimension pdf(-5:5)
      character*80 outfile, outdir, mkdir_cmd
      character*20 iset_str, Q2_str
      integer nQ, last
      nQ = 1

      print*, 'ISET = '
      read*, iset
      call setCJ(iset)

      write(iset_str, '(I0)') iset

      print*, 'Q2 = '
      read*, Q2

C Format Q2 string and clean it
      write(Q2_str, '(F6.3)') Q2
      Q2_str = adjustl(Q2_str)

C Strip trailing zeros
99    last = len_trim(Q2_str)
      if (Q2_str(last:last) == '0' .and. index(Q2_str, '.') > 0) then
         Q2_str(last:last) = ''
         goto 99
      endif
C Strip trailing dot if any
      last = len_trim(Q2_str)
      if (Q2_str(last:last) == '.') then
         Q2_str(last:last) = ''
      endif

C Create output subdirectory "output/Q2=<Q2_str>"
      outdir = 'output/Q2=' // trim(Q2_str)
      mkdir_cmd = 'mkdir -p ' // trim(outdir)
      call system(trim(mkdir_cmd))

C Build output file path
      outfile = trim(outdir) // '/tst_CJpdf_ISET=' // trim(iset_str) //
     1 '_Q2=' // trim(Q2_str) // '.dat'
      open(unit=1, file=outfile, status='unknown')

C Write header
      write(1,4)
 4    format(5x,'Q**2',10x,'x',9x,'u',11x,'d',11x,'g',10x,'ub',10x,'db',/
     1 10x,'sb',10x,'s',10x,'cb',10x,'c',10x,'bb',10x,'b')

      do 10 j = 1, nQ
         Q = sqrt(Q2)
         do 20 l = 1, 99
            x = l * 0.01
            do 30 k = -5, 5
               pdf(k) = x * CJpdf(k, x, Q)
30          continue
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
 3          format(f10.3, f10.3, 11(1x,e12.3))
20       continue
         write(1,*)
10    continue

      print*, 'Output PDFs in ', outfile
      call exit
      end
