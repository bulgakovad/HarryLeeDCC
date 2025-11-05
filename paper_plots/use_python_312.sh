#!/bin/tcsh
module purge
module use /scigroup/cvmfs/hallb/clas12/sw/modulefiles
module avail python          # проверь, что виден python/3.12.4
module load python/3.12.4
python -V