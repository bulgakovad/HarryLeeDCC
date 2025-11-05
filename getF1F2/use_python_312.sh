# 1) clean slate
module purge

# 2) add the CLAS12 modulefiles path (the one your script printed)
module use /scigroup/cvmfs/hallb/clas12/sw/modulefiles

# 3) confirm it’s visible, then load
module avail python
module load python/3.12.4

# 4) verify
which python
python -V