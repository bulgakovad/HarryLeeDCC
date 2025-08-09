import sys
import lhapdf
from theory import IDIS

# Initialize LHAPDF
print("LHAPDF version:", lhapdf.__version__)
pdf = lhapdf.mkPDF("CJ15nlo", 0)  # We will install this PDF soon

# Create an IDIS instance
idis = IDIS("CJ15nlo")

# Test values
x = 0.1
Q2 = 10.0

try:
    F2_p = idis.get_F2(x, Q2, 'p')
    F1_p = idis.get_F1(x, Q2, 'p')
    print(f"Test x={x}, Q2={Q2}:")
    print(f"  F2^p = {F2_p}")
    print(f"  F1^p = {F1_p}")
except Exception as e:
    print("Error while running theory.py:", e)
