#!/usr/bin/env python3
import os
import math
import lhapdf

# Proton mass [GeV]
M_PROTON = 0.93891897

# Charge^2 map for flavors: u,c -> 4/9 ; d,s,b -> 1/9
E2 = {1: 1.0/9.0, 2: 4.0/9.0, 3: 1.0/9.0, 4: 4.0/9.0, 5: 1.0/9.0}

def active_flavors(pdf, Q2):
    """Return list of active PDG IDs (1..Nf) based on charm/bottom thresholds."""
    mc = pdf.quarkThreshold(4)  # charm threshold (GeV)
    mb = pdf.quarkThreshold(5)  # bottom threshold (GeV)
    nf = 3
    if Q2 > mc*mc: nf += 1
    if Q2 > mb*mb: nf += 1
    return list(range(1, nf+1))

def f2_lo(pdf, x, Q2):
    """Pure LO, massless F2 using LHAPDF x*f(x)."""
    if x <= 0.0 or x >= 1.0:
        return float('nan')
    f2 = 0.0
    for pid in active_flavors(pdf, Q2):
        # xfi and x fbar_i
        xfi   = pdf.xfxQ2(+pid, x, Q2)
        xfbar = pdf.xfxQ2(-pid, x, Q2)
        f2 += E2[pid] * (xfi + xfbar)
    return f2

def f1_lo_from_f2(F2, x):
    """Massless LO relation."""
    if x <= 0.0 or not math.isfinite(F2):
        return float('nan')
    return F2 / (2.0 * x) # Callan-Gross relation

def x_from_WQ2(W, Q2, M=M_PROTON):
    """Bjorken x from W and Q^2: x = Q^2 / (W^2 - M^2 + Q^2)."""
    denom = W*W - M*M + Q2
    return Q2 / denom if denom > 0 else float('nan')

def main():

#-----------------------Params------------------------------------------#

    pdf_set = "CT18LO"
    
    outdir = f"Output/Output_{pdf_set}_LO"
    os.makedirs(outdir, exist_ok=True)

    # Load LO PDF
    pdf = lhapdf.mkPDF(pdf_set, 0)

    # Q2 grid (your list)
    Q2_list = [1.025, 2.025, 2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699, 15.0, 20.0, 30.0]
    
    wmin = 1.07
    wmax = 31.0
    wstep = 0.01
    
    q2min, q2max = pdf.q2Min, pdf.q2Max
#-----------------------Params------------------------------------------#
    # Outputs
    f2_path = os.path.join(outdir, "F2_LO.txt")
    f1_path = os.path.join(outdir, "F1_LO.txt")
    with open(f2_path, "w") as f2out, open(f1_path, "w") as f1out:
        for Q2 in Q2_list:
          if not ( q2min <= Q2 <= q2max):
              print("Q2 =", Q2, "out of PDF range:", q2min, "-", q2max)
          W = wmin
          while W <= wmax + 1e-12:
            x = x_from_WQ2(W, Q2, M_PROTON)
            # Skip unphysical/unsupported x
            if not (0.0 < x < 1.0):
                continue
            F2 = f2_lo(pdf, x, Q2)
            F1 = f1_lo_from_f2(F2, x)
            f2out.write(f"{Q2}\t{W:.2f}\t{F2:.8e}\n")
            f1out.write(f"{Q2}\t{W:.2f}\t{F1:.8e}\n")
            W = round(W + wstep, 10)  # avoid FP drift


    print(f"Wrote:\n  {f2_path}\n  {f1_path}")

if __name__ == "__main__":
    main()
