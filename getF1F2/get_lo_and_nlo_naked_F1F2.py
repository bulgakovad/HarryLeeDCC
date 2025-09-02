#!/usr/bin/env python3
import os
import math
import argparse
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
    return F2 / (2.0 * x)

def x_from_WQ2(W, Q2, M=M_PROTON):
    """Bjorken x from W and Q^2: x = Q^2 / (W^2 - M^2 + Q^2)."""
    denom = W*W - M*M + Q2
    return Q2 / denom if denom > 0 else float('nan')

def main():
    ap = argparse.ArgumentParser(description="Compute LO F1 and F2 (massless) directly from LHAPDF.")
    ap.add_argument("--pdf", default="CJ15lo", help="LO PDF set name (default: CJ15lo)")
    ap.add_argument("--outdir", default="Output_naked", help="Output directory")
    ap.add_argument("--wmin", type=float, default=1.00, help="min W [GeV]")
    ap.add_argument("--wmax", type=float, default=4.00, help="max W [GeV]")
    ap.add_argument("--wstep", type=float, default=0.01, help="W step [GeV]")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load LO PDF
    pdf = lhapdf.mkPDF(args.pdf, 0)

    # Q2 grid (your list)
    Q2_list = [0.5, 0.75, 1, 1.75, 2, 2.5, 3,
               2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699,
               12, 14, 16, 18, 20]

    # Outputs
    f2_path = os.path.join(args.outdir, "ALL_Q2_broad_W_F2_LO.txt")
    f1_path = os.path.join(args.outdir, "ALL_Q2_broad_W_F1_LO.txt")
    with open(f2_path, "w") as f2out, open(f1_path, "w") as f1out:
        f2out.write("# Q2\tW\tx\tF2_LO\n")
        f1out.write("# Q2\tW\tx\tF1_LO\n")

        
        for Q2 in Q2_list:
          W = args.wmin
          while W <= args.wmax + 1e-12:
            x = x_from_WQ2(W, Q2, M_PROTON)
            # Skip unphysical/unsupported x
            if not (0.0 < x < 1.0):
                continue
            F2 = f2_lo(pdf, x, Q2)
            F1 = f1_lo_from_f2(F2, x)
            f2out.write(f"{Q2}\t{W:.2f}\t{x:.8e}\t{F2:.8e}\n")
            f1out.write(f"{Q2}\t{W:.2f}\t{x:.8e}\t{F1:.8e}\n")
            W = round(W + args.wstep, 10)  # avoid FP drift

    print(f"Wrote:\n  {f2_path}\n  {f1_path}")

if __name__ == "__main__":
    main()
