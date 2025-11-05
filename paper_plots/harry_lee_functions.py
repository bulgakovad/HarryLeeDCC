import numpy as np
import math
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os
import pandas as pd


from functions_pdf import get_lo_pdf_interpolators, get_nlo_pdf_interpolators, compute_pdf_cross_sections, compute_pdf_cross_sections_from_F2_FL
from functions_anl_osaka import compute_cross_section_model, compute_1pi_cross_section_model, compute_2pi_cross_section_model



def compare_F1(Q2_list, pdf_set_lo, pdf_set_nlo, num_points=400, W_cutoff=4.0):
    """
    For each Q² in Q2_list, plot:
        - F1 LO (CJ15) 
        - F1 NLO LT (CJ15)
        - F1 NLO TMC only (CJ15)
        - F1 NLO TMC + HT (CJ15)
    """
    out_dir = f"compare_F1_{pdf_set_nlo}"
    os.makedirs(out_dir, exist_ok=True)

    for Q2 in Q2_list:
        have_lo = have_nlo = False
        # LO: F1(W) 
        try:
            F1_LO, _, W_lo_rng = get_lo_pdf_interpolators(Q2,pdf_set=pdf_set_lo) 
            W_lo_min, W_lo_max = float(np.min(W_lo_rng)), float(np.max(W_lo_rng))
            have_lo = True
        except Exception:
            pass
        # NLO , NLO TMC, NLO TMC+HT
        
        try:
            F1_NLO, F1_NLO_TMC, F1_NLO_TMC_alt, F1_NLO_TMC_HT, _, _, _, _, _, _, W_nlo_rng = get_nlo_pdf_interpolators(Q2,pdf_set=pdf_set_nlo)
            # (we don't use F1_b_alt here
            W_nlo_min, W_nlo_max = float(np.min(W_nlo_rng)), float(np.max(W_nlo_rng))
            have_nlo = True
        except Exception:
            pass

        if not (have_lo or have_nlo):
            print(f"[WARNING] Q²={Q2}: no LO/NLO F1 available — skipping.")
            continue

        # W grid up to cutoff (start from min available W among datasets)
        wmins = []
        if have_lo:  wmins.append(W_lo_min)
        if have_nlo: wmins.append(W_nlo_min)
        W_min_global = max(1.0, min(wmins)) if wmins else 1.0
        W_vals = np.linspace(W_min_global, W_cutoff, num_points)

        # Evaluate only where each dataset is defined
        F1_LO_vals = np.full_like(W_vals, np.nan, dtype=float)
        if have_lo:
            m = (W_vals >= W_lo_min) & (W_vals <= W_lo_max)
            try:
                F1_LO_vals[m] = F1_LO(W_vals[m])
            except Exception:
                pass

        F1_NLO_vals = np.full_like(W_vals, np.nan, dtype=float)
        F1_NLO_TMC_vals = np.full_like(W_vals, np.nan, dtype=float)
        F1_NLO_TMC_HT_vals = np.full_like(W_vals, np.nan, dtype=float)
        if have_nlo:
            m = (W_vals >= W_nlo_min) & (W_vals <= W_nlo_max)
            try: F1_NLO_vals[m]  = F1_NLO(W_vals[m])     # NLO LT
            except Exception: pass
            try: F1_NLO_TMC_vals[m]  = F1_NLO_TMC(W_vals[m])         # TMC only
            except Exception: pass
            try: F1_NLO_TMC_HT_vals[m] = F1_NLO_TMC_HT(W_vals[m])       # TMC + HT
            except Exception: pass
            

        # Plot
        plt.figure(figsize=(8, 6))
        handles = [plt.Line2D([], [], color='white', label=f"Q² = {Q2:.3f} GeV²")]

        # LO 
        if np.isfinite(F1_LO_vals).any():
            good = np.isfinite(F1_LO_vals)
            h_lo, = plt.plot(W_vals[good], F1_LO_vals[good],
                             label=f"{pdf_set_lo}: LO + LT ", color="blue", ls="dotted", lw=2)
            handles.append(h_lo)

        # NLO 
        if np.isfinite(F1_NLO_vals).any():
            good = np.isfinite(F1_NLO_vals)
            h_naked, = plt.plot(W_vals[good], F1_NLO_vals[good],
                                label=f"{pdf_set_nlo}: NLO + LT ", color="magenta", ls="-", lw=2)
            handles.append(h_naked)

        #NLO TMC only
        if np.isfinite(F1_NLO_TMC_vals).any():
            good = np.isfinite(F1_NLO_TMC_vals)
            h_b, = plt.plot(W_vals[good], F1_NLO_TMC_vals[good],
                            label=f"{pdf_set_nlo}: NLO + LT + TMC ", color="green", ls="dashdot", lw=1.5)
            handles.append(h_b)
        


        # NLO TMC + HT
        if np.isfinite(F1_NLO_TMC_HT_vals).any():
            good = np.isfinite(F1_NLO_TMC_HT_vals)
            h_bht, = plt.plot(W_vals[good], F1_NLO_TMC_HT_vals[good],
                              label=f"{pdf_set_nlo}: NLO LT + TMC + HT  ", color="orange", ls="dashdot", lw=2)
            handles.append(h_bht)

        

        plt.xlabel("W (GeV)")
        plt.ylabel(r"$F_1(W; Q^2)$")
        plt.grid(True)
        plt.legend(handles=handles, loc="upper left", fontsize="small")

        q2_str = str(Q2).rstrip("0").rstrip(".")
        out_path = f"{out_dir}/compare_F1_Q2={q2_str}_Wmax={W_cutoff}.pdf"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print("Saved →", out_path)


def compare_F2(Q2_list, pdf_set_lo, pdf_set_nlo, num_points=400, W_cutoff=4.0):
    """
    For each Q² in Q2_list, plot:
        - F2 LO (CJ15) 
        - F2 NLO LT (CJ15)
        - F2 NLO TMC only (CJ15)
        - F2 NLO TMC + HT (CJ15)
    """
     
    out_dir = f"compare_F2_{pdf_set_nlo}"
    os.makedirs(out_dir, exist_ok=True)

    for Q2 in Q2_list:
        have_lo = have_nlo = False

        # LO
        try:
            _, F2_LO, W_lo_rng = get_lo_pdf_interpolators(Q2, pdf_set=pdf_set_lo)
            W_lo_min, W_lo_max = float(np.min(W_lo_rng)), float(np.max(W_lo_rng))
            have_lo = True
        except Exception:
            pass

        # NLO , NLO TMC, NLO TMC+HT
        try:
            _, _, _, _, F2_NLO, F2_NLO_TMC, F2_NLO_TMC_HT, _, _, _, W_nlo_rng = get_nlo_pdf_interpolators(Q2,pdf_set=pdf_set_nlo)
            W_nlo_min, W_nlo_max = float(np.min(W_nlo_rng)), float(np.max(W_nlo_rng))
            have_nlo = True
        except Exception:
            pass

        if not (have_lo or have_nlo):
            print(f"[WARNING] Q²={Q2}: no LO/NLO F2 available — skipping.")
            continue

        # W grid
        wmins = []
        if have_lo:  wmins.append(W_lo_min)
        if have_nlo: wmins.append(W_nlo_min)
        W_min_global = max(1.0, min(wmins)) if wmins else 1.0
        W_vals = np.linspace(W_min_global, W_cutoff, num_points)

        # Evaluate LO
        F2_LO_vals = np.full_like(W_vals, np.nan, dtype=float)
        if have_lo:
            m = (W_vals >= W_lo_min) & (W_vals <= W_lo_max)
            try:
                F2_LO_vals[m] = F2_LO(W_vals[m])
            except Exception:
                pass

        # Evaluate NLO (LT, TMC only, TMC+HT, HT-only)
        F2_NLO_vals = np.full_like(W_vals, np.nan, dtype=float)
        F2_NLO_TMC_vals = np.full_like(W_vals, np.nan, dtype=float)
        F2_NLO_TMC_HT_vals = np.full_like(W_vals, np.nan, dtype=float)
        if have_nlo:
            m = (W_vals >= W_nlo_min) & (W_vals <= W_nlo_max)
            try: F2_NLO_vals[m]  = F2_NLO(W_vals[m])
            except Exception: pass
            try: F2_NLO_TMC_vals[m]      = F2_NLO_TMC(W_vals[m])
            except Exception: pass
            try: F2_NLO_TMC_HT_vals[m]    = F2_NLO_TMC_HT(W_vals[m])
            except Exception: pass
           

        # Plot
        plt.figure(figsize=(8, 6))
        handles = [plt.Line2D([], [], color='white', label=f"Q² = {Q2:.3f} GeV²")]

        if np.isfinite(F2_LO_vals).any():
            good = np.isfinite(F2_LO_vals)
            h_lo, = plt.plot(W_vals[good], F2_LO_vals[good],
                             label=f"{pdf_set_lo}: LO + LT", color="blue", ls="dotted", lw=2)
            handles.append(h_lo)

        if np.isfinite(F2_NLO_vals).any():
            good = np.isfinite(F2_NLO_vals)
            h_naked, = plt.plot(W_vals[good], F2_NLO_vals[good],
                                label=f"{pdf_set_nlo}: NLO + LT", color="magenta", ls="solid", lw=2)
            handles.append(h_naked)

        if np.isfinite(F2_NLO_TMC_vals).any():
            good = np.isfinite(F2_NLO_TMC_vals)
            h_b, = plt.plot(W_vals[good], F2_NLO_TMC_vals[good],
                            label=f"{pdf_set_nlo}: NLO + LT + TMC (OPE)", color="green", ls="dashdot", lw=1)
            handles.append(h_b)

        if np.isfinite(F2_NLO_TMC_HT_vals).any():
            good = np.isfinite(F2_NLO_TMC_HT_vals)
            h_bht, = plt.plot(W_vals[good], F2_NLO_TMC_HT_vals[good],
                              label=f"{pdf_set_nlo}: NLO + LT + TMC (OPE) + HT", color="orange", ls="dashdot", lw=2)
            handles.append(h_bht)

        

        plt.xlabel("W (GeV)")
        plt.ylabel(r"$F_2(W; Q^2)$")
        plt.grid(True)
        plt.legend(handles=handles, loc="upper left", fontsize="small")

        q2_str = str(Q2).rstrip("0").rstrip(".")
        out_path = f"{out_dir}/compare_F2_Q2={q2_str}_Wmax={W_cutoff}.pdf"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print("Saved →", out_path)


def compare_xsecs(fixed_Q2, beam_energy, pdf_set_lo, pdf_set_nlo,
                   W_cutoff = 2.5,
                   interp_file="input_data/wempx.dat",
                   onepi_file="input_data/wemp-pi.dat",
                   num_points=200):
    
    out_dir = f"compare_xsecs_{pdf_set_nlo}"
    os.makedirs(out_dir, exist_ok=True)

    # ---------- Kinematics & constants ----------
    
    data_anl_model = np.loadtxt(interp_file)
    W_grid = np.unique(data_anl_model[:, 0])

    M  = 0.9385

    # lab kinematic W-limit
    w_kin_max = math.sqrt(max(M**2 + 2*M*beam_energy - fixed_Q2, 0.0))
    W_hi = min(W_cutoff, w_kin_max - 1e-6)
    W_lo = W_grid.min()
    W_vals = np.linspace(W_lo, W_hi, num_points)


    # Containers
    anl_full_xs, anl_onepi_xs = [], []
    pdf_lo_xs, pdf_lo_err = [], []
    pdf_nlo_xs, pdf_nlo_tmc_xs, pdf_nlo_tmc_ht_xs  = [], [], []
    pdf_nlo_tmc_ht_F2FL_xs = []  # new curve - calculation from F2 and FL

    # ---------- LO PDF ----------
    try:
        F1_LO, F2_LO, W_lo_range = get_lo_pdf_interpolators(fixed_Q2,pdf_set=pdf_set_lo)
        have_lo = True
        W_lo_min, W_lo_max = float(np.min(W_lo_range)), float(np.max(W_lo_range))
    except Exception:
        have_lo = False
        W_lo_min = W_lo_max = None

    # ---------- NLO, NLO+TMC, NLO+TMC+HT PDF (includes FL) ----------
    try:
        (F1_NLO, F1_NLO_TMC, F1_NLO_TMC_alt, F1_NLO_TMC_HT,
         F2_NLO, F2_NLO_TMC, F2_NLO_TMC_HT,
         FL_NLO, FL_NLO_TMC, FL_NLO_TMC_HT,
         W_nlo_range) = get_nlo_pdf_interpolators(fixed_Q2, pdf_set = pdf_set_nlo)

        have_nlo = True
        W_nlo_min, W_nlo_max = float(np.min(W_nlo_range)), float(np.max(W_nlo_range))
    except Exception:
        have_nlo = False
        W_nlo_min = W_nlo_max = None
        # dummies so loop runs
        F1_NLO = F1_NLO_TMC = F1_NLO_TMC_HT = F2_NLO = F2_NLO_TMC = F2_NLO_TMC_HT = FL_NLO = FL_NLO_TMC = FL_NLO_TMC_HT = lambda w: np.nan

    # ---------- Build curves on W grid ----------
    for w in W_vals:
        # ANL total xsec
        try:
            anl_full_xs.append(compute_cross_section_model(w, fixed_Q2, beam_energy, file_path=interp_file, verbose=False))
        except Exception:
            anl_full_xs.append(np.nan)

        # 1π ANL xsec
        try:
            anl_onepi_xs.append(compute_1pi_cross_section_model(w, fixed_Q2, beam_energy, file_path=onepi_file, verbose=False))
        except Exception:
            anl_onepi_xs.append(np.nan)

        # LO with band (limit to native W range)
        if have_lo and (W_lo_min <= w <= W_lo_max):
            try:
                val = compute_pdf_cross_sections(w, fixed_Q2, beam_energy, F1_interp=F1_LO, F2_interp=F2_LO)
                pdf_lo_xs.append(val)
            except Exception:
                pdf_lo_xs.append(np.nan)
        else:
            pdf_lo_xs.append(np.nan)

        # NLO sets (limit to native W range from the function)
        if have_nlo and (W_nlo_min <= w <= W_nlo_max):
            # NLO 
            try:
                pdf_nlo_xs.append(compute_pdf_cross_sections(w, fixed_Q2, beam_energy, F1_interp=F1_NLO, F2_interp=F2_NLO))
            except Exception:
                pdf_nlo_xs.append(np.nan)
            # NLO + TMC
            try:
                pdf_nlo_tmc_xs.append(compute_pdf_cross_sections(w, fixed_Q2, beam_energy, F1_interp=F1_NLO_TMC, F2_interp=F2_NLO_TMC))
            except Exception:
                pdf_nlo_tmc_xs.append(np.nan)
            # LT + TMC(OPE) + HT
            try:
                pdf_nlo_tmc_ht_xs.append(compute_pdf_cross_sections(w, fixed_Q2, beam_energy, F1_interp=F1_NLO_TMC_HT, F2_interp=F2_NLO_TMC_HT))
            except Exception:
                pdf_nlo_tmc_ht_xs.append(np.nan)

            # NEW: σ from (F2, FL) with HT+TMC set
            try:
                pdf_nlo_tmc_ht_F2FL_xs.append(compute_pdf_cross_sections_from_F2_FL(w, fixed_Q2, beam_energy, F2_NLO_TMC_HT, FL_NLO_TMC_HT))
            except Exception:
                pdf_nlo_tmc_ht_F2FL_xs.append(np.nan)
        else:
            pdf_nlo_xs.append(np.nan)
            pdf_nlo_tmc_xs.append(np.nan)
            pdf_nlo_tmc_ht_xs.append(np.nan)
            pdf_nlo_tmc_ht_F2FL_xs.append(np.nan)

    # ---------- Arrays ----------
    anl_full_xs           = np.asarray(anl_full_xs)
    anl_onepi_xs          = np.asarray(anl_onepi_xs)
    pdf_lo_xs             = np.asarray(pdf_lo_xs)        if have_lo  else np.array([])
    pdf_nlo_xs            = np.asarray(pdf_nlo_xs)    if have_nlo else np.array([])
    pdf_nlo_tmc_xs        = np.asarray(pdf_nlo_tmc_xs)       if have_nlo else np.array([])
    pdf_nlo_tmc_ht_xs      = np.asarray(pdf_nlo_tmc_ht_xs)    if have_nlo else np.array([])
    pdf_nlo_tmc_ht_F2FL_xs    = np.asarray(pdf_nlo_tmc_ht_F2FL_xs) if have_nlo else np.array([])

    # ---------- RGA data (unchanged) ----------
    have_rga = False
    try:
        rga_file = f"exp_data/InclusiveExpValera_Q2={fixed_Q2}.dat"
        if os.path.isfile(rga_file):
            rga = np.genfromtxt(rga_file, names=["W", "eps", "sigma", "error", "sys_error"],
                                delimiter="\t", skip_header=1)
            m = (rga["W"] >= W_lo) & (rga["W"] <= W_hi)
            W_rga = rga["W"][m]
            sigma_rga = rga["sigma"][m] * 1e-3
            err_rga = np.sqrt(rga["error"][m]**2 + rga["sys_error"][m]**2) * 1e-3
            have_rga = (W_rga.size > 0)
    except Exception:
        have_rga = False

    # ---------- Plot ----------
    plt.figure(figsize=(8, 6))
    handles = [plt.Line2D([], [], color='white',
               label=f"Q² = {fixed_Q2:.3f} GeV², E = {beam_energy} GeV")]
    
    #good_anl_full = np.isfinite(anl_full_xs)
    #h_model_full, = plt.plot(W_vals[good_anl_full], anl_full_xs[good_anl_full],
    #                     label="ANL-Osaka full", color="black", ls="solid", lw=2)
    #handles.append(h_model_full)
    #
    #good_anl_1pi = np.isfinite(anl_onepi_xs)
    #h_model_1pi, = plt.plot(W_vals[good_anl_1pi], anl_onepi_xs[good_anl_1pi],
    #                     label=r"ANL-Osaka 1$\pi$ contribution", color="black", ls="dashed", lw=2)
    #handles.append(h_model_1pi)

    if have_lo and np.isfinite(pdf_lo_xs).any():
        good_lo = np.isfinite(pdf_lo_xs)
        h_pdf_lo, = plt.plot(W_vals[good_lo], pdf_lo_xs[good_lo],
                             label=f"{pdf_set_lo}: LO + LT", color="blue", ls="dotted", lw=2)
        handles.append(h_pdf_lo)

    if np.isfinite(pdf_nlo_xs).any():
        good_nlo = np.isfinite(pdf_nlo_xs)
        h_pdf_nlo_lt, = plt.plot(W_vals[good_nlo], pdf_nlo_xs[good_nlo],
                                 label=f"{pdf_set_nlo}: NLO + LT", color="purple", ls="dashdot", lw=2)
        handles.append(h_pdf_nlo_lt)
        
    if np.isfinite(pdf_nlo_tmc_xs).any():
        good_nlo_tmc = np.isfinite(pdf_nlo_tmc_xs)
        h_pdf_nlo, = plt.plot(W_vals[good_nlo_tmc], pdf_nlo_tmc_xs[good_nlo_tmc],
                              label=f"{pdf_set_nlo}: NLO + LT + TMC(OPE)", color="green", ls="dashdot", lw=2)
        handles.append(h_pdf_nlo)

    if np.isfinite(pdf_nlo_tmc_ht_xs).any():
        good_nlo_tmc_ht = np.isfinite(pdf_nlo_tmc_ht_xs)
        h_pdf_nlo_ht, = plt.plot(W_vals[good_nlo_tmc_ht], pdf_nlo_tmc_ht_xs[good_nlo_tmc_ht],
                                 label=f"{pdf_set_nlo}: NLO + LT + TMC(OPE) + HT", color="orange", ls="dashed", lw=2)
        handles.append(h_pdf_nlo_ht)

    # NEW curve from (F2, FL)
    #if np.isfinite(pdf_nlo_tmc_ht_F2FL_xs).any():
    #    good_f2fl = np.isfinite(pdf_nlo_tmc_ht_F2FL_xs)
    #    h_f2fl, = plt.plot(W_vals[good_f2fl], pdf_nlo_tmc_ht_F2FL_xs[good_f2fl],
    #                       label=f"{pdf_set_nlo}: NLO + LT + TMC(OPE) + HT from F2, F_L", color="red", ls="solid", lw=2)
    #    handles.append(h_f2fl)

    if have_rga:
        h_rga = plt.errorbar(W_rga, sigma_rga, yerr=err_rga,
                             fmt="s", color="magenta", capsize=1, ms=2,
                             label="RGA data (V. Klimenko)")
        handles.append(h_rga)

    plt.xlabel("W (GeV)")
    plt.ylabel(r"$d \sigma / dW dQ^2$ ($\mathrm{\mu bn/GeV^3}$)")
    plt.grid(True)
    plt.xlim(1, W_cutoff + 0.1)
    if handles:
        plt.legend(handles=handles, loc="upper left", fontsize="small")

    fname = f"{out_dir}/compare_xsecs_Q2={fixed_Q2}_E={beam_energy}.pdf"
    plt.savefig(fname, dpi=300)
    plt.close()
    print("Saved →", fname)




#for q2 in [2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699]:
#    compare_xsecs(fixed_Q2=q2, beam_energy=10.6, pdf_set_lo="CT18LO", pdf_set_nlo="CT18NLO", W_cutoff=2.5)
#    compare_xsecs(fixed_Q2=q2, beam_energy=10.6, pdf_set_lo="CJ15lo", pdf_set_nlo="CJ15nlo", W_cutoff=2.5)
    
#compare_F2([1.025,2.025, 2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699, 15.0, 20.0], pdf_set_lo="CJ15lo", pdf_set_nlo="CJ15nlo", W_cutoff=1.8)
#compare_F2([1.025,2.025, 2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699, 15.0, 20.0],pdf_set="CJ15nlo", W_cutoff=2.5)
#compare_F2([1.025,2.025, 2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699, 15.0, 20.0],pdf_set="CJ15nlo", W_cutoff=5.0)
#compare_F2([1.025,2.025, 2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699, 15.0, 20.0],pdf_set="CJ15nlo", W_cutoff=10.0)
#compare_F2([1.025,2.025, 2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699, 15.0, 20.0],pdf_set="CJ15nlo", W_cutoff=20.0)
#compare_F2([1.025, 2.025, 2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699, 15.0, 20.0],pdf_set="CJ15nlo", W_cutoff=30.0)

compare_F1([2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699], pdf_set_lo="CJ15lo", pdf_set_nlo="CJ15nlo", W_cutoff=30.0)
compare_F1([2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699], pdf_set_lo="CT18LO", pdf_set_nlo="CT18NLO", W_cutoff=30.0)




