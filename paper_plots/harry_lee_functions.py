import numpy as np
import math
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os
import pandas as pd
from matplotlib.ticker import ScalarFormatter


from functions_pdf import get_lo_pdf_interpolators, get_nlo_pdf_interpolators, compute_pdf_cross_sections, compute_pdf_cross_sections_from_F2_FL, get_R_from_F1F2, calculate_moment_LO_pdf
from functions_anl_osaka import compute_cross_section_model, compute_1pi_cross_section_model, compute_2pi_cross_section_model, interpolate_structure_functions, sigma_LT_to_F2_AO_model, calculate_moment_AO_model
from functions_data import calc_trunc_moment_data, F2_from_xsect_data, x_of_W, estimate_bin_size_err_data


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
                   W_cutoff ,
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
    if fixed_Q2 > 3.0:
        have_AO = have_AO_1pi = False
    else:
        have_AO = have_AO_1pi = True
    
    pdf_lo_xs = []
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

        # LO 
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

    # ---------- RGA data  ----------
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
        
       # ---------- AO extended model ----------
    have_AO_ext = False
    try:
        AO_ext_file = f"tables_from_Yannick/fine_binning/AO/Wdist_Q2_{fixed_Q2}_GLOBAL_LT.dat"
        if os.path.isfile(AO_ext_file):
            AO_ext = np.genfromtxt(
                AO_ext_file,
                names=["W", "sigma"],   # second column is the cross section
                delimiter=None,         # whitespace-separated
                skip_header=1           # set to 1 only if there's a header line
            )
            m = (AO_ext["W"] >= W_lo) & (AO_ext["W"] <= W_hi)
            W_AO_ext     = AO_ext["W"][m]
            sigma_AO_ext = AO_ext["sigma"][m]
            have_AO_ext  = (W_AO_ext.size > 0)
    except Exception:
        have_AO_ext = False
        print("No AO extended model data found:")
  
    # ---------- Plot ----------
    plt.figure(figsize=(8, 6))
    handles = [plt.Line2D([], [], color='white',
               label=f"Q² = {fixed_Q2:.3f} GeV², E = {beam_energy} GeV")]
    
    have_AO = False # disabled for now
    if have_AO:
        good_anl_full = np.isfinite(anl_full_xs) & (W_vals <= 2.0)
        h_model_full, = plt.plot(W_vals[good_anl_full], anl_full_xs[good_anl_full],
                             label="ANL-Osaka full", color="red", ls="solid", lw=2)
        handles.append(h_model_full)
        
    if have_AO_ext:
            h_AO_ext = plt.errorbar(W_AO_ext, sigma_AO_ext,
                                 color="red", ls="solid", lw=2,
                                 label="ANL-Osaka full (extended)")
            handles.append(h_AO_ext)
    #
    #if have_AO_1pi:
    #    good_anl_1pi = np.isfinite(anl_onepi_xs)
    #    h_model_1pi, = plt.plot(W_vals[good_anl_1pi], anl_onepi_xs[good_anl_1pi],
    #                         label=r"ANL-Osaka 1$\pi$ contribution", color="black", ls="dashed", lw=2)
    #    handles.append(h_model_1pi)

    if have_lo and np.isfinite(pdf_lo_xs).any():
        good_lo = np.isfinite(pdf_lo_xs)
        h_pdf_lo, = plt.plot(W_vals[good_lo], pdf_lo_xs[good_lo],
                             label=f"{pdf_set_lo}: LO + LT", color="blue", ls="dotted", lw=2)
        handles.append(h_pdf_lo)
#
    if np.isfinite(pdf_nlo_xs).any():
        good_nlo = np.isfinite(pdf_nlo_xs)
        h_pdf_nlo_lt, = plt.plot(W_vals[good_nlo], pdf_nlo_xs[good_nlo],
                                 label=f"{pdf_set_nlo}: NLO + LT", color="green", ls="dashed", lw=2)
        handles.append(h_pdf_nlo_lt)
        
    #if np.isfinite(pdf_nlo_tmc_xs).any():
    #    good_nlo_tmc = np.isfinite(pdf_nlo_tmc_xs)
    #    h_pdf_nlo_tmc, = plt.plot(W_vals[good_nlo_tmc], pdf_nlo_tmc_xs[good_nlo_tmc],
    #                          label=f"{pdf_set_nlo}: NLO + LT + TMC(OPE)", color="green", ls="dashdot", lw=2)
    #    handles.append(h_pdf_nlo_tmc)

    if np.isfinite(pdf_nlo_tmc_ht_xs).any():
        good_nlo_tmc_ht = np.isfinite(pdf_nlo_tmc_ht_xs)
        h_pdf_nlo_ht, = plt.plot(W_vals[good_nlo_tmc_ht], pdf_nlo_tmc_ht_xs[good_nlo_tmc_ht],
                                 label=f"{pdf_set_nlo}: NLO + LT + TMC(OPE) + HT", color="orange", ls="solid", lw=2)
        handles.append(h_pdf_nlo_ht)

    # NEW curve from (F2, FL)
    #if np.isfinite(pdf_nlo_tmc_ht_F2FL_xs).any():
    #    good_f2fl = np.isfinite(pdf_nlo_tmc_ht_F2FL_xs)
    #    h_f2fl, = plt.plot(W_vals[good_f2fl], pdf_nlo_tmc_ht_F2FL_xs[good_f2fl],
    #                       label=f"{pdf_set_nlo}: NLO + LT + TMC(OPE) + HT from F2, F_L", color="red", ls="solid", lw=2)
    #    handles.append(h_f2fl)

    if have_rga:
        h_rga = plt.errorbar(W_rga, sigma_rga, yerr=err_rga,
                             fmt="s", color="black", capsize=1, ms=2,
                             label="RGA data (V. Klimenko)")
        handles.append(h_rga)
        
    

    plt.xlabel("W (GeV)")
    plt.ylabel(r"$d \sigma / dW dQ^2$ ($\mathrm{\mu bn/GeV^3}$)")
    plt.title(f"Comparison of cross sections at Q²={fixed_Q2} GeV², E={beam_energy} GeV")
    plt.grid(True)
    if W_cutoff == 2.5:
        plt.xlim(1, W_cutoff+0.05)
    elif W_cutoff == 5.0:
        plt.xlim(1, 4.5)
    
    if handles:
        plt.legend(handles=handles, loc="lower right", fontsize="small")
        
    
    ax = plt.gca()  # get current axes

    ax.text(
        0.02, 0.98,                      # (x, y) in axes coordinates
        f"{pdf_set_nlo}",               # the text
        transform=ax.transAxes,  
        fontsize=20,         
        fontweight="bold",
        ha="left", va="top",
    )
    
    fname = f"{out_dir}/compare_xsecs_Q2={fixed_Q2}_E={beam_energy}_W_max={W_cutoff}.pdf"
    plt.savefig(fname, dpi=300)
    plt.close()
    print("Saved →", fname)



def plot_sigmaLT_and_R_from_F1F2(
        fixed_Q2,
        pdf_set_nlo,
        flag_nlo,
        beam_energy=10.6,
        W_cutoff=2.5,
        num_points=200
        ):
    """
    Plot sigma_T, sigma_L and R = sigma_L/sigma_T as functions of W
    using F1, F2 interpolators and your get_R_from_F1F2() helper.

    Args:
        fixed_Q2   (float): Q^2 in GeV^2
        F1_interp  (callable): F1(W) interpolator
        F2_interp  (callable): F2(W) interpolator
        W_min      (float): minimum W (GeV)
        W_max      (float): maximum W (GeV)
        num_points (int): number of W points
        pdf_label  (str): label for legend / filename (e.g. "CT18NLO")
        out_dir    (str): output directory for the plot
    """

    out_dir   = f"sigmaLT_R_plots_{pdf_set_nlo}_{flag_nlo}"
    table_dir = f"sigmaLT_R_tables_{pdf_set_nlo}_{flag_nlo}"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(table_dir, exist_ok=True)

   

    # Containers
    sigma_L_list = []
    sigma_T_list = []
    R_list       = []
    xsect_list   = []
    
    try:
        F1_NLO, _, _, F1_NLO_TMC_HT, F2_NLO, _, F2_NLO_TMC_HT, _, _, _, W_nlo_rng = get_nlo_pdf_interpolators(fixed_Q2,pdf_set=pdf_set_nlo)
        W_nlo_min, W_nlo_max = float(np.min(W_nlo_rng)), float(np.max(W_nlo_rng))
    except Exception:
        pass
    
     # W grid
    W_vals = np.linspace(W_nlo_min, W_cutoff, num_points)

    if flag_nlo == "NLO_only":
        F1 = F1_NLO
        F2 = F2_NLO
    elif flag_nlo == "NLO_TMC_HT":
        F1 = F1_NLO_TMC_HT
        F2 = F2_NLO_TMC_HT
        
    for W in W_vals:
        R, sigma_L, sigma_T, xsect, _, _ = get_R_from_F1F2(W, fixed_Q2, beam_energy, F1, F2)
        sigma_L_list.append(sigma_L)
        sigma_T_list.append(sigma_T)
        R_list.append(R)
        xsect_list.append(xsect)

    sigma_L = np.asarray(sigma_L_list)
    sigma_T = np.asarray(sigma_T_list)
    R       = np.asarray(R_list)
    xsect   = np.asarray(xsect_list)

     # ----- Figure with three panels -----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
    ax_LT, ax_R, ax_xsect = axes

    # 1) Left: sigma_T and sigma_L
    mask_LT = np.isfinite(sigma_L) & np.isfinite(sigma_T)
    ax_LT.plot(W_vals[mask_LT], sigma_T[mask_LT], label=r"$\sigma_T$", color="orange")
    ax_LT.plot(W_vals[mask_LT], sigma_L[mask_LT], label=r"$\sigma_L$", color="blue")

    ax_LT.set_title("$\sigma_L, \sigma_T$")
    ax_LT.set_xlabel(r"$W\ (\mathrm{GeV})$")
    ax_LT.set_ylabel(r"$\sigma_{L,T}\ (\mathrm{GeV^{-2}})$")
    ax_LT.grid(True)
    ax_LT.legend()

    # Small label in the corner
    pdf_label = pdf_set_nlo
    txt = f"$Q^2 = {fixed_Q2} GeV^2$,\n E = {beam_energy} GeV,\n {flag_nlo}, \n"
    if pdf_label:
        txt += f"{pdf_label}"
    ax_LT.text(
    0.5, 0.95, txt,              # x = center, y = near top
    transform=ax_LT.transAxes,   # axes coordinates
    ha="center",                 # horizontal center
    va="top",                    # text sits just below y=0.98
    fontsize=9,
    fontweight="bold",
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
)

    # 2) Middle: R = sigma_L / sigma_T
    mask_R = np.isfinite(R)
    ax_R.set_title("R coefficient")
    ax_R.plot(W_vals[mask_R], R[mask_R], label=r"$R = \sigma_L / \sigma_T$")
    ax_R.set_xlabel(r"$W\ (\mathrm{GeV})$")
    ax_R.set_ylabel(r"$R$")
    ax_R.grid(True)
    ax_R.legend()

    # 3) Right: d^2σ / dW dQ^2
    mask_x = np.isfinite(xsect)
    ax_xsect.plot(W_vals[mask_x], xsect[mask_x],
                  label=r"$\frac{d^2\sigma}{dW\,dQ^2}$")
    ax_xsect.set_xlabel(r"$W\ (\mathrm{GeV})$")
    ax_xsect.set_ylabel(r"$\mathrm{d}^2\sigma / (\mathrm{d}W\,\mathrm{d}Q^2)\ "
                         r"(\mathrm{\mu b/GeV^3})$")
    ax_xsect.grid(True)
    ax_xsect.set_title("Differential cross section from $\sigma_L, \sigma_T$")
    ax_xsect.legend()

    fig.tight_layout()

    # Save
    fname = (f"{out_dir}/sigmaLT_R_xsec_Q2={fixed_Q2}_"
             f"E={beam_energy}_{pdf_label}_{flag_nlo}.pdf")
    plt.savefig(fname, dpi=300)
    plt.close(fig)
    print("Saved →", fname)
    
    # ----- Save table: Q2, W, sigma_L, sigma_T, R -----
    table = np.column_stack([
        np.full_like(W_vals, fixed_Q2, dtype=float),
        W_vals,
        sigma_L,
        sigma_T,
        R
    ])
    fname_table = (f"{table_dir}/sigmaLT_R_Q2={fixed_Q2}_"
                   f"{pdf_label}_{flag_nlo}.dat")
    header = "Q2\tW\tSigma_L(GeV^-2)\tSigma_T(GeV^-2)\tR"
    np.savetxt(fname_table, table,
               fmt="%.6e",
               delimiter="\t",
               header=header)
    print("Saved table →", fname_table)
    

def compare_W1W2_pdf_vs_AO(
        fixed_Q2,
        pdf_set_nlo,
        flag_nlo,
        beam_energy=10.6,
        W_cutoff=2.5,
        num_points=200,
        anl_file="input_data/wempx.dat"
    ):
    """
    Compare W1, W2 reconstructed from σ_T and R (PDF-based F1,F2)
    with W1, W2 from the ANL-Osaka model (via interpolate_structure_functions).

    Two-panel plot:
      left:  W1_pdf vs W and W1_AO vs W
      right: W2_pdf vs W and W2_AO vs W
    """

    out_dir = f"compare_W1W2_{pdf_set_nlo}_{flag_nlo}"
    os.makedirs(out_dir, exist_ok=True)

    # ---- Get W-range from ANL-Osaka table ----
    data = np.loadtxt(anl_file)
    W_anl = data[:, 0]
    Q2_anl = data[:, 1]

    W_anl_min, W_anl_max = W_anl.min(), W_anl.max()
    Q2_anl_min, Q2_anl_max = Q2_anl.min(), Q2_anl.max()

    if not (Q2_anl_min <= fixed_Q2 <= Q2_anl_max):
        print(f"[compare_W1W2] Q2={fixed_Q2} outside ANL range "
              f"[{Q2_anl_min}, {Q2_anl_max}]")
        return

    # ---- PDF-based F1,F2 interpolators ----
    (F1_NLO, F1_NLO_TMC, F1_NLO_TMC_alt, F1_NLO_TMC_HT,
     F2_NLO, F2_NLO_TMC, F2_NLO_TMC_HT,
     FL_NLO, FL_NLO_TMC, FL_NLO_TMC_HT,
     W_nlo_rng) = get_nlo_pdf_interpolators(fixed_Q2, pdf_set=pdf_set_nlo)

    W_nlo_min, W_nlo_max = float(np.min(W_nlo_rng)), float(np.max(W_nlo_rng))

    if flag_nlo == "NLO_only":
        F1 = F1_NLO
        F2 = F2_NLO
    elif flag_nlo == "NLO_TMC_HT":
        F1 = F1_NLO_TMC_HT
        F2 = F2_NLO_TMC_HT
    else:
        print(f"[compare_W1W2] Unknown flag_nlo='{flag_nlo}'")
        return

    # ---- Common W range ----
    W_lo = max(W_anl_min, W_nlo_min)
    W_hi = min(W_anl_max, W_nlo_max, W_cutoff)
    if W_hi <= W_lo:
        print("[compare_W1W2] No overlapping W-range between ANL and PDF")
        return

    W_vals = np.linspace(W_lo, W_hi, num_points)

    # ---- Arrays ----
    W1_pdf_list, W2_pdf_list = [], []
    W1_AO_list,  W2_AO_list  = [], []

    for W in W_vals:
        # PDF: reconstruct W1, W2 from σ_T and R
        R, sigma_L, sigma_T, xsec, W1_pdf, W2_pdf = get_R_from_F1F2(
            W, fixed_Q2, beam_energy, F1, F2
        )
        W1_pdf_list.append(W1_pdf)
        W2_pdf_list.append(W2_pdf)

        # ANL-Osaka W1, W2 via your interpolator
        try:
            W1_AO, W2_AO = interpolate_structure_functions(
                anl_file, target_W=W, target_Q2=fixed_Q2
            )
        except ValueError as e:
            # This should not happen if W range was chosen correctly,
            # but just in case, append NaNs.
            W1_AO, W2_AO = np.nan, np.nan
        W1_AO_list.append(W1_AO)
        W2_AO_list.append(W2_AO)

    W1_pdf = np.asarray(W1_pdf_list)
    W2_pdf = np.asarray(W2_pdf_list)
    W1_AO  = np.asarray(W1_AO_list)
    W2_AO  = np.asarray(W2_AO_list)

    # ---- Plot ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    ax_W1, ax_W2 = axes
    
    txt = (rf"$Q^2 = {fixed_Q2:.3f}\,\mathrm{{GeV}}^2$ " rf"$E = {beam_energy:.1f}\,\mathrm{{GeV}}$ " rf"{pdf_set_nlo }, {flag_nlo}")   

    # Left: W1
    mask1 = np.isfinite(W1_pdf) & np.isfinite(W1_AO)
    ax_W1.plot(W_vals[mask1], W1_AO[mask1],
               label="ANL-Osaka $W_1$", color="black")
    ax_W1.plot(W_vals[mask1], W1_pdf[mask1],
               label=f"{pdf_set_nlo} ({flag_nlo}) $W_1$", linestyle="--")
    ax_W1.set_xlabel(r"$W\ (\mathrm{GeV})$")
    ax_W1.set_ylabel(r"$W_1(Q^2,W)$")
    ax_W1.set_title(txt)
    ax_W1.grid(True)
    ax_W1.legend()

    # Right: W2
    mask2 = np.isfinite(W2_pdf) & np.isfinite(W2_AO)
    ax_W2.plot(W_vals[mask2], W2_AO[mask2],
               label="ANL-Osaka $W_2$", color="black")
    ax_W2.plot(W_vals[mask2], W2_pdf[mask2],
               label=f"{pdf_set_nlo} ({flag_nlo}) $W_2$", linestyle="--")
    ax_W2.set_xlabel(r"$W\ (\mathrm{GeV})$")
    ax_W2.set_ylabel(r"$W_2(Q^2,W)$")
    ax_W2.grid(True)
    ax_W2.legend()



    fig.tight_layout()

    fname = (f"{out_dir}/compare_W1W2_Q2={fixed_Q2}_"
             f"E={beam_energy}_{pdf_set_nlo}_{flag_nlo}.png")
    plt.savefig(fname, dpi=300)
    plt.close(fig)
    print("Saved →", fname)
    
    
    
    

def plot_F2_from_data_AO_PDF(Q2_value, W_cutoff = 2.0, show_lines = False, vs_what = "w", pdf_set_lo = "CJ15lo", pdf_set_nlo="CJ15nlo"):
    
    # ------------------------------------ data (exp) -------------------------------------

    df_data = F2_from_xsect_data(Q2_value,R_source="AO")
    W = df_data["W"].to_numpy()
    x = df_data["x"].to_numpy()
    F2 = df_data["F2"].to_numpy()
    F2_err = df_data["F2_err"].to_numpy()
    
    mask_data = W <= W_cutoff
    W = W[mask_data]
    x = x[mask_data]
    F2 = F2[mask_data]
    F2_err = F2_err[mask_data]
    
    
    #------------------------------------PDF predictions----------------------------------------
    have_lo = False
    try:
        _, F2_LO, W_lo_rng = get_lo_pdf_interpolators(Q2_value,pdf_set=pdf_set_lo)
        W_lo_min = float(np.min(W_lo_rng))
        have_lo = True
    except Exception:
        pass
    have_nlo = False
    try:
          _, _, _, _, F2_NLO, F2_NLO_TMC, F2_NLO_TMC_HT, _, _, _, W_nlo_rng = get_nlo_pdf_interpolators(Q2_value,pdf_set=pdf_set_nlo)
          W_nlo_min = float(np.min(W_nlo_rng))
          have_nlo = True
    except Exception:
          pass
      
    # W grid
    wmins = []
    if have_nlo: wmins.append(W_nlo_min)
    W_min_global = max(1.0, min(wmins)) if wmins else 1.0
    W_max_global = W_cutoff
    W_vals = np.linspace(W_min_global, W_max_global, 400)
    # Evaluate LO
    F2_LO_vals = np.full_like(W_vals, np.nan, dtype=float)
     # Evaluate NLO (LT, TMC only, TMC+HT, HT-only)
    F2_NLO_vals = np.full_like(W_vals, np.nan, dtype=float)
    F2_NLO_TMC_vals = np.full_like(W_vals, np.nan, dtype=float)
    F2_NLO_TMC_HT_vals = np.full_like(W_vals, np.nan, dtype=float)
    m = (W_vals >= W_nlo_min) & (W_vals <= W_max_global)
    if have_lo:
        try: F2_LO_vals[m]  = F2_LO(W_vals[m])
        except Exception: pass
    if have_nlo:
        try: F2_NLO_vals[m]  = F2_NLO(W_vals[m])
        except Exception: pass
        try: F2_NLO_TMC_vals[m] = F2_NLO_TMC(W_vals[m])
        except Exception: pass
        try: F2_NLO_TMC_HT_vals[m] = F2_NLO_TMC_HT(W_vals[m])
        except Exception: pass
    # ------------------------------------AO model prediction----------------------------------------
    have_ao = False
    F2_AO_vals = np.full_like(W_vals, np.nan, dtype=float)
    try:
        # Native AO grid -> interpolate to our W_vals without extrapolation
        W_AO, F2_AO_native = sigma_LT_to_F2_AO_model(Q2_value)  # returns native (W, F2)
        F2_AO_vals = np.interp(W_vals, W_AO, F2_AO_native, left=np.nan, right=np.nan)
        F2_AO_vals = F2_AO_vals[m]
        have_ao = True
    except Exception:
        pass
    
    # ---------------------ranges for future integration ()----------------------------------------
    W_min = 1.15 # now corresponds to data range
    Wmax1 = 1.35 # end of 1st resonance region
    Wmin2 = Wmax1+0.004 # CRUTCH for visibility 
    Wmax2 = 1.6 # end of 2nd resonance region
    Wmin3 = Wmax2+0.004 # CRUTCH for visibility
    Wmax3 = 2.0 # end of 3rd resonance region
    W_max = 2.5 
    if Q2_value == 9.699:
      W_max = 2.25
    xmax = x_of_W(W_min, Q2_value)

    x1 = x_of_W(Wmax1, Q2_value) # W = 1.35 GeV
    xmin2 = x_of_W(Wmin2, Q2_value) # W = 1.35 GeV
    x2 = x_of_W(Wmax2, Q2_value) # W = 1.6 GeV
    xmin3 = x_of_W(Wmin3, Q2_value) # W = 1.6 GeV
    x3 = x_of_W(Wmax3, Q2_value) # W =2.0 GeV
    xmin = x_of_W(W_max, Q2_value) # W = 2.5 GeV (2.25 GeV at highest Q2)
        # -----------------------------
    # Plot (exp points with errors and PDF-based predictions)
    # -----------------------------
    plt.figure(figsize=(7, 5))
    ax = plt.gca()
    vs = vs_what.lower().strip()
    if vs in ["w", "W"]:
        order = np.argsort(W)
        x_axis = W[order]
        xlab = r"$W$ [GeV]"
        tag = "W"
    elif vs in ["x", "X"]:
        order = np.argsort(x)  # increasing x
        x_axis = x[order]
        xlab = r"$x_{B}$"
        tag = "x"
    else:
        raise ValueError(f"vs_what must be 'w' or 'x' (got '{vs_what}')")
    F2_plot = F2[order]
    F2e_plot = F2_err[order]
    plt.errorbar(x_axis, F2_plot, yerr=F2e_plot, label="RGA data (V.Klimenko)", color="black", fmt="o", linestyle="none", markersize=2.5, capsize=2)
    if tag == "W" and have_lo:
        good = np.isfinite(F2_LO_vals)
        if good.any():
            plt.plot(W_vals[good], F2_LO_vals[good], label=f"{pdf_set_lo}: LO + LT", color="blue", ls="dotted", lw=1.3)
    if tag == "W" and have_nlo:
        if np.isfinite(F2_NLO_vals).any():
            good = np.isfinite(F2_NLO_vals)
            h_naked, = plt.plot(W_vals[good], F2_NLO_vals[good], label=f"{pdf_set_nlo}: NLO + LT", color="green", ls="dashed", lw=1.3)
        if np.isfinite(F2_NLO_TMC_HT_vals).any():
            good = np.isfinite(F2_NLO_TMC_HT_vals)
            h_bht, = plt.plot(W_vals[good], F2_NLO_TMC_HT_vals[good], label=f"{pdf_set_nlo}: NLO + LT + TMC (OPE) + HT", color="red", ls="-.", lw=1.3)
    if tag == "W" and have_ao:
        good = np.isfinite(F2_AO_vals)
        if good.any():
            plt.plot(W_vals[good], F2_AO_vals[good], label="AO model extended", color="black", ls="solid", lw=1.3)
     # --- PDF curves on x-axis ---
    if tag == "x" and have_lo:
       x_pdf = x_of_W(W_vals, Q2_value)
       good = np.isfinite(F2_LO_vals) & np.isfinite(x_pdf)
       if good.any():
           p = np.argsort(x_pdf[good])
           plt.plot(x_pdf[good][p], F2_LO_vals[good][p], label=f"{pdf_set_lo}: LO + LT", color="blue", ls="dotted", lw=1.3)
    if tag == "x" and have_nlo:
        x_pdf = x_of_W(W_vals, Q2_value)
        # NLO + LT
        good = np.isfinite(F2_NLO_vals) & np.isfinite(x_pdf)
        if good.any():
            p = np.argsort(x_pdf[good])  # increasing x
            plt.plot(x_pdf[good][p], F2_NLO_vals[good][p], label=f"{pdf_set_nlo}: NLO + LT", color="green", ls="dashed", lw=1.3)
        # NLO + LT + TMC + HT
        good = np.isfinite(F2_NLO_TMC_HT_vals) & np.isfinite(x_pdf)
        if good.any():
            p = np.argsort(x_pdf[good])
            plt.plot(x_pdf[good][p], F2_NLO_TMC_HT_vals[good][p], label=f"{pdf_set_nlo}: NLO + LT + TMC (OPE) + HT",color="red", ls="-.", lw=1.3)
    if tag == "x" and have_ao:
        x_ao = x_of_W(W_vals, Q2_value)
        good = np.isfinite(F2_AO_vals) & np.isfinite(x_ao)
        if good.any():
            p = np.argsort(x_ao[good])
            plt.plot(x_ao[good][p], F2_AO_vals[good][p], label="AO model extended", color="black", ls="solid", lw=1.3)
    # --------------------- Vertical lines and labels for different W (or x) regions --------------------------
    if show_lines:
        y_top = 0.95
        def label_between(x_left, x_right, txt, color):
            x_mid = 0.5 * (x_left + x_right)
            ax.text( x_mid, y_top, txt, transform=ax.get_xaxis_transform(),  # x in data coords, y in axes coords
            ha="center", va="top", color=color, fontsize=7)
        if tag == "W":
            ax.axvline(W_min, linestyle="--", linewidth=1, color = "red")
            ax.axvline(Wmax1, linestyle="--", linewidth=1, color = "red")
            ax.axvline(Wmin2, linestyle="--", linewidth=1, color = "green")
            ax.axvline(Wmax2, linestyle="--", linewidth=1, color = "green")
            ax.axvline(Wmin3, linestyle="--", linewidth=1, color = "blue")
            ax.axvline(Wmax3, linestyle="--", linewidth=1, color = "blue")
            ax.axvline(W_max, linestyle="--", linewidth=1, color = "black")
            label_between(W_min,  Wmax1, "1st region",  "red")
            label_between(Wmin2,  Wmax2, "2nd region",  "green")
            label_between(Wmin3,  Wmax3, "3rd region",  "blue")
            label_between(Wmax3,  W_max, "Tail region",  "black")
            plt.legend(frameon=False, fontsize=7, loc="lower right")
        if tag == "x":
            ax.axvline(xmax, linestyle="--", linewidth=1, color = "red")
            ax.axvline(x1, linestyle="--", linewidth=1, color = "red")
            ax.axvline(xmin2, linestyle="--", linewidth=1, color = "green")
            ax.axvline(x2, linestyle="--", linewidth=1, color = "green")
            ax.axvline(xmin3, linestyle="--", linewidth=1, color = "blue")
            ax.axvline(x3, linestyle="--", linewidth=1, color = "blue")
            ax.axvline(xmin, linestyle="--", linewidth=1, color = "black")
            label_between(xmax,  x1,   "1st region", "red")
            label_between(xmin2, x2,   "2nd region", "green")
            label_between(xmin3, x3,   "3rd region", "blue")
            label_between(x3, xmin,   "Tail region", "black")
            plt.legend(frameon=False, fontsize=7, loc="lower left")
    plt.xlabel(xlab)
    plt.ylabel(r"$F_2 \; (GeV^{-2})$")
    plt.title(rf"$F_2$ structure function; $Q^2 = {Q2_value}$ GeV$^2$")
    plt.grid(True)
    if not show_lines:
        plt.legend(frameon=False, fontsize=10, loc="best")
    plt.tight_layout()
    out_dir = "F2_from_data_plots"
    os.makedirs(out_dir, exist_ok=True)
    out_pdf = os.path.join(out_dir, f"F2_Q2={Q2_value}_vs_{tag}_show_lines-{show_lines}.pdf")
    plt.savefig(out_pdf, dpi=200)
    plt.close()
    print(f"Saved → {out_pdf}")
    
    

def plot_F2_from_data_diff_R_sources(Q2_value, vs_what = "w"):
    
    # ------------------------------------ data (exp) -------------------------------------
    df_data_AO = F2_from_xsect_data(Q2_value,R_source="AO")
    df_data_Astrid = F2_from_xsect_data(Q2_value,R_source="Astrid")
    df_data_CJ15 = F2_from_xsect_data(Q2_value,R_source="CJ15")
    W = df_data_AO["W"].to_numpy()
    x = df_data_AO["x"].to_numpy()
    
    F2_AO = df_data_AO["F2"].to_numpy()
    F2_err_AO = df_data_AO["F2_err"].to_numpy()

    F2_Astrid = df_data_Astrid["F2"].to_numpy()
    F2_err_Astrid = df_data_Astrid["F2_err"].to_numpy()

    F2_CJ15 = df_data_CJ15["F2"].to_numpy()
    F2_err_CJ15 = df_data_CJ15["F2_err"].to_numpy()
    # ----------------------------------------------Plotting-----------------------------------------------

    plt.figure(figsize=(7, 5))
    ax = plt.gca()
    vs = vs_what.lower().strip()
    if vs in ["w", "W"]:
        order = np.argsort(W)
        x_axis = W[order]
        xlab = r"$W$ [GeV]"
        tag = "W"
    elif vs in ["x", "X"]:
        order = np.argsort(x)  # increasing x
        x_axis = x[order]
        xlab = r"$x_{B}$"
        tag = "x"
    else:
        raise ValueError(f"vs_what must be 'w' or 'x' (got '{vs_what}')")

    plt.scatter(x_axis, F2_AO[order], label=r"RGA data with $R_{LT}$ from AO", color="black", s=5)
    plt.scatter(x_axis, F2_Astrid[order],  label=r"RGA data with $R_{LT}$ from Astrid", color="green", s=5)
    plt.scatter(x_axis, F2_CJ15[order],  label=r"RGA data with $R_{LT}$ from CJ15", color="red", s=5)

    plt.xlabel(xlab)
    plt.ylabel(r"$F_2 \; (GeV^{-2})$")
    plt.title(rf"$F_2$ structure function; $Q^2 = {Q2_value}$ GeV$^2$")
    plt.grid(True)
    plt.legend(frameon=False, fontsize=10, loc="best")
    plt.tight_layout()
    out_dir = "F2_diff_R_sources_plots"
    os.makedirs(out_dir, exist_ok=True)
    out_pdf = os.path.join(out_dir, f"F2_Q2={Q2_value}_vs_{tag}.pdf")
    plt.savefig(out_pdf, dpi=200)
    plt.close()
    print(f"Saved → {out_pdf}")



def plot_M2_truncated_vs_Q2(pdf_set, error_mode="correlated", in_dir="../getF1F2/Output/truncated_moments", out_dir="Moment_vs_Q2"):
    in_path = os.path.join(in_dir, f"M2_{pdf_set}.txt")
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"Cannot find input file: {in_path}")

    os.makedirs(out_dir, exist_ok=True)

    data = np.loadtxt(in_path)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 13:
        raise ValueError(f"Expected >= 13 columns in {in_path}, got {data.shape[1]}.")

    Q2 = data[:, 0].astype(float)

    brady = {
        "1st":  data[:, 1],
        "2nd":  data[:, 2],
        "3rd":  data[:, 3],
        "tail": data[:, 4],
        "all":  data[:, 5],
        "part": data[:, 6]
    }
    naked = {
        "1st":  data[:, 7],
        "2nd":  data[:, 8],
        "3rd":  data[:, 9],
        "tail": data[:, 10],
        "all":  data[:, 11],
        "part": data[:, 12]
    }

    # Data
    regions = ["1st", "2nd", "3rd", "tail", "all", "part"]

    # Sort by Q2 just in case
    idx = np.argsort(Q2)
    Q2s = Q2[idx]
    for k in brady:
        brady[k] = brady[k][idx]
        naked[k] = naked[k][idx]
        
    # ---- experimental moments from data ----
    exp_m2 = {r: np.full_like(Q2s, np.nan, dtype=float) for r in regions}
    exp_e2 = {r: np.full_like(Q2s, np.nan, dtype=float) for r in regions}

    for i, q2v in enumerate(Q2s):
        for r in regions:
            out = calc_trunc_moment_data(q2v, r,R_source="AO", n=2, error_mode=error_mode).iloc[0]   # n=2 -> M2
            m  = float(out["moment"])
            de = float(out["error"])

            # if your calc returns 0 when no overlap, avoid plotting fake zeros
            if (m == 0.0 and de == 0.0):
                continue

            exp_m2[r][i] = m
            exp_e2[r][i] = de
    # -------------------------  AO model moments -----------------------------
    ao_m2 = {r: np.full_like(Q2s, np.nan, dtype=float) for r in regions}

    for i, q2v in enumerate(Q2s):
        for r in regions:
            out_ao = calculate_moment_AO_model(
                q2v, r, n=2,
                E_beam=10.6,
                in_dir="tables_from_Yannick/fine_binning/AO",
                convert_ub_to_GeV2=True,
                divide_by_Gamma=True
            ).iloc[0]

            m_ao = float(out_ao["moment"])
            if m_ao == 0.0:
                continue
            ao_m2[r][i] = m_ao
    # ---------------------------LO PDF moments ------------------------------
    lo_pdf_m2 = {r: np.full_like(Q2s, np.nan, dtype=float) for r in regions}
    for i, q2v in enumerate(Q2s):
        for r in regions:
            out_lo = calculate_moment_LO_pdf(
                q2v, r
            ).iloc[0]
            m_lo = float(out_lo["moment"])
            if m_lo == 0.0:
                continue
            lo_pdf_m2[r][i] = m_lo

    # -------------------- ###  Wmax info for title --------------------
    Q2_special = 9.699
    Wmax_default = 2.5
    Wmax_special = 2.25

    has_special = np.any(np.isclose(Q2s, Q2_special, rtol=0, atol=1e-6))
    if has_special:
        title_suffix = (rf"$W_\max={Wmax_default}\,\mathrm{{GeV}}$ "
                        rf"(for $Q^2={Q2_special}$: $W_\max={Wmax_special}\,\mathrm{{GeV}}$)")
    else:
        title_suffix = rf"$W_\max={Wmax_default}\,\mathrm{{GeV}}$"
    # ---------------------------------------------------------------------

    region_titles = {
        "1st":  r"1st resonance region $W \in [1.15, 1.35]$",
        "2nd":  r"2nd resonance region $W \in [1.35, 1.6]$",
        "3rd":  r"3rd resonance region $W \in [1.6, 2.0]$",
        "tail": r"Tail region $W \in [2.0, W_{max}]$",
        "part": r"Partial resonance region $W \in [1.15, 2.0]$",
        "all":  r"Full resonance region $W \in [1.15, W_{max}]$",
    }

    for region in ["1st", "2nd", "3rd", "tail", "part", "all"]:
        plt.figure()
         # -------------------------  AO model prediction ---------------------
        good_ao = np.isfinite(ao_m2[region])
        if np.any(good_ao):
            plt.plot(Q2s[good_ao], ao_m2[region][good_ao], color="magenta",marker="o", linestyle="-", markersize=3, label="AO model")
            
            # -------------------------  LO PDF prediction -----------------------
        good_lo = np.isfinite(lo_pdf_m2[region])
        if np.any(good_lo):
            plt.plot(Q2s[good_lo], lo_pdf_m2[region][good_lo], color="blue",marker="d", linestyle="-", markersize=3, label="CJ15lo: LO+LT")
        
        #--------------------NLO PDF-based  prediction--------------------

        plt.plot(Q2s, naked[region], marker="^", markersize=3, linestyle="-",color = "green", label="CJ15nlo: NLO+LT")
        plt.plot(Q2s, brady[region], marker="s", markersize=3, linestyle="-",color = "red", label="CJ15nlo: NLO+LT+TMC+HT")
        # ----  experimental points with error bars ----
        good = np.isfinite(exp_m2[region]) & np.isfinite(exp_e2[region])
        if np.any(good):
            plt.errorbar(Q2s[good], exp_m2[region][good], yerr=exp_e2[region][good], color = "black", fmt="o", linestyle="none", markersize=3, capsize=2,label=f"RGA data (V.Klimenko)\n R_LT from AO model \n{error_mode} error estimation")
    

        plt.xlabel(r"$Q^2\ \mathrm{[GeV^2]}$")
        plt.ylabel(r"$M_2$ (truncated)")
        if region in ["all", "tail"]:
            plt.title(f"{region_titles[region]}\n{title_suffix}")
        else:
            plt.title(f"{region_titles[region]}")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()

        out_path = os.path.join(out_dir, f"M2_vs_Q2_{pdf_set}_{region}_error_{error_mode}.pdf")
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        
    
    print(f"Saved to: {out_dir}")
    
    
    
    
def plot_bin_size_ratio_vs_Q2(out_dir="Bin_size_ratio_plots",
                              out_name="AO_bin_size_ratio_vs_Q2_4panel.png",
                              regions=("1st", "2nd", "3rd", "part"),
                              title=None):
    """
    Make a 4-panel (2x2) plot of ratio_cont_over_trapz vs Q2 for selected regions:
      "1st", "2nd", "3rd", "part" (default)

    Parameters
    ----------
    df_bin : pandas.DataFrame
        Output of estimate_bin_size_err_data(). Must contain columns:
          - Q2, region, ratio_cont_over_trapz
    out_dir : str
        Output directory.
    out_name : str
        Output filename.
    regions : tuple/list of str
        Regions to plot (in this order). Must be 4 for a 2x2 layout.
    title : str or None
        Optional overall title.

    Saves
    -----
    out_dir/out_name
    """

    df_bin = estimate_bin_size_err_data([2.774,3.244,3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699],regions, R_source="AO")
    required = {"Q2", "region", "ratio_cont_over_trapz"}
    missing = required - set(df_bin.columns)
    if missing:
        raise ValueError(f"df_bin is missing required columns: {sorted(missing)}")

    if len(regions) != 4:
        raise ValueError(f"Expected 4 regions for 2x2 plot, got {len(regions)}: {regions}")

    os.makedirs(out_dir, exist_ok=True)

    # Normalize region strings for selection, but keep original labels for titles
    df = df_bin.copy()
    df["region_norm"] = df["region"].astype(str).str.lower().str.strip()

    region_norm_map = {r: str(r).lower().strip() for r in regions}

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    axes = axes.ravel()

    for ax, r in zip(axes, regions):
        fmt = ScalarFormatter(useOffset=False)
        fmt.set_scientific(False)
        ax.yaxis.set_major_formatter(fmt)
        rnorm = region_norm_map[r]
        d = df[df["region_norm"] == rnorm].copy()

        # If region not present, just annotate and continue
        if d.empty:
            ax.text(0.5, 0.5, f"No data for '{r}'", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(str(r))
            ax.axhline(1.0, linewidth=1)
            ax.grid(True, alpha=0.3)
            continue

        d = d.sort_values("Q2")
        Q2 = d["Q2"].to_numpy(dtype=float)
        ratio = d["ratio_cont_over_trapz"].to_numpy(dtype=float)

        # Plot points only (no connecting line)
        ax.plot(Q2, ratio, linestyle="None", marker="o", markersize=4)

        ax.set_title(str(r))
        ax.axhline(1.0, linewidth=1)
        ax.grid(True, alpha=0.3)

        ax.set_ylabel(r"$I_{\mathrm{continious}}/I_{\mathrm{trapz(data\ grid)}}$")

    for ax in axes[-2:]:
        ax.set_xlabel(r"$Q^2\ (\mathrm{GeV}^2)$")

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    out_path = os.path.join(out_dir, out_name)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return out_path


#-----------------------------------------------------------------------------------------------------------
plot_bin_size_ratio_vs_Q2()

#for Q2 in [2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699]:
    #plot_F2_from_data_diff_R_sources(Q2_value=Q2, vs_what = "w")
    #plot_F2_from_data_diff_R_sources(Q2_value=Q2, vs_what = "x")


#for Q2 in [2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699]:
#    plot_F2_from_data_AO_PDF(Q2_value=Q2, W_cutoff=2.0, show_lines = False, vs_what = "x", pdf_set_nlo="CJ15nlo")
#    plot_F2_from_data_AO_PDF(Q2_value=Q2, W_cutoff=2.0, show_lines = False, vs_what = "w", pdf_set_nlo="CJ15nlo")


#plot_M2_truncated_vs_Q2(pdf_set="CJ15nlo", error_mode="point_uncorrelated")
#plot_M2_truncated_vs_Q2(pdf_set="CJ15nlo", error_mode="correlated")
#plot_M2_truncated_vs_Q2(pdf_set="CJ15nlo", error_mode="segment_uncorrelated")


    
#compare_F2([1.025,2.025, 2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699, 15.0, 20.0], pdf_set_lo="CJ15lo", pdf_set_nlo="CJ15nlo", W_cutoff=1.8)
#compare_F2([2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699],pdf_set_lo = "CJ15lo", pdf_set_nlo="CJ15nlo", W_cutoff=2.5)
#compare_F2([1.025,2.025, 2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699, 15.0, 20.0],pdf_set="CJ15nlo", W_cutoff=5.0)
#compare_F2([1.025,2.025, 2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699, 15.0, 20.0],pdf_set="CJ15nlo", W_cutoff=10.0)
#compare_F2([1.025,2.025, 2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699, 15.0, 20.0],pdf_set="CJ15nlo", W_cutoff=20.0)
#compare_F2([1.025, 2.025, 2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699, 15.0, 20.0],pdf_set="CJ15nlo", W_cutoff=30.0)

#for Q2 in [2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699]:
    #compare_xsecs(fixed_Q2=Q2, beam_energy=10.6, pdf_set_lo="CJ15lo", pdf_set_nlo="CJ15nlo", W_cutoff=2.0)    




