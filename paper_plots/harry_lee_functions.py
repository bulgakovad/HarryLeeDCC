import numpy as np
import math
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os
import pandas as pd
from matplotlib.ticker import MultipleLocator, ScalarFormatter


from functions_pdf import get_lo_pdf_interpolators, get_nlo_pdf_interpolators, compute_pdf_cross_sections, compute_pdf_cross_sections_from_F2_FL, get_R_from_F1F2, calculate_moment_LO_pdf, get_nlo_HT_only_pdf_interpolators
from functions_anl_osaka import compute_cross_section_model, compute_1pi_cross_section_model, compute_2pi_cross_section_model, interpolate_structure_functions, sigma_LT_to_F2_AO_model, calculate_moment_AO_ext, get_AO_interpolators, calculate_moment_AO_original
from functions_data import calc_trunc_moment_data, F2_from_xsect_data, x_of_W, estimate_bin_size_err_data, read_patrick_data, read_stas_data, strfun_F2_to_W2


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


def compare_F2(Q2_list: list, what_to_show: list, pdf_set_lo: str, pdf_set_nlo: str, W_cutoff=2.5, num_points=400):
    """
    For each Q² in Q2_list, plot:
        - F2 LO (CJ15) 
        - F2 NLO LT (CJ15)
        - F2 NLO TMC only (CJ15)
        - F2 NLO TMC + HT (CJ15)
        - Patrick's prediction (NEW)
        - AO model (original)
        - AO model (extended)
        - AO 1π contribution (original)
        
    """
     
    out_dir = f"compare_F2_{pdf_set_nlo}"
    os.makedirs(out_dir, exist_ok=True)

    for Q2 in Q2_list:
        have_lo = have_nlo = have_pat_HT = have_pat_no_HT = have_strfun_world = have_strfun_clas = have_ao_original = have_ao_ext = have_ao_2pi = have_rga = have_stas = False
        
        # AO original
        try:
            _, F2_AO, W_ao_rng = get_AO_interpolators(
                file_path="input_data/wempx.dat",
                fixed_Q2=Q2,
                W_min=1.1,
                W_max=W_cutoff,
                num_points=num_points
            )
            W_ao_min, W_ao_max = float(np.min(W_ao_rng)), float(np.max(W_ao_rng))
            have_ao_original = True
        except Exception as e:
            have_ao_original = False
            print(f"No AO model data found for Q2={Q2}: {e}")
            
                # AO extended model from sigma_T + sigma_L table
        try:
            W_ao_ext_native, F2_ao_ext_native = sigma_LT_to_F2_AO_model(Q2)

            m = (
                np.isfinite(W_ao_ext_native) &
                np.isfinite(F2_ao_ext_native) &
                (W_ao_ext_native >= 1.1) &
                (W_ao_ext_native <= W_cutoff)
            )

            W_ao_ext_native = W_ao_ext_native[m]
            F2_ao_ext_native = F2_ao_ext_native[m]

            if W_ao_ext_native.size > 1:
                W_ao_ext_min = float(np.min(W_ao_ext_native))
                W_ao_ext_max = float(np.max(W_ao_ext_native))
                have_ao_ext = True
            else:
                have_ao_ext = False

        except Exception as e:
            have_ao_ext = False
            print(f"No AO extended model data found for Q2={Q2}: {e}")
            
            
        # AO 1 pi contribution
        try:
            _, F2_AO_1pi, W_ao_rng_1pi = get_AO_interpolators(
                file_path="input_data/wemp-pi.dat",
                fixed_Q2=Q2,
                W_min=1.1,
                W_max=W_cutoff,
                num_points=num_points
            )
            W_ao_min_1pi, W_ao_max_1pi = float(np.min(W_ao_rng_1pi)), float(np.max(W_ao_rng_1pi))
            have_ao_1pi = True
        except Exception as e:
            have_ao_1pi = False
            print(f"No AO 1 pi data found for Q2={Q2}: {e}")
            
        

        #CLAS+World interpolation data from strfun website
        try:
            strfun_file = f"strfun_F1F2_data/vs_w/clas_and_world_data/F2_vs_w_Q2={Q2}.dat"
            if os.path.isfile(strfun_file):
                strfun_world = np.genfromtxt(strfun_file, names=["W", "Quantity", "Uncertainty"],
                                    delimiter="\t", skip_header=1)
                m = (strfun_world["W"] >= 1.1) & (strfun_world["W"] <= W_cutoff)
                W_strfun_world = strfun_world["W"][m]
                sigma_strfun_world = strfun_world["Quantity"][m] 
                err_strfun_world = strfun_world["Uncertainty"][m] 
                have_strfun_world = (W_strfun_world.size > 0)
        except Exception:
            have_strfun_world = False
            
            
         #CLAS only interpolation data from strfun website
        try:
            strfun_file = f"strfun_F1F2_data/vs_w/clas_only_data/F2_vs_w_Q2={Q2}.dat"
            if os.path.isfile(strfun_file):
                strfun_clas = np.genfromtxt(strfun_file, names=["W", "Quantity", "Uncertainty"],
                                    delimiter="\t", skip_header=1)
                m = (strfun_clas["W"] >= 1.1) & (strfun_clas["W"] <= W_cutoff)
                W_strfun_clas = strfun_clas["W"][m]
                sigma_strfun_clas = strfun_clas["Quantity"][m] 
                err_strfun_clas = strfun_clas["Uncertainty"][m]
                have_strfun_clas = (W_strfun_clas.size > 0)
        except Exception:
            have_strfun_clas = False  
            
            
        # F2 from Stas
                # F2 from Stas
        try:
            W_stas, F2_stas, F2_err_stas, have_stas = read_stas_data(
                fixed_Q2=Q2,
                channel="pi+ n + pi0 p",
                W_min=1.1,
                W_max=W_cutoff, # changed H Lee
                csv_path="from_Stas/F2_interpolated.csv",
            )
        except Exception as e:
            have_stas = False
            print(f"No Stas F2 data found for Q2={Q2}: {e}")
            
        # F2  from RGA data
        try:
            data_df = F2_from_xsect_data(Q2_value=Q2, R_source="AO")

            m = (data_df["W"] >= 1.1) & (data_df["W"] <= W_cutoff)
            data_df = data_df.loc[m].copy()

            W_data   = data_df["W"].to_numpy()
            F2_data  = data_df["F2"].to_numpy()
            err_data = data_df["F2_err"].to_numpy()

            have_rga = (len(W_data) > 0)

        except Exception as e:
            have_rga = False
            print(f"No extracted F2 data found for Q2={Q2}: {e}")
            

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
            
        
        # Patrick with HT
        try :
            W_pat_HT, F2_pat_HT, err_pat_HT, have_pat_HT = read_patrick_data(
                series="f2",
                fixed_Q2=Q2,
                beam_energy=None,   # ignored inside helper, kept for interface compatibility
                W_min=1.15,
                W_max=W_cutoff,
                xlsx_path="from_patrick/CLAS12.xlsx",
                q2_tol=1e-3,
                convert_nb_to_mub=False,
            )
        except Exception:
            have_pat_HT = False
            print(f"No Patrick HT F2 data found for Q2={Q2}")
            
        # Patrick no HT
        try :
            W_pat_no_HT, F2_pat_no_HT, err_pat_no_HT, have_pat_no_HT = read_patrick_data(
                series="f2",
                fixed_Q2=Q2,
                beam_energy=None,   # ignored inside helper, kept for interface compatibility
                W_min=1.15,
                W_max=W_cutoff,
                xlsx_path="from_patrick/CLAS12_no_HT.xlsx",
                q2_tol=1e-3,
                convert_nb_to_mub=False,
            )
        except Exception:
            have_pat_no_HT = False
            print(f"No Patrick without HT F2 data found for Q2={Q2}")

        wmins = []
        if have_lo:
            wmins.append(W_lo_min)
        if have_nlo:
            wmins.append(W_nlo_min)
        if have_ao_original:
            wmins.append(W_ao_min)
        if have_ao_1pi:
            wmins.append(W_ao_min_1pi)
        if have_ao_ext:
            wmins.append(W_ao_ext_min)

        W_min_global = max(1.0, min(wmins)) if wmins else 1.0
        W_vals = np.linspace(W_min_global, W_cutoff, num_points)
        
        
        #Evaluate AO original model
        F2_AO_vals = np.full_like(W_vals, np.nan, dtype=float)
        if have_ao_original:
            m = (W_vals >= W_ao_min) & (W_vals <= 2.0)
            try:
                F2_AO_vals[m] = F2_AO(W_vals[m])
            except Exception:
                pass
            
        # Evaluate AO extended model
        F2_AO_ext_vals = np.full_like(W_vals, np.nan, dtype=float)
        if have_ao_ext:
            try:
                _, F2_AO_ext_vals = sigma_LT_to_F2_AO_model(
                    fixed_Q2=Q2,
                    W_out=W_vals
                )
            except Exception as e:
                print(f"Unable to evaluate AO extended model for Q2={Q2}: {e}")
        
            
        # Evaluate AO 1 pi contribution
        F2_AO_1pi_vals = np.full_like(W_vals, np.nan, dtype=float)
        if have_ao_1pi:
            m = (W_vals >= W_ao_min_1pi) & (W_vals <= 2.0)
            try:
                F2_AO_1pi_vals[m] = F2_AO_1pi(W_vals[m])
            except Exception:
                pass
            
        # AO 2 pi contribution (full - 1pi)
        try:
            if have_ao_original and have_ao_1pi:
                F2_AO_2pi_vals = np.full_like(W_vals, np.nan, dtype=float)
                m = (W_vals >= W_ao_min) & (W_vals <= 2.0)
                if m.any():
                    F2_AO_2pi_vals[m] = F2_AO(W_vals[m]) - F2_AO_1pi(W_vals[m])
                have_ao_2pi = True
        except Exception as e:
            pass


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
        
        if "AO_model" in what_to_show:
            if np.isfinite(F2_AO_vals).any():
                good = np.isfinite(F2_AO_vals)
                h_ao, = plt.plot(
                    W_vals[good], F2_AO_vals[good],
                    label="ANL-Osaka model full", color="red", ls="solid", lw=2
                )
                handles.append(h_ao)
            else:
                print("Unable to plot AO original model curve")
                
        if "AO_model_ext" in what_to_show:
            if np.isfinite(F2_AO_ext_vals).any():
                w_ao_ext_max = 2.25 if Q2 == 9.699 else 2.5
                good = np.isfinite(F2_AO_ext_vals) & (W_vals >= W_ao_ext_min) & (W_vals <= w_ao_ext_max)
                h_ao_ext, = plt.plot(
                    W_vals[good], F2_AO_ext_vals[good],
                    label="ANL-Osaka model full (extended)", color="red", ls="solid", lw=2
                )
                handles.append(h_ao_ext)
            else:
                print("Unable to plot AO extended model curve")
                
        if "AO_model_1pi" in what_to_show:
            if np.isfinite(F2_AO_1pi_vals).any():
                good = np.isfinite(F2_AO_1pi_vals)
                h_ao_1pi, = plt.plot(
                    W_vals[good], F2_AO_1pi_vals[good],
                    label="ANL-Osaka model 1π contribution", color="black", ls="dashed", lw=2
                )
                handles.append(h_ao_1pi)
            else:
                print("Unable to plot AO 1π contribution curve")
                
        if "Stas" in what_to_show:
            if have_stas:
                h_stas = plt.errorbar(
                    W_stas,
                    F2_stas,
                    yerr=F2_err_stas,
                    fmt="o",
                    color="blue",
                    label=f"from Stas interpolation: {"pi+ n + pi0 p"} channel",
                    markersize=3,
                    capsize=2
                )
                handles.append(h_stas)
            else:
                print(f"Unable to plot Stas F2 for Q2={Q2}")
                
        if "AO_model_2pi" in what_to_show:
            if np.isfinite(F2_AO_2pi_vals).any():
                good = np.isfinite(F2_AO_2pi_vals)
                h_ao_2pi, = plt.plot(
                    W_vals[good], F2_AO_2pi_vals[good],
                    label="ANL-Osaka (full - 1π) contribution", color="blue", ls="dashed", lw=2
                )
                handles.append(h_ao_2pi)
            else:
                print("Unable to plot AO 2π contribution curve")

 
        
        if "strfun_world" in what_to_show:
            if have_strfun_world:
                h_strfun = plt.errorbar(W_strfun_world, sigma_strfun_world, yerr=err_strfun_world, fmt='o', color='black', label="CLAS + World data interpolation", markersize=2)
                handles.append(h_strfun)
            else:
                print("Unable to plot CLAS + World data interpolation")
                
        if "strfun_clas" in what_to_show:
            if have_strfun_clas:
                h_strfun_clas = plt.errorbar(W_strfun_clas, sigma_strfun_clas, yerr=err_strfun_clas, fmt='o', color='purple', label="CLAS-only data interpolation", markersize=2)
                handles.append(h_strfun_clas)
            else:
                print("Unable to plot CLAS-only data interpolation")
                
                
        if "rga_data" in what_to_show:
            if have_rga:
                h_rga = plt.errorbar(
                    W_data,
                    F2_data,
                    yerr=err_data,
                    fmt='o',
                    color='black',
                    label="RGA data (V. Klimenko)",
                    markersize=3,
                    capsize=2
                )
                handles.append(h_rga)
            else:
                print("Unable to plot extracted F2 data")

        if "LO_LT" in what_to_show:
            if np.isfinite(F2_LO_vals).any():
                good = np.isfinite(F2_LO_vals)
                h_lo, = plt.plot(W_vals[good], F2_LO_vals[good],
                                 label=f"{pdf_set_lo}: LO + LT", color="blue", ls="dotted", lw=2)
                handles.append(h_lo)
            else:
                print("Unable to plot LO + LT PDF curve")

        
        if "NLO_LT" in what_to_show:
            if np.isfinite(F2_NLO_vals).any():
                good = np.isfinite(F2_NLO_vals)
                h_naked, = plt.plot(W_vals[good], F2_NLO_vals[good],
                                    label=f"{pdf_set_nlo}: NLO + LT", color="blue", ls="dashed", lw=2)
                handles.append(h_naked)
            else:
                print("Unable to plot NLO + LT PDF curve")

        if "NLO_TMC" in what_to_show:
            if np.isfinite(F2_NLO_TMC_vals).any():
                good = np.isfinite(F2_NLO_TMC_vals)
                h_b, = plt.plot(W_vals[good], F2_NLO_TMC_vals[good],
                                label=f"{pdf_set_nlo}: NLO + LT + TMC (OPE)", color="green", ls="dashdot", lw=1)
                handles.append(h_b)
            else:
                print("Unable to plot NLO + LT + TMC (OPE) PDF curve")

        if "NLO_TMC_HT" in what_to_show:
            if np.isfinite(F2_NLO_TMC_HT_vals).any():
                good = np.isfinite(F2_NLO_TMC_HT_vals)
                h_bht, = plt.plot(W_vals[good], F2_NLO_TMC_HT_vals[good],
                              label=f"{pdf_set_nlo}: NLO + LT + TMC (OPE) + HT", color="orange", ls="dashdot", lw=2)
            handles.append(h_bht)
        
        if "Patrick_HT" in what_to_show:
            if have_pat_HT:
                h_pat_ht = plt.errorbar(W_pat_HT, F2_pat_HT, yerr=err_pat_HT, fmt='o', color='red', label="Patrick's F2 with HT", markersize=2)
                handles.append(h_pat_ht)
            else:
                print("Unable to plot Patrick's F2 with HT")
                
        if "Patrick_no_HT" in what_to_show:
            if have_pat_no_HT:
                h_pat_no_ht = plt.errorbar(W_pat_no_HT, F2_pat_no_HT, yerr=err_pat_no_HT, fmt='o', color='purple', label="Patrick's F2 without HT", markersize=2)
                handles.append(h_pat_no_ht)
            else:
                print("Unable to plot Patrick's F2 without HT")

        

        plt.xlabel("W (GeV)")
        plt.ylabel(r"$F_2$")
        plt.title(f"Comparison of F2 structure functions at Q²={Q2} GeV² ")

        ax = plt.gca()
        #ax.xaxis.set_major_locator(MultipleLocator(0.1))
        #ax.yaxis.set_major_locator(MultipleLocator(0.01))
        ax.grid(True, which="major")

        plt.legend(handles=handles, loc="upper left", fontsize="small")

        q2_str = str(Q2).rstrip("0").rstrip(".")
        out_path = f"{out_dir}/compare_F2_Q2={q2_str}_Wmax={W_cutoff}.pdf"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print("Saved →", out_path)




def compare_W2(Q2_list: list,
               what_to_show: list,
               W_cutoff=2.5,
               num_points=400,
               ao_file_path="input_data/wempx.dat",
               out_dir="compare_W2"):
    """
    Compare W2 structure function for:

        - AO model
        - strfun CLAS + World
        - strfun CLAS only

    AO W2 is taken directly from interpolate_structure_functions().
    strfun W2 is obtained from F2 using:

        W2 = F2 / nu

    where:

        nu = (W^2 - Mp^2 + Q2) / (2 Mp)

    what_to_show options:
        "AO_model"
        "strfun_world"
        "strfun_clas"
    """

    os.makedirs(out_dir, exist_ok=True)

    for Q2 in Q2_list:

        have_ao = False
        have_strfun_world = False
        have_strfun_clas = False

        # ------------------------------------------------------------
        # AO model W2
        # ------------------------------------------------------------
        W_vals = np.linspace(1.1, W_cutoff, num_points)
        W2_AO_vals = np.full_like(W_vals, np.nan, dtype=float)

        if "AO_model" in what_to_show:
            for i, W in enumerate(W_vals):
                try:
                    _, W2_val = interpolate_structure_functions(
                        file_path=ao_file_path,
                        target_W=W,
                        target_Q2=Q2
                    )
                    W2_AO_vals[i] = W2_val
                except Exception:
                    # outside grid or failed interpolation
                    W2_AO_vals[i] = np.nan

            have_ao = np.isfinite(W2_AO_vals).any()

            if not have_ao:
                print(f"No AO W2 data found for Q2={Q2}")

        # ------------------------------------------------------------
        # strfun CLAS + World
        # ------------------------------------------------------------
        if "strfun_world" in what_to_show:
            try:
                W_strfun_world, W2_strfun_world, W2err_strfun_world, have_strfun_world = strfun_F2_to_W2(
                    fixed_Q2=Q2,
                    data_type="clas_and_world",
                    W_min=1.1,
                    W_max=W_cutoff,
                    base_dir="strfun_F1F2_data/vs_w"
                )
            except Exception as e:
                have_strfun_world = False
                print(f"No strfun CLAS + World W2 data found for Q2={Q2}: {e}")

        # ------------------------------------------------------------
        # strfun CLAS only
        # ------------------------------------------------------------
        if "strfun_clas" in what_to_show:
            try:
                W_strfun_clas, W2_strfun_clas, W2err_strfun_clas, have_strfun_clas = strfun_F2_to_W2(
                    fixed_Q2=Q2,
                    data_type="clas_only",
                    W_min=1.1,
                    W_max=W_cutoff,
                    base_dir="strfun_F1F2_data/vs_w"
                )
            except Exception as e:
                have_strfun_clas = False
                print(f"No strfun CLAS-only W2 data found for Q2={Q2}: {e}")

        # ------------------------------------------------------------
        # Plot
        # ------------------------------------------------------------
        plt.figure(figsize=(8, 6))

        handles = [
            plt.Line2D([], [], color="white", label=f"Q² = {Q2:.3f} GeV²")
        ]

        if "AO_model" in what_to_show:
            if have_ao:
                good = np.isfinite(W2_AO_vals)

                h_ao, = plt.plot(
                    W_vals[good],
                    W2_AO_vals[good],
                    color="red",
                    ls="solid",
                    lw=2,
                    label="ANL-Osaka model"
                )
                handles.append(h_ao)
            else:
                print(f"Unable to plot AO W2 for Q2={Q2}")

        if "strfun_world" in what_to_show:
            if have_strfun_world:
                h_world = plt.errorbar(
                    W_strfun_world,
                    W2_strfun_world,
                    yerr=W2err_strfun_world,
                    fmt="o",
                    color="black",
                    markersize=2,
                    capsize=2,
                    label="strfun CLAS + World"
                )
                handles.append(h_world)
            else:
                print(f"Unable to plot strfun CLAS + World W2 for Q2={Q2}")

        if "strfun_clas" in what_to_show:
            if have_strfun_clas:
                h_clas = plt.errorbar(
                    W_strfun_clas,
                    W2_strfun_clas,
                    yerr=W2err_strfun_clas,
                    fmt="o",
                    color="purple",
                    markersize=2,
                    capsize=2,
                    label="strfun CLAS only"
                )
                handles.append(h_clas)
            else:
                print(f"Unable to plot strfun CLAS-only W2 for Q2={Q2}")

        plt.xlabel("W (GeV)")
        plt.ylabel(r"$W_2$ (GeV$^{-1}$)")
        plt.title(f"Comparison of W2 structure functions at Q²={Q2} GeV²")

        ax = plt.gca()
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        ax.grid(True, which="major")

        plt.legend(handles=handles, loc="upper left", fontsize="small")
        plt.tight_layout()

        q2_str = str(Q2).rstrip("0").rstrip(".")
        out_path = f"{out_dir}/compare_W2_Q2={q2_str}_Wmax={W_cutoff}.pdf"

        plt.savefig(out_path, dpi=300)
        plt.close()

        print("Saved →", out_path)


def compare_xsecs( what_to_plot: list,
                   fixed_Q2, beam_energy, pdf_set_lo, pdf_set_nlo,
                   W_cutoff,
                   interp_file="input_data/wempx.dat",
                   onepi_file="input_data/wemp-pi.dat",
                   num_points=200,
                   patrick_series=None          # None, "prediction", "thy", or ["prediction","thy"]
                   ):
    
    out_dir = f"compare_xsecs_{pdf_set_nlo}"
    os.makedirs(out_dir, exist_ok=True)
    patrick_q2_tol=1e-3
    # ---------- Kinematics & constants ----------
    
    data_anl_model = np.loadtxt(interp_file)
    W_grid = np.unique(data_anl_model[:, 0])

    M  = 0.9385

    # lab kinematic W-limit
    w_kin_max = math.sqrt(max(M**2 + 2*M*beam_energy - fixed_Q2, 0.0))
    W_hi = min(W_cutoff, w_kin_max - 1e-6)
    W_lo = W_grid.min()
    W_vals = np.linspace(W_lo, W_hi, num_points)
    
    # Patrick controls
    if patrick_series is None:
        patrick_series = []
    elif isinstance(patrick_series, str):
        patrick_series = [patrick_series]


    # Containers
    anl_full_xs, anl_onepi_xs, anl_two_pi_xs = [], [], []
    if fixed_Q2 > 3.0:
        have_AO = have_AO_1pi = have_AO_2pi = False
        have_AO_ext = True

    else:
        have_AO = have_AO_1pi = have_AO_2pi = True
        have_AO_ext = False  # Do not plot Ext model for Q2 < 3 there is original model!
        if fixed_Q2 == 2.774:
            have_AO_ext = True  # Exception for 2.774 GeV^2 where we have both original and extended models -> show only extended
            have_AO = True # For now
    
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
    #----------------------------------------NLO + HT only. No TMC ---------------------------------------
    try:
        (F1_NLO_HT_only, F2_NLO_HT_only, W_nlo_ht_only_range) = get_nlo_HT_only_pdf_interpolators(fixed_Q2, pdf_set=pdf_set_nlo)
        have_nlo_ht_only = True
    except Exception:
        have_nlo_ht_only = False
        #W_nlo_min = W_nlo_max = None
        F1_NLO_HT_only = F2_NLO_HT_only = lambda w: np.nan
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
            
        # 2π ANL xsec 
        try:
            anl_two_pi_xs.append(compute_2pi_cross_section_model(w, fixed_Q2, beam_energy))
        except Exception:
            anl_two_pi_xs.append(np.nan)

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
    anl_two_pi_xs         = np.asarray(anl_two_pi_xs)
    pdf_lo_xs             = np.asarray(pdf_lo_xs)        if have_lo  else np.array([])
    pdf_nlo_xs            = np.asarray(pdf_nlo_xs)    if have_nlo else np.array([])
    pdf_nlo_tmc_xs        = np.asarray(pdf_nlo_tmc_xs)       if have_nlo else np.array([])
    pdf_nlo_tmc_ht_xs      = np.asarray(pdf_nlo_tmc_ht_xs)    if have_nlo else np.array([])
    pdf_nlo_tmc_ht_F2FL_xs    = np.asarray(pdf_nlo_tmc_ht_F2FL_xs) if have_nlo else np.array([])
    #New: HT only, no TMC
    pdf_nlo_ht_only_xs = np.asarray([compute_pdf_cross_sections(w, fixed_Q2, beam_energy, F1_interp=F1_NLO_HT_only, F2_interp=F2_NLO_HT_only) for w in W_vals]) if have_nlo_ht_only else np.array([])

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
    if have_AO_ext:
        AO_ext_file = f"tables_from_Yannick/fine_binning/AO/Wdist_Q2_{fixed_Q2}_GLOBAL_LT.dat"
        try:
            AO_ext = np.genfromtxt(
                AO_ext_file,
                usecols=(0, 1),
                names=["W", "sigma"],
                delimiter=None,
                skip_header=1
            )
            m = (AO_ext["W"] >= W_lo) & (AO_ext["W"] <= 2.25) # changed the range to 2.0 (original W range)
            W_AO_ext = AO_ext["W"][m]
            sigma_AO_ext = AO_ext["sigma"][m]
            have_AO_ext = (W_AO_ext.size > 0)
        except Exception as e:
            have_AO_ext = False
            print(f"Failed to load {AO_ext_file}: {e}")
        
        # ---------- Patrick data ----------
    patrick_data_HT = {}
    patrick_data_no_HT = {}

    for series in patrick_series:
        try:
            W_pat_HT, sigma_pat_HT, err_pat_HT, have_pat_HT = read_patrick_data(
                series=series,
                fixed_Q2=fixed_Q2,
                beam_energy=beam_energy,   # ignored inside helper, kept for interface compatibility
                W_min=W_lo,
                W_max=W_hi,
                xlsx_path="from_patrick/CLAS12.xlsx",
                q2_tol=patrick_q2_tol,
                convert_nb_to_mub=True,
            )

            if have_pat_HT:
                patrick_data_HT[series] = {
                    "W": W_pat_HT,
                    "y": sigma_pat_HT,
                    "yerr": err_pat_HT,
                }
            W_pat_no_HT, sigma_pat_no_HT, err_pat_no_HT, have_pat_no_HT = read_patrick_data(
                series=series,
                fixed_Q2=fixed_Q2,
                beam_energy=beam_energy,   # ignored inside helper, kept for interface compatibility
                W_min=W_lo,
                W_max=W_hi,
                xlsx_path="from_patrick/CLAS12_no_HT.xlsx",
                q2_tol=patrick_q2_tol,
                convert_nb_to_mub=True,
            )
            if have_pat_no_HT:
                patrick_data_no_HT[series] = {
                    "W": W_pat_no_HT,
                    "y": sigma_pat_no_HT,
                    "yerr": err_pat_no_HT,
                }

        except Exception as e:
            print(f"No Patrick {series} data found for Q2={fixed_Q2}: {e}")
  
    # ---------- Plot ----------
    plt.figure(figsize=(8, 6))
    handles = [plt.Line2D([], [], color='white',
               label=f"Q² = {fixed_Q2:.3f} GeV², E = {beam_energy} GeV")]
    
    if "AO_original" in what_to_plot:
        have_AO == True # For now!
        if have_AO:
            good_anl_full = np.isfinite(anl_full_xs) & (W_vals <= 2.0)
            h_model_full, = plt.plot(W_vals[good_anl_full], anl_full_xs[good_anl_full],
                                 label="ANL-Osaka model full", color="red", ls="solid", lw=2)
            handles.append(h_model_full)
        else: 
            print("Unable to plot ANL-Osaka original model")
            
    if "AO_ext" in what_to_plot:
        if have_AO_ext:
                h_AO_ext = plt.errorbar(W_AO_ext, sigma_AO_ext,
                                     color="red", ls="solid", lw=2,
                                     label="ANL-Osaka model full (extended)")
                handles.append(h_AO_ext)
        else:
            print("Unable to plot ANL-Osaka extended model")
            
    if "AO_1pi" in what_to_plot:
        if have_AO_1pi:
            good_anl_1pi = np.isfinite(anl_onepi_xs) & (W_vals <= 2.0)
            h_model_1pi, = plt.plot(W_vals[good_anl_1pi], anl_onepi_xs[good_anl_1pi], 
                                    label=r"ANL-Osaka model 1$\pi$ contribution", color="black", ls="dashed", lw=2)
            handles.append(h_model_1pi)
        else:
            print("Unable to plot ANL-Osaka model 1π contribution")
            
    if "AO_2pi" in what_to_plot:
        if have_AO_2pi:
            good_anl_2pi = np.isfinite(anl_two_pi_xs) & (W_vals <= 2.0)
            h_model_2pi, = plt.plot(W_vals[good_anl_2pi], anl_two_pi_xs[good_anl_2pi], label="ANL-Osaka (full - 1π) contribution", color="blue", ls="dashed", lw=2)
            handles.append(h_model_2pi)
        else:
            print("Unable to plot ANL-Osaka (full - 1π) contribution")

    if "LO_LT" in what_to_plot:
        if have_lo and np.isfinite(pdf_lo_xs).any():
            good_lo = np.isfinite(pdf_lo_xs)
            h_pdf_lo, = plt.plot(W_vals[good_lo], pdf_lo_xs[good_lo],
                                 label=f"{pdf_set_lo}: LO + LT", color="blue", ls="dotted", lw=2)
            handles.append(h_pdf_lo)
        else:
            print("Unable to plot LO + LT PDF curve")

    if "NLO_LT" in what_to_plot:
        if np.isfinite(pdf_nlo_xs).any():
            good_nlo = np.isfinite(pdf_nlo_xs)
            h_pdf_nlo_lt, = plt.plot(W_vals[good_nlo], pdf_nlo_xs[good_nlo],
                                     label=f"{pdf_set_nlo}: NLO + LT", color="blue", ls="dashed", lw=2)
            handles.append(h_pdf_nlo_lt)
        else:
            print("Unable to plot NLO + LT PDF curve")
        
    if "NLO_TMC" in what_to_plot:
        if np.isfinite(pdf_nlo_tmc_xs).any():
            good_nlo_tmc = np.isfinite(pdf_nlo_tmc_xs)
            h_pdf_nlo_tmc, = plt.plot(W_vals[good_nlo_tmc], pdf_nlo_tmc_xs[good_nlo_tmc],
                                  label=f"{pdf_set_nlo}: NLO + LT + TMC(OPE)", color="purple", ls="dashed", lw=2)
            handles.append(h_pdf_nlo_tmc)
        else:
            print("Unable to plot NLO + LT + TMC PDF curve")

    if "NLO_TMC_HT" in what_to_plot:
        if np.isfinite(pdf_nlo_tmc_ht_xs).any():
            good_nlo_tmc_ht = np.isfinite(pdf_nlo_tmc_ht_xs)
            h_pdf_nlo_ht, = plt.plot(W_vals[good_nlo_tmc_ht], pdf_nlo_tmc_ht_xs[good_nlo_tmc_ht],
                                     label=f"{pdf_set_nlo}: NLO + LT + TMC(OPE) + HT", color="orange", 
                                     ls="dashdot", lw=2)
            handles.append(h_pdf_nlo_ht)
        else:
            print("Unable to plot NLO + TMC + HT PDF curve")

    if "NLO_TMC_HT_F2FL" in what_to_plot:
        #NEW curve from (F2, FL)
        if np.isfinite(pdf_nlo_tmc_ht_F2FL_xs).any():
            good_f2fl = np.isfinite(pdf_nlo_tmc_ht_F2FL_xs)
            h_f2fl, = plt.plot(W_vals[good_f2fl], pdf_nlo_tmc_ht_F2FL_xs[good_f2fl],
                               label=f"{pdf_set_nlo}: NLO + LT + TMC(OPE) + HT from F2, F_L", color="red", ls="solid", lw=2)
            handles.append(h_f2fl)
        else:
            print("Unable to plot NLO + TMC + HT PDF curve calculated from F2 and FL")
    if "NLO_HT_only" in what_to_plot:
        if np.isfinite(pdf_nlo_ht_only_xs).any():
            good_ht_only = np.isfinite(pdf_nlo_ht_only_xs)
            h_nlo_ht_only, = plt.plot(W_vals[good_ht_only], pdf_nlo_ht_only_xs[good_ht_only],
                                     label=f"{pdf_set_nlo}: NLO + LT + HT (no TMC)", color="purple", ls="dashdot", lw=2)
            handles.append(h_nlo_ht_only)
        else:
            print("Unable to plot NLO + HT (no TMC) PDF curve")

    if "data_rga" in what_to_plot:
        if have_rga:
            h_rga = plt.errorbar(W_rga, sigma_rga, yerr=err_rga,
                                 fmt="s", color="black", capsize=1, ms=2,
                                 label="RGA data (V. Klimenko)")
            handles.append(h_rga)
        else:
            print("Unable to plot RGA data points")
      
    if "patrick_HT" in what_to_plot:  
        if "prediction" in patrick_data_HT:
            h_pat_pred = plt.errorbar(
                patrick_data_HT["prediction"]["W"],
                patrick_data_HT["prediction"]["y"],
                yerr=patrick_data_HT["prediction"]["yerr"],
                fmt="o",
                color="magenta",
                capsize=1,
                ms=1.5,
                label="Patrick prediction with HT"
            )
            handles.append(h_pat_pred)

        if "thy" in patrick_data_HT:
            h_pat_thy = plt.errorbar(
                patrick_data_HT["thy"]["W"],
                patrick_data_HT["thy"]["y"],
                yerr=patrick_data_HT["thy"]["yerr"],
                fmt="^",
                color="blue",
                capsize=1,
                ms=1.5,
                label="Patrick thy with HT"
            )
            handles.append(h_pat_thy)
        else:
            print("Unable to plot Patrick HT data points")
    
    if "patrick_no_HT" in what_to_plot:
        if "prediction" in patrick_data_no_HT:
            h_pat_pred = plt.errorbar(
                patrick_data_no_HT["prediction"]["W"],
                patrick_data_no_HT["prediction"]["y"],
                yerr=patrick_data_no_HT["prediction"]["yerr"],
                fmt="o",
                color="green",
                capsize=1,
                ms=1.5,
                label="Patrick prediction without HT"
            )
            handles.append(h_pat_pred)

        if "thy" in patrick_data_no_HT:
            h_pat_thy = plt.errorbar(
                patrick_data_no_HT["thy"]["W"],
                patrick_data_no_HT["thy"]["y"],
                yerr=patrick_data_no_HT["thy"]["yerr"],
                fmt="^",
                color="red",
                capsize=1,
                ms=1.5,
                label="Patrick thy without HT"
            )
            handles.append(h_pat_thy)
        else:
            print("Unable to plot Patrick no HT data points")
    

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
        
    #if any(x in what_to_plot for x in ["LO_LT", "NLO_LT", "NLO_TMC", "NLO_TMC_HT", "NLO_TMC_HT_F2FL"]):
    #    ax = plt.gca()  # get current axes
    #    ax.text(
    #        0.02, 0.98,                      # (x, y) in axes coordinates
    #        f"{pdf_set_nlo}",               # the text
    #        transform=ax.transAxes,  
    #        fontsize=20,         
    #        fontweight="bold",
    #        ha="left", va="top",
    #    )
    
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

    ax_LT.set_title("$#sigma_L, #sigma_T$")
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
    ax_xsect.set_title("Differential cross section from $#sigma_L, #sigma_T$")
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



def plot_M2_truncated_vs_Q2(pdf_set,
                            error_mode="correlated",
                            in_dir="../getF1F2/Output/truncated_moments",
                            out_dir="Moment_vs_Q2",
                            ao_original_file="input_data/wempx.dat",
                            ao_q2_switch=2.774):
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

    def _first_row_or_none(df):
        if df is None:
            return None
        if hasattr(df, "empty") and df.empty:
            return None
        if len(df) == 0:
            return None
        return df.iloc[0]

    def _safe_moment_and_error(func, *args, verbose=False, **kwargs):
        try:
            df = func(*args, **kwargs)
            row = _first_row_or_none(df)
            if row is None:
                return np.nan, np.nan

            m = float(row["moment"])
            de = float(row["error"])

            if not np.isfinite(m) or not np.isfinite(de):
                return np.nan, np.nan

            # treat fake "no content" output as missing
            if m == 0.0 and de == 0.0:
                return np.nan, np.nan

            return m, de

        except Exception as e:
            if verbose:
                print(f"[skip] {func.__name__}{args}: {e}")
            return np.nan, np.nan

    def _safe_moment_only(func, *args, verbose=False, zero_is_missing=True, **kwargs):
        try:
            df = func(*args, **kwargs)
            row = _first_row_or_none(df)
            if row is None:
                return np.nan

            m = float(row["moment"])

            if not np.isfinite(m):
                return np.nan

            if zero_is_missing and m == 0.0:
                return np.nan

            return m

        except Exception as e:
            if verbose:
                print(f"[skip] {func.__name__}{args}: {e}")
            return np.nan

    def _safe_ao_moment(q2v, region, verbose=False):
        """
        AO logic:
          - tail: always removed
          - Q2 < ao_q2_switch:
                use ORIGINAL AO for 1st, 2nd, 3rd, part
                keep EXTENDED AO for all (because original AO is not defined for W > 2.0)
          - Q2 >= ao_q2_switch:
                keep current EXTENDED AO logic, except tail stays removed
        """
        r = str(region).lower().strip()

        # Never show AO in tail
        if r == "tail":
            return np.nan

        # Below switch: use original AO where it is actually defined
        if q2v < ao_q2_switch:
            if r in {"1st", "2nd", "3rd", "part"}:
                return _safe_moment_only(
                    calculate_moment_AO_original,
                    q2v, r,
                    n=2,
                    file_path=ao_original_file,
                    verbose=verbose
                )
            elif r == "all":
                return _safe_moment_only(
                    calculate_moment_AO_ext,
                    q2v, r,
                    n=2,
                    E_beam=10.6,
                    in_dir="tables_from_Yannick/fine_binning/AO",
                    convert_ub_to_GeV2=True,
                    divide_by_Gamma=True,
                    verbose=verbose
                )
            else:
                return np.nan

        # Above switch: keep existing extended AO logic, except tail already removed
        return _safe_moment_only(
            calculate_moment_AO_ext,
            q2v, r,
            n=2,
            E_beam=10.6,
            in_dir="tables_from_Yannick/fine_binning/AO",
            convert_ub_to_GeV2=True,
            divide_by_Gamma=True,
            verbose=verbose
        )

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

    regions = ["1st", "2nd", "3rd", "tail", "all", "part"]

    # Sort by Q2 just in case
    idx = np.argsort(Q2)
    Q2s = Q2[idx]
    for k in brady:
        brady[k] = brady[k][idx]
        naked[k] = naked[k][idx]
        
    # to plot only from Q2 = 2.5 GeV as Dr Joo  asked
    mask_q2 = Q2s >= 2.5
    Q2s = Q2s[mask_q2]
    for k in brady:
        brady[k] = brady[k][mask_q2]
        naked[k] = naked[k][mask_q2]

    # ---- experimental moments from data ----
    exp_m2 = {r: np.full_like(Q2s, np.nan, dtype=float) for r in regions}
    exp_e2 = {r: np.full_like(Q2s, np.nan, dtype=float) for r in regions}

    for i, q2v in enumerate(Q2s):
        for r in regions:
            try:
                df = calc_trunc_moment_data(q2v, r, R_source="AO", n=2, error_mode=error_mode)

                if df is None or len(df) == 0:
                    continue
                if hasattr(df, "empty") and df.empty:
                    continue

                out = df.iloc[0]

                m = float(out["moment"])
                de = float(out["error"])

                if not np.isfinite(m) or not np.isfinite(de):
                    continue
                if m == 0.0 and de == 0.0:
                    continue

                exp_m2[r][i] = m
                exp_e2[r][i] = de

            except (FileNotFoundError, IndexError, KeyError):
                continue

    for r in regions:
        good = np.isfinite(exp_m2[r]) & np.isfinite(exp_e2[r])
        print(f"{r}: Q2 with data =", Q2s[good])

    # ------------------------- AO model moments -----------------------------
    ao_m2 = {r: np.full_like(Q2s, np.nan, dtype=float) for r in regions}

    for i, q2v in enumerate(Q2s):
        for r in regions:
            ao_m2[r][i] = _safe_ao_moment(q2v, r, verbose=False)

    for r in regions:
        good_ao = np.isfinite(ao_m2[r])
        print(f"AO {r}: Q2 with data =", Q2s[good_ao])

    # --------------------------- LO PDF moments ------------------------------
    lo_pdf_m2 = {r: np.full_like(Q2s, np.nan, dtype=float) for r in regions}

    for i, q2v in enumerate(Q2s):
        for r in regions:
            lo_pdf_m2[r][i] = _safe_moment_only(
                calculate_moment_LO_pdf,
                q2v, r,
                verbose=False
            )

    # -------------------- Wmax info for title --------------------
    Q2_special = 9.699
    Wmax_default = 2.5
    Wmax_special = 2.25

    has_special = np.any(np.isclose(Q2s, Q2_special, rtol=0, atol=1e-6))
    if has_special:
        title_suffix = (rf"$W_\max={Wmax_default}\,\mathrm{{GeV}}$ "
                        rf"(for $Q^2={Q2_special}$: $W_\max={Wmax_special}\,\mathrm{{GeV}}$)")
    else:
        title_suffix = rf"$W_\max={Wmax_default}\,\mathrm{{GeV}}$"

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

        good_exp = np.isfinite(exp_m2[region]) & np.isfinite(exp_e2[region])
        if np.any(good_exp):
            plt.errorbar(
                Q2s[good_exp],
                exp_m2[region][good_exp],
                yerr=exp_e2[region][good_exp],
                color="black",
                fmt="o",
                linestyle="none",
                markersize=3,
                capsize=2,
                label=f"RGA data (V.Klimenko)\nR_LT from AO model\n{error_mode} error estimation"
            )

        # ------------------------- AO model prediction ---------------------
        good_ao = np.isfinite(ao_m2[region])
        if np.any(good_ao):
            if region in ["1st", "2nd", "3rd", "part"]:
                ao_label = f"ANL-Osaka model: \n original for $Q^2<{ao_q2_switch}$, extended otherwise"
            else:
                ao_label = "ANL-Osaka model extended"

            plt.plot(
                Q2s[good_ao],
                ao_m2[region][good_ao],
                color="red",
                marker="o",
                linestyle="-",
                markersize=3,
                label=ao_label
            )

        # ------------------------- LO PDF prediction -----------------------
        # good_lo = np.isfinite(lo_pdf_m2[region])
        # if np.any(good_lo):
        #     plt.plot(Q2s[good_lo], lo_pdf_m2[region][good_lo],
        #              color="orange", marker="d", linestyle="-",
        #              markersize=3, label="CJ15lo: LO+LT")

        # -------------------- NLO PDF-based prediction --------------------
        good_naked = np.isfinite(naked[region])
        if np.any(good_naked):
            plt.plot(
                Q2s[good_naked],
                naked[region][good_naked],
                marker="^",
                markersize=3,
                linestyle="-",
                color="blue",
                label="CJ15nlo: NLO+LT"
            )

        good_brady = np.isfinite(brady[region])
        if np.any(good_brady):
            plt.plot(
                Q2s[good_brady],
                brady[region][good_brady],
                marker="s",
                markersize=3,
                linestyle="-",
                color="orange",
                label="CJ15nlo: NLO+LT+TMC+HT"
            )

        plt.xlabel(r"$Q^2\ \mathrm{[GeV^2]}$")
        ax = plt.gca()
        ax.xaxis.set_major_locator(MultipleLocator(0.5))   # major ticks every 0.5 in Q2
        ax.xaxis.set_minor_locator(MultipleLocator(0.25))  # minor ticks every 0.25
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
#plot_bin_size_ratio_vs_Q2()
    
#for Q2 in [9.699]:
#    compare_xsecs(
#        [
#         "data_rga",
#         #"AO_original",
#        "AO_ext",
#        #"AO_1pi", 
#         #"AO_2pi",
#         "NLO_TMC_HT", 
#         "NLO_LT", 
#         #"LO_LT", 
#         #"NLO_TMC",
#         #"NLO_HT_only",
#         #"NLO_TMC_HT_F2FL",
#         #"patrick_HT",
#         #"patrick_no_HT"
#         ],
#        fixed_Q2=Q2,
#        beam_energy=10.6,
#        pdf_set_lo="CJ15lo",
#        pdf_set_nlo="CJ15nlo",
#        W_cutoff=2.75,
#        patrick_series=["thy"]
#    )


#for Q2 in [2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699]:
    #plot_F2_from_data_diff_R_sources(Q2_value=Q2, vs_what = "w")
    #plot_F2_from_data_diff_R_sources(Q2_value=Q2, vs_what = "x")


#for Q2 in [2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699]:
#    plot_F2_from_data_AO_PDF(Q2_value=Q2, W_cutoff=2.0, show_lines = False, vs_what = "x", pdf_set_nlo="CJ15nlo")
#    plot_F2_from_data_AO_PDF(Q2_value=Q2, W_cutoff=2.0, show_lines = False, vs_what = "w", pdf_set_nlo="CJ15nlo")


#plot_M2_truncated_vs_Q2(pdf_set="CJ15nlo", error_mode="point_uncorrelated")
#plot_M2_truncated_vs_Q2(pdf_set="CJ15nlo", error_mode="correlated")
#plot_M2_truncated_vs_Q2(pdf_set="CJ15nlo", error_mode="segment_uncorrelated")


    
#compare_W2(
#    Q2_list=[1.0, 2.0, 3.0],
#    what_to_show=[
#        "AO_model",
#        #"strfun_world",
#        "strfun_clas"
#    ],
#    W_cutoff=2.0,
#    num_points=400
#)


compare_F2(Q2_list = [2.774], what_to_show = ["AO_model", "rga_data", "AO_model_1pi", "NLO_TMC_HT"], pdf_set_lo = "CJ15lo", pdf_set_nlo="CJ15nlo", W_cutoff= 2.0 )



