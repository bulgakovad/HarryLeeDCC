import numpy as np
import math
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os
import pandas as pd


from functions_pdf import get_pdf_interpolators_with_error, get_nlo_pdf_interpolators, compute_cross_section_pdf_with_error, get_nlo_pdf_cross_sections
from functions_anl_osaka import compute_cross_section, calculate_1pi_cross_section, compute_2pi_cross_section

from functions_fitting import (
    read_and_prepare_data,
    compute_reference_shapes_for_df,
    build_least_squares_from_df,
    run_minuit,
    make_predict_curve_function,
    make_component_curve_functions   # NEW
)


def generate_table_struct_funcs(file_path, fixed_Q2, onepi_file="input_data/wemp-pi.dat"):
    """
    Generates a table of structure functions W1 and W2 for full and 1π channels,
    for a fixed Q². The format is:
        W   W1_full   W1_1pi   W2_full   W2_1pi

    Parameters:
        file_path (str): Path to the full-channel structure function file (W, Q², W1, W2)
        fixed_Q2 (float): Q² value for which to extract and interpolate structure functions
        onepi_file (str): Path to 1π-channel structure function file
    """
    # Load full-channel data
    data = np.loadtxt(file_path)
    W = data[:, 0]
    Q2 = data[:, 1]
    W1 = data[:, 2]
    W2 = data[:, 3]

    # Identify the closest grid Q² value
    Q2_unique = np.unique(Q2)
    idx = np.argmin(np.abs(Q2_unique - fixed_Q2))
    grid_Q2 = Q2_unique[idx]
    tol = 1e-6
    rows = data[np.abs(Q2 - grid_Q2) < tol]

    output_rows = []
    for row in rows:
        W_val = row[0]
        W1_full = row[2]
        W2_full = row[3]

        # Try to interpolate W1 and W2 for 1π at this W, Q²
        try:
            W1_1pi, W2_1pi = interpolate_structure_functions_1pi(onepi_file, W_val, fixed_Q2)
        except Exception:
            W1_1pi, W2_1pi = float('nan'), float('nan')

        output_rows.append([W_val, W1_full, W1_1pi, W2_full, W2_1pi])

    # Save to file
    header = "W\tW1_full\tW1_1pi\tW2_full\tW2_1pi"
    output_filename = f"tables_struct_funcs/struct_funcs_W12_for_Q2={fixed_Q2}.txt"
    np.savetxt(output_filename, np.array(output_rows), header=header, fmt="%.6e", delimiter="\t")
    print(f"Table saved as {output_filename}")

def generate_table_xsecs(file_path, fixed_Q2, beam_energy, onepi_file="input_data/wemp-pi.dat"):
    """
    Generates a table of differential cross sections dσ/dWdQ² for full and 1π channels,
    at a fixed Q². The format is:
        W   xsec_full   xsec_1pi

    Parameters:
        file_path (str): Path to the full-channel structure function file (W, Q², W1, W2)
        fixed_Q2 (float): Q² value at which cross sections are evaluated
        beam_energy (float): Beam energy (E) in GeV
        onepi_file (str): Path to 1π-channel structure function file
    """
    data = np.loadtxt(file_path)
    W = data[:, 0]
    Q2 = data[:, 1]

    Q2_unique = np.unique(Q2)
    idx = np.argmin(np.abs(Q2_unique - fixed_Q2))
    grid_Q2 = Q2_unique[idx]
    tol = 1e-6
    rows = data[np.abs(Q2 - grid_Q2) < tol]

    output_rows = []
    for row in rows:
        W_val = row[0]

        try:
            xsec_full = compute_cross_section(W_val, fixed_Q2, beam_energy, file_path, verbose=False)
        except Exception:
            xsec_full = float('nan')

        try:
            xsec_1pi = calculate_1pi_cross_section(W_val, fixed_Q2, beam_energy, onepi_file, verbose=False)
        except Exception:
            xsec_1pi = float('nan')

        output_rows.append([W_val, xsec_full, xsec_1pi])

    header = "W\txsec_full\txsec_1pi"
    output_filename = f"tables_xsecs/xsecs_for_Q2={fixed_Q2}.txt"
    os.makedirs("tables_xsecs", exist_ok=True)
    np.savetxt(output_filename, np.array(output_rows), header=header, fmt="%.6e", delimiter="\t")
    print(f"Table saved as {output_filename}")

    
def compare_strfun(fixed_Q2, beam_energy,
                   interp_file="input_data/wempx.dat",
                   onepi_file="input_data/wemp-pi.dat",
                   num_points=200):
    """
    Plot whatever is available among:
      - ANL-Osaka (full) and 1π
      - LO PDF (CJ15) with error band
      - NLO PDF (Brady)
      - CLAS+World (strfun) smoothed band
      - RGA data (Valera)
    For any source that is missing or invalid at this Q², just skip it.
    """

    W_cutoff = 2.0
    data = np.loadtxt(interp_file)
    W_grid = np.unique(data[:, 0])
    W_vals = np.linspace(W_grid.min(), min(W_grid.max(), W_cutoff), num_points)

    # Containers
    anl_xs, onepi_xs = [], []
    pdf_lo_xs, pdf_lo_err = [], []
    pdf_nlo_xs = []
    pdf_nlo_ht_xs = []  # For NLO HT cross sections

    # ---------- LO PDF (with error) ----------
    try:
        F1_lo, F2_lo, F1_lo_err, F2_lo_err, _ = get_pdf_interpolators_with_error(fixed_Q2, central_iset=400)
        have_lo = True
    except Exception:
        have_lo = False

    # ---------- NLO PDF (Brady tables) ----------
    try:
        F1_brady, F1_brady_alt, F1_HT, F2_brady, F2_HT, _ = get_nlo_pdf_interpolators(fixed_Q2)
        have_nlo = True
    except Exception:
        have_nlo = False

    # ---------- Curves on W grid ----------
    for w in W_vals:
        # ANL full
        try:
            anl_xs.append(compute_cross_section(w, fixed_Q2, beam_energy,
                                                file_path=interp_file, verbose=False))
        except Exception:
            anl_xs.append(np.nan)
        # 1π
        try:
            onepi_xs.append(calculate_1pi_cross_section(w, fixed_Q2, beam_energy,
                                                        file_path=onepi_file, verbose=False))
        except Exception:
            onepi_xs.append(np.nan)
        # LO PDF
        if have_lo:
            try:
                pdf_val, pdf_err = compute_cross_section_pdf_with_error(
                    w, fixed_Q2, beam_energy, F1_lo, F2_lo, F1_lo_err, F2_lo_err
                )
                pdf_lo_xs.append(pdf_val)
                pdf_lo_err.append(pdf_err)
            except Exception:
                pdf_lo_xs.append(np.nan)
                pdf_lo_err.append(np.nan)
        # NLO PDF
        if have_nlo:
            try:
                pdf_nlo_xs.append(get_nlo_pdf_cross_sections(w, fixed_Q2, beam_energy, F1_interp=F1_brady, F2_interp=F2_brady))
                pdf_nlo_ht_xs.append(get_nlo_pdf_cross_sections(w, fixed_Q2, beam_energy, F1_interp=F1_HT, F2_interp=F2_HT))
            except Exception:
                pdf_nlo_xs.append(np.nan)
                pdf_nlo_ht_xs.append(np.nan)

    anl_xs     = np.asarray(anl_xs)
    onepi_xs   = np.asarray(onepi_xs)
    pdf_lo_xs  = np.asarray(pdf_lo_xs)  if have_lo  else np.array([])
    pdf_lo_err = np.asarray(pdf_lo_err) if have_lo  else np.array([])
    pdf_nlo_xs = np.asarray(pdf_nlo_xs) if have_nlo else np.array([])
    pdf_nlo_ht_xs = np.asarray(pdf_nlo_ht_xs) if have_nlo else np.array([])

    # ---------- strfun smoothed band (optional) ----------
    have_strfun = False
    try:
        meas_file = f"strfun_data/cs_Q2={fixed_Q2}_E={beam_energy}.dat"
        if os.path.isfile(meas_file):
            mdat = np.genfromtxt(meas_file, names=["W", "Quantity", "Uncertainty"],
                                 delimiter="\t", skip_header=1)
            mask_meas = mdat["W"] <= W_cutoff
            W_meas = mdat["W"][mask_meas]
            cs_meas = mdat["Quantity"][mask_meas]
            err_meas = mdat["Uncertainty"][mask_meas]

            sort_idx = np.argsort(W_meas)
            W_sorted = W_meas[sort_idx]
            cs_sorted = cs_meas[sort_idx]
            err_sorted = err_meas[sort_idx]

            cs_interp  = interp1d(W_sorted, cs_sorted, kind='cubic', bounds_error=False, fill_value="extrapolate")
            err_interp = interp1d(W_sorted, err_sorted, kind='cubic', bounds_error=False, fill_value="extrapolate")

            W_data_min = W_sorted[0]
            W_data_max = W_sorted[-1]
            mask_within_data = (W_vals >= W_data_min) & (W_vals <= W_data_max)
            W_vals_data = W_vals[mask_within_data]

            cs_vals  = cs_interp(W_vals_data)
            err_vals = err_interp(W_vals_data)
            have_strfun = True
        else:
            W_vals_data = np.array([])
            cs_vals = err_vals = np.array([])
    except Exception:
        W_vals_data = np.array([])
        cs_vals = err_vals = np.array([])

    # ---------- RGA data (optional) ----------
    have_rga = False
    try:
        rga_file = f"exp_data/InclusiveExpValera_Q2={fixed_Q2}.dat"
        if os.path.isfile(rga_file):
            rga_data = np.genfromtxt(rga_file,
                                     names=["W", "eps", "sigma", "error", "sys_error"],
                                     delimiter="\t", skip_header=1)
            mask_rga = rga_data["W"] <= W_cutoff
            W_rga = rga_data["W"][mask_rga]
            sigma_rga = rga_data["sigma"][mask_rga] * 1e-3
            err_rga = np.sqrt(rga_data["error"][mask_rga]**2 + rga_data["sys_error"][mask_rga]**2) * 1e-3
            have_rga = (len(W_rga) > 0)
    except Exception:
        pass

    # ---------- Plot ----------
    plt.figure(figsize=(8, 6))
    handles = [plt.Line2D([], [], color='white',
               label=f"Q² = {fixed_Q2:.3f} GeV², E = {beam_energy} GeV")]

    # ANL and 1π (only if any finite values)
    if np.isfinite(anl_xs).any():
        h_anl, = plt.plot(W_vals, anl_xs, label="ANL-Osaka model: full cross section",
                          color="black", lw=2)
        handles.append(h_anl)
    if np.isfinite(onepi_xs).any():
        good = np.isfinite(onepi_xs)
        h_1pi, = plt.plot(W_vals[good], onepi_xs[good],
                          label="ANL-Osaka model: 1π contribution", color="black", ls="--", lw=2)
        handles.append(h_1pi)

    # strfun band
    if have_strfun and len(W_vals_data) > 0:
        h_strfun_line, = plt.plot(W_vals_data, cs_vals, color="grey", lw=2,
                                  label="CLAS+World data smoothed")
        plt.fill_between(W_vals_data, cs_vals - err_vals, cs_vals + err_vals,
                         color="grey", alpha=0.3)
        handles.append(h_strfun_line)

    # LO PDF
    if have_lo and np.isfinite(pdf_lo_xs).any():
        good_lo = np.isfinite(pdf_lo_xs)
        h_pdf_lo, = plt.plot(W_vals[good_lo], pdf_lo_xs[good_lo],
                             label="LO PDF (CJ15)", color="blue", ls="dotted", lw=1)
        try:
            plt.fill_between(W_vals[good_lo],
                             pdf_lo_xs[good_lo] - pdf_lo_err[good_lo],
                             pdf_lo_xs[good_lo] + pdf_lo_err[good_lo],
                             color="blue", alpha=0.3)
        except Exception:
            pass
        handles.append(h_pdf_lo)

    # NLO PDF
    if have_nlo and np.isfinite(pdf_nlo_xs).any():
        good_nlo = np.isfinite(pdf_nlo_xs)
        good_nlo_ht = np.isfinite(pdf_nlo_ht_xs)
        h_pdf_nlo, = plt.plot(W_vals[good_nlo], pdf_nlo_xs[good_nlo],label="NLO PDF (CJ15 TMC only)", color="green", ls="dashdot", lw=2)
        h_pdf_nlo_ht, = plt.plot(W_vals[good_nlo_ht], pdf_nlo_ht_xs[good_nlo_ht], label="NLO PDF (CJ15 TMC + HT)", color="orange", ls="dashdot", lw=2)
        handles.append(h_pdf_nlo)
        handles.append(h_pdf_nlo_ht)

    # RGA points
    if have_rga:
        h_rga = plt.errorbar(W_rga, sigma_rga, yerr=err_rga,
                             fmt="s", color="magenta", capsize=1, ms=2,
                             label="RGA data (V. Klimenko)")
        handles.append(h_rga)

    plt.xlabel("W (GeV)")
    plt.ylabel(r"$d \sigma / dW dQ^2$ ($\mathrm{\mu bn/GeV^3}$)")
    plt.grid(True)
    if len(handles) > 0:
        plt.legend(handles=handles, loc="upper left", fontsize="small")

    os.makedirs("compare_strfun", exist_ok=True)
    fname = f"compare_strfun/compare_strfun_Q2={fixed_Q2}_E={beam_energy}.pdf"
    plt.savefig(fname, dpi=300)
    plt.close()
    print("Saved →", fname)

def compare_exp_model_pdf_bjorken_x(
    fixed_Q2,
    beam_energy,
    interp_file="input_data/wempx.dat",
    onepi_file="input_data/wemp-pi.dat",
    num_points=200
):
    Mp = 0.9385

    valid_Q2_list = [3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699]
    plot_rga = abs(fixed_Q2 - 2.774) < 1e-3
    only_pdf_and_klim = (fixed_Q2 > 3.0) and any(abs(fixed_Q2 - q2) < 1e-3 for q2 in valid_Q2_list)

    # W → x
    W_cutoff = 2.5 if only_pdf_and_klim else 2.0
    W_vals = np.linspace(1.1, W_cutoff, num_points)
    x_vals = fixed_Q2 / (W_vals**2 - Mp**2 + fixed_Q2)
    mask = x_vals > 0
    W_vals = W_vals[mask]
    x_vals = x_vals[mask]

    # Load PDFs
    F1_lo, F2_lo, F1_lo_err, F2_lo_err, _ = get_pdf_interpolators_with_error(fixed_Q2, central_iset=400)
    pdf_lo_xs, pdf_lo_err = [], []
    for w in W_vals:
        try:
            lo, lo_err = compute_cross_section_pdf_with_error(
                w, fixed_Q2, beam_energy, F1_lo, F2_lo, F1_lo_err, F2_lo_err
            )
        except Exception:
            lo, lo_err = np.nan, np.nan
        pdf_lo_xs.append(lo)
        pdf_lo_err.append(lo_err)

    jacobian = fixed_Q2 / (2.0 * W_vals * x_vals**2)
    pdf_lo_xs = np.array(pdf_lo_xs) * jacobian
    pdf_lo_err = np.array(pdf_lo_err) * jacobian

    anl_xs, onepi_xs = [], []
    if not only_pdf_and_klim:
        # Load strfun data
        strfun_file = f"strfun_data/cs_Q2={fixed_Q2}_E={beam_energy}.dat"
        if not os.path.isfile(strfun_file):
            raise FileNotFoundError(f"Strfun data not found: {strfun_file}")
        strfun_data = np.genfromtxt(strfun_file, delimiter="\t", skip_header=1)
        W_exp = strfun_data[:, 0]
        sigma_exp = strfun_data[:, 1]
        sigma_err = strfun_data[:, 2]

        x_exp = fixed_Q2 / (W_exp**2 - Mp**2 + fixed_Q2)
        dWdx_exp = fixed_Q2 / (2.0 * W_exp * x_exp**2)
        sigma_dx = sigma_exp * dWdx_exp
        sigma_dx_err = sigma_err * dWdx_exp

        # Interpolation of strfun data
        sort_idx = np.argsort(x_exp)
        x_sorted = x_exp[sort_idx]
        sigma_sorted = sigma_dx[sort_idx]
        err_sorted = sigma_dx_err[sort_idx]

        cs_interp = interp1d(x_sorted, sigma_sorted, kind='cubic', bounds_error=False, fill_value=np.nan)
        err_interp = interp1d(x_sorted, err_sorted, kind='cubic', bounds_error=False, fill_value=np.nan)

        cs_vals = cs_interp(x_vals)
        err_vals = err_interp(x_vals)

        # Load ANL + 1π
        for w in W_vals:
            try:
                anl = compute_cross_section(w, fixed_Q2, beam_energy, file_path=interp_file, verbose=False)
            except Exception:
                anl = np.nan
            try:
                onepi = calculate_1pi_cross_section(w, fixed_Q2, beam_energy, file_path=onepi_file, verbose=False)
            except Exception:
                onepi = np.nan
            anl_xs.append(anl)
            onepi_xs.append(onepi)
        anl_xs = np.array(anl_xs) * jacobian
        onepi_xs = np.array(onepi_xs) * jacobian

    # Klimenko data if needed
    if plot_rga or only_pdf_and_klim:
        klim_file = f"exp_data/InclusiveExpValera_Q2={fixed_Q2}.dat"
        if not os.path.isfile(klim_file):
            raise FileNotFoundError("Klimenko RGA data not found.")
        klim = np.genfromtxt(klim_file, delimiter="\t", skip_header=1)
        W_klim = klim[:, 0]
        sigma_klim = klim[:, 2] * 1e-3
        err_stat = klim[:, 3] * 1e-3
        err_sys  = klim[:, 4] * 1e-3
        err_total = np.sqrt(err_stat**2 + err_sys**2)
        x_klim = fixed_Q2 / (W_klim**2 - Mp**2 + fixed_Q2)
        dWdx_klim = fixed_Q2 / (2.0 * W_klim * x_klim**2)
        sigma_klim_dx = sigma_klim * dWdx_klim
        err_klim_dx = err_total * dWdx_klim

    # ----------- PLOT ------------
    plt.figure(figsize=(8, 6))
    legend_text = f"$Q^2$ = {fixed_Q2:.3f} GeV$^2$, $E_{{beam}}$ = {beam_energy:.2f} GeV"
    plt.plot([], [], ' ', label=legend_text)

    if not only_pdf_and_klim:
        plt.plot(x_vals, anl_xs, label="ANL-Osaka model: full cross section", color="black")
        plt.plot(x_vals, onepi_xs, "--", label="ANL-Osaka model: $1\\pi$ contribution", color="black", ls="--")
        plt.plot(x_vals, cs_vals, color="grey", lw=1, label="CLAS+World data (smoothed)")
        plt.fill_between(x_vals, cs_vals - err_vals, cs_vals + err_vals, color="grey", alpha=0.3)

    plt.plot(x_vals, pdf_lo_xs, label="LO PDF (CJ15)", color="blue", ls="dotted", lw=1)
    plt.fill_between(x_vals, pdf_lo_xs - pdf_lo_err, pdf_lo_xs + pdf_lo_err, color="blue", alpha=0.3)

    if plot_rga or only_pdf_and_klim:
        plt.errorbar(
            x_klim, sigma_klim_dx, yerr=err_klim_dx,
            fmt="s", ms=3, capsize=2, color="magenta", label="RGA data (V. Klimenko)"
        )

    plt.xlabel("Bjorken x")
    plt.ylabel(r"$d\sigma/dQ^2dx$ [$\mu b/GeV^2$]")
    plt.grid(True)
    plt.legend(fontsize="small")

    # Axis limits
    xlim_min = np.min(x_vals) * 0.95
    xlim_max = np.max(x_vals) * 1.05
    ylim_max = 1.05 * np.nanmax([
        np.nanmax(pdf_lo_xs + pdf_lo_err),
        np.nanmax(sigma_klim_dx) if (plot_rga or only_pdf_and_klim) else 0,
        np.nanmax(cs_vals + err_vals) if not only_pdf_and_klim else 0,
        np.nanmax(anl_xs) if not only_pdf_and_klim else 0
    ])
    plt.xlim(xlim_min, xlim_max)
    plt.ylim(0, ylim_max)
    plt.tight_layout()

    # Save plot
    os.makedirs("compare_strfun_x_xi", exist_ok=True)
    fname = f"compare_strfun_x_xi/compare_strfun_vs_x_Q2={fixed_Q2}_E={beam_energy}.pdf"
    plt.savefig(fname, dpi=300)
    plt.close()
    print("Saved →", fname)


def compare_exp_model_pdf_nachtmann_xi(fixed_Q2, beam_energy,
                                       interp_file="input_data/wempx.dat",
                                       onepi_file="input_data/wemp-pi.dat",
                                       num_points=200):

    Mp = 0.9385
    

    def xi_from_x(x, Q2):
        t = 4 * x**2 * Mp**2 / Q2
        return 2 * x / (1 + np.sqrt(1 + t))

    def dxdxi(x, Q2):
        t = 4 * x**2 * Mp**2 / Q2
        return 0.5 * (1 + t + np.sqrt(1 + t))

    def dWdx(W, x):
        return (W**2 + fixed_Q2 - Mp**2) / (2.0 * W * x)

    valid_Q2_list = [3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699]
    plot_rga = abs(fixed_Q2 - 2.774) < 1e-3
    only_pdf_and_klim = (fixed_Q2 > 3.0) and any(abs(fixed_Q2 - q2) < 1e-3 for q2 in valid_Q2_list)

    # W → x → ξ
    W_cutoff = 2.5 if only_pdf_and_klim else 2.0
    W_vals = np.linspace(1.1, W_cutoff, num_points)
    x_vals = fixed_Q2 / (W_vals**2 - Mp**2 + fixed_Q2)
    mask = x_vals > 0
    W_vals = W_vals[mask]
    x_vals = x_vals[mask]
    xi_vals = xi_from_x(x_vals, fixed_Q2)

    F1_lo, F2_lo, F1_lo_err, F2_lo_err, _ = get_pdf_interpolators_with_error(fixed_Q2, central_iset=400)

    anl_xs, onepi_xs = [], []
    pdf_lo_xs, pdf_lo_err = [], []

    for w in W_vals:
        try:
            lo, lo_err = compute_cross_section_pdf_with_error(w, fixed_Q2, beam_energy, F1_lo, F2_lo, F1_lo_err, F2_lo_err)
        except Exception:
            lo, lo_err = np.nan, np.nan
        pdf_lo_xs.append(lo)
        pdf_lo_err.append(lo_err)

        if not only_pdf_and_klim:
            try:
                anl = compute_cross_section(w, fixed_Q2, beam_energy, file_path=interp_file, verbose=False)
            except Exception:
                anl = np.nan
            try:
                onepi = calculate_1pi_cross_section(w, fixed_Q2, beam_energy, file_path=onepi_file, verbose=False)
            except Exception:
                onepi = np.nan
            anl_xs.append(anl)
            onepi_xs.append(onepi)

    dWdx_vals = dWdx(W_vals, x_vals)
    dxdxi_vals = dxdxi(x_vals, fixed_Q2)
    dWdXi_vals = dWdx_vals * dxdxi_vals
    pdf_lo_xs = np.array(pdf_lo_xs) * dWdXi_vals
    pdf_lo_err = np.array(pdf_lo_err) * dWdXi_vals

    if not only_pdf_and_klim:
        anl_xs = np.array(anl_xs) * dWdXi_vals
        onepi_xs = np.array(onepi_xs) * dWdXi_vals

        strfun_file = f"strfun_data/cs_Q2={fixed_Q2}_E={beam_energy}.dat"
        if not os.path.isfile(strfun_file):
            raise FileNotFoundError(f"Strfun data not found: {strfun_file}")
        strfun_data = np.genfromtxt(strfun_file, delimiter="\t", skip_header=1)
        W_exp = strfun_data[:, 0]
        sigma_exp = strfun_data[:, 1]
        sigma_err = strfun_data[:, 2]

        x_exp = fixed_Q2 / (W_exp**2 - Mp**2 + fixed_Q2)
        xi_exp = xi_from_x(x_exp, fixed_Q2)
        dWdx_exp = dWdx(W_exp, x_exp)
        dxdxi_exp = dxdxi(x_exp, fixed_Q2)
        dWdXi_exp = dWdx_exp * dxdxi_exp
        sigma_dxi = sigma_exp * dWdXi_exp
        sigma_dxi_err = sigma_err * dWdXi_exp

        sort_idx = np.argsort(xi_exp)
        xi_sorted = xi_exp[sort_idx]
        sigma_sorted = sigma_dxi[sort_idx]
        err_sorted = sigma_dxi_err[sort_idx]

        cs_interp = interp1d(xi_sorted, sigma_sorted, kind='cubic', bounds_error=False, fill_value=np.nan)
        err_interp = interp1d(xi_sorted, err_sorted, kind='cubic', bounds_error=False, fill_value=np.nan)

        cs_vals = cs_interp(xi_vals)
        err_vals = err_interp(xi_vals)

    if plot_rga or only_pdf_and_klim:
        klim_file = f"exp_data/InclusiveExpValera_Q2={fixed_Q2}.dat"
        if not os.path.isfile(klim_file):
            raise FileNotFoundError("Klimenko RGA data not found.")
        klim = np.genfromtxt(klim_file, delimiter="\t", skip_header=1)
        W_klim = klim[:, 0]
        sigma_klim = klim[:, 2] * 1e-3
        err_stat = klim[:, 3] * 1e-3
        err_sys = klim[:, 4] * 1e-3
        err_total = np.sqrt(err_stat**2 + err_sys**2)

        x_klim = fixed_Q2 / (W_klim**2 - Mp**2 + fixed_Q2)
        xi_klim = xi_from_x(x_klim, fixed_Q2)
        dWdx_klim = dWdx(W_klim, x_klim)
        dxdxi_klim = dxdxi(x_klim, fixed_Q2)
        dWdXi_klim = dWdx_klim * dxdxi_klim
        sigma_klim_dxi = sigma_klim * dWdXi_klim
        err_klim_dxi = err_total * dWdXi_klim

    # Plot
    plt.figure(figsize=(8, 6))
    legend_text = f"$Q^2$ = {fixed_Q2:.3f} GeV$^2$, $E_{{beam}}$ = {beam_energy:.2f} GeV"
    plt.plot([], [], ' ', label=legend_text)

    if not only_pdf_and_klim:
        plt.plot(xi_vals, anl_xs, label="ANL-Osaka model: full cross section", color="black")
        plt.plot(xi_vals, onepi_xs, "--", label="ANL-Osaka model: $1\\pi$ contribution", color="black", ls="--")
        plt.plot(xi_vals, cs_vals, color="grey", lw=1, label="CLAS+World data (smoothed)")
        plt.fill_between(xi_vals, cs_vals - err_vals, cs_vals + err_vals, color="grey", alpha=0.3)

    plt.plot(xi_vals, pdf_lo_xs, label="LO PDF (CJ15)", color="blue", ls="dotted", lw=1)
    plt.fill_between(xi_vals, pdf_lo_xs - pdf_lo_err, pdf_lo_xs + pdf_lo_err, color="blue", alpha=0.3)

    if plot_rga or only_pdf_and_klim:
        plt.errorbar(xi_klim, sigma_klim_dxi, yerr=err_klim_dxi, fmt="s", ms=3, capsize=2, color="magenta", label="RGA data (V. Klimenko)")

    plt.xlabel("Nachtmann ξ")

    xlim_min = np.min(xi_vals) * 0.95
    xlim_max = np.max(xi_vals) * 1.05
    ylim_max = 1.05 * np.nanmax([
        np.nanmax(pdf_lo_xs + pdf_lo_err),
        np.nanmax(sigma_klim_dxi) if (plot_rga or only_pdf_and_klim) else 0,
        np.nanmax(cs_vals + err_vals) if not only_pdf_and_klim else 0,
        np.nanmax(anl_xs) if not only_pdf_and_klim else 0
    ])
    plt.xlim(xlim_min, xlim_max)
    plt.ylim(0, ylim_max)
    plt.ylabel(r"$d\sigma/dQ^2d\xi$ [$\mu b/GeV^2$]")
    plt.grid(True)
    plt.legend(fontsize="small")
    plt.tight_layout()

    os.makedirs("compare_strfun_x_xi", exist_ok=True)
    fname = f"compare_strfun_x_xi/compare_strfun_vs_xi_Q2={fixed_Q2}_E={beam_energy}.pdf"
    plt.savefig(fname, dpi=300)
    plt.close()
    print("Saved →", fname)
    
    
def anl_osaka_model(fixed_Q2,
                    beam_energy,
                    interp_file="input_data/wempx.dat",
                    onepi_file="input_data/wemp-pi.dat",
                    num_points=200,
                    W_cutoff=2.0,
                    clamp_nonneg=False):
    """
    Plot ANL-Osaka full cross section, 1π contribution, and (full − 1π) ≈ 2π contribution
    vs W at a fixed Q² and beam energy.

    Parameters
    ----------
    fixed_Q2 : float
        Q² value (GeV²).
    beam_energy : float
        Beam energy E (GeV).
    interp_file : str
        Path to the full-channel (W, Q², W1, W2) table (default: "input_data/wempx.dat").
    onepi_file : str
        Path to the 1π-channel (W, Q², W1, W2) table (default: "input_data/wemp-pi.dat").
    num_points : int
        Number of W points for the curve.
    W_cutoff : float
        Upper limit for W in the plot (GeV). The actual upper limit used is min(table_max_W, W_cutoff).
    clamp_nonneg : bool
        If True, max(full − 1π, 0) is plotted/saved for the “2π” curve to avoid small negative artifacts.

    Output
    ------
    - Figure saved to:  anl_osaka_model/anl_osaka_model_Q2={fixed_Q2}_E={beam_energy}.pdf
    - Data table saved to: tables_xsecs/anl_osaka_model_Q2={fixed_Q2}_E={beam_energy}.txt
      Columns: W, xsec_full, xsec_1pi, xsec_2pi
    """
    # Build a W grid from the full table, capped at W_cutoff
    data = np.loadtxt(interp_file)
    W_grid = np.unique(data[:, 0])
    W_vals = np.linspace(W_grid.min(), min(W_grid.max(), W_cutoff), num_points)

    xsec_full, xsec_1pi, xsec_2pi = [], [], []

    for w in W_vals:
        # Full
        try:
            d_full = compute_cross_section(w, fixed_Q2, beam_energy, file_path=interp_file, verbose=False)
        except Exception:
            d_full = np.nan

        # 1π
        try:
            d_1pi = calculate_1pi_cross_section(w, fixed_Q2, beam_energy, file_path=onepi_file, verbose=False)
        except Exception:
            d_1pi = np.nan

        # 2π ≈ (full − 1π)
        d_2pi = np.nan
        # Prefer the dedicated helper; fall back to (full − 1π) if both are available
        try:
            d_2pi = compute_2pi_cross_section(
                W=w, Q2=fixed_Q2, beam_energy=beam_energy,
                full_file_path=interp_file, onepi_file_path=onepi_file,
                verbose=False, clamp_nonneg=clamp_nonneg
            )
        except Exception:
            if not (np.isnan(d_full) or np.isnan(d_1pi)):
                d_2pi = d_full - d_1pi
                if clamp_nonneg and d_2pi < 0:
                    d_2pi = 0.0

        xsec_full.append(d_full)
        xsec_1pi.append(d_1pi)
        xsec_2pi.append(d_2pi)

    xsec_full = np.asarray(xsec_full)
    xsec_1pi  = np.asarray(xsec_1pi)
    xsec_2pi  = np.asarray(xsec_2pi)

    # --------- Plot ---------
    plt.figure(figsize=(8, 6))
    header_text = f"$Q^2$ = {fixed_Q2:.3f} GeV$^2$,  E = {beam_energy:.2f} GeV"
    plt.plot([], [], ' ', label=header_text)

    h_full,  = plt.plot(W_vals, xsec_full,  color="black", lw=2, label="ANL-Osaka: full")
    h_1pi,   = plt.plot(W_vals, xsec_1pi,   color="black", lw=2, ls="--", label="ANL-Osaka: 1π")
    h_2pi,   = plt.plot(W_vals, xsec_2pi,   color="red",   lw=2, ls="-.", label="ANL-Osaka: 2π (full − 1π)")

    plt.xlabel("W (GeV)")
    plt.ylabel(r"$d \sigma / dW\, dQ^2$ ($\mathrm{\mu bn/GeV^3}$)")
    plt.grid(True)
    plt.legend(fontsize="small", loc="upper left")
    plt.tight_layout()

    os.makedirs("anl_osaka_model", exist_ok=True)
    fig_name = f"anl_osaka_model/anl_osaka_model_Q2={fixed_Q2}_E={beam_energy}.pdf"
    plt.savefig(fig_name, dpi=300)
    plt.close()
    print("Saved →", fig_name)


def fit_inclusive_scaling(              # global fit - all Q2 bins 
    exp_folder="exp_data_to_fit",
    Q2_ref=2.774,
    E_ref=10.6,
    W_max_fit=2.0,
    full_file="input_data/wempx.dat",
    onepi_file="input_data/wemp-pi.dat",
    clamp_nonneg_2pi=True,
    save_tag="scalingfit"
):
    """
    Final driver that:
      1) loads and prepares exp data,
      2) evaluates 1π/2π reference shapes at (Q2_ref, E_ref),
      3) builds and runs the global least-squares fit,
      4) writes a text results file,
      5) creates an overview plot across Q² bins including components:
         F2×1π and F2'×2π.
    """
    # 1) Load & prep
    df = read_and_prepare_data(exp_folder, W_max_fit)

    # 2) Reference shapes
    df = compute_reference_shapes_for_df(
        df, Q2_ref=Q2_ref, E_ref=E_ref,
        full_file=full_file, onepi_file=onepi_file,
        clamp_nonneg_2pi=clamp_nonneg_2pi,
    )

    # 3) Build cost & fit
    cost, x_tuple, y, yerr = build_least_squares_from_df(df)
    m = run_minuit(cost)

    # 4) Save results file
    os.makedirs("fit_results", exist_ok=True)
    tag = f"{save_tag}_Q2ref={Q2_ref}_E={E_ref}"
    res_path = f"fit_results/fit_{tag}.txt"
    with open(res_path, "w") as f:
        f.write(f"Scaling fit with fixed 1π/2π shapes at Q²_ref={Q2_ref}, E_ref={E_ref}\n")
        f.write("=" * 72 + "\n")
        f.write(f"Valid minimization: {m.fmin.is_valid}\n")
        f.write(f"EDM: {m.fmin.edm:.3e}  (tol={m.tol})\n")
        f.write(f"Chi2 / NDF: {m.fval:.2f} / {m.ndof} = {m.fval / m.ndof:.3f}\n\n")
        f.write("Parameters:\n")
        for name in m.parameters:
            f.write(f"  {name:>4} = {m.values[name]: .6e} ± {m.errors[name]: .6e}\n")

        f.write("\nCorrelation matrix (full):\n\n")
        names = list(m.parameters)
        f.write(f"{'':>8}" + "".join(f"{n:>12}" for n in names) + "\n")
        for r in names:
            f.write(f"{r:>8}")
            for c in names:
                f.write(f"{m.covariance[r, c]:12.3f}")
            f.write("\n")
    print("Saved →", res_path)

    # 5) Overview plot with components
    os.makedirs("fit_plots", exist_ok=True)
    q2_bins = [float(q) for q in sorted(df["Q2"].unique())]
    n = len(q2_bins)
    ncols = 3 if n >= 3 else n
    nrows = int(np.ceil(n / ncols)) if ncols else 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False)
    axes = axes.flatten()

    params = {k: m.values[k] for k in m.parameters}
    predict_total = make_predict_curve_function(
        Q2_ref=Q2_ref, E_ref=E_ref,
        onepi_file=onepi_file, full_file=full_file,
        clamp_nonneg_2pi=clamp_nonneg_2pi,
        fitted_params=params,
    )
    predict_1pi, predict_2pi = make_component_curve_functions(
        Q2_ref=Q2_ref, E_ref=E_ref,
        onepi_file=onepi_file, full_file=full_file,
        clamp_nonneg_2pi=clamp_nonneg_2pi,
        fitted_params=params,
    )

    W_dense = np.linspace(max(1.05, float(df["W"].min())), W_max_fit, 300)

    for ax, q2 in zip(axes, q2_bins):
        sub = df[df["Q2"] == q2]
        x = sub["W"].to_numpy(dtype=float)
        y = sub["XSEC"].to_numpy(dtype=float)
        ye = sub["uncertainty"].to_numpy(dtype=float)

        # Data
        ax.errorbar(x, y, yerr=ye, fmt="o", ms=3, capsize=1, label=f"Data Q²={q2:.3f}")

        # Fit and components
        ax.plot(W_dense, predict_total(W_dense, q2), "k-",  lw=1.8, label="Fit (total)")
        ax.plot(W_dense, predict_1pi(W_dense, q2),   "--",  lw=1.4, color="tab:blue",   label=r"$F_2\times 1\pi$")
        ax.plot(W_dense, predict_2pi(W_dense, q2),   ":",   lw=1.6, color="tab:orange", label=r"$F_2'\times 2\pi$")

        ax.set_title(f"Q² = {q2:.3f} GeV²")
        ax.set_xlabel("W (GeV)")
        ax.set_ylabel(r"$d^2\sigma/dW\,dQ^2$ [$\mu b/\mathrm{GeV}^3$]")
        ax.grid(True)
        ax.legend(fontsize=8)
        ax.axvline(W_max_fit, color="gray", ls="--", lw=1)

    # hide any unused axes
    for j in range(len(q2_bins), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"Global scaling fit (Q²_ref={Q2_ref}, E_ref={E_ref})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = f"fit_plots/fit_{tag}_overview.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print("Saved →", plot_path)
    
    
def fit_inclusive_scaling_per_bin(
    exp_folder="exp_data_to_fit",
    Q2_ref=2.774,
    E_ref=10.6,
    W_max_fit=2.0,
    full_file="input_data/wempx.dat",
    onepi_file="input_data/wemp-pi.dat",
    clamp_nonneg_2pi=True,
    save_tag="individual_Q2_fit",
):
    """
    Fit each Q² bin independently and:
      - Save ONE compact text file with all bins' results
      - Save ONE 3x3 overview plot (all bins) with Data, Total, F2×1π, F2'×2π, and χ²/ndf on each panel
      - Save a 2x3 plot of coefficients vs Q² (c1..c3, c1'..c3')

    Outputs:
      - fit_results/fit_individual_Q2_fit_Q2ref=<Q2_ref>_E=<E_ref>.txt
      - fit_results/fit_individual_Q2_fit_summary_Q2ref=<Q2_ref>_E=<E_ref>.csv
      - fit_plots/fit_individual_Q2_fit_Q2ref=<Q2_ref>_E=<E_ref>_overview.png
      - fit_plots/fit_individual_Q2_fit_coeffs_Q2ref=<Q2_ref>_E=<E_ref>.png
    """
    print(f"→ Running per-Q² (individual) scaling fits over {exp_folder}/ …")

    # 1) Load & prepare once
    df = read_and_prepare_data(exp_folder, W_max_fit)

    # 2) Precompute W-only reference shapes once at (Q2_ref, E_ref)
    df = compute_reference_shapes_for_df(
        df, Q2_ref=Q2_ref, E_ref=E_ref,
        full_file=full_file, onepi_file=onepi_file,
        clamp_nonneg_2pi=clamp_nonneg_2pi,
    )

    os.makedirs("fit_results", exist_ok=True)
    os.makedirs("fit_plots",   exist_ok=True)

    # 3) Fit each bin
    q2_bins = [float(q) for q in sorted(df["Q2"].unique())]
    bin_records = []   # to plot later
    summary_rows = []  # to write CSV and compact TXT

    for q2 in q2_bins:
        sub = df[df["Q2"] == q2].copy()
        npts = len(sub)
        if npts < 8:
            print(f"  [warn] Q²={q2:.3f} has only {npts} points; 6 params may be underconstrained.")

        cost, *_ = build_least_squares_from_df(sub)
        m = run_minuit(
        cost,
        multi_start=100,        # try e.g. 20 random restarts
        jitter_scale=0.25,     # tweak if needed
        strategy=2,            # more thorough line searches
        seed=50                # reproducible randomness (optional)
        )
        # Capture minimization status
        min_valid = bool(m.fmin.is_valid)
        edm = float(m.fmin.edm)
        tol = float(m.tol)
        converged = edm < tol

        # Store results for later plotting/printing
        params = {k: m.values[k] for k in m.parameters}
        perr   = {k: m.errors[k] for k in m.parameters}
        rec = {
            "q2": q2,
            "sub": sub,
            "params": params,
            "perr": perr,
            "chi2": m.fval,
            "ndf": m.ndof,
            "chi2_ndf": (m.fval / m.ndof if m.ndof else np.nan),
            "npts": npts,
            "min_valid": min_valid,
            "edm": edm,
            "tol": tol,
            "converged": converged
        }
        bin_records.append(rec)

        # Summary row
        row = {"Q2": q2, "N": npts, "chi2": m.fval, "ndf": m.ndof,
               "chi2_ndf": (m.fval / m.ndof if m.ndof else np.nan)}
        for name in m.parameters:
            row[name] = m.values[name]
            row[name + "_err"] = m.errors[name]
        summary_rows.append(row)

    # 4) Write ONE compact TXT with all bins
    tag = f"{save_tag}_Q2ref={Q2_ref}_E={E_ref}".replace(" ", "")
    txt_path = f"fit_results/fit_{tag}.txt"
    with open(txt_path, "w") as f:
        f.write(f"Individual Q² scaling fits  (reference shapes at Q²_ref={Q2_ref}, E_ref={E_ref})\n")
        f.write("=" * 86 + "\n\n")
        for rec in bin_records:
            q2 = rec["q2"]
            f.write(f"Q² = {q2:.3f} GeV²   N = {rec['npts']}   χ²/ndf = {rec['chi2_ndf']:.3f}"
                    f"  (χ²={rec['chi2']:.2f}, ndf={rec['ndf']})\n")

            # NEW: status lines
            f.write("Minimization status:\n")
            f.write(f"  Valid minimization: {rec['min_valid']}\n")
            f.write(f"  Converged (EDM < tol): {rec['converged']} "
                    f"(EDM = {rec['edm']:.2e}, tol = {rec['tol']})\n")

            p, e = rec["params"], rec["perr"]
            f.write("  c1  = {: .6e} ± {: .6e}   c2  = {: .6e} ± {: .6e}   c3  = {: .6e} ± {: .6e}\n"
                    .format(p["c1"], e["c1"], p["c2"], e["c2"], p["c3"], e["c3"]))
            f.write("  c1' = {: .6e} ± {: .6e}   c2' = {: .6e} ± {: .6e}   c3' = {: .6e} ± {: .6e}\n\n"
                    .format(p["c1p"], e["c1p"], p["c2p"], e["c2p"], p["c3p"], e["c3p"]))
    print("Saved →", txt_path)

    # 5) Write summary CSV as well (handy for further analysis)
    summ = pd.DataFrame(summary_rows).sort_values("Q2")
    csv_path = f"fit_results/fit_{tag}_summary.csv"
    summ.to_csv(csv_path, index=False)
    print("Saved →", csv_path)

    # 6) One 3x3 overview plot with Data + Total + components + χ²/ndf per panel
    n = len(q2_bins)
    ncols = 3 if n >= 3 else n
    nrows = int(np.ceil(n / ncols)) if ncols else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows), squeeze=False)
    axes = axes.flatten()

    for ax, rec in zip(axes, bin_records):
        q2 = rec["q2"]
        sub = rec["sub"]

        predict_total = make_predict_curve_function(
            Q2_ref=Q2_ref, E_ref=E_ref,
            onepi_file=onepi_file, full_file=full_file,
            clamp_nonneg_2pi=clamp_nonneg_2pi,
            fitted_params=rec["params"],
        )
        predict_1pi, predict_2pi = make_component_curve_functions(
            Q2_ref=Q2_ref, E_ref=E_ref,
            onepi_file=onepi_file, full_file=full_file,
            clamp_nonneg_2pi=clamp_nonneg_2pi,
            fitted_params=rec["params"],
        )

        W_dense = np.linspace(max(1.05, float(sub["W"].min())), W_max_fit, 300)
        x  = sub["W"].to_numpy(float)
        y  = sub["XSEC"].to_numpy(float)
        ye = sub["uncertainty"].to_numpy(float)

        # data + curves
        ax.errorbar(x, y, yerr=ye, fmt="o", ms=3, capsize=1, label=f"Data")
        ax.plot(W_dense, predict_total(W_dense, q2), "k-",  lw=1.8, label="Fit (total)")
        ax.plot(W_dense, predict_1pi(W_dense, q2),   "--",  lw=1.4, color="tab:blue",   label=r"$F_2\times 1\pi$")
        ax.plot(W_dense, predict_2pi(W_dense, q2),   ":",   lw=1.6, color="tab:orange", label=r"$F_2'\times 2\pi$")

        # cosmetics
        chi2_ndf = rec["chi2_ndf"]
        ax.set_title(f"Q² = {q2:.3f} GeV² "rf" $\chi^2/\mathrm{{ndf}} = {chi2_ndf:.3f}$")
        ax.set_xlabel("W (GeV)")
        ax.set_ylabel(r"$d^2\sigma/dW\,dQ^2$ [$\mu b/\mathrm{GeV}^3$]")
        ax.grid(True)
        ax.axvline(W_max_fit, color="gray", ls="--", lw=1)

        # Lean legend to first row only (avoids repetition); or just keep it small
        ax.legend(fontsize=8)

    # hide unused axes
    for j in range(len(bin_records), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"Individual Q² fits (reference: Q²={Q2_ref}, E={E_ref})", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    overview_path = f"fit_plots/fit_{tag}_overview.png"
    plt.savefig(overview_path, dpi=300)
    plt.close()
    print("Saved →", overview_path)

    # 7) Coefficient-vs-Q² plots (6 panels with error bars)
    pretty = {
        "c1": r"$c_1$", "c2": r"$c_2$", "c3": r"$c_3$",
        "c1p": r"$c'_1$", "c2p": r"$c'_2$", "c3p": r"$c'_3$",
    }
    keys = ["c1", "c2", "c3", "c1p", "c2p", "c3p"]

    fig2, axes2 = plt.subplots(2, 3, figsize=(12, 7), squeeze=False)
    axes2 = axes2.flatten()

    q2_arr = np.array([rec["q2"] for rec in bin_records])
    for i, key in enumerate(keys):
        ax = axes2[i]
        vals = np.array([rec["params"][key] for rec in bin_records])
        errs = np.array([rec["perr"][key]   for rec in bin_records])
        ax.errorbar(q2_arr, vals, yerr=errs, fmt="o-", ms=4, capsize=2)
        ax.set_xlabel(r"$Q^2$ (GeV$^2$)")
        ax.set_ylabel(pretty[key])
        ax.set_title(f"{pretty[key]} vs $Q^2$")
        ax.grid(True)

    fig2.tight_layout()
    coeffs_path = f"fit_plots/fit_{tag}_coeffs.png"
    plt.savefig(coeffs_path, dpi=300)
    plt.close()
    print("Saved →", coeffs_path)
