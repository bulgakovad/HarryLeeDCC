import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions_pdf import  get_nlo_pdf_interpolators
from functions_anl_osaka import sigma_LT_to_F2_AO_model

import os
from scipy.interpolate import CubicSpline
from scipy.integrate import quad

from pathlib import Path

import os


E_beam = 10.6
M = 0.9382720813
alpha = 1/137.035999084
four_pi2_alpha = 4.0*np.pi**2*alpha

def x_of_W(W,Q2): return Q2 / (W*W - M*M + Q2)


def F2_from_xsect_data(Q2_value, R_source):
    

    # --- read file (headerless) and coerce numerics ---
    in_path = f"exp_data/InclusiveExpValera_Q2={Q2_value}.dat"
    df = pd.read_csv(in_path, sep=r"\s+", header=None,
                     names=["W", "eps", "sigma_dW_dQ2", "err_stat", "err_sys"])
    df = df.apply(pd.to_numeric, errors="coerce").dropna()

    # arrays
    W      = df["W"].to_numpy()
    eps    = df["eps"].to_numpy()
    s_dWQ  = df["sigma_dW_dQ2"].to_numpy()
    e_stat = df["err_stat"].to_numpy()
    e_sys  = df["err_sys"].to_numpy()
    e_full = np.sqrt(e_stat**2 + e_sys**2)
    
    # R_LT
    if R_source == "AO":
        r_path = f"tables_from_Yannick/exp_binning/{R_source}/Wdist_Q2_{Q2_value}_GLOBAL_LT.dat"
    elif R_source == "Astrid":
        r_path = f"tables_from_Yannick/exp_binning/{R_source}/Wdist_Q2_{Q2_value}_GLOBAL_LT.dat"
    elif R_source == "CJ15":
        r_path = f"tables_from_Yannick/exp_binning/{R_source}/Wdist_Q2_{Q2_value}_GLOBAL_LT.dat"
    else:
        raise ValueError(f"Unknown R_source='{R_source}'. Use 'AO', 'Astrid', or 'CJ15'.")

    r_df = pd.read_csv(r_path, sep=r"\s+", comment="#", header=None)

    # columns (0-based): 0=W, 5=R_LT, 6=dR_LT
    W_R  = r_df.iloc[:, 0].to_numpy()
    R_LT = r_df.iloc[:, 5].to_numpy()
    dR_LT   = r_df.iloc[:, 6].to_numpy()

    # kinematics
    nu   = (W**2 + Q2_value - M**2) / (2.0*M)
    E_p  = E_beam - nu
    K    = (W**2 - M**2) / (2.0*M)
    x    = x_of_W(W, Q2_value)
    rho2 = 1.0 + (4.0*M**2*x**2) / Q2_value

    # flux & Jacobian
    Gamma_v = (alpha/(2.0*np.pi**2)) * (E_p/E_beam) * (K/Q2_value) * (1.0/(1.0 - eps))
    J = (W * np.pi) / (M * E_beam * E_p)

    # σ_U and units
    sigma_U = s_dWQ / (Gamma_v * J)
    sigma_U_err = e_full / (Gamma_v * J)
    NB_TO_GEV2 = 1.0 / 389_379.365
    sigma_U *= NB_TO_GEV2
    sigma_U_err *= NB_TO_GEV2
    
    pref = (K*M) / four_pi2_alpha  # KM/(4π²α)

    F2 = pref * (2.0*x/rho2) * ((1.0 + R_LT)/(1.0 + eps*R_LT)) * sigma_U
    
    A = pref * (2.0*x/rho2)
    f = (1.0 + R_LT) / (1.0 + eps*R_LT) # Auxillary funcrion f(R) to calculate uncertainty

    df_dR = (1.0 - eps) / (1.0 + eps*R_LT)**2
    f_err = np.abs(df_dR) * dR_LT

    F2_err_from_sigma = A * f * sigma_U_err
    F2_err_from_R     = A * sigma_U * f_err

    F2_err = np.sqrt(F2_err_from_sigma**2 + F2_err_from_R**2)
    
    
    return pd.DataFrame({"W": W, "x": x, "F2": F2, "F2_err": F2_err})




def calc_trunc_moment_data(Q2_value, region, R_source, n=2, error_mode="correlated"):
    """
    Truncated Cornwall–Norton moment from data:
        M_n(Q2; region) = ∫_{x_lo}^{x_hi} x^{n-2} F2(x,Q2) dx
    Region is defined via W-bounds, converted to x-bounds at fixed Q2.

    error_mode:
      - "segment_uncorrelated" : uncorrelated trapezoid propagation
      - "point_uncorrelated": uncorrelated pointwise propagation 
      - "correlated": fully correlated envelope (y -> y ± dy)

    Returns DataFrame with: Q2, region, n, x_lo, x_hi, moment, error
    """

    M = 0.9382720813

    # --- get data (x, F2, dF2) ---
    df = F2_from_xsect_data(Q2_value, R_source = R_source)
    x  = df["x"].to_numpy(dtype=float)
    F2 = df["F2"].to_numpy(dtype=float)
    dF = df["F2_err"].to_numpy(dtype=float)

    mask = np.isfinite(x) & np.isfinite(F2) & np.isfinite(dF)
    x, F2, dF = x[mask], F2[mask], dF[mask]
    if x.size < 2:
        return pd.DataFrame([{
            "Q2": Q2_value, "region": region, "n": n,
            "x_lo": np.nan, "x_hi": np.nan,
            "moment": 0.0, "error": 0.0
        }])

    # sort by increasing x
    o = np.argsort(x)
    x, F2, dF = x[o], F2[o], dF[o]

    # --- region -> W bounds ---
    W_min_data = 1.15
    Wmax1 = 1.35
    Wmin2 = Wmax1
    Wmax2 = 1.60
    Wmin3 = Wmax2  # No crutch here, use exact boundary
    Wmax3 = 2.0
    W_max = 2.25 if np.isclose(Q2_value, 9.699, atol=1e-3) else 2.50

    reg = str(region).lower().strip()
    region_map = {
        "1": (W_min_data, Wmax1), "r1": (W_min_data, Wmax1), "first": (W_min_data, Wmax1), "1st": (W_min_data, Wmax1),
        "2": (Wmin2, Wmax2),      "r2": (Wmin2, Wmax2),      "second": (Wmin2, Wmax2),      "2nd": (Wmin2, Wmax2),
        "3": (Wmin3, Wmax3),      "r3": (Wmin3, Wmax3),      "third": (Wmin3, Wmax3),       "3rd": (Wmin3, Wmax3),
        "partial": (W_min_data, Wmax3), "part": (W_min_data, Wmax3),
        "tail": (Wmax3, W_max),
        "full": (W_min_data, W_max), "all": (W_min_data, W_max),
    }
    if reg not in region_map:
        raise ValueError(f"Unknown region='{region}'. Use one of: {sorted(region_map.keys())}")

    W_lo, W_hi = region_map[reg]

    # --- W -> x bounds at this Q2 ---
    def x_of_W(W):
        return Q2_value / (W*W - M*M + Q2_value)

    x1 = x_of_W(W_lo)
    x2 = x_of_W(W_hi)
    x_lo_bound = min(x1, x2)
    x_hi_bound = max(x1, x2)

    # --- intersect with available data x-range ---
    lo = max(x_lo_bound, x[0])
    hi = min(x_hi_bound, x[-1])
    if lo >= hi:
        return pd.DataFrame([{
            "Q2": Q2_value, "region": region, "n": n,
            "x_lo": lo, "x_hi": hi,
            "moment": 0.0, "error": 0.0
        }])

    # --- build segment including interpolated endpoints ---
    F2_lo = np.interp(lo, x, F2); F2_hi = np.interp(hi, x, F2)
    dF_lo = np.interp(lo, x, dF); dF_hi = np.interp(hi, x, dF)

    mid = (x > lo) & (x < hi)
    x_seg  = np.concatenate(([lo], x[mid], [hi]))
    F2_seg = np.concatenate(([F2_lo], F2[mid], [F2_hi]))
    dF_seg = np.concatenate(([dF_lo], dF[mid], [dF_hi]))

    # --- moment integrand and trapezoid integral ---
    w  = x_seg**(n - 2)      # CN weight
    y  = w * F2_seg
    dy = w * dF_seg

    moment = float(np.trapz(y, x_seg))

    # --- error estimate ---
    mode = str(error_mode).lower().strip()
    if mode in ["segment_uncorrelated"]:
        # (keep your existing method unchanged)
        dx = np.diff(x_seg)
        err2 = np.sum((0.5*dx)**2 * (dy[:-1]**2 + dy[1:]**2))
        error = float(np.sqrt(err2))
    elif mode in ["point_uncorrelated", "points", "pointwise", "wts"]:
        # Correct for uncorrelated point-to-point errors: Var(I)=sum_k (w_k*dy_k)^2
        dx = np.diff(x_seg)

        wts = np.zeros_like(x_seg, dtype=float)
        wts[0] = 0.5 * dx[0]
        wts[1:-1] = 0.5 * (dx[:-1] + dx[1:])
        wts[-1] = 0.5 * dx[-1]

        err2 = np.sum((wts * dy)**2)
        error = float(np.sqrt(err2))

    elif mode in ["correlated", "corr", "c", "envelope"]:
        # fully correlated envelope: y -> y ± dy
        I_hi = float(np.trapz(y + dy, x_seg))
        I_lo = float(np.trapz(y - dy, x_seg))
        error = 0.5 * (I_hi - I_lo)

    else:
        raise ValueError("error_mode must be 'uncorrelated' or 'correlated'")

    return pd.DataFrame([{
        "Q2": Q2_value, "region": region, "n": n,
        "x_lo": lo, "x_hi": hi,
        "moment": moment, "error": error
    }])



def plot_epsilon_vs_W_grid(
    q2_values=(2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699),
    data_dir="exp_data",
    out_png="epsilon_vs_W_Q2grid.png",
    w_min=1.07,
    w_max=2.50,
):
    """
    Make a 3x3 panel plot of ε (virtual-photon polarization) vs W
    from Klimenko files: {data_dir}/InclusiveExpValera_Q2={Q2}.dat

    Each file is parsed as whitespace-separated with columns:
        W, eps, sigma_dW_dQ2, err_stat, err_sys

    Args:
        q2_values: iterable of Q^2 values (GeV^2) to plot (len should be 9 for a 3x3 grid).
        data_dir: directory where the Klimenko files live.
        out_png : filename for the saved figure.
        w_min, w_max: x-range to display (GeV).

    Returns:
        dict mapping Q2 -> DataFrame with columns ["W","eps"] actually plotted.
    """
    q2_values = list(q2_values)
    if len(q2_values) != 9:
        raise ValueError("Please provide exactly 9 Q^2 values for a 3x3 panel plot.")

    results = {}

    fig, axes = plt.subplots(3, 3, figsize=(11, 9), sharex=True, sharey=True, constrained_layout=True)

    for ax, Q2 in zip(axes.ravel(), q2_values):
        fpath = Path(data_dir) / f"InclusiveExpValera_Q2={Q2}.dat"

        if not fpath.exists():
            ax.text(0.5, 0.5, "missing file", ha="center", va="center", transform=ax.transAxes, color="crimson")
            ax.set_title(f"Q²={Q2:g} GeV²")
            ax.grid(alpha=0.25)
            continue

        # read and coerce numerics
        df = pd.read_csv(
            fpath,
            sep=r"\s+",
            header=None,
            names=["W", "eps", "sigma_dW_dQ2", "err_stat", "err_sys"],
            engine="python",
        )
        df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=["W", "eps"])

        # restrict W-range for display
        df = df[(df["W"] >= w_min) & (df["W"] <= w_max)].copy()
        if df.empty:
            ax.text(0.5, 0.5, "no points in range", ha="center", va="center", transform=ax.transAxes, color="gray")
            ax.set_title(f"Q²={Q2:g} GeV²")
            ax.grid(alpha=0.25)
            continue

        df = df.sort_values("W")
        results[Q2] = df[["W", "eps"]].reset_index(drop=True)

        ax.plot(df["W"].to_numpy(), df["eps"].to_numpy(), marker="o", ms=3, lw=1)
        ax.set_title(f"Q²={Q2:g} GeV²")
        ax.grid(alpha=0.25)

    # common labels and limits
    for ax in axes[-1]:
        ax.set_xlabel("W  [GeV]")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\varepsilon$")

    axes[0, 0].set_ylim(0.0, 1.05)
    axes[0, 0].set_xlim(w_min, w_max)

    fig.suptitle(r"$\varepsilon(W)$ from Klimenko data", y=1.02, fontsize=14)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    return results


def calculate_epsilon_valerii(Q2_value):
    Q2 = float(Q2_value)
    q2_tag = str(Q2_value)
    data_path = "exp_data"
    filename = f"InclusiveExpValera_Q2={q2_tag}.dat"
    in_path = os.path.join(data_path, filename)
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"Cannot find input file: {in_path}")

    data = np.loadtxt(in_path, skiprows=1)
    W = data[:, 0]
    eps = data[:, 1]
    sigma = data[:, 2]
    error = data[:, 3]
    sys_error = data[:, 4]
    
    #---
    # kinematics needed for epsilon & Gamma
    nu = (W**2 + Q2 - M**2) / (2.0 * M)
    Eprime = E_beam - nu

    # theta from Q2 = 4 E E' sin^2(theta/2)
    sin2 = Q2 / (4.0 * E_beam * Eprime)

    tan2 = sin2 / (1.0 - sin2)

    my_eps = 1.0 / (1.0 + 2.0 * (1.0 + (nu**2)/Q2) * tan2)
    
    df = pd.DataFrame({"W": W, "sigma": sigma, "error": error, "sys_error": sys_error, "eps": eps, "my_eps": my_eps})
    
    out_dir = "checking_epsilon/"
    os.makedirs(out_dir, exist_ok=True)
    out_filename = out_dir + filename + "_TEST_EPSILON.csv"
    df.to_csv(out_filename, sep="\t", index=False)
    #print(df)

def calculate_epsilon_yannick(Q2_value):
    Q2 = float(Q2_value)
    q2_tag = str(Q2_value)
    data_path = "tables_from_Yannick/fine_binning/AO/"
    filename = f"Wdist_Q2_{q2_tag}_GLOBAL_LT.dat"
    in_path = os.path.join(data_path, filename)
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"Cannot find input file: {in_path}")

    data = np.loadtxt(in_path, skiprows=1)
    W = data[:, 0]
    sigma_tot = data[:, 1]
    sigma_T = data[:, 2]
    eps_sigma_L = data[:, 3]
    sigma_L = data[:, 4]
    R_LT = data[:, 5]
    dR_LT = data[:, 6]
    
    #New column
    yannick_eps = eps_sigma_L / sigma_L

    #---
    # kinematics needed for epsilon & Gamma
    nu = (W**2 + Q2 - M**2) / (2.0 * M)
    Eprime = E_beam - nu

    # theta from Q2 = 4 E E' sin^2(theta/2)
    sin2 = Q2 / (4.0 * E_beam * Eprime)

    tan2 = sin2 / (1.0 - sin2)

    my_eps = 1.0 / (1.0 + 2.0 * (1.0 + (nu**2)/Q2) * tan2)

    df = pd.DataFrame({"W": W,  "eps_sigma_L": eps_sigma_L, "sigma_L": sigma_L, "yannick_eps = eps_sigma_L / sigma_L": yannick_eps, "my_eps": my_eps})

    out_dir = "checking_epsilon/"
    os.makedirs(out_dir, exist_ok=True)
    out_filename = out_dir + filename + "_TEST_EPSILON.csv"
    df.to_csv(out_filename, sep="\t", index=False)
    #print(df.head(50))

def plot_R_vs_W_grid(Q2_value, out_dir = "checking_R_LT", W_cutoff=2.0):
    Q2_value = float(Q2_value)
    data_dir_AO="tables_from_Yannick/exp_binning/AO"
    data_dir_Astrid="tables_from_Yannick/exp_binning/Astrid"
    data_dir_CJ15 = "tables_from_Yannick/exp_binning/CJ15"
    
    fpath_AO = Path(data_dir_AO) / f"Wdist_Q2_{Q2_value}_GLOBAL_LT.dat"
    fpath_Astrid = Path(data_dir_Astrid) / f"Wdist_Q2_{Q2_value}_GLOBAL_LT.dat"
    fpath_CJ15 = Path(data_dir_CJ15) / f"Wdist_Q2_{Q2_value}_GLOBAL_LT.dat"
    
    data_AO = np.loadtxt(fpath_AO, skiprows=1)
    W = data_AO[:, 0]
    R_LT_AO = data_AO[:, 5]
    dR_LT_AO = data_AO[:, 6]

    data_Astrid = np.loadtxt(fpath_Astrid, skiprows=1)
    R_LT_Astrid = data_Astrid[:, 5]
    dR_LT_Astrid = data_Astrid[:, 6]

    data_CJ15 = np.loadtxt(fpath_CJ15, skiprows=1)
    R_LT_CJ15 = data_CJ15[:, 5]
    dR_LT_CJ15 = data_CJ15[:, 6]
    # Plotting
    mask = (W <= W_cutoff)
    W = W[mask]
    R_LT_AO = R_LT_AO[mask]
    R_LT_Astrid = R_LT_Astrid[mask]
    R_LT_CJ15 = R_LT_CJ15[mask]  
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(W, R_LT_AO, marker='o', label='AO')
    ax.scatter(W, R_LT_Astrid, marker='s', label='Astrid')
    ax.scatter(W, R_LT_CJ15, marker='^', label='CJ15')
    ax.set_xlabel('W [GeV]')
    ax.set_ylabel('R_LT')
    ax.set_title(f'R_LT vs W at Q²={Q2_value} GeV²')
    ax.legend()
    ax.grid(alpha=0.25)
    os.makedirs(out_dir, exist_ok=True)
    out_png=f"R_LT_vs_W_{Q2_value}_comparison.pdf"
    out_filepath = Path(out_dir) / out_png
    fig.savefig(out_filepath, dpi=300)
    plt.close(fig)
    

    
    
def estimate_bin_size_err_data(Q2_values, regions,
                               R_source,
                               n=2,
                               E_beam=10.6,
                               ao_in_dir="tables_from_Yannick/fine_binning/AO",
                               convert_ub_to_GeV2=True,
                               divide_by_Gamma=True,
                               epsabs=1e-8,
                               epsrel=1e-6,
                               quad_limit=200):
    """
    Estimate finite-binning (data-grid) integration bias for truncated CN moments,
    using AO model as the 'truth' shape.

    For each Q2 and each region:
      1) "continuous" AO moment: integrate x^{n-2} * F2_AO(x,Q2) dx via quad over spline
      2) "coarse" AO moment: evaluate the same spline on THE SAME x-grid used for data trapz,
         then integrate with np.trapz on that grid
      3) return ratio = continuous / coarse and frac_diff = (continuous - coarse)/continuous

    Parameters
    ----------
    Q2_values : iterable of float
        Q2 values to process.
    regions : iterable of str
        Regions: "1st","2nd","3rd","partial","tail","full", etc. (same mapping as your code).
    R_source : str
        Passed to F2_from_xsect_data(Q2_value, R_source=R_source) to get the DATA x-grid.
        (We only use its x grid; the AO model provides y-values.)
    n : int
        CN moment order.
    E_beam, ao_in_dir, convert_ub_to_GeV2, divide_by_Gamma : as in your AO extraction.
    epsabs, epsrel, quad_limit : quad settings.

    Returns
    -------
    pandas.DataFrame with columns:
        Q2, region, n, x_lo, x_hi,
        moment_AO_cont, moment_AO_trapz_on_data_grid,
        ratio_cont_over_trapz, frac_diff
    """

    M = 0.9382720813

    # --- region -> W bounds (same logic as in your functions) ---
    W_min_data = 1.15
    Wmax1 = 1.35
    Wmin2 = Wmax1
    Wmax2 = 1.60
    Wmin3 = Wmax2
    Wmax3 = 2.0

    def region_bounds_W(Q2, region):
        W_max = 2.25 if np.isclose(Q2, 9.699, atol=1e-3) else 2.50
        reg = str(region).lower().strip()
        region_map = {
            "1": (W_min_data, Wmax1), "r1": (W_min_data, Wmax1), "first": (W_min_data, Wmax1), "1st": (W_min_data, Wmax1),
            "2": (Wmin2, Wmax2),      "r2": (Wmin2, Wmax2),      "second": (Wmin2, Wmax2),      "2nd": (Wmin2, Wmax2),
            "3": (Wmin3, Wmax3),      "r3": (Wmin3, Wmax3),      "third": (Wmin3, Wmax3),       "3rd": (Wmin3, Wmax3),
            "partial": (W_min_data, Wmax3), "part": (W_min_data, Wmax3),
            "tail": (Wmax3, W_max),
            "full": (W_min_data, W_max), "all": (W_min_data, W_max),
        }
        if reg not in region_map:
            raise ValueError(f"Unknown region='{region}'. Use one of: {sorted(region_map.keys())}")
        return region_map[reg]

    def x_of_W(W, Q2):
        return Q2 / (W*W - M*M + Q2)

    rows = []

    for Q2_value in Q2_values:
        Q2 = float(Q2_value)

        # --- DATA x-grid (we only need x) ---
        df_data = F2_from_xsect_data(Q2_value, R_source=R_source)
        x_data = df_data["x"].to_numpy(dtype=float)
        mask = np.isfinite(x_data)
        x_data = x_data[mask]
        if x_data.size < 2:
            # no usable grid -> fill NaNs for all regions
            for region in regions:
                rows.append({
                    "Q2": Q2_value, "region": region, "n": n,
                    "x_lo": np.nan, "x_hi": np.nan,
                    "moment_AO_cont": np.nan,
                    "moment_AO_trapz_on_data_grid": np.nan,
                    "ratio_cont_over_trapz": np.nan,
                    "frac_diff": np.nan
                })
            continue

        # sort increasing x
        x_data = np.sort(x_data)

        # --- AO model F2(W) on its native grid, then build spline in x ---
        W_ao, F2W_ao = sigma_LT_to_F2_AO_model(
            fixed_Q2=Q2,
            W_out=None,
            E_beam=E_beam,
            in_dir=ao_in_dir,
            convert_ub_to_GeV2=convert_ub_to_GeV2,
            divide_by_Gamma=divide_by_Gamma
        )
        W_ao = np.asarray(W_ao, dtype=float)
        F2W_ao = np.asarray(F2W_ao, dtype=float)

        x_ao = Q2 / (W_ao*W_ao - M*M + Q2)
        m2 = np.isfinite(x_ao) & np.isfinite(F2W_ao)
        x_ao = x_ao[m2]
        F2_ao = F2W_ao[m2]

        if x_ao.size < 2:
            for region in regions:
                rows.append({
                    "Q2": Q2_value, "region": region, "n": n,
                    "x_lo": np.nan, "x_hi": np.nan,
                    "moment_AO_cont": np.nan,
                    "moment_AO_trapz_on_data_grid": np.nan,
                    "ratio_cont_over_trapz": np.nan,
                    "frac_diff": np.nan
                })
            continue

        o = np.argsort(x_ao)
        x_ao = x_ao[o]
        F2_ao = F2_ao[o]

        x_u, idx = np.unique(x_ao, return_index=True)
        F2_u = F2_ao[idx]

        if x_u.size < 2:
            for region in regions:
                rows.append({
                    "Q2": Q2_value, "region": region, "n": n,
                    "x_lo": np.nan, "x_hi": np.nan,
                    "moment_AO_cont": np.nan,
                    "moment_AO_trapz_on_data_grid": np.nan,
                    "ratio_cont_over_trapz": np.nan,
                    "frac_diff": np.nan
                })
            continue

        F2_spline_x = CubicSpline(x_u, F2_u, bc_type="natural", extrapolate=False)

        def integrand(xv):
            f2 = F2_spline_x(xv)
            if not np.isfinite(f2):
                return 0.0
            return (xv ** (n - 2)) * float(f2)

        # --- loop over regions ---
        for region in regions:
            W_lo, W_hi = region_bounds_W(Q2, region)
            xb1 = x_of_W(W_lo, Q2)
            xb2 = x_of_W(W_hi, Q2)
            x_lo_bound = min(xb1, xb2)
            x_hi_bound = max(xb1, xb2)

            # intersect bounds with BOTH: data-x coverage and AO-x coverage
            lo = max(x_lo_bound, float(x_data[0]), float(x_u[0]))
            hi = min(x_hi_bound, float(x_data[-1]), float(x_u[-1]))

            if not (lo < hi):
                rows.append({
                    "Q2": Q2_value, "region": region, "n": n,
                    "x_lo": lo, "x_hi": hi,
                    "moment_AO_cont": 0.0,
                    "moment_AO_trapz_on_data_grid": 0.0,
                    "ratio_cont_over_trapz": np.nan,
                    "frac_diff": np.nan
                })
                continue

            # --- 1) continuous AO integral over [lo, hi] ---
            I_cont, _ = quad(integrand, lo, hi,
                             epsabs=epsabs, epsrel=epsrel, limit=quad_limit)
            I_cont = float(I_cont)

            # --- 2) AO integrated with trapz on the SAME x-grid as data integration ---
            # Build x_seg exactly like you do for data: endpoints + interior data points in (lo,hi)
            mid = (x_data > lo) & (x_data < hi)
            x_seg = np.concatenate(([lo], x_data[mid], [hi]))

            # Evaluate AO spline at those points
            F2_seg = F2_spline_x(x_seg)
            # Safety: if any NaNs due to extrapolate=False (shouldn't happen due to intersection)
            finite = np.isfinite(F2_seg) & np.isfinite(x_seg)
            x_seg = x_seg[finite]
            F2_seg = F2_seg[finite]

            if x_seg.size < 2:
                rows.append({
                    "Q2": Q2_value, "region": region, "n": n,
                    "x_lo": lo, "x_hi": hi,
                    "moment_AO_cont": I_cont,
                    "moment_AO_trapz_on_data_grid": np.nan,
                    "ratio_cont_over_trapz": np.nan,
                    "frac_diff": np.nan
                })
                continue

            w = x_seg ** (n - 2)
            y = w * F2_seg
            I_trapz = float(np.trapz(y, x_seg))

            ratio = (I_cont / I_trapz) if (np.isfinite(I_trapz) and I_trapz != 0.0) else np.nan
            frac_diff = ((I_cont - I_trapz) / I_cont) if (np.isfinite(I_cont) and I_cont != 0.0) else np.nan

            rows.append({
                "Q2": Q2_value, "region": region, "n": n,
                "x_lo": lo, "x_hi": hi,
                "moment_AO_continious": I_cont,
                "moment_AO_trapz_on_data_grid": I_trapz,
                "ratio_cont_over_trapz": ratio,
                "frac_diff": frac_diff
            })

    return pd.DataFrame(rows)

#------------------------------------------------------Patrick data helper function -------
from functools import lru_cache
@lru_cache(maxsize=None)
def _load_patrick_table(xlsx_path):
    """
    Read Patrick's xlsx once, normalize column names, and return a clean DataFrame.
    Cached so repeated compare_xsecs() calls do not keep re-reading the spreadsheet.
    """
    xlsx_path = os.path.abspath(xlsx_path)
    df = pd.read_excel(xlsx_path)

    # Normalize headers: "mean prediction" -> "mean_prediction", etc.
    df = df.rename(columns={
        col: col.strip().lower().replace(" ", "_")
        for col in df.columns
    })

    needed = [
        "q2", "w",
        "mean_prediction", "lower_prediction", "upper_prediction",
        "mean_thy", "lower_thy", "upper_thy",
    ]

    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(
            f"Patrick table is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # Force numeric where it matters
    for col in needed:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def read_patrick_data(
    series,          # "prediction" or "thy"
    fixed_Q2,
    beam_energy=None,
    W_min=None,
    W_max=None,
    xlsx_path="from_patrick/CLAS12.xlsx",
    q2_tol=1e-3,
    convert_nb_to_mub=True,
):
    """
    Return Patrick data in a format directly usable by plt.errorbar().

    Parameters
    ----------
    series : str
        Either "prediction" or "thy".
    fixed_Q2 : float
        Desired Q2 slice.
    beam_energy : ignored
        Kept only for interface compatibility with compare_xsecs().
    W_min, W_max : float or None
        Optional W window.
    xlsx_path : str
        Path to Patrick xlsx file.
    q2_tol : float
        Absolute tolerance for Q2 matching via np.isclose.
    convert_nb_to_mub : bool
        If True, convert nb/GeV^3 -> microbarn/GeV^3 by multiplying by 1e-3.

    Returns
    -------
    W : np.ndarray
    y : np.ndarray
    yerr : np.ndarray of shape (2, N)
        Asymmetric error bars in matplotlib format:
        yerr[0] = lower error = mean - lower
        yerr[1] = upper error = upper - mean
    have_data : bool
    """
    _ = beam_energy  # intentionally ignored

    series = series.strip().lower()
    series_map = {
        "prediction": ("mean_prediction", "lower_prediction", "upper_prediction"),
        "thy": ("mean_thy", "lower_thy", "upper_thy"),
    }

    if series not in series_map:
        raise ValueError(
            f"Unsupported series='{series}'. Use 'prediction' or 'thy'."
        )

    mean_col, low_col, up_col = series_map[series]
    df = _load_patrick_table(xlsx_path)

    # Select requested Q2 slice
    q2_mask = np.isclose(df["q2"].to_numpy(dtype=float), fixed_Q2,
                         atol=q2_tol, rtol=0.0)

    sub = df.loc[q2_mask, ["w", mean_col, low_col, up_col]].copy()

    # Optional W cuts
    if W_min is not None:
        sub = sub[sub["w"] >= W_min]
    if W_max is not None:
        sub = sub[sub["w"] <= W_max]

    # Drop junk rows, sort, and avoid accidental duplicate W entries
    sub = (
        sub.dropna(subset=["w", mean_col, low_col, up_col])
           .sort_values("w")
           .drop_duplicates(subset="w", keep="first")
    )

    if sub.empty:
        return np.array([]), np.array([]), np.empty((2, 0)), False

    scale = 1e-3 if convert_nb_to_mub else 1.0

    W = sub["w"].to_numpy(dtype=float)
    y = sub[mean_col].to_numpy(dtype=float) * scale
    y_low_abs = sub[low_col].to_numpy(dtype=float) * scale
    y_up_abs  = sub[up_col].to_numpy(dtype=float) * scale

    # Convert absolute bounds -> matplotlib asymmetric errors
    yerr_low = np.maximum(y - y_low_abs, 0.0)
    yerr_up  = np.maximum(y_up_abs - y, 0.0)
    yerr = np.vstack([yerr_low, yerr_up])

    return W, y, yerr, True


    
#print(estimate_bin_size_err_data([2.774,3.244,3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699],["1st"], R_source="AO"))
    
    
#calculate_epsilon_yannick(2.774)
#calculate_epsilon_valerii(2.774)