import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import os

MP = 0.9382720813  # GeV

def x_bj_from_WQ2(W, Q2, M=MP):
    """x_Bj(Q2, W) = Q2 / (W^2 - M^2 + Q2)."""
    W = np.asarray(W, dtype=float)
    Q2 = np.asarray(Q2, dtype=float)
    return Q2 / (W**2 - M**2 + Q2)

def R_osipenko(W, Q2, M=MP, W_ref=2.5):

    W  = np.asarray(W,  dtype=float)
    Q2 = np.asarray(Q2, dtype=float)

    # DIS core term (all GeV units); guard tiny/invalid Q2.
    Q2_safe = np.maximum(Q2, 1e-12)
    zeta = np.log(Q2_safe / 0.04)
    zeta = np.where(zeta <= 0, np.nan, zeta)
    core = (0.041 / zeta) + (0.592 / Q2_safe) - (0.331 / (0.09 + Q2_safe**2))

    # Threshold factor normalized at W_ref=2.5 GeV
    x     = x_bj_from_WQ2(W,     Q2_safe, M=M)
    x_ref = x_bj_from_WQ2(W_ref, Q2_safe, M=M)
    denom = (1.0 - x_ref)**3
    denom = np.where(np.abs(denom) < 1e-12, np.nan, denom)
    thresh = ((1.0 - x)**3) / denom

    R = thresh * core
    # Only defined/used for W <= 2.5 GeV
    R = np.where(W <= W_ref, R, np.nan)
    return R

def dR_osipenko(W, Q2):
    """Parametrization uncertainty δR in the resonance region: constant 0.08."""
    W = np.asarray(W, dtype=float)
    out = np.full_like(W, 0.08, dtype=float)
    out = np.where(W <= 2.5, out, np.nan)
    return out





def calculate_R_LT_from_AO_W1W2(
    Q2_value,
    table_path="input_data/wempx.dat",
    q2_tol=1e-6,
    return_interpolator=False,
    W_grid=None,
):
    """
    Calculate R_LT = sigma_L / sigma_T from AO model W1, W2 table.

    Expected table format:
        col 0: W
        col 1: Q2
        col 2: W1
        col 3: W2

    No header row.

    Formula:
        R_LT = (1 + nu^2 / Q2) * W2 / W1 - 1

    where:
        nu = (W^2 + Q2 - M^2) / (2M)

    Parameters
    ----------
    Q2_value : float
        Fixed Q2 value to select.
    table_path : str
        Path to AO W1/W2 table.
    q2_tol : float
        Tolerance for selecting rows with Q2 close to Q2_value.
    return_interpolator : bool
        If True, also return scipy interpolator R_LT(W).
    W_grid : array-like or None
        If given, evaluate interpolated R_LT on this W grid.

    Returns
    -------
    df_out : pandas.DataFrame
        Columns: W, Q2, W1, W2, nu, x, R_LT

    If return_interpolator=True:
        return df_out, R_LT_interp

    If W_grid is not None:
        return df_grid
    """
    
    M_PROTON = MP

    df = pd.read_csv(
        table_path,
        delim_whitespace=True,
        header=None,
        names=["W", "Q2", "W1", "W2"],
        comment="#",
    )

    # Keep only requested Q2
    df_q2 = df[np.abs(df["Q2"] - Q2_value) < q2_tol].copy()

    if df_q2.empty:
        available_q2 = np.sort(df["Q2"].unique())
        raise ValueError(
            f"No AO W1/W2 data found for Q2 = {Q2_value}. "
            f"Try increasing q2_tol. Available Q2 values include:\n"
            f"{available_q2[:20]}"
        )

    df_q2 = df_q2.sort_values("W").reset_index(drop=True)

    W = df_q2["W"].to_numpy(dtype=float)
    Q2 = df_q2["Q2"].to_numpy(dtype=float)
    W1 = df_q2["W1"].to_numpy(dtype=float)
    W2 = df_q2["W2"].to_numpy(dtype=float)

    nu = (W**2 + Q2 - M_PROTON**2) / (2.0 * M_PROTON)
    x = Q2 / (2.0 * M_PROTON * nu)

    R_LT = (1.0 + nu**2 / Q2) * (W2 / W1) - 1.0

    df_out = pd.DataFrame({
        "W": W,
        "Q2": Q2,
        "W1": W1,
        "W2": W2,
        "nu": nu,
        "x": x,
        "R_LT": R_LT,
    })

    # Remove pathological points if W1 is zero/tiny or result is nan/inf
    df_out = df_out.replace([np.inf, -np.inf], np.nan)
    df_out = df_out.dropna(subset=["R_LT"]).reset_index(drop=True)

    if W_grid is not None:
        interp = interp1d(
            df_out["W"],
            df_out["R_LT"],
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

        W_grid = np.asarray(W_grid, dtype=float)

        return pd.DataFrame({
            "W": W_grid,
            "Q2": Q2_value,
            "R_LT": interp(W_grid),
        })

    if return_interpolator:
        R_LT_interp = interp1d(
            df_out["W"],
            df_out["R_LT"],
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )

        return df_out, R_LT_interp

    return df_out


def plot_R_LT_AO_vs_W_for_Q2_list(
    Q2_list,
    table_path="input_data/wempx.dat",
    output_dir="RLT_AO_Osipenko",
    q2_tol=1e-4,
    W_min=None,
    W_max=None,
    overlay_osipenko=True,
    save_pdf=True,
    save_png=False,
    show=False,
    skip_missing=True,
):
    """
    Plot AO-model R_LT(W) for each Q2 in Q2_list and save each plot as PDF.

    Uses existing function:
        calculate_R_LT_from_AO_W1W2(...)

    Parameters
    ----------
    Q2_list : list or array
        List of Q2 values to plot.
    table_path : str
        Path to AO W1/W2 table.
    output_dir : str
        Folder where plots are saved.
    q2_tol : float
        Tolerance for selecting Q2 rows from AO table.
    W_min, W_max : float or None
        Optional W range cut.
    overlay_osipenko : bool
        If True, overlay R_osipenko(W, Q2).
    save_pdf : bool
        Save each plot as PDF.
    save_png : bool
        Optionally also save PNG.
    show : bool
        If True, display plots interactively.
    skip_missing : bool
        If True, skip Q2 values missing from AO table instead of crashing.

    Returns
    -------
    saved_files : list
        List of saved plot paths.
    """

    os.makedirs(output_dir, exist_ok=True)

    saved_files = []

    for Q2_value in Q2_list:

        try:
            df_rlt = calculate_R_LT_from_AO_W1W2(
                Q2_value=Q2_value,
                table_path=table_path,
                q2_tol=q2_tol,
            )

        except Exception as e:
            msg = f"[plot_R_LT_AO] Skipping Q2 = {Q2_value}: {e}"
            if skip_missing:
                print(msg)
                continue
            else:
                raise RuntimeError(msg)

        # Optional W cuts
        if W_min is not None:
            df_rlt = df_rlt[df_rlt["W"] >= W_min]

        if W_max is not None:
            df_rlt = df_rlt[df_rlt["W"] <= W_max]

        if df_rlt.empty:
            print(f"[plot_R_LT_AO] Skipping Q2 = {Q2_value}: no points after W cuts.")
            continue

        W = df_rlt["W"].to_numpy(dtype=float)
        R_LT = df_rlt["R_LT"].to_numpy(dtype=float)

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.plot(
            W,
            R_LT,
            "-",
            lw=2,
            label="AO from $W_1, W_2$"
        )

        # Optional Osipenko comparison
        if overlay_osipenko:
            W_osi = np.linspace(np.nanmin(W), np.nanmax(W), 500)
            R_osi = R_osipenko(W_osi, Q2_value)

            ax.plot(
                W_osi,
                R_osi,
                "--",
                lw=2,
                label="Osipenko"
            )

            dR_osi = dR_osipenko(W_osi, Q2_value)

            ax.fill_between(
                W_osi,
                R_osi - dR_osi,
                R_osi + dR_osi,
                alpha=0.20,
                label=r"Osipenko $\delta R = 0.08$"
            )

        ax.axhline(0.0, color="black", lw=1, alpha=0.5)

        ax.set_xlabel(r"$W$ [GeV]", fontsize=13)
        ax.set_ylabel(r"$R_{LT} = \sigma_L / \sigma_T$", fontsize=13)

        ax.set_title(
            rf"$R_{{LT}}(W)$, $Q^2 = {Q2_value:.3f}$ GeV$^2$",
            fontsize=14
        )

        ax.grid(alpha=0.3)
        ax.legend(fontsize=11)
        fig.tight_layout()

        q2_tag = f"{Q2_value:.3f}".replace(".", "p")

        if save_pdf:
            pdf_path = os.path.join(output_dir, f"R_LT_AO_Q2_{q2_tag}.pdf")
            fig.savefig(pdf_path)
            saved_files.append(pdf_path)
            print(f"Saved: {pdf_path}")

        if save_png:
            png_path = os.path.join(output_dir, f"R_LT_AO_Q2_{q2_tag}.png")
            fig.savefig(png_path, dpi=200)
            saved_files.append(png_path)
            print(f"Saved: {png_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    return saved_files


plot_R_LT_AO_vs_W_for_Q2_list(Q2_list=[1.6, 1.8, 1.4])

def plot_R_vs_W_scan(out_png="R_vs_W_resonance_Q2scan.png"):
    """
    Plot R(W,Q^2)=sigma_L/sigma_T vs W for the 9 Q^2 values (resonance region only).
    Uses user-provided function: R_osipenko(W, Q2).
    No error bars/shading.
    """
    # Q^2 list and W range (resonance only)
    q2_values = [2.774, 3.244, 3.793, 4.435, 5.187, 6.065, 7.093, 8.294, 9.699]
    W_min, W_max = 1.07, 2.50
    W_fine = np.linspace(W_min, W_max, 400)

    # optional discrete points to overlay (your terminal grid)
    W_nodes = np.array([1.15, 1.20, 1.25, 1.30, 1.40, 1.50, 1.60, 1.70,
                        1.80, 1.90, 2.00, 2.10, 2.20, 2.30, 2.40, 2.50])

    # vectorize in case R_osipenko is scalar-only
    Rv = np.vectorize(R_osipenko)

    fig, axes = plt.subplots(3, 3, figsize=(11, 9), sharex=True, sharey=True, constrained_layout=True)

    for ax, Q2 in zip(axes.ravel(), q2_values):
        R_f = Rv(W_fine, Q2)
        mask = np.isfinite(R_f)
        ax.plot(W_fine[mask], R_f[mask], lw=1.8, color="tab:blue", label=r"$R(W,Q^2)$")

        # overlay node points (no errors)
        R_pts = Rv(W_nodes, Q2)
        ax.plot(W_nodes, R_pts, "ko", ms=3)

        ax.set_title(f"$Q^2={Q2:.3g}")
        ax.grid(alpha=0.25)

    for ax in axes[-1]:
        ax.set_xlabel("$W$ [GeV]")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$R=\sigma_L/\sigma_T$")

    fig.suptitle(r"$R(W,Q^2)$ in the resonance region", y=1.02, fontsize=14)

    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Saved → {out_png}")


