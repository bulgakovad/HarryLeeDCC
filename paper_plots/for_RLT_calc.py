import numpy as np

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



import numpy as np
import matplotlib.pyplot as plt

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


plot_R_vs_W_scan()