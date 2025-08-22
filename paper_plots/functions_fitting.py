# functions_fitting.py

import os
import numpy as np
import pandas as pd
import math
from iminuit import Minuit
from iminuit.cost import LeastSquares
from typing import Optional

from functions_anl_osaka import (
    calculate_1pi_cross_section,
    compute_2pi_cross_section,
    compute_cross_section,
)


def F2_of_omega(omega, c1, c2, c3):
    """
    Scaling function F2(omega') = c1 z^3 + c2 z^4 + c3 z^5,  z = 1 - 1/omega'.
    """
    z = 1.0 - 1.0 / omega
    return c1 * z**3 + c2 * z**4 + c3 * z**5


def read_and_prepare_data(exp_folder: str, W_max_fit: float, filename: str = "exp_data_all.dat") -> pd.DataFrame:
    """
    Minimal loader for a single CSV:
        exp_data_to_fit/exp_data_all.dat
    Expected columns (exact): Q2, W, epsilon, XSEC, Stat, Sys, ScaleType

    Returns a dataframe with the needed columns and an 'uncertainty' column,
    filtered to W <= W_max_fit and Q2 > 0.
    """
    # Allow either a folder (default) or a direct file path
    path = exp_folder
    if os.path.isdir(exp_folder):
        path = os.path.join(exp_folder, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    df = pd.read_csv(path)  # comma-separated

    # Ensure required columns exist
    required = ["Q2", "W", "XSEC", "Stat", "Sys"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {path}: {missing}")

    # Keep only what we need; cast to float and compute total uncertainty
    df = df[["Q2", "W", "XSEC", "Stat", "Sys"]].astype(float).copy()
    df["uncertainty"] = np.sqrt(df["Stat"]**2 + df["Sys"]**2)

    # Clean and filter
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["Q2", "W", "XSEC", "uncertainty"], inplace=True)
    df = df[(df["W"] <= W_max_fit) & (df["Q2"] > 0) & (df["uncertainty"] > 0)].copy()

    if df.empty:
        raise ValueError("No valid points after filtering (check W_max_fit or data).")

    return df

def compute_reference_shapes_for_df(
    df: pd.DataFrame,
    Q2_ref: float,
    E_ref: float,
    full_file: str,
    onepi_file: str,
    clamp_nonneg_2pi: bool = True,
) -> pd.DataFrame:
    """
    For each W in df, compute sigma_1pi_ref(W) and sigma_2pi_ref(W) at fixed (Q2_ref, E_ref).
    Adds two columns: 'sigma_1pi_ref', 'sigma_2pi_ref'.
    """
    def _onepi_ref(w):
        return calculate_1pi_cross_section(w, Q2_ref, E_ref, file_path=onepi_file, verbose=False)

    def _twopi_ref(w):
        try:
            return compute_2pi_cross_section(
                W=w, Q2=Q2_ref, beam_energy=E_ref,
                full_file_path=full_file, onepi_file_path=onepi_file,
                verbose=False, clamp_nonneg=clamp_nonneg_2pi
            )
        except Exception:
            # Fallback: full - 1π
            full = compute_cross_section(w, Q2_ref, E_ref, file_path=full_file, verbose=False)
            one  = calculate_1pi_cross_section(w, Q2_ref, E_ref, file_path=onepi_file, verbose=False)
            d = full - one
            return max(d, 0.0) if clamp_nonneg_2pi else d

    v1 = np.vectorize(_onepi_ref, otypes=[float])
    v2 = np.vectorize(_twopi_ref, otypes=[float])

    W_arr = df["W"].to_numpy(dtype=float)
    df = df.copy()
    df["sigma_1pi_ref"] = v1(W_arr)
    df["sigma_2pi_ref"] = v2(W_arr)

    df = df[np.isfinite(df["sigma_1pi_ref"]) & np.isfinite(df["sigma_2pi_ref"])].copy()
    if df.empty:
        raise RuntimeError("Reference channel evaluation failed for all points (check W range vs tables).")
    return df


def build_least_squares_from_df(df: pd.DataFrame) -> tuple:
    """
    Builds LeastSquares(cost) for the global model using the df with
    columns: W, Q2, XSEC, uncertainty, sigma_1pi_ref, sigma_2pi_ref.
    Returns (cost, x_tuple, y, yerr).
    """
    def model_scaling(x_tuple, c1, c2, c3, c1p, c2p, c3p):
        W, Q2, s1, s2 = x_tuple
        omega = 1.0 + (W**2) / Q2
        return F2_of_omega(omega, c1, c2, c3) * s1 + F2_of_omega(omega, c1p, c2p, c3p) * s2

    x_tuple = (
        df["W"].to_numpy(dtype=float),
        df["Q2"].to_numpy(dtype=float),
        df["sigma_1pi_ref"].to_numpy(dtype=float),
        df["sigma_2pi_ref"].to_numpy(dtype=float),
    )
    y     = df["XSEC"].to_numpy(dtype=float)
    yerr  = df["uncertainty"].to_numpy(dtype=float)
    cost  = LeastSquares(x_tuple, y, yerr, model_scaling)
    return cost, x_tuple, y, yerr


def run_minuit(
    cost,
    *,
    multi_start: int = 1,
    jitter_scale: float = 0.20,
    strategy: int = 1,
    limits: Optional[dict] = None,   # e.g. {"c1": (-100, 100), "c1p": (-100, 100), ...}
    seed: Optional[int] = None,
    base_start: Optional[dict] = None
):
    """
    Run Minuit on the provided LeastSquares cost.

    Backward compatible:
      - multi_start=1 reproduces your previous single-start behavior.
    Set multi_start>1 to random-restart around base_start and keep the best χ².
    """

    # Default starting values (same as before; edit if your originals differ)
    if base_start is None:
        base_start = dict(c1=1.0, c2=1.0, c3=1.0, c1p=1.0, c2p=1.0, c3p=1.0)

    rng = np.random.default_rng(seed)

    def _fit_once(start_dict):
        m = Minuit(cost, **start_dict)
        if limits:
            for k, lim in limits.items():
                if k in m.parameters:
                    m.limits[k] = lim
        m.strategy = strategy  # 1 = default, 2 = more thorough
        m.migrad()
        m.hesse()              # keep your current covariance behavior
        return m

    # Single start (exactly like before)
    if multi_start <= 1:
        return _fit_once(base_start)

    # Multi-start: jitter around base_start, keep best valid χ²
    best_m = None
    best_val = math.inf

    names = list(base_start.keys())
    for k in range(multi_start):
        start = {}
        for p in names:
            v = float(base_start[p])
            # multiplicative jitter if |v|>0, otherwise additive
            if abs(v) > 0:
                start[p] = v * (1.0 + jitter_scale * rng.standard_normal())
            else:
                start[p] = jitter_scale * rng.standard_normal()

        m = _fit_once(start)
        # prefer valid minima; still consider non-valid if nothing valid yet
        score = m.fval if m.fmin.is_valid else (m.fval + 1e6)
        if score < best_val:
            best_val = score
            best_m = m

    return best_m


def make_predict_curve_function(
    Q2_ref: float,
    E_ref: float,
    onepi_file: str,
    full_file: str,
    clamp_nonneg_2pi: bool,
    fitted_params: dict,
):
    """
    Returns a function predict(W_arr, q2) that computes the post-fit total curve:
      F2(omega'; c)*sigma_1pi_ref(W) + F2(omega'; c')*sigma_2pi_ref(W)
    for any W_arr, q2.
    """
    def _onepi_ref(w):
        return calculate_1pi_cross_section(w, Q2_ref, E_ref, file_path=onepi_file, verbose=False)

    def _twopi_ref(w):
        try:
            return compute_2pi_cross_section(
                W=w, Q2=Q2_ref, beam_energy=E_ref,
                full_file_path=full_file, onepi_file_path=onepi_file,
                verbose=False, clamp_nonneg=clamp_nonneg_2pi
            )
        except Exception:
            full = compute_cross_section(w, Q2_ref, E_ref, file_path=full_file, verbose=False)
            one  = calculate_1pi_cross_section(w, Q2_ref, E_ref, file_path=onepi_file, verbose=False)
            d = full - one
            return max(d, 0.0) if clamp_nonneg_2pi else d

    v1 = np.vectorize(_onepi_ref, otypes=[float])
    v2 = np.vectorize(_twopi_ref, otypes=[float])

    p = dict(fitted_params)

    def predict(W_arr, q2):
        W_arr = np.asarray(W_arr, dtype=float)
        s1 = v1(W_arr)
        s2 = v2(W_arr)
        omega = 1.0 + (W_arr**2) / float(q2)
        z = 1.0 - 1.0 / omega
        f2a = p["c1"]  * z**3 + p["c2"]  * z**4 + p["c3"]  * z**5
        f2b = p["c1p"] * z**3 + p["c2p"] * z**4 + p["c3p"] * z**5
        return f2a * s1 + f2b * s2

    return predict


def make_component_curve_functions(
    Q2_ref: float,
    E_ref: float,
    onepi_file: str,
    full_file: str,
    clamp_nonneg_2pi: bool,
    fitted_params: dict,
):
    """
    Returns two functions:
      predict_1pi(W_arr, q2) = F2(omega'; c)*sigma_1pi_ref(W)
      predict_2pi(W_arr, q2) = F2(omega'; c')*sigma_2pi_ref(W)
    so you can plot the components separately.
    """
    def _onepi_ref(w):
        return calculate_1pi_cross_section(w, Q2_ref, E_ref, file_path=onepi_file, verbose=False)

    def _twopi_ref(w):
        try:
            return compute_2pi_cross_section(
                W=w, Q2=Q2_ref, beam_energy=E_ref,
                full_file_path=full_file, onepi_file_path=onepi_file,
                verbose=False, clamp_nonneg=clamp_nonneg_2pi
            )
        except Exception:
            full = compute_cross_section(w, Q2_ref, E_ref, file_path=full_file, verbose=False)
            one  = calculate_1pi_cross_section(w, Q2_ref, E_ref, file_path=onepi_file, verbose=False)
            d = full - one
            return max(d, 0.0) if clamp_nonneg_2pi else d

    v1 = np.vectorize(_onepi_ref, otypes=[float])
    v2 = np.vectorize(_twopi_ref, otypes=[float])

    p = dict(fitted_params)

    def predict_1pi(W_arr, q2):
        W_arr = np.asarray(W_arr, dtype=float)
        s1 = v1(W_arr)
        omega = 1.0 + (W_arr**2) / float(q2)
        z = 1.0 - 1.0 / omega
        f2a = p["c1"]  * z**3 + p["c2"]  * z**4 + p["c3"]  * z**5
        return f2a * s1

    def predict_2pi(W_arr, q2):
        W_arr = np.asarray(W_arr, dtype=float)
        s2 = v2(W_arr)
        omega = 1.0 + (W_arr**2) / float(q2)
        z = 1.0 - 1.0 / omega
        f2b = p["c1p"] * z**3 + p["c2p"] * z**4 + p["c3p"] * z**5
        return f2b * s2

    return predict_1pi, predict_2pi
