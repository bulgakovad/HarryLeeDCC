#!/usr/bin/env python3

import os
import math
import lhapdf
import numpy as np
import matplotlib.pyplot as plt

from plot_settings import *


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

PDF_SET = "CJ15nlo"
PDF_MEMBER = 0
Q2 = 4.0  # GeV^2

# If True, evaluate all CJ15 error members and draw the PDF uncertainty band.
# If False, plot central PDF_MEMBER 
USE_UNCERTAINTY_CJ15 = False

# Confidence level requested from LHAPDF for the CJ15 uncertainty band.
# 68.268949 corresponds to one standard deviation for a Gaussian.
CJ15_CONFIDENCE_LEVEL = 68.268949


# Argonne lattice momentum index:
# p5 corresponds to P^z = 3.04 GeV in the supplied plotting code.
ARGONNE_P_INDEX = 5
ARGONNE_PZ_GEV = 3.04

ARGONNE_DATA_DIR = "../unpol_PDF_for_harry/data"

OUTPUT_DIR = "Output/LHAPDF_vs_Argonne_isovector_comparison"
OUTPUT_NAME = f"{PDF_SET}_Argonne_signed_isovector_Q2_4.png"

# LHAPDF evaluation range.
XMIN = 1.0e-3
XMAX = 0.99
NPOINTS = 700
LOG_SPACING = True

# Approximate reliability regions shown in the Argonne plot.
LATTICE_XMIN = 0.18
LATTICE_XMAX = 0.82

YMIN = -0.5
YMAX = 4.0


# ----------------------------------------------------------------------
# CJ15 functions
# ----------------------------------------------------------------------

def make_positive_x_grid(xmin, xmax, npoints, log_spacing=True):
    """
    Generate a positive-x grid for LHAPDF evaluation.
    """
    if xmin <= 0.0:
        raise ValueError("xmin must be positive.")

    if xmax <= xmin:
        raise ValueError("xmax must be greater than xmin.")

    if npoints < 2:
        raise ValueError("npoints must be at least 2.")

    if log_spacing:
        return np.logspace(
            math.log10(xmin),
            math.log10(xmax),
            npoints,
        )

    return np.linspace(xmin, xmax, npoints)


def get_cj15_signed_isovector_branches(pdf, x_positive, Q2):
    r"""
    Construct the signed isovector PDF from one CJ15 member.

    For x > 0:

        f^{u-d}(x,Q^2) = u(x,Q^2) - d(x,Q^2)

    For x < 0:

        f^{u-d}(x,Q^2)
        = -[\bar{u}(|x|,Q^2) - \bar{d}(|x|,Q^2)]

    LHAPDF xfxQ2 returns x*f(x,Q^2), so division by x is required.
    """

    f_positive = np.array([
        (
            pdf.xfxQ2(+2, x, Q2)
            - pdf.xfxQ2(+1, x, Q2)
        ) / x
        for x in x_positive
    ])

    f_negative_at_abs_x = np.array([
        -(
            pdf.xfxQ2(-2, x, Q2)
            - pdf.xfxQ2(-1, x, Q2)
        ) / x
        for x in x_positive
    ])

    # Reverse arrays so the negative branch runs from -XMAX to -XMIN.
    x_negative = -x_positive[::-1]
    f_negative = f_negative_at_abs_x[::-1]

    return x_negative, f_negative, x_positive, f_positive


def get_cj15_signed_isovector_with_uncertainty(
    pdf_set_name,
    x_positive,
    Q2,
    confidence_level=68.268949,
):
    r"""
    Evaluate the signed isovector PDF for all members of a PDF set and
    calculate the set-prescribed PDF uncertainty using LHAPDF.

    The uncertainty is calculated directly for the derived observable

        u - d

    and

        -(\bar{u} - \bar{d}),

    preserving the correlations among the individual flavors.

    Returns central values and asymmetric upper/lower uncertainties for
    the positive- and negative-x branches.
    """

    pdf_set = lhapdf.getPDFSet(pdf_set_name)
    pdf_members = pdf_set.mkPDFs()

    if len(pdf_members) < 2:
        raise RuntimeError(
            f"{pdf_set_name} contains only {len(pdf_members)} member(s); "
            "an uncertainty band cannot be calculated."
        )

    print(
        f"[{pdf_set_name}] Using {len(pdf_members)} total members "
        f"for the PDF uncertainty."
    )
    print(
        f"[{pdf_set_name}] Requested confidence level: "
        f"{confidence_level:g}%"
    )

    npoints = len(x_positive)

    central_positive = np.empty(npoints)
    errplus_positive = np.empty(npoints)
    errminus_positive = np.empty(npoints)

    central_negative_abs_x = np.empty(npoints)
    errplus_negative_abs_x = np.empty(npoints)
    errminus_negative_abs_x = np.empty(npoints)

    for ix, x in enumerate(x_positive):
        positive_member_values = np.array([
            (
                pdf.xfxQ2(+2, x, Q2)
                - pdf.xfxQ2(+1, x, Q2)
            ) / x
            for pdf in pdf_members
        ])

        negative_member_values = np.array([
            -(
                pdf.xfxQ2(-2, x, Q2)
                - pdf.xfxQ2(-1, x, Q2)
            ) / x
            for pdf in pdf_members
        ])

        positive_uncertainty = pdf_set.uncertainty(
            positive_member_values.tolist(),
            confidence_level,
        )

        negative_uncertainty = pdf_set.uncertainty(
            negative_member_values.tolist(),
            confidence_level,
        )

        central_positive[ix] = positive_uncertainty.central
        errplus_positive[ix] = positive_uncertainty.errplus
        errminus_positive[ix] = positive_uncertainty.errminus

        central_negative_abs_x[ix] = negative_uncertainty.central
        errplus_negative_abs_x[ix] = negative_uncertainty.errplus
        errminus_negative_abs_x[ix] = negative_uncertainty.errminus

    # Reverse all negative-side arrays so x increases from -XMAX to -XMIN.
    x_negative = -x_positive[::-1]

    central_negative = central_negative_abs_x[::-1]
    errplus_negative = errplus_negative_abs_x[::-1]
    errminus_negative = errminus_negative_abs_x[::-1]

    return {
        "x_positive": x_positive,
        "central_positive": central_positive,
        "errplus_positive": errplus_positive,
        "errminus_positive": errminus_positive,
        "x_negative": x_negative,
        "central_negative": central_negative,
        "errplus_negative": errplus_negative,
        "errminus_negative": errminus_negative,
        "number_of_members": len(pdf_members),
    }


# ----------------------------------------------------------------------
# Argonne loading functions
# ----------------------------------------------------------------------

def load_argonne_branch(
    data_dir,
    matching_order,
    p_index,
    branch,
):
    """
    Load one Argonne branch.

    Parameters
    ----------
    matching_order : str
        "nlo" or "nll"

    p_index : int
        For example, 5 for the supplied p5 files.

    branch : str
        "quark" or "anti_quark"

    Expected filenames
    ------------------
    unpol_nlo_p5_quark_x_ls.txt
    unpol_nlo_p5_quark_mean.txt
    unpol_nlo_p5_quark_sdev.txt

    and similarly for NLL and anti_quark.
    """

    prefix = os.path.join(
        data_dir,
        f"unpol_{matching_order}_p{p_index}_{branch}",
    )

    x_path = f"{prefix}_x_ls.txt"
    mean_path = f"{prefix}_mean.txt"
    sdev_path = f"{prefix}_sdev.txt"

    for path in (x_path, mean_path, sdev_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Argonne input file not found: {path}"
            )

    x_values = np.atleast_1d(np.loadtxt(x_path))
    mean_values = np.atleast_1d(np.loadtxt(mean_path))
    sdev_values = np.atleast_1d(np.loadtxt(sdev_path))

    if not (
        len(x_values)
        == len(mean_values)
        == len(sdev_values)
    ):
        raise ValueError(
            f"Array-length mismatch for {matching_order}, {branch}: "
            f"x={len(x_values)}, "
            f"mean={len(mean_values)}, "
            f"sdev={len(sdev_values)}"
        )

    if not (
        np.all(np.isfinite(x_values))
        and np.all(np.isfinite(mean_values))
        and np.all(np.isfinite(sdev_values))
    ):
        raise ValueError(
            f"Non-finite value found in Argonne "
            f"{matching_order} {branch} input."
        )

    order = np.argsort(x_values)

    return (
        x_values[order],
        mean_values[order],
        sdev_values[order],
    )


def load_all_argonne_data(data_dir, p_index):
    """
    Load NLO and NLL quark and antiquark branches.
    """

    data = {}

    for matching_order in ("nlo", "nll"):
        for branch in ("quark", "anti_quark"):
            data[(matching_order, branch)] = load_argonne_branch(
                data_dir=data_dir,
                matching_order=matching_order,
                p_index=p_index,
                branch=branch,
            )

    return data


# ----------------------------------------------------------------------
# Main plotting function
# ----------------------------------------------------------------------

def plot_cj15_argonne_comparison():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------------- CJ15 ----------------

    central_pdf = lhapdf.mkPDF(PDF_SET, PDF_MEMBER)

    print(
        f"[{PDF_SET}, member {PDF_MEMBER}] "
        f"x range = [{central_pdf.xMin}, {central_pdf.xMax}], "
        f"Q^2 range = [{central_pdf.q2Min}, "
        f"{central_pdf.q2Max}] GeV^2"
    )

    if not (central_pdf.q2Min <= Q2 <= central_pdf.q2Max):
        print(
            f"WARNING: Q^2={Q2} GeV^2 is outside the nominal "
            f"range [{central_pdf.q2Min}, "
            f"{central_pdf.q2Max}] GeV^2."
        )

    if XMIN < central_pdf.xMin:
        print(
            f"WARNING: XMIN={XMIN} is below the nominal PDF limit "
            f"xMin={central_pdf.xMin}."
        )

    if XMAX > central_pdf.xMax:
        print(
            f"WARNING: XMAX={XMAX} is above the nominal PDF limit "
            f"xMax={central_pdf.xMax}."
        )

    x_positive = make_positive_x_grid(
        xmin=XMIN,
        xmax=XMAX,
        npoints=NPOINTS,
        log_spacing=LOG_SPACING,
    )

    if USE_UNCERTAINTY_CJ15:
        cj15 = get_cj15_signed_isovector_with_uncertainty(
            pdf_set_name=PDF_SET,
            x_positive=x_positive,
            Q2=Q2,
            confidence_level=CJ15_CONFIDENCE_LEVEL,
        )
    else:
        (
            cj_x_negative,
            cj_f_negative,
            cj_x_positive,
            cj_f_positive,
        ) = get_cj15_signed_isovector_branches(
            pdf=central_pdf,
            x_positive=x_positive,
            Q2=Q2,
        )

    # ---------------- Argonne ----------------

    argonne = load_all_argonne_data(
        data_dir=ARGONNE_DATA_DIR,
        p_index=ARGONNE_P_INDEX,
    )

    (
        nll_quark_x,
        nll_quark_mean,
        nll_quark_sdev,
    ) = argonne[("nll", "quark")]

    (
        nll_anti_x,
        nll_anti_mean,
        nll_anti_sdev,
    ) = argonne[("nll", "anti_quark")]

    (
        nlo_quark_x,
        nlo_quark_mean,
        nlo_quark_sdev,
    ) = argonne[("nlo", "quark")]

    (
        nlo_anti_x,
        nlo_anti_mean,
        nlo_anti_sdev,
    ) = argonne[("nlo", "anti_quark")]

    # ---------------- Plot ----------------

    fig, ax = default_plot()

    # Argonne NLL quark branch.
    ax.fill_between(
        nll_quark_x,
        nll_quark_mean - nll_quark_sdev,
        nll_quark_mean + nll_quark_sdev,
        color=blue,
        alpha=0.30,
        hatch="///",
        edgecolor="blue",
        label=rf"Argonne, $P^z={ARGONNE_PZ_GEV:g}$ GeV, NLL",
        zorder=2,
    )

    ax.plot(
        nll_quark_x,
        nll_quark_mean,
        color=blue,
        linewidth=1.2,
        zorder=3,
    )

    # Argonne NLL antiquark branch.
    ax.fill_between(
        nll_anti_x,
        nll_anti_mean - nll_anti_sdev,
        nll_anti_mean + nll_anti_sdev,
        color=blue,
        alpha=0.30,
        hatch="///",
        edgecolor="blue",
        zorder=2,
    )

    ax.plot(
        nll_anti_x,
        nll_anti_mean,
        color=blue,
        linewidth=1.2,
        zorder=3,
    )

    # Argonne NLO quark branch.
    ax.fill_between(
        nlo_quark_x,
        nlo_quark_mean - nlo_quark_sdev,
        nlo_quark_mean + nlo_quark_sdev,
        color=blue,
        alpha=0.18,
        label=rf"Argonne, $P^z={ARGONNE_PZ_GEV:g}$ GeV, NLO",
        zorder=1,
    )

    ax.plot(
        nlo_quark_x,
        nlo_quark_mean,
        color=blue,
        linewidth=1.2,
        zorder=3,
    )

    # Argonne NLO antiquark branch.
    ax.fill_between(
        nlo_anti_x,
        nlo_anti_mean - nlo_anti_sdev,
        nlo_anti_mean + nlo_anti_sdev,
        color=blue,
        alpha=0.18,
        zorder=1,
    )

    ax.plot(
        nlo_anti_x,
        nlo_anti_mean,
        color=blue,
        linewidth=1.2,
        zorder=3,
    )

    # ---------------- CJ15 ----------------

    if USE_UNCERTAINTY_CJ15:
        # Negative-x CJ15 uncertainty band.
        ax.fill_between(
            cj15["x_negative"],
            (
                cj15["central_negative"]
                - cj15["errminus_negative"]
            ),
            (
                cj15["central_negative"]
                + cj15["errplus_negative"]
            ),
            color=red,
            alpha=0.5,
            linewidth=0.0,
            label=(
                rf"{PDF_SET}, "
                rf"{CJ15_CONFIDENCE_LEVEL:.0f} % C.L."
            ),
            zorder=4,
        )

        # Positive-x CJ15 uncertainty band.
        ax.fill_between(
            cj15["x_positive"],
            (
                cj15["central_positive"]
                - cj15["errminus_positive"]
            ),
            (
                cj15["central_positive"]
                + cj15["errplus_positive"]
            ),
            color=red,
            alpha=0.5,
            linewidth=0.0,
            zorder=4,
        )

        # Negative-x central curve.
        ax.plot(
            cj15["x_negative"],
            cj15["central_negative"],
            color=red,
            linewidth=1.0,
            linestyle="-",
            zorder=5,
        )

        # Positive-x central curve.
        ax.plot(
            cj15["x_positive"],
            cj15["central_positive"],
            color=red,
            linewidth=1.0,
            linestyle="-",
            zorder=5,
        )

    else:
        # Original behavior: one selected CJ15 member and no uncertainty.
        ax.plot(
            cj_x_negative,
            cj_f_negative,
            color=red,
            linewidth=1.0,
            linestyle="-",
            label=rf"{PDF_SET}",
            zorder=5,
        )

        ax.plot(
            cj_x_positive,
            cj_f_positive,
            color=red,
            linewidth=1.0,
            linestyle="-",
            zorder=5,
        )

    # Reference line.
    ax.axhline(
        0.0,
        color="black",
        linewidth=0.8,
        zorder=0,
    )

    # Regions shaded in the Argonne plot.
    ax.axvspan(
        -LATTICE_XMIN,
        LATTICE_XMIN,
        color=grey,
        alpha=0.30,
        zorder=-2,
    )

    ax.axvspan(
        LATTICE_XMAX,
        1.0,
        color=grey,
        alpha=0.30,
        zorder=-2,
    )

    ax.axvspan(
        -1.0,
        -LATTICE_XMAX,
        color=grey,
        alpha=0.30,
        zorder=-2,
    )

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(YMIN, YMAX)

    ax.set_xlabel(
        r"$x$",
        **fs_p,
    )

    ax.set_ylabel(
        r"$f^{u-d}(x;\ Q^2=\mu^2=4~\mathrm{GeV}^2)$",
        **fs_p,
    )

    ax.set_title(
        r"Argonne LQCD and CJ15nlo signed isovector PDF",
        **fs_small_p,
    )

    ax.legend(
        loc="upper left",
        fontsize=9,
        frameon=True,
        borderpad=0.4,
        labelspacing=0.4,
        handlelength=2.0,
        handletextpad=0.6,
    )

    fig.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        OUTPUT_NAME,
    )

    fig.savefig(
        output_path,
        dpi=250,
        bbox_inches="tight",
    )

    pdf_output_path = os.path.splitext(output_path)[0] + ".pdf"

    fig.savefig(
        pdf_output_path,
        bbox_inches="tight",
        transparent=True,
    )

    plt.show()
    plt.close(fig)

    print(f"Saved PNG: {output_path}")
    print(f"Saved PDF: {pdf_output_path}")


if __name__ == "__main__":
    plot_cj15_argonne_comparison()