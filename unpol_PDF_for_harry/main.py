# %%
from plot_settings import *

px = 5
px_gev = 3.04 # GeV
color = blue
edgecolor = "blue"

fig, ax = default_plot()


#! Note: plotting next-to-leading-logarithmic (NLL) order results of unpolarized valence quark PDF. The order represents the perturbative order of the matching kernel in LaMET formalism.
unpol_nll_quark_x_ls = np.loadtxt(f"./data/unpol_nll_p{px}_quark_x_ls.txt")
unpol_nll_quark_mean = np.loadtxt(f"./data/unpol_nll_p{px}_quark_mean.txt")
unpol_nll_quark_sdev = np.loadtxt(f"./data/unpol_nll_p{px}_quark_sdev.txt")

ax.fill_between(
    unpol_nll_quark_x_ls,
    unpol_nll_quark_mean - unpol_nll_quark_sdev,
    unpol_nll_quark_mean + unpol_nll_quark_sdev,
    color=color,
    alpha=0.3,
    hatch="///",
    edgecolor=edgecolor,
    label=r"$P^z=$" + f"{px_gev} GeV, NLL",
)
ax.plot(unpol_nll_quark_x_ls, unpol_nll_quark_mean, color=color, linewidth=1)


#! Note: plotting next-to-leading-logarithmic (NLL) order results of unpolarized anti-quark PDF. The anti-quark PDF has an extra minus sign to maintain the normalization condition: integrate the PDF from -1 to 1 gives q - qbar contribution. 
unpol_nll_anti_quark_x_ls = np.loadtxt(f"./data/unpol_nll_p{px}_anti_quark_x_ls.txt")
unpol_nll_anti_quark_mean = np.loadtxt(f"./data/unpol_nll_p{px}_anti_quark_mean.txt")
unpol_nll_anti_quark_sdev = np.loadtxt(f"./data/unpol_nll_p{px}_anti_quark_sdev.txt")

ax.fill_between(
    unpol_nll_anti_quark_x_ls,
    unpol_nll_anti_quark_mean - unpol_nll_anti_quark_sdev,
    unpol_nll_anti_quark_mean + unpol_nll_anti_quark_sdev,
    color=color,
    alpha=0.3,
    hatch="///",
    edgecolor=edgecolor,
)
ax.plot(unpol_nll_anti_quark_x_ls, unpol_nll_anti_quark_mean, color=color, linewidth=1)


#! Note: plotting next-to-leading order (NLO) order results of unpolarized valence quark PDF.
unpol_nlo_quark_x_ls = np.loadtxt(f"./data/unpol_nlo_p{px}_quark_x_ls.txt")
unpol_nlo_quark_mean = np.loadtxt(f"./data/unpol_nlo_p{px}_quark_mean.txt")
unpol_nlo_quark_sdev = np.loadtxt(f"./data/unpol_nlo_p{px}_quark_sdev.txt")

ax.fill_between(
    unpol_nlo_quark_x_ls,
    unpol_nlo_quark_mean - unpol_nlo_quark_sdev,
    unpol_nlo_quark_mean + unpol_nlo_quark_sdev,
    color=color,
    alpha=0.2,
    label=r"$P^z=$" + f"{px_gev} GeV, NLO",
)
ax.plot(unpol_nlo_quark_x_ls, unpol_nlo_quark_mean, color=color, linewidth=1)


#! Note: plotting next-to-leading order (NLO) order results of unpolarized anti-quark PDF. The anti-quark PDF has an extra minus sign to maintain the normalization condition: integrate the PDF from -1 to 1 gives q - qbar contribution. 
unpol_nlo_anti_quark_x_ls = np.loadtxt(f"./data/unpol_nlo_p{px}_anti_quark_x_ls.txt")
unpol_nlo_anti_quark_mean = np.loadtxt(f"./data/unpol_nlo_p{px}_anti_quark_mean.txt")
unpol_nlo_anti_quark_sdev = np.loadtxt(f"./data/unpol_nlo_p{px}_anti_quark_sdev.txt")

ax.fill_between(
    unpol_nlo_anti_quark_x_ls,
    unpol_nlo_anti_quark_mean - unpol_nlo_anti_quark_sdev,
    unpol_nlo_anti_quark_mean + unpol_nlo_anti_quark_sdev,
    alpha=0.2,
    color=color,
)
ax.plot(unpol_nlo_anti_quark_x_ls, unpol_nlo_anti_quark_mean, color=color, linewidth=1)

ax.axvspan(-0.18, 0.18, color='gray', alpha=0.3)
ax.axvspan(0.82, 1.0, color='gray', alpha=0.3)

ax.set_xlabel(r"$x$", **fs_p)
ax.set_ylabel(r"$f^{u-d} (x;~Q^2 = \mu^2 = 4~\rm{GeV}^2)$", **fs_p)
ax.legend(loc="upper right", **fs_small_p)
ax.set_xlim(-1, 1.)
ax.set_ylim(-0.5, 4)
plt.tight_layout()
plt.savefig(f"plot/unpolarized_valence_PDF.pdf", transparent=True)
plt.show()

# %%
