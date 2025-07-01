import numpy as np
import matplotlib.pyplot as plt

# === Load bg_params.dat ===
def load_bg_params(path="bg_res_params/bg_params.dat"):
    q2_vals = []
    p1_vals = []
    p2_vals = []

    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            tokens = line.strip().split(",")
            q2 = float(tokens[0])
            p1 = float(tokens[1])
            p2 = float(tokens[2])
            q2_vals.append(q2)
            p1_vals.append(p1)
            p2_vals.append(p2)

    return np.array(q2_vals), np.array(p1_vals), np.array(p2_vals)

# === Load res_params.dat ===
def load_res_params(path="bg_res_params/res_params.dat"):
    q2_vals = []
    param_matrix = []

    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            tokens = [float(x) for x in line.strip().split(",")]
            q2 = tokens[0]
            params = tokens[1:]
            q2_vals.append(q2)
            param_matrix.append(params)

    return np.array(q2_vals), np.array(param_matrix)  # shape: (N, 11)


# === Plot BG params ===
def plot_bg(q2_vals, p1_vals, p2_vals):
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    axes[0].plot(q2_vals, p1_vals, marker='o', color='tab:blue')
    axes[0].set_ylabel("Background p₁")
    axes[0].grid(True)

    axes[1].plot(q2_vals, p2_vals, marker='s', color='tab:green')
    axes[1].set_xlabel(r"$Q^2$ (GeV$^2$)")
    axes[1].set_ylabel("Background p₂")
    axes[1].grid(True)

    fig.suptitle("Fitted Background Parameters vs $Q^2$")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("bg_fit_params_vs_q2.png", dpi=150)
    plt.close(fig)
    print("Saved: bg_fit_params_vs_q2.png")


# === Plot RES params ===
def plot_res(q2_vals, param_matrix):
    param_labels = [f"p{i+1}" for i in range(11)]
    nrows, ncols = 4, 3

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(11):
        axes[i].plot(q2_vals, param_matrix[:, i], marker='o')
        axes[i].set_title(param_labels[i])
        axes[i].set_xlabel(r"$Q^2$ (GeV$^2$)")
        axes[i].grid(True)

    for i in range(11, len(axes)):
        fig.delaxes(axes[i])

    fig.suptitle("Fitted Resonance Parameters vs $Q^2$")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("res_fit_params_vs_q2.png", dpi=150)
    plt.close(fig)
    print("Saved: res_fit_params_vs_q2.png")


# === Main execution ===
if __name__ == "__main__":
    q2_bg, p1_bg, p2_bg = load_bg_params()
    plot_bg(q2_bg, p1_bg, p2_bg)

    q2_res, res_matrix = load_res_params()
    plot_res(q2_res, res_matrix)
