#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import math

def main():
    # Ask user for Q2 value (in GeV^2)
    try:
        q2_target = float(input("Enter Q2 value (in GeV^2) to plot: "))
    except ValueError:
        print("Invalid input. Please enter a numerical value for Q2.")
        return

    # Nucleon mass in GeV
    m = 0.9383

    filename = "output/tst_CJpdf_ISET=400.out"
    xi_list = []
    F1_trans_list = []
    F2_trans_list = []

    tol = 1e-6  # tolerance for Q2 matching

    # Read the file line by line
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip empty lines
            try:
                # Parse the line into floats
                data = [float(val) for val in line.split()]
            except ValueError:
                continue  # skip non-numeric lines (e.g. header)
            # Expect at least 13 columns: Q2, x, then 11 PDF values.
            if len(data) < 13:
                continue

            Q2_val = data[0]
            # Process only rows with Q2 close to the target value
            if abs(Q2_val - q2_target) < tol:
                x_val = data[1]
                # Table ordering (0-indexed):
                #   0: Q2, 1: x, 2: x*u, 3: x*d, 4: x*g, 5: x*ubar, 6: x*dbar, ...
                u    = data[2]  # x*u
                d    = data[3]  # x*d
                ub   = data[5]  # x*ubar
                dbar = data[6]  # x*dbar

                # Compute structure functions (with x already multiplied)
                F2 = (4.0/9.0) * (u + ub) + (1.0/9.0) * (d + dbar)
                F1 = F2 / (2 * x_val) if x_val != 0 else 0.0

                # Compute the Nachtmann variable ξ:
                # t = 4*x^2*m^2/Q2_target
                t = 4 * x_val**2 * m**2 / q2_target
                xi = 2 * x_val / (1 + math.sqrt(1 + t))

                F2_trans = F2 
                F1_trans = F1 

                xi_list.append(xi)
                F2_trans_list.append(F2_trans)
                F1_trans_list.append(F1_trans)

    if not xi_list:
        print("No data found for Q2 =", q2_target)
        return

    # Convert lists to numpy arrays and sort by ξ
    xi_arr = np.array(xi_list)
    F1_arr = np.array(F1_trans_list)
    F2_arr = np.array(F2_trans_list)
    order = np.argsort(xi_arr)
    xi_arr = xi_arr[order]
    F1_arr = F1_arr[order]
    F2_arr = F2_arr[order]

    # Create a canvas with 1 row and 2 columns of subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Left subplot: F1 vs ξ
    axs[0].plot(xi_arr, F1_arr, label="F1 (Nachtmann)", color="blue")
    axs[0].set_xlabel(r"$\xi$")
    axs[0].set_ylabel("F1")
    axs[0].set_title(r"F1 vs $\xi$ for Q2 = {:.3f} GeV$^2$".format(q2_target))
    axs[0].grid(True)
    axs[0].legend()

    # Right subplot: F2 vs ξ
    axs[1].plot(xi_arr, F2_arr, label="F2 (Nachtmann)", color="red")
    axs[1].set_xlabel(r"$\xi$")
    axs[1].set_ylabel("F2")
    axs[1].set_title(r"F2 vs $\xi$ for Q2 = {:.3f} GeV$^2$".format(q2_target))
    axs[1].grid(True)
    axs[1].legend()

    plt.tight_layout()
    save_filename = "F1_F2_vs_xi_separate_Q2_{:.3f}.png".format(q2_target)
    plt.savefig(save_filename)
    print("Plot saved as", save_filename)
    plt.show()

if __name__ == "__main__":
    main()
