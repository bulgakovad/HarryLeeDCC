#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Ask user for Q² value to plot
    try:
        q2_target = float(input("Enter Q2 value to plot: "))
    except ValueError:
        print("Invalid input. Please enter a numerical value for Q2.")
        return

    filename = "output/tst_CJpdf_ISET=400.out"
    xs = []
    F1_vals = []
    F2_vals = []

    tol = 1e-6  # tolerance for Q² matching

    # Read the file line by line
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
            # Try to parse the line into floats
            try:
                data = [float(val) for val in line.split()]
            except ValueError:
                # Skip lines that cannot be parsed (e.g. header)
                continue
            # Expect at least 13 columns: Q², x, then 11 PDF values.
            if len(data) < 13:
                continue

            Q2_val = data[0]
            # Only use rows that match the requested Q2 value
            if abs(Q2_val - q2_target) < tol:
                x_val = data[1]
                # According to the output table ordering:
                #   0: Q²
                #   1: x
                #   2: u    (i.e. x*u)
                #   3: d    (i.e. x*d)
                #   4: g
                #   5: ub   (i.e. x*ubar)
                #   6: db   (i.e. x*dbar)
                u   = data[2]
                d   = data[3]
                ub  = data[5]
                dbar = data[6]

                # Compute F2 and F1.
                # Since the table already contains x*u, etc., we use:
                F2 = (4.0/9.0) * (u + ub) + (1.0/9.0) * (d + dbar)
                F1 = F2 / (2 * x_val)
                xs.append(x_val)
                F2_vals.append(F2)
                F1_vals.append(F1)

    if not xs:
        print("No data found for Q2 =", q2_target)
        return

    # Convert lists to numpy arrays and sort them by x
    xs = np.array(xs)
    F1_vals = np.array(F1_vals)
    F2_vals = np.array(F2_vals)
    order = np.argsort(xs)
    xs = xs[order]
    F1_vals = F1_vals[order]
    F2_vals = F2_vals[order]

    # Create a canvas with 1 row and 2 columns of subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Left subplot: F1 vs x
    ax1.plot(xs, F1_vals, label="F1", color="blue")
    ax1.set_xlabel("x")
    ax1.set_ylabel("F1")
    ax1.set_title("F1 vs x for Q2 = {:.3f}".format(q2_target))
    ax1.grid(True)
    ax1.legend()
    
    # Right subplot: F2 vs x
    ax2.plot(xs, F2_vals, label="F2", color="red")
    ax2.set_xlabel("x")
    ax2.set_ylabel("F2")
    ax2.set_title("F2 vs x for Q2 = {:.3f}".format(q2_target))
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    
    # Save the plot to a PNG file
    save_filename = "F1_F2_vs_x_Q2_{:.3f}.png".format(q2_target)
    plt.savefig(save_filename)
    print("Plot saved as", save_filename)
    
    plt.show()

if __name__ == "__main__":
    main()
