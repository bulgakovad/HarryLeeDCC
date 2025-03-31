import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

def interpolate_structure_functions(file_path, target_W, target_Q2):
    """
    Interpolates the structure functions W1 and W2 for given W and Q2 values
    using bicubic (cubic spline) interpolation.
    
    Parameters:
        file_path (str): Path to the input .dat file containing the data.
        target_W (float): The W (invariant mass) value at which to interpolate.
        target_Q2 (float): The Q2 (virtuality) value at which to interpolate.
        
    Returns:
        tuple: Interpolated (W1, W2) values.
    """
    # Load the data (assumed columns: W, Q2, W1, W2)
    data = np.loadtxt(file_path)
    
    # Extract columns
    W = data[:, 0]
    Q2 = data[:, 1]
    W1 = data[:, 2]
    W2 = data[:, 3]
    
    # Get unique grid points
    W_unique = np.unique(W)
    Q2_unique = np.unique(Q2)
    
    # Determine grid dimensions
    nW = len(W_unique)
    nQ2 = len(Q2_unique)
    
    # Reshape structure function arrays into 2D grids.
    W1_grid = W1.reshape(nW, nQ2)
    W2_grid = W2.reshape(nW, nQ2)
    
    # Build bicubic spline interpolators for W1 and W2
    interp_W1 = RectBivariateSpline(W_unique, Q2_unique, W1_grid, kx=3, ky=3)
    interp_W2 = RectBivariateSpline(W_unique, Q2_unique, W2_grid, kx=3, ky=3)
    
    # Evaluate the interpolators at the target values.
    W1_interp = interp_W1(target_W, target_Q2)[0, 0]
    W2_interp = interp_W2(target_W, target_Q2)[0, 0]
    
    return W1_interp, W2_interp

def plot_spline_vs_W(file_path, fixed_Q2, num_points=200):
    """
    Plots W1(W, Q2=fixed_Q2) and W2(W, Q2=fixed_Q2) as 1D splines
    over the range of W values in the data and saves the plot.
    
    Additionally, the original grid values (at the Q2 slice closest to fixed_Q2)
    are displayed as markers.
    
    Parameters:
        file_path (str): Path to the input data file (W, Q2, W1, W2)
        fixed_Q2 (float): The Q2 value at which to slice.
        num_points (int): Number of W points for the smooth spline curve.
    """
    # Load data
    data = np.loadtxt(file_path)
    W = data[:, 0]
    Q2 = data[:, 1]
    W1 = data[:, 2]
    W2 = data[:, 3]
    
    # Extract unique grid points
    W_unique = np.unique(W)
    Q2_unique = np.unique(Q2)
    
    # Reshape data into grids
    nW = len(W_unique)
    nQ2 = len(Q2_unique)
    W1_grid = W1.reshape(nW, nQ2)
    W2_grid = W2.reshape(nW, nQ2)
    
    # Build 2D splines
    interp_W1 = RectBivariateSpline(W_unique, Q2_unique, W1_grid, kx=3, ky=3)
    interp_W2 = RectBivariateSpline(W_unique, Q2_unique, W2_grid, kx=3, ky=3)
    
    # Find the grid Q2 value closest to fixed_Q2
    idx_q2 = np.argmin(np.abs(Q2_unique - fixed_Q2))
    grid_Q2 = Q2_unique[idx_q2]
    
    # Generate a fine array of W values for the smooth curve
    W_min, W_max = W_unique[0], W_unique[-1]
    W_vals = np.linspace(W_min, W_max, num_points)
    W1_smooth = [interp_W1(w, fixed_Q2)[0, 0] for w in W_vals]
    W2_smooth = [interp_W2(w, fixed_Q2)[0, 0] for w in W_vals]
    
    # Extract original grid data for the slice at grid_Q2
    W1_grid_vals = W1_grid[:, idx_q2]
    W2_grid_vals = W2_grid[:, idx_q2]
    
    # Plot smooth curves and overlay grid markers
    plt.figure(figsize=(7, 5))
    plt.plot(W_vals, W1_smooth, label='W1 spline', color='red', linewidth=2)
    plt.plot(W_vals, W2_smooth, label='W2 spline', color='blue', linewidth=2)
    plt.scatter(W_unique, W1_grid_vals, label='W1 grid', color='red', marker='o', s=5)
    plt.scatter(W_unique, W2_grid_vals, label='W2 grid', color='blue', marker='s', s=5)
    plt.xlabel('W (GeV)')
    plt.ylabel('Structure Functions')
    plt.title(f"W1, W2 vs W at fixed Q² = {fixed_Q2:.3f} GeV²\n(Grid Q² = {grid_Q2:.3f} GeV²)")
    plt.legend()
    plt.grid(True)
    
    filename = f"spline_vs_W_fixed_Q2_{fixed_Q2:.3f}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Plot saved as {filename}")

def plot_spline_vs_Q2(file_path, fixed_W, num_points=200):
    """
    Plots W1(W=fixed_W, Q2) and W2(W=fixed_W, Q2) as 1D splines
    over the range of Q² values in the data and saves the plot.
    
    Additionally, the original grid values (at the W slice closest to fixed_W)
    are displayed as markers.
    
    Parameters:
        file_path (str): Path to the input data file (W, Q2, W1, W2)
        fixed_W (float): The W value at which to slice.
        num_points (int): Number of Q² points for the smooth spline curve.
    """
    # Load data
    data = np.loadtxt(file_path)
    W = data[:, 0]
    Q2 = data[:, 1]
    W1 = data[:, 2]
    W2 = data[:, 3]
    
    # Extract unique grid points
    W_unique = np.unique(W)
    Q2_unique = np.unique(Q2)
    
    # Reshape data into grids
    nW = len(W_unique)
    nQ2 = len(Q2_unique)
    W1_grid = W1.reshape(nW, nQ2)
    W2_grid = W2.reshape(nW, nQ2)
    
    # Build 2D splines
    interp_W1 = RectBivariateSpline(W_unique, Q2_unique, W1_grid, kx=3, ky=3)
    interp_W2 = RectBivariateSpline(W_unique, Q2_unique, W2_grid, kx=3, ky=3)
    
    # Find the grid W value closest to fixed_W
    idx_w = np.argmin(np.abs(W_unique - fixed_W))
    grid_W = W_unique[idx_w]
    
    # Generate a fine array of Q² values for the smooth curve
    Q2_min, Q2_max = Q2_unique[0], Q2_unique[-1]
    Q2_vals = np.linspace(Q2_min, Q2_max, num_points)
    W1_smooth = [interp_W1(fixed_W, q2)[0, 0] for q2 in Q2_vals]
    W2_smooth = [interp_W2(fixed_W, q2)[0, 0] for q2 in Q2_vals]
    
    # Extract original grid data for the slice at grid_W
    W1_grid_vals = W1_grid[idx_w, :]
    W2_grid_vals = W2_grid[idx_w, :]
    
    # Plot smooth curves and overlay grid markers
    plt.figure(figsize=(7, 5))
    plt.plot(Q2_vals, W1_smooth, label='W1 spline', color='green', linewidth=2)
    plt.plot(Q2_vals, W2_smooth, label='W2 spline', color='orange', linewidth=2)
    plt.scatter(Q2_unique, W1_grid_vals, label='W1 grid', color='green', marker='o', s=5)
    plt.scatter(Q2_unique, W2_grid_vals, label='W2 grid', color='orange', marker='s', s=5)
    plt.xlabel('Q² (GeV²)')
    plt.ylabel('Structure Functions')
    plt.title(f"W1, W2 vs Q² at fixed W = {fixed_W:.3f} GeV\n(Grid W = {grid_W:.3f} GeV)")
    plt.legend()
    plt.grid(True)
    
    filename = f"spline_vs_Q2_fixed_W_{fixed_W:.3f}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Plot saved as {filename}")

def plot_2d_heatmap(file_path, function="W1", num_points=200):
    """
    Plots a 2D heatmap of the chosen structure function (W1 or W2) over the entire
    (W, Q²) domain using bicubic spline interpolation. The heatmap shows the
    interpolated values as a color map.
    
    Parameters:
        file_path (str): Path to the input data file (W, Q², W1, W2)
        function (str) : Which structure function to plot ("W1" or "W2")
        num_points (int): Number of points in each dimension for the interpolated grid.
    """
    # Load data
    data = np.loadtxt(file_path)
    W = data[:, 0]
    Q2 = data[:, 1]
    W1 = data[:, 2]
    W2 = data[:, 3]
    
    # Extract unique grid points
    W_unique = np.unique(W)
    Q2_unique = np.unique(Q2)
    
    # Reshape data into 2D arrays
    nW = len(W_unique)
    nQ2 = len(Q2_unique)
    W1_grid = W1.reshape(nW, nQ2)
    W2_grid = W2.reshape(nW, nQ2)
    
    # Create bicubic spline interpolators
    interp_W1 = RectBivariateSpline(W_unique, Q2_unique, W1_grid, kx=3, ky=3)
    interp_W2 = RectBivariateSpline(W_unique, Q2_unique, W2_grid, kx=3, ky=3)
    
    # Create a fine grid for plotting
    W_min, W_max = W_unique[0], W_unique[-1]
    Q2_min, Q2_max = Q2_unique[0], Q2_unique[-1]
    W_vals = np.linspace(W_min, W_max, num_points)
    Q2_vals = np.linspace(Q2_min, Q2_max, num_points)
    X, Y = np.meshgrid(W_vals, Q2_vals, indexing='xy')
    
    # Evaluate the chosen function on the fine grid
    Z = np.zeros_like(X)
    for i in range(num_points):
        for j in range(num_points):
            if function.upper() == "W1":
                Z[j, i] = interp_W1(X[j, i], Y[j, i])
            else:
                Z[j, i] = interp_W2(X[j, i], Y[j, i])
    
    # Plot the heatmap without overlaying grid points
    plt.figure(figsize=(7, 6))
    plt.pcolormesh(X, Y, Z, cmap="viridis", shading="auto")
    plt.colorbar(label=function.upper())
    plt.xlabel("W (GeV)")
    plt.ylabel("Q² (GeV²)")
    plt.title(f"{function.upper()} Bicubic Interpolation Heatmap")
    
    filename = f"heatmap_{function.upper()}.png"
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"2D heatmap saved as {filename}")

if __name__ == "__main__":
    file_path = "input_data/wempx.dat"
    
    # Example: 2D interpolation at a single point
    target_W = 1.505    # example value
    target_Q2 = 2.674     # example value
    w1_val, w2_val = interpolate_structure_functions(file_path, target_W, target_Q2)
    print(f"Interpolated W1({target_W}, {target_Q2}) = {w1_val}")
    print(f"Interpolated W2({target_W}, {target_Q2}) = {w2_val}")
    
    # Example: 1D spline plot vs W at fixed Q²
    plot_spline_vs_W(file_path, target_Q2, num_points=200)
    
    # Example: 1D spline plot vs Q² at fixed W
    plot_spline_vs_Q2(file_path, target_W, num_points=200)
    
    # Example: 2D heatmaps for W1 and W2 with increased bins
    plot_2d_heatmap(file_path, function="W1", num_points=200)
    plot_2d_heatmap(file_path, function="W2", num_points=200)
