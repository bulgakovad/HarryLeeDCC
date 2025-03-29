import numpy as np
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
    # Load the data from the file
    data = np.loadtxt(file_path)
    
    # Extract the columns: W, Q2, W1, and W2
    W = data[:, 0]
    Q2 = data[:, 1]
    W1 = data[:, 2]
    W2 = data[:, 3]
    
    # Get unique grid points. We assume the file is structured on a grid.
    W_unique = np.unique(W)
    Q2_unique = np.unique(Q2)
    
    # Determine grid dimensions
    nW = len(W_unique)
    nQ2 = len(Q2_unique)
    
    # Reshape the structure function arrays into 2D grids.
    # This assumes that for each unique W, the Q2 values appear in sorted order.
    W1_grid = W1.reshape(nW, nQ2)
    W2_grid = W2.reshape(nW, nQ2)
    
    # Build bicubic spline interpolators for W1 and W2
    interp_W1 = RectBivariateSpline(W_unique, Q2_unique, W1_grid, kx=3, ky=3)
    interp_W2 = RectBivariateSpline(W_unique, Q2_unique, W2_grid, kx=3, ky=3)
    
    # Evaluate the interpolators at the target W and Q2 values.
    W1_interp = interp_W1(target_W, target_Q2)[0, 0]
    W2_interp = interp_W2(target_W, target_Q2)[0, 0]
    
    return W1_interp, W2_interp

# Example usage:
if __name__ == "__main__":
    file_path = "input_data/wempx.dat"
    # Choose target values within the valid range of W and Q2 from the file
    target_W = 5   # example value
    target_Q2 = 0.1   # example value
    w1, w2 = interpolate_structure_functions(file_path, target_W, target_Q2)
    print("Interpolated W1:", w1)
    print("Interpolated W2:", w2)
