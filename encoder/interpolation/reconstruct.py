import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

def reconstruct_from_minimal_storage(compressed_key_points, num_points=2379):
    """
    Reconstruct the full boundary from compressed key points
    """
    # Remove duplicates while preserving order
    unique_points = []
    for point in compressed_key_points:
        if len(unique_points) == 0 or not np.allclose(point, unique_points[-1]):
            unique_points.append(point)
    compressed_key_points = np.array(unique_points)
    
    # Ensure we have enough points for spline fitting
    if len(compressed_key_points) < 4:
        # Fallback: linear interpolation if too few points
        t_original = np.linspace(0, 1, len(compressed_key_points))
        t_new = np.linspace(0, 1, num_points)
        
        x_interp = np.interp(t_new, t_original, compressed_key_points[:, 0])
        y_interp = np.interp(t_new, t_original, compressed_key_points[:, 1])
        
        reconstructed_boundary = np.column_stack([x_interp, y_interp])
        return reconstructed_boundary, None
    
    # Ensure the shape is closed (connect last point to first)
    if not np.allclose(compressed_key_points[0], compressed_key_points[-1]):
        closed_points = np.vstack([compressed_key_points, compressed_key_points[0]])
    else:
        closed_points = compressed_key_points
    
    # Use spline interpolation to reconstruct the full boundary
    from scipy.interpolate import splprep, splev
    
    x = closed_points[:, 0]
    y = closed_points[:, 1]
    
    try:
        # Fit spline through the compressed key points
        # Use smoothing to handle potential duplicate points
        smoothing = len(closed_points) * 0.1  # Small smoothing
        tck, u = splprep([x, y], s=smoothing, per=1)  # periodic spline for closed shapes
        
        # Generate reconstructed points
        t_new = np.linspace(0, 1, num_points)
        x_reconstructed, y_reconstructed = splev(t_new, tck)
        
        reconstructed_boundary = np.column_stack([x_reconstructed, y_reconstructed])
        
        return reconstructed_boundary, tck
        
    except Exception as e:
        print(f"⚠️  Spline fitting failed, using linear interpolation: {e}")
        # Fallback to linear interpolation
        t_original = np.linspace(0, 1, len(closed_points))
        t_new = np.linspace(0, 1, num_points)
        
        x_interp = np.interp(t_new, t_original, closed_points[:, 0])
        y_interp = np.interp(t_new, t_original, closed_points[:, 1])
        
        reconstructed_boundary = np.column_stack([x_interp, y_interp])
        return reconstructed_boundary, None


def save_compressed_matrix(compressed_matrix, filename):
    """
    Save the compressed matrix to file
    """
    # Save as numpy array (most efficient)
    np.save(f"{filename}.npy", compressed_matrix)
    
    # Also save as CSV for readability
    np.savetxt(f"{filename}.csv", compressed_matrix, delimiter=",", fmt='%.3f')
    
    print(f"✅ Compressed matrix saved:")
    print(f"   - {filename}.npy ({compressed_matrix.nbytes} bytes)")
    print(f"   - {filename}.csv (human readable)")

def load_and_reconstruct(filename, original_point_count):
    """
    Load compressed matrix and reconstruct full boundary
    """
    # Load the compressed matrix
    compressed_matrix = np.load(f"{filename}.npy")
    
    # Reconstruct the full boundary
    reconstructed_boundary, _ = reconstruct_from_minimal_storage(
        compressed_matrix, 
        num_points=original_point_count
    )
    
    print(f"✅ Reconstruction from compressed matrix:")
    print(f"   - Loaded {len(compressed_matrix)} key points")
    print(f"   - Reconstructed to {len(reconstructed_boundary)} points")
    
    return reconstructed_boundary, compressed_matrix

def analyze_final_storage(original_boundary, compressed_matrix):
    """
    Final storage analysis comparing all options
    """
    original_size = original_boundary.nbytes
    compressed_size = compressed_matrix.nbytes
    
    print("\n" + "="*60)
    print("💾 FINAL STORAGE COMPARISON")
    print("="*60)
    
    print(f"{'Representation':<25} {'Points':<10} {'Size':<15} {'Reduction':<15}")
    print("-" * 65)
    print(f"{'Original Boundary':<25} {len(original_boundary):<10} {original_size:<15,} {'0%':<15}")
    print(f"{'Compressed Matrix':<25} {len(compressed_matrix):<10} {compressed_size:<15,} {(1-compressed_size/original_size)*100:<14.1f}%")
    
    # Calculate bits per point
    original_bpp = (original_size * 8) / len(original_boundary)
    compressed_bpp = (compressed_size * 8) / len(original_boundary)
    
    print(f"\n📈 Efficiency Metrics:")
    print(f"  - Original: {original_bpp:.1f} bits per point")
    print(f"  - Compressed: {compressed_bpp:.1f} bits per point")
    print(f"  - Efficiency: {original_bpp/compressed_bpp:.1f}x more efficient")


