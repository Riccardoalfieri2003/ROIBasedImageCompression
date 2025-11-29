import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def interpolate_shape_spline(coordinates, smoothing=0, num_points=100):
    """
    Simple parametric spline interpolation for ANY irregular shape
    
    Parameters:
    -----------
    coordinates : list of (x,y) tuples
    smoothing : 0 for exact interpolation, >0 for smoothing
    num_points : number of points in reconstructed shape
    
    Returns:
    --------
    reconstructed coordinates and spline coefficients
    """
    # Convert to numpy array
    coords = np.array(coordinates)
    
    # Ensure closed shape
    if not np.allclose(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0]])
    
    # Separate x and y coordinates
    x = coords[:, 0]
    y = coords[:, 1]
    
    # Create parameter t (0 to 1) along the curve
    t = np.linspace(0, 1, len(coords))
    
    # Fit parametric spline
    try:
        # splprep does parametric spline: x(t) and y(t)
        tck, u = splprep([x, y], s=smoothing, per=1)  # per=1 for closed curves
        
        # Generate interpolated points
        t_new = np.linspace(0, 1, num_points)
        x_new, y_new = splev(t_new, tck)
        
        # Calculate error (for exact interpolation, should be near 0)
        if smoothing == 0:
            # Sample at original points to check accuracy
            x_orig_interp, y_orig_interp = splev(t, tck)
            error = np.mean(np.sqrt((x_orig_interp - x)**2 + (y_orig_interp - y)**2))
        else:
            error = 0  # With smoothing, we don't expect exact match
        
        return {
            'reconstructed': np.column_stack([x_new, y_new]),
            'spline_coefficients': tck,
            'error': error,
            'original_points': len(coords),
            'reconstructed_points': num_points
        }
        
    except Exception as e:
        print(f"Spline interpolation failed: {e}")
        return None

def compress_shape_spline(coordinates, compression_ratio=0.1, num_points=100):
    """
    Compress shape by keeping only key points + spline interpolation
    """
    coords = np.array(coordinates)
    
    if not np.allclose(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0]])
    
    # Select key points for compression
    n_key_points = max(4, int(len(coords) * compression_ratio))
    key_indices = select_key_points(coords, n_key_points)
    key_points = coords[key_indices]
    
    # Interpolate using only key points
    result = interpolate_shape_spline(key_points, smoothing=0, num_points=num_points)
    
    if result:
        # Calculate compression metrics
        original_size = len(coords)
        compressed_size = n_key_points
        storage_reduction = (1 - compressed_size/original_size) * 100
        
        result['compression_info'] = {
            'original_points': original_size,
            'key_points': n_key_points,
            'compression_ratio': compression_ratio,
            'storage_reduction': f"{storage_reduction:.1f}%"
        }
    
    return result

def select_key_points(coords, n_points):
    """
    Smart selection of key points for spline compression
    """
    if n_points >= len(coords):
        return np.arange(len(coords))
    
    # Always include first point
    selected = [0]
    
    # Calculate cumulative distance along curve
    distances = np.zeros(len(coords))
    for i in range(1, len(coords)):
        distances[i] = distances[i-1] + np.linalg.norm(coords[i] - coords[i-1])
    
    # Select evenly spaced points by arc length
    target_distances = np.linspace(0, distances[-1], n_points)
    
    for dist in target_distances[1:]:  # Skip first (already included)
        idx = np.argmin(np.abs(distances - dist))
        if idx not in selected:
            selected.append(idx)
    
    return np.array(sorted(selected))

def create_test_irregular_shapes():
    """
    Create various irregular shapes for testing
    """
    shapes = {}
    
    # 1. Very irregular blob
    theta = np.linspace(0, 2*np.pi, 100, endpoint=False)
    r = 1 + 0.5 * np.sin(3*theta) + 0.3 * np.cos(5*theta) + 0.2 * np.sin(7*theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    shapes['Complex Blob'] = list(zip(x, y))
    
    # 2. Kidney bean (highly asymmetric)
    t = np.linspace(0, 2*np.pi, 80, endpoint=False)
    x = 1.5 * np.cos(t) + 0.8 * np.cos(2*t)
    y = np.sin(t) + 0.5 * np.sin(3*t)
    shapes['Kidney Bean'] = list(zip(x, y))
    
    # 3. Random walk shape (very irregular)
    np.random.seed(42)
    t = np.linspace(0, 2*np.pi, 120, endpoint=False)
    r = 1 + 0.6 * np.cumsum(np.random.randn(len(t))) * 0.1
    r = np.convolve(r, np.ones(10)/10, mode='same')  # Smooth a bit
    x = r * np.cos(t)
    y = r * np.sin(t)
    shapes['Random Walk'] = list(zip(x, y))
    
    # 4. Star with many points
    t = np.linspace(0, 2*np.pi, 150, endpoint=False)
    r = 1 + 0.4 * np.sin(7*t)
    x = r * np.cos(t)
    y = r * np.sin(t)
    shapes['7-pointed Star'] = list(zip(x, y))
    
    return shapes

def test_spline_interpolation():
    """
    Test spline interpolation on various irregular shapes
    """
    shapes = create_test_irregular_shapes()
    
    print("SPLINE INTERPOLATION TEST")
    print("=" * 60)
    
    for shape_name, coordinates in shapes.items():
        print(f"\n🔷 Testing: {shape_name}")
        print(f"   Original points: {len(coordinates)}")
        
        # Test exact interpolation (should be perfect)
        result = interpolate_shape_spline(coordinates, smoothing=0)
        if result:
            print(f"   Exact interpolation error: {result['error']:.6f}")
        
        # Test compression
        compression_ratios = [0.1, 0.2, 0.3]
        for ratio in compression_ratios:
            compressed = compress_shape_spline(coordinates, ratio)
            if compressed:
                info = compressed['compression_info']
                print(f"   Compression {ratio:.0%}: {info['key_points']} key points, "
                      f"storage: {info['storage_reduction']}")
        
        # Visualize
        visualize_spline_results(coordinates, shape_name)

def visualize_spline_results(coordinates, shape_name):
    """
    Visualize spline interpolation results
    """
    # Test different compression ratios
    ratios = [0.05, 0.1, 0.2, 0.5]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot original
    original_x, original_y = zip(*coordinates)
    axes[0].plot(original_x, original_y, 'bo-', linewidth=2, markersize=3, alpha=0.7)
    axes[0].set_title(f'{shape_name}\nOriginal\n{len(coordinates)} points')
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)
    
    # Plot compressed versions
    for i, ratio in enumerate(ratios):
        result = compress_shape_spline(coordinates, ratio)
        if result:
            recon = result['reconstructed']
            key_points = result['compression_info']['key_points']
            
            recon_x, recon_y = recon[:, 0], recon[:, 1]
            axes[i+1].plot(recon_x, recon_y, 'r-', linewidth=2, label='Spline')
            
            # Plot key points used
            compressed_coords = select_key_points(np.array(coordinates), key_points)
            key_x = [coordinates[idx][0] for idx in compressed_coords]
            key_y = [coordinates[idx][1] for idx in compressed_coords]
            axes[i+1].plot(key_x, key_y, 'go', markersize=4, 
                          markerfacecolor='none', markeredgewidth=2, label='Key points')
            
            axes[i+1].set_title(f'Compression: {ratio:.0%}\n{key_points} key points')
            axes[i+1].set_aspect('equal')
            axes[i+1].grid(True, alpha=0.3)
            axes[i+1].legend()
    
    # Hide empty subplots
    for i in range(len(ratios) + 1, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def compare_spline_vs_original(coordinates, shape_name):
    """
    Detailed comparison of original vs spline reconstruction
    """
    # Test with 10% compression
    result = compress_shape_spline(coordinates, 0.1, num_points=len(coordinates))
    
    if not result:
        return
    
    original = np.array(coordinates)
    reconstructed = result['reconstructed']
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original
    ax1.plot(original[:, 0], original[:, 1], 'bo-', linewidth=1, markersize=2)
    ax1.set_title('Original Shape')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Reconstructed
    ax2.plot(reconstructed[:, 0], reconstructed[:, 1], 'ro-', linewidth=1, markersize=2)
    ax2.set_title('Spline Reconstruction\n(10% compression)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # Overlay
    ax3.plot(original[:, 0], original[:, 1], 'b-', linewidth=2, label='Original', alpha=0.7)
    ax3.plot(reconstructed[:, 0], reconstructed[:, 1], 'r--', linewidth=2, label='Spline')
    ax3.set_title('Overlay Comparison')
    ax3.set_aspect('equal')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Simple usage for your boundaries
def compress_boundary_simple(boundary_coordinates, compression_ratio=0.1):
    """
    Simple one-function compression for your boundaries
    """
    return compress_shape_spline(boundary_coordinates, compression_ratio)

# Test the spline approach
if __name__ == "__main__":
    print("Testing Parametric Spline Interpolation on Irregular Shapes")
    print("=" * 70)
    
    """# Run comprehensive tests
    test_spline_interpolation()
    
    # Test on a specific shape with detailed comparison
    shapes = create_test_irregular_shapes()
    test_shape = 'Complex Blob'
    if test_shape in shapes:
        compare_spline_vs_original(shapes[test_shape], test_shape)
    
    print("\n" + "=" * 70)
    print("SUMMARY: Splines work excellently for irregular shapes because:")
    print("✅ Parametric approach handles any shape (not just functions)")
    print("✅ Exact interpolation possible (zero error)")
    print("✅ Excellent compression ratios (5-20% typical)")
    print("✅ Preserves smooth curves naturally")
    print("✅ Fast and robust computation")"""


   



















def print_spline_coefficients(result):
    """
    Print the spline coefficients in a readable format
    """
    if not result or 'spline_coefficients' not in result:
        print("No spline coefficients found in result")
        return
    
    tck = result['spline_coefficients']
    knots, coefficients, degree = tck
    
    print("\n📊 SPLINE COEFFICIENTS ANALYSIS:")
    print("=" * 50)
    print(f"Degree: {degree}")
    print(f"Number of knots: {len(knots)}")
    print(f"Number of coefficients: {len(coefficients[0])} for x, {len(coefficients[1])} for y")
    
    print(f"\n🔧 Knot vector (first 10):")
    print(f"  {knots[:10]}...")
    
    print(f"\n📈 X coefficients (first 10):")
    print(f"  {coefficients[0][:10]}...")
    
    print(f"\n📉 Y coefficients (first 10):")
    print(f"  {coefficients[1][:10]}...")
    
    # Storage analysis
    original_size = result['original_points'] * 2  # x and y
    compressed_size = len(knots) + len(coefficients[0]) + len(coefficients[1])
    compression_ratio = compressed_size / original_size
    
    print(f"\n💾 STORAGE ANALYSIS:")
    print(f"  Original: {original_size} values ({result['original_points']} points × 2)")
    print(f"  Compressed: {compressed_size} values")
    print(f"  Compression ratio: {compression_ratio:.1%}")
    print(f"  Storage reduction: {(1 - compression_ratio)*100:.1f}%")

def print_compressed_coefficients(compressed_result):
    """
    Print coefficients for compressed spline
    """
    if not compressed_result or 'spline_coefficients' not in compressed_result:
        print("No compressed coefficients found")
        return
    
    tck = compressed_result['spline_coefficients']
    knots, coefficients, degree = tck
    info = compressed_result.get('compression_info', {})
    
    print(f"\n🎯 COMPRESSED SPLINE (Ratio: {info.get('compression_ratio', 'N/A'):.0%}):")
    print("=" * 50)
    print(f"Key points used: {info.get('key_points', 'N/A')}")
    print(f"Degree: {degree}")
    print(f"Knots: {len(knots)}")
    print(f"X coefficients: {len(coefficients[0])}")
    print(f"Y coefficients: {len(coefficients[1])}")
    
    # Show actual values
    print(f"\n📋 First 5 knots:")
    for i, knot in enumerate(knots[:5]):
        print(f"  Knot[{i}]: {knot:.6f}")
    
    print(f"\n📋 First 5 X coefficients:")
    for i, coeff in enumerate(coefficients[0][:5]):
        print(f"  X[{i}]: {coeff:.6f}")
    
    print(f"\n📋 First 5 Y coefficients:")
    for i, coeff in enumerate(coefficients[1][:5]):
        print(f"  Y[{i}]: {coeff:.6f}")

# Your modified main
def main():
    # Your coordinates here
    coordinates = []  # Your boundary coordinates
    
    print("SPLINE INTERPOLATION WITH COEFFICIENT ANALYSIS")
    print("=" * 60)
    
    # Test exact interpolation
    result = interpolate_shape_spline(coordinates, smoothing=0)

    if result:
        print(f"✅ Exact interpolation error: {result['error']:.6f}")
        
        # Print coefficients for exact interpolation
        print_spline_coefficients(result)
        
        # Test compression
        compression_ratios = [0.1, 0.2, 0.3]
        for ratio in compression_ratios:
            compressed = compress_shape_spline(coordinates, ratio)
            if compressed:
                info = compressed['compression_info']
                print(f"\n{'='*50}")
                print(f"Compression {ratio:.0%}: {info['key_points']} key points, "
                      f"storage: {info['storage_reduction']}")
                
                # Print compressed coefficients
                print_compressed_coefficients(compressed)
        
        # Visualize
        visualize_spline_results(coordinates, "Custom")

# Alternative: Simple coefficient extraction
def extract_spline_data_for_storage(result):
    """
    Extract spline data in a format suitable for storage
    """
    if not result or 'spline_coefficients' not in result:
        return None
    
    tck = result['spline_coefficients']
    knots, coefficients, degree = tck
    
    storage_data = {
        'degree': int(degree),
        'knots': knots.tolist(),
        'x_coefficients': coefficients[0].tolist(),
        'y_coefficients': coefficients[1].tolist(),
        'original_points': result.get('original_points', 0),
        'reconstructed_points': result.get('reconstructed_points', 0)
    }
    
    # Add compression info if available
    if 'compression_info' in result:
        storage_data['compression_info'] = result['compression_info']
    
    return storage_data

def print_storage_format(storage_data):
    """
    Print the data in a storage-friendly format
    """
    print("\n💾 STORAGE-FRIENDLY FORMAT:")
    print("=" * 50)
    print(f"Degree: {storage_data['degree']}")
    print(f"Knots: {len(storage_data['knots'])} values")
    print(f"X coefficients: {len(storage_data['x_coefficients'])} values")
    print(f"Y coefficients: {len(storage_data['y_coefficients'])} values")
    
    print(f"\n📦 Total values to store: {len(storage_data['knots']) + len(storage_data['x_coefficients']) + len(storage_data['y_coefficients']) + 1}")
    print(f"📦 vs Original: {storage_data['original_points'] * 2} values")
    
    compression = 1 - (len(storage_data['knots']) + len(storage_data['x_coefficients']) + len(storage_data['y_coefficients']) + 1) / (storage_data['original_points'] * 2)
    print(f"📦 Compression: {compression:.1%}")


















































import numpy as np
from scipy.interpolate import splprep, splev

def compress_shape_spline_direct(coordinates, compression_ratio=0.1, num_points=100):
    """
    Direct spline compression in one function call
    """
    coords = np.array(coordinates)
    
    # Ensure closed shape
    if not np.allclose(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0]])
    
    # Select key points
    n_key_points = max(4, int(len(coords) * compression_ratio))
    
    # Smart point selection (curvature-based)
    def select_key_points(coords, n_points):
        if n_points >= len(coords):
            return np.arange(len(coords))
        
        # Calculate curvature
        curvature = np.zeros(len(coords))
        for i in range(1, len(coords)-1):
            v1 = coords[i] - coords[i-1]
            v2 = coords[i+1] - coords[i]
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1, 1)
                curvature[i] = np.arccos(cos_angle)
        
        curvature[0] = curvature[1]
        curvature[-1] = curvature[-2]
        
        # Select points with highest curvature + evenly spaced
        selected = [0]  # Always include first point
        high_curve_indices = np.argsort(curvature)[-(n_points-1):][::-1]
        selected.extend(high_curve_indices.tolist())
        
        return np.array(sorted(selected))
    
    key_indices = select_key_points(coords, n_key_points)
    key_points = coords[key_indices]
    
    # Fit parametric spline
    x_key = key_points[:, 0]
    y_key = key_points[:, 1]
    
    tck, u = splprep([x_key, y_key], s=0, per=1)
    
    # Reconstruct
    t_new = np.linspace(0, 1, num_points)
    x_new, y_new = splev(t_new, tck)
    
    # Calculate error
    t_original = np.linspace(0, 1, len(coords))
    x_orig_interp, y_orig_interp = splev(t_original, tck)
    error = np.mean(np.sqrt((x_orig_interp - coords[:, 0])**2 + 
                           (y_orig_interp - coords[:, 1])**2))
    
    return {
        'reconstructed': np.column_stack([x_new, y_new]),
        'spline_coefficients': tck,
        'key_points': key_points,
        'key_indices': key_indices,
        'metrics': {
            'mean_error': error,
            'compression_ratio': compression_ratio,
            'original_points': len(coords),
            'key_points_used': n_key_points,
            'storage_reduction': f"{(1 - n_key_points/len(coords))*100:.1f}%"
        }
    }

def print_all_coefficients_detailed(result):
    """
    Print ALL coefficients in detail
    """
    if not result or 'spline_coefficients' not in result:
        print("❌ No spline coefficients found")
        return
    
    tck = result['spline_coefficients']
    knots, coefficients, degree = tck
    metrics = result['metrics']
    
    print("\n" + "="*70)
    print("📊 COMPLETE SPLINE COEFFICIENTS ANALYSIS")
    print("="*70)
    
    print(f"🎯 Compression: {metrics['compression_ratio']:.1%}")
    print(f"📈 Original points: {metrics['original_points']}")
    print(f"🔑 Key points used: {metrics['key_points_used']}")
    print(f"💾 Storage reduction: {metrics['storage_reduction']}")
    print(f"📏 Degree: {degree}")
    print(f"🎯 Reconstruction error: {metrics['mean_error']:.6f}")
    
    print(f"\n🔧 KNOT VECTOR ({len(knots)} values):")
    print("-" * 50)
    for i, knot in enumerate(knots):
        print(f"  Knot[{i:3d}]: {knot:.8f}")
    
    print(f"\n📈 X COEFFICIENTS ({len(coefficients[0])} values):")
    print("-" * 50)
    for i, coeff in enumerate(coefficients[0]):
        print(f"  X[{i:3d}]: {coeff:12.8f}")
    
    print(f"\n📉 Y COEFFICIENTS ({len(coefficients[1])} values):")
    print("-" * 50)
    for i, coeff in enumerate(coefficients[1]):
        print(f"  Y[{i:3d}]: {coeff:12.8f}")
    
    # Storage summary
    total_storage = len(knots) + len(coefficients[0]) + len(coefficients[1])
    original_storage = metrics['original_points'] * 2
    compression_ratio_storage = total_storage / original_storage
    
    print(f"\n💾 STORAGE SUMMARY:")
    print(f"  Original: {original_storage} values")
    print(f"  Compressed: {total_storage} values")
    print(f"  Compression ratio: {compression_ratio_storage:.1%}")
    print(f"  Actual storage reduction: {(1-compression_ratio_storage)*100:.1f}%")

def filter_coefficients_by_threshold(result, threshold=0.001):
    """
    Filter out coefficients below a certain threshold
    """
    if not result or 'spline_coefficients' not in result:
        return None
    
    tck = result['spline_coefficients']
    knots, coefficients, degree = tck
    
    # Filter X coefficients
    x_coeffs_filtered = []
    x_kept_indices = []
    for i, coeff in enumerate(coefficients[0]):
        if abs(coeff) >= threshold:
            x_coeffs_filtered.append(coeff)
            x_kept_indices.append(i)
    
    # Filter Y coefficients
    y_coeffs_filtered = []
    y_kept_indices = []
    for i, coeff in enumerate(coefficients[1]):
        if abs(coeff) >= threshold:
            y_coeffs_filtered.append(coeff)
            y_kept_indices.append(i)
    
    # Create new tck with filtered coefficients
    # Note: This is simplified - in practice, you'd need to adjust knots too
    filtered_tck = (knots, [np.array(x_coeffs_filtered), np.array(y_coeffs_filtered)], degree)
    
    print(f"\n🎯 COEFFICIENT FILTERING (threshold: {threshold}):")
    print(f"  Original X coefficients: {len(coefficients[0])}")
    print(f"  Filtered X coefficients: {len(x_coeffs_filtered)} ({len(x_coeffs_filtered)/len(coefficients[0]):.1%} kept)")
    print(f"  Original Y coefficients: {len(coefficients[1])}")
    print(f"  Filtered Y coefficients: {len(y_coeffs_filtered)} ({len(y_coeffs_filtered)/len(coefficients[1]):.1%} kept)")
    
    # Create filtered result
    filtered_result = result.copy()
    filtered_result['spline_coefficients'] = filtered_tck
    filtered_result['filtering_info'] = {
        'threshold': threshold,
        'original_x_coeffs': len(coefficients[0]),
        'filtered_x_coeffs': len(x_coeffs_filtered),
        'original_y_coeffs': len(coefficients[1]),
        'filtered_y_coeffs': len(y_coeffs_filtered),
        'reduction_ratio': (len(x_coeffs_filtered) + len(y_coeffs_filtered)) / (len(coefficients[0]) + len(coefficients[1]))
    }
    
    return filtered_result

def visualize_compression_result(coordinates, result, title="Spline Compression"):
    """
    Simple visualization of the compression result
    """
    import matplotlib.pyplot as plt
    
    coords = np.array(coordinates)
    reconstructed = result['reconstructed']
    key_points = result['key_points']
    metrics = result['metrics']
    
    plt.figure(figsize=(12, 4))
    
    # Original
    plt.subplot(1, 3, 1)
    plt.plot(coords[:, 0], coords[:, 1], 'bo-', markersize=2, linewidth=1)
    plt.title(f'Original\n{metrics["original_points"]} points')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    # Reconstructed
    plt.subplot(1, 3, 2)
    plt.plot(reconstructed[:, 0], reconstructed[:, 1], 'r-', linewidth=2)
    plt.title(f'Reconstructed\n{metrics["key_points_used"]} key points')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    # Key points
    plt.subplot(1, 3, 3)
    plt.plot(coords[:, 0], coords[:, 1], 'b-', alpha=0.3, linewidth=1)
    plt.plot(key_points[:, 0], key_points[:, 1], 'ro', markersize=4, 
             markerfacecolor='none', markeredgewidth=2)
    plt.title(f'Key Points\nError: {metrics["mean_error"]:.4f}')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()












    











import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

def divide_shape_into_sublists(coordinates, num_sublists=3):
    """
    Divide the shape coordinates into sequential sublists
    """
    coords = np.array(coordinates)
    
    # Ensure closed shape
    if not np.allclose(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0]])
    
    total_points = len(coords)
    points_per_sublist = total_points // num_sublists
    
    sublists = []
    for i in range(num_sublists):
        start_idx = i * points_per_sublist
        if i == num_sublists - 1:  # Last sublist gets remaining points
            end_idx = total_points
        else:
            end_idx = (i + 1) * points_per_sublist
        
        # Ensure each sublist is closed by adding first point at the end
        sublist = coords[start_idx:end_idx]
        if not np.allclose(sublist[0], sublist[-1]):
            sublist = np.vstack([sublist, sublist[0]])
        
        sublists.append(sublist.tolist())
    
    print(f"📊 Divided {total_points} points into {num_sublists} sublists:")
    for i, sublist in enumerate(sublists):
        print(f"  Sublist {i+1}: {len(sublist)} points")
    
    return sublists

def compress_sublist_spline(sublist_coordinates, compression_ratio=0.2, num_points=100):
    """
    Compress a single sublist using spline interpolation
    """
    coords = np.array(sublist_coordinates)
    
    # Select key points
    n_key_points = max(4, int(len(coords) * compression_ratio))
    
    # Simple point selection (evenly spaced)
    key_indices = np.linspace(0, len(coords)-1, n_key_points, dtype=int)
    key_points = coords[key_indices]
    
    # Fit parametric spline
    x_key = key_points[:, 0]
    y_key = key_points[:, 1]
    
    try:
        tck, u = splprep([x_key, y_key], s=0, per=0)  # per=0 for open curves
        
        # Reconstruct
        t_new = np.linspace(0, 1, num_points)
        x_new, y_new = splev(t_new, tck)
        
        # Calculate error on original sublist points
        t_original = np.linspace(0, 1, len(coords))
        x_orig_interp, y_orig_interp = splev(t_original, tck)
        error = np.mean(np.sqrt((x_orig_interp - coords[:, 0])**2 + 
                               (y_orig_interp - coords[:, 1])**2))
        
        return {
            'reconstructed': np.column_stack([x_new, y_new]),
            'spline_coefficients': tck,
            'key_points': key_points,
            'key_indices': key_indices,
            'metrics': {
                'mean_error': error,
                'compression_ratio': compression_ratio,
                'original_points': len(coords),
                'key_points_used': n_key_points,
                'storage_reduction': f"{(1 - n_key_points/len(coords))*100:.1f}%"
            }
        }
    except Exception as e:
        print(f"❌ Sublist compression failed: {e}")
        return None

def compress_shape_divided(coordinates, num_sublists=3, compression_ratio=0.2, points_per_sublist=50):
    """
    Main function: Divide shape and compress each part
    """
    # Divide into sublists
    sublists = divide_shape_into_sublists(coordinates, num_sublists)
    
    results = []
    all_reconstructed = []
    
    print(f"\n🎯 COMPRESSING {num_sublists} SUBLISTS:")
    print("=" * 50)
    
    for i, sublist in enumerate(sublists):
        print(f"\n📦 Processing sublist {i+1}/{num_sublists}...")
        result = compress_sublist_spline(sublist, compression_ratio, points_per_sublist)
        
        if result:
            results.append(result)
            all_reconstructed.append(result['reconstructed'])
            
            metrics = result['metrics']
            print(f"   ✅ Key points: {metrics['key_points_used']}")
            print(f"   ✅ Error: {metrics['mean_error']:.6f}")
            print(f"   ✅ Storage: {metrics['storage_reduction']}")
        else:
            print(f"   ❌ Failed to compress sublist {i+1}")
    
    # Combine all reconstructed parts
    if all_reconstructed:
        combined_reconstructed = np.vstack(all_reconstructed)
        
        # Calculate overall error
        original_coords = np.array(coordinates)
        if not np.allclose(original_coords[0], original_coords[-1]):
            original_coords = np.vstack([original_coords, original_coords[0]])
        
        # Sample combined reconstruction at original points for error calculation
        from scipy.interpolate import interp1d
        
        # Create parameter along combined curve
        t_combined = np.linspace(0, 1, len(combined_reconstructed))
        t_original = np.linspace(0, 1, len(original_coords))
        
        # Interpolate to get values at original points
        f_x = interp1d(t_combined, combined_reconstructed[:, 0], kind='linear', 
                       bounds_error=False, fill_value='extrapolate')
        f_y = interp1d(t_combined, combined_reconstructed[:, 1], kind='linear', 
                       bounds_error=False, fill_value='extrapolate')
        
        x_interp = f_x(t_original)
        y_interp = f_y(t_original)
        
        overall_error = np.mean(np.sqrt((x_interp - original_coords[:, 0])**2 + 
                                       (y_interp - original_coords[:, 1])**2))
        
        # Calculate overall storage
        total_original_points = len(original_coords)
        total_key_points = sum(len(res['key_points']) for res in results)
        overall_storage_reduction = (1 - total_key_points/total_original_points) * 100
        
        return {
            'sublist_results': results,
            'combined_reconstructed': combined_reconstructed,
            'overall_metrics': {
                'mean_error': overall_error,
                'total_original_points': total_original_points,
                'total_key_points': total_key_points,
                'storage_reduction': f"{overall_storage_reduction:.1f}%",
                'num_sublists': num_sublists,
                'compression_ratio': compression_ratio
            }
        }
    
    return None

def print_divided_compression_analysis(result):
    """
    Print detailed analysis of divided compression
    """
    if not result:
        print("❌ No results to analyze")
        return
    
    overall = result['overall_metrics']
    sublist_results = result['sublist_results']
    
    print("\n" + "="*70)
    print("📊 DIVIDED COMPRESSION ANALYSIS")
    print("="*70)
    
    print(f"🎯 Overall Results:")
    print(f"   Number of sublists: {overall['num_sublists']}")
    print(f"   Compression ratio: {overall['compression_ratio']:.1%}")
    print(f"   Total original points: {overall['total_original_points']}")
    print(f"   Total key points: {overall['total_key_points']}")
    print(f"   Overall storage reduction: {overall['storage_reduction']}")
    print(f"   Overall reconstruction error: {overall['mean_error']:.6f}")
    
    print(f"\n📈 Sublist Details:")
    for i, sub_res in enumerate(sublist_results):
        metrics = sub_res['metrics']
        print(f"   Sublist {i+1}:")
        print(f"     - Original points: {metrics['original_points']}")
        print(f"     - Key points used: {metrics['key_points_used']}")
        print(f"     - Sublist error: {metrics['mean_error']:.6f}")
        print(f"     - Storage: {metrics['storage_reduction']}")

def visualize_divided_compression(coordinates, result):
    """
    Visualize the divided compression results
    """
    original_coords = np.array(coordinates)
    if not np.allclose(original_coords[0], original_coords[-1]):
        original_coords = np.vstack([original_coords, original_coords[0]])
    
    combined_reconstructed = result['combined_reconstructed']
    sublist_results = result['sublist_results']
    overall = result['overall_metrics']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original shape
    ax1.plot(original_coords[:, 0], original_coords[:, 1], 'b-', linewidth=2, label='Original')
    ax1.set_title(f'Original Shape\n{overall["total_original_points"]} points')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Combined reconstruction
    ax2.plot(combined_reconstructed[:, 0], combined_reconstructed[:, 1], 'r-', linewidth=2, label='Reconstructed')
    ax2.set_title(f'Combined Reconstruction\n{overall["total_key_points"]} key points\nError: {overall["mean_error"]:.4f}')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Sublist key points
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    ax3.plot(original_coords[:, 0], original_coords[:, 1], 'k-', alpha=0.3, linewidth=1, label='Original')
    for i, sub_res in enumerate(sublist_results):
        color = colors[i % len(colors)]
        key_points = sub_res['key_points']
        ax3.plot(key_points[:, 0], key_points[:, 1], 'o', color=color, markersize=6,
                markerfacecolor='none', markeredgewidth=2, label=f'Sublist {i+1}')
    ax3.set_title(f'Key Points by Sublist\n{overall["num_sublists"]} sublists')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Overlay comparison
    ax4.plot(original_coords[:, 0], original_coords[:, 1], 'b-', linewidth=2, label='Original', alpha=0.7)
    ax4.plot(combined_reconstructed[:, 0], combined_reconstructed[:, 1], 'r--', linewidth=2, label='Reconstructed')
    ax4.set_title(f'Overlay Comparison\nStorage: {overall["storage_reduction"]}')
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.show()

# MAIN EXECUTION
if __name__ == "__main__":
    # Your sample coordinates (replace with your actual data)
    sample_coordinates =[  (750.5,697.0),  (750.5,696.0),  (750.5,695.0),  (750.5,694.0),  (750.0,693.5), 
  (749.5,693.0),  (749.5,692.0),  (749.5,691.0),  (749.0,690.5),  (748.5,690.0), 
  (748.5,689.0),  (748.5,688.0),  (748.5,687.0),  (748.5,686.0),  (748.5,685.0), 
  (748.5,684.0),  (748.5,683.0),  (749.0,682.5),  (749.5,682.0),  (749.5,681.0), 
  (749.5,680.0),  (749.0,679.5),  (748.5,679.0),  (748.5,678.0),  (748.5,677.0), 
  (748.5,676.0),  (748.0,675.5),  (747.0,675.5),  (746.5,676.0),  (746.5,677.0), 
  (746.5,678.0),  (746.0,678.5),  (745.0,678.5),  (744.0,678.5),  (743.0,678.5), 
  (742.0,678.5),  (741.0,678.5),  (740.0,678.5),  (739.5,679.0),  (739.0,679.5), 
  (738.0,679.5),  (737.0,679.5),  (736.0,679.5),  (735.0,679.5),  (734.5,680.0), 
  (734.0,680.5),  (733.0,680.5),  (732.0,680.5),  (731.5,680.0),  (731.0,679.5), 
  (730.0,679.5),  (729.5,679.0),  (729.0,678.5),  (728.0,678.5),  (727.0,678.5), 
  (726.0,678.5),  (725.5,678.0),  (725.0,677.5),  (724.5,677.0),  (724.0,676.5), 
  (723.5,676.0),  (723.0,675.5),  (722.5,675.0),  (722.5,674.0),  (722.0,673.5), 
  (721.5,673.0),  (721.0,672.5),  (720.0,672.5),  (719.0,672.5),  (718.5,672.0), 
  (718.5,671.0),  (718.0,670.5),  (717.0,670.5),  (716.0,670.5),  (715.5,670.0), 
  (715.0,669.5),  (714.0,669.5),  (713.0,669.5),  (712.0,669.5),  (711.0,669.5), 
  (710.0,669.5),  (709.0,669.5),  (708.0,669.5),  (707.5,670.0),  (707.0,670.5), 
  (706.0,670.5),  (705.5,670.0),  (705.0,669.5),  (704.0,669.5),  (703.0,669.5), 
  (702.5,670.0),  (702.0,670.5),  (701.5,671.0),  (701.5,672.0),  (701.5,673.0), 
  (701.0,673.5),  (700.5,674.0),  (700.0,674.5),  (699.0,674.5),  (698.5,675.0), 
  (698.0,675.5),  (697.5,676.0),  (697.0,676.5),  (696.5,677.0),  (696.5,678.0), 
  (696.0,678.5),  (695.0,678.5),  (694.0,678.5),  (693.0,678.5),  (692.5,679.0), 
  (692.5,680.0),  (693.0,680.5),  (693.5,681.0),  (694.0,681.5),  (694.5,682.0), 
  (694.5,683.0),  (694.5,684.0),  (694.5,685.0),  (695.0,685.5),  (695.5,686.0), 
  (695.0,686.5),  (694.5,687.0),  (694.0,687.5),  (693.0,687.5),  (692.0,687.5), 
  (691.0,687.5),  (690.5,688.0),  (690.0,688.5),  (689.0,688.5),  (688.0,688.5), 
  (687.5,689.0),  (687.5,690.0),  (687.5,691.0),  (687.5,692.0),  (688.0,692.5), 
  (689.0,692.5),  (689.5,693.0),  (689.0,693.5),  (688.5,694.0),  (688.0,694.5), 
  (687.5,695.0),  (687.5,696.0),  (687.0,696.5),  (686.5,697.0),  (686.0,697.5), 
  (685.5,698.0),  (685.0,698.5),  (684.5,699.0),  (684.0,699.5),  (683.5,700.0), 
  (683.0,700.5),  (682.5,701.0),  (682.0,701.5),  (681.0,701.5),  (680.0,701.5), 
  (679.5,701.0),  (679.0,700.5),  (678.5,701.0),  (678.5,702.0),  (678.0,702.5), 
  (677.0,702.5),  (676.0,702.5),  (675.5,703.0),  (675.0,703.5),  (674.5,704.0), 
  (674.5,705.0),  (674.5,706.0),  (674.5,707.0),  (674.5,708.0),  (674.0,708.5), 
  (673.5,709.0),  (673.5,710.0),  (673.0,710.5),  (672.5,711.0),  (672.0,711.5), 
  (671.5,712.0),  (671.0,712.5),  (670.5,713.0),  (670.5,714.0),  (670.0,714.5), 
  (669.5,715.0),  (669.0,715.5),  (668.5,716.0),  (668.0,716.5),  (667.5,717.0), 
  (667.5,718.0),  (668.0,718.5),  (668.5,719.0),  (669.0,719.5),  (669.5,720.0), 
  (670.0,720.5),  (671.0,720.5),  (671.5,721.0),  (671.5,722.0),  (671.0,722.5), 
  (670.5,723.0),  (670.0,723.5),  (669.0,723.5),  (668.0,723.5),  (667.0,723.5), 
  (666.0,723.5),  (665.0,723.5),  (664.5,723.0),  (664.0,722.5),  (663.0,722.5), 
  (662.5,723.0),  (662.0,723.5),  (661.0,723.5),  (660.5,723.0),  (660.0,722.5), 
  (659.0,722.5),  (658.5,722.0),  (658.0,721.5),  (657.0,721.5),  (656.0,721.5), 
  (655.5,722.0),  (655.0,722.5),  (654.0,722.5),  (653.0,722.5),  (652.0,722.5), 
  (651.5,723.0),  (651.0,723.5),  (650.5,723.0),  (650.0,722.5),  (649.0,722.5), 
  (648.5,723.0),  (648.0,723.5),  (647.0,723.5),  (646.5,724.0),  (646.0,724.5), 
  (645.0,724.5),  (644.5,725.0),  (644.5,726.0),  (645.0,726.5),  (645.5,727.0), 
  (646.0,727.5),  (646.5,728.0),  (646.5,729.0),  (646.5,730.0),  (646.0,730.5), 
  (645.0,730.5),  (644.5,730.0),  (644.0,729.5),  (643.5,729.0),  (643.0,728.5), 
  (642.5,728.0),  (642.5,727.0),  (642.0,726.5),  (641.5,726.0),  (641.5,725.0), 
  (641.5,724.0),  (641.0,723.5),  (640.0,723.5),  (639.0,723.5),  (638.5,724.0), 
  (638.0,724.5),  (637.5,725.0),  (637.0,725.5),  (636.5,726.0),  (636.0,726.5), 
  (635.5,727.0),  (635.0,727.5),  (634.0,727.5),  (633.0,727.5),  (632.5,728.0), 
  (632.0,728.5),  (631.5,729.0),  (631.5,730.0),  (631.0,730.5),  (630.0,730.5), 
  (629.0,730.5),  (628.0,730.5),  (627.5,731.0),  (627.5,732.0),  (627.5,733.0), 
  (627.5,734.0),  (628.0,734.5),  (629.0,734.5),  (630.0,734.5),  (631.0,734.5), 
  (632.0,734.5),  (632.5,735.0),  (632.5,736.0),  (632.5,737.0),  (632.5,738.0), 
  (633.0,738.5),  (633.5,739.0),  (633.5,740.0),  (634.0,740.5),  (634.5,741.0), 
  (634.5,742.0),  (635.0,742.5),  (635.5,743.0),  (635.5,744.0),  (636.0,744.5), 
  (637.0,744.5),  (638.0,744.5),  (638.5,745.0),  (639.0,745.5),  (639.5,746.0), 
  (640.0,746.5),  (640.5,747.0),  (641.0,747.5),  (641.5,748.0),  (642.0,748.5), 
  (642.5,749.0),  (643.0,749.5),  (643.5,750.0),  (643.5,751.0),  (643.5,752.0), 
  (644.0,752.5),  (645.0,752.5),  (646.0,752.5),  (647.0,752.5),  (647.5,753.0), 
  (647.5,754.0),  (647.5,755.0),  (647.5,756.0),  (647.0,756.5),  (646.5,756.0), 
  (646.0,755.5),  (645.5,755.0),  (645.0,754.5),  (644.0,754.5),  (643.0,754.5), 
  (642.0,754.5),  (641.0,754.5),  (640.5,754.0),  (640.5,753.0),  (640.0,752.5), 
  (639.0,752.5),  (638.0,752.5),  (637.0,752.5),  (636.5,752.0),  (636.0,751.5), 
  (635.0,751.5),  (634.0,751.5),  (633.0,751.5),  (632.0,751.5),  (631.0,751.5), 
  (630.0,751.5),  (629.0,751.5),  (628.5,751.0),  (628.0,750.5),  (627.5,750.0), 
  (627.5,749.0),  (628.0,748.5),  (628.5,748.0),  (628.0,747.5),  (627.5,747.0), 
  (627.5,746.0),  (627.0,745.5),  (626.5,745.0),  (626.0,744.5),  (625.0,744.5), 
  (624.5,745.0),  (624.5,746.0),  (624.5,747.0),  (624.0,747.5),  (623.5,748.0), 
  (623.0,748.5),  (622.5,749.0),  (622.5,750.0),  (622.0,750.5),  (621.5,751.0), 
  (621.0,751.5),  (620.0,751.5),  (619.5,751.0),  (619.0,750.5),  (618.0,750.5), 
  (617.0,750.5),  (616.0,750.5),  (615.0,750.5),  (614.0,750.5),  (613.0,750.5), 
  (612.0,750.5),  (611.0,750.5),  (610.5,751.0),  (610.0,751.5),  (609.0,751.5), 
  (608.0,751.5),  (607.0,751.5),  (606.0,751.5),  (605.0,751.5),  (604.0,751.5), 
  (603.0,751.5),  (602.0,751.5),  (601.0,751.5),  (600.0,751.5),  (599.5,751.0), 
  (600.0,750.5),  (600.5,750.0),  (601.0,749.5),  (602.0,749.5),  (602.5,749.0), 
  (602.5,748.0),  (602.5,747.0),  (602.0,746.5),  (601.0,746.5),  (600.5,746.0), 
  (600.0,745.5),  (599.5,746.0),  (599.0,746.5),  (598.0,746.5),  (597.0,746.5), 
  (596.0,746.5),  (595.0,746.5),  (594.0,746.5),  (593.5,747.0),  (593.0,747.5), 
  (592.0,747.5),  (591.0,747.5),  (590.0,747.5),  (589.5,748.0),  (589.0,748.5), 
  (588.0,748.5),  (587.5,748.0),  (587.0,747.5),  (586.0,747.5),  (585.0,747.5), 
  (584.0,747.5),  (583.5,747.0),  (583.0,746.5),  (582.5,746.0),  (582.5,745.0), 
  (582.5,744.0),  (582.5,743.0),  (582.0,742.5),  (581.5,742.0),  (581.5,741.0), 
  (581.0,740.5),  (580.5,740.0),  (580.0,739.5),  (579.0,739.5),  (578.5,739.0), 
  (578.0,738.5),  (577.5,739.0),  (577.0,739.5),  (576.5,740.0),  (576.0,740.5), 
  (575.5,740.0),  (575.5,739.0),  (575.0,738.5),  (574.5,738.0),  (574.0,737.5), 
  (573.5,738.0),  (573.0,738.5),  (572.5,739.0),  (572.5,740.0),  (572.0,740.5), 
  (571.5,741.0),  (571.0,741.5),  (570.5,742.0),  (570.0,742.5),  (569.5,743.0), 
  (569.0,743.5),  (568.5,744.0),  (568.0,744.5),  (567.5,745.0),  (567.5,746.0), 
  (567.5,747.0),  (567.0,747.5),  (566.5,748.0),  (566.0,748.5),  (565.0,748.5), 
  (564.5,749.0),  (564.0,749.5),  (563.5,750.0),  (563.5,751.0),  (564.0,751.5), 
  (564.5,752.0),  (564.0,752.5),  (563.5,753.0),  (563.0,753.5),  (562.0,753.5), 
  (561.0,753.5),  (560.0,753.5),  (559.0,753.5),  (558.0,753.5),  (557.0,753.5), 
  (556.5,753.0),  (556.0,752.5),  (555.5,752.0),  (555.5,751.0),  (556.0,750.5), 
  (556.5,751.0),  (557.0,751.5),  (557.5,751.0),  (558.0,750.5),  (558.5,750.0), 
  (558.5,749.0),  (558.5,748.0),  (559.0,747.5),  (559.5,747.0),  (559.5,746.0), 
  (560.0,745.5),  (560.5,745.0),  (560.5,744.0),  (561.0,743.5),  (561.5,743.0), 
  (562.0,742.5),  (562.5,742.0),  (562.5,741.0),  (563.0,740.5),  (563.5,740.0), 
  (563.5,739.0),  (563.5,738.0),  (563.5,737.0),  (564.0,736.5),  (564.5,736.0), 
  (564.5,735.0),  (564.5,734.0),  (565.0,733.5),  (565.5,733.0),  (565.5,732.0), 
  (565.5,731.0),  (565.5,730.0),  (565.5,729.0),  (565.5,728.0),  (565.5,727.0), 
  (565.5,726.0),  (565.5,725.0),  (565.5,724.0),  (565.5,723.0),  (565.5,722.0), 
  (566.0,721.5),  (566.5,721.0),  (566.5,720.0),  (567.0,719.5),  (567.5,720.0), 
  (567.5,721.0),  (567.5,722.0),  (567.5,723.0),  (568.0,723.5),  (569.0,723.5), 
  (569.5,723.0),  (570.0,722.5),  (571.0,722.5),  (571.5,722.0),  (572.0,721.5), 
  (572.5,721.0),  (572.5,720.0),  (573.0,719.5),  (573.5,719.0),  (573.5,718.0), 
  (573.5,717.0),  (573.5,716.0),  (574.0,715.5),  (574.5,715.0),  (575.0,714.5), 
  (575.5,714.0),  (576.0,713.5),  (576.5,713.0),  (576.5,712.0),  (576.0,711.5), 
  (575.5,712.0),  (575.0,712.5),  (574.5,713.0),  (574.0,713.5),  (573.5,714.0), 
  (573.0,714.5),  (572.5,715.0),  (572.5,716.0),  (572.5,717.0),  (572.0,717.5), 
  (571.5,718.0),  (571.5,719.0),  (571.0,719.5),  (570.0,719.5),  (569.5,719.0), 
  (569.0,718.5),  (568.5,718.0),  (568.0,717.5),  (567.5,717.0),  (567.0,716.5), 
  (566.5,716.0),  (566.0,715.5),  (565.0,715.5),  (564.5,715.0),  (564.5,714.0), 
  (564.0,713.5),  (563.5,713.0),  (563.5,712.0),  (563.0,711.5),  (562.5,711.0), 
  (562.5,710.0),  (562.0,709.5),  (561.5,709.0),  (561.5,708.0),  (561.5,707.0), 
  (561.0,706.5),  (560.5,706.0),  (560.5,705.0),  (560.5,704.0),  (560.0,703.5), 
  (559.5,703.0),  (559.5,702.0),  (559.5,701.0),  (559.5,700.0),  (559.5,699.0), 
  (559.5,698.0),  (559.0,697.5),  (558.5,697.0),  (558.0,696.5),  (557.5,696.0), 
  (557.0,695.5),  (556.5,695.0),  (556.5,694.0),  (556.5,693.0),  (556.5,692.0), 
  (556.5,691.0),  (556.0,690.5),  (555.5,690.0),  (555.5,689.0),  (556.0,688.5), 
  (556.5,688.0),  (556.5,687.0),  (556.0,686.5),  (555.5,686.0),  (555.5,685.0), 
  (555.0,684.5),  (554.5,684.0),  (554.0,683.5),  (553.5,683.0),  (553.0,682.5), 
  (552.5,682.0),  (552.5,681.0),  (553.0,680.5),  (554.0,680.5),  (555.0,680.5), 
  (556.0,680.5),  (556.5,680.0),  (556.5,679.0),  (556.5,678.0),  (556.5,677.0), 
  (556.5,676.0),  (556.5,675.0),  (556.5,674.0),  (557.0,673.5),  (557.5,673.0), 
  (557.0,672.5),  (556.5,672.0),  (556.5,671.0),  (556.5,670.0),  (556.0,669.5), 
  (555.0,669.5),  (554.5,670.0),  (554.0,670.5),  (553.5,671.0),  (553.5,672.0), 
  (553.0,672.5),  (552.5,673.0),  (552.5,674.0),  (552.0,674.5),  (551.5,675.0), 
  (551.5,676.0),  (551.5,677.0),  (551.5,678.0),  (551.5,679.0),  (551.5,680.0), 
  (551.0,680.5),  (550.5,681.0),  (550.0,681.5),  (549.5,682.0),  (549.5,683.0), 
  (549.0,683.5),  (548.5,684.0),  (548.5,685.0),  (548.0,685.5),  (547.5,686.0), 
  (547.0,686.5),  (546.5,686.0),  (546.5,685.0),  (546.5,684.0),  (546.0,683.5), 
  (545.5,683.0),  (545.5,682.0),  (545.0,681.5),  (544.5,681.0),  (544.5,680.0), 
  (544.0,679.5),  (543.5,679.0),  (543.0,678.5),  (542.5,678.0),  (542.5,677.0), 
  (542.0,676.5),  (541.5,676.0),  (541.0,675.5),  (540.5,675.0),  (540.0,674.5), 
  (539.0,674.5),  (538.5,674.0),  (538.0,673.5),  (537.5,673.0),  (537.5,672.0), 
  (537.0,671.5),  (536.5,671.0),  (536.5,670.0),  (536.5,669.0),  (536.5,668.0), 
  (537.0,667.5),  (537.5,667.0),  (537.5,666.0),  (537.5,665.0),  (537.5,664.0), 
  (537.5,663.0),  (537.5,662.0),  (537.5,661.0),  (537.5,660.0),  (538.0,659.5), 
  (538.5,659.0),  (539.0,658.5),  (540.0,658.5),  (540.5,658.0),  (540.5,657.0), 
  (540.5,656.0),  (540.5,655.0),  (540.5,654.0),  (540.5,653.0),  (540.5,652.0), 
  (540.0,651.5),  (539.5,651.0),  (539.0,650.5),  (538.5,650.0),  (538.0,649.5), 
  (537.5,649.0),  (537.5,648.0),  (537.5,647.0),  (537.5,646.0),  (537.5,645.0), 
  (537.5,644.0),  (537.5,643.0),  (537.5,642.0),  (537.5,641.0),  (537.5,640.0), 
  (537.0,639.5),  (536.5,639.0),  (536.0,638.5),  (535.0,638.5),  (534.5,638.0), 
  (534.0,637.5),  (533.0,637.5),  (532.5,638.0),  (532.5,639.0),  (532.0,639.5), 
  (531.5,640.0),  (531.0,640.5),  (530.0,640.5),  (529.5,641.0),  (529.5,642.0), 
  (529.0,642.5),  (528.5,643.0),  (528.0,643.5),  (527.5,644.0),  (527.5,645.0), 
  (527.0,645.5),  (526.5,646.0),  (526.5,647.0),  (526.0,647.5),  (525.5,648.0), 
  (525.5,649.0),  (525.5,650.0),  (525.0,650.5),  (524.5,651.0),  (524.0,651.5), 
  (523.0,651.5),  (522.0,651.5),  (521.5,652.0),  (521.0,652.5),  (520.5,653.0), 
  (520.0,653.5),  (519.5,654.0),  (519.0,654.5),  (518.5,655.0),  (518.5,656.0), 
  (518.5,657.0),  (518.0,657.5),  (517.5,658.0),  (517.5,659.0),  (517.5,660.0), 
  (517.5,661.0),  (517.0,661.5),  (516.5,662.0),  (516.0,662.5),  (515.5,663.0), 
  (515.0,663.5),  (514.5,664.0),  (514.0,664.5),  (513.5,665.0),  (513.5,666.0), 
  (513.0,666.5),  (512.5,667.0),  (512.5,668.0),  (512.5,669.0),  (512.0,669.5), 
  (511.5,670.0),  (511.0,670.5),  (510.5,671.0),  (510.0,671.5),  (509.5,672.0), 
  (509.0,672.5),  (508.5,673.0),  (508.5,674.0),  (508.0,674.5),  (507.5,675.0), 
  (507.5,676.0),  (507.5,677.0),  (507.0,677.5),  (506.5,678.0),  (506.0,678.5), 
  (505.5,679.0),  (505.5,680.0),  (505.0,680.5),  (504.5,681.0),  (504.5,682.0), 
  (504.0,682.5),  (503.5,683.0),  (503.5,684.0),  (503.0,684.5),  (502.5,685.0), 
  (502.5,686.0),  (502.5,687.0),  (502.5,688.0),  (502.5,689.0),  (503.0,689.5), 
  (503.5,690.0),  (503.5,691.0),  (504.0,691.5),  (504.5,692.0),  (505.0,692.5), 
  (505.5,693.0),  (505.5,694.0),  (506.0,694.5),  (506.5,695.0),  (507.0,695.5), 
  (508.0,695.5),  (508.5,696.0),  (509.0,696.5),  (509.5,697.0),  (510.0,697.5), 
  (511.0,697.5),  (512.0,697.5),  (512.5,698.0),  (513.0,698.5),  (514.0,698.5), 
  (514.5,699.0),  (515.0,699.5),  (515.5,700.0),  (516.0,700.5),  (517.0,700.5), 
  (517.5,700.0),  (518.0,699.5),  (518.5,699.0),  (519.0,698.5),  (519.5,698.0), 
  (519.5,697.0),  (520.0,696.5),  (520.5,696.0),  (521.0,695.5),  (522.0,695.5), 
  (522.5,696.0),  (523.0,696.5),  (523.5,697.0),  (524.0,697.5),  (524.5,698.0), 
  (525.0,698.5),  (525.5,699.0),  (526.0,699.5),  (526.5,700.0),  (526.5,701.0), 
  (527.0,701.5),  (527.5,702.0),  (527.5,703.0),  (528.0,703.5),  (528.5,704.0), 
  (529.0,704.5),  (529.5,705.0),  (530.0,705.5),  (531.0,705.5),  (531.5,706.0), 
  (532.0,706.5),  (532.5,707.0),  (533.0,707.5),  (533.5,708.0),  (534.0,708.5), 
  (534.5,709.0),  (534.5,710.0),  (535.0,710.5),  (535.5,711.0),  (536.0,711.5), 
  (536.5,712.0),  (536.5,713.0),  (536.5,714.0),  (536.5,715.0),  (537.0,715.5), 
  (537.5,716.0),  (537.5,717.0),  (537.5,718.0),  (537.0,718.5),  (536.5,719.0), 
  (536.5,720.0),  (536.5,721.0),  (536.5,722.0),  (536.0,722.5),  (535.5,723.0), 
  (535.5,724.0),  (535.5,725.0),  (535.5,726.0),  (535.5,727.0),  (535.5,728.0), 
  (536.0,728.5),  (536.5,729.0),  (537.0,729.5),  (537.5,730.0),  (538.0,730.5), 
  (538.5,731.0),  (539.0,731.5),  (539.5,732.0),  (540.0,732.5),  (540.5,733.0), 
  (540.5,734.0),  (541.0,734.5),  (541.5,735.0),  (542.0,735.5),  (542.5,736.0), 
  (542.5,737.0),  (542.5,738.0),  (542.0,738.5),  (541.0,738.5),  (540.0,738.5), 
  (539.0,738.5),  (538.0,738.5),  (537.0,738.5),  (536.0,738.5),  (535.0,738.5), 
  (534.0,738.5),  (533.0,738.5),  (532.0,738.5),  (531.0,738.5),  (530.5,738.0), 
  (530.0,737.5),  (529.0,737.5),  (528.5,738.0),  (528.0,738.5),  (527.0,738.5), 
  (526.0,738.5),  (525.5,738.0),  (525.0,737.5),  (524.0,737.5),  (523.0,737.5), 
  (522.0,737.5),  (521.0,737.5),  (520.0,737.5),  (519.0,737.5),  (518.5,737.0), 
  (518.0,736.5),  (517.5,736.0),  (517.5,735.0),  (517.5,734.0),  (517.5,733.0), 
  (517.5,732.0),  (518.0,731.5),  (518.5,731.0),  (518.5,730.0),  (518.5,729.0), 
  (518.0,728.5),  (517.5,728.0),  (517.0,727.5),  (516.5,727.0),  (517.0,726.5), 
  (517.5,726.0),  (518.0,725.5),  (519.0,725.5),  (519.5,725.0),  (519.5,724.0), 
  (519.0,723.5),  (518.5,723.0),  (518.5,722.0),  (518.0,721.5),  (517.5,721.0), 
  (517.5,720.0),  (517.5,719.0),  (517.5,718.0),  (517.0,717.5),  (516.5,717.0), 
  (516.0,716.5),  (515.5,716.0),  (515.5,715.0),  (515.0,714.5),  (514.5,714.0), 
  (514.5,713.0),  (514.5,712.0),  (514.5,711.0),  (514.0,710.5),  (513.5,710.0), 
  (513.0,709.5),  (512.5,709.0),  (512.5,708.0),  (512.0,707.5),  (511.5,707.0), 
  (511.5,706.0),  (511.5,705.0),  (511.5,704.0),  (511.5,703.0),  (511.5,702.0), 
  (511.0,701.5),  (510.5,701.0),  (510.5,700.0),  (510.5,699.0),  (510.0,698.5), 
  (509.0,698.5),  (508.5,699.0),  (508.0,699.5),  (507.5,700.0),  (507.0,700.5), 
  (506.0,700.5),  (505.5,701.0),  (505.0,701.5),  (504.0,701.5),  (503.5,702.0), 
  (503.0,702.5),  (502.5,703.0),  (502.0,703.5),  (501.5,704.0),  (501.0,704.5), 
  (500.5,705.0),  (500.5,706.0),  (501.0,706.5),  (501.5,707.0),  (501.5,708.0), 
  (502.0,708.5),  (502.5,709.0),  (502.5,710.0),  (502.5,711.0),  (503.0,711.5), 
  (503.5,712.0),  (503.5,713.0),  (503.5,714.0),  (503.5,715.0),  (503.5,716.0), 
  (503.5,717.0),  (503.5,718.0),  (503.5,719.0),  (503.0,719.5),  (502.5,720.0), 
  (502.5,721.0),  (502.0,721.5),  (501.5,722.0),  (501.5,723.0),  (501.5,724.0), 
  (501.5,725.0),  (501.5,726.0),  (501.5,727.0),  (501.5,728.0),  (501.5,729.0), 
  (501.0,729.5),  (500.0,729.5),  (499.5,729.0),  (499.0,728.5),  (498.0,728.5), 
  (497.5,728.0),  (497.0,727.5),  (496.5,727.0),  (496.0,726.5),  (495.0,726.5), 
  (494.5,727.0),  (494.0,727.5),  (493.0,727.5),  (492.0,727.5),  (491.0,727.5), 
  (490.0,727.5),  (489.5,727.0),  (489.0,726.5),  (488.0,726.5),  (487.5,726.0), 
  (487.0,725.5),  (486.5,725.0),  (486.0,724.5),  (485.5,724.0),  (485.5,723.0), 
  (485.5,722.0),  (485.5,721.0),  (485.5,720.0),  (485.0,719.5),  (484.5,719.0), 
  (484.0,718.5),  (483.0,718.5),  (482.0,718.5),  (481.0,718.5),  (480.0,718.5), 
  (479.5,718.0),  (479.5,717.0),  (480.0,716.5),  (480.5,716.0),  (480.5,715.0), 
  (480.0,714.5),  (479.5,714.0),  (479.0,713.5),  (478.5,714.0),  (478.0,714.5), 
  (477.5,715.0),  (477.0,715.5),  (476.0,715.5),  (475.5,715.0),  (475.0,714.5), 
  (474.0,714.5),  (473.0,714.5),  (472.5,715.0),  (472.0,715.5),  (471.5,716.0), 
  (471.0,716.5),  (470.0,716.5),  (469.0,716.5),  (468.5,717.0),  (468.0,717.5), 
  (467.5,718.0),  (467.0,718.5),  (466.5,719.0),  (466.0,719.5),  (465.5,720.0), 
  (465.5,721.0),  (465.5,722.0),  (465.5,723.0),  (466.0,723.5),  (466.5,724.0), 
  (466.5,725.0),  (467.0,725.5),  (467.5,726.0),  (468.0,726.5),  (469.0,726.5), 
  (469.5,727.0),  (470.0,727.5),  (471.0,727.5),  (472.0,727.5),  (472.5,727.0), 
  (472.5,726.0),  (473.0,725.5),  (474.0,725.5),  (474.5,726.0),  (474.5,727.0), 
  (474.5,728.0),  (475.0,728.5),  (475.5,729.0),  (476.0,729.5),  (476.5,729.0), 
  (477.0,728.5),  (478.0,728.5),  (478.5,729.0),  (479.0,729.5),  (479.5,730.0), 
  (479.5,731.0),  (479.5,732.0),  (479.5,733.0),  (480.0,733.5),  (480.5,734.0), 
  (480.5,735.0),  (480.0,735.5),  (479.5,736.0),  (479.5,737.0),  (479.0,737.5), 
  (478.5,737.0),  (478.0,736.5),  (477.5,736.0),  (477.0,735.5),  (476.5,735.0), 
  (476.0,734.5),  (475.5,734.0),  (475.0,733.5),  (474.5,733.0),  (474.5,732.0), 
  (474.0,731.5),  (473.5,732.0),  (473.0,732.5),  (472.5,733.0),  (472.5,734.0), 
  (472.0,734.5),  (471.5,735.0),  (471.0,735.5),  (470.0,735.5),  (469.0,735.5), 
  (468.0,735.5),  (467.0,735.5),  (466.0,735.5),  (465.0,735.5),  (464.0,735.5), 
  (463.5,735.0),  (463.0,734.5),  (462.5,735.0),  (462.0,735.5),  (461.0,735.5), 
  (460.5,735.0),  (460.0,734.5),  (459.5,734.0),  (459.0,733.5),  (458.5,733.0), 
  (458.0,732.5),  (457.5,732.0),  (457.5,731.0),  (457.5,730.0),  (457.5,729.0), 
  (457.0,728.5),  (456.0,728.5),  (455.0,728.5),  (454.5,729.0),  (454.5,730.0), 
  (454.5,731.0),  (454.5,732.0),  (454.0,732.5),  (453.5,732.0),  (453.5,731.0), 
  (453.0,730.5),  (452.5,731.0),  (452.0,731.5),  (451.0,731.5),  (450.5,731.0), 
  (450.0,730.5),  (449.5,731.0),  (449.0,731.5),  (448.0,731.5),  (447.5,732.0), 
  (447.0,732.5),  (446.5,732.0),  (446.0,731.5),  (445.0,731.5),  (444.0,731.5), 
  (443.0,731.5),  (442.0,731.5),  (441.5,732.0),  (441.0,732.5),  (440.5,733.0), 
  (440.0,733.5),  (439.5,734.0),  (439.5,735.0),  (439.0,735.5),  (438.5,736.0), 
  (438.5,737.0),  (438.5,738.0),  (438.0,738.5),  (437.5,738.0),  (437.5,737.0), 
  (437.0,736.5),  (436.5,737.0),  (436.0,737.5),  (435.5,738.0),  (435.5,739.0), 
  (435.0,739.5),  (434.5,739.0),  (434.5,738.0),  (434.0,737.5),  (433.5,737.0), 
  (433.0,736.5),  (432.0,736.5),  (431.5,737.0),  (431.0,737.5),  (430.5,738.0), 
  (430.0,738.5),  (429.0,738.5),  (428.5,739.0),  (428.0,739.5),  (427.0,739.5), 
  (426.5,740.0),  (426.0,740.5),  (425.5,741.0),  (425.0,741.5),  (424.5,742.0), 
  (424.0,742.5),  (423.5,743.0),  (423.0,743.5),  (422.5,744.0),  (422.0,744.5), 
  (421.5,745.0),  (421.0,745.5),  (420.5,746.0),  (420.0,746.5),  (419.5,747.0), 
  (419.0,747.5),  (418.0,747.5),  (417.0,747.5),  (416.5,748.0),  (416.0,748.5), 
  (415.0,748.5),  (414.5,749.0),  (414.0,749.5),  (413.0,749.5),  (412.0,749.5), 
  (411.5,750.0),  (412.0,750.5),  (412.5,751.0),  (412.5,752.0),  (412.0,752.5), 
  (411.5,753.0),  (411.5,754.0),  (412.0,754.5),  (412.5,755.0),  (412.5,756.0), 
  (413.0,756.5),  (413.5,757.0),  (414.0,757.5),  (415.0,757.5),  (415.5,758.0), 
  (416.0,758.5),  (416.5,759.0),  (417.0,759.5),  (417.5,759.0),  (418.0,758.5), 
  (418.5,758.0),  (418.5,757.0),  (418.5,756.0),  (418.5,755.0),  (418.5,754.0), 
  (418.5,753.0),  (418.5,752.0),  (418.0,751.5),  (417.5,751.0),  (417.0,750.5), 
  (416.5,750.0),  (417.0,749.5),  (417.5,749.0),  (418.0,748.5),  (418.5,749.0), 
  (419.0,749.5),  (420.0,749.5),  (420.5,750.0),  (421.0,750.5),  (422.0,750.5), 
  (423.0,750.5),  (423.5,751.0),  (423.5,752.0),  (424.0,752.5),  (425.0,752.5), 
  (425.5,753.0),  (426.0,753.5),  (426.5,754.0),  (427.0,754.5),  (427.5,755.0), 
  (427.5,756.0),  (427.5,757.0),  (427.5,758.0),  (428.0,758.5),  (428.5,759.0), 
  (428.5,760.0),  (429.0,760.5),  (430.0,760.5),  (430.5,760.0),  (430.5,759.0), 
  (430.5,758.0),  (430.0,757.5),  (429.5,757.0),  (429.5,756.0),  (429.5,755.0), 
  (429.5,754.0),  (429.0,753.5),  (428.5,753.0),  (428.0,752.5),  (427.5,752.0), 
  (427.0,751.5),  (426.5,751.0),  (426.5,750.0),  (426.5,749.0),  (427.0,748.5), 
  (427.5,749.0),  (428.0,749.5),  (429.0,749.5),  (429.5,750.0),  (430.0,750.5), 
  (430.5,750.0),  (430.5,749.0),  (431.0,748.5),  (431.5,749.0),  (431.5,750.0), 
  (431.5,751.0),  (432.0,751.5),  (432.5,752.0),  (432.5,753.0),  (432.5,754.0), 
  (433.0,754.5),  (433.5,755.0),  (433.5,756.0),  (433.5,757.0),  (433.5,758.0), 
  (433.0,758.5),  (432.5,759.0),  (432.5,760.0),  (432.5,761.0),  (432.5,762.0), 
  (432.5,763.0),  (432.5,764.0),  (432.5,765.0),  (432.5,766.0),  (432.0,766.5), 
  (431.0,766.5),  (430.5,766.0),  (430.0,765.5),  (429.5,765.0),  (429.0,764.5), 
  (428.5,765.0),  (428.0,765.5),  (427.5,766.0),  (428.0,766.5),  (428.5,767.0), 
  (428.5,768.0),  (429.0,768.5),  (429.5,769.0),  (429.5,770.0),  (429.5,771.0), 
  (429.5,772.0),  (429.5,773.0),  (429.5,774.0),  (429.5,775.0),  (429.5,776.0), 
  (430.0,776.5),  (430.5,777.0),  (430.5,778.0),  (430.5,779.0),  (430.0,779.5), 
  (429.0,779.5),  (428.5,780.0),  (428.0,780.5),  (427.5,781.0),  (427.0,781.5), 
  (426.5,782.0),  (426.5,783.0),  (426.0,783.5),  (425.0,783.5),  (424.5,784.0), 
  (424.0,784.5),  (423.5,785.0),  (423.5,786.0),  (424.0,786.5),  (424.5,787.0), 
  (425.0,787.5),  (425.5,788.0),  (426.0,788.5),  (426.5,789.0),  (427.0,789.5), 
  (427.5,790.0),  (427.5,791.0),  (427.5,792.0),  (427.0,792.5),  (426.5,793.0), 
  (426.5,794.0),  (426.5,795.0),  (426.5,796.0),  (426.5,797.0),  (427.0,797.5), 
  (427.5,798.0),  (427.5,799.0),  (427.5,800.0),  (428.0,800.5),  (428.5,801.0), 
  (428.5,802.0),  (429.0,802.5),  (429.5,803.0),  (429.5,804.0),  (429.0,804.5), 
  (428.5,805.0),  (428.0,805.5),  (427.5,806.0),  (427.0,806.5),  (426.5,807.0), 
  (427.0,807.5),  (427.5,808.0),  (428.0,808.5),  (428.5,809.0),  (428.5,810.0), 
  (429.0,810.5),  (429.5,811.0),  (429.5,812.0),  (429.5,813.0),  (430.0,813.5), 
  (430.5,813.0),  (430.5,812.0),  (430.5,811.0),  (430.5,810.0),  (430.0,809.5), 
  (429.5,809.0),  (429.5,808.0),  (429.5,807.0),  (430.0,806.5),  (430.5,806.0), 
  (431.0,805.5),  (432.0,805.5),  (433.0,805.5),  (434.0,805.5),  (434.5,805.0), 
  (434.5,804.0),  (434.0,803.5),  (433.5,803.0),  (433.0,802.5),  (432.5,802.0), 
  (432.5,801.0),  (432.0,800.5),  (431.5,800.0),  (431.0,799.5),  (430.5,799.0), 
  (430.0,798.5),  (429.5,798.0),  (429.0,797.5),  (428.5,797.0),  (428.5,796.0), 
  (428.5,795.0),  (429.0,794.5),  (429.5,794.0),  (430.0,793.5),  (430.5,793.0), 
  (430.5,792.0),  (430.5,791.0),  (430.5,790.0),  (430.5,789.0),  (431.0,788.5), 
  (431.5,788.0),  (431.5,787.0),  (431.5,786.0),  (431.5,785.0),  (431.5,784.0), 
  (431.5,783.0),  (431.5,782.0),  (431.5,781.0),  (431.5,780.0),  (431.5,779.0), 
  (431.5,778.0),  (431.5,777.0),  (431.5,776.0),  (431.5,775.0),  (432.0,774.5), 
  (432.5,774.0),  (432.5,773.0),  (433.0,772.5),  (433.5,772.0),  (434.0,771.5), 
  (434.5,772.0),  (434.5,773.0),  (434.5,774.0),  (434.5,775.0),  (434.5,776.0), 
  (434.0,776.5),  (433.5,777.0),  (433.5,778.0),  (434.0,778.5),  (434.5,779.0), 
  (434.5,780.0),  (434.5,781.0),  (434.5,782.0),  (435.0,782.5),  (435.5,783.0), 
  (435.5,784.0),  (436.0,784.5),  (436.5,785.0),  (437.0,785.5),  (437.5,786.0), 
  (438.0,786.5),  (439.0,786.5),  (440.0,786.5),  (441.0,786.5),  (441.5,786.0), 
  (442.0,785.5),  (442.5,786.0),  (443.0,786.5),  (443.5,787.0),  (443.5,788.0), 
  (443.5,789.0),  (443.5,790.0),  (444.0,790.5),  (444.5,791.0),  (444.5,792.0), 
  (445.0,792.5),  (445.5,793.0),  (446.0,793.5),  (447.0,793.5),  (448.0,793.5), 
  (449.0,793.5),  (450.0,793.5),  (451.0,793.5),  (452.0,793.5),  (453.0,793.5), 
  (453.5,793.0),  (454.0,792.5),  (455.0,792.5),  (456.0,792.5),  (456.5,793.0), 
  (456.5,794.0),  (457.0,794.5),  (457.5,795.0),  (458.0,795.5),  (458.5,796.0), 
  (458.5,797.0),  (459.0,797.5),  (459.5,798.0),  (460.0,798.5),  (460.5,799.0), 
  (461.0,799.5),  (462.0,799.5),  (462.5,800.0),  (463.0,800.5),  (463.5,801.0), 
  (464.0,801.5),  (465.0,801.5),  (466.0,801.5),  (466.5,801.0),  (466.5,800.0), 
  (466.5,799.0),  (466.5,798.0),  (466.5,797.0),  (467.0,796.5),  (467.5,796.0), 
  (468.0,795.5),  (469.0,795.5),  (470.0,795.5),  (470.5,795.0),  (470.5,794.0), 
  (471.0,793.5),  (472.0,793.5),  (473.0,793.5),  (474.0,793.5),  (475.0,793.5), 
  (476.0,793.5),  (477.0,793.5),  (477.5,794.0),  (478.0,794.5),  (478.5,795.0), 
  (479.0,795.5),  (479.5,796.0),  (480.0,796.5),  (480.5,797.0),  (480.5,798.0), 
  (481.0,798.5),  (482.0,798.5),  (483.0,798.5),  (484.0,798.5),  (485.0,798.5), 
  (485.5,798.0),  (485.5,797.0),  (485.5,796.0),  (485.5,795.0),  (485.5,794.0), 
  (486.0,793.5),  (486.5,793.0),  (486.5,792.0),  (487.0,791.5),  (487.5,792.0), 
  (488.0,792.5),  (488.5,793.0),  (489.0,793.5),  (489.5,793.0),  (490.0,792.5), 
  (490.5,792.0),  (490.5,791.0),  (490.5,790.0),  (490.5,789.0),  (491.0,788.5), 
  (491.5,789.0),  (492.0,789.5),  (493.0,789.5),  (493.5,790.0),  (494.0,790.5), 
  (494.5,791.0),  (495.0,791.5),  (496.0,791.5),  (496.5,792.0),  (497.0,792.5), 
  (498.0,792.5),  (498.5,792.0),  (499.0,791.5),  (500.0,791.5),  (501.0,791.5), 
  (501.5,792.0),  (501.5,793.0),  (501.5,794.0),  (502.0,794.5),  (502.5,795.0), 
  (503.0,795.5),  (504.0,795.5),  (505.0,795.5),  (506.0,795.5),  (506.5,795.0), 
  (507.0,794.5),  (508.0,794.5),  (508.5,794.0),  (509.0,793.5),  (509.5,793.0), 
  (510.0,792.5),  (511.0,792.5),  (511.5,793.0),  (512.0,793.5),  (513.0,793.5), 
  (513.5,794.0),  (514.0,794.5),  (515.0,794.5),  (516.0,794.5),  (517.0,794.5), 
  (518.0,794.5),  (518.5,794.0),  (518.5,793.0),  (519.0,792.5),  (519.5,792.0), 
  (520.0,791.5),  (521.0,791.5),  (521.5,792.0),  (521.5,793.0),  (521.5,794.0), 
  (521.0,794.5),  (520.5,795.0),  (520.0,795.5),  (519.5,796.0),  (519.5,797.0), 
  (519.5,798.0),  (519.0,798.5),  (518.5,799.0),  (518.0,799.5),  (517.5,800.0), 
  (517.5,801.0),  (517.5,802.0),  (517.5,803.0),  (517.5,804.0),  (517.5,805.0), 
  (517.0,805.5),  (516.5,806.0),  (516.5,807.0),  (516.0,807.5),  (515.5,808.0), 
  (515.5,809.0),  (515.0,809.5),  (514.5,810.0),  (514.5,811.0),  (514.5,812.0), 
  (514.0,812.5),  (513.5,813.0),  (513.5,814.0),  (513.5,815.0),  (513.0,815.5), 
  (512.5,816.0),  (512.5,817.0),  (512.5,818.0),  (512.5,819.0),  (512.5,820.0), 
  (512.5,821.0),  (512.5,822.0),  (512.5,823.0),  (512.5,824.0),  (512.5,825.0), 
  (513.0,825.5),  (513.5,826.0),  (514.0,826.5),  (515.0,826.5),  (516.0,826.5), 
  (516.5,826.0),  (517.0,825.5),  (517.5,825.0),  (518.0,824.5),  (518.5,824.0), 
  (518.5,823.0),  (518.5,822.0),  (518.5,821.0),  (519.0,820.5),  (519.5,820.0), 
  (519.5,819.0),  (519.5,818.0),  (520.0,817.5),  (520.5,817.0),  (521.0,816.5), 
  (521.5,816.0),  (522.0,815.5),  (522.5,815.0),  (522.5,814.0),  (522.5,813.0), 
  (523.0,812.5),  (523.5,812.0),  (523.5,811.0),  (523.5,810.0),  (524.0,809.5), 
  (524.5,809.0),  (525.0,808.5),  (525.5,808.0),  (526.0,807.5),  (526.5,807.0), 
  (526.5,806.0),  (527.0,805.5),  (527.5,805.0),  (528.0,804.5),  (528.5,804.0), 
  (528.5,803.0),  (529.0,802.5),  (529.5,802.0),  (530.0,801.5),  (531.0,801.5), 
  (532.0,801.5),  (532.5,802.0),  (532.5,803.0),  (533.0,803.5),  (534.0,803.5), 
  (535.0,803.5),  (536.0,803.5),  (536.5,803.0),  (537.0,802.5),  (537.5,803.0), 
  (538.0,803.5),  (539.0,803.5),  (540.0,803.5),  (540.5,803.0),  (541.0,802.5), 
  (541.5,802.0),  (541.5,801.0),  (541.0,800.5),  (540.5,800.0),  (540.5,799.0), 
  (541.0,798.5),  (542.0,798.5),  (542.5,799.0),  (542.5,800.0),  (542.5,801.0), 
  (542.5,802.0),  (543.0,802.5),  (544.0,802.5),  (544.5,803.0),  (545.0,803.5), 
  (546.0,803.5),  (546.5,803.0),  (547.0,802.5),  (548.0,802.5),  (549.0,802.5), 
  (550.0,802.5),  (550.5,802.0),  (551.0,801.5),  (551.5,801.0),  (552.0,800.5), 
  (552.5,801.0),  (553.0,801.5),  (554.0,801.5),  (555.0,801.5),  (556.0,801.5), 
  (557.0,801.5),  (558.0,801.5),  (558.5,801.0),  (559.0,800.5),  (560.0,800.5), 
  (561.0,800.5),  (562.0,800.5),  (563.0,800.5),  (563.5,801.0),  (563.5,802.0), 
  (564.0,802.5),  (564.5,803.0),  (564.5,804.0),  (565.0,804.5),  (566.0,804.5), 
  (567.0,804.5),  (568.0,804.5),  (569.0,804.5),  (570.0,804.5),  (571.0,804.5), 
  (572.0,804.5),  (573.0,804.5),  (573.5,804.0),  (574.0,803.5),  (574.5,803.0), 
  (575.0,802.5),  (576.0,802.5),  (577.0,802.5),  (578.0,802.5),  (579.0,802.5), 
  (580.0,802.5),  (581.0,802.5),  (582.0,802.5),  (583.0,802.5),  (583.5,802.0), 
  (584.0,801.5),  (585.0,801.5),  (585.5,801.0),  (586.0,800.5),  (587.0,800.5), 
  (588.0,800.5),  (589.0,800.5),  (590.0,800.5),  (591.0,800.5),  (592.0,800.5), 
  (593.0,800.5),  (594.0,800.5),  (595.0,800.5),  (595.5,800.0),  (596.0,799.5), 
  (597.0,799.5),  (598.0,799.5),  (599.0,799.5),  (600.0,799.5),  (601.0,799.5), 
  (602.0,799.5),  (603.0,799.5),  (604.0,799.5),  (605.0,799.5),  (606.0,799.5), 
  (607.0,799.5),  (608.0,799.5),  (609.0,799.5),  (610.0,799.5),  (611.0,799.5), 
  (612.0,799.5),  (613.0,799.5),  (614.0,799.5),  (615.0,799.5),  (616.0,799.5), 
  (617.0,799.5),  (618.0,799.5),  (619.0,799.5),  (620.0,799.5),  (621.0,799.5), 
  (622.0,799.5),  (623.0,799.5),  (623.5,800.0),  (624.0,800.5),  (625.0,800.5), 
  (626.0,800.5),  (626.5,800.0),  (627.0,799.5),  (628.0,799.5),  (629.0,799.5), 
  (630.0,799.5),  (630.5,800.0),  (631.0,800.5),  (632.0,800.5),  (633.0,800.5), 
  (634.0,800.5),  (635.0,800.5),  (636.0,800.5),  (637.0,800.5),  (638.0,800.5), 
  (639.0,800.5),  (640.0,800.5),  (641.0,800.5),  (642.0,800.5),  (643.0,800.5), 
  (644.0,800.5),  (645.0,800.5),  (646.0,800.5),  (647.0,800.5),  (648.0,800.5), 
  (649.0,800.5),  (649.5,800.0),  (650.0,799.5),  (651.0,799.5),  (652.0,799.5), 
  (653.0,799.5),  (654.0,799.5),  (655.0,799.5),  (655.5,799.0),  (656.0,798.5), 
  (657.0,798.5),  (657.5,798.0),  (658.0,797.5),  (659.0,797.5),  (660.0,797.5), 
  (660.5,797.0),  (661.0,796.5),  (662.0,796.5),  (663.0,796.5),  (664.0,796.5), 
  (664.5,796.0),  (665.0,795.5),  (666.0,795.5),  (667.0,795.5),  (668.0,795.5), 
  (668.5,795.0),  (669.0,794.5),  (669.5,794.0),  (670.0,793.5),  (670.5,793.0), 
  (671.0,792.5),  (671.5,792.0),  (672.0,791.5),  (672.5,791.0),  (672.5,790.0), 
  (673.0,789.5),  (673.5,789.0),  (674.0,788.5),  (674.5,788.0),  (675.0,787.5), 
  (675.5,787.0),  (675.5,786.0),  (676.0,785.5),  (676.5,785.0),  (677.0,784.5), 
  (678.0,784.5),  (678.5,784.0),  (679.0,783.5),  (680.0,783.5),  (681.0,783.5), 
  (681.5,783.0),  (682.0,782.5),  (683.0,782.5),  (683.5,782.0),  (684.0,781.5), 
  (684.5,781.0),  (685.0,780.5),  (685.5,780.0),  (685.5,779.0),  (686.0,778.5), 
  (687.0,778.5),  (687.5,778.0),  (688.0,777.5),  (689.0,777.5),  (690.0,777.5), 
  (691.0,777.5),  (691.5,777.0),  (692.0,776.5),  (693.0,776.5),  (694.0,776.5), 
  (695.0,776.5),  (695.5,776.0),  (696.0,775.5),  (696.5,775.0),  (697.0,774.5), 
  (697.5,774.0),  (697.5,773.0),  (698.0,772.5),  (699.0,772.5),  (699.5,772.0), 
  (699.5,771.0),  (700.0,770.5),  (700.5,770.0),  (701.0,769.5),  (702.0,769.5), 
  (703.0,769.5),  (703.5,769.0),  (704.0,768.5),  (704.5,768.0),  (705.0,767.5), 
  (706.0,767.5),  (706.5,767.0),  (706.0,766.5),  (705.5,766.0),  (705.5,765.0), 
  (705.5,764.0),  (705.0,763.5),  (704.5,763.0),  (704.5,762.0),  (704.0,761.5), 
  (703.5,761.0),  (703.0,760.5),  (702.5,760.0),  (702.5,759.0),  (702.5,758.0), 
  (702.0,757.5),  (701.5,757.0),  (701.0,756.5),  (700.5,756.0),  (700.0,755.5), 
  (699.5,755.0),  (699.0,754.5),  (698.0,754.5),  (697.5,754.0),  (697.0,753.5), 
  (696.0,753.5),  (695.0,753.5),  (694.0,753.5),  (693.5,754.0),  (694.0,754.5), 
  (695.0,754.5),  (696.0,754.5),  (696.5,755.0),  (696.0,755.5),  (695.5,756.0), 
  (696.0,756.5),  (696.5,757.0),  (696.5,758.0),  (696.0,758.5),  (695.5,758.0), 
  (695.0,757.5),  (694.5,757.0),  (694.0,756.5),  (693.5,757.0),  (693.0,757.5), 
  (692.0,757.5),  (691.5,758.0),  (691.0,758.5),  (690.5,758.0),  (690.0,757.5), 
  (689.5,757.0),  (689.0,756.5),  (688.5,757.0),  (688.5,758.0),  (688.5,759.0), 
  (688.0,759.5),  (687.5,760.0),  (687.0,760.5),  (686.5,761.0),  (686.5,762.0), 
  (686.5,763.0),  (686.0,763.5),  (685.5,764.0),  (685.0,764.5),  (684.0,764.5), 
  (683.5,765.0),  (683.5,766.0),  (683.0,766.5),  (682.5,767.0),  (683.0,767.5), 
  (683.5,768.0),  (683.5,769.0),  (683.5,770.0),  (683.5,771.0),  (683.5,772.0), 
  (684.0,772.5),  (684.5,773.0),  (685.0,773.5),  (685.5,774.0),  (686.0,774.5), 
  (687.0,774.5),  (688.0,774.5),  (688.5,775.0),  (688.5,776.0),  (688.0,776.5), 
  (687.0,776.5),  (686.0,776.5),  (685.0,776.5),  (684.5,777.0),  (684.0,777.5), 
  (683.5,778.0),  (683.5,779.0),  (683.0,779.5),  (682.0,779.5),  (681.0,779.5), 
  (680.5,779.0),  (680.0,778.5),  (679.5,778.0),  (680.0,777.5),  (680.5,777.0), 
  (681.0,776.5),  (682.0,776.5),  (682.5,776.0),  (682.5,775.0),  (682.5,774.0), 
  (682.0,773.5),  (681.5,773.0),  (681.0,772.5),  (680.5,772.0),  (680.0,771.5), 
  (679.0,771.5),  (678.5,771.0),  (678.5,770.0),  (679.0,769.5),  (679.5,769.0), 
  (679.5,768.0),  (679.5,767.0),  (679.0,766.5),  (678.5,766.0),  (678.5,765.0), 
  (678.0,764.5),  (677.5,764.0),  (677.5,763.0),  (677.0,762.5),  (676.5,762.0), 
  (676.0,761.5),  (675.5,761.0),  (675.0,760.5),  (674.5,760.0),  (674.5,759.0), 
  (674.5,758.0),  (674.5,757.0),  (674.5,756.0),  (674.5,755.0),  (674.5,754.0), 
  (674.5,753.0),  (674.5,752.0),  (674.5,751.0),  (674.5,750.0),  (674.5,749.0), 
  (674.5,748.0),  (674.5,747.0),  (674.0,746.5),  (673.5,746.0),  (673.5,745.0), 
  (673.5,744.0),  (673.0,743.5),  (672.5,743.0),  (672.5,742.0),  (672.0,741.5), 
  (671.5,741.0),  (672.0,740.5),  (672.5,740.0),  (673.0,739.5),  (674.0,739.5), 
  (675.0,739.5),  (675.5,739.0),  (675.5,738.0),  (675.0,737.5),  (674.5,737.0), 
  (674.0,736.5),  (673.5,736.0),  (673.5,735.0),  (673.5,734.0),  (673.5,733.0), 
  (674.0,732.5),  (674.5,732.0),  (675.0,731.5),  (675.5,731.0),  (675.5,730.0), 
  (675.5,729.0),  (676.0,728.5),  (676.5,728.0),  (677.0,727.5),  (677.5,727.0), 
  (678.0,726.5),  (678.5,727.0),  (679.0,727.5),  (680.0,727.5),  (680.5,728.0), 
  (681.0,728.5),  (682.0,728.5),  (682.5,729.0),  (683.0,729.5),  (684.0,729.5), 
  (685.0,729.5),  (686.0,729.5),  (687.0,729.5),  (687.5,730.0),  (688.0,730.5), 
  (689.0,730.5),  (690.0,730.5),  (691.0,730.5),  (692.0,730.5),  (693.0,730.5), 
  (694.0,730.5),  (694.5,730.0),  (695.0,729.5),  (695.5,729.0),  (696.0,728.5), 
  (697.0,728.5),  (698.0,728.5),  (699.0,728.5),  (699.5,729.0),  (700.0,729.5), 
  (700.5,729.0),  (700.5,728.0),  (700.0,727.5),  (699.5,727.0),  (699.5,726.0), 
  (700.0,725.5),  (701.0,725.5),  (702.0,725.5),  (703.0,725.5),  (704.0,725.5), 
  (705.0,725.5),  (706.0,725.5),  (707.0,725.5),  (708.0,725.5),  (708.5,726.0), 
  (709.0,726.5),  (710.0,726.5),  (710.5,727.0),  (711.0,727.5),  (711.5,728.0), 
  (712.0,728.5),  (712.5,729.0),  (713.0,729.5),  (713.5,730.0),  (713.5,731.0), 
  (714.0,731.5),  (714.5,732.0),  (714.5,733.0),  (714.5,734.0),  (714.0,734.5), 
  (713.0,734.5),  (712.0,734.5),  (711.5,735.0),  (711.0,735.5),  (710.5,736.0), 
  (710.0,736.5),  (709.5,737.0),  (709.5,738.0),  (709.5,739.0),  (710.0,739.5), 
  (710.5,740.0),  (711.0,740.5),  (712.0,740.5),  (712.5,740.0),  (713.0,739.5), 
  (713.5,739.0),  (714.0,738.5),  (714.5,738.0),  (714.5,737.0),  (715.0,736.5), 
  (716.0,736.5),  (716.5,737.0),  (717.0,737.5),  (717.5,738.0),  (718.0,738.5), 
  (718.5,739.0),  (719.0,739.5),  (719.5,740.0),  (720.0,740.5),  (720.5,741.0), 
  (721.0,741.5),  (721.5,742.0),  (722.0,742.5),  (722.5,743.0),  (723.0,743.5), 
  (723.5,744.0),  (723.5,745.0),  (724.0,745.5),  (724.5,746.0),  (725.0,746.5), 
  (725.5,747.0),  (726.0,747.5),  (727.0,747.5),  (728.0,747.5),  (728.5,747.0), 
  (728.5,746.0),  (729.0,745.5),  (729.5,745.0),  (729.5,744.0),  (730.0,743.5), 
  (730.5,743.0),  (730.5,742.0),  (730.5,741.0),  (731.0,740.5),  (731.5,740.0), 
  (731.5,739.0),  (731.5,738.0),  (731.5,737.0),  (731.5,736.0),  (731.5,735.0), 
  (731.5,734.0),  (732.0,733.5),  (732.5,733.0),  (733.0,732.5),  (734.0,732.5), 
  (735.0,732.5),  (736.0,732.5),  (737.0,732.5),  (737.5,732.0),  (738.0,731.5), 
  (739.0,731.5),  (739.5,731.0),  (740.0,730.5),  (740.5,730.0),  (741.0,729.5), 
  (741.5,729.0),  (742.0,728.5),  (742.5,728.0),  (743.0,727.5),  (743.5,727.0), 
  (744.0,726.5),  (744.5,726.0),  (744.5,725.0),  (745.0,724.5),  (745.5,724.0), 
  (746.0,723.5),  (746.5,723.0),  (746.5,722.0),  (747.0,721.5),  (747.5,721.0), 
  (747.5,720.0),  (747.5,719.0),  (747.5,718.0),  (747.5,717.0),  (747.5,716.0), 
  (747.5,715.0),  (747.5,714.0),  (747.0,713.5),  (746.5,713.0),  (746.5,712.0), 
  (746.5,711.0),  (746.5,710.0),  (746.5,709.0),  (746.5,708.0),  (747.0,707.5), 
  (747.5,707.0),  (747.5,706.0),  (748.0,705.5),  (748.5,705.0),  (748.5,704.0), 
  (748.5,703.0),  (748.5,702.0),  (748.5,701.0),  (749.0,700.5),  (749.5,700.0), 
  (749.5,699.0),  (749.5,698.0),  (750.0,697.5),  (750.5,697.0), 
    ]
    
    print("🚀 DIVIDED SPLINE COMPRESSION TEST")
    print("=" * 50)
    
    # Parameters
    num_sublists = 3
    compression_ratio = 0.2
    
    # Single function call that divides and compresses
    result = compress_shape_divided(sample_coordinates, num_sublists, compression_ratio)
    
    if result:
        # Print analysis
        print_divided_compression_analysis(result)
        
        # Visualize
        print("\n" + "="*70)
        print("📊 VISUALIZATION")
        print("="*70)
        visualize_divided_compression(sample_coordinates, result)
        
        # Print coefficients for each sublist
        print("\n" + "="*70)
        print("🔧 SUBLIST COEFFICIENTS")
        print("="*70)
        for i, sub_res in enumerate(result['sublist_results']):
            tck = sub_res['spline_coefficients']
            knots, coefficients, degree = tck
            print(f"\n📦 Sublist {i+1}:")
            print(f"   Degree: {degree}")
            print(f"   Knots: {len(knots)}")
            print(f"   X coefficients: {len(coefficients[0])}")
            print(f"   Y coefficients: {len(coefficients[1])}")
            print(f"   First 3 X coefficients: {coefficients[0][:3]}")
            print(f"   First 3 Y coefficients: {coefficients[1][:3]}")
    else:
        print("❌ Divided compression failed!")