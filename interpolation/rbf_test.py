import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from scipy.spatial.distance import cdist

def create_test_shapes():
    """
    Create various test shapes for RBF compression testing
    """
    shapes = {}
    
    # 1. Circle (simple smooth shape)
    theta = np.linspace(0, 2*np.pi, 100, endpoint=False)
    x_circle = np.cos(theta)
    y_circle = np.sin(theta)
    shapes['Circle'] = list(zip(x_circle, y_circle))
    
    # 2. Ellipse
    x_ellipse = 1.5 * np.cos(theta)
    y_ellipse = 0.7 * np.sin(theta)
    shapes['Ellipse'] = list(zip(x_ellipse, y_ellipse))
    
    # 3. Star shape
    x_star = np.cos(theta) * (1 + 0.5 * np.cos(5*theta))
    y_star = np.sin(theta) * (1 + 0.5 * np.cos(5*theta))
    shapes['Star'] = list(zip(x_star, y_star))
    
    # 4. Random blob (irregular shape)
    r_blob = 1 + 0.3 * np.random.normal(size=len(theta))
    r_blob = np.convolve(r_blob, np.ones(5)/5, mode='same')  # Smooth
    x_blob = r_blob * np.cos(theta)
    y_blob = r_blob * np.sin(theta)
    shapes['Random Blob'] = list(zip(x_blob, y_blob))
    
    # 5. Square (sharp corners)
    t_square = np.linspace(0, 4, 100, endpoint=False)
    x_square = np.where(t_square < 1, 1, np.where(t_square < 2, 2-t_square, 
                     np.where(t_square < 3, -1, t_square-4)))
    y_square = np.where(t_square < 1, t_square-0.5, np.where(t_square < 2, 0.5,
                     np.where(t_square < 3, 2.5-t_square, -0.5)))
    shapes['Square'] = list(zip(x_square, y_square))
    
    # 6. Heart shape
    t_heart = np.linspace(0, 2*np.pi, 80, endpoint=False)
    x_heart = 16 * np.sin(t_heart)**3
    y_heart = 13 * np.cos(t_heart) - 5 * np.cos(2*t_heart) - 2 * np.cos(3*t_heart) - np.cos(4*t_heart)
    x_heart, y_heart = x_heart/16, y_heart/16  # Normalize
    shapes['Heart'] = list(zip(x_heart, y_heart))
    
    return shapes

def compress_shape_rbf(coordinates, compression_ratio=0.1, rbf_function='thin_plate'):
    """
    Compress a shape using RBF interpolation
    """
    coords = np.array(coordinates)
    
    # Ensure closed shape
    if not np.allclose(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0]])
    
    # Create parameter t (0 to 1) along the curve
    t_original = np.linspace(0, 1, len(coords))
    
    # Select centers strategically
    n_centers = max(4, int(len(coords) * compression_ratio))
    
    # Use curvature-based sampling for centers
    centers_indices = select_centers_by_curvature(coords, n_centers)
    centers_t = t_original[centers_indices]
    centers_coords = coords[centers_indices]
    
    # Separate coordinates
    x_original = coords[:, 0]
    y_original = coords[:, 1]
    
    try:
        # Create RBF interpolators
        rbf_x = Rbf(centers_t, centers_coords[:, 0], function=rbf_function)
        rbf_y = Rbf(centers_t, centers_coords[:, 1], function=rbf_function)
        
        # Reconstruct
        t_test = np.linspace(0, 1, len(coords))
        x_reconstructed = rbf_x(t_test)
        y_reconstructed = rbf_y(t_test)
        
        # Calculate error
        error = np.mean(np.sqrt((x_reconstructed - x_original)**2 + 
                               (y_reconstructed - y_original)**2))
        
        return {
            'compressed_data': {
                'centers_t': centers_t.tolist(),
                'centers_x': centers_coords[:, 0].tolist(),
                'centers_y': centers_coords[:, 1].tolist(),
                'rbf_function': rbf_function
            },
            'reconstruction': {
                'x': x_reconstructed,
                'y': y_reconstructed
            },
            'metrics': {
                'mean_error': error,
                'max_error': np.max(np.sqrt((x_reconstructed - x_original)**2 + 
                                          (y_reconstructed - y_original)**2)),
                'compression_ratio': compression_ratio,
                'original_points': len(coords),
                'centers_used': n_centers
            }
        }
        
    except Exception as e:
        print(f"RBF failed: {e}")
        return None

def select_centers_by_curvature(coords, n_centers):
    """
    Select centers based on curvature (more centers where shape changes rapidly)
    """
    n_points = len(coords)
    
    if n_centers >= n_points:
        return np.arange(n_points)
    
    # Calculate approximate curvature
    curvature = np.zeros(n_points)
    for i in range(1, n_points-1):
        v1 = coords[i] - coords[i-1]
        v2 = coords[i+1] - coords[i]
        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1, 1)
            curvature[i] = np.arccos(cos_angle)
    
    curvature[0] = curvature[1]
    curvature[-1] = curvature[-2]
    
    # Always include first point
    selected = [0]
    
    # Select points with highest curvature
    remaining_indices = list(range(1, n_points))
    curvature_remaining = curvature[1:]
    
    # Get high curvature points
    high_curve_indices = np.argsort(curvature_remaining)[-(n_centers-1):][::-1]
    selected.extend([i+1 for i in high_curve_indices])
    
    return np.array(selected)

def test_rbf_functions_on_shape(shape_coords, shape_name):
    """
    Test different RBF functions on a single shape
    """
    rbf_functions = ['thin_plate', 'multiquadric', 'gaussian', 'inverse', 'linear']
    compression_ratio = 0.1
    
    print(f"\nTesting {shape_name} ({len(shape_coords)} points):")
    print("=" * 50)
    
    results = {}
    
    for rbf_func in rbf_functions:
        result = compress_shape_rbf(shape_coords, compression_ratio, rbf_func)
        if result:
            error = result['metrics']['mean_error']
            centers = result['metrics']['centers_used']
            results[rbf_func] = result
            print(f"  {rbf_func:12} | Error: {error:7.4f} | Centers: {centers:2d}")
        else:
            print(f"  {rbf_func:12} | Failed")
    
    return results

def visualize_shape_comparison(shape_coords, results, shape_name):
    """
    Visualize original vs reconstructed shapes for different RBF functions
    """
    n_functions = len(results)
    if n_functions == 0:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot original shape
    original_x, original_y = zip(*shape_coords)
    axes[0].plot(original_x, original_y, 'b-', linewidth=2, label='Original')
    axes[0].set_title(f'{shape_name}\nOriginal\n{len(shape_coords)} points')
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot each RBF function result
    for i, (rbf_func, result) in enumerate(results.items()):
        if i >= 5:  # Max 5 subplots
            break
            
        recon_x = result['reconstruction']['x']
        recon_y = result['reconstruction']['y']
        centers_x = result['compressed_data']['centers_x']
        centers_y = result['compressed_data']['centers_y']
        error = result['metrics']['mean_error']
        centers = result['metrics']['centers_used']
        
        axes[i+1].plot(recon_x, recon_y, 'r-', linewidth=2, label='Reconstructed')
        axes[i+1].plot(centers_x, centers_y, 'go', markersize=4, 
                      markerfacecolor='none', markeredgewidth=2, label='Centers')
        axes[i+1].plot(original_x, original_y, 'b--', linewidth=1, alpha=0.5, label='Original')
        
        axes[i+1].set_title(f'{rbf_func}\n{centers} centers\nError: {error:.4f}')
        axes[i+1].set_aspect('equal')
        axes[i+1].grid(True, alpha=0.3)
        axes[i+1].legend()
    
    # Hide empty subplots
    for i in range(n_functions + 1, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def test_compression_ratios(shape_coords, shape_name):
    """
    Test different compression ratios on a shape
    """
    compression_ratios = [0.05, 0.1, 0.2, 0.3, 0.5]
    rbf_function = 'thin_plate'  # Use the generally best one
    
    print(f"\nTesting compression ratios on {shape_name}:")
    print("=" * 50)
    
    results = {}
    
    for ratio in compression_ratios:
        result = compress_shape_rbf(shape_coords, ratio, rbf_function)
        if result:
            error = result['metrics']['mean_error']
            centers = result['metrics']['centers_used']
            original_points = result['metrics']['original_points']
            storage_reduction = (1 - centers/original_points) * 100
            
            results[ratio] = result
            print(f"  Ratio {ratio:5.0%} | Centers: {centers:3d} | "
                  f"Error: {error:7.4f} | Storage: {storage_reduction:5.1f}%")
    
    return results

def plot_compression_analysis(compression_results, shape_name):
    """
    Plot compression ratio vs error analysis
    """
    ratios = list(compression_results.keys())
    errors = [compression_results[r]['metrics']['mean_error'] for r in ratios]
    centers = [compression_results[r]['metrics']['centers_used'] for r in ratios]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Error vs compression ratio
    ax1.plot(ratios, errors, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Compression Ratio')
    ax1.set_ylabel('Reconstruction Error')
    ax1.set_title(f'{shape_name}\nError vs Compression Ratio')
    ax1.grid(True, alpha=0.3)
    
    # Centers used vs compression ratio
    ax2.plot(ratios, centers, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Compression Ratio')
    ax2.set_ylabel('Centers Used')
    ax2.set_title(f'{shape_name}\nCenters vs Compression Ratio')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main_rbf_test_suite():
    """
    Main function to run the complete RBF test suite
    """
    print("RBF COMPRESSION TEST SUITE")
    print("=" * 60)
    
    # Create test shapes
    shapes = create_test_shapes()
    
    print(f"Created {len(shapes)} test shapes:")
    for name, coords in shapes.items():
        print(f"  {name}: {len(coords)} points")
    
    # Test 1: Compare RBF functions on each shape
    print("\n" + "=" * 60)
    print("TEST 1: COMPARING RBF FUNCTIONS")
    print("=" * 60)
    
    all_function_results = {}
    for shape_name, shape_coords in shapes.items():
        results = test_rbf_functions_on_shape(shape_coords, shape_name)
        all_function_results[shape_name] = results
        visualize_shape_comparison(shape_coords, results, shape_name)
    
    # Test 2: Test compression ratios on selected shapes
    print("\n" + "=" * 60)
    print("TEST 2: COMPRESSION RATIO ANALYSIS")
    print("=" * 60)
    
    test_shapes = ['Circle', 'Star', 'Random Blob']  # Representative shapes
    compression_results = {}
    
    for shape_name in test_shapes:
        if shape_name in shapes:
            results = test_compression_ratios(shapes[shape_name], shape_name)
            compression_results[shape_name] = results
            plot_compression_analysis(results, shape_name)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("RBF performs best on smooth shapes (Circle, Ellipse)")
    print("Higher compression ratios work well for simple shapes")
    print("Complex shapes need more centers for good reconstruction")
    print("'thin_plate' and 'multiquadric' generally work best")

# Run the test suite
if __name__ == "__main__":
    main_rbf_test_suite()