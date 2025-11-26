import numpy as np
import matplotlib.pyplot as plt









def compress_irregular_shape(coordinates, compression_ratio=0.1, decimal_places=3):
    """
    Compress an irregular shape using Fourier descriptors
    
    Parameters:
    -----------
    coordinates : list of tuples or numpy array
        List of (x,y) coordinates representing the irregular shape
    compression_ratio : float (default: 0.1)
        Fraction of coefficients to keep (0.1 = keep top 10%)
    decimal_places : int or None (default: 3)
        Number of decimal places to store (None for full precision)
    
    Returns:
    --------
    dict with:
        'compressed_coeffs' : list of tuples (index, real, imag)
        'storage_info' : dict with storage statistics
        'reconstruction_info' : dict with error metrics
        'original_length' : int (important for reconstruction!)
    """
    
    # Convert to numpy array and ensure closed shape
    coords = np.array(coordinates)
    if not np.allclose(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0]])
    
    # Convert to complex numbers and compute Fourier coefficients
    z_points = coords[:, 0] + 1j * coords[:, 1]
    fourier_coeffs = np.fft.fft(z_points)
    
    # Keep only the most important coefficients
    n_keep = max(1, int(len(fourier_coeffs) * compression_ratio))
    magnitudes = np.abs(fourier_coeffs)
    largest_indices = np.argsort(magnitudes)[-n_keep:]
    
    # Create compressed coefficients with rounding
    compressed_coeffs = np.zeros_like(fourier_coeffs)
    compressed_coeffs[largest_indices] = fourier_coeffs[largest_indices]
    
    if decimal_places is not None:
        compressed_coeffs = np.round(compressed_coeffs, decimals=decimal_places)
    
    # Prepare the output in compact format: (index, real, imag)
    compressed_output = []
    kept_indices = np.where(compressed_coeffs != 0)[0]
    for idx in kept_indices:
        compressed_output.append((
            int(idx),  # index
            float(compressed_coeffs[idx].real),  # real part
            float(compressed_coeffs[idx].imag)   # imaginary part
        ))
    
    # Calculate storage statistics
    original_size = len(coords) * 2  # x and y coordinates
    compressed_size = len(compressed_output) * 3  # index + real + imag
    
    # Calculate reconstruction error
    z_original = np.fft.ifft(fourier_coeffs)
    z_reconstructed = np.fft.ifft(compressed_coeffs)
    reconstruction_error = np.mean(np.sqrt(
        (z_reconstructed.real - z_original.real)**2 + 
        (z_reconstructed.imag - z_original.imag)**2
    ))
    
    # Prepare return dictionary
    result = {
        'compressed_coeffs': compressed_output,
        'original_length': len(fourier_coeffs),  # CRITICAL: store original length!
        'storage_info': {
            'original_points': len(coords),
            'compressed_coeffs_count': len(compressed_output),
            'compression_ratio': compression_ratio,
            'decimal_places': decimal_places,
            'storage_reduction': f"{(1 - compressed_size/original_size)*100:.1f}%",
            'bytes_saved_estimate': (original_size - compressed_size) * 8  # 8 bytes per float
        },
        'reconstruction_info': {
            'max_error': np.max(np.abs(z_reconstructed - z_original)),
            'mean_error': reconstruction_error,
            'rmse': np.sqrt(np.mean(np.abs(z_reconstructed - z_original)**2))
        }
    }
    
    return result

def reconstruct_from_compressed(compressed_coeffs, original_length):
    """
    Reconstruct shape from compressed coefficients
    
    Parameters:
    -----------
    compressed_coeffs : list of tuples (index, real, imag)
        Output from compress_irregular_shape function
    original_length : int
        The original length of the Fourier coefficients array
    
    Returns:
    --------
    numpy array of (x, y) coordinates
    """
    # Recreate the Fourier coefficients array with the EXACT original length
    fourier_coeffs = np.zeros(original_length, dtype=complex)
    for idx, real, imag in compressed_coeffs:
        if idx < original_length:
            fourier_coeffs[idx] = real + 1j * imag
    
    # Inverse FFT to get the shape - this will have original_length points
    z_reconstructed = np.fft.ifft(fourier_coeffs)
    
    # Return as (x, y) coordinates
    return np.column_stack([z_reconstructed.real, z_reconstructed.imag])























def create_test_shapes():
    """
    Create various types of irregular shapes for testing compression
    """
    shapes = {}
    
    # 1. Organic blob shape
    theta = np.linspace(0, 2*np.pi, 50, endpoint=False)
    r = 1 + 0.4 * np.sin(2*theta) + 0.3 * np.cos(5*theta) + 0.2 * np.sin(7*theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    shapes['Organic Blob'] = list(zip(x, y))
    
    # 2. Starfish-like shape
    theta = np.linspace(0, 2*np.pi, 60, endpoint=False)
    r = 1 + 0.5 * np.cos(5*theta) + 0.2 * np.cos(10*theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    shapes['Starfish'] = list(zip(x, y))
    
    # 3. Kidney bean shape (asymmetric)
    theta = np.linspace(0, 2*np.pi, 45, endpoint=False)
    x = 1.5 * np.cos(theta) + 0.8 * np.cos(2*theta)
    y = np.sin(theta) + 0.5 * np.sin(3*theta)
    shapes['Kidney Bean'] = list(zip(x, y))
    
    # 4. Spiral-like irregular shape
    theta = np.linspace(0, 2*np.pi, 55, endpoint=False)
    r = 1 + 0.3 * np.sin(3*theta) * np.cos(2*theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    shapes['Spiral Irregular'] = list(zip(x, y))
    
    # 5. Very jagged shape
    theta = np.linspace(0, 2*np.pi, 40, endpoint=False)
    r = 1 + 0.6 * np.random.normal(size=len(theta))
    # Smooth slightly but keep jaggedness
    from scipy.ndimage import gaussian_filter1d
    r = gaussian_filter1d(r, sigma=1.2)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    shapes['Jagged Polygon'] = list(zip(x, y))
    
    # 6. Heart-like shape (mathematical heart curve)
    t = np.linspace(0, 2*np.pi, 50, endpoint=False)
    x = 16 * np.sin(t)**3
    y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
    # Normalize
    x = x / 20
    y = y / 20
    shapes['Heart'] = list(zip(x, y))
    
    # 7. Flower shape
    theta = np.linspace(0, 2*np.pi, 65, endpoint=False)
    r = 1 + 0.4 * np.cos(6*theta) + 0.2 * np.cos(12*theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    shapes['Flower'] = list(zip(x, y))
    
    # 8. Gear-like shape
    theta = np.linspace(0, 2*np.pi, 48, endpoint=False)
    r = 1 + 0.3 * np.sign(np.sin(8*theta))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    shapes['Gear'] = list(zip(x, y))
    
    # 9. Random amoeba shape
    theta = np.linspace(0, 2*np.pi, 35, endpoint=False)
    r = 1 + 0.5 * np.random.normal(size=len(theta))
    r = gaussian_filter1d(r, sigma=2.5)  # More smoothing for organic look
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    shapes['Amoeba'] = list(zip(x, y))
    
    # 10. Complex multi-lobed shape
    theta = np.linspace(0, 2*np.pi, 70, endpoint=False)
    r = (1 + 0.4 * np.sin(2*theta) + 
         0.3 * np.cos(3*theta) + 
         0.2 * np.sin(5*theta) + 
         0.1 * np.cos(7*theta))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    shapes['Multi-lobed'] = list(zip(x, y))
    
    return shapes

def test_shape_compression(shape_name, coordinates, compression_ratios=[1.0, 0.3, 0.1, 0.05]):
    
    """
    Test compression on a single shape with multiple compression levels
    """
    print(f"\n{'='*60}")
    print(f"TESTING: {shape_name}")
    print(f"{'='*60}")
    
    results = {}
    
    for ratio in compression_ratios:
        # Compress the shape
        result = compress_irregular_shape(
            coordinates, 
            compression_ratio=ratio,
            decimal_places=3
        )
        
        storage = result['storage_info']
        reconstruction = result['reconstruction_info']
        
        print(f"Compression {ratio:.0%}:")
        print(f"  Coefficients: {storage['compressed_coeffs_count']:2d}")
        print(f"  Storage reduction: {storage['storage_reduction']:>8}")
        print(f"  Mean error: {reconstruction['mean_error']:.6f}")
        
        results[ratio] = result
    
    # Plot the results
    plot_multiple_compressions(shape_name, coordinates, results)
    
    return results

def plot_multiple_compressions(shape_name, original_coords, compression_results):
    """
    Plot original shape and multiple compression levels in one figure
    """
    n_plots = len(compression_results) + 1
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot original
    original_array = np.array(original_coords)
    axes[0].plot(original_array[:, 0], original_array[:, 1], 'b-o', 
                linewidth=2, markersize=3, label='Original')
    axes[0].set_title(f'{shape_name}\nOriginal\n{len(original_coords)} points')
    axes[0].axis('equal')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot compressed versions
    ratios = list(compression_results.keys())
    for i, ratio in enumerate(ratios, 1):
        result = compression_results[ratio]
        
        # Reconstruct shape
        reconstructed_coords = reconstruct_from_compressed(
            result['compressed_coeffs'],
            result['original_length']
        )
        
        axes[i].plot(reconstructed_coords[:, 0], reconstructed_coords[:, 1], 'r-o', 
                    linewidth=2, markersize=3, label='Reconstructed')
        axes[i].plot(original_array[:, 0], original_array[:, 1], 'b--', 
                    linewidth=1, alpha=0.3, label='Original')
        
        storage = result['storage_info']
        reconstruction = result['reconstruction_info']
        
        axes[i].set_title(f'Compression: {ratio:.0%}\n'
                         f'{storage["compressed_coeffs_count"]} coeffs\n'
                         f'Error: {reconstruction["mean_error"]:.4f}')
        axes[i].axis('equal')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    
    # Hide empty subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def compare_all_shapes_compression():
    """
    Compare compression performance across all test shapes
    """
    shapes = create_test_shapes()
    compression_ratio = 0.1  # Test with 10% compression
    
    print("COMPARING COMPRESSION ACROSS ALL SHAPES")
    print("=" * 70)
    print(f"Testing with {compression_ratio:.0%} compression ratio")
    print("=" * 70)
    
    results = {}
    
    for shape_name, coordinates in shapes.items():
        result = compress_irregular_shape(
            coordinates, 
            compression_ratio=compression_ratio,
            decimal_places=3
        )
        
        results[shape_name] = {
            'error': result['reconstruction_info']['mean_error'],
            'coefficients': result['storage_info']['compressed_coeffs_count'],
            'storage_reduction': result['storage_info']['storage_reduction'],
            'original_points': result['storage_info']['original_points']
        }
    
    # Print comparison table
    print("\nCOMPRESSION RESULTS (10% coefficients kept):")
    print("-" * 85)
    print(f"{'Shape':<20} {'Points':<8} {'Coeffs':<8} {'Storage':<12} {'Mean Error':<12}")
    print("-" * 85)
    
    for shape_name, data in results.items():
        print(f"{shape_name:<20} {data['original_points']:<8} {data['coefficients']:<8} "
              f"{data['storage_reduction']:<12} {data['error']:.6f}")
    
    return results

def plot_compression_efficiency(all_results):
    """
    Plot compression efficiency across all shapes
    """
    shapes = list(all_results.keys())
    errors = [all_results[shape]['error'] for shape in shapes]
    coefficients = [all_results[shape]['coefficients'] for shape in shapes]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Reconstruction error by shape
    bars1 = ax1.bar(shapes, errors, color='lightcoral', alpha=0.7)
    ax1.set_title('Reconstruction Error by Shape\n(10% Compression)')
    ax1.set_ylabel('Mean Reconstruction Error')
    ax1.set_xlabel('Shape Type')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, error in zip(bars1, errors):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                f'{error:.4f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Coefficients needed by shape
    bars2 = ax2.bar(shapes, coefficients, color='lightblue', alpha=0.7)
    ax2.set_title('Coefficients Needed by Shape\n(10% Compression)')
    ax2.set_ylabel('Number of Coefficients')
    ax2.set_xlabel('Shape Type')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, coeff in zip(bars2, coefficients):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{coeff}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()

# Main testing function
def run_comprehensive_tests():
    """
    Run comprehensive tests on all shapes
    """
    shapes = create_test_shapes()
    
    print("COMPREHENSIVE FOURIER COMPRESSION TESTING")
    print("=" * 70)
    print(f"Testing {len(shapes)} different irregular shapes")
    print("=" * 70)
    
    # Test a few representative shapes in detail
    test_shapes = ['Organic Blob', 'Starfish', 'Jagged Polygon', 'Heart', 'Amoeba']
    
    all_detailed_results = {}
    for shape_name in test_shapes:
        if shape_name in shapes:
            results = test_shape_compression(
                shape_name, 
                shapes[shape_name],
                compression_ratios=[1.0, 0.3, 0.1, 0.05]
            )
            all_detailed_results[shape_name] = results
    
    # Compare all shapes with fixed compression
    all_results = compare_all_shapes_compression()
    
    # Plot efficiency comparison
    plot_compression_efficiency(all_results)
    
    return all_detailed_results, all_results

# Quick individual shape test
def test_specific_shape(shape_name):
    """
    Test compression on a specific shape
    """
    shapes = create_test_shapes()
    
    if shape_name in shapes:
        print(f"Testing {shape_name}...")
        coordinates = shapes[shape_name]
        
        # Test multiple compression levels
        results = test_shape_compression(
            shape_name, 
            coordinates,
            compression_ratios=[1.0, 0.2, 0.1, 0.05]
        )
        return results
    else:
        print(f"Shape '{shape_name}' not found. Available shapes: {list(shapes.keys())}")
        return None

# Run the tests
if __name__ == "__main__":
    # Option 1: Run comprehensive tests on all shapes
    detailed_results, comparison_results = run_comprehensive_tests()
    
    # Option 2: Test a specific shape
    # results = test_specific_shape('Heart')
    
    # Option 3: Test multiple specific shapes
    # for shape_name in ['Flower', 'Gear', 'Multi-lobed']:
    #     test_specific_shape(shape_name)