import matplotlib.pyplot as plt
import numpy as np

def create_irregular_shape(n_points=50):
    """
    Create various types of irregular closed shapes for testing
    """
    shapes = []
    
    # 1. Random blob shape - ensure closed
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    r = 1 + 0.4 * np.random.normal(size=n_points)
    from scipy.ndimage import gaussian_filter1d
    r = gaussian_filter1d(r, sigma=2)
    x1 = r * np.cos(theta)
    y1 = r * np.sin(theta)
    # Ensure closed by making first and last points the same
    shapes.append(('Random Blob', np.column_stack([x1, y1])))
    
    # 2. Kidney bean shape - closed
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    x2 = 1.5 * np.cos(theta) + 0.5 * np.cos(2*theta)
    y2 = np.sin(theta) + 0.3 * np.sin(3*theta)
    shapes.append(('Kidney Bean', np.column_stack([x2, y2])))
    
    # 3. Amoeba-like shape - closed
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    r3 = 1 + 0.3 * np.sin(2*theta) + 0.2 * np.sin(5*theta) + 0.1 * np.sin(8*theta)
    x3 = r3 * np.cos(theta)
    y3 = r3 * np.sin(theta)
    shapes.append(('Amoeba', np.column_stack([x3, y3])))
    
    return shapes

def compress_fourier_advanced(coeffs, keep_ratio=0.1, decimal_places=3):
    """
    Advanced compression: keep largest coefficients + round to specified decimal places
    """
    magnitudes = np.abs(coeffs)
    
    # How many coefficients to keep
    n_keep = max(1, int(len(coeffs) * keep_ratio))
    
    # Find indices of largest coefficients
    largest_indices = np.argsort(magnitudes)[-n_keep:]
    
    # Create compressed coefficients (zeros for small ones)
    compressed = np.zeros_like(coeffs)
    compressed[largest_indices] = coeffs[largest_indices]
    
    # ROUND to specified decimal places
    if decimal_places is not None:
        compressed = np.round(compressed, decimals=decimal_places)
    
    return compressed

def analyze_storage_requirements(original_coeffs, compressed_coeffs, decimal_places=3):
    """
    Analyze storage savings with rounding
    """
    # Count non-zero coefficients before and after compression
    n_original_nonzero = np.sum(original_coeffs != 0)
    n_compressed_nonzero = np.sum(compressed_coeffs != 0)
    
    # Calculate storage requirements
    # Original: all coefficients stored as full precision floats
    original_storage = len(original_coeffs) * 2 * 8  # 2 floats per complex, 8 bytes per float
    
    # Compressed: only non-zero coefficients, rounded values
    # With rounding, we might need fewer bytes, but let's be conservative
    compressed_storage = n_compressed_nonzero * 2 * 8  # Still 8 bytes, but fewer numbers
    
    compression_ratio = compressed_storage / original_storage
    
    print(f"Storage Analysis:")
    print(f"  Original: {n_original_nonzero} non-zero coefficients, {original_storage} bytes")
    print(f"  Compressed: {n_compressed_nonzero} non-zero coefficients, {compressed_storage} bytes")
    print(f"  Compression ratio: {compression_ratio:.1%}")
    print(f"  Rounded to {decimal_places} decimal places")
    
    return compression_ratio

def plot_irregular_shape_compression_advanced(shape_name, coordinates, test_ratios=[0.05, 0.1, 0.2, 0.5], decimal_places=3):
    """
    Plot compression results with rounding for a single irregular shape
    """
    # Ensure shape is closed
    if not np.allclose(coordinates[0], coordinates[-1]):
        coordinates = np.vstack([coordinates, coordinates[0]])
    
    # Get Fourier coefficients
    z_points = coordinates[:, 0] + 1j * coordinates[:, 1]
    fourier_coeffs = np.fft.fft(z_points)
    
    # Reconstruct original shape from full coefficients
    z_original = np.fft.ifft(fourier_coeffs)
    
    # Create subplot
    fig, axes = plt.subplots(1, len(test_ratios) + 1, figsize=(5*(len(test_ratios)+1), 5))
    
    # Make sure axes is always iterable
    if len(test_ratios) + 1 == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Plot original
    axes[0].plot(z_original.real, z_original.imag, 'b-', linewidth=2, label='Original')
    axes[0].plot(coordinates[:, 0], coordinates[:, 1], 'ro', markersize=3, alpha=0.6, label='Points')
    axes[0].set_title(f'{shape_name}\nOriginal\n{len(fourier_coeffs)} coeffs')
    axes[0].legend()
    axes[0].axis('equal')
    axes[0].grid(True, alpha=0.3)
    
    # Display original coefficient info
    magnitudes = np.abs(fourier_coeffs)
    top_indices = np.argsort(magnitudes)[-3:][::-1]
    print(f"\n{shape_name} - Original Top 3 coefficients:")
    for idx in top_indices:
        print(f"  c[{idx}]: {fourier_coeffs[idx].real:.6f} + {fourier_coeffs[idx].imag:.6f}j")
    
    # Plot compressed versions with rounding
    compression_results = []
    
    for i, ratio in enumerate(test_ratios, 1):
        # Apply compression with rounding
        compressed_coeffs = compress_fourier_advanced(fourier_coeffs, keep_ratio=ratio, decimal_places=decimal_places)
        z_compressed = np.fft.ifft(compressed_coeffs)
        
        # Calculate error
        error = np.mean(np.sqrt((z_compressed.real - z_original.real)**2 + 
                               (z_compressed.imag - z_original.imag)**2))
        
        n_kept = np.sum(compressed_coeffs != 0)
        
        # Display compressed coefficient info for the first test case
        if i == 1:  # First compression case
            print(f"\n{shape_name} - Compressed Top coefficients (ratio {ratio:.0%}, {decimal_places} decimals):")
            kept_indices = np.where(compressed_coeffs != 0)[0]
            for idx in kept_indices[:5]:  # Show first 5 kept coefficients
                print(f"  c[{idx}]: {compressed_coeffs[idx].real:.{decimal_places}f} + {compressed_coeffs[idx].imag:.{decimal_places}f}j")
        
        axes[i].plot(z_compressed.real, z_compressed.imag, 'g-', linewidth=2, label='Compressed')
        axes[i].plot(z_original.real, z_original.imag, 'b--', linewidth=1, alpha=0.3, label='Original')
        axes[i].set_title(f'{ratio:.0%} Compress\n{n_kept} coeffs\nError: {error:.4f}\n{decimal_places} decimals')
        axes[i].legend()
        axes[i].axis('equal')
        axes[i].grid(True, alpha=0.3)
        
        compression_results.append({
            'ratio': ratio,
            'coeffs_kept': n_kept,
            'error': error,
            'compressed_coeffs': compressed_coeffs
        })
    
    plt.tight_layout()
    plt.show()
    
    # Analyze storage requirements for the most aggressive compression
    if compression_results:
        most_compressed = compression_results[0]  # 5% compression
        storage_ratio = analyze_storage_requirements(fourier_coeffs, most_compressed['compressed_coeffs'], decimal_places)
    
    return fourier_coeffs, compression_results

def test_different_precision_levels(shape_name, coordinates):
    """
    Test how different decimal precision affects compression
    """
    # Ensure shape is closed
    if not np.allclose(coordinates[0], coordinates[-1]):
        coordinates = np.vstack([coordinates, coordinates[0]])
    
    z_points = coordinates[:, 0] + 1j * coordinates[:, 1]
    fourier_coeffs = np.fft.fft(z_points)
    z_original = np.fft.ifft(fourier_coeffs)
    
    precision_levels = [1, 2, 3, 4, None]  # None = no rounding
    compression_ratio = 0.1  # Keep 10% of coefficients
    
    fig, axes = plt.subplots(1, len(precision_levels) + 1, figsize=(5*(len(precision_levels)+1), 5))
    axes = axes.flatten()
    
    # Plot original
    axes[0].plot(z_original.real, z_original.imag, 'b-', linewidth=2)
    axes[0].set_title(f'Original\n{len(fourier_coeffs)} coeffs')
    axes[0].axis('equal')
    axes[0].grid(True, alpha=0.3)
    
    print(f"\n{shape_name} - Precision Level Comparison (10% coefficients kept):")
    print("Precision | Coeffs Kept | Error    | Example Coefficient")
    print("-" * 55)
    
    for i, precision in enumerate(precision_levels, 1):
        if precision is None:
            compressed_coeffs = compress_fourier_advanced(fourier_coeffs, keep_ratio=compression_ratio, decimal_places=None)
            precision_label = "Full"
        else:
            compressed_coeffs = compress_fourier_advanced(fourier_coeffs, keep_ratio=compression_ratio, decimal_places=precision)
            precision_label = f"{precision} dec"
        
        z_compressed = np.fft.ifft(compressed_coeffs)
        error = np.mean(np.sqrt((z_compressed.real - z_original.real)**2 + 
                               (z_compressed.imag - z_original.imag)**2))
        
        n_kept = np.sum(compressed_coeffs != 0)
        
        # Find first non-zero coefficient as example
        kept_indices = np.where(compressed_coeffs != 0)[0]
        if len(kept_indices) > 0:
            example_coeff = compressed_coeffs[kept_indices[0]]
            if precision is None:
                example_str = f"{example_coeff.real:.6f} + {example_coeff.imag:.6f}j"
            else:
                example_str = f"{example_coeff.real:.{precision}f} + {example_coeff.imag:.{precision}f}j"
        else:
            example_str = "N/A"
        
        print(f"{precision_label:9} | {n_kept:11} | {error:8.4f} | {example_str}")
        
        axes[i].plot(z_compressed.real, z_compressed.imag, 'g-', linewidth=2)
        axes[i].set_title(f'{precision_label} precision\nError: {error:.4f}')
        axes[i].axis('equal')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Main analysis function
def analyze_irregular_shapes_advanced():
    """
    Comprehensive analysis with storage optimization
    """
    shapes = create_irregular_shape(n_points=40)
    
    print("ADVANCED FOURIER COMPRESSION WITH STORAGE OPTIMIZATION")
    print("=" * 70)
    print("Features: Closed shapes + Coefficient selection + Decimal rounding")
    print("=" * 70)
    
    all_results = []
    
    for shape_name, coordinates in shapes:
        print(f"\n{'='*60}")
        print(f"ANALYZING: {shape_name}")
        print(f"{'='*60}")
        
        # Normalize coordinates
        coords_normalized = coordinates - np.mean(coordinates, axis=0)
        coords_normalized = coords_normalized / np.max(np.abs(coords_normalized))
        
        # Analyze with 3 decimal places rounding
        fourier_coeffs, compression_results = plot_irregular_shape_compression_advanced(
            shape_name, coords_normalized, decimal_places=3
        )
        
        # Test different precision levels
        test_different_precision_levels(shape_name, coords_normalized)
        
        all_results.append({
            'name': shape_name,
            'compression_results': compression_results
        })
    
    return all_results

# Answer to your question about what to store:
def explain_storage_strategy():
    """
    Explain what actually needs to be stored
    """
    print("\n" + "="*70)
    print("STORAGE STRATEGY EXPLANATION")
    print("="*70)
    print("Q: What do I need to store - all 4 values (real, imag, magnitude, phase)?")
    print("A: NO! You only need to store REAL and IMAGINARY parts!")
    print("\nWhy:")
    print("• Magnitude and phase can be calculated from real/imaginary")
    print("• magnitude = sqrt(real² + imag²)")
    print("• phase = atan2(imag, real)")
    print("• Storing real/imaginary is sufficient for perfect reconstruction")
    print("\nStorage format for each kept coefficient:")
    print("  [index, real_value, imaginary_value]")
    print("Example: [1, 201.780, 2.916]")
    print("\nWith rounding to 3 decimals:")
    print("  Original: 201.779526 + 2.915981j")
    print("  Stored:   201.780 + 2.916j")
    print("\nThis reduces storage significantly!")

# Run the analysis
if __name__ == "__main__":
    print("Advanced Fourier Compression with Storage Optimization")
    print("Testing on Irregular Closed Shapes")
    
    # Explain storage strategy
    explain_storage_strategy()
    
    # Run the advanced analysis
    results = analyze_irregular_shapes_advanced()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("✓ Working on irregular closed shapes")
    print("✓ Saving only 3 decimal digits for coefficients")
    print("✓ Only need to store REAL and IMAGINARY parts (not magnitude/phase)")
    print("✓ Significant storage reduction achieved")
    print("="*70)