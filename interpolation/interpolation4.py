import numpy as np

# Example usage and testing
def test_compression_function():
    """
    Test the compression function with example shapes
    """
    # Create an irregular shape for testing
    np.random.seed(42)
    theta = np.linspace(0, 2*np.pi, 50, endpoint=False)
    r = 1 + 0.4 * np.random.normal(size=len(theta))
    from scipy.ndimage import gaussian_filter1d
    r = gaussian_filter1d(r, sigma=2)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    coordinates = list(zip(x, y))
    
    print("TESTING COMPRESSION FUNCTION")
    print("=" * 50)
    
    # Test different compression settings
    test_settings = [
        (0.05, 2),  # 5% coefficients, 2 decimal places
        (0.1, 3),   # 10% coefficients, 3 decimal places  
        (0.2, 4),   # 20% coefficients, 4 decimal places
    ]
    
    for compression_ratio, decimal_places in test_settings:
        print(f"\nCompression: {compression_ratio:.0%}, Decimals: {decimal_places}")
        print("-" * 40)
        
        # Compress the shape
        result = compress_irregular_shape(
            coordinates, 
            compression_ratio=compression_ratio,
            decimal_places=decimal_places
        )
        
        # Print results
        storage = result['storage_info']
        reconstruction = result['reconstruction_info']
        
        print(f"Original points: {storage['original_points']}")
        print(f"Compressed coefficients: {storage['compressed_coeffs_count']}")
        print(f"Storage reduction: {storage['storage_reduction']}")
        print(f"Reconstruction error: {reconstruction['mean_error']:.6f}")
        
        # Show first few coefficients
        print("First 3 coefficients stored:")
        for i, (idx, real, imag) in enumerate(result['compressed_coeffs'][:3]):
            print(f"  [{idx}]: {real:.{decimal_places}f} + {imag:.{decimal_places}f}j")
    
    return result












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

def plot_compression_results(original_coords, compressed_result, title="Compression Results"):
    """
    Plot original vs reconstructed shape - FIXED VERSION
    """
    # Reconstruct shape from compressed coefficients using the correct length
    reconstructed_coords = reconstruct_from_compressed(
        compressed_result['compressed_coeffs'],
        compressed_result['original_length']  # Use the stored original length!
    )
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot original
    original_array = np.array(original_coords)
    ax1.plot(original_array[:, 0], original_array[:, 1], 'b-o', linewidth=2, markersize=4, label='Original')
    ax1.set_title('Original Shape')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot reconstructed
    ax2.plot(reconstructed_coords[:, 0], reconstructed_coords[:, 1], 'r-o', linewidth=2, markersize=4, label='Reconstructed')
    ax2.set_title('Reconstructed from Compressed Coefficients')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    
    # Add compression info to plot
    storage = compressed_result['storage_info']
    reconstruction = compressed_result['reconstruction_info']
    info_text = f"Compression: {storage['compression_ratio']:.0%}\n"
    info_text += f"Coefficients: {storage['compressed_coeffs_count']}\n"
    info_text += f"Decimals: {storage['decimal_places']}\n"
    info_text += f"Mean Error: {reconstruction['mean_error']:.6f}"
    
    ax2.text(0.05, 0.95, info_text, transform=ax2.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def test_perfect_reconstruction():
    """
    Test that we can achieve perfect reconstruction with no compression
    """
    # Create test shape for visualization
    theta = np.linspace(0, 2*np.pi, 40, endpoint=False)
    x = np.cos(theta) * (1 + 0.3 * np.sin(3*theta))
    y = np.sin(theta) * (1 + 0.2 * np.cos(4*theta))
    test_coords = list(zip(x, y))
    
    print("Testing perfect reconstruction (compression_ratio=1.0)")
    print("=" * 50)
    
    # Test with no compression (keep 100% of coefficients)
    viz_result = compress_irregular_shape(test_coords, compression_ratio=1.0, decimal_places=10)
    
    print(f"Original points: {viz_result['storage_info']['original_points']}")
    print(f"Compressed coefficients: {viz_result['storage_info']['compressed_coeffs_count']}")
    print(f"Mean reconstruction error: {viz_result['reconstruction_info']['mean_error']:.10f}")
    print(f"Max reconstruction error: {viz_result['reconstruction_info']['max_error']:.10f}")
    
    # The error should be very close to zero with no compression
    if viz_result['reconstruction_info']['mean_error'] < 1e-10:
        print("✅ SUCCESS: Perfect reconstruction achieved!")
    else:
        print("❌ FAILED: Reconstruction error too high")
    
    plot_compression_results(test_coords, viz_result, "Perfect Reconstruction Test")

def test_progressive_compression():
    """
    Test different compression levels on the same shape
    """
    # Create test shape
    theta = np.linspace(0, 2*np.pi, 40, endpoint=False)
    x = np.cos(theta) * (1 + 0.3 * np.sin(3*theta))
    y = np.sin(theta) * (1 + 0.2 * np.cos(4*theta))
    test_coords = list(zip(x, y))
    
    print("\nTesting progressive compression")
    print("=" * 50)
    
    compression_levels = [1.0, 0.5, 0.2, 0.1, 0.05]
    
    for ratio in compression_levels:
        result = compress_irregular_shape(test_coords, compression_ratio=ratio, decimal_places=3)
        error = result['reconstruction_info']['mean_error']
        coeffs_count = result['storage_info']['compressed_coeffs_count']
        storage_reduction = result['storage_info']['storage_reduction']
        
        print(f"Ratio {ratio:.0%}: {coeffs_count:2d} coeffs, Error: {error:.6f}, Storage: {storage_reduction}")
        
        # Plot each compression level
        if ratio in [1.0, 0.2, 0.05]:  # Plot a few representative ones
            plot_compression_results(test_coords, result, f"Compression: {ratio:.0%}")

# Quick usage example
if __name__ == "__main__":
    # First test perfect reconstruction
    test_perfect_reconstruction()
    
    # Then test progressive compression
    test_progressive_compression()