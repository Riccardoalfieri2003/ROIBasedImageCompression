import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib as plt
import matplotlib.pyplot as plt

def compress_irregular_border(border_coordinates, method='fourier'):
    """
    Compress any irregular border from your (x,y) coordinate list
    """
    # Your border coordinates - can be any shape!
    # border_coordinates = [(x1,y1), (x2,y2), (x3,y3), ...]
    
    x_points = [coord[0] for coord in border_coordinates]
    y_points = [coord[1] for coord in border_coordinates]
    
    if method == 'fourier':
        return adaptive_fourier_compression(x_points, y_points)
    elif method == 'spline':
        return compress_with_spline(x_points, y_points)

def adaptive_fourier_compression(x_points, y_points, quality=0.95):
    """
    Automatically choose how many coefficients to keep based on shape complexity
    """
    z_points = np.array(x_points) + 1j * np.array(y_points)
    
    if not np.allclose(z_points[0], z_points[-1]):
        z_points = np.append(z_points, z_points[0])
    
    coeffs = np.fft.fft(z_points)
    
    # Calculate energy distribution
    energy = np.abs(coeffs) ** 2
    total_energy = np.sum(energy)
    cumulative_energy = np.cumsum(energy) / total_energy
    
    # Find how many coefficients needed to capture 'quality' of energy
    num_coeffs_needed = np.argmax(cumulative_energy >= quality) + 1
    
    # But don't use more than half the original points
    max_reasonable = min(len(x_points) // 2, num_coeffs_needed)
    num_coeffs = max(5, max_reasonable)  # At least 5 coefficients
    
    compressed_coeffs = coeffs[:num_coeffs]
    
    original_size = len(x_points) * 2
    compressed_size = num_coeffs * 2
    compression_ratio = compressed_size / original_size
    
    # Reconstruct
    full_coeffs = np.zeros_like(coeffs, dtype=complex)
    full_coeffs[:num_coeffs] = compressed_coeffs
    z_reconstructed = np.fft.ifft(full_coeffs)
    
    return {
        'coefficients': compressed_coeffs,
        'x_reconstructed': z_reconstructed.real,
        'y_reconstructed': z_reconstructed.imag,
        'num_coeffs_used': num_coeffs,
        'energy_captured': cumulative_energy[num_coeffs-1],
        'compression_ratio': compression_ratio,
        'savings': f"{(1-compression_ratio)*100:.1f}%"
    }

def compress_with_spline(x_points, y_points, num_points=100):
    """Spline compression for any shape"""
    tck, u = splprep([x_points, y_points], s=0, per=1)
    
    # Generate smooth border
    u_new = np.linspace(0, 1, num_points)
    x_new, y_new = splev(u_new, tck)
    
    return {
        'coefficients': tck,  # Spline coefficients
        'x_reconstructed': x_new,
        'y_reconstructed': y_new,
        'compression_ratio': len(tck[1]) / len(x_points)  # Approximate
    }


def analyze_stored_data(compressed_data, shape_name):
    """
    Examine the compressed coefficients
    """
    print(f"\n=== {shape_name} Storage Analysis ===")
    #print(f"Original points: {compressed_data['original_size'] / 2:.0f}")
    print(f"Stored coefficients: {len(compressed_data['coefficients'])}")
    print(f"Compression ratio: {compressed_data['compression_ratio']:.1%}")
    print(f"Storage savings: {(1 - compressed_data['compression_ratio'])*100:.1f}%")
    
    # Show the actual stored values
    coeffs = compressed_data['coefficients']
    print(f"\nFirst 5 stored coefficients (complex numbers):")
    for i in range(min(5, len(coeffs))):
        print(f"  Coefficient {i}: {coeffs[i]:.3f} {coeffs[i].real:+.3f} + {coeffs[i].imag:+.3f}j")
    
    # Show magnitude and phase
    print(f"\nMagnitude of coefficients:")
    magnitudes = np.abs(coeffs)
    for i in range(min(5, len(coeffs))):
        print(f"  Coeff {i}: |{magnitudes[i]:.3f}|")
    
    return coeffs


def uncompress_fourier(fourier_coeffs, num_points=100):
    """
    Reconstruct border from Fourier coefficients
    """
    # Create a full coefficient array with zeros for high frequencies
    full_coeffs = np.zeros(num_points, dtype=complex)
    full_coeffs[:len(fourier_coeffs)] = fourier_coeffs
    
    # Inverse Fourier transform
    z_reconstructed = np.fft.ifft(full_coeffs)
    
    return z_reconstructed.real, z_reconstructed.imag

def uncompress_and_visualize(compressed_data, original_x, original_y, title):
    """
    Complete uncompression and comparison
    """
    # Uncompress
    x_reconstructed, y_reconstructed = uncompress_fourier(compressed_data['coefficients'])
    
    # Plot comparison
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(original_x, original_y, 'ro-', markersize=4)
    plt.title(f'Original {title}\n{len(original_x)} points')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(x_reconstructed, y_reconstructed, 'b-', linewidth=2)
    plt.title(f'Reconstructed {title}\n{len(compressed_data["coefficients"])} coefficients')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    plt.plot(original_x, original_y, 'ro-', markersize=4, alpha=0.5, label='Original')
    plt.plot(x_reconstructed, y_reconstructed, 'b-', linewidth=2, label='Reconstructed')
    plt.title('Overlay Comparison')
    plt.axis('equal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return x_reconstructed, y_reconstructed


# Example 1: Star shape (definitely not a function!)
theta = np.linspace(0, 2*np.pi, 50, endpoint=False)
x_star = np.cos(theta) * (1 + 0.5 * np.cos(5*theta))
y_star = np.sin(theta) * (1 + 0.5 * np.cos(5*theta))

# Example 2: Random irregular shape (like your regions)
x_irregular = [10, 45, 80, 75, 60, 30, 10, 10, 25, 45]  # Back-and-forth in x!
y_irregular = [10, 5, 20, 50, 80, 90, 70, 40, 20, 10]   # Back-and-forth in y!

# Both work perfectly!
# Your compressed data
result_star = adaptive_fourier_compression(x_star, y_star, quality=1)
result_irregular = adaptive_fourier_compression(x_irregular, y_irregular, quality=1)

# Analyze what's stored
star_coeffs = analyze_stored_data(result_star, "Star")
irregular_coeffs = analyze_stored_data(result_irregular, "Irregular Region")

# Uncompress and visualize
print("\n=== RECONSTRUCTION ===")
star_x, star_y = uncompress_and_visualize(result_star, x_star, y_star, "Star")
irregular_x, irregular_y = uncompress_and_visualize(result_irregular, x_irregular, y_irregular, "Irregular Region")