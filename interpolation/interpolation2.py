import numpy as np
from scipy.interpolate import splprep, splev, BSpline, Rbf
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt

class ShapeInterpolator:
    def __init__(self, coordinates):
        """
        Initialize with list of coordinates [(x1,y1), (x2,y2), ...]
        """
        self.coords = np.array(coordinates)
        self.x = self.coords[:, 0]
        self.y = self.coords[:, 1]
        self.n_points = len(coordinates)
        
        # Parameter t (0 to 1)
        self.t = np.linspace(0, 1, self.n_points)
    
    # 1. PARAMETRIC POLYNOMIAL INTERPOLATION
    def parametric_polynomial(self, degree=None):
        """
        Parametric polynomial interpolation
        Returns coefficients for x(t) and y(t) polynomials
        """
        if degree is None:
            degree = min(10, self.n_points - 1)
        
        # Fit polynomials x(t) and y(t)
        x_poly = np.polyfit(self.t, self.x, degree)
        y_poly = np.polyfit(self.t, self.y, degree)
        
        return {
            'method': 'parametric_polynomial',
            'x_coeffs': x_poly,  # Highest degree first
            'y_coeffs': y_poly,
            'degree': degree
        }
    
    def evaluate_polynomial(self, coeffs, t_values):
        """Evaluate polynomial interpolation"""
        x_vals = np.polyval(coeffs['x_coeffs'], t_values)
        y_vals = np.polyval(coeffs['y_coeffs'], t_values)
        return x_vals, y_vals
    
    # 2. SPLINE METHODS
    def parametric_spline(self, s=0, k=3):
        """
        Parametric cubic spline interpolation
        s: smoothing factor (0 for exact interpolation)
        k: spline degree (3 for cubic)
        """
        tck, u = splprep([self.x, self.y], s=s, k=k, per=0)
        
        return {
            'method': 'parametric_spline',
            'tck': tck,  # (knots, coefficients, degree)
            'parameter': u
        }
    
    def evaluate_spline(self, coeffs, t_values):
        """Evaluate spline interpolation"""
        x_vals, y_vals = splev(t_values, coeffs['tck'])
        return x_vals, y_vals
    
    # 3. B-SPLINE
    def b_spline(self, degree=3, n_control=20):
        """
        B-spline interpolation
        """
        tck, u = splprep([self.x, self.y], s=0, k=degree)
        
        # Create B-spline object
        spline_x = BSpline(tck[0], tck[1][0], tck[2])
        spline_y = BSpline(tck[0], tck[1][1], tck[2])
        
        return {
            'method': 'b_spline',
            'spline_x': spline_x,
            'spline_y': spline_y,
            'knots': tck[0],
            'coeffs_x': tck[1][0],
            'coeffs_y': tck[1][1],
            'degree': tck[2]
        }
    
    def evaluate_bspline(self, coeffs, t_values):
        """Evaluate B-spline"""
        x_vals = coeffs['spline_x'](t_values)
        y_vals = coeffs['spline_y'](t_values)
        return x_vals, y_vals
    
    # 4. FOURIER DESCRIPTORS
    def fourier_descriptors(self, n_coeffs=None):
        """
        Fourier descriptors for closed contours
        """
        if n_coeffs is None:
            n_coeffs = self.n_points
        
        # Convert to complex numbers
        z = self.x + 1j * self.y
        
        # Ensure closed contour
        if not np.allclose(z[0], z[-1]):
            z = np.append(z, z[0])
            t_closed = np.linspace(0, 1, len(z))
        else:
            t_closed = self.t
        
        # Fourier transform
        coeffs = fft(z)
        
        return {
            'method': 'fourier_descriptors',
            'coefficients': coeffs,
            'n_coeffs': len(coeffs),
            'is_closed': True
        }
    
    def evaluate_fourier(self, coeffs, t_values, n_coeffs=None):
        """Evaluate Fourier reconstruction"""
        if n_coeffs is None:
            n_coeffs = len(coeffs['coefficients'])
        
        # Use all coefficients for maximum accuracy
        z_reconstructed = ifft(coeffs['coefficients'])
        
        # Interpolate to desired t_values
        from scipy.interpolate import interp1d
        t_original = np.linspace(0, 1, len(z_reconstructed))
        
        f_x = interp1d(t_original, z_reconstructed.real, kind='cubic')
        f_y = interp1d(t_original, z_reconstructed.imag, kind='cubic')
        
        x_vals = f_x(t_values)
        y_vals = f_y(t_values)
        
        return x_vals, y_vals
    
    # 5. RADIAL BASIS FUNCTIONS (RBF)
    def radial_basis(self, function='thin_plate'):
        """
        Radial Basis Function interpolation
        """
        # For RBF, we need to handle x and y separately
        rbf_x = Rbf(self.t, self.x, function=function)
        rbf_y = Rbf(self.t, self.y, function=function)
        
        return {
            'method': 'radial_basis',
            'rbf_x': rbf_x,
            'rbf_y': rbf_y,
            'function': function
        }
    
    def evaluate_rbf(self, coeffs, t_values):
        """Evaluate RBF interpolation"""
        x_vals = coeffs['rbf_x'](t_values)
        y_vals = coeffs['rbf_y'](t_values)
        return x_vals, y_vals
    
    # 6. BEZIER CURVE
    def bezier_curve(self):
        """
        Bezier curve through all points (high degree)
        """
        from scipy.special import comb
        
        n = self.n_points - 1
        t = self.t
        
        # Bernstein polynomials
        bezier_x = np.zeros_like(t)
        bezier_y = np.zeros_like(t)
        
        for i in range(self.n_points):
            binom = comb(n, i)
            bernstein = binom * (t**i) * ((1 - t)**(n - i))
            bezier_x += self.x[i] * bernstein
            bezier_y += self.y[i] * bernstein
        
        # Store control points (all original points)
        return {
            'method': 'bezier_curve',
            'control_points': self.coords,
            'degree': n
        }
    
    def evaluate_bezier(self, coeffs, t_values):
        """Evaluate Bezier curve"""
        from scipy.special import comb
        
        n = coeffs['degree']
        control_points = coeffs['control_points']
        
        x_vals = np.zeros_like(t_values)
        y_vals = np.zeros_like(t_values)
        
        for i in range(len(control_points)):
            binom = comb(n, i)
            bernstein = binom * (t_values**i) * ((1 - t_values)**(n - i))
            x_vals += control_points[i, 0] * bernstein
            y_vals += control_points[i, 1] * bernstein
        
        return x_vals, y_vals

# USAGE EXAMPLE
def test_all_methods():
    # Create a test shape (irregular star)
    theta = np.linspace(0, 2*np.pi, 50, endpoint=False)
    x = np.cos(theta) * (1 + 0.3 * np.cos(5*theta) + 0.1 * np.random.normal(size=50))
    y = np.sin(theta) * (1 + 0.3 * np.cos(5*theta) + 0.1 * np.random.normal(size=50))
    coordinates = list(zip(x, y))
    
    # Initialize interpolator
    interpolator = ShapeInterpolator(coordinates)
    
    # Test points for evaluation
    t_test = np.linspace(0, 1, 200)
    
    # Apply all methods
    methods = {}
    
    print("=== SHAPE INTERPOLATION METHODS ===")
    
    # 1. Polynomial
    print("1. Computing parametric polynomial...")
    methods['polynomial'] = interpolator.parametric_polynomial()
    
    # 2. Spline
    print("2. Computing parametric spline...")
    methods['spline'] = interpolator.parametric_spline()
    
    # 3. B-spline
    print("3. Computing B-spline...")
    methods['bspline'] = interpolator.b_spline()
    
    # 4. Fourier
    print("4. Computing Fourier descriptors...")
    methods['fourier'] = interpolator.fourier_descriptors()
    
    # 5. RBF
    print("5. Computing radial basis functions...")
    methods['rbf'] = interpolator.radial_basis()
    
    # 6. Bezier
    print("6. Computing Bezier curve...")
    methods['bezier'] = interpolator.bezier_curve()
    
    # Evaluate all methods
    results = {}
    for name, coeffs in methods.items():
        print(f"Evaluating {name}...")
        if name == 'polynomial':
            x_eval, y_eval = interpolator.evaluate_polynomial(coeffs, t_test)
        elif name == 'spline':
            x_eval, y_eval = interpolator.evaluate_spline(coeffs, t_test)
        elif name == 'bspline':
            x_eval, y_eval = interpolator.evaluate_bspline(coeffs, t_test)
        elif name == 'fourier':
            x_eval, y_eval = interpolator.evaluate_fourier(coeffs, t_test)
        elif name == 'rbf':
            x_eval, y_eval = interpolator.evaluate_rbf(coeffs, t_test)
        elif name == 'bezier':
            x_eval, y_eval = interpolator.evaluate_bezier(coeffs, t_test)
        
        results[name] = (x_eval, y_eval)
    
    return methods, results, coordinates, t_test

def plot_comparison(methods, results, original_coords, t_test):
    """Plot comparison of all methods"""
    original_x, original_y = zip(*original_coords)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    methods_list = list(methods.keys())
    
    for i, method in enumerate(methods_list):
        x_eval, y_eval = results[method]
        
        axes[i].plot(original_x, original_y, 'ro', markersize=4, alpha=0.7, label='Original')
        axes[i].plot(x_eval, y_eval, 'b-', linewidth=2, label='Interpolated')
        axes[i].set_title(f'{method.upper()} Interpolation')
        axes[i].legend()
        axes[i].axis('equal')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()













def main():
    """Comprehensive test with detailed coefficient analysis"""
    # Create a test shape (irregular star)
    theta = np.linspace(0, 2*np.pi, 200, endpoint=False)  # Fewer points for clarity
    x = np.cos(theta) * (1 + 0.3 * np.cos(5*theta) + 0.05 * np.random.normal(size=200))
    y = np.sin(theta) * (1 + 0.3 * np.cos(5*theta) + 0.05 * np.random.normal(size=200))
    coordinates = list(zip(x, y))
    
    print("=" * 70)
    print("SHAPE INTERPOLATION COEFFICIENT ANALYSIS")
    print("=" * 70)
    print(f"Original shape: {len(coordinates)} points")
    print(f"Points: {coordinates}")
    print()
    
    # Initialize interpolator
    interpolator = ShapeInterpolator(coordinates)
    
    # Test points for evaluation
    t_test = np.linspace(0, 1, 200)
    
    # Apply all methods
    methods = {}
    
    print("COMPUTING INTERPOLATION METHODS...")
    print("-" * 50)
    
    # 1. Polynomial
    print("1. PARAMETRIC POLYNOMIAL")
    methods['polynomial'] = interpolator.parametric_polynomial(degree=8)
    coeffs = methods['polynomial']
    print(f"   • Degree: {coeffs['degree']}")
    print(f"   • X coefficients: {len(coeffs['x_coeffs'])}")
    print(f"   • Y coefficients: {len(coeffs['y_coeffs'])}")
    print(f"   • Total coefficients: {len(coeffs['x_coeffs']) + len(coeffs['y_coeffs'])}")
    print(f"   • Compression ratio: {(len(coeffs['x_coeffs']) + len(coeffs['y_coeffs'])) / (2 * len(coordinates)):.1%}")
    print(f"   • First 3 X coefficients: {[f'{c:.6f}' for c in coeffs['x_coeffs'][:3]]}...")
    print(f"   • First 3 Y coefficients: {[f'{c:.6f}' for c in coeffs['y_coeffs'][:3]]}...")
    print()
    
    # 2. Spline
    print("2. PARAMETRIC SPLINE")
    methods['spline'] = interpolator.parametric_spline()
    coeffs = methods['spline']
    knots, ctrl_points, degree = coeffs['tck']
    print(f"   • Degree: {degree}")
    print(f"   • Number of knots: {len(knots)}")
    print(f"   • Control points for X: {len(ctrl_points[0])}")
    print(f"   • Control points for Y: {len(ctrl_points[1])}")
    print(f"   • Total storage elements: {len(knots) + len(ctrl_points[0]) + len(ctrl_points[1]) + 1}")
    print(f"   • Compression ratio: {(len(knots) + len(ctrl_points[0]) + len(ctrl_points[1]) + 1) / (2 * len(coordinates)):.1%}")
    print(f"   • First 3 knots: {[f'{k:.3f}' for k in knots[:3]]}...")
    print(f"   • First 3 X control points: {[f'{c:.3f}' for c in ctrl_points[0][:3]]}...")
    print()
    
    # 3. B-spline
    print("3. B-SPLINE")
    methods['bspline'] = interpolator.b_spline()
    coeffs = methods['bspline']
    print(f"   • Degree: {coeffs['degree']}")
    print(f"   • Number of knots: {len(coeffs['knots'])}")
    print(f"   • X coefficients: {len(coeffs['coeffs_x'])}")
    print(f"   • Y coefficients: {len(coeffs['coeffs_y'])}")
    print(f"   • Total coefficients: {len(coeffs['knots']) + len(coeffs['coeffs_x']) + len(coeffs['coeffs_y'])}")
    print(f"   • Compression ratio: {(len(coeffs['knots']) + len(coeffs['coeffs_x']) + len(coeffs['coeffs_y'])) / (2 * len(coordinates)):.1%}")
    print(f"   • First 3 X coefficients: {[f'{c:.3f}' for c in coeffs['coeffs_x'][:3]]}...")
    print()
    
    # 4. Fourier
    print("4. FOURIER DESCRIPTORS")
    methods['fourier'] = interpolator.fourier_descriptors()
    coeffs = methods['fourier']
    print(f"   • Number of coefficients: {coeffs['n_coeffs']}")
    print(f"   • Is closed contour: {coeffs['is_closed']}")
    print(f"   • Compression ratio: {coeffs['n_coeffs'] / len(coordinates):.1%}")  # Complex numbers count as 2
    magnitudes = np.abs(coeffs['coefficients'])
    print(f"   • Most important coefficients (by magnitude):")
    important_indices = np.argsort(magnitudes)[-5:][::-1]  # Top 5
    for idx in important_indices:
        print(f"     [{idx}]: {coeffs['coefficients'][idx]:.3f} (mag: {magnitudes[idx]:.3f})")
    print()
    
    # 5. RBF
    print("5. RADIAL BASIS FUNCTIONS")
    methods['rbf'] = interpolator.radial_basis()
    coeffs = methods['rbf']
    print(f"   • Basis function: {coeffs['function']}")
    print(f"   • Storage: RBF functions (not simple coefficients)")
    print(f"   • Nodes: {len(coordinates)}")
    print("   • Note: RBF stores implicit functions, not explicit coefficients")
    print()
    
    # 6. Bezier
    print("6. BEZIER CURVE")
    methods['bezier'] = interpolator.bezier_curve()
    coeffs = methods['bezier']
    print(f"   • Degree: {coeffs['degree']}")
    print(f"   • Control points: {len(coeffs['control_points'])}")
    print(f"   • Compression ratio: {len(coeffs['control_points']) / len(coordinates):.1%}")
    print(f"   • First 3 control points:")
    for i, (cx, cy) in enumerate(coeffs['control_points'][:3]):
        print(f"     P{i}: ({cx:.3f}, {cy:.3f})")
    print()
    
    # Evaluate all methods and calculate errors
    print("ACCURACY ANALYSIS")
    print("-" * 50)
    
    results = {}
    errors = {}
    
    original_array = np.array(coordinates)
    
    for name, coeffs in methods.items():
        if name == 'polynomial':
            x_eval, y_eval = interpolator.evaluate_polynomial(coeffs, t_test)
        elif name == 'spline':
            x_eval, y_eval = interpolator.evaluate_spline(coeffs, t_test)
        elif name == 'bspline':
            x_eval, y_eval = interpolator.evaluate_bspline(coeffs, t_test)
        

        # In your main function, after Fourier computation:
        elif name == 'fourier':
            x_eval, y_eval = interpolator.evaluate_fourier(coeffs, t_test)
            
            # PRINT ALL FOURIER COEFFICIENTS
            print("\n   • ALL FOURIER COEFFICIENTS:")
            fourier_coeffs = coeffs['coefficients']
            for i in range(len(fourier_coeffs)):
                real_part = fourier_coeffs[i].real
                imag_part = fourier_coeffs[i].imag
                magnitude = np.abs(fourier_coeffs[i])
                phase = np.angle(fourier_coeffs[i])
                print(f"     c[{i:2d}]: {real_part:10.6f} + {imag_part:10.6f}j  | mag: {magnitude:10.6f}  | phase: {phase:7.3f}")


        elif name == 'rbf':
            x_eval, y_eval = interpolator.evaluate_rbf(coeffs, t_test)
        elif name == 'bezier':
            x_eval, y_eval = interpolator.evaluate_bezier(coeffs, t_test)
        
        results[name] = (x_eval, y_eval)
        
        # Calculate error at original points
        from scipy.interpolate import interp1d
        t_original = np.linspace(0, 1, len(coordinates))
        x_interp = interp1d(t_test, x_eval, kind='linear')(t_original)
        y_interp = interp1d(t_test, y_eval, kind='linear')(t_original)
        
        error = np.sqrt((x_interp - original_array[:, 0])**2 + (y_interp - original_array[:, 1])**2)
        errors[name] = {
            'max_error': np.max(error),
            'mean_error': np.mean(error),
            'rmse': np.sqrt(np.mean(error**2))
        }
    
    # Print error summary
    print("RECONSTRUCTION ERRORS (at original points):")
    print("Method            | Max Error | Mean Error | RMSE    ")
    print("-" * 55)
    for name in methods.keys():
        err = errors[name]
        print(f"{name:15} | {err['max_error']:8.4f}  | {err['mean_error']:9.4f} | {err['rmse']:6.4f}")
    
    print()
    
    # Summary table
    print("METHOD SUMMARY")
    print("-" * 70)
    print("Method            | Coeffs | Storage | Compress | Max Error")
    print("-" * 70)
    for name in methods.keys():
        coeffs = methods[name]
        err = errors[name]
        
        if name == 'polynomial':
            storage = len(coeffs['x_coeffs']) + len(coeffs['y_coeffs'])
            compress = storage / (2 * len(coordinates))
        elif name == 'spline':
            knots, ctrl, deg = coeffs['tck']
            storage = len(knots) + len(ctrl[0]) + len(ctrl[1])
            compress = storage / (2 * len(coordinates))
        elif name == 'bspline':
            storage = len(coeffs['knots']) + len(coeffs['coeffs_x']) + len(coeffs['coeffs_y'])
            compress = storage / (2 * len(coordinates))
        elif name == 'fourier':
            storage = len(coeffs['coefficients'])
            compress = storage / len(coordinates)  # Complex numbers
        elif name == 'rbf':
            storage = len(coordinates)  # RBF nodes
            compress = 1.0  # No compression
        elif name == 'bezier':
            storage = len(coeffs['control_points'])
            compress = storage / len(coordinates)
        
        print(f"{name:15} | {storage:6} | {storage:7} | {compress:7.1%} | {err['max_error']:8.4f}")
    
    return methods, results, coordinates, t_test, errors

def plot_detailed_comparison(methods, results, original_coords, errors):
    """Plot detailed comparison with error information"""
    original_x, original_y = zip(*original_coords)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    methods_list = list(methods.keys())
    
    for i, method in enumerate(methods_list):
        x_eval, y_eval = results[method]
        err = errors[method]
        
        axes[i].plot(original_x, original_y, 'ro', markersize=6, alpha=0.8, label='Original', markeredgecolor='black')
        axes[i].plot(x_eval, y_eval, 'b-', linewidth=2, label='Interpolated', alpha=0.8)
        axes[i].set_title(f'{method.upper()}\nMax Error: {err["max_error"]:.4f}, Mean: {err["mean_error"]:.4f}', 
                         fontweight='bold', fontsize=12)
        axes[i].legend()
        axes[i].axis('equal')
        axes[i].grid(True, alpha=0.3)
        
        # Add coefficient count to plot
        coeffs = methods[method]
        if method == 'polynomial':
            coeff_count = len(coeffs['x_coeffs']) + len(coeffs['y_coeffs'])
        elif method == 'spline':
            knots, ctrl, deg = coeffs['tck']
            coeff_count = len(knots) + len(ctrl[0]) + len(ctrl[1])
        elif method == 'bspline':
            coeff_count = len(coeffs['knots']) + len(coeffs['coeffs_x']) + len(coeffs['coeffs_y'])
        elif method == 'fourier':
            coeff_count = len(coeffs['coefficients'])
        elif method == 'rbf':
            coeff_count = len(original_coords)
        elif method == 'bezier':
            coeff_count = len(coeffs['control_points'])
        
        axes[i].text(0.02, 0.98, f'Coefficients: {coeff_count}', 
                    transform=axes[i].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

"""# Run the comprehensive analysis
if __name__ == "__main__":
    methods, results, original_coords, t_test, errors = main()
    plot_detailed_comparison(methods, results, original_coords, errors)
    
    # Final recommendations
    print("\n" + "=" * 70)
    print("FINAL RECOMMENDATIONS")
    print("=" * 70)
    
    # Find best method by error
    best_error = min(errors.items(), key=lambda x: x[1]['max_error'])
    best_compression = min(methods.items(), key=lambda x: 
        len(x[1]['x_coeffs']) + len(x[1]['y_coeffs']) if 'x_coeffs' in x[1] else float('inf'))
    
    print(f"• Most accurate: {best_error[0]} (max error: {best_error[1]['max_error']:.4f})")
    print(f"• Best compression: parametric_polynomial")
    print("• Recommended for general use: SPLINE (good balance of accuracy and simplicity)")
    print("• Recommended for closed shapes: FOURIER (excellent compression)")
    print("• Recommended for very irregular shapes: RBF (most flexible)")"""

















import matplotlib.pyplot as plt
import numpy as np

def plot_fourier_compression_comparison(original_coeffs, original_coords, test_ratios=[0.05, 0.1, 0.2, 0.5]):
    """
    Plot original shape vs compressed reconstructions for different ratios
    """
    # Reconstruct original shape from full coefficients
    z_original = np.fft.ifft(original_coeffs)
    x_original = z_original.real
    y_original = z_original.imag
    
    # Create subplot grid - FIXED: Calculate proper grid size
    n_plots = len(test_ratios) + 1  # +1 for original
    n_cols = min(3, n_plots)  # Max 3 columns
    n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    # Handle single subplot case
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot original
    axes[0].plot(x_original, y_original, 'b-', linewidth=2, label='Original')
    if original_coords is not None:
        axes[0].plot(original_coords[:, 0], original_coords[:, 1], 'ro', markersize=3, alpha=0.6)
    axes[0].set_title(f'Original Shape\n{len(original_coeffs)} coefficients')
    axes[0].legend()
    axes[0].axis('equal')
    axes[0].grid(True, alpha=0.3)
    
    # Plot compressed versions - FIXED: Use correct indexing
    for idx, ratio in enumerate(test_ratios):
        i = idx + 1  # Start from index 1 (0 is original)
        
        # Compress
        compressed_coeffs = compress_fourier(original_coeffs, keep_ratio=ratio)
        
        # Reconstruct
        z_compressed = np.fft.ifft(compressed_coeffs)
        x_compressed = z_compressed.real
        y_compressed = z_compressed.imag
        
        # Calculate how many coefficients were kept
        n_kept = np.sum(compressed_coeffs != 0)
        
        # Calculate error
        error = np.mean(np.sqrt((x_compressed - x_original)**2 + (y_compressed - y_original)**2))
        
        # Plot
        axes[i].plot(x_compressed, y_compressed, 'g-', linewidth=2, label='Compressed')
        axes[i].plot(x_original, y_original, 'b--', linewidth=1, alpha=0.5, label='Original')
        if original_coords is not None:
            axes[i].plot(original_coords[:, 0], original_coords[:, 1], 'ro', markersize=3, alpha=0.6)
        axes[i].set_title(f'Ratio: {ratio:.0%}\n{n_kept}/{len(original_coeffs)} coeffs\nError: {error:.4f}')
        axes[i].legend()
        axes[i].axis('equal')
        axes[i].grid(True, alpha=0.3)
    
    # Hide empty subplots if any - FIXED: Correct range
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def compress_fourier(coeffs, keep_ratio=0.1):
    """
    Compress by keeping only the largest coefficients
    """
    magnitudes = np.abs(coeffs)
    
    # How many coefficients to keep
    n_keep = max(1, int(len(coeffs) * keep_ratio))
    
    # Find indices of largest coefficients
    largest_indices = np.argsort(magnitudes)[-n_keep:]
    
    # Create compressed coefficients (zeros for small ones)
    compressed = np.zeros_like(coeffs)
    compressed[largest_indices] = coeffs[largest_indices]
    
    return compressed

# Simplified test function
def test_compression_visualization():
    """
    Test the compression visualization with your Fourier coefficients
    """
    num_points=200
    # Create a sample shape for testing
    theta = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    x = np.cos(theta) * (1 + 0.3 * np.cos(5*theta))
    y = np.sin(theta) * (1 + 0.3 * np.cos(5*theta))
    coordinates = np.column_stack([x, y])
    
    # Get Fourier coefficients
    z_points = coordinates[:, 0] + 1j * coordinates[:, 1]
    fourier_coeffs = np.fft.fft(z_points)
    
    print("Fourier Compression Visualization")
    print("=" * num_points)
    print(f"Original shape: {len(coordinates)} points")
    print(f"Fourier coefficients: {len(fourier_coeffs)}")
    
    # Show coefficient magnitudes for reference
    magnitudes = np.abs(fourier_coeffs)
    print(f"\nTop 5 coefficients by magnitude:")
    top_indices = np.argsort(magnitudes)[-5:][::-1]
    for idx in top_indices:
        print(f"  c[{idx}]: magnitude = {magnitudes[idx]:.3f}")
    
    # Create the comparison plots
    plot_fourier_compression_comparison(fourier_coeffs, coordinates)
    
    return fourier_coeffs

# Even simpler version - single row plot
def plot_simple_compression_comparison(original_coeffs, original_coords, test_ratios=[0.05, 0.1, 0.2, 0.5]):
    """
    Simple one-row comparison that definitely works
    """
    z_original = np.fft.ifft(original_coeffs)
    
    fig, axes = plt.subplots(1, len(test_ratios) + 1, figsize=(5*(len(test_ratios)+1), 5))
    
    # Make sure axes is always iterable
    if len(test_ratios) + 1 == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Plot original
    axes[0].plot(z_original.real, z_original.imag, 'b-', linewidth=2)
    axes[0].set_title(f'Original\n{len(original_coeffs)} coeffs')
    axes[0].axis('equal')
    axes[0].grid(True, alpha=0.3)
    
    # Plot compressed versions
    for i, ratio in enumerate(test_ratios, 1):
        compressed_coeffs = compress_fourier(original_coeffs, keep_ratio=ratio)
        z_compressed = np.fft.ifft(compressed_coeffs)
        
        axes[i].plot(z_compressed.real, z_compressed.imag, 'g-', linewidth=2)
        n_kept = np.sum(compressed_coeffs != 0)
        axes[i].set_title(f'{ratio:.0%} Compression\n{n_kept} coeffs')
        axes[i].axis('equal')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Run the visualization
if __name__ == "__main__":
    print("Testing Fourier compression...")
    
    # Use the simple version first to test
    theta = np.linspace(0, 2*np.pi, 50, endpoint=False)
    x = np.cos(theta) * (1 + 0.3 * np.cos(5*theta))
    y = np.sin(theta) * (1 + 0.3 * np.cos(5*theta))
    coordinates = np.column_stack([x, y])
    
    z_points = coordinates[:, 0] + 1j * coordinates[:, 1]
    fourier_coeffs = np.fft.fft(z_points)
    
    # Test with simple version first
    plot_simple_compression_comparison(fourier_coeffs, coordinates)
    
    # Then try the detailed version
    test_compression_visualization()