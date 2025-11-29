import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

def divide_shape_smart(coordinates, num_sublists=3):
    """
    Divide shape smartly by arc length to maintain continuity
    """
    coords = np.array(coordinates)
    
    # Ensure closed shape
    if not np.allclose(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0]])
    
    # Calculate cumulative arc length
    arc_lengths = np.zeros(len(coords))
    for i in range(1, len(coords)):
        arc_lengths[i] = arc_lengths[i-1] + np.linalg.norm(coords[i] - coords[i-1])
    
    total_length = arc_lengths[-1]
    segment_length = total_length / num_sublists
    
    # Find division points by arc length
    division_indices = [0]  # Start with first point
    for i in range(1, num_sublists):
        target_length = i * segment_length
        # Find point closest to target arc length
        idx = np.argmin(np.abs(arc_lengths - target_length))
        division_indices.append(idx)
    division_indices.append(len(coords) - 1)  # End with last point
    
    # Create overlapping sublists for smooth transitions
    sublists = []
    for i in range(len(division_indices) - 1):
        start_idx = division_indices[i]
        end_idx = division_indices[i + 1] + 1  # Include endpoint
        
        # Add overlap for smooth transitions (except for first and last)
        if i > 0:
            start_idx = max(0, start_idx - 2)  # Overlap with previous
        if i < len(division_indices) - 2:
            end_idx = min(len(coords), end_idx + 2)  # Overlap with next
        
        sublist = coords[start_idx:end_idx]
        sublists.append(sublist.tolist())
    
    print(f"📊 Divided {len(coords)} points into {len(sublists)} sublists by arc length")
    for i, sublist in enumerate(sublists):
        print(f"  Sublist {i+1}: {len(sublist)} points")
    
    return sublists, division_indices

def compress_sublist_with_continuity(sublist_coordinates, sublist_index, total_sublists, compression_ratio=0.2):
    """
    Fixed version: Properly handles compression_ratio = 1.0
    """
    coords = np.array(sublist_coordinates)
    
    # CRITICAL FIX: If compression_ratio is 1.0, use ALL points
    if compression_ratio >= 1.0:
        n_key_points = len(coords)
        key_indices = np.arange(len(coords))
    else:
        # Adjust compression based on position
        if sublist_index == 0 or sublist_index == total_sublists - 1:
            adjusted_ratio = compression_ratio * 0.8
        else:
            adjusted_ratio = compression_ratio
        
        n_key_points = max(4, int(len(coords) * adjusted_ratio))
        
        # Select key points with emphasis on segment boundaries
        if len(coords) <= n_key_points:
            key_indices = np.arange(len(coords))
        else:
            # Always include first and last points for continuity
            key_indices = [0, len(coords)-1]
            
            # Add interior points based on curvature
            remaining_slots = n_key_points - 2
            if remaining_slots > 0:
                curvature = np.zeros(len(coords))
                for i in range(1, len(coords)-1):
                    v1 = coords[i] - coords[i-1]
                    v2 = coords[i+1] - coords[i]
                    if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        cos_angle = np.clip(cos_angle, -1, 1)
                        curvature[i] = np.arccos(cos_angle)
                
                interior_indices = list(range(1, len(coords)-1))
                high_curve_indices = np.argsort(curvature[1:-1])[-remaining_slots:][::-1]
                key_indices.extend([i+1 for i in high_curve_indices])
            
            """# Instead of just curvature, consider even spacing
            if remaining_slots > 0:
                # Mix curvature and even spacing
                curvature_slots = remaining_slots // 2
                spacing_slots = remaining_slots - curvature_slots
                
                # High curvature points
                high_curve_indices = np.argsort(curvature[1:-1])[-curvature_slots:][::-1]
                
                # Evenly spaced points
                spacing_indices = np.linspace(1, len(coords)-2, spacing_slots, dtype=int)
                
                key_indices.extend([i+1 for i in high_curve_indices])
                key_indices.extend(spacing_indices)"""
            
            key_indices = sorted(key_indices)
    
    key_points = coords[key_indices]
    
    # DEBUG: Print key point info
    print(f"   DEBUG: Original points: {len(coords)}, Key points: {len(key_points)}, Ratio: {compression_ratio}")
    
    # Fit parametric spline
    x_key = key_points[:, 0]
    y_key = key_points[:, 1]
    
    try:
        # Use small smoothing only if we're compressing
        #smoothing = 0 if len(key_points) == len(coords) else len(key_points) * 0.001
        #tck, u = splprep([x_key, y_key], s=smoothing, per=0)

        # Use stricter smoothing and degree control
        smoothing = max(1.0, len(key_points) * 0.1)  # Increased smoothing
        tck, u = splprep([x_key, y_key], s=smoothing, per=0, k=min(3, len(key_points)-1))
        #tck, u = splprep([x_key, y_key], s=0, per=0, k=min(3, len(key_points)-1))
        
        # Reconstruct with same number of points as original sublist
        num_recon_points = len(coords)
        t_new = np.linspace(0, 1, num_recon_points)
        x_new, y_new = splev(t_new, tck)
        
        reconstructed = np.column_stack([x_new, y_new])
        
        # Calculate error
        t_original = np.linspace(0, 1, len(coords))
        x_orig_interp, y_orig_interp = splev(t_original, tck)
        error = np.mean(np.sqrt((x_orig_interp - coords[:, 0])**2 + 
                               (y_orig_interp - coords[:, 1])**2))
        
        return {
            'reconstructed': reconstructed,
            'spline_coefficients': tck,
            'key_points': key_points,
            'key_indices': key_indices,
            'metrics': {
                'mean_error': error,
                'compression_ratio': compression_ratio,
                'original_points': len(coords),
                'key_points_used': len(key_points),
                'storage_reduction': f"{(1 - len(key_points)/len(coords))*100:.1f}%"
            }
        }
    except Exception as e:
        print(f"❌ Sublist {sublist_index + 1} compression failed: {e}")
        return None



def combine_sublists_smoothly(sublist_results):
    """
    Combine sublist reconstructions smoothly by removing overlaps
    """
    if not sublist_results:
        return None
    
    all_reconstructed = []
    
    for i, result in enumerate(sublist_results):
        reconstructed = result['reconstructed']
        
        if i == 0:
            # First sublist: take all points
            all_reconstructed.append(reconstructed)
        else:
            # Subsequent sublists: remove overlapping points (take from middle to end)
            # Remove first 20% to avoid overlap with previous segment
            start_idx = max(1, len(reconstructed) // 5)
            all_reconstructed.append(reconstructed[start_idx:])
    
    combined = np.vstack(all_reconstructed)
    return combined

def divide_shape_smart_fixed(coordinates, num_sublists=3):
    """
    Fixed division that ensures proper point counting
    """
    coords = np.array(coordinates)
    
    # Ensure closed shape and count properly
    original_length = len(coords)
    if not np.allclose(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0]])
        print(f"🔒 Closed shape: {original_length} → {len(coords)} points")
    
    # Calculate cumulative arc length
    arc_lengths = np.zeros(len(coords))
    for i in range(1, len(coords)):
        arc_lengths[i] = arc_lengths[i-1] + np.linalg.norm(coords[i] - coords[i-1])
    
    total_length = arc_lengths[-1]
    segment_length = total_length / num_sublists
    
    # Find division points by arc length
    division_indices = [0]
    for i in range(1, num_sublists):
        target_length = i * segment_length
        idx = np.argmin(np.abs(arc_lengths - target_length))
        division_indices.append(idx)
    division_indices.append(len(coords) - 1)
    
    # Create sublists with exact point accounting
    sublists = []
    total_points_in_sublists = 0
    
    for i in range(len(division_indices) - 1):
        start_idx = division_indices[i]
        end_idx = division_indices[i + 1] + 1
        
        # Add overlap for smooth transitions
        if i > 0:
            start_idx = max(0, start_idx - 2)
        if i < len(division_indices) - 2:
            end_idx = min(len(coords), end_idx + 2)
        
        sublist = coords[start_idx:end_idx]
        sublists.append(sublist.tolist())
        total_points_in_sublists += len(sublist)
    
    print(f"📊 Division summary:")
    print(f"  Original points: {len(coords)}")
    print(f"  Sublists created: {len(sublists)}")
    print(f"  Total points in sublists: {total_points_in_sublists}")
    print(f"  Overlap points: {total_points_in_sublists - len(coords)}")
    
    for i, sublist in enumerate(sublists):
        print(f"  Sublist {i+1}: {len(sublist)} points")
    
    return sublists, division_indices



def compress_shape_single_fallback(coordinates, compression_ratio=0.2):
    """
    Fallback: Use single spline for the whole shape
    """
    print("🔄 Falling back to single spline compression...")
    
    coords = np.array(coordinates)
    if not np.allclose(coords[0], coords[-1]):
        coords = np.vstack([coords, coords[0]])
    
    # Simple compression with single spline
    n_key_points = max(4, int(len(coords) * compression_ratio))
    key_indices = np.linspace(0, len(coords)-1, n_key_points, dtype=int)
    key_points = coords[key_indices]
    
    x_key = key_points[:, 0]
    y_key = key_points[:, 1]
    
    tck, u = splprep([x_key, y_key], s=0, per=1)
    
    t_new = np.linspace(0, 1, len(coords))
    x_new, y_new = splev(t_new, tck)
    
    error = np.mean(np.sqrt((x_new - coords[:, 0])**2 + (y_new - coords[:, 1])**2))
    
    # Create single sublist result structure
    single_result = {
        'reconstructed': np.column_stack([x_new, y_new]),
        'spline_coefficients': tck,
        'key_points': key_points,
        'metrics': {
            'mean_error': error,
            'compression_ratio': compression_ratio,
            'original_points': len(coords),
            'key_points_used': n_key_points,
            'storage_reduction': f"{(1 - n_key_points/len(coords))*100:.1f}%"
        }
    }
    
    return {
        'sublist_results': [single_result],
        'combined_reconstructed': np.column_stack([x_new, y_new]),
        'overall_metrics': {
            'mean_error': error,
            'total_original_points': len(coords),
            'total_key_points': n_key_points,
            'storage_reduction': f"{(1 - n_key_points/len(coords))*100:.1f}%",
            'num_sublists': 1,  # Single spline
            'compression_ratio': compression_ratio
        }
    }




def compress_shape_divided_exact(coordinates, num_sublists=3, compression_ratio=0.2):
    """
    Exact version with proper point counting
    """
    # Divide shape smartly
    sublists, division_indices = divide_shape_smart_fixed(coordinates, num_sublists)
    
    results = []
    total_key_points = 0
    total_original_points_in_sublists = 0
    
    print(f"\n🎯 COMPRESSION WITH EXACT COUNTING:")
    print("=" * 50)
    print(f"Target compression ratio: {compression_ratio:.0%}")
    
    # Compress each sublist
    for i, sublist in enumerate(sublists):
        print(f"\n📦 Processing sublist {i+1}/{len(sublists)}...")
        result = compress_sublist_with_continuity(sublist, i, len(sublists), compression_ratio)
        
        if result:
            results.append(result)
            metrics = result['metrics']
            total_key_points += metrics['key_points_used']
            total_original_points_in_sublists += metrics['original_points']
            
            print(f"   ✅ Original: {metrics['original_points']} points")
            print(f"   ✅ Key points: {metrics['key_points_used']}")
            print(f"   ✅ Actual ratio: {metrics['key_points_used']/metrics['original_points']:.1%}")
            print(f"   ✅ Error: {metrics['mean_error']:.6f}")
        else:
            print(f"   ❌ Failed to compress sublist {i+1}")
            return compress_shape_single_fallback(coordinates, compression_ratio)
    
    # Combine results
    combined_reconstructed = combine_sublists_smoothly(results)
    
    if combined_reconstructed is None:
        return compress_shape_single_fallback(coordinates, compression_ratio)
    
    # Calculate overall metrics with exact counting
    original_coords = np.array(coordinates)
    if not np.allclose(original_coords[0], original_coords[-1]):
        original_coords = np.vstack([original_coords, original_coords[0]])
    
    total_original_points = len(original_coords)
    
    # Calculate exact error
    from scipy.interpolate import interp1d
    t_combined = np.linspace(0, 1, len(combined_reconstructed))
    t_original = np.linspace(0, 1, len(original_coords))
    
    f_x = interp1d(t_combined, combined_reconstructed[:, 0], kind='linear', 
                   bounds_error=False, fill_value='extrapolate')
    f_y = interp1d(t_combined, combined_reconstructed[:, 1], kind='linear', 
                   bounds_error=False, fill_value='extrapolate')
    
    x_interp = f_x(t_original)
    y_interp = f_y(t_original)
    
    overall_error = np.mean(np.sqrt((x_interp - original_coords[:, 0])**2 + 
                                   (y_interp - original_coords[:, 1])**2))
    
    # Calculate exact storage reduction
    actual_compression_ratio = total_key_points / total_original_points
    overall_storage_reduction = (1 - actual_compression_ratio) * 100
    
    print(f"\n📊 EXACT COMPRESSION SUMMARY:")
    print(f"  Total original points: {total_original_points}")
    print(f"  Total key points: {total_key_points}")
    print(f"  Target compression ratio: {compression_ratio:.1%}")
    print(f"  Actual compression ratio: {actual_compression_ratio:.1%}")
    print(f"  Storage reduction: {overall_storage_reduction:.1f}%")
    
    return {
        'sublist_results': results,
        'combined_reconstructed': combined_reconstructed,
        'overall_metrics': {
            'mean_error': overall_error,
            'total_original_points': total_original_points,
            'total_key_points': total_key_points,
            'storage_reduction': f"{overall_storage_reduction:.1f}%",
            'num_sublists': num_sublists,
            'compression_ratio': compression_ratio,
            'actual_compression_ratio': actual_compression_ratio
        }
    }










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
    """for i, sub_res in enumerate(sublist_results):
        color = colors[i % len(colors)]
        key_points = sub_res['key_points']
        ax3.plot(key_points[:, 0], key_points[:, 1], 'o', color=color, markersize=6,
                markerfacecolor='none', markeredgewidth=2, label=f'Sublist {i+1}')"""
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


























































import sys
import pickle

def build_coefficient_matrix(compression_result):
    """
    Build a matrix of spline coefficients from compression results
    Returns: coefficient_matrix, size_info
    """
    sublist_results = compression_result['sublist_results']
    num_sublists = len(sublist_results)
    
    coefficient_rows = []
    
    for i, result in enumerate(sublist_results):
        tck = result['spline_coefficients']
        key_points = result['key_points']
        
        # Extract spline coefficients and knots
        # tck structure: (t, c, k) where:
        # - t: knot vector
        # - c: coefficients (2 arrays for x and y)
        # - k: degree
        
        knots, coeffs, degree = tck
        coeffs_x, coeffs_y = coeffs
        
        # Create a row for this sublist containing all coefficients
        row_data = {
            'sublist_index': i,
            'knots': knots,
            'coeffs_x': coeffs_x,
            'coeffs_y': coeffs_y,
            'degree': degree,
            'num_key_points': len(key_points),
            'key_points': key_points  # Optional: include actual key points
        }
        
        coefficient_rows.append(row_data)
    
    return coefficient_rows

def analyze_storage_savings(original_coordinates, coefficient_matrix, compression_result):
    """
    Compare storage requirements between original and compressed data
    """
    # Calculate original data size
    original_array = np.array(original_coordinates)
    original_size_bytes = original_array.nbytes
    
    # Calculate compressed data size
    # Method 1: Estimate from coefficient matrix
    compressed_size_estimate = 0
    for row in coefficient_matrix:
        compressed_size_estimate += (
            row['knots'].nbytes + 
            row['coeffs_x'].nbytes + 
            row['coeffs_y'].nbytes + 
            sys.getsizeof(row['degree']) +
            row['key_points'].nbytes
        )
    
    # Method 2: Actual serialized size (more accurate)
    serialized_data = pickle.dumps(coefficient_matrix)
    compressed_actual_size = len(serialized_data)
    
    # Method 3: Just key points size (minimal representation)
    all_key_points = np.vstack([row['key_points'] for row in coefficient_matrix])
    key_points_size = all_key_points.nbytes
    
    # Get metrics from compression result
    overall_metrics = compression_result['overall_metrics']
    
    print("💾 STORAGE ANALYSIS")
    print("=" * 50)
    print(f"Original Data:")
    print(f"  - Points: {overall_metrics['total_original_points']}")
    print(f"  - Size: {original_size_bytes:,} bytes")
    print(f"  - Shape: {original_array.shape}")
    
    print(f"\nCompressed Data (Coefficient Matrix):")
    print(f"  - Number of sublists: {len(coefficient_matrix)}")
    print(f"  - Total key points: {overall_metrics['total_key_points']}")
    print(f"  - Estimated size: {compressed_size_estimate:,} bytes")
    print(f"  - Serialized size: {compressed_actual_size:,} bytes")
    print(f"  - Key points only: {key_points_size:,} bytes")
    
    print(f"\n📊 STORAGE REDUCTION:")
    reduction_estimated = (1 - compressed_size_estimate / original_size_bytes) * 100
    reduction_actual = (1 - compressed_actual_size / original_size_bytes) * 100
    reduction_keypoints = (1 - key_points_size / original_size_bytes) * 100
    
    print(f"  - Estimated: {reduction_estimated:.1f}%")
    print(f"  - Actual (serialized): {reduction_actual:.1f}%")
    print(f"  - Key points only: {reduction_keypoints:.1f}%")
    print(f"  - Theoretical: {overall_metrics['storage_reduction']}")
    
    return {
        'original_size': original_size_bytes,
        'compressed_estimated': compressed_size_estimate,
        'compressed_actual': compressed_actual_size,
        'key_points_size': key_points_size,
        'coefficient_matrix': coefficient_matrix
    }

def visualize_coefficient_matrix(coefficient_matrix):
    """
    Visualize the structure of the coefficient matrix
    """
    print("\n📈 COEFFICIENT MATRIX STRUCTURE:")
    print("=" * 50)
    
    # Create a summary table
    print(f"{'Sublist':<8} {'Knots':<8} {'Coeffs X':<10} {'Coeffs Y':<10} {'Key Pts':<8} {'Degree':<8}")
    print("-" * 60)
    
    total_coeffs = 0
    for i, row in enumerate(coefficient_matrix):
        num_knots = len(row['knots'])
        num_coeffs_x = len(row['coeffs_x'])
        num_coeffs_y = len(row['coeffs_y'])
        num_key_pts = row['num_key_points']
        degree = row['degree']
        
        total_coeffs += num_coeffs_x + num_coeffs_y
        
        print(f"{i+1:<8} {num_knots:<8} {num_coeffs_x:<10} {num_coeffs_y:<10} {num_key_pts:<8} {degree:<8}")
    
    print(f"\nTotal coefficients stored: {total_coeffs}")
    
    # Plot the matrix structure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Number of coefficients per sublist
    coeffs_per_sublist = [len(row['coeffs_x']) + len(row['coeffs_y']) for row in coefficient_matrix]
    axes[0, 0].bar(range(len(coefficient_matrix)), coeffs_per_sublist)
    axes[0, 0].set_title('Coefficients per Sublist')
    axes[0, 0].set_xlabel('Sublist Index')
    axes[0, 0].set_ylabel('Number of Coefficients')
    
    # Plot 2: Key points distribution
    key_points_per_sublist = [row['num_key_points'] for row in coefficient_matrix]
    axes[0, 1].bar(range(len(coefficient_matrix)), key_points_per_sublist, color='orange')
    axes[0, 1].set_title('Key Points per Sublist')
    axes[0, 1].set_xlabel('Sublist Index')
    axes[0, 1].set_ylabel('Number of Key Points')
    
    # Plot 3: Storage breakdown
    """sizes = [original_size_bytes, compressed_actual_size, key_points_size]
    labels = ['Original Points', 'Compressed Data', 'Key Points Only']
    colors = ['red', 'blue', 'green']
    axes[1, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
    axes[1, 0].set_title('Storage Breakdown')"""
    
    # Plot 4: Coefficient types distribution
    total_coeffs_x = sum(len(row['coeffs_x']) for row in coefficient_matrix)
    total_coeffs_y = sum(len(row['coeffs_y']) for row in coefficient_matrix)
    total_knots = sum(len(row['knots']) for row in coefficient_matrix)
    
    coeff_types = [total_coeffs_x, total_coeffs_y, total_knots]
    coeff_labels = ['X Coefficients', 'Y Coefficients', 'Knots']
    axes[1, 1].bar(coeff_labels, coeff_types, color=['lightblue', 'lightcoral', 'lightgreen'])
    axes[1, 1].set_title('Compressed Data Composition')
    axes[1, 1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.show()

































import numpy as np

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


def get_minimal_storage_with_rounding(compression_result, decimal_places=3):
    """
    Get absolute minimal storage by storing only key points with rounding
    Returns: compressed_data, storage_info
    """
    sublist_results = compression_result['sublist_results']
    
    # Extract all key points from all sublists
    all_key_points = []
    for result in sublist_results:
        all_key_points.extend(result['key_points'])
    
    # Convert to numpy array
    all_key_points = np.array(all_key_points)
    
    # Remove duplicates while preserving order (important for reconstruction)
    unique_points = []
    for point in all_key_points:
        if len(unique_points) == 0 or not np.allclose(point, unique_points[-1], atol=1e-6):
            unique_points.append(point)
    unique_points = np.array(unique_points)
    
    # Round to specified decimal places
    rounded_key_points = np.round(unique_points, decimal_places)
    
    # Remove duplicates again after rounding
    final_points = []
    for point in rounded_key_points:
        if len(final_points) == 0 or not np.allclose(point, final_points[-1], atol=1e-6):
            final_points.append(point)
    final_points = np.array(final_points)
    
    # Calculate storage
    original_size = len(compression_result['combined_reconstructed']) * 2 * 8  # 8 bytes per float
    rounded_size = final_points.nbytes
    
    storage_info = {
        'original_points': len(compression_result['combined_reconstructed']),
        'compressed_points': len(final_points),
        'original_size_bytes': original_size,
        'rounded_size_bytes': rounded_size,
        'compression_ratio': len(final_points) / len(compression_result['combined_reconstructed']),
        'storage_reduction': (1 - rounded_size / original_size) * 100,
        'decimal_places': decimal_places
    }
    
    return final_points, storage_info

def analyze_rounding_impact(compression_result):
    """
    Analyze how rounding affects storage and accuracy
    """
    print("🔍 ROUNDING IMPACT ANALYSIS")
    print("=" * 50)
    
    decimal_options = [1, 2, 3, 4, 5]  # Start from 1 to avoid too much precision loss
    
    results = []
    for decimals in decimal_options:
        try:
            compressed_data, storage_info = get_minimal_storage_with_rounding(
                compression_result, decimal_places=decimals
            )
            
            print(f"Compressed to {len(compressed_data)} points with {decimals} decimal places")
            
            # Reconstruct and calculate error
            reconstructed, _ = reconstruct_from_minimal_storage(compressed_data)
            original = compression_result['combined_reconstructed']
            
            # Ensure same length for comparison
            min_length = min(len(reconstructed), len(original))
            reconstructed = reconstructed[:min_length]
            original = original[:min_length]
            
            # Calculate reconstruction error
            error = np.mean(np.sqrt(
                (reconstructed[:, 0] - original[:, 0])**2 + 
                (reconstructed[:, 1] - original[:, 1])**2
            ))
            
            results.append({
                'decimal_places': decimals,
                'storage_bytes': storage_info['rounded_size_bytes'],
                'storage_reduction': storage_info['storage_reduction'],
                'reconstruction_error': error,
                'compressed_points': len(compressed_data)
            })
            
            print(f"Decimals: {decimals} | Size: {storage_info['rounded_size_bytes']:>6,} bytes | "
                  f"Reduction: {storage_info['storage_reduction']:5.1f}% | Error: {error:.6f}")
                  
        except Exception as e:
            print(f"❌ Failed with {decimals} decimals: {e}")
            continue
    
    return results


# Simple test function to debug
def test_reconstruction_simple(compression_result, decimal_places=3):
    """Simple test without the full analysis"""
    print("🧪 SIMPLE RECONSTRUCTION TEST")
    print("=" * 50)
    
    # Get compressed data
    compressed_data, storage_info = get_minimal_storage_with_rounding(
        compression_result, decimal_places=decimal_places
    )
    
    print(f"Compressed from {storage_info['original_points']} to {len(compressed_data)} points")
    print(f"Storage: {storage_info['rounded_size_bytes']:,} bytes ({storage_info['storage_reduction']:.1f}% reduction)")
    
    # Reconstruct
    reconstructed, _ = reconstruct_from_minimal_storage(compressed_data)
    
    # Calculate error
    original = compression_result['combined_reconstructed']
    error = np.mean(np.sqrt(
        (reconstructed[:, 0] - original[:, 0])**2 + 
        (reconstructed[:, 1] - original[:, 1])**2
    ))
    
    print(f"Reconstruction error: {error:.6f}")
    
    # Simple visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(original[:, 0], original[:, 1], 'b-', label='Original', linewidth=2)
    plt.plot(compressed_data[:, 0], compressed_data[:, 1], 'ro', markersize=3, label='Key Points')
    plt.legend()
    plt.title(f'Compression: {len(compressed_data)} key points')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.subplot(1, 2, 2)
    plt.plot(original[:, 0], original[:, 1], 'b-', label='Original', linewidth=2)
    plt.plot(reconstructed[:, 0], reconstructed[:, 1], 'g--', label='Reconstructed', linewidth=2)
    plt.legend()
    plt.title(f'Reconstruction (Error: {error:.6f})')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.show()
    
    return compressed_data, reconstructed, error

def visualize_minimal_storage_results(original, compressed, reconstructed, storage_info):
    """
    Visualize original vs compressed vs reconstructed boundaries
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    #fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot 1: Original vs Compressed
    axes[0, 0].plot(original[:, 0], original[:, 1], 'b-', alpha=0.7, label='Original', linewidth=2)
    axes[0, 0].plot(compressed[:, 0], compressed[:, 1], 'ro', markersize=4, label='Compressed Key Points')
    axes[0, 0].set_title(f'Original vs Compressed\n({len(compressed)} key points)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_aspect('equal')
    
    # Plot 2: Original vs Reconstructed
    axes[0, 1].plot(original[:, 0], original[:, 1], 'b-', alpha=0.7, label='Original', linewidth=2)
    axes[0, 1].plot(reconstructed[:, 0], reconstructed[:, 1], 'g--', alpha=0.8, label='Reconstructed', linewidth=2)
    axes[0, 1].set_title('Original vs Reconstructed')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_aspect('equal')
    
    # Plot 3: Storage comparison
    sizes = [storage_info['original_size_bytes'], storage_info['rounded_size_bytes']]
    labels = [f'Original\n{storage_info["original_size_bytes"]:,} bytes', 
              f'Compressed\n{storage_info["rounded_size_bytes"]:,} bytes']
    colors = ['lightcoral', 'lightgreen']
    axes[1, 0].bar(labels, sizes, color=colors, alpha=0.8)
    axes[1, 0].set_title('Storage Comparison')
    axes[1, 0].set_ylabel('Bytes')
    
    # Add value labels on bars
    for i, v in enumerate(sizes):
        axes[1, 0].text(i, v + max(sizes)*0.01, f'{v:,}', ha='center', va='bottom')
    
    # Plot 4: Error visualization
    error = np.sqrt(
        (reconstructed[:, 0] - original[:, 0])**2 + 
        (reconstructed[:, 1] - original[:, 1])**2
    )
    axes[1, 1].plot(error, 'r-', alpha=0.7)
    axes[1, 1].set_title('Reconstruction Error per Point')
    axes[1, 1].set_xlabel('Point Index')
    axes[1, 1].set_ylabel('Error')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(bottom=0)
    
    # Add mean error to plot
    mean_error = np.mean(error)
    axes[1, 1].axhline(y=mean_error, color='blue', linestyle='--', label=f'Mean Error: {mean_error:.6f}')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()

















































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






















def visualize_reconstruction_comparison(original, compressed, reconstructed):
    """
    Simple side-by-side visualization
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Original vs Compressed Points
    ax1.plot(original[:, 0], original[:, 1], 'b-', alpha=0.7, linewidth=2, label='Original Boundary')
    ax1.plot(compressed[:, 0], compressed[:, 1], 'ro', markersize=4, label='Compressed Points')
    ax1.set_title(f'Compression: {len(original)} → {len(compressed)} points')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot 2: Original vs Reconstructed
    ax2.plot(original[:, 0], original[:, 1], 'b-', alpha=0.7, linewidth=2, label='Original')
    ax2.plot(reconstructed[:, 0], reconstructed[:, 1], 'g--', alpha=0.8, linewidth=2, label='Reconstructed')
    ax2.set_title('Original vs Reconstructed')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Plot 3: Error Visualization
    error = np.sqrt((reconstructed[:, 0] - original[:, 0])**2 + 
                   (reconstructed[:, 1] - original[:, 1])**2)
    ax3.plot(error, 'r-', alpha=0.7)
    ax3.axhline(y=np.mean(error), color='blue', linestyle='--', 
                label=f'Mean Error: {np.mean(error):.6f}')
    ax3.set_title('Reconstruction Error per Point')
    ax3.set_xlabel('Point Index')
    ax3.set_ylabel('Error')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print error statistics
    print(f"📊 Reconstruction Quality:")
    print(f"   Mean Error: {np.mean(error):.6f}")
    print(f"   Max Error: {np.max(error):.6f}")
    print(f"   Std Error: {np.std(error):.6f}")

def visualize_interactive_comparison(original, compressed, reconstructed):
    """
    Interactive visualization with zoom capability
    """
    from matplotlib.widgets import Slider
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Full view
    ax1.plot(original[:, 0], original[:, 1], 'b-', linewidth=2, label='Original')
    ax1.plot(reconstructed[:, 0], reconstructed[:, 1], 'g--', linewidth=1, label='Reconstructed')
    ax1.plot(compressed[:, 0], compressed[:, 1], 'ro', markersize=3, label='Key Points')
    ax1.set_title('Full Boundary - Original vs Reconstructed')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot 2: Zoomed view
    zoom_factor = 0.1  # Show 10% of the boundary
    zoom_start = int(len(original) * 0.5)  # Start from middle
    zoom_end = int(zoom_start + len(original) * zoom_factor)
    
    ax2.plot(original[zoom_start:zoom_end, 0], original[zoom_start:zoom_end, 1], 
             'b-', linewidth=3, label='Original')
    ax2.plot(reconstructed[zoom_start:zoom_end, 0], reconstructed[zoom_start:zoom_end, 1], 
             'g--', linewidth=2, label='Reconstructed')
    ax2.plot(compressed[:, 0], compressed[:, 1], 'ro', markersize=4, alpha=0.5, label='Key Points')
    ax2.set_title(f'Zoomed View (points {zoom_start}-{zoom_end})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()


def visualize_reconstruction_overlay(original, compressed, reconstructed):
    """
    Overlay visualization showing all three together
    """
    plt.figure(figsize=(12, 10))
    
    # Plot original boundary
    plt.plot(original[:, 0], original[:, 1], 'b-', linewidth=3, alpha=0.5, label='Original Boundary')
    
    # Plot compressed points
    plt.plot(compressed[:, 0], compressed[:, 1], 'ro', markersize=6, label=f'Compressed Points ({len(compressed)})')
    
    # Plot reconstructed boundary
    plt.plot(reconstructed[:, 0], reconstructed[:, 1], 'g--', linewidth=2, alpha=0.8, label='Reconstructed')
    
    plt.title(f'Boundary Reconstruction\n{len(original)} points → {len(compressed)} points → {len(reconstructed)} points', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def visualize_quality_metrics(original, reconstructed):
    """
    Detailed quality metrics visualization
    """
    # Calculate errors
    errors = np.sqrt((reconstructed[:, 0] - original[:, 0])**2 + 
                    (reconstructed[:, 1] - original[:, 1])**2)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Error distribution
    ax1.hist(errors, bins=50, alpha=0.7, color='red', edgecolor='black')
    ax1.axvline(np.mean(errors), color='blue', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(errors):.6f}')
    ax1.set_xlabel('Error')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Error Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Cumulative error
    cumulative_error = np.cumsum(errors)
    ax2.plot(cumulative_error, 'purple', linewidth=2)
    ax2.set_xlabel('Point Index')
    ax2.set_ylabel('Cumulative Error')
    ax2.set_title('Cumulative Reconstruction Error')
    ax2.grid(True, alpha=0.3)
    
    # 3. Error along boundary
    ax3.plot(errors, 'orange', linewidth=1)
    ax3.axhline(y=np.mean(errors), color='red', linestyle='--', 
                label=f'Mean: {np.mean(errors):.6f}')
    ax3.set_xlabel('Point Index along Boundary')
    ax3.set_ylabel('Error')
    ax3.set_title('Error Along Boundary')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Quality summary
    metrics = {
        'Mean Error': f'{np.mean(errors):.6f}',
        'Max Error': f'{np.max(errors):.6f}',
        'Std Dev': f'{np.std(errors):.6f}',
        '95th Percentile': f'{np.percentile(errors, 95):.6f}',
        'Points > 0.001': f'{np.sum(errors > 0.001):,}',
        'Compression Ratio': f'{len(reconstructed)/len(original):.1%}'
    }
    
    ax4.axis('off')
    text_str = '\n'.join([f'{k}: {v}' for k, v in metrics.items()])
    ax4.text(0.1, 0.9, text_str, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.set_title('Quality Metrics Summary')
    
    plt.tight_layout()
    plt.show()
















# MAIN EXECUTION - TEST WITH YOUR DATA
if __name__ == "__main__":
    # Your sample coordinates (replace with your actual problematic data)
    sample_coordinates =[  (661.5,510.0),  (661.5,509.0),  (661.0,508.5),  (660.5,508.0),  (660.5,507.0), 
  (660.5,506.0),  (660.5,505.0),  (660.0,504.5),  (659.0,504.5),  (658.0,504.5), 
  (657.5,505.0),  (657.0,505.5),  (656.0,505.5),  (655.0,505.5),  (654.5,506.0), 
  (654.0,506.5),  (653.5,507.0),  (653.0,507.5),  (652.0,507.5),  (651.5,508.0), 
  (651.0,508.5),  (650.5,509.0),  (650.5,510.0),  (650.0,510.5),  (649.5,511.0), 
  (649.5,512.0),  (649.5,513.0),  (649.5,514.0),  (649.5,515.0),  (649.5,516.0), 
  (649.5,517.0),  (650.0,517.5),  (650.5,518.0),  (650.5,519.0),  (650.0,519.5), 
  (649.5,520.0),  (649.5,521.0),  (649.5,522.0),  (649.0,522.5),  (648.0,522.5), 
  (647.0,522.5),  (646.0,522.5),  (645.0,522.5),  (644.0,522.5),  (643.5,522.0), 
  (643.0,521.5),  (642.0,521.5),  (641.0,521.5),  (640.0,521.5),  (639.5,522.0), 
  (639.0,522.5),  (638.0,522.5),  (637.0,522.5),  (636.5,523.0),  (636.0,523.5), 
  (635.0,523.5),  (634.0,523.5),  (633.0,523.5),  (632.5,524.0),  (632.5,525.0), 
  (632.5,526.0),  (632.5,527.0),  (632.0,527.5),  (631.5,528.0),  (631.0,528.5), 
  (630.5,529.0),  (630.5,530.0),  (630.0,530.5),  (629.5,531.0),  (629.5,532.0), 
  (629.5,533.0),  (629.5,534.0),  (629.5,535.0),  (629.5,536.0),  (629.0,536.5), 
  (628.5,537.0),  (628.5,538.0),  (628.0,538.5),  (627.5,539.0),  (627.0,539.5), 
  (626.0,539.5),  (625.0,539.5),  (624.0,539.5),  (623.0,539.5),  (622.0,539.5), 
  (621.5,539.0),  (621.5,538.0),  (622.0,537.5),  (622.5,538.0),  (623.0,538.5), 
  (623.5,538.0),  (624.0,537.5),  (624.5,537.0),  (624.5,536.0),  (625.0,535.5), 
  (625.5,535.0),  (625.5,534.0),  (625.5,533.0),  (625.5,532.0),  (625.5,531.0), 
  (625.5,530.0),  (625.5,529.0),  (625.5,528.0),  (625.5,527.0),  (626.0,526.5), 
  (626.5,526.0),  (626.5,525.0),  (626.5,524.0),  (627.0,523.5),  (627.5,523.0), 
  (627.5,522.0),  (628.0,521.5),  (628.5,521.0),  (629.0,520.5),  (629.5,520.0), 
  (629.5,519.0),  (629.5,518.0),  (629.5,517.0),  (629.5,516.0),  (629.0,515.5), 
  (628.5,515.0),  (628.5,514.0),  (628.0,513.5),  (627.5,513.0),  (627.5,512.0), 
  (627.0,511.5),  (626.5,511.0),  (626.5,510.0),  (626.5,509.0),  (626.5,508.0), 
  (626.5,507.0),  (626.5,506.0),  (626.0,505.5),  (625.5,505.0),  (625.5,504.0), 
  (625.5,503.0),  (625.5,502.0),  (625.5,501.0),  (625.5,500.0),  (625.5,499.0), 
  (626.0,498.5),  (626.5,498.0),  (626.5,497.0),  (626.5,496.0),  (627.0,495.5), 
  (627.5,495.0),  (627.5,494.0),  (627.5,493.0),  (627.5,492.0),  (628.0,491.5), 
  (628.5,491.0),  (628.5,490.0),  (628.5,489.0),  (628.5,488.0),  (628.5,487.0), 
  (629.0,486.5),  (629.5,486.0),  (629.5,485.0),  (629.5,484.0),  (630.0,483.5), 
  (630.5,483.0),  (630.5,482.0),  (630.5,481.0),  (631.0,480.5),  (631.5,480.0), 
  (631.5,479.0),  (631.5,478.0),  (632.0,477.5),  (632.5,477.0),  (632.5,476.0), 
  (633.0,475.5),  (633.5,475.0),  (633.5,474.0),  (633.5,473.0),  (633.5,472.0), 
  (633.5,471.0),  (633.5,470.0),  (633.5,469.0),  (633.5,468.0),  (633.5,467.0), 
  (633.5,466.0),  (633.5,465.0),  (633.5,464.0),  (633.5,463.0),  (633.5,462.0), 
  (634.0,461.5),  (634.5,461.0),  (635.0,460.5),  (635.5,460.0),  (636.0,459.5), 
  (636.5,459.0),  (636.5,458.0),  (636.5,457.0),  (636.5,456.0),  (636.5,455.0), 
  (636.5,454.0),  (636.0,453.5),  (635.5,453.0),  (635.5,452.0),  (635.5,451.0), 
  (635.0,450.5),  (634.5,450.0),  (634.5,449.0),  (634.5,448.0),  (634.5,447.0), 
  (634.5,446.0),  (634.5,445.0),  (634.5,444.0),  (634.5,443.0),  (634.0,442.5), 
  (633.5,442.0),  (633.0,441.5),  (632.0,441.5),  (631.5,441.0),  (631.0,440.5), 
  (630.0,440.5),  (629.5,441.0),  (629.0,441.5),  (628.5,442.0),  (628.0,442.5), 
  (627.0,442.5),  (626.5,442.0),  (626.0,441.5),  (625.0,441.5),  (624.0,441.5), 
  (623.0,441.5),  (622.0,441.5),  (621.0,441.5),  (620.5,442.0),  (620.0,442.5), 
  (619.0,442.5),  (618.0,442.5),  (617.0,442.5),  (616.0,442.5),  (615.5,442.0), 
  (615.0,441.5),  (614.0,441.5),  (613.5,441.0),  (613.0,440.5),  (612.5,440.0), 
  (612.0,439.5),  (611.0,439.5),  (610.0,439.5),  (609.0,439.5),  (608.0,439.5), 
  (607.5,439.0),  (607.0,438.5),  (606.0,438.5),  (605.0,438.5),  (604.5,438.0), 
  (604.0,437.5),  (603.5,437.0),  (603.0,436.5),  (602.0,436.5),  (601.0,436.5), 
  (600.5,436.0),  (600.0,435.5),  (599.0,435.5),  (598.0,435.5),  (597.0,435.5), 
  (596.5,436.0),  (596.0,436.5),  (595.5,437.0),  (595.0,437.5),  (594.5,438.0), 
  (594.0,438.5),  (593.5,439.0),  (593.0,439.5),  (592.0,439.5),  (591.5,440.0), 
  (591.0,440.5),  (590.5,441.0),  (590.0,441.5),  (589.0,441.5),  (588.5,441.0), 
  (588.5,440.0),  (588.5,439.0),  (588.5,438.0),  (588.0,437.5),  (587.5,437.0), 
  (587.0,436.5),  (586.5,436.0),  (586.0,435.5),  (585.5,436.0),  (585.5,437.0), 
  (585.5,438.0),  (585.5,439.0),  (585.5,440.0),  (585.0,440.5),  (584.0,440.5), 
  (583.5,440.0),  (583.0,439.5),  (582.5,439.0),  (582.0,438.5),  (581.0,438.5), 
  (580.0,438.5),  (579.0,438.5),  (578.5,438.0),  (578.0,437.5),  (577.5,437.0), 
  (578.0,436.5),  (578.5,436.0),  (578.5,435.0),  (578.0,434.5),  (577.5,434.0), 
  (577.5,433.0),  (577.5,432.0),  (577.5,431.0),  (578.0,430.5),  (579.0,430.5), 
  (580.0,430.5),  (580.5,431.0),  (581.0,431.5),  (581.5,431.0),  (581.5,430.0), 
  (581.5,429.0),  (581.5,428.0),  (581.5,427.0),  (581.0,426.5),  (580.5,427.0), 
  (580.0,427.5),  (579.5,428.0),  (579.0,428.5),  (578.0,428.5),  (577.5,428.0), 
  (577.0,427.5),  (576.0,427.5),  (575.5,428.0),  (575.0,428.5),  (574.5,429.0), 
  (574.0,429.5),  (573.0,429.5),  (572.5,430.0),  (572.0,430.5),  (571.0,430.5), 
  (570.0,430.5),  (569.0,430.5),  (568.0,430.5),  (567.5,431.0),  (567.0,431.5), 
  (566.0,431.5),  (565.5,431.0),  (565.5,430.0),  (566.0,429.5),  (566.5,429.0), 
  (566.0,428.5),  (565.0,428.5),  (564.5,429.0),  (564.0,429.5),  (563.5,430.0), 
  (563.0,430.5),  (562.0,430.5),  (561.5,431.0),  (561.0,431.5),  (560.0,431.5), 
  (559.0,431.5),  (558.5,432.0),  (558.0,432.5),  (557.0,432.5),  (556.0,432.5), 
  (555.5,433.0),  (555.0,433.5),  (554.5,434.0),  (554.0,434.5),  (553.5,435.0), 
  (553.0,435.5),  (552.5,436.0),  (552.0,436.5),  (551.0,436.5),  (550.0,436.5), 
  (549.0,436.5),  (548.5,437.0),  (548.0,437.5),  (547.0,437.5),  (546.5,438.0), 
  (546.0,438.5),  (545.0,438.5),  (544.0,438.5),  (543.0,438.5),  (542.5,439.0), 
  (542.0,439.5),  (541.5,439.0),  (541.0,438.5),  (540.5,438.0),  (540.0,437.5), 
  (539.5,438.0),  (539.0,438.5),  (538.5,439.0),  (539.0,439.5),  (539.5,440.0), 
  (539.5,441.0),  (540.0,441.5),  (540.5,442.0),  (540.5,443.0),  (540.0,443.5), 
  (539.5,444.0),  (539.5,445.0),  (539.5,446.0),  (539.5,447.0),  (539.5,448.0), 
  (539.0,448.5),  (538.5,449.0),  (538.5,450.0),  (538.0,450.5),  (537.0,450.5), 
  (536.5,451.0),  (536.0,451.5),  (535.5,452.0),  (535.0,452.5),  (534.5,453.0), 
  (534.5,454.0),  (534.0,454.5),  (533.5,455.0),  (533.0,455.5),  (532.0,455.5), 
  (531.5,456.0),  (531.0,456.5),  (530.5,457.0),  (530.0,457.5),  (529.0,457.5), 
  (528.0,457.5),  (527.5,457.0),  (527.0,456.5),  (526.5,456.0),  (526.5,455.0), 
  (526.5,454.0),  (526.0,453.5),  (525.5,453.0),  (525.5,452.0),  (525.0,451.5), 
  (524.5,451.0),  (524.5,450.0),  (524.0,449.5),  (523.5,449.0),  (523.5,448.0), 
  (523.0,447.5),  (522.5,447.0),  (522.0,446.5),  (521.5,446.0),  (521.0,445.5), 
  (520.0,445.5),  (519.0,445.5),  (518.0,445.5),  (517.0,445.5),  (516.0,445.5), 
  (515.0,445.5),  (514.0,445.5),  (513.5,445.0),  (513.0,444.5),  (512.5,445.0), 
  (512.0,445.5),  (511.5,446.0),  (511.5,447.0),  (512.0,447.5),  (513.0,447.5), 
  (513.5,448.0),  (513.5,449.0),  (513.5,450.0),  (513.5,451.0),  (513.5,452.0), 
  (514.0,452.5),  (514.5,453.0),  (514.0,453.5),  (513.0,453.5),  (512.5,453.0), 
  (512.0,452.5),  (511.5,452.0),  (511.5,451.0),  (511.0,450.5),  (510.5,450.0), 
  (510.5,449.0),  (510.0,448.5),  (509.5,449.0),  (509.0,449.5),  (508.5,450.0), 
  (508.5,451.0),  (508.5,452.0),  (508.5,453.0),  (508.5,454.0),  (509.0,454.5), 
  (509.5,455.0),  (509.5,456.0),  (510.0,456.5),  (511.0,456.5),  (512.0,456.5), 
  (512.5,457.0),  (513.0,457.5),  (514.0,457.5),  (514.5,458.0),  (515.0,458.5), 
  (515.5,459.0),  (515.5,460.0),  (516.0,460.5),  (516.5,461.0),  (517.0,461.5), 
  (517.5,462.0),  (518.0,462.5),  (518.5,463.0),  (519.0,463.5),  (520.0,463.5), 
  (521.0,463.5),  (521.5,463.0),  (521.5,462.0),  (522.0,461.5),  (522.5,461.0), 
  (523.0,460.5),  (523.5,461.0),  (523.5,462.0),  (523.5,463.0),  (523.5,464.0), 
  (524.0,464.5),  (525.0,464.5),  (525.5,465.0),  (526.0,465.5),  (527.0,465.5), 
  (527.5,466.0),  (528.0,466.5),  (528.5,467.0),  (528.5,468.0),  (528.5,469.0), 
  (528.5,470.0),  (528.5,471.0),  (528.5,472.0),  (528.5,473.0),  (528.5,474.0), 
  (528.5,475.0),  (529.0,475.5),  (529.5,476.0),  (529.5,477.0),  (529.0,477.5), 
  (528.5,478.0),  (528.5,479.0),  (528.0,479.5),  (527.5,480.0),  (527.0,480.5), 
  (526.0,480.5),  (525.5,481.0),  (525.0,481.5),  (524.0,481.5),  (523.5,482.0), 
  (523.0,482.5),  (522.5,483.0),  (522.5,484.0),  (522.5,485.0),  (522.0,485.5), 
  (521.0,485.5),  (520.5,486.0),  (520.5,487.0),  (521.0,487.5),  (522.0,487.5), 
  (522.5,488.0),  (523.0,488.5),  (524.0,488.5),  (525.0,488.5),  (526.0,488.5), 
  (527.0,488.5),  (528.0,488.5),  (528.5,489.0),  (529.0,489.5),  (529.5,490.0), 
  (529.5,491.0),  (530.0,491.5),  (530.5,492.0),  (530.5,493.0),  (530.5,494.0), 
  (530.0,494.5),  (529.5,495.0),  (529.0,495.5),  (528.5,496.0),  (528.5,497.0), 
  (528.5,498.0),  (528.5,499.0),  (528.5,500.0),  (528.0,500.5),  (527.5,501.0), 
  (527.5,502.0),  (527.5,503.0),  (527.0,503.5),  (526.5,504.0),  (526.5,505.0), 
  (526.0,505.5),  (525.5,506.0),  (525.5,507.0),  (525.0,507.5),  (524.5,508.0), 
  (524.0,508.5),  (523.5,509.0),  (523.0,509.5),  (522.5,510.0),  (522.0,510.5), 
  (521.5,511.0),  (521.5,512.0),  (521.0,512.5),  (520.5,512.0),  (520.5,511.0), 
  (520.5,510.0),  (520.5,509.0),  (520.5,508.0),  (520.5,507.0),  (520.5,506.0), 
  (520.5,505.0),  (521.0,504.5),  (521.5,504.0),  (521.5,503.0),  (521.5,502.0), 
  (521.0,501.5),  (520.5,501.0),  (520.0,500.5),  (519.5,501.0),  (519.5,502.0), 
  (519.5,503.0),  (519.5,504.0),  (519.5,505.0),  (519.0,505.5),  (518.5,506.0), 
  (518.5,507.0),  (518.5,508.0),  (518.0,508.5),  (517.5,509.0),  (517.0,509.5), 
  (516.5,510.0),  (517.0,510.5),  (517.5,511.0),  (517.5,512.0),  (517.5,513.0), 
  (517.0,513.5),  (516.5,514.0),  (516.5,515.0),  (517.0,515.5),  (517.5,516.0), 
  (517.5,517.0),  (517.0,517.5),  (516.0,517.5),  (515.0,517.5),  (514.0,517.5), 
  (513.0,517.5),  (512.0,517.5),  (511.5,518.0),  (511.0,518.5),  (510.5,518.0), 
  (510.5,517.0),  (510.5,516.0),  (510.5,515.0),  (510.5,514.0),  (510.5,513.0), 
  (510.5,512.0),  (510.5,511.0),  (510.0,510.5),  (509.5,510.0),  (509.5,509.0), 
  (509.0,508.5),  (508.5,508.0),  (508.0,507.5),  (507.5,507.0),  (507.0,506.5), 
  (506.5,506.0),  (506.0,505.5),  (505.0,505.5),  (504.5,505.0),  (504.5,504.0), 
  (504.0,503.5),  (503.5,503.0),  (503.0,502.5),  (502.0,502.5),  (501.0,502.5), 
  (500.5,502.0),  (500.0,501.5),  (499.0,501.5),  (498.0,501.5),  (497.5,502.0), 
  (498.0,502.5),  (499.0,502.5),  (499.5,503.0),  (500.0,503.5),  (501.0,503.5), 
  (501.5,504.0),  (502.0,504.5),  (502.5,505.0),  (503.0,505.5),  (503.5,506.0), 
  (504.0,506.5),  (504.5,507.0),  (505.0,507.5),  (505.5,508.0),  (506.0,508.5), 
  (506.5,509.0),  (507.0,509.5),  (507.5,510.0),  (507.5,511.0),  (507.5,512.0), 
  (507.5,513.0),  (507.5,514.0),  (507.5,515.0),  (507.5,516.0),  (507.5,517.0), 
  (507.0,517.5),  (506.5,518.0),  (506.0,518.5),  (505.5,519.0),  (505.0,519.5), 
  (504.5,519.0),  (504.5,518.0),  (504.5,517.0),  (504.0,516.5),  (503.0,516.5), 
  (502.5,516.0),  (502.0,515.5),  (501.5,515.0),  (501.0,514.5),  (500.5,514.0), 
  (500.0,513.5),  (499.0,513.5),  (498.0,513.5),  (497.0,513.5),  (496.5,513.0), 
  (496.0,512.5),  (495.5,512.0),  (495.0,511.5),  (494.0,511.5),  (493.0,511.5), 
  (492.0,511.5),  (491.0,511.5),  (490.0,511.5),  (489.5,511.0),  (489.5,510.0), 
  (489.5,509.0),  (489.0,508.5),  (488.5,508.0),  (488.0,507.5),  (487.5,507.0), 
  (487.0,506.5),  (486.0,506.5),  (485.0,506.5),  (484.5,507.0),  (484.0,507.5), 
  (483.0,507.5),  (482.5,508.0),  (483.0,508.5),  (484.0,508.5),  (484.5,509.0), 
  (485.0,509.5),  (486.0,509.5),  (486.5,510.0),  (487.0,510.5),  (487.5,511.0), 
  (487.5,512.0),  (487.0,512.5),  (486.5,513.0),  (486.5,514.0),  (487.0,514.5), 
  (488.0,514.5),  (488.5,515.0),  (489.0,515.5),  (489.5,516.0),  (489.0,516.5), 
  (488.5,517.0),  (488.0,517.5),  (487.5,518.0),  (487.0,518.5),  (486.0,518.5), 
  (485.0,518.5),  (484.5,519.0),  (484.5,520.0),  (485.0,520.5),  (486.0,520.5), 
  (487.0,520.5),  (487.5,521.0),  (488.0,521.5),  (488.5,522.0),  (488.0,522.5), 
  (487.5,523.0),  (487.0,523.5),  (486.0,523.5),  (485.5,524.0),  (485.0,524.5), 
  (484.0,524.5),  (483.5,524.0),  (483.0,523.5),  (482.0,523.5),  (481.0,523.5), 
  (480.0,523.5),  (479.0,523.5),  (478.5,523.0),  (478.0,522.5),  (477.5,523.0), 
  (477.0,523.5),  (476.5,524.0),  (476.5,525.0),  (476.5,526.0),  (476.5,527.0), 
  (476.5,528.0),  (476.5,529.0),  (476.0,529.5),  (475.5,530.0),  (475.5,531.0), 
  (475.5,532.0),  (475.5,533.0),  (475.0,533.5),  (474.5,534.0),  (474.0,534.5), 
  (473.5,535.0),  (473.5,536.0),  (473.0,536.5),  (472.5,537.0),  (472.5,538.0), 
  (472.0,538.5),  (471.5,539.0),  (471.0,539.5),  (470.0,539.5),  (469.0,539.5), 
  (468.0,539.5),  (467.0,539.5),  (466.5,540.0),  (466.0,540.5),  (465.5,541.0), 
  (465.0,541.5),  (464.5,542.0),  (464.5,543.0),  (465.0,543.5),  (465.5,543.0), 
  (466.0,542.5),  (466.5,542.0),  (467.0,541.5),  (467.5,541.0),  (468.0,540.5), 
  (469.0,540.5),  (470.0,540.5),  (471.0,540.5),  (472.0,540.5),  (472.5,540.0), 
  (473.0,539.5),  (473.5,539.0),  (474.0,538.5),  (474.5,538.0),  (475.0,537.5), 
  (475.5,537.0),  (476.0,536.5),  (476.5,536.0),  (477.0,535.5),  (477.5,535.0), 
  (477.5,534.0),  (478.0,533.5),  (478.5,533.0),  (479.0,532.5),  (480.0,532.5), 
  (480.5,533.0),  (480.5,534.0),  (480.5,535.0),  (480.5,536.0),  (481.0,536.5), 
  (482.0,536.5),  (483.0,536.5),  (484.0,536.5),  (484.5,536.0),  (484.5,535.0), 
  (485.0,534.5),  (485.5,534.0),  (485.0,533.5),  (484.5,533.0),  (484.5,532.0), 
  (484.0,531.5),  (483.0,531.5),  (482.0,531.5),  (481.0,531.5),  (480.5,531.0), 
  (480.0,530.5),  (479.5,530.0),  (479.5,529.0),  (479.0,528.5),  (478.5,528.0), 
  (479.0,527.5),  (479.5,527.0),  (480.0,526.5),  (481.0,526.5),  (481.5,527.0), 
  (482.0,527.5),  (483.0,527.5),  (483.5,528.0),  (484.0,528.5),  (484.5,529.0), 
  (485.0,529.5),  (485.5,530.0),  (486.0,530.5),  (487.0,530.5),  (487.5,530.0), 
  (488.0,529.5),  (489.0,529.5),  (490.0,529.5),  (490.5,530.0),  (491.0,530.5), 
  (491.5,530.0),  (492.0,529.5),  (492.5,529.0),  (492.5,528.0),  (493.0,527.5), 
  (493.5,527.0),  (494.0,526.5),  (495.0,526.5),  (495.5,527.0),  (496.0,527.5), 
  (496.5,528.0),  (497.0,528.5),  (497.5,529.0),  (498.0,529.5),  (499.0,529.5), 
  (500.0,529.5),  (500.5,529.0),  (500.5,528.0),  (500.5,527.0),  (500.5,526.0), 
  (500.5,525.0),  (500.5,524.0),  (500.5,523.0),  (501.0,522.5),  (502.0,522.5), 
  (503.0,522.5),  (503.5,523.0),  (503.5,524.0),  (504.0,524.5),  (504.5,525.0), 
  (505.0,525.5),  (505.5,526.0),  (505.5,527.0),  (506.0,527.5),  (506.5,528.0), 
  (506.5,529.0),  (506.5,530.0),  (507.0,530.5),  (507.5,531.0),  (507.5,532.0), 
  (508.0,532.5),  (508.5,533.0),  (509.0,533.5),  (510.0,533.5),  (510.5,534.0), 
  (511.0,534.5),  (511.5,534.0),  (511.5,533.0),  (512.0,532.5),  (512.5,532.0), 
  (513.0,531.5),  (513.5,531.0),  (513.5,530.0),  (513.5,529.0),  (513.5,528.0), 
  (513.5,527.0),  (513.5,526.0),  (513.5,525.0),  (513.5,524.0),  (514.0,523.5), 
  (515.0,523.5),  (516.0,523.5),  (517.0,523.5),  (517.5,523.0),  (518.0,522.5), 
  (519.0,522.5),  (519.5,522.0),  (520.0,521.5),  (521.0,521.5),  (521.5,521.0), 
  (522.0,520.5),  (523.0,520.5),  (523.5,520.0),  (523.5,519.0),  (523.5,518.0), 
  (523.5,517.0),  (523.5,516.0),  (524.0,515.5),  (524.5,515.0),  (525.0,514.5), 
  (525.5,514.0),  (525.5,513.0),  (526.0,512.5),  (526.5,512.0),  (527.0,511.5), 
  (528.0,511.5),  (529.0,511.5),  (530.0,511.5),  (530.5,511.0),  (530.5,510.0), 
  (530.0,509.5),  (529.5,509.0),  (529.5,508.0),  (530.0,507.5),  (530.5,507.0), 
  (530.5,506.0),  (530.5,505.0),  (530.5,504.0),  (530.5,503.0),  (531.0,502.5), 
  (532.0,502.5),  (532.5,502.0),  (532.5,501.0),  (532.5,500.0),  (532.5,499.0), 
  (533.0,498.5),  (533.5,498.0),  (533.5,497.0),  (533.5,496.0),  (533.0,495.5), 
  (532.5,495.0),  (532.5,494.0),  (532.5,493.0),  (532.5,492.0),  (532.5,491.0), 
  (532.5,490.0),  (532.5,489.0),  (532.0,488.5),  (531.5,488.0),  (531.5,487.0), 
  (531.5,486.0),  (532.0,485.5),  (532.5,485.0),  (533.0,484.5),  (534.0,484.5), 
  (535.0,484.5),  (536.0,484.5),  (537.0,484.5),  (538.0,484.5),  (539.0,484.5), 
  (539.5,485.0),  (540.0,485.5),  (540.5,486.0),  (541.0,486.5),  (541.5,487.0), 
  (541.5,488.0),  (541.5,489.0),  (541.5,490.0),  (541.5,491.0),  (541.5,492.0), 
  (542.0,492.5),  (543.0,492.5),  (543.5,493.0),  (543.5,494.0),  (544.0,494.5), 
  (544.5,495.0),  (544.5,496.0),  (544.0,496.5),  (543.5,497.0),  (543.5,498.0), 
  (543.5,499.0),  (543.5,500.0),  (543.5,501.0),  (543.5,502.0),  (543.0,502.5), 
  (542.5,503.0),  (542.5,504.0),  (542.5,505.0),  (542.0,505.5),  (541.5,506.0), 
  (541.0,506.5),  (540.5,507.0),  (540.5,508.0),  (540.0,508.5),  (539.5,509.0), 
  (539.5,510.0),  (539.5,511.0),  (539.5,512.0),  (539.5,513.0),  (540.0,513.5), 
  (540.5,514.0),  (540.5,515.0),  (540.5,516.0),  (541.0,516.5),  (541.5,517.0), 
  (541.0,517.5),  (540.5,518.0),  (540.0,518.5),  (539.0,518.5),  (538.0,518.5), 
  (537.5,518.0),  (537.0,517.5),  (536.5,517.0),  (536.0,516.5),  (535.5,516.0), 
  (535.0,515.5),  (534.5,516.0),  (534.5,517.0),  (534.5,518.0),  (534.5,519.0), 
  (534.0,519.5),  (533.5,520.0),  (533.5,521.0),  (533.5,522.0),  (533.5,523.0), 
  (534.0,523.5),  (534.5,524.0),  (535.0,524.5),  (535.5,524.0),  (535.5,523.0), 
  (536.0,522.5),  (537.0,522.5),  (538.0,522.5),  (538.5,523.0),  (539.0,523.5), 
  (540.0,523.5),  (541.0,523.5),  (542.0,523.5),  (542.5,523.0),  (543.0,522.5), 
  (543.5,522.0),  (544.0,521.5),  (544.5,521.0),  (545.0,520.5),  (545.5,520.0), 
  (545.5,519.0),  (546.0,518.5),  (546.5,518.0),  (547.0,517.5),  (547.5,518.0), 
  (548.0,518.5),  (548.5,519.0),  (548.5,520.0),  (549.0,520.5),  (549.5,521.0), 
  (549.5,522.0),  (549.5,523.0),  (549.5,524.0),  (549.5,525.0),  (549.5,526.0), 
  (549.5,527.0),  (549.5,528.0),  (549.5,529.0),  (549.5,530.0),  (549.5,531.0), 
  (549.5,532.0),  (549.5,533.0),  (549.5,534.0),  (549.5,535.0),  (549.5,536.0), 
  (549.5,537.0),  (549.5,538.0),  (549.5,539.0),  (549.5,540.0),  (549.5,541.0), 
  (550.0,541.5),  (550.5,542.0),  (550.5,543.0),  (551.0,543.5),  (551.5,544.0), 
  (552.0,544.5),  (552.5,545.0),  (553.0,545.5),  (553.5,545.0),  (554.0,544.5), 
  (554.5,544.0),  (555.0,543.5),  (555.5,543.0),  (556.0,542.5),  (556.5,542.0), 
  (557.0,541.5),  (557.5,541.0),  (558.0,540.5),  (558.5,540.0),  (559.0,539.5), 
  (559.5,539.0),  (559.5,538.0),  (559.5,537.0),  (559.5,536.0),  (560.0,535.5), 
  (560.5,535.0),  (560.5,534.0),  (561.0,533.5),  (561.5,533.0),  (562.0,532.5), 
  (563.0,532.5),  (564.0,532.5),  (565.0,532.5),  (565.5,532.0),  (565.5,531.0), 
  (566.0,530.5),  (566.5,530.0),  (567.0,529.5),  (567.5,529.0),  (568.0,528.5), 
  (568.5,528.0),  (568.5,527.0),  (568.5,526.0),  (568.5,525.0),  (568.0,524.5), 
  (567.5,524.0),  (568.0,523.5),  (568.5,523.0),  (568.5,522.0),  (568.5,521.0), 
  (569.0,520.5),  (569.5,520.0),  (570.0,519.5),  (570.5,519.0),  (571.0,518.5), 
  (571.5,518.0),  (571.0,517.5),  (570.5,517.0),  (571.0,516.5),  (571.5,516.0), 
  (572.0,515.5),  (573.0,515.5),  (574.0,515.5),  (574.5,515.0),  (575.0,514.5), 
  (576.0,514.5),  (577.0,514.5),  (578.0,514.5),  (578.5,514.0),  (579.0,513.5), 
  (579.5,513.0),  (580.0,512.5),  (580.5,512.0),  (581.0,511.5),  (582.0,511.5), 
  (582.5,511.0),  (582.5,510.0),  (582.5,509.0),  (582.5,508.0),  (582.0,507.5), 
  (581.0,507.5),  (580.0,507.5),  (579.5,507.0),  (579.5,506.0),  (579.0,505.5), 
  (578.5,505.0),  (578.0,504.5),  (577.0,504.5),  (576.5,504.0),  (576.5,503.0), 
  (576.5,502.0),  (576.5,501.0),  (577.0,500.5),  (578.0,500.5),  (579.0,500.5), 
  (579.5,500.0),  (580.0,499.5),  (580.5,499.0),  (581.0,498.5),  (582.0,498.5), 
  (582.5,498.0),  (583.0,497.5),  (584.0,497.5),  (584.5,498.0),  (585.0,498.5), 
  (585.5,498.0),  (586.0,497.5),  (586.5,497.0),  (586.5,496.0),  (587.0,495.5), 
  (588.0,495.5),  (589.0,495.5),  (590.0,495.5),  (590.5,495.0),  (590.5,494.0), 
  (590.5,493.0),  (590.5,492.0),  (591.0,491.5),  (592.0,491.5),  (592.5,491.0), 
  (593.0,490.5),  (593.5,490.0),  (593.5,489.0),  (594.0,488.5),  (594.5,488.0), 
  (595.0,487.5),  (596.0,487.5),  (597.0,487.5),  (597.5,487.0),  (598.0,486.5), 
  (599.0,486.5),  (599.5,487.0),  (599.5,488.0),  (599.5,489.0),  (599.5,490.0), 
  (599.5,491.0),  (599.5,492.0),  (599.5,493.0),  (599.5,494.0),  (599.5,495.0), 
  (599.5,496.0),  (599.5,497.0),  (599.5,498.0),  (600.0,498.5),  (601.0,498.5), 
  (601.5,499.0),  (602.0,499.5),  (602.5,500.0),  (603.0,500.5),  (603.5,501.0), 
  (603.5,502.0),  (604.0,502.5),  (604.5,503.0),  (605.0,503.5),  (606.0,503.5), 
  (606.5,504.0),  (606.5,505.0),  (607.0,505.5),  (607.5,506.0),  (607.5,507.0), 
  (607.5,508.0),  (607.5,509.0),  (608.0,509.5),  (608.5,510.0),  (608.5,511.0), 
  (608.5,512.0),  (608.5,513.0),  (608.5,514.0),  (609.0,514.5),  (610.0,514.5), 
  (611.0,514.5),  (612.0,514.5),  (613.0,514.5),  (614.0,514.5),  (614.5,515.0), 
  (614.5,516.0),  (614.5,517.0),  (614.5,518.0),  (614.5,519.0),  (614.5,520.0), 
  (614.0,520.5),  (613.5,521.0),  (613.0,521.5),  (612.5,522.0),  (612.0,522.5), 
  (611.5,523.0),  (611.5,524.0),  (611.5,525.0),  (611.5,526.0),  (611.5,527.0), 
  (611.5,528.0),  (612.0,528.5),  (612.5,529.0),  (612.5,530.0),  (612.5,531.0), 
  (612.5,532.0),  (612.5,533.0),  (612.5,534.0),  (613.0,534.5),  (613.5,535.0), 
  (614.0,535.5),  (615.0,535.5),  (615.5,536.0),  (616.0,536.5),  (616.5,537.0), 
  (617.0,537.5),  (617.5,538.0),  (618.0,538.5),  (618.5,539.0),  (618.5,540.0), 
  (619.0,540.5),  (619.5,541.0),  (620.0,541.5),  (620.5,542.0),  (621.0,542.5), 
  (622.0,542.5),  (623.0,542.5),  (623.5,543.0),  (623.0,543.5),  (622.5,544.0), 
  (622.5,545.0),  (622.5,546.0),  (622.0,546.5),  (621.5,547.0),  (621.5,548.0), 
  (621.5,549.0),  (621.5,550.0),  (621.0,550.5),  (620.5,551.0),  (620.0,551.5), 
  (619.5,552.0),  (619.0,552.5),  (618.5,553.0),  (618.5,554.0),  (618.0,554.5), 
  (617.5,555.0),  (617.5,556.0),  (617.0,556.5),  (616.5,557.0),  (616.5,558.0), 
  (616.5,559.0),  (616.0,559.5),  (615.5,560.0),  (615.5,561.0),  (615.0,561.5), 
  (614.5,562.0),  (614.5,563.0),  (614.5,564.0),  (614.5,565.0),  (614.5,566.0), 
  (614.5,567.0),  (614.5,568.0),  (614.5,569.0),  (614.5,570.0),  (615.0,570.5), 
  (615.5,571.0),  (615.5,572.0),  (616.0,572.5),  (616.5,573.0),  (616.5,574.0), 
  (617.0,574.5),  (617.5,575.0),  (618.0,575.5),  (619.0,575.5),  (619.5,576.0), 
  (620.0,576.5),  (620.5,577.0),  (620.0,577.5),  (619.0,577.5),  (618.5,578.0), 
  (619.0,578.5),  (620.0,578.5),  (621.0,578.5),  (622.0,578.5),  (622.5,578.0), 
  (622.5,577.0),  (622.0,576.5),  (621.5,576.0),  (621.0,575.5),  (620.5,575.0), 
  (620.0,574.5),  (619.5,574.0),  (619.0,573.5),  (618.5,573.0),  (618.0,572.5), 
  (617.5,572.0),  (617.5,571.0),  (617.0,570.5),  (616.5,570.0),  (616.0,569.5), 
  (615.5,569.0),  (616.0,568.5),  (616.5,568.0),  (616.5,567.0),  (617.0,566.5), 
  (617.5,566.0),  (618.0,565.5),  (618.5,565.0),  (618.5,564.0),  (618.5,563.0), 
  (619.0,562.5),  (620.0,562.5),  (621.0,562.5),  (622.0,562.5),  (622.5,563.0), 
  (623.0,563.5),  (624.0,563.5),  (624.5,563.0),  (625.0,562.5),  (625.5,562.0), 
  (625.5,561.0),  (625.5,560.0),  (625.5,559.0),  (625.5,558.0),  (626.0,557.5), 
  (626.5,557.0),  (627.0,556.5),  (628.0,556.5),  (629.0,556.5),  (630.0,556.5), 
  (631.0,556.5),  (632.0,556.5),  (633.0,556.5),  (633.5,556.0),  (634.0,555.5), 
  (634.5,556.0),  (635.0,556.5),  (636.0,556.5),  (636.5,557.0),  (637.0,557.5), 
  (638.0,557.5),  (639.0,557.5),  (639.5,558.0),  (640.0,558.5),  (641.0,558.5), 
  (642.0,558.5),  (642.5,558.0),  (642.5,557.0),  (642.0,556.5),  (641.5,556.0), 
  (641.0,555.5),  (640.0,555.5),  (639.5,555.0),  (639.5,554.0),  (639.0,553.5), 
  (638.5,553.0),  (638.0,552.5),  (637.5,552.0),  (638.0,551.5),  (639.0,551.5), 
  (639.5,552.0),  (640.0,552.5),  (640.5,552.0),  (640.5,551.0),  (640.5,550.0), 
  (641.0,549.5),  (641.5,549.0),  (641.5,548.0),  (642.0,547.5),  (642.5,547.0), 
  (642.5,546.0),  (642.5,545.0),  (643.0,544.5),  (643.5,544.0),  (644.0,543.5), 
  (644.5,543.0),  (644.5,542.0),  (645.0,541.5),  (645.5,541.0),  (645.5,540.0), 
  (646.0,539.5),  (646.5,539.0),  (647.0,538.5),  (647.5,538.0),  (647.5,537.0), 
  (648.0,536.5),  (649.0,536.5),  (649.5,536.0),  (649.5,535.0),  (650.0,534.5), 
  (650.5,534.0),  (650.5,533.0),  (651.0,532.5),  (651.5,532.0),  (651.0,531.5), 
  (650.5,531.0),  (651.0,530.5),  (652.0,530.5),  (652.5,530.0),  (652.5,529.0), 
  (652.5,528.0),  (653.0,527.5),  (654.0,527.5),  (654.5,527.0),  (655.0,526.5), 
  (656.0,526.5),  (656.5,526.0),  (656.5,525.0),  (656.5,524.0),  (657.0,523.5), 
  (657.5,523.0),  (657.5,522.0),  (658.0,521.5),  (658.5,521.0),  (658.5,520.0), 
  (659.0,519.5),  (659.5,519.0),  (659.5,518.0),  (659.5,517.0),  (659.5,516.0), 
  (659.5,515.0),  (660.0,514.5),  (660.5,514.0),  (660.5,513.0),  (660.5,512.0), 
  (660.5,511.0),  (661.0,510.5),  (661.5,510.0), 
]
    
    print("🚀 FIXED DIVIDED SPLINE COMPRESSION TEST")
    print("=" * 50)
    
    # Parameters - start with fewer sublists
    num_sublists = 50  # Start with 2 instead of 3
    compression_ratio = 0.3  # Use more conservative compression
    
    # Use the fixed version
    #result = compress_sublist_with_continuity(sample_coordinates, num_sublists, compression_ratio)
    result = compress_shape_divided_exact(sample_coordinates, num_sublists, compression_ratio)
    
    if result:
        # Print analysis
        print_divided_compression_analysis(result)
        
        overall = result['overall_metrics']
        print(f"\n🎯 FINAL RESULTS:")
        print(f"  Original points: {overall['total_original_points']}")
        print(f"  Key points: {overall['total_key_points']}")
        print(f"  Target ratio: {overall['compression_ratio']:.1%}")
        print(f"  Actual ratio: {overall['actual_compression_ratio']:.1%}")
        print(f"  Storage reduction: {overall['storage_reduction']}")
        print(f"  Reconstruction error: {overall['mean_error']:.6f}")

        visualize_divided_compression(sample_coordinates, result)
        
        # 1. Get optimal minimal storage (3 decimal places as requested)
        print("\n" + "="*60)
        print("💾 MINIMAL STORAGE WITH ROUNDING")
        print("="*60)
        
        compressed_data, storage_info = get_minimal_storage_with_rounding(
            result, decimal_places=3
        )
        
        print(f"Original data:")
        print(f"  - Points: {storage_info['original_points']:,}")
        print(f"  - Size: {storage_info['original_size_bytes']:,} bytes")
        
        print(f"\nCompressed data (rounded to 3 decimals):")
        print(f"  - Points: {storage_info['compressed_points']:,}")
        print(f"  - Size: {storage_info['rounded_size_bytes']:,} bytes")
        print(f"  - Compression ratio: {storage_info['compression_ratio']:.1%}")
        print(f"  - Storage reduction: {storage_info['storage_reduction']:.1f}%")
        
        # 3. Reconstruct from compressed data
        print("\n" + "="*60)
        print("🔄 RECONSTRUCTION FROM COMPRESSED DATA")
        print("="*60)
        
        # Use the EXACT same number of points as original for reconstruction
        reconstructed_boundary, spline_data = reconstruct_from_minimal_storage(
            compressed_data, 
            num_points=storage_info['original_points']  # This is key!
        )
        
        # Get original boundary - ensure we have the right one
        original_boundary = result['combined_reconstructed']
        
        # Ensure both arrays have the same length
        min_length = min(len(original_boundary), len(reconstructed_boundary))
        original_boundary = original_boundary[:min_length]
        reconstructed_boundary = reconstructed_boundary[:min_length]
        
        print(f"Length check - Original: {len(original_boundary)}, Reconstructed: {len(reconstructed_boundary)}")
        
        # Calculate final reconstruction quality
        final_error = np.mean(np.sqrt(
            (reconstructed_boundary[:, 0] - original_boundary[:, 0])**2 + 
            (reconstructed_boundary[:, 1] - original_boundary[:, 1])**2
        ))
        
        print(f"Reconstruction quality:")
        print(f"  - Original points: {len(original_boundary)}")
        print(f"  - Reconstructed points: {len(reconstructed_boundary)}")
        print(f"  - Reconstruction error: {final_error:.6f}")
        
        # 4. Visualize the results
        print("\n" + "="*60)
        print("📊 VISUALIZATION")
        print("="*60)
        
        visualize_minimal_storage_results(
            original_boundary, 
            compressed_data, 
            reconstructed_boundary, 
            storage_info
        )



        compressed_matrix=compressed_data

        original_boundary = result['combined_reconstructed']
        
        # Save the matrix
        save_compressed_matrix(compressed_matrix, "compressed_boundary")
        
        # Demonstrate reconstruction
        reconstructed, loaded_matrix = load_and_reconstruct("compressed_boundary", storage_info['original_points'])
        
        # Final analysis
        analyze_final_storage(result['combined_reconstructed'], compressed_matrix)

        # Ensure same length for comparison
        min_length = min(len(original_boundary), len(reconstructed))
        original_boundary = original_boundary[:min_length]
        reconstructed = reconstructed[:min_length]
        
        # Show all visualizations
        visualize_reconstruction_comparison(original_boundary, compressed_matrix, reconstructed)
        visualize_reconstruction_overlay(original_boundary, compressed_matrix, reconstructed)
        visualize_quality_metrics(original_boundary, reconstructed)

    

    else:
        print("❌ Compression failed!")












