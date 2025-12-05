import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt


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
