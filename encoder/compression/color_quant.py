import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import mark_boundaries
import scipy.fft

from encoder.ROI.roi import get_regions, extract_regions

from encoder.subregions.split_score import calculate_split_score, normalize_result
from encoder.subregions.slic import enhanced_slic_with_texture, extract_slic_segment_boundaries, visualize_split_analysis
from encoder.subregions.visualize import plot_regions
from encoder.interpolation.reconstruct import reconstruct_from_minimal_storage,save_compressed_matrix, load_and_reconstruct, analyze_final_storage



def coordinates_to_mask(boundary_coords, image_shape, plot_debug=True):
    """
    Convert boundary coordinates to mask - handles numpy array input.
    """
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    
    print(f"\n{'='*60}")
    print(f"DEBUG: coordinates_to_mask_fixed")
    print(f"{'='*60}")
    print(f"Input type: {type(boundary_coords)}")
    
    # Handle different input formats
    points_array = None
    
    # Case 1: boundary_coords is already a numpy array
    if isinstance(boundary_coords, np.ndarray):
        print(f"Input is numpy array with shape: {boundary_coords.shape}")
        points_array = boundary_coords.astype(np.float32)
    
    # Case 2: boundary_coords is a list containing numpy arrays
    elif isinstance(boundary_coords, list) and len(boundary_coords) > 0:
        print(f"Input is list with {len(boundary_coords)} elements")
        print(f"First element type: {type(boundary_coords[0])}")
        
        # If first element is a numpy array
        if isinstance(boundary_coords[0], np.ndarray):
            print(f"First element shape: {boundary_coords[0].shape}")
            
            # It could be a list of multiple arrays (for multiple contours)
            # Let's try to combine them
            all_points = []
            for i, arr in enumerate(boundary_coords):
                if isinstance(arr, np.ndarray):
                    print(f"  Array {i}: shape {arr.shape}, dtype {arr.dtype}")
                    # Check if it's 2D array of points
                    if len(arr.shape) == 2 and arr.shape[1] >= 2:
                        # Take first two columns (x, y)
                        all_points.append(arr[:, :2])
                    else:
                        print(f"  Warning: Unexpected array shape {arr.shape}")
            
            if all_points:
                # Combine all points
                points_array = np.vstack(all_points).astype(np.float32)
                print(f"Combined points shape: {points_array.shape}")
        
        # If first element is a list/tuple
        elif isinstance(boundary_coords[0], (list, tuple)):
            # Convert list of lists to numpy array
            try:
                points_array = np.array(boundary_coords, dtype=np.float32)
                print(f"Converted list to array with shape: {points_array.shape}")
            except:
                print("Failed to convert list to array")
    
    # If we still don't have points_array, try to extract from the nested structure
    if points_array is None:
        print(f"\nTrying to extract points from nested structure...")
        
        # Based on your error output, it looks like boundary_coords[0] is a 2D array
        # and boundary_coords[1] is a list of arrays
        if len(boundary_coords) >= 1 and isinstance(boundary_coords[0], np.ndarray):
            if len(boundary_coords[0].shape) == 2:
                # This is likely the points array
                points_array = boundary_coords[0].astype(np.float32)
                print(f"Extracted points from boundary_coords[0], shape: {points_array.shape}")
        
        # If that didn't work, try to flatten everything
        if points_array is None:
            print(f"Trying to flatten structure...")
            try:
                # Recursively flatten lists/arrays
                def flatten_coords(data):
                    points = []
                    if isinstance(data, np.ndarray):
                        if len(data.shape) == 2 and data.shape[1] >= 2:
                            return data[:, :2]
                        else:
                            return data.flatten()[:2] if data.size >= 2 else []
                    elif isinstance(data, (list, tuple)):
                        for item in data:
                            points.extend(flatten_coords(item))
                        return points
                    else:
                        return [data]
                
                flat_points = flatten_coords(boundary_coords)
                if len(flat_points) >= 2:
                    points_array = np.array(flat_points, dtype=np.float32).reshape(-1, 2)
                    print(f"Flattened to array shape: {points_array.shape}")
            except Exception as e:
                print(f"Flattening failed: {e}")
    
    if points_array is None or len(points_array) < 3:
        print(f"❌ ERROR: Could not extract valid points array")
        print(f"  points_array: {points_array}")
        if points_array is not None:
            print(f"  Shape: {points_array.shape}")
        return np.zeros(image_shape[:2], dtype=bool)
    
    print(f"\n✅ Successfully extracted points array")
    print(f"  Shape: {points_array.shape}")
    print(f"  First few points:")
    for i in range(min(5, len(points_array))):
        print(f"    [{i}] ({points_array[i, 0]:.1f}, {points_array[i, 1]:.1f})")
    
    # Now we have points_array with shape (N, 2)
    # But we need to check: are these (row, col) or (x, y)?
    # Let's check the range
    print(f"\nPoints range:")
    print(f"  Dimension 0 (likely rows): [{points_array[:, 0].min():.1f}, {points_array[:, 0].max():.1f}]")
    print(f"  Dimension 1 (likely cols): [{points_array[:, 1].min():.1f}, {points_array[:, 1].max():.1f}]")
    print(f"  Image shape: {image_shape} (rows, cols)")
    
    # Check if points need to be swapped
    # If dimension 0 range is much larger than image rows, they might be swapped
    if points_array[:, 0].max() > image_shape[0] * 10:  # Way too large for rows
        print(f"⚠️  Dimension 0 values seem too large for rows, might need swapping")
        # Try swapping
        points_swapped = points_array[:, [1, 0]]  # Swap columns
        print(f"  Swapped range - Rows: [{points_swapped[:, 0].min():.1f}, {points_swapped[:, 0].max():.1f}]")
        print(f"  Swapped range - Cols: [{points_swapped[:, 1].min():.1f}, {points_swapped[:, 1].max():.1f}]")
        
        # Use whichever seems more reasonable
        if points_swapped[:, 0].max() <= image_shape[0] * 1.5:  # More reasonable for rows
            print(f"  Using swapped coordinates (seems more reasonable)")
            points_array = points_swapped
    
    # Create empty mask
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    # Convert to integer for OpenCV
    # OpenCV expects points as (x, y) where x=col, y=row
    points_cv = points_array[:, [1, 0]].copy()  # Swap to (col, row) for OpenCV
    
    # Scale if necessary (points might be normalized or in wrong scale)
    print(f"\nPoints for OpenCV (col, row):")
    print(f"  Cols: [{points_cv[:, 0].min():.1f}, {points_cv[:, 0].max():.1f}]")
    print(f"  Rows: [{points_cv[:, 1].min():.1f}, {points_cv[:, 1].max():.1f}]")
    
    # Check if points need scaling
    if points_cv[:, 0].max() < 1.0 or points_cv[:, 1].max() < 1.0:
        print(f"⚠️  Points seem normalized (< 1.0), scaling to image size")
        points_cv[:, 0] *= image_shape[1]  # Scale columns
        points_cv[:, 1] *= image_shape[0]  # Scale rows
    
    # Convert to integers
    points_int = np.round(points_cv).astype(np.int32)
    
    # Clip to image bounds
    points_int[:, 0] = np.clip(points_int[:, 0], 0, image_shape[1] - 1)
    points_int[:, 1] = np.clip(points_int[:, 1], 0, image_shape[0] - 1)
    
    print(f"\nFinal integer points (col, row):")
    print(f"  Cols: [{points_int[:, 0].min():.1f}, {points_int[:, 0].max():.1f}]")
    print(f"  Rows: [{points_int[:, 1].min():.1f}, {points_int[:, 1].max():.1f}]")
    
    # Fill the polygon
    try:
        cv2.fillPoly(mask, [points_int], color=255)
        mask_true_count = np.sum(mask > 0)
        print(f"✅ Polygon filled successfully")
        print(f"Mask has {mask_true_count} True pixels")
    except Exception as e:
        print(f"❌ fillPoly failed: {e}")
        
        # Try convex hull
        try:
            hull = cv2.convexHull(points_int)
            cv2.fillPoly(mask, [hull], color=255)
            mask_true_count = np.sum(mask > 0)
            print(f"✅ Convex hull filled successfully")
            print(f"Convex hull has {mask_true_count} True pixels")
        except Exception as e2:
            print(f"❌ Convex hull also failed: {e2}")
            
            # Last resort: bounding box
            if len(points_int) > 0:
                min_col, min_row = points_int.min(axis=0)
                max_col, max_row = points_int.max(axis=0)
                
                min_col = max(0, min_col)
                min_row = max(0, min_row)
                max_col = min(image_shape[1] - 1, max_col)
                max_row = min(image_shape[0] - 1, max_row)
                
                if min_col < max_col and min_row < max_row:
                    mask[min_row:max_row+1, min_col:max_col+1] = 255
                    mask_true_count = np.sum(mask > 0)
                    print(f"✅ Using bounding box")
                    print(f"Bounding box: cols [{min_col}, {max_col}], rows [{min_row}, {max_row}]")
                    print(f"Bounding box has {mask_true_count} True pixels")
    
    # Convert to boolean
    mask_bool = mask.astype(bool)
    
    if plot_debug:
        # Create debug plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot 1: Original points
        axes[0].scatter(points_array[:, 1], points_array[:, 0], s=10, c='red')
        axes[0].plot(points_array[:, 1], points_array[:, 0], 'b-', alpha=0.5)
        axes[0].set_xlim(0, image_shape[1])
        axes[0].set_ylim(0, image_shape[0])
        axes[0].invert_yaxis()
        axes[0].set_title(f'Original Points ({len(points_array)} points)')
        axes[0].set_xlabel('Column')
        axes[0].set_ylabel('Row')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Mask
        axes[1].imshow(mask_bool, cmap='gray')
        axes[1].set_title(f'Generated Mask\n{np.sum(mask_bool):,} pixels')
        axes[1].axis('off')
        
        # Plot 3: Points on mask
        axes[2].imshow(mask_bool, cmap='gray')
        axes[2].scatter(points_cv[:, 0], points_cv[:, 1], s=10, c='red', alpha=0.7)
        axes[2].plot(points_cv[:, 0], points_cv[:, 1], 'b-', alpha=0.5, linewidth=1)
        axes[2].set_title('Points Overlay')
        axes[2].axis('off')
        axes[2].set_xlim(0, image_shape[1])
        axes[2].set_ylim(image_shape[0], 0)  # Invert y-axis for image coordinates
        
        plt.suptitle('Debug: Boundary to Mask Conversion', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    return mask_bool















def compress_roi_content_only(roi_mask, region_image, quality=75):
    """
    Compress ONLY the ROI content (no borders, minimal metadata)
    Returns: compressed_data, compression_ratio, psnr
    """
    # Get ROI pixels only
    roi_pixels = region_image[roi_mask]
    original_size_bytes = len(roi_pixels) * 3  # RGB bytes
    
    print(f"  ROI Content:")
    print(f"    Pixels: {len(roi_pixels):,}")
    print(f"    Original size: {original_size_bytes:,} bytes")
    
    # Convert to normalized 0-1 range
    roi_float = roi_pixels.astype(np.float32) / 255.0
    
    # Check if image is too dark (common issue)
    avg_brightness = np.mean(roi_float)
    print(f"    Average brightness: {avg_brightness:.3f} (0-1 scale)")
    
    # If image is too dark, apply brightness correction for DCT
    if avg_brightness < 0.2:
        print(f"    ⚠️ Image is very dark. Applying brightness boost for DCT...")
        brightness_boost = 0.2 / avg_brightness if avg_brightness > 0 else 2.0
        roi_float = np.clip(roi_float * brightness_boost, 0, 1)
        print(f"    Brightness boosted by {brightness_boost:.1f}x")
    
    # Get unique colors in ROI (for simple compression estimate)
    unique_colors = len(np.unique(roi_pixels.reshape(-1, 3), axis=0))
    print(f"    Unique colors: {unique_colors:,}")
    
    # SIMPLE DCT-based compression (content only, no borders)
    compressed_blocks = []
    
    # Process ROI in 8x8 blocks (aligned to image grid)
    block_size = 8
    height, width = region_image.shape[:2]
    
    # Find all 8x8 blocks that contain ROI pixels
    roi_blocks = []
    
    for i in range(0, height - block_size + 1, block_size):
        for j in range(0, width - block_size + 1, block_size):
            # Check if this block contains any ROI pixels
            block_mask = roi_mask[i:i+block_size, j:j+block_size]
            if np.any(block_mask):
                roi_blocks.append((i, j, block_mask))
    
    print(f"    Processing {len(roi_blocks)} content blocks...")
    
    # Simple compression: store only blocks with content
    total_compressed_bytes = 0
    compressed_coefficients = []
    
    for block_idx, (i, j, block_mask) in enumerate(roi_blocks):
        block_data = []
        
        for channel in range(3):
            # Extract channel data
            channel_block = region_image[i:i+block_size, j:j+block_size, channel].astype(np.float32) / 255.0
            
            # Apply DCT
            dct_block = scipy.fft.dctn(channel_block, norm='ortho')
            
            # SIMPLE adaptive quantization based on block brightness
            block_mean = np.mean(channel_block[block_mask])
            
            # Gentle quantization table
            if block_mean < 0.1:  # Very dark block
                quant_factor = 0.01  # Keep almost everything
            elif block_mean < 0.3:  # Dark block
                quant_factor = 0.05
            else:  # Normal block
                quant_factor = 0.1 + (1.0 - quality/100) * 0.9
            
            # Quantize (keep more coefficients for dark blocks)
            quantized = np.round(dct_block / quant_factor)
            
            # Count non-zero coefficients
            non_zero = np.sum(quantized != 0)
            
            # Simple entropy estimate: store only non-zero coefficients
            # Format: (value, position) pairs
            if non_zero > 0:
                # Find non-zero positions and values
                nonzero_pos = np.where(quantized != 0)
                nonzero_vals = quantized[nonzero_pos]
                
                # Each coefficient: 2 bytes for position + 2 bytes for value = 4 bytes
                block_compressed_size = non_zero * 4
                total_compressed_bytes += block_compressed_size
                
                block_data.append({
                    'channel': channel,
                    'non_zero': non_zero,
                    'positions': list(zip(nonzero_pos[0], nonzero_pos[1])),
                    'values': nonzero_vals.astype(np.int16).tolist(),
                    'quant_factor': quant_factor
                })
        
        if block_data:
            compressed_coefficients.append({
                'position': (i, j),
                'block_size': block_size,
                'channels': block_data
            })
    
    # Calculate compression ratio (CONTENT ONLY)
    compression_ratio = original_size_bytes / total_compressed_bytes if total_compressed_bytes > 0 else 0
    
    print(f"  Compression (content only):")
    print(f"    Compressed size: {total_compressed_bytes:,} bytes")
    print(f"    Ratio: {compression_ratio:.2f}:1")
    
    # Reconstruct to calculate PSNR
    reconstructed = reconstruct_roi_content(compressed_coefficients, region_image.shape, roi_mask)
    
    # Calculate PSNR
    psnr = calculate_psnr_simple(region_image, reconstructed, roi_mask)
    
    print(f"    Quality: {psnr:.2f} dB PSNR")
    
    return {
        'compressed_coefficients': compressed_coefficients,
        'original_size': original_size_bytes,
        'compressed_size': total_compressed_bytes,
        'compression_ratio': compression_ratio,
        'psnr': psnr,
        'roi_mask': roi_mask
    }

def calculate_content_compression_stats(compression_result):
    """
    Calculate meaningful compression statistics for ROI content
    """
    original = compression_result['original_size']
    compressed = compression_result['compressed_size']
    ratio = compression_result['compression_ratio']
    psnr = compression_result['psnr']
    
    print(f"\n  CONTENT COMPRESSION SUMMARY:")
    print(f"  {'Metric':<20} {'Value':<15} {'Assessment':<20}")
    print(f"  {'-'*20} {'-'*15} {'-'*20}")
    
    # Original size
    print(f"  {'Original size':<20} {original:,} bytes {'':<20}")
    
    # Compressed size
    print(f"  {'Compressed size':<20} {compressed:,} bytes {'':<20}")
    
    # Compression ratio
    if ratio > 1.0:
        assessment = f"✅ Compression achieved"
        space_savings = (1 - compressed/original) * 100
        print(f"  {'Compression ratio':<20} {ratio:.2f}:1 {assessment}")
        print(f"  {'Space savings':<20} {space_savings:.1f}% {'':<20}")
    else:
        assessment = f"❌ File size INCREASED"
        increase = (compressed/original - 1) * 100
        print(f"  {'Compression ratio':<20} {ratio:.2f}:1 {assessment}")
        print(f"  {'Size increase':<20} {increase:.1f}% {'':<20}")
    
    # Quality
    if psnr > 40:
        quality_assess = "✅ Excellent quality"
    elif psnr > 35:
        quality_assess = "✓ Good quality"
    elif psnr > 30:
        quality_assess = "⚠️ Acceptable"
    elif psnr > 25:
        quality_assess = "❌ Poor quality"
    else:
        quality_assess = "❌❌ Very poor"
    
    print(f"  {'PSNR':<20} {psnr:.2f} dB {quality_assess}")
    
    # Bits per pixel (bpp)
    roi_pixels = np.sum(compression_result['roi_mask'])
    bpp = (compressed * 8) / roi_pixels if roi_pixels > 0 else 0
    print(f"  {'Bits per pixel':<20} {bpp:.2f} bpp {'':<20}")
    
    return {
        'original_bytes': original,
        'compressed_bytes': compressed,
        'ratio': ratio,
        'psnr': psnr,
        'bpp': bpp
    }

def reconstruct_roi_content(compressed_coefficients, image_shape, roi_mask):
    """
    Reconstruct ROI from compressed coefficients
    """
    height, width, _ = image_shape
    reconstructed = np.zeros((height, width, 3), dtype=np.float32)
    
    for block_info in compressed_coefficients:
        i, j = block_info['position']
        block_size = block_info['block_size']
        
        for channel_data in block_info['channels']:
            channel = channel_data['channel']
            quant_factor = channel_data['quant_factor']
            
            # Create empty DCT block
            dct_block = np.zeros((block_size, block_size), dtype=np.float32)
            
            # Fill with reconstructed coefficients
            for (pos_i, pos_j), value in zip(channel_data['positions'], channel_data['values']):
                dct_block[pos_i, pos_j] = value * quant_factor
            
            # Inverse DCT
            recon_block = scipy.fft.idctn(dct_block, norm='ortho')
            
            # Clip to valid range and place in reconstruction
            recon_block = np.clip(recon_block, 0, 1)
            reconstructed[i:i+block_size, j:j+block_size, channel] = recon_block
    
    # Convert back to 0-255
    reconstructed_uint8 = (reconstructed * 255).astype(np.uint8)
    
    return reconstructed_uint8

def calculate_psnr_for_region_correct_1(original_region, reconstructed_region, roi_mask):
    """
    CORRECT PSNR calculation - compares ONLY ROI pixels
    """
    # Get ONLY the pixels within the ROI mask
    original_roi_pixels = original_region[roi_mask]
    reconstructed_roi_pixels = reconstructed_region[roi_mask]
    
    if len(original_roi_pixels) == 0 or len(reconstructed_roi_pixels) == 0:
        return 0
    
    # Calculate MSE and PSNR ONLY for ROI pixels
    mse = np.mean((original_roi_pixels.astype(np.float32) - reconstructed_roi_pixels.astype(np.float32)) ** 2)
    
    if mse > 0:
        psnr = 10 * np.log10(255**2 / mse)
    else:
        psnr = float('inf')
    
    return psnr

def calculate_psnr_for_region_correct(original, reconstructed, mask):
    """
    Robust PSNR calculation with error handling.
    
    Args:
        original: Original image (uint8)
        reconstructed: Reconstructed image (uint8)
        mask: Optional boolean mask
    
    Returns:
        PSNR in dB
    """
    # Ensure images are same shape and type
    if original.shape != reconstructed.shape:
        raise ValueError(f"Shape mismatch: {original.shape} vs {reconstructed.shape}")
    
    # Convert to float for calculation
    original_float = original.astype(np.float64)
    reconstructed_float = reconstructed.astype(np.float64)
    
    # Apply mask if provided
    if mask is not None:
        if mask.shape != original.shape[:2]:
            raise ValueError(f"Mask shape mismatch: {mask.shape} vs {original.shape[:2]}")
        
        # Get indices of masked pixels
        rows, cols = np.where(mask)
        
        if len(rows) == 0:
            return float('inf')  # No pixels to compare
        
        # Extract only masked pixels from all channels
        original_masked = original_float[rows, cols, :]
        reconstructed_masked = reconstructed_float[rows, cols, :]
        
        # Flatten for MSE calculation
        original_flat = original_masked.flatten()
        reconstructed_flat = reconstructed_masked.flatten()
        
        mse = np.mean((original_flat - reconstructed_flat) ** 2)
    else:
        # Consider all pixels
        original_flat = original_float.flatten()
        reconstructed_flat = reconstructed_float.flatten()
        mse = np.mean((original_flat - reconstructed_flat) ** 2)
    
    # Handle zero MSE
    if mse == 0:
        return float('inf')
    
    # PSNR calculation (max value is 255 for uint8)
    psnr = 10 * np.log10((255.0 ** 2) / mse)
    return psnr








































import cv2
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
from scipy import interpolate


def fill_black_pixels_with_neighbors(image, mask, reference_image=None):
    """
    Fill black pixels using neighborhood information.
    """
    result = image.copy()
    black_pixels = np.all(image < 10, axis=2) & mask
    
    if not np.any(black_pixels):
        return result
    
    # Get coordinates of black pixels
    black_coords = np.argwhere(black_pixels)
    
    for y, x in black_coords:
        # Define neighborhood (3x3 window)
        y_min = max(0, y - 1)
        y_max = min(image.shape[0], y + 2)
        x_min = max(0, x - 1)
        x_max = min(image.shape[1], x + 2)
        
        # Get non-black pixels in neighborhood
        neighborhood = image[y_min:y_max, x_min:x_max]
        non_black = np.any(neighborhood >= 10, axis=2)
        
        if np.any(non_black):
            # Average of non-black neighbors
            valid_pixels = neighborhood[non_black]
            result[y, x] = np.mean(valid_pixels, axis=0).astype(np.uint8)
        elif reference_image is not None:
            # Fallback to reference image
            result[y, x] = reference_image[y, x]
    
    return result

def create_edge_blending_mask(mask, blend_width=3):
    """
    Create a mask for edge blending.
    """
    # Create distance transform from mask boundary
    mask_boundary = binary_dilation(mask) & ~binary_erosion(mask)
    
    # Compute distance from boundary
    distance = distance_transform_edt(~mask_boundary)
    
    # Create smooth blending mask (1.0 at center, 0.0 at edges)
    blend_mask = np.clip(distance / blend_width, 0, 1)
    blend_mask = blend_mask[:, :, np.newaxis]  # Add channel dimension
    
    return blend_mask

def blend_edges(reconstructed, original, edge_mask, blend_width=3):
    """
    Blend edges between reconstructed and original image.
    """
    blended = edge_mask * reconstructed + (1 - edge_mask) * original
    return blended.astype(np.uint8)


































# ==============================================
# SHARED QUANTIZATION TABLE OPTIMIZATION
# ==============================================

def analyze_subregions_frequency_content(region_image, segment_boundaries, roi_segments, bbox_mask):
    """
    Analyze frequency content of all subregions to design optimal shared quantization table.
    Returns: frequency statistics and recommended quantization table
    """
    print(f"  Analyzing frequency content of {len(segment_boundaries)} subregions...")
    
    # Initialize frequency statistics
    dct_coefficients = []
    block_count = 0
    
    # Standard JPEG luminance quantization table (will be adapted)
    std_qtable = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ], dtype=np.float32)
    
    # Collect DCT coefficients from all subregions
    for seg_idx, seg_data in enumerate(segment_boundaries[:20]):  # Sample first 20 segments
        segment_id = seg_data.get('segment_id', seg_idx)
        segment_mask = (roi_segments == segment_id) & bbox_mask
        
        segment_pixels = np.sum(segment_mask)
        if segment_pixels < 64:  # Need at least 8x8 block
            continue
        
        # Extract region
        rows, cols = np.where(segment_mask)
        min_r, max_r = rows.min(), rows.max()
        min_c, max_c = cols.min(), cols.max()
        
        # Get region image
        region_crop = region_image[min_r:max_r+1, min_c:max_c+1]
        mask_crop = segment_mask[min_r:max_r+1, min_c:max_c+1]
        
        # Process 8x8 blocks
        h, w = region_crop.shape[:2]
        for i in range(0, h-7, 8):
            for j in range(0, w-7, 8):
                # Check if block has enough masked pixels
                block_mask = mask_crop[i:i+8, j:j+8]
                if np.sum(block_mask) < 32:  # At least half the block
                    continue
                
                # Get block
                block = region_crop[i:i+8, j:j+8]
                
                # Convert to YCbCr and take Y channel (luminance)
                if block.shape[2] == 3:
                    block_ycbcr = cv2.cvtColor(block, cv2.COLOR_RGB2YCrCb)
                    block_y = block_ycbcr[:, :, 0].astype(np.float32) - 128
                else:
                    block_y = block[:, :, 0].astype(np.float32) - 128
                
                # Apply DCT
                dct_block = cv2.dct(block_y)
                dct_coefficients.append(dct_block)
                block_count += 1
    
    print(f"    Analyzed {block_count} 8x8 blocks")
    
    if block_count == 0:
        print(f"    Warning: No valid blocks found, using standard table")
        return std_qtable
    
    # Analyze coefficient statistics
    dct_coefficients = np.array(dct_coefficients)
    
    # Calculate average magnitude for each frequency
    avg_magnitude = np.mean(np.abs(dct_coefficients), axis=0)
    
    # Normalize (DC coefficient is much larger)
    dc_value = avg_magnitude[0, 0]
    normalized_magnitude = avg_magnitude / dc_value if dc_value > 0 else avg_magnitude
    
    # Design adaptive quantization table
    # Principle: Preserve frequencies with high average magnitude, discard others
    adaptive_qtable = std_qtable.copy()
    
    # Adjust based on frequency importance
    for i in range(8):
        for j in range(8):
            freq_importance = normalized_magnitude[i, j]
            
            # If this frequency is important (carries texture/color info), use milder quantization
            if freq_importance > 0.1:  # Threshold
                adaptive_qtable[i, j] = max(1, std_qtable[i, j] * (1.0 - freq_importance))
            else:
                # Unimportant frequency - stronger quantization
                adaptive_qtable[i, j] = std_qtable[i, j] * 2.0
    
    # Ensure DC coefficient has minimal quantization (preserve base color)
    adaptive_qtable[0, 0] = max(1, adaptive_qtable[0, 0] * 0.5)
    
    # Clip to reasonable range
    adaptive_qtable = np.clip(adaptive_qtable, 1, 255).astype(np.uint8)
    
    print(f"    Designed adaptive quantization table")
    print(f"    DC coefficient quantization: {adaptive_qtable[0, 0]} (vs standard {std_qtable[0, 0]})")
    
    return adaptive_qtable




"""def fill_border_gaps(reconstructed_image, segment_mask, region_image):
    # Create a binary mask of black pixels (all channels near 0)
    black_mask = np.all(reconstructed_image < 10, axis=2) & segment_mask
    
    if not np.any(black_mask):
        return reconstructed_image
    
    # Use morphological dilation to find border regions
    from scipy import ndimage
    
    # Create kernel for dilation
    kernel = np.ones((3, 3), dtype=bool)
    
    # Dilate the non-black area within the mask
    non_black_mask = ~black_mask & segment_mask
    dilated_non_black = ndimage.binary_dilation(non_black_mask, structure=kernel, iterations=2)
    
    # Find border pixels that are black but adjacent to non-black
    border_pixels = black_mask & dilated_non_black
    
    # For each border pixel, take average of neighboring non-black pixels
    rows, cols = np.where(border_pixels)
    
    for r, c in zip(rows, cols):
        # Get 3x3 neighborhood
        r_start = max(0, r-1)
        r_end = min(reconstructed_image.shape[0], r+2)
        c_start = max(0, c-1)
        c_end = min(reconstructed_image.shape[1], c+2)
        
        # Get non-black neighbors within the mask
        neighborhood = reconstructed_image[r_start:r_end, c_start:c_end]
        neighbor_mask = ~np.all(neighborhood < 10, axis=2) & segment_mask[r_start:r_end, c_start:c_end]
        
        if np.any(neighbor_mask):
            # Average of valid neighbors
            valid_pixels = neighborhood[neighbor_mask]
            avg_color = np.mean(valid_pixels, axis=0).astype(np.uint8)
            reconstructed_image[r, c] = avg_color
        else:
            # No valid neighbors, use original image
            reconstructed_image[r, c] = region_image[r, c]
    
    return reconstructed_image
"""
def calculate_minimal_compressed_size(seg_compression):
    """
    Calculate ACTUAL minimal storage size for a subregion.
    Only counts what we actually need to store.
    """
    # 1. ext_bbox: 4 integers × 2 bytes = 8 bytes
    size = 8
    
    # 2. For each block
    for block_data in seg_compression['compressed_blocks']:
        # Block position: i, j (2 × 2 bytes = 4 bytes)
        size += 4
        
        block_compressed = block_data['data']
        
        # 3. For each channel (Y, Cb, Cr)
        for ch in range(3):
            channel_key = f'channel_{ch}'
            if channel_key in block_compressed:
                channel_data = block_compressed[channel_key]
                
                # Number of coefficients: 1 byte
                size += 1
                
                num_coeffs = len(channel_data.get('values', []))
                
                # For each coefficient: index (1 byte) + value (2 bytes) = 3 bytes
                size += num_coeffs * 3
    
    return size


"""def fill_border_gaps(final_image, reconstructed_mask, kernel_size=3):

    from scipy import ndimage
    import cv2
    
    # Convert mask to uint8 for OpenCV
    mask_uint8 = (reconstructed_mask * 255).astype(np.uint8)
    
    # Create kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Apply morphological closing to fill small holes
    closed_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    
    # Convert back to boolean
    filled_mask = closed_mask > 0
    
    # Count how many pixels were added
    added_pixels = np.sum(filled_mask & ~reconstructed_mask)
    print(f"  Morphological closing added {added_pixels:,} pixels")
    
    if added_pixels == 0:
        return final_image, reconstructed_mask
    
    # For newly added pixels, fill with average of neighbors
    filled_image = final_image.copy()
    new_pixel_mask = filled_mask & ~reconstructed_mask
    
    if np.any(new_pixel_mask):
        # Use distance transform to fill from nearest valid pixel
        from scipy.ndimage import distance_transform_edt
        
        # Get coordinates of valid pixels
        valid_rows, valid_cols = np.where(reconstructed_mask)
        
        if len(valid_rows) > 0:
            # For each new pixel, find nearest valid pixel
            new_rows, new_cols = np.where(new_pixel_mask)
            
            for i, (r, c) in enumerate(zip(new_rows, new_cols)):
                # Find nearest valid pixel (simple 3x3 neighborhood)
                r_min = max(0, r-1)
                r_max = min(final_image.shape[0], r+2)
                c_min = max(0, c-1)
                c_max = min(final_image.shape[1], c+2)
                
                neighborhood = reconstructed_mask[r_min:r_max, c_min:c_max]
                if np.any(neighborhood):
                    # Get colors from valid neighbors
                    neighbor_colors = final_image[r_min:r_max, c_min:c_max][neighborhood]
                    avg_color = np.mean(neighbor_colors, axis=0).astype(np.uint8)
                    filled_image[r, c] = avg_color
            
            print(f"  Filled {len(new_rows):,} new pixels with neighbor colors")
    
    return filled_image, filled_mask

"""


def fill_all_gaps(reconstructed_image, segment_mask, region_image, max_gap_distance=3):
    """
    Fill all gaps: both holes within mask AND gaps between segments.
    """
    from scipy import ndimage
    import numpy as np
    
    # 1. Find black pixels in the reconstructed area
    black_threshold = 10
    black_pixels = np.all(reconstructed_image < black_threshold, axis=2)
    
    # Pixels that should be filled but are black
    pixels_to_fill = black_pixels & segment_mask
    
    if not np.any(pixels_to_fill):
        return reconstructed_image
    
    print(f"    Found {np.sum(pixels_to_fill):,} black pixels to fill")
    
    # 2. Create distance transform to find nearest valid pixel
    valid_pixels = ~black_pixels & segment_mask
    
    if not np.any(valid_pixels):
        # No valid pixels, fill with original image
        reconstructed_image[segment_mask] = region_image[segment_mask]
        return reconstructed_image
    
    # 3. For each black pixel, find nearest valid pixel within max_gap_distance
    # Create kernel for neighborhood search
    kernel_size = max_gap_distance * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), dtype=bool)
    
    # Dilate valid pixels to cover gaps
    dilated_valid = ndimage.binary_dilation(valid_pixels, structure=kernel)
    
    # Pixels that can be filled (black pixels near valid pixels)
    fillable_pixels = pixels_to_fill & dilated_valid
    
    if not np.any(fillable_pixels):
        print(f"    No fillable pixels found within {max_gap_distance} pixels")
        return reconstructed_image
    
    # 4. Fill pixels using nearest neighbor interpolation
    filled_image = reconstructed_image.copy()
    rows_to_fill, cols_to_fill = np.where(fillable_pixels)
    
    # Get coordinates and colors of valid pixels
    valid_rows, valid_cols = np.where(valid_pixels)
    valid_coords = np.column_stack([valid_rows, valid_cols])
    valid_colors = reconstructed_image[valid_rows, valid_cols]
    
    # Create KDTree for fast nearest neighbor search
    from scipy.spatial import KDTree
    tree = KDTree(valid_coords)
    
    # For each pixel to fill, find nearest valid pixel
    for i, (r, c) in enumerate(zip(rows_to_fill, cols_to_fill)):
        # Find k nearest valid pixels
        distances, indices = tree.query([r, c], k=min(4, len(valid_coords)))
        
        # Weighted average based on distance (closer = more weight)
        weights = 1.0 / (distances + 1e-6)  # Add small epsilon to avoid division by zero
        weights = weights / np.sum(weights)
        
        # Calculate weighted average color
        weighted_color = np.zeros(3, dtype=np.float32)
        for j, idx in enumerate(indices):
            weighted_color += valid_colors[idx] * weights[j]
        
        filled_image[r, c] = np.clip(weighted_color, 0, 255).astype(np.uint8)
    
    print(f"    Filled {len(rows_to_fill):,} pixels")
    return filled_image



def fill_all_holes_in_mask(mask):
    """
    Fill all holes in a binary mask.
    """
    from scipy import ndimage
    
    # Label background (False) regions
    labeled_bg, num_bg = ndimage.label(~mask)
    
    # The largest background region is the outside (should not be filled)
    bg_sizes = []
    for bg_id in range(1, num_bg + 1):
        bg_size = np.sum(labeled_bg == bg_id)
        bg_sizes.append(bg_size)
    
    if not bg_sizes:
        return mask
    
    # Find largest background region (the outside)
    largest_bg_id = np.argmax(bg_sizes) + 1
    
    # Fill all other background regions (holes)
    filled_mask = mask.copy()
    
    for bg_id in range(1, num_bg + 1):
        if bg_id != largest_bg_id:  # Skip the outside
            hole_mask = (labeled_bg == bg_id)
            filled_mask[hole_mask] = True
    
    holes_filled = np.sum(filled_mask & ~mask)
    print(f"  Filled {holes_filled:,} hole pixels")
    
    return filled_mask










def calculate_psnr_simple(original, reconstructed, mask):
    """
    Simple PSNR calculation for ROI area only
    """
    original_pixels = original[mask]
    recon_pixels = reconstructed[mask]
    
    if len(original_pixels) == 0:
        return 0
    
    mse = np.mean((original_pixels.astype(np.float32) - recon_pixels.astype(np.float32)) ** 2)
    
    if mse > 0:
        psnr = 10 * np.log10(255**2 / mse)
    else:
        psnr = float('inf')
    
    return psnr














































































"""
def compress_cropped_segment_with_shared_qtable(cropped_segment, top_left_coords, shared_qtable, quality_factor=75):

    if cropped_segment.size == 0:
        return None
    
    h, w = cropped_segment.shape[:2]
    num_pixels = h * w
    
    # Create transparency mask (non-black pixels)
    # IMPORTANT: We consider a pixel "non-black" if ANY channel > threshold
    transparency_mask = np.any(cropped_segment > 10, axis=2)  # Threshold to avoid noise
    
    # Count only actual content pixels (non-black)
    content_pixels = np.sum(transparency_mask)
    original_size = content_pixels * 3  # Only count content pixels
    
    print(f"      Crop size: {h}x{w} ({num_pixels:,} total, {content_pixels:,} content)")
    
    # Prepare compressed data structure
    compressed_data = {
        'method': 'dct_shared_qtable_cropped',
        'original_size': original_size,
        'cropped_shape': (h, w),
        'top_left_coords': top_left_coords,  # (row, col) in full image
        'transparency_mask': transparency_mask.astype(np.uint8),  # Store mask
        'content_pixels': int(content_pixels),
        'compressed_blocks': [],
        'use_shared_qtable': True,
        'quality_factor': quality_factor
    }
    
    # Adjust quantization table
    scale_factor = 1.0
    if quality_factor < 50:
        scale_factor = 50.0 / quality_factor
    else:
        scale_factor = (100 - quality_factor) / 50.0
    
    qtable_scaled = np.clip(np.round(shared_qtable * scale_factor), 1, 255).astype(np.uint8)
    
    # Process 8x8 blocks in the cropped segment
    total_blocks = 0
    compressed_size = 0
    
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            i_end = min(i + 8, h)
            j_end = min(j + 8, w)
            block_height = i_end - i
            block_width = j_end - j
            
            # Check if block contains any content pixels (non-black)
            block_mask = transparency_mask[i:i_end, j:j_end]
            if np.sum(block_mask) == 0:
                continue  # Skip all-black blocks
            
            # Get block
            block = cropped_segment[i:i_end, j:j_end]
            
            # Pad if necessary
            if block_height < 8 or block_width < 8:
                block_padded = np.zeros((8, 8, 3), dtype=np.uint8)
                block_padded[:block_height, :block_width] = block
                block = block_padded
            
            # Convert to YCbCr
            block_ycbcr = cv2.cvtColor(block, cv2.COLOR_RGB2YCrCb)
            
            compressed_block = {}
            
            # Process each channel
            for ch in range(3):
                channel_data = block_ycbcr[:, :, ch].astype(np.float32) - 128
                dct_coeffs = cv2.dct(channel_data)
                quantized = np.round(dct_coeffs / qtable_scaled).astype(np.int16)
                
                # Store non-zero coefficients
                non_zero_mask = quantized != 0
                non_zero_coeffs = quantized[non_zero_mask]
                
                if len(non_zero_coeffs) > 0:
                    positions = np.argwhere(non_zero_mask).tolist()
                    values = non_zero_coeffs.tolist()
                    
                    compressed_block[f'channel_{ch}'] = {
                        'positions': positions,
                        'values': values,
                        'num_coeffs': len(values)
                    }
                    
                    compressed_size += len(values) * 4  # 2 bytes per position + 2 per value
                else:
                    compressed_block[f'channel_{ch}'] = {
                        'positions': [],
                        'values': [],
                        'num_coeffs': 0
                    }
            
            compressed_data['compressed_blocks'].append({
                'position': (i, j),  # Position within cropped image
                'data': compressed_block,
                'block_size': (block_height, block_width)
            })
            total_blocks += 1
    
    compressed_size += 100  # Metadata overhead (mask, coords, etc.)
    
    compressed_data['compressed_size'] = compressed_size
    compressed_data['compression_ratio'] = original_size / compressed_size if compressed_size > 0 else 0
    compressed_data['num_blocks'] = total_blocks
    
    print(f"      Blocks to compress: {total_blocks}")
    
    return compressed_data
"""

import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans

def compress_small_region_color_quantization(region_image, top_left_coords, quality_factor=50):
    """
    Compress small region using color quantization with quality-to-colors mapping.
    """
    if region_image is None or region_image.size == 0:
        return None
    
    h, w, _ = region_image.shape
    total_pixels = h * w
    
    print(f"Compressing {h}x{w} region with color quantization")
    print(f"Top-left: {top_left_coords}")
    
    # ==============================================
    # 1. QUALITY FACTOR TO NUMBER OF COLORS MAPPING
    # ==============================================
    # Quality factor controls number of colors in palette
    # Higher quality = more colors = better fidelity = larger size
    
    # Map quality (0-100) to number of colors (2-256)
    if quality_factor <= 10:
        n_colors = 2  # Binary image
    elif quality_factor <= 30:
        n_colors = 4  # Very low quality
    elif quality_factor <= 50:
        n_colors = 8  # Low quality
    elif quality_factor <= 70:
        n_colors = 16  # Medium quality
    elif quality_factor <= 85:
        n_colors = 32  # Good quality
    elif quality_factor <= 95:
        n_colors = 64  # High quality
    else:
        n_colors = 128  # Very high quality
    
    # Cap based on actual pixel count (can't have more colors than pixels)
    unique_colors = len(np.unique(region_image.reshape(-1, 3), axis=0))
    n_colors = min(n_colors, unique_colors, 256)
    
    print(f"Quality {quality_factor} → {n_colors} colors (max possible: {unique_colors})")
    
    # ==============================================
    # 2. COLOR QUANTIZATION USING K-MEANS
    # ==============================================
    # Reshape image to list of pixels
    pixels = region_image.reshape(-1, 3).astype(np.float32)
    
    # Special case: Single color region
    if n_colors == 1 or unique_colors == 1:
        avg_color = np.mean(pixels, axis=0).astype(np.uint8)
        
        compressed_data = {
            'method': 'color_quantization',
            'top_left': top_left_coords,
            'shape': (h, w),
            'palette': [avg_color.tolist()],  # Single color palette
            'indices': [0] * total_pixels,  # All indices are 0
            'n_colors': 1,
            'quality': quality_factor,
            'original_size': total_pixels * 3
        }
        
        # Calculate compressed size
        compressed_size = (
            3 +  # 1 color * 3 bytes
            1 * total_pixels +  # 1 byte per pixel for index (could be optimized)
            50  # Metadata overhead
        )
        compressed_data['compressed_size'] = compressed_size
        
        print(f"Single color region: {avg_color}")
        return compressed_data
    
    # Apply K-means clustering for color quantization
    kmeans = MiniBatchKMeans(
        n_clusters=n_colors,
        random_state=42,
        batch_size=min(100, len(pixels)),
        max_iter=20
    )
    
    # Fit and predict
    labels = kmeans.fit_predict(pixels)
    
    # Get palette (cluster centers)
    palette = kmeans.cluster_centers_.astype(np.uint8)
    
    # ==============================================
    # 3. ENCODE INDICES EFFICIENTLY
    # ==============================================
    # Reshape labels back to image shape
    index_map = labels.reshape(h, w)
    
    # Choose optimal data type for indices
    if n_colors <= 256:
        index_dtype = np.uint8
        bytes_per_index = 1
    elif n_colors <= 65536:
        index_dtype = np.uint16
        bytes_per_index = 2
    else:
        index_dtype = np.uint32
        bytes_per_index = 4
    
    index_map = index_map.astype(index_dtype)
    
    # Option 1: Simple flatten (good for small images)
    indices_flat = index_map.flatten().tolist()
    
    # Option 2: Run-length encoding for better compression
    # (Uncomment if your regions have large uniform areas)
    # indices_rle = run_length_encode(index_map.flatten())
    
    # ==============================================
    # 4. CALCULATE COMPRESSION STATS
    # ==============================================
    original_size = total_pixels * 3  # RGB bytes
    
    # Compressed size calculation:
    # - Palette: n_colors * 3 bytes
    # - Indices: total_pixels * bytes_per_index
    # - Metadata: ~50 bytes
    palette_size = n_colors * 3
    indices_size = total_pixels * bytes_per_index
    metadata_size = 50
    
    compressed_size = palette_size + indices_size + metadata_size
    
    # ==============================================
    # 5. CREATE COMPRESSED DATA STRUCTURE
    # ==============================================
    compressed_data = {
        'method': 'color_quantization',
        'top_left': top_left_coords,  # Essential for placement
        'shape': (h, w),              # Essential for reconstruction
        'palette': palette.tolist(),  # Color palette
        'indices': indices_flat,      # Pixel indices
        # 'indices_rle': indices_rle,  # Alternative: RLE encoded
        'n_colors': n_colors,
        'quality': quality_factor,
        'index_dtype': str(index_dtype),
        'original_size': original_size,
        'compressed_size': compressed_size,
        'compression_ratio': original_size / compressed_size if compressed_size > 0 else 0
    }
    
    # ==============================================
    # 6. VALIDATE AND ADD VISUALIZATION
    # ==============================================
    # Quick reconstruction for validation
    reconstructed = palette[labels].reshape(h, w, 3)
    
    # Calculate PSNR
    mse = np.mean((region_image.astype(float) - reconstructed.astype(float)) ** 2)
    psnr = 10 * np.log10(255*255/mse) if mse > 0 else float('inf')
    
    compressed_data['mse'] = float(mse)
    compressed_data['psnr'] = float(psnr)
    
    print(f"Original: {original_size:,} bytes")
    print(f"Compressed: {compressed_size:,} bytes")
    print(f"Ratio: {original_size/compressed_size:.2f}:1")
    print(f"PSNR: {psnr:.2f} dB")
    
    return compressed_data


def decompress_color_quantization(compressed_data):
    """
    Decompress region compressed with color quantization.
    """
    method = compressed_data.get('method', '')
    if method != 'color_quantization':
        print(f"Warning: Expected 'color_quantization' but got '{method}'")
    
    # Extract essential data
    top_left = compressed_data['top_left']
    h, w = compressed_data['shape']
    palette = np.array(compressed_data['palette'], dtype=np.uint8)
    indices = compressed_data['indices']
    
    # Handle different index formats
    if isinstance(indices, list):
        # Flat list of indices
        index_array = np.array(indices, dtype=np.uint8).reshape(h, w)
    else:
        # Already an array (shouldn't happen with current implementation)
        index_array = indices
    
    # Reconstruct image
    reconstructed = palette[index_array].reshape(h, w, 3)
    
    # Return with placement info
    return {
        'image': reconstructed,
        'top_left': top_left,
        'shape': (h, w),
        'method': 'color_quantization',
        'quality': compressed_data.get('quality', 50)
    }














if __name__ == "__main__":

    image_name = 'images/Lenna.webp'
    image = cv2.imread(image_name)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    unified, regions, roi_image, nonroi_image, roi_mask, nonroi_mask = get_regions(image_rgb)
    roi_regions, nonroi_regions = extract_regions(image_rgb, roi_mask, nonroi_mask)


    print(f"Found {len(roi_regions)} ROI regions")
    print(f"Found {len(nonroi_regions)} non-ROI regions")

    # Display some statistics
    print("\nROI Regions (sorted by area):")
    for i, region in enumerate(sorted(roi_regions, key=lambda x: x['area'], reverse=True)[:5]):
        print(f"  Region {i+1}: Area = {region['area']} pixels")

    print("\nNon-ROI Regions (sorted by area):")
    for i, region in enumerate(sorted(nonroi_regions, key=lambda x: x['area'], reverse=True)[:5]):
        print(f"  Region {i+1}: Area = {region['area']} pixels")


   





























    # 🆕 COMPRESS EACH SLIC SEGMENT INDIVIDUALLY
    print("  Applying DCT compression to SLIC segments...")
    all_segments_compressed = []
    total_segments_original = 0
    total_segments_compressed = 0

    # ==============================================
    # MAIN PROCESSING LOOP - One ROI Region at a time
    # ==============================================
    for i, region in enumerate(roi_regions):
        print(f"\n{'='*60}")
        print(f"PROCESSING ROI REGION {i+1}/{len(roi_regions)}")
        print(f"{'='*60}")
        
        # ==============================================
        # 1. EXTRACT THE ROI REGION
        # ==============================================
        minr, minc, maxr, maxc = region['bbox']
        bbox_region = image_rgb[minr:maxr, minc:maxc]
        bbox_mask = region['bbox_mask']  # Irregular region mask
        region_image = bbox_region
        
        print(f"Region {i+1}: {bbox_region.shape[1]}x{bbox_region.shape[0]} pixels")
        print(f"Mask area: {np.sum(bbox_mask):,} pixels")

        
        # ==============================================
        # 2. ANALYZE REGION FOR SEGMENTATION
        # ==============================================
        overall_score, color_score, texture_score = calculate_split_score(bbox_region, bbox_mask)
        
        print(f"  Analysis scores:")
        print(f"    Overall: {overall_score:.3f}")
        print(f"    Color: {color_score:.3f}")
        print(f"    Texture: {texture_score:.3f}")
        
        window = math.ceil(math.ceil(math.log(bbox_region.size, 10)) * math.log(bbox_region.size))
        normalized_overall_score = normalize_result(overall_score, window)
        optimal_segments = math.ceil(normalized_overall_score)
        
        if optimal_segments <= 0: 
            optimal_segments = 1
        
        print(f"  Optimal segments: {optimal_segments}")
        
        # Visualize analysis (optional)
        visualize_split_analysis(
            region_image=region_image,
            overall_score=overall_score,
            color_score=color_score, 
            texture_score=texture_score,
            optimal_segments=optimal_segments
        )
        


        
        # ==============================================
        # 3. APPLY SLIC SEGMENTATION TO THE ROI
        # ==============================================
        print(f"\n  Applying SLIC segmentation...")
        roi_segments, texture_map = enhanced_slic_with_texture(bbox_region, n_segments=optimal_segments)
        segment_boundaries = extract_slic_segment_boundaries(roi_segments, bbox_mask)
        
        print(f"  Found {len(segment_boundaries)} sub-regions")
        print(f"  Total boundary points: {sum(seg['num_points'] for seg in segment_boundaries):,}")
        


        
        # ==============================================
        # MODIFIED MAIN LOOP WITH SHARED QUANTIZATION TABLE
        # ==============================================
       
        print(f"\n{'='*60}")
        print(f"METHOD 2: SHARED QTABLE SUBREGION COMPRESSION")
        print(f"{'='*60}")

        # PHASE 1: Analyze all subregions and design shared quantization table
        shared_qtable = analyze_subregions_frequency_content(
            region_image, 
            segment_boundaries, 
            roi_segments, 
            bbox_mask
        )

        print(f"\n  Shared quantization table (8x8):")
        for i in range(8):
            row_str = "    " + " ".join(f"{val:6.1f}" for val in shared_qtable[i])  # 6 chars, 1 decimal
            print(row_str)

        # PHASE 2: Compress all subregions using shared table
        subregion_results = []
        subregion_reconstructed = np.zeros_like(region_image)
        total_subregion_original = 0
        total_subregion_compressed = len(shared_qtable.tobytes())  # Start with table size
        subregion_psnrs = []

        print(f"\n  Compressing {len(segment_boundaries)} subregions with shared table...")

    













        for seg_idx, seg_data in enumerate(segment_boundaries):
            segment_id = seg_data.get('segment_id', seg_idx)
            segment_mask = (roi_segments == segment_id) & bbox_mask
            
            segment_pixels = np.sum(segment_mask)
            if segment_pixels < 64:
                continue
            
            print(f"\n    Segment {seg_idx+1}/{len(segment_boundaries)} (ID: {segment_id}):")
            print(f"      Pixels: {segment_pixels:,}")
            
            # ==============================================
            # 1. CREATE ISOLATED SEGMENT IMAGE
            # ==============================================
            # Create image with only this segment visible
            segment_image_full = region_image.copy()
            segment_image_full[bbox_mask & ~segment_mask] = [0, 0, 0]  # Other segments in bbox -> black
            segment_image_full[~bbox_mask] = [0, 0, 0]  # Outside bbox -> black
            
            # ==============================================
            # 2. FIND TIGHT BOUNDING BOX (CROP TO NON-BLACK PIXELS)
            # ==============================================
            # Find coordinates of non-black pixels
            # Non-black = at least one channel is > 0
            non_black_mask = np.any(segment_image_full > 0, axis=2)
            
            if np.sum(non_black_mask) == 0:
                print(f"      Warning: Segment has no non-black pixels!")
                continue
            
            # Get row and column indices of non-black pixels
            rows, cols = np.where(non_black_mask)
            
            # Find bounding box coordinates
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()
            
            # Add small padding (optional, e.g., 2 pixels)
            pad = 0
            h, w = segment_image_full.shape[:2]
            min_row = max(0, min_row - pad) 
            max_row = min(h - 1, max_row + pad) 
            min_col = max(0, min_col - pad)
            max_col = min(w - 1, max_col + pad) 
            
            # Calculate dimensions of cropped region
            crop_height = max_row - min_row + 1
            crop_width = max_col - min_col + 1
            
            # ==============================================
            # 3. CROP THE IMAGE TO TIGHT BOUNDING BOX
            # ==============================================
            segment_image_cropped = segment_image_full[min_row:max_row+1, min_col:max_col+1]

            
            
            # ==============================================
            # 4. CALCULATE ORIGINAL COORDINATES
            # ==============================================            
            absolute_min_row = min_row + minr
            absolute_min_col = min_col + minc 
            absolute_max_row = max_row + minr 
            absolute_max_col = max_col + minc
            
            # The top-left corner (left-northernmost point) coordinates in full image:
            top_left_abs_row = absolute_min_row
            top_left_abs_col = absolute_min_col
            
            print(f"      Original segment shape: {segment_image_full.shape[:2]}")
            print(f"      Cropped segment shape: {segment_image_cropped.shape[:2]}")
            print(f"      Crop reduction: {(1 - (crop_height*crop_width)/(h*w))*100:.1f}%")
            print(f"      Bbox in region: ({min_row}, {min_col}) to ({max_row}, {max_col})")
            print(f"      Bbox in full image: ({absolute_min_row}, {absolute_min_col}) to ({absolute_max_row}, {absolute_max_col})")
            print(f"      Top-left in full image: ({top_left_abs_row}, {top_left_abs_col})")
            
            # ==============================================
            # 5. VISUALIZE BEFORE/AFTER CROPPING
            # ==============================================
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            import matplotlib.patches as patches  # For drawing the Rectangle
            # Original (full region with black background)
            axes[0].imshow(segment_image_full)
            # Draw red bounding box
            rect = patches.Rectangle(
                (min_col, min_row), 
                max_col - min_col, 
                max_row - min_row,
                linewidth=2, 
                edgecolor='red', 
                facecolor='none'
            )
            axes[0].add_patch(rect)
            axes[0].set_title(f'Original Segment\nFull region: {h}x{w}')
            axes[0].axis('off')
            
            # Cropped version
            axes[1].imshow(segment_image_cropped)
            axes[1].set_title(f'Cropped Segment\nCropped: {crop_height}x{crop_width}')
            axes[1].axis('off')
            
            # Binary mask (for reference)
            axes[2].imshow(non_black_mask, cmap='gray')
            # Draw bounding box on mask too
            rect_mask = patches.Rectangle(
                (min_col, min_row), 
                max_col - min_col, 
                max_row - min_row,
                linewidth=2, 
                edgecolor='red', 
                facecolor='none'
            )
            axes[2].add_patch(rect_mask)
            axes[2].set_title(f'Non-black Mask\nPixels: {np.sum(non_black_mask):,}')
            axes[2].axis('off')
            
            plt.suptitle(f'Segment {seg_idx+1} - Tight Bounding Box Extraction', fontsize=14)
            plt.tight_layout()
            plt.show()
            
     
            
            

        
            print(f"segment_image_cropped shape: {segment_image_cropped.shape}")


            # ==============================================
            # 2. COMPRESS CROPPED SEGMENT
            # ==============================================
            seg_compression = compress_small_region_color_quantization(
                segment_image_cropped,
                (top_left_abs_row, top_left_abs_col),
                quality_factor=75  # Control fidelity vs compression
            )


            if seg_compression is None:
                continue
            
            # ==============================================
            # 3. RECONSTRUCT 
            # ==============================================
            reconstruction_result = decompress_color_quantization(
                seg_compression,
            )
            

            # ==============================================
            # 4. VISUALIZE COMPARISON WITH STATS
            # ==============================================
            print(f"\n{'='*60}")
            print(f"SEGMENT {seg_idx+1} - COLOR QUANTIZATION RESULTS")
            print(f"{'='*60}")

            # Get stats from compression
            h, w, _ = segment_image_cropped.shape
            original_size = h * w * 3
            compressed_size = seg_compression['compressed_size']
            compression_ratio = seg_compression['compression_ratio']
            psnr = seg_compression['psnr']
            n_colors = seg_compression['n_colors']

            # Print stats in a clean format
            print(f"Segment {seg_idx+1} Summary:")
            print(f"  Shape: {h}x{w}")
            print(f"  Top-left: {top_left_abs_row, top_left_abs_col}")
            print(f"  Colors in palette: {n_colors}")
            print(f"  Original size: {original_size:,} bytes")
            print(f"  Compressed size: {compressed_size:,} bytes")
            print(f"  Compression ratio: {compression_ratio:.2f}:1")
            print(f"  Space savings: {(1 - compressed_size/original_size)*100:.1f}%")
            print(f"  Quality (PSNR): {psnr:.2f} dB")

            # Create visualization figure
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 3, figsize=(15, 10))

            # Original image
            axes[0, 0].imshow(segment_image_cropped)
            axes[0, 0].set_title(f'Original\n{h}x{w} pixels')
            axes[0, 0].axis('off')

            # Add pixel info
            non_black = np.sum(np.any(segment_image_cropped > 10, axis=2))
            axes[0, 0].text(0.02, 0.98, f'Content: {non_black:,} px', 
                        transform=axes[0, 0].transAxes, fontsize=9, color='white',
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

            # Reconstructed image
            reconstructed_img = reconstruction_result['image']
            axes[0, 1].imshow(reconstructed_img)
            axes[0, 1].set_title(f'Reconstructed\n{n_colors} colors')
            axes[0, 1].axis('off')

            # Difference (amplified for visibility)
            diff = np.abs(segment_image_cropped.astype(float) - reconstructed_img.astype(float))
            diff_display = np.clip(diff * 3, 0, 255).astype(np.uint8)
            axes[0, 2].imshow(diff_display)
            axes[0, 2].set_title(f'Difference (×3)\nPSNR: {psnr:.2f} dB')
            axes[0, 2].axis('off')

            # Add MSE/PSNR to difference plot
            mse = np.mean(diff ** 2)
            axes[0, 2].text(0.02, 0.98, f'MSE: {mse:.2f}', 
                        transform=axes[0, 2].transAxes, fontsize=9, color='white',
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))

            # Color palette visualization
            palette = np.array(seg_compression['palette'])
            axes[1, 0].axis('off')

            # Create palette visualization
            if n_colors > 0:
                # Create a color swatch
                palette_h = max(1, min(10, n_colors // 10))
                palette_w = (n_colors + palette_h - 1) // palette_h
                
                palette_img = np.zeros((palette_h * 20, palette_w * 20, 3), dtype=np.uint8)
                
                for idx, color in enumerate(palette):
                    row = (idx // palette_w) * 20
                    col = (idx % palette_w) * 20
                    palette_img[row:row+18, col:col+18] = color
                
                # Display in inset axes
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                ax_inset = inset_axes(axes[1, 0], width="60%", height="60%", loc='center')
                ax_inset.imshow(palette_img)
                ax_inset.set_title(f'Color Palette ({n_colors} colors)')
                ax_inset.axis('off')
                
                # Add palette stats
                unique_original = len(np.unique(segment_image_cropped.reshape(-1, 3), axis=0))
                palette_text = f'Original: {unique_original} colors\nCompressed: {n_colors} colors\nReduction: {(1 - n_colors/unique_original)*100:.1f}%'
                axes[1, 0].text(0.5, 0.1, palette_text, transform=axes[1, 0].transAxes,
                            fontsize=9, horizontalalignment='center',
                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

            # Compression statistics visualization
            axes[1, 1].axis('off')

            # Create bar chart for size comparison
            size_data = [original_size, compressed_size]
            size_labels = ['Original', 'Compressed']
            colors = ['skyblue', 'lightcoral']

            # Create inset axes for bar chart
            ax_bar = inset_axes(axes[1, 1], width="60%", height="60%", loc='center')
            bars = ax_bar.bar(size_labels, size_data, color=colors, alpha=0.7)
            ax_bar.set_title('Size Comparison')
            ax_bar.set_ylabel('Bytes')

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax_bar.text(bar.get_x() + bar.get_width()/2., height + max(size_data)*0.05,
                        f'{height:,}', ha='center', va='bottom', fontsize=9)

            # Add ratio text
            ratio_text = f'Compression Ratio: {compression_ratio:.2f}:1\nSavings: {(1 - compressed_size/original_size)*100:.1f}%'
            axes[1, 1].text(0.5, 0.1, ratio_text, transform=axes[1, 1].transAxes,
                        fontsize=9, horizontalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

            # Index map visualization (shows which palette index each pixel uses)
            axes[1, 2].axis('off')

            # Get index map
            index_map = np.array(seg_compression['indices']).reshape(h, w)

            # Display index map as grayscale
            ax_index = inset_axes(axes[1, 2], width="60%", height="60%", loc='center')
            im = ax_index.imshow(index_map, cmap='viridis', aspect='auto')
            ax_index.set_title('Index Map (Palette Indices)')
            ax_index.axis('off')

            # Add colorbar for index map
            plt.colorbar(im, ax=ax_index, fraction=0.046, pad=0.04)

            # Add index map stats
            index_unique = len(np.unique(index_map))
            axes[1, 2].text(0.5, 0.1, f'Unique indices: {index_unique}/{n_colors}',
                        transform=axes[1, 2].transAxes,
                        fontsize=9, horizontalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

            # Main title
            plt.suptitle(f'Segment {seg_idx+1}: Color Quantization Results\nTop-left: {top_left_abs_row, top_left_abs_col}, Size: {h}x{w}', 
                        fontsize=14, fontweight='bold')

            plt.tight_layout()
            plt.show()

            # ==============================================
            # 5. CONSOLE SUMMARY (CLEAN FORMAT)
            # ==============================================
            print(f"\n📊 COMPRESSION SUMMARY:")
            print(f"   Original:    {original_size:>8,} bytes")
            print(f"   Compressed:  {compressed_size:>8,} bytes")
            print(f"   Ratio:       {compression_ratio:>8.2f}:1")
            print(f"   Savings:     {(1 - compressed_size/original_size)*100:>7.1f}%")
            print(f"   PSNR:        {psnr:>8.2f} dB")
            print(f"   Palette:     {n_colors:>8} colors")
            print(f"{'='*60}")
            