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










































































import numpy as np
import cv2
from sklearn.cluster import MiniBatchKMeans

"""
def compress_small_region_color_quantization(region_image, top_left_coords, max_colors=32):

    if region_image is None or region_image.size == 0:
        return None
    
    h, w, _ = region_image.shape
    total_pixels = h * w
    
    print(f"Compressing {h}x{w} region with color quantization")
    print(f"Top-left: {top_left_coords}")
    print(f"Max colors: {max_colors}")
    
    # ==============================================
    # 1. GET UNIQUE COLORS (WITHOUT DUPLICATES)
    # ==============================================
    # Get all unique colors from the image
    pixels = region_image.reshape(-1, 3)
    
    # Method 1: Use np.unique - removes duplicates
    unique_colors_array = np.unique(pixels, axis=0)
    unique_colors = len(unique_colors_array)
    
    print(f"Unique colors in image: {unique_colors:,}")
    
    # Cap max_colors based on actual unique colors
    if max_colors > unique_colors:
        print(f"Reducing max_colors from {max_colors} to {unique_colors} (actual unique colors)")
        max_colors = unique_colors
    
    # Special case: Single color region
    if max_colors == 1 or unique_colors == 1:
        avg_color = np.mean(pixels, axis=0).astype(np.uint8)
        
        compressed_data = {
            'method': 'color_quantization',
            'top_left': top_left_coords,
            'shape': (h, w),
            'palette': [avg_color.tolist()],  # Single color palette
            'indices': [0] * total_pixels,  # All indices are 0
            'max_colors': 1,
            'actual_colors': 1,
            'original_size': total_pixels * 3
        }
        
        compressed_size = 3 + total_pixels + 50
        compressed_data['compressed_size'] = compressed_size
        
        # Calculate PSNR (perfect for single color)
        compressed_data['psnr'] = float('inf')
        
        print(f"Single color region: {avg_color}")
        return compressed_data
    
    # ==============================================
    # 2. COLOR QUANTIZATION WITH UNIQUE COLORS ENFORCED
    # ==============================================
    # If we have few colors already, just use them directly
    if unique_colors <= max_colors:
        print(f"Using {unique_colors} unique colors directly (no quantization needed)")
        
        # Create palette from unique colors
        palette = unique_colors_array[:max_colors]
        
        # Create mapping from color to index
        color_to_index = {}
        for idx, color in enumerate(palette):
            color_tuple = tuple(color)
            color_to_index[color_tuple] = idx
        
        # Create index map
        indices_flat = []
        for pixel in pixels:
            color_tuple = tuple(pixel)
            indices_flat.append(color_to_index[color_tuple])
        
        index_map = np.array(indices_flat).reshape(h, w)
        
    else:
        # Need to quantize to reduce colors
        print(f"Quantizing {unique_colors:,} colors → {max_colors} colors")
        
        # Apply K-means clustering
        pixels_float = pixels.astype(np.float32)
        
        kmeans = MiniBatchKMeans(
            n_clusters=max_colors,
            random_state=42,
            batch_size=min(100, len(pixels_float)),
            max_iter=20
        )
        
        # Fit and predict
        labels = kmeans.fit_predict(pixels_float)
        
        # Get palette (cluster centers)
        palette_raw = kmeans.cluster_centers_.astype(np.uint8)
        
        # ==============================================
        # 3. REMOVE DUPLICATE COLORS FROM PALETTE
        # ==============================================
        print("Removing duplicate colors from palette...")
        
        # Convert palette to list of tuples for easy duplicate detection
        palette_tuples = [tuple(color) for color in palette_raw]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_palette_tuples = []
        
        for color_tuple in palette_tuples:
            if color_tuple not in seen:
                seen.add(color_tuple)
                unique_palette_tuples.append(color_tuple)
        
        # Count actual unique colors after deduplication
        actual_colors = len(unique_palette_tuples)
        
        print(f"Palette after deduplication: {actual_colors} colors (from {max_colors})")
        
        # Convert back to numpy array
        palette = np.array([list(color) for color in unique_palette_tuples], dtype=np.uint8)
        
        # ==============================================
        # 4. RE-INDEX LABELS AFTER DEDUPLICATION
        # ==============================================
        # Need to remap labels because we removed duplicate palette entries
        
        # Create mapping from old cluster index to new palette index
        old_to_new = {}
        
        # Track which new index each old cluster should map to
        for old_idx, color_tuple in enumerate(palette_tuples):
            if color_tuple not in old_to_new:
                # Find new index for this color
                new_idx = unique_palette_tuples.index(color_tuple)
                old_to_new[old_idx] = new_idx
        
        # Remap all labels
        remapped_labels = np.zeros_like(labels)
        for old_idx, new_idx in old_to_new.items():
            mask = labels == old_idx
            remapped_labels[mask] = new_idx
        
        labels = remapped_labels
        index_map = labels.reshape(h, w)
        
        # Create indices list
        indices_flat = index_map.flatten().tolist()
    
    # ==============================================
    # 5. CALCULATE COMPRESSION STATS
    # ==============================================
    actual_colors = len(palette)
    
    # Choose optimal data type for indices
    if actual_colors <= 256:
        index_dtype = np.uint8
        bytes_per_index = 1
    else:
        index_dtype = np.uint16
        bytes_per_index = 2
    
    # Recalculate with actual_colors
    original_size = total_pixels * 3
    palette_size = actual_colors * 3
    indices_size = total_pixels * bytes_per_index
    metadata_size = 50
    
    compressed_size = palette_size + indices_size + metadata_size
    
    # ==============================================
    # 6. CREATE COMPRESSED DATA STRUCTURE
    # ==============================================
    compressed_data = {
        'method': 'color_quantization',
        'top_left': top_left_coords,
        'shape': (h, w),
        'palette': palette.tolist(),  # No duplicate colors
        'indices': indices_flat,
        'max_colors': max_colors,
        'actual_colors': actual_colors,  # Actual number of unique colors in palette
        'index_dtype': str(index_dtype),
        'original_size': original_size,
        'compressed_size': compressed_size,
        'compression_ratio': original_size / compressed_size if compressed_size > 0 else 0
    }
    
    # ==============================================
    # 7. VALIDATION AND PSNR CALCULATION
    # ==============================================
    # Reconstruct image for validation
    reconstructed = palette[index_map].reshape(h, w, 3)
    
    # Calculate PSNR
    mse = np.mean((region_image.astype(float) - reconstructed.astype(float)) ** 2)
    psnr = 10 * np.log10(255*255/mse) if mse > 0 else float('inf')
    
    compressed_data['mse'] = float(mse)
    compressed_data['psnr'] = float(psnr)
    
    print(f"Palette: {actual_colors} unique colors")
    print(f"Original: {original_size:,} bytes")
    print(f"Compressed: {compressed_size:,} bytes")
    print(f"Ratio: {original_size/compressed_size:.2f}:1")
    print(f"PSNR: {psnr:.2f} dB")
    print(f"Color reduction: {unique_colors} → {actual_colors} colors")
    
    return compressed_data
"""

def hierarchical_color_quantization(region_image, top_left_coords, quality=85):
    """
    Hierarchical color quantization starting from ALL unique colors.
    Clusters similar colors based on quality parameter.
    """
    if region_image is None or region_image.size == 0:
        return None
    
    h, w, _ = region_image.shape
    total_pixels = h * w
    
    print(f"Hierarchical color quantization: {h}x{w}, quality={quality}%")
    
    # ==============================================
    # 1. GET ALL UNIQUE COLORS WITH FREQUENCIES
    # ==============================================
    pixels = region_image.reshape(-1, 3)
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    unique_count = len(unique_colors)
    
    print(f"  Found {unique_count:,} unique colors")
    
    # Special cases
    if unique_count == 0:
        return None
    elif unique_count == 1:
        # Single color region
        palette = unique_colors
        indices_flat = [0] * total_pixels
        actual_colors = 1
    else:
        # ==============================================
        # 2. DETERMINE TARGET CLUSTERS BASED ON QUALITY
        # ==============================================
        # Quality 100% = keep all unique colors
        # Quality 0% = reduce to 2 colors (minimum)
        target_clusters = max(2, int(unique_count * quality / 100))
        
        print(f"  Quality {quality}% → target: {target_clusters} clusters "
              f"(from {unique_count} colors)")
        
        # ==============================================
        # 3. PERFORM CLUSTERING BASED ON SIZE
        # ==============================================
        if unique_count <= 2000:  # Reasonable for hierarchical clustering
            palette, indices_flat, actual_colors = cluster_hierarchical_small(
                unique_colors, counts, target_clusters, pixels
            )
        else:
            # Too many unique colors for hierarchical clustering
            print(f"  Too many unique colors ({unique_count:,}), using optimized clustering")
            palette, indices_flat, actual_colors = cluster_large_dataset(
                unique_colors, counts, target_clusters, pixels
            )
    
    # After clustering functions, we should have:
    # palette, indices_flat, actual_colors defined
    
    # ==============================================
    # 4. CALCULATE SIMILARITY METRICS
    # ==============================================
    # Calculate color similarity preservation
    original_colors_set = set(tuple(color) for color in unique_colors)
    compressed_colors_set = set(tuple(color) for color in palette)
    
    # Find closest matches for each original color
    avg_color_error = 0
    sample_size = min(100, len(unique_colors))
    sample_indices = np.random.choice(len(unique_colors), sample_size, replace=False)
    
    for idx in sample_indices:
        orig_color = unique_colors[idx]
        distances = np.sqrt(np.sum((palette - orig_color) ** 2, axis=1))
        avg_color_error += distances.min()
    
    avg_color_error = avg_color_error / sample_size
    
    print(f"  Average color error: {avg_color_error:.1f}")
    print(f"  Color reduction: {unique_count:,} → {actual_colors}")
    print(f"  Reduction ratio: {unique_count/actual_colors:.1f}:1")
    
    # ==============================================
    # 5. CREATE COMPRESSED DATA STRUCTURE
    # ==============================================
    compressed_data = {
        'method': 'hierarchical_quantization',
        'top_left': top_left_coords,
        'shape': (h, w),
        'palette': palette.tolist(),
        'indices': indices_flat,
        'original_unique_colors': unique_count,
        'compressed_colors': actual_colors,
        'quality': quality,
        'avg_color_error': float(avg_color_error),
        'original_size': total_pixels * 3
    }
    
    # Calculate compression stats
    index_dtype = np.uint8 if actual_colors <= 256 else np.uint16
    bytes_per_index = 1 if actual_colors <= 256 else 2
    
    # Size calculations
    palette_size = actual_colors * 3
    indices_size = total_pixels * bytes_per_index
    metadata_size = 100  # Approximate
    
    compressed_size = palette_size + indices_size + metadata_size
    original_size = total_pixels * 3
    
    compressed_data['compressed_size'] = compressed_size
    compressed_data['compression_ratio'] = original_size / compressed_size if compressed_size > 0 else 0
    compressed_data['index_dtype'] = str(index_dtype)
    
    # Calculate PSNR
    indices_array = np.array(indices_flat).reshape(h, w)
    reconstructed = palette[indices_array]
    mse = np.mean((region_image.astype(float) - reconstructed.astype(float)) ** 2)
    psnr = 10 * np.log10(255*255/mse) if mse > 0 else float('inf')
    
    compressed_data['mse'] = float(mse)
    compressed_data['psnr'] = float(psnr)
    
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  Compression: {original_size:,} → {compressed_size:,} bytes")
    print(f"  Ratio: {compressed_data['compression_ratio']:.2f}:1")
    
    return compressed_data
def cluster_hierarchical_small(unique_colors, counts, target_clusters, pixels):
    """
    Cluster small datasets (<2000 colors) using hierarchical clustering.
    """
    # Convert to Lab color space for perceptual clustering
    try:
        from skimage.color import rgb2lab
        colors_for_clustering = rgb2lab(unique_colors.reshape(-1, 1, 3)).reshape(-1, 3)
    except ImportError:
        colors_for_clustering = unique_colors.astype(float)
    
    # Weight colors by frequency
    weights = counts / counts.max()
    weighted_colors = colors_for_clustering * weights[:, np.newaxis]
    
    # Calculate pair-wise distances
    from scipy.spatial.distance import pdist
    from scipy.cluster.hierarchy import linkage, fcluster
    
    distances = pdist(weighted_colors, metric='euclidean')
    
    # Perform hierarchical clustering
    Z = linkage(distances, method='ward')
    
    # Cut dendrogram to get desired number of clusters
    cluster_labels = fcluster(Z, t=target_clusters, criterion='maxclust')
    
    # Create cluster centers (palette)
    palette = []
    
    for cluster_id in range(1, target_clusters + 1):
        mask = cluster_labels == cluster_id
        if np.any(mask):
            # Weighted average of colors in this cluster
            cluster_colors_rgb = unique_colors[mask]
            cluster_counts = counts[mask]
            
            weighted_avg = np.average(
                cluster_colors_rgb.astype(float), 
                axis=0, 
                weights=cluster_counts
            )
            palette.append(weighted_avg.astype(np.uint8))
    
    palette = np.array(palette)
    actual_clusters = len(palette)
    
    print(f"  Created {actual_clusters} color clusters (hierarchical)")
    
    # Map each unique color to its cluster
    color_to_cluster = {}
    for i, color in enumerate(unique_colors):
        cluster_idx = cluster_labels[i] - 1  # Convert to 0-based
        color_to_cluster[tuple(color)] = cluster_idx
    
    # Create indices for all pixels
    indices_flat = [color_to_cluster[tuple(pixel)] for pixel in pixels]
    
    return palette, indices_flat, actual_clusters


def cluster_large_dataset(unique_colors, counts, target_clusters, pixels):
    """
    Cluster large datasets (>2000 colors) using optimized methods.
    """
    # Cap target_clusters to sample size
    max_sample_size = 2000
    if target_clusters > max_sample_size:
        print(f"  Adjusting target clusters from {target_clusters} to {max_sample_size} "
              f"(max sample size)")
        target_clusters = max_sample_size
    
    # Ensure target_clusters is less than unique_count
    target_clusters = min(target_clusters, len(unique_colors) - 1)
    if target_clusters < 2:
        target_clusters = 2
    
    # Sample weighted by frequency
    sample_size = min(max_sample_size, len(unique_colors))
    
    # If we have more unique colors than sample size, we need to sample
    if len(unique_colors) > sample_size:
        print(f"  Sampling {sample_size:,} colors from {len(unique_colors):,} unique colors")
        
        probs = counts / counts.sum()
        sample_indices = np.random.choice(
            len(unique_colors), 
            size=sample_size, 
            replace=False, 
            p=probs
        )
        
        sample_colors = unique_colors[sample_indices]
        sample_counts = counts[sample_indices]
    else:
        # Use all colors
        sample_colors = unique_colors
        sample_counts = counts
        sample_size = len(unique_colors)
    
    # Apply K-means to samples
    from sklearn.cluster import MiniBatchKMeans
    
    # Weight samples by frequency
    sample_weights = sample_counts / sample_counts.max()
    weighted_samples = sample_colors.astype(float)
    
    # Adjust target_clusters if needed
    if target_clusters > sample_size:
        target_clusters = max(2, sample_size // 2)
        print(f"  Adjusted target clusters to {target_clusters} (sample size: {sample_size})")
    
    # Ensure we have enough samples for clustering
    if target_clusters >= sample_size:
        # Use all colors as palette (no clustering)
        print(f"  Using all {sample_size} sampled colors as palette")
        palette = sample_colors
        actual_clusters = len(palette)
        
        # Create color mapping
        color_to_idx = {tuple(color): i for i, color in enumerate(sample_colors)}
        
        # For colors not in sample, find nearest
        indices_flat = []
        for pixel in pixels:
            pixel_tuple = tuple(pixel)
            if pixel_tuple in color_to_idx:
                indices_flat.append(color_to_idx[pixel_tuple])
            else:
                # Find nearest color
                distances = np.sqrt(np.sum((sample_colors - pixel) ** 2, axis=1))
                nearest_idx = np.argmin(distances)
                indices_flat.append(nearest_idx)
    else:
        # Perform K-means clustering
        print(f"  Applying K-means to {sample_size} samples → {target_clusters} clusters")
        
        kmeans = MiniBatchKMeans(
            n_clusters=target_clusters, 
            random_state=42, 
            n_init=3,
            batch_size=256
        )
        
        kmeans.fit(weighted_samples, sample_weight=sample_weights)
        
        # Get palette from K-means
        palette = kmeans.cluster_centers_.astype(np.uint8)
        actual_clusters = len(palette)
        
        # Map all colors to nearest palette color
        print(f"  Mapping {len(unique_colors):,} colors to {actual_clusters} palette colors")
        
        # Process in batches to avoid memory issues
        batch_size = 10000
        indices_flat = []
        
        for i in range(0, len(pixels), batch_size):
            batch = pixels[i:i+batch_size]
            
            # Vectorized distance calculation
            # Reshape for broadcasting: (batch_size, 1, 3) - (1, palette_size, 3)
            batch_expanded = batch[:, np.newaxis, :]  # (batch_size, 1, 3)
            palette_expanded = palette[np.newaxis, :, :]  # (1, palette_size, 3)
            
            distances = np.sqrt(np.sum((batch_expanded - palette_expanded) ** 2, axis=2))
            nearest_indices = np.argmin(distances, axis=1)
            
            indices_flat.extend(nearest_indices.tolist())
    
    return palette, indices_flat, actual_clusters


def adaptive_hierarchical_quantization(region_image, top_left_coords):
    """
    Adaptive version: automatically determines quality based on region characteristics.
    """
    h, w, _ = region_image.shape
    
    # Analyze region to determine optimal quality
    if h * w < 1000:  # Small region
        quality = 95  # High quality
    else:
        # Calculate color complexity
        pixels = region_image.reshape(-1, 3)
        unique_colors = len(np.unique(pixels, axis=0))
        color_density = unique_colors / (h * w)
        
        if color_density > 0.5:  # Very colorful
            quality = 90
        elif color_density > 0.2:  # Moderately colorful
            quality = 80
        else:  # Mostly uniform
            quality = 70
    
    print(f"Adaptive quality selection: {quality}%")
    return hierarchical_color_quantization(region_image, top_left_coords, quality)


def quality_controlled_quantization(region_image, top_left_coords, min_quality=70, max_quality=95):
    """
    Find optimal quality that meets size constraints.
    """
    # Try different quality levels
    qualities = [max_quality, (max_quality + min_quality) // 2, min_quality]
    best_result = None
    best_score = -float('inf')
    
    for quality in qualities:
        result = hierarchical_color_quantization(region_image, top_left_coords, quality)
        
        if result:
            # Score = PSNR * compression ratio
            score = result['psnr'] * result['compression_ratio']
            
            if score > best_score:
                best_score = score
                best_result = result
    
    return best_result

def apply_edge_aware_dithering(image, indices, palette):
    """
    Apply Floyd-Steinberg dithering only to smooth areas, not edges.
    """
    h, w = indices.shape
    dithered = indices.copy()
    
    # Convert to float for error diffusion
    img_float = image.astype(float)
    
    # Detect edges using simple gradient
    from scipy.ndimage import sobel
    
    # Calculate edge strength
    gray = np.mean(img_float, axis=2)
    edge_x = sobel(gray, axis=0)
    edge_y = sobel(gray, axis=1)
    edge_strength = np.sqrt(edge_x**2 + edge_y**2)
    edge_threshold = edge_strength.mean()
    
    # Apply Floyd-Steinberg dithering only to non-edge pixels
    for y in range(h):
        for x in range(w):
            # Skip edge pixels
            if edge_strength[y, x] > edge_threshold:
                continue
                
            old_pixel = img_float[y, x]
            
            # Find nearest palette color
            distances = np.sqrt(np.sum((palette - old_pixel) ** 2, axis=1))
            new_idx = np.argmin(distances)
            new_pixel = palette[new_idx]
            
            dithered[y, x] = new_idx
            
            # Calculate quantization error
            quant_error = old_pixel - new_pixel
            
            # Diffuse error to neighbors (Floyd-Steinberg)
            if x + 1 < w:
                img_float[y, x+1] += quant_error * 7/16
            if y + 1 < h:
                if x > 0:
                    img_float[y+1, x-1] += quant_error * 3/16
                img_float[y+1, x] += quant_error * 5/16
                if x + 1 < w:
                    img_float[y+1, x+1] += quant_error * 1/16
    
    return dithered


def adaptive_color_quantization(region_image, top_left_coords, quality=85):
    """
    Adaptive quantization: automatically determines optimal color count.
    """
    h, w, _ = region_image.shape
    total_pixels = h * w
    
    # Get unique colors
    pixels = region_image.reshape(-1, 3)
    unique_colors = np.unique(pixels, axis=0)
    unique_count = len(unique_colors)
    
    # Start with conservative estimate
    max_colors = min(32, unique_count)
    
    # Try different color counts to find optimal
    best_psnr = 0
    best_result = None
    
    for colors in [max_colors, max_colors//2, max_colors//4]:
        if colors < 2:
            colors = 2
            
        result = hierarchical_color_quantization(
            region_image, 
            top_left_coords, 
            max_colors=colors,
            quality=quality
        )
        
        if result and result['psnr'] > best_psnr:
            best_psnr = result['psnr']
            best_result = result
    
    return best_result

def decompress_color_quantization(compressed_data):
    """
    Decompress region compressed with color quantization.
    """
    # Extract essential data
    top_left = compressed_data['top_left']
    h, w = compressed_data['shape']
    palette = np.array(compressed_data['palette'], dtype=np.uint8)
    indices = compressed_data['indices']
    
    # Determine the correct data type based on palette size
    # If palette has more than 256 colors, we need uint16
    if len(palette) > 256:
        # Use uint16 for large palettes
        dtype = np.uint16
    else:
        # Use uint8 for small palettes
        dtype = np.uint8
    
    # Handle different index formats
    if isinstance(indices, list):
        # Flat list of indices - use the determined dtype
        try:
            index_array = np.array(indices, dtype=dtype).reshape(h, w)
        except OverflowError:
            # Fallback to uint16 if uint8 fails (just in case)
            print(f"Warning: Overflow with {dtype}, falling back to uint16")
            index_array = np.array(indices, dtype=np.uint16).reshape(h, w)
    else:
        # Already an array
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





















































import struct
import pickle
import zlib

"""
def save_segments_a2f(all_segments_compressed, filename="output.a2f"):

    print(f"Saving to {filename}")
    
    # Create minimal segments list
    segments_to_save = []
    
    for seg in all_segments_compressed:
        # Store ONLY these 4 fields (nothing else!)
        minimal_seg = {
            't': seg['top_left'],      # top_left (tuple)
            's': seg['shape'],         # shape (tuple)
            'p': seg['palette'],       # palette (list of RGB)
        }
        
        # Store either RLE or raw indices (use shortest key)
        if 'indices_rle' in seg:
            minimal_seg['r'] = seg['indices_rle']  # RLE indices
        else:
            minimal_seg['i'] = seg['indices']      # Raw indices
        
        segments_to_save.append(minimal_seg)
    
    print(f"Preparing {len(segments_to_save)} segments")
    
    # ==============================================
    # OPTIMIZED SERIALIZATION (No extra metadata!)
    # ==============================================
    # Create minimal save structure
    save_data = segments_to_save  # Just the list, nothing else!
    
    # Serialize with highest efficiency
    serialized = pickle.dumps(save_data, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Pickle size: {len(serialized):,} bytes")
    
    # Apply zlib compression
    compressed = zlib.compress(serialized, level=9)  # Max compression
    print(f"After zlib: {len(compressed):,} bytes")
    
    # ==============================================
    # WRITE MINIMAL .a2f FILE
    # ==============================================
    with open(filename, 'wb') as f:
        # Header (8 bytes total - absolute minimum)
        f.write(b'A2F2')  # Magic number (4 bytes)
        f.write(struct.pack('<I', len(compressed)))  # Data size (4 bytes)
        
        # Write compressed data
        f.write(compressed)
    
    # Calculate stats
    total_pixels = sum(seg['s'][0] * seg['s'][1] for seg in segments_to_save)
    original_bytes = total_pixels * 3
    file_size = len(compressed) + 8
    
    print(f"\n✅ Saved {len(segments_to_save)} segments")
    print(f"   File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
    print(f"   Total pixels: {total_pixels:,}")
    print(f"   Compression: {original_bytes/file_size:.1f}:1")
    print(f"   Savings: {(1 - file_size/original_bytes)*100:.1f}%")
    
    return filename 
"""


def save_segments_a2f(all_segments_compressed, filename="output.a2f"):
    """
    Save segments with ONLY essential reconstruction data.
    """
    print(f"\nSaving to {filename}")
    
    # Create minimal segments list
    segments_to_save = []
    
    for seg in all_segments_compressed:
        # Store minimal data
        minimal_seg = {
            't': tuple(seg['top_left']),      # top_left (tuple)
            's': tuple(seg['shape']),         # shape (tuple)
            'p': seg['palette'],              # palette (list of RGB)
        }
        
        # Store indices in most efficient format
        if 'indices_rle' in seg:
            minimal_seg['r'] = seg['indices_rle']  # RLE indices
            minimal_seg['e'] = 'r'                 # encoding: rle
        elif 'indices' in seg:
            # Convert numpy array to list for smaller pickle size if small
            if isinstance(seg['indices'], np.ndarray):
                if seg['indices'].size < 1000:
                    minimal_seg['i'] = seg['indices'].tolist()
                else:
                    minimal_seg['i'] = seg['indices']
            else:
                minimal_seg['i'] = seg['indices']
            minimal_seg['e'] = 'r' if 'encoding' in seg and seg['encoding'] == 'rle' else 'i'
        
        segments_to_save.append(minimal_seg)
    
    print(f"Preparing {len(segments_to_save)} segments")
    
    # Calculate statistics before compression
    total_pixels = 0
    total_indices_size = 0
    for seg in segments_to_save:
        h, w = seg['s']
        total_pixels += h * w
        if 'i' in seg:
            if isinstance(seg['i'], list):
                total_indices_size += len(seg['i'])
            elif isinstance(seg['i'], np.ndarray):
                total_indices_size += seg['i'].size
        elif 'r' in seg:
            total_indices_size += len(seg['r']) // 2
    
    print(f"Total pixels: {total_pixels:,}")
    print(f"Total runs/indices: {total_indices_size:,}")
    
    # Serialize with highest efficiency
    serialized = pickle.dumps(segments_to_save, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Pickle size: {len(serialized):,} bytes")
    
    # Apply zlib compression
    compressed = zlib.compress(serialized, level=9)
    print(f"After zlib: {len(compressed):,} bytes")
    
    # Write .a2f file
    with open(filename, 'wb') as f:
        # Header
        f.write(b'A2F2')  # Magic number
        f.write(struct.pack('<I', len(compressed)))  # Data size
        
        # Write compressed data
        f.write(compressed)
    
    # Calculate final stats
    original_bytes = total_pixels * 3  # Original RGB image
    file_size = len(compressed) + 8
    
    print(f"\n✅ Saved {len(segments_to_save)} segments")
    print(f"   File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
    print(f"   Original (RGB): {original_bytes:,} bytes ({original_bytes/1024:.1f} KB)")
    print(f"   Compression ratio: {original_bytes/file_size:.2f}:1")
    print(f"   Space savings: {(1 - file_size/original_bytes)*100:.1f}%")
    
    return filename

def load_segments_a2f(filename):
    """
    Load segments from .a2f file.
    """
    print(f"Loading {filename}")
    
    try:
        with open(filename, 'rb') as f:
            # ==========================================
            # READ HEADER (MUST match save function)
            # ==========================================
            # 1. Read and verify magic number
            magic = f.read(4)
            if magic != b'A2F1':
                raise ValueError(f"Invalid .a2f file. Expected 'A2F1', got {magic}")
            
            # 2. Read compressed data size
            compressed_size = struct.unpack('<I', f.read(4))[0]
            
            # 3. Read exactly that many bytes of compressed data
            compressed_data = f.read(compressed_size)
            
            if len(compressed_data) != compressed_size:
                raise ValueError(f"File truncated. Expected {compressed_size} bytes, got {len(compressed_data)}")
        
        # ==========================================
        # DECOMPRESS AND LOAD DATA
        # ==========================================
        serialized_data = zlib.decompress(compressed_data)
        loaded_data = pickle.loads(serialized_data)
        
        segments = loaded_data['segments']
        print(f"✅ Loaded {len(segments)} segments")
        
        return segments
        
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None


def reconstruct_from_segments(segments, full_image_shape):
    """
    Reconstruct full image from loaded segments.
    """
    h, w = full_image_shape[:2]
    reconstructed = np.zeros((h, w, 3), dtype=np.uint8)
    
    for seg_idx, seg in enumerate(segments):
        top_row, left_col = seg['top_left']
        seg_h, seg_w = seg['shape']
        
        # Convert to numpy arrays
        palette = np.array(seg['palette'], dtype=np.uint8)
        indices = np.array(seg['indices'], dtype=np.uint8).reshape(seg_h, seg_w)
        
        # Reconstruct segment
        segment_img = palette[indices]
        
        # Place in final image (with bounds checking)
        bottom_row = min(top_row + seg_h, h)
        right_col = min(left_col + seg_w, w)
        
        if bottom_row > top_row and right_col > left_col:
            valid_h = bottom_row - top_row
            valid_w = right_col - left_col
            reconstructed[top_row:bottom_row, left_col:right_col] = segment_img[:valid_h, :valid_w]
    
    print(f"Reconstructed image: {w}x{h}")
    return reconstructed

def save_binary_a2f(all_segments_compressed, filename="output.a2f"):
    """
    Save using custom binary format (maximum compression).
    """
    print(f"Saving binary .a2f: {filename}")
    
    with open(filename, 'wb') as f:
        # Header
        f.write(b'A2FB')  # Magic
        f.write(struct.pack('<H', len(all_segments_compressed)))  # Segment count (2 bytes)
        
        total_bytes = 6  # Header size
        
        for seg in all_segments_compressed:
            # 1. Top-left (4 bytes: 2× uint16)
            top, left = seg['top_left']
            f.write(struct.pack('<HH', top, left))
            
            # 2. Shape (4 bytes: 2× uint16)
            h, w = seg['shape']
            f.write(struct.pack('<HH', h, w))
            
            # 3. Palette
            palette = seg['palette']
            f.write(struct.pack('<B', len(palette)))  # Color count (1 byte)
            
            for r, g, b in palette:
                f.write(struct.pack('<BBB', r, g, b))  # 3 bytes per color
            
            # 4. Indices (RLE or raw)
            if 'indices_rle' in seg:
                # RLE format: flag + pairs
                f.write(b'R')  # RLE flag (1 byte)
                rle_pairs = seg['indices_rle']
                f.write(struct.pack('<H', len(rle_pairs)))  # Pair count (2 bytes)
                
                for value, count in rle_pairs:
                    f.write(struct.pack('<BH', value, count))  # 3 bytes per pair
            else:
                # Raw format
                f.write(b'I')  # Raw flag (1 byte)
                indices = seg['indices']
                f.write(struct.pack('<H', len(indices)))  # Index count (2 bytes)
                
                # Pack indices efficiently
                for idx in indices:
                    f.write(struct.pack('<B', idx))  # 1 byte per index
            
            total_bytes = f.tell()
    
    print(f"✅ Binary .a2f: {total_bytes:,} bytes")
    return filename






































import numpy as np
from collections import defaultdict

def fast_border_smoothing(region_components, roi_bbox):
    """
    Fast border smoothing: Border pixels get averaged with surrounding non-border pixels.
    Much faster than previous implementations.
    """
    if not region_components:
        return []
    
    if len(region_components) == 1:
        return region_components
    
    print(f"\n{'='*60}")
    print(f"FAST BORDER SMOOTHING: {len(region_components)} components")
    print(f"{'='*60}")
    
    minr, minc, maxr, maxc = roi_bbox
    roi_height = maxr - minr
    roi_width = maxc - minc
    
    # ==============================================
    # 1. SIMPLE MERGE (NO SMOOTHING YET)
    # ==============================================
    print("Merging components...")
    
    # Create canvas
    roi_indices = np.ones((roi_height, roi_width), dtype=np.uint16)
    
    # Collect all colors
    all_colors = []
    color_to_index = {}
    
    # Add black
    black_color = (0, 0, 0)
    all_colors.append(black_color)
    color_to_index[black_color] = 1
    
    # Process segments (last segment has priority)
    for seg in reversed(region_components):  # Reverse so last segments win
        seg_top_left = seg['top_left']
        seg_shape = seg['shape']
        seg_palette = seg['palette']
        seg_indices = np.array(seg['indices']).reshape(seg_shape)
        
        # Calculate position relative to ROI
        rel_row = seg_top_left[0] - minr
        rel_col = seg_top_left[1] - minc
        
        # Place segment (colored pixels override anything)
        for r in range(seg_shape[0]):
            for c in range(seg_shape[1]):
                roi_r = rel_row + r
                roi_c = rel_col + c
                
                if 0 <= roi_r < roi_height and 0 <= roi_c < roi_width:
                    seg_pixel_idx = seg_indices[r, c]
                    
                    if seg_pixel_idx != 1:  # Colored pixel
                        color = tuple(seg_palette[seg_pixel_idx])
                        
                        # Map color to combined palette
                        if color not in color_to_index:
                            color_to_index[color] = len(all_colors)
                            all_colors.append(color)
                        
                        # Place the color index (overwrites anything)
                        roi_indices[roi_r, roi_c] = color_to_index[color]
    
    # ==============================================
    # 2. IDENTIFY BORDER PIXELS FAST
    # ==============================================
    print("Identifying border pixels...")
    
    # Border pixels: pixels adjacent to a different colored pixel
    border_mask = np.zeros((roi_height, roi_width), dtype=bool)
    non_border_mask = np.zeros((roi_height, roi_width), dtype=bool)
    
    # Simple 4-connected neighbor check
    for r in range(roi_height):
        for c in range(roi_width):
            current_idx = roi_indices[r, c]
            if current_idx != 1:  # Not black
                is_border = False
                
                # Check 4 neighbors
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < roi_height and 0 <= nc < roi_width:
                        neighbor_idx = roi_indices[nr, nc]
                        if neighbor_idx != 1 and neighbor_idx != current_idx:
                            is_border = True
                            break
                
                if is_border:
                    border_mask[r, c] = True
                else:
                    non_border_mask[r, c] = True
    
    border_count = np.sum(border_mask)
    print(f"  Border pixels: {border_count:,}")
    
    # ==============================================
    # 3. FAST BORDER SMOOTHING
    # ==============================================
    print("Smoothing border pixels...")
    
    # Create a copy for smoothing
    smoothed_indices = roi_indices.copy()
    
    # Pre-cache palette as numpy array for faster access
    palette_array = np.array(all_colors)
    
    # Process only border pixels
    border_coords = np.where(border_mask)
    smoothed_count = 0
    
    for r, c in zip(border_coords[0], border_coords[1]):
        current_idx = roi_indices[r, c]
        
        # Collect non-border neighbor colors (excluding black)
        neighbor_colors = []
        
        # Check 8-connected neighbors
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                
                nr, nc = r + dr, c + dc
                if 0 <= nr < roi_height and 0 <= nc < roi_width:
                    neighbor_idx = roi_indices[nr, nc]
                    
                    # If neighbor is non-border and not black, use its color
                    if neighbor_idx != 1 and non_border_mask[nr, nc]:
                        neighbor_colors.append(palette_array[neighbor_idx])
        
        if neighbor_colors:
            # Average the neighbor colors
            avg_color = np.mean(neighbor_colors, axis=0).astype(np.uint8)
            
            # Find closest color in palette (skip black at index 1)
            # Simple linear search is fine since palette is small
            min_dist = float('inf')
            best_idx = current_idx
            
            for i in range(len(all_colors)):
                if i != 1:  # Skip black
                    dist = np.sum((palette_array[i] - avg_color) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = i
            
            if best_idx != current_idx:
                smoothed_indices[r, c] = best_idx
                smoothed_count += 1
    
    # ==============================================
    # 4. CREATE FINAL MERGED SEGMENT
    # ==============================================
    black_pixels = np.sum(smoothed_indices == 1)
    colored_pixels = roi_height * roi_width - black_pixels
    
    print(f"\nFast border smoothing complete:")
    print(f"  ROI size: {roi_height}x{roi_width} = {roi_height*roi_width:,} pixels")
    print(f"  Colored pixels: {colored_pixels:,}")
    print(f"  Black pixels: {black_pixels:,} ({black_pixels/(roi_height*roi_width)*100:.1f}%)")
    print(f"  Palette size: {len(all_colors)} colors")
    print(f"  Border pixels smoothed: {smoothed_count:,}/{border_count:,}")
    
    merged_segment = {
        'top_left': (minr, minc),
        'shape': (roi_height, roi_width),
        'palette': all_colors,
        'indices': smoothed_indices.flatten().tolist(),
        'encoding': 'roi_merged_fast_smooth'
    }
    
    return [merged_segment]


def fast_border_smoothing_simple(region_components, roi_bbox):
    """
    Fast border smoothing using vectorized operations.
    Border pixels are replaced with average of their neighbors.
    """
    if not region_components:
        return []
    
    if len(region_components) == 1:
        return region_components
    
    print(f"\n{'='*60}")
    print(f"FAST BORDER SMOOTHING: {len(region_components)} components")
    print(f"{'='*60}")
    
    minr, minc, maxr, maxc = roi_bbox
    roi_height = maxr - minr
    roi_width = maxc - minc
    
    # ==============================================
    # 1. FAST MERGE (NO SMOOTHING)
    # ==============================================
    print("Merging components...")
    
    # Create canvas
    roi_indices = np.ones((roi_height, roi_width), dtype=np.uint16)
    
    # Collect colors
    all_colors = [(0, 0, 0)]  # Black at index 1
    color_to_idx = {(0, 0, 0): 1}
    
    # Place segments (last wins)
    for seg in reversed(region_components):
        seg_top_left = seg['top_left']
        seg_shape = seg['shape']
        seg_palette = seg['palette']
        seg_indices = np.array(seg['indices']).reshape(seg_shape)
        
        rel_row = seg_top_left[0] - minr
        rel_col = seg_top_left[1] - minc
        
        for r in range(seg_shape[0]):
            for c in range(seg_shape[1]):
                roi_r = rel_row + r
                roi_c = rel_col + c
                
                if 0 <= roi_r < roi_height and 0 <= roi_c < roi_width:
                    seg_idx = seg_indices[r, c]
                    if seg_idx != 1:
                        color = tuple(seg_palette[seg_idx])
                        if color not in color_to_idx:
                            color_to_idx[color] = len(all_colors)
                            all_colors.append(color)
                        roi_indices[roi_r, roi_c] = color_to_idx[color]
    
    print(f"  ROI: {roi_height}x{roi_width} pixels")
    print(f"  Colors: {len(all_colors)}")
    
    # ==============================================
    # 2. VECTORIZED BORDER DETECTION
    # ==============================================
    print("Detecting border pixels...")
    
    # Create mask of non-black pixels
    non_black_mask = (roi_indices != 1)
    
    # Detect borders using convolution - MUCH FASTER
    from scipy.ndimage import convolve
    
    # Create border mask: pixel is border if any neighbor is different
    kernel = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]]) / 4.0
    
    # Use convolution to check neighbors
    # This is approximate but fast
    convolved = convolve(roi_indices.astype(float), kernel, mode='reflect')
    
    # Border if value differs from center
    border_mask = np.abs(convolved - roi_indices) > 0.1
    border_mask &= non_black_mask  # Only non-black pixels can be borders
    
    border_count = np.sum(border_mask)
    print(f"  Border pixels detected: {border_count:,}")
    
    if border_count == 0:
        print("  No borders to smooth!")
        return [{
            'top_left': (minr, minc),
            'shape': (roi_height, roi_width),
            'palette': all_colors,
            'indices': roi_indices.flatten().tolist(),
            'encoding': 'roi_merged_no_smooth'
        }]
    
    # ==============================================
    # 3. FAST BORDER REPLACEMENT
    # ==============================================
    print("Replacing border pixels...")
    
    # Convert to RGB image for averaging
    palette_array = np.array(all_colors, dtype=np.uint8)
    rgb_image = palette_array[roi_indices]
    
    # Create smoothed copy
    smoothed_rgb = rgb_image.copy()
    
    # Get border coordinates
    border_coords = np.where(border_mask)
    
    # For each border pixel, average its 4 neighbors
    for idx in range(len(border_coords[0])):
        r, c = border_coords[0][idx], border_coords[1][idx]
        
        # Get 4-connected neighbors
        neighbors = []
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < roi_height and 0 <= nc < roi_width:
                if not border_mask[nr, nc]:  # Only use non-border neighbors
                    neighbors.append(rgb_image[nr, nc])
        
        if neighbors:
            # Average the neighbor colors
            avg_color = np.mean(neighbors, axis=0).astype(np.uint8)
            smoothed_rgb[r, c] = avg_color
    
    # ==============================================
    # 4. CONVERT BACK TO INDEXES
    # ==============================================
    print("Converting back to palette indexes...")
    
    # Create color to index mapping for fast lookup
    color_map = {tuple(color): idx for idx, color in enumerate(all_colors)}
    
    # Convert smoothed RGB back to indices
    smoothed_indices = np.ones((roi_height, roi_width), dtype=np.uint16)
    
    # Only update border pixels (non-border pixels keep original index)
    smoothed_indices = roi_indices.copy()
    
    # Update border pixels
    for idx in range(len(border_coords[0])):
        r, c = border_coords[0][idx], border_coords[1][idx]
        color = tuple(smoothed_rgb[r, c])
        
        if color in color_map:
            smoothed_indices[r, c] = color_map[color]
        else:
            # Find closest color
            min_dist = float('inf')
            best_idx = 1  # Default to black
            
            color_np = np.array(color)
            for i, pal_color in enumerate(all_colors):
                if i != 1:  # Skip black
                    dist = np.sum((np.array(pal_color) - color_np) ** 2)
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = i
            
            smoothed_indices[r, c] = best_idx
    
    # ==============================================
    # 5. CREATE RESULT
    # ==============================================
    print(f"\nFast border smoothing complete!")
    
    return [{
        'top_left': (minr, minc),
        'shape': (roi_height, roi_width),
        'palette': all_colors,
        'indices': smoothed_indices.flatten().tolist(),
        'encoding': 'roi_merged_fast_smooth'
    }]












def merge_region_components_better(region_components, roi_bbox):
    """
    Better merging: Sort segments by area (largest first) to give priority.
    This ensures important/large segments aren't obscured by small ones.
    """
    if not region_components:
        return []
    
    if len(region_components) == 1:
        return region_components
    
    print(f"\n{'='*60}")
    print(f"MERGING {len(region_components)} REGION COMPONENTS (sorted by area)")
    print(f"{'='*60}")
    
    minr, minc, maxr, maxc = roi_bbox
    roi_height = maxr - minr
    roi_width = maxc - minc
    
    # Sort segments by area (largest first)
    region_components_sorted = sorted(
        region_components,
        key=lambda seg: seg['shape'][0] * seg['shape'][1],
        reverse=True
    )
    
    # ==============================================
    # 1. CREATE EMPTY CANVAS FOR ROI
    # ==============================================
    roi_indices = np.ones((roi_height, roi_width), dtype=np.uint16)
    segment_ownership = np.zeros((roi_height, roi_width), dtype=np.int32)  # Which segment placed each pixel
    
    # Collect all colors
    all_colors = []
    color_to_index = {}
    
    # Ensure black is at index 1
    black_color = (0, 0, 0)
    if black_color not in all_colors:
        all_colors.append(black_color)
        color_to_index[black_color] = 1
    
    # ==============================================
    # 2. PLACE SEGMENTS (largest first)
    # ==============================================
    print(f"ROI canvas: {roi_height}x{roi_width} pixels")
    print(f"Processing {len(region_components_sorted)} segments (largest first)...")
    
    for seg_idx, seg in enumerate(region_components_sorted):
        seg_top_left = seg['top_left']
        seg_shape = seg['shape']
        seg_palette = seg['palette']
        seg_indices = np.array(seg['indices']).reshape(seg_shape)
        
        # Calculate position relative to ROI
        rel_row = seg_top_left[0] - minr
        rel_col = seg_top_left[1] - minc
        
        placed_pixels = 0
        skipped_pixels = 0
        
        # Place segment pixels
        for r in range(seg_shape[0]):
            for c in range(seg_shape[1]):
                roi_r = rel_row + r
                roi_c = rel_col + c
                
                if 0 <= roi_r < roi_height and 0 <= roi_c < roi_width:
                    seg_pixel_idx = seg_indices[r, c]
                    
                    if seg_pixel_idx != 1:  # Colored pixel
                        color = tuple(seg_palette[seg_pixel_idx])
                        
                        # Always place if pixel is black OR if no one has claimed it yet
                        if roi_indices[roi_r, roi_c] == 1 or segment_ownership[roi_r, roi_c] == 0:
                            # Map color to combined palette
                            if color not in color_to_index:
                                color_to_index[color] = len(all_colors)
                                all_colors.append(color)
                            
                            # Place the color
                            roi_indices[roi_r, roi_c] = color_to_index[color]
                            segment_ownership[roi_r, roi_c] = seg_idx + 1
                            placed_pixels += 1
                        else:
                            skipped_pixels += 1
        
        seg_area = seg_shape[0] * seg_shape[1]
        print(f"  Segment {seg_idx+1} ({seg_area:,} px): {placed_pixels} placed, {skipped_pixels} skipped")
    
    # ==============================================
    # 3. CREATE MERGED SEGMENT
    # ==============================================
    black_pixels = np.sum(roi_indices == 1)
    colored_pixels = roi_height * roi_width - black_pixels
    
    print(f"\nMerging statistics:")
    print(f"  ROI pixels: {roi_height * roi_width:,}")
    print(f"  Colored pixels after merge: {colored_pixels:,}")
    print(f"  Black pixels remaining: {black_pixels:,} ({black_pixels/(roi_height*roi_width)*100:.1f}%)")
    print(f"  Unique colors in merged palette: {len(all_colors)}")
    
    merged_segment = {
        'top_left': (minr, minc),
        'shape': (roi_height, roi_width),
        'palette': all_colors,
        'indices': roi_indices.flatten().tolist(),
        'method': 'color_quantization',
        'max_colors': len(all_colors),
        'actual_colors': len(all_colors),
        'encoding': 'roi_merged_sorted'
    }
    
    return [merged_segment]


def merge_region_components_overlap(region_components, roi_bbox):
    """
    Handle overlaps by prioritizing segments that have more colored pixels at overlap locations.
    """
    if not region_components:
        return []
    
    if len(region_components) == 1:
        return region_components
    
    print(f"\n{'='*60}")
    print(f"MERGING WITH OVERLAP HANDLING: {len(region_components)} components")
    print(f"{'='*60}")
    
    minr, minc, maxr, maxc = roi_bbox
    roi_height = maxr - minr
    roi_width = maxc - minc
    
    # ==============================================
    # 1. CREATE DATA STRUCTURES
    # ==============================================
    # We'll track multiple candidates for each pixel
    pixel_candidates = {}  # (r, c) -> list of (segment_idx, color)
    
    # Process all segments to collect candidates
    for seg_idx, seg in enumerate(region_components):
        seg_top_left = seg['top_left']
        seg_shape = seg['shape']
        seg_palette = seg['palette']
        seg_indices = np.array(seg['indices']).reshape(seg_shape)
        
        rel_row = seg_top_left[0] - minr
        rel_col = seg_top_left[1] - minc
        
        for r in range(seg_shape[0]):
            for c in range(seg_shape[1]):
                roi_r = rel_row + r
                roi_c = rel_col + c
                
                if 0 <= roi_r < roi_height and 0 <= roi_c < roi_width:
                    seg_pixel_idx = seg_indices[r, c]
                    
                    if seg_pixel_idx != 1:  # Colored pixel
                        color = tuple(seg_palette[seg_pixel_idx])
                        
                        key = (roi_r, roi_c)
                        if key not in pixel_candidates:
                            pixel_candidates[key] = []
                        
                        pixel_candidates[key].append((seg_idx, color))
    
    # ==============================================
    # 2. RESOLVE OVERLAPS
    # ==============================================
    # Create final canvas
    roi_indices = np.ones((roi_height, roi_width), dtype=np.uint16)
    
    # Collect all colors
    all_colors = []
    color_to_index = {}
    
    # Add black
    black_color = (0, 0, 0)
    all_colors.append(black_color)
    color_to_index[black_color] = 1
    
    # Resolve each pixel
    overlap_count = 0
    for (r, c), candidates in pixel_candidates.items():
        if len(candidates) == 1:
            # No overlap, just use this color
            _, color = candidates[0]
            
            if color not in color_to_index:
                color_to_index[color] = len(all_colors)
                all_colors.append(color)
            
            roi_indices[r, c] = color_to_index[color]
        else:
            # Overlap - choose the "best" candidate
            overlap_count += 1
            
            # Simple strategy: choose the first segment's color
            # (could be improved to choose based on segment importance)
            _, color = candidates[0]
            
            if color not in color_to_index:
                color_to_index[color] = len(all_colors)
                all_colors.append(color)
            
            roi_indices[r, c] = color_to_index[color]
    
    # ==============================================
    # 3. CREATE MERGED SEGMENT
    # ==============================================
    black_pixels = np.sum(roi_indices == 1)
    colored_pixels = roi_height * roi_width - black_pixels
    
    print(f"\nOverlap statistics:")
    print(f"  Total pixels: {roi_height * roi_width:,}")
    print(f"  Pixels with candidates: {len(pixel_candidates):,}")
    print(f"  Overlap pixels (multiple candidates): {overlap_count:,}")
    print(f"  Final colored pixels: {colored_pixels:,}")
    print(f"  Unique colors: {len(all_colors)}")
    
    merged_segment = {
        'top_left': (minr, minc),
        'shape': (roi_height, roi_width),
        'palette': all_colors,
        'indices': roi_indices.flatten().tolist(),
        'encoding': 'roi_merged_overlap'
    }
    
    return [merged_segment]


















def visualize_roi_components(overlap_canvas, component_count):
    """
    Visualize the current arrangement of components in ROI.
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. Show the combined ROI
        axes[0].imshow(overlap_canvas)
        axes[0].set_title('Combined ROI Components')
        axes[0].axis('off')
        
        # 2. Show black pixel mask (gaps)
        black_mask = (overlap_canvas.sum(axis=2) == 0)
        axes[1].imshow(black_mask, cmap='gray')
        axes[1].set_title(f'Black Pixels (Gaps): {np.sum(black_mask):,}')
        axes[1].axis('off')
        
        # 3. Show overlap regions
        overlap_mask = (component_count > 1)
        axes[2].imshow(overlap_mask, cmap='hot')
        axes[2].set_title(f'Overlap Regions: {np.sum(overlap_mask):,}')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("Matplotlib not available for visualization")

def merge_region_components_simple(region_components, roi_bbox):
    """
    Merge region components by placing them on ROI canvas.
    Colored pixels override black pixels.
    
    Args:
        region_components: List of compressed segments from same ROI
        roi_bbox: (minr, minc, maxr, maxc) of ROI in original image
    
    Returns:
        List of merged segments (ideally 1 segment)
    """
    if not region_components:
        return []
    
    if len(region_components) == 1:
        return region_components
    
    print(f"\n{'='*60}")
    print(f"MERGING {len(region_components)} REGION COMPONENTS")
    print(f"{'='*60}")
    
    minr, minc, maxr, maxc = roi_bbox
    roi_height = maxr - minr
    roi_width = maxc - minc
    
    # ==============================================
    # 1. CREATE EMPTY CANVAS FOR ROI
    # ==============================================
    # Start with all pixels as black (index 0)
    roi_indices = np.zeros((roi_height, roi_width), dtype=np.uint16)
    
    # Collect all colors from all segments
    all_colors = []
    color_to_index = {}
    
    # ALWAYS add black as the first color (index 0)
    black_color = (0, 0, 0)
    all_colors.append(black_color)
    color_to_index[black_color] = 0
    
    print(f"ROI canvas: {roi_height}x{roi_width} pixels")
    print(f"Placing {len(region_components)} segments...")
    
    # ==============================================
    # 2. PLACE ALL SEGMENTS (last segment wins)
    # ==============================================
    # Process in reverse so last segments have priority
    for seg_idx, seg in enumerate(reversed(region_components)):
        seg_top_left = seg['top_left']  # Absolute coordinates
        seg_shape = seg['shape']
        seg_palette = seg['palette']  # List of [R, G, B] colors
        seg_indices = np.array(seg['indices']).reshape(seg_shape)
        
        # Calculate position relative to ROI
        rel_row = seg_top_left[0] - minr
        rel_col = seg_top_left[1] - minc
        
        colored_placed = 0
        
        # Place segment on ROI canvas
        for r in range(seg_shape[0]):
            for c in range(seg_shape[1]):
                roi_r = rel_row + r
                roi_c = rel_col + c
                
                # Check bounds
                if 0 <= roi_r < roi_height and 0 <= roi_c < roi_width:
                    seg_pixel_idx = seg_indices[r, c]
                    
                    # Get the actual color from segment's palette
                    if seg_pixel_idx < len(seg_palette):
                        color_tuple = tuple(seg_palette[seg_pixel_idx])
                        
                        # Check if this pixel is black
                        is_black = (color_tuple == black_color)
                        
                        if not is_black:
                            # This is a colored pixel
                            
                            # Map color to combined palette
                            if color_tuple not in color_to_index:
                                color_to_index[color_tuple] = len(all_colors)
                                all_colors.append(color_tuple)
                            
                            # Always place colored pixel (overwrites whatever was there)
                            roi_indices[roi_r, roi_c] = color_to_index[color_tuple]
                            colored_placed += 1
        
        print(f"  Segment {len(region_components)-seg_idx}: {colored_placed} colored pixels placed")
    
    # ==============================================
    # 3. CREATE MERGED SEGMENT
    # ==============================================
    # Count statistics
    black_pixels = np.sum(roi_indices == 0)  # Index 0 is black
    colored_pixels_total = roi_height * roi_width - black_pixels
    
    print(f"\nMerging complete:")
    print(f"  ROI size: {roi_height}x{roi_width} = {roi_height*roi_width:,} pixels")
    print(f"  Black pixels: {black_pixels:,} ({black_pixels/(roi_height*roi_width)*100:.1f}%)")
    print(f"  Colored pixels: {colored_pixels_total:,}")
    print(f"  Unique colors: {len(all_colors)}")
    
    # Create the merged segment
    merged_segment = {
        'top_left': (minr, minc),
        'shape': (roi_height, roi_width),
        'palette': all_colors,
        'indices': roi_indices.flatten().tolist(),
        'method': 'merged',
        'actual_colors': len(all_colors),
        'encoding': 'roi_merged'
    }
    
    return [merged_segment]


def visualize_merged_result(merged_segments, roi_shape, offset_r, offset_c, original_components=None):
    """
    Visualize the merged result for a ROI with side-by-side comparison.
    
    Args:
        merged_segments: List of merged segments (usually 1)
        roi_shape: (height, width) of the ROI
        offset_r, offset_c: ROI offset in original image
        original_components: Original components for comparison (optional)
    """
    import matplotlib.pyplot as plt
    
    h, w = roi_shape
    
    # ==============================================
    # 1. CREATE MERGED CANVAS
    # ==============================================
    merged_canvas = np.zeros((h, w, 3), dtype=np.uint8)
    
    for seg in merged_segments:
        top_left = seg['top_left']
        shape = seg['shape']
        indices = np.array(seg['indices']).reshape(shape)
        palette = seg['palette']
        
        # Calculate position relative to ROI
        rel_r = top_left[0] - offset_r
        rel_c = top_left[1] - offset_c
        
        # Create RGB image from indices
        for r in range(shape[0]):
            for c in range(shape[1]):
                canvas_r = rel_r + r
                canvas_c = rel_c + c
                
                if 0 <= canvas_r < h and 0 <= canvas_c < w:
                    idx = indices[r, c]
                    if idx < len(palette):
                        color = palette[idx]
                        merged_canvas[canvas_r, canvas_c] = color
    
    # ==============================================
    # 2. CREATE BLACK MASK
    # ==============================================
    black_mask = np.all(merged_canvas == [0, 0, 0], axis=2)
    colored_mask = ~black_mask
    
    # Create colored-only image (black pixels become transparent in visualization)
    colored_only = merged_canvas.copy()
    
    # ==============================================
    # 3. CREATE ORIGINAL COMPONENTS CANVAS (if provided)
    # ==============================================
    if original_components:
        original_canvas = np.zeros((h, w, 3), dtype=np.uint8)
        
        for seg in original_components:
            top_left = seg['top_left']
            shape = seg['shape']
            indices = np.array(seg['indices']).reshape(shape)
            palette = seg['palette']
            
            rel_r = top_left[0] - offset_r
            rel_c = top_left[1] - offset_c
            
            for r in range(shape[0]):
                for c in range(shape[1]):
                    canvas_r = rel_r + r
                    canvas_c = rel_c + c
                    
                    if 0 <= canvas_r < h and 0 <= canvas_c < w:
                        idx = indices[r, c]
                        if idx < len(palette):
                            color = palette[idx]
                            original_canvas[canvas_r, canvas_c] = color
    
    # ==============================================
    # 4. CREATE VISUALIZATION
    # ==============================================
    if original_components:
        # Show 4 subplots: original, merged, colored only, black mask
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original components
        axes[0, 0].imshow(original_canvas)
        axes[0, 0].set_title(f'Original: {len(original_components)} segments')
        axes[0, 0].axis('off')
        
        # Merged result
        axes[0, 1].imshow(merged_canvas)
        axes[0, 1].set_title(f'Merged: {len(merged_segments)} segment(s)')
        axes[0, 1].axis('off')
        
        # Colored pixels only (black as white for visibility)
        colored_display = colored_only.copy()
        colored_display[black_mask] = [255, 255, 255]  # White background
        axes[0, 2].imshow(colored_display)
        axes[0, 2].set_title(f'Colored pixels: {np.sum(colored_mask):,}')
        axes[0, 2].axis('off')
        
        # Black pixel mask
        axes[1, 0].imshow(black_mask, cmap='gray')
        axes[1, 0].set_title(f'Black pixels: {np.sum(black_mask):,} ({np.sum(black_mask)/(h*w)*100:.1f}%)')
        axes[1, 0].axis('off')
        
        # Color distribution
        if np.sum(colored_mask) > 0:
            # Get all colored pixels
            colored_pixels = merged_canvas[colored_mask].reshape(-1, 3)
            
            # Create histogram of color intensities
            intensities = np.mean(colored_pixels, axis=1)
            axes[1, 1].hist(intensities, bins=50, color='skyblue', edgecolor='black')
            axes[1, 1].set_title('Color intensity distribution')
            axes[1, 1].set_xlabel('Intensity (0-255)')
            axes[1, 1].set_ylabel('Count')
        else:
            axes[1, 1].text(0.5, 0.5, 'No colored pixels', 
                          ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].axis('off')
        
        # Stats text
        stats_text = f"""
        ROI Size: {h}x{w} = {h*w:,} pixels
        Colored: {np.sum(colored_mask):,} ({np.sum(colored_mask)/(h*w)*100:.1f}%)
        Black: {np.sum(black_mask):,} ({np.sum(black_mask)/(h*w)*100:.1f}%)
        Colors: {len(merged_segments[0]['palette']) if merged_segments else 0}
        """
        axes[1, 2].text(0.1, 0.5, stats_text, fontsize=10, 
                       verticalalignment='center', transform=axes[1, 2].transAxes)
        axes[1, 2].axis('off')
        
    else:
        # Show 2x2 grid for merged only
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Merged result
        axes[0, 0].imshow(merged_canvas)
        axes[0, 0].set_title(f'Merged ROI ({len(merged_segments)} segment(s))')
        axes[0, 0].axis('off')
        
        # Colored pixels only
        colored_display = colored_only.copy()
        colored_display[black_mask] = [255, 255, 255]  # White background
        axes[0, 1].imshow(colored_display)
        axes[0, 1].set_title(f'Colored pixels only ({np.sum(colored_mask):,})')
        axes[0, 1].axis('off')
        
        # Black pixel mask
        axes[1, 0].imshow(black_mask, cmap='gray')
        axes[1, 0].set_title(f'Black pixels: {np.sum(black_mask):,} ({np.sum(black_mask)/(h*w)*100:.1f}%)')
        axes[1, 0].axis('off')
        
        # Color preview (first few colors in palette)
        if merged_segments and len(merged_segments[0]['palette']) > 1:
            palette = merged_segments[0]['palette']
            # Skip black (first color)
            colors_to_show = min(10, len(palette) - 1)
            
            # Create color swatches
            swatch_height = 1
            swatch_width = colors_to_show
            
            color_swatches = np.zeros((swatch_height, swatch_width, 3), dtype=np.uint8)
            for i in range(colors_to_show):
                color_swatches[0, i] = palette[i + 1]  # Skip black
            
            axes[1, 1].imshow(color_swatches)
            axes[1, 1].set_title(f'Top {colors_to_show} colors (excl. black)')
            axes[1, 1].axis('off')
        else:
            axes[1, 1].text(0.5, 0.5, 'No colors to display', 
                          ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed stats
    print(f"\n{'='*60}")
    print(f"VISUALIZATION STATISTICS")
    print(f"{'='*60}")
    print(f"ROI Size: {h}x{w} = {h*w:,} pixels")
    print(f"Colored pixels: {np.sum(colored_mask):,} ({np.sum(colored_mask)/(h*w)*100:.1f}%)")
    print(f"Black pixels: {np.sum(black_mask):,} ({np.sum(black_mask)/(h*w)*100:.1f}%)")
    
    if merged_segments:
        seg = merged_segments[0]
        print(f"Segment shape: {seg['shape']}")
        print(f"Palette size: {len(seg['palette'])} colors")
        print(f"Top-left position: {seg['top_left']}")
        
        # Count how many unique colors actually appear
        indices = np.array(seg['indices'])
        unique_indices = set(indices)
        print(f"Unique color indices used: {len(unique_indices)}")
        
        # Show color frequencies
        if len(seg['palette']) <= 20:  # Only if palette is small
            print("\nColor frequencies:")
            for idx in sorted(unique_indices):
                count = np.sum(indices == idx)
                color = seg['palette'][idx]
                if tuple(color) == (0, 0, 0):
                    color_name = "BLACK"
                else:
                    color_name = f"RGB{tuple(color)}"
                print(f"  Index {idx}: {color_name} - {count:,} pixels ({count/len(indices)*100:.1f}%)")
    
    return merged_canvas



def debug_merge_components(region_components, roi_bbox):
    """
    Debug version with extra information.
    """
    if not region_components:
        return []
    
    print(f"\n{'='*60}")
    print(f"DEBUG MERGING {len(region_components)} COMPONENTS")
    print(f"{'='*60}")
    
    # First, print info about each component
    for i, seg in enumerate(region_components):
        print(f"\nSegment {i}:")
        print(f"  Top-left: {seg['top_left']}")
        print(f"  Shape: {seg['shape']}")
        print(f"  Palette size: {len(seg['palette'])}")
        
        # Count black pixels in this segment
        indices = np.array(seg['indices'])
        palette = seg['palette']
        
        black_count = 0
        for idx in indices:
            if idx < len(palette):
                color = tuple(palette[idx])
                if color == (0, 0, 0):
                    black_count += 1
        
        print(f"  Black pixels: {black_count}/{len(indices)} ({black_count/len(indices)*100:.1f}%)")
    
    # Now do the merge
    return merge_region_components_simple(region_components, roi_bbox)















































































def get_all_unique_colors(region_image, top_left_coords):
    """
    Get ALL unique colors from the region without any clustering.
    Returns the same data structure as compression functions.
    """
    if region_image is None or region_image.size == 0:
        return None
    
    h, w, _ = region_image.shape
    total_pixels = h * w
    
    print(f"Getting all unique colors: {h}x{w} region")
    print(f"Top-left: {top_left_coords}")
    
    # ==============================================
    # 1. GET ALL UNIQUE COLORS
    # ==============================================
    pixels = region_image.reshape(-1, 3)
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    unique_count = len(unique_colors)
    
    print(f"  Found {unique_count:,} unique colors")
    
    # ==============================================
    # 2. CREATE COLOR TO INDEX MAPPING
    # ==============================================
    # Create a dictionary mapping each color to its index
    color_to_index = {}
    palette = []
    
    for i, color in enumerate(unique_colors):
        color_tuple = tuple(color)
        color_to_index[color_tuple] = i
        palette.append(color.tolist())  # Convert to list for JSON serialization
    
    print(f"  Created palette with {len(palette)} colors")
    
    # ==============================================
    # 3. CREATE INDICES FOR EACH PIXEL
    # ==============================================
    # Map each pixel to its color index
    indices_flat = []
    for pixel in pixels:
        color_tuple = tuple(pixel)
        indices_flat.append(color_to_index[color_tuple])
    
    # ==============================================
    # 4. CALCULATE BASIC STATISTICS
    # ==============================================
    actual_colors = len(palette)
    original_size = total_pixels * 3
    
    # Determine index data type
    if actual_colors <= 256:
        index_dtype = np.uint8
        bytes_per_index = 1
    else:
        index_dtype = np.uint16
        bytes_per_index = 2
    
    # Calculate compressed size
    palette_size = actual_colors * 3
    indices_size = total_pixels * bytes_per_index
    metadata_size = 50  # Approximate metadata size
    
    compressed_size = palette_size + indices_size + metadata_size
    
    # Calculate compression ratio
    compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
    
    # PSNR is perfect since we're using all original colors
    psnr = float('inf')
    
    # ==============================================
    # 5. CREATE OUTPUT DATA STRUCTURE
    # ==============================================
    compressed_data = {
        'method': 'exact_colors',
        'top_left': top_left_coords,
        'shape': (h, w),
        'palette': palette,  # All unique colors
        'indices': indices_flat,
        'max_colors': actual_colors,
        'actual_colors': actual_colors,
        'index_dtype': str(index_dtype),
        'original_size': original_size,
        'compressed_size': compressed_size,
        'compression_ratio': compression_ratio,
        'mse': 0.0,  # Perfect reconstruction
        'psnr': psnr,
        'encoding': 'exact'  # Mark as exact color representation
    }
    
    print(f"  Palette size: {actual_colors} colors")
    print(f"  Original: {original_size:,} bytes")
    print(f"  Compressed: {compressed_size:,} bytes")
    print(f"  Ratio: {compression_ratio:.2f}:1")
    print(f"  PSNR: Perfect (exact colors)")
    
    return compressed_data


def get_all_colors_fast(region_image, top_left_coords):
    """
    Faster version using numpy operations.
    Returns all unique colors without compression.
    """
    if region_image is None or region_image.size == 0:
        return None
    
    h, w, _ = region_image.shape
    total_pixels = h * w
    
    print(f"Fast exact colors: {h}x{w} region")
    
    # Reshape to pixel array
    pixels = region_image.reshape(-1, 3)
    
    # Get unique colors and their indices
    unique_colors, inverse_indices = np.unique(pixels, axis=0, return_inverse=True)
    unique_count = len(unique_colors)
    
    print(f"  Unique colors: {unique_count:,}")
    
    # Convert palette to list format
    palette = unique_colors.tolist()
    
    # Indices are already in inverse_indices
    indices_flat = inverse_indices.tolist()
    
    # Calculate stats
    actual_colors = unique_count
    original_size = total_pixels * 3
    
    # Data type for indices
    if actual_colors <= 256:
        index_dtype = np.uint8
        bytes_per_index = 1
    else:
        index_dtype = np.uint16
        bytes_per_index = 2
    
    compressed_size = actual_colors * 3 + total_pixels * bytes_per_index + 50
    
    # Create result
    result = {
        'method': 'exact_colors_fast',
        'top_left': top_left_coords,
        'shape': (h, w),
        'palette': palette,
        'indices': indices_flat,
        'actual_colors': actual_colors,
        'original_size': original_size,
        'compressed_size': compressed_size,
        'compression_ratio': original_size / compressed_size,
        'psnr': float('inf'),
        'mse': 0.0,
        'encoding': 'exact_fast'
    }
    
    print(f"  Compression: {original_size:,} → {compressed_size:,} bytes")
    
    return result










































from sklearn.cluster import DBSCAN
import numpy as np

def cluster_palette_colors(compressed_data, eps=10.0, min_samples=2, max_colors_per_cluster=5):
    """
    Cluster similar colors in palette using DBSCAN.
    Colors within 'eps' distance are clustered together.
    Each cluster is replaced with its average color.
    
    SPECIAL RULE: Black [0, 0, 0] is NEVER clustered - always kept as is.
    """
    print(f"\n{'='*60}")
    print(f"CLUSTERING PALETTE COLORS (Black preserved)")
    print(f"{'='*60}")
    
    # Extract original data
    palette = np.array(compressed_data['palette'], dtype=np.uint8)
    indices = np.array(compressed_data['indices'])
    h, w = compressed_data['shape']
    top_left = compressed_data['top_left']
    
    original_colors = len(palette)
    print(f"Original palette: {original_colors} colors")
    print(f"Clustering parameters: eps={eps}, min_samples={min_samples}")
    
    # ==============================================
    # 1. SEPARATE BLACK FROM OTHER COLORS
    # ==============================================
    # Find which indices correspond to black [0, 0, 0]
    black_indices = []
    non_black_indices = []
    
    for i, color in enumerate(palette):
        if np.array_equal(color, [0, 0, 0]):
            black_indices.append(i)
        else:
            non_black_indices.append(i)
    
    print(f"  Black colors found: {len(black_indices)}")
    print(f"  Non-black colors: {len(non_black_indices)}")
    
    if len(non_black_indices) == 0:
        print("  Only black colors - nothing to cluster!")
        return compressed_data
    
    # ==============================================
    # 2. CLUSTER ONLY NON-BLACK COLORS
    # ==============================================
    non_black_palette = palette[non_black_indices]
    
    # Normalize colors to 0-1 for better distance calculation
    palette_normalized = non_black_palette.astype(float) / 255.0
    
    # Apply DBSCAN (only to non-black colors)
    dbscan = DBSCAN(eps=eps/255.0, min_samples=min_samples, metric='euclidean')
    cluster_labels = dbscan.fit_predict(palette_normalized)
    
    # Count clusters (-1 means noise/outlier colors)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = np.sum(cluster_labels == -1)
    
    print(f"DBSCAN found {n_clusters} clusters + {n_noise} noise colors (excluding black)")
    
    # ==============================================
    # 3. CREATE NEW PALETTE (INCLUDING BLACK)
    # ==============================================
    new_palette = []
    color_mapping = {}  # Old index → new index
    
    # FIRST: Add all black colors exactly as they are
    for old_black_idx in black_indices:
        new_palette.append(palette[old_black_idx])
        color_mapping[old_black_idx] = len(new_palette) - 1
    
    # SECOND: Process non-black clusters
    unique_labels = set(cluster_labels)
    
    for label in unique_labels:
        if label == -1:
            # Noise points - keep them as individual colors
            noise_mask = cluster_labels == -1
            noise_relative_indices = np.where(noise_mask)[0]
            
            for rel_idx in noise_relative_indices:
                # Convert relative index back to original palette index
                old_idx = non_black_indices[rel_idx]
                new_palette.append(palette[old_idx])
                color_mapping[old_idx] = len(new_palette) - 1
        else:
            # Regular cluster
            cluster_mask = cluster_labels == label
            cluster_relative_indices = np.where(cluster_mask)[0]
            
            # Convert relative indices to original palette indices
            cluster_original_indices = [non_black_indices[idx] for idx in cluster_relative_indices]
            
            if len(cluster_original_indices) > max_colors_per_cluster:
                print(f"  Cluster {label} has {len(cluster_original_indices)} colors > {max_colors_per_cluster}, splitting...")
                
                # Get the actual colors for this cluster
                cluster_colors = palette[cluster_original_indices]
                
                # Split large cluster
                split_clusters = split_large_cluster(cluster_colors, int(max_colors_per_cluster))
                
                for split_cluster in split_clusters:
                    avg_color = np.mean(split_cluster, axis=0).astype(np.uint8)
                    new_palette.append(avg_color)
                    
                    # Map all colors in this split to the same new index
                    for color in split_cluster:
                        # Find original index for this color
                        old_idx = find_color_index(palette, color)
                        if old_idx is not None:
                            color_mapping[old_idx] = len(new_palette) - 1
            else:
                # Average colors in cluster
                cluster_colors = palette[cluster_original_indices]
                avg_color = np.mean(cluster_colors, axis=0).astype(np.uint8)
                new_palette.append(avg_color)
                
                # Map all colors in cluster to the same new index
                for old_idx in cluster_original_indices:
                    color_mapping[old_idx] = len(new_palette) - 1
    
    new_palette = np.array(new_palette)
    new_color_count = len(new_palette)
    
    # Verify black is preserved
    black_preserved = any(np.array_equal(color, [0, 0, 0]) for color in new_palette)
    print(f"  Black preserved: {black_preserved}")
    
    print(f"New palette: {new_color_count} colors")
    print(f"Color reduction: {original_colors} → {new_color_count} "
          f"({(original_colors - new_color_count)/original_colors*100:.1f}%)")
    
    # ==============================================
    # 4. UPDATE INDICES WITH NEW COLOR MAPPING
    # ==============================================
    # Create a mapping array for fast lookup
    mapping_array = np.zeros(original_colors, dtype=np.uint16)
    for old_idx, new_idx in color_mapping.items():
        mapping_array[old_idx] = new_idx
    
    # Update indices
    new_indices = mapping_array[indices].tolist()
    
    # ==============================================
    # 5. CALCULATE STATISTICS
    # ==============================================
    total_pixels = h * w
    original_size = compressed_data.get('original_size', total_pixels * 3)
    
    # Calculate new compressed size
    if new_color_count <= 256:
        bytes_per_index = 1
        index_dtype = 'uint8'
    else:
        bytes_per_index = 2
        index_dtype = 'uint16'
    
    new_palette_size = new_color_count * 3
    new_indices_size = total_pixels * bytes_per_index
    metadata_size = 100  # Approximate
    
    new_compressed_size = new_palette_size + new_indices_size + metadata_size
    compression_ratio = original_size / new_compressed_size if new_compressed_size > 0 else 0
    
    # Estimate PSNR
    mse = estimate_clustering_mse(palette, new_palette, color_mapping, indices, h, w)
    psnr = 10 * np.log10(255*255/mse) if mse > 0 else float('inf')
    
    # ==============================================
    # 6. CREATE NEW COMPRESSED DATA
    # ==============================================
    new_compressed_data = {
        'method': 'clustered_colors',
        'top_left': top_left,
        'shape': (h, w),
        'palette': new_palette.tolist(),
        'indices': new_indices,
        'original_unique_colors': original_colors,
        'compressed_colors': new_color_count,
        'index_dtype': index_dtype,
        'original_size': original_size,
        'compressed_size': new_compressed_size,
        'compression_ratio': compression_ratio,
        'mse': float(mse),
        'psnr': float(psnr),
        'clustering_params': {
            'eps': eps,
            'min_samples': min_samples,
            'max_colors_per_cluster': max_colors_per_cluster
        },
        'encoding': 'dbscan_clustered',
        'black_preserved': True  # Flag to indicate black was preserved
    }
    
    print(f"\nClustering complete:")
    print(f"  Colors: {original_colors} → {new_color_count}")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  Compression: {original_size:,} → {new_compressed_size:,} bytes")
    print(f"  Ratio: {compression_ratio:.2f}:1")
    
    return new_compressed_data

def split_large_cluster(cluster_colors, max_size):
    """
    Split a large cluster into smaller sub-clusters.
    Uses simple K-means to split.
    """
    from sklearn.cluster import KMeans
    
    n_splits = max(2, len(cluster_colors) // max_size + 1)
    kmeans = KMeans(n_clusters=n_splits, random_state=42, n_init=3)
    
    labels = kmeans.fit_predict(cluster_colors.astype(float))
    
    splits = []
    for i in range(n_splits):
        mask = labels == i
        if np.any(mask):
            splits.append(cluster_colors[mask])
    
    return splits


def find_color_index(palette, color):
    """Find the index of a color in the palette."""
    matches = np.all(palette == color, axis=1)
    if np.any(matches):
        return np.where(matches)[0][0]
    return None


def estimate_clustering_mse(orig_palette, new_palette, mapping, indices, h, w):
    """
    Estimate MSE from clustering.
    Simplified calculation: average squared color distance.
    """
    total_error = 0
    pixel_count = h * w
    
    # Calculate average error per color mapping
    for old_idx, new_idx in mapping.items():
        orig_color = orig_palette[old_idx].astype(float)
        new_color = new_palette[new_idx].astype(float)
        error = np.sum((orig_color - new_color) ** 2)
        total_error += error
    
    # Average error
    avg_error = total_error / len(mapping) if mapping else 0
    
    # Scale by pixel count (simplified)
    estimated_mse = avg_error * 0.3  # Empirical scaling factor
    
    return max(estimated_mse, 0.1)  # Avoid zero


def hierarchical_color_clustering(compressed_data, quality=85):
    """
    Alternative: Hierarchical clustering with quality control.
    """
    palette = np.array(compressed_data['palette'], dtype=np.uint8)
    indices = np.array(compressed_data['indices'])
    
    original_colors = len(palette)
    target_colors = max(2, int(original_colors * quality / 100))
    
    print(f"Hierarchical clustering: {original_colors} → {target_colors} colors")
    
    # Use K-means for hierarchical-like clustering
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(n_clusters=target_colors, random_state=42, n_init=3)
    kmeans.fit(palette.astype(float))
    
    # Get new palette and mapping
    new_palette = kmeans.cluster_centers_.astype(np.uint8)
    new_indices = kmeans.labels_[indices].tolist()
    
    # Create new compressed data
    return create_clustered_result(
        compressed_data, new_palette, new_indices, 'hierarchical_clustered'
    )


def create_clustered_result(original_data, new_palette, new_indices, method_name):
    """Helper to create clustered result structure."""
    h, w = original_data['shape']
    total_pixels = h * w
    
    new_color_count = len(new_palette)
    
    # Calculate sizes
    original_size = original_data.get('original_size', total_pixels * 3)
    
    if new_color_count <= 256:
        bytes_per_index = 1
        index_dtype = 'uint8'
    else:
        bytes_per_index = 2
        index_dtype = 'uint16'
    
    new_compressed_size = new_color_count * 3 + total_pixels * bytes_per_index + 100
    
    return {
        'method': method_name,
        'top_left': original_data['top_left'],
        'shape': (h, w),
        'palette': new_palette.tolist(),
        'indices': new_indices,
        'compressed_colors': new_color_count,
        'index_dtype': index_dtype,
        'original_size': original_size,
        'compressed_size': new_compressed_size,
        'compression_ratio': original_size / new_compressed_size,
        'encoding': method_name
    }


























































def visualize_all_roi_components(all_roi_components, image_shape):
    """
    Visualize all ROI components placed in their correct positions in the full image.
    
    Args:
        all_roi_components: List of ROI components (each is a merged segment)
        image_shape: (height, width) of the original image
    """
    import matplotlib.pyplot as plt
    
    h, w = image_shape
    
    print(f"\n{'='*60}")
    print(f"VISUALIZING ALL ROI COMPONENTS")
    print(f"{'='*60}")
    print(f"Image size: {h}x{w}")
    print(f"Number of ROIs: {len(all_roi_components)}")
    
    # ==============================================
    # 1. CREATE FULL IMAGE CANVAS WITH ALL ROIS
    # ==============================================
    full_canvas = np.zeros((h, w, 3), dtype=np.uint8)
    roi_mask = np.zeros((h, w), dtype=bool)
    
    for roi_idx, roi in enumerate(all_roi_components):
        top_left = roi['top_left']
        shape = roi['shape']
        indices = np.array(roi['indices']).reshape(shape)
        palette = roi['palette']
        
        tl_r, tl_c = top_left
        roi_h, roi_w = shape
        
        # Place ROI on canvas
        for r in range(roi_h):
            for c in range(roi_w):
                img_r = tl_r + r
                img_c = tl_c + c
                
                if 0 <= img_r < h and 0 <= img_c < w:
                    idx = indices[r, c]
                    if idx < len(palette):
                        color = palette[idx]
                        # Only place if not black
                        if tuple(color) != (0, 0, 0):
                            full_canvas[img_r, img_c] = color
                            roi_mask[img_r, img_c] = True
    
    # ==============================================
    # 2. CALCULATE STATISTICS
    # ==============================================
    total_pixels = h * w
    roi_pixels = np.sum(roi_mask)
    non_roi_pixels = total_pixels - roi_pixels
    
    print(f"\nGlobal Statistics:")
    print(f"  Total image pixels: {total_pixels:,}")
    print(f"  ROI covered pixels: {roi_pixels:,} ({roi_pixels/total_pixels*100:.1f}%)")
    print(f"  Non-ROI pixels: {non_roi_pixels:,} ({non_roi_pixels/total_pixels*100:.1f}%)")
    
    # ROI-specific stats
    print(f"\nROI-specific statistics:")
    for i, roi in enumerate(all_roi_components):
        tl_r, tl_c = roi['top_left']
        roi_h, roi_w = roi['shape']
        indices = np.array(roi['indices'])
        
        # Count non-black pixels
        palette = roi['palette']
        colored_pixels = 0
        for idx in indices:
            if idx < len(palette):
                color = palette[idx]
                if tuple(color) != (0, 0, 0):
                    colored_pixels += 1
        
        roi_total = roi_h * roi_w
        print(f"  ROI {i}: {roi_h}x{roi_w} at ({tl_r},{tl_c})")
        print(f"    Colored pixels: {colored_pixels:,}/{roi_total:,} ({colored_pixels/roi_total*100:.1f}%)")
        print(f"    Palette size: {len(palette)} colors")
    
    # ==============================================
    # 3. CREATE VISUALIZATION
    # ==============================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 3.1 Full reconstruction
    axes[0, 0].imshow(full_canvas)
    axes[0, 0].set_title(f'All ROIs ({len(all_roi_components)} total)')
    axes[0, 0].axis('off')
    
    # 3.2 ROI mask
    axes[0, 1].imshow(roi_mask, cmap='viridis')
    axes[0, 1].set_title(f'ROI Coverage: {roi_pixels:,} pixels ({roi_pixels/total_pixels*100:.1f}%)')
    axes[0, 1].axis('off')
    
    # 3.3 Individual ROI preview
    roi_preview = np.zeros((h, w, 3), dtype=np.uint8)
    # Use different colors for different ROIs
    roi_colors = [
        [255, 0, 0],    # Red
        [0, 255, 0],    # Green
        [0, 0, 255],    # Blue
        [255, 255, 0],  # Yellow
        [255, 0, 255],  # Magenta
        [0, 255, 255],  # Cyan
        [128, 0, 0],    # Dark red
        [0, 128, 0],    # Dark green
        [0, 0, 128],    # Dark blue
    ]
    
    for i, roi in enumerate(all_roi_components):
        color = roi_colors[i % len(roi_colors)]
        tl_r, tl_c = roi['top_left']
        roi_h, roi_w = roi['shape']
        
        # Draw ROI bounding box
        roi_preview[tl_r:tl_r+roi_h, tl_c] = color
        roi_preview[tl_r:tl_r+roi_h, tl_c+roi_w-1] = color
        roi_preview[tl_r, tl_c:tl_c+roi_w] = color
        roi_preview[tl_r+roi_h-1, tl_c:tl_c+roi_w] = color
    
    axes[0, 2].imshow(roi_preview)
    axes[0, 2].set_title('ROI Boundaries (different colors)')
    axes[0, 2].axis('off')
    
    # 3.4 Colored pixels only (white background)
    colored_display = full_canvas.copy()
    colored_mask = np.any(full_canvas != [0, 0, 0], axis=2)
    colored_display[~colored_mask] = [255, 255, 255]  # White background
    
    axes[1, 0].imshow(colored_display)
    axes[1, 0].set_title(f'Colored pixels only ({np.sum(colored_mask):,})')
    axes[1, 0].axis('off')
    
    # 3.5 Palette sizes bar chart
    palette_sizes = [len(roi['palette']) for roi in all_roi_components]
    roi_indices = list(range(len(all_roi_components)))
    
    axes[1, 1].bar(roi_indices, palette_sizes, color='skyblue', edgecolor='black')
    axes[1, 1].set_title('Palette sizes per ROI')
    axes[1, 1].set_xlabel('ROI Index')
    axes[1, 1].set_ylabel('Number of colors')
    axes[1, 1].set_xticks(roi_indices)
    
    # 3.6 Statistics text
    stats_text = f"""
    Image: {h}x{w} = {total_pixels:,} px
    ROIs: {len(all_roi_components)}
    ROI Coverage: {roi_pixels:,} px ({roi_pixels/total_pixels*100:.1f}%)
    Empty: {non_roi_pixels:,} px ({non_roi_pixels/total_pixels*100:.1f}%)
    
    Average palette: {np.mean(palette_sizes):.1f} colors
    Min palette: {min(palette_sizes) if palette_sizes else 0}
    Max palette: {max(palette_sizes) if palette_sizes else 0}
    """
    
    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=10, 
                   verticalalignment='center', transform=axes[1, 2].transAxes)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return full_canvas, roi_mask


def visualize_individual_roi(roi_component, roi_index=None):
    """
    Visualize a single ROI component in detail.
    
    Args:
        roi_component: Single ROI component dictionary
        roi_index: Optional index for labeling
    """
    import matplotlib.pyplot as plt
    
    top_left = roi_component['top_left']
    shape = roi_component['shape']
    indices = np.array(roi_component['indices']).reshape(shape)
    palette = roi_component['palette']
    
    h, w = shape
    
    # Create ROI image
    roi_image = np.zeros((h, w, 3), dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            idx = indices[r, c]
            if idx < len(palette):
                roi_image[r, c] = palette[idx]
    
    # Calculate statistics
    black_mask = np.all(roi_image == [0, 0, 0], axis=2)
    colored_mask = ~black_mask
    colored_pixels = np.sum(colored_mask)
    
    # Create visualization
    title = f"ROI Component"
    if roi_index is not None:
        title = f"ROI Component {roi_index}"
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # 1. ROI image
    axes[0, 0].imshow(roi_image)
    axes[0, 0].set_title(f'{title}\n{shape} at {top_left}')
    axes[0, 0].axis('off')
    
    # 2. Colored only (white background)
    colored_display = roi_image.copy()
    colored_display[black_mask] = [255, 255, 255]
    axes[0, 1].imshow(colored_display)
    axes[0, 1].set_title(f'Colored: {colored_pixels:,}/{h*w:,} ({colored_pixels/(h*w)*100:.1f}%)')
    axes[0, 1].axis('off')
    
    # 3. Black mask
    axes[0, 2].imshow(black_mask, cmap='gray')
    axes[0, 2].set_title(f'Black: {np.sum(black_mask):,} pixels')
    axes[0, 2].axis('off')
    
    # 4. Color palette swatches
    palette_size = len(palette)
    colors_to_show = min(20, palette_size)
    
    # Create palette grid (4x5 max)
    grid_h = min(4, (colors_to_show + 4) // 5)
    grid_w = min(5, colors_to_show)
    
    palette_grid = np.zeros((grid_h * 20, grid_w * 20, 3), dtype=np.uint8)
    
    for i in range(colors_to_show):
        row = i // grid_w
        col = i % grid_w
        color = palette[i]
        palette_grid[row*20:(row+1)*20, col*20:(col+1)*20] = color
    
    axes[1, 0].imshow(palette_grid)
    axes[1, 0].set_title(f'Palette: {palette_size} colors')
    axes[1, 0].axis('off')
    
    # 5. Color frequency
    color_counts = np.bincount(indices.flatten())
    unique_colors = len([c for c in color_counts if c > 0])
    
    # Show top 10 most frequent colors
    top_n = min(10, len(color_counts))
    top_indices = np.argsort(color_counts)[-top_n:][::-1]
    
    y_pos = np.arange(top_n)
    frequencies = [color_counts[i] for i in top_indices]
    
    axes[1, 1].barh(y_pos, frequencies, color='lightcoral')
    axes[1, 1].set_yticks(y_pos)
    axes[1, 1].set_yticklabels([f'Idx {i}' for i in top_indices])
    axes[1, 1].set_xlabel('Pixel count')
    axes[1, 1].set_title(f'Top {top_n} color frequencies')
    axes[1, 1].invert_yaxis()
    
    # 6. Statistics text
    stats_text = f"""
    Position: {top_left}
    Size: {h}x{w} = {h*w:,} px
    Colored: {colored_pixels:,} ({colored_pixels/(h*w)*100:.1f}%)
    Black: {np.sum(black_mask):,} ({np.sum(black_mask)/(h*w)*100:.1f}%)
    
    Palette: {palette_size} colors
    Unique used: {unique_colors}
    Encoding: {roi_component.get('encoding', 'unknown')}
    """
    
    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=9, 
                   verticalalignment='center', transform=axes[1, 2].transAxes)
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return roi_image, top_left





























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
    ROI_components=[]
    for i, region in enumerate(roi_regions):

        region_components=[]

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
            

     
            
        
            print(f"segment_image_cropped shape: {segment_image_cropped.shape}")


            # ==============================================
            # 2. COMPRESS CROPPED SEGMENT
            # ==============================================
            seg_compression = get_all_unique_colors(
                segment_image_cropped,
                (top_left_abs_row, top_left_abs_col)
            )

            quality=50
            n_colors = seg_compression['actual_colors']

            distance= 256 - (256*quality / 100)
            eps=math.pow(100/quality,3)

            coefficient_max_samples=quality/100
            max_sample_pre=math.pow(n_colors, coefficient_max_samples )
            #max_colors_per_cluster=math.ceil( math.pow(math.e,max_sample_pre) * 3*100/quality  )
            max_colors_per_cluster=math.ceil(max_sample_pre * 3*100/quality  )

            # Then cluster the colors
            seg_compression = cluster_palette_colors(
                seg_compression,
                eps=eps,           # Distance threshold (0-255 scale)
                min_samples=1,      # Min colors to form cluster
                max_colors_per_cluster=max_colors_per_cluster  # Split large clusters
            )

            print(f"Eps: {eps}")
            print(f"n_colors: {n_colors}")
            print(f"max_sample_pre: {max_sample_pre}")
            print(f"max_colors_per_cluster: {max_colors_per_cluster}")

            
                        
            
            show_reconstruction_result=False
            if show_reconstruction_result:
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
                n_colors = seg_compression['compressed_colors']

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
            
                
            
            

            if seg_compression is None:
                continue
        
            

            all_segments_compressed.append(seg_compression)
            region_components.append(seg_compression)


        # ==============================================
        # 4. MERGE COMPONENTS WITHIN THIS ROI
        # ==============================================
        print(f"\n{'='*60}")
        print(f"MERGING COMPONENTS FOR ROI {i+1}")
        print(f"{'='*60}")

                
        if len(region_components) > 1:
            # Get ROI bbox
            minr, minc, maxr, maxc = region['bbox']
            roi_bbox = (minr, minc, maxr, maxc)
            roi_height = maxr - minr
            roi_width = maxc - minc
            
            # Choose merging strategy
            # Option 1: Simple merge (colored pixels override black)
            merged_components = merge_region_components_simple(region_components, roi_bbox)
            ROI_components.append(merged_components)

            #visualize_merged_result(merged_components, (roi_height, roi_width), minr, minc)
            
            # Option 2: Merge with segment sorting
            # merged_components = merge_region_components_better(region_components, roi_bbox)
            
            # Option 3: Merge with explicit overlap handling
            # merged_components = merge_region_components_overlap(region_components, roi_bbox)
            
            # Add to final list
            all_segments_compressed.extend(merged_components)
            
            # Calculate statistics
            original_pixels = sum(seg['shape'][0] * seg['shape'][1] for seg in region_components)
            original_black = sum(seg['indices'].count(1) for seg in region_components)
            
            merged_pixels = sum(seg['shape'][0] * seg['shape'][1] for seg in merged_components)
            merged_black = sum(seg['indices'].count(1) for seg in merged_components)
            
            print(f"\nSummary:")
            print(f"  Original: {len(region_components)} segments, {original_pixels:,} pixels")
            print(f"  Merged: {len(merged_components)} segments, {merged_pixels:,} pixels")
            print(f"  Black reduction: {original_black - merged_black:,} pixels")
            print(f"  Pixel reduction: {original_pixels - merged_pixels:,} pixels")
            
        else:
            # Just add the single component
            ROI_components.append(region_components)
            all_segments_compressed.extend(region_components)
            print(f"Only 1 component in ROI {i+1}, no merging needed")



        """
        Fine subregion delle ROI
        """






































    """Inizio subregion delle nonROI"""
    # ==============================================
    # MAIN PROCESSING LOOP - One nonROI Region at a time
    # ==============================================
    nonROI_components=[]
    for i, region in enumerate(nonroi_regions):

        region_components=[]

        print(f"\n{'='*60}")
        print(f"PROCESSING nonROI REGION {i+1}/{len(nonroi_regions)}")
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
        optimal_segments = math.ceil(normalized_overall_score)/3
        
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
            

    
            
        
            print(f"segment_image_cropped shape: {segment_image_cropped.shape}")


            # ==============================================
            # 2. COMPRESS CROPPED SEGMENT
            # ==============================================
            seg_compression = get_all_unique_colors(
                segment_image_cropped,
                (top_left_abs_row, top_left_abs_col)
            )

            quality=10
            n_colors = seg_compression['actual_colors']

            distance= 256 - (256*quality / 100)
            eps=math.pow(100/quality,3)

            coefficient_max_samples=quality/100
            max_sample_pre=math.pow(n_colors, coefficient_max_samples )
            #max_colors_per_cluster=math.ceil( math.pow(math.e,max_sample_pre) * 3*100/quality  )
            max_colors_per_cluster=math.ceil(max_sample_pre * 3*100/quality  )

            # Then cluster the colors
            seg_compression = cluster_palette_colors(
                seg_compression,
                eps=eps,           # Distance threshold (0-255 scale)
                min_samples=1,      # Min colors to form cluster
                max_colors_per_cluster=max_colors_per_cluster  # Split large clusters
            )

            print(f"Eps: {eps}")
            print(f"n_colors: {n_colors}")
            print(f"max_sample_pre: {max_sample_pre}")
            print(f"max_colors_per_cluster: {max_colors_per_cluster}")

            
                        
            
            show_reconstruction_result=False
            if show_reconstruction_result:
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
                n_colors = seg_compression['compressed_colors']

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
            
                
            
            

            if seg_compression is None:
                continue
        
            

            all_segments_compressed.append(seg_compression)
            region_components.append(seg_compression)


        # ==============================================
        # 4. MERGE COMPONENTS WITHIN THIS ROI
        # ==============================================
        print(f"\n{'='*60}")
        print(f"MERGING COMPONENTS FOR ROI {i+1}")
        print(f"{'='*60}")

                
        if len(region_components) > 1:
            # Get ROI bbox
            minr, minc, maxr, maxc = region['bbox']
            roi_bbox = (minr, minc, maxr, maxc)
            roi_height = maxr - minr
            roi_width = maxc - minc
            
            # Choose merging strategy
            # Option 1: Simple merge (colored pixels override black)
            merged_components = merge_region_components_simple(region_components, roi_bbox)
            nonROI_components.append(merged_components)

            #visualize_merged_result(merged_components, (roi_height, roi_width), minr, minc)
            
            # Option 2: Merge with segment sorting
            # merged_components = merge_region_components_better(region_components, roi_bbox)
            
            # Option 3: Merge with explicit overlap handling
            # merged_components = merge_region_components_overlap(region_components, roi_bbox)
            
            # Add to final list
            all_segments_compressed.extend(merged_components)
            
            # Calculate statistics
            original_pixels = sum(seg['shape'][0] * seg['shape'][1] for seg in region_components)
            original_black = sum(seg['indices'].count(1) for seg in region_components)
            
            merged_pixels = sum(seg['shape'][0] * seg['shape'][1] for seg in merged_components)
            merged_black = sum(seg['indices'].count(1) for seg in merged_components)
            
            print(f"\nSummary:")
            print(f"  Original: {len(region_components)} segments, {original_pixels:,} pixels")
            print(f"  Merged: {len(merged_components)} segments, {merged_pixels:,} pixels")
            print(f"  Black reduction: {original_black - merged_black:,} pixels")
            print(f"  Pixel reduction: {original_pixels - merged_pixels:,} pixels")
            
        else:
            # Just add the single component
            nonROI_components.append(region_components)
            all_segments_compressed.extend(region_components)
            print(f"Only 1 component in ROI {i+1}, no merging needed")














    original_image_height, original_image_width, _ = image_rgb.shape



    """# Extract the first (and only) element from each list
    all_roi_components_flat = [roi[0] for roi in ROI_components]

    # 1. Visualize all ROIs together in the image context
    full_reconstruction, roi_mask = visualize_all_roi_components(
        all_roi_components_flat, 
        image_shape=(original_image_height, original_image_width)
    )

    # 2. Visualize individual ROIs in detail
    for i, roi in enumerate(all_roi_components_flat):
        print(f"\n{'='*60}")
        print(f"VISUALIZING ROI {i}")
        print(f"{'='*60}")
        
        roi_image, roi_topleft = visualize_individual_roi(roi, roi_index=i)





    


    # Extract the first (and only) element from each list
    all_nonroi_components_flat = [roi[0] for roi in nonROI_components]

    # 1. Visualize all ROIs together in the image context
    full_reconstruction, roi_mask = visualize_all_roi_components(
        all_nonroi_components_flat, 
        image_shape=(original_image_height, original_image_width)
    )

    # 2. Visualize individual ROIs in detail
    for i, nonroi in enumerate(all_nonroi_components_flat):
        print(f"\n{'='*60}")
        print(f"VISUALIZING ROI {i}")
        print(f"{'='*60}")
        
        nonroi_image = visualize_individual_roi(nonroi, roi_index=i)
    """



















































    image_components=[]

    """
    Second phase of hierarchical clustering:
    Clustering colors between each ROI / nonROI
    """

    #ROI

    # First, collect all ROI components
    all_roi_components_flat = [roi[0] for roi in ROI_components]

    # Use the entire image as the bbox
    image_bbox = (0, 0, original_image_height, original_image_width)

    roi_image = merge_region_components_simple(
        all_roi_components_flat,
        roi_bbox=image_bbox
    )

    # Extract the merged segment dictionary
    merged_segment = roi_image[0]  # This contains palette and indices, NOT the image!

    quality=85
    n_colors = merged_segment['actual_colors']

    distance= 256 - (256*quality / 100)
    eps=math.pow(100/quality,3)

    coefficient_max_samples=quality/100
    max_sample_pre=math.pow(n_colors, coefficient_max_samples )
    #max_colors_per_cluster=math.ceil( math.pow(math.e,max_sample_pre) * 3*100/quality  )
    max_colors_per_cluster=math.ceil(max_sample_pre * 3*100/quality  )

    # ==============================================
    # OPTION 1: If you want to cluster the merged palette
    # ==============================================
    # You already have the merged palette in merged_segment
    # Just cluster it directly:
    ROI_seg_compression = cluster_palette_colors(
        merged_segment,  # Pass the dictionary
        eps=eps,
        min_samples=1,
        max_colors_per_cluster=max_colors_per_cluster
    )

    image_components.append(ROI_seg_compression)

    


    show_reconstruction_result=True
    if show_reconstruction_result:
        # ==============================================
        # 3. RECONSTRUCT 
        # ==============================================
        reconstruction_result = decompress_color_quantization(ROI_seg_compression)
        
        # ==============================================
        # 4. SIMPLE VISUALIZATION
        # ==============================================
        print(f"\n{'='*60}")
        print(f"ROI RECONSTRUCTION")
        print(f"{'='*60}")
        
        # Get basic info
        h, w = reconstruction_result['shape']
        top_left = reconstruction_result['top_left']
        n_colors = ROI_seg_compression['compressed_colors']
        psnr = ROI_seg_compression.get('psnr', 0)
        
        # Get reconstructed image
        reconstructed_img = reconstruction_result['image']
        
        # Simple display - just the reconstruction
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # 1. Reconstructed ROI
        axes[0].imshow(reconstructed_img)
        axes[0].set_title(f'Reconstructed ROI\n{h}x{w} pixels\n{n_colors} colors\nPSNR: {psnr:.1f} dB')
        axes[0].axis('off')
        
        # 2. Colored pixels only (white background)
        colored_only = reconstructed_img.copy()
        black_mask = np.all(reconstructed_img == [0, 0, 0], axis=2)
        colored_only[black_mask] = [255, 255, 255]  # White background for black areas
        
        axes[1].imshow(colored_only)
        axes[1].set_title(f'Colored Pixels Only\nBlack pixels: {np.sum(black_mask):,}\n({np.sum(black_mask)/(h*w)*100:.1f}%)')
        axes[1].axis('off')
        
        plt.suptitle(f'ROI at position {top_left}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Console summary
        print(f"Position: {top_left}")
        print(f"Size: {h}x{w} = {h*w:,} pixels")
        print(f"Colors: {n_colors}")
        print(f"Black pixels: {np.sum(black_mask):,} ({np.sum(black_mask)/(h*w)*100:.1f}%)")
        print(f"Colored pixels: {h*w - np.sum(black_mask):,} ({(h*w - np.sum(black_mask))/(h*w)*100:.1f}%)")
        print(f"PSNR: {psnr:.1f} dB")







    #nonROI

    # First, collect all ROI components
    all_nonroi_components_flat = [roi[0] for roi in nonROI_components]

    # Use the entire image as the bbox
    image_bbox = (0, 0, original_image_height, original_image_width)

    nonroi_image = merge_region_components_simple(
        all_nonroi_components_flat,
        roi_bbox=image_bbox
    )

    # Extract the merged segment dictionary
    merged_segment = nonroi_image[0]  # This contains palette and indices, NOT the image!

    quality=75
    n_colors = merged_segment['actual_colors']

    distance= 256 - (256*quality / 100)
    eps=math.pow(100/quality,3)

    coefficient_max_samples=quality/100
    max_sample_pre=math.pow(n_colors, coefficient_max_samples )
    #max_colors_per_cluster=math.ceil( math.pow(math.e,max_sample_pre) * 3*100/quality  )
    max_colors_per_cluster=math.ceil(max_sample_pre * 3*100/quality  )

    # ==============================================
    # OPTION 1: If you want to cluster the merged palette
    # ==============================================
    # You already have the merged palette in merged_segment
    # Just cluster it directly:
    nonROI_seg_compression = cluster_palette_colors(
        merged_segment,  # Pass the dictionary
        eps=eps,
        min_samples=1,
        max_colors_per_cluster=max_colors_per_cluster
    )

    image_components.append(nonROI_seg_compression)

    


    show_reconstruction_result=True
    if show_reconstruction_result:
        # ==============================================
        # 3. RECONSTRUCT 
        # ==============================================
        reconstruction_result = decompress_color_quantization(nonROI_seg_compression)
        
        # ==============================================
        # 4. SIMPLE VISUALIZATION
        # ==============================================
        print(f"\n{'='*60}")
        print(f"ROI RECONSTRUCTION")
        print(f"{'='*60}")
        
        # Get basic info
        h, w = reconstruction_result['shape']
        top_left = reconstruction_result['top_left']
        n_colors = nonROI_seg_compression['compressed_colors']
        psnr = nonROI_seg_compression.get('psnr', 0)
        
        # Get reconstructed image
        reconstructed_img = reconstruction_result['image']
        
        # Simple display - just the reconstruction
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # 1. Reconstructed ROI
        axes[0].imshow(reconstructed_img)
        axes[0].set_title(f'Reconstructed ROI\n{h}x{w} pixels\n{n_colors} colors\nPSNR: {psnr:.1f} dB')
        axes[0].axis('off')
        
        # 2. Colored pixels only (white background)
        colored_only = reconstructed_img.copy()
        black_mask = np.all(reconstructed_img == [0, 0, 0], axis=2)
        colored_only[black_mask] = [255, 255, 255]  # White background for black areas
        
        axes[1].imshow(colored_only)
        axes[1].set_title(f'Colored Pixels Only\nBlack pixels: {np.sum(black_mask):,}\n({np.sum(black_mask)/(h*w)*100:.1f}%)')
        axes[1].axis('off')
        
        plt.suptitle(f'ROI at position {top_left}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Console summary
        print(f"Position: {top_left}")
        print(f"Size: {h}x{w} = {h*w:,} pixels")
        print(f"Colors: {n_colors}")
        print(f"Black pixels: {np.sum(black_mask):,} ({np.sum(black_mask)/(h*w)*100:.1f}%)")
        print(f"Colored pixels: {h*w - np.sum(black_mask):,} ({(h*w - np.sum(black_mask))/(h*w)*100:.1f}%)")
        print(f"PSNR: {psnr:.1f} dB")





















    """
    Final peak of clustering: whole image
    """


    # First, collect all ROI components
    all_image_components_flat = [roi for roi in image_components]

    # Use the entire image as the bbox
    image_bbox = (0, 0, original_image_height, original_image_width)

    regions_image = merge_region_components_simple(
        all_image_components_flat,
        roi_bbox=image_bbox
    )

    # Extract the merged segment dictionary
    merged_segment = regions_image[0]  # This contains palette and indices, NOT the image!

    quality=95
    n_colors = merged_segment['actual_colors']

    distance= 256 - (256*quality / 100)
    eps=math.pow(100/quality,3)

    coefficient_max_samples=quality/100
    max_sample_pre=math.pow(n_colors, coefficient_max_samples )
    #max_colors_per_cluster=math.ceil( math.pow(math.e,max_sample_pre) * 3*100/quality  )
    max_colors_per_cluster=math.ceil(max_sample_pre * 3*100/quality  )

    # ==============================================
    # OPTION 1: If you want to cluster the merged palette
    # ==============================================
    # You already have the merged palette in merged_segment
    # Just cluster it directly:
    image_seg_compression = cluster_palette_colors(
        merged_segment,  # Pass the dictionary
        eps=eps,
        min_samples=1,
        max_colors_per_cluster=max_colors_per_cluster
    )

    image_components.append(nonROI_seg_compression)

    


    show_reconstruction_result=True
    if show_reconstruction_result:
        # ==============================================
        # 3. RECONSTRUCT 
        # ==============================================
        reconstruction_result = decompress_color_quantization(image_seg_compression)
        
        # ==============================================
        # 4. SIMPLE VISUALIZATION
        # ==============================================
        print(f"\n{'='*60}")
        print(f"ROI RECONSTRUCTION")
        print(f"{'='*60}")
        
        # Get basic info
        h, w = reconstruction_result['shape']
        top_left = reconstruction_result['top_left']
        n_colors = image_seg_compression['compressed_colors']
        psnr = image_seg_compression.get('psnr', 0)
        
        # Get reconstructed image
        reconstructed_img = reconstruction_result['image']
        
        # Simple display - just the reconstruction
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # 1. Reconstructed ROI
        axes[0].imshow(reconstructed_img)
        axes[0].set_title(f'Reconstructed ROI\n{h}x{w} pixels\n{n_colors} colors\nPSNR: {psnr:.1f} dB')
        axes[0].axis('off')
        
        # 2. Colored pixels only (white background)
        colored_only = reconstructed_img.copy()
        black_mask = np.all(reconstructed_img == [0, 0, 0], axis=2)
        colored_only[black_mask] = [255, 255, 255]  # White background for black areas
        
        axes[1].imshow(colored_only)
        axes[1].set_title(f'Colored Pixels Only\nBlack pixels: {np.sum(black_mask):,}\n({np.sum(black_mask)/(h*w)*100:.1f}%)')
        axes[1].axis('off')
        
        plt.suptitle(f'ROI at position {top_left}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Console summary
        print(f"Position: {top_left}")
        print(f"Size: {h}x{w} = {h*w:,} pixels")
        print(f"Colors: {n_colors}")
        print(f"Black pixels: {np.sum(black_mask):,} ({np.sum(black_mask)/(h*w)*100:.1f}%)")
        print(f"Colored pixels: {h*w - np.sum(black_mask):,} ({(h*w - np.sum(black_mask))/(h*w)*100:.1f}%)")
        print(f"PSNR: {psnr:.1f} dB")




    save=True
    if save:
        from PIL import Image
        import numpy as np

        # Assuming reconstruction_result is your decompressed result
        reconstruction_result = decompress_color_quantization(image_seg_compression)

        # Extract the image
        reconstructed_image = reconstruction_result['image']  # This is a numpy array

        # Convert numpy array to PIL Image and save
        pil_image = Image.fromarray(reconstructed_image)
        pil_image.save('reconstructed_image.jpg', quality=95)  # quality 1-100

        print(f"✅ Image saved as 'reconstructed_image.jpg'")
        print(f"   Size: {reconstructed_image.shape[1]}x{reconstructed_image.shape[0]}")


