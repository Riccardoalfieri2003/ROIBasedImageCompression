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

def compress_subregion_with_shared_qtable(region_image, segment_mask, shared_qtable, quality_factor=75):
    """
    Compress subregion using shared quantization table with better edge handling.
    """
    rows, cols = np.where(segment_mask)
    if len(rows) == 0:
        return None
    
    num_pixels = len(rows)
    original_size = num_pixels * 3
    
    # Get region bounds with some padding for edge blocks
    min_r, max_r = rows.min(), rows.max()
    min_c, max_c = cols.min(), cols.max()
    
    # Add padding to ensure we cover edge pixels
    pad_r = 8 - ((max_r - min_r + 1) % 8) if (max_r - min_r + 1) % 8 != 0 else 0
    pad_c = 8 - ((max_c - min_c + 1) % 8) if (max_c - min_c + 1) % 8 != 0 else 0
    
    # Extended bounds to include padding
    ext_min_r = max(0, min_r - 4)  # Pad a bit for safety
    ext_max_r = min(region_image.shape[0] - 1, max_r + pad_r + 4)
    ext_min_c = max(0, min_c - 4)
    ext_max_c = min(region_image.shape[1] - 1, max_c + pad_c + 4)
    
    # Extract extended region
    region_ext = region_image[ext_min_r:ext_max_r+1, ext_min_c:ext_max_c+1]
    mask_ext = segment_mask[ext_min_r:ext_max_r+1, ext_min_c:ext_max_c+1]
    
    # Prepare compressed data structure
    compressed_data = {
        'method': 'dct_shared_qtable',
        'original_size': original_size,
        'true_bbox': (min_r, min_c, max_r, max_c),  # Actual region bounds
        'ext_bbox': (ext_min_r, ext_min_c, ext_max_r, ext_max_c),  # Extended bounds
        'shape': region_ext.shape,
        'compressed_blocks': [],
        'use_shared_qtable': True
    }
    
    # Adjust quantization table based on quality factor
    scale_factor = 1.0
    if quality_factor < 50:
        scale_factor = 50.0 / quality_factor
    else:
        scale_factor = (100 - quality_factor) / 50.0
    
    qtable_scaled = np.clip(np.round(shared_qtable * scale_factor), 1, 255).astype(np.uint8)
    
    # Process ALL 8x8 blocks in extended region (not just masked ones)
    h, w = region_ext.shape[:2]
    total_blocks = 0
    compressed_size = 0
    
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            # Get block bounds (handle partial blocks at edges)
            i_end = min(i + 8, h)
            j_end = min(j + 8, w)
            block_height = i_end - i
            block_width = j_end - j
            
            # Skip blocks with no masked pixels
            block_mask = mask_ext[i:i_end, j:j_end]
            if np.sum(block_mask) == 0:
                continue
            
            # Get block (pad if necessary for DCT)
            block = region_ext[i:i_end, j:j_end]
            
            # If block is smaller than 8x8, pad it
            if block_height < 8 or block_width < 8:
                block_padded = np.zeros((8, 8, 3), dtype=np.uint8)
                block_padded[:block_height, :block_width] = block
                block = block_padded
            
            # Convert to YCbCr
            block_ycbcr = cv2.cvtColor(block, cv2.COLOR_RGB2YCrCb)
            
            compressed_block = {}
            
            # Process each channel
            for ch in range(3):
                # Extract channel and center around 0
                channel_data = block_ycbcr[:, :, ch].astype(np.float32) - 128
                
                # Apply DCT
                dct_coeffs = cv2.dct(channel_data)
                
                # Quantize using shared table
                quantized = np.round(dct_coeffs / qtable_scaled).astype(np.int16)
                
                # Run-length encode (store non-zero coefficients)
                non_zero_mask = quantized != 0
                non_zero_coeffs = quantized[non_zero_mask]
                
                if len(non_zero_coeffs) > 0:
                    # Store position and value of non-zero coefficients
                    positions = np.argwhere(non_zero_mask)
                    values = non_zero_coeffs
                    
                    compressed_block[f'channel_{ch}'] = {
                        'positions': positions,
                        'values': values,
                        'num_coeffs': len(values),
                        'block_shape': (block_height, block_width)  # Store original block size
                    }
                    
                    # Estimate storage
                    compressed_size += len(values) * 4  # 2 bytes for position, 2 for value
                else:
                    compressed_block[f'channel_{ch}'] = {
                        'positions': np.array([]),
                        'values': np.array([]),
                        'num_coeffs': 0,
                        'block_shape': (block_height, block_width)
                    }
            
            compressed_data['compressed_blocks'].append({
                'position': (i, j),
                'data': compressed_block,
                'block_size': (block_height, block_width)  # Store for reconstruction
            })
            total_blocks += 1
    
    # Add overhead
    compressed_size += 50  # Increased for extra metadata
    
    compressed_data['compressed_size'] = compressed_size
    compressed_data['compression_ratio'] = original_size / compressed_size if compressed_size > 0 else 0
    compressed_data['num_blocks'] = total_blocks
    
    return compressed_data

def reconstruct_subregion_with_shared_qtable(compressed_data, shared_qtable, full_image_shape, quality_factor=75):
    """
    Reconstruct subregion with proper edge handling.
    """
    # Get extended bounding box
    ext_min_r, ext_min_c, ext_max_r, ext_max_c = compressed_data['ext_bbox']
    ext_height = ext_max_r - ext_min_r + 1
    ext_width = ext_max_c - ext_min_c + 1
    
    # Create reconstruction of extended region
    reconstructed_ext = np.zeros((ext_height, ext_width, 3), dtype=np.uint8)
    
    # Adjust quantization table
    scale_factor = 1.0
    if quality_factor < 50:
        scale_factor = 50.0 / quality_factor
    else:
        scale_factor = (100 - quality_factor) / 50.0
    
    qtable_scaled = np.clip(np.round(shared_qtable * scale_factor), 1, 255).astype(np.float32)
    
    # Reconstruct each block
    for block_data in compressed_data['compressed_blocks']:
        i, j = block_data['position']
        block_compressed = block_data['data']
        block_height, block_width = block_data['block_size']
        
        # Reconstruct full 8x8 block
        block_reconstructed = np.zeros((8, 8, 3), dtype=np.float32)
        
        for ch in range(3):
            channel_data = block_compressed.get(f'channel_{ch}', {})
            positions = channel_data.get('positions', np.array([]))
            values = channel_data.get('values', np.array([]))
            
            if len(positions) > 0:
                # Create quantized coefficient matrix
                quantized = np.zeros((8, 8), dtype=np.float32)
                for idx, (pos, val) in enumerate(zip(positions, values)):
                    quantized[pos[0], pos[1]] = val
                
                # Dequantize
                dct_coeffs = quantized * qtable_scaled
                
                # Inverse DCT
                channel_recon = cv2.idct(dct_coeffs) + 128
                block_reconstructed[:, :, ch] = channel_recon
            else:
                # If no coefficients, fill with 128 (neutral value)
                block_reconstructed[:, :, ch] = 128
        
        # Convert from YCbCr to RGB
        block_reconstructed = np.clip(block_reconstructed, 0, 255).astype(np.uint8)
        block_rgb = cv2.cvtColor(block_reconstructed, cv2.COLOR_YCrCb2RGB)
        
        # Place only the valid part (original block size)
        i_end = min(i + block_height, ext_height)
        j_end = min(j + block_width, ext_width)
        
        if i_end > i and j_end > j:
            reconstructed_ext[i:i_end, j:j_end] = block_rgb[:i_end-i, :j_end-j]
    
    # Create full-sized reconstruction
    full_reconstructed = np.zeros(full_image_shape, dtype=np.uint8)
    
    # Place extended reconstruction back into full image
    full_h, full_w = full_image_shape[:2]
    
    # Calculate valid bounds within full image
    r_start = ext_min_r
    r_end = min(ext_max_r + 1, full_h)
    c_start = ext_min_c
    c_end = min(ext_max_c + 1, full_w)
    
    # Calculate corresponding bounds in reconstructed_ext
    ext_r_start = 0
    ext_r_end = min(ext_height, r_end - ext_min_r)
    ext_c_start = 0
    ext_c_end = min(ext_width, c_end - ext_min_c)
    
    # Copy only the overlapping region
    if r_start < r_end and c_start < c_end:
        full_reconstructed[r_start:r_end, c_start:c_end] = \
            reconstructed_ext[ext_r_start:ext_r_end, ext_c_start:ext_c_end]
    
    return full_reconstructed

def fill_border_gaps(reconstructed_image, segment_mask, region_image):
    """
    Fill black borders by interpolating from neighboring pixels.
    """
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


        from encoder.interpolation.spline import compress_shape_divided_exact, get_minimal_storage_with_rounding
        # ==============================================
        # 1. INTERPOLATION OF BORDERS
        # ==============================================
        result = compress_shape_divided_exact(bbox_mask, num_sublists=3, compression_ratio=0.3)
        compressed_data, storage_info = get_minimal_storage_with_rounding(
            result, decimal_places=3
        )
        
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
        


        """# ==============================================
        # 4A. DIRECT ROI COMPRESSION (Whole ROI)
        # ==============================================
        print(f"\n{'='*60}")
        print(f"METHOD 1: DIRECT ROI COMPRESSION (Whole ROI)")
        print(f"{'='*60}")
        
        # Compress ROI content only
        compression_result = compress_roi_content_only(
            bbox_mask,  # ROI mask
            region_image,
            quality=75
        )
        
        # Calculate and display statistics
        content_stats = calculate_content_compression_stats(compression_result)
        
        # Get reconstructed image
        content_reconstructed = reconstruct_roi_content(
            compression_result['compressed_coefficients'],
            region_image.shape,
            bbox_mask
        )
        
        # CORRECT PSNR calculation (ROI area only)
        content_psnr = calculate_psnr_for_region_correct_1(region_image, content_reconstructed, bbox_mask)
        content_stats['psnr'] = content_psnr
        
        # Store ROI compression results
        roi_compression_results = {
            'method': 'Direct ROI',
            'original_bytes': content_stats['original_bytes'],
            'compressed_bytes': content_stats['compressed_bytes'],
            'ratio': content_stats['ratio'],
            'psnr': content_psnr,
            'reconstructed': content_reconstructed
        }"""
        
        
        












        
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
            if segment_pixels < 64:  # Need at least one 8x8 block
                continue
            
            #print(f"\n    Segment {seg_idx+1}/{len(segment_boundaries)} (ID: {segment_id}):")
            #print(f"      Pixels: {segment_pixels:,}")
            
            # Compress with shared quantization table
            seg_compression = compress_subregion_with_shared_qtable(
                region_image, 
                segment_mask, 
                shared_qtable,
                quality_factor=50
            )
            
            if seg_compression is None:
                continue

            # Calculate ACTUAL minimal compressed size
            actual_compressed_size = calculate_minimal_compressed_size(seg_compression)
            seg_compression['actual_compressed_size'] = actual_compressed_size
            seg_compression['actual_ratio'] = seg_compression['original_size'] / actual_compressed_size
            
            #print(f"      Estimated in function: {seg_compression['compressed_size']:,} bytes")
            #print(f"      Actual minimal: {actual_compressed_size:,} bytes")
            #print(f"      Savings: {seg_compression['compressed_size'] - actual_compressed_size:,} bytes")
            
            # Reconstruct
            seg_reconstructed = reconstruct_subregion_with_shared_qtable(
                seg_compression,
                shared_qtable,
                region_image.shape,  # Pass the full shape
                quality_factor=50
            )

            # Fill border gaps
            seg_reconstructed = fill_border_gaps(seg_reconstructed, segment_mask, region_image)
                            
            # Calculate PSNR
            rows, cols = np.where(segment_mask)
            if len(rows) > 0:
                # Extract only the overlapping area
                seg_recon_masked = np.zeros_like(region_image)
                seg_recon_masked[rows, cols] = seg_reconstructed[rows, cols]
                
                # Simple PSNR calculation
                original_vals = region_image[rows, cols].flatten().astype(float)
                recon_vals = seg_recon_masked[rows, cols].flatten().astype(float)
                
                mse = np.mean((original_vals - recon_vals) ** 2)
                seg_psnr = 10 * np.log10(255*255/mse) if mse > 0 else float('inf')
            else:
                seg_psnr = 0
            
            # Update reconstructed image
            subregion_reconstructed[segment_mask] = seg_reconstructed[segment_mask]
            
            # Store results
            subregion_results.append({
                'segment_id': segment_id,
                'pixels': segment_pixels,
                'original_bytes': seg_compression['original_size'],
                'compressed_bytes': actual_compressed_size,  # Use ACTUAL size
                'estimated_bytes': seg_compression['compressed_size'],  # Keep for comparison
                'ratio': seg_compression['actual_ratio'],  # Use ACTUAL ratio
                'psnr': seg_psnr,
                'num_blocks': seg_compression.get('num_blocks', 0)
            })
            
            total_subregion_original += seg_compression['original_size']
            total_subregion_compressed += actual_compressed_size  # Use ACTUAL size
            subregion_psnrs.append(seg_psnr)
            
            # Progress
            if (seg_idx + 1) % 5 == 0 or (seg_idx + 1) == len(segment_boundaries):
                print(f"    Processed {seg_idx + 1}/{len(segment_boundaries)} segments")

        # Calculate overall stats (include shared table in compressed size)
        if total_subregion_compressed > 0:
            subregion_ratio = total_subregion_original / total_subregion_compressed
            avg_psnr = np.mean(subregion_psnrs) if subregion_psnrs else 0
            
            # Overall PSNR
            overall_psnr = calculate_psnr_for_region_correct(
                region_image, subregion_reconstructed, bbox_mask
            )
            
            print(f"\n  SHARED QTABLE COMPRESSION SUMMARY:")
            print(f"    Shared table size: {len(shared_qtable.tobytes()):,} bytes")
            print(f"    Total original: {total_subregion_original:,} bytes")
            print(f"    Total compressed (including table): {total_subregion_compressed:,} bytes")
            print(f"    Overall ratio: {subregion_ratio:.2f}:1")
            print(f"    Average segment PSNR: {avg_psnr:.2f} dB")
            print(f"    Overall PSNR: {overall_psnr:.2f} dB")
            
            # Calculate savings
            if len(subregion_results) > 1:
                # Without optimization: each segment would need its own table (64 bytes)
                naive_size = total_subregion_compressed + (len(subregion_results) - 1) * 64
                savings = naive_size - total_subregion_compressed
                print(f"    Storage savings from shared table: {savings:,} bytes")
            
            subregion_compression_results = {
                'method': 'Shared QTable',
                'original_bytes': total_subregion_original,
                'compressed_bytes': total_subregion_compressed,
                'ratio': subregion_ratio,
                'avg_segment_psnr': avg_psnr,
                'overall_psnr': overall_psnr,
                'reconstructed': subregion_reconstructed,
                'segment_results': subregion_results,
                'shared_qtable': shared_qtable
            }
        else:
            print(f"  ❌ No subregions were successfully compressed")
            subregion_compression_results = None











        # Calculate overall stats
        if total_subregion_compressed > 0 and len(subregion_psnrs) > 0:
            subregion_ratio = total_subregion_original / total_subregion_compressed
            avg_subregion_psnr = np.mean(subregion_psnrs)
            
            # Calculate overall PSNR
            subregion_overall_psnr = calculate_psnr_for_region_correct(
                region_image, 
                subregion_reconstructed, 
                bbox_mask
            )
            
            print(f"\n  FIXED SUBREGION COMPRESSION SUMMARY:")
            print(f"    Total original: {total_subregion_original:,} bytes")
            print(f"    Total compressed: {total_subregion_compressed:,} bytes")
            print(f"    Overall ratio: {subregion_ratio:.2f}:1")
            print(f"    Average segment PSNR: {avg_subregion_psnr:.2f} dB")
            print(f"    Overall reconstruction PSNR: {subregion_overall_psnr:.2f} dB")
            
            # Show PSNR improvement analysis
            if avg_subregion_psnr < 20:
                print(f"\n  ⚠️  PSNR ANALYSIS:")
                print(f"    Current PSNR ({avg_subregion_psnr:.1f} dB) is too low.")
                print(f"    Target should be >20 dB for acceptable quality.")
                print(f"    Issues detected: Color mismatch in reconstruction.")
            
            subregion_compression_results = {
                'method': 'Fixed Subregion',
                'original_bytes': total_subregion_original,
                'compressed_bytes': total_subregion_compressed,
                'ratio': subregion_ratio,
                'avg_segment_psnr': avg_subregion_psnr,
                'overall_psnr': subregion_overall_psnr,
                'reconstructed': subregion_reconstructed,
                'segment_results': subregion_results,
                #'compression_methods': compression_methods
            }
        else:
            print(f"  ❌ No successful compressions")
            subregion_compression_results = None



        # ==============================================
        # SIMPLIFIED DEBUG VISUALIZATION (No figure modification)
        # ==============================================
        if subregion_results and len(subregion_results) > 0:
            print(f"\n  DEBUG: MEAN COLOR ANALYSIS")
            
            # Create a simple debug visualization
            debug_image = np.zeros_like(region_image)
            
            for seg in subregion_results[:min(5, len(subregion_results))]:  # Show first 5 segments or less
                segment_id = seg['segment_id']
                segment_mask = (roi_segments == segment_id) & bbox_mask
                
                if seg.get('mean_color'):
                    mean_color = np.array(seg['mean_color'], dtype=np.uint8)
                    rows, cols = np.where(segment_mask)
                    
                    if len(rows) > 0:
                        # Show what we're reconstructing
                        debug_image[rows, cols] = mean_color
                        
                        # Get actual average from original
                        actual_pixels = region_image[rows, cols]
                        actual_mean = np.mean(actual_pixels, axis=0).astype(np.uint8)
                        
                        print(f"    Segment {segment_id}:")
                        print(f"      Computed mean: {mean_color}")
                        print(f"      Actual mean: {actual_mean}")
                        print(f"      Difference: {np.abs(mean_color - actual_mean)}")
                        print(f"      Match within tolerance: {np.allclose(mean_color, actual_mean, atol=10)}")
            
            # Create a NEW figure for debug visualization (don't modify existing one)
            if np.any(debug_image > 0):  # Only create if we have data
                try:
                    fig_debug, axes_debug = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # Original segment
                    axes_debug[0].imshow(region_image)
                    axes_debug[0].set_title('Original Region')
                    axes_debug[0].axis('off')
                    
                    # Mean color reconstruction
                    axes_debug[1].imshow(debug_image)
                    axes_debug[1].set_title('Mean Color Reconstruction (Debug)')
                    axes_debug[1].axis('off')
                    
                    # Difference
                    diff = np.abs(region_image.astype(float) - debug_image.astype(float))
                    diff_display = np.mean(diff, axis=2) if diff.shape[2] == 3 else diff
                    im = axes_debug[2].imshow(diff_display, cmap='hot', vmin=0, vmax=100)
                    axes_debug[2].set_title('Color Difference (Mean vs Original)')
                    axes_debug[2].axis('off')
                    plt.colorbar(im, ax=axes_debug[2], fraction=0.046, pad=0.04)
                    
                    plt.suptitle('Debug: Mean Color Accuracy Check', fontsize=14)
                    plt.tight_layout()
                    plt.show()
                except Exception as e:
                    print(f"  ⚠️  Debug visualization failed (non-critical): {e}")
            
            # Also show PSNR distribution
            if len(subregion_psnrs) > 1:
                print(f"\n  PSNR DISTRIBUTION ACROSS SEGMENTS:")
                print(f"    Min: {min(subregion_psnrs):.1f} dB")
                print(f"    Max: {max(subregion_psnrs):.1f} dB")
                print(f"    Median: {np.median(subregion_psnrs):.1f} dB")
                print(f"    Std Dev: {np.std(subregion_psnrs):.1f} dB")
                
                # Check for very low PSNR segments
                low_psnr_segments = [(i, psnr) for i, psnr in enumerate(subregion_psnrs) if psnr < 15]
                if low_psnr_segments:
                    print(f"  ⚠️  LOW PSNR SEGMENTS (<15 dB):")
                    for idx, psnr in low_psnr_segments[:3]:  # Show first 3
                        seg_id = subregion_results[idx]['segment_id']
                        print(f"    Segment {seg_id}: {psnr:.1f} dB")

        # ==============================================
        # ADDITIONAL VISUALIZATION FOR HOMOGENEOUS REGIONS
        # ==============================================

        # In the visualization section, add method information
        if subregion_compression_results and subregion_results:
            # Create method visualization
            method_visualization = np.zeros_like(region_image)
            method_colors = {
                'mean_color': [0, 255, 0],  # Green for mean color
                'dct': [255, 0, 0],         # Red for DCT
                'texture_aware': [0, 0, 255]  # Blue for texture aware
            }
            
            for seg in subregion_results:
                segment_id = seg['segment_id']
                segment_mask = (roi_segments == segment_id) & bbox_mask
                method = seg.get('method', 'dct')
                
                rows, cols = np.where(segment_mask)
                if len(rows) > 0:
                    method_visualization[rows, cols] = method_colors.get(method, [255, 255, 255])
            
            # Create a SEPARATE figure for method visualization (don't modify existing one)
            try:
                fig_methods, ax_methods = plt.subplots(figsize=(6, 5))
                ax_methods.imshow(method_visualization)
                ax_methods.set_title('Compression Methods\nGreen=Mean Color, Red=DCT, Blue=Texture', fontsize=11)
                ax_methods.axis('off')
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"  ⚠️  Method visualization failed (non-critical): {e}")
            
            # Also show separate analysis of homogeneous segments
            homogeneous_segments = [s for s in subregion_results if s.get('method') == 'mean_color']
            dct_segments = [s for s in subregion_results if s.get('method') == 'dct']
            texture_segments = [s for s in subregion_results if s.get('method') == 'texture_aware']
            
            if homogeneous_segments:
                print(f"\n  HOMOGENEOUS SEGMENTS ANALYSIS ({len(homogeneous_segments)} segments):")
                print(f"    Average homogeneity: {np.mean([s['homogeneity'] for s in homogeneous_segments]):.3f}")
                print(f"    Average ratio: {np.mean([s['ratio'] for s in homogeneous_segments]):.2f}:1")
                print(f"    Average PSNR: {np.mean([s['psnr'] for s in homogeneous_segments]):.2f} dB")
                
                # Show best homogeneous segment
                best_homo = max(homogeneous_segments, key=lambda x: x['ratio'])
                print(f"    Best homogeneous segment: ID={best_homo['segment_id']}, "
                    f"Ratio={best_homo['ratio']:.2f}:1, PSNR={best_homo['psnr']:.2f} dB")
            
            if dct_segments:
                print(f"\n  COMPLEX SEGMENTS ANALYSIS ({len(dct_segments)} segments):")
                print(f"    Average homogeneity: {np.mean([s['homogeneity'] for s in dct_segments]):.3f}")
                print(f"    Average ratio: {np.mean([s['ratio'] for s in dct_segments]):.2f}:1")
                print(f"    Average PSNR: {np.mean([s['psnr'] for s in dct_segments]):.2f} dB")
            
            if texture_segments:
                print(f"\n  TEXTURE-AWARE SEGMENTS ANALYSIS ({len(texture_segments)} segments):")
                print(f"    Average ratio: {np.mean([s['ratio'] for s in texture_segments]):.2f}:1")
                print(f"    Average PSNR: {np.mean([s['psnr'] for s in texture_segments]):.2f} dB")










        
        """# ==============================================
        # 5. COMPREHENSIVE COMPARISON
        # ==============================================
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE COMPARISON")
        print(f"{'='*60}")
        
        # Create comparison table
        print(f"\n  COMPRESSION METHODS COMPARISON:")
        print(f"  {'Method':<20} {'Original':<12} {'Compressed':<12} {'Ratio':<10} {'PSNR':<10} {'BPP':<10}")
        print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
        
        # ROI method
        roi_bpp = (roi_compression_results['compressed_bytes'] * 8) / np.sum(bbox_mask)
        print(f"  {'Direct ROI':<20} {roi_compression_results['original_bytes']:<12,} "
            f"{roi_compression_results['compressed_bytes']:<12,} "
            f"{roi_compression_results['ratio']:<10.2f} "
            f"{roi_compression_results['psnr']:<10.2f} "
            f"{roi_bpp:<10.2f}")
        
        # Subregion method (if available)
        if subregion_compression_results:
            subregion_bpp = (subregion_compression_results['compressed_bytes'] * 8) / np.sum(bbox_mask)
            print(f"  {'Subregion':<20} {subregion_compression_results['original_bytes']:<12,} "
                f"{subregion_compression_results['compressed_bytes']:<12,} "
                f"{subregion_compression_results['ratio']:<10.2f} "
                f"{subregion_compression_results['overall_psnr']:<10.2f} "
                f"{subregion_bpp:<10.2f}")
            
            # Compare which is better
            print(f"\n  COMPARISON ANALYSIS:")
            
            # Ratio comparison
            if subregion_compression_results['ratio'] > roi_compression_results['ratio']:
                ratio_diff = subregion_compression_results['ratio'] - roi_compression_results['ratio']
                print(f"  ✅ Subregion compression gives BETTER ratio: +{ratio_diff:.2f}:1 better")
            else:
                ratio_diff = roi_compression_results['ratio'] - subregion_compression_results['ratio']
                print(f"  ✅ Direct ROI compression gives BETTER ratio: +{ratio_diff:.2f}:1 better")
            
            # PSNR comparison
            if subregion_compression_results['overall_psnr'] > roi_compression_results['psnr']:
                psnr_diff = subregion_compression_results['overall_psnr'] - roi_compression_results['psnr']
                print(f"  ✅ Subregion compression gives BETTER quality: +{psnr_diff:.2f} dB better")
            else:
                psnr_diff = roi_compression_results['psnr'] - subregion_compression_results['overall_psnr']
                print(f"  ✅ Direct ROI compression gives BETTER quality: +{psnr_diff:.2f} dB better")
            
            # File size comparison
            size_saving_roi = roi_compression_results['original_bytes'] - roi_compression_results['compressed_bytes']
            size_saving_sub = subregion_compression_results['original_bytes'] - subregion_compression_results['compressed_bytes']
            
            if size_saving_sub > size_saving_roi:
                print(f"  ✅ Subregion compression saves MORE space: {size_saving_sub:,} vs {size_saving_roi:,} bytes")
            else:
                print(f"  ✅ Direct ROI compression saves MORE space: {size_saving_roi:,} vs {size_saving_sub:,} bytes")
        
        # ==============================================
        # 6. VISUAL COMPARISON OF ALL METHODS
        # ==============================================
        print(f"\n  VISUAL COMPARISON OF ALL METHODS:")
        
        # Create displays with gray background
        gray_bg = np.array([200, 200, 200], dtype=np.uint8)
        
        original_display = region_image.copy()
        roi_display = roi_compression_results['reconstructed'].copy()
        
        for c in range(3):
            original_display[:, :, c] = np.where(bbox_mask, region_image[:, :, c], gray_bg[c])
            roi_display[:, :, c] = np.where(bbox_mask, roi_compression_results['reconstructed'][:, :, c], gray_bg[c])
        
        # Create figure
        n_methods = 2 if subregion_compression_results else 1
        fig_width = 5 * (n_methods + 1)  # +1 for original
        
        fig, axes = plt.subplots(2, n_methods + 1, figsize=(fig_width, 8))
        axes = axes.flatten()
        
        # Original
        axes[0].imshow(original_display)
        axes[0].set_title(f'Original ROI\n{np.sum(bbox_mask):,} pixels', fontsize=11)
        axes[0].axis('off')
        
        # Direct ROI
        axes[1].imshow(roi_display)
        axes[1].set_title(f'Direct ROI Compression\n'
                        f'Ratio: {roi_compression_results["ratio"]:.2f}:1\n'
                        f'PSNR: {roi_compression_results["psnr"]:.2f} dB', fontsize=11)
        axes[1].axis('off')
        
        # Subregion (if available)
        if subregion_compression_results:
            subregion_display = subregion_compression_results['reconstructed'].copy()
            for c in range(3):
                subregion_display[:, :, c] = np.where(bbox_mask, subregion_compression_results['reconstructed'][:, :, c], gray_bg[c])
            
            axes[2].imshow(subregion_display)
            axes[2].set_title(f'Subregion Compression\n'
                            f'Ratio: {subregion_compression_results["ratio"]:.2f}:1\n'
                            f'PSNR: {subregion_compression_results["overall_psnr"]:.2f} dB', fontsize=11)
            axes[2].axis('off')
        
        # Difference maps
        start_idx = n_methods + 1
        
        # ROI difference
        diff_roi = np.abs(region_image.astype(np.float32) - roi_compression_results['reconstructed'].astype(np.float32))
        diff_roi_masked = diff_roi * bbox_mask[:, :, np.newaxis].astype(np.float32)
        
        axes[start_idx].imshow(np.mean(diff_roi_masked, axis=2), cmap='hot')
        axes[start_idx].set_title(f'Direct ROI Error\nMax: {np.max(diff_roi_masked):.1f}', fontsize=11)
        axes[start_idx].axis('off')
        
        # Subregion difference (if available)
        if subregion_compression_results:
            diff_sub = np.abs(region_image.astype(np.float32) - subregion_compression_results['reconstructed'].astype(np.float32))
            diff_sub_masked = diff_sub * bbox_mask[:, :, np.newaxis].astype(np.float32)
            
            axes[start_idx + 1].imshow(np.mean(diff_sub_masked, axis=2), cmap='hot')
            axes[start_idx + 1].set_title(f'Subregion Error\nMax: {np.max(diff_sub_masked):.1f}', fontsize=11)
            axes[start_idx + 1].axis('off')
        
        plt.suptitle(f'ROI Region {i+1} - Compression Methods Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # ==============================================
        # 7. DETAILED SUBREGION ANALYSIS
        # ==============================================
        if subregion_compression_results and subregion_results:
            print(f"\n  DETAILED SUBREGION ANALYSIS:")
            print(f"  {'ID':<5} {'Pixels':<10} {'Original':<12} {'Compressed':<12} {'Ratio':<10} {'PSNR':<10}")
            print(f"  {'-'*5} {'-'*10} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")
            
            for seg in subregion_results[:10]:  # Show first 10 segments
                print(f"  {seg['segment_id']:<5} {seg['pixels']:<10,} "
                    f"{seg['original_bytes']:<12,} {seg['compressed_bytes']:<12,} "
                    f"{seg['ratio']:<10.2f} {seg['psnr']:<10.2f}")
            
            if len(subregion_results) > 10:
                print(f"  ... and {len(subregion_results) - 10} more segments")
            
            # Segment statistics
            seg_ratios = [s['ratio'] for s in subregion_results]
            seg_psnrs = [s['psnr'] for s in subregion_results]
            
            print(f"\n  SEGMENT STATISTICS:")
            print(f"    Best ratio: {max(seg_ratios):.2f}:1 (Segment {subregion_results[np.argmax(seg_ratios)]['segment_id']})")
            print(f"    Worst ratio: {min(seg_ratios):.2f}:1 (Segment {subregion_results[np.argmin(seg_ratios)]['segment_id']})")
            print(f"    Best PSNR: {max(seg_psnrs):.2f} dB (Segment {subregion_results[np.argmax(seg_psnrs)]['segment_id']})")
            print(f"    Worst PSNR: {min(seg_psnrs):.2f} dB (Segment {subregion_results[np.argmin(seg_psnrs)]['segment_id']})")
            print(f"    Ratio std dev: {np.std(seg_ratios):.2f}")
            print(f"    PSNR std dev: {np.std(seg_psnrs):.2f} dB")
            
            # Plot segment performance
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Ratio histogram
            axes[0].hist(seg_ratios, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0].axvline(np.mean(seg_ratios), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(seg_ratios):.2f}')
            axes[0].set_xlabel('Compression Ratio', fontsize=11)
            axes[0].set_ylabel('Number of Segments', fontsize=11)
            axes[0].set_title('Segment Compression Ratio Distribution', fontsize=12)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # PSNR histogram
            axes[1].hist(seg_psnrs, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[1].axvline(np.mean(seg_psnrs), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(seg_psnrs):.2f} dB')
            axes[1].set_xlabel('PSNR (dB)', fontsize=11)
            axes[1].set_ylabel('Number of Segments', fontsize=11)
            axes[1].set_title('Segment Quality Distribution', fontsize=12)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.suptitle(f'ROI Region {i+1} - Segment Performance Analysis', fontsize=13, fontweight='bold')
            plt.tight_layout()
            plt.show()"""
        
        # ==============================================
        # 8. FINAL SUMMARY AND SAVE OPTIONS
        # ==============================================
        """print(f"\n{'='*60}")
        print(f"FINAL SUMMARY - ROI REGION {i+1}")
        print(f"{'='*60}")
        
        # Determine best method
        if subregion_compression_results:
            if subregion_compression_results['ratio'] > roi_compression_results['ratio']:
                best_method = "SUBREGION COMPRESSION"
                reason = f"Better compression ratio ({subregion_compression_results['ratio']:.2f}:1 vs {roi_compression_results['ratio']:.2f}:1)"
            elif subregion_compression_results['overall_psnr'] > roi_compression_results['psnr']:
                best_method = "SUBREGION COMPRESSION"
                reason = f"Better quality ({subregion_compression_results['overall_psnr']:.2f} dB vs {roi_compression_results['psnr']:.2f} dB)"
            else:
                best_method = "DIRECT ROI COMPRESSION"
                reason = f"Better overall performance"
        else:
            best_method = "DIRECT ROI COMPRESSION"
            reason = "Subregion compression failed or not available"
        
        print(f"  RECOMMENDED METHOD: {best_method}")
        print(f"  Reason: {reason}")
        
        if roi_compression_results['ratio'] > 1.0:
            print(f"  ✅ Direct ROI compression achieved: {roi_compression_results['ratio']:.2f}:1 ratio")
            print(f"     Saved {roi_compression_results['original_bytes'] - roi_compression_results['compressed_bytes']:,} bytes")
        else:
            print(f"  ❌ Direct ROI compression FAILED: Ratio {roi_compression_results['ratio']:.2f}:1 (file size increased)")
        
        if subregion_compression_results:
            if subregion_compression_results['ratio'] > 1.0:
                print(f"  ✅ Subregion compression achieved: {subregion_compression_results['ratio']:.2f}:1 ratio")
                print(f"     Saved {subregion_compression_results['original_bytes'] - subregion_compression_results['compressed_bytes']:,} bytes")
            else:
                print(f"  ❌ Subregion compression FAILED: Ratio {subregion_compression_results['ratio']:.2f}:1")"""
        















    # Display SLIC results WITH BOUNDARIES
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 5, 1)
    plt.imshow(region_image)
    plt.title(f'ROI Region {i+1}')
    plt.axis('off')

    plt.subplot(1, 5, 2)
    # Show segments only within the irregular region
    segments_display = roi_segments.copy()
    segments_display[~bbox_mask] = 0  # Set background to 0
    plt.imshow(segments_display, cmap='nipy_spectral')
    plt.title(f'SLIC Segments: {roi_segments.max()}')
    plt.axis('off')

    plt.subplot(1, 5, 3)
    # Create boundaries only within the irregular region
    boundaries_image = mark_boundaries(bbox_region, roi_segments)
    boundaries_image[~bbox_mask] = 0  # Set background to black
    plt.imshow(boundaries_image)
    plt.title('SLIC Boundaries (Region Only)')
    plt.axis('off')

    plt.subplot(1, 5, 4)
    plt.imshow(texture_map)
    plt.title('Texture Map')
    plt.axis('off')
    
    # 🆕 PLOT EXTRACTED BOUNDARIES
    plt.subplot(1, 5, 5)
    plt.imshow(bbox_mask, cmap='gray')
    
    # Plot each segment boundary
    """colors = plt.cm.Set3(np.linspace(0, 1, len(segment_boundaries)))
    for j, segment in enumerate(segment_boundaries):
        coords = np.array(segment['boundary_coords'])
        if len(coords) > 0:
            plt.plot(coords[:, 1], coords[:, 0], color=colors[j], linewidth=2, 
                    label=f'Seg {segment["segment_id"]}')"""
    
    plt.title(f'Extracted Boundaries\n{len(segment_boundaries)} segments')
    plt.axis('off')
    plt.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.show()




    

