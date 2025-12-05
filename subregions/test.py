

#from regions.division import should_split
from subregions.split_score import calculate_split_score, calculate_optimal_segments
from roi.clahe import get_enhanced_image
from roi.new_roi import get_edge_map, compute_local_density, suggest_automatic_threshold, process_and_unify_borders

import sys
import math
import cv2
from matplotlib import pyplot as plt
import numpy as np


from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import numpy as np






from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import numpy as np

def extract_connected_regions(mask, original_image):
    """Extract all connected components from a mask"""
    labeled_mask = label(mask)
    regions = regionprops(labeled_mask)
    
    region_data = []
    for region in regions:
        # Create mask for this specific region
        single_region_mask = np.zeros_like(mask)
        single_region_mask[region.coords[:, 0], region.coords[:, 1]] = True
        
        # Extract the region from original image
        region_image = original_image.copy()
        region_image[~single_region_mask] = 0
        
        # Get bounding box coordinates
        minr, minc, maxr, maxc = region.bbox
        bbox_image = original_image[minr:maxr, minc:maxc]
        bbox_mask = single_region_mask[minr:maxr, minc:maxc]
        
        region_data.append({
            'mask': single_region_mask,
            'full_image': region_image,
            'bbox_image': bbox_image,
            'bbox_mask': bbox_mask,
            'bbox': region.bbox,
            'area': region.area,
            'coords': region.coords,
            'label': region.label
        })
    
    return region_data



def plot_regions(regions, title, max_display=12):
    """Plot multiple regions in a grid"""
    n_regions = min(len(regions), max_display)
    if n_regions == 0:
        print(f"No regions to display for {title}")
        return
    
    cols = 4
    rows = (n_regions + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(rows * cols):
        row = i // cols
        col = i % cols
        
        if i < n_regions:
            region = regions[i]
            axes[row, col].imshow(region['bbox_image'])
            axes[row, col].set_title(f'Region {i+1}\nArea: {region["area"]} px')
            axes[row, col].axis('off')
        else:
            axes[row, col].axis('off')
    
    # Hide empty subplots
    for i in range(n_regions, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.suptitle(f'{title} - {len(regions)} regions found', fontsize=16)
    plt.tight_layout()
    plt.show()






from skimage.color import rgb2gray, rgb2lab
from skimage import filters
import numpy as np
from skimage.segmentation import slic

def enhanced_slic_with_texture(image, n_segments, compactness=10, texture_weight=0.3):
    """
    Enhanced SLIC that considers both color and texture
    """
    # Calculate texture features
    gray = rgb2gray(image)
    texture_map = np.zeros_like(gray)
    
    # Calculate texture using Gabor filters
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        gabor_real, gabor_imag = filters.gabor(gray, frequency=0.1, theta=theta)
        gabor_mag = np.sqrt(gabor_real**2 + gabor_imag**2)
        texture_map += gabor_mag
    
    texture_map /= 4  # Average across orientations
    
    # Normalize texture map
    if texture_map.max() > 0:
        texture_map = texture_map / texture_map.max()
    
    # Convert image to LAB for better color perception
    lab_image = rgb2lab(image)
    
    # Use standard SLIC but you could extend this to custom distance function
    segments = slic(
        image, 
        n_segments=n_segments,
        compactness=compactness,
        sigma=1,
        channel_axis=2
    )
    
    return segments, texture_map




def visualize_split_analysis(region_image, overall_score, color_score, texture_score, optimal_segments):
    """Visualize the split analysis results"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original region
    axes[0, 0].imshow(region_image)
    axes[0, 0].set_title(f'Original Region\nArea: {region_image.shape[0]}x{region_image.shape[1]}')
    axes[0, 0].axis('off')
    
    # Color analysis
    lab_image = rgb2lab(region_image)
    axes[0, 1].imshow(lab_image[:, :, 0], cmap='viridis')
    axes[0, 1].set_title(f'Color Complexity: {color_score:.3f}')
    axes[0, 1].axis('off')
    
    # Texture analysis
    gray = rgb2gray(region_image)
    texture = filters.sobel(gray)
    axes[0, 2].imshow(texture, cmap='hot')
    axes[0, 2].set_title(f'Texture Complexity: {texture_score:.3f}')
    axes[0, 2].axis('off')
    
    # Scores
    scores = [overall_score, color_score, texture_score]
    labels = ['Overall', 'Color', 'Texture']
    colors = ['blue', 'green', 'red']
    axes[1, 0].bar(labels, scores, color=colors)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title('Split Scores')
    
    # Recommended segments
    axes[1, 1].text(0.5, 0.5, f'Optimal Segments:\n{optimal_segments}', 
                   ha='center', va='center', fontsize=20)
    axes[1, 1].set_title('SLIC Recommendation')
    axes[1, 1].axis('off')
    
    # Weights (you can modify these)
    weights = [0.6, 0.4]  # color, texture
    axes[1, 2].pie(weights, labels=['Color', 'Texture'], autopct='%1.1f%%')
    axes[1, 2].set_title('Feature Weights')
    
    plt.tight_layout()
    plt.show()




def normalize_result(score, window_size):
        return window_size/( 1+ math.exp(-12*(score-0.5)) ) 













from skimage.segmentation import find_boundaries
from skimage.measure import find_contours
import numpy as np

def extract_slic_segment_boundaries(roi_segments, bbox_mask):
    """
    Extract boundaries of all SLIC segments within the irregular region
    
    Returns:
    --------
    list of dictionaries, each containing:
        'segment_id': ID of the segment
        'boundary_coords': list of (y,x) coordinates
        'area': number of pixels in segment
    """
    segment_boundaries = []
    
    # Get unique segment IDs (excluding background)
    segment_ids = np.unique(roi_segments)
    segment_ids = segment_ids[segment_ids != 0]  # Remove background
    
    for seg_id in segment_ids:
        # Create mask for this specific segment
        segment_mask = (roi_segments == seg_id) & bbox_mask
        
        if np.sum(segment_mask) > 0:  # If segment exists in our region
            # Find contours/boundaries
            contours = find_contours(segment_mask, level=0.5)
            
            # Take the longest contour (main boundary)
            if len(contours) > 0:
                main_contour = max(contours, key=len)
                
                # Convert to list of (y,x) coordinates
                boundary_coords = [(coord[0], coord[1]) for coord in main_contour]
                
                segment_boundaries.append({
                    'segment_id': int(seg_id),
                    'boundary_coords': boundary_coords,
                    'area': np.sum(segment_mask),
                    'num_points': len(boundary_coords)
                })
    
    return segment_boundaries

def save_boundaries_to_file(segment_boundaries, filename):
    """
    Save segment boundaries to a text file
    """
    with open(filename, 'w') as f:
        f.write("SLIC Segment Boundaries\n")
        f.write("=" * 50 + "\n")
        
        for segment in segment_boundaries:
            f.write(f"\nSegment {segment['segment_id']}:\n")
            f.write(f"Area: {segment['area']} pixels\n")
            f.write(f"Boundary points: {segment['num_points']}\n")
            f.write("Coordinates (y,x):\n")
            
            # Write coordinates in batches for readability
            f.write("[")
            for i in range(0, len(segment['boundary_coords']), 5):
                batch = segment['boundary_coords'][i:i+5]
                coord_str = "  ".join([f"({y:.1f},{x:.1f})," for y, x in batch])
                f.write(f"  {coord_str} \n")
            f.write("]")



































































import scipy.fft

def compress_region_with_dct(region_image, region_mask, quality=75):
    """
    Apply DCT compression to a region
    Returns compressed data and reconstruction metrics
    """
    # Convert to YUV color space (like JPEG)
    if region_image.shape[-1] == 3:  # RGB image
        yuv_image = rgb_to_yuv(region_image)
    else:
        yuv_image = region_image
    
    compressed_channels = {}
    reconstruction_metrics = {}
    
    for channel_idx, channel_name in enumerate(['Y', 'U', 'V']):
        if channel_idx < yuv_image.shape[-1]:
            channel_data = yuv_image[:, :, channel_idx]
            
            # Apply mask to get only the region pixels
            masked_channel = channel_data * region_mask
            
            # Get bounding box of non-zero region to avoid compressing empty areas
            rows, cols = np.where(region_mask)
            if len(rows) == 0:
                # Empty region, skip compression
                compressed_channels[channel_name] = None
                continue
                
            min_row, max_row = np.min(rows), np.max(rows)
            min_col, max_col = np.min(cols), np.max(cols)
            
            # Extract the actual region content (with padding for block alignment)
            region_content = masked_channel[min_row:max_row+1, min_col:max_col+1]
            bbox_height, bbox_width = region_content.shape
            
            # Pad to multiple of 8 for block processing
            pad_bottom = (8 - (bbox_height % 8)) % 8
            pad_right = (8 - (bbox_width % 8)) % 8
            padded_region = np.pad(region_content, ((0, pad_bottom), (0, pad_right)), mode='constant')
            
            # Apply 8x8 block DCT
            compressed_blocks = []
            original_blocks = []
            
            for i in range(0, padded_region.shape[0], 8):
                for j in range(0, padded_region.shape[1], 8):
                    block = padded_region[i:i+8, j:j+8]
                    if block.shape == (8, 8):
                        # Apply DCT
                        dct_block = scipy.fft.dctn(block, norm='ortho')
                        original_blocks.append(block.copy())
                        
                        # Quantize (simplified - you can use JPEG quantization tables)
                        quantized_block = quantize_dct_block(dct_block, quality)
                        compressed_blocks.append(quantized_block)
            
            compressed_channels[channel_name] = {
                'compressed_blocks': compressed_blocks,
                'bbox': (min_row, min_col, bbox_height, bbox_width),
                'padding': (pad_bottom, pad_right),
                'original_shape': region_content.shape
            }
            
            # Calculate compression metrics for this channel
            if original_blocks:
                original_size = sum(block.size for block in original_blocks) * 8  # 8 bytes per float64
                compressed_size = estimate_compressed_size(compressed_blocks)
                reconstruction_metrics[channel_name] = {
                    'original_size': original_size,
                    'compressed_size': compressed_size,
                    'compression_ratio': compressed_size / original_size if original_size > 0 else 0
                }
    
    return compressed_channels, reconstruction_metrics

def quantize_dct_block(dct_block, quality):
    """
    Simple quantization - you can replace with standard JPEG quantization tables
    """
    # Simplified quantization: keep only significant coefficients
    threshold = (100 - quality) / 100.0 * np.max(np.abs(dct_block))
    quantized = np.where(np.abs(dct_block) > threshold, dct_block, 0)
    return quantized

def estimate_compressed_size(compressed_blocks):
    """
    Estimate storage size for compressed blocks
    In real implementation, you'd use run-length + Huffman encoding
    """
    total_elements = sum(np.count_nonzero(block) for block in compressed_blocks)
    # Estimate: each non-zero element needs value + position
    return total_elements * (8 + 4)  # 8 bytes value + 4 bytes position

def rgb_to_yuv(rgb_image):
    """
    Convert RGB to YUV color space
    """
    # Simple conversion (you can use proper color matrix)
    r, g, b = rgb_image[:,:,0], rgb_image[:,:,1], rgb_image[:,:,2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = 0.492 * (b - y)  # Simplified
    v = 0.877 * (r - y)  # Simplified
    return np.stack([y, u, v], axis=-1)













def decompress_region_from_dct(compressed_data, original_mask, quality=75):
    """
    Reconstruct region from DCT compressed data
    """
    # Create empty reconstruction
    reconstructed_yuv = np.zeros((original_mask.shape[0], original_mask.shape[1], 3))
    
    for channel_idx, channel_name in enumerate(['Y', 'U', 'V']):
        if channel_name in compressed_data and compressed_data[channel_name] is not None:
            channel_info = compressed_data[channel_name]
            compressed_blocks = channel_info['compressed_blocks']
            min_row, min_col, bbox_height, bbox_width = channel_info['bbox']
            pad_bottom, pad_right = channel_info['padding']
            
            # Reconstruct the padded region
            padded_height = bbox_height + pad_bottom
            padded_width = bbox_width + pad_right
            reconstructed_channel = np.zeros((padded_height, padded_width))
            
            block_idx = 0
            for i in range(0, padded_height, 8):
                for j in range(0, padded_width, 8):
                    if block_idx < len(compressed_blocks):
                        # Get the compressed block
                        compressed_block = compressed_blocks[block_idx]
                        
                        # Apply inverse DCT
                        idct_block = scipy.fft.idctn(compressed_block, norm='ortho')
                        
                        # Place back in reconstructed image
                        if i+8 <= padded_height and j+8 <= padded_width:
                            reconstructed_channel[i:i+8, j:j+8] = idct_block
                        
                        block_idx += 1
            
            # Remove padding and place in original position
            original_region = reconstructed_channel[:bbox_height, :bbox_width]
            reconstructed_yuv[min_row:min_row+bbox_height, min_col:min_col+bbox_width, channel_idx] = original_region
    
    # Convert back to RGB
    reconstructed_rgb = yuv_to_rgb(reconstructed_yuv)
    
    # Apply the original mask to get exact region shape
    reconstructed_rgb[~original_mask] = 0
    
    return reconstructed_rgb

def yuv_to_rgb(yuv_image):
    """
    Convert YUV back to RGB
    """
    y, u, v = yuv_image[:,:,0], yuv_image[:,:,1], yuv_image[:,:,2]
    
    # Inverse of our simple conversion
    r = y + 1.140 * v
    g = y - 0.395 * u - 0.581 * v
    b = y + 2.032 * u
    
    # Clip to valid range
    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0, 1)  # Assuming values were normalized 0-1
    
    return rgb

def quantize_dct_block(dct_block, quality):
    """
    Improved quantization function (should match compression)
    """
    # Better quantization: use threshold based on quality
    max_coeff = np.max(np.abs(dct_block))
    threshold = (100 - quality) / 100.0 * max_coeff * 0.1  # Adjusted scaling
    
    # Keep only coefficients above threshold
    quantized = np.where(np.abs(dct_block) > threshold, dct_block, 0)
    return quantized



































def compress_segment_with_dct(segment_mask, region_image, quality=85):  # Increased quality default
    """
    Apply DCT compression to a single SLIC segment with proper color handling
    """
    # Extract only the pixels belonging to this segment
    rows, cols = np.where(segment_mask)
    if len(rows) == 0:
        return None, None
    
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)
    
    # Extract segment content (keep original colors!)
    segment_content = region_image[min_row:max_row+1, min_col:max_col+1].copy()
    segment_mask_cropped = segment_mask[min_row:max_row+1, min_col:max_col+1]
    
    # Only process the actual segment area (set background to black)
    for c in range(3):  # For each RGB channel
        channel = segment_content[:, :, c]
        channel[~segment_mask_cropped] = 0
        segment_content[:, :, c] = channel
    
    # Convert to float for DCT processing
    segment_float = segment_content.astype(np.float32) / 255.0
    
    compressed_channels = {}
    reconstruction_metrics = {}
    
    for channel_idx, channel_name in enumerate(['R', 'G', 'B']):  # Keep RGB for simplicity
        channel_data = segment_float[:, :, channel_idx]
        
        segment_height, segment_width = segment_content.shape[:2]
        
        # Pad to multiple of 8 for block processing
        pad_bottom = (8 - (segment_height % 8)) % 8
        pad_right = (8 - (segment_width % 8)) % 8
        padded_segment = np.pad(channel_data, ((0, pad_bottom), (0, pad_right)), mode='constant')
        
        # Apply 8x8 block DCT
        compressed_blocks = []
        
        for i in range(0, padded_segment.shape[0], 8):
            for j in range(0, padded_segment.shape[1], 8):
                block = padded_segment[i:i+8, j:j+8]
                if block.shape == (8, 8):
                    # Only compress if this block has content
                    block_mask = segment_mask_cropped[i:min(i+8, segment_height), 
                                                     j:min(j+8, segment_width)]
                    if np.any(block_mask):
                        dct_block = scipy.fft.dctn(block, norm='ortho')
                        
                        # GENTLE quantization - keep most coefficients
                        quantized_block = quantize_dct_gentle(dct_block, quality)
                        compressed_blocks.append({
                            'block_data': quantized_block,
                            'position': (i, j)
                        })
        
        compressed_channels[channel_name] = {
            'compressed_blocks': compressed_blocks,
            'bbox': (min_row, min_col, segment_height, segment_width),
            'padding': (pad_bottom, pad_right),
            'segment_mask': segment_mask_cropped
        }
        
        # Calculate metrics
        original_size = segment_height * segment_width * 4  # 4 bytes per float32
        compressed_size = len(compressed_blocks) * 64 * 4  # 64 coefficients per block
        reconstruction_metrics[channel_name] = {
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compressed_size / original_size if original_size > 0 else 0
        }
    
    return compressed_channels, reconstruction_metrics

def quantize_dct_gentle(dct_block, quality):
    """
    MUCH gentler quantization - keep most detail
    """
    # Very mild quantization table
    mild_quant_table = np.array([
        [1, 1, 1, 1, 2, 2, 3, 4],
        [1, 1, 1, 1, 2, 3, 4, 5],
        [1, 1, 1, 2, 2, 3, 5, 6],
        [1, 1, 2, 2, 3, 4, 6, 7],
        [2, 2, 2, 3, 4, 5, 7, 8],
        [2, 3, 3, 4, 5, 6, 8, 9],
        [3, 4, 5, 6, 7, 8, 9, 10],
        [4, 5, 6, 7, 8, 9, 10, 11]
    ], dtype=np.float32)
    
    # Scale based on quality (higher quality = less quantization)
    scale = (100 - quality) / 50.0  # Much gentler scaling
    quant_table = np.maximum(1, mild_quant_table * scale)
    
    # Quantize gently - round to nearest integer but keep most values
    quantized = np.round(dct_block / quant_table) * quant_table
    
    return quantized

def decompress_segment_from_dct(compressed_data, segment_mask, quality=85):
    """
    Decompress a single segment with proper color reconstruction
    """
    reconstructed_rgb = np.zeros((segment_mask.shape[0], segment_mask.shape[1], 3), dtype=np.float32)
    
    for channel_idx, channel_name in enumerate(['R', 'G', 'B']):
        if channel_name in compressed_data:
            channel_info = compressed_data[channel_name]
            compressed_blocks = channel_info['compressed_blocks']
            min_row, min_col, seg_height, seg_width = channel_info['bbox']
            pad_bottom, pad_right = channel_info['padding']
            
            padded_height = seg_height + pad_bottom
            padded_width = seg_width + pad_right
            reconstructed_channel = np.zeros((padded_height, padded_width), dtype=np.float32)
            
            for block_info in compressed_blocks:
                block_data = block_info['block_data']
                i, j = block_info['position']
                
                # Inverse DCT (no dequantization needed since we stored quantized values)
                idct_block = scipy.fft.idctn(block_data, norm='ortho')
                
                # Place in reconstructed image
                if i+8 <= padded_height and j+8 <= padded_width:
                    reconstructed_channel[i:i+8, j:j+8] = idct_block
            
            # Remove padding and place in position
            original_segment = reconstructed_channel[:seg_height, :seg_width]
            reconstructed_rgb[min_row:min_row+seg_height, min_col:min_col+seg_width, channel_idx] = original_segment
    
    # Convert back to 0-255 range and apply mask
    reconstructed_rgb = np.clip(reconstructed_rgb, 0, 1)
    reconstructed_uint8 = (reconstructed_rgb * 255).astype(np.uint8)
    
    # Only keep the segment area
    final_reconstruction = np.zeros_like(reconstructed_uint8)
    final_reconstruction[segment_mask] = reconstructed_uint8[segment_mask]
    
    return final_reconstruction

def rgb_to_yuv_proper(rgb_image):
    """
    Proper RGB to YUV conversion
    """
    if rgb_image.dtype != np.float32 and rgb_image.dtype != np.float64:
        rgb_normalized = rgb_image.astype(np.float32) / 255.0
    else:
        rgb_normalized = rgb_image
    
    # Standard conversion matrix
    r, g, b = rgb_normalized[:,:,0], rgb_normalized[:,:,1], rgb_normalized[:,:,2]
    
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.14713 * r - 0.28886 * g + 0.436 * b
    v = 0.615 * r - 0.51499 * g - 0.10001 * b
    
    return np.stack([y, u, v], axis=-1)

def yuv_to_rgb_proper(yuv_image):
    """
    Proper YUV to RGB conversion
    """
    y, u, v = yuv_image[:,:,0], yuv_image[:,:,1], yuv_image[:,:,2]
    
    r = y + 1.13983 * v
    g = y - 0.39465 * u - 0.58060 * v
    b = y + 2.03211 * u
    
    rgb = np.stack([r, g, b], axis=-1)
    rgb = np.clip(rgb, 0, 1)
    
    # Convert back to 0-255 range
    return (rgb * 255).astype(np.uint8)

def quantize_dct_block_proper(dct_block, quality):
    """
    Better quantization using JPEG-like quantization tables
    """
    # Standard JPEG luminance quantization table
    luminance_quant_table = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])
    
    # Scale quantization table based on quality
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality
    
    quant_table = np.maximum(1, (luminance_quant_table * scale + 50) / 100)
    
    # Quantize
    quantized = np.round(dct_block / quant_table)
    
    return quantized


"""def get_quantization_table(quality):

    # Standard JPEG luminance quantization table
    base_table = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])
    
    # Scale based on quality
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality
    
    return np.maximum(1, (base_table * scale + 50) / 100)

"""












def calculate_real_compressed_size(compressed_channels):
    """
    Calculate REAL compressed size (what would be stored in a file)
    """
    if compressed_channels is None:
        return 0
    
    total_size = 0
    
    for channel_idx in range(3):
        if channel_idx in compressed_channels:
            channel_info = compressed_channels[channel_idx]
            
            # 1. Quantization table (8x8 float32)
            if 'quantization_table' in channel_info:
                total_size += channel_info['quantization_table'].nbytes
            
            # 2. Original brightness (1 float32)
            if 'original_brightness' in channel_info:
                total_size += 4  # 4 bytes for float32
            
            # 3. Block positions (list of tuples - 2 ints each)
            if 'block_positions' in channel_info:
                total_size += len(channel_info['block_positions']) * 8  # 2 * 4 bytes
            
            # 4. Compressed blocks (THIS IS THE MAIN PART)
            if 'compressed_blocks' in channel_info:
                for block in channel_info['compressed_blocks']:
                    # In real compression, these would be entropy coded
                    # For estimation, count them as int16
                    total_size += len(block) * 2  # 2 bytes per coefficient
    
    # Add overhead for data structure (headers, etc.)
    total_size += 100  # Approximate overhead
    
    return total_size

def calculate_compression_statistics(compressed_channels, original_size_bytes):
    """
    Calculate REAL compression statistics
    """
    if compressed_channels is None:
        return 0, 0, 1.0
    
    # Calculate REAL compressed size
    compressed_size_bytes = calculate_real_compressed_size(compressed_channels)
    
    # Calculate compression ratio
    if compressed_size_bytes > 0:
        compression_ratio = original_size_bytes / compressed_size_bytes
    else:
        compression_ratio = 1.0
    
    return compressed_size_bytes, original_size_bytes, compression_ratio




















def compress_segment_adaptive_blocks(segment_mask, region_image, quality=90):
    """
    Use adaptive block sizes based on segment characteristics
    """
    rows, cols = np.where(segment_mask)
    if len(rows) == 0:
        return None, None
    
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)
    
    segment_content = region_image[min_row:max_row+1, min_col:max_col+1].copy()
    segment_mask_cropped = segment_mask[min_row:max_row+1, min_col:max_col+1]
    
    # Analyze segment texture to choose block size
    segment_std = np.std(segment_content[segment_mask_cropped])
    segment_size = np.sum(segment_mask_cropped)
    
    # Choose block size based on content
    if segment_std < 10:  # Smooth area
        block_size = 16  # Larger blocks for smooth areas
    elif segment_std > 50:  # High texture
        block_size = 4   # Smaller blocks for detailed areas
    else:
        block_size = 8   # Default
    
    # For very small segments, use even smaller blocks
    if segment_size < 100:
        block_size = max(4, block_size // 2)
    
    segment_float = segment_content.astype(np.float32) / 255.0
    compressed_channels = {}
    
    for channel_idx, channel_name in enumerate(['R', 'G', 'B']):
        channel_data = segment_float[:, :, channel_idx]
        segment_height, segment_width = segment_content.shape[:2]
        
        # Pad to multiple of block_size
        pad_bottom = (block_size - (segment_height % block_size)) % block_size
        pad_right = (block_size - (segment_width % block_size)) % block_size
        padded_segment = np.pad(channel_data, ((0, pad_bottom), (0, pad_right)), mode='edge')  # Edge padding
        
        compressed_blocks = []
        
        for i in range(0, padded_segment.shape[0], block_size):
            for j in range(0, padded_segment.shape[1], block_size):
                block = padded_segment[i:i+block_size, j:j+block_size]
                if block.shape == (block_size, block_size):
                    block_mask = segment_mask_cropped[i:min(i+block_size, segment_height), 
                                                     j:min(j+block_size, segment_width)]
                    if np.any(block_mask):
                        dct_block = scipy.fft.dctn(block, norm='ortho')
                        quantized_block = quantize_adaptive(dct_block, quality, block_size)
                        compressed_blocks.append({
                            'block_data': quantized_block,
                            'position': (i, j),
                            'block_size': block_size
                        })
        
        compressed_channels[channel_name] = {
            'compressed_blocks': compressed_blocks,
            'bbox': (min_row, min_col, segment_height, segment_width),
            'padding': (pad_bottom, pad_right),
            'segment_mask': segment_mask_cropped,
            'block_size': block_size
        }
    
    return compressed_channels, {'block_size_used': block_size}

def quantize_adaptive(dct_block, quality, block_size):
    """
    Adaptive quantization based on block size and frequency
    """
    # Create custom quantization table based on block size
    if block_size == 4:
        quant_table = np.array([
            [1, 1, 2, 3],
            [1, 1, 2, 3],
            [2, 2, 3, 4],
            [3, 3, 4, 5]
        ], dtype=np.float32)
    elif block_size == 16:
        quant_table = np.array([
            [1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6],
            [1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
            [1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7],
            [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7],
            [1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8],
            [2, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8],
            [2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9],
            [2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9],
            [3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10],
            [3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10],
            [3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11],
            [4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11],
            [4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12],
            [5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12],
            [5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13],
            [6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13]
        ], dtype=np.float32)
    else:  # 8x8
        quant_table = np.array([
            [1, 1, 1, 2, 2, 3, 4, 5],
            [1, 1, 1, 2, 3, 4, 5, 6],
            [1, 1, 2, 2, 3, 4, 6, 7],
            [2, 2, 2, 3, 4, 5, 7, 8],
            [2, 3, 3, 4, 5, 6, 8, 9],
            [3, 4, 4, 5, 6, 7, 9, 10],
            [4, 5, 6, 7, 8, 9, 10, 11],
            [5, 6, 7, 8, 9, 10, 11, 12]
        ], dtype=np.float32)
    
    # Gentle scaling
    scale = max(0.1, (100 - quality) / 200.0)  # Even gentler
    quant_table = np.maximum(0.5, quant_table * scale)  # Minimum quantization of 0.5
    
    quantized = dct_block / quant_table
    # Don't round - keep floating point for better quality
    return quantized















def create_cosine_window(size):
    """
    Create cosine window for smooth overlapping
    """
    window = np.outer(
        np.hanning(size),
        np.hanning(size)
    )
    return window

def decompress_segment_from_dct_improved(compressed_data, segment_mask, quality=95):
    """
    Improved decompression for adaptive blocks
    """
    reconstructed_rgb = np.zeros((segment_mask.shape[0], segment_mask.shape[1], 3), dtype=np.float32)
    
    for channel_idx, channel_name in enumerate(['R', 'G', 'B']):
        if channel_name in compressed_data:
            channel_info = compressed_data[channel_name]
            compressed_blocks = channel_info['compressed_blocks']
            min_row, min_col, seg_height, seg_width = channel_info['bbox']
            pad_bottom, pad_right = channel_info['padding']
            block_size = channel_info.get('block_size', 8)  # Default to 8 if not specified
            
            padded_height = seg_height + pad_bottom
            padded_width = seg_width + pad_right
            reconstructed_channel = np.zeros((padded_height, padded_width), dtype=np.float32)
            
            for block_info in compressed_blocks:
                block_data = block_info['block_data']
                i, j = block_info['position']
                current_block_size = block_info.get('block_size', block_size)
                
                # Inverse DCT
                idct_block = scipy.fft.idctn(block_data, norm='ortho')
                
                # Place in reconstructed image
                if i+current_block_size <= padded_height and j+current_block_size <= padded_width:
                    reconstructed_channel[i:i+current_block_size, j:j+current_block_size] = idct_block
            
            # Remove padding and place in position
            original_segment = reconstructed_channel[:seg_height, :seg_width]
            reconstructed_rgb[min_row:min_row+seg_height, min_col:min_col+seg_width, channel_idx] = original_segment
    
    # Convert back to 0-255 range and apply mask
    reconstructed_rgb = np.clip(reconstructed_rgb, 0, 1)
    reconstructed_uint8 = (reconstructed_rgb * 255).astype(np.uint8)
    
    # Apply gentle smoothing to reduce block artifacts
    from scipy.ndimage import gaussian_filter
    
    reconstructed_smoothed = reconstructed_uint8.astype(np.float32)
    for c in range(3):
        channel = reconstructed_smoothed[:, :, c]
        # Only smooth non-zero areas
        non_zero_mask = channel > 0
        if np.any(non_zero_mask):
            # Very gentle Gaussian smoothing (sigma=0.5 is very light)
            smoothed_channel = gaussian_filter(channel, sigma=0.5)
            # Blend original and smoothed (only 20% smoothing)
            alpha = 0.2
            channel[non_zero_mask] = (1-alpha) * channel[non_zero_mask] + alpha * smoothed_channel[non_zero_mask]
        reconstructed_smoothed[:, :, c] = channel
    
    final_reconstruction = np.clip(reconstructed_smoothed, 0, 255).astype(np.uint8)
    
    # Only keep the segment area
    result = np.zeros_like(final_reconstruction)
    result[segment_mask] = final_reconstruction[segment_mask]
    
    return result























"""def compress_segment_with_dct_fixed(segment_mask, region_image, quality=95):
 
    rows, cols = np.where(segment_mask)
    if len(rows) == 0:
        return None, None
    
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)
    
    # Extract segment and ensure we keep original brightness
    segment_content = region_image[min_row:max_row+1, min_col:max_col+1].copy()
    segment_mask_cropped = segment_mask[min_row:max_row+1, min_col:max_col+1]
    
    # Store original mean brightness for each channel
    original_means = []
    for c in range(3):
        channel_data = segment_content[:, :, c]
        original_mean = np.mean(channel_data[segment_mask_cropped])
        original_means.append(original_mean)
        
        # Zero out non-segment areas
        channel_data[~segment_mask_cropped] = 0
        segment_content[:, :, c] = channel_data
    
    # Convert to float for DCT (0-1 range)
    segment_float = segment_content.astype(np.float32) / 255.0
    
    compressed_channels = {}
    
    for channel_idx, channel_name in enumerate(['R', 'G', 'B']):
        channel_data = segment_float[:, :, channel_idx]
        segment_height, segment_width = segment_content.shape[:2]
        
        # Use smaller blocks for better quality (4x4 instead of 8x8)
        block_size = 4
        pad_bottom = (block_size - (segment_height % block_size)) % block_size
        pad_right = (block_size - (segment_width % block_size)) % block_size
        
        # Use edge padding to avoid dark borders
        padded_segment = np.pad(channel_data, ((0, pad_bottom), (0, pad_right)), mode='edge')
        
        compressed_blocks = []
        
        for i in range(0, padded_segment.shape[0], block_size):
            for j in range(0, padded_segment.shape[1], block_size):
                block = padded_segment[i:i+block_size, j:j+block_size]
                if block.shape == (block_size, block_size):
                    block_mask = segment_mask_cropped[i:min(i+block_size, segment_height), 
                                                     j:min(j+block_size, segment_width)]
                    if np.any(block_mask):
                        dct_block = scipy.fft.dctn(block, norm='ortho')
                        
                        # VERY GENTLE quantization - preserve most coefficients
                        quantized_block = quantize_ultra_gentle(dct_block, quality)
                        compressed_blocks.append({
                            'block_data': quantized_block,
                            'position': (i, j)
                        })
        
        compressed_channels[channel_name] = {
            'compressed_blocks': compressed_blocks,
            'bbox': (min_row, min_col, segment_height, segment_width),
            'padding': (pad_bottom, pad_right),
            'segment_mask': segment_mask_cropped,
            'original_mean': original_means[channel_idx],  # Store original brightness
            'block_size': block_size
        }
    
    return compressed_channels, {'original_means': original_means}
"""
"""def quantize_ultra_gentle(dct_block, quality):

    block_size = dct_block.shape[0]
    
    # Create extremely gentle quantization table
    if block_size == 4:
        quant_table = np.array([
            [0.5, 0.5, 0.7, 0.9],
            [0.5, 0.5, 0.7, 0.9],
            [0.7, 0.7, 0.9, 1.1],
            [0.9, 0.9, 1.1, 1.3]
        ], dtype=np.float32)
    else:  # 8x8
        quant_table = np.array([
            [0.5, 0.5, 0.5, 0.7, 0.7, 0.9, 1.1, 1.3],
            [0.5, 0.5, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
            [0.5, 0.5, 0.7, 0.7, 0.9, 1.1, 1.5, 1.7],
            [0.7, 0.7, 0.7, 0.9, 1.1, 1.3, 1.7, 1.9],
            [0.7, 0.9, 0.9, 1.1, 1.3, 1.5, 1.9, 2.1],
            [0.9, 1.1, 1.1, 1.3, 1.5, 1.7, 2.1, 2.3],
            [1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5],
            [1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7]
        ], dtype=np.float32)
    
    # Even gentler scaling
    scale = max(0.05, (100 - quality) / 500.0)  # Much gentler!
    quant_table = np.maximum(0.1, quant_table * scale)  # Minimum 0.1
    
    # Don't quantize DC coefficient (brightness) at all!
    quant_table[0, 0] = 0.1  # Almost no quantization for brightness
    
    quantized = dct_block / quant_table
    return quantized
"""

def quantize_ultra_gentle(dct_block, quality):
    """
    CORRECT quantization - higher quality = smaller quantization values
    """
    block_size = dct_block.shape[0]
    
    # Standard JPEG quantization table (8x8)
    if block_size == 8:
        std_table = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=np.float32)
    else:  # 4x4 - scaled down version
        std_table = np.array([
            [8, 6, 5, 8],
            [6, 6, 7, 10],
            [7, 7, 8, 12],
            [9, 11, 14, 16]
        ], dtype=np.float32)
    
    # Quality scaling (0-100 to 0.01-2.0 scale)
    if quality >= 50:
        scale_factor = (100 - quality) / 50.0
    else:
        scale_factor = 50.0 / quality
    
    # Adjust quantization table based on quality
    quant_table = std_table * scale_factor
    quant_table = np.clip(quant_table, 1, 255)  # Never go below 1
    
    # For high quality (95), scale_factor ≈ 0.1, so quant_table values get SMALLER
    # For low quality (10), scale_factor ≈ 5.0, so quant_table values get LARGER
    
    # Quantize (this is where compression happens)
    quantized = np.round(dct_block / quant_table)
    
    return quantized

def get_quantization_table_fixed(quality, block_size):
    """
    Get the proper quantization table for dequantization
    """
    if block_size == 4:
        quant_table = np.array([
            [0.5, 0.5, 0.7, 0.9],
            [0.5, 0.5, 0.7, 0.9],
            [0.7, 0.7, 0.9, 1.1],
            [0.9, 0.9, 1.1, 1.3]
        ], dtype=np.float32)
    else:  # 8x8 or default
        quant_table = np.array([
            [0.5, 0.5, 0.5, 0.7, 0.7, 0.9, 1.1, 1.3],
            [0.5, 0.5, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
            [0.5, 0.5, 0.7, 0.7, 0.9, 1.1, 1.5, 1.7],
            [0.7, 0.7, 0.7, 0.9, 1.1, 1.3, 1.7, 1.9],
            [0.7, 0.9, 0.9, 1.1, 1.3, 1.5, 1.9, 2.1],
            [0.9, 1.1, 1.1, 1.3, 1.5, 1.7, 2.1, 2.3],
            [1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5],
            [1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7]
        ], dtype=np.float32)
    
    # Apply quality scaling (must match the compression side!)
    scale = max(0.05, (100 - quality) / 500.0)
    quant_table = np.maximum(0.1, quant_table * scale)
    
    # Preserve DC coefficient (brightness)
    quant_table[0, 0] = 0.1
    
    return quant_table


def decompress_segment_fixed(compressed_data, segment_mask, quality=95):
    """
    Fixed decompression with brightness preservation
    """
    reconstructed_rgb = np.zeros((segment_mask.shape[0], segment_mask.shape[1], 3), dtype=np.float32)
    
    for channel_idx, channel_name in enumerate(['R', 'G', 'B']):
        if channel_name in compressed_data:
            channel_info = compressed_data[channel_name]
            compressed_blocks = channel_info['compressed_blocks']
            min_row, min_col, seg_height, seg_width = channel_info['bbox']
            pad_bottom, pad_right = channel_info['padding']
            original_mean = channel_info['original_mean']
            block_size = channel_info.get('block_size', 4)
            
            padded_height = seg_height + pad_bottom
            padded_width = seg_width + pad_right
            reconstructed_channel = np.zeros((padded_height, padded_width), dtype=np.float32)
            
            for block_info in compressed_blocks:
                block_data = block_info['block_data']
                i, j = block_info['position']
                
                # Get quantization table for dequantization
                quant_table = get_quantization_table_fixed(quality, block_size)
                
                # Dequantize
                dequantized_block = block_data * quant_table
                
                # Inverse DCT
                idct_block = scipy.fft.idctn(dequantized_block, norm='ortho')
                
                # Place in reconstructed image
                if i+block_size <= padded_height and j+block_size <= padded_width:
                    reconstructed_channel[i:i+block_size, j:j+block_size] = idct_block
            
            # Remove padding
            original_segment = reconstructed_channel[:seg_height, :seg_width]
            
            # Adjust brightness to match original
            current_mean = np.mean(original_segment[channel_info['segment_mask']])
            if current_mean > 0 and original_mean > 0:
                brightness_ratio = original_mean / 255.0 / (current_mean + 1e-8)
                original_segment *= brightness_ratio
            
            reconstructed_rgb[min_row:min_row+seg_height, min_col:min_col+seg_width, channel_idx] = original_segment
    
    # Convert back to 0-255 range
    reconstructed_rgb = np.clip(reconstructed_rgb, 0, 1)
    reconstructed_uint8 = (reconstructed_rgb * 255).astype(np.uint8)
    
    # Only keep the segment area
    final_reconstruction = np.zeros_like(reconstructed_uint8)
    final_reconstruction[segment_mask] = reconstructed_uint8[segment_mask]
    
    return final_reconstruction









def compress_segment_with_dct_efficient(segment_mask, region_image, quality=50):
    """
    Efficient DCT compression with actual size reduction
    """
    rows, cols = np.where(segment_mask)
    if len(rows) == 0:
        return None, None
    
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)
    
    segment_content = region_image[min_row:max_row+1, min_col:max_col+1].copy()
    segment_mask_cropped = segment_mask[min_row:max_row+1, min_col:max_col+1]
    
    # Store original means for brightness correction
    original_means = []
    for c in range(3):
        channel_data = segment_content[:, :, c]
        original_mean = np.mean(channel_data[segment_mask_cropped])
        original_means.append(original_mean)
        channel_data[~segment_mask_cropped] = 0
    
    segment_float = segment_content.astype(np.float32) / 255.0
    compressed_channels = {}
    
    for channel_idx, channel_name in enumerate(['R', 'G', 'B']):
        channel_data = segment_float[:, :, channel_idx]
        segment_height, segment_width = segment_content.shape[:2]
        
        # Use 8x8 blocks for better compression
        block_size = 8
        pad_bottom = (block_size - (segment_height % block_size)) % block_size
        pad_right = (block_size - (segment_width % block_size)) % block_size
        
        padded_segment = np.pad(channel_data, ((0, pad_bottom), (0, pad_right)), mode='constant')
        
        compressed_blocks = []
        
        for i in range(0, padded_segment.shape[0], block_size):
            for j in range(0, padded_segment.shape[1], block_size):
                block = padded_segment[i:i+block_size, j:j+block_size]
                if block.shape == (block_size, block_size):
                    block_mask = segment_mask_cropped[i:min(i+block_size, segment_height), 
                                                     j:min(j+block_size, segment_width)]
                    if np.any(block_mask):
                        dct_block = scipy.fft.dctn(block, norm='ortho')
                        
                        # ACTUAL COMPRESSION: Threshold and store sparse representation
                        quantized_block = quantize_with_threshold(dct_block, quality)
                        
                        # Convert to sparse representation (only store non-zero values)
                        nonzero_indices = np.where(np.abs(quantized_block) > 1e-6)
                        if len(nonzero_indices[0]) > 0:
                            sparse_data = {
                                'values': quantized_block[nonzero_indices].astype(np.float16),  # Half precision!
                                'indices': np.ravel_multi_index(nonzero_indices, (block_size, block_size)),
                                'count': len(nonzero_indices[0])
                            }
                            
                            # Only store if we actually achieve compression
                            if sparse_data['count'] < block_size * block_size * 0.8:  # At least 20% reduction
                                compressed_blocks.append({
                                    'sparse_data': sparse_data,
                                    'position': (i, j)
                                })
                            else:
                                # Store full block if sparse doesn't help
                                compressed_blocks.append({
                                    'full_block': quantized_block.astype(np.float16),
                                    'position': (i, j)
                                })
        
        compressed_channels[channel_name] = {
            'compressed_blocks': compressed_blocks,
            'bbox': (min_row, min_col, segment_height, segment_width),
            'padding': (pad_bottom, pad_right),
            'segment_mask': segment_mask_cropped,
            'original_mean': original_means[channel_idx],
            'block_size': block_size
        }
    
    return compressed_channels, {'original_means': original_means}

def quantize_with_threshold(dct_block, quality):
    """
    Real compression: threshold small coefficients to zero
    """
    # Calculate threshold based on quality and block energy
    block_energy = np.sum(np.abs(dct_block))
    threshold = (100 - quality) / 100.0 * block_energy * 0.01  # Adaptive threshold
    
    # Zero out coefficients below threshold
    compressed_block = np.where(np.abs(dct_block) > threshold, dct_block, 0)
    
    return compressed_block

def calculate_actual_compressed_size(compressed_channels):
    """
    Calculate ACTUAL compressed size (not theoretical)
    """
    total_size = 0
    
    for channel_name in ['R', 'G', 'B']:
        if channel_name in compressed_channels:
            channel_info = compressed_channels[channel_name]
            
            for block_info in channel_info['compressed_blocks']:
                if 'sparse_data' in block_info:
                    # Sparse storage: values (2 bytes each) + indices (2 bytes each) + count (4 bytes)
                    sparse_data = block_info['sparse_data']
                    total_size += sparse_data['count'] * (2 + 2) + 4  # 2 bytes per value + 2 bytes per index
                else:
                    # Full block storage: 64 coefficients * 2 bytes each
                    total_size += 64 * 2
                
                # Add position storage (2 * 2 bytes)
                total_size += 4
    
    return total_size

def calculate_original_size(segment_mask, segment_content):
    """
    Calculate original size of the segment
    """
    segment_pixels = np.sum(segment_mask)
    return segment_pixels * 3  # 3 bytes per pixel (RGB)

def decompress_segment_efficient(compressed_data, segment_mask, quality=75):
    """
    Decompress from efficient sparse representation
    """
    reconstructed_rgb = np.zeros((segment_mask.shape[0], segment_mask.shape[1], 3), dtype=np.float32)
    
    for channel_idx, channel_name in enumerate(['R', 'G', 'B']):
        if channel_name in compressed_data:
            channel_info = compressed_data[channel_name]
            compressed_blocks = channel_info['compressed_blocks']
            min_row, min_col, seg_height, seg_width = channel_info['bbox']
            pad_bottom, pad_right = channel_info['padding']
            block_size = channel_info.get('block_size', 8)
            original_mean = channel_info['original_mean']
            
            padded_height = seg_height + pad_bottom
            padded_width = seg_width + pad_right
            reconstructed_channel = np.zeros((padded_height, padded_width), dtype=np.float32)
            
            for block_info in compressed_blocks:
                i, j = block_info['position']
                
                if 'sparse_data' in block_info:
                    # Reconstruct from sparse representation
                    sparse_data = block_info['sparse_data']
                    block = np.zeros((block_size, block_size), dtype=np.float32)
                    
                    # Reconstruct indices
                    indices = np.unravel_index(sparse_data['indices'], (block_size, block_size))
                    block[indices] = sparse_data['values']
                    
                else:
                    # Full block
                    block = block_info['full_block']
                
                # Place in reconstructed image
                if i+block_size <= padded_height and j+block_size <= padded_width:
                    reconstructed_channel[i:i+block_size, j:j+block_size] += block
            
            # Remove padding and brightness correction
            original_segment = reconstructed_channel[:seg_height, :seg_width]
            reconstructed_rgb[min_row:min_row+seg_height, min_col:min_col+seg_width, channel_idx] = original_segment
    
    # Final conversion
    reconstructed_rgb = np.clip(reconstructed_rgb, 0, 1)
    reconstructed_uint8 = (reconstructed_rgb * 255).astype(np.uint8)
    
    final_reconstruction = np.zeros_like(reconstructed_uint8)
    final_reconstruction[segment_mask] = reconstructed_uint8[segment_mask]
    
    return final_reconstruction


















































import numpy as np
import scipy.fft

def compress_segment_simple_dct(segment_mask, region_image, quality=50):
    """
    SIMPLE, WORKING DCT compression with proper brightness preservation
    """
    rows, cols = np.where(segment_mask)
    if len(rows) == 0:
        return None, None
    
    # Get bounding box
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)
    
    # Extract segment
    segment_content = region_image[min_row:max_row+1, min_col:max_col+1].copy()
    segment_mask_cropped = segment_mask[min_row:max_row+1, min_col:max_col+1]
    
    # Convert to float (0-1 range)
    segment_float = segment_content.astype(np.float32) / 255.0
    
    # Store original brightness
    original_brightness = []
    for c in range(3):
        channel_data = segment_float[:, :, c]
        mask_area = channel_data[segment_mask_cropped]
        if len(mask_area) > 0:
            original_brightness.append(np.mean(mask_area))
        else:
            original_brightness.append(0.5)  # Default
    
    # Use 8x8 blocks (standard JPEG size)
    block_size = 8
    height, width = segment_content.shape[:2]
    
    # Pad to multiple of block_size
    pad_h = (block_size - (height % block_size)) % block_size
    pad_w = (block_size - (width % block_size)) % block_size
    
    compressed_data = {}
    
    for channel_idx in range(3):
        channel = segment_float[:, :, channel_idx]
        
        # Pad with edge values, NOT zeros
        padded = np.pad(channel, ((0, pad_h), (0, pad_w)), mode='edge')
        h_pad, w_pad = padded.shape
        
        # Create empty array for DCT coefficients
        dct_coeffs = np.zeros_like(padded)
        
        # Standard JPEG quantization table (8x8)
        quant_table = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=np.float32)
        
        # Adjust quality (50 = standard, 100 = best, 1 = worst)
        if quality <= 0:
            quality = 1
        if quality > 100:
            quality = 100
            
        # Scale factor: quality 50 = scale 1.0, quality 100 = scale 0.01, quality 1 = scale 100
        if quality >= 50:
            scale = (100 - quality) / 50.0
        else:
            scale = 50.0 / quality
        
        # Apply scale (but never let it go below 0.01)
        scaled_table = quant_table * max(scale, 0.01)
        
        # Process each 8x8 block
        compressed_blocks = []
        block_positions = []
        
        for i in range(0, h_pad, block_size):
            for j in range(0, w_pad, block_size):
                block = padded[i:i+block_size, j:j+block_size]
                
                # Check if block contains any mask area
                block_mask_area = segment_mask_cropped[
                    i:min(i+block_size, height), 
                    j:min(j+block_size, width)
                ]
                
                if np.any(block_mask_area) and block.shape == (block_size, block_size):
                    # Apply DCT
                    dct_block = scipy.fft.dctn(block, norm='ortho')
                    
                    # QUANTIZE (divide by numbers ≥ 1)
                    quantized = np.round(dct_block / scaled_table)
                    
                    compressed_blocks.append(quantized.flatten())
                    block_positions.append((i, j))
        
        compressed_data[channel_idx] = {
            'blocks': compressed_blocks,
            'positions': block_positions,
            'quant_table': scaled_table,
            'original_shape': (h_pad, w_pad),
            'block_size': block_size,
            'original_brightness': original_brightness[channel_idx]
        }
    
    return compressed_data, {
        'original_means': original_brightness,
        'segment_bbox': (min_row, min_col, height, width),
        'segment_mask_shape': segment_mask_cropped.shape
    }


def decompress_segment_simple_dct(compressed_data, metadata):
    """
    Decompress the DCT-compressed segment
    """
    if compressed_data is None:
        return None
    
    height, width = metadata['segment_bbox'][2:]
    
    reconstructed = np.zeros((height, width, 3), dtype=np.float32)
    
    for channel_idx in range(3):
        if channel_idx not in compressed_data:
            continue
            
        channel_info = compressed_data[channel_idx]
        blocks = channel_info['blocks']
        positions = channel_info['positions']
        quant_table = channel_info['quant_table']
        h_pad, w_pad = channel_info['original_shape']
        block_size = channel_info['block_size']
        
        # Create padded reconstruction
        padded_recon = np.zeros((h_pad, w_pad), dtype=np.float32)
        
        for idx, (i, j) in enumerate(positions):
            if idx < len(blocks):
                # Reshape flattened block
                quantized_flat = blocks[idx]
                quantized_block = quantized_flat.reshape((block_size, block_size))
                
                # Dequantize (MULTIPLY by quantization table)
                dct_block = quantized_block * quant_table
                
                # Inverse DCT
                recon_block = scipy.fft.idctn(dct_block, norm='ortho')
                
                # Place in padded reconstruction
                padded_recon[i:i+block_size, j:j+block_size] = recon_block
        
        # Crop to original size
        channel_recon = padded_recon[:height, :width]
        
        # Adjust brightness to match original
        current_mean = np.mean(channel_recon)
        target_mean = channel_info['original_brightness']
        if current_mean > 0:
            channel_recon = channel_recon * (target_mean / current_mean)
        
        # Clip to valid range
        channel_recon = np.clip(channel_recon, 0, 1)
        reconstructed[:, :, channel_idx] = channel_recon
    
    # Convert back to 0-255 uint8
    reconstructed_uint8 = (reconstructed * 255).astype(np.uint8)
    
    return reconstructed_uint8






































































import numpy as np
import scipy.fft
import math

"""def compress_segment_with_dct_fixed(segment_mask, region_image, quality=50):

    # Get the coordinates where segment_mask is True
    rows, cols = np.where(segment_mask)
    if len(rows) == 0:
        return None, None
    
    # Get bounding box of this segment
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)
    
    # Extract the segment from the region image
    segment_content = region_image[min_row:max_row+1, min_col:max_col+1].copy()
    segment_mask_cropped = segment_mask[min_row:max_row+1, min_col:max_col+1]
    
    # Check if we have a valid segment
    height, width = segment_content.shape[:2]
    if height == 0 or width == 0:
        return None, None
    
    # Store original brightness for brightness preservation
    original_brightness = []
    for c in range(3):
        channel_data = segment_content[:, :, c]
        # Get mean brightness only in the masked area
        mask_area = channel_data[segment_mask_cropped]
        if len(mask_area) > 0:
            original_brightness.append(np.mean(mask_area))
        else:
            original_brightness.append(128.0)  # Default mid-gray
    
    # Convert to float (0-1 range) for DCT
    segment_float = segment_content.astype(np.float32) / 255.0
    
    # Use standard JPEG block size
    block_size = 8
    
    # Pad to multiple of block_size (use edge padding to avoid dark borders)
    pad_h = (block_size - (height % block_size)) % block_size
    pad_w = (block_size - (width % block_size)) % block_size
    
    compressed_channels = {}
    
    for channel_idx in range(3):
        channel_data = segment_float[:, :, channel_idx]
        
        # Pad with edge values (not zeros!)
        padded_channel = np.pad(channel_data, ((0, pad_h), (0, pad_w)), mode='edge')
        h_pad, w_pad = padded_channel.shape
        
        # Standard JPEG quantization table (8x8)
        std_quant_table = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=np.float32)
        
        # Adjust quantization table based on quality (0-100)
        # quality=50 uses standard table, quality=100 uses gentle table, quality=1 uses aggressive table
        if quality <= 0:
            quality = 1
        if quality > 100:
            quality = 100
            
        # Calculate scale factor
        if quality >= 50:
            scale_factor = (100 - quality) / 50.0  # 50→1.0, 75→0.5, 100→0.0
        else:
            scale_factor = 50.0 / quality  # 25→2.0, 10→5.0, 1→50.0
        
        # Apply scale but keep minimum of 0.01
        scale_factor = max(scale_factor, 0.01)
        quant_table = std_quant_table * scale_factor
        
        # Process each 8x8 block
        compressed_blocks = []
        block_positions = []
        
        for i in range(0, h_pad, block_size):
            for j in range(0, w_pad, block_size):
                block = padded_channel[i:i+block_size, j:j+block_size]
                
                # Check if this block contains any of our segment mask
                # We need to check the corresponding area in the original mask
                block_mask_area = segment_mask_cropped[
                    i:min(i+block_size, height), 
                    j:min(j+block_size, width)
                ]
                
                if np.any(block_mask_area) and block.shape == (block_size, block_size):
                    # Apply DCT
                    dct_block = scipy.fft.dctn(block, norm='ortho')
                    
                    # QUANTIZE (divide by numbers ≥ 1)
                    quantized_block = np.round(dct_block / quant_table)
                    
                    compressed_blocks.append(quantized_block.flatten())
                    block_positions.append((i, j))
        
        # Store compressed data for this channel
        compressed_channels[channel_idx] = {
            'compressed_blocks': compressed_blocks,
            'block_positions': block_positions,
            'quantization_table': quant_table,
            'padded_shape': (h_pad, w_pad),
            'block_size': block_size,
            'original_brightness': original_brightness[channel_idx]
        }
    
    # Metadata for reconstruction
    metadata = {
        'original_means': original_brightness,
        'segment_bbox': (min_row, min_col, height, width),
        'segment_mask_shape': segment_mask_cropped.shape,
        'padding': (pad_h, pad_w)
    }
    
    return compressed_channels, metadata
"""

def decompress_segment_dct(compressed_channels, metadata, original_region_shape):
    """
    Decompress a DCT-compressed segment
    Returns the reconstructed segment in its original position within the region
    """
    if compressed_channels is None:
        return None
    
    # Get the original segment's bounding box and shape
    min_row, min_col, height, width = metadata['segment_bbox']
    
    # Initialize reconstruction at the correct size
    reconstructed = np.zeros((height, width, 3), dtype=np.float32)
    
    for channel_idx in range(3):
        if channel_idx not in compressed_channels:
            continue
            
        channel_info = compressed_channels[channel_idx]
        blocks = channel_info['compressed_blocks']
        positions = channel_info['block_positions']
        quant_table = channel_info['quantization_table']
        h_pad, w_pad = channel_info['padded_shape']
        block_size = channel_info['block_size']
        target_brightness = channel_info['original_brightness']
        
        # Create padded reconstruction space
        padded_recon = np.zeros((h_pad, w_pad), dtype=np.float32)
        
        for block_idx, (i, j) in enumerate(positions):
            if block_idx < len(blocks):
                # Get the quantized block data
                quantized_flat = blocks[block_idx]
                quantized_block = quantized_flat.reshape((block_size, block_size))
                
                # DEQUANTIZE (multiply by quantization table)
                dct_block = quantized_block * quant_table
                
                # Inverse DCT
                recon_block = scipy.fft.idctn(dct_block, norm='ortho')
                
                # Place in padded reconstruction
                if i + block_size <= h_pad and j + block_size <= w_pad:
                    padded_recon[i:i+block_size, j:j+block_size] = recon_block
        
        # Crop to original size
        channel_recon = padded_recon[:height, :width]
        
        # Simple brightness adjustment (optional)
        # You can comment this out if it causes issues
        current_mean = np.mean(channel_recon)
        if current_mean > 0 and target_brightness > 0:
            # target_brightness is stored as 0-255 range, convert to 0-1
            target_norm = target_brightness / 255.0
            channel_recon = channel_recon * (target_norm / current_mean)
        
        # Clip to valid range
        channel_recon = np.clip(channel_recon, 0, 1)
        reconstructed[:, :, channel_idx] = channel_recon
    
    # Convert back to 0-255 uint8
    reconstructed_uint8 = (reconstructed * 255).astype(np.uint8)
    
    return reconstructed_uint8


def calculate_psnr_for_segment(original_region, recon_segment, segment_mask, segment_bbox):
    """
    Calculate PSNR for a specific segment
    """
    min_row, min_col, height, width = segment_bbox
    
    # Extract the segment area from original region
    original_segment = original_region[min_row:min_row+height, min_col:min_col+width]
    
    # Get mask for just this segment area
    segment_mask_cropped = segment_mask[min_row:min_row+height, min_col:min_col+width]
    
    # Get pixels from both images
    original_pixels = original_segment[segment_mask_cropped]
    recon_pixels = recon_segment[segment_mask_cropped]
    
    if len(original_pixels) == 0 or len(recon_pixels) == 0:
        return 0
    
    # Calculate MSE and PSNR
    mse = np.mean((original_pixels - recon_pixels) ** 2)
    if mse > 0:
        psnr = 10 * np.log10(255**2 / mse)
    else:
        psnr = float('inf')
    
    return psnr


"""def calculate_compression_statistics(compressed_channels, original_size_bytes):

    if compressed_channels is None:
        return 0, 0, 1.0
    
    compressed_size_bytes = 0
    original_pixels = 0
    
    for channel_idx in range(3):
        if channel_idx in compressed_channels:
            blocks = compressed_channels[channel_idx]['compressed_blocks']
            for block in blocks:
                compressed_size_bytes += block.nbytes
    
    # Calculate compression ratio
    if compressed_size_bytes > 0:
        compression_ratio = original_size_bytes / compressed_size_bytes
    else:
        compression_ratio = 1.0
    
    return compressed_size_bytes, original_size_bytes, compression_ratio
"""



import matplotlib.pyplot as plt

def visualize_roi_comparison(original_region, reconstructed_region, segment_masks, 
                            region_idx, quality, overall_ratio, avg_psnr):
    """
    Visualize comparison between original and reconstructed ROI
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Region {region_idx} - Quality: {quality} - Compression Ratio: {overall_ratio:.2f}:1 - Avg PSNR: {avg_psnr:.2f} dB', 
                 fontsize=14, fontweight='bold')
    
    # 1. Original ROI
    axes[0, 0].imshow(original_region)
    axes[0, 0].set_title('Original ROI Region')
    axes[0, 0].axis('off')
    
    # 2. Reconstructed ROI
    axes[0, 1].imshow(reconstructed_region)
    axes[0, 1].set_title('Reconstructed ROI Region')
    axes[0, 1].axis('off')
    
    # 3. Difference (error) image
    if original_region.shape == reconstructed_region.shape:
        diff = np.abs(original_region.astype(np.float32) - reconstructed_region.astype(np.float32))
        diff_normalized = (diff / 255.0) * 255
        axes[0, 2].imshow(diff_normalized.astype(np.uint8), cmap='hot')
        axes[0, 2].set_title(f'Difference (Max error: {np.max(diff):.1f})')
        axes[0, 2].axis('off')
    
    # 4. Original with segment boundaries
    axes[1, 0].imshow(original_region)
    if segment_masks and len(segment_masks) > 0:
        # Create overlay of segment boundaries
        boundaries = np.zeros(original_region.shape[:2], dtype=bool)
        for mask in segment_masks:
            # Find edges of each segment
            from scipy import ndimage
            eroded = ndimage.binary_erosion(mask, structure=np.ones((3,3)))
            boundary = mask & ~eroded
            boundaries = boundaries | boundary
        
        # Plot boundaries in red
        y_coords, x_coords = np.where(boundaries)
        axes[1, 0].scatter(x_coords, y_coords, s=1, c='red', alpha=0.6)
    axes[1, 0].set_title(f'Original with {len(segment_masks)} segments')
    axes[1, 0].axis('off')
    
    # 5. Close-up comparison (center crop)
    h, w = original_region.shape[:2]
    crop_size = min(100, h//3, w//3)
    center_y, center_x = h//2, w//2
    
    if crop_size > 0:
        crop_y1 = max(0, center_y - crop_size//2)
        crop_y2 = min(h, center_y + crop_size//2)
        crop_x1 = max(0, center_x - crop_size//2)
        crop_x2 = min(w, center_x + crop_size//2)
        
        original_crop = original_region[crop_y1:crop_y2, crop_x1:crop_x2]
        recon_crop = reconstructed_region[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # Create side-by-side comparison
        comparison = np.hstack([original_crop, recon_crop])
        axes[1, 1].imshow(comparison)
        axes[1, 1].set_title(f'Close-up Comparison ({crop_size}x{crop_size})')
        axes[1, 1].axis('off')
        
        # Add dividing line
        divider_x = original_crop.shape[1]
        axes[1, 1].axvline(x=divider_x, color='white', linewidth=2)
        axes[1, 1].text(divider_x//2, -10, 'Original', ha='center', color='white', 
                       fontweight='bold', backgroundcolor='black')
        axes[1, 1].text(divider_x + original_crop.shape[1]//2, -10, 'Reconstructed', 
                       ha='center', color='white', fontweight='bold', backgroundcolor='black')
    
    """# 6. PSNR histogram for segments
    if 'segment_psnrs' in locals():
        axes[1, 2].hist(segment_psnrs, bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[1, 2].axvline(avg_psnr, color='red', linestyle='--', linewidth=2, label=f'Avg: {avg_psnr:.2f} dB')
        axes[1, 2].set_xlabel('PSNR (dB)')
        axes[1, 2].set_ylabel('Number of Segments')
        axes[1, 2].set_title('Segment Quality Distribution')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)"""
    
    plt.tight_layout()
    plt.show()
    
    # Also show single segment comparison if available
    if segment_masks and len(segment_masks) > 0:
        show_single_segment_comparison(original_region, reconstructed_region, segment_masks[0])

def show_single_segment_comparison(original_region, reconstructed_region, segment_mask):
    """
    Show detailed comparison for a single segment
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Get bounding box of the segment
    rows, cols = np.where(segment_mask)
    if len(rows) == 0:
        return
    
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)
    height = max_row - min_row + 1
    width = max_col - min_col + 1
    
    # Extract the segment from both images
    orig_segment = original_region[min_row:max_row+1, min_col:max_col+1]
    recon_segment = reconstructed_region[min_row:max_row+1, min_col:max_col+1]
    
    # Mask for just this segment
    segment_mask_cropped = segment_mask[min_row:max_row+1, min_col:max_col+1]
    
    # 1. Original segment
    axes[0].imshow(orig_segment)
    axes[0].set_title(f'Original Segment\n{height}x{width} pixels')
    axes[0].axis('off')
    
    # 2. Reconstructed segment
    axes[1].imshow(recon_segment)
    axes[1].set_title('Reconstructed Segment')
    axes[1].axis('off')
    
    # 3. Difference with mask overlay
    if orig_segment.shape == recon_segment.shape:
        diff = np.abs(orig_segment.astype(np.float32) - recon_segment.astype(np.float32))
        diff_rgb = np.zeros_like(orig_segment, dtype=np.uint8)
        
        # Create red overlay for errors
        for c in range(3):
            diff_channel = diff[:, :, c]
            # Normalize to 0-255
            if np.max(diff_channel) > 0:
                diff_channel = (diff_channel / np.max(diff_channel)) * 255
            
            # Create overlay: error in red channel, original in green/blue
            diff_rgb[:, :, 0] = np.minimum(255, diff_channel).astype(np.uint8)
            diff_rgb[:, :, 1] = orig_segment[:, :, 1]
            diff_rgb[:, :, 2] = orig_segment[:, :, 2]
        
        axes[2].imshow(diff_rgb)
        axes[2].set_title(f'Error Overlay (Red=error)\nMax error: {np.max(diff):.1f}')
        axes[2].axis('off')
    
    # 4. Segment mask
    mask_display = np.zeros((height, width, 3), dtype=np.uint8)
    mask_display[segment_mask_cropped] = [255, 255, 255]  # White for mask
    mask_display[~segment_mask_cropped] = [0, 0, 0]       # Black for non-mask
    
    axes[3].imshow(mask_display)
    axes[3].set_title(f'Segment Mask\n{np.sum(segment_mask_cropped)} pixels')
    axes[3].axis('off')
    
    plt.suptitle(f'Single Segment Detail View', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()














def get_quantization_table_simple(quality, block_size=8):
    """
    SIMPLE quantization that GUARANTEES non-zero coefficients
    """
    if block_size == 8:
        # GENTLE table for small segments
        quant_table = np.array([
            [0.1, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 4.0],
            [0.1, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 4.0],
            [0.2, 0.2, 0.3, 0.5, 1.0, 2.0, 4.0, 8.0],
            [0.3, 0.3, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
            [0.5, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
            [1.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0],
            [2.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0],
            [4.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]
        ], dtype=np.float32)
    else:  # 4x4
        quant_table = np.array([
            [0.1, 0.1, 0.2, 0.5],
            [0.1, 0.1, 0.2, 0.5],
            [0.2, 0.2, 0.5, 1.0],
            [0.5, 0.5, 1.0, 2.0]
        ], dtype=np.float32)
    
    # Quality adjustment (minimal)
    if quality > 80:
        # Almost no additional scaling
        scale = 0.8
    elif quality > 60:
        scale = 1.0
    elif quality > 40:
        scale = 1.2
    elif quality > 20:
        scale = 1.5
    else:
        scale = 2.0

    """if quality>0 and quality<=100: scale=quality/100
    else: scale=1"""
    
    quant_table = quant_table * scale
    
    print(f"    [QUANT] quality={quality}, scale={scale}, "
          f"DC={quant_table[0,0]:.3f}, max={np.max(quant_table):.1f}")
    
    return quant_table


def compress_segment_guaranteed_working(segment_mask, region_image, quality=75):
    """
    DCT compression that GUARANTEES non-black output
    """
    rows, cols = np.where(segment_mask)
    if len(rows) == 0:
        return None, None
    
    # Get bounding box
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)
    height = max_row - min_row + 1
    width = max_col - min_col + 1
    
    print(f"    [INFO] Segment: {width}x{height} pixels")
    
    # Extract segment
    segment_content = region_image[min_row:max_row+1, min_col:max_col+1].copy()
    segment_mask_cropped = segment_mask[min_row:max_row+1, min_col:max_col+1]
    
    # Store original brightness
    original_brightness = []
    for c in range(3):
        channel_data = segment_content[:, :, c]
        mask_area = channel_data[segment_mask_cropped]
        if len(mask_area) > 0:
            original_brightness.append(np.mean(mask_area))
        else:
            original_brightness.append(128.0)
    
    print(f"    [INFO] Brightness: R={original_brightness[0]:.1f}, "
          f"G={original_brightness[1]:.1f}, B={original_brightness[2]:.1f}")
    
    # Convert to float (0-1)
    segment_float = segment_content.astype(np.float32) / 255.0
    
    # Force 4x4 blocks for small segments
    if width < 16 or height < 16:
        block_size = 4
        print(f"    [INFO] Using 4x4 blocks (segment too small for 8x8)")
    else:
        block_size = 8
    
    # Pad
    pad_h = (block_size - (height % block_size)) % block_size
    pad_w = (block_size - (width % block_size)) % block_size
    
    compressed_channels = {}
    
    for channel_idx in range(3):
        channel_data = segment_float[:, :, channel_idx]
        
        # Check range
        mask_values = channel_data[segment_mask_cropped]
        if len(mask_values) > 0:
            ch_min, ch_max = np.min(mask_values), np.max(mask_values)
            print(f"    [CH{channel_idx}] Range: [{ch_min:.3f}, {ch_max:.3f}]")
        
        # Pad
        padded = np.pad(channel_data, ((0, pad_h), (0, pad_w)), mode='edge')
        
        # Get GENTLE quantization table
        quant_table = get_quantization_table_simple(quality, block_size)
        
        # Process blocks
        compressed_blocks = []
        block_positions = []
        
        for i in range(0, padded.shape[0], block_size):
            for j in range(0, padded.shape[1], block_size):
                block = padded[i:i+block_size, j:j+block_size]
                
                # Check if block contains mask
                block_mask_area = segment_mask_cropped[
                    i:min(i+block_size, height), 
                    j:min(j+block_size, width)
                ]
                
                if np.any(block_mask_area) and block.shape == (block_size, block_size):
                    # Apply DCT
                    dct_block = scipy.fft.dctn(block, norm='ortho')
                    
                    # DEBUG: Print DC coefficient (brightness)
                    dc_value = dct_block[0, 0]
                    
                    # GENTLE quantization
                    quantized_block = np.round(dct_block / quant_table)
                    
                    # FORCE DC coefficient to be non-zero
                    if quantized_block[0, 0] == 0 and dc_value != 0:
                        print(f"    [WARN] Channel {channel_idx} block ({i},{j}): "
                              f"DC={dc_value:.3f} was quantized to 0! Keeping as 1.")
                        quantized_block[0, 0] = 1 if dc_value > 0 else -1
                    
                    # Count non-zero
                    non_zero = np.sum(quantized_block != 0)
                    if non_zero == 0:
                        print(f"    [ERROR] All zeros in block! Adding DC=1")
                        quantized_block[0, 0] = 1
                    
                    # Store
                    compressed_blocks.append(quantized_block.astype(np.int16).flatten())
                    block_positions.append((i, j))
        
        if compressed_blocks:
            compressed_channels[channel_idx] = {
                'compressed_blocks': compressed_blocks,
                'block_positions': block_positions,
                'quantization_table': quant_table,
                'padded_shape': padded.shape,
                'block_size': block_size,
                'original_brightness': original_brightness[channel_idx]
            }
    
    if not compressed_channels:
        return None, None
    
    metadata = {
        'original_means': original_brightness,
        'segment_bbox': (min_row, min_col, height, width),
        'segment_mask_shape': segment_mask_cropped.shape,
        'padding': (pad_h, pad_w),
        'block_size': block_size
    }
    
    return compressed_channels, metadata


def compress_segment_with_dct_fixed(segment_mask, region_image, quality=50):
    """
    FIXED DCT compression with DEBUG output
    """
    rows, cols = np.where(segment_mask)
    if len(rows) == 0:
        return None, None
    
    # Get bounding box
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)
    height = max_row - min_row + 1
    width = max_col - min_col + 1
    
    print(f"    Segment size: {width}x{height} pixels")
    
    # Extract segment
    segment_content = region_image[min_row:max_row+1, min_col:max_col+1].copy()
    segment_mask_cropped = segment_mask[min_row:max_row+1, min_col:max_col+1]
    
    # Store original brightness
    original_brightness = []
    for c in range(3):
        channel_data = segment_content[:, :, c]
        mask_area = channel_data[segment_mask_cropped]
        if len(mask_area) > 0:
            original_brightness.append(np.mean(mask_area))
        else:
            original_brightness.append(128.0)
    
    print(f"    Original brightness: R={original_brightness[0]:.1f}, "
          f"G={original_brightness[1]:.1f}, B={original_brightness[2]:.1f}")
    
    # Convert to float (0-1)
    segment_float = segment_content.astype(np.float32) / 255.0
    
    # Use block size based on segment size
    if min(height, width) < 16:
        block_size = 4
    else:
        block_size = 8
    
    print(f"    Using block size: {block_size}x{block_size}")
    
    # Pad to multiple of block_size
    pad_h = (block_size - (height % block_size)) % block_size
    pad_w = (block_size - (width % block_size)) % block_size
    
    compressed_channels = {}
    total_non_zero = 0
    total_coeffs = 0
    
    for channel_idx in range(3):
        channel_data = segment_float[:, :, channel_idx]
        
        # Check channel range
        mask_values = channel_data[segment_mask_cropped]
        if len(mask_values) > 0:
            print(f"    Channel {channel_idx}: min={np.min(mask_values):.4f}, "
                  f"max={np.max(mask_values):.4f}, mean={np.mean(mask_values):.4f}")
        
        # Pad with edge values
        padded_channel = np.pad(channel_data, ((0, pad_h), (0, pad_w)), mode='edge')
        h_pad, w_pad = padded_channel.shape
        
        # Get quantization table
        quant_table = get_quantization_table(quality, block_size)
        
        # Process blocks
        compressed_blocks = []
        block_positions = []
        channel_non_zero = 0
        channel_coeffs = 0
        
        for i in range(0, h_pad, block_size):
            for j in range(0, w_pad, block_size):
                block = padded_channel[i:i+block_size, j:j+block_size]
                
                # Check if block contains mask
                block_mask_area = segment_mask_cropped[
                    i:min(i+block_size, height), 
                    j:min(j+block_size, width)
                ]
                
                if np.any(block_mask_area) and block.shape == (block_size, block_size):
                    # Apply DCT
                    dct_block = scipy.fft.dctn(block, norm='ortho')
                    
                    # Debug DCT range
                    dct_min, dct_max = np.min(dct_block), np.max(dct_block)
                    dct_abs_max = np.max(np.abs(dct_block))
                    
                    # Quantize
                    quantized_block = np.round(dct_block / quant_table)
                    
                    # Count non-zero coefficients
                    non_zero = np.sum(quantized_block != 0)
                    channel_non_zero += non_zero
                    channel_coeffs += block_size * block_size
                    
                    if non_zero == 0:
                        print(f"      WARNING: Block ({i},{j}) - ALL coefficients quantized to zero!")
                        print(f"        DCT range: [{dct_min:.4f}, {dct_max:.4f}], max abs: {dct_abs_max:.4f}")
                        print(f"        DC coefficient: {dct_block[0,0]:.4f}, "
                              f"quantized to: {quantized_block[0,0]}")
                    
                    # Convert to int16
                    quantized_block_int16 = quantized_block.astype(np.int16)
                    
                    compressed_blocks.append(quantized_block_int16.flatten())
                    block_positions.append((i, j))
        
        if compressed_blocks:
            compressed_channels[channel_idx] = {
                'compressed_blocks': compressed_blocks,
                'block_positions': block_positions,
                'quantization_table': quant_table,
                'padded_shape': (h_pad, w_pad),
                'block_size': block_size,
                'original_brightness': original_brightness[channel_idx]
            }
            
            print(f"    Channel {channel_idx}: {len(compressed_blocks)} blocks, "
                  f"{channel_non_zero}/{channel_coeffs} non-zero coefficients "
                  f"({channel_non_zero/channel_coeffs*100:.1f}%)")
            
            total_non_zero += channel_non_zero
            total_coeffs += channel_coeffs
    
    if not compressed_channels:
        print("    ERROR: No blocks were compressed!")
        return None, None
    
    print(f"    TOTAL: {total_non_zero}/{total_coeffs} non-zero coefficients "
          f"({total_non_zero/total_coeffs*100:.1f}%)")
    
    if total_non_zero == 0:
        print("    ⚠️ CRITICAL: ALL coefficients are zero! Image will be black!")
    
    metadata = {
        'original_means': original_brightness,
        'segment_bbox': (min_row, min_col, height, width),
        'segment_mask_shape': segment_mask_cropped.shape,
        'padding': (pad_h, pad_w),
        'block_size': block_size
    }
    
    return compressed_channels, metadata



def calculate_psnr_for_region(original_region, recon_region, region_mask, region_bbox):
    """
    Calculate PSNR for an entire ROI region
    """
    # Get pixels from both images within the mask
    original_pixels = original_region[region_mask]
    recon_pixels = recon_region[region_mask]
    
    if len(original_pixels) == 0 or len(recon_pixels) == 0:
        return 0
    
    # Calculate MSE and PSNR
    mse = np.mean((original_pixels - recon_pixels) ** 2)
    if mse > 0:
        psnr = 10 * np.log10(255**2 / mse)
    else:
        psnr = float('inf')
    
    return psnr














    
























































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


def visualize_roi_comparison_correct(original_region, reconstructed_region, roi_mask, region_idx, ratio, psnr, method='content_only'):
    """
    CORRECT visualization - shows ONLY ROI area comparison
    """
    # Create mask for displaying only ROI area
    roi_display = np.zeros_like(original_region)
    for c in range(3):
        roi_display[:, :, c] = roi_mask * 255  # White mask
    
    # Extract ONLY the ROI area from both images
    original_roi_only = original_region.copy()
    recon_roi_only = reconstructed_region.copy()
    
    # Set non-ROI areas to gray for visualization
    gray_bg = np.array([128, 128, 128], dtype=np.uint8)
    
    for c in range(3):
        original_roi_only[:, :, c] = np.where(roi_mask, original_region[:, :, c], gray_bg[c])
        recon_roi_only[:, :, c] = np.where(roi_mask, reconstructed_region[:, :, c], gray_bg[c])
    
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # 1. Original ROI only (with gray background)
    axes[0].imshow(original_roi_only)
    axes[0].set_title(f'Original ROI Area\n{np.sum(roi_mask):,} pixels')
    axes[0].axis('off')
    
    # 2. Reconstructed ROI only (with gray background)
    axes[1].imshow(recon_roi_only)
    axes[1].set_title(f'Reconstructed ({method})\nRatio: {ratio:.2f}:1')
    axes[1].axis('off')
    
    # 3. Side-by-side ROI comparison
    # Create a composite showing both
    h, w = original_region.shape[:2]
    composite = np.ones((h, w*2 + 20, 3), dtype=np.uint8) * 200  # Light gray background
    
    # Place original on left
    composite[:, :w] = original_roi_only
    # Place reconstructed on right
    composite[:, w+20:] = recon_roi_only
    
    axes[2].imshow(composite)
    axes[2].set_title(f'Side-by-side Comparison\nPSNR: {psnr:.2f} dB')
    axes[2].axis('off')
    
    # Add dividing line to composite
    axes[2].axvline(x=w, color='black', linewidth=2, alpha=0.5)
    
    # 4. Difference (error) map ONLY in ROI area
    diff = np.abs(original_region.astype(np.float32) - reconstructed_region.astype(np.float32))
    diff_rgb = np.zeros_like(original_region, dtype=np.uint8)
    
    # Create heatmap: red = error, original image = background
    for c in range(3):
        diff_channel = diff[:, :, c]
        # Apply mask: only show errors in ROI area
        diff_channel = diff_channel * roi_mask.astype(np.float32)
        
        # Normalize for display
        if np.max(diff_channel) > 0:
            diff_channel = (diff_channel / np.max(diff_channel)) * 255
        
        # Overlay: error in red, original in green/blue
        diff_rgb[:, :, 0] = np.minimum(255, diff_channel).astype(np.uint8)
        diff_rgb[:, :, 1] = original_region[:, :, 1]
        diff_rgb[:, :, 2] = original_region[:, :, 2]
    
    axes[3].imshow(diff_rgb)
    max_error = np.max(diff[roi_mask]) if np.any(roi_mask) else 0
    axes[3].set_title(f'Error Overlay (Red=error)\nMax error in ROI: {max_error:.1f}')
    axes[3].axis('off')
    
    plt.suptitle(f'ROI Region {region_idx} - Correct Comparison (ROI Area Only)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
























































# ==============================================
# NEW FUNCTIONS FOR HOMOGENEOUS REGION COMPRESSION
# ==============================================

def analyze_region_homogeneity(region_image, region_mask):
    """
    Analyze how homogeneous a region is by checking color variance.
    Returns homogeneity score (0-1, where 1 is perfectly homogeneous)
    and mean color.
    """
    # Extract only the masked pixels
    rows, cols = np.where(region_mask)
    masked_pixels = region_image[rows, cols]
    
    if len(masked_pixels) == 0:
        return 0.0, np.array([0, 0, 0])
    
    # Calculate color variance for each channel
    mean_color = np.mean(masked_pixels, axis=0)
    variance = np.var(masked_pixels, axis=0)
    
    # Normalize variance to 0-1 range (assuming 0-255 pixel values)
    max_variance = 255 * 255  # Max possible variance
    normalized_variance = np.mean(variance) / max_variance
    
    # Homogeneity score = 1 - normalized variance
    homogeneity_score = 1.0 - normalized_variance
    
    return homogeneity_score, mean_color.astype(np.uint8)


def compress_homogeneous_region_mean_color(region_image, region_mask, store_residual=True):
    """
    Fixed version: Properly handle mean color and residual.
    """
    # Get mean color - ensure it's calculated correctly
    _, mean_color = analyze_region_homogeneity(region_image, region_mask)
    
    # Get mask coordinates
    rows, cols = np.where(region_mask)
    num_pixels = len(rows)
    
    if num_pixels == 0:
        return None
    
    # Create compressed data structure
    compressed_data = {
        'method': 'mean_color',
        'mean_color': mean_color.astype(np.uint8),  # Ensure uint8
        'mask_rows': rows,
        'mask_cols': cols,
        'store_residual': store_residual,
        'original_size': num_pixels * 3,
        'shape': region_image.shape
    }
    
    # Calculate residual if requested
    if store_residual:
        # Reconstruct mean image
        mean_image = np.zeros_like(region_image)
        mean_image[rows, cols] = mean_color
        
        # Calculate residual (difference from original)
        residual = region_image.astype(np.int16) - mean_image.astype(np.int16)
        residual_values = residual[rows, cols]
        
        # Quantize residual to save space but maintain quality
        # Use 2-bit quantization (values: -64, 0, 64) for significant compression
        residual_quantized = np.clip(np.round(residual_values / 64), -2, 2).astype(np.int8)
        
        compressed_data['residual'] = residual_quantized
        compressed_data['quantization_step'] = 64
    
    return compressed_data


def reconstruct_homogeneous_region_mean_color(compressed_data, image_shape):
    """
    Fixed reconstruction: Properly apply mean color and residual.
    """
    shape = compressed_data['shape']
    mean_color = compressed_data['mean_color']
    rows = compressed_data['mask_rows']
    cols = compressed_data['mask_cols']
    
    # Create blank image
    reconstructed = np.zeros(shape, dtype=np.uint8)
    
    # Fill with mean color
    reconstructed[rows, cols] = mean_color
    
    # Add residual if available
    if compressed_data.get('store_residual', False) and 'residual' in compressed_data:
        residual_quantized = compressed_data['residual']
        quantization_step = compressed_data.get('quantization_step', 64)
        
        # Dequantize residual
        residual = residual_quantized.astype(np.float32) * quantization_step
        
        # Add to reconstruction and clip to valid range
        reconstructed_values = reconstructed[rows, cols].astype(np.float32) + residual
        reconstructed[rows, cols] = np.clip(reconstructed_values, 0, 255).astype(np.uint8)
    
    return reconstructed


def compress_subregion_smart(segment_image, segment_mask, homogeneity_threshold=0.85):
    """
    Smart subregion compression: use mean color for homogeneous regions,
    fall back to DCT for complex regions.
    """
    # Analyze homogeneity
    homogeneity_score, _ = analyze_region_homogeneity(segment_image, segment_mask)
    
    # Decide which compression method to use
    if homogeneity_score >= homogeneity_threshold:
        # Use mean color compression for homogeneous regions
        return compress_homogeneous_region_mean_color(
            segment_image, 
            segment_mask,
            store_residual=True  # Store residual for quality
        )
    else:
        # Fall back to standard DCT compression for complex regions
        return compress_roi_content_only(segment_mask, segment_image, quality=75)


def reconstruct_subregion_smart(compressed_data, image_shape, segment_mask):
    """
    Reconstruct from smart compression based on method used.
    """
    if compressed_data.get('method') == 'mean_color':
        return reconstruct_homogeneous_region_mean_color(compressed_data, image_shape)
    else:
        # Fall back to standard reconstruction
        return reconstruct_roi_content(
            compressed_data['compressed_coefficients'],
            image_shape,
            segment_mask
        )
    
























# ==============================================
# FIXED RECONSTRUCTION FUNCTIONS
# ==============================================
def compress_homogeneous_region_fixed(region_image, region_mask, store_residual=True):
    """
    Fixed compression for homogeneous regions.
    Returns compression data structure.
    """
    # Get mean color
    homogeneity_score, mean_color = analyze_region_homogeneity(region_image, region_mask)
    
    # Get mask coordinates
    rows, cols = np.where(region_mask)
    num_pixels = len(rows)
    
    if num_pixels == 0:
        return None
    
    # Create compressed data structure
    compressed_data = {
        'method': 'mean_color',
        'mean_color': mean_color.astype(np.uint8),
        'mask_rows': rows,
        'mask_cols': cols,
        'store_residual': store_residual,
        'original_size': num_pixels * 3,
        'image_shape': region_image.shape,
        'homogeneity_score': homogeneity_score
    }
    
    # Calculate residual if requested
    if store_residual:
        # Create mean image for this segment
        mean_image = np.zeros(region_image.shape, dtype=np.uint8)
        mean_image[rows, cols] = mean_color
        
        # Calculate residual (difference from original)
        residual = region_image.astype(np.int16) - mean_image.astype(np.int16)
        residual_values = residual[rows, cols]
        
        # Quantize residual (use 3-bit quantization: values from -4 to 3)
        # This gives us 3 bits per channel (instead of 8)
        residual_quantized = np.clip(np.round(residual_values / 32), -4, 3).astype(np.int8)
        
        compressed_data['residual'] = residual_quantized
        compressed_data['quantization_step'] = 32
        compressed_data['residual_range'] = (-4, 3)
    
    return compressed_data

def reconstruct_homogeneous_region_fixed(compressed_data):
    """
    Reconstruct image from homogeneous region compression data.
    Returns full reconstructed image.
    """
    # Get parameters from compressed data
    image_shape = compressed_data['image_shape']
    mean_color = compressed_data['mean_color']
    rows = compressed_data['mask_rows']
    cols = compressed_data['mask_cols']
    
    # Create blank image
    reconstructed = np.zeros(image_shape, dtype=np.uint8)
    
    # Fill with mean color
    reconstructed[rows, cols] = mean_color
    
    # Add residual if available
    if compressed_data.get('store_residual', False) and 'residual' in compressed_data:
        residual_quantized = compressed_data['residual']
        quantization_step = compressed_data.get('quantization_step', 32)
        
        # Dequantize residual
        residual = residual_quantized.astype(np.float32) * quantization_step
        
        # Apply residual and clip to valid range
        reconstructed_values = reconstructed[rows, cols].astype(np.float32) + residual
        reconstructed[rows, cols] = np.clip(reconstructed_values, 0, 255).astype(np.uint8)
    
    return reconstructed


def calculate_actual_compression_size_fixed(compressed_data):
    """
    Calculate actual compressed size in bytes.
    """
    if compressed_data['method'] != 'mean_color':
        return compressed_data.get('compressed_size', 0)
    
    size = 0
    
    # Mean color: 3 bytes
    size += 3
    
    # Homogeneity score: 4 bytes (float)
    size += 4
    
    # Number of pixels: 4 bytes (int32)
    size += 4
    
    # Mask coordinates: 
    num_pixels = len(compressed_data['mask_rows'])
    # We can store coordinates as uint16 if image is < 65536 pixels in each dimension
    size += num_pixels * 4  # 2 bytes for row, 2 bytes for col
    
    # Residual if stored
    if compressed_data.get('store_residual', False) and 'residual' in compressed_data:
        # Residual is int8 (1 byte per value)
        residual_size = compressed_data['residual'].size
        size += residual_size
    
    return size


























































# ==============================================
# TEXTURE-AWARE HOMOGENEOUS COMPRESSION
# ==============================================

def analyze_region_texture(region_image, region_mask):
    """
    Analyze if region has simple texture pattern.
    Returns: base_color, texture_pattern (small representative patch), texture_complexity
    """
    rows, cols = np.where(region_mask)
    if len(rows) == 0:
        return None, None, 1.0
    
    # Get region bounding box
    min_r, max_r = rows.min(), rows.max()
    min_c, max_c = cols.min(), cols.max()
    
    # Extract region
    region_crop = region_image[min_r:max_r+1, min_c:max_c+1]
    mask_crop = region_mask[min_r:max_r+1, min_c:max_c+1]
    
    # Calculate base color (mean)
    masked_pixels = region_image[rows, cols]
    base_color = np.mean(masked_pixels, axis=0).astype(np.uint8)
    
    # Calculate deviation from mean (texture pattern)
    base_image = np.full_like(region_crop, base_color)
    texture = region_crop.astype(np.int16) - base_image.astype(np.int16)
    
    # Only consider masked area
    texture_masked = texture * mask_crop[:, :, np.newaxis]
    
    # Analyze texture complexity
    # If texture is mostly zero or simple pattern, we can compress it efficiently
    texture_energy = np.mean(np.abs(texture_masked))
    texture_complexity = min(1.0, texture_energy / 50.0)  # Normalize
    
    # Find a small representative patch (e.g., 4x4 pixels)
    patch_size = 4
    if region_crop.shape[0] >= patch_size and region_crop.shape[1] >= patch_size:
        # Take top-left corner (or find most representative area)
        texture_patch = texture_masked[:patch_size, :patch_size, :]
    else:
        texture_patch = texture_masked
    
    return base_color, texture_patch, texture_complexity


def compress_textured_homogeneous_region(region_image, region_mask, texture_quality=4):
    """
    Compress homogeneous region with texture.
    Strategy: Base color + Quantized texture pattern
    """
    rows, cols = np.where(region_mask)
    if len(rows) == 0:
        return None
    
    num_pixels = len(rows)
    original_size = num_pixels * 3
    
    # Analyze texture
    base_color, texture_patch, complexity = analyze_region_texture(region_image, region_mask)
    
    # Get bounding box
    min_r, max_r = rows.min(), rows.max()
    min_c, max_c = cols.min(), cols.max()
    bbox = (min_r, min_c, max_r, max_c)
    
    compressed_data = {
        'method': 'texture_aware',
        'base_color': base_color,
        'bbox': bbox,
        'num_pixels': num_pixels,
        'original_size': original_size,
        'texture_complexity': complexity
    }
    
    # STRATEGY 1: If texture is very simple (complexity < 0.2), just store base color
    if complexity < 0.2:
        # Very homogeneous - just base color
        compressed_size = 3 + 16  # color + bbox
        compressed_data['compressed_size'] = compressed_size
        compressed_data['has_texture'] = False
        compressed_data['texture_data'] = None
    
    # STRATEGY 2: Simple texture - store quantized pattern
    else:
        # Quantize texture patch to reduce size
        # texture_patch is int16 (-255 to 255), we quantize to fewer bits
        if texture_patch is not None:
            # Quantize to 3-bit per channel (-3, -2, -1, 0, 1, 2, 3)
            quantized_texture = np.clip(np.round(texture_patch / texture_quality), -3, 3).astype(np.int8)
            
            # Flatten and store
            texture_bytes = quantized_texture.tobytes()
            texture_size = len(texture_bytes)
            
            compressed_data['texture_data'] = texture_bytes
            compressed_data['texture_shape'] = texture_patch.shape
            compressed_data['quantization_step'] = texture_quality
            compressed_data['has_texture'] = True
            
            compressed_size = 3 + 16 + texture_size  # color + bbox + texture
        else:
            compressed_size = 3 + 16
            compressed_data['has_texture'] = False
    
    compressed_data['compressed_size'] = compressed_size
    compressed_data['compression_ratio'] = original_size / compressed_size
    
    return compressed_data


def reconstruct_textured_homogeneous_region(compressed_data, region_mask, image_shape):
    """
    Reconstruct region using the provided mask.
    """
    reconstructed = np.zeros(image_shape, dtype=np.uint8)
    base_color = compressed_data['base_color']
    
    # Get bounding box
    min_r, min_c, max_r, max_c = compressed_data['bbox']
    
    # Fill masked area with base color
    rows, cols = np.where(region_mask)
    reconstructed[rows, cols] = base_color
    
    # Add texture if available
    if compressed_data.get('has_texture', False) and 'texture_data' in compressed_data:
        # Reconstruct texture pattern
        texture_bytes = compressed_data['texture_data']
        texture_shape = compressed_data['texture_shape']
        quant_step = compressed_data.get('quantization_step', 4)
        
        # Convert bytes back to array
        quantized_texture = np.frombuffer(texture_bytes, dtype=np.int8).reshape(texture_shape)
        
        # Dequantize
        texture_pattern = quantized_texture.astype(np.int16) * quant_step
        
        # Tile texture pattern
        patch_h, patch_w = texture_shape[:2]
        bbox_height = max_r - min_r + 1
        bbox_width = max_c - min_c + 1
        
        # Create texture for each masked pixel in bounding box
        for r, c in zip(rows, cols):
            if min_r <= r <= max_r and min_c <= c <= max_c:
                # Calculate position in texture tile
                tex_r = (r - min_r) % patch_h
                tex_c = (c - min_c) % patch_w
                
                # Apply texture
                pixel_value = reconstructed[r, c].astype(np.int16) + texture_pattern[tex_r, tex_c]
                reconstructed[r, c] = np.clip(pixel_value, 0, 255).astype(np.uint8)
    
    return reconstructed





























































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

































if __name__ == "__main__":

    image_name = 'images/waikiki.jpg'
    image = cv2.imread(image_name)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #enhanced_image_rgb=get_enhanced_image(image_rgb, shadow_threshold=100)
    enhanced_image_rgb=image_rgb

    print(f"Size: {image_rgb.size}")
    factor = math.ceil(math.log(image_rgb.size, 10)) * math.log(image_rgb.size)
    print(f"factor: {factor}")


    # Compare with your ROI detection
    edge_map = get_edge_map(enhanced_image_rgb)
    edge_density = compute_local_density(edge_map, kernel_size=3)

    threshold = suggest_automatic_threshold(edge_density, edge_map, method="mean") / 100
    
    window_size = math.floor(factor)
    min_region_size= math.ceil( image_rgb.size / math.pow(10, math.ceil(math.log(image_rgb.size, 10))-3 ) ) 
    print(f"min_region_size: {min_region_size}")

    print(f"\nWindow: {window_size}x{window_size}, Threshold: {threshold:.3f} ===")

    unified, regions, roi_image, nonroi_image, roi_mask, nonroi_mask = process_and_unify_borders(
        edge_map, edge_density, enhanced_image_rgb,
        density_threshold=threshold,
        #unification_method=method,
        min_region_size=min_region_size
    )


    


    # Create a version where only ROI regions are visible (non-ROI is black)
    roi_only_image = image_rgb.copy()
    roi_only_image[~roi_mask] = 0

    # Create a version where only non-ROI regions are visible
    nonroi_only_image = image_rgb.copy()
    nonroi_only_image[~nonroi_mask] = 0




    # Extract all connected regions for ROI and non-ROI
    roi_regions = extract_connected_regions(roi_mask, image_rgb)
    nonroi_regions = extract_connected_regions(nonroi_mask, image_rgb)

    print(f"Found {len(roi_regions)} ROI regions")
    print(f"Found {len(nonroi_regions)} non-ROI regions")

    # Display some statistics
    print("\nROI Regions (sorted by area):")
    for i, region in enumerate(sorted(roi_regions, key=lambda x: x['area'], reverse=True)[:5]):
        print(f"  Region {i+1}: Area = {region['area']} pixels")

    print("\nNon-ROI Regions (sorted by area):")
    for i, region in enumerate(sorted(nonroi_regions, key=lambda x: x['area'], reverse=True)[:5]):
        print(f"  Region {i+1}: Area = {region['area']} pixels")


    # Display ROI regions
    plot_regions(roi_regions, "ROI Regions")

    # Display non-ROI regions
    plot_regions(nonroi_regions, "Non-ROI Regions")



    # Display both
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(roi_only_image)
    plt.title('ROI Regions Only')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(nonroi_only_image)
    plt.title('Non-ROI Regions Only')
    plt.axis('off')

    plt.tight_layout()
    plt.show()



    """
    Change the split score
    """



    roi_subregions=[]
    nonroi_subregions=[]




    for i, region in enumerate(roi_regions):
        # Apply SLIC only to the bounding box region
        minr, minc, maxr, maxc = region['bbox']
        
        # Extract the region from the original image
        bbox_region = image_rgb[minr:maxr, minc:maxc]
        bbox_mask = region['bbox_mask']  # This masks only the actual irregular region

        region_image = bbox_region
        
        # Calculate split score ONLY on the irregular region (using the mask)
        overall_score, color_score, texture_score = calculate_split_score(bbox_region, bbox_mask)
        
        print(f"Region {i+1}:")
        print(f"  Overall score: {overall_score:.3f}")
        print(f"  Color score: {color_score:.3f}")
        print(f"  Texture score: {texture_score:.3f}")

        window = math.ceil(math.ceil(math.log(bbox_region.size, 10)) * math.log(bbox_region.size))
        print(f"Window: {window} px")
        normalized_overall_score = normalize_result(overall_score, window)
        optimal_segments = math.ceil(normalized_overall_score)
        
        if optimal_segments <= 0: 
            optimal_segments = 1

        roi_segments, texture_map = enhanced_slic_with_texture(bbox_region, n_segments=optimal_segments)
        
        # 🆕 EXTRACT SEGMENT BOUNDARIES
        segment_boundaries = extract_slic_segment_boundaries(roi_segments, bbox_mask)
        
        print(f"  Found {len(segment_boundaries)} sub-regions")
        
        
        # 🆕 SAVE TO FILE
        """filename = f"region_{i+1}_boundaries.txt"
        save_boundaries_to_file(segment_boundaries, filename)
        print(f"  Boundaries saved to: {filename}")"""
        
        # 🆕 PRINT SUMMARY
        total_boundary_points = sum(seg['num_points'] for seg in segment_boundaries)
        print(f"  Total boundary points: {total_boundary_points}")
        
        # Visualize
        visualize_split_analysis(
            region_image=region_image,
            overall_score=overall_score,
            color_score=color_score, 
            texture_score=texture_score,
            optimal_segments=optimal_segments
        )




























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
            }
            
            
























            


            
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
                row_str = "    " + " ".join(f"{val:3d}" for val in shared_qtable[i])
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
                
                print(f"\n    Segment {seg_idx+1}/{len(segment_boundaries)} (ID: {segment_id}):")
                print(f"      Pixels: {segment_pixels:,}")
                
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
                
                print(f"      Estimated in function: {seg_compression['compressed_size']:,} bytes")
                print(f"      Actual minimal: {actual_compressed_size:,} bytes")
                print(f"      Savings: {seg_compression['compressed_size'] - actual_compressed_size:,} bytes")
                
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










            
            # ==============================================
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
                plt.show()
            
            # ==============================================
            # 8. FINAL SUMMARY AND SAVE OPTIONS
            # ==============================================
            print(f"\n{'='*60}")
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
                    print(f"  ❌ Subregion compression FAILED: Ratio {subregion_compression_results['ratio']:.2f}:1")
            















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






        """# 🆕 DECOMPRESS AND VISUALIZE ALL SEGMENTS
        print("  Decompressing segments to verify quality...")
        reconstructed_region = np.zeros_like(bbox_region)

        for segment_data in all_segments_compressed:
            segment_id = segment_data['segment_id']
            compressed_data = segment_data['compressed_data']
            
            # Create mask for this segment
            segment_mask = (roi_segments == segment_id) & bbox_mask
            
            # Decompress this segment
            segment_reconstructed = decompress_segment_from_dct(compressed_data, segment_mask, quality=95)
            
            # Add to final reconstruction
            reconstructed_region[segment_mask] = segment_reconstructed[segment_mask]

        # Calculate quality metrics
        original_masked = bbox_region * bbox_mask[:, :, np.newaxis]
        reconstructed_masked = reconstructed_region * bbox_mask[:, :, np.newaxis]

        # Convert to proper range for PSNR calculation
        if original_masked.dtype == np.uint8:
            max_pixel = 255
        else:
            max_pixel = 1.0

        mse = np.mean((original_masked - reconstructed_masked) ** 2)
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse)) if mse > 0 else 100

        print(f"  Reconstruction quality: PSNR = {psnr:.2f} dB")"""


        # 🆕 DECOMPRESS AND VISUALIZE ALL SEGMENTS
        """print("  Decompressing segments to verify quality...")
        reconstructed_region = np.zeros_like(bbox_region)

        for segment_data in all_segments_compressed:
            segment_id = segment_data['segment_id']
            segment_mask = (roi_segments == segment_id) & bbox_mask
            
            # Use the improved decompression
            segment_reconstructed = decompress_segment_simple_dct(
                segment_data['compressed_data'], segment_mask
            )
            
            # Add to final reconstruction
            reconstructed_region[segment_mask] = segment_reconstructed[segment_mask]

        # Calculate quality metrics
        original_masked = bbox_region * bbox_mask[:, :, np.newaxis]
        reconstructed_masked = reconstructed_region * bbox_mask[:, :, np.newaxis]

        # Convert to proper range for PSNR calculation
        if original_masked.dtype == np.uint8:
            max_pixel = 255
        else:
            max_pixel = 1.0

        mse = np.mean((original_masked.astype(float) - reconstructed_masked.astype(float)) ** 2)
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse)) if mse > 0 else 100

        print(f"  Reconstruction quality: PSNR = {psnr:.2f} dB")"""







        

        """# 🆕 VISUALIZE ORIGINAL VS RECONSTRUCTED
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(bbox_region)
        plt.title('Original Region')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(reconstructed_region)
        plt.title('Reconstructed Region')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        # Difference visualization
        difference = np.abs(original_region_masked - reconstructed_masked)
        plt.imshow(difference, cmap='hot')
        plt.title(f'Difference (PSNR: {psnr:.2f} dB)')
        plt.colorbar()
        plt.axis('off')

        plt.tight_layout()
        plt.show()"""


        """# 🆕 VISUALIZE RESULTS
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 4, 1)
        plt.imshow(bbox_region)
        plt.title('Original Region')
        plt.axis('off')

        plt.subplot(1, 4, 2)
        plt.imshow(reconstructed_region.astype(np.uint8))
        plt.title('Reconstructed Region')
        plt.axis('off')

        plt.subplot(1, 4, 3)
        difference = np.abs(original_masked.astype(float) - reconstructed_masked.astype(float))
        plt.imshow(difference, cmap='hot', vmin=0, vmax=100)
        plt.title(f'Difference\nPSNR: {psnr:.2f} dB')
        plt.colorbar()
        plt.axis('off')

        plt.subplot(1, 4, 4)
        # Show which segments were compressed
        segments_display = roi_segments.copy()
        segments_display[~bbox_mask] = 0
        plt.imshow(segments_display, cmap='nipy_spectral')
        plt.title(f'{len(all_segments_compressed)} Compressed\nSegments')
        plt.axis('off')

        plt.tight_layout()
        plt.show()"""




        """# 🆕 VISUALIZE RESULTS
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(bbox_region)
        plt.title('Original Region')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(reconstructed_region)
        plt.title(f'Reconstructed Region\nPSNR: {psnr:.2f} dB')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        difference = np.abs(original_masked.astype(float) - reconstructed_masked.astype(float))
        plt.imshow(difference, cmap='hot', vmin=0, vmax=50)  # Reduced max for better visualization
        plt.title('Difference (Enhanced)')
        plt.colorbar()
        plt.axis('off')

        plt.tight_layout()
        plt.show()"""

        


    
    for i, region in enumerate(nonroi_regions):
        # Apply SLIC only to the bounding box region
        minr, minc, maxr, maxc = region['bbox']
        
        # Extract the region from the original image
        bbox_region = image_rgb[minr:maxr, minc:maxc]
        bbox_mask = region['bbox_mask']  # This masks only the actual irregular region

        region_image = bbox_region
        
        # Calculate split score ONLY on the irregular region (using the mask)
        overall_score, color_score, texture_score = calculate_split_score(bbox_region, bbox_mask)
        
        print(f"Region {i+1}:")
        print(f"  Overall score: {overall_score:.3f}")
        print(f"  Color score: {color_score:.3f}")
        print(f"  Texture score: {texture_score:.3f}")

        #if overall_score == 0: 
        #    continue

        window =math.ceil( math.ceil(math.log(bbox_region.size, 10)) * math.log(bbox_region.size) )
        print(f"Window: {window} px")
        normalized_overall_score=normalize_result(overall_score, window)
        optimal_segments=math.ceil(normalized_overall_score)
        
        # Calculate optimal segments based on score
        #optimal_segments = calculate_optimal_segments(overall_score, region['area'], min_segments=1, max_segments=factor)
        if optimal_segments<=0: optimal_segments=1

        nonroi_segments, texture_map = enhanced_slic_with_texture(bbox_region, n_segments=optimal_segments)
        
        # Visualize
        visualize_split_analysis(
            region_image=region_image,
            overall_score=overall_score,
            color_score=color_score, 
            texture_score=texture_score,
            optimal_segments=optimal_segments
        )

        # Display SLIC results
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 4, 1)
        plt.imshow(region_image)
        plt.title(f'ROI Region {i+1}')
        plt.axis('off')

        plt.subplot(1, 4, 2)
        # Show segments only within the irregular region
        segments_display = nonroi_segments.copy()
        segments_display[~bbox_mask] = 0  # Set background to 0
        plt.imshow(segments_display, cmap='nipy_spectral')
        plt.title(f'SLIC Segments: {nonroi_segments.max()}')
        plt.axis('off')

        plt.subplot(1, 4, 3)
        # Create boundaries only within the irregular region
        boundaries_image = mark_boundaries(bbox_region, nonroi_segments)
        boundaries_image[~bbox_mask] = 0  # Set background to black
        plt.imshow(boundaries_image)
        plt.title('SLIC Boundaries (Region Only)')
        plt.axis('off')

        plt.subplot(1, 4, 4)
        plt.imshow(texture_map)
        plt.title('texture_map')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

