

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


def get_quantization_table(quality):
    """
    Get quantization table for given quality
    """
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























def compress_segment_with_dct_fixed(segment_mask, region_image, quality=95):
    """
    Fixed DCT compression with proper brightness handling
    """
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

def quantize_ultra_gentle(dct_block, quality):
    """
    Ultra-gentle quantization - preserve almost all detail
    """
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




















































if __name__ == "__main__":
    image_name = 'images/kauai.jpg'
    image = cv2.imread(image_name)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    enhanced_image_rgb=get_enhanced_image(image_rgb, shadow_threshold=100)

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


    """for region in roi_regions:

        # Apply SLIC only to the bounding box region
        minr, minc, maxr, maxc = region['bbox']
        
        # Extract the region from the original image
        bbox_region = image_rgb[minr:maxr, minc:maxc]

        region_image=bbox_region
        #should_split_roi, roi_score = should_split(bbox_region)
        roi_score=calculate_split_score(bbox_region)

        print(f"roi_score {roi_score}")

        if roi_score==0: continue

            # Apply SLIC only on ROI regions
        roi_segments = slic(bbox_region, 
                        n_segments=math.ceil(roi_score*2),  # Adjust based on your needs
                        compactness=10,
                        sigma=1,
                        mask=region["bbox_mask"])  # This ensures SLIC only works on ROI areas
        
        roi_segments, texture_map = enhanced_slic_with_texture(bbox_region, math.ceil(roi_score*2))

        visualize_split_analysis(region_image, overall_score, color_score, texture_score, optimal_segments):"""
    

    """    
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

        #if overall_score == 0: 
        #    continue

        window =math.ceil( math.ceil(math.log(bbox_region.size, 10)) * math.log(bbox_region.size) )
        print(f"Window: {window} px")
        normalized_overall_score=normalize_result(overall_score, window)
        optimal_segments=math.ceil(normalized_overall_score)
        
        # Calculate optimal segments based on score
        #optimal_segments = calculate_optimal_segments(overall_score, region['area'], min_segments=1, max_segments=factor)
        if optimal_segments<=0: optimal_segments=1

        roi_segments, texture_map = enhanced_slic_with_texture(bbox_region, n_segments=optimal_segments)
        
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
        segments_display = roi_segments.copy()
        segments_display[~bbox_mask] = 0  # Set background to 0
        plt.imshow(segments_display, cmap='nipy_spectral')
        plt.title(f'SLIC Segments: {roi_segments.max()}')
        plt.axis('off')

        plt.subplot(1, 4, 3)
        # Create boundaries only within the irregular region
        boundaries_image = mark_boundaries(bbox_region, roi_segments)
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

    """



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



        """# 🆕 COMPRESS EACH SLIC SEGMENT INDIVIDUALLY
        print("  Applying DCT compression to SLIC segments...")
        all_segments_compressed = []
        total_segments_original = 0
        total_segments_compressed = 0

        for segment_idx, segment_info in enumerate(segment_boundaries):
            # Create mask for this specific segment
            segment_mask = (roi_segments == segment_info['segment_id']) & bbox_mask
            
            if np.sum(segment_mask) > 0:  # Only compress non-empty segments
                compressed_segment, segment_metrics = compress_segment_with_dct(
                    segment_mask, bbox_region, quality=75
                )
                
                if compressed_segment:
                    all_segments_compressed.append({
                        'segment_id': segment_info['segment_id'],
                        'compressed_data': compressed_segment,
                        'metrics': segment_metrics,
                        'boundary_coords': segment_info['boundary_coords']
                    })
                    
                    # Accumulate metrics
                    for channel_metrics in segment_metrics.values():
                        total_segments_original += channel_metrics['original_size']
                        total_segments_compressed += channel_metrics['compressed_size']

        # Print segment compression results
        if total_segments_original > 0:
            overall_ratio = total_segments_compressed / total_segments_original
            print(f"    Segments compression: {total_segments_original:,} → {total_segments_compressed:,} bytes ({overall_ratio:.1%})")
            print(f"    {len(all_segments_compressed)} segments compressed")
        """





        # 🆕 COMPRESS EACH SLIC SEGMENT INDIVIDUALLY
        print("  Applying DCT compression to SLIC segments...")
        all_segments_compressed = []
        total_segments_original = 0
        total_segments_compressed = 0

        for segment_idx, segment_info in enumerate(segment_boundaries):
            # Create mask for this specific segment
            segment_mask = (roi_segments == segment_info['segment_id']) & bbox_mask
            
            if np.sum(segment_mask) > 0:  # Only compress non-empty segments
                compressed_segment, segment_metrics = compress_segment_with_dct_fixed(
                    segment_mask, bbox_region, quality=50
                )
                
                """if compressed_segment:
                    # Calculate size metrics manually since function returns different format
                    segment_size = 0
                    compressed_size = 0
                    
                    for channel_name in ['R', 'G', 'B']:
                        if channel_name in compressed_segment:
                            channel_info = compressed_segment[channel_name]
                            segment_height, segment_width = channel_info['bbox'][2], channel_info['bbox'][3]
                            segment_size += segment_height * segment_width * 4  # 4 bytes per float32
                            compressed_size += len(channel_info['compressed_blocks']) * 64 * 4  # Approximate
                    
                    all_segments_compressed.append({
                        'segment_id': segment_info['segment_id'],
                        'compressed_data': compressed_segment,
                        'metrics': {
                            'original_size': segment_size,
                            'compressed_size': compressed_size,
                            'compression_ratio': compressed_size / segment_size if segment_size > 0 else 0
                        }
                    })
                    
                    # Accumulate metrics
                    total_segments_original += segment_size
                    total_segments_compressed += compressed_size
                    """
                


                if compressed_segment:
                    # Calculate ACTUAL sizes
                    segment_original_size = calculate_original_size(segment_mask, bbox_region)
                    segment_compressed_size = calculate_actual_compressed_size(compressed_segment)
                    
                    all_segments_compressed.append({
                        'segment_id': segment_info['segment_id'],
                        'compressed_data': compressed_segment,
                        'metrics': {
                            'original_size': segment_original_size,
                            'compressed_size': segment_compressed_size,
                            'compression_ratio': segment_compressed_size / segment_original_size
                        }
                    })
                    
                    total_segments_original += segment_original_size
                    total_segments_compressed += segment_compressed_size

                """ # Print segment compression results
                if total_segments_original > 0:
                    overall_ratio = total_segments_compressed / total_segments_original
                    print(f"    Segments compression: {total_segments_original:,} → {total_segments_compressed:,} bytes ({overall_ratio:.1%})")
                    print(f"    {len(all_segments_compressed)} segments compressed")"""

                
                # Print REAL compression results
                if total_segments_original > 0:
                    overall_ratio = total_segments_compressed / total_segments_original
                    print(f"    REAL compression: {total_segments_original:,} → {total_segments_compressed:,} bytes ({overall_ratio:.1%})")
                    print(f"    {len(all_segments_compressed)} segments compressed")
                    
                    if overall_ratio < 1.0:  print(f"    ✅ ACTUAL SPACE SAVING: {(1-overall_ratio)*100:.1f}%")
                    else:  print(f"    ❌ NO COMPRESSION: {((overall_ratio-1)*100):.1f}% SIZE INCREASE")




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
        colors = plt.cm.Set3(np.linspace(0, 1, len(segment_boundaries)))
        for j, segment in enumerate(segment_boundaries):
            coords = np.array(segment['boundary_coords'])
            if len(coords) > 0:
                plt.plot(coords[:, 1], coords[:, 0], color=colors[j], linewidth=2, 
                        label=f'Seg {segment["segment_id"]}')
        
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
        print("  Decompressing segments to verify quality...")
        reconstructed_region = np.zeros_like(bbox_region)

        for segment_data in all_segments_compressed:
            segment_id = segment_data['segment_id']
            segment_mask = (roi_segments == segment_id) & bbox_mask
            
            # Use the improved decompression
            segment_reconstructed = decompress_segment_fixed(
                segment_data['compressed_data'], segment_mask, quality=100
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

        print(f"  Reconstruction quality: PSNR = {psnr:.2f} dB")







        

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




        # 🆕 VISUALIZE RESULTS
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
        plt.show()

        


    
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

