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


import cv2
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
from scipy import interpolate










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






































import numpy as np
from collections import defaultdict




def merge_region_components_simple(region_components, roi_bbox):
    """
    Merge region components by placing them on ROI canvas.
    Colored pixels override black pixels.
    """
    if not region_components:
        return []
    
    if len(region_components) == 1:
        # For single segment, ensure it has the 'actual_colors' key
        single_seg = region_components[0].copy()
        if 'actual_colors' not in single_seg:
            single_seg['actual_colors'] = len(single_seg.get('palette', []))
        return [single_seg]
    
    print(f"\n{'='*60}")
    print(f"MERGING {len(region_components)} REGION COMPONENTS")
    print(f"{'='*60}")
    
    minr, minc, maxr, maxc = roi_bbox
    roi_height = maxr - minr
    roi_width = maxc - minc
    
    # ==============================================
    # 1. CREATE EMPTY CANVAS FOR ROI
    # ==============================================
    # Use uint32 to support up to ~4 billion colors (more than enough)
    roi_indices = np.zeros((roi_height, roi_width), dtype=np.uint32)
    
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
    for seg_idx, seg in enumerate(reversed(region_components)):
        seg_top_left = seg['top_left']
        seg_shape = seg['shape']
        seg_palette = seg['palette']
        seg_indices = np.array(seg['indices']).reshape(seg_shape)
        
        rel_row = seg_top_left[0] - minr
        rel_col = seg_top_left[1] - minc
        
        colored_placed = 0
        
        # Vectorized placement for better performance
        for r in range(seg_shape[0]):
            for c in range(seg_shape[1]):
                roi_r = rel_row + r
                roi_c = rel_col + c
                
                if 0 <= roi_r < roi_height and 0 <= roi_c < roi_width:
                    seg_pixel_idx = seg_indices[r, c]
                    
                    if seg_pixel_idx < len(seg_palette):
                        color_tuple = tuple(seg_palette[seg_pixel_idx])
                        
                        if color_tuple != black_color:
                            if color_tuple not in color_to_index:
                                color_to_index[color_tuple] = len(all_colors)
                                all_colors.append(color_tuple)
                            
                            roi_indices[roi_r, roi_c] = color_to_index[color_tuple]
                            colored_placed += 1
        
        print(f"  Segment {len(region_components)-seg_idx}: {colored_placed} colored pixels placed")
    
    # ==============================================
    # 3. CREATE MERGED SEGMENT
    # ==============================================
    black_pixels = np.sum(roi_indices == 0)
    colored_pixels_total = roi_height * roi_width - black_pixels
    
    print(f"\nMerging complete:")
    print(f"  ROI size: {roi_height}x{roi_width} = {roi_height*roi_width:,} pixels")
    print(f"  Black pixels: {black_pixels:,} ({black_pixels/(roi_height*roi_width)*100:.1f}%)")
    print(f"  Colored pixels: {colored_pixels_total:,}")
    print(f"  Unique colors: {len(all_colors)}")
    
    # Check if we can use smaller dtype for indices
    if len(all_colors) <= 256:
        indices_dtype = np.uint8
    elif len(all_colors) <= 65536:
        indices_dtype = np.uint16
    else:
        indices_dtype = np.uint32
    
    # Convert indices to appropriate dtype
    indices_converted = roi_indices.astype(indices_dtype)
    
    merged_segment = {
        'top_left': (minr, minc),
        'shape': (roi_height, roi_width),
        'palette': all_colors,
        'indices': indices_converted.flatten().tolist(),
        'indices_dtype': str(indices_dtype),
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
    mse = 0
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


def split_large_cluster(cluster_colors, max_colors_per_cluster):
    """
    Split a large cluster into smaller sub-clusters.
    Uses simple K-means to split.
    
    Args:
        cluster_colors: Array of colors in the cluster [n_colors, 3]
        max_colors_per_cluster: Maximum allowed colors per sub-cluster
    
    Returns:
        List of sub-cluster arrays
    """
    n_colors = len(cluster_colors)
    
    # If cluster is already small enough, return it as single cluster
    if n_colors <= max_colors_per_cluster:
        return [cluster_colors]
    
    # Calculate how many sub-clusters we need
    n_splits = max(2, (n_colors + max_colors_per_cluster - 1) // max_colors_per_cluster)
    
    # Ensure we don't ask for more clusters than colors
    n_splits = min(n_splits, n_colors)
    
    # Special case: very small clusters
    if n_colors <= 2 or n_splits < 2:
        return [cluster_colors]
    
    try:
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=n_splits, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(cluster_colors.astype(float))
        
        splits = []
        for i in range(n_splits):
            mask = labels == i
            if np.any(mask):
                splits.append(cluster_colors[mask])
        
        # Double-check that all splits are within size limit
        # If any split is still too large, recursively split it
        final_splits = []
        for split in splits:
            if len(split) > max_colors_per_cluster:
                final_splits.extend(split_large_cluster(split, max_colors_per_cluster))
            else:
                final_splits.append(split)
        
        return final_splits
        
    except Exception as e:
        print(f"Warning: KMeans splitting failed: {e}")
        print(f"  Cluster size: {n_colors}, Requested splits: {n_splits}")
        # Fallback: Simple split by luminance
        return split_by_luminance(cluster_colors, max_colors_per_cluster)


def split_by_luminance(cluster_colors, max_colors_per_cluster):
    """
    Fallback splitting method: sort by luminance and split evenly
    """
    n_colors = len(cluster_colors)
    
    if n_colors <= max_colors_per_cluster:
        return [cluster_colors]
    
    # Calculate luminance (Y = 0.299*R + 0.587*G + 0.114*B)
    luminance = (0.299 * cluster_colors[:, 0] + 
                 0.587 * cluster_colors[:, 1] + 
                 0.114 * cluster_colors[:, 2])
    
    # Sort by luminance
    sorted_indices = np.argsort(luminance)
    sorted_colors = cluster_colors[sorted_indices]
    
    # Split into approximately equal parts
    n_splits = max(2, (n_colors + max_colors_per_cluster - 1) // max_colors_per_cluster)
    split_points = np.array_split(sorted_colors, n_splits)
    
    # Filter out empty splits
    return [split for split in split_points if len(split) > 0]

def find_color_index(palette, color):
    """Find the index of a color in the palette."""
    matches = np.all(palette == color, axis=1)
    if np.any(matches):
        return np.where(matches)[0][0]
    return None









import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from sklearn.cluster import DBSCAN

def cluster_palette_colors_parallel(quality, compressed_data, eps=10.0, min_samples=2, 
                                   max_colors_per_cluster=5, num_workers=None):
    """
    Parallel version of color clustering.
    Large clusters are split in parallel using multiple threads.
    """
    print(f"\n{'='*60}")
    print(f"CLUSTERING PALETTE COLORS (Parallel - Black preserved)")
    print(f"{'='*60}")
    
    # Extract original data
    palette = np.array(compressed_data['palette'], dtype=np.uint8)
    indices = np.array(compressed_data['indices'])
    h, w = compressed_data['shape']
    top_left = compressed_data['top_left']
    
    # Get the count of colors (scalar, not array)
    original_colors_count = len(palette)  # This is a scalar integer
    num_workers= max(5, math.ceil(original_colors_count/2500) )
    print(f"Original palette: {original_colors_count} colors")
    print(f"Clustering parameters: eps={eps}, min_samples={min_samples}")
    
    # ==============================================
    # 1. SEPARATE BLACK FROM OTHER COLORS
    # ==============================================
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
    palette_normalized = non_black_palette.astype(float) / 255.0

    if len(non_black_palette)>=10000:

        # Use MiniBatch K-Means for speed
        n_clusters=math.ceil(len(non_black_palette)*(quality/100)/10)
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=1000,
            random_state=42,
            n_init='auto'
        )
        
        cluster_labels = kmeans.fit_predict(non_black_palette.astype(float))
    
        # Get the actual number of clusters found (K-means always finds n_clusters)
        actual_clusters = len(np.unique(cluster_labels))
        
        # Get the cluster centers (these are your reduced colors)
        cluster_centers = kmeans.cluster_centers_
        
        # Round and convert to uint8
        reduced_colors = np.round(cluster_centers).astype(np.uint8)
        
        print(f"  K-Means reduction: {len(non_black_palette)} → {len(reduced_colors)} colors")
        print(f"  Actual clusters found: {actual_clusters}")
    
    else:
        dbscan = DBSCAN(eps=eps/255.0, min_samples=min_samples, metric='euclidean')
        #dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        cluster_labels = dbscan.fit_predict(palette_normalized)


    
    

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = np.sum(cluster_labels == -1)
    
    print(f"DBSCAN found {n_clusters} clusters + {n_noise} noise colors")
    
    # ==============================================
    # 3. PREPARE CLUSTERS FOR PARALLEL PROCESSING
    # ==============================================
    new_palette = []
    color_mapping = {}
    
    # Add black colors
    for old_black_idx in black_indices:
        new_palette.append(palette[old_black_idx])
        color_mapping[old_black_idx] = len(new_palette) - 1
    
    # Process noise points (not parallelizable - small)
    noise_mask = cluster_labels == -1
    noise_relative_indices = np.where(noise_mask)[0]
    
    for rel_idx in noise_relative_indices:
        old_idx = non_black_indices[rel_idx]
        new_palette.append(palette[old_idx])
        color_mapping[old_idx] = len(new_palette) - 1
    
    # ==============================================
    # 4. PARALLEL PROCESSING OF LARGE CLUSTERS
    # ==============================================
    # Identify clusters that need splitting
    clusters_to_process = []
    small_clusters_info = []  # Store info for small clusters
    
    unique_labels = set(cluster_labels)
    
    for label in unique_labels:
        if label == -1:
            continue  # Already processed noise
        
        cluster_mask = cluster_labels == label
        cluster_relative_indices = np.where(cluster_mask)[0]
        cluster_original_indices = [non_black_indices[idx] for idx in cluster_relative_indices]
        cluster_colors = palette[cluster_original_indices]
        
        if len(cluster_original_indices) > max_colors_per_cluster:
            # Large cluster - needs parallel splitting
            clusters_to_process.append({
                'label': label,
                'colors': cluster_colors,
                'original_indices': cluster_original_indices,
                'size': len(cluster_original_indices)
            })
        else:
            # Small cluster - store for immediate processing
            small_clusters_info.append({
                'label': label,
                'colors': cluster_colors,
                'original_indices': cluster_original_indices
            })
    
    print(f"Found {len(clusters_to_process)} large clusters for parallel processing")
    print(f"Found {len(small_clusters_info)} small clusters for immediate processing")
    
    # Process small clusters immediately
    for cluster_info in small_clusters_info:
        avg_color = np.mean(cluster_info['colors'], axis=0).astype(np.uint8)
        new_idx = len(new_palette)
        new_palette.append(avg_color)
        
        for old_idx in cluster_info['original_indices']:
            color_mapping[old_idx] = new_idx
    
    # ==============================================
    # 5. PARALLEL SPLITTING OF LARGE CLUSTERS
    # ==============================================
    if clusters_to_process:
        # Determine number of workers
        if num_workers is None:
            num_workers = min(multiprocessing.cpu_count(), len(clusters_to_process))
        
        print(f"Using {num_workers} workers for parallel processing")
        
        # Process clusters in parallel
        results = process_clusters_parallel(
            clusters_to_process, 
            max_colors_per_cluster, 
            num_workers
        )
        
        # Process results from parallel workers
        for cluster_info, split_clusters in results:
            label = cluster_info['label']
            original_indices = cluster_info['original_indices']
            original_colors = cluster_info['colors']
            
            if split_clusters is None:
                # Fallback: use simple averaging
                print(f"  Cluster {label}: parallel processing failed, using fallback")
                avg_color = np.mean(original_colors, axis=0).astype(np.uint8)
                new_idx = len(new_palette)
                new_palette.append(avg_color)
                
                for old_idx in original_indices:
                    color_mapping[old_idx] = new_idx
            else:
                # Process each split cluster
                for split_cluster in split_clusters:
                    avg_color = np.mean(split_cluster, axis=0).astype(np.uint8)
                    new_idx = len(new_palette)
                    new_palette.append(avg_color)
                    
                    # Map colors in this split
                    for color in split_cluster:
                        old_idx = find_color_index(palette, color)
                        if old_idx is not None:
                            color_mapping[old_idx] = new_idx
    
    # ==============================================
    # 6. COMPLETE PROCESSING AND CALCULATIONS
    # ==============================================
    new_palette = np.array(new_palette)
    new_color_count = len(new_palette)  # This is a scalar integer
    
    # Verify black is preserved
    black_preserved = any(np.array_equal(color, [0, 0, 0]) for color in new_palette)
    print(f"  Black preserved: {black_preserved}")
    
    # FIXED: Use scalar integers for printing
    print(f"New palette: {new_color_count} colors")
    print(f"Color reduction: {original_colors_count} → {new_color_count} "
          f"({(original_colors_count - new_color_count)/original_colors_count*100:.1f}%)")
    
    # Update indices
    mapping_array = np.zeros(original_colors_count, dtype=np.uint16)
    for old_idx, new_idx in color_mapping.items():
        mapping_array[old_idx] = new_idx
    
    new_indices = mapping_array[indices].tolist()
    
    # ==============================================
    # 7. CALCULATE STATISTICS (MISSING IN ORIGINAL)
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
    
    # Estimate PSNR - FIXED: pass proper parameters
    mse = 0
    psnr = 10 * np.log10(255*255/mse) if mse > 0 else float('inf')
    
    # ==============================================
    # 8. CREATE NEW COMPRESSED DATA
    # ==============================================
    new_compressed_data = {
        'method': 'clustered_colors',
        'top_left': top_left,
        'shape': (h, w),
        'palette': new_palette.tolist(),
        'indices': new_indices,
        'original_unique_colors': original_colors_count,  # Use scalar
        'compressed_colors': new_color_count,  # Use scalar
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
        'black_preserved': True,
        'parallel_processed': True  # Flag to indicate parallel processing was used
    }
    
    print(f"\nParallel clustering complete:")
    print(f"  Colors: {original_colors_count} → {new_color_count}")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  Compression: {original_size:,} → {new_compressed_size:,} bytes")
    print(f"  Ratio: {compression_ratio:.2f}:1")
    
    return new_compressed_data

def process_clusters_parallel(clusters_to_process, max_colors_per_cluster, num_workers):
    """
    Process multiple clusters in parallel using ThreadPoolExecutor.
    """
    results = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_cluster = {}
        for cluster_info in clusters_to_process:
            future = executor.submit(
                split_cluster_worker,
                cluster_info['colors'],
                max_colors_per_cluster,
                cluster_info['label']
            )
            future_to_cluster[future] = cluster_info
        
        # Process results as they complete
        for future in as_completed(future_to_cluster):
            cluster_info = future_to_cluster[future]
            try:
                split_clusters = future.result(timeout=30)  # 30 second timeout
                results.append((cluster_info, split_clusters))
            except Exception as e:
                print(f"Error processing cluster {cluster_info['label']}: {e}")
                results.append((cluster_info, None))  # Mark as failed
    
    return results

def split_cluster_worker(cluster_colors, max_colors_per_cluster, cluster_label=None):
    """
    Worker function for parallel cluster splitting.
    """
    try:
        return split_large_cluster(cluster_colors, max_colors_per_cluster)
    except Exception as e:
        if cluster_label:
            print(f"Worker error for cluster {cluster_label}: {e}")
        return None











































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








































import math
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist

def compute_clustering_params(n_colors, quality, color_space='rgb'):
    """
    Compute optimized DBSCAN parameters for color clustering
    
    Args:
        n_colors: Number of distinct colors in palette
        quality: 0-100 quality parameter
        color_space: 'rgb' or 'lab' (LAB is perceptually uniform)
    
    Returns:
        eps, min_samples, max_colors_per_cluster
    """
    # ==============================================
    # 1. EPSILON (maximum distance within cluster)
    # ==============================================
    # Quality 0-100 maps to perceptual thresholds
    # Higher quality = smaller epsilon (tighter clusters)
    
    """
    if color_space == 'lab':
        # LAB space: ΔE*ab perceptual distances
        # ΔE < 1: Not perceptible
        # ΔE 1-2: Perceptible by trained observers
        # ΔE 2-10: Perceptible at glance
        # ΔE > 10: Different colors
        
        # Map quality to ΔE threshold
        min_eps = 2.0    # Quality 100 = JND threshold
        max_eps = 20.0   # Quality 0 = very different colors
        
        # Exponential decay for better control
        eps = max_eps * math.exp(-quality / 25.0) + min_eps
        
    else:  # RGB space
        # RGB Euclidean distance (0-441.67 max)
        # Quality 100: tight clusters (eps=5)
        # Quality 0: loose clusters (eps=100)
        min_eps = 5.0
        max_eps = 100.0
        
        # Non-linear mapping for better control
        eps = min_eps + (max_eps - min_eps) * ((100 - quality) / 100) ** 2
    
    # ==============================================
    # 2. MIN_SAMPLES (minimum points to form cluster)
    # ==============================================
    # Higher quality = more strict clustering (require more samples)
    if n_colors < 10:
        min_samples = 1
    elif n_colors < 100:
        # Scale with number of colors
        min_samples = max(1, int(2 * quality / 100))
    else:
        # For large palettes, be more selective
        min_samples = max(2, int(3 * quality / 100))
    
    # ==============================================
    # 3. MAX COLORS PER CLUSTER (compression factor)
    # ==============================================
    # Higher quality = preserve more colors (smaller clusters)
    # Lower quality = merge more colors (larger clusters)
    
    # Base compression ratio target based on quality
    target_compression_ratio = 1.0 + (100 - quality) / 25.0
    # quality=100 → ratio=1.0 (no merging)
    # quality=75 → ratio=2.0 (merge 2:1)
    # quality=50 → ratio=3.0 (merge 3:1)
    # quality=25 → ratio=4.0 (merge 4:1)
    # quality=0 → ratio=5.0 (merge 5:1)
    
    # Adjust based on palette size
    if n_colors < 50:
        # Small palettes: be conservative
        max_colors_per_cluster = int(target_compression_ratio * 1.5)
    elif n_colors < 200:
        # Medium palettes: normal merging
        max_colors_per_cluster = int(target_compression_ratio)
    else:
        # Large palettes: can be more aggressive
        max_colors_per_cluster = int(target_compression_ratio * 0.8)
    
    # Ensure reasonable bounds
    max_colors_per_cluster = max(1, min(10, max_colors_per_cluster))
    
    # ==============================================
    # 4. ADAPTIVE EPSILON BASED ON LOCAL DENSITY
    # ==============================================
    # Option: Compute k-distance to determine eps automatically
    # eps = compute_k_distance(colors, k=min_samples)

    """

    #eps=256*100/(quality*2)
    eps=128 - 1.28 * quality
    #max_colors_per_cluster=math.ceil(n_colors*100/quality)
    max_colors_per_cluster= math.ceil( ( -(quality / 100) * n_colors + n_colors) / quality )

    if eps==0: eps=1
    if max_colors_per_cluster==0: max_colors_per_cluster=1
    min_samples=1
    
    return eps, min_samples, max_colors_per_cluster


def print_compressed_data_types(compressed_data, name="Compressed Data"):
    """
    Print detailed type information about compressed data
    """
    print(f"\n{'='*60}")
    print(f"DATA TYPE ANALYSIS: {name}")
    print(f"{'='*60}")
    
    if not isinstance(compressed_data, dict):
        print(f"ERROR: Expected dict, got {type(compressed_data)}")
        return
    
    for key, value in compressed_data.items():
        # Handle special cases
        if key == 'palette':
            if isinstance(value, list):
                print(f"{key:20}: list of {len(value)} colors")
                if value:
                    first_color = value[0]
                    print(f"  First color type: {type(first_color)}, length: {len(first_color)}")
                    print(f"  Color values: {first_color[:3]}...")
            elif isinstance(value, np.ndarray):
                print(f"{key:20}: numpy array {value.shape}, dtype={value.dtype}")
            else:
                print(f"{key:20}: {type(value)}")
        
        elif key == 'indices':
            if isinstance(value, list):
                print(f"{key:20}: list of {len(value)} indices")
                if value:
                    max_val = max(value)
                    min_val = min(value)
                    print(f"  Value range: {min_val} to {max_val}")
                    
                    # Determine optimal dtype
                    if max_val < 256:
                        optimal_dtype = 'uint8'
                    elif max_val < 65536:
                        optimal_dtype = 'uint16'
                    else:
                        optimal_dtype = 'uint32'
                    print(f"  Optimal dtype: {optimal_dtype}")
                    
                    # Estimate memory usage
                    list_bytes = len(value) * 28  # Approx bytes per Python int
                    optimal_bytes = len(value) * (1 if optimal_dtype == 'uint8' else 2 if optimal_dtype == 'uint16' else 4)
                    print(f"  Memory: {list_bytes:,}B (list) → {optimal_bytes:,}B ({optimal_dtype})")
                    
            elif isinstance(value, np.ndarray):
                print(f"{key:20}: numpy array {value.shape}, dtype={value.dtype}")
                print(f"  Value range: {value.min()} to {value.max()}")
                
                # Check if can be downgraded
                max_val = value.max()
                if max_val < 256 and value.dtype != np.uint8:
                    print(f"  ⚠️  Can be downgraded: {value.dtype} → uint8")
                elif max_val < 65536 and value.dtype not in [np.uint16, np.uint8]:
                    print(f"  ⚠️  Can be downgraded: {value.dtype} → uint16")
            else:
                print(f"{key:20}: {type(value)}")
        
        else:
            # For other keys
            if isinstance(value, (list, tuple)):
                print(f"{key:20}: {type(value).__name__} of {len(value)} items")
            elif isinstance(value, np.ndarray):
                print(f"{key:20}: numpy array {value.shape}, dtype={value.dtype}")
            else:
                print(f"{key:20}: {type(value).__name__}: {repr(value)[:50]}...")
    
    # Calculate total estimated size
    print(f"\n{'='*60}")
    print("MEMORY ESTIMATE:")
    total_size = 0
    
    for key, value in compressed_data.items():
        if key == 'indices' and isinstance(value, list):
            total_size += len(value) * 28  # Approx 28 bytes per Python int
        elif key == 'indices' and isinstance(value, np.ndarray):
            total_size += value.nbytes
        elif key == 'palette' and isinstance(value, list):
            total_size += len(value) * 3  # 3 bytes per RGB color
        elif key == 'palette' and isinstance(value, np.ndarray):
            total_size += value.nbytes
        elif isinstance(value, str):
            total_size += len(value)
        elif isinstance(value, (int, float)):
            total_size += 28  # Approx for Python numbers
    
    print(f"Total estimated size: {total_size:,} bytes")
    print(f"{'='*60}")



def optimize_compressed_dtype(compressed_data):
    """
    Downgrade uint32 indices to uint8 or uint16 when possible.
    
    Args:
        compressed_data: Dictionary with 'indices' and 'palette'
    
    Returns:
        Optimized dictionary with smallest dtype
    """
    if 'indices' not in compressed_data:
        return compressed_data
    
    indices = compressed_data['indices']
    
    # Convert to numpy array if it's a list
    if isinstance(indices, list):
        indices_array = np.array(indices)
    elif isinstance(indices, np.ndarray):
        indices_array = indices.copy()
    else:
        print(f"Warning: Unknown indices type: {type(indices)}")
        return compressed_data
    
    # Get current dtype info
    current_dtype = indices_array.dtype
    max_value = indices_array.max() if len(indices_array) > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"DTYPE OPTIMIZATION")
    print(f"{'='*60}")
    print(f"Current dtype: {current_dtype}")
    print(f"Max index value: {max_value}")
    
    # Determine optimal dtype
    if max_value < 256:
        optimal_dtype = np.uint8
        dtype_name = 'uint8'
        bytes_per_index = 1
    elif max_value < 65536:
        optimal_dtype = np.uint16
        dtype_name = 'uint16'
        bytes_per_index = 2
    else:
        optimal_dtype = np.uint32
        dtype_name = 'uint32'
        bytes_per_index = 4
    
    # Calculate savings
    current_bytes = indices_array.nbytes
    if current_dtype == np.uint8:
        current_bytes_per_index = 1
    elif current_dtype == np.uint16:
        current_bytes_per_index = 2
    elif current_dtype == np.uint32:
        current_bytes_per_index = 4
    else:
        current_bytes_per_index = 4  # Default assumption
    
    optimized_bytes = len(indices_array) * bytes_per_index
    savings = current_bytes - optimized_bytes
    
    print(f"Optimal dtype: {dtype_name}")
    print(f"Current size: {current_bytes:,} bytes ({current_bytes_per_index} bytes/index)")
    print(f"Optimized size: {optimized_bytes:,} bytes ({bytes_per_index} bytes/index)")
    print(f"Savings: {savings:,} bytes ({savings/current_bytes*100:.1f}% smaller)")
    
    # Only convert if there's a benefit
    if optimal_dtype != current_dtype:
        print(f"✅ Converting: {current_dtype} → {optimal_dtype}")
        indices_optimized = indices_array.astype(optimal_dtype)
        
        # Update the compressed data
        optimized_data = compressed_data.copy()
        optimized_data['indices'] = indices_optimized.tolist()  # Or keep as array
        optimized_data['indices_dtype'] = dtype_name
        optimized_data['indices_optimized'] = True
        
        # Update palette count if available
        if 'palette' in optimized_data:
            actual_colors = len(optimized_data['palette'])
            optimized_data['actual_colors'] = actual_colors
            print(f"Palette size: {actual_colors} colors")
        
        return optimized_data
    else:
        print(f"✓ Already optimal dtype: {current_dtype}")
        return compressed_data





















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
        if optimal_segments<1: optimal_segments=1
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

            quality=15
            n_colors = seg_compression['actual_colors']

            """
            distance= 256 - (256*quality / 100)
            eps=math.pow(100/quality,3)

            coefficient_max_samples=quality/100
            max_sample_pre=math.pow(n_colors, coefficient_max_samples )
            #max_colors_per_cluster=math.ceil( math.pow(math.e,max_sample_pre) * 3*100/quality  )
            max_colors_per_cluster=math.ceil(max_sample_pre * 3*100/quality  )
            """

            palette_colors = np.array(seg_compression['palette'])
            
            # Option A: Simple improved formulas
            eps, min_samples, max_colors_per_cluster = compute_clustering_params(
                n_colors, quality, color_space='lab'
            )

            

            # Then cluster the colors
            seg_compression = cluster_palette_colors_parallel(
                quality,
                seg_compression,
                eps=eps,           # Distance threshold (0-255 scale)
                min_samples=1,      # Min colors to form cluster
                max_colors_per_cluster=max_colors_per_cluster  # Split large clusters
            )

            print(f"Eps: {eps}")
            print(f"n_colors: {n_colors}")
            #print(f"max_sample_pre: {max_sample_pre}")
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
        if optimal_segments<1: optimal_segments=1
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

            quality=5

            """
            distance= 256 - (256*quality / 100)
            eps=math.pow(100/quality,3)

            coefficient_max_samples=quality/100
            max_sample_pre=math.pow(n_colors, coefficient_max_samples )
            #max_colors_per_cluster=math.ceil( math.pow(math.e,max_sample_pre) * 3*100/quality  )
            max_colors_per_cluster=math.ceil(max_sample_pre * 3*100/quality  )
            """

            n_colors = seg_compression['actual_colors']
            palette_colors = np.array(seg_compression['palette'])
            
            # Option A: Simple improved formulas
            eps, min_samples, max_colors_per_cluster = compute_clustering_params(
                n_colors, quality, color_space='lab'
            )


            # Then cluster the colors
            seg_compression = cluster_palette_colors_parallel(
                quality,
                seg_compression,
                eps=eps,           # Distance threshold (0-255 scale)
                min_samples=1,      # Min colors to form cluster
                max_colors_per_cluster=max_colors_per_cluster  # Split large clusters
            )

            print(f"Eps: {eps}")
            print(f"n_colors: {n_colors}")
            #print(f"max_sample_pre: {max_sample_pre}")
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







































    







    image_components=[]

    """
    Second phase of hierarchical clustering:
    Clustering colors between each ROI / nonROI
    """

    #ROI

    # First, collect all ROI components
    try: all_roi_components_flat = [roi[0] for roi in ROI_components]
    except: all_roi_components_flat = [roi for roi in ROI_components]

    # More robust flattening
    all_roi_components_flat = []
    for roi in ROI_components:
        if isinstance(roi, dict):
            all_roi_components_flat.append(roi)
        elif isinstance(roi, list):
            for item in roi:
                if isinstance(item, dict):
                    all_roi_components_flat.append(item)
                else:
                    print(f"Warning: Skipping non-dict item in list: {type(item)}")
        else:
            print(f"Warning: Skipping unexpected type: {type(roi)}")

    print(f"Processed {len(ROI_components)} inputs → {len(all_roi_components_flat)} segments")

    # Use the entire image as the bbox
    image_bbox = (0, 0, original_image_height, original_image_width)

    roi_image = merge_region_components_simple(
        all_roi_components_flat,
        roi_bbox=image_bbox
    )


    # Extract the merged segment dictionary
    merged_segment = roi_image[0]  # This contains palette and indices, NOT the image!

    quality=25
    n_colors = merged_segment['actual_colors']

    """
    distance= 256 - (256*quality / 100)
    eps=math.pow(100/quality,3)

    coefficient_max_samples=quality/100
    max_sample_pre=math.pow(n_colors, coefficient_max_samples )
    #max_colors_per_cluster=math.ceil( math.pow(math.e,max_sample_pre) * 3*100/quality  )
    max_colors_per_cluster=math.ceil(max_sample_pre * 3*100/quality  )
    """

    n_colors = merged_segment['actual_colors']
    palette_colors = np.array(merged_segment['palette'])
    
    # Option A: Simple improved formulas
    eps, min_samples, max_colors_per_cluster = compute_clustering_params(
        n_colors, quality, color_space='lab'
    )


    # ==============================================
    # OPTION 1: If you want to cluster the merged palette
    # ==============================================
    # You already have the merged palette in merged_segment
    # Just cluster it directly:
    ROI_seg_compression = cluster_palette_colors_parallel(
        quality,
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
    try: all_nonroi_components_flat = [roi[0] for roi in nonROI_components]
    except:  all_nonroi_components_flat = [roi for roi in nonROI_components]

    # More robust flattening
    all_nonroi_components_flat = []
    for nonroi in nonROI_components:
        if isinstance(nonroi, dict):
            all_nonroi_components_flat.append(nonroi)
        elif isinstance(nonroi, list):
            for item in nonroi:
                if isinstance(item, dict):
                    all_nonroi_components_flat.append(item)
                else:
                    print(f"Warning: Skipping non-dict item in list: {type(item)}")
        else:
            print(f"Warning: Skipping unexpected type: {type(nonroi)}")

    print(f"Processed {len(nonROI_components)} inputs → {len(all_nonroi_components_flat)} segments")

    # Use the entire image as the bbox
    image_bbox = (0, 0, original_image_height, original_image_width)

    nonroi_image = merge_region_components_simple(
        all_nonroi_components_flat,
        roi_bbox=image_bbox
    )

    # Extract the merged segment dictionary
    merged_segment = nonroi_image[0]  # This contains palette and indices, NOT the image!

    quality=15
    n_colors = merged_segment['actual_colors']

    """
    distance= 256 - (256*quality / 100)
    eps=math.pow(100/quality,3)

    coefficient_max_samples=quality/100
    max_sample_pre=math.pow(n_colors, coefficient_max_samples )
    #max_colors_per_cluster=math.ceil( math.pow(math.e,max_sample_pre) * 3*100/quality  )
    max_colors_per_cluster=math.ceil(max_sample_pre * 3*100/quality  )
    """

    n_colors = merged_segment['actual_colors']
    palette_colors = np.array(merged_segment['palette'])
    
    # Option A: Simple improved formulas
    eps, min_samples, max_colors_per_cluster = compute_clustering_params(
        n_colors, quality, color_space='lab'
    )

    # ==============================================
    # OPTION 1: If you want to cluster the merged palette
    # ==============================================
    # You already have the merged palette in merged_segment
    # Just cluster it directly:
    nonROI_seg_compression = cluster_palette_colors_parallel(
        quality,
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

    quality=35
    n_colors = merged_segment['actual_colors']

    """
    distance= 256 - (256*quality / 100)
    eps=math.pow(100/quality,3)

    coefficient_max_samples=quality/100
    max_sample_pre=math.pow(n_colors, coefficient_max_samples )
    #max_colors_per_cluster=math.ceil( math.pow(math.e,max_sample_pre) * 3*100/quality  )
    max_colors_per_cluster=math.ceil(max_sample_pre * 3*100/quality  )
    """

    n_colors = merged_segment['actual_colors']
    palette_colors = np.array(merged_segment['palette'])
    
    # Option A: Simple improved formulas
    eps, min_samples, max_colors_per_cluster = compute_clustering_params(
        n_colors, quality, color_space='lab'
    )

    # ==============================================
    # OPTION 1: If you want to cluster the merged palette
    # ==============================================
    # You already have the merged palette in merged_segment
    # Just cluster it directly:
    image_seg_compression = cluster_palette_colors_parallel(
        quality,
        merged_segment,  # Pass the dictionary
        eps=eps,
        min_samples=1,
        max_colors_per_cluster=max_colors_per_cluster
    )

    # Print current types
    print_compressed_data_types(image_seg_compression, "BEFORE OPTIMIZATION")

    # Optimize dtype
    image_seg_compression = optimize_compressed_dtype(image_seg_compression)

    # Print optimized types
    print_compressed_data_types(image_seg_compression, "AFTER OPTIMIZATION")



    


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



    savePicture=False
    if savePicture:
        from PIL import Image
        import numpy as np

        # Assuming reconstruction_result is your decompressed result
        reconstruction_result = decompress_color_quantization(image_seg_compression)

        # Extract the image
        reconstructed_image = reconstruction_result['image']  # This is a numpy array

        # Convert numpy array to PIL Image and save
        pil_image = Image.fromarray(reconstructed_image)
        pil_image.save('reconstructed_image.jpg', quality=25)  # quality 1-100

        print(f"✅ Image saved as 'reconstructed_image.jpg'")
        print(f"   Size: {reconstructed_image.shape[1]}x{reconstructed_image.shape[0]}")






    saveCompression = True
    if saveCompression:
        print(f"\n{'='*60}")
        print(f"FINAL LOSS LESS COMPRESSION")
        print(f"{'='*60}")
        
        # Get your final data
        final_data = image_seg_compression  # or clustered_result
        
        # Extract components
        shape = final_data['shape']
        palette = final_data['palette']
        indices_flat = final_data['indices']
        
        # Convert to matrix
        h, w = shape
        indices_matrix = np.array(indices_flat).reshape(h, w)
        
        print(f"Compressing: {w}x{h} image, {len(palette)} colors")
        
        # Compress
        from encoder.compression.compression import lossless_compress_optimized, save_compressed
        
        compressed_data = lossless_compress_optimized(palette, indices_matrix, shape)
        
        # Save
        filename = "compressed_lenna.rhccq"
        file_size = save_compressed(compressed_data, filename)
        
        # Stats
        original_size = h * w * 3
        compression_ratio = original_size / file_size
        
        print(f"✅ Saved: {filename}")
        print(f"   Original: {original_size:,} bytes")
        print(f"   Compressed: {file_size:,} bytes")
        print(f"   Ratio: {compression_ratio:.2f}:1")
        print(f"   Savings: {(1 - file_size/original_size)*100:.1f}%")            
