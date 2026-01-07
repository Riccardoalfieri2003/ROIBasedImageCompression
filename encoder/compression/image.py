import numpy as np
from encoder.compression.clustering import compute_clustering_params,cluster_palette_colors_parallel
from encoder.compression.compression import optimize_compressed_dtype, print_compressed_data_types
from encoder.compression.merging import merge_region_components_simple

from decoder.uncompression.uncompression import lossless_decompress, load_compressed, decompress_color_quantization, partial_decompress_color_quantization


import numpy as np
from scipy import ndimage

def fill_black_holes_in_segment(merged_segment, max_hole_size=10, connectivity=4):
    """
    Fill isolated black pixels and small black holes in a merged segment.
    
    Parameters:
    - merged_segment: Dictionary with 'palette', 'indices', and 'shape'
    - max_hole_size: Maximum size of black region to fill (in pixels)
    - connectivity: 4 or 8 connectivity for hole detection
    
    Returns:
    - Updated merged_segment with filled holes
    """
    print(f"\n{'='*60}")
    print(f"FILLING BLACK HOLES (max size: {max_hole_size} pixels)")
    print(f"{'='*60}")
    
    # Extract data from segment
    palette = np.array(merged_segment['palette'])
    indices = np.array(merged_segment['indices'])
    shape = merged_segment['shape']
    
    # Reshape indices to 2D
    indices_2d = indices.reshape(shape)
    
    # Find black color index in palette
    black_index = None
    for i, color in enumerate(palette):
        if np.array_equal(color, [0, 0, 0]):
            black_index = i
            break
    
    if black_index is None:
        print("No black color found in palette. Nothing to fill.")
        return merged_segment.copy()
    
    print(f"Black color at palette index: {black_index}")
    
    # Create binary mask of black pixels
    black_mask = (indices_2d == black_index)
    
    # Count initial black pixels
    initial_black = np.sum(black_mask)
    total_pixels = shape[0] * shape[1]
    
    print(f"Initial black pixels: {initial_black:,} ({initial_black/total_pixels*100:.2f}%)")
    
    if initial_black == 0:
        print("No black pixels to fill.")
        return merged_segment.copy()
    
    # Label connected components of black pixels
    if connectivity == 4:
        structure = np.array([[0, 1, 0],
                             [1, 1, 1],
                             [0, 1, 0]])
    else:  # connectivity == 8
        structure = np.ones((3, 3), dtype=int)
    
    labeled_black, num_features = ndimage.label(black_mask, structure=structure)
    
    print(f"Found {num_features} connected black regions")
    
    # Create copy of indices to modify
    new_indices = indices_2d.copy()
    filled_regions = 0
    filled_pixels = 0
    
    # Process each black region
    for label in range(1, num_features + 1):
        region_mask = (labeled_black == label)
        region_size = np.sum(region_mask)
        
        if region_size <= max_hole_size:
            # This is a small hole that should be filled
            
            # Get coordinates of this region
            region_coords = np.where(region_mask)
            
            # Find the most common non-black color among neighbors
            # Dilate the region to get its neighbors
            dilated = ndimage.binary_dilation(region_mask, structure=structure)
            neighbor_mask = dilated & ~region_mask
            
            if np.any(neighbor_mask):
                # Get neighbor indices (excluding black)
                neighbor_indices = new_indices[neighbor_mask]
                non_black_neighbors = neighbor_indices[neighbor_indices != black_index]
                
                if len(non_black_neighbors) > 0:
                    # Find the most common neighbor color
                    from collections import Counter
                    color_counts = Counter(non_black_neighbors)
                    most_common_color = max(color_counts, key=color_counts.get)
                    
                    # Fill the region with this color
                    new_indices[region_mask] = most_common_color
                    filled_regions += 1
                    filled_pixels += region_size
                    
                    # Debug output for larger fills
                    if region_size > 1:
                        print(f"  Filled region {label}: {region_size} pixels with color {most_common_color}")
    
    # Statistics
    final_black = np.sum(new_indices == black_index)
    black_removed = initial_black - final_black
    
    print(f"\nHole filling complete:")
    print(f"  Filled {filled_regions} regions")
    print(f"  Filled {filled_pixels:,} pixels")
    print(f"  Black pixels removed: {black_removed:,}")
    print(f"  Remaining black pixels: {final_black:,} ({final_black/total_pixels*100:.2f}%)")
    
    # Create updated segment
    updated_segment = merged_segment.copy()
    updated_segment['indices'] = new_indices.flatten().tolist()
    updated_segment['method'] = updated_segment.get('method', 'merged') + '_filled'
    
    return updated_segment


def fill_black_holes_vectorized(merged_segment, max_hole_size=10):
    """
    Fill small black holes with appropriate neighboring colors.
    Each hole gets filled with the most common color from its own neighbors.
    """
    import numpy as np
    from scipy import ndimage
    from collections import Counter
    
    palette = np.array(merged_segment['palette'])
    indices = np.array(merged_segment['indices'])
    shape = merged_segment['shape']
    
    indices_2d = indices.reshape(shape)
    
    # Find black index
    black_index = None
    for i, color in enumerate(palette):
        if np.array_equal(color, [0, 0, 0]):
            black_index = i
            break
    
    if black_index is None:
        print("No black color in palette")
        return merged_segment.copy()
    
    print(f"Found black at palette index {black_index}")
    
    # Create black mask
    black_mask = (indices_2d == black_index)
    initial_black = black_mask.sum()
    print(f"Initial black pixels: {initial_black:,}")
    
    if initial_black == 0:
        return merged_segment.copy()
    
    # Label connected black regions
    structure = np.ones((3, 3), dtype=int)  # 8-connectivity
    labeled, num_regions = ndimage.label(black_mask, structure=structure)
    
    print(f"Found {num_regions} black regions")
    
    # Get region sizes
    region_sizes = ndimage.sum(black_mask, labeled, range(1, num_regions + 1))
    
    # Identify small regions to fill (size <= max_hole_size)
    small_region_ids = []
    for i, size in enumerate(region_sizes):
        if size <= max_hole_size:
            small_region_ids.append(i + 1)  # +1 because labels start at 1
    
    if not small_region_ids:
        print("No small black regions to fill")
        return merged_segment.copy()
    
    print(f"Found {len(small_region_ids)} small regions (size <= {max_hole_size})")
    
    # Process each small region
    new_indices = indices_2d.copy()
    filled_count = 0
    
    for region_id in small_region_ids:
        # Create mask for this specific region
        region_mask = (labeled == region_id)
        region_size = region_mask.sum()
        
        # Get dilated region to find neighbors
        dilated = ndimage.binary_dilation(region_mask, structure=structure)
        neighbor_mask = dilated & ~region_mask
        
        if not neighbor_mask.any():
            continue  # No neighbors, can't fill properly
            
        # Get neighbor indices (excluding black)
        neighbor_indices = indices_2d[neighbor_mask]
        non_black_neighbors = neighbor_indices[neighbor_indices != black_index]
        
        if len(non_black_neighbors) == 0:
            continue  # All neighbors are also black
            
        # Find most common color among THIS region's neighbors
        color_counts = Counter(non_black_neighbors)
        most_common_color = max(color_counts.items(), key=lambda x: x[1])[0]
        
        # Fill the entire region with this color
        new_indices[region_mask] = most_common_color
        filled_count += region_size
        
        # Debug for larger fills
        if region_size > 1:
            # Get the actual RGB color for debugging
            fill_color_rgb = palette[most_common_color]
            print(f"  Region {region_id} ({region_size} pixels) → Color {most_common_color} = {fill_color_rgb}")
    
    # Statistics
    final_black = (new_indices == black_index).sum()
    black_removed = initial_black - final_black
    
    print(f"\nFilled {filled_count} pixels in {len(small_region_ids)} regions")
    print(f"Black pixels removed: {black_removed:,}")
    print(f"Remaining black: {final_black:,}")
    
    # Update segment
    updated_segment = merged_segment.copy()
    updated_segment['indices'] = new_indices.flatten().tolist()
    updated_segment['method'] = updated_segment.get('method', '') + '_holes_filled'
    
    return updated_segment


def quantize_image(image_components, original_image_height, original_image_width, quality=100):

    # First, collect all Regions components
    all_image_components_flat = [regions for regions in image_components]

    # Use the entire image as the bbox
    image_bbox = (0, 0, original_image_height, original_image_width)

    regions_image = merge_region_components_simple(
        all_image_components_flat,
        roi_bbox=image_bbox
    )

    # Extract the merged segment dictionary
    merged_segment = regions_image[0]  # This contains palette and indices, NOT the image!

    merged_segment=fill_black_holes_vectorized(merged_segment, max_hole_size=50)

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
        reconstruction_result = partial_decompress_color_quantization(image_seg_compression)
        
        # ==============================================
        # 4. SIMPLE VISUALIZATION
        # ==============================================
        print(f"\n{'='*60}")
        print(f"Regions RECONSTRUCTION")
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
        
        # 1. Reconstructed Regions
        axes[0].imshow(reconstructed_img)
        axes[0].set_title(f'Reconstructed Regions\n{h}x{w} pixels\n{n_colors} colors\nPSNR: {psnr:.1f} dB')
        axes[0].axis('off')
        
        # 2. Colored pixels only (white background)
        colored_only = reconstructed_img.copy()
        black_mask = np.all(reconstructed_img == [0, 0, 0], axis=2)
        colored_only[black_mask] = [255, 255, 255]  # White background for black areas
        
        axes[1].imshow(colored_only)
        axes[1].set_title(f'Colored Pixels Only\nBlack pixels: {np.sum(black_mask):,}\n({np.sum(black_mask)/(h*w)*100:.1f}%)')
        axes[1].axis('off')
        
        plt.suptitle(f'Regions at position {top_left}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Console summary
        print(f"Position: {top_left}")
        print(f"Size: {h}x{w} = {h*w:,} pixels")
        print(f"Colors: {n_colors}")
        print(f"Black pixels: {np.sum(black_mask):,} ({np.sum(black_mask)/(h*w)*100:.1f}%)")
        print(f"Colored pixels: {h*w - np.sum(black_mask):,} ({(h*w - np.sum(black_mask))/(h*w)*100:.1f}%)")
        print(f"PSNR: {psnr:.1f} dB")

    return image_seg_compression