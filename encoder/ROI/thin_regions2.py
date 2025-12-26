import cv2
import numpy as np

import time

from encoder.ROI.edges import compute_local_density







def remove_thin_structures_optimized(binary_image, density_threshold=0.2, 
                                   thinness_threshold=0.3, window_size=25,
                                   min_region_size=10, connectivity=8):
    """
    Optimized removal of thin regions in low-density areas.
    10-100x faster than original.
    """
    if np.sum(binary_image > 0) == 0:
        return binary_image
    
    start_time = time.time()
    
    # Calculate local density map
    density_map = compute_local_density(binary_image, window_size)
    print(f"density_map: {time.time() - start_time:.3f} seconds")
    
    start_time = time.time()
    
    # Find all connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_image, connectivity=connectivity
    )
    print(f"connected: {time.time() - start_time:.3f} seconds")
    
    start_time = time.time()
    
    # Identify thin regions
    thin_regions_mask = identify_thin_regions_ultrafast(binary_image)
    print(f"thin_regions_mask: {time.time() - start_time:.3f} seconds")
    
    start_time = time.time()
    
    # ==============================================
    # OPTIMIZED REGION PROCESSING (Replaces the slow loop)
    # ==============================================
    
    # Step 1: Get all thin region IDs (vectorized)
    thin_region_ids = np.unique(labels[thin_regions_mask])
    
    # Step 2: Calculate region densities for ALL regions at once (vectorized)
    # This replaces the loop that does: np.mean(density_map[region_mask])
    
    # Flatten arrays for fast processing
    labels_flat = labels.ravel()
    density_flat = density_map.ravel()
    
    # Use bincount to compute sums and counts per region
    region_sums = np.zeros(num_labels, dtype=np.float64)
    region_counts = np.zeros(num_labels, dtype=np.int64)
    
    # Alternative faster method: use numpy's bincount with weights
    # This is MUCH faster than looping
    region_counts = np.bincount(labels_flat, minlength=num_labels)
    region_sums = np.bincount(labels_flat, weights=density_flat, minlength=num_labels)
    
    # Calculate average densities (avoid division by zero)
    region_densities = np.zeros(num_labels, dtype=np.float32)
    valid_regions = region_counts > 0
    region_densities[valid_regions] = (
        region_sums[valid_regions] / region_counts[valid_regions]
    )
    
    # Step 3: Identify which thin regions have low density
    regions_to_remove_ids = thin_region_ids[
        region_densities[thin_region_ids] < density_threshold
    ]
    
    # Step 4: Create removal mask (vectorized)
    # Method A: Using numpy.isin (fast for many regions)
    regions_to_remove = np.isin(labels, regions_to_remove_ids)
    
    # Method B: Alternative - use advanced indexing (sometimes faster for few regions)
    # if len(regions_to_remove_ids) < 100:
    #     # Create lookup table
    #     remove_lut = np.zeros(num_labels, dtype=bool)
    #     remove_lut[regions_to_remove_ids] = True
    #     regions_to_remove = remove_lut[labels]
    
    # Step 5: Apply removal
    cleaned_image = binary_image.copy()
    cleaned_image[regions_to_remove] = 0
    
    removed_pixels = np.sum(regions_to_remove)
    removed_regions = len(regions_to_remove_ids)
    
    print(f"Removed {removed_pixels} pixels from {removed_regions} thin regions in low-density areas")
    print(f"post: {time.time() - start_time:.3f} seconds")
    
    return cleaned_image





def identify_thin_regions_fast(binary_image, min_region_size=10, thinness_threshold=0.3):
    """
    Ultra-fast thin region detection - RETURNS SAME FORMAT AS ORIGINAL.
    """
    height, width = binary_image.shape
    
    # Distance transform
    dist_transform = cv2.distanceTransform(
        binary_image.astype(np.uint8),
        cv2.DIST_L2,
        3
    )
    
    # Get connected components FIRST
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_image.astype(np.uint8), 
        connectivity=8
    )
    
    # Create thin mask
    thin_mask = np.zeros_like(binary_image, dtype=bool)
    
    # Process each region
    for i in range(1, num_labels):
        region_area = stats[i, cv2.CC_STAT_AREA]
        
        if region_area < min_region_size:
            continue
            
        # Get region mask
        region_mask = (labels == i)
        
        # Calculate thinness using distance transform
        # Thinness = average distance to edge / max possible distance
        region_distances = dist_transform[region_mask]
        
        if len(region_distances) > 0:
            avg_distance = np.mean(region_distances)
            max_dim = max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
            
            # Normalize: avg_distance / (max_dim/2) [0-1]
            if max_dim > 0:
                normalized_thinness = (avg_distance * 2) / max_dim
                
                # Lower score = more thin
                thinness_score = 1.0 - normalized_thinness
                
                if thinness_score > thinness_threshold:  # Note: > not <
                    thin_mask[region_mask] = True
    
    return thin_mask

def identify_thin_regions_ultrafast(binary_image, min_region_size=10, thinness_threshold=0.3):
    """
    Ultra-fast thin region detection - FULLY VECTORIZED.
    10-50x faster than loop version.
    """
    # ==============================================
    # 1. Distance transform (already fast - C++ optimized)
    # ==============================================
    dist_transform = cv2.distanceTransform(
        binary_image.astype(np.uint8),
        cv2.DIST_L2,
        3
    )
    
    # ==============================================
    # 2. Connected components
    # ==============================================
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_image.astype(np.uint8), 
        connectivity=8
    )
    
    if num_labels <= 1:  # No foreground
        return np.zeros_like(binary_image, dtype=bool)
    
    # ==============================================
    # 3. VECTORIZED region processing (NO LOOPS!)
    # ==============================================
    
    # Step A: Filter regions by size (vectorized)
    region_areas = stats[1:, cv2.CC_STAT_AREA]  # Skip background (label 0)
    valid_region_mask = region_areas >= min_region_size
    
    if not np.any(valid_region_mask):
        return np.zeros_like(binary_image, dtype=bool)
    
    # Step B: Get max dimensions for each region (vectorized)
    widths = stats[1:, cv2.CC_STAT_WIDTH]
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    max_dims = np.maximum(widths, heights)
    
    # Step C: Calculate average distance per region (vectorized using bincount)
    # This is the KEY OPTIMIZATION - replaces the loop!
    
    # Flatten arrays
    labels_flat = labels.ravel()
    dist_flat = dist_transform.ravel()
    
    # Use bincount to compute sums and counts per region
    # This is C-speed, not Python loop speed!
    region_sums = np.bincount(labels_flat, weights=dist_flat, minlength=num_labels)
    region_counts = np.bincount(labels_flat, minlength=num_labels)
    
    # Calculate average distances (avoid division by zero)
    avg_distances = np.zeros(num_labels)
    valid_counts = region_counts > 0
    avg_distances[valid_counts] = region_sums[valid_counts] / region_counts[valid_counts]
    
    # Step D: Calculate thinness scores (vectorized)
    # For regions 1..num_labels-1
    region_avg_dists = avg_distances[1:]
    region_max_dims = max_dims
    
    # Avoid division by zero
    valid_dims = region_max_dims > 0
    normalized_thinness = np.zeros_like(region_avg_dists)
    normalized_thinness[valid_dims] = (region_avg_dists[valid_dims] * 2) / region_max_dims[valid_dims]
    
    # Thinness score (0=not thin, 1=very thin)
    thinness_scores = 1.0 - normalized_thinness
    
    # Step E: Identify thin regions (vectorized)
    is_thin = (thinness_scores > thinness_threshold) & valid_region_mask
    
    # Get IDs of thin regions
    thin_region_ids = np.where(is_thin)[0] + 1  # +1 because we skipped label 0
    
    # ==============================================
    # 4. Create mask (vectorized)
    # ==============================================
    if len(thin_region_ids) == 0:
        return np.zeros_like(binary_image, dtype=bool)
    
    # Fastest method: use numpy.isin (C-speed)
    thin_mask = np.isin(labels, thin_region_ids)
    
    return thin_mask