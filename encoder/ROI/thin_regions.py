import cv2
import numpy as np

from encoder.ROI.edges import compute_local_density

def remove_thin_structures(binary_image, density_threshold=0.2, 
                          thinness_threshold=0.3, window_size=25,
                          min_region_size=10, connectivity=8):
    """
    Remove thin regions (any shape) in low-density areas.
    
    Args:
        binary_image: Binary image (0 and 255)
        density_threshold: Maximum density to consider for removal (0-1)
        thinness_threshold: How thin a region must be to be removed (0-1, lower = more thin)
        window_size: Size of window for density calculation
        min_region_size: Minimum region size to consider
        connectivity: 4 or 8 connectivity
    
    Returns:
        cleaned_image: Image without thin regions in sparse areas
    """
    if np.sum(binary_image > 0) == 0:
        return binary_image
    
    # Calculate local density map
    density_map = compute_local_density(binary_image, window_size)
    
    # Find all connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=connectivity)
    
    # Identify thin regions
    thin_regions_mask = identify_thin_regions(binary_image, labels, stats, thinness_threshold, min_region_size)
    
    # Only remove thin regions in low-density areas
    regions_to_remove = np.zeros_like(binary_image, dtype=bool)
    
    for region_id in range(1, num_labels):
        if thin_regions_mask[labels == region_id].any():
            # Get the region mask
            region_mask = (labels == region_id)
            
            # Calculate average density for this region
            region_density = np.mean(density_map[region_mask])
            
            # Remove if in low-density area
            if region_density < density_threshold:
                regions_to_remove[region_mask] = True
    
    # Remove the identified regions
    cleaned_image = binary_image.copy()
    cleaned_image[regions_to_remove] = 0
    
    removed_pixels = np.sum(regions_to_remove)
    print(f"Removed {removed_pixels} pixels from {np.sum([np.any(regions_to_remove[labels == i]) for i in range(1, num_labels)])} thin regions in low-density areas")
    
    return cleaned_image

def identify_thin_regions(binary_image, labels, stats, thinness_threshold=0.3, min_region_size=10):
    """
    Identify regions that are thin based on aspect ratio and skeleton analysis.
    """
    thin_mask = np.zeros_like(binary_image, dtype=bool)
    height, width = binary_image.shape
    
    for i in range(1, len(stats)):
        region_area = stats[i, cv2.CC_STAT_AREA]
        
        if region_area < min_region_size:
            continue
            
        region_mask = (labels == i)
        
        # Calculate thinness using multiple methods
        thinness_score = calculate_region_thinness(region_mask, stats[i])
        
        # Region is considered thin if below threshold
        if thinness_score < thinness_threshold:
            thin_mask[region_mask] = True
    
    return thin_mask

def calculate_region_thinness(region_mask, region_stats):
    """
    Calculate how thin a region is using multiple metrics.
    Returns a score between 0 (very thin) and 1 (not thin).
    """
    # Method 1: Aspect ratio thinness
    width = region_stats[cv2.CC_STAT_WIDTH]
    height = region_stats[cv2.CC_STAT_HEIGHT]
    area = region_stats[cv2.CC_STAT_AREA]
    
    # Compactness measure (thin regions have high perimeter-to-area ratio)
    perimeter = calculate_region_perimeter(region_mask)
    compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    
    # Aspect ratio thinness
    max_dim = max(width, height)
    min_dim = min(width, height)
    aspect_thinness = min_dim / max_dim if max_dim > 0 else 0
    
    # Area-to-bounding-box ratio (thin regions have low ratio)
    bbox_area = width * height
    area_ratio = area / bbox_area if bbox_area > 0 else 0
    
    # Combined thinness score (lower = more thin)
    thinness_score = (compactness + aspect_thinness + area_ratio) / 3.0
    
    return thinness_score

def calculate_region_perimeter(region_mask):
    """
    Calculate the perimeter of a region using contour detection.
    """
    # Convert mask to uint8
    region_uint8 = region_mask.astype(np.uint8) * 255
    
    # Find contours
    contours, _ = cv2.findContours(region_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return 0
    
    # Use the largest contour
    perimeter = cv2.arcLength(contours[0], True)
    return perimeter

