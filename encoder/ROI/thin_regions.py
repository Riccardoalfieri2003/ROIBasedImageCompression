import cv2
import numpy as np

import time

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
    
    start_time=time.time()
    # Calculate local density map
    density_map = compute_local_density(binary_image, window_size)
    print("density_map: %s seconds ---" % (time.time() - start_time))
    
    start_time=time.time()
    # Find all connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=connectivity)
    print("connected: %s seconds ---" % (time.time() - start_time))
    
    start_time=time.time()
    # Identify thin regions
    thin_regions_mask = identify_thin_regions(binary_image, labels, stats, thinness_threshold, min_region_size)
    #thin_regions_mask = identify_thin_regions_optimized(binary_image, labels, stats, thinness_threshold) 
    print("thin_regions_mask: %s seconds ---" % (time.time() - start_time))


    start_time=time.time()
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
    print("post: %s seconds ---" % (time.time() - start_time))
    
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
    #perimeter = calculate_perimeter_approximate(region_mask)
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



def identify_thin_regions_heuristic(binary_image, labels, stats, 
                                    thinness_threshold=0.3):
    """
    Use simple heuristics to quickly identify thin regions.
    """
    thin_mask = np.zeros_like(binary_image, dtype=bool)
    
    for i in range(1, len(stats)):
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        # QUICK HEURISTIC 1: Aspect ratio
        max_dim = max(width, height)
        min_dim = min(width, height)
        
        if min_dim == 0:
            continue
            
        aspect_ratio = max_dim / min_dim
        
        # QUICK HEURISTIC 2: Area to bounding box ratio
        bbox_area = width * height
        if bbox_area == 0:
            continue
            
        fill_ratio = area / bbox_area
        
        # Combined quick score
        quick_score = (1.0 / aspect_ratio) * fill_ratio
        
        # Only do detailed analysis if likely thin
        if quick_score < thinness_threshold * 1.5:  # Looser threshold
            region_mask = (labels == i)
            
            # Now do detailed analysis
            thinness_score = calculate_region_thinness(region_mask, stats[i])
            
            if thinness_score < thinness_threshold:
                thin_mask[region_mask] = True
    
    return thin_mask













def identify_thin_regions_optimized(binary_image, labels, stats, 
                                    thinness_threshold=0.3, 
                                    min_region_size=10,
                                    max_region_size=1000):  # ADD THIS!
    """
    Optimized version with pre-filtering.
    """
    height, width = binary_image.shape
    thin_mask = np.zeros_like(binary_image, dtype=bool)
    
    # Pre-calculate all contours ONCE
    contours = get_all_contours(binary_image, labels)
    
    for i in range(1, len(stats)):
        region_area = stats[i, cv2.CC_STAT_AREA]
        
        # QUICK REJECTION: Skip if too small or too large
        if region_area < min_region_size or region_area > max_region_size:
            continue
        
        # Get contour from pre-computed list
        contour = contours[i] if i < len(contours) else None
        if contour is None or len(contour) == 0:
            continue
        
        # Calculate thinness using pre-computed contour
        thinness_score = calculate_region_thinness_ultrafast(
            stats[i], contour
        )
        
        if thinness_score < thinness_threshold:
            # Only create mask for thin regions
            region_mask = (labels == i)
            thin_mask[region_mask] = True
    
    return thin_mask

def get_all_contours(binary_image, labels):
    """
    Compute contours for ALL regions at once.
    """
    contours_dict = {}
    unique_labels = np.unique(labels)
    
    # Process each label
    for label in unique_labels:
        if label == 0:  # Skip background
            continue
            
        region_mask = (labels == label)
        region_uint8 = region_mask.astype(np.uint8) * 255
        
        contour, _ = cv2.findContours(
            region_uint8, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if contour:
            contours_dict[label] = contour[0]
    
    return contours_dict

def calculate_region_thinness_fast(region_stats, contour=None):
    """
    Fast thinness calculation using only bounding box stats.
    No mask creation, no contour finding needed.
    
    Args:
        region_stats: Stats from cv2.connectedComponentsWithStats
        contour: Optional pre-computed contour (can be None)
    
    Returns:
        float: Thinness score (0=very thin, 1=not thin)
    """
    # Extract stats
    width = region_stats[cv2.CC_STAT_WIDTH]
    height = region_stats[cv2.CC_STAT_HEIGHT]
    area = region_stats[cv2.CC_STAT_AREA]
    
    # Quick return for impossible cases
    if area == 0 or width == 0 or height == 0:
        return 1.0  # Not thin
    
    # ==============================================
    # 1. Aspect Ratio Thinness (FAST)
    # ==============================================
    max_dim = max(width, height)
    min_dim = min(width, height)
    
    # Aspect ratio: 0=line, 1=square
    aspect_ratio = min_dim / max_dim if max_dim > 0 else 0
    
    # ==============================================
    # 2. Area to Bounding Box Ratio (FAST)
    # ==============================================
    bbox_area = width * height
    area_ratio = area / bbox_area if bbox_area > 0 else 0
    
    # ==============================================
    # 3. Perimeter Approximation (FAST)
    # ==============================================
    if contour is not None and len(contour) > 0:
        # Use pre-computed contour if available
        perimeter = cv2.arcLength(contour, True)
    else:
        # FAST approximation: perimeter ≈ 2*(width + height)
        # This is accurate for rectangular shapes
        perimeter_approx = 2 * (width + height)
        
        # Adjust for non-rectangular shapes using area
        # For a perfect circle: perimeter = 2√(π*area)
        # For a thin line: perimeter ≈ 2*max_dim
        circle_perimeter = 2 * np.sqrt(np.pi * area)
        line_perimeter = 2 * max_dim
        
        # Blend between extremes based on aspect ratio
        perimeter = (perimeter_approx * 0.5 + 
                    circle_perimeter * (1 - aspect_ratio) * 0.25 +
                    line_perimeter * aspect_ratio * 0.25)
    
    # Compactness: 1=perfect circle, 0=very thin
    if perimeter > 0:
        compactness = (4 * np.pi * area) / (perimeter ** 2)
        # Clamp to [0, 1]
        compactness = max(0, min(1, compactness))
    else:
        compactness = 1.0
    
    # ==============================================
    # 4. Additional Fast Heuristics
    # ==============================================
    
    # Line-likeness: ratio of max dimension to area
    # Thin lines have high max_dim/area ratio
    line_likeness = max_dim / area if area > 0 else 0
    # Normalize (empirical threshold)
    line_likeness_norm = min(1.0, line_likeness / 10.0)
    
    # Solidity: area / convex hull area (approximate)
    # For thin regions, convex hull is much larger than area
    if contour is not None and len(contour) > 0:
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 1.0
    else:
        # Approximation: solidity ≈ area_ratio
        solidity = area_ratio ** 0.5  # Square root gives better spread
    
    # ==============================================
    # 5. Combine Scores (Weighted Average)
    # ==============================================
    # Weights can be adjusted based on your needs
    weights = {
        'aspect_ratio': 0.3,      # Simple and effective
        'area_ratio': 0.25,       # Captures filling
        'compactness': 0.25,      # Shape regularity
        'solidity': 0.15,         # Convexity
        'line_likeness': 0.05,    # Additional hint
    }
    
    thinness_score = (
        weights['aspect_ratio'] * aspect_ratio +
        weights['area_ratio'] * area_ratio +
        weights['compactness'] * compactness +
        weights['solidity'] * solidity +
        weights['line_likeness'] * (1 - line_likeness_norm)  # Inverted
    )
    
    # Normalize to [0, 1]
    thinness_score = max(0, min(1, thinness_score))
    
    return thinness_score

def calculate_region_thinness_ultrafast(region_stats, contour):
    """
    Ultra-fast thinness calculation using ONLY bounding box.
    No contours, no approximations.
    
    Returns score where lower = more thin.
    """
    width = region_stats[cv2.CC_STAT_WIDTH]
    height = region_stats[cv2.CC_STAT_HEIGHT]
    area = region_stats[cv2.CC_STAT_AREA]
    
    if area <= 4:  # Tiny regions
        return 0.0 if width == 1 or height == 1 else 0.5
    
    # 1. Extreme aspect ratio check
    max_dim = max(width, height)
    min_dim = min(width, height)
    
    if min_dim == 1:  # Definitely thin in one dimension
        thinness_due_to_width = 1.0 / width if width > 1 else 0
        thinness_due_to_height = 1.0 / height if height > 1 else 0
        return (thinness_due_to_width + thinness_due_to_height) / 2
    
    # 2. Area to perimeter ratio approximation
    # For a perfect square: perimeter = 4√area
    # For a line: perimeter ≈ 2*max_dim
    expected_square_perimeter = 4 * np.sqrt(area)
    actual_perimeter_approx = 2 * (width + height)
    
    # Ratio > 1 means more line-like, < 1 means more square-like
    perimeter_ratio = actual_perimeter_approx / expected_square_perimeter
    
    # 3. Combined score
    aspect_score = min_dim / max_dim  # 0=line, 1=square
    perimeter_score = 1.0 / perimeter_ratio  # Inverse
    
    # Blend: aspect ratio is more important
    thinness_score = 0.7 * aspect_score + 0.3 * perimeter_score
    
    return min(1.0, max(0.0, thinness_score))