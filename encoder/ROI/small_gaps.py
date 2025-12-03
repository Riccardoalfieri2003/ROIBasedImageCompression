from encoder.ROI.edges import compute_local_density
import numpy as np
import cv2

def bridge_small_gaps(binary_image, max_gap=3, density_threshold=0.3, 
                                   local_window=5, regional_window=25,
                                   method='density_aware'):
    """
    2D directional gap bridging that considers both local surroundings and regional density.
    
    Args:
        binary_image: Binary image (0 and 255)
        max_gap: Maximum gap size to bridge
        density_threshold: Minimum regional density to consider bridging
        local_window: Window size for local directionality check
        regional_window: Window size for regional density calculation
        method: 'density_aware', 'strict', or 'relaxed'
    
    Returns:
        bridged_image: Image with internal gaps bridged
    """
    bridged_image = binary_image.copy()
    
    # Calculate regional density (larger context)
    regional_density = compute_local_density(binary_image, regional_window)
    
    # Find candidate pixels: black pixels in dense regions
    candidates = (binary_image == 0) & (regional_density > density_threshold)
    
    if not np.any(candidates):
        return bridged_image
    
    # Find internal gaps based on method
    if method == 'strict':
        internal_gaps = find_internal_gaps_strict_2d(binary_image, candidates, max_gap, local_window)
    elif method == 'relaxed':
        internal_gaps = find_internal_gaps_relaxed_2d(binary_image, candidates, max_gap, local_window)
    else:  # 'density_aware'
        internal_gaps = find_internal_gaps_density_aware_2d(binary_image, candidates, max_gap, local_window)
    
    # Bridge the gaps
    bridged_image[internal_gaps] = 255
    
    print(f"Bridged {np.sum(internal_gaps)} internal gap pixels")
    return bridged_image

def find_internal_gaps_strict_2d(binary_image, candidates, max_gap, local_window):
    """
    Strict: pixel must be completely surrounded by white in local neighborhood.
    """
    internal_gaps = np.zeros_like(binary_image, dtype=bool)
    height, width = binary_image.shape
    
    for y in range(height):
        for x in range(width):
            if candidates[y, x]:
                if is_locally_surrounded_2d(binary_image, x, y, local_window, max_gap):
                    internal_gaps[y, x] = True
    
    return internal_gaps

def find_internal_gaps_relaxed_2d(binary_image, candidates, max_gap, local_window):
    """
    Relaxed: pixel must have white pixels in opposite directions within local area.
    """
    internal_gaps = np.zeros_like(binary_image, dtype=bool)
    height, width = binary_image.shape
    
    for y in range(height):
        for x in range(width):
            if candidates[y, x]:
                if has_white_in_opposite_directions_2d(binary_image, x, y, max_gap, local_window):
                    internal_gaps[y, x] = True
    
    return internal_gaps

def find_internal_gaps_density_aware_2d(binary_image, candidates, max_gap, local_window):
    """
    Density-aware: combine local connectivity with density gradient.
    """
    internal_gaps = np.zeros_like(binary_image, dtype=bool)
    height, width = binary_image.shape
    
    # Calculate local density for more precise analysis
    local_density = compute_local_density(binary_image, local_window)
    
    for y in range(height):
        for x in range(width):
            if candidates[y, x]:
                # Check if this is an internal gap using multiple criteria
                if is_internal_gap_2d(binary_image, local_density, x, y, max_gap, local_window):
                    internal_gaps[y, x] = True
    
    return internal_gaps

def is_locally_surrounded_2d(binary_image, x, y, window_size, max_gap):
    """
    Check if pixel is surrounded by white pixels in local neighborhood.
    """
    height, width = binary_image.shape
    half_window = window_size // 2
    
    # Define local neighborhood
    y_start = max(0, y - half_window)
    y_end = min(height, y + half_window + 1)
    x_start = max(0, x - half_window)
    x_end = min(width, x + half_window + 1)
    
    neighborhood = binary_image[y_start:y_end, x_start:x_end]
    
    # Count white pixels in neighborhood (excluding center)
    white_count = np.sum(neighborhood > 0)
    total_pixels = neighborhood.size - 1  # exclude center
    
    # Consider surrounded if most pixels in neighborhood are white
    return white_count / total_pixels > 0.7  # 70% white in local area

def has_white_in_opposite_directions_2d(binary_image, x, y, max_gap, local_window):
    """
    Check for white pixels in opposite directions within local constraints.
    """
    height, width = binary_image.shape
    
    # Define search patterns (pairs of opposite directions)
    direction_pairs = [
        [(-1, 0), (1, 0)],   # left-right
        [(0, -1), (0, 1)],   # up-down
        [(-1, -1), (1, 1)],  # diagonal 
        [(-1, 1), (1, -1)]   # anti-diagonal
    ]
    
    for pair in direction_pairs:
        found_both = True
        for dx, dy in pair:
            found_white = False
            for dist in range(1, max_gap + 1):
                nx, ny = x + dx * dist, y + dy * dist
                # Check if within local window and image bounds
                if (abs(nx - x) <= local_window and abs(ny - y) <= local_window and
                    0 <= nx < width and 0 <= ny < height):
                    if binary_image[ny, nx] > 0:
                        found_white = True
                        break
            if not found_white:
                found_both = False
                break
        
        if found_both:
            return True
    
    return False

def is_internal_gap_2d(binary_image, local_density, x, y, max_gap, local_window):
    """
    Comprehensive check for internal gaps using multiple criteria.
    """
    height, width = binary_image.shape
    
    # Criterion 1: Local density should be high around this pixel
    half_window = local_window // 2
    y_start = max(0, y - half_window)
    y_end = min(height, y + half_window + 1)
    x_start = max(0, x - half_window)
    x_end = min(width, x + half_window + 1)
    
    local_density_around = np.mean(local_density[y_start:y_end, x_start:x_end])
    if local_density_around < 0.4:  # Not dense enough locally
        return False
    
    # Criterion 2: Should have white pixels in multiple directions
    directions_with_white = 0
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        found_white = False
        for dist in range(1, max_gap + 1):
            nx, ny = x + dx * dist, y + dy * dist
            if 0 <= nx < width and 0 <= ny < height:
                if binary_image[ny, nx] > 0:
                    found_white = True
                    break
        if found_white:
            directions_with_white += 1
    
    # Criterion 3: Should not be on an edge (gradient check)
    is_on_edge = is_pixel_on_edge(binary_image, x, y, local_window)
    
    # Combine criteria
    return (directions_with_white >= 3) and not is_on_edge

def is_pixel_on_edge(binary_image, x, y, window_size):
    """
    Check if pixel is on an edge between regions using gradient.
    """
    height, width = binary_image.shape
    half_window = window_size // 2
    
    y_start = max(0, y - half_window)
    y_end = min(height, y + half_window + 1)
    x_start = max(0, x - half_window)
    x_end = min(width, x + half_window + 1)
    
    neighborhood = binary_image[y_start:y_end, x_start:x_end].astype(np.float32)
    
    if neighborhood.size < 4:  # Too small for gradient
        return False
    
    # Calculate gradient magnitude
    grad_x = cv2.Sobel(neighborhood, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(neighborhood, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # High gradient indicates edge
    return np.max(grad_mag) > 100
