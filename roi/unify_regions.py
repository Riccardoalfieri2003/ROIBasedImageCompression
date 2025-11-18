import math
import cv2
from matplotlib import pyplot as plt
import numpy as np





"""def unify_with_morphology(binary_image, gap_size=2, min_region_size=50):
    # Convert to binary
    binary = binary_image.copy() if binary_image.max() > 1 else binary_image * 255
    binary = (binary > 0).astype(np.uint8) * 255
    
    # Use closing to fill small gaps in existing edge regions
    kernel_size = max(1, gap_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Close small gaps within edge regions
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Only keep the filled pixels that are connected to original edges
    # This prevents creating new disconnected regions
    filled_gaps = cv2.bitwise_and(closed, cv2.bitwise_not(binary))
    
    # Only keep filled gaps that are surrounded by edges (dilate original and check)
    dilated_edges = cv2.dilate(binary, kernel, iterations=1)
    valid_fills = cv2.bitwise_and(filled_gaps, dilated_edges)
    
    # Combine original edges with valid filled gaps
    unified = cv2.bitwise_or(binary, valid_fills)
    
    # Remove small regions
    cleaned = remove_small_regions(unified, min_size=min_region_size)
    region_map = (cleaned > 0).astype(np.uint8)
    
    return cleaned, region_map"""


def unify_with_morphology(binary_image, gap_size=2, min_region_size=50):
    # Convert to binary
    binary = binary_image.copy() if binary_image.max() > 1 else binary_image * 255
    binary = (binary > 0).astype(np.uint8) * 255
    
    # Use closing to fill small gaps in existing edge regions
    kernel_size = max(1, gap_size)
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    kernel = cv2.getStructuringElement(cv2.MORPH_ERODE, (kernel_size, kernel_size))
    
    # Close small gaps within edge regions
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Only keep the filled pixels that are connected to original edges
    # This prevents creating new disconnected regions
    filled_gaps = cv2.bitwise_and(closed, cv2.bitwise_not(binary))
    
    # Only keep filled gaps that are surrounded by edges (dilate original and check)
    dilated_edges = cv2.dilate(binary, kernel, iterations=1)
    valid_fills = cv2.bitwise_and(filled_gaps, dilated_edges)
    
    # Combine original edges with valid filled gaps
    unified = cv2.bitwise_or(binary, valid_fills)
    
    # Remove small regions
    cleaned = remove_small_regions(unified, min_size=min_region_size)
    region_map = (cleaned > 0).astype(np.uint8)
    
    return cleaned, region_map




"""def unify_with_morphology(binary_image, gap_size=2, min_region_size=50, kernel_type='rect'):

    binary = binary_image.copy() if binary_image.max() > 1 else binary_image * 255
    binary = (binary > 0).astype(np.uint8) * 255
    
    # Choose different kernel types
    kernel_size = max(1, gap_size)
    
    if kernel_type == 'rect':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    elif kernel_type == 'cross':
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    elif kernel_type == 'ellipse':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    # Alternative approach: Use multiple iterations with smaller kernel
    if kernel_type == 'multi_pass':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        closed = binary.copy()
        for _ in range(gap_size):
            closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)
    else:
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # More conservative gap filling
    filled_gaps = cv2.bitwise_and(closed, cv2.bitwise_not(binary))
    
    # Stricter validation: gaps must be completely surrounded
    dilated_edges = cv2.dilate(binary, kernel, iterations=2)
    valid_fills = cv2.bitwise_and(filled_gaps, dilated_edges)
    
    # Only add gaps that don't create isolated islands
    from scipy import ndimage
    filled_labels, num_fills = ndimage.label(valid_fills)
    
    final_fills = np.zeros_like(valid_fills)
    for label in range(1, num_fills + 1):
        fill_region = (filled_labels == label)
        # Check if this fill region touches multiple edge components
        if np.sum(cv2.dilate(fill_region.astype(np.uint8), kernel) & binary) > np.sum(fill_region) * 2:
            final_fills[fill_region] = 255
    
    unified = cv2.bitwise_or(binary, final_fills)
    cleaned = remove_small_regions(unified, min_size=min_region_size)
    region_map = (cleaned > 0).astype(np.uint8)
    
    return cleaned, region_map
"""

def unify_with_distance_transform(binary_image, max_gap_distance=3, min_region_size=50):
    """
    Use distance transform to fill gaps that are very close to existing edges.
    More precise - only fills pixels that are within a specific distance of edges.
    """
    binary = binary_image.copy() if binary_image.max() > 1 else binary_image * 255
    binary = (binary > 0).astype(np.uint8)
    
    # Calculate distance to nearest edge
    distance_map = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 5)
    
    # Fill pixels that are very close to edges (small gaps)
    gap_mask = (distance_map > 0) & (distance_map <= max_gap_distance)
    
    # Only fill gaps that are surrounded by edges (avoid filling isolated areas)
    from scipy import ndimage
    
    # Label the gap regions
    gap_labels, num_gaps = ndimage.label(gap_mask)
    
    unified = binary.copy()
    
    for label in range(1, num_gaps + 1):
        gap_region = (gap_labels == label)
        
        # Check if this gap is surrounded by edges
        dilated_gap = cv2.dilate(gap_region.astype(np.uint8), np.ones((3, 3), np.uint8))
        edge_neighbors = cv2.bitwise_and(dilated_gap, binary)
        
        # Only fill if the gap is mostly surrounded by edges
        if np.sum(edge_neighbors) > np.sum(gap_region) * 0.7:  # 70% surrounded
            unified[gap_region] = 1
    
    unified = unified * 255
    cleaned = remove_small_regions(unified, min_size=min_region_size)
    region_map = (cleaned > 0).astype(np.uint8)
    
    return cleaned, region_map




def unify_with_connected_components(binary_image, min_gap_size=5, min_region_size=50):
    """
    Only fill gaps that are completely enclosed within edge regions.
    Most conservative - only fills holes that are fully surrounded.
    """
    binary = binary_image.copy() if binary_image.max() > 1 else binary_image * 255
    binary = (binary > 0).astype(np.uint8)
    
    # Find connected components of background (holes)
    background = 1 - binary
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(background, connectivity=4)
    
    unified = binary.copy()
    
    for i in range(1, num_labels):  # Skip background label 0
        hole_size = stats[i, cv2.CC_STAT_AREA]
        
        # Only fill small holes (gaps) within edge regions
        if hole_size <= min_gap_size:
            hole_mask = (labels == i).astype(np.uint8)
            
            # Check if this hole is completely surrounded by edges
            dilated_hole = cv2.dilate(hole_mask, np.ones((3, 3), np.uint8))
            # Remove the hole itself to get only the border
            border = dilated_hole - hole_mask
            
            # If the border is entirely edges, fill the hole
            border_pixels = border * binary
            if np.sum(border_pixels) == np.sum(border):
                unified[hole_mask.astype(bool)] = 1
    
    unified = unified * 255
    cleaned = remove_small_regions(unified, min_size=min_region_size)
    region_map = (cleaned > 0).astype(np.uint8)
    
    return cleaned, region_map


def unify_hybrid_method(binary_image, gap_size=2, min_region_size=50):
    """
    Combine morphological and connected component approaches.
    Good balance between filling real gaps and avoiding over-filling.
    """
    binary = binary_image.copy() if binary_image.max() > 1 else binary_image * 255
    binary = (binary > 0).astype(np.uint8)
    
    # Step 1: Small morphological closing for tiny gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap_size, gap_size))
    closed_small = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Step 2: Fill only completely enclosed small holes
    background = 1 - closed_small
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(background, connectivity=4)
    
    unified = closed_small.copy()
    
    for i in range(1, num_labels):
        hole_size = stats[i, cv2.CC_STAT_AREA]
        if hole_size <= 10:  # Only fill very small holes
            hole_mask = (labels == i).astype(np.uint8)
            dilated_hole = cv2.dilate(hole_mask, np.ones((3, 3), np.uint8))
            border = dilated_hole - hole_mask
            
            # Must be completely surrounded
            if np.sum(border * closed_small) == np.sum(border):
                unified[hole_mask.astype(bool)] = 1
    
    unified = unified * 255
    cleaned = remove_small_regions(unified, min_size=min_region_size)
    region_map = (cleaned > 0).astype(np.uint8)
    
    return cleaned, region_map


def extract_roi_nonroi(original_image, region_map):
    """
    Extract ROI and non-ROI regions from original image.
    
    Args:
        original_image: Original RGB image
        region_map: Binary region map (1=ROI, 0=non-ROI)
    
    Returns:
        roi_image: Image showing only ROI regions
        nonroi_image: Image showing only non-ROI regions
        roi_mask: Binary mask for ROI
        nonroi_mask: Binary mask for non-ROI
    """
    # Create masks
    roi_mask = (region_map == 1)
    nonroi_mask = (region_map == 0)
    
    # Create ROI image (keep ROI pixels, black out non-ROI)
    roi_image = original_image.copy()
    roi_image[~roi_mask] = 0  # Set non-ROI to black
    
    # Create non-ROI image (keep non-ROI pixels, black out ROI)
    nonroi_image = original_image.copy()
    nonroi_image[~nonroi_mask] = 0  # Set ROI to black
    
    return roi_image, nonroi_image, roi_mask, nonroi_mask


def process_and_unify_borders(edge_map, edge_density, original_image, density_threshold=0.3, 
                            unification_method='hybrid', min_region_size=50, **kwargs):
    # Step 1: Get high-density borders
    high_density_mask = edge_density > density_threshold
    intensity_borders = edge_map.copy()
    intensity_borders[~high_density_mask] = 0
    
    binary_borders = (intensity_borders > 0).astype(np.uint8) * 255
    
    # Step 2: Use selected unification method
    if unification_method == 'morphology':
        unified_borders, region_map = unify_with_morphology(binary_borders, gap_size=25 ,min_region_size=min_region_size, **kwargs)

        # Test different kernels in your main:
        """kernel_types = ['rect', 'cross', 'ellipse', 'multi_pass']

        for kernel_type in kernel_types:
            print(f"\n=== Testing {kernel_type} kernel ===")
            unified_borders, region_map = unify_with_morphology(
                binary_borders, 
                gap_size=25, 
                min_region_size=min_region_size,
                kernel_type=kernel_type
            )

            # Extract ROI and non-ROI
            roi_image, nonroi_image, roi_mask, nonroi_mask = extract_roi_nonroi(original_image, region_map)
            
            # Create comprehensive visualization
            visualize_unification_process(original_image, binary_borders, unified_borders, region_map, roi_mask, unification_method)"""
                




    elif unification_method == 'distance':
        unified_borders, region_map = unify_with_distance_transform(
            binary_borders, max_gap_distance=100, min_region_size=min_region_size, **kwargs)
    elif unification_method == 'components':
        unified_borders, region_map = unify_with_connected_components(
            binary_borders, min_region_size=min_region_size, **kwargs)
    elif unification_method == 'hybrid':
        unified_borders, region_map = unify_hybrid_method(
            binary_borders, min_region_size=min_region_size, **kwargs)
    """else:
        # Fallback to original
        unified_borders, region_map = unify_black_pixels_in_white_regions(
            binary_borders, min_region_size=min_region_size, **kwargs)"""
    
    # Extract ROI and non-ROI
    roi_image, nonroi_image, roi_mask, nonroi_mask = extract_roi_nonroi(original_image, region_map)
    
    # Create comprehensive visualization
    visualize_unification_process(original_image, binary_borders, unified_borders, region_map, roi_mask, unification_method)
    
    
    
    
    print(f"Method: {unification_method}, ROI coverage: {np.sum(roi_mask)/roi_mask.size*100:.1f}%")
    
    return unified_borders, region_map, roi_image, nonroi_image, roi_mask, nonroi_mask


def visualize_unification_process(original_image, binary_borders, unified_borders, 
                                region_map, roi_mask, method_name):
    """
    Visualize the entire unification process with overlays.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Edge processing pipeline
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(binary_borders, cmap='gray')
    initial_edges = np.sum(binary_borders > 0)
    axes[0, 1].set_title(f'High-Density Edges\n{initial_edges} pixels')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(unified_borders, cmap='gray')
    final_edges = np.sum(unified_borders > 0)
    added_edges = final_edges - initial_edges
    axes[0, 2].set_title(f'After {method_name}\n{final_edges} pixels (+{added_edges})')
    axes[0, 2].axis('off')
    
    # Row 2: ROI overlays
    axes[1, 0].imshow(original_image)
    # Overlay initial edges in blue
    edge_overlay_initial = np.zeros_like(original_image)
    edge_overlay_initial[binary_borders > 0] = [0, 0, 255]  # Blue for initial edges
    axes[1, 0].imshow(edge_overlay_initial, alpha=0.7)
    axes[1, 0].set_title('Initial Edges (Blue)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(original_image)
    # Overlay added edges in green
    added_edges_mask = (unified_borders > 0) & (binary_borders == 0)
    edge_overlay_added = np.zeros_like(original_image)
    edge_overlay_added[added_edges_mask] = [0, 255, 0]  # Green for added edges
    axes[1, 1].imshow(edge_overlay_added, alpha=0.7)
    axes[1, 1].set_title(f'Added Edges (Green)\n{np.sum(added_edges_mask)} pixels')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(original_image)
    # Overlay final ROI in red
    roi_overlay = np.zeros_like(original_image)
    roi_overlay[roi_mask] = [255, 0, 0]  # Red for ROI
    axes[1, 2].imshow(roi_overlay, alpha=0.5)
    roi_coverage = np.sum(roi_mask) / roi_mask.size * 100
    axes[1, 2].set_title(f'Final ROI (Red)\n{roi_coverage:.1f}% coverage')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed statistics
    print(f"\n=== {method_name.upper()} Method Statistics ===")
    print(f"Initial edge pixels: {initial_edges}")
    print(f"Final edge pixels: {final_edges}")
    print(f"Pixels added by unification: {added_edges}")
    print(f"ROI coverage: {roi_coverage:.1f}%")
    print(f"ROI pixels: {np.sum(roi_mask)}")








def compute_local_density(binary_map, kernel_size=15):
    """
    Compute local density of non-zero pixels in a binary map.
    
    Args:
        binary_map: Binary image (0s and 1s or 0s and 255s)
        kernel_size: Size of the local neighborhood
    
    Returns:
        density_map: Float array with values 0-1 representing local density
    """
    # Ensure binary map is in 0-1 range
    if binary_map.max() > 1:
        binary_map = binary_map / 255.0
    
    # Create a normalized kernel (sums to 1)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)
    kernel /= kernel.sum()
    
    # Compute local density using convolution
    density_map = cv2.filter2D(binary_map.astype(np.float32), -1, kernel)
    
    return density_map





def suggest_automatic_threshold(edge_density_map, edge_map, method='mean'):
    """
    Suggest automatic threshold based on edge density statistics.
    
    Args:
        edge_density_map: Density map from compute_local_density
        edge_map: Canny edge map
        method: 'mean', 'median', or 'percentile'
    
    Returns:
        suggested_threshold: Automatic threshold value
    """
    # Get density values only at edge locations
    edge_mask = edge_map > 0
    edge_density_values = edge_density_map[edge_mask]
    
    if len(edge_density_values) == 0:
        return 0.1  # Default fallback
    
    if method == 'mean':
        threshold = np.mean(edge_density_values)
    elif method == 'median':
        threshold = np.median(edge_density_values)
    elif method == 'percentile':
        threshold = np.percentile(edge_density_values, 70)  # 70th percentile
    else:
        threshold = np.mean(edge_density_values)
    
    return threshold


















def remove_small_regions(binary_image, min_size=50, remove_thin_lines=True, kernel_size=3):
    """
    Remove small connected components and thin lines from binary image.
    
    Args:
        binary_image: Binary image (0 and 255)
        min_size: Minimum number of pixels for a region to be kept
        remove_thin_lines: Whether to remove single-pixel lines
        kernel_size: Size of morphological kernel
    
    Returns:
        cleaned_image: Binary image with small regions and thin lines removed
    """
    # Find all connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    
    # Create output image
    cleaned_image = np.zeros_like(binary_image)
    
    # Iterate through all components (skip background at index 0)
    for i in range(1, num_labels):
        # Check if the region is large enough
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            # Keep this region
            cleaned_image[labels == i] = 255
    
    # Remove thin lines and single pixels using morphological operations
    if remove_thin_lines:
        cleaned_image = remove_thin_structures(cleaned_image, kernel_size)
    
    return cleaned_image

def remove_thin_structures(binary_image, kernel_size=3):
    """
    Remove single-pixel lines and isolated pixels using morphological operations.
    
    Args:
        binary_image: Binary image (0 and 255)
        kernel_size: Size of the morphological kernel
    
    Returns:
        cleaned_image: Binary image without thin structures
    """
    # Create morphological kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Method 1: Opening to remove small objects and thin lines
    cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    
    # Method 2: Additional erosion to break thin connections
    eroded = cv2.erode(cleaned, kernel, iterations=1)
    
    # Method 3: Reconstruction to restore original shapes but without thin parts
    # This preserves larger regions while removing thin extensions
    reconstructed = reconstruct_larger_regions(eroded, binary_image, kernel)
    
    return reconstructed

def reconstruct_larger_regions(eroded_image, original_image, kernel):
    """
    Reconstruct regions while preserving only substantial parts.
    """
    # Use dilation to rebuild regions, but only where they were substantial
    marker = eroded_image.copy()
    reconstructed = cv2.dilate(marker, kernel, iterations=1)
    reconstructed = cv2.bitwise_and(reconstructed, original_image)
    
    # Repeat to ensure good reconstruction
    for _ in range(2):
        reconstructed = cv2.dilate(reconstructed, kernel, iterations=1)
        reconstructed = cv2.bitwise_and(reconstructed, original_image)
    
    return reconstructed











from edges import get_edge_map

if __name__ == "__main__":
    image_name = 'images/Lenna.webp'
    image = cv2.imread(image_name)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print(f"Size: {image_rgb.size}")
    factor = math.ceil(math.log(image_rgb.size, 10)) * math.log(image_rgb.size)
    print(f"factor: {factor}")
    
    

    # Compare with your ROI detection
    edge_map = get_edge_map(image_rgb)
    edge_density = compute_local_density(edge_map, kernel_size=3)

    threshold = suggest_automatic_threshold(edge_density, edge_map, method="mean") /2.5
    
    window_size = math.floor(factor)
    min_region_size= math.ceil( image_rgb.size / math.pow(10, math.ceil(math.log(image_rgb.size, 10))-3 ) )
    print(f"min_region_size: {min_region_size}")

    print(f"\nWindow: {window_size}x{window_size}, Threshold: {threshold:.3f} ===")
    
    # Test different methods
    #methods = ['morphology', 'distance', 'components', 'hybrid']
    methods = ['morphology']

    for method in methods:

        print(f"\n=== Testing {method} method ===")
        unified, regions, roi_image, nonroi_image, roi_mask, nonroi_mask = process_and_unify_borders(
            edge_map, edge_density, image_rgb,
            density_threshold=threshold,
            unification_method=method,
            min_region_size=min_region_size
        )