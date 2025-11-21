import math
import cv2
from matplotlib import pyplot as plt
import numpy as np

from edges import get_edge_map
from clahe import get_enhanced_image



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



def process_and_unify_borders(edge_map, edge_density, original_image, 
                            density_threshold=0.3,  # Your existing parameter
                            # NEW PARAMETERS TO CONTROL ROI DETECTION:
                            border_sensitivity=0.3,    # LOWER = less strict borders
                            min_region_size=30,        # SMALLER = keep more regions  
                            max_gap_to_bridge=10,      # LARGER = bridge more gaps
                            noise_aggressiveness=2,    # SMALLER = less noise removal
                            unification_strength=0.4): # Controls how much to unify
    """
    Complete pipeline: filter edges by density, then unify regions, remove small regions.
    Returns ROI and non-ROI separately.
    """
    # Step 1: Get high-density borders (your existing approach)
    high_density_mask = edge_density > density_threshold
    intensity_borders = edge_map.copy()
    intensity_borders[~high_density_mask] = 0
    
    # Convert to binary (0 and 255)
    binary_borders = (intensity_borders > 0).astype(np.uint8) * 255
    
    binary_thin_bordersless=remove_thin_structures(binary_borders, density_threshold=0.15, thinness_threshold=0.3, window_size=25, min_region_size=10)

    plt.imshow(binary_thin_bordersless)
    plt.show()

    

    noiseless_binary_borders=remove_small_noise_regions(binary_thin_bordersless, min_size=75)
    #noiseless_binary_borders=remove_small_noise_regions(noiseless_binary_borders, min_size=1)

    plt.imshow(noiseless_binary_borders)
    plt.show()


    # Step 2: Use the new directional region unification
    unified_borders, region_map = directional_region_unification(
        noiseless_binary_borders,  # This was the missing variable - using binary_borders instead of binary_image
        border_sensitivity=border_sensitivity,
        min_region_size=min_region_size,
        max_gap_to_bridge=max_gap_to_bridge
    )
        
    # Extract ROI and non-ROI
    roi_image, nonroi_image, roi_mask, nonroi_mask = extract_roi_nonroi(original_image, region_map)
    
    # Create visualization
    visualize_roi_nonroi_comparison(original_image, roi_image, nonroi_image, region_map)
    
    print(f"=== ROI/non-ROI Statistics ===")
    print(f"ROI coverage: {np.sum(roi_mask)} pixels ({np.sum(roi_mask)/roi_mask.size*100:.1f}%)")
    print(f"non-ROI coverage: {np.sum(nonroi_mask)} pixels ({np.sum(nonroi_mask)/nonroi_mask.size*100:.1f}%)")
    
    return unified_borders, region_map, roi_image, nonroi_image, roi_mask, nonroi_mask


def visualize_roi_nonroi_comparison(original_image, roi_image, nonroi_image, region_map):
    """
    Visualize ROI and non-ROI regions side by side.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Original and masks
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(region_map, cmap='tab10')
    axes[0, 1].set_title('Region Map\n(1=ROI, 0=non-ROI)')
    axes[0, 1].axis('off')
    
    # Overlay visualization
    axes[0, 2].imshow(original_image)
    overlay = np.zeros_like(original_image)
    overlay[region_map == 1] = [255, 0, 0]  # Red for ROI
    axes[0, 2].imshow(overlay, alpha=0.6)
    axes[0, 2].set_title('ROI Overlay (Red)')
    axes[0, 2].axis('off')
    
    # Row 2: ROI and non-ROI extracted
    axes[1, 0].imshow(roi_image)
    roi_pixels = np.sum(region_map == 1)
    axes[1, 0].set_title(f'ROI Regions\n{roi_pixels} pixels')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(nonroi_image)
    nonroi_pixels = np.sum(region_map == 0)
    axes[1, 1].set_title(f'non-ROI Regions\n{nonroi_pixels} pixels')
    axes[1, 1].axis('off')
    
    # Statistics
    axes[1, 2].axis('off')
    axes[1, 2].text(0.1, 0.9, 'Region Statistics:', fontsize=12, fontweight='bold')
    axes[1, 2].text(0.1, 0.7, f'Total pixels: {original_image.shape[0] * original_image.shape[1]}', fontsize=10)
    axes[1, 2].text(0.1, 0.6, f'ROI pixels: {roi_pixels} ({roi_pixels/(roi_pixels+nonroi_pixels)*100:.1f}%)', fontsize=10)
    axes[1, 2].text(0.1, 0.5, f'non-ROI pixels: {nonroi_pixels} ({nonroi_pixels/(roi_pixels+nonroi_pixels)*100:.1f}%)', fontsize=10)
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].set_facecolor('lightgray')
    
    plt.tight_layout()
    plt.show()

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









































import numpy as np
import cv2
from scipy import ndimage
from skimage import morphology

def directional_region_unification(binary_image, 
                                 border_sensitivity=0.3,      # KEY: Lower = fewer borders detected
                                 min_region_size=30,          # KEY: Smaller = keep more small regions
                                 max_gap_to_bridge=50,        # KEY: Larger = connect more regions
                                 noise_aggressiveness=2,      # KEY: Smaller = remove less noise
                                 unification_strength=0.4):   # KEY: Higher = unify more aggressively
    """
    Unify regions while respecting natural borders and directional context.
    
    Args:
        binary_image: Binary image (0 and 255)
        border_sensitivity: How sensitive to detect borders (0-1, higher = more sensitive)
        min_region_size: Minimum number of pixels for a region to be kept
        noise_removal_kernel: Size of kernel for noise removal
        max_gap_to_bridge: Maximum gap size to bridge between regions
    
    Returns:
        unified_image: Binary image with unified regions
        region_map: Array showing unified regions
    """
    # Ensure binary image is 0-255 uint8
    if binary_image.max() <= 1:
        binary_image = (binary_image * 255).astype(np.uint8)

    
    
    # Step 1: Initial noise removal
    #denoised = remove_small_noise_regions(binary_image, min_size=min_region_size//10)
    #denoised = remove_small_noise_regions(binary_image, min_size=min_region_size)

    denoised=binary_image
    
    # Step 2: Detect strong borders using gradient
    border_mask = detect_meaningful_borders(denoised, sensitivity=0.5)

    plt.imshow(border_mask)
    plt.show()
    
    # Step 3: Protect borders from being unified
    protected_image = protect_border_regions(denoised, border_mask)
    
    # Step 4: Bridge small gaps within regions (not across borders)
    bridged_image = bridge_small_gaps(protected_image, border_mask, max_gap=5)
    #bridged_image = bridge_small_gaps(bridged_image, border_mask, max_gap=20)

    plt.imshow(bridged_image)
    plt.show()

    #bridged_image=binary_image
   
    # Step 5: Final region cleaning
    #cleaned_image = remove_small_regions(bridged_image, min_size=min_region_size/10, kernel_size=1)
    #cleaned_image=bridged_image


    """cleaned_image = remove_thin_structures(
        bridged_image, 
        #min_size=5,              # Much lower than 50!
        #remove_thin_lines=True,  # Disable initially
        kernel_size=25            # Small kernel
    )

    cleaned_image=remove_small_regions(
        cleaned_image,
        min_size=5,
        remove_thin_lines=True,
        kernel_size=30
    )"""


    # INSTEAD OF:
    # cleaned_image = remove_thin_structures(bridged_image, kernel_size=25)
    # cleaned_image = remove_small_regions(cleaned_image, min_size=5, remove_thin_lines=True, kernel_size=30)

    # USE THIS:
    cleaned_image = contextual_region_cleaning(
        bridged_image,
        thin_kernel_size=3,           # For thin structure removal (3-7)
        min_relative_size=0.01,       # 2% of parent region size
        absolute_min_size=100,         # Absolute minimum size
        connectivity=8
    )

    #plt.imshow(cleaned_image)
    #plt.show()
    
    # Create region map
    region_map = (cleaned_image > 0).astype(np.uint8)
    
    return cleaned_image, region_map

def detect_meaningful_borders(binary_image, sensitivity=0.7):
    """
    Detect meaningful borders using gradient analysis and structural patterns.
    
    Args:
        binary_image: Binary image (0 and 255)
        sensitivity: Border detection sensitivity (0-1)
    
    Returns:
        border_mask: Binary mask of meaningful borders
    """
    # Convert to float for processing
    img_float = binary_image.astype(np.float32) / 255.0
    
    # Calculate gradient magnitude using Sobel
    grad_x = cv2.Sobel(img_float, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_float, cv2.CV_32F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize gradient
    if gradient_mag.max() > 0:
        gradient_mag = gradient_mag / gradient_mag.max()
    
    # Threshold gradient based on sensitivity
    gradient_threshold = sensitivity * 0.5
    strong_edges = gradient_mag > gradient_threshold
    
    # Use morphological operations to enhance border continuity
    kernel = np.ones((3, 3), np.uint8)
    enhanced_edges = cv2.morphologyEx(strong_edges.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    
    # Find regions where we have significant gradient changes
    # These represent meaningful borders between different regions
    border_regions = enhanced_edges.astype(bool)
    
    # Dilate borders to create protection zone
    border_dilation = cv2.dilate(border_regions.astype(np.uint8), kernel, iterations=2)
    
    return border_dilation > 0

def protect_border_regions(binary_image, border_mask, kernel_size=18):
    """
    Protect border regions from unification while allowing internal noise removal.
    
    Args:
        binary_image: Binary image (0 and 255)
        border_mask: Mask of regions to protect
    
    Returns:
        protected_image: Image with protected borders
    """
    protected_image = binary_image.copy()
    
    # Create a safe zone around borders where we won't modify pixels
    safe_zone = border_mask
    
    # For black pixels in white regions (potential noise), only remove them
    # if they're not near meaningful borders
    white_regions = binary_image > 0
    black_pixels = binary_image == 0
    
    # Find black pixels that are completely surrounded by white (internal noise)
    # and not near borders
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    white_neighborhood = cv2.morphologyEx(white_regions.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    
    # Internal black pixels (potential noise) are those surrounded by white
    # but not in border regions
    internal_black_pixels = black_pixels & (white_neighborhood > 0) & (~safe_zone)
    
    # Convert internal noise to white
    protected_image[internal_black_pixels] = 255
    
    return protected_image

def bridge_small_gaps(binary_image, border_mask, max_gap=5):
    """
    Bridge small gaps within regions while respecting borders.
    
    Args:
        binary_image: Binary image (0 and 255)
        border_mask: Mask of protected border regions
        max_gap: Maximum gap size to bridge
    
    Returns:
        bridged_image: Image with small gaps bridged
    """
    bridged_image = binary_image.copy()
    
    # Create safe working area (not near borders)
    """safe_area = ~border_mask
    
    # Only bridge gaps in safe areas
    if np.any(safe_area):
        # Use morphological closing to bridge small gaps, but only in safe areas
        #kernel_size = max(1, min(max_gap, 5))
        kernel_size = max_gap
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Apply closing only to safe areas
        safe_region = bridged_image.copy()
        safe_region[~safe_area] = 0  # Mask out border areas
        
        closed_safe = cv2.morphologyEx(safe_region, cv2.MORPH_CLOSE, kernel)
        
        # Combine results: use closed version in safe areas, original near borders
        bridged_image[safe_area] = closed_safe[safe_area]"""
    
    kernel_size = max_gap
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Apply closing only to safe areas
    safe_region = bridged_image.copy()
    #safe_region[~safe_area] = 0  # Mask out border areas
    
    closed_safe = cv2.morphologyEx(safe_region, cv2.MORPH_GRADIENT, kernel)
    
    # Combine results: use closed version in safe areas, original near borders
    #bridged_image[safe_area] = closed_safe[safe_area]

    return closed_safe

def remove_small_noise_regions(binary_image, min_size=5):
    """
    Remove very small noise regions before main processing.
    
    Args:
        binary_image: Binary image (0 and 255)
        min_size: Minimum size for noise regions
    
    Returns:
        denoised_image: Image with small noise removed
    """
    # Remove small white noise
    denoised_white = remove_small_components(binary_image, min_size, foreground=255)
    
    # Remove small black noise in white regions
    # Invert to treat black regions as foreground for removal
    inverted = 255 - denoised_white
    denoised_black = remove_small_components(inverted, min_size, foreground=255)
    
    # Convert back
    denoised_image = 255 - denoised_black
    
    return denoised_image

def remove_small_components(binary_image, min_size, foreground=255):
    """
    Remove small connected components from binary image.
    
    Args:
        binary_image: Binary image
        min_size: Minimum component size
        foreground: Value representing foreground
    
    Returns:
        cleaned_image: Image with small components removed
    """
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (binary_image == foreground).astype(np.uint8), connectivity=8
    )
    
    # Create mask for components large enough to keep
    component_mask = np.zeros_like(binary_image, dtype=np.uint8)
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            component_mask[labels == i] = 255
    
    # Preserve the original values for kept components
    if foreground == 255:
        cleaned_image = component_mask
    else:
        cleaned_image = 255 - component_mask
    
    return cleaned_image

# Keep your existing remove_small_regions function
def remove_small_regions(binary_image, min_size=10, remove_thin_lines=False, kernel_size=3):
   
    # For sparse edge images, use closing to connect nearby edges
    kernel = np.ones((3, 3), np.uint8)
    
    # Connect nearby edge pixels
    connected_edges = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    
    # Only then remove very small components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(connected_edges, connectivity=8)
    
    cleaned_image = np.zeros_like(binary_image)
    
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            cleaned_image[labels == i] = 255
    
    return cleaned_image
















"""
def remove_thin_structures(binary_image, kernel_size=2, method='erosion', density_threshold=0.3, window_size=15):
    if np.sum(binary_image > 0) == 0:
        return binary_image
    
    # Calculate local density map
    density_map = calculate_local_density(binary_image, window_size)
    
    # Identify thin structures
    thin_structures_mask = identify_thin_structures(binary_image, method, kernel_size)
    
    # Only remove thin structures in low-density areas
    low_density_areas = density_map < density_threshold
    thin_structures_to_remove = thin_structures_mask & low_density_areas
    
    # Remove only the identified thin structures in low-density regions
    cleaned_image = binary_image.copy()
    cleaned_image[thin_structures_to_remove] = 0
    
    print(f"Removed {np.sum(thin_structures_to_remove)} thin pixels in low-density areas")
    
    return cleaned_image

def calculate_local_density(binary_image, window_size=15):

    # Convert to float for processing
    binary_float = (binary_image > 0).astype(np.float32)
    
    # Create averaging kernel
    kernel = np.ones((window_size, window_size), dtype=np.float32)
    kernel /= np.sum(kernel)
    
    # Compute local density using convolution
    density_map = cv2.filter2D(binary_float, -1, kernel)
    
    return density_map

def identify_thin_structures(binary_image, method='erosion', kernel_size=2):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    if method == 'erosion':
        # Thin structures are those that disappear after erosion+dilation
        eroded = cv2.erode(binary_image, kernel, iterations=1)
        dilated = cv2.dilate(eroded, kernel, iterations=1)
        thin_mask = (binary_image > 0) & (dilated == 0)
        
    elif method == 'opening':
        # Thin structures removed by opening
        opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
        thin_mask = (binary_image > 0) & (opened == 0)
        
    elif method == 'skeleton':
        # Identify skeleton endpoints and isolated pixels
        thin_mask = identify_skeleton_thin_parts(binary_image)
    
    elif method == 'neighbor_count':
        # Thin structures have few neighbors
        thin_mask = identify_low_connectivity_pixels(binary_image)
    
    return thin_mask

def identify_skeleton_thin_parts(binary_image):

    kernel = np.ones((3, 3), np.uint8)
    
    # Count neighbors for each pixel
    neighbor_count = cv2.filter2D(binary_image / 255, -1, kernel) - (binary_image / 255)
    
    # Endpoints have exactly 1 neighbor, isolated pixels have 0
    endpoints = (binary_image > 0) & (neighbor_count <= 1)
    
    return endpoints

def identify_low_connectivity_pixels(binary_image, min_neighbors=2):

    kernel = np.ones((3, 3), np.uint8)
    
    # Count neighbors for each pixel
    neighbor_count = cv2.filter2D(binary_image / 255, -1, kernel) - (binary_image / 255)
    
    # Pixels with too few neighbors
    low_connectivity = (binary_image > 0) & (neighbor_count < min_neighbors)
    
    return low_connectivity


"""





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
    density_map = calculate_local_density(binary_image, window_size)
    
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

def identify_elongated_regions(binary_image, labels, stats, elongation_threshold=0.2):
    """
    Alternative method: Identify elongated regions.
    """
    elongated_mask = np.zeros_like(binary_image, dtype=bool)
    
    for i in range(1, len(stats)):
        region_mask = (labels == i)
        
        # Fit an ellipse to the region
        points = np.column_stack(np.where(region_mask))
        if len(points) < 5:  # Need minimum points for ellipse fitting
            continue
            
        ellipse = cv2.fitEllipse(points)
        (center, axes, angle) = ellipse
        
        # Calculate elongation (ratio of minor to major axis)
        major_axis = max(axes)
        minor_axis = min(axes)
        elongation = minor_axis / major_axis if major_axis > 0 else 0
        
        if elongation < elongation_threshold:
            elongated_mask[region_mask] = True
    
    return elongated_mask

def calculate_local_density(binary_image, window_size=25):
    """
    Calculate local density of white pixels.
    """
    binary_float = (binary_image > 0).astype(np.float32)
    kernel = np.ones((window_size, window_size), dtype=np.float32)
    kernel /= np.sum(kernel)
    density_map = cv2.filter2D(binary_float, -1, kernel)
    
    print(f"Density stats - Min: {density_map.min():.3f}, Max: {density_map.max():.3f}, Mean: {density_map.mean():.3f}")
    return density_map



































def contextual_region_cleaning(binary_image, 
                             thin_kernel_size=3,
                             min_relative_size=0.02,  # 2% of parent region size
                             absolute_min_size=10,
                             connectivity=8):
    """
    Contextually clean regions based on hierarchical relationships.
    
    Args:
        binary_image: Binary image (0=nonROI, 255=ROI)
        thin_kernel_size: Kernel size for thin structure removal
        min_relative_size: Minimum relative size to keep (0-1)
        absolute_min_size: Absolute minimum size in pixels
        connectivity: 4 or 8 connectivity
    
    Returns:
        cleaned_image: Contextually cleaned binary image
    """
    # Step 1: Remove thin structures first
    cleaned = remove_thin_structures_contextual(binary_image, kernel_size=thin_kernel_size)
    plt.imshow(cleaned)
    plt.show()
    

    # Step 2: Find all connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=connectivity)
    print(num_labels)
    
    if num_labels <= 1:
        return cleaned
    
    # Step 3: Create hierarchy map
    hierarchy_map = build_region_hierarchy(labels, stats)
    
    # Step 4: Contextual cleaning based on hierarchy
    final_image = apply_contextual_cleaning(cleaned, labels, stats, hierarchy_map, min_relative_size, absolute_min_size)
    
    return final_image

def remove_thin_structures_contextual(binary_image, kernel_size=3):
    """
    Remove thin structures while preserving important boundaries.
    """
    if np.sum(binary_image > 0) == 0:
        return binary_image
    
    # Use gentle opening to remove thin structures
    kernel = cv2.getStructuringElement(cv2.MORPH_CLOSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

    plt.imshow(cleaned)
    plt.show()
    
    return cleaned

def build_region_hierarchy(labels, stats):
    """
    Build hierarchy to understand which regions contain which other regions.
    Returns a dictionary mapping region_id to its containing region_id.
    """
    hierarchy = {}
    height, width = labels.shape
    
    # For each region, find which other region contains its centroid
    for region_id in range(1, len(stats)):
        centroid_x = stats[region_id, cv2.CC_STAT_LEFT] + stats[region_id, cv2.CC_STAT_WIDTH] // 2
        centroid_y = stats[region_id, cv2.CC_STAT_TOP] + stats[region_id, cv2.CC_STAT_HEIGHT] // 2
        
        # Ensure centroid is within bounds
        centroid_x = max(0, min(centroid_x, width-1))
        centroid_y = max(0, min(centroid_y, height-1))
        
        # Find which region contains the centroid
        containing_region = labels[centroid_y, centroid_x]
        
        if containing_region != region_id and containing_region != 0:
            hierarchy[region_id] = containing_region
    
    return hierarchy

def apply_contextual_cleaning(binary_image, labels, stats, hierarchy, 
                            min_relative_size, absolute_min_size):
    """
    Apply contextual cleaning based on region hierarchy and relative sizes.
    """
    cleaned_image = np.zeros_like(binary_image)
    
    # First pass: identify regions to keep based on hierarchy
    regions_to_keep = set()
    regions_to_convert = {}  # region_id -> new_value
    
    for region_id in range(1, len(stats)):
        region_area = stats[region_id, cv2.CC_STAT_AREA]
        region_value = binary_image[labels == region_id][0] if np.any(labels == region_id) else 0
        
        # Check if this region is inside another region
        if region_id in hierarchy:
            parent_id = hierarchy[region_id]
            parent_area = stats[parent_id, cv2.CC_STAT_AREA]
            parent_value = binary_image[labels == parent_id][0] if np.any(labels == parent_id) else 0
            
            # Case 1: nonROI inside ROI
            if region_value == 0 and parent_value == 255:
                relative_size = region_area / parent_area
                if relative_size < min_relative_size or region_area < absolute_min_size:
                    # Convert small nonROI inside ROI to ROI
                    regions_to_convert[region_id] = 255
                else:
                    regions_to_keep.add(region_id)
            
            # Case 2: ROI inside nonROI  
            elif region_value == 255 and parent_value == 0:
                relative_size = region_area / parent_area
                if relative_size < min_relative_size or region_area < absolute_min_size:
                    # Convert small ROI inside nonROI to nonROI
                    regions_to_convert[region_id] = 0
                else:
                    regions_to_keep.add(region_id)
            
            else:
                regions_to_keep.add(region_id)
        
        else:
            # Independent region - keep based on absolute size
            if region_area >= absolute_min_size:
                regions_to_keep.add(region_id)
    
    # Second pass: apply the decisions
    for region_id in range(1, len(stats)):
        if region_id in regions_to_keep:
            # Keep original value
            mask = labels == region_id
            if np.any(mask):
                original_value = binary_image[mask][0]
                cleaned_image[mask] = original_value
        elif region_id in regions_to_convert:
            # Convert to new value
            cleaned_image[labels == region_id] = regions_to_convert[region_id]
        else:
            # Remove small independent regions
            if stats[region_id, cv2.CC_STAT_AREA] >= absolute_min_size:
                cleaned_image[labels == region_id] = binary_image[labels == region_id][0]
            # else: region is removed (stays 0)
    
    return cleaned_image








































if __name__ == "__main__":
    image_name = 'images/kauai.jpg'
    image = cv2.imread(image_name)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb=get_enhanced_image(image_rgb, shadow_threshold=100)

    print(f"Size: {image_rgb.size}")
    factor = math.ceil(math.log(image_rgb.size, 10)) * math.log(image_rgb.size)
    print(f"factor: {factor}")
    
    

    # Compare with your ROI detection
    edge_map = get_edge_map(image_rgb)
    edge_density = compute_local_density(edge_map, kernel_size=3)

    """    # Method 1: Simplest
    plt.imshow(edge_density)
    plt.show()"""



    threshold = suggest_automatic_threshold(edge_density, edge_map, method="mean") /100
    
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
            #unification_method=method,
            min_region_size=min_region_size
        )