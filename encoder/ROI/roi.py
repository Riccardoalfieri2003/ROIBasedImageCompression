import math
import cv2
from matplotlib import pyplot as plt
import numpy as np

from encoder.ROI.edges import compute_local_density, suggest_automatic_threshold, get_edge_map
from encoder.ROI.small_regions import remove_small_regions, connect_nearby_pixels
from encoder.ROI.thin_regions import remove_thin_structures
from encoder.ROI.small_gaps import bridge_small_gaps




def get_regions(image_rgb):

    print(f"Size: {image_rgb.size}")
    factor = math.ceil(math.log(image_rgb.size, 10)) * math.log(image_rgb.size)
    print(f"factor: {factor}")


    edge_map = get_edge_map(image_rgb)
    edge_density = compute_local_density(edge_map, kernel_size=3)


    threshold = suggest_automatic_threshold(edge_density, edge_map, method="mean") / 100
    
    window_size = math.floor(factor)
    min_region_size= math.ceil( image_rgb.size / math.pow(10, math.ceil(math.log(image_rgb.size, 10))-3 ) ) 
    print(f"min_region_size: {min_region_size}")

    print(f"\nWindow: {window_size}x{window_size}, Threshold: {threshold:.3f} ===")

    unified, regions, roi_image, nonroi_image, roi_mask, nonroi_mask = process_and_unify_borders(
        edge_map, edge_density, image_rgb,
        density_threshold=threshold,
        min_region_size=min_region_size
    )

    return unified, regions, roi_image, nonroi_image, roi_mask, nonroi_mask


from skimage.measure import label, regionprops
def extract_regions(image_rgb, roi_mask, nonroi_mask):
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

    return roi_regions, nonroi_regions




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
    
    #binary_thin_bordersless=remove_thin_structures(binary_borders, density_threshold=0.10, thinness_threshold=0.3, window_size=25, min_region_size=25)

    binary_thin_bordersless=binary_borders
    noiseless_binary_borders=remove_small_noise_regions(binary_thin_bordersless, min_size=75)

    # Skeleton-based connection
    connected_1 = connect_nearby_pixels(
        noiseless_binary_borders,
        connection_distance=25,
        method='skeleton',
        min_region_size=25
    )

    connected = bridge_small_gaps(connected_1, max_gap=100, density_threshold=0.2, local_window=15, regional_window=25, method="relaxed")



    plt.subplot(1, 2, 1)  # 2 rows, 2 columns, first position
    plt.imshow(connected_1)  
    plt.axis('off')  # Hide the axis labels
    plt.title("connected_1") 

    # Add the second image to the figure (top-right position)
    plt.subplot(1, 2, 2)  # 2 rows, 2 columns, second position
    plt.imshow(connected)  
    plt.axis('off')  # Hide the axis labels
    plt.title("connected") 

    plt.show()


    

    # Step 2: Use the new directional region unification
    unified_borders, region_map = directional_region_unification(
        connected,  # This was the missing variable - using binary_borders instead of binary_image
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


    denoised=binary_image
    
    # Step 2: Detect strong borders using gradient
    border_mask = detect_meaningful_borders(denoised, sensitivity=0.5)

    # Step 3: Protect borders from being unified
    protected_image = protect_border_regions(denoised, border_mask, kernel_size=15)

    plt.imshow(protected_image)
    plt.title("protected_image")
    plt.show()
    
    # Step 4: Bridge small gaps within regions (not across borders)
    bridged_image = bridge_small_gaps(protected_image, max_gap=25, density_threshold=0.2, local_window=15, regional_window=25, method="relaxed")

    plt.imshow(bridged_image)
    plt.title("bridged_image")
    plt.show()

    closed_regions=fill_closed_regions(bridged_image, min_hole_size=10, max_hole_size=10000, connectivity=4)
    plt.imshow(closed_regions)
    plt.title("closed_regions")
    plt.show()

    # INSTEAD OF:
    cleaned_image = remove_small_regions(closed_regions, min_size=5, remove_thin_lines=True, kernel_size=30)


    
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























def fill_closed_regions(binary_image, min_hole_size=10, max_hole_size=1000, connectivity=4):
    """
    Fill holes in closed regions while respecting size constraints.
    
    Args:
        binary_image: Binary image (0 and 255)
        min_hole_size: Minimum hole size to fill (avoid noise)
        max_hole_size: Maximum hole size to fill (avoid filling large backgrounds)
        connectivity: 4 or 8 connectivity
    
    Returns:
        filled_image: Image with closed regions filled
    """
    # Ensure binary image is 0-255 uint8
    if binary_image.max() <= 1:
        binary_image = (binary_image * 255).astype(np.uint8)
    
    # Invert to find holes (black regions surrounded by white)
    inverted = 255 - binary_image
    
    # Find connected components in the inverted image
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=connectivity)
    
    # Create mask of holes to fill
    holes_to_fill = np.zeros_like(binary_image, dtype=np.uint8)
    
    for i in range(1, num_labels):  # Skip background (index 0)
        hole_area = stats[i, cv2.CC_STAT_AREA]
        
        # Only fill holes within size constraints
        if min_hole_size <= hole_area <= max_hole_size:
            holes_to_fill[labels == i] = 255
    
    # Combine original with filled holes
    filled_image = cv2.bitwise_or(binary_image, holes_to_fill)
    
    print(f"Filled {np.sum(holes_to_fill > 0)} pixels in {num_labels-1} holes")
    return filled_image

def remove_small_noise_regions(binary_image, min_size=5, density_threshold=0.2, window_size=15):
    """
    Remove small noise regions ONLY in low-density areas.
    Preserves small regions that are part of dense areas.
    
    Args:
        binary_image: Binary image (0 and 255)
        min_size: Minimum size for noise regions
        density_threshold: Only remove in areas with density below this (0-1)
        window_size: Window size for density calculation
    
    Returns:
        denoised_image: Image with small noise removed from sparse areas
    """
    # Calculate local density
    density_map = compute_local_density(binary_image, window_size)
    
    # Remove small white noise in low-density areas
    denoised_white = remove_small_components_density_aware(
        binary_image, min_size, foreground=255, 
        density_map=density_map, density_threshold=density_threshold
    )
    
    # Remove small black noise in low-density areas
    inverted = 255 - denoised_white
    denoised_black = remove_small_components_density_aware(
        inverted, min_size, foreground=255,
        density_map=density_map, density_threshold=density_threshold
    )
    
    # Convert back
    denoised_image = 255 - denoised_black
    
    return denoised_image

def remove_small_components_density_aware(binary_image, min_size, foreground=255, 
                                        density_map=None, density_threshold=0.2):
    """
    Remove small connected components ONLY in low-density regions.
    
    Args:
        binary_image: Binary image
        min_size: Minimum component size
        foreground: Value representing foreground
        density_map: Precomputed density map (optional)
        density_threshold: Only remove in areas with density below this
    
    Returns:
        cleaned_image: Image with small components removed from sparse areas
    """
    # Calculate density map if not provided
    if density_map is None:
        density_map = compute_local_density(binary_image, window_size=15)
    
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (binary_image == foreground).astype(np.uint8), connectivity=8
    )
    
    # Create output image (start with original)
    cleaned_image = binary_image.copy()
    
    # Track removal statistics
    removed_count = 0
    preserved_count = 0
    
    for i in range(1, num_labels):
        region_mask = (labels == i)
        region_area = stats[i, cv2.CC_STAT_AREA]
        
        # Only consider small regions for removal
        if region_area < min_size:
            # Calculate average density for this region
            region_density = np.mean(density_map[region_mask])
            
            # Only remove if in low-density area
            if region_density < density_threshold:
                # Remove this component
                cleaned_image[region_mask] = 0 if foreground == 255 else 255
                removed_count += 1
            else:
                # Preserve - it's in a dense area (likely important)
                preserved_count += 1
    
    print(f"Removed {removed_count} small components in low-density areas")
    print(f"Preserved {preserved_count} small components in dense areas")
    
    return cleaned_image

