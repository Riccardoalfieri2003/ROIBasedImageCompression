from matplotlib import pyplot as plt
import numpy as np
import cv2

def create_clean_region_display(bbox_region, bbox_mask):
    """Create a clean display with transparent background instead of black"""
    # Create RGBA image with transparency
    clean_image = np.zeros((bbox_region.shape[0], bbox_region.shape[1], 4), dtype=np.uint8)
    
    # Copy RGB data where mask is True
    clean_image[bbox_mask, :3] = bbox_region[bbox_mask]
    
    # Set alpha channel: 255 where mask is True, 0 where False
    clean_image[bbox_mask, 3] = 255
    
    return clean_image
"""
def comprehensive_region_cleaning(bbox_region, bbox_mask, min_area_ratio=0.02, show_cleaning=False):

    original_mask = bbox_mask.copy()
    total_pixels = np.sum(original_mask)
    
    print(f"Before cleaning: {total_pixels} pixels, {cv2.connectedComponents(original_mask.astype(np.uint8))[0]-1} components")
    
    # Step 1: Remove very small components
    min_area = max(int(total_pixels * min_area_ratio), 20)  # At least 20 pixels
    cleaned_mask = remove_small_components(original_mask, min_area_ratio)
    
    # Step 2: Morphological cleaning
    cleaned_mask = clean_region_mask(cleaned_mask, min_area=min_area, kernel_size=3)
    
    # Step 3: Smooth boundaries
    cleaned_mask = smooth_region_boundary(cleaned_mask, smoothing_iterations=1)
    
    # Step 4: Final component filtering
    cleaned_mask = remove_small_components(cleaned_mask, min_area_ratio)
    
    final_pixels = np.sum(cleaned_mask)
    final_components = cv2.connectedComponents(cleaned_mask.astype(np.uint8))[0] - 1
    print(f"After cleaning: {final_pixels} pixels, {final_components} components")
    
    # Create cleaned region image
    cleaned_region = np.zeros_like(bbox_region)
    cleaned_region[cleaned_mask] = bbox_region[cleaned_mask]
    
    if show_cleaning:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original
        original_display = create_clean_region_display(bbox_region, original_mask)
        axes[0, 0].imshow(original_display)
        axes[0, 0].set_title(f'Original\n{total_pixels} pixels')
        axes[0, 0].axis('off')
        
        # Mask comparison
        axes[0, 1].imshow(original_mask, cmap='gray')
        axes[0, 1].set_title('Original Mask')
        axes[0, 1].axis('off')
        
        # Cleaned
        cleaned_display = create_clean_region_display(bbox_region, cleaned_mask)
        axes[1, 0].imshow(cleaned_display)
        axes[1, 0].set_title(f'Cleaned\n{final_pixels} pixels')
        axes[1, 0].axis('off')
        
        # Cleaned mask
        axes[1, 1].imshow(cleaned_mask, cmap='gray')
        axes[1, 1].set_title('Cleaned Mask')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return cleaned_region, cleaned_mask
"""





def clean_region_mask(mask, min_area=50, kernel_size=3):
    """
    Clean a region mask by removing small artifacts and filling holes.
    
    Args:
        mask: Binary mask (True/False or 0/1)
        min_area: Minimum area in pixels to keep a connected component
        kernel_size: Size of morphological kernel for cleaning
    
    Returns:
        Cleaned mask
    """
    mask_uint8 = mask.astype(np.uint8) * 255
    
    # Remove small artifacts (opening: erosion followed by dilation)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    
    # Fill small holes (closing: dilation followed by erosion)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    
    # Remove small connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    
    # Create new mask keeping only large enough components
    final_mask = np.zeros_like(cleaned)
    for i in range(1, num_labels):  # Skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            final_mask[labels == i] = 255
    
    return final_mask.astype(bool)

"""
def remove_small_components(mask, min_area_ratio=0.01):

    mask_uint8 = mask.astype(np.uint8) * 255
    total_area = np.sum(mask)
    min_area = int(total_area * min_area_ratio)
    
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    
    # Keep only components larger than min_area
    cleaned_mask = np.zeros_like(mask_uint8)
    for i in range(1, num_labels):  # Skip background
        if stats[i, cv2.CC_STAT_AREA] >= max(min_area, 10):  # At least 10 pixels
            cleaned_mask[labels == i] = 255
    
    return cleaned_mask.astype(bool)
"""
    
def smooth_region_boundary(mask, smoothing_iterations=1):
    """
    Smooth the boundary of the region to remove jagged edges.
    
    Args:
        mask: Binary mask
        smoothing_iterations: Number of smoothing passes
    
    Returns:
        Smoothed mask
    """
    mask_uint8 = mask.astype(np.uint8) * 255
    
    for _ in range(smoothing_iterations):
        # Use a larger kernel for more aggressive smoothing
        kernel = np.ones((5, 5), np.uint8)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    
    return mask_uint8.astype(bool)























































def comprehensive_region_cleaning(bbox_region, bbox_mask, min_area_ratio=0.03, border_cleaning=True, show_cleaning=False):
    """
    Comprehensive cleaning of region mask with enhanced border artifact removal.
    
    Args:
        bbox_region: The image region
        bbox_mask: The mask to clean
        min_area_ratio: Minimum area ratio for keeping components
        border_cleaning: Whether to perform aggressive border cleaning
        show_cleaning: Whether to show before/after comparison
    
    Returns:
        Tuple of (cleaned_region_image, cleaned_mask)
    """
    original_mask = bbox_mask.copy()
    total_pixels = np.sum(original_mask)
    
    print(f"Before cleaning: {total_pixels} pixels, {cv2.connectedComponents(original_mask.astype(np.uint8))[0]-1} components")
    
    # Step 1: Remove very small components
    min_area = max(int(total_pixels * min_area_ratio), 20)  # At least 20 pixels
    cleaned_mask = remove_small_components(original_mask, min_area_ratio)
    
    # Step 2: MORPHOLOGICAL CLEANING WITH BORDER FOCUS
    if border_cleaning and total_pixels > 200:  # Only for substantial regions
        cleaned_mask = aggressive_border_cleaning(cleaned_mask, bbox_region)
    
    # Step 3: Remove small components again after border cleaning
    cleaned_mask = remove_small_components(cleaned_mask, min_area_ratio)
    
    # Step 4: Smooth boundaries
    cleaned_mask = smooth_region_boundary(cleaned_mask, smoothing_iterations=1)
    
    # Step 5: Final component filtering
    cleaned_mask = remove_small_components(cleaned_mask, min_area_ratio)
    
    final_pixels = np.sum(cleaned_mask)
    final_components = cv2.connectedComponents(cleaned_mask.astype(np.uint8))[0] - 1
    print(f"After cleaning: {final_pixels} pixels, {final_components} components")
    
    # Create cleaned region image
    cleaned_region = np.zeros_like(bbox_region)
    cleaned_region[cleaned_mask] = bbox_region[cleaned_mask]
    
    if show_cleaning:
        visualize_cleaning_process(bbox_region, original_mask, cleaned_region, cleaned_mask)
    
    return cleaned_region, cleaned_mask




def aggressive_border_cleaning(mask, region_image):
    """
    Aggressively clean border artifacts and stray pixels from SLIC boundaries.
    """
    mask_uint8 = mask.astype(np.uint8) * 255
    
    # Step 1: Find the main connected component (largest one)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    
    if num_labels <= 1:  # Only background
        return mask
    
    # Find the largest component
    largest_component_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
    main_component_mask = (labels == largest_component_idx)
    
    # Step 2: Analyze color consistency of border pixels
    border_mask = find_problematic_border_pixels(main_component_mask, region_image)
    
    # Step 3: Remove problematic border pixels
    cleaned_mask = main_component_mask & (~border_mask)
    
    # Step 4: Fill small holes created by border removal
    kernel = np.ones((3, 3), np.uint8)
    cleaned_mask_uint8 = cleaned_mask.astype(np.uint8) * 255
    cleaned_mask_uint8 = cv2.morphologyEx(cleaned_mask_uint8, cv2.MORPH_CLOSE, kernel)
    
    return cleaned_mask_uint8.astype(bool)

def find_problematic_border_pixels(mask, region_image):
    """
    Identify border pixels that are likely artifacts based on color inconsistency.
    """
    mask_uint8 = mask.astype(np.uint8) * 255
    
    # Find the border of the mask
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask_uint8, kernel, iterations=1)
    border_mask = mask_uint8 - eroded
    
    if np.sum(border_mask) == 0:
        return np.zeros_like(mask, dtype=bool)
    
    # Convert to LAB for better color analysis
    lab_image = cv2.cvtColor(region_image, cv2.COLOR_RGB2LAB)
    
    # Get coordinates of border pixels
    border_coords = np.where(border_mask > 0)
    border_pixels = lab_image[border_coords]
    
    # Get coordinates of interior pixels (slightly inside the border)
    interior_mask = cv2.erode(mask_uint8, kernel, iterations=2)
    interior_coords = np.where(interior_mask > 0)
    interior_pixels = lab_image[interior_coords]
    
    if len(interior_pixels) == 0 or len(border_pixels) == 0:
        return np.zeros_like(mask, dtype=bool)
    
    # Calculate mean color of interior region
    interior_mean = np.mean(interior_pixels, axis=0)
    
    # Find border pixels that are significantly different from interior
    problematic_border = np.zeros_like(mask, dtype=bool)
    
    for i, (y, x) in enumerate(zip(border_coords[0], border_coords[1])):
        border_color = lab_image[y, x]
        
        # Calculate color distance in LAB space (more perceptually accurate)
        color_distance = np.linalg.norm(border_color - interior_mean)
        
        # High threshold for LAB space (typically 5-10 for noticeable differences)
        if color_distance > 20:  # Very conservative threshold
            problematic_border[y, x] = True
    
    print(f"Found {np.sum(problematic_border)} problematic border pixels")
    return problematic_border

def remove_small_components(mask, min_area_ratio):
    """Remove small connected components from the mask."""
    mask_uint8 = mask.astype(np.uint8) * 255
    total_area = np.sum(mask)
    min_area = int(total_area * min_area_ratio)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    
    cleaned_mask = np.zeros_like(mask_uint8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= max(min_area, 10):
            cleaned_mask[labels == i] = 255
    
    return cleaned_mask.astype(bool)

def smooth_region_boundary(mask, smoothing_iterations=1):
    """Smooth the boundary of the region."""
    mask_uint8 = mask.astype(np.uint8) * 255
    
    for _ in range(smoothing_iterations):
        kernel = np.ones((3, 3), np.uint8)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    
    return mask_uint8.astype(bool)

def visualize_cleaning_process(original_region, original_mask, cleaned_region, cleaned_mask):
    """Visualize the cleaning process."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original region with mask
    axes[0, 0].imshow(original_region)
    original_overlay = np.zeros((*original_region.shape[:2], 4))
    original_overlay[original_mask] = [1, 0, 0, 0.5]
    axes[0, 0].imshow(original_overlay)
    axes[0, 0].set_title(f'Original\n{np.sum(original_mask)} pixels')
    axes[0, 0].axis('off')
    
    # Original mask
    axes[0, 1].imshow(original_mask, cmap='gray')
    axes[0, 1].set_title('Original Mask')
    axes[0, 1].axis('off')
    
    # Border analysis
    border_mask = find_problematic_border_pixels(original_mask, original_region)
    border_display = original_region.copy()
    border_display[border_mask] = [255, 0, 0]
    axes[0, 2].imshow(border_display)
    axes[0, 2].set_title(f'Problematic Border\n{np.sum(border_mask)} pixels')
    axes[0, 2].axis('off')
    
    # Cleaned region
    axes[1, 0].imshow(cleaned_region)
    cleaned_overlay = np.zeros((*cleaned_region.shape[:2], 4))
    cleaned_overlay[cleaned_mask] = [0, 1, 0, 0.5]
    axes[1, 0].imshow(cleaned_overlay)
    axes[1, 0].set_title(f'Cleaned\n{np.sum(cleaned_mask)} pixels')
    axes[1, 0].axis('off')
    
    # Cleaned mask
    axes[1, 1].imshow(cleaned_mask, cmap='gray')
    axes[1, 1].set_title('Cleaned Mask')
    axes[1, 1].axis('off')
    
    # Difference
    difference = original_mask.astype(int) - cleaned_mask.astype(int)
    diff_display = np.zeros((*difference.shape, 3))
    diff_display[difference > 0] = [1, 0, 0]   # Red: removed pixels
    diff_display[difference < 0] = [0, 1, 0]   # Green: added pixels (should be minimal)
    axes[1, 2].imshow(diff_display)
    axes[1, 2].set_title(f'Removed Pixels\n{np.sum(difference > 0)} pixels')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()