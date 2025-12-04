import math
import cv2
from matplotlib import pyplot as plt
import numpy as np

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
