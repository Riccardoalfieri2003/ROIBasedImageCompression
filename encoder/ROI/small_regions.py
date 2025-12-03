import numpy as np
import cv2

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

def connect_nearby_pixels(binary_image, connection_distance=3, method='dilation', min_region_size=5):
    """
    Connect nearby pixels by filling gaps between them.
    
    Args:
        binary_image: Binary image (0 and 255)
        connection_distance: Maximum distance to connect pixels (kernel size)
        method: 'dilation', 'voronoi', 'skeleton', or 'region_growing'
        min_region_size: Minimum region size to consider for connection
    
    Returns:
        connected_image: Image with nearby pixels connected
    """
    if binary_image.max() <= 1:
        binary_image = (binary_image * 255).astype(np.uint8)
    
    if method == 'dilation':
        return connect_by_dilation(binary_image, connection_distance, min_region_size)
    elif method == 'voronoi':
        return connect_by_voronoi(binary_image, connection_distance, min_region_size)
    elif method == 'skeleton':
        return connect_by_skeleton(binary_image, connection_distance, min_region_size)
    elif method == 'region_growing':
        return connect_by_region_growing(binary_image, connection_distance, min_region_size)
    else:
        return connect_by_dilation(binary_image, connection_distance, min_region_size)

def connect_by_dilation(binary_image, connection_distance, min_region_size):
    """
    Connect pixels using morphological dilation.
    Simple and effective for most cases.
    """
    # Remove very small regions first
    cleaned = remove_small_regions(binary_image, min_size=min_region_size)
    
    # Create kernel based on connection distance
    kernel_size = connection_distance * 2 + 1  # Ensure odd number
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Dilate to connect nearby regions
    dilated = cv2.dilate(cleaned, kernel, iterations=1)
    
    # Erode back to original scale but with connections
    connected = cv2.erode(dilated, kernel, iterations=1)
    
    return connected

def connect_by_voronoi(binary_image, connection_distance, min_region_size):
    """
    Connect pixels using Voronoi diagram - creates natural-looking connections.
    """
    from scipy.spatial import Voronoi
    from scipy.ndimage import distance_transform_edt
    
    # Remove small regions
    cleaned = remove_small_regions(binary_image, min_size=min_region_size)
    
    # Get coordinates of white pixels
    y_coords, x_coords = np.where(cleaned > 0)
    
    if len(x_coords) == 0:
        return cleaned
    
    # Create Voronoi diagram
    points = np.column_stack((x_coords, y_coords))
    vor = Voronoi(points)
    
    # Create output image
    connected = np.zeros_like(cleaned)
    
    # Fill Voronoi regions that are close to multiple points
    for i, region in enumerate(vor.regions):
        if not region or -1 in region:  # Invalid region
            continue
        
        # Get polygon vertices
        polygon = vor.vertices[region]
        
        # Check if polygon is close to multiple points
        if is_polygon_connecting(polygon, points, connection_distance):
            # Convert polygon to integer coordinates and fill
            polygon_int = polygon.astype(np.int32)
            cv2.fillPoly(connected, [polygon_int], 255)
    
    return connected

def is_polygon_connecting(polygon, points, max_distance):
    """
    Check if a Voronoi polygon connects multiple points within max_distance.
    """
    if len(polygon) == 0:
        return False
    
    # Find points close to this polygon
    polygon_center = np.mean(polygon, axis=0)
    distances = np.linalg.norm(points - polygon_center, axis=1)
    
    close_points = points[distances <= max_distance * 2]
    
    return len(close_points) >= 2  # Connects at least 2 points

def connect_by_skeleton(binary_image, connection_distance, min_region_size):
    """
    Connect pixels using skeletonization and distance transform.
    """
    from skimage import morphology
    
    # Remove small regions
    cleaned = remove_small_regions(binary_image, min_size=min_region_size)
    
    # Calculate distance transform
    dist_transform = cv2.distanceTransform(255 - cleaned, cv2.DIST_L2, 5)
    
    # Create skeleton of the distance transform
    skeleton = morphology.skeletonize(dist_transform <= connection_distance)
    
    # Combine original with skeleton
    connected = np.maximum(cleaned, skeleton.astype(np.uint8) * 255)
    
    return connected

def connect_by_region_growing(binary_image, connection_distance, min_region_size):
    """
    Connect pixels using region growing algorithm.
    """
    # Remove small regions
    cleaned = remove_small_regions(binary_image, min_size=min_region_size)
    
    # Label connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    
    # Create distance map for each region
    connected = cleaned.copy()
    
    for region_id in range(1, num_labels):
        region_mask = (labels == region_id)
        
        # Grow region up to connection_distance
        grown_region = grow_region(region_mask, connection_distance)
        
        # Add grown region to result
        connected = np.maximum(connected, grown_region)
    
    return connected

def grow_region(region_mask, growth_distance):
    """
    Grow a region by specified distance.
    """
    kernel_size = growth_distance * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Dilate the region
    grown = cv2.dilate(region_mask.astype(np.uint8), kernel, iterations=1)
    
    return grown * 255

