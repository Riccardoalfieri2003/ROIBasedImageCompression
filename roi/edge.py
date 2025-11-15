

import math
import sys
import cv2
from matplotlib import pyplot as plt
import numpy as np





"""def edge_preserving_preprocess(image):
    # Step 1: Strong bilateral filter to blur interiors while preserving edges
    smoothed = cv2.bilateralFilter(image, 15, 80, 80)
    
    # Step 2: Edge enhancement using Laplacian
    gray = cv2.cvtColor(smoothed, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpened_gray = gray - 0.5 * laplacian
    sharpened_gray = np.clip(sharpened_gray, 0, 255).astype(np.uint8)
    
    # Step 3: Merge enhanced edges back with smoothed color
    lab = cv2.cvtColor(smoothed, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    enhanced_l = cv2.addWeighted(l, 0.7, sharpened_gray, 0.3, 0)
    final_lab = cv2.merge([enhanced_l, a, b])
    final = cv2.cvtColor(final_lab, cv2.COLOR_LAB2RGB)
    
    return final"""


def edge_preserving_preprocess(image):

    # Step 2: Edge enhancement using Laplacian
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpened_gray = gray - 0.5 * laplacian
    sharpened_gray = np.clip(sharpened_gray, 0, 255).astype(np.uint8)
    
    # Step 3: Merge enhanced edges back with smoothed color
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    enhanced_l = cv2.addWeighted(l, 0.7, sharpened_gray, 0.3, 0)
    final_lab = cv2.merge([enhanced_l, a, b])
    final = cv2.cvtColor(final_lab, cv2.COLOR_LAB2RGB)
    
    return final



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




"""
if __name__ == "__main__":
    image_name = 'images/cerchio.png'
    image = cv2.imread(image_name)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mod_image=edge_preserving_preprocess(image_rgb)

    # Compute all components
    edge_map = cv2.Canny(mod_image, 50, 150)
    edge_density = compute_local_density(edge_map, kernel_size=15)"""



def visualize_intensity_borders(edge_map, edge_density, density_threshold=0.3):
    """
    Visualize edge intensities only for high-density border regions.
    
    Args:
        edge_map: Original Canny edge map (0-255)
        edge_density: Density map from compute_local_density (0-1)
        density_threshold: Threshold for considering high-density areas
    """
    # Create mask for high-density regions
    high_density_mask = edge_density > density_threshold
    
    # Apply mask to edge map - keep only edges in high-density areas
    intensity_borders = edge_map.copy()
    intensity_borders[~high_density_mask] = 0  # Set low-density areas to 0
    
    # Create visualization
    plt.figure(figsize=(12, 4))
    
    # Plot 1: Original edge map
    plt.subplot(1, 3, 1)
    plt.imshow(edge_map, cmap='gray')
    plt.title('Original Canny Edges')
    plt.axis('off')
    
    # Plot 2: Edge density map
    plt.subplot(1, 3, 2)
    plt.imshow(edge_density, cmap='hot')
    plt.title(f'Edge Density\nThreshold: {density_threshold}')
    plt.colorbar()
    plt.axis('off')
    
    # Plot 3: Filtered edge intensities
    plt.subplot(1, 3, 3)
    plt.imshow(intensity_borders, cmap='gray')
    plt.title(f'High-Density Borders\n({np.sum(intensity_borders > 0)} pixels)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return intensity_borders













def calculate_average_edge_intensity(edge_map):
    """
    Calculate average intensity of edges, excluding zero (non-edge) pixels.
    
    Args:
        edge_map: Canny edge map (0-255)
    
    Returns:
        avg_intensity: Average intensity of edge pixels only
        edge_pixels_count: Number of edge pixels
    """
    # Get only edge pixels (non-zero)
    edge_pixels = edge_map[edge_map > 0]
    
    if len(edge_pixels) == 0:
        return 0, 0
    
    avg_intensity = np.mean(edge_pixels)
    edge_pixels_count = len(edge_pixels)
    
    return avg_intensity, edge_pixels_count

def calculate_average_edge_density(edge_density_map, edge_map):
    """
    Calculate average density only at edge locations.
    
    Args:
        edge_density_map: Density map from compute_local_density (0-1)
        edge_map: Canny edge map to identify edge locations
    
    Returns:
        avg_density: Average density at edge pixels
        edge_pixels_count: Number of edge pixels considered
    """
    # Create mask for edge locations
    edge_mask = edge_map > 0
    
    # Get density values only at edge locations
    edge_density_values = edge_density_map[edge_mask]
    
    if len(edge_density_values) == 0:
        return 0, 0
    
    avg_density = np.mean(edge_density_values)
    edge_pixels_count = len(edge_density_values)
    
    return avg_density, edge_pixels_count

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







def unify_black_pixels_in_white_regions(binary_image, window_size=30, white_threshold=0.5, min_region_size=50):
    """
    Unite black pixels in predominantly white regions by converting them to white,
    then remove small isolated regions.
    
    Args:
        binary_image: Binary image (0 and 255)
        window_size: Size of the neighborhood window (odd number recommended)
        white_threshold: Percentage of white pixels needed to convert black to white (0-1)
        min_region_size: Minimum number of pixels for a region to be kept
    
    Returns:
        unified_image: Binary image with unified regions and small regions removed
        region_map: Array showing region 1 (unified white) and region 0 (background)
    """
    # Ensure binary image is 0-1
    if binary_image.max() > 1:
        binary_01 = (binary_image > 0).astype(np.float32)
    else:
        binary_01 = binary_image.astype(np.float32)
    
    # Create a normalized kernel for averaging
    kernel = np.ones((window_size, window_size), dtype=np.float32)
    kernel /= np.sum(kernel)
    
    # Compute local white percentage using convolution
    local_white_ratio = cv2.filter2D(binary_01, -1, kernel)
    
    # Create mask: black pixels in predominantly white regions
    black_pixels_mask = (binary_01 == 0)  # Original black pixels
    white_dominant_regions = (local_white_ratio > white_threshold)  # Regions with mostly white
    
    # Convert black pixels to white if they're in white-dominant regions
    unified_image = binary_image.copy()
    unified_image[black_pixels_mask & white_dominant_regions] = 255
    
    # Step 3: Remove small regions
    cleaned_image = remove_small_regions(unified_image, min_size=min_region_size)
    
    # Create region map: 1 for unified white region, 0 for background
    region_map = (cleaned_image > 0).astype(np.uint8)
    
    return cleaned_image, region_map

def remove_small_regions(binary_image, min_size=50):
    """
    Remove small connected components from binary image.
    
    Args:
        binary_image: Binary image (0 and 255)
        min_size: Minimum number of pixels for a region to be kept
    
    Returns:
        cleaned_image: Binary image with small regions removed
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
    
    return cleaned_image

# Alternative version using morphological operations for faster processing
def remove_small_regions_fast(binary_image, min_size=50):
    """
    Faster method using morphological operations (good for very small noise).
    """
    # Use opening to remove small objects
    kernel_size = max(1, int(np.sqrt(min_size) // 2))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    
    return cleaned_image


def visualize_unification_process(original_binary, unified_image, region_map, window_size):
    """
    Visualize the unification process.
    """
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Original binary image
    plt.subplot(1, 3, 1)
    plt.imshow(original_binary, cmap='gray')
    plt.title('Original Binary Image')
    plt.axis('off')
    
    # Plot 2: Unified image
    plt.subplot(1, 3, 2)
    plt.imshow(unified_image, cmap='gray')
    changed_pixels = np.sum((original_binary == 0) & (unified_image == 255))
    plt.title(f'Unified Image\n{changed_pixels} pixels converted')
    plt.axis('off')
    
    # Plot 3: Region map
    plt.subplot(1, 3, 3)
    plt.imshow(region_map, cmap='tab10')
    plt.title('Region Map\n(1=unified white, 0=background)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def process_and_unify_borders(edge_map, edge_density, original_image, density_threshold=0.3, 
                            unification_window=30, unification_threshold=0.5, min_region_size=50):
    """
    Complete pipeline: filter edges by density, then unify regions, remove small regions.
    """
    # Step 1: Get high-density borders (your existing approach)
    high_density_mask = edge_density > density_threshold
    intensity_borders = edge_map.copy()
    intensity_borders[~high_density_mask] = 0
    
    # Convert to binary (0 and 255)
    binary_borders = (intensity_borders > 0).astype(np.uint8) * 255
    
    # Step 2: Unify black pixels in white regions with small region removal
    unified_borders, region_map = unify_black_pixels_in_white_regions(
        binary_borders, 
        window_size=unification_window, 
        white_threshold=unification_threshold,
        min_region_size=min_region_size
    )
    
    # Create overlay visualization
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Original image
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot 2: Before small region removal
    plt.subplot(1, 3, 2)
    plt.imshow(binary_borders, cmap='gray')
    initial_regions = np.sum(binary_borders > 0)
    plt.title(f'Before Cleaning\n{initial_regions} pixels')
    plt.axis('off')
    
    # Plot 3: After small region removal
    plt.subplot(1, 3, 3)
    plt.imshow(unified_borders, cmap='gray')
    final_regions = np.sum(unified_borders > 0)
    removed_pixels = initial_regions - final_regions
    plt.title(f'After Cleaning\n{final_regions} pixels\n{removed_pixels} removed')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Overlay visualization
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(original_image)
    # Create colored overlay (red for region 1)
    overlay = np.zeros_like(original_image)
    overlay[region_map == 1] = [255, 0, 0]  # Red color for region 1
    plt.imshow(overlay, alpha=0.6)
    plt.title(f'Cleaned ROI Overlay\n{final_regions} pixels (min size: {min_region_size})')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"=== Region Cleaning Statistics ===")
    print(f"Initial white pixels: {initial_regions}")
    print(f"Final white pixels: {final_regions}")
    print(f"Pixels removed: {removed_pixels}")
    print(f"Regions removed: {removed_pixels} small components")
    print(f"Region 1 size: {np.sum(region_map == 1)} pixels")
    print(f"Region 0 size: {np.sum(region_map == 0)} pixels")
    
    return unified_borders, region_map

# Alternative visualization with more detailed overlay
def visualize_detailed_regions(original_image, region_map, unified_borders):
    """
    Create a more detailed visualization showing different aspects of the regions.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Binary borders
    axes[0, 1].imshow(unified_borders, cmap='gray')
    axes[0, 1].set_title('Detected Regions (Binary)')
    axes[0, 1].axis('off')
    
    # Region map with colors
    axes[1, 0].imshow(region_map, cmap='tab10')
    axes[1, 0].set_title('Region Map\n(0=Background, 1=ROI)')
    axes[1, 0].axis('off')
    
    # Overlay on original
    axes[1, 1].imshow(original_image)
    # Create a colored overlay where region 1 is semi-transparent red
    overlay_mask = np.zeros((*region_map.shape, 3))
    overlay_mask[region_map == 1] = [1, 0, 0]  # Red in RGB
    axes[1, 1].imshow(overlay_mask, alpha=0.4)
    axes[1, 1].set_title('ROI Overlay on Original\n(Red = Detected Regions)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

"""
# Updated main function with overlay visualization
if __name__ == "__main__":
    image_name = 'images/Hawaii.jpg'
    image = cv2.imread(image_name)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print(f"Size: {image_rgb.size}")
    factor = math.ceil(math.log(image_rgb.size, 10)) * math.log(image_rgb.size)
    print(f"factor: {factor}")

    # Compute edges and density
    edge_map = cv2.Canny(image_rgb, 50, 150)
    edge_density = compute_local_density(edge_map, kernel_size=15)
    
    threshold = suggest_automatic_threshold(edge_density, edge_map, method="percentile")
    thresholds = [threshold/5, threshold/2.5, threshold]

    window_size = math.floor(factor)
    min_region_size= math.ceil( image_rgb.size / math.pow(10, math.ceil(math.log(image_rgb.size, 10))-3 ) )
    print(f"min_region_size: {min_region_size}")


    for i, threshold in enumerate(thresholds):
        print(f"\n=== Test {i+1}: Window: {window_size}x{window_size}, Threshold: {threshold:.3f} ===")
        unified, regions = process_and_unify_borders(
            edge_map, edge_density, image_rgb,  # Pass original image for overlay
            density_threshold=0.05,
            unification_window=window_size,
            unification_threshold=threshold,
            min_region_size=min_region_size
        )
        
        # Additional detailed visualization
        visualize_detailed_regions(image_rgb, regions, unified)
"""



def optimize_threshold_automatically(edge_density_map, edge_map, method='adaptive_mean'):
    """
    Optimize threshold using various mathematical optimization techniques.
    
    Args:
        edge_density_map: Density map from compute_local_density
        edge_map: Canny edge map
        method: Optimization method
    
    Returns:
        optimized_threshold: Mathematically optimized threshold
    """
    # Get density values only at edge locations
    edge_mask = edge_map > 0
    edge_density_values = edge_density_map[edge_mask]
    
    if len(edge_density_values) == 0:
        return 0.1  # Default fallback
    
    if method == 'adaptive_mean':
        # Method 1: Adaptive mean based on distribution shape
        mean_val = np.mean(edge_density_values)
        std_val = np.std(edge_density_values)
        skewness = np.mean((edge_density_values - mean_val) ** 3) / (std_val ** 3)
        
        # Adjust based on skewness: if distribution is skewed, threshold should be lower
        if skewness > 1:  # Right-skewed (many low-density edges)
            return mean_val * 0.7
        elif skewness < -1:  # Left-skewed (many high-density edges)
            return mean_val * 1.2
        else:
            return mean_val
    
    elif method == 'bimodal_optimization':
        # Method 2: Detect bimodal distribution and find valley
        from scipy.signal import find_peaks
        
        hist, bin_edges = np.histogram(edge_density_values, bins=50, density=True)
        peaks, _ = find_peaks(hist, height=0.01)
        
        if len(peaks) >= 2:
            # Find the valley between two highest peaks
            sorted_peaks = peaks[np.argsort(hist[peaks])[-2:]]  # Two highest peaks
            valley_start, valley_end = min(sorted_peaks), max(sorted_peaks)
            valley_region = hist[valley_start:valley_end]
            valley_pos = valley_start + np.argmin(valley_region)
            threshold = bin_edges[valley_pos]
            return threshold
        else:
            # Fallback to percentile if not bimodal
            return np.percentile(edge_density_values, 60)
    
    elif method == 'entropy_optimization':
        # Method 3: Maximize information gain (entropy)
        best_entropy = -1
        best_threshold = 0.3
        
        for threshold in np.linspace(0.1, 0.8, 50):
            # Create binary classification
            high_density = edge_density_values > threshold
            low_density = ~high_density
            
            n_high = np.sum(high_density)
            n_low = np.sum(low_density)
            n_total = len(edge_density_values)
            
            if n_high == 0 or n_low == 0:
                continue
                
            # Calculate entropy
            p_high = n_high / n_total
            p_low = n_low / n_total
            entropy = - (p_high * np.log2(p_high) + p_low * np.log2(p_low))
            
            if entropy > best_entropy:
                best_entropy = entropy
                best_threshold = threshold
        
        return best_threshold
    
    elif method == 'otsu_optimization':
        # Method 5: Otsu's method adapted for density maps
        # Normalize to 0-255 for Otsu
        normalized_density = (edge_density_values - edge_density_values.min())
        if normalized_density.max() > 0:
            normalized_density = normalized_density / normalized_density.max() * 255
        
        normalized_density = normalized_density.astype(np.uint8)
        
        # Use Otsu's threshold
        otsu_threshold, _ = cv2.threshold(normalized_density, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Convert back to 0-1 range
        return otsu_threshold / 255.0
    
    else:
        # Fallback to your original method
        return suggest_automatic_threshold(edge_density_map, edge_map, method='mean')

def find_best_threshold_method(edge_density_map, edge_map):
    """
    Test multiple optimization methods and suggest the best one.
    """
    methods = ['adaptive_mean', 'entropy_optimization', 'otsu_optimization']
    
    best_method = 'adaptive_mean'
    best_score = -1
    
    for method in methods:
        try:
            threshold = optimize_threshold_automatically(edge_density_map, edge_map, method)
            
            # Score the threshold (higher = better separation)
            edge_density_values = edge_density_map[edge_map > 0]
            high_density = edge_density_values > threshold
            low_density = ~high_density
            
            if np.sum(high_density) > 0 and np.sum(low_density) > 0:
                # Score based on separation (between-class variance)
                mean_high = np.mean(edge_density_values[high_density])
                mean_low = np.mean(edge_density_values[low_density])
                score = abs(mean_high - mean_low)  # Larger separation = better
                
                if score > best_score:
                    best_score = score
                    best_method = method
                    
        except Exception as e:
            continue
    
    final_threshold = optimize_threshold_automatically(edge_density_map, edge_map, best_method)
    return final_threshold, best_method, best_score

# Updated usage in your main function:
if __name__ == "__main__":
    image_name = 'images/Lenna.webp'
    image = cv2.imread(image_name)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Compute edges and density
    edge_map = cv2.Canny(image_rgb, 50, 150)
    edge_density = compute_local_density(edge_map, kernel_size=15)
    
    # Test different optimization methods
    methods = ['adaptive_mean', 'entropy_optimization', 'otsu_optimization']
    
    for method in methods:
        print(f"\n=== Testing {method} ===")
        optimized_threshold = optimize_threshold_automatically(edge_density, edge_map, method)
        print(f"Optimized threshold: {optimized_threshold:.3f}")
        
        unified, regions = process_and_unify_borders(
            edge_map, edge_density, image_rgb,
            density_threshold=0.05,
            unification_window=30,
            unification_threshold=optimized_threshold,
            min_region_size=50
        )
    
    # Find the best method automatically
    print(f"\n=== Finding Best Method ===")
    best_threshold, best_method, best_score = find_best_threshold_method(edge_density, edge_map)
    print(f"Best method: {best_method}")
    print(f"Best threshold: {best_threshold:.3f}")
    print(f"Separation score: {best_score:.3f}")