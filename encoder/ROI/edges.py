import cv2
import numpy as np

def get_edge_map(image_rgb):
    best_edges, best_low, best_high, best_method = find_best_edges_by_quality(image_rgb)
    edge_map = cv2.Canny(image_rgb, best_low, best_high)
    return edge_map

def find_best_edges_by_quality(image_rgb, debug=False):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    methods = ['otsu', 'percentile', 'gradient', 'hybrid']
    
    best_score = -1
    best_edges = None
    
    for method in methods:
        for sensitivity in [0.5, 0.7, 1.0, 1.3, 1.5]:
            low, high = compute_adaptive_canny_thresholds(gray, method, sensitivity)
            edges = cv2.Canny(gray, low, high)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Score based on edge continuity and noise
            score = evaluate_edge_quality(edges, gray)
            
            if score > best_score:
                best_score = score
                best_edges = edges
                best_method = method
                best_low, best_high = low, high
                best_sensitivity = sensitivity

            if debug: print(f"{method} (sens: {sensitivity}): ({low}, {high}) -> density: {edge_density:.4f}, score: {score:.4f}")

    if debug:
        print(f"\nBest method: {best_method} with sensitivity {best_sensitivity}")
        print(f"Best thresholds: ({best_low}, {best_high})")
        print(f"Edge density: {np.sum(cv2.Canny(gray, best_low, best_high) > 0) / (gray.shape[0] * gray.shape[1]):.4f}")
    
    return best_edges, best_low, best_high, best_method

def evaluate_edge_quality(edges, gray):
    """Score edges based on continuity and signal-to-noise ratio"""
    # Longer edges are better (less fragmentation)
    from skimage.measure import label
    labeled_edges = label(edges, connectivity=2)
    region_sizes = np.bincount(labeled_edges.ravel())
    avg_region_size = np.mean(region_sizes[1:])  # Exclude background
    
    # Higher contrast edges are better
    edge_intensities = gray[edges > 0]
    contrast_score = np.std(edge_intensities) if len(edge_intensities) > 0 else 0
    
    return avg_region_size * contrast_score

def compute_adaptive_canny_thresholds(image, method='otsu', sensitivity=1.0):
    """
    Compute adaptive Canny thresholds based on image characteristics.
    
    Args:
        image: Grayscale or RGB image
        method: 'otsu', 'percentile', 'gradient', 'hybrid'
        sensitivity: Factor to adjust threshold sensitivity (0.5-2.0)
    
    Returns:
        low_threshold, high_threshold
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    if method == 'otsu':
        # Use Otsu's method to find optimal threshold
        otsu_threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        low_threshold = max(10, int(otsu_threshold * 0.5 * sensitivity))
        high_threshold = min(255, int(otsu_threshold * 1.5 * sensitivity))
    
    elif method == 'percentile':
        # Use percentiles of gradient magnitude
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Remove zeros for percentile calculation
        non_zero_gradients = gradient_magnitude[gradient_magnitude > 0]
        
        if len(non_zero_gradients) > 0:
            low_percentile = np.percentile(non_zero_gradients, 70) * sensitivity
            high_percentile = np.percentile(non_zero_gradients, 90) * sensitivity
        else:
            low_percentile = 50 * sensitivity
            high_percentile = 150 * sensitivity
        
        low_threshold = max(10, int(low_percentile))
        high_threshold = min(255, int(high_percentile))
    
    elif method == 'gradient':
        # Based on gradient statistics
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        mean_grad = np.mean(gradient_magnitude)
        std_grad = np.std(gradient_magnitude)
        
        low_threshold = max(10, int((mean_grad - 0.5 * std_grad) * sensitivity))
        high_threshold = min(255, int((mean_grad + 0.5 * std_grad) * sensitivity))
    
    elif method == 'hybrid':
        # Combine multiple methods
        # Method 1: Otsu-based
        otsu_threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Method 2: Gradient-based
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        mean_grad = np.mean(gradient_magnitude)
        
        # Combine both methods
        combined_low = (otsu_threshold * 0.5 + mean_grad * 0.5) * sensitivity
        combined_high = (otsu_threshold * 1.5 + mean_grad * 1.0) * sensitivity
        
        low_threshold = max(10, int(combined_low))
        high_threshold = min(255, int(combined_high))
    
    else:
        # Default fallback
        low_threshold = 50
        high_threshold = 150
    
    # Ensure high > low and reasonable values
    low_threshold = max(10, min(200, low_threshold))
    high_threshold = max(low_threshold + 10, min(255, high_threshold))
    
    return low_threshold, high_threshold



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