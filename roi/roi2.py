import cv2
from matplotlib import pyplot as plt
import numpy as np
import cv2
import numpy as np
from scipy import ndimage
from scipy import fftpack






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

def spectral_residual_saliency(image):
    """
    Classical spectral residual saliency detection.
    Based on the paper: "Saliency Detection: A Spectral Residual Approach"
    
    Args:
        image: RGB image (H, W, 3)
    
    Returns:
        saliency_map: Saliency map highlighting visually salient regions
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    gray = gray.astype(np.float32)
    
    # Step 1: Compute log amplitude spectrum
    fft = fftpack.fft2(gray)
    amplitude_spectrum = np.abs(fft)
    log_amplitude = np.log(amplitude_spectrum + 1e-8)  # Avoid log(0)
    
    # Step 2: Compute spectral residual (difference from local average)
    # Smooth the log amplitude spectrum
    kernel_size = 3
    smoothed_log_amp = cv2.blur(log_amplitude, (kernel_size, kernel_size))
    spectral_residual = log_amplitude - smoothed_log_amp
    
    # Step 3: Reconstruct saliency map
    # Combine residual phase with original phase
    saliency_fft = np.exp(spectral_residual + 1j * np.angle(fft))
    saliency_spatial = np.abs(fftpack.ifft2(saliency_fft))
    
    # Step 4: Post-process
    saliency_map = cv2.GaussianBlur(saliency_spatial, (5, 5), 3)
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
    
    return saliency_map


def detect_roi_coarse(image):
    """
    Find potential ROIs without using segmentation
    """
    methods = []
    
    # 1. Edge density (your approach)
    edge_map = cv2.Canny(image, 50, 150)
    edge_density = compute_local_density(edge_map, kernel_size=15)
    methods.append(edge_density)
    
    # 2. Local contrast
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    contrast_map = cv2.Laplacian(lab[:,:,0], cv2.CV_32F)
    methods.append(np.abs(contrast_map))
    
    # 3. Classical saliency
    saliency_map = spectral_residual_saliency(image)
    methods.append(saliency_map)
    
    # Combine evidence
    combined_roi_map = np.mean(methods, axis=0)
    
    # Return both binary and continuous maps
    binary_roi_map = combined_roi_map > np.percentile(combined_roi_map, 70)
    return binary_roi_map, combined_roi_map





def find_texture_homogeneous_regions(image):
    """Find regions with consistent texture patterns"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Compute local texture variance
    texture_variance = compute_local_variance(gray, kernel_size=15)
    
    # Low variance = homogeneous texture = potential ROI
    homogeneous_regions = texture_variance < np.percentile(texture_variance, 30)
    
    return homogeneous_regions

def compute_local_variance(image, kernel_size=15):
    """Compute local variance to find texture homogeneity"""
    mean = cv2.blur(image, (kernel_size, kernel_size))
    mean_sq = cv2.blur(image.astype(np.float32)**2, (kernel_size, kernel_size))
    variance = mean_sq - mean.astype(np.float32)**2
    return variance

def find_color_uniform_regions(image):
    """Find regions with consistent color"""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Compute color variance in LAB space (perceptually uniform)
    color_variance = np.zeros(image.shape[:2])
    for channel in range(3):
        channel_var = compute_local_variance(lab[:,:,channel], kernel_size=20)
        color_variance += channel_var
    
    # Low color variance = uniform color region = potential ROI
    uniform_color_regions = color_variance < np.percentile(color_variance, 40)
    
    return uniform_color_regions


def filter_by_size(binary_map, min_size_ratio=0.005, max_size_ratio=0.8):
    """Keep only regions of meaningful size"""
    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_map.astype(np.uint8), connectivity=8
    )
    
    # Filter by size
    total_pixels = binary_map.size
    filtered_mask = np.zeros_like(binary_map)
    
    for i in range(1, num_labels):  # Skip background
        area = stats[i, cv2.CC_STAT_AREA]
        area_ratio = area / total_pixels
        
        if min_size_ratio <= area_ratio <= max_size_ratio:
            filtered_mask[labels == i] = True
    
    return filtered_mask

def detect_complete_roi_regions(image):
    """
    Find complete ROI regions, not just boundaries
    """
    roi_evidence = []
    
    # 1. Texture homogeneity (low variance = uniform regions)
    texture_homogeneous = find_texture_homogeneous_regions(image)
    roi_evidence.append(texture_homogeneous.astype(float))
    
    # 2. Color uniformity (low variance = solid color regions)
    color_uniform = find_color_uniform_regions(image)
    roi_evidence.append(color_uniform.astype(float))
    
    # 3. Edge evidence (your existing method - but for completeness)
    edge_based_roi, _ = detect_roi_coarse(image)
    roi_evidence.append(edge_based_roi.astype(float))
    
    # 4. Size-based filtering (remove noise and background)
    combined_evidence = np.mean(roi_evidence, axis=0)
    size_filtered = filter_by_size(combined_evidence > 0.5)
    
    return size_filtered

def visualize_region_vs_edge_detection(original, edge_roi, region_roi):
    """Compare edge-based vs region-based ROI detection"""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original
    axes[0].imshow(original)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Edge-based ROI (your current approach)
    edge_overlay = original.copy()
    edge_overlay[edge_roi] = [255, 0, 0]  # Red edges
    axes[1].imshow(edge_overlay)
    axes[1].set_title('Edge-Based ROI\n(Finds boundaries)')
    axes[1].axis('off')
    
    # Region-based ROI (new approach)
    region_overlay = original.copy()
    region_overlay[region_roi] = [0, 255, 0]  # Green regions
    axes[2].imshow(region_overlay)
    axes[2].set_title('Region-Based ROI\n(Finds complete areas)')
    axes[2].axis('off')
    
    # Combined
    combined_overlay = original.copy()
    combined_overlay[edge_roi] = [255, 0, 0]    # Red edges
    combined_overlay[region_roi] = [0, 255, 0]  # Green regions
    axes[3].imshow(combined_overlay)
    axes[3].set_title('Combined: Red=Edges, Green=Regions')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()



def find_object_regions(image):
    """Find complete objects rather than just boundaries"""
    # Use morphological operations to close boundaries
    edges = cv2.Canny(image, 50, 150)
    
    # Close gaps in edges to form complete regions
    kernel = np.ones((5, 5), np.uint8)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Fill regions
    contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    object_mask = np.zeros_like(edges)
    for contour in contours:
        # Filter by size and shape
        area = cv2.contourArea(contour)
        if area > 1000:  # Minimum object size
            cv2.fillPoly(object_mask, [contour], 255)
    
    return object_mask > 0


def find_salient_regions(image):
    """Find regions that visually stand out"""
    # Your existing saliency
    saliency_map = spectral_residual_saliency(image)
    
    # But now find connected regions, not just points
    binary_saliency = saliency_map > np.percentile(saliency_map, 80)
    
    # Morphological cleaning to form regions
    kernel = np.ones((10, 10), np.uint8)
    cleaned_regions = cv2.morphologyEx(
        binary_saliency.astype(np.uint8), 
        cv2.MORPH_CLOSE, kernel
    )
    
    return cleaned_regions > 0


def test_complete_roi_detection(test_image):
    """Compare different ROI detection approaches"""
    
    # Your current edge-based approach
    edge_roi, _ = detect_roi_coarse(test_image)
    
    # New region-based approaches
    texture_roi = find_texture_homogeneous_regions(test_image)
    color_roi = find_color_uniform_regions(test_image)
    object_roi = find_object_regions(test_image)
    salient_roi = find_salient_regions(test_image)
    
    # Combine evidence
    combined_region_roi = (texture_roi | color_roi | object_roi | salient_roi)
    combined_region_roi = filter_by_size(combined_region_roi)
    
    # Visualize comparison
    visualize_region_vs_edge_detection(test_image, edge_roi, combined_region_roi)
    
    return edge_roi, combined_region_roi

# Usage
if __name__ == "__main__":
    image_name = 'images/waikiki.jpg'
    image = cv2.imread(image_name)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    edge_roi, region_roi = test_complete_roi_detection(image_rgb)