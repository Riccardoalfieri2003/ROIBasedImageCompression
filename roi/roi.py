

import cv2
import numpy as np





import cv2
import numpy as np
from scipy import ndimage

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


import numpy as np
import cv2
from scipy import fftpack

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

# Simpler alternative if you want something faster
def simple_saliency(image):
    """
    Simple saliency based on color and intensity uniqueness
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Compute mean color
    mean_color = cv2.blur(lab, (15, 15))
    
    # Compute saliency as distance from mean
    saliency = np.sqrt(
        np.sum((lab.astype(np.float32) - mean_color.astype(np.float32)) ** 2, axis=2)
    )
    
    # Normalize
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    
    return saliency




















import matplotlib.pyplot as plt

def visualize_roi_detection(original_image, binary_roi, combined_roi, edges, density, saliency):
    """
    Create a comprehensive visualization of the ROI detection process
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Original image
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Edge map
    axes[0, 1].imshow(edges, cmap='gray')
    axes[0, 1].set_title('Canny Edge Detection')
    axes[0, 1].axis('off')
    
    # Edge density
    im1 = axes[0, 2].imshow(density, cmap='hot')
    axes[0, 2].set_title('Edge Density Map')
    axes[0, 2].axis('off')
    plt.colorbar(im1, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # Saliency map
    im2 = axes[0, 3].imshow(saliency, cmap='viridis')
    axes[0, 3].set_title('Spectral Residual Saliency')
    axes[0, 3].axis('off')
    plt.colorbar(im2, ax=axes[0, 3], fraction=0.046, pad=0.04)
    
    # Combined ROI map (continuous)
    im3 = axes[1, 0].imshow(combined_roi, cmap='plasma')
    axes[1, 0].set_title('Combined ROI Map (Continuous)')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Binary ROI mask
    axes[1, 1].imshow(binary_roi, cmap='gray')
    axes[1, 1].set_title(f'Binary ROI Mask\n({np.sum(binary_roi)}/{binary_roi.size} pixels)')
    axes[1, 1].axis('off')
    
    # ROI overlay on original image
    axes[1, 2].imshow(original_image)
    # Create a red overlay for ROI regions
    roi_overlay = np.zeros((*original_image.shape[:2], 4))
    roi_overlay[binary_roi] = [1, 0, 0, 0.5]  # Red with transparency
    axes[1, 2].imshow(roi_overlay)
    axes[1, 2].set_title('ROI Overlay (Red = ROI)')
    axes[1, 2].axis('off')
    
    # Non-ROI overlay (background)
    axes[1, 3].imshow(original_image)
    # Create a blue overlay for non-ROI regions
    nonroi_overlay = np.zeros((*original_image.shape[:2], 4))
    nonroi_overlay[~binary_roi] = [0, 0, 1, 0.3]  # Blue with transparency
    axes[1, 3].imshow(nonroi_overlay)
    axes[1, 3].set_title('Non-ROI Overlay (Blue = Background)')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_individual_components(original_image, combined_roi):
    """
    Show how each component contributes to the final ROI map
    """
    # Recompute individual components for detailed analysis
    edge_map = cv2.Canny(original_image, 50, 150)
    edge_density = compute_local_density(edge_map, kernel_size=15)
    
    lab = cv2.cvtColor(original_image, cv2.COLOR_RGB2LAB)
    contrast_map = np.abs(cv2.Laplacian(lab[:,:,0], cv2.CV_32F))
    contrast_map = (contrast_map - contrast_map.min()) / (contrast_map.max() - contrast_map.min() + 1e-8)
    
    saliency_map = spectral_residual_saliency(original_image)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Edge density
    im1 = axes[0, 0].imshow(edge_density, cmap='hot')
    axes[0, 0].set_title(f'Edge Density\nrange: [{edge_density.min():.3f}, {edge_density.max():.3f}]')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Contrast map
    im2 = axes[0, 1].imshow(contrast_map, cmap='cool')
    axes[0, 1].set_title(f'Local Contrast\nrange: [{contrast_map.min():.3f}, {contrast_map.max():.3f}]')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Saliency map
    im3 = axes[0, 2].imshow(saliency_map, cmap='viridis')
    axes[0, 2].set_title(f'Saliency\nrange: [{saliency_map.min():.3f}, {saliency_map.max():.3f}]')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Histograms
    axes[1, 0].hist(edge_density.flatten(), bins=50, alpha=0.7, color='red')
    axes[1, 0].axvline(np.percentile(edge_density, 70), color='black', linestyle='--', label='70th percentile')
    axes[1, 0].set_title('Edge Density Distribution')
    axes[1, 0].legend()
    
    axes[1, 1].hist(contrast_map.flatten(), bins=50, alpha=0.7, color='blue')
    axes[1, 1].axvline(np.percentile(contrast_map, 70), color='black', linestyle='--', label='70th percentile')
    axes[1, 1].set_title('Contrast Distribution')
    axes[1, 1].legend()
    
    axes[1, 2].hist(saliency_map.flatten(), bins=50, alpha=0.7, color='green')
    axes[1, 2].axvline(np.percentile(saliency_map, 70), color='black', linestyle='--', label='70th percentile')
    axes[1, 2].set_title('Saliency Distribution')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.show()


















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




def plot_detailed_roi_analysis(original_image, binary_roi):
    """
    More detailed visualization with separate views
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 2. Pure color mask (no blending)
    color_mask = np.zeros((*original_image.shape[:2], 3), dtype=np.uint8)
    color_mask[binary_roi] = [255, 0, 0]    # Pure red for ROI
    color_mask[~binary_roi] = [0, 0, 255]   # Pure blue for nonROI
    
    axes[1].imshow(color_mask)
    axes[1].set_title('Pure ROI/nonROI Mask\n(Red=ROI, Blue=Background)')
    axes[1].axis('off')
    
    # 3. Blended overlay
    alpha = 0.3
    blended = cv2.addWeighted(original_image, 1-alpha, color_mask, alpha, 0)
    axes[2].imshow(blended)
    axes[2].set_title(f'Overlay on Original\nROI: {np.sum(binary_roi)/binary_roi.size*100:.1f}%')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

    # Print statistics
    roi_pixels = np.sum(binary_roi)
    total_pixels = binary_roi.size
    print(f"\n=== ROI Classification Results ===")
    print(f"ROI pixels (Red):    {roi_pixels} ({roi_pixels/total_pixels*100:.1f}%)")
    print(f"nonROI pixels (Blue): {total_pixels - roi_pixels} ({(total_pixels-roi_pixels)/total_pixels*100:.1f}%)")


def test_functions(test_image, visualization=False):
    """
    Test ROI detection with comprehensive visualization
    """
    # Compute all components
    edges = cv2.Canny(test_image, 50, 150)
    density = compute_local_density(edges, kernel_size=15)
    saliency = spectral_residual_saliency(test_image)
    
    # Get ROI results
    binary_roi, combined_roi = detect_roi_coarse(test_image)
    
    # Print statistics
    print("=== ROI Detection Statistics ===")
    print(f"Density map:     [{density.min():.3f}, {density.max():.3f}]")
    print(f"Saliency map:    [{saliency.min():.3f}, {saliency.max():.3f}]")
    print(f"Combined ROI:    [{combined_roi.min():.3f}, {combined_roi.max():.3f}]")
    print(f"ROI threshold:   {np.percentile(combined_roi, 70):.3f}")
    print(f"ROI coverage:    {np.sum(binary_roi)}/{binary_roi.size} pixels ({np.sum(binary_roi)/binary_roi.size*100:.1f}%)")
    
    # Create visualizations
    if visualization:
        #visualize_roi_detection(test_image, binary_roi, combined_roi, edges, density, saliency)
        visualize_individual_components(test_image, combined_roi)

        # Create the red/blue plot
        plot_detailed_roi_analysis(test_image, binary_roi)
    
    return binary_roi, combined_roi




def edge_preserving_preprocess(image):
    """
    Specifically designed for your use case: enhance edges, blur interiors
    """
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
    
    return final


if __name__ == "__main__":
    image_name = 'images/cerchio.png'
    image = cv2.imread(image_name)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mod_image=edge_preserving_preprocess(image_rgb)
    test_functions(mod_image, visualization=True)



