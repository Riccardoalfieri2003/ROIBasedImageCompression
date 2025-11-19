import cv2
import matplotlib.pyplot  as plt
import numpy as np


import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def create_circular_kernel(radius):
    """
    Create a circular kernel of given radius.
    """
    size = 2 * radius + 1
    kernel = np.zeros((size, size))
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    kernel[mask] = 1
    kernel /= kernel.sum()  # Normalize
    return kernel

def create_gaussian_circular_kernel(radius, sigma_ratio=0.3):
    """
    Create a circular kernel with Gaussian weighting.
    Closer pixels have higher weights than farther ones.
    """
    size = 2 * radius + 1
    kernel = np.zeros((size, size))
    
    # Create coordinate grid
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    distance = np.sqrt(x**2 + y**2)
    
    # Create circular mask
    circle_mask = distance <= radius
    
    # Apply Gaussian weighting within the circle
    sigma = radius * sigma_ratio  # Control falloff rate
    gaussian_weights = np.exp(-(distance**2) / (2 * sigma**2))
    
    # Combine circle mask with Gaussian weights
    kernel[circle_mask] = gaussian_weights[circle_mask]
    kernel /= kernel.sum()  # Normalize
    
    return kernel

def create_distance_weighted_kernel(radius, power=2):
    """
    Create kernel weighted by inverse distance.
    power=1: linear falloff, power=2: quadratic falloff, etc.
    """
    size = 2 * radius + 1
    kernel = np.zeros((size, size))
    
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    distance = np.sqrt(x**2 + y**2)
    
    circle_mask = distance <= radius
    
    # Inverse distance weighting (closer = higher weight)
    with np.errstate(divide='ignore', invalid='ignore'):
        weights = 1.0 / (distance**power + 1e-6)  # Avoid division by zero
    
    weights[~circle_mask] = 0
    weights[distance == 0] = 1.0  # Center pixel gets highest weight
    
    kernel = weights / weights.sum()
    
    return kernel

def create_exponential_kernel(radius, decay_rate=0.1):
    """
    Create kernel with exponential distance decay.
    """
    size = 2 * radius + 1
    kernel = np.zeros((size, size))
    
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    distance = np.sqrt(x**2 + y**2)
    
    circle_mask = distance <= radius
    
    # Exponential decay
    weights = np.exp(-decay_rate * distance)
    weights[~circle_mask] = 0
    
    kernel = weights / weights.sum()
    
    return kernel



def visualize_weighted_kernels(radius=25):
    """
    Compare different weighted kernel types.
    """
    kernels = {
        'Uniform': create_circular_kernel(radius),
        'Gaussian (σ=0.3r)': create_gaussian_circular_kernel(radius, 0.3),
        'Gaussian (σ=0.5r)': create_gaussian_circular_kernel(radius, 0.5),
        'Inverse Distance²': create_distance_weighted_kernel(radius, power=2),
        'Inverse Distance¹': create_distance_weighted_kernel(radius, power=1),
        'Exponential (fast)': create_exponential_kernel(radius, decay_rate=0.2),
        'Exponential (slow)': create_exponential_kernel(radius, decay_rate=0.05)
    }
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    
    for idx, (name, kernel) in enumerate(kernels.items()):
        if idx < len(axes):
            im = axes[idx].imshow(kernel, cmap='viridis')
            axes[idx].set_title(f'{name}\nSum: {kernel.sum():.3f}')
            axes[idx].axis('off')
            plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for idx in range(len(kernels), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()



















"""def compute_gradient_density(gradient_map, radius=50):

    # Create circular kernel
    kernel = create_circular_kernel(radius)
    
    # Compute local density using convolution
    density_map = cv2.filter2D(gradient_map.astype(np.float32), -1, kernel)
    
    return density_map"""


def compute_weighted_gradient_density(gradient_map, radius=50, kernel_type='gaussian', **kernel_params):
    """
    Compute local gradient density using weighted circular kernel.
    
    Args:
        gradient_map: Normalized gradient magnitude (0-1)
        radius: Radius of circular neighborhood
        kernel_type: 'gaussian', 'distance', 'exponential', 'uniform'
        **kernel_params: Parameters for the specific kernel type
    
    Returns:
        density_map: Weighted local gradient density (0-1)
    """
    # Create appropriate weighted kernel
    if kernel_type == 'gaussian':
        sigma_ratio = kernel_params.get('sigma_ratio', 0.3)
        kernel = create_gaussian_circular_kernel(radius, sigma_ratio)
    elif kernel_type == 'distance':
        power = kernel_params.get('power', 2)
        kernel = create_distance_weighted_kernel(radius, power)
    elif kernel_type == 'exponential':
        decay_rate = kernel_params.get('decay_rate', 0.1)
        kernel = create_exponential_kernel(radius, decay_rate)
    else:  # uniform fallback
        kernel = create_circular_kernel(radius)
    
    # Compute weighted density using convolution
    density_map = cv2.filter2D(gradient_map.astype(np.float32), -1, kernel)
    
    return density_map, kernel



def identify_roi_from_density(density_map, density_threshold=0.1):
    """
    Identify ROI pixels based on density threshold.
    
    Args:
        density_map: Local gradient density map
        density_threshold: Threshold for considering as ROI (0-1)
    
    Returns:
        roi_mask: Binary mask where True = ROI
    """
    roi_mask = density_map > density_threshold
    return roi_mask

def clean_roi_mask(roi_mask, min_region_size=100):
    """
    Remove small isolated regions from ROI mask.
    """
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        roi_mask.astype(np.uint8), connectivity=8
    )
    
    # Create cleaned mask
    cleaned_mask = np.zeros_like(roi_mask)
    
    # Keep only regions larger than min_region_size
    for i in range(1, num_labels):  # Skip background (0)
        if stats[i, cv2.CC_STAT_AREA] >= min_region_size:
            cleaned_mask[labels == i] = True
    
    return cleaned_mask

def visualize_roi_detection(original_image, gradient_normalized, density_map, roi_mask, radius):
    """
    Comprehensive visualization of the ROI detection process.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Basic images
    axes[0, 0].imshow(original_image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gradient_normalized, cmap='hot')
    axes[0, 1].set_title('Gradient Magnitude\n(Normalized 0-1)')
    axes[0, 1].axis('off')
    plt.colorbar(axes[0, 1].images[0], ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    axes[0, 2].imshow(density_map, cmap='viridis')
    axes[0, 2].set_title(f'Gradient Density\n(Radius: {radius}px)')
    axes[0, 2].axis('off')
    plt.colorbar(axes[0, 2].images[0], ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # Row 2: ROI results
    axes[1, 0].imshow(roi_mask, cmap='gray')
    roi_pixels = np.sum(roi_mask)
    total_pixels = roi_mask.size
    axes[1, 0].set_title(f'ROI Mask\n{roi_pixels}/{total_pixels} pixels\n({roi_pixels/total_pixels*100:.1f}%)')
    axes[1, 0].axis('off')
    
    # ROI overlay on original
    axes[1, 1].imshow(original_image)
    overlay = np.zeros_like(original_image)
    overlay[roi_mask] = [255, 0, 0]  # Red for ROI
    axes[1, 1].imshow(overlay, alpha=0.5)
    axes[1, 1].set_title('ROI Overlay (Red)')
    axes[1, 1].axis('off')
    
    # ROI only
    roi_only = original_image.copy()
    roi_only[~roi_mask] = 0
    axes[1, 2].imshow(roi_only)
    axes[1, 2].set_title('ROI Regions Only')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

# Quick function to get final ROI for a specific configuration
def get_final_roi(image_rgb, radius=50, density_threshold=0.1, min_region_size=100):
    """
    One-shot function to get ROI for given parameters.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # Compute gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    gradient_normalized = (gradient_magnitude - gradient_magnitude.min()) / (gradient_magnitude.max() - gradient_magnitude.min())
    
    # Compute density and ROI
    density_map = compute_weighted_gradient_density(gradient_normalized, radius)
    roi_mask = identify_roi_from_density(density_map, density_threshold)
    cleaned_roi_mask = clean_roi_mask(roi_mask, min_region_size)
    
    return cleaned_roi_mask, gradient_normalized, density_map







def compare_weighted_kernels(image_rgb, radius=50, density_threshold=0.1):
    """
    Compare different weighted kernels for ROI detection.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # Compute gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    gradient_normalized = (gradient_magnitude - gradient_magnitude.min()) / (gradient_magnitude.max() - gradient_magnitude.min())
    
    # Test different kernel types
    kernel_configs = [
        ('Uniform', 'uniform', {}),
        ('Gaussian (fast)', 'gaussian', {'sigma_ratio': 0.2}),
        ('Gaussian (slow)', 'gaussian', {'sigma_ratio': 0.5}),
        ('Inv Dist²', 'distance', {'power': 2}),
        ('Inv Dist¹', 'distance', {'power': 1}),
        ('Exp Fast', 'exponential', {'decay_rate': 0.2}),
        ('Exp Slow', 'exponential', {'decay_rate': 0.05})
    ]
    
    fig, axes = plt.subplots(3, len(kernel_configs), figsize=(20, 12))
    
    for col, (name, k_type, params) in enumerate(kernel_configs):
        # Compute weighted density
        density_map, kernel = compute_weighted_gradient_density(
            gradient_normalized, radius, k_type, **params
        )
        
        # Get ROI
        roi_mask = identify_roi_from_density(density_map, density_threshold)
        cleaned_roi_mask = clean_roi_mask(roi_mask, min_region_size=100)
        
        # Plot kernel
        axes[0, col].imshow(kernel, cmap='viridis')
        axes[0, col].set_title(f'{name}\nKernel')
        axes[0, col].axis('off')
        
        # Plot density map
        axes[1, col].imshow(density_map, cmap='viridis')
        roi_coverage = np.sum(cleaned_roi_mask) / cleaned_roi_mask.size * 100
        axes[1, col].set_title(f'Density Map\n{roi_coverage:.1f}% ROI')
        axes[1, col].axis('off')
        
        # Plot ROI overlay
        axes[2, col].imshow(image_rgb)
        overlay = np.zeros_like(image_rgb)
        overlay[cleaned_roi_mask] = [255, 0, 0]
        axes[2, col].imshow(overlay, alpha=0.5)
        axes[2, col].set_title(f'ROI Overlay\n{name}')
        axes[2, col].axis('off')
        
        print(f"{name:15}: {roi_coverage:.1f}% ROI coverage")
    
    plt.tight_layout()
    plt.show()

    #print("=== Comparing Weighted Kernels ===")
    #compare_weighted_kernels(image_rgb, radius=50, density_threshold=0.1)







def enhance_gradient_magnitude(gradient_magnitude, enhancement_type='power', factor=2.0):
    """
    Enhance gradient magnitudes, especially weak ones.
    
    Args:
        gradient_magnitude: Raw gradient magnitude from Sobel
        enhancement_type: 'power', 'exponential', 'sigmoid', 'contrast'
        factor: Enhancement strength factor
    
    Returns:
        enhanced_gradient: Enhanced gradient magnitude
    """
    if enhancement_type == 'power':
        # Power law transformation: boosts weak gradients more than strong ones
        enhanced = gradient_magnitude ** (1/factor)
        # Renormalize to original range
        enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min()) * gradient_magnitude.max()
        
    elif enhancement_type == 'exponential':
        # Exponential enhancement
        enhanced = np.exp(gradient_magnitude * factor / gradient_magnitude.max()) - 1
        enhanced = enhanced / enhanced.max() * gradient_magnitude.max()
        
    elif enhancement_type == 'sigmoid':
        # Sigmoid contrast enhancement
        normalized = gradient_magnitude / gradient_magnitude.max()
        enhanced = 1 / (1 + np.exp(-factor * (normalized - 0.5)))
        enhanced = enhanced * gradient_magnitude.max()
        
    elif enhancement_type == 'contrast':
        # Simple contrast stretching for weak gradients
        weak_mask = gradient_magnitude < gradient_magnitude.mean()
        enhanced = gradient_magnitude.copy()
        enhanced[weak_mask] = enhanced[weak_mask] * factor
        
    elif enhancement_type == 'adaptive':
        # Adaptive enhancement based on local statistics
        local_mean = cv2.blur(gradient_magnitude, (15, 15))
        local_std = np.sqrt(cv2.blur(gradient_magnitude**2, (15, 15)) - local_mean**2)
        
        # Boost gradients that are above noise level but weak
        noise_level = local_std.mean() * 0.5
        weak_but_significant = (gradient_magnitude > noise_level) & (gradient_magnitude < gradient_magnitude.mean())
        enhanced = gradient_magnitude.copy()
        enhanced[weak_but_significant] = enhanced[weak_but_significant] * factor
        
    else:
        enhanced = gradient_magnitude
    
    return enhanced

def compute_enhanced_gradients(gray, enhancement_type='power', factor=2.0):
    """
    Compute Sobel gradients with enhancement.
    """
    # Compute raw gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Enhance gradient magnitudes
    enhanced_magnitude = enhance_gradient_magnitude(gradient_magnitude, enhancement_type, factor)
    
    return enhanced_magnitude, gradient_magnitude

def compute_enhanced_sobel(gray, dx=1, dy=1, ksize=3, scale=1.0, delta=0):
    """
    Compute Sobel with adjustable parameters for stronger response.
    """
    # You can increase dx, dy values or use scale parameter
    sobelx = cv2.Sobel(gray, cv2.CV_64F, dx, 0, ksize=ksize, scale=scale, delta=delta)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, dy, ksize=ksize, scale=scale, delta=delta)
    
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    return gradient_magnitude

def multi_scale_sobel(gray, scales=[1.0, 2.0, 3.0]):
    """
    Combine Sobel responses from multiple scales.
    """
    gradients = []
    
    for scale in scales:
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3, scale=scale)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3, scale=scale)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        gradients.append(grad_mag)
    
    # Combine by taking maximum response at each pixel
    combined = np.max(gradients, axis=0)
    return combined


from clahe import get_enhanced_image

if __name__ == "__main__":
    image_name = 'images/kauai.jpg'
    image = cv2.imread(image_name)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb=get_enhanced_image(image_rgb)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # Test different enhancement methods
    enhancement_methods = [
        ('None', 'none', 1.0),
        ('Power (weak boost)', 'power', 1.5),
        #('Power (strong boost)', 'power', 2.5),
        #('Exponential', 'exponential', 3.0),
        #('Sigmoid', 'sigmoid', 5.0),
        #('Adaptive', 'adaptive', 2.0),
        #('Multi-scale', 'multiscale', 1.0)
    ]
    
    fig, axes = plt.subplots(2, len(enhancement_methods), figsize=(20, 12))
    
    for col, (name, enh_type, factor) in enumerate(enhancement_methods):

        if enh_type == 'multiscale': gradient_magnitude = multi_scale_sobel(gray)
        else: gradient_magnitude, original_magnitude = compute_enhanced_gradients(gray, enh_type, factor)
        
        # Normalize
        gradient_normalized = (gradient_magnitude - gradient_magnitude.min()) / (gradient_magnitude.max() - gradient_magnitude.min())
        
        # Compute density and ROI
        density_map, kernel = compute_weighted_gradient_density(
            gradient_normalized, radius=10, kernel_type='gaussian', sigma_ratio=0.05
        )
        
        roi_mask = identify_roi_from_density(density_map, density_threshold=0.1)
        final_roi_mask = clean_roi_mask(roi_mask, min_region_size=10)
        
        # Plot gradient comparison
        axes[0, col].imshow(gradient_normalized, cmap='hot')
        axes[0, col].set_title(f'{name}\nGradient')
        axes[0, col].axis('off')
        
        # Plot density map
        """axes[1, col].imshow(density_map, cmap='viridis')
        axes[1, col].set_title('Density Map')
        axes[1, col].axis('off')"""
        
        # Plot ROI overlay
        axes[1, col].imshow(image_rgb)
        overlay = np.zeros_like(image_rgb)
        overlay[final_roi_mask] = [255, 0, 0]
        axes[1, col].imshow(overlay, alpha=0.5)
        roi_coverage = np.sum(final_roi_mask) / final_roi_mask.size * 100
        axes[1, col].set_title(f'ROI: {roi_coverage:.1f}%')
        axes[1, col].axis('off')
        
        print(f"{name:20}: {roi_coverage:.1f}% ROI coverage")
    
    plt.tight_layout()
    plt.show()