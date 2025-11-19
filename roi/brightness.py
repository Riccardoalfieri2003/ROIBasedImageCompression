import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage


import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def detect_shadow_regions_by_gradient_smoothness(image_rgb, smoothness_threshold=0.1, shadow_intensity_threshold=80):
    """
    Detect shadow regions based on smooth transitions from bright areas.
    
    Args:
        image_rgb: Input RGB image
        smoothness_threshold: How smooth the transition must be (0-1, lower = smoother required)
        shadow_intensity_threshold: Maximum brightness to consider as potential shadow
    
    Returns:
        shadow_mask: Binary mask of shadow regions
        gradient_smoothness: Map of gradient smoothness values
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a, b = cv2.split(lab)
    
    # Step 1: Find dark regions (potential shadows)
    dark_mask = l_channel < shadow_intensity_threshold
    
    # Step 2: Compute gradient magnitude
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Step 3: Find bright regions adjacent to dark regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_dark = cv2.dilate(dark_mask.astype(np.uint8), kernel)
    bright_boundaries = dilated_dark & ~dark_mask & (l_channel > shadow_intensity_threshold + 20)
    
    # Step 4: Compute gradient smoothness at boundaries
    # Use local average gradient around boundaries
    kernel_avg = np.ones((7, 7)) / 49
    local_avg_gradient = cv2.filter2D(gradient_magnitude, -1, kernel_avg)
    
    # Normalize gradient values to 0-1 range
    gradient_normalized = local_avg_gradient / local_avg_gradient.max()
    
    # Step 5: Shadow regions are dark areas with smooth boundaries to bright regions
    shadow_mask = np.zeros_like(dark_mask, dtype=bool)
    
    # For each dark pixel, check if it has smooth transitions to bright areas
    for y in range(l_channel.shape[0]):
        for x in range(l_channel.shape[1]):
            if dark_mask[y, x]:
                # Check local neighborhood for smooth bright boundaries
                if has_smooth_bright_boundary(l_channel, gradient_normalized, bright_boundaries, 
                                            x, y, smoothness_threshold):
                    shadow_mask[y, x] = True
    
    return shadow_mask, gradient_normalized

def has_smooth_bright_boundary(l_channel, gradient_smoothness, bright_boundaries, x, y, threshold):
    """
    Check if a dark pixel has smooth transitions to nearby bright regions.
    """
    height, width = l_channel.shape
    search_radius = 15
    
    # Search in local neighborhood
    x_min = max(0, x - search_radius)
    x_max = min(width, x + search_radius + 1)
    y_min = max(0, y - search_radius)
    y_max = min(height, y + search_radius + 1)
    
    local_bright = bright_boundaries[y_min:y_max, x_min:x_max]
    local_smoothness = gradient_smoothness[y_min:y_max, x_min:x_max]
    
    # If there are bright boundaries nearby with low gradients (smooth transitions)
    if np.any(local_bright):
        smooth_bright_pixels = local_bright & (local_smoothness < threshold)
        return np.any(smooth_bright_pixels)
    
    return False

def brighten_shadow_regions_adaptive(image_rgb, shadow_mask, target_brightness_ratio=0.7):
    """
    Brighten shadow regions based on nearby bright areas.
    
    Args:
        image_rgb: Input RGB image
        shadow_mask: Binary mask of shadow regions
        target_brightness_ratio: Target brightness as fraction of nearby bright areas (0-1)
    
    Returns:
        enhanced_rgb: Image with brightened shadow regions
        brightness_reference: Map showing reference brightness used
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Find bright regions (non-shadows that are bright)
    bright_regions = (l > 120) & (~shadow_mask)
    
    # Create distance transform to find nearest bright pixel for each shadow pixel
    from scipy import ndimage
    
    # For each shadow pixel, find distance and index to nearest bright pixel
    distances, indices = ndimage.distance_transform_edt(~bright_regions, return_indices=True)
    
    # Get reference brightness from nearest bright pixel
    reference_brightness = l[indices[0], indices[1]]
    
    # Create enhanced image
    l_enhanced = l.copy().astype(np.float32)
    
    # Only enhance shadow pixels that have a reasonable reference nearby
    max_search_distance = 50  # pixels
    valid_shadows = shadow_mask & (distances < max_search_distance)
    
    # Apply adaptive brightening
    current_brightness = l[valid_shadows]
    target_brightness = reference_brightness[valid_shadows] * target_brightness_ratio
    
    # Smooth enhancement to avoid artifacts
    enhancement = target_brightness / np.maximum(current_brightness, 1)
    enhancement = np.minimum(enhancement, 3.0)  # Cap enhancement
    
    l_enhanced[valid_shadows] = np.clip(current_brightness * enhancement, 0, 255)
    
    # Convert back
    l_enhanced = l_enhanced.astype(np.uint8)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced_rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    # Create brightness reference map for visualization
    brightness_reference = np.zeros_like(l, dtype=np.float32)
    brightness_reference[valid_shadows] = reference_brightness[valid_shadows]
    
    return enhanced_rgb, brightness_reference

def fast_shadow_detection_and_enhancement(image_rgb, smoothness_threshold=0.1, target_ratio=0.7):
    """
    Fast version using vectorized operations.
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Simple shadow detection: dark areas with smooth local gradients
    dark_mask = l < 80
    
    # Compute local gradient variance (smoothness measure)
    sobelx = cv2.Sobel(l, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(l, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    
    # Local gradient standard deviation (low = smooth, high = sharp)
    kernel = np.ones((7, 7)) / 49
    local_mean = cv2.filter2D(gradient_mag, -1, kernel)
    local_sq_mean = cv2.filter2D(gradient_mag**2, -1, kernel)
    local_std = np.sqrt(local_sq_mean - local_mean**2)
    
    # Normalize
    local_std_norm = local_std / local_std.max()
    
    # Shadow = dark + smooth gradients
    shadow_mask = dark_mask & (local_std_norm < smoothness_threshold)
    
    # Brighten shadows based on global bright regions
    bright_regions = l > 150
    distances, indices = ndimage.distance_transform_edt(~bright_regions, return_indices=True)
    reference_brightness = l[indices[0], indices[1]]
    
    # Apply enhancement
    l_enhanced = l.copy().astype(np.float32)
    valid_shadows = shadow_mask & (distances < 100)
    
    current = l[valid_shadows]
    target = reference_brightness[valid_shadows] * target_ratio
    enhancement = target / np.maximum(current, 1)
    enhancement = np.minimum(enhancement, 4.0)
    
    l_enhanced[valid_shadows] = np.clip(current * enhancement, 0, 255)
    l_enhanced = l_enhanced.astype(np.uint8)
    
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced_rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    return enhanced_rgb, shadow_mask, local_std_norm

# Visualization function
def visualize_shadow_processing(original_rgb, shadow_mask, gradient_smoothness, enhanced_rgb, brightness_reference):
    """
    Visualize the shadow detection and enhancement process.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Detection process
    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(shadow_mask, cmap='gray')
    shadow_pixels = np.sum(shadow_mask)
    axes[0, 1].set_title(f'Shadow Mask\n{shadow_pixels} pixels')
    axes[0, 1].axis('off')
    
    im1 = axes[0, 2].imshow(gradient_smoothness, cmap='viridis')
    axes[0, 2].set_title('Gradient Smoothness\n(Low = Smooth, High = Sharp)')
    axes[0, 2].axis('off')
    plt.colorbar(im1, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # Row 2: Enhancement results
    axes[1, 0].imshow(enhanced_rgb)
    axes[1, 0].set_title('Enhanced Image\n(Shadows Brightened)')
    axes[1, 0].axis('off')
    
    # Difference
    diff = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)[:, :, 0] - \
           cv2.cvtColor(original_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)[:, :, 0]
    im2 = axes[1, 1].imshow(diff, cmap='coolwarm', vmin=-50, vmax=50)
    axes[1, 1].set_title('Brightness Change\n(Blue=Darkened, Red=Brightened)')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # Brightness reference
    im3 = axes[1, 2].imshow(brightness_reference, cmap='hot')
    axes[1, 2].set_title('Reference Brightness\nUsed for Enhancement')
    axes[1, 2].axis('off')
    plt.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()












def brighten_shadow_regions_uniform(image_rgb, shadow_mask, target_brightness_ratio=0.7):
    """
    Brighten entire shadow regions uniformly using region growing.
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Find connected shadow regions
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        shadow_mask.astype(np.uint8), connectivity=8
    )
    
    l_enhanced = l.copy().astype(np.float32)
    
    for i in range(1, num_labels):  # Skip background
        # Get the current shadow region
        region_mask = labels == i
        
        # Find the brightest boundary pixels of this region
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated_region = cv2.dilate(region_mask.astype(np.uint8), kernel)
        boundary = dilated_region & ~region_mask
        
        # Get reference brightness from surrounding bright areas
        boundary_brightness = l[boundary]
        if len(boundary_brightness) > 0:
            reference_brightness = np.median(boundary_brightness[boundary_brightness > 100])
            
            # Calculate enhancement for the entire region
            current_region_brightness = np.median(l[region_mask])
            target_brightness = reference_brightness * target_brightness_ratio
            
            if target_brightness > current_region_brightness:
                enhancement = target_brightness / max(current_region_brightness, 1)
                enhancement = min(enhancement, 3.0)
                
                # Apply same enhancement to entire region
                l_enhanced[region_mask] = np.clip(l[region_mask].astype(np.float32) * enhancement, 0, 255)
    
    l_enhanced = l_enhanced.astype(np.uint8)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced_rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    return enhanced_rgb





















def brighten_shadows_distance_based(image_rgb, shadow_mask, target_ratio=0.7):
    """
    Use distance transform to create smooth brightness enhancement across entire shadow regions.
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Find bright regions around shadows
    bright_regions = (l > 150) & (~shadow_mask)
    
    # For each shadow pixel, compute enhancement based on distance to nearest bright pixel
    distances, indices = ndimage.distance_transform_edt(~bright_regions, return_indices=True)
    reference_brightness = l[indices[0], indices[1]]
    
    # Create enhancement map for entire shadow regions
    l_enhanced = l.copy().astype(np.float32)
    
    # Calculate current average brightness of each shadow region
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        shadow_mask.astype(np.uint8), connectivity=8
    )
    
    for i in range(1, num_labels):
        region_mask = labels == i
        
        # Get reference brightness from nearest bright area
        region_reference = reference_brightness[region_mask]
        if len(region_reference) > 0:
            avg_reference = np.median(region_reference)
            
            # Current average brightness of this shadow region
            current_avg = np.mean(l[region_mask])
            
            # Calculate uniform enhancement for entire region
            if avg_reference > current_avg:
                target_brightness = avg_reference * target_ratio
                enhancement = target_brightness / max(current_avg, 1)
                enhancement = min(enhancement, 4.0)
                
                # Apply uniform enhancement to entire region
                l_enhanced[region_mask] = np.clip(l[region_mask] * enhancement, 0, 255)
    
    l_enhanced = l_enhanced.astype(np.uint8)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced_rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    return enhanced_rgb


def iterative_shadow_brightening(image_rgb, max_iterations=3, shadow_threshold=80, target_ratio=0.7):
    """
    Iteratively detect and brighten shadows until no significant shadows remain.
    """
    current_image = image_rgb.copy()
    enhancement_history = []
    
    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}")
        
        # Convert to LAB
        lab = cv2.cvtColor(current_image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Detect shadows in current image
        shadow_mask = detect_shadows_fast(l, shadow_threshold)
        
        # Check if we have shadows to process
        shadow_pixels = np.sum(shadow_mask)
        if shadow_pixels == 0:
            print("No more shadows detected - stopping iteration")
            break
        
        print(f"  Shadows detected: {shadow_pixels} pixels")
        
        # Brighten shadow regions uniformly
        enhanced_rgb = brighten_shadow_regions_uniform(current_image, shadow_mask, target_ratio)
        enhancement_history.append((current_image.copy(), shadow_mask, enhanced_rgb.copy()))
        
        # Update for next iteration
        current_image = enhanced_rgb
    
    return current_image, enhancement_history

def detect_shadows_fast(l_channel, shadow_threshold=80):
    """
    Fast shadow detection based on intensity and local contrast.
    """
    # Simple approach: dark areas with low local contrast
    dark_mask = l_channel < shadow_threshold
    
    # Compute local contrast (standard deviation)
    kernel = np.ones((7, 7)) / 49
    local_mean = cv2.filter2D(l_channel.astype(np.float32), -1, kernel)
    local_sq_mean = cv2.filter2D(l_channel.astype(np.float32)**2, -1, kernel)
    local_contrast = np.sqrt(local_sq_mean - local_mean**2)
    
    # Shadows are dark areas with low contrast
    shadow_mask = dark_mask & (local_contrast < 20)
    
    return shadow_mask

def poisson_shadow_enhancement(image_rgb, shadow_mask, target_ratio=0.7):
    """
    Use Poisson equation to create smooth, uniform brightness enhancement.
    More computationally expensive but very high quality.
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Find bright boundaries around shadow regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_shadows = cv2.dilate(shadow_mask.astype(np.uint8), kernel)
    shadow_boundaries = dilated_shadows & ~shadow_mask
    
    # Set boundary conditions: shadow boundaries should match surrounding brightness
    l_enhanced = l.copy().astype(np.float32)
    
    # For each connected shadow region
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        shadow_mask.astype(np.uint8), connectivity=8
    )
    
    for i in range(1, num_labels):
        region_mask = labels == i
        
        # Get boundary brightness
        region_dilated = cv2.dilate(region_mask.astype(np.uint8), kernel)
        region_boundary = region_dilated & ~region_mask
        boundary_brightness = l[region_boundary]
        
        if len(boundary_brightness) > 0:
            target_brightness = np.median(boundary_brightness) * target_ratio
            current_avg = np.mean(l[region_mask])
            
            if target_brightness > current_avg:
                # Simple uniform enhancement (approximation of Poisson solution)
                enhancement = target_brightness / max(current_avg, 1)
                enhancement = min(enhancement, 3.0)
                
                # Apply to entire region
                l_enhanced[region_mask] = np.clip(l[region_mask] * enhancement, 0, 255)
    
    l_enhanced = l_enhanced.astype(np.uint8)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced_rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    return enhanced_rgb


def compare_shadow_enhancement_methods(image_rgb):
    """
    Compare different shadow enhancement methods.
    """
    # Detect shadows first
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l = lab[:, :, 0]
    shadow_mask = detect_shadows_fast(l)
    
    methods = [
        ('Region Growing', brighten_shadow_regions_uniform),
        ('Distance-Based', brighten_shadows_distance_based),
        ('Poisson-Based', poisson_shadow_enhancement),
    ]
    
    fig, axes = plt.subplots(2, len(methods) + 1, figsize=(20, 10))
    
    # Original image and shadow mask
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(shadow_mask, cmap='gray')
    axes[1, 0].set_title(f'Shadow Mask\n{np.sum(shadow_mask)} pixels')
    axes[1, 0].axis('off')
    
    # Test each method
    for col, (name, method) in enumerate(methods, 1):

        enhanced_rgb = method(image_rgb, shadow_mask)
        
        # Show enhanced result
        axes[0, col].imshow(enhanced_rgb)
        axes[0, col].set_title(f'{name}\nEnhanced')
        axes[0, col].axis('off')
        
        # Show brightness difference
        lab_orig = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)[:, :, 0]
        lab_enh = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2LAB)[:, :, 0]
        brightness_diff = lab_enh.astype(np.float32) - lab_orig.astype(np.float32)
        
        im = axes[1, col].imshow(brightness_diff, cmap='coolwarm', vmin=-50, vmax=50)
        axes[1, col].set_title('Brightness Change')
        axes[1, col].axis('off')
        plt.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)
        
        # Calculate statistics
        shadow_enhancement = brightness_diff[shadow_mask]
        avg_enhancement = np.mean(shadow_enhancement) if len(shadow_enhancement) > 0 else 0
        print(f"{name}: Average shadow brightening = {avg_enhancement:.1f} levels")
    
    plt.tight_layout()
    plt.show()
    
    # Test iterative approach
    print("\n=== Testing Iterative Approach ===")
    final_image, history = iterative_shadow_brightening(image_rgb, max_iterations=3)
    
    # Show iterative results
    fig, axes = plt.subplots(2, len(history) + 1, figsize=(20, 8))
    
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(detect_shadows_fast(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)[:, :, 0]))
    axes[1, 0].set_title('Initial Shadows')
    axes[1, 0].axis('off')
    
    for i, (original, mask, enhanced) in enumerate(history, 1):
        axes[0, i].imshow(enhanced)
        axes[0, i].set_title(f'Iteration {i}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(detect_shadows_fast(cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)[:, :, 0]))
        remaining_shadows = np.sum(detect_shadows_fast(cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)[:, :, 0]))
        axes[1, i].set_title(f'Remaining: {remaining_shadows} px')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()





































import time

def ultra_fast_shadow_enhancement(image_rgb, shadow_intensity_threshold=80):
    """
    Super fast shadow enhancement using CLAHE on LAB space.
    """
    # Convert to LAB color space (much faster than iterative methods)
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Simple shadow detection: just use intensity threshold
    shadow_mask = l < shadow_intensity_threshold
    
    # Apply CLAHE only to shadow regions
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    # Only replace shadow regions
    l_final = l.copy()
    l_final[shadow_mask] = l_enhanced[shadow_mask]
    
    # Convert back to RGB
    lab_enhanced = cv2.merge([l_final, a, b])
    enhanced_rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    return enhanced_rgb, shadow_mask


def fast_gamma_shadow_enhancement(image_rgb, shadow_threshold=80, gamma=0.6):
    """
    Lightning-fast gamma correction for shadow regions.
    """
    # Convert to LAB
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Create shadow mask
    shadow_mask = l < shadow_threshold
    
    # Apply gamma correction only to shadow regions
    l_enhanced = l.copy().astype(np.float32)
    l_enhanced[shadow_mask] = 255 * (l_enhanced[shadow_mask] / 255) ** gamma
    
    # Convert back
    l_enhanced = l_enhanced.astype(np.uint8)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced_rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    return enhanced_rgb, shadow_mask

def lut_shadow_enhancement(image_rgb, shadow_threshold=80, boost_factor=2.0):
    """
    Fastest method using Lookup Tables - real-time performance.
    """
    # Convert to LAB
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Create lookup table for shadow enhancement
    lut = np.arange(256, dtype=np.uint8)
    
    # Apply boost only to dark pixels (shadows)
    shadow_pixels = lut < shadow_threshold
    lut[shadow_pixels] = np.clip(lut[shadow_pixels] * boost_factor, 0, 255).astype(np.uint8)
    
    # Apply LUT (extremely fast)
    l_enhanced = cv2.LUT(l, lut)
    
    # Create shadow mask for visualization
    shadow_mask = l < shadow_threshold
    
    # Convert back
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced_rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    return enhanced_rgb, shadow_mask

def fast_adaptive_brightness(image_rgb, shadow_threshold=80, target_brightness=120):
    """
    Fast adaptive brightness adjustment for shadows.
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Simple shadow mask
    shadow_mask = l < shadow_threshold
    
    # Calculate required brightness adjustment
    l_enhanced = l.copy().astype(np.float32)
    
    # Vectorized operation - very fast
    shadow_pixels = shadow_mask
    current_shadow_brightness = np.mean(l[shadow_pixels]) if np.any(shadow_pixels) else 0
    
    if current_shadow_brightness > 0:
        enhancement = target_brightness / current_shadow_brightness
        enhancement = min(enhancement, 3.0)  # Cap enhancement
        
        l_enhanced[shadow_pixels] = np.clip(l[shadow_pixels] * enhancement, 0, 255)
    
    l_enhanced = l_enhanced.astype(np.uint8)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced_rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    return enhanced_rgb, shadow_mask










def benchmark_shadow_methods(image_rgb):
    """
    Benchmark all fast shadow enhancement methods.
    """
    methods = [
        ('CLAHE', ultra_fast_shadow_enhancement),
        #('Gamma Correction', fast_gamma_shadow_enhancement),
        #('LUT (Fastest)', lut_shadow_enhancement),
        #('Adaptive Brightness', fast_adaptive_brightness),
    ]
    
    results = []
    
    for name, method in methods:
        start_time = time.time()
        
        if name == 'CLAHE':
            enhanced, mask = ultra_fast_shadow_enhancement(image_rgb)
        elif name == 'Gamma Correction':
            enhanced, mask = fast_gamma_shadow_enhancement(image_rgb, gamma=0.5)
        elif name == 'LUT (Fastest)':
            enhanced, mask = lut_shadow_enhancement(image_rgb, boost_factor=2.0)
        else:
            enhanced, mask = fast_adaptive_brightness(image_rgb)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Calculate improvement
        lab_orig = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)[:, :, 0]
        lab_enh = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)[:, :, 0]
        shadow_improvement = np.mean(lab_enh[mask] - lab_orig[mask]) if np.any(mask) else 0
        
        results.append((name, enhanced, mask, processing_time, shadow_improvement))
        print(f"{name:20}: {processing_time:.4f}s, Shadow improvement: {shadow_improvement:.1f} levels")
    
    return results

def visualize_fast_results(image_rgb, results):
    """
    Visualize results from fast methods.
    """
    fig, axes = plt.subplots(3, len(results) + 1, figsize=(20, 12))
    
    # Original image
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Original shadows
    lab_orig = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    shadow_mask_orig = lab_orig[:, :, 0] < 80
    axes[1, 0].imshow(shadow_mask_orig, cmap='gray')
    axes[1, 0].set_title('Original Shadows')
    axes[1, 0].axis('off')
    
    # Empty for brightness diff
    axes[2, 0].axis('off')
    
    # Results for each method
    for col, (name, enhanced, mask, time_val, improvement) in enumerate(results, 1):
        # Enhanced image
        axes[0, col].imshow(enhanced)
        axes[0, col].set_title(f'{name}\n{time_val:.3f}s')
        axes[0, col].axis('off')
        
        # Shadow mask
        axes[1, col].imshow(mask, cmap='gray')
        shadow_pixels = np.sum(mask)
        axes[1, col].set_title(f'Shadows: {shadow_pixels} px')
        axes[1, col].axis('off')
        
        # Brightness difference
        lab_enh = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)[:, :, 0]
        brightness_diff = lab_enh.astype(np.float32) - lab_orig[:, :, 0].astype(np.float32)
        
        im = axes[2, col].imshow(brightness_diff, cmap='coolwarm', vmin=-30, vmax=30)
        axes[2, col].set_title(f'Improvement: {improvement:.1f} levels')
        axes[2, col].axis('off')
        plt.colorbar(im, ax=axes[2, col], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()


def enhanced_clahe_for_shadows(image_rgb, shadow_threshold=80, clip_limit=4.0, tile_size=4):
    """
    CLAHE optimized for shadow enhancement.
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Identify shadow regions first
    shadow_mask = l < shadow_threshold
    
    # Apply aggressive CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_enhanced = clahe.apply(l)
    
    # Only keep the enhancement in shadow regions
    l_final = l.copy()
    l_final[shadow_mask] = l_enhanced[shadow_mask]
    
    # Convert back
    lab_enhanced = cv2.merge([l_final, a, b])
    enhanced_rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    return enhanced_rgb, shadow_mask

def clahe_with_shadow_boost(image_rgb, shadow_boost=2.0):
    """
    Combine CLAHE with additional shadow boosting.
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Step 1: Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    
    # Step 2: Additional boost for very dark areas
    very_dark_mask = l < 60
    l_final = l_clahe.copy()
    l_final[very_dark_mask] = np.clip(l_clahe[very_dark_mask] * shadow_boost, 0, 255)
    
    # Convert back
    lab_enhanced = cv2.merge([l_final, a, b])
    enhanced_rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    return enhanced_rgb, very_dark_mask


"""
# Main execution - SUPER FAST
if __name__ == "__main__":
    image_name = 'images/kauai.jpg'
    image = cv2.imread(image_name)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print("=== FAST Shadow Enhancement Methods ===")
    print(f"Image size: {image_rgb.shape[1]}x{image_rgb.shape[0]}")
    
    # Benchmark all methods
    results = benchmark_shadow_methods(image_rgb)
    
    # Visualize results
    visualize_fast_results(image_rgb, results)
    
    # Use the fastest method for your pipeline
    print("\n=== Using Fastest Method (LUT) for ROI Pipeline ===")
    enhanced_rgb, shadow_mask = lut_shadow_enhancement(image_rgb, boost_factor=2.0)
    
    # Continue with your gradient-based ROI detection...
    gray_enhanced = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray_enhanced, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_enhanced, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    print("Enhanced image ready for ROI detection!")"""


def clahe_shadow_debugger(image_rgb):
    """
    Comprehensive debugger for CLAHE shadow enhancement.
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Test different CLAHE parameters
    parameters = [
        {'clipLimit': 2.0, 'tileGridSize': (8, 8), 'name': 'Default'},
        {'clipLimit': 4.0, 'tileGridSize': (8, 8), 'name': 'High Clip'},
        {'clipLimit': 2.0, 'tileGridSize': (4, 4), 'name': 'Small Tiles'},
        {'clipLimit': 4.0, 'tileGridSize': (4, 4), 'name': 'Aggressive'},
    ]
    
    shadow_threshold = 80
    
    fig, axes = plt.subplots(3, len(parameters), figsize=(20, 12))
    
    for col, params in enumerate(parameters):
        # Apply CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=params['clipLimit'], 
            tileGridSize=params['tileGridSize']
        )
        l_enhanced = clahe.apply(l)
        
        # Calculate statistics
        shadow_mask = l < shadow_threshold
        shadow_improvement = np.mean(l_enhanced[shadow_mask] - l[shadow_mask])
        still_dark = np.sum((l_enhanced < shadow_threshold) & shadow_mask)
        
        # Plot results
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced_rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        
        axes[0, col].imshow(enhanced_rgb)
        axes[0, col].set_title(f"{params['name']}\nClip: {params['clipLimit']}, Tile: {params['tileGridSize']}")
        axes[0, col].axis('off')
        
        # Shadow enhancement
        diff = l_enhanced.astype(np.float32) - l.astype(np.float32)
        im = axes[1, col].imshow(diff, cmap='coolwarm', vmin=-30, vmax=30)
        axes[1, col].set_title(f'Improvement: {shadow_improvement:.1f} levels')
        axes[1, col].axis('off')
        plt.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)
        
        # Remaining dark areas
        remaining_dark = (l_enhanced < shadow_threshold) & shadow_mask
        axes[2, col].imshow(remaining_dark, cmap='gray')
        axes[2, col].set_title(f'Still Dark: {still_dark} pixels')
        axes[2, col].axis('off')
    
    plt.tight_layout()
    plt.show()




























def selective_clahe_shadows_only(image_rgb, shadow_threshold=80, clip_limit=3.0, tile_size=8):
    """
    Apply CLAHE ONLY to shadow regions, leave other areas untouched.
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Identify shadow regions
    shadow_mask = l < shadow_threshold
    print(f"Shadow pixels detected: {np.sum(shadow_mask)}")
    
    # Extract ONLY the shadow regions as a 1D array
    shadow_indices = np.where(shadow_mask)
    shadow_pixels = l[shadow_mask]  # This is already 1D
    
    if len(shadow_pixels) > 0:
        # Apply CLAHE only to the shadow pixels
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        
        # Reshape to 2D for CLAHE (it expects 2D input)
        height, width = shadow_pixels.shape[0], 1
        shadow_pixels_2d = shadow_pixels.reshape(height, width)
        
        # Apply CLAHE
        enhanced_shadows_2d = clahe.apply(shadow_pixels_2d)
        
        # Convert back to 1D
        enhanced_shadows = enhanced_shadows_2d.flatten()
        
        # Put enhanced shadows back into the original image
        l_enhanced = l.copy()
        l_enhanced[shadow_mask] = enhanced_shadows
    else:
        l_enhanced = l.copy()
    
    # Convert back to RGB
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced_rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    return enhanced_rgb, shadow_mask

def isolated_shadow_clahe(image_rgb, shadow_threshold=80, clip_limit=4.0):
    """
    Process each shadow region individually to avoid affecting non-shadow areas.
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Find connected shadow regions
    shadow_mask = l < shadow_threshold
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        shadow_mask.astype(np.uint8), connectivity=8
    )
    
    l_enhanced = l.copy()
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(4, 4))
    
    for i in range(1, num_labels):  # Skip background (0)
        # Get individual shadow region
        region_mask = labels == i
        region_pixels = l[region_mask]
        
        # Apply CLAHE only to this region
        # Reshape to 2D for CLAHE, then flatten back to 1D
        region_pixels_2d = region_pixels.reshape(-1, 1)
        enhanced_region_2d = clahe.apply(region_pixels_2d)
        enhanced_region = enhanced_region_2d.flatten()  # Convert back to 1D

        l_enhanced[region_mask] = enhanced_region
    
    # Convert back
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced_rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    return enhanced_rgb, shadow_mask

def simple_shadow_brightening(image_rgb, shadow_threshold=80, brightness_boost=1.8):
    """
    Simple brightness boost for shadows without contrast enhancement.
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Identify shadows
    shadow_mask = l < shadow_threshold
    
    # Simple brightness boost (no contrast change)
    l_enhanced = l.copy().astype(np.float32)
    l_enhanced[shadow_mask] = np.clip(l_enhanced[shadow_mask] * brightness_boost, 0, 255)
    l_enhanced = l_enhanced.astype(np.uint8)
    
    # Convert back
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced_rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    return enhanced_rgb, shadow_mask

def compare_shadow_methods(image_rgb):
    """
    Compare different shadow brightening methods.
    """
    methods = [
        ('Selective CLAHE', selective_clahe_shadows_only),
        ('Isolated CLAHE', isolated_shadow_clahe), 
        ('Simple Brightening', simple_shadow_brightening),
    ]
    
    fig, axes = plt.subplots(2, len(methods) + 1, figsize=(18, 8))
    
    # Original image
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Original shadows
    lab_orig = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    shadow_mask_orig = lab_orig[:, :, 0] < 80
    axes[1, 0].imshow(shadow_mask_orig, cmap='gray')
    axes[1, 0].set_title('Original Shadows')
    axes[1, 0].axis('off')
    
    for col, (name, method) in enumerate(methods, 1):
        if 'CLAHE' in name:
            enhanced, mask = method(image_rgb, clip_limit=4.0)
        else:
            enhanced, mask = method(image_rgb, brightness_boost=2.0)
        
        # Enhanced image
        axes[0, col].imshow(enhanced)
        axes[0, col].set_title(name)
        axes[0, col].axis('off')
        
        # Brightness difference (shadows only)
        lab_orig = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)[:, :, 0]
        lab_enh = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)[:, :, 0]
        
        # Only show difference in shadow regions
        brightness_diff = np.zeros_like(lab_orig, dtype=np.float32)
        brightness_diff[mask] = lab_enh[mask] - lab_orig[mask]
        
        im = axes[1, col].imshow(brightness_diff, cmap='RdYlBu', vmin=0, vmax=50)
        axes[1, col].set_title('Shadow Brightening')
        axes[1, col].axis('off')
        plt.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)
        
        # Statistics
        shadow_improvement = np.mean(lab_enh[mask] - lab_orig[mask])
        print(f"{name}: Average shadow brightening = {shadow_improvement:.1f} levels")
    
    plt.tight_layout()
    plt.show()

# Test which areas are actually being modified
def debug_shadow_enhancement(image_rgb):
    """
    Debug exactly what's being enhanced.
    """
    # Use selective CLAHE
    enhanced, shadow_mask = selective_clahe_shadows_only(image_rgb)
    
    lab_orig = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    lab_enh = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
    
    # Calculate changes
    l_orig = lab_orig[:, :, 0]
    l_enh = lab_enh[:, :, 0]
    
    # Find pixels that actually changed
    changed_pixels = l_enh != l_orig
    shadow_pixels_changed = changed_pixels & shadow_mask
    non_shadow_pixels_changed = changed_pixels & ~shadow_mask
    
    print(f"Total pixels changed: {np.sum(changed_pixels)}")
    print(f"Shadow pixels changed: {np.sum(shadow_pixels_changed)}")
    print(f"Non-shadow pixels changed: {np.sum(non_shadow_pixels_changed)}")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(enhanced)
    axes[0].set_title('Enhanced Image')
    axes[0].axis('off')
    
    axes[1].imshow(shadow_pixels_changed, cmap='hot')
    axes[1].set_title('Shadow Pixels That Changed')
    axes[1].axis('off')
    
    axes[2].imshow(non_shadow_pixels_changed, cmap='hot')
    axes[2].set_title('Non-Shadow Pixels That Changed\n(Should be ZERO)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
"""
# Main execution
if __name__ == "__main__":
    image_name = 'images/kauai.jpg'
    image = cv2.imread(image_name)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print("=== Shadow-Only Enhancement Methods ===")
    
    # Debug first
    debug_shadow_enhancement(image_rgb)
    
    # Compare methods
    compare_shadow_methods(image_rgb)
    
    # Use the best method for your pipeline
    print("\n=== Using Selective CLAHE for ROI Pipeline ===")
    enhanced_rgb, shadow_mask = selective_clahe_shadows_only(image_rgb, clip_limit=4.0)
    
    # Continue with your gradient computation...
    gray_enhanced = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2GRAY)
    # ... your ROI detection code"""







































def aggressive_shadow_brightening(image_rgb, shadow_threshold=80, clip_limit=8.0, tile_size=4, manual_boost=2.0):
    """
    Aggressive shadow brightening with manual boost for very dark areas.
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Identify shadow regions
    shadow_mask = l < shadow_threshold
    print(f"Shadow pixels detected: {np.sum(shadow_mask)}")
    
    if np.sum(shadow_mask) > 0:
        # Extract shadow pixels
        shadow_pixels = l[shadow_mask]
        
        # Step 1: Apply aggressive CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        shadow_pixels_2d = shadow_pixels.reshape(-1, 1)
        enhanced_shadows_clahe = clahe.apply(shadow_pixels_2d).flatten()
        
        # Step 2: Manual boost for very dark areas
        very_dark_mask = shadow_pixels < 40  # Very dark pixels within shadows
        enhanced_shadows = enhanced_shadows_clahe.copy()
        
        if np.any(very_dark_mask):
            # Apply extra boost to very dark pixels
            enhanced_shadows[very_dark_mask] = np.clip(
                enhanced_shadows_clahe[very_dark_mask] * manual_boost, 0, 255
            )
        
        # Step 3: Ensure minimum brightness
        min_brightness = 60  # Force shadows to be at least this bright
        too_dark = enhanced_shadows < min_brightness
        enhanced_shadows[too_dark] = min_brightness
        
        # Put back into image
        l_enhanced = l.copy()
        l_enhanced[shadow_mask] = enhanced_shadows
    else:
        l_enhanced = l.copy()
    
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced_rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    return enhanced_rgb, shadow_mask

def target_brightness_shadows(image_rgb, shadow_threshold=80, target_brightness=120):
    """
    Force all shadow regions to reach a target brightness level.
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Identify shadow regions
    shadow_mask = l < shadow_threshold
    
    l_enhanced = l.copy().astype(np.float32)
    
    if np.any(shadow_mask):
        # Calculate required boost for each shadow pixel
        shadow_pixels = l[shadow_mask]
        
        # Simple: boost all shadows to at least target brightness
        current_avg = np.mean(shadow_pixels)
        if current_avg > 0:
            # Individual pixel boost (not average-based)
            for i, pixel_val in enumerate(shadow_pixels):
                if pixel_val < target_brightness:
                    # Stronger boost for darker pixels
                    darkness = 1.0 - (pixel_val / target_brightness)
                    boost = 1.0 + (darkness * 3.0)  # 1x to 4x boost
                    new_val = pixel_val * boost
                    shadow_pixels[i] = min(new_val, target_brightness)
            
            l_enhanced[shadow_mask] = np.clip(shadow_pixels, 0, 255)
    
    l_enhanced = l_enhanced.astype(np.uint8)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced_rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    return enhanced_rgb, shadow_mask


def histogram_stretch_shadows(image_rgb, shadow_threshold=80, stretch_factor=0.9):
    """
    Stretch the histogram of shadow regions to use the full brightness range.
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Identify shadow regions
    shadow_mask = l < shadow_threshold
    
    l_enhanced = l.copy()
    
    if np.any(shadow_mask):
        shadow_pixels = l[shadow_mask]
        
        # Get current range of shadow pixels
        min_val = shadow_pixels.min()
        max_val = shadow_pixels.max()
        
        print(f"Shadow range: {min_val}-{max_val}")
        
        if max_val > min_val:
            # Stretch to use more of the brightness range
            new_min = min_val
            new_max = min(255, int(max_val + (255 - max_val) * stretch_factor))
            
            # Apply linear stretch
            stretched = (shadow_pixels - min_val) * (new_max - new_min) / (max_val - min_val) + new_min
            stretched = np.clip(stretched, 0, 255).astype(np.uint8)
            
            l_enhanced[shadow_mask] = stretched
            
            print(f"Stretched to: {new_min}-{new_max}")
    
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced_rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    return enhanced_rgb, shadow_mask

def lut_aggressive_shadows(image_rgb, shadow_threshold=80, dark_boost=3.0, medium_boost=2.0):
    """
    Use lookup tables with different boosts for different darkness levels.
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Create aggressive lookup table
    lut = np.arange(256, dtype=np.uint8)
    
    # Different boosts for different intensity levels
    very_dark = lut < 40
    medium_dark = (lut >= 40) & (lut < shadow_threshold)
    
    # Apply aggressive boosts
    lut[very_dark] = np.clip(lut[very_dark] * dark_boost, 0, 255).astype(np.uint8)
    lut[medium_dark] = np.clip(lut[medium_dark] * medium_boost, 0, 255).astype(np.uint8)
    
    # Apply LUT
    l_enhanced = cv2.LUT(l, lut)
    
    # Shadow mask for visualization
    shadow_mask = l < shadow_threshold
    
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced_rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    return enhanced_rgb, shadow_mask


def compare_aggressive_methods(image_rgb):
    """
    Compare different aggressive shadow brightening methods.
    """
    methods = [
        ('Aggressive CLAHE + Boost', lambda img: aggressive_shadow_brightening(img, clip_limit=4.0, manual_boost=1.5)),
        ('Target Brightness', lambda img: target_brightness_shadows(img, target_brightness=130)),
        ('Histogram Stretch', lambda img: histogram_stretch_shadows(img, stretch_factor=0.95)),
        ('LUT Aggressive', lambda img: lut_aggressive_shadows(img, dark_boost=4.0, medium_boost=2.5)),
    ]
    
    fig, axes = plt.subplots(2, len(methods) + 1, figsize=(20, 8))
    
    # Original
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    lab_orig = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    shadow_orig = lab_orig[:, :, 0] < 80
    axes[1, 0].imshow(shadow_orig, cmap='gray')
    axes[1, 0].set_title(f'Original Shadows\n{np.sum(shadow_orig)} px')
    axes[1, 0].axis('off')
    
    for col, (name, method) in enumerate(methods, 1):
        enhanced, shadow_mask = method(image_rgb)
        
        # Enhanced image
        axes[0, col].imshow(enhanced)
        axes[0, col].set_title(name)
        axes[0, col].axis('off')
        
        # Brightness improvement
        lab_enh = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)[:, :, 0]
        brightness_diff = np.zeros_like(lab_orig[:, :, 0], dtype=np.float32)
        brightness_diff[shadow_mask] = lab_enh[shadow_mask] - lab_orig[:, :, 0][shadow_mask]
        
        im = axes[1, col].imshow(brightness_diff, cmap='RdYlBu', vmin=0, vmax=80)
        axes[1, col].set_title('Brightening')
        axes[1, col].axis('off')
        plt.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)
        
        # Statistics
        shadow_improvement = np.mean(lab_enh[shadow_mask] - lab_orig[:, :, 0][shadow_mask])
        min_after = lab_enh[shadow_mask].min()
        print(f"{name}: Avg boost = {shadow_improvement:.1f}, Min after = {min_after}")
    
    plt.tight_layout()
    plt.show()

# Test the aggressive methods
if __name__ == "__main__":
    image_name = 'images/kauai.jpg'
    image = cv2.imread(image_name)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print("=== Aggressive Shadow Brightening Methods ===")
    compare_aggressive_methods(image_rgb)