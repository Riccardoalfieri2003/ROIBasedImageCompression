import cv2
import matplotlib.pyplot  as plt
import numpy as np


def detect_shadow_regions(image_rgb, shadow_threshold=40):
    """
    Detect shadow regions based on low intensity and low local contrast.
    """
    # Convert to LAB and get lightness channel
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a, b = cv2.split(lab)
    
    # Shadow mask: regions with low lightness
    low_light_mask = l_channel < shadow_threshold
    
    # Also consider low local contrast regions
    local_std = cv2.blur((l_channel - cv2.blur(l_channel, (15, 15)))**2, (15, 15))
    low_contrast_mask = np.sqrt(local_std) < 15
    
    # Combine masks
    shadow_mask = low_light_mask & low_contrast_mask
    
    return shadow_mask, l_channel

def selective_shadow_enhancement(image_rgb, enhancement_factor=2.0):
    """
    Enhance ONLY the shadow regions, leave other regions untouched.
    """
    # Detect shadow regions
    shadow_mask, l_channel = detect_shadow_regions(image_rgb)
    
    # Convert to LAB for processing
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Create enhanced version
    l_enhanced = l.copy()
    
    # Apply enhancement ONLY to shadow regions
    shadow_indices = shadow_mask
    l_enhanced[shadow_indices] = np.clip(l[shadow_indices] * enhancement_factor, 0, 255)
    
    # Merge back
    lab_enhanced = cv2.merge([l_enhanced.astype(np.uint8), a, b])
    enhanced_rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    return enhanced_rgb, shadow_mask


def adaptive_local_enhancement(image_rgb):
    """
    Apply different enhancement levels based on local brightness.
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Calculate local brightness
    local_brightness = cv2.blur(l, (30, 30))  # Large kernel for regional brightness
    
    # Create enhancement map - stronger enhancement for darker regions
    enhancement_map = np.ones_like(l, dtype=np.float32)
    enhancement_map[local_brightness < 50] = 2.0    # Very dark: 2x enhancement
    enhancement_map[local_brightness < 100] = 1.5   # Dark: 1.5x enhancement  
    enhancement_map[local_brightness < 150] = 1.2   # Medium: 1.2x enhancement
    # Bright regions: no enhancement (1.0)
    
    # Apply adaptive enhancement
    l_enhanced = np.clip(l.astype(np.float32) * enhancement_map, 0, 255).astype(np.uint8)
    
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced_rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    return enhanced_rgb, enhancement_map

def enhance_gradients_in_shadows(image_rgb):
    """
    Compute gradients first, then enhance weak gradients in shadow regions.
    """
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # Compute original gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag_original = np.sqrt(sobelx**2 + sobely**2)
    
    # Detect shadow regions (low intensity areas)
    shadow_mask = gray < np.percentile(gray, 30)  # Bottom 30% intensity
    
    # Enhance gradients ONLY in shadow regions
    grad_mag_enhanced = grad_mag_original.copy()
    
    # Boost weak gradients in shadows
    shadow_gradients = grad_mag_original[shadow_mask]
    if len(shadow_gradients) > 0:
        shadow_mean = np.mean(shadow_gradients)
        shadow_std = np.std(shadow_gradients)
        
        # Enhance gradients in shadows that are above noise level
        enhancement_mask = shadow_mask & (grad_mag_original > shadow_std)
        grad_mag_enhanced[enhancement_mask] = grad_mag_original[enhancement_mask] * 2.0
    
    return grad_mag_original, grad_mag_enhanced, shadow_mask



























def brighten_shadow_regions(image_rgb, shadow_mask, brightness_boost=1.5):
    """
    Brighten ONLY the shadow regions identified by the mask.
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Create enhanced version
    l_enhanced = l.copy().astype(np.float32)
    
    # Apply brightness boost ONLY to shadow regions
    l_enhanced[shadow_mask] = np.clip(l_enhanced[shadow_mask] * brightness_boost, 0, 255)
    
    # Convert back to uint8 and merge
    l_enhanced = l_enhanced.astype(np.uint8)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced_rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    return enhanced_rgb

def adaptive_brightness_boost(image_rgb, shadow_mask, max_boost=2.0):
    """
    Apply adaptive brightness boost - stronger for darker shadow regions.
    """
    # Convert to LAB
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    l_enhanced = l.copy().astype(np.float32)
    
    # Calculate how dark each shadow region is
    shadow_intensities = l[shadow_mask]
    if len(shadow_intensities) > 0:
        min_intensity = shadow_intensities.min()
        max_intensity = shadow_intensities.max()
        
        # Adaptive boost: darker pixels get more boost
        for i in range(l.shape[0]):
            for j in range(l.shape[1]):
                if shadow_mask[i, j]:
                    intensity = l[i, j]
                    # Normalize to 0-1 based on shadow darkness
                    darkness = 1.0 - (intensity - min_intensity) / (max_intensity - min_intensity + 1e-6)
                    boost = 1.0 + (max_boost - 1.0) * darkness
                    l_enhanced[i, j] = np.clip(l[i, j] * boost, 0, 255)
    
    l_enhanced = l_enhanced.astype(np.uint8)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced_rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    return enhanced_rgb

def compute_gradients_on_enhanced_image(original_rgb, shadow_mask):
    """
    Brighten shadow regions first, then compute gradients.
    """
    # Brighten shadow regions
    enhanced_rgb = brighten_shadow_regions(original_rgb, shadow_mask, brightness_boost=1.8)
    
    # Convert to grayscale for gradient computation
    gray_original = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
    gray_enhanced = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2GRAY)
    
    # Compute gradients on both versions
    sobelx_orig = cv2.Sobel(gray_original, cv2.CV_64F, 1, 0, ksize=3)
    sobely_orig = cv2.Sobel(gray_original, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag_orig = np.sqrt(sobelx_orig**2 + sobely_orig**2)
    
    sobelx_enh = cv2.Sobel(gray_enhanced, cv2.CV_64F, 1, 0, ksize=3)
    sobely_enh = cv2.Sobel(gray_enhanced, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag_enh = np.sqrt(sobelx_enh**2 + sobely_enh**2)
    
    return enhanced_rgb, grad_mag_orig, grad_mag_enh


















def detect_shadows_with_smooth_transitions(image_rgb, transition_threshold=0.3):
    """
    Detect shadows only when there are smooth transitions to brighter regions.
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a, b = cv2.split(lab)
    
    # Step 1: Find dark regions (potential shadows)
    dark_mask = l_channel < 80
    
    # Step 2: Analyze gradient smoothness around dark regions
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Step 3: Check if dark regions have smooth boundaries
    shadow_mask = np.zeros_like(dark_mask, dtype=bool)
    
    # Dilate dark regions to include their boundaries
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_dark = cv2.dilate(dark_mask.astype(np.uint8), kernel)
    boundaries = dilated_dark & ~dark_mask  # Boundary pixels around dark regions
    
    for y in range(l_channel.shape[0]):
        for x in range(l_channel.shape[1]):
            if dark_mask[y, x]:
                # Check if this dark pixel has smooth transitions to brighter areas
                if is_smooth_transition(l_channel, gradient_magnitude, x, y, boundaries, transition_threshold):
                    shadow_mask[y, x] = True
    
    return shadow_mask

def is_smooth_transition(l_channel, gradient_magnitude, x, y, boundaries, threshold):
    """
    Check if a dark pixel has smooth transitions to brighter regions.
    """
    height, width = l_channel.shape
    window_size = 10
    
    # Get local neighborhood
    x_min = max(0, x - window_size)
    x_max = min(width, x + window_size + 1)
    y_min = max(0, y - window_size)
    y_max = min(height, y + window_size + 1)
    
    local_l = l_channel[y_min:y_max, x_min:x_max]
    local_grad = gradient_magnitude[y_min:y_max, x_min:x_max]
    local_boundaries = boundaries[y_min:y_max, x_min:x_max]
    
    # Check if there are brighter regions nearby with smooth gradients
    bright_pixels = local_l > l_channel[y, x] + 20  # At least 20 levels brighter
    boundary_gradients = local_grad[local_boundaries & bright_pixels]
    
    if len(boundary_gradients) == 0:
        return False  # No bright boundaries nearby
    
    # Calculate smoothness metric (low gradients indicate smooth transitions)
    smoothness = np.mean(boundary_gradients) / 255.0  # Normalize
    
    return smoothness < threshold  # Lower gradient mean = smoother transition



def brightness_aware_shadow_enhancement(image_rgb, shadow_mask, adaptation_radius=50):
    """
    Enhance shadow regions based on nearby bright region brightness.
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    l_enhanced = l.copy().astype(np.float32)
    
    # Find bright regions (non-shadows that are bright)
    bright_regions = (l > 150) & (~shadow_mask)
    
    # For each shadow pixel, find reference brightness from nearby bright regions
    for y in range(l.shape[0]):
        for x in range(l.shape[1]):
            if shadow_mask[y, x]:
                reference_brightness = find_reference_brightness(l, bright_regions, x, y, adaptation_radius)
                
                if reference_brightness > 0:  # Found a reference
                    current_brightness = l[y, x]
                    target_ratio = 0.6  # Target: reach 60% of reference brightness
                    target_brightness = reference_brightness * target_ratio
                    
                    # Smooth enhancement to avoid artifacts
                    enhancement = target_brightness / max(current_brightness, 1)
                    enhancement = min(enhancement, 3.0)  # Cap at 3x enhancement
                    
                    l_enhanced[y, x] = np.clip(current_brightness * enhancement, 0, 255)
    
    l_enhanced = l_enhanced.astype(np.uint8)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced_rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    return enhanced_rgb

def find_reference_brightness(l_channel, bright_regions, x, y, radius):
    """
    Find appropriate reference brightness from nearby bright regions.
    """
    height, width = l_channel.shape
    
    # Search in expanding circles around the shadow pixel
    for r in range(10, radius + 1, 10):
        x_min = max(0, x - r)
        x_max = min(width, x + r + 1)
        y_min = max(0, y - r)
        y_max = min(height, y + r + 1)
        
        # Get bright pixels in this region
        local_bright = bright_regions[y_min:y_max, x_min:x_max]
        local_l = l_channel[y_min:y_max, x_min:x_max]
        
        bright_pixels = local_l[local_bright]
        
        if len(bright_pixels) > 0:
            # Use median to avoid outliers
            return np.median(bright_pixels)
    
    return -1  # No reference found


def advanced_shadow_processing(image_rgb, transition_threshold=0.2, adaptation_radius=50):
    """
    Complete shadow processing with smooth transition detection and brightness adaptation.
    """

    # Step 1: Detect shadows with smooth transition requirement
    shadow_mask = detect_shadows_with_smooth_transitions(image_rgb, transition_threshold)
    
    # Step 2: Apply brightness-aware enhancement
    enhanced_rgb = brightness_aware_shadow_enhancement(image_rgb, shadow_mask, adaptation_radius)

    """# Step 1: Detect shadows with smooth transition requirement
    shadow_mask = fast_detect_shadows_with_smooth_transitions(image_rgb, transition_threshold)
    
    # Step 2: Apply brightness-aware enhancement
    enhanced_rgb = fast_brightness_aware_enhancement(image_rgb, shadow_mask, adaptation_radius)"""
    
    return enhanced_rgb, shadow_mask












def fast_detect_shadows_with_smooth_transitions(image_rgb, transition_threshold=0.3):
    """
    Vectorized version - MUCH faster.
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a, b = cv2.split(lab)
    
    # Step 1: Find dark regions
    dark_mask = l_channel < 80
    
    # Step 2: Compute gradients (vectorized)
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    # Step 3: Use morphological operations to find boundaries (vectorized)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_dark = cv2.dilate(dark_mask.astype(np.uint8), kernel)
    boundaries = dilated_dark & ~dark_mask
    
    # Step 4: Compute local gradient statistics using convolution (vectorized)
    # Average gradient in boundary regions around each pixel
    kernel_avg = np.ones((15, 15)) / 225  # 15x15 averaging kernel
    local_avg_gradient = cv2.filter2D(gradient_magnitude, -1, kernel_avg)
    
    # Step 5: Find bright boundaries nearby (vectorized)
    bright_boundaries = boundaries & (l_channel > (l_channel + 20))  # Vectorized condition
    
    # Step 6: Combined smoothness condition (vectorized)
    smooth_shadows = dark_mask & (local_avg_gradient < transition_threshold * 255)
    
    return smooth_shadows

def fast_brightness_aware_enhancement(image_rgb, shadow_mask, adaptation_radius=50):
    """
    Vectorized brightness-aware enhancement.
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Find bright regions
    bright_regions = (l > 150) & (~shadow_mask)
    
    # Create distance transform to find nearest bright pixel for each shadow pixel
    from scipy import ndimage
    
    # Create binary mask of bright regions
    bright_mask = bright_regions.astype(np.uint8)
    
    # For each shadow pixel, find distance to nearest bright region
    distances, indices = ndimage.distance_transform_edt(~bright_regions, return_indices=True)
    
    # Get reference brightness from nearest bright pixel
    reference_brightness = l[indices[0], indices[1]]
    
    # Only enhance shadow pixels that have a reasonable reference nearby
    valid_reference = distances < adaptation_radius
    shadow_with_reference = shadow_mask & valid_reference
    
    # Apply enhancement (vectorized)
    l_enhanced = l.copy().astype(np.float32)
    target_ratio = 0.6
    
    current_brightness = l[shadow_with_reference]
    target_brightness = reference_brightness[shadow_with_reference] * target_ratio
    enhancement = target_brightness / np.maximum(current_brightness, 1)
    enhancement = np.minimum(enhancement, 3.0)  # Cap enhancement
    
    l_enhanced[shadow_with_reference] = np.clip(current_brightness * enhancement, 0, 255)
    
    l_enhanced = l_enhanced.astype(np.uint8)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced_rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    return enhanced_rgb



















# Visualization and testing
if __name__ == "__main__":
    image_name = 'images/Lenna.webp'
    image = cv2.imread(image_name)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Test different parameters
    transition_thresholds = [0.1, 0.2, 0.3]  # Lower = stricter smoothness requirement
    adaptation_radii = [30, 50, 70]  # Search radius for reference brightness

    transition_thresholds = [0.3]  # Lower = stricter smoothness requirement
    adaptation_radii = [70]  # Search radius for reference brightness
    
    fig, axes = plt.subplots(len(transition_thresholds), 4, figsize=(20, 5 * len(transition_thresholds)))
    
    if len(transition_thresholds) == 1:
        axes = axes.reshape(1, -1)
    
    for i, trans_thresh in enumerate(transition_thresholds):
        for j, adapt_radius in enumerate(adaptation_radii):
            enhanced_rgb, shadow_mask = advanced_shadow_processing(
                image_rgb, trans_thresh, adapt_radius)
            
            # Compute gradients for comparison
            gray_orig = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            gray_enh = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2GRAY)
            
            grad_orig = np.sqrt(cv2.Sobel(gray_orig, cv2.CV_64F, 1, 0, ksize=3)**2 + 
                               cv2.Sobel(gray_orig, cv2.CV_64F, 0, 1, ksize=3)**2)
            grad_enh = np.sqrt(cv2.Sobel(gray_enh, cv2.CV_64F, 1, 0, ksize=3)**2 + 
                              cv2.Sobel(gray_enh, cv2.CV_64F, 0, 1, ksize=3)**2)
            
            # Plot results
            col = j
            if j == 0:
                axes[i, 0].imshow(shadow_mask, cmap='gray')
                axes[i, 0].set_title(f'Shadow Mask\nTransThresh: {trans_thresh}')
                axes[i, 0].axis('off')
            
            axes[i, 1].imshow(enhanced_rgb)
            axes[i, 1].set_title(f'Enhanced\nAdaptRadius: {adapt_radius}')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(grad_orig, cmap='hot')
            axes[i, 2].set_title('Original Gradients')
            axes[i, 2].axis('off')
            
            axes[i, 3].imshow(grad_enh, cmap='hot')
            axes[i, 3].set_title('Enhanced Gradients')
            axes[i, 3].axis('off')
            
            # Statistics
            shadow_grad_orig = grad_orig[shadow_mask]
            shadow_grad_enh = grad_enh[shadow_mask]
            improvement = ((np.mean(shadow_grad_enh) - np.mean(shadow_grad_orig)) / np.mean(shadow_grad_orig) * 100 )
            
            print(f"TransThresh: {trans_thresh}, AdaptRadius: {adapt_radius}")
            print(f"  Shadow pixels: {np.sum(shadow_mask)}, Improvement: {improvement:.1f}%")
            print()
    
    plt.tight_layout()
    plt.show()