import math
import cv2
from matplotlib import pyplot as plt
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.segmentation import slic, mark_boundaries
from scipy.stats import entropy
from scipy.signal import find_peaks




def create_clean_region_display(bbox_region, bbox_mask):
    """Create a clean display with transparent background instead of black"""
    # Create RGBA image with transparency
    clean_image = np.zeros((bbox_region.shape[0], bbox_region.shape[1], 4), dtype=np.uint8)
    
    # Copy RGB data where mask is True
    clean_image[bbox_mask, :3] = bbox_region[bbox_mask]
    
    # Set alpha channel: 255 where mask is True, 0 where False
    clean_image[bbox_mask, 3] = 255
    
    return clean_image

def create_compact_region(bbox_region, clean_mask):
    """
    Create a compact rectangular image containing only the irregular region.
    This eliminates the black background problem for analysis functions.
    """
    ys, xs = np.where(clean_mask)
    
    if len(ys) == 0:
        return None
    
    # Find the tight bounding box around the region
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    
    # Extract the tight region
    tight_region = bbox_region[y_min:y_max+1, x_min:x_max+1]
    tight_mask = clean_mask[y_min:y_max+1, x_min:x_max+1]
    
    # Create output image with only the region pixels
    compact_region = np.zeros_like(tight_region)
    compact_region[tight_mask] = tight_region[tight_mask]
    
    return compact_region

def should_split_irregular_region(bbox_region, clean_mask, debug=False):
    """
    Analyze an irregular region without black background contamination.
    
    Args:
        bbox_region: The rectangular region containing the irregular shape
        clean_mask: Boolean mask indicating the actual irregular region
    
    Returns:
        Same as should_split but for irregular regions
    """


    BLUR_KERNEL_SIZE = (5, 5) 
    
    # Create a copy to avoid modifying the original image outside this function
    blurred_bbox_region = bbox_region.copy() 
    
    # Apply Gaussian Blur
    blurred_bbox_region = cv2.GaussianBlur(blurred_bbox_region, BLUR_KERNEL_SIZE, 0)
    # Note: The blur is applied to all three color channels simultaneously.
    # ------------------------------------

    # Extract only the pixels in the actual region (using original clean_mask)
    region_pixels = blurred_bbox_region[clean_mask]




    # Extract only the pixels in the actual region
    region_pixels = bbox_region[clean_mask]
    
    if len(region_pixels) == 0:
        return False, 0, 0, 0, 0, 0, 0, 0, 0
    
    # Create a compact representation of just the region
    compact_region = create_compact_region(bbox_region, clean_mask)
    
    # Use your existing analysis functions on the compact region
    """color_score, entropy_norm, peaks_norm = color_variation_score_irregular(bbox_region, clean_mask, show_hist=debug)
    normalized_texture, fine_texture, pattern_regularity = enhanced_texture_analysis_irregular(bbox_region, clean_mask, show_analysis=debug)
    smoothness_score = gradient_smoothness_score_irregular(bbox_region, clean_mask, show_grad=debug)
    edge_score, density, strength = edge_density_score_irregular(bbox_region, clean_mask, show_edges=debug)"""

    color_score, entropy_norm, peaks_norm = color_variation_score_irregular(bbox_region, clean_mask, show_hist=debug)
    normalized_texture, fine_texture, pattern_regularity = enhanced_texture_analysis_irregular(blurred_bbox_region, clean_mask, show_analysis=debug)
    smoothness_score = gradient_smoothness_score_irregular(blurred_bbox_region, clean_mask, show_grad=debug)
    edge_score, density, strength = edge_density_score_irregular(blurred_bbox_region, clean_mask, show_edges=debug)
    
    print(f"\n=== REGION ANALYSIS ===")
    print(f"Color score: {color_score:.3f} (entropy: {entropy_norm:.3f}, peaks: {peaks_norm:.3f})")
    print(f"normalized_texture variation: {normalized_texture:.3f} ( fine_texture: {fine_texture:.3f}, pattern_regularity: {pattern_regularity:.3f} )") 
    print(f"Gradient smoothness: {smoothness_score:.3f}")
    print(f"Edge presence: {edge_score:.3f} (density: {density:.3f}, strength: {strength:.3f})")
    
    # Decision matrix with enhanced edge importance
    split_score = 0
    reasons = []
    
    # 1. ENHANCED EDGE ANALYSIS (Increased Importance)
    edge_importance_multiplier = 1.5  # Boost edge significance
    
    # Strong, well-defined edges (clear boundaries)
    if edge_score > 0.35:
        split_score += 0.5 * edge_importance_multiplier
        reasons.append("Strong edge presence - likely object boundary")
    elif edge_score > 0.15:
        split_score += 0.5  * edge_importance_multiplier
        reasons.append("Moderate edge presence")
    
    # Smooth, continuous edges (very important for realistic shapes)
    if strength > 0.5 and density < 0.15:  # Strong but not too dense = smooth edges
        split_score += 2.5 * edge_importance_multiplier
        reasons.append("Smooth continuous edges - important for shape preservation")
    
    # Edge clusters that might indicate complex boundaries
    if density > 0.1 and strength > 0.4:
        split_score += 2.0 * edge_importance_multiplier
        reasons.append("Edge clusters suggesting complex boundary")
    
    # 2. COLOR-BASED DECISIONS (slightly reduced relative importance)
    if color_score > 0.7:
        split_score += 2.5  # Reduced from 3.0
        reasons.append("Very high color variation")
    elif color_score > 0.5:
        split_score += 1.5  # Reduced from 2.0
        reasons.append("High color variation")
    elif color_score > 0.3:
        split_score += 0.8  # Reduced from 1.0
        reasons.append("Moderate color variation")
    
    # Multiple color peaks indicate distinct color regions
    if peaks_norm > 0.6:
        split_score += 1.2  # Reduced from 1.5
        reasons.append("Multiple distinct color modes")
    
    # 3. TEXTURE-BASED DECISIONS
    if normalized_texture < 0.6:
        split_score += 2.5
        reasons.append("Strong texture pattern")
    elif normalized_texture < 0.3:
        split_score += 1.5
        reasons.append("Moderate texture")
    
    # Fine textures often need more subdivision
    if fine_texture < 1.5 and normalized_texture > 0.4:
        split_score += 1.0
        reasons.append("Fine detailed texture")
    
    # Very regular patterns might not need splitting
    if pattern_regularity > 0.8 and normalized_texture > 0.3:
        split_score -= 1.0
        reasons.append("Very regular pattern - less need to split")
    
    # 4. GRADIENT ANALYSIS with edge consideration
    if smoothness_score > 0.9:
        # Very smooth gradient - but check if there are edges
        if edge_score < 0.25:  # No significant edges
            split_score -= 2.5  # Strong penalty for splitting pure gradients
            reasons.append("Pure smooth gradient - preserve continuity")
        else:
            split_score -= 1.0  # Smaller penalty if edges are present
            reasons.append("Mostly smooth but has edges")
    elif smoothness_score < 0.3:
        # Rough, non-smooth area - encourage splitting
        split_score += 1.5
        reasons.append("Rough/non-smooth area")
    
    # 5. ENHANCED COMBINATION CASES (with edge focus)
    
    # Case A: Edges + Color variation = very strong split signal
    if edge_score > 0.35 and color_score > 0.4:
        split_score += 2.0
        reasons.append("Edges with color variation - strong boundary signal")
    
    # Case B: Edges + Texture = likely textured object boundary
    if edge_score > 0.35 and normalized_texture < 0.3:
        split_score += 1.5
        reasons.append("Edges with texture - textured object boundary")
    
    # Case C: Strong edges in smooth color regions = object boundary
    if edge_score > 0.5 and color_score < 0.3 and smoothness_score < 0.7:
        split_score += 2.5
        reasons.append("Clear object boundary in uniform region")
    
    # Case D: The "perfect storm" - edges, color, and texture
    if edge_score > 0.35 and color_score > 0.4 and normalized_texture < 0.4:
        split_score += 3.0
        reasons.append("Complex region with edges, color, and texture")
    
    # 6. EDGE-DRIVEN CONTRADICTION RESOLUTION
    
    # Strong edges override smoothness concerns
    if smoothness_score < 0.7 and edge_score > 0.15:
        # Remove the smoothness penalty and add bonus
        split_score += 2.0  # Override and bonus
        reasons.append("Overriding: strong edges define important boundaries")
    
    # Strong edges override texture regularity
    """
    if pattern_regularity < 0.7 and edge_score > 0.12:
        split_score += 1.0
        reasons.append("Overriding: edges in regular pattern indicate boundaries")
    """
    
    # FINAL DECISION with edge-adaptive threshold
    print(f"Split score: {split_score:.2f}")
    print("Reasons:", ", ".join(reasons))
    
    # Adaptive threshold that considers edge presence
    base_threshold = 4.0
    
    # Lower threshold for edge-rich regions (we want to preserve boundaries)
    if edge_score > 0.1:
        if edge_score > 0.2:
            base_threshold = 2.5  # Very low threshold for strong edges
            print("Very low threshold: strong edge presence")
        else:
            base_threshold = 3.0  # Low threshold for moderate edges
            print("Low threshold: edge presence")
    
    # Even lower for the "perfect boundary" case
    if edge_score > 0.15 and color_score > 0.3 and smoothness_score > 0.6:
        base_threshold = 2.0
        print("Minimum threshold: ideal boundary conditions")
    
    # Higher threshold for very uniform regions without edges
    if color_score < 0.2 and normalized_texture > 0.1 and smoothness_score > 0.9 and edge_score < 0.05:
        base_threshold = 6.0
        print("High threshold: completely uniform region")
    
    decision = split_score >= base_threshold
    print(f"Decision: {'SPLIT' if decision else 'KEEP'} (threshold: {base_threshold})")
    
    return decision, math.floor(split_score)
            



    

def color_variation_score_irregular(bbox_region, clean_mask, show_hist=True, include_lightness=False):
    """
    Color variation analysis for irregular regions - excludes border pixels from analysis.
    
    Args:
        bbox_region: The rectangular region containing the irregular shape
        clean_mask: Boolean mask indicating the actual irregular region
        show_hist: Whether to show histograms
        include_lightness: Whether to include L channel in analysis
    
    Returns:
        color_score, normalized_entropy, normalized_peaks
    """
    # Create a compact representation of just the region
    compact_region = create_compact_region(bbox_region, clean_mask)
    
    if compact_region is None:
        return 0.0, 0.0, 0.0
    
    # Create mask for the compact region
    compact_mask = (cv2.cvtColor(compact_region, cv2.COLOR_RGB2GRAY) > 0).astype(np.uint8)
    
    # Create INTERIOR mask by eroding the compact mask (exclude border)
    kernel = np.ones((5, 5), np.uint8)
    interior_mask = cv2.erode(compact_mask, kernel, iterations=2)
    
    # Skip if interior is too small after erosion
    interior_size = np.sum(interior_mask)
    if interior_size < 20:
        # Fall back to less aggressive erosion
        interior_mask = cv2.erode(compact_mask, np.ones((3, 3), np.uint8), iterations=1)
        interior_size = np.sum(interior_mask)
        if interior_size < 10:
            # If still too small, use the whole region but with warning
            interior_mask = compact_mask
            interior_size = np.sum(compact_mask)
            print("Warning: Using entire region (interior too small)")
    
    # Extract ONLY the pixels that are in the INTERIOR region (exclude border)
    interior_pixels = compact_region[interior_mask > 0]
    
    if len(interior_pixels) == 0:
        return 0.0, 0.0, 0.0
    
    # Convert the interior pixels to LAB color space
    interior_pixels_2d = interior_pixels.reshape(-1, 1, 3)
    lab_pixels = cv2.cvtColor(interior_pixels_2d, cv2.COLOR_RGB2LAB)
    lab_pixels = lab_pixels.reshape(-1, 3)
    
    # Split into channels
    l, a, b = lab_pixels[:, 0], lab_pixels[:, 1], lab_pixels[:, 2]
    
    # Compute histograms only for the INTERIOR region pixels
    hists = []
    for channel in [l, a, b]:
        hist = np.histogram(channel, bins=256, range=(0, 255))[0]
        hist = hist / (hist.sum() + 1e-8)  # Normalize
        hists.append(hist)
    
    # Select which channels to analyze
    if include_lightness:
        channels_to_use = hists  # [L, A, B]
        channel_names = ['L', 'A', 'B']
    else:
        channels_to_use = hists[1:]  # [A, B] only
        channel_names = ['A', 'B']

    # Compute entropy & number of peaks for selected channels
    color_entropy = np.mean([entropy(h) for h in channels_to_use])
    num_peaks = np.mean([len(find_peaks(h, height=0.005)[0]) for h in channels_to_use])

    # --- Visualization ---
    if show_hist:
        fig = plt.figure(figsize=(15, 10))

        # Top left: original rectangular region with mask overlay
        ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2)
        display_with_mask = bbox_region.copy()
        # Create a red overlay for the mask
        overlay = np.zeros_like(bbox_region)
        overlay[clean_mask] = [255, 0, 0]
        ax1.imshow(display_with_mask)
        ax1.imshow(overlay, alpha=0.3)
        ax1.set_title("Region with Mask Overlay")
        ax1.axis("off")

        # Top right: clean region only (no black background)
        ax2 = plt.subplot2grid((3, 3), (0, 2))
        clean_display = create_clean_region_display(bbox_region, clean_mask)
        ax2.imshow(clean_display)
        ax2.set_title("Clean Region\n(analysis area)")
        ax2.axis("off")

        # Bottom: histograms for each LAB channel
        colors = ['k', 'r', 'b']
        titles = ['L (Lightness)', 'A (Green–Red)', 'B (Blue–Yellow)']
        for i, (h, c, title) in enumerate(zip(hists, colors, titles)):
            ax = plt.subplot2grid((3, 3), (1 + i//2, i%2))
            ax.plot(h, color=c)
            ax.set_title(f'{title}\n' + ("← used" if (channel_names[i] if include_lightness else f'AB[{i-1}]') in title else ""))
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            # Add some stats to the histogram
            ax.text(0.05, 0.95, f'Entropy: {entropy(h):.3f}', transform=ax.transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        # Add summary info
        ax_summary = plt.subplot2grid((3, 3), (2, 2))
        ax_summary.axis('off')
        ax_summary.text(0.1, 0.9, f"Region Analysis Summary", fontsize=12, weight='bold')
        #ax_summary.text(0.1, 0.7, f"Pixels analyzed: {len(region_pixels)}")
        ax_summary.text(0.1, 0.5, f"Color entropy: {color_entropy:.3f}")
        ax_summary.text(0.1, 0.3, f"Number of peaks: {num_peaks:.1f}")
        ax_summary.text(0.1, 0.1, f"Channels used: {', '.join(channel_names)}")

        plt.tight_layout()
        plt.show()

    # Normalize scores
    normalized_entropy = min(color_entropy / 4.0, 1.0)
    normalized_peaks = min(num_peaks / 8.0, 1.0)
    
    # Combined color variation score (0-1)
    color_score = 0.7 * normalized_entropy + 0.3 * normalized_peaks
    
    print(f"Irregular Region - Color score: {color_score:.3f} (entropy: {normalized_entropy:.3f}, peaks: {normalized_peaks:.3f})")
    return color_score, normalized_entropy, normalized_peaks

    """
    Simplified version for irregular regions - faster computation.
    """
    # Extract ONLY the pixels that are in the actual region
    region_pixels = bbox_region[clean_mask]
    
    if len(region_pixels) == 0:
        return 0.0, 0.0, 0.0
    
    # Convert to LAB and analyze
    region_pixels_2d = region_pixels.reshape(-1, 1, 3)
    lab_pixels = cv2.cvtColor(region_pixels_2d, cv2.COLOR_RGB2LAB)
    lab_pixels = lab_pixels.reshape(-1, 3)
    
    l, a, b = lab_pixels[:, 0], lab_pixels[:, 1], lab_pixels[:, 2]
    
    # Compute histograms
    channels = [l, a, b] if include_lightness else [a, b]
    hists = [np.histogram(ch, bins=256, range=(0, 255))[0] for ch in channels]
    hists = [h / (h.sum() + 1e-8) for h in hists]
    
    # Compute metrics
    color_entropy = np.mean([entropy(h) for h in hists])
    num_peaks = np.mean([len(find_peaks(h, height=0.005)[0]) for h in hists])
    
    # Normalize and return
    normalized_entropy = min(color_entropy / 4.0, 1.0)
    normalized_peaks = min(num_peaks / 8.0, 1.0)
    color_score = 0.7 * normalized_entropy + 0.3 * normalized_peaks
    
    return color_score, normalized_entropy, normalized_peaks

def enhanced_texture_analysis_irregular(bbox_region, clean_mask, show_analysis=False):
    """
    Texture analysis for irregular regions - excludes border pixels from analysis.
    
    Args:
        bbox_region: The rectangular region containing the irregular shape
        clean_mask: Boolean mask indicating the actual irregular region
        show_analysis: Whether to show analysis visualization
    
    Returns:
        normalized_texture, fine_texture, pattern_regularity
    """
    # Create a compact representation of just the region
    compact_region = create_compact_region(bbox_region, clean_mask)
    
    if compact_region is None:
        return 0.0, 0.0, 0.0
    
    # Create mask for the compact region
    compact_mask = (cv2.cvtColor(compact_region, cv2.COLOR_RGB2GRAY) > 0).astype(np.uint8)
    
    # Create INTERIOR mask by eroding the compact mask (exclude border)
    kernel = np.ones((5, 5), np.uint8)
    interior_mask = cv2.erode(compact_mask, kernel, iterations=2)
    
    # Skip if interior is too small after erosion
    interior_size = np.sum(interior_mask)
    if interior_size < 50:  # Need more pixels for texture analysis
        # Fall back to less aggressive erosion
        interior_mask = cv2.erode(compact_mask, np.ones((3, 3), np.uint8), iterations=1)
        interior_size = np.sum(interior_mask)
        if interior_size < 25:
            # If still too small, use the whole region but with warning
            interior_mask = compact_mask
            interior_size = np.sum(compact_mask)
            print("Warning: Using entire region for texture (interior too small)")
    
    # Create interior-only region for texture analysis
    interior_region = np.zeros_like(compact_region)
    interior_region[interior_mask > 0] = compact_region[interior_mask > 0]
    
    # Convert to grayscale for texture analysis
    gray = cv2.cvtColor(interior_region, cv2.COLOR_RGB2GRAY)
    
    # Skip if the interior region is mostly empty
    if np.sum(gray > 0) < 25:
        return 0.0, 0.0, 0.0
    
    # Resize for consistent analysis (maintain aspect ratio to preserve texture)
    original_height, original_width = gray.shape
    target_size = 128
    
    # Calculate resize dimensions maintaining aspect ratio
    if original_height > original_width:
        new_height = target_size
        new_width = int(original_width * target_size / original_height)
    else:
        new_width = target_size
        new_height = int(original_height * target_size / original_width)
    
    # Ensure minimum dimensions
    new_width = max(new_width, 32)
    new_height = max(new_height, 32)
    
    # Resize the interior region for analysis
    gray_small = cv2.resize(gray, (new_width, new_height))
    
    # Create a mask for the resized interior region
    resized_interior_mask = (gray_small > 0).astype(np.uint8)
    
    # Skip if resized interior is too small
    if np.sum(resized_interior_mask) < 20:
        return 0.0, 0.0, 0.0
    
    # GLCM at multiple distances (adjust distances based on image size)
    distances = [1, min(3, new_width//10)]  # Adaptive distances
    
    try:
        # Only compute GLCM on the interior region
        glcm1 = graycomatrix(gray_small, [distances[0]], [0, np.pi/4, np.pi/2, 3*np.pi/4], 
                           256, symmetric=True, normed=True)
        glcm3 = graycomatrix(gray_small, [distances[1]], [0, np.pi/4, np.pi/2, 3*np.pi/4], 
                           256, symmetric=True, normed=True)
        
        # Multi-scale contrast
        contrast1 = graycoprops(glcm1, 'contrast').mean()
        contrast3 = graycoprops(glcm3, 'contrast').mean()
        
        energy = graycoprops(glcm1, 'energy').mean()
        homogeneity = graycoprops(glcm1, 'homogeneity').mean()
        
        # Detect different texture types:
        fine_texture = contrast1 / (contrast3 + 1e-8)  # Fine vs coarse texture
        pattern_regularity = energy  # High = regular pattern
        
        # Combined score that emphasizes texture presence
        texture_strength = (contrast1 + (1 - homogeneity)) * (1 - pattern_regularity)
        
        # Adaptive normalization based on interior region size
        normalization_factor = 0.4 * (min(original_height, original_width) / 100.0)
        normalization_factor = max(normalization_factor, 0.2)  # Minimum threshold
        
        normalized_texture = min(texture_strength / normalization_factor, 1.0)
        
    except Exception as e:
        print(f"Texture analysis error: {e}")
        return 0.0, 0.0, 0.0
    
    # Visualization
    if show_analysis:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original region with mask
        axes[0, 0].imshow(bbox_region)
        mask_overlay = np.zeros((*bbox_region.shape[:2], 4))
        mask_overlay[clean_mask] = [1, 0, 0, 0.3]  # Red overlay
        axes[0, 0].imshow(mask_overlay)
        axes[0, 0].set_title('Original Region with Mask')
        axes[0, 0].axis('off')
        
        # Clean region
        clean_display = create_clean_region_display(bbox_region, clean_mask)
        axes[0, 1].imshow(clean_display)
        axes[0, 1].set_title('Clean Region for Analysis')
        axes[0, 1].axis('off')
        
        # Compact region used for analysis
        axes[0, 2].imshow(compact_region)
        axes[0, 2].set_title(f'Compact Region\n{compact_region.shape[:2]}')
        axes[0, 2].axis('off')
        
        # Grayscale for texture analysis
        axes[1, 0].imshow(gray_small, cmap='gray')
        axes[1, 0].set_title(f'Resized for Analysis\n{gray_small.shape[:2]}')
        axes[1, 0].axis('off')
        
        # Texture metrics
        axes[1, 1].axis('off')
        axes[1, 1].text(0.1, 0.9, 'Texture Analysis Results', fontsize=12, weight='bold')
        axes[1, 1].text(0.1, 0.7, f'Contrast (d=1): {contrast1:.3f}')
        axes[1, 1].text(0.1, 0.6, f'Contrast (d=3): {contrast3:.3f}')
        axes[1, 1].text(0.1, 0.5, f'Energy: {energy:.3f}')
        axes[1, 1].text(0.1, 0.4, f'Homogeneity: {homogeneity:.3f}')
        axes[1, 1].text(0.1, 0.3, f'Fine Texture Ratio: {fine_texture:.3f}')
        axes[1, 1].text(0.1, 0.2, f'Pattern Regularity: {pattern_regularity:.3f}')
        axes[1, 1].text(0.1, 0.1, f'Texture Score: {normalized_texture:.3f}')
        
        # Texture strength visualization
        axes[1, 2].bar(['Contrast1', '1-Homogeneity', '1-Energy', 'Total'], 
                      [contrast1, 1-homogeneity, 1-energy, texture_strength])
        axes[1, 2].set_title('Texture Strength Components')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    print(f"Irregular Region - Texture score: {normalized_texture:.3f} "
          f"(fine: {fine_texture:.3f}, regular: {pattern_regularity:.3f})")
    
    return normalized_texture, fine_texture, pattern_regularity




def gradient_smoothness_score_irregular(bbox_region, clean_mask, show_grad=False):
    """
    Compute smoothness score for irregular regions - only analyzes pixels within the mask.
    
    Args:
        bbox_region: The rectangular region containing the irregular shape
        clean_mask: Boolean mask indicating the actual irregular region
        show_grad: Whether to show gradient visualization
    
    Returns:
        smoothness_score: 0-1 where 1 = perfectly smooth gradient
    """
    # Extract ONLY the pixels that are in the actual region
    region_pixels = bbox_region[clean_mask]
    
    if len(region_pixels) < 10:  # Need minimum pixels for gradient analysis
        return 1.0  # Treat very small regions as smooth
    
    # Create a compact representation of just the region
    compact_region = create_compact_region(bbox_region, clean_mask)
    
    if compact_region is None:
        return 1.0
    
    # Convert to grayscale for gradient analysis
    gray = cv2.cvtColor(compact_region, cv2.COLOR_RGB2GRAY)
    
    # Compute gradients ONLY within the compact region
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)
    
    # Create a mask for the compact region (where we have actual pixels)
    compact_mask = (cv2.cvtColor(compact_region, cv2.COLOR_RGB2GRAY) > 0).astype(np.uint8)
    
    # Only consider gradients within the actual region (ignore black background)
    valid_grad_mask = compact_mask > 0
    valid_gradients = grad_mag[valid_grad_mask]
    
    if len(valid_gradients) == 0:
        return 1.0  # No valid gradients = smooth
    
    # Compute smoothness metric only on valid gradients
    mean_grad = np.mean(valid_gradients)
    std_grad = np.std(valid_gradients)
    
    # Handle edge case where mean gradient is very small
    if mean_grad < 1e-6:
        gradient_roughness = 0.0  # Essentially no variation = very smooth
    else:
        gradient_roughness = std_grad / mean_grad  # coefficient of variation
    
    # Normalize roughness to 0-1 and invert to get smoothness
    # Use adaptive normalization based on region characteristics
    normalization_factor = 3.0  # More conservative for irregular regions
    normalized_roughness = min(gradient_roughness / normalization_factor, 1.0)
    smoothness_score = 1.0 - normalized_roughness  # Now high = smooth, low = rough
    
    # Visualization
    if show_grad:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original region with mask
        axes[0, 0].imshow(bbox_region)
        mask_overlay = np.zeros((*bbox_region.shape[:2], 4))
        mask_overlay[clean_mask] = [1, 0, 0, 0.3]  # Red overlay
        axes[0, 0].imshow(mask_overlay)
        axes[0, 0].set_title('Original Region with Mask')
        axes[0, 0].axis('off')
        
        # Clean region
        clean_display = create_clean_region_display(bbox_region, clean_mask)
        axes[0, 1].imshow(clean_display)
        axes[0, 1].set_title('Clean Region for Analysis')
        axes[0, 1].axis('off')
        
        # Compact region used for analysis
        axes[0, 2].imshow(compact_region)
        axes[0, 2].set_title(f'Compact Region\n{compact_region.shape[:2]}')
        axes[0, 2].axis('off')
        
        # Gradient magnitude within valid region
        grad_display = np.zeros_like(grad_mag)
        grad_display[valid_grad_mask] = grad_mag[valid_grad_mask]
        im1 = axes[1, 0].imshow(grad_display, cmap='viridis')
        axes[1, 0].set_title(f'Gradient Magnitude\n(within region)')
        axes[1, 0].axis('off')
        plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # Gradient statistics
        axes[1, 1].axis('off')
        axes[1, 1].text(0.1, 0.9, 'Gradient Analysis Results', fontsize=12, weight='bold')
        axes[1, 1].text(0.1, 0.7, f'Mean Gradient: {mean_grad:.3f}')
        axes[1, 1].text(0.1, 0.6, f'Std Gradient: {std_grad:.3f}')
        axes[1, 1].text(0.1, 0.5, f'Gradient Roughness: {gradient_roughness:.3f}')
        axes[1, 1].text(0.1, 0.4, f'Normalized Roughness: {normalized_roughness:.3f}')
        axes[1, 1].text(0.1, 0.3, f'Smoothness Score: {smoothness_score:.3f}')
        axes[1, 1].text(0.1, 0.2, f'Valid Pixels: {len(valid_gradients)}')
        
        # Gradient distribution
        axes[1, 2].hist(valid_gradients, bins=50, alpha=0.7, color='blue')
        axes[1, 2].axvline(mean_grad, color='red', linestyle='--', label=f'Mean: {mean_grad:.2f}')
        axes[1, 2].axvline(mean_grad + std_grad, color='orange', linestyle='--', 
                          label=f'Mean+Std: {mean_grad+std_grad:.2f}')
        axes[1, 2].set_xlabel('Gradient Magnitude')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_title('Gradient Distribution')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    print(f"Irregular Region - Smoothness: {smoothness_score:.3f} "
          f"(roughness: {gradient_roughness:.3f}, mean grad: {mean_grad:.3f})")
    
    return smoothness_score

"""
def edge_density_score_irregular(bbox_region, clean_mask, low_thresh=50, high_thresh=150, show_edges=False):

    # Extract ONLY the pixels that are in the actual region
    region_pixels = bbox_region[clean_mask]
    
    if len(region_pixels) < 10:  # Need minimum pixels for edge detection
        return 0.0, 0.0, 0.0
    
    # Create a compact representation of just the region
    compact_region = create_compact_region(bbox_region, clean_mask)
    
    if compact_region is None:
        return 0.0, 0.0, 0.0
    
    # Convert to appropriate color spaces
    lab = cv2.cvtColor(compact_region, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(compact_region, cv2.COLOR_RGB2GRAY)
    
    # Create mask for the compact region (where we have actual pixels)
    compact_mask = (cv2.cvtColor(compact_region, cv2.COLOR_RGB2GRAY) > 0).astype(np.uint8)
    
    # Detect edges in the compact region
    edges = cv2.Canny(gray, low_thresh, high_thresh)
    
    # Remove edges that are on the boundary with black background
    # Dilate the mask to exclude edges near the boundary
    kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(compact_mask, kernel, iterations=2)
    
    # Only keep edges that are well inside the actual region
    valid_edges = edges & dilated_mask
    
    # Compute gradients
    gx = cv2.Sobel(lab, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lab, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)
    
    # Compute metrics ONLY within the actual region
    region_size = np.sum(compact_mask)
    if region_size > 0:
        # Density: fraction of edge pixels relative to region size (not image size)
        density = np.sum(valid_edges > 0) / region_size
        
        # Mean strength: only consider gradients at valid edge locations
        edge_locations = valid_edges > 0
        if np.sum(edge_locations) > 0:
            mean_strength = np.mean(grad_mag[edge_locations])
        else:
            mean_strength = 0.0
    else:
        density = 0.0
        mean_strength = 0.0
    
    # Visualization
    if show_edges:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original region with mask
        axes[0, 0].imshow(bbox_region)
        mask_overlay = np.zeros((*bbox_region.shape[:2], 4))
        mask_overlay[clean_mask] = [1, 0, 0, 0.3]  # Red overlay
        axes[0, 0].imshow(mask_overlay)
        axes[0, 0].set_title('Original Region with Mask')
        axes[0, 0].axis('off')
        
        # Clean region
        clean_display = create_clean_region_display(bbox_region, clean_mask)
        axes[0, 1].imshow(clean_display)
        axes[0, 1].set_title('Clean Region for Analysis')
        axes[0, 1].axis('off')
        
        # Compact region used for analysis
        axes[0, 2].imshow(compact_region)
        axes[0, 2].set_title(f'Compact Region\n{compact_region.shape[:2]}')
        axes[0, 2].axis('off')
        
        # Valid edges within region
        edge_display = np.zeros_like(compact_region)
        edge_display[valid_edges > 0] = [255, 0, 0]  # Red edges
        combined_display = compact_region.copy()
        combined_display[valid_edges > 0] = [255, 0, 0]  # Overlay edges in red
        axes[1, 0].imshow(combined_display)
        axes[1, 0].set_title(f'Edges Inside Region\n(density={density:.4f})')
        axes[1, 0].axis('off')
        
        # Edge statistics
        axes[1, 1].axis('off')
        axes[1, 1].text(0.1, 0.9, 'Edge Analysis Results', fontsize=12, weight='bold')
        axes[1, 1].text(0.1, 0.7, f'Region Size: {region_size} pixels')
        axes[1, 1].text(0.1, 0.6, f'Edge Pixels: {np.sum(valid_edges > 0)}')
        axes[1, 1].text(0.1, 0.5, f'Edge Density: {density:.4f}')
        axes[1, 1].text(0.1, 0.4, f'Mean Strength: {mean_strength:.2f}')
        axes[1, 1].text(0.1, 0.3, f'Valid/Total Edges: {np.sum(valid_edges > 0)}/{np.sum(edges > 0)}')
        
        # Gradient magnitude within region
        grad_display = np.zeros_like(grad_mag)
        grad_display[compact_mask > 0] = grad_mag[compact_mask > 0]
        im = axes[1, 2].imshow(grad_display, cmap='inferno')
        axes[1, 2].set_title(f'Gradient Magnitude\n(mean={mean_strength:.2f})')
        axes[1, 2].axis('off')
        plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.show()
    
    # Normalize and combine scores
    normalized_strength = min(mean_strength / 100.0, 1.0)
    
    # Scale density since it's now relative to region size (typically smaller values)
    scaled_density = min(density * 5.0, 1.0)  # Scale by 5x to match full-image density range
    
    #edge_score = 0.6 * scaled_density + 0.4 * normalized_strength
    edge_score = 0.95 * scaled_density + 0.05 * normalized_strength
    
    print(f"Irregular Region - Edge score: {edge_score:.3f} "
          f"(density: {density:.4f}, strength: {normalized_strength:.3f})")
    
    return edge_score, density, normalized_strength
"""


def edge_density_score_irregular(bbox_region, clean_mask, low_thresh=50, high_thresh=150, show_edges=False):
    """
    Edge density analysis for irregular regions - excludes region border from calculations.
    
    Args:
        bbox_region: The rectangular region containing the irregular shape
        clean_mask: Boolean mask indicating the actual irregular region
        low_thresh: Canny low threshold
        high_thresh: Canny high threshold
        show_edges: Whether to show edge visualization
    
    Returns:
        edge_score, density, normalized_strength
    """
    # Extract ONLY the pixels that are in the actual region
    region_pixels = bbox_region[clean_mask]
    
    if len(region_pixels) < 10:  # Need minimum pixels for edge detection
        return 0.0, 0.0, 0.0
    
    # Create a compact representation of just the region
    compact_region = create_compact_region(bbox_region, clean_mask)
    
    if compact_region is None:
        return 0.0, 0.0, 0.0
    
    # Convert to appropriate color spaces
    lab = cv2.cvtColor(compact_region, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(compact_region, cv2.COLOR_RGB2GRAY)
    
    # Create mask for the compact region (where we have actual pixels)
    compact_mask = (cv2.cvtColor(compact_region, cv2.COLOR_RGB2GRAY) > 0).astype(np.uint8)
    
    # Create INTERIOR mask by eroding the compact mask (exclude border)
    kernel = np.ones((5, 5), np.uint8)  # Larger kernel for more aggressive border removal
    interior_mask = cv2.erode(compact_mask, kernel, iterations=2)
    
    # Skip if interior is too small after erosion
    interior_size = np.sum(interior_mask)
    if interior_size < 20:  # Minimum interior size
        # Fall back to less aggressive erosion
        interior_mask = cv2.erode(compact_mask, np.ones((3, 3), np.uint8), iterations=1)
        interior_size = np.sum(interior_mask)
        if interior_size < 10:
            return 0.0, 0.0, 0.0
    
    # Detect edges in the compact region
    edges = cv2.Canny(gray, low_thresh, high_thresh)
    
    # Only keep edges that are in the INTERIOR (not near any border)
    valid_edges = edges & interior_mask
    
    # Compute gradients
    gx = cv2.Sobel(lab, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lab, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)
    
    # Compute metrics ONLY within the INTERIOR region
    if interior_size > 0:
        # Density: fraction of edge pixels relative to interior size
        density = np.sum(valid_edges > 0) / interior_size
        
        # Mean strength: only consider gradients at valid edge locations in interior
        edge_locations = valid_edges > 0
        if np.sum(edge_locations) > 0:
            mean_strength = np.mean(grad_mag[edge_locations])
        else:
            mean_strength = 0.0
    else:
        density = 0.0
        mean_strength = 0.0
    
    # Visualization
    if show_edges:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original region with mask
        axes[0, 0].imshow(bbox_region)
        mask_overlay = np.zeros((*bbox_region.shape[:2], 4))
        mask_overlay[clean_mask] = [1, 0, 0, 0.3]  # Red overlay
        axes[0, 0].imshow(mask_overlay)
        axes[0, 0].set_title('Original Region with Mask')
        axes[0, 0].axis('off')
        
        # Clean region
        clean_display = create_clean_region_display(bbox_region, clean_mask)
        axes[0, 1].imshow(clean_display)
        axes[0, 1].set_title('Clean Region for Analysis')
        axes[0, 1].axis('off')
        
        # Compact region with interior mask overlay
        compact_display = compact_region.copy()
        # Highlight interior in green
        interior_overlay = np.zeros((*compact_region.shape[:2], 4))
        interior_overlay[interior_mask > 0] = [0, 1, 0, 0.3]  # Green interior
        axes[0, 2].imshow(compact_display)
        axes[0, 2].imshow(interior_overlay)
        axes[0, 2].set_title(f'Interior Region\n{interior_size} pixels')
        axes[0, 2].axis('off')
        
        # Valid edges within INTERIOR only
        edge_display = compact_region.copy()
        edge_display[valid_edges > 0] = [255, 0, 0]  # Red edges
        # Also show the border that we excluded
        border_mask = compact_mask & ~interior_mask
        edge_display[border_mask > 0] = [255, 255, 0]  # Yellow border (excluded)
        axes[1, 0].imshow(edge_display)
        axes[1, 0].set_title(f'Edges in Interior Only\n(density={density:.4f})')
        axes[1, 0].axis('off')
        
        # Edge statistics
        axes[1, 1].axis('off')
        axes[1, 1].text(0.1, 0.9, 'Edge Analysis Results', fontsize=12, weight='bold')
        axes[1, 1].text(0.1, 0.7, f'Total Region: {np.sum(compact_mask)} pixels')
        axes[1, 1].text(0.1, 0.6, f'Interior Region: {interior_size} pixels')
        axes[1, 1].text(0.1, 0.5, f'Border Excluded: {np.sum(border_mask)} pixels')
        axes[1, 1].text(0.1, 0.4, f'Edge Pixels: {np.sum(valid_edges > 0)}')
        axes[1, 1].text(0.1, 0.3, f'Edge Density: {density:.4f}')
        axes[1, 1].text(0.1, 0.2, f'Mean Strength: {mean_strength:.2f}')
        
        # Gradient magnitude within interior
        grad_display = np.zeros_like(grad_mag)
        grad_display[interior_mask > 0] = grad_mag[interior_mask > 0]
        im = axes[1, 2].imshow(grad_display, cmap='inferno')
        axes[1, 2].set_title(f'Gradient in Interior\n(mean={mean_strength:.2f})')
        axes[1, 2].axis('off')
        plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.show()
    
    # Normalize and combine scores
    normalized_strength = min(mean_strength / 100.0, 1.0)
    
    # Scale density since it's now relative to interior size
    scaled_density = min(density * 8.0, 1.0)  # Increased scaling for interior-only analysis
    
    edge_score = 0.95 * scaled_density + 0.05 * normalized_strength
    
    print(f"Irregular Region - Edge score: {edge_score:.3f} "
          f"(density: {density:.4f}, strength: {normalized_strength:.3f}, "
          f"interior: {interior_size}/{np.sum(compact_mask)} pixels)")
    
    return edge_score, density, normalized_strength
















































































   