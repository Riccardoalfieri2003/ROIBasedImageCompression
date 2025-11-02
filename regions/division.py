import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops



import matplotlib.pyplot as plt

def color_variation_score(image_rgb, debug=True, show_hist=True):
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # --- Compute histograms ---
    hist_l = cv2.calcHist([l], [0], None, [256], [0, 256]).flatten()
    hist_a = cv2.calcHist([a], [0], None, [256], [0, 256]).flatten()
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256]).flatten()

    # --- Plot histograms if requested ---
    if show_hist:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.plot(hist_l, color='black')
        plt.title('L Channel Histogram (Lightness)')
        plt.xlabel('Intensity'); plt.ylabel('Frequency')

        plt.subplot(1, 3, 2)
        plt.plot(hist_a, color='red')
        plt.title('A Channel Histogram (Green–Red)')
        plt.xlabel('Value'); plt.ylabel('Frequency')

        plt.subplot(1, 3, 3)
        plt.plot(hist_b, color='blue')
        plt.title('B Channel Histogram (Blue–Yellow)')
        plt.xlabel('Value'); plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()
    
    # --- Normalize histograms ---
    hist_l /= hist_l.sum() + 1e-8
    hist_a /= hist_a.sum() + 1e-8
    hist_b /= hist_b.sum() + 1e-8
    
    

    # --- Compute weighted std ---
    x = np.arange(256)
    def weighted_std(hist, x, label=""):
        mean = np.sum(hist * x)
        std = np.sqrt(np.sum(hist * (x - mean) ** 2))
        if debug:
            print(f"[DEBUG] {label} mean={mean:.2f}, std={std:.2f}")
        return std

    std_l = weighted_std(hist_l, x, "L")
    std_a = weighted_std(hist_a, x, "A")
    std_b = weighted_std(hist_b, x, "B")

    color_score = np.mean([std_l, std_a, std_b])
    if debug:
        print(f"[DEBUG] Final color_score = {color_score:.2f}")

    return color_score


def texture_variation_score(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # Reduce resolution slightly for speed
    gray_small = cv2.resize(gray, (128, 128))
    
    # Compute co-occurrence matrix (GLCM)
    glcm = graycomatrix(gray_small, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast').mean()
    energy = graycoprops(glcm, 'energy').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    
    # Instead of the current combination, consider:
    texture_variability = contrast * (1 - homogeneity) * (1 - energy)
    
    # Normalize to 0-1 (empirical bounds from testing)
    normalized_texture = min(texture_variability / 0.3, 1.0)  # Adjust divisor based on your image set
    
    return normalized_texture












from scipy.signal import find_peaks
from scipy.stats import entropy



def color_variation_score_v2(image_rgb, show_hist=True, include_lightness=False):
    # Convert to LAB color space
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Compute normalized histograms
    hists = [cv2.calcHist([ch], [0], None, [256], [0, 256]).flatten() for ch in [l, a, b]]
    hists = [h / (h.sum() + 1e-8) for h in hists]

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
        fig = plt.figure(figsize=(12, 8))

        # Top: image
        ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=3)
        ax1.imshow(image_rgb)
        ax1.set_title("Original Image")
        ax1.axis("off")

        # Bottom: histograms for each LAB channel
        colors = ['k', 'r', 'b']
        titles = ['L (Lightness)', 'A (Green–Red)', 'B (Blue–Yellow)']
        for i, (h, c, title) in enumerate(zip(hists, colors, titles)):
            ax = plt.subplot2grid((2, 3), (1, i))
            ax.plot(h, color=c)
            ax.set_title(title + (" ← used" if channel_names[i if include_lightness else i-1] in title else ""))
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")

        plt.tight_layout()
        plt.show()
    

    # Add normalization at the end:
    # Normalize entropy to 0-1 scale (assuming max practical entropy ~4)
    normalized_entropy = min(color_entropy / 4.0, 1.0)
    # Normalize peaks (assuming max practical peaks ~8)
    normalized_peaks = min(num_peaks / 8.0, 1.0)
    
    # Combined color variation score (0-1)
    color_score = 0.7 * normalized_entropy + 0.3 * normalized_peaks
    
    #print(f"Color score: {color_score:.3f} (entropy: {normalized_entropy:.3f}, peaks: {normalized_peaks:.3f})")
    return color_score, normalized_entropy, normalized_peaks














import cv2
import numpy as np

def gradient_smoothness_score(image_rgb, show_grad=False):
    """
    Compute a smoothness score based on image gradients.
    Low score = smooth gradient (likely not multiple regions)
    High score = strong/sudden changes (likely multiple regions)
    """
    # Convert to grayscale for simplicity
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    
    # Compute gradients
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)
    
    # Compute smoothness metric
    mean_grad = np.mean(grad_mag)
    std_grad = np.std(grad_mag)
    smoothness_score = std_grad / (mean_grad + 1e-8)  # coefficient of variation

    
    # Rename for clarity and invert the scale
    gradient_roughness = std_grad / (mean_grad + 1e-8)
    
    # Normalize roughness to 0-1 and invert to get smoothness
    normalized_roughness = min(gradient_roughness / 5.0, 1.0)  # Empirical scaling
    smoothness_score = 1.0 - normalized_roughness  # Now high = smooth, low = rough
    
    """
    if show_grad:
        plt.title(f"Gradient Map (smoothness={smoothness_score:.3f}, roughness={gradient_roughness:.3f})")
        plt.figure(figsize=(8, 6))
        plt.imshow(grad_mag, cmap='gray')
        plt.axis('off')
        plt.show()
    """
    
    return smoothness_score  # Now 0-1, where 1 = perfectly smooth gradient



import cv2
import numpy as np
import matplotlib.pyplot as plt

def edge_density_score(image_rgb, low_thresh=50, high_thresh=150, show_edges=False):
    """
    Measure both edge density and edge intensity using Canny + gradient magnitude.
    
    Returns:
        density (float): fraction of edge pixels
        mean_strength (float): average gradient magnitude along edges
    """
    # Convert to LAB space (better color distance perception)
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Detect edges (binary mask)
    edges = cv2.Canny(gray, low_thresh, high_thresh)

    # Compute gradients in color space (use only lightness or full color magnitude)
    gx = cv2.Sobel(lab, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(lab, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(np.sum(gx**2 + gy**2, axis=2))

    # Compute edge density and average strength
    density = np.sum(edges > 0) / edges.size
    if np.sum(edges > 0) > 0:
        mean_strength = np.mean(grad_mag[edges > 0])
    else:
        mean_strength = 0.0

    

    # Optional visualization
    if show_edges:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(edges, cmap='gray')
        plt.title(f"Edges Detected (density={density:.4f})")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(grad_mag, cmap='inferno')
        plt.title(f"Gradient Magnitude (mean={mean_strength:.2f})")
        plt.axis('off')

        plt.tight_layout()
        plt.show()
    
    # Normalize strength to 0-1 (assuming max practical ~100)
    normalized_strength = min(mean_strength / 100.0, 1.0)
    
    # Combine density and strength into single score
    edge_score = 0.6 * density + 0.4 * normalized_strength
    
    return edge_score, density, normalized_strength





def enhanced_texture_analysis(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray_small = cv2.resize(gray, (128, 128))
    
    # GLCM at multiple distances
    glcm1 = graycomatrix(gray_small, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 256, symmetric=True, normed=True)
    glcm3 = graycomatrix(gray_small, [3], [0, np.pi/4, np.pi/2, 3*np.pi/4], 256, symmetric=True, normed=True)
    
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
    
    normalized_texture = min(texture_strength / 0.4, 1.0)
    
    return normalized_texture, fine_texture, pattern_regularity

















def should_split(image_rgb):
    color_score, entropy_norm, peaks_norm = color_variation_score_v2(image_rgb, show_hist=True)
    normalized_texture, fine_texture, pattern_regularity = enhanced_texture_analysis(image_rgb)
    smoothness_score = gradient_smoothness_score(image_rgb, show_grad=True)  # Note: now high = smooth
    edge_score, density, strength = edge_density_score(image_rgb, show_edges=False)
    
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
    if edge_score > 0.15:
        split_score += 3.0 * edge_importance_multiplier
        reasons.append("Strong edge presence - likely object boundary")
    elif edge_score > 0.08:
        split_score += 2.0 * edge_importance_multiplier
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
    if normalized_texture > 0.6:
        split_score += 2.5
        reasons.append("Strong texture pattern")
    elif normalized_texture > 0.3:
        split_score += 1.5
        reasons.append("Moderate texture")
    
    # Fine textures often need more subdivision
    if fine_texture > 1.5 and normalized_texture > 0.4:
        split_score += 1.0
        reasons.append("Fine detailed texture")
    
    # Very regular patterns might not need splitting
    if pattern_regularity > 0.8 and normalized_texture < 0.3:
        split_score -= 1.0
        reasons.append("Very regular pattern - less need to split")
    
    # 4. GRADIENT ANALYSIS with edge consideration
    if smoothness_score > 0.9:
        # Very smooth gradient - but check if there are edges
        if edge_score < 0.05:  # No significant edges
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
    if edge_score > 0.1 and color_score > 0.4:
        split_score += 2.0
        reasons.append("Edges with color variation - strong boundary signal")
    
    # Case B: Edges + Texture = likely textured object boundary
    if edge_score > 0.1 and normalized_texture > 0.3:
        split_score += 1.5
        reasons.append("Edges with texture - textured object boundary")
    
    # Case C: Strong edges in smooth color regions = object boundary
    if edge_score > 0.15 and color_score < 0.3 and smoothness_score > 0.7:
        split_score += 2.5
        reasons.append("Clear object boundary in uniform region")
    
    # Case D: The "perfect storm" - edges, color, and texture
    if edge_score > 0.1 and color_score > 0.4 and normalized_texture > 0.4:
        split_score += 3.0
        reasons.append("Complex region with edges, color, and texture")
    
    # 6. EDGE-DRIVEN CONTRADICTION RESOLUTION
    
    # Strong edges override smoothness concerns
    if smoothness_score > 0.8 and edge_score > 0.15:
        # Remove the smoothness penalty and add bonus
        split_score += 2.0  # Override and bonus
        reasons.append("Overriding: strong edges define important boundaries")
    
    # Strong edges override texture regularity
    if pattern_regularity > 0.7 and edge_score > 0.12:
        split_score += 1.0
        reasons.append("Overriding: edges in regular pattern indicate boundaries")
    
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
    if color_score < 0.2 and normalized_texture < 0.1 and smoothness_score > 0.9 and edge_score < 0.05:
        base_threshold = 6.0
        print("High threshold: completely uniform region")
    
    decision = split_score >= base_threshold
    print(f"Decision: {'SPLIT' if decision else 'KEEP'} (threshold: {base_threshold})")
    
    return decision



def main():
    image_name='images/nero.png'
    image = cv2.imread(image_name)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image_rgb = cv2.GaussianBlur(image, (51, 51), 50)


    plt.title(f"Picture")
    plt.imshow(image_rgb, cmap='gray')
    plt.axis('off')
    plt.show()

    print(image_name)
    split=should_split(image_rgb)

    print(f"Split: {split}")