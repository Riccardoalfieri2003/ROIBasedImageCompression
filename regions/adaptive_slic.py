import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from skimage.segmentation import slic, mark_boundaries

from division import color_variation_score_v2, edge_density_score, enhanced_texture_analysis, gradient_smoothness_score, texture_variation_score, should_split
from irregular_region import should_split_irregular_region

"""
def recursive_slic_adaptive(image, mask=None, depth_limit=None, depth=0, compactness=1, sigma=1, base_segments=3):

    # Base case for recursion depth
    if depth_limit is not None and depth >= depth_limit:
        return []

    # If no mask is given, use the whole image
    if mask is None:
        mask = np.ones(image.shape[:2], dtype=bool)

    # Extract the masked region for analysis
    ys, xs = np.where(mask)
    if len(ys) == 0:  # Empty mask
        return []
    
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    region_image = image[y_min:y_max, x_min:x_max]
    region_mask = mask[y_min:y_max, x_min:x_max]



    # --- DECISION: Should we split this region? ---
    should_split_result = should_split(region_image)

    # --- VISUALIZATION ---
    if True:
        
        # Create the visualization
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Irregular region with border
        plt.subplot(1, 3, 1)
        display_image = np.zeros_like(region_image)
        display_image[region_mask] = region_image[region_mask]
        
        # Find and draw contour
        contours, _ = cv2.findContours(region_mask.astype(np.uint8), 
                                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(display_image, contours, -1, (255, 0, 0), 2)
        
        plt.imshow(display_image)
        plt.title(f'Region at Depth {depth}')
        plt.axis('off')
        
        # Plot 2: Original region image for reference
        plt.subplot(1, 3, 2)
        plt.imshow(region_image)
        plt.title('Cropped Region Image')
        plt.axis('off')
        
        # Plot 3: Decision information (FIXED)
        plt.subplot(1, 3, 3)
        
        # Clear the axis and set limits
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')  # Turn off the axis
        
        decision_color = 'red' if should_split_result else 'green'
        decision_text = "SPLIT" if should_split_result else "KEEP"
        
        # Add text with proper positioning
        plt.text(0.5, 0.9, f"DECISION: {decision_text}", fontsize=16, 
                color=decision_color, ha='center', va='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=decision_color, alpha=0.3))
        
        
        plt.title('Analysis Results')
        
        plt.tight_layout()
        plt.show()
    
    if not should_split_result:
        # Region is uniform enough - stop recursion here
        return [{
            "mask": region_mask,
            "bbox": (x_min, y_min, x_max, y_max),
            "depth": depth,
            "terminal": True  # Mark as terminal node
        }]

    # --- ADAPTIVE SEGMENTATION STRATEGY ---
    all_segments = []
    
    # Strategy 1: Try with 2 segments first (since we know it's worth dividing)
    n_segments = 3
    sub_segments = slic(
        image,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        start_label=1,
        mask=mask
    )
    
    unique_labels = np.unique(sub_segments[mask])
    
    print(f"Depth {depth}: Splitting into {len(unique_labels)} regions")

    # Process each sub-region
    for label in unique_labels:
        sub_mask = (sub_segments == label)

        # Skip empty or invalid masks
        if np.count_nonzero(sub_mask) < 10:
            continue

        # Extract the sub-region
        ys_sub, xs_sub = np.where(sub_mask)
        y_min_sub, y_max_sub = ys_sub.min(), ys_sub.max()
        x_min_sub, x_max_sub = xs_sub.min(), xs_sub.max()

        cropped_image = image[y_min_sub:y_max_sub, x_min_sub:x_max_sub]
        cropped_mask = sub_mask[y_min_sub:y_max_sub, x_min_sub:x_max_sub]

        # Store this region info
        all_segments.append({
            "mask": cropped_mask,
            "bbox": (x_min_sub, y_min_sub, x_max_sub, y_max_sub),
            "depth": depth,
            "terminal": False  # We'll determine this in recursion
        })

        # Recurse further with adaptive strategy
        child_segments = recursive_slic_adaptive(
            image,
            mask=sub_mask,
            depth_limit=depth_limit,
            depth=depth + 1,
            compactness=compactness,
            sigma=sigma,
            base_segments=base_segments
        )
        all_segments.extend(child_segments)

    return all_segments
"""














from image_cleaning import comprehensive_region_cleaning, remove_small_components


def recursive_slic_adaptive(image, mask=None, depth_limit=None, depth=0, compactness=1, sigma=1, base_segments=3):
    """
    Recursively apply SLIC segmentation, adapting n_segments based on should_split_improved.
    FIXED: Handles first iteration properly where mask covers entire region.
    """
    # Base case for recursion depth
    if depth_limit is not None and depth >= depth_limit:
        return []

    # If no mask is given, use the whole image
    if mask is None:
        mask = np.ones(image.shape[:2], dtype=bool)

    # Extract the masked region for analysis
    ys, xs = np.where(mask)
    if len(ys) == 0:  # Empty mask
        return []
    
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    bbox_region = image[y_min:y_max, x_min:x_max]
    bbox_mask = mask[y_min:y_max, x_min:x_max]

    # --- DECISION: Should we split this region? ---

    # SPECIAL HANDLING FOR FIRST ITERATION: Check if mask covers entire bbox
    if depth == 0 or np.all(bbox_mask):
        # For full masks, use the regular analysis on the entire region
        should_split_result = should_split(bbox_region)
        print(f"Depth {depth}: Analyzing full rectangular region")



    else:

        """AGGIUNGERE: Se l'immagibne è piccola, mantinei"""

        if np.sum(bbox_mask) > 100:  # Only clean if region is substantial
            cleaned_region, cleaned_mask = comprehensive_region_cleaning(
                bbox_region, bbox_mask, 
                min_area_ratio=0.02,  # More aggressive cleaning
                border_cleaning=True,  # Enable border cleaning
                show_cleaning=(depth <= 2)  # Show cleaning process only for first few levels
            )
            
            # Use cleaned region for display AND analysis
            clean_display = create_clean_region_display(cleaned_region, cleaned_mask)
            
            # Update the variables for analysis
            bbox_region = cleaned_region
            bbox_mask = cleaned_mask

            should_split_result = should_split_irregular_region(bbox_region, bbox_mask)
            
        else:

            """
            # For very small regions, use basic cleaning without border analysis
            basic_cleaned_mask = remove_small_components(bbox_mask, min_area_ratio=0.05)
            basic_cleaned_region = np.zeros_like(bbox_region)
            basic_cleaned_region[basic_cleaned_mask] = bbox_region[basic_cleaned_mask]
            
            clean_display = create_clean_region_display(basic_cleaned_region, basic_cleaned_mask)

            cleaned_region, cleaned_mask = comprehensive_region_cleaning(
                bbox_region, bbox_mask, 
                min_area_ratio=0.02,  # More aggressive cleaning
                border_cleaning=True,  # Enable border cleaning
                show_cleaning=(depth <= 2)  # Show cleaning process only for first few levels
            )

            should_split_result = should_split_irregular_region(cleaned_region, cleaned_mask)
            """

            should_split_result=False



        # For irregular masks, use the irregular region analysis
        #clean_display = create_clean_region_display(bbox_region, bbox_mask) 
        #should_split_result = should_split_irregular_region(bbox_region, bbox_mask)
        #should_split_result = should_split(clean_display)
        
        print(f"Depth {depth}: Analyzing irregular region with {np.sum(bbox_mask)} pixels")

    # --- VISUALIZATION ---
    if True:
        # Create the visualization
        plt.figure(figsize=(15, 5))




        
        # Plot 1: Show the region context
        plt.subplot(1, 3, 1)
        if depth == 0 or np.all(bbox_mask):
            # Full region - just show the image
            plt.imshow(bbox_region)
            plt.title(f'Full Region at Depth {depth}')
        else:
            # Irregular region - show with border
            display_image = np.zeros_like(bbox_region)
            display_image[bbox_mask] = bbox_region[bbox_mask]
            
            # Find and draw contour
            contours, _ = cv2.findContours(bbox_mask.astype(np.uint8), 
                                          cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display_image, contours, -1, (255, 0, 0), 2)
            
            plt.imshow(display_image)
            plt.title(f'Irregular Region at Depth {depth}')
        plt.axis('off')



        

        # Plot 2: Show what we're actually analyzing
        """
        plt.subplot(1, 3, 2)
        if depth == 0 or np.all(bbox_mask):
            # Analyzing the full rectangle
            plt.imshow(bbox_region)
            plt.title('Analyzing: Full Region')
        else:
            # Analyzing only the irregular part
            clean_display = create_clean_region_display(bbox_region, bbox_mask)
            plt.imshow(clean_display)
            plt.title(f'Analyzing: Irregular Region\n({np.sum(bbox_mask)} pixels)')
        plt.axis('off')
        """

        # In your recursive function, update the visualization part:
        if depth == 0 or np.all(bbox_mask):
            # Analyzing the full rectangle
            plt.imshow(bbox_region)
            plt.title('Analyzing: Full Region')
        else:
            # Clean the mask before analysis and display
            if np.sum(bbox_mask) > 100:  # Only clean if region is substantial
                
                """cleaned_region, cleaned_mask = comprehensive_region_cleaning(
                    bbox_region, bbox_mask, 
                    min_area_ratio=0.03,  # Keep components with at least 3% of total area
                    show_cleaning=(depth <= 2)  # Show cleaning process only for first few levels
                )"""
                
                # Use cleaned region for display AND analysis
                #clean_display = create_clean_region_display(cleaned_region, cleaned_mask)
                plt.imshow(clean_display)
                plt.title(f'Cleaned Irregular Region\n({np.sum(cleaned_mask)} pixels)')
                
                # Update the variables for analysis
                bbox_region = cleaned_region
                bbox_mask = cleaned_mask
            else:
                # For very small regions, just use basic cleaning
                clean_display = create_clean_region_display(bbox_region, bbox_mask)
                plt.imshow(clean_display)
                plt.title(f'Irregular Region\n({np.sum(bbox_mask)} pixels)')
        




        # Plot 3: Decision information
        plt.subplot(1, 3, 3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        
        decision_color = 'red' if should_split_result else 'green'
        decision_text = "SPLIT" if should_split_result else "KEEP"
        
        plt.text(0.5, 0.9, f"DECISION: {decision_text}", fontsize=16, 
                color=decision_color, ha='center', va='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor=decision_color, alpha=0.3))
        
        plt.title('Analysis Results')
        plt.tight_layout()
        plt.show()
    
    if not should_split_result:
        # Region is uniform enough - stop recursion here
        return [{
            "mask": mask,
            "bbox": (x_min, y_min, x_max, y_max),
            "depth": depth,
            "terminal": True
        }]

    # --- ADAPTIVE SEGMENTATION STRATEGY ---
    all_segments = []
    
    # Apply SLIC only within the irregular mask
    n_segments = 3
    sub_segments = slic(
        image,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        start_label=1,
        mask=mask
    )
    
    unique_labels = np.unique(sub_segments[mask])
    
    print(f"Depth {depth}: Splitting into {len(unique_labels)} regions")

    # Process each sub-region
    for label in unique_labels:
        sub_mask = (sub_segments == label)

        # Skip empty or invalid masks
        if np.count_nonzero(sub_mask) < 10:
            continue

        # Recurse with the new irregular mask
        child_segments = recursive_slic_adaptive(
            image,
            mask=sub_mask,
            depth_limit=depth_limit,
            depth=depth + 1,
            compactness=compactness,
            sigma=sigma,
            base_segments=base_segments
        )
        all_segments.extend(child_segments)

    return all_segments




def create_clean_region_display(bbox_region, bbox_mask):
    """Create a clean display with transparent background instead of black"""
    # Create RGBA image with transparency
    clean_image = np.zeros((bbox_region.shape[0], bbox_region.shape[1], 4), dtype=np.uint8)
    
    # Copy RGB data where mask is True
    clean_image[bbox_mask, :3] = bbox_region[bbox_mask]
    
    # Set alpha channel: 255 where mask is True, 0 where False
    clean_image[bbox_mask, 3] = 255
    
    return clean_image


def get_adaptive_parameters(region_image):
    """Adjust SLIC parameters based on region properties"""
    color_score = color_variation_score_v2(region_image)
    texture_score = texture_variation_score(region_image)
    
    # More compact for textured regions, less for smooth color regions
    if texture_score > 0.4:
        compactness = 20  # Higher compactness for textures
    elif color_score > 0.6:
        compactness = 5   # Lower compactness for color regions
    else:
        compactness = 10  # Default
    
    # Adjust n_segments based on region size and complexity
    region_size = region_image.shape[0] * region_image.shape[1]
    base_segments = max(2, min(5, int(np.sqrt(region_size) / 50)))
    
    return compactness, base_segments




def visualize_segments_on_image(base_image, all_segments):
    """
    Draw red contours for every segment found at all recursion levels.
    FIXED: Handles dimension mismatches between masks and target regions.
    """
    contour_image = base_image.copy()
    base_height, base_width = base_image.shape[:2]

    for i, seg in enumerate(all_segments):
        try:
            mask = seg["mask"]
            x_min, y_min, x_max, y_max = seg["bbox"]
            
            # Calculate expected region dimensions
            region_height = y_max - y_min
            region_width = x_max - x_min
            
            # Get actual mask dimensions
            mask_height, mask_width = mask.shape[:2]
            
            # Handle dimension mismatches by taking the minimum
            copy_height = min(region_height, mask_height, base_height - y_min)
            copy_width = min(region_width, mask_width, base_width - x_min)
            
            # Skip if dimensions are invalid
            if copy_height <= 0 or copy_width <= 0:
                print(f"Warning: Skipping segment {i} with invalid dimensions")
                continue
            
            # Create full mask and copy only the valid portion
            full_mask = np.zeros(base_image.shape[:2], dtype=np.uint8)
            
            # Copy the compatible portion of the mask
            full_mask[y_min:y_min+copy_height, x_min:x_min+copy_width] = \
                mask[:copy_height, :copy_width].astype(np.uint8) * 255

            # Find and draw contours
            contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 1)
            
        except Exception as e:
            print(f"Error processing segment {i}: {e}")
            print(f"BBox: ({x_min}, {y_min}, {x_max}, {y_max})")
            print(f"Mask shape: {mask.shape}")
            print(f"Base image shape: {base_image.shape}")
            continue

    return contour_image


def plot_irregular_region(region_image, region_mask):
    """Plot only the irregular region border, not the rectangle"""
    plt.figure(figsize=(6, 6))
    
    # Create a black background
    display_image = np.zeros_like(region_image)
    
    # Find the contour of the mask
    contours, _ = cv2.findContours(region_mask.astype(np.uint8), 
                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw the region image only inside the mask
    display_image[region_mask] = region_image[region_mask]
    
    # Draw the contour in red
    cv2.drawContours(display_image, contours, -1, (255, 0, 0), 2)
    
    plt.imshow(display_image)
    plt.title(f'Irregular Region - Size: {np.sum(region_mask)} pixels')
    plt.axis('off')
    plt.show()
    
    return display_image





















"""
def should_split_irregular_region(bbox_region, region_mask):

    # Create masked version for analysis
    masked_region = np.zeros_like(bbox_region)
    masked_region[region_mask] = bbox_region[region_mask]
    
    # Extract only the pixels that are actually in the region
    region_pixels = bbox_region[region_mask]
    
    # For color analysis - we need to work with the actual pixels
    if len(region_pixels) > 0:
        # Convert to appropriate format for analysis
        region_for_analysis = masked_region
        
        # Now use your existing analysis functions
        color_score, entropy_norm, peaks_norm = color_variation_score_v2(region_for_analysis, show_hist=True)
        normalized_texture, fine_texture, pattern_regularity = enhanced_texture_analysis(region_for_analysis)
        smoothness_score = gradient_smoothness_score(region_for_analysis, show_grad=False)
        edge_score, density, strength = edge_density_score(region_for_analysis, show_edges=False)
        
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
        


    else:
        return False
"""






















# --- Example usage ---
if __name__ == "__main__":
    image = cv2.imread("images/Lenna.webp")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    """lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)

    lab_clahe = cv2.merge((l_clahe, a, b))
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)"""


    all_segments = recursive_slic_adaptive(
        image_rgb,
        depth_limit=6,  # optional
        base_segments=3,
        compactness=1,
        sigma=1
    )

    print(f"Numero di regioni: {len(all_segments)}")

    result = visualize_segments_on_image(image_rgb, all_segments)

    plt.figure(figsize=(12, 8))
    plt.imshow(result)
    plt.title("Recursive SLIC with Clean Region Isolation")
    plt.axis("off")
    plt.show()

