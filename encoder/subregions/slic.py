from skimage.color import rgb2gray, rgb2lab
from skimage import filters
import numpy as np
from skimage.segmentation import slic
import matplotlib.pyplot as plt
import math

"""def enhanced_slic_with_texture(image, n_segments, compactness=10, texture_weight=0.3):
    # Calculate texture features
    gray = rgb2gray(image)
    texture_map = np.zeros_like(gray)
    
    # Calculate texture using Gabor filters
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        gabor_real, gabor_imag = filters.gabor(gray, frequency=0.1, theta=theta)
        gabor_mag = np.sqrt(gabor_real**2 + gabor_imag**2)
        texture_map += gabor_mag
    
    texture_map /= 4  # Average across orientations
    
    # Normalize texture map
    if texture_map.max() > 0:
        texture_map = texture_map / texture_map.max()
    
    # Convert image to LAB for better color perception
    lab_image = rgb2lab(image)
    
    # Use standard SLIC but you could extend this to custom distance function
    segments = slic(
        image, 
        n_segments=n_segments,
        compactness=compactness,
        sigma=1,
        channel_axis=2
    )
    
    return segments, texture_map
"""

#def enhanced_slic_with_texture(image, n_segments, compactness=10, texture_weight=0.3):
def enhanced_slic_with_texture(image, n_segments=100, compactness=10, scale_factor=0.3):

    # Calculate texture features
    gray = rgb2gray(image)
    texture_map = np.zeros_like(gray)
    
    # Calculate texture using Gabor filters
    for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
        gabor_real, gabor_imag = filters.gabor(gray, frequency=0.1, theta=theta)
        gabor_mag = np.sqrt(gabor_real**2 + gabor_imag**2)
        texture_map += gabor_mag
    
    texture_map /= 4  # Average across orientations
    
    # Normalize texture map
    if texture_map.max() > 0:
        texture_map = texture_map / texture_map.max()



    from skimage.transform import resize
    
    # Downscale image
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    
    small_image = resize(image, (new_h, new_w), 
                        preserve_range=True, 
                        anti_aliasing=True).astype(np.uint8)
    
    n_segments=math.ceil(n_segments * scale_factor * scale_factor)  # Adjust for area
    # Run SLIC on small image
    segments_small = slic(
        small_image,
        n_segments=n_segments,
        compactness=compactness,
        sigma=1,
        channel_axis=2
    )
    
    # Upscale segmentation
    segments = resize(segments_small, (h, w),
                     order=0,  # Nearest-neighbor for labels
                     preserve_range=True,
                     anti_aliasing=False).astype(np.int32)
    
    return segments,texture_map


from skimage.segmentation import find_boundaries
from skimage.measure import find_contours
import numpy as np

"""
def extract_slic_segment_boundaries(roi_segments, bbox_mask):

    segment_boundaries = []
    
    # Get unique segment IDs (excluding background)
    segment_ids = np.unique(roi_segments)
    segment_ids = segment_ids[segment_ids != 0]  # Remove background
    
    for seg_id in segment_ids:
        # Create mask for this specific segment
        segment_mask = (roi_segments == seg_id) & bbox_mask
        
        if np.sum(segment_mask) > 0:  # If segment exists in our region
            # Find contours/boundaries
            contours = find_contours(segment_mask, level=0.5)
            
            # Take the longest contour (main boundary)
            if len(contours) > 0:
                main_contour = max(contours, key=len)
                
                # Convert to list of (y,x) coordinates
                boundary_coords = [(coord[0], coord[1]) for coord in main_contour]
                
                segment_boundaries.append({
                    'segment_id': int(seg_id),
                    'boundary_coords': boundary_coords,
                    'area': np.sum(segment_mask),
                    'num_points': len(boundary_coords)
                })
    
    return segment_boundaries
"""

def extract_slic_segment_boundaries(roi_segments, bbox_mask):
    """
    Extract boundaries of all SLIC segments within the irregular region
    
    Returns:
    --------
    list of dictionaries, each containing:
        'segment_id': ID of the segment
        'boundary_coords': list of (y,x) coordinates
        'area': number of pixels in segment
    """
    segment_boundaries = []
    
    # Get unique segment IDs (excluding background)
    segment_ids = np.unique(roi_segments)
    segment_ids = segment_ids[segment_ids != 0]  # Remove background
    
    for seg_id in segment_ids:
        # Create mask for this specific segment
        segment_mask = (roi_segments == seg_id) & bbox_mask
        
        # Check if segment has enough pixels
        if np.sum(segment_mask) > 0:  # If segment exists in our region
            # Get mask dimensions
            rows, cols = segment_mask.shape
            
            # Skip if mask is too small for contour detection
            if rows < 2 or cols < 2:
                # Handle tiny segments - either skip or create a point boundary
                # Get the coordinates of the single pixel
                y_coords, x_coords = np.where(segment_mask)
                if len(y_coords) > 0:
                    # Create a small square boundary around the single point
                    y, x = y_coords[0], x_coords[0]
                    boundary_coords = [
                        (y-0.5, x-0.5), (y-0.5, x+0.5),
                        (y+0.5, x+0.5), (y+0.5, x-0.5)
                    ]
                    segment_boundaries.append({
                        'segment_id': int(seg_id),
                        'boundary_coords': boundary_coords,
                        'area': np.sum(segment_mask),
                        'num_points': len(boundary_coords),
                        'note': 'tiny_segment'
                    })
                continue  # Skip to next segment
            
            # Find contours/boundaries
            try:
                contours = find_contours(segment_mask, level=0.5)
                
                # Take the longest contour (main boundary)
                if len(contours) > 0:
                    main_contour = max(contours, key=len)
                    
                    # Convert to list of (y,x) coordinates
                    boundary_coords = [(coord[0], coord[1]) for coord in main_contour]
                    
                    segment_boundaries.append({
                        'segment_id': int(seg_id),
                        'boundary_coords': boundary_coords,
                        'area': np.sum(segment_mask),
                        'num_points': len(boundary_coords),
                        'note': 'normal_segment'
                    })
            except ValueError as e:
                # Log the error and handle tiny segments
                print(f"Warning: Segment {seg_id} too small for contour detection: {e}")
                # Skip this segment or handle as above
                continue
    
    return segment_boundaries


def visualize_split_analysis(region_image, overall_score, color_score, texture_score, optimal_segments):
    """Visualize the split analysis results"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original region
    axes[0, 0].imshow(region_image)
    axes[0, 0].set_title(f'Original Region\nArea: {region_image.shape[0]}x{region_image.shape[1]}')
    axes[0, 0].axis('off')
    
    # Color analysis
    lab_image = rgb2lab(region_image)
    axes[0, 1].imshow(lab_image[:, :, 0], cmap='viridis')
    axes[0, 1].set_title(f'Color Complexity: {color_score:.3f}')
    axes[0, 1].axis('off')
    
    # Texture analysis
    gray = rgb2gray(region_image)
    texture = filters.sobel(gray)
    axes[0, 2].imshow(texture, cmap='hot')
    axes[0, 2].set_title(f'Texture Complexity: {texture_score:.3f}')
    axes[0, 2].axis('off')
    
    # Scores
    scores = [overall_score, color_score, texture_score]
    labels = ['Overall', 'Color', 'Texture']
    colors = ['blue', 'green', 'red']
    axes[1, 0].bar(labels, scores, color=colors)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title('Split Scores')
    
    # Recommended segments
    axes[1, 1].text(0.5, 0.5, f'Optimal Segments:\n{optimal_segments}', 
                   ha='center', va='center', fontsize=20)
    axes[1, 1].set_title('SLIC Recommendation')
    axes[1, 1].axis('off')
    
    # Weights (you can modify these)
    weights = [0.6, 0.4]  # color, texture
    axes[1, 2].pie(weights, labels=['Color', 'Texture'], autopct='%1.1f%%')
    axes[1, 2].set_title('Feature Weights')
    
    plt.tight_layout()
    plt.show()

