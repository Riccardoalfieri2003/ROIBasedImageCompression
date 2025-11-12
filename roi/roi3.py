import cv2
from matplotlib import pyplot as plt
import numpy as np
import cv2
import numpy as np
from scipy import ndimage
from scipy import fftpack
from skimage.segmentation import slic


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





def classify_roi_regions(image, regions_list, edge_roi_map):
    """
    Classify each region as ROI or nonROI based on multiple criteria
    """
    roi_regions = []
    nonroi_regions = []
    
    for i, region in enumerate(regions_list):
        region_mask = region['mask']
        bbox = region['bbox']
        
        # Extract region image
        x_min, y_min, x_max, y_max = bbox
        region_image = image[y_min:y_max, x_min:x_max]
        
        # Compute ROI score for this region
        roi_score = compute_region_roi_score(region_image, region_mask, edge_roi_map, bbox)
        
        # Classify based on score
        if roi_score > 0.6:  # Threshold can be adjusted
            roi_regions.append({
                'region_data': region,
                'roi_score': roi_score,
                'type': 'ROI'
            })
        else:
            nonroi_regions.append({
                'region_data': region, 
                'roi_score': roi_score,
                'type': 'nonROI'
            })
    
    return roi_regions, nonroi_regions

def compute_region_roi_score(region_image, region_mask, edge_roi_map, bbox):
    """
    Compute how likely this region is to be ROI (0-1 score)
    """
    scores = []
    
    # 1. Edge density within region
    edge_density_score = compute_region_edge_density(region_mask, edge_roi_map, bbox)
    scores.append(edge_density_score)
    
    # 2. Color complexity
    color_complexity_score = compute_color_complexity(region_image, region_mask)
    scores.append(color_complexity_score)
    
    # 3. Texture richness
    texture_score = compute_texture_richness(region_image, region_mask)
    scores.append(texture_score)
    
    # 4. Size appropriateness
    size_score = compute_size_score(region_mask, region_image.shape)
    scores.append(size_score)
    
    # 5. Boundary strength
    boundary_score = compute_boundary_strength(region_mask, edge_roi_map, bbox)
    scores.append(boundary_score)
    
    return np.mean(scores)






























def compute_color_complexity(region_image, region_mask):
    """How complex is the color distribution in this region?"""
    # Ensure shapes match
    region_mask = ensure_shapes_match(region_mask, region_image.shape[:2])
    
    # Extract pixels only from this region
    region_pixels = region_image[region_mask]
    
    if len(region_pixels) == 0:
        return 0.0
    
    # Convert to LAB for perceptual color difference
    region_pixels_2d = region_pixels.reshape(-1, 1, 3)
    lab_pixels = cv2.cvtColor(region_pixels_2d, cv2.COLOR_RGB2LAB)
    lab_pixels = lab_pixels.reshape(-1, 3)
    
    # Compute color variance
    color_variance = np.var(lab_pixels, axis=0)
    avg_variance = np.mean(color_variance)
    
    # Normalize (empirical values - adjust based on your images)
    return min(avg_variance / 1000, 1.0)

def compute_texture_richness(region_image, region_mask):
    """How rich is the texture in this region?"""
    # Ensure shapes match
    region_mask = ensure_shapes_match(region_mask, region_image.shape[:2])
    
    gray = cv2.cvtColor(region_image, cv2.COLOR_RGB2GRAY)
    
    # Compute local variance as texture measure
    texture_variance = compute_local_variance(gray, kernel_size=5)
    
    # Ensure texture_variance matches region_mask shape
    texture_variance = ensure_shapes_match(texture_variance, region_mask.shape)
    
    # Average texture variance in the region
    region_texture = np.mean(texture_variance[region_mask])
    
    # Normalize
    return min(region_texture / 1000, 1.0)

def ensure_shapes_match(smaller_array, target_shape):
    """
    Ensure two arrays have the same shape by cropping the larger one
    """
    if smaller_array.shape == target_shape:
        return smaller_array
    
    # Crop to the minimum dimensions
    min_height = min(smaller_array.shape[0], target_shape[0])
    min_width = min(smaller_array.shape[1], target_shape[1])
    
    return smaller_array[:min_height, :min_width]

def compute_region_edge_density(region_mask, edge_roi_map, bbox):
    """How many strong edges are inside this region?"""
    # Extract the corresponding portion of the edge map
    x_min, y_min, x_max, y_max = bbox
    edge_roi_portion = edge_roi_map[y_min:y_max, x_min:x_max]
    
    # Ensure both masks have the same shape
    region_mask = ensure_shapes_match(region_mask, edge_roi_portion.shape)
    edge_roi_portion = ensure_shapes_match(edge_roi_portion, region_mask.shape)
    
    # Count edge pixels within this region
    edge_pixels_in_region = np.sum(edge_roi_portion & region_mask)
    total_pixels_in_region = np.sum(region_mask)
    
    if total_pixels_in_region == 0:
        return 0.0
    
    edge_density = edge_pixels_in_region / total_pixels_in_region
    return min(edge_density * 5, 1.0)

def compute_boundary_strength(region_mask, edge_roi_map, bbox):
    """How strong are the boundaries around this region?"""
    # Extract the corresponding portion of the edge map
    x_min, y_min, x_max, y_max = bbox
    edge_roi_portion = edge_roi_map[y_min:y_max, x_min:x_max]
    
    # Ensure both masks have the same shape
    region_mask = ensure_shapes_match(region_mask, edge_roi_portion.shape)
    edge_roi_portion = ensure_shapes_match(edge_roi_portion, region_mask.shape)
    
    # Find region boundary
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(region_mask.astype(np.uint8), kernel)
    boundary = region_mask.astype(np.uint8) - eroded
    
    # Count strong edges on the boundary
    strong_edges_on_boundary = np.sum(edge_roi_portion & boundary.astype(bool))
    total_boundary_pixels = np.sum(boundary)
    
    if total_boundary_pixels == 0:
        return 0.0
    
    return min(strong_edges_on_boundary / total_boundary_pixels * 3, 1.0)






















def compute_local_variance(image, kernel_size=15):
    """Compute local variance to find texture homogeneity"""
    mean = cv2.blur(image, (kernel_size, kernel_size))
    mean_sq = cv2.blur(image.astype(np.float32)**2, (kernel_size, kernel_size))
    variance = mean_sq - mean.astype(np.float32)**2
    return variance


def compute_size_score(region_mask, image_shape):
    """Is this region a reasonable size?"""
    region_pixels = np.sum(region_mask)
    total_pixels = image_shape[0] * image_shape[1]
    size_ratio = region_pixels / total_pixels
    
    # Ideal size range: 1% to 30% of image
    if 0.01 <= size_ratio <= 0.3:
        return 1.0
    elif 0.005 <= size_ratio <= 0.5:
        return 0.7
    elif 0.001 <= size_ratio <= 0.7:
        return 0.3
    else:
        return 0.0



def visualize_roi_classification(image, roi_regions, nonroi_regions):
    """
    Create a clear visualization of ROI vs nonROI regions with robust shape handling
    """
    # Create overlay images
    roi_overlay = image.copy()
    nonroi_overlay = image.copy()
    combined_overlay = image.copy()
    
    def safe_mask_insert(full_mask, mask, bbox):
        """Safely insert a mask into a full-size mask accounting for shape mismatches"""
        x_min, y_min, x_max, y_max = bbox
        
        # Calculate expected shape from bbox
        expected_height = y_max - y_min
        expected_width = x_max - x_min
        
        # Ensure the mask fits in the target area
        actual_height, actual_width = mask.shape
        
        # Use the minimum dimensions to avoid shape mismatches
        copy_height = min(expected_height, actual_height, full_mask.shape[0] - y_min)
        copy_width = min(expected_width, actual_width, full_mask.shape[1] - x_min)
        
        # Crop the mask if necessary
        mask_cropped = mask[:copy_height, :copy_width]
        
        # Insert into full mask
        full_mask[y_min:y_min+copy_height, x_min:x_min+copy_width] = mask_cropped
        return full_mask
    
    # Mark ROI regions in Green
    for region_data in roi_regions:
        region = region_data['region_data']
        mask = region['mask']
        bbox = region['bbox']
        
        # Create full-size mask safely
        full_mask = np.zeros(image.shape[:2], dtype=bool)
        full_mask = safe_mask_insert(full_mask, mask, bbox)
        
        # Color the region
        roi_overlay[full_mask] = [0, 255, 0]  # Green for ROI
        combined_overlay[full_mask] = [0, 255, 0]  # Green for ROI
    
    # Mark nonROI regions in Blue
    for region_data in nonroi_regions:
        region = region_data['region_data']
        mask = region['mask']
        bbox = region['bbox']
        
        # Create full-size mask safely
        full_mask = np.zeros(image.shape[:2], dtype=bool)
        full_mask = safe_mask_insert(full_mask, mask, bbox)
        
        # Color the region
        nonroi_overlay[full_mask] = [0, 0, 255]  # Blue for nonROI
        combined_overlay[full_mask] = [0, 0, 255]  # Blue for nonROI
    
    # Create the visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # ROI regions only (Green)
    axes[0, 1].imshow(roi_overlay)
    axes[0, 1].set_title(f'ROI Regions (Green)\n{len(roi_regions)} regions')
    axes[0, 1].axis('off')
    
    # nonROI regions only (Blue)
    axes[1, 0].imshow(nonroi_overlay)
    axes[1, 0].set_title(f'nonROI Regions (Blue)\n{len(nonroi_regions)} regions')
    axes[1, 0].axis('off')
    
    # Combined view
    axes[1, 1].imshow(combined_overlay)
    axes[1, 1].set_title(f'Combined: Green=ROI, Blue=nonROI\nTotal: {len(roi_regions) + len(nonroi_regions)} regions')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\n=== ROI Classification Results ===")
    print(f"ROI regions: {len(roi_regions)}")
    print(f"nonROI regions: {len(nonroi_regions)}")
    print(f"Total regions: {len(roi_regions) + len(nonroi_regions)}")
    
    # Show ROI scores for each region
    if roi_regions:
        roi_scores = [r['roi_score'] for r in roi_regions]
        print(f"ROI scores range: {min(roi_scores):.3f} - {max(roi_scores):.3f}")
        for i, region in enumerate(roi_regions[:5]):  # Show first 5
            print(f"  ROI region {i}: score {region['roi_score']:.3f}")
    
    if nonroi_regions:
        nonroi_scores = [r['roi_score'] for r in nonroi_regions]
        print(f"nonROI scores range: {min(nonroi_scores):.3f} - {max(nonroi_scores):.3f}")



def complete_roi_classification_pipeline(image):
    """
    Complete pipeline: detect edges → find regions → classify ROI/nonROI
    """
    # Step 1: Get edge-based ROI map (your existing code)
    edge_roi_map, _ = detect_roi_coarse(image)
    
    # Step 2: Segment image into regions (using SLIC or your method)
    regions_list = segment_image_into_regions(image)
    
    # Step 3: Classify each region as ROI or nonROI
    roi_regions, nonroi_regions = classify_roi_regions(image, regions_list, edge_roi_map)
    
    # Step 4: Visualize results
    visualize_roi_classification(image, roi_regions, nonroi_regions)
    
    return roi_regions, nonroi_regions

def segment_image_into_regions(image):
    """
    Segment image into regions (using SLIC or your preferred method)
    """
    # Using SLIC for segmentation
    segments = slic(image, n_segments=100, compactness=10, sigma=1)
    
    regions_list = []
    for segment_id in np.unique(segments):
        # Create mask for this segment
        mask = (segments == segment_id)
        
        # Get bounding box
        ys, xs = np.where(mask)
        if len(ys) == 0:
            continue
            
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        
        regions_list.append({
            'mask': mask[y_min:y_max+1, x_min:x_max+1],
            'bbox': (x_min, y_min, x_max, y_max),
            'id': segment_id
        })
    
    return regions_list




# Run the complete pipeline
if __name__ == "__main__":
    image_name = 'images/Lenna.webp'
    image = cv2.imread(image_name)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    roi_regions, nonroi_regions = complete_roi_classification_pipeline(image_rgb)
    
    # Now you have clear ROI/nonROI classification!
    print(f"Found {len(roi_regions)} ROI regions and {len(nonroi_regions)} nonROI regions")