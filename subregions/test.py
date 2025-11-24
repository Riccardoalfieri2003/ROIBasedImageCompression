

#from regions.division import should_split
from subregions.split_score import calculate_split_score, calculate_optimal_segments
from roi.clahe import get_enhanced_image
from roi.new_roi import get_edge_map, compute_local_density, suggest_automatic_threshold, process_and_unify_borders

import sys
import math
import cv2
from matplotlib import pyplot as plt
import numpy as np


from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import numpy as np






from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import numpy as np

def extract_connected_regions(mask, original_image):
    """Extract all connected components from a mask"""
    labeled_mask = label(mask)
    regions = regionprops(labeled_mask)
    
    region_data = []
    for region in regions:
        # Create mask for this specific region
        single_region_mask = np.zeros_like(mask)
        single_region_mask[region.coords[:, 0], region.coords[:, 1]] = True
        
        # Extract the region from original image
        region_image = original_image.copy()
        region_image[~single_region_mask] = 0
        
        # Get bounding box coordinates
        minr, minc, maxr, maxc = region.bbox
        bbox_image = original_image[minr:maxr, minc:maxc]
        bbox_mask = single_region_mask[minr:maxr, minc:maxc]
        
        region_data.append({
            'mask': single_region_mask,
            'full_image': region_image,
            'bbox_image': bbox_image,
            'bbox_mask': bbox_mask,
            'bbox': region.bbox,
            'area': region.area,
            'coords': region.coords,
            'label': region.label
        })
    
    return region_data



def plot_regions(regions, title, max_display=12):
    """Plot multiple regions in a grid"""
    n_regions = min(len(regions), max_display)
    if n_regions == 0:
        print(f"No regions to display for {title}")
        return
    
    cols = 4
    rows = (n_regions + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(rows * cols):
        row = i // cols
        col = i % cols
        
        if i < n_regions:
            region = regions[i]
            axes[row, col].imshow(region['bbox_image'])
            axes[row, col].set_title(f'Region {i+1}\nArea: {region["area"]} px')
            axes[row, col].axis('off')
        else:
            axes[row, col].axis('off')
    
    # Hide empty subplots
    for i in range(n_regions, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.suptitle(f'{title} - {len(regions)} regions found', fontsize=16)
    plt.tight_layout()
    plt.show()






from skimage.color import rgb2gray, rgb2lab
from skimage import filters
import numpy as np
from skimage.segmentation import slic

def enhanced_slic_with_texture(image, n_segments, compactness=10, texture_weight=0.3):
    """
    Enhanced SLIC that considers both color and texture
    """
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




if __name__ == "__main__":
    image_name = 'images/kauai.jpg'
    image = cv2.imread(image_name)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    enhanced_image_rgb=get_enhanced_image(image_rgb, shadow_threshold=100)

    print(f"Size: {image_rgb.size}")
    factor = math.ceil(math.log(image_rgb.size, 10)) * math.log(image_rgb.size)
    print(f"factor: {factor}")


    # Compare with your ROI detection
    edge_map = get_edge_map(enhanced_image_rgb)
    edge_density = compute_local_density(edge_map, kernel_size=3)

    threshold = suggest_automatic_threshold(edge_density, edge_map, method="mean") / 100
    
    window_size = math.floor(factor)
    min_region_size= math.ceil( image_rgb.size / math.pow(10, math.ceil(math.log(image_rgb.size, 10))-3 ) ) 
    print(f"min_region_size: {min_region_size}")

    print(f"\nWindow: {window_size}x{window_size}, Threshold: {threshold:.3f} ===")

    unified, regions, roi_image, nonroi_image, roi_mask, nonroi_mask = process_and_unify_borders(
        edge_map, edge_density, enhanced_image_rgb,
        density_threshold=threshold,
        #unification_method=method,
        min_region_size=min_region_size
    )


    


    # Create a version where only ROI regions are visible (non-ROI is black)
    roi_only_image = image_rgb.copy()
    roi_only_image[~roi_mask] = 0

    # Create a version where only non-ROI regions are visible
    nonroi_only_image = image_rgb.copy()
    nonroi_only_image[~nonroi_mask] = 0




    # Extract all connected regions for ROI and non-ROI
    roi_regions = extract_connected_regions(roi_mask, image_rgb)
    nonroi_regions = extract_connected_regions(nonroi_mask, image_rgb)

    print(f"Found {len(roi_regions)} ROI regions")
    print(f"Found {len(nonroi_regions)} non-ROI regions")

    # Display some statistics
    print("\nROI Regions (sorted by area):")
    for i, region in enumerate(sorted(roi_regions, key=lambda x: x['area'], reverse=True)[:5]):
        print(f"  Region {i+1}: Area = {region['area']} pixels")

    print("\nNon-ROI Regions (sorted by area):")
    for i, region in enumerate(sorted(nonroi_regions, key=lambda x: x['area'], reverse=True)[:5]):
        print(f"  Region {i+1}: Area = {region['area']} pixels")


    # Display ROI regions
    plot_regions(roi_regions, "ROI Regions")

    # Display non-ROI regions
    plot_regions(nonroi_regions, "Non-ROI Regions")



    # Display both
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(roi_only_image)
    plt.title('ROI Regions Only')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(nonroi_only_image)
    plt.title('Non-ROI Regions Only')
    plt.axis('off')

    plt.tight_layout()
    plt.show()



    """
    Change the split score
    """



    roi_subregions=[]
    nonroi_subregions=[]


    """for region in roi_regions:

        # Apply SLIC only to the bounding box region
        minr, minc, maxr, maxc = region['bbox']
        
        # Extract the region from the original image
        bbox_region = image_rgb[minr:maxr, minc:maxc]

        region_image=bbox_region
        #should_split_roi, roi_score = should_split(bbox_region)
        roi_score=calculate_split_score(bbox_region)

        print(f"roi_score {roi_score}")

        if roi_score==0: continue

            # Apply SLIC only on ROI regions
        roi_segments = slic(bbox_region, 
                        n_segments=math.ceil(roi_score*2),  # Adjust based on your needs
                        compactness=10,
                        sigma=1,
                        mask=region["bbox_mask"])  # This ensures SLIC only works on ROI areas
        
        roi_segments, texture_map = enhanced_slic_with_texture(bbox_region, math.ceil(roi_score*2))

        visualize_split_analysis(region_image, overall_score, color_score, texture_score, optimal_segments):"""
    

    for i, region in enumerate(roi_regions):
        # Apply SLIC only to the bounding box region
        minr, minc, maxr, maxc = region['bbox']
        
        # Extract the region from the original image
        bbox_region = image_rgb[minr:maxr, minc:maxc]
        bbox_mask = region['bbox_mask']  # This masks only the actual irregular region

        region_image = bbox_region
        
        # Calculate split score ONLY on the irregular region (using the mask)
        overall_score, color_score, texture_score = calculate_split_score(bbox_region, bbox_mask)
        
        print(f"Region {i+1}:")
        print(f"  Overall score: {overall_score:.3f}")
        print(f"  Color score: {color_score:.3f}")
        print(f"  Texture score: {texture_score:.3f}")

        #if overall_score == 0: 
        #    continue
        
        # Calculate optimal segments based on score
        optimal_segments = calculate_optimal_segments(overall_score, region['area'])
        if optimal_segments<=0: optimal_segments=1
        
        # Apply enhanced SLIC ONLY to the irregular region using the mask
        roi_segments = slic(
            bbox_region, 
            n_segments=optimal_segments,
            compactness=10,
            sigma=1,
            mask=bbox_mask,  # ← THIS IS THE KEY: Only segment the irregular region
            channel_axis=2
        )
        
        # Visualize
        visualize_split_analysis(
            region_image=region_image,
            overall_score=overall_score,
            color_score=color_score, 
            texture_score=texture_score,
            optimal_segments=optimal_segments
        )

        # Display SLIC results
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(region_image)
        plt.title(f'ROI Region {i+1}')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        # Show segments only within the irregular region
        segments_display = roi_segments.copy()
        segments_display[~bbox_mask] = 0  # Set background to 0
        plt.imshow(segments_display, cmap='nipy_spectral')
        plt.title(f'SLIC Segments: {roi_segments.max()}')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        # Create boundaries only within the irregular region
        boundaries_image = mark_boundaries(bbox_region, roi_segments)
        boundaries_image[~bbox_mask] = 0  # Set background to black
        plt.imshow(boundaries_image)
        plt.title('SLIC Boundaries (Region Only)')
        plt.axis('off')

        plt.tight_layout()
        plt.show()


    sys.exit(0)




    for region in nonroi_regions:

        # Apply SLIC only to the bounding box region
        minr, minc, maxr, maxc = region['bbox']
        
        # Extract the region from the original image
        bbox_region = image_rgb[minr:maxr, minc:maxc]

        region_image=bbox_region
        #nonroi_score=calculate_split_score(bbox_region)
        overall_score, color_score, texture_score = calculate_split_score(bbox_region)

        print(f"nonroi_score {overall_score}")

        if overall_score==0: continue

        # Apply SLIC only on non-ROI regions
        nonroi_segments = slic(bbox_region, 
                            n_segments=math.ceil(overall_score),
                            compactness=10,
                            sigma=1,
                            mask=region["bbox_mask"])

        # Display results
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(region_image)
        plt.title('Non-ROI Regions Only')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(nonroi_segments, cmap='nipy_spectral')
        plt.title('SLIC on Non-ROI Regions')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(mark_boundaries(region_image, nonroi_segments))
        plt.title('SLIC Boundaries on Non-ROI')
        plt.axis('off')

        plt.tight_layout()
        plt.show()