

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




def normalize_result(score, window_size):
        return window_size/( 1+ math.exp(-12*(score-0.5)) ) 













from skimage.segmentation import find_boundaries
from skimage.measure import find_contours
import numpy as np

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

def save_boundaries_to_file(segment_boundaries, filename):
    """
    Save segment boundaries to a text file
    """
    with open(filename, 'w') as f:
        f.write("SLIC Segment Boundaries\n")
        f.write("=" * 50 + "\n")
        
        for segment in segment_boundaries:
            f.write(f"\nSegment {segment['segment_id']}:\n")
            f.write(f"Area: {segment['area']} pixels\n")
            f.write(f"Boundary points: {segment['num_points']}\n")
            f.write("Coordinates (y,x):\n")
            
            # Write coordinates in batches for readability
            f.write("[")
            for i in range(0, len(segment['boundary_coords']), 5):
                batch = segment['boundary_coords'][i:i+5]
                coord_str = "  ".join([f"({y:.1f},{x:.1f})," for y, x in batch])
                f.write(f"  {coord_str} \n")
            f.write("]")









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
    

    """    
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

        window =math.ceil( math.ceil(math.log(bbox_region.size, 10)) * math.log(bbox_region.size) )
        print(f"Window: {window} px")
        normalized_overall_score=normalize_result(overall_score, window)
        optimal_segments=math.ceil(normalized_overall_score)
        
        # Calculate optimal segments based on score
        #optimal_segments = calculate_optimal_segments(overall_score, region['area'], min_segments=1, max_segments=factor)
        if optimal_segments<=0: optimal_segments=1

        roi_segments, texture_map = enhanced_slic_with_texture(bbox_region, n_segments=optimal_segments)
        
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

        plt.subplot(1, 4, 1)
        plt.imshow(region_image)
        plt.title(f'ROI Region {i+1}')
        plt.axis('off')

        plt.subplot(1, 4, 2)
        # Show segments only within the irregular region
        segments_display = roi_segments.copy()
        segments_display[~bbox_mask] = 0  # Set background to 0
        plt.imshow(segments_display, cmap='nipy_spectral')
        plt.title(f'SLIC Segments: {roi_segments.max()}')
        plt.axis('off')

        plt.subplot(1, 4, 3)
        # Create boundaries only within the irregular region
        boundaries_image = mark_boundaries(bbox_region, roi_segments)
        boundaries_image[~bbox_mask] = 0  # Set background to black
        plt.imshow(boundaries_image)
        plt.title('SLIC Boundaries (Region Only)')
        plt.axis('off')

        plt.subplot(1, 4, 4)
        plt.imshow(texture_map)
        plt.title('texture_map')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    """



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

        window = math.ceil(math.ceil(math.log(bbox_region.size, 10)) * math.log(bbox_region.size))
        print(f"Window: {window} px")
        normalized_overall_score = normalize_result(overall_score, window)
        optimal_segments = math.ceil(normalized_overall_score)
        
        if optimal_segments <= 0: 
            optimal_segments = 1

        roi_segments, texture_map = enhanced_slic_with_texture(bbox_region, n_segments=optimal_segments)
        
        # 🆕 EXTRACT SEGMENT BOUNDARIES
        segment_boundaries = extract_slic_segment_boundaries(roi_segments, bbox_mask)
        
        print(f"  Found {len(segment_boundaries)} sub-regions")
        
        # 🆕 SAVE TO FILE
        filename = f"region_{i+1}_boundaries.txt"
        save_boundaries_to_file(segment_boundaries, filename)
        print(f"  Boundaries saved to: {filename}")
        
        # 🆕 PRINT SUMMARY
        total_boundary_points = sum(seg['num_points'] for seg in segment_boundaries)
        print(f"  Total boundary points: {total_boundary_points}")
        
        # Visualize
        visualize_split_analysis(
            region_image=region_image,
            overall_score=overall_score,
            color_score=color_score, 
            texture_score=texture_score,
            optimal_segments=optimal_segments
        )

        # Display SLIC results WITH BOUNDARIES
        plt.figure(figsize=(18, 5))

        plt.subplot(1, 5, 1)
        plt.imshow(region_image)
        plt.title(f'ROI Region {i+1}')
        plt.axis('off')

        plt.subplot(1, 5, 2)
        # Show segments only within the irregular region
        segments_display = roi_segments.copy()
        segments_display[~bbox_mask] = 0  # Set background to 0
        plt.imshow(segments_display, cmap='nipy_spectral')
        plt.title(f'SLIC Segments: {roi_segments.max()}')
        plt.axis('off')

        plt.subplot(1, 5, 3)
        # Create boundaries only within the irregular region
        boundaries_image = mark_boundaries(bbox_region, roi_segments)
        boundaries_image[~bbox_mask] = 0  # Set background to black
        plt.imshow(boundaries_image)
        plt.title('SLIC Boundaries (Region Only)')
        plt.axis('off')

        plt.subplot(1, 5, 4)
        plt.imshow(texture_map)
        plt.title('Texture Map')
        plt.axis('off')
        
        # 🆕 PLOT EXTRACTED BOUNDARIES
        plt.subplot(1, 5, 5)
        plt.imshow(bbox_mask, cmap='gray')
        
        # Plot each segment boundary
        colors = plt.cm.Set3(np.linspace(0, 1, len(segment_boundaries)))
        for j, segment in enumerate(segment_boundaries):
            coords = np.array(segment['boundary_coords'])
            if len(coords) > 0:
                plt.plot(coords[:, 1], coords[:, 0], color=colors[j], linewidth=2, 
                        label=f'Seg {segment["segment_id"]}')
        
        plt.title(f'Extracted Boundaries\n{len(segment_boundaries)} segments')
        plt.axis('off')
        plt.legend(loc='upper right', fontsize=8)

        plt.tight_layout()
        plt.show()
        
        """# 🆕 COMPRESS BOUNDARIES USING OUR FOURIER METHOD
        print(f"  Compressing boundaries with Fourier...")
        compressed_boundaries = []
        
        for j, segment in enumerate(segment_boundaries):
            if len(segment['boundary_coords']) >= 4:  # Need minimum points for Fourier
                compressed = compress_irregular_shape(
                    segment['boundary_coords'],
                    compression_ratio=0.1,
                    decimal_places=3
                )
                
                compressed_boundaries.append({
                    'segment_id': segment['segment_id'],
                    'compressed_data': compressed,
                    'original_points': segment['num_points'],
                    'compressed_coeffs': len(compressed['compressed_coeffs'])
                })
                
                print(f"    Segment {segment['segment_id']}: {segment['num_points']} points → {len(compressed['compressed_coeffs'])} coefficients")
        
        # 🆕 SAVE COMPRESSED BOUNDARIES
        compressed_filename = f"region_{i+1}_compressed_boundaries.npy"
        np.save(compressed_filename, compressed_boundaries)
        print(f"  Compressed boundaries saved to: {compressed_filename}")"""



    
    for i, region in enumerate(nonroi_regions):
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

        window =math.ceil( math.ceil(math.log(bbox_region.size, 10)) * math.log(bbox_region.size) )
        print(f"Window: {window} px")
        normalized_overall_score=normalize_result(overall_score, window)
        optimal_segments=math.ceil(normalized_overall_score)
        
        # Calculate optimal segments based on score
        #optimal_segments = calculate_optimal_segments(overall_score, region['area'], min_segments=1, max_segments=factor)
        if optimal_segments<=0: optimal_segments=1

        nonroi_segments, texture_map = enhanced_slic_with_texture(bbox_region, n_segments=optimal_segments)
        
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

        plt.subplot(1, 4, 1)
        plt.imshow(region_image)
        plt.title(f'ROI Region {i+1}')
        plt.axis('off')

        plt.subplot(1, 4, 2)
        # Show segments only within the irregular region
        segments_display = nonroi_segments.copy()
        segments_display[~bbox_mask] = 0  # Set background to 0
        plt.imshow(segments_display, cmap='nipy_spectral')
        plt.title(f'SLIC Segments: {nonroi_segments.max()}')
        plt.axis('off')

        plt.subplot(1, 4, 3)
        # Create boundaries only within the irregular region
        boundaries_image = mark_boundaries(bbox_region, nonroi_segments)
        boundaries_image[~bbox_mask] = 0  # Set background to black
        plt.imshow(boundaries_image)
        plt.title('SLIC Boundaries (Region Only)')
        plt.axis('off')

        plt.subplot(1, 4, 4)
        plt.imshow(texture_map)
        plt.title('texture_map')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

