import cv2
import math
from encoder.ROI.roi import get_regions, extract_regions

from encoder.subregions.split_score import calculate_split_score, normalize_result
from encoder.subregions.slic import enhanced_slic_with_texture, extract_slic_segment_boundaries, visualize_split_analysis

if __name__ == "__main__":

    image_name = 'images/Lenna.webp'
    image = cv2.imread(image_name)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    unified, regions, roi_image, nonroi_image, roi_mask, nonroi_mask = get_regions(image_rgb)
    roi_regions, nonroi_regions = extract_regions(image_rgb, roi_mask, nonroi_mask)



    roi_subregions=[]
    nonroi_subregions=[]

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
        """filename = f"region_{i+1}_boundaries.txt"
        save_boundaries_to_file(segment_boundaries, filename)
        print(f"  Boundaries saved to: {filename}")"""
        
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