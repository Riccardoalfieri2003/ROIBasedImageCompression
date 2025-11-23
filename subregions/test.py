

from regions.division import should_split
from roi.clahe import get_enhanced_image
from roi.new_roi import get_edge_map, compute_local_density, suggest_automatic_threshold, process_and_unify_borders

import math
import cv2
from matplotlib import pyplot as plt
import numpy as np


from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import numpy as np





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

    """
    Add separate regions are computed differently, not in one
    """

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


    should_split_roi, roi_score =should_split(roi_only_image)
    should_split_nonroi, nonroi_score=should_split(nonroi_only_image)

    print(f"should_split_roi: {should_split_roi}")
    print(f"roi_score {roi_score}")
    print()
    print(f"should_split_nonroi: {should_split_nonroi}")
    print(f"nonroi_score {nonroi_score}")




    # Apply SLIC only on ROI regions
    roi_segments = slic(roi_only_image, 
                    n_segments=math.ceil(roi_score*2),  # Adjust based on your needs
                    compactness=10,
                    sigma=1,
                    mask=roi_mask)  # This ensures SLIC only works on ROI areas

    # Display results
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(roi_only_image)
    plt.title('ROI Regions Only')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(roi_segments, cmap='nipy_spectral')
    plt.title('SLIC on ROI Regions')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(mark_boundaries(roi_only_image, roi_segments))
    plt.title('SLIC Boundaries on ROI')
    plt.axis('off')

    plt.tight_layout()
    plt.show()



    # Apply SLIC only on non-ROI regions
    nonroi_segments = slic(nonroi_only_image, 
                        n_segments=math.ceil(nonroi_score/5),
                        compactness=10,
                        sigma=1,
                        mask=nonroi_mask)

    # Display results
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(nonroi_only_image)
    plt.title('Non-ROI Regions Only')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(nonroi_segments, cmap='nipy_spectral')
    plt.title('SLIC on Non-ROI Regions')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(mark_boundaries(nonroi_only_image, nonroi_segments))
    plt.title('SLIC Boundaries on Non-ROI')
    plt.axis('off')

    plt.tight_layout()
    plt.show()