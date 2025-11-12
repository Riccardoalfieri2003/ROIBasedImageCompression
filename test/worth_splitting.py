import cv2
from matplotlib import pyplot as plt
import numpy as np



import sys
import os

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from regions.irregular_region import should_split_irregular_region


def main():
    image_name = 'images/cerchio.png'
    image = cv2.imread(image_name)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



    # Display original image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis('off')

    # Option 1: Test with the entire image as a "rectangle" region
    bbox_region = image_rgb  # The entire image
    clean_mask = np.ones(image_rgb.shape[:2], dtype=bool)  # Full mask
    
    # Option 2: Test with a specific rectangular region within the image
    # height, width = image_rgb.shape[:2]
    # x1, y1, x2, y2 = width//4, height//4, 3*width//4, 3*height//4  # Center region
    # bbox_region = image_rgb[y1:y2, x1:x2]
    # clean_mask = np.ones(bbox_region.shape[:2], dtype=bool)  # Full mask for the subregion

    # Option 3: Test with an irregular mask (circular for example)
    # bbox_region = image_rgb
    # clean_mask = np.zeros(image_rgb.shape[:2], dtype=bool)
    # center_y, center_x = image_rgb.shape[0]//2, image_rgb.shape[1]//2
    # radius = min(center_x, center_y) // 2
    # y, x = np.ogrid[:image_rgb.shape[0], :image_rgb.shape[1]]
    # mask_circle = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    # clean_mask[mask_circle] = True
    # bbox_region = image_rgb  # Keep full image as bbox_region

    # Display the region we're testing
    plt.subplot(1, 2, 2)
    display_image = image_rgb.copy()
    if bbox_region.shape != image_rgb.shape:  # If we're using a subregion

        # Create overlay for subregion
        overlay = np.zeros_like(image_rgb)
        y1, y2 = image_rgb.height//4, 3*image_rgb.height//4
        x1, x2 = image_rgb.width//4, 3*image_rgb.width//4
        overlay[y1:y2, x1:x2] = [255, 0, 0]
        plt.imshow(display_image)
        plt.imshow(overlay, alpha=0.3)
        plt.title("Testing Region (red overlay)")
    else:
        # Create overlay for irregular mask
        overlay = np.zeros((*image_rgb.shape[:2], 4))
        overlay[clean_mask] = [1, 0, 0, 0.5]  # Red overlay for mask
        plt.imshow(display_image)
        plt.imshow(overlay)
        plt.title("Testing Region (red mask)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

    print(f"Image: {image_name}")
    print(f"Region shape: {bbox_region.shape}")
    print(f"Mask coverage: {np.sum(clean_mask)}/{clean_mask.size} pixels ({np.sum(clean_mask)/clean_mask.size*100:.1f}%)")
    
    # Test the function
    split, n_segments = should_split_irregular_region(bbox_region, clean_mask, debug=True)
    print(f"Split decision: {split}")

if __name__ == "__main__":
    main()