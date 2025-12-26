import cv2
import math

from encoder.ROI.roi import get_regions, extract_regions

if __name__ == "__main__":

    image_name = 'images/Hawaii.jpg'
    image = cv2.imread(image_name)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    unified, regions, roi_image, nonroi_image, roi_mask, nonroi_mask = get_regions(image_rgb)
    roi_regions, nonroi_regions = extract_regions(image_rgb, roi_mask, nonroi_mask)