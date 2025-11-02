
import cv2
import numpy as np
import matplotlib.pyplot as plt




# Load the image
#image = cv2.imread('cerchio.png')
image = cv2.imread('Hawaii.jpg')
# Convert from BGR to RGB for matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to grayscale (optional)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)








# Initialize SLIC superpixels
slic = cv2.ximgproc.createSuperpixelSLIC(image_rgb, algorithm=cv2.ximgproc.SLICO, region_size=100, ruler=30.0)
slic.iterate(10)  # Number of iterations

# Get the superpixel segmentation
superpixels_mask = slic.getLabels()
superpixels_contour = slic.getLabelContourMask()

# Create result with superpixel boundaries
superpixel_result = image_rgb.copy()
superpixel_result[superpixels_contour == 255] = [255, 0, 0]  # Draw boundaries in red


plt.imshow(superpixel_result)
plt.title('SLIC Superpixels')
plt.axis('off')

plt.tight_layout()
plt.show()