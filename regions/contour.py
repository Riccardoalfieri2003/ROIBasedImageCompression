import cv2
import numpy as np
import matplotlib.pyplot as plt


# Load the image
image = cv2.imread('Hawaii.jpg')
# Convert from BGR to RGB for matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to grayscale (optional)
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Compute gradient magnitude
gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
gradient_magnitude = cv2.magnitude(grad_x, grad_y)

# Normalize to 8-bit
gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)




# Preprocessing: noise reduction and thresholding
blur = cv2.GaussianBlur(gray, (5, 5), 0)


#_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# Use gradient for thresholding instead of gray
_, thresh = cv2.threshold(gradient_magnitude, 30, 255, cv2.THRESH_BINARY)

# Remove noise with morphological operations
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Finding sure foreground area using distance transform
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.35 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)

# Find unknown region (boundary area)
unknown = cv2.subtract(sure_bg, sure_fg)

# Marker labelling for watershed
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

# Apply watershed
markers = cv2.watershed(image_rgb, markers)
watershed_result = image_rgb.copy()
watershed_result[markers == -1] = [255, 0, 0]  # Mark boundaries in red




plt.imshow(watershed_result)
plt.title('Contour-Based (Watershed)')
plt.axis('off')
plt.show()