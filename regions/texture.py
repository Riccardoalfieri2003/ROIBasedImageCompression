import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np


"""
Da controllare, non sembra terminare
"""




# Load the image
image = cv2.imread('Hawaii.jpg')
# Convert from BGR to RGB for matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to grayscale (optional)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)





# Calculate texture measure using Local Binary Patterns
def calculate_lbp(gray_image):
    lbp = np.zeros_like(gray_image)
    for i in range(1, gray_image.shape[0]-1):
        for j in range(1, gray_image.shape[1]-1):
            center = gray_image[i, j]
            code = 0
            code |= (gray_image[i-1, j-1] > center) << 7
            code |= (gray_image[i-1, j] > center) << 6
            code |= (gray_image[i-1, j+1] > center) << 5
            code |= (gray_image[i, j+1] > center) << 4
            code |= (gray_image[i+1, j+1] > center) << 3
            code |= (gray_image[i+1, j] > center) << 2
            code |= (gray_image[i+1, j-1] > center) << 1
            code |= (gray_image[i, j-1] > center) << 0
            lbp[i, j] = code
    return lbp

# Calculate LBP texture map
lbp_image = calculate_lbp(gray)

# Cluster textures using K-Means

# Define criteria and apply K-Means
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

texture_values = lbp_image.reshape((-1, 1))
texture_values = np.float32(texture_values)
_, texture_labels, texture_centers = cv2.kmeans(texture_values, 4, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Create texture-based segmentation
texture_segmented = texture_centers[texture_labels.flatten()].reshape(gray.shape)
texture_segmented = np.uint8(texture_segmented)


overlay = cv2.addWeighted(image_rgb, 0, texture_segmented, 1, 0)

# Show the overlayed image full-size
plt.figure(figsize=(12, 8))
plt.imshow(overlay)
plt.axis('off')
plt.title(f'Overlay: Original + Segmented', fontsize=16)
plt.tight_layout()
plt.show()