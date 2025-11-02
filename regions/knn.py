import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('Lenna.webp')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_rgb = cv2.GaussianBlur(image, (5, 5), 1.0)



# --- K-MEANS SEGMENTATION ---
pixel_values = image_rgb.reshape((-1, 3))
pixel_values = np.float32(pixel_values)




criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 4
_, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
labels_2d = labels.reshape(image_rgb.shape[:2])

# --- CONNECTED REGIONS ---
output_colored = np.zeros_like(image_rgb)  # blank canvas for all colored regions

region_count = 0
for cluster_id in range(k):
    mask = np.uint8(labels_2d == cluster_id)
    num_labels, components = cv2.connectedComponents(mask)

    print(f"Cluster {cluster_id}: {num_labels - 1} connected regions found")

    for region_id in range(1, num_labels):
        region_mask = np.uint8(components == region_id)
        color = np.random.randint(0, 255, size=3).tolist()  # random color
        output_colored[region_mask == 1] = color
        region_count += 1



print(f"\nTotal regions detected: {region_count}")

# --- DISPLAY RESULTS ---
plt.figure(figsize=(12, 8))
plt.imshow(output_colored)
plt.title('Distinct Regions (Each in a Different Color)')
plt.axis('off')
plt.tight_layout()
plt.show()
