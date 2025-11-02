import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Load and convert
image = cv2.imread('Lenna.webp')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



def adaptive_contrast_enhancement(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    contrast = np.std(gray)
    min_val, max_val = np.min(gray), np.max(gray)
    dynamic_range = max_val - min_val

    # Adapt CLAHE strength: higher for lower contrast
    if contrast < 40:
        clip = 4.0
    elif contrast < 60:
        clip = 2.5
    else:
        clip = 1.5  # image already has good contrast

    # Adapt global scale: stronger boost for narrow range
    alpha = 1.2 + (80 - contrast) / 100.0
    alpha = np.clip(alpha, 1.0, 1.8)
    beta = (128 - np.mean(gray)) * 0.2  # adjust brightness slightly

    # Apply CLAHE
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

    # Global scaling
    enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
    return enhanced






"""# Convert to LAB for perceptual lightness contrast
lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
l, a, b = cv2.split(lab)

# Equalize only the lightness channel
l_eq = cv2.equalizeHist(l)

# Merge back and convert to RGB
lab_eq = cv2.merge((l_eq, a, b))
enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
"""


"""lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
l, a, b = cv2.split(lab)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
l_clahe = clahe.apply(l)

lab_clahe = cv2.merge((l_clahe, a, b))
contrast_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

# Optional global intensity boost
enhanced = cv2.convertScaleAbs(contrast_clahe, alpha=1.3, beta=5)"""



"""
enhanced=adaptive_contrast_enhancement(image_rgb)

smoothed = cv2.bilateralFilter(enhanced, d=20, sigmaColor=75, sigmaSpace=75)
#smoothed = cv2.pyrMeanShiftFiltering(enhanced, sp=25, sr=50)

plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.imshow(enhanced)
plt.title("After Contrast Enhancement")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(smoothed)
plt.title("After Edge-Preserving Smoothing")
plt.axis('off')
plt.show()"""







"""Knn



# --- K-MEANS SEGMENTATION ---
pixel_values = smoothed.reshape((-1, 3))
pixel_values = np.float32(pixel_values)




criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 4
_, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
labels_2d = labels.reshape(smoothed.shape[:2])

# --- CONNECTED REGIONS ---
output_colored = np.zeros_like(smoothed)  # blank canvas for all colored regions

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


"""




"""
# --- SLIC SUPERPIXELS (OpenCV) ---
image = smoothed.copy()
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.shape[2] == 3 else image

try:
    slic = cv2.ximgproc.createSuperpixelSLIC(
        image_rgb, algorithm=cv2.ximgproc.SLICO, region_size=20, ruler=10.0
    )
    slic.iterate(10)

    # Retrieve labels and contour mask
    labels = slic.getLabels()
    mask_slic = slic.getLabelContourMask()
    num_superpixels = slic.getNumberOfSuperpixels()

    print("Number of superpixels:", num_superpixels)

    # Visualize boundaries
    output = image_rgb.copy()
    output[mask_slic == 255] = [255, 0, 0]  # red boundaries

    plt.figure(figsize=(12, 8))
    plt.imshow(output)
    plt.title(f"SLIC Superpixels ({num_superpixels} regions)")
    plt.axis("off")
    plt.show()

except AttributeError:
    print("⚠️ cv2.ximgproc not available. Try installing opencv-contrib-python:")
    print("   pip install opencv-contrib-python")
"""





from skimage.segmentation import slic, mark_boundaries

# --- Run SLIC segmentation ---
segments = slic(image_rgb, n_segments=3, compactness=1, sigma=1, start_label=1)
num_segments = len(np.unique(segments))
print("Number of superpixels:", num_segments)

# --- Create a copy to draw contours ---
contour_image = image_rgb.copy()

unique_labels = np.unique(segments)
for label in unique_labels:
    mask = np.uint8(segments == label) * 255  # binary mask for this region

    # Find contours for the region
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the image
    cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 1)  # red contours


# --- Show the result ---
plt.figure(figsize=(10, 8))
plt.imshow(contour_image)
plt.title(f"SLIC Contours - {num_segments} Regions")
plt.axis("off")
plt.show()




# --- Select the region you want to isolate ---
for i in range(1, num_segments+1):
    target_label = i  # <--- change this to the region you want
    mask = (segments == target_label).astype(np.uint8)

    # --- Find the bounding box of the region ---
    ys, xs = np.where(mask)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    # --- Crop both image and mask ---
    cropped_image = image_rgb[y_min:y_max, x_min:x_max]
    cropped_mask = mask[y_min:y_max, x_min:x_max]

    # --- Apply the mask cleanly ---
    # Instead of black outside the region, we can use transparency or a white background.
    region_only = cropped_image.copy()
    background = np.ones_like(region_only, dtype=np.uint8) * 255  # white background
    region_only = np.where(cropped_mask[..., None].astype(bool), region_only, background)

    # --- Show the result ---
    plt.figure(figsize=(6, 6))
    plt.imshow(region_only)
    plt.title(f"Isolated Region {target_label}")
    plt.axis("off")
    plt.show()










"""

for label in unique_labels:
    # Binary mask for this region
    mask = (segments == label).astype(np.uint8)

    # Get bounding box of the region
    ys, xs = np.where(mask)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    # Crop both the image and the mask to the bounding box
    cropped_image = image_rgb[y_min:y_max, x_min:x_max]
    cropped_mask = mask[y_min:y_max, x_min:x_max]

    # Keep only the region (mask everything else out, but don't feed black pixels to slic)
    # Convert to float and normalize to avoid 0–black confusion
    region_pixels = cropped_image.copy().astype(np.float32)
    region_pixels[~cropped_mask.astype(bool)] = np.nan  # mark outside region as NaN

    # Fill NaNs by interpolation or by mean color of valid pixels (optional)
    valid_mask = np.isfinite(region_pixels).all(axis=-1)
    mean_color = np.nanmean(region_pixels[valid_mask], axis=0)
    region_pixels[~valid_mask] = mean_color  # fill outside with mean color

    # Run SLIC on this cropped and cleaned region
    subsegments = slic(
        region_pixels,
        n_segments=2,
        compactness=1,
        sigma=1,
        start_label=1,
        mask=cropped_mask.astype(bool)  # ensures slic works only inside the region
    )

    plt.figure(figsize=(6, 6))
    plt.imshow(mark_boundaries(cropped_image, subsegments))
    plt.title(f"Sub-SLIC inside region {label}")
    plt.axis("off")
    plt.show()

"""




