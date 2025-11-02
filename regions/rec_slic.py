import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.util import img_as_ubyte


image = cv2.imread('images/Lenna.webp')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



def recursive_slic(
    image,
    n_segments=3,
    compactness=1,
    sigma=1,
    depth_limit=None,
    current_depth=0,
    parent_label=None,
    all_segments=None,
):
    if all_segments is None:
        all_segments = []

    # --- Run SLIC on current image ---
    try:
        segments = slic(
            image,
            n_segments=n_segments,
            compactness=compactness,
            sigma=sigma,
            start_label=1,
        )
    except Exception as e:
        # safety fallback if SLIC fails
        print(f"Skipping region at depth {current_depth} due to error: {e}")
        return all_segments

    unique_labels = np.unique(segments)

    # --- For each region ---
    for label in unique_labels:
        mask = (segments == label).astype(np.uint8)

        # Bounding box
        ys, xs = np.where(mask)
        if ys.size == 0 or xs.size == 0:
            continue  # skip empty
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        cropped_image = image[y_min:y_max, x_min:x_max]
        cropped_mask = mask[y_min:y_max, x_min:x_max]

        # Skip too small regions
        if cropped_image.shape[0] < 5 or cropped_image.shape[1] < 5:
            continue

        # Skip uniform regions
        if np.std(cropped_image) < 2:
            continue

        # Fill background with mean color
        region_only = cropped_image.copy().astype(np.float32)
        region_pixels = region_only[cropped_mask.astype(bool)]
        if len(region_pixels) == 0:
            continue
        mean_color = np.mean(region_pixels, axis=0)
        region_only[~cropped_mask.astype(bool)] = mean_color

        # Save region info
        all_segments.append({
            "depth": current_depth,
            "label": label,
            "parent": parent_label,
            "bbox": (x_min, y_min, x_max, y_max),
            "mask": cropped_mask,
            "image": region_only.astype(np.uint8),
        })

        # --- Stop recursion conditions ---
        if depth_limit is not None and current_depth >= depth_limit:
            continue

        # --- Subdivide only if valid ---
        try:
            sub_segments = slic(
                region_only.astype(np.uint8),
                n_segments=n_segments,
                compactness=compactness,
                sigma=sigma,
                start_label=1,
                mask=cropped_mask.astype(bool),
            )
        except Exception as e:
            print(f"Sub-SLIC failed at depth {current_depth}: {e}")
            continue

        # Stop if we only got 1 region
        if len(np.unique(sub_segments)) <= 1:
            continue

        # Recurse
        recursive_slic(
            region_only.astype(np.uint8),
            n_segments=n_segments,
            compactness=compactness,
            sigma=sigma,
            depth_limit=depth_limit,
            current_depth=current_depth + 1,
            parent_label=label,
            all_segments=all_segments,
        )

    return all_segments


def visualize_segments_on_image(base_image, all_segments):
    """
    Draw red contours for every segment (from all recursion depths).
    """
    contour_image = base_image.copy()

    for seg in all_segments:
        mask = seg["mask"]
        x_min, y_min, x_max, y_max = seg["bbox"]
        full_mask = np.zeros(base_image.shape[:2], dtype=np.uint8)
        full_mask[y_min:y_max, x_min:x_max] = mask * 255

        contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 1)  # red contours

    return contour_image


# --- Run recursive segmentation ---
all_segments = recursive_slic(
    image_rgb,
    n_segments=3,
    compactness=1,
    sigma=1,
    depth_limit=3  # or None for auto-depth
)





print(f"Total regions found: {len(all_segments)}")

# Visualize results by depth
for seg in all_segments:
    plt.figure(figsize=(5, 5))
    plt.imshow(seg["image"])
    plt.title(f"Depth {seg['depth']} | Parent {seg['parent']}")
    plt.axis("off")
    plt.show()



"""# --- Draw all contours ---
contour_result = visualize_segments_on_image(image_rgb, all_segments)

plt.figure(figsize=(10, 8))
plt.imshow(contour_result)
plt.title("Recursive SLIC Contours (red = detected regions)")
plt.axis("off")
plt.show()"""
