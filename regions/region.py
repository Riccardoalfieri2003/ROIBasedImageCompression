import numpy as np
import cv2
from skimage.segmentation import slic, mark_boundaries
import matplotlib.pyplot as plt









def recursive_slic(image, mask=None, depth_limit=None, depth=0, compactness=1, sigma=1, n_segments=3):
    """
    Recursively apply SLIC segmentation inside each region, isolating subregions cleanly.
    """
    # Base case for recursion depth
    if depth_limit is not None and depth >= depth_limit:
        return []

    # If no mask is given, use the whole image
    if mask is None:
        mask = np.ones(image.shape[:2], dtype=bool)

    # --- Apply SLIC on the masked area only ---
    sub_segments = slic(
        image,
        n_segments=n_segments,
        compactness=compactness,
        sigma=sigma,
        start_label=1,
        mask=mask
    )

    all_segments = []
    unique_labels = np.unique(sub_segments[mask])

    for label in unique_labels:
        sub_mask = (sub_segments == label)

        # Skip empty or invalid masks
        if np.count_nonzero(sub_mask) < 10:
            continue

        # --- Extract only the region (no filling!) ---
        ys, xs = np.where(sub_mask)
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        cropped_image = image[y_min:y_max, x_min:x_max]
        cropped_mask = sub_mask[y_min:y_max, x_min:x_max]

        # Store this region info
        all_segments.append({
            "mask": cropped_mask,
            "bbox": (x_min, y_min, x_max, y_max),
            "depth": depth
        })

        # Recurse further
        all_segments += recursive_slic(
            image,
            mask=sub_mask,
            depth_limit=depth_limit,
            depth=depth + 1,
            compactness=compactness,
            sigma=sigma,
            n_segments=n_segments
        )

    return all_segments





def visualize_segments_on_image(base_image, all_segments):
    """
    Draw red contours for every segment found at all recursion levels.
    """
    contour_image = base_image.copy()

    for seg in all_segments:
        mask = seg["mask"]
        x_min, y_min, x_max, y_max = seg["bbox"]

        full_mask = np.zeros(base_image.shape[:2], dtype=np.uint8)
        full_mask[y_min:y_max, x_min:x_max] = mask.astype(np.uint8) * 255

        contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 1)

    return contour_image



# --- Example usage ---
if __name__ == "__main__":
    image = cv2.imread("images/waikiki.jpg")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    all_segments = recursive_slic(
        image_rgb,
        depth_limit=3,  # optional
        n_segments=3,
        compactness=1,
        sigma=1
    )

    print(f"Numero di regioni: {len(all_segments)}")

    result = visualize_segments_on_image(image_rgb, all_segments)

    plt.figure(figsize=(12, 8))
    plt.imshow(result)
    plt.title("Recursive SLIC with Clean Region Isolation")
    plt.axis("off")
    plt.show()
