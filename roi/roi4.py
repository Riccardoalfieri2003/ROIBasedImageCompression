import cv2
from matplotlib import pyplot as plt
import numpy as np
import cv2
import numpy as np
from scipy import ndimage
from scipy import fftpack
from skimage.segmentation import slic



def show_canny_edges(image, low_threshold=50, high_threshold=150, show_comparison=True):
    """
    Apply Canny edge detection and visualize the results
    
    Args:
        image: RGB image
        low_threshold: Lower threshold for hysteresis
        high_threshold: Upper threshold for hysteresis  
        show_comparison: Whether to show original + edges side by side
    """
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    
    if show_comparison:
        # Show original and edges side by side
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Grayscale image
        axes[1].imshow(gray, cmap='gray')
        axes[1].set_title('Grayscale')
        axes[1].axis('off')
        
        # Canny edges
        axes[2].imshow(edges, cmap='gray')
        axes[2].set_title(f'Canny Edges\nThreshold: {low_threshold}-{high_threshold}')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()
    else:
        # Just show the edges
        plt.figure(figsize=(10, 8))
        plt.imshow(edges, cmap='gray')
        plt.title(f'Canny Edges (Threshold: {low_threshold}-{high_threshold})')
        plt.axis('off')
        plt.show()
    
    # Print edge statistics
    edge_pixels = np.sum(edges > 0)
    total_pixels = edges.size
    print(f"Edge pixels: {edge_pixels}/{total_pixels} ({edge_pixels/total_pixels*100:.2f}%)")
    print(f"Edge threshold range: {low_threshold}-{high_threshold}")
    
    return edges

# Interactive version to test different thresholds
def interactive_canny_test(image):
    """
    Test different Canny thresholds interactively
    """
    thresholds = [
        (30, 100),   # Low threshold - more edges
        (50, 150),   # Medium threshold - balanced
        (100, 200),  # High threshold - fewer edges
        (30, 200),   # Wide range
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for i, (low, high) in enumerate(thresholds):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, low, high)
        
        axes[i].imshow(edges, cmap='gray')
        axes[i].set_title(f'Threshold: {low}-{high}\nEdges: {np.sum(edges>0)} pixels')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Overlay edges on original image
def show_edges_overlay(image, low_threshold=50, high_threshold=150):
    """
    Show edges overlaid on the original image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    
    # Create red edges overlay
    overlay = image.copy()
    overlay[edges > 0] = [255, 0, 0]  # Red edges
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original with edge overlay
    axes[0].imshow(overlay)
    axes[0].set_title(f'Edges Overlay (Red)\nThreshold: {low_threshold}-{high_threshold}')
    axes[0].axis('off')
    
    # Just edges
    axes[1].imshow(edges, cmap='gray')
    axes[1].set_title('Canny Edges Only')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return edges










def edge_preserving_preprocess(image):
    """
    Specifically designed for your use case: enhance edges, blur interiors
    """
    # Step 1: Strong bilateral filter to blur interiors while preserving edges
    smoothed = cv2.bilateralFilter(image, 5, 25, 50)
    
    # Step 2: Edge enhancement using Laplacian
    gray = cv2.cvtColor(smoothed, cv2.COLOR_RGB2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpened_gray = gray - 1 * laplacian
    sharpened_gray = np.clip(sharpened_gray, 0, 255).astype(np.uint8)
    
    # Step 3: Merge enhanced edges back with smoothed color
    lab = cv2.cvtColor(smoothed, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    enhanced_l = cv2.addWeighted(l, 0.5, sharpened_gray, 0.5, 0)
    final_lab = cv2.merge([enhanced_l, a, b])
    final = cv2.cvtColor(final_lab, cv2.COLOR_LAB2RGB)
    
    return final




image_name = 'images/Hawaii.jpg'
image = cv2.imread(image_name)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image_rgb=edge_preserving_preprocess(image_rgb)


# Basic Canny edges
#edges = show_canny_edges(image_rgb)

# Test different thresholds
interactive_canny_test(image_rgb)



def get_canny_edges(image, low_threshold=100, high_threshold=200):
    """
    Get Canny edges for specific thresholds
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return edges

# Usage:
edges_100_200 = get_canny_edges(image_rgb, 100, 200)
# And if you want to see it:
plt.imshow(edges_100_200, cmap='gray')
plt.title(f'Canny Edges (100-200) - {np.sum(edges_100_200>0)} pixels')
plt.axis('off')
plt.show()







def smooth_edges_distance(edges, distance_threshold=2):
    """
    Use distance transform to create smoother, thicker edges
    """
    # Compute distance to nearest edge
    dist_transform = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 5)
    
    # Keep pixels close to original edges
    smoothed = np.zeros_like(edges)
    smoothed[dist_transform <= distance_threshold] = 255
    
    return smoothed

# Usage
cleaned_edges = smooth_edges_distance(edges_100_200)
# And if you want to see it:
plt.imshow(cleaned_edges, cmap='gray')
plt.title(f'Cleaned Edges (100-200) - {np.sum(cleaned_edges>0)} pixels')
plt.axis('off')
plt.show()








# Edges overlaid on original
#edges = show_edges_overlay(image_rgb, low_threshold=50, high_threshold=150)

# Custom thresholds
#edges = show_canny_edges(image_rgb, low_threshold=30, high_threshold=100)