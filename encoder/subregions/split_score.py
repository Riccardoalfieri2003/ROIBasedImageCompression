import numpy as np
from skimage import filters, feature
from scipy import ndimage as ndi
from skimage.color import rgb2lab, rgb2gray
import math


def calculate_optimal_segments(split_score, region_area, min_segments=5, max_segments=50):
    base_segments = min_segments + (max_segments - min_segments) * split_score
    area_factor = np.sqrt(region_area / 1000)  # Normalize by area
    optimal_segments = int(base_segments * area_factor)
    
    return np.clip(optimal_segments, min_segments, max_segments)

def calculate_split_score(region_image, mask=None):
    """
    Calculate split score based on color and texture complexity
    If mask is provided, only analyze pixels where mask is True
    """
    # If no mask provided, create one that excludes black regions
    if mask is None:
        gray_image = rgb2gray(region_image)
        mask = gray_image > 0.01
    
    # Ensure we're working with a valid region
    if np.sum(mask) < 100:  # Too small region
        return 0.0, 0.0, 0.0
    
    # Convert to grayscale for texture analysis
    gray_image = rgb2gray(region_image)
    
    # 1. COLOR COMPLEXITY SCORE (only on masked region)
    def color_complexity_score(image, mask):
        # Convert to LAB color space
        lab_image = rgb2lab(image)
        
        # Calculate standard deviation in each channel within the mask
        l_std = np.std(lab_image[mask, 0])
        a_std = np.std(lab_image[mask, 1]) 
        b_std = np.std(lab_image[mask, 2])
        
        # Combine channel variances
        color_variance = (l_std/100 + a_std/128 + b_std/128) / 3
        
        # Calculate color gradient magnitude only in masked area
        gradient_magnitude = 0
        for channel in range(3):
            grad_x = filters.sobel(lab_image[:, :, channel])
            grad_y = filters.sobel(lab_image[:, :, channel])
            # Only consider gradients in the masked region
            gradient_magnitude += np.sqrt(grad_x**2 + grad_y**2)
        
        gradient_score = np.mean(gradient_magnitude[mask]) / 3
        
        # Combine variance and gradient scores
        color_score = 0.7 * color_variance + 0.3 * gradient_score
        return np.clip(color_score, 0, 1)
    
    # 2. TEXTURE COMPLEXITY SCORE (only on masked region)
    def texture_complexity_score(gray_image, mask):
        """
        Calculate texture complexity score only on masked region
        """
        # Ensure we have enough pixels to analyze
        if np.sum(mask) < 100:
            return 0.0
        
        # Extract only the masked region
        masked_gray = gray_image[mask]
        
        texture_scores = []
        
        # Method 1: Local Binary Patterns (LBP) - FIXED
        try:
            # Apply LBP to entire image first, then mask
            lbp = feature.local_binary_pattern(gray_image, 8, 1, method='uniform')
            lbp_masked = lbp[mask]
            
            if len(lbp_masked) > 10:  # Need enough samples
                # Calculate histogram of LBP values in masked region
                lbp_hist, _ = np.histogram(lbp_masked, bins=10, range=(0, 10), density=True)
                # More uniform distribution = higher complexity = higher score
                lbp_entropy = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-8))  # Add small epsilon to avoid log(0)
                lbp_score = np.clip(lbp_entropy / 3.0, 0, 1)  # Normalize by max possible entropy
            else:
                lbp_score = 0
        except:
            lbp_score = 0
        
        # Method 2: Gradient-based texture - FIXED
        try:
            grad_magnitude = filters.sobel(gray_image)
            grad_masked = grad_magnitude[mask]
            
            if len(grad_masked) > 10:
                # Use variance of gradients as texture measure
                grad_variance = np.var(grad_masked)
                grad_score = np.clip(grad_variance * 50, 0, 1)  # Scale appropriately
            else:
                grad_score = 0
        except:
            grad_score = 0
        
        # Method 3: Entropy of intensities - NEW
        try:
            if len(masked_gray) > 10:
                hist, _ = np.histogram(masked_gray, bins=32, range=(0, 1), density=True)
                intensity_entropy = -np.sum(hist * np.log2(hist + 1e-8))
                entropy_score = np.clip(intensity_entropy / 5.0, 0, 1)  # Normalize
            else:
                entropy_score = 0
        except:
            entropy_score = 0
        
        # Method 4: Standard deviation of intensities - NEW
        try:
            std_score = np.clip(np.std(masked_gray) * 2, 0, 1)
        except:
            std_score = 0
        
        # Combine all texture measures
        texture_score = (lbp_score + grad_score + entropy_score + std_score) / 4
        
        return np.clip(texture_score, 0, 1)


    def calculate_entropy(data):
        """Calculate entropy of data"""
        if len(data) == 0:
            return 0
        hist, _ = np.histogram(data, bins=32, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        return -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0
    
    # Calculate individual scores (only on masked region)
    color_score = color_complexity_score(region_image, mask)
    texture_score = texture_complexity_score(gray_image, mask)
    
    # Combine with weights
    overall_score = (0.4 * color_score + 0.6 * texture_score)
    
    return overall_score, color_score, texture_score

def normalize_result(score, window_size):
        return window_size/( 1+ math.exp(-12*(score-0.5)) ) 