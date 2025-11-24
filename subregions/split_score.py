import numpy as np
from skimage import filters, feature
from scipy import ndimage as ndi
from skimage.color import rgb2lab, rgb2gray
import math

"""
def calculate_split_score(region_image, color_weight=0.6, texture_weight=0.4):

    
    # Ensure we're working with a valid region (not all zeros)
    if np.all(region_image == 0):
        return 0.0, 0.0, 0.0
    
    # Convert to grayscale for texture analysis
    gray_image = rgb2gray(region_image)
    
    # Remove zero regions from analysis
    mask = gray_image > 0.01
    if np.sum(mask) < 100:  # Too small region
        return 0.0, 0.0, 0.0
    
    # 1. COLOR COMPLEXITY SCORE
    def color_complexity_score(image, mask):
        # Convert to LAB color space for better perceptual uniformity
        lab_image = rgb2lab(image)
        
        # Calculate standard deviation in each channel within the mask
        l_std = np.std(lab_image[mask, 0])
        a_std = np.std(lab_image[mask, 1]) 
        b_std = np.std(lab_image[mask, 2])
        
        # Combine channel variances (normalize by typical ranges)
        color_variance = (l_std/100 + a_std/128 + b_std/128) / 3
        
        # Calculate color gradient magnitude
        gradient_magnitude = 0
        for channel in range(3):
            grad_x = filters.sobel(lab_image[:, :, channel])
            grad_y = filters.sobel(lab_image[:, :, channel])
            gradient_magnitude += np.sqrt(grad_x**2 + grad_y**2)
        
        gradient_score = np.mean(gradient_magnitude[mask]) / 3
        
        # Combine variance and gradient scores
        color_score = 0.7 * color_variance + 0.3 * gradient_score
        return np.clip(color_score, 0, 1)
    
    # 2. TEXTURE COMPLEXITY SCORE
    def texture_complexity_score(gray_image, mask):
        texture_scores = []
        
        # Method 1: Local Binary Patterns (LBP)
        lbp = feature.local_binary_pattern(gray_image, 8, 1, method='uniform')
        lbp_hist, _ = np.histogram(lbp[mask].flatten(), bins=10, density=True)
        lbp_score = 1 - np.max(lbp_hist)  # More uniform = lower score
        
        # Method 2: Gabor filter responses
        gabor_responses = []
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            gabor_real, gabor_imag = filters.gabor(gray_image, frequency=0.1, theta=theta)
            gabor_magnitude = np.sqrt(gabor_real**2 + gabor_imag**2)
            gabor_responses.append(gabor_magnitude[mask])
        
        gabor_variance = np.mean([np.std(resp) for resp in gabor_responses])
        gabor_score = np.clip(gabor_variance * 10, 0, 1)
        
        # Method 3: Entropy of gradients
        grad_magnitude = filters.sobel(gray_image)
        entropy_score = calculate_entropy(grad_magnitude[mask])
        
        # Combine texture measures
        texture_score = (lbp_score + gabor_score + entropy_score) / 3
        return np.clip(texture_score, 0, 1)
    
    def calculate_entropy(data):
        hist, _ = np.histogram(data, bins=32, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        return -np.sum(hist * np.log2(hist))
    
    def normalize_result(score, window_size):
        return window_size/( 1+ math.exp(-12*(score-0.5)) ) 
    
    window_size = math.ceil(math.log(region_image.size, 10)) * math.log(region_image.size)
    print(f"WindowSize: {window_size}")
    
    # Calculate individual scores
    color_score =  normalize_result( color_complexity_score(region_image, mask), window_size)
    texture_score = normalize_result( texture_complexity_score(gray_image, mask), window_size)
    
    # Combine with weights
    overall_score = (color_weight * color_score + texture_weight * texture_score)
    
    return overall_score, color_score, texture_score
"""


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
        # Only analyze texture in the masked region
        texture_scores = []
        
        # Method 1: Local Binary Patterns (LBP) on masked region
        lbp = feature.local_binary_pattern(gray_image, 8, 1, method='uniform')
        lbp_masked = lbp[mask]
        if len(lbp_masked) > 0:
            lbp_hist, _ = np.histogram(lbp_masked.flatten(), bins=10, density=True)
            lbp_score = 1 - np.max(lbp_hist) if len(lbp_hist) > 0 else 0
        else:
            lbp_score = 0
        
        # Method 2: Gradient magnitude in masked region
        grad_magnitude = filters.sobel(gray_image)
        entropy_score = calculate_entropy(grad_magnitude[mask]) if np.sum(mask) > 0 else 0
        
        # Combine texture measures
        texture_score = (lbp_score + entropy_score) / 2
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
    overall_score = (0.6 * color_score + 0.4 * texture_score)
    
    return overall_score, color_score, texture_score