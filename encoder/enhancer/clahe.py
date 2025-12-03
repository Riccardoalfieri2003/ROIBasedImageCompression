import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage



def get_enhanced_image(image_rgb, shadow_threshold=100, clip_limit=3.0, tile_size=16):
    """
        Function to get the enhanced image
    """

    """
    Create Shadow Mask:
        Converts RGB to LAB color space (L: lightness, A: green-red, B: blue-yellow)
        lab_orig[:, :, 0] gets the L (lightness) channel
        Creates a boolean mask where pixels with L < 100 are considered "shadows"
        This mask identifies dark regions in the image
    """
    lab_orig = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    custom_shadow_mask = lab_orig[:, :, 0] < shadow_threshold 

    # Apply CLAHE with custom shadows
    enhanced_rgb = clahe_custom_shadows(image_rgb, custom_shadow_mask, clip_limit=clip_limit, tile_size=tile_size, debug=True)

    return enhanced_rgb

def clahe_custom_shadows(image_rgb, custom_shadow_mask, clip_limit=4.0, tile_size=4, debug=False):
    """
    Apply CLAHE ONLY to regions specified by custom shadow mask.
    
    Args:
        image_rgb: Input RGB image
        custom_shadow_mask: Your manually defined shadow mask (boolean array)
        clip_limit: CLAHE contrast limit
        tile_size: CLAHE tile size
        debug: If True, print debug information
    
    Returns:
        enhanced_rgb: Image with enhanced shadow regions
    """
    if debug:
        print("=" * 50)
        print("CLAHE CUSTOM SHADOWS DEBUG INFO")
        print("=" * 50)
    
    # Convert to LAB color space
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    if debug:
        print(f"Input image shape: {image_rgb.shape}")
        print(f"L channel range: [{l.min():.1f}, {l.max():.1f}]")
        print(f"Total pixels: {l.size:,}")
        print(f"Shadow mask pixels (L < threshold): {np.sum(custom_shadow_mask):,}")
        print(f"Shadow percentage: {np.sum(custom_shadow_mask)/l.size*100:.1f}%")
        print(f"CLAHE params - Clip limit: {clip_limit}, Tile size: {tile_size}x{tile_size}")
    
    if np.sum(custom_shadow_mask) > 0:
        # Extract shadow pixels using your custom mask
        shadow_pixels = l[custom_shadow_mask]
        
        if debug:
            print(f"\nShadow region L channel range: [{shadow_pixels.min():.1f}, {shadow_pixels.max():.1f}]")
            print(f"Mean L in shadows before: {shadow_pixels.mean():.1f}")
        
        # Apply CLAHE only to shadow regions
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        
        # Reshape for CLAHE and apply
        shadow_pixels_2d = shadow_pixels.reshape(-1, 1)
        enhanced_shadows_2d = clahe.apply(shadow_pixels_2d)
        enhanced_shadows = enhanced_shadows_2d.flatten()
        
        if debug:
            print(f"Mean L in shadows after CLAHE: {enhanced_shadows.mean():.1f}")
            print(f"L improvement: {enhanced_shadows.mean() - shadow_pixels.mean():+.1f}")
        
        # Put enhanced shadows back
        l_enhanced = l.copy()
        l_enhanced[custom_shadow_mask] = enhanced_shadows
        
        if debug:
            print(f"\nApplied CLAHE to {len(enhanced_shadows):,} shadow pixels")
    else:
        l_enhanced = l.copy()
        if debug: 
            print("No shadow regions detected in custom mask!")
            print("Returning original image (no enhancement applied)")
    
    # Convert back to RGB
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced_rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    if debug:
        print("=" * 50)
        print("Processing complete")
        print("=" * 50)
    
    return enhanced_rgb














def aggressive_clahe_custom_shadows(image_rgb, custom_shadow_mask, clip_limit=8.0, tile_size=4, min_brightness=80):
    """
    Aggressive CLAHE using your custom shadow mask with minimum brightness guarantee.
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    print(f"Your shadow regions: {np.sum(custom_shadow_mask)} pixels")
    
    if np.sum(custom_shadow_mask) > 0:
        # Extract your shadow regions
        shadow_pixels = l[custom_shadow_mask]
        
        # Apply aggressive CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        shadow_pixels_2d = shadow_pixels.reshape(-1, 1)
        enhanced_shadows = clahe.apply(shadow_pixels_2d).flatten()
        
        # Force minimum brightness in shadows
        too_dark = enhanced_shadows < min_brightness
        enhanced_shadows[too_dark] = min_brightness
        
        # Put back
        l_enhanced = l.copy()
        l_enhanced[custom_shadow_mask] = enhanced_shadows
        
        print(f"Shadow brightness - Before: {shadow_pixels.mean():.1f}, After: {enhanced_shadows.mean():.1f}")
    else:
        l_enhanced = l.copy()
    
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced_rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    return enhanced_rgb

def clahe_custom_regions(image_rgb, custom_shadow_mask, clip_limit=6.0):
    """
    Process each connected region in your custom mask individually.
    """
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Find connected components in YOUR custom mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        custom_shadow_mask.astype(np.uint8), connectivity=8
    )
    
    l_enhanced = l.copy()
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(4, 4))
    
    print(f"Processing {num_labels-1} shadow regions from your mask")
    
    for i in range(1, num_labels):  # Skip background
        region_mask = labels == i
        region_pixels = l[region_mask]
        
        if len(region_pixels) > 10:  # Only process substantial regions
            # Apply CLAHE to this specific region
            region_pixels_2d = region_pixels.reshape(-1, 1)
            enhanced_region = clahe.apply(region_pixels_2d).flatten()
            
            l_enhanced[region_mask] = enhanced_region
            
            print(f"Region {i}: {len(region_pixels)} pixels, "
                  f"brightness {region_pixels.mean():.1f} -> {enhanced_region.mean():.1f}")
    
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced_rgb = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    return enhanced_rgb


def demonstrate_custom_shadow_enhancement(image_rgb):
    """
    Demonstrate using your custom shadow mask with CLAHE.
    """
    # Step 1: Create your custom shadow mask (as you described)
    lab_orig = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    custom_shadow_mask = lab_orig[:, :, 0] < 100  # YOUR manual threshold
    
    print(f"Your shadow mask covers {np.sum(custom_shadow_mask)} pixels "
          f"({np.sum(custom_shadow_mask)/custom_shadow_mask.size*100:.1f}% of image)")
    
    # Step 2: Apply different CLAHE methods using YOUR mask
    methods = [
        ('Standard CLAHE',  lambda: clahe_custom_shadows(image_rgb, custom_shadow_mask, clip_limit=4.0)),
        
        #('Aggressive CLAHE', lambda: aggressive_clahe_custom_shadows(image_rgb, custom_shadow_mask, clip_limit=8.0, min_brightness=90)),
        
        #('Region-based CLAHE', lambda: clahe_custom_regions(image_rgb, custom_shadow_mask, clip_limit=6.0)),
    ]
    
    # Visualize results
    fig, axes = plt.subplots(2, len(methods) + 1, figsize=(18, 8))
    
    # Original image and your mask
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(custom_shadow_mask, cmap='gray')
    axes[1, 0].set_title('Your Custom Shadow Mask')
    axes[1, 0].axis('off')
    
    for col, (name, method) in enumerate(methods, 1):
        enhanced_rgb = method()
        
        # Enhanced image
        axes[0, col].imshow(enhanced_rgb)
        axes[0, col].set_title(name)
        axes[0, 0].axis('off')
        
        # Brightness improvement in your shadow regions
        lab_enh = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2LAB)[:, :, 0]
        brightness_diff = np.zeros_like(lab_orig[:, :, 0], dtype=np.float32)
        brightness_diff[custom_shadow_mask] = lab_enh[custom_shadow_mask] - lab_orig[:, :, 0][custom_shadow_mask]
        
        im = axes[1, col].imshow(brightness_diff, cmap='RdYlBu', vmin=0, vmax=80)
        axes[1, col].set_title('Brightening in Your Regions')
        axes[1, col].axis('off')
        plt.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)
        
        # Statistics for your specific regions
        original_brightness = lab_orig[:, :, 0][custom_shadow_mask].mean()
        enhanced_brightness = lab_enh[custom_shadow_mask].mean()
        improvement = enhanced_brightness - original_brightness
        
        print(f"{name}: {original_brightness:.1f} → {enhanced_brightness:.1f} "
              f"(+{improvement:.1f} levels)")
    
    plt.tight_layout()
    plt.show()
    
    return enhanced_rgb, custom_shadow_mask








def test_clahe_parameters(image_rgb, shadow_mask):
    """
    Test different CLAHE parameter combinations.
    """
    combinations = [
        ('Conservative', {'clip_limit': 2.0, 'tile_size': 8}),
        ('Balanced', {'clip_limit': 4.0, 'tile_size': 8}),
        ('Aggressive', {'clip_limit': 8.0, 'tile_size': 4}),
        ('Fine Detail', {'clip_limit': 6.0, 'tile_size': 4}),
        ('Smooth', {'clip_limit': 3.0, 'tile_size': 16}),
        ('Personal', {'clip_limit': 3.0, 'tile_size': 16})
    ]
    
    fig, axes = plt.subplots(2, len(combinations), figsize=(20, 8))
    
    for col, (name, params) in enumerate(combinations):
        enhanced = clahe_custom_shadows(image_rgb, shadow_mask, **params)
        
        # Show enhanced image
        axes[0, col].imshow(enhanced)
        axes[0, col].set_title(f'{name}\nClip: {params["clip_limit"]}, Tile: {params["tile_size"]}')
        axes[0, col].axis('off')
        
        # Show difference from original (only in shadow regions)
        lab_orig = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)[:, :, 0]
        lab_enh = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)[:, :, 0]
        
        diff = np.zeros_like(lab_orig, dtype=np.float32)
        diff[shadow_mask] = lab_enh[shadow_mask] - lab_orig[shadow_mask]
        
        im = axes[1, col].imshow(diff, cmap='RdYlBu', vmin=0, vmax=80)
        axes[1, col].set_title(f'Brightening')
        axes[1, col].axis('off')
        plt.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()





