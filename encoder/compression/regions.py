import math
import numpy as np

from encoder.compression.clustering import compute_clustering_params,cluster_palette_colors_parallel
from encoder.compression.merging import merge_region_components_simple

from decoder.uncompression.uncompression import lossless_decompress, load_compressed, decompress_color_quantization, partial_decompress_color_quantization

def region_quantization(regions_components, original_image_height, original_image_width, quality=50):

    image_components=[]

    # First, collect all regions components
    try: all_regions_components_flat = [regions[0] for regions in regions_components]
    except: all_regions_components_flat = [regions for regions in regions_components]

    # More robust flattening
    all_regions_components_flat = []
    for regions in regions_components:
        if isinstance(regions, dict):
            all_regions_components_flat.append(regions)
        elif isinstance(regions, list):
            for item in regions:
                if isinstance(item, dict):
                    all_regions_components_flat.append(item)
                else:
                    print(f"Warning: Skipping non-dict item in list: {type(item)}")
        else:
            print(f"Warning: Skipping unexpected type: {type(regions)}")

    print(f"Processed {len(regions_components)} inputs → {len(all_regions_components_flat)} segments")

    # Use the entire image as the bbox
    image_bbox = (0, 0, original_image_height, original_image_width)

    regions_image = merge_region_components_simple(
        all_regions_components_flat,
        roi_bbox=image_bbox
    )


    # Extract the merged segment dictionary
    merged_segment = regions_image[0]  # This contains palette and indices, NOT the image!

    n_colors = merged_segment['actual_colors']


    n_colors = merged_segment['actual_colors']
    palette_colors = np.array(merged_segment['palette'])
    
    # Option A: Simple improved formulas
    eps, min_samples, max_colors_per_cluster = compute_clustering_params(
        n_colors, quality, color_space='lab'
    )


    # ==============================================
    # OPTION 1: If you want to cluster the merged palette
    # ==============================================
    # You already have the merged palette in merged_segment
    # Just cluster it directly:
    regions_seg_compression = cluster_palette_colors_parallel(
        quality,
        merged_segment,  # Pass the dictionary
        eps=eps,
        min_samples=1,
        max_colors_per_cluster=max_colors_per_cluster
    )

    image_components.append(regions_seg_compression)

    


    show_reconstruction_result=True
    if show_reconstruction_result:
        # ==============================================
        # 3. RECONSTRUCT 
        # ==============================================
        reconstruction_result = partial_decompress_color_quantization(regions_seg_compression)
        
        # ==============================================
        # 4. SIMPLE VISUALIZATION
        # ==============================================
        print(f"\n{'='*60}")
        print(f"regions RECONSTRUCTION")
        print(f"{'='*60}")
        
        # Get basic info
        h, w = reconstruction_result['shape']
        top_left = reconstruction_result['top_left']
        n_colors = regions_seg_compression['compressed_colors']
        psnr = regions_seg_compression.get('psnr', 0)
        
        # Get reconstructed image
        reconstructed_img = reconstruction_result['image']
        
        # Simple display - just the reconstruction
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # 1. Reconstructed regions
        axes[0].imshow(reconstructed_img)
        axes[0].set_title(f'Reconstructed regions\n{h}x{w} pixels\n{n_colors} colors\nPSNR: {psnr:.1f} dB')
        axes[0].axis('off')
        
        # 2. Colored pixels only (white background)
        colored_only = reconstructed_img.copy()
        black_mask = np.all(reconstructed_img == [0, 0, 0], axis=2)
        colored_only[black_mask] = [255, 255, 255]  # White background for black areas
        
        axes[1].imshow(colored_only)
        axes[1].set_title(f'Colored Pixels Only\nBlack pixels: {np.sum(black_mask):,}\n({np.sum(black_mask)/(h*w)*100:.1f}%)')
        axes[1].axis('off')
        
        plt.suptitle(f'regions at position {top_left}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Console summary
        print(f"Position: {top_left}")
        print(f"Size: {h}x{w} = {h*w:,} pixels")
        print(f"Colors: {n_colors}")
        print(f"Black pixels: {np.sum(black_mask):,} ({np.sum(black_mask)/(h*w)*100:.1f}%)")
        print(f"Colored pixels: {h*w - np.sum(black_mask):,} ({(h*w - np.sum(black_mask))/(h*w)*100:.1f}%)")
        print(f"PSNR: {psnr:.1f} dB")

    return image_components