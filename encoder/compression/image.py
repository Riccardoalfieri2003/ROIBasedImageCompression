import numpy as np
from encoder.compression.clustering import compute_clustering_params,cluster_palette_colors_parallel
from encoder.compression.compression import optimize_compressed_dtype, print_compressed_data_types
from encoder.compression.merging import merge_region_components_simple

from decoder.uncompression.uncompression import lossless_decompress, load_compressed, decompress_color_quantization, partial_decompress_color_quantization

def quantize_image(image_components, original_image_height, original_image_width, quality=100):

    # First, collect all Regions components
    all_image_components_flat = [regions for regions in image_components]

    # Use the entire image as the bbox
    image_bbox = (0, 0, original_image_height, original_image_width)

    regions_image = merge_region_components_simple(
        all_image_components_flat,
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
    image_seg_compression = cluster_palette_colors_parallel(
        quality,
        merged_segment,  # Pass the dictionary
        eps=eps,
        min_samples=1,
        max_colors_per_cluster=max_colors_per_cluster
    )

    # Print current types
    print_compressed_data_types(image_seg_compression, "BEFORE OPTIMIZATION")

    # Optimize dtype
    image_seg_compression = optimize_compressed_dtype(image_seg_compression)

    # Print optimized types
    print_compressed_data_types(image_seg_compression, "AFTER OPTIMIZATION")



    


    show_reconstruction_result=True
    if show_reconstruction_result:
        # ==============================================
        # 3. RECONSTRUCT 
        # ==============================================
        reconstruction_result = partial_decompress_color_quantization(image_seg_compression)
        
        # ==============================================
        # 4. SIMPLE VISUALIZATION
        # ==============================================
        print(f"\n{'='*60}")
        print(f"Regions RECONSTRUCTION")
        print(f"{'='*60}")
        
        # Get basic info
        h, w = reconstruction_result['shape']
        top_left = reconstruction_result['top_left']
        n_colors = image_seg_compression['compressed_colors']
        psnr = image_seg_compression.get('psnr', 0)
        
        # Get reconstructed image
        reconstructed_img = reconstruction_result['image']
        
        # Simple display - just the reconstruction
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # 1. Reconstructed Regions
        axes[0].imshow(reconstructed_img)
        axes[0].set_title(f'Reconstructed Regions\n{h}x{w} pixels\n{n_colors} colors\nPSNR: {psnr:.1f} dB')
        axes[0].axis('off')
        
        # 2. Colored pixels only (white background)
        colored_only = reconstructed_img.copy()
        black_mask = np.all(reconstructed_img == [0, 0, 0], axis=2)
        colored_only[black_mask] = [255, 255, 255]  # White background for black areas
        
        axes[1].imshow(colored_only)
        axes[1].set_title(f'Colored Pixels Only\nBlack pixels: {np.sum(black_mask):,}\n({np.sum(black_mask)/(h*w)*100:.1f}%)')
        axes[1].axis('off')
        
        plt.suptitle(f'Regions at position {top_left}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Console summary
        print(f"Position: {top_left}")
        print(f"Size: {h}x{w} = {h*w:,} pixels")
        print(f"Colors: {n_colors}")
        print(f"Black pixels: {np.sum(black_mask):,} ({np.sum(black_mask)/(h*w)*100:.1f}%)")
        print(f"Colored pixels: {h*w - np.sum(black_mask):,} ({(h*w - np.sum(black_mask))/(h*w)*100:.1f}%)")
        print(f"PSNR: {psnr:.1f} dB")

    return image_seg_compression