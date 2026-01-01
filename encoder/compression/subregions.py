import math
import numpy as np

from encoder.subregions.split_score import calculate_split_score, normalize_result
from encoder.subregions.slic import enhanced_slic_with_texture, extract_slic_segment_boundaries, visualize_split_analysis
from encoder.compression.clustering import compute_clustering_params,cluster_palette_colors_parallel, get_all_unique_colors
from encoder.compression.merging import merge_region_components_simple

from decoder.uncompression.uncompression import decompress_color_quantization

def subregion_quantization(image_rgb, subregions, quality=10, subregion_type=None, debug=False):

    # ==============================================
    # MAIN PROCESSING LOOP - One Sub Region at a time
    # ==============================================

    subregions_components=[]

    for i, region in enumerate(subregions):

        region_components=[]


        if debug:
            print(f"\n{'='*60}")
            if subregion_type!=None: print(f"PROCESSING {subregion_type} subregion {i+1}/{len(subregions)}")
            else: print(f"PROCESSING subregion {i+1}/{len(subregions)}")
            print(f"{'='*60}")
        
        # ==============================================
        # 1. EXTRACT THE subregions REGION
        # ==============================================
        minr, minc, maxr, maxc = region['bbox']
        bbox_region = image_rgb[minr:maxr, minc:maxc]
        bbox_mask = region['bbox_mask']  # Irregular region mask
        region_image = bbox_region
        
        if debug:
            print(f"Region {i+1}: {bbox_region.shape[1]}x{bbox_region.shape[0]} pixels")
            print(f"Mask area: {np.sum(bbox_mask):,} pixels")

        
        # ==============================================
        # 2. ANALYZE REGION FOR SEGMENTATION
        # ==============================================
        overall_score, color_score, texture_score = calculate_split_score(bbox_region, bbox_mask)
        
        if debug:
            print(f"  Analysis scores:")
            print(f"    Overall: {overall_score:.3f}")
            print(f"    Color: {color_score:.3f}")
            print(f"    Texture: {texture_score:.3f}")
        
        window = math.ceil(math.ceil(math.log(bbox_region.size, 10)) * math.log(bbox_region.size))
        normalized_overall_score = normalize_result(overall_score, window)
        optimal_segments = math.ceil(normalized_overall_score)
        
        if optimal_segments <= 0: 
            optimal_segments = 1
        
        if debug:
            print(f"  Optimal segments: {optimal_segments}")
            
            # Visualize analysis (optional)
            visualize_split_analysis(
                region_image=region_image,
                overall_score=overall_score,
                color_score=color_score, 
                texture_score=texture_score,
                optimal_segments=optimal_segments
            )
        


        
        # ==============================================
        # 3. APPLY SLIC SEGMENTATION TO THE subregions
        # ==============================================
        if debug: print(f"\n  Applying SLIC segmentation...")
        if optimal_segments<1: optimal_segments=1
        subregions_segments, texture_map = enhanced_slic_with_texture(bbox_region, n_segments=optimal_segments)
        segment_boundaries = extract_slic_segment_boundaries(subregions_segments, bbox_mask)
        
        if debug:
            print(f"  Found {len(segment_boundaries)} sub-regions")
            print(f"  Total boundary points: {sum(seg['num_points'] for seg in segment_boundaries):,}")
        




        for seg_idx, seg_data in enumerate(segment_boundaries):
            segment_id = seg_data.get('segment_id', seg_idx)
            segment_mask = (subregions_segments == segment_id) & bbox_mask
            
            segment_pixels = np.sum(segment_mask)
            if segment_pixels < 64:
                continue
            
            if debug:
                print(f"\n    Segment {seg_idx+1}/{len(segment_boundaries)} (ID: {segment_id}):")
                print(f"      Pixels: {segment_pixels:,}")
            
            # ==============================================
            # 1. CREATE ISOLATED SEGMENT IMAGE
            # ==============================================
            # Create image with only this segment visible
            segment_image_full = region_image.copy()
            segment_image_full[bbox_mask & ~segment_mask] = [0, 0, 0]  # Other segments in bbox -> black
            segment_image_full[~bbox_mask] = [0, 0, 0]  # Outside bbox -> black
            
            # ==============================================
            # 2. FIND TIGHT BOUNDING BOX (CROP TO NON-BLACK PIXELS)
            # ==============================================
            # Find coordinates of non-black pixels
            # Non-black = at least one channel is > 0
            non_black_mask = np.any(segment_image_full > 0, axis=2)
            
            if np.sum(non_black_mask) == 0:
                print(f"      Warning: Segment has no non-black pixels!")
                continue
            
            # Get row and column indices of non-black pixels
            rows, cols = np.where(non_black_mask)
            
            # Find bounding box coordinates
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()
            
            # Add small padding (optional, e.g., 2 pixels)
            pad = 0
            h, w = segment_image_full.shape[:2]
            min_row = max(0, min_row - pad) 
            max_row = min(h - 1, max_row + pad) 
            min_col = max(0, min_col - pad)
            max_col = min(w - 1, max_col + pad) 
            
            # Calculate dimensions of cropped region
            crop_height = max_row - min_row + 1
            crop_width = max_col - min_col + 1
            
            # ==============================================
            # 3. CROP THE IMAGE TO TIGHT BOUNDING BOX
            # ==============================================
            segment_image_cropped = segment_image_full[min_row:max_row+1, min_col:max_col+1]

            
            
            # ==============================================
            # 4. CALCULATE ORIGINAL COORDINATES
            # ==============================================            
            absolute_min_row = min_row + minr
            absolute_min_col = min_col + minc 
            absolute_max_row = max_row + minr 
            absolute_max_col = max_col + minc
            
            # The top-left corner (left-northernmost point) coordinates in full image:
            top_left_abs_row = absolute_min_row
            top_left_abs_col = absolute_min_col
            
            if debug:
                print(f"      Original segment shape: {segment_image_full.shape[:2]}")
                print(f"      Cropped segment shape: {segment_image_cropped.shape[:2]}")
                print(f"      Crop reduction: {(1 - (crop_height*crop_width)/(h*w))*100:.1f}%")
                print(f"      Bbox in region: ({min_row}, {min_col}) to ({max_row}, {max_col})")
                print(f"      Bbox in full image: ({absolute_min_row}, {absolute_min_col}) to ({absolute_max_row}, {absolute_max_col})")
                print(f"      Top-left in full image: ({top_left_abs_row}, {top_left_abs_col})")
            
                print(f"segment_image_cropped shape: {segment_image_cropped.shape}")


            # ==============================================
            # 2. COMPRESS CROPPED SEGMENT
            # ==============================================
            seg_compression = get_all_unique_colors(
                segment_image_cropped,
                (top_left_abs_row, top_left_abs_col)
            )


            n_colors = seg_compression['actual_colors']

            palette_colors = np.array(seg_compression['palette'])
            
            # Option A: Simple improved formulas
            eps, min_samples, max_colors_per_cluster = compute_clustering_params(
                n_colors, quality, color_space='lab'
            )

            

            # Then cluster the colors
            seg_compression = cluster_palette_colors_parallel(
                quality,
                seg_compression,
                eps=eps,           # Distance threshold (0-255 scale)
                min_samples=1,      # Min colors to form cluster
                max_colors_per_cluster=max_colors_per_cluster  # Split large clusters
            )

            if debug:
                print(f"Eps: {eps}")
                print(f"n_colors: {n_colors}")
                print(f"max_colors_per_cluster: {max_colors_per_cluster}")

            
                        
            
            show_reconstruction_result=False
            if show_reconstruction_result:
                # ==============================================
                # 3. RECONSTRUCT 
                # ==============================================
                reconstruction_result = decompress_color_quantization(
                    seg_compression,
                )

                

                # ==============================================
                # 4. VISUALIZE COMPARISON WITH STATS
                # ==============================================
                print(f"\n{'='*60}")
                print(f"SEGMENT {seg_idx+1} - COLOR QUANTIZATION RESULTS")
                print(f"{'='*60}")

                # Get stats from compression
                h, w, _ = segment_image_cropped.shape
                original_size = h * w * 3
                compressed_size = seg_compression['compressed_size']
                compression_ratio = seg_compression['compression_ratio']
                psnr = seg_compression['psnr']
                n_colors = seg_compression['compressed_colors']

                # Print stats in a clean format
                print(f"Segment {seg_idx+1} Summary:")
                print(f"  Shape: {h}x{w}")
                print(f"  Top-left: {top_left_abs_row, top_left_abs_col}")
                print(f"  Colors in palette: {n_colors}")
                print(f"  Original size: {original_size:,} bytes")
                print(f"  Compressed size: {compressed_size:,} bytes")
                print(f"  Compression ratio: {compression_ratio:.2f}:1")
                print(f"  Space savings: {(1 - compressed_size/original_size)*100:.1f}%")
                print(f"  Quality (PSNR): {psnr:.2f} dB")

                # Create visualization figure
                import matplotlib.pyplot as plt

                fig, axes = plt.subplots(2, 3, figsize=(15, 10))

                # Original image
                axes[0, 0].imshow(segment_image_cropped)
                axes[0, 0].set_title(f'Original\n{h}x{w} pixels')
                axes[0, 0].axis('off')

                # Add pixel info
                non_black = np.sum(np.any(segment_image_cropped > 10, axis=2))
                axes[0, 0].text(0.02, 0.98, f'Content: {non_black:,} px', 
                            transform=axes[0, 0].transAxes, fontsize=9, color='white',
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

                # Reconstructed image
                reconstructed_img = reconstruction_result['image']
                axes[0, 1].imshow(reconstructed_img)
                axes[0, 1].set_title(f'Reconstructed\n{n_colors} colors')
                axes[0, 1].axis('off')

                # Difference (amplified for visibility)
                diff = np.abs(segment_image_cropped.astype(float) - reconstructed_img.astype(float))
                diff_display = np.clip(diff * 3, 0, 255).astype(np.uint8)
                axes[0, 2].imshow(diff_display)
                axes[0, 2].set_title(f'Difference (×3)\nPSNR: {psnr:.2f} dB')
                axes[0, 2].axis('off')

                # Add MSE/PSNR to difference plot
                mse = np.mean(diff ** 2)
                axes[0, 2].text(0.02, 0.98, f'MSE: {mse:.2f}', 
                            transform=axes[0, 2].transAxes, fontsize=9, color='white',
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))

                # Color palette visualization
                palette = np.array(seg_compression['palette'])
                axes[1, 0].axis('off')

                # Create palette visualization
                if n_colors > 0:
                    # Create a color swatch
                    palette_h = max(1, min(10, n_colors // 10))
                    palette_w = (n_colors + palette_h - 1) // palette_h
                    
                    palette_img = np.zeros((palette_h * 20, palette_w * 20, 3), dtype=np.uint8)
                    
                    for idx, color in enumerate(palette):
                        row = (idx // palette_w) * 20
                        col = (idx % palette_w) * 20
                        palette_img[row:row+18, col:col+18] = color
                    
                    # Display in inset axes
                    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
                    ax_inset = inset_axes(axes[1, 0], width="60%", height="60%", loc='center')
                    ax_inset.imshow(palette_img)
                    ax_inset.set_title(f'Color Palette ({n_colors} colors)')
                    ax_inset.axis('off')
                    
                    # Add palette stats
                    unique_original = len(np.unique(segment_image_cropped.reshape(-1, 3), axis=0))
                    palette_text = f'Original: {unique_original} colors\nCompressed: {n_colors} colors\nReduction: {(1 - n_colors/unique_original)*100:.1f}%'
                    axes[1, 0].text(0.5, 0.1, palette_text, transform=axes[1, 0].transAxes,
                                fontsize=9, horizontalalignment='center',
                                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

                # Compression statistics visualization
                axes[1, 1].axis('off')

                # Create bar chart for size comparison
                size_data = [original_size, compressed_size]
                size_labels = ['Original', 'Compressed']
                colors = ['skyblue', 'lightcoral']

                # Create inset axes for bar chart
                ax_bar = inset_axes(axes[1, 1], width="60%", height="60%", loc='center')
                bars = ax_bar.bar(size_labels, size_data, color=colors, alpha=0.7)
                ax_bar.set_title('Size Comparison')
                ax_bar.set_ylabel('Bytes')

                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax_bar.text(bar.get_x() + bar.get_width()/2., height + max(size_data)*0.05,
                            f'{height:,}', ha='center', va='bottom', fontsize=9)

                # Add ratio text
                ratio_text = f'Compression Ratio: {compression_ratio:.2f}:1\nSavings: {(1 - compressed_size/original_size)*100:.1f}%'
                axes[1, 1].text(0.5, 0.1, ratio_text, transform=axes[1, 1].transAxes,
                            fontsize=9, horizontalalignment='center',
                            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

                # Index map visualization (shows which palette index each pixel uses)
                axes[1, 2].axis('off')

                # Get index map
                index_map = np.array(seg_compression['indices']).reshape(h, w)

                # Display index map as grayscale
                ax_index = inset_axes(axes[1, 2], width="60%", height="60%", loc='center')
                im = ax_index.imshow(index_map, cmap='viridis', aspect='auto')
                ax_index.set_title('Index Map (Palette Indices)')
                ax_index.axis('off')

                # Add colorbar for index map
                plt.colorbar(im, ax=ax_index, fraction=0.046, pad=0.04)

                # Add index map stats
                index_unique = len(np.unique(index_map))
                axes[1, 2].text(0.5, 0.1, f'Unique indices: {index_unique}/{n_colors}',
                            transform=axes[1, 2].transAxes,
                            fontsize=9, horizontalalignment='center',
                            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

                # Main title
                plt.suptitle(f'Segment {seg_idx+1}: Color Quantization Results\nTop-left: {top_left_abs_row, top_left_abs_col}, Size: {h}x{w}', 
                            fontsize=14, fontweight='bold')

                plt.tight_layout()
                plt.show()

                # ==============================================
                # 5. CONSOLE SUMMARY (CLEAN FORMAT)
                # ==============================================
                print(f"\n📊 COMPRESSION SUMMARY:")
                print(f"   Original:    {original_size:>8,} bytes")
                print(f"   Compressed:  {compressed_size:>8,} bytes")
                print(f"   Ratio:       {compression_ratio:>8.2f}:1")
                print(f"   Savings:     {(1 - compressed_size/original_size)*100:>7.1f}%")
                print(f"   PSNR:        {psnr:>8.2f} dB")
                print(f"   Palette:     {n_colors:>8} colors")
                print(f"{'='*60}")
            
                
            
            

            if seg_compression is None:
                continue
        
            

            #all_segments_compressed.append(seg_compression)
            region_components.append(seg_compression)


        # ==============================================
        # 4. MERGE COMPONENTS WITHIN THIS subregions
        # ==============================================
        if debug:
            print(f"\n{'='*60}")
            print(f"MERGING COMPONENTS FOR subregions {i+1}")
            print(f"{'='*60}")

                
        if len(region_components) > 1:
            # Get subregions bbox
            minr, minc, maxr, maxc = region['bbox']
            subregions_bbox = (minr, minc, maxr, maxc)
            subregions_height = maxr - minr
            subregions_width = maxc - minc
            
            # Choose merging strategy
            # Option 1: Simple merge (colored pixels override black)
            merged_components = merge_region_components_simple(region_components, subregions_bbox)
            subregions_components.append(merged_components)

            #visualize_merged_result(merged_components, (subregions_height, subregions_width), minr, minc)
            
            # Option 2: Merge with segment sorting
            # merged_components = merge_region_components_better(region_components, subregions_bbox)
            
            # Option 3: Merge with explicit overlap handling
            # merged_components = merge_region_components_overlap(region_components, subregions_bbox)
            
            # Add to final list
            #all_segments_compressed.extend(merged_components)
            
            # Calculate statistics
            original_pixels = sum(seg['shape'][0] * seg['shape'][1] for seg in region_components)
            original_black = sum(seg['indices'].count(1) for seg in region_components)
            
            merged_pixels = sum(seg['shape'][0] * seg['shape'][1] for seg in merged_components)
            merged_black = sum(seg['indices'].count(1) for seg in merged_components)
            
            if debug:
                print(f"\nSummary:")
                print(f"  Original: {len(region_components)} segments, {original_pixels:,} pixels")
                print(f"  Merged: {len(merged_components)} segments, {merged_pixels:,} pixels")
                print(f"  Black reduction: {original_black - merged_black:,} pixels")
                print(f"  Pixel reduction: {original_pixels - merged_pixels:,} pixels")
            
        else:
            # Just add the single component
            subregions_components.append(region_components)
            #all_segments_compressed.extend(region_components)
            if debug: print(f"Only 1 component in subregions {i+1}, no merging needed")

    return subregions_components

