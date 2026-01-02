
import numpy as np
from collections import defaultdict




def merge_region_components_simple(region_components, roi_bbox):
    """
    Merge region components by placing them on ROI canvas.
    Colored pixels override black pixels.
    """
    if not region_components:
        return []
    
    if len(region_components) == 1:
        # For single segment, ensure it has the 'actual_colors' key
        single_seg = region_components[0].copy()
        if 'actual_colors' not in single_seg:
            single_seg['actual_colors'] = len(single_seg.get('palette', []))
        return [single_seg]
    
    print(f"\n{'='*60}")
    print(f"MERGING {len(region_components)} REGION COMPONENTS")
    print(f"{'='*60}")
    
    minr, minc, maxr, maxc = roi_bbox
    roi_height = maxr - minr
    roi_width = maxc - minc
    
    # ==============================================
    # 1. CREATE EMPTY CANVAS FOR ROI
    # ==============================================
    # Use uint32 to support up to ~4 billion colors (more than enough)
    roi_indices = np.zeros((roi_height, roi_width), dtype=np.uint32)
    
    # Collect all colors from all segments
    all_colors = []
    color_to_index = {}
    
    # ALWAYS add black as the first color (index 0)
    black_color = (0, 0, 0)
    all_colors.append(black_color)
    color_to_index[black_color] = 0
    
    print(f"ROI canvas: {roi_height}x{roi_width} pixels")
    print(f"Placing {len(region_components)} segments...")
    
    # ==============================================
    # 2. PLACE ALL SEGMENTS (last segment wins)
    # ==============================================
    for seg_idx, seg in enumerate(reversed(region_components)):
        seg_top_left = seg['top_left']
        seg_shape = seg['shape']
        seg_palette = seg['palette']
        seg_indices = np.array(seg['indices']).reshape(seg_shape)
        
        rel_row = seg_top_left[0] - minr
        rel_col = seg_top_left[1] - minc
        
        colored_placed = 0
        
        # Vectorized placement for better performance
        for r in range(seg_shape[0]):
            for c in range(seg_shape[1]):
                roi_r = rel_row + r
                roi_c = rel_col + c
                
                if 0 <= roi_r < roi_height and 0 <= roi_c < roi_width:
                    seg_pixel_idx = seg_indices[r, c]
                    
                    if seg_pixel_idx < len(seg_palette):
                        color_tuple = tuple(seg_palette[seg_pixel_idx])
                        
                        if color_tuple != black_color:
                            if color_tuple not in color_to_index:
                                color_to_index[color_tuple] = len(all_colors)
                                all_colors.append(color_tuple)
                            
                            roi_indices[roi_r, roi_c] = color_to_index[color_tuple]
                            colored_placed += 1
        
        print(f"  Segment {len(region_components)-seg_idx}: {colored_placed} colored pixels placed")
    
    # ==============================================
    # 3. CREATE MERGED SEGMENT
    # ==============================================
    black_pixels = np.sum(roi_indices == 0)
    colored_pixels_total = roi_height * roi_width - black_pixels
    
    print(f"\nMerging complete:")
    print(f"  ROI size: {roi_height}x{roi_width} = {roi_height*roi_width:,} pixels")
    print(f"  Black pixels: {black_pixels:,} ({black_pixels/(roi_height*roi_width)*100:.1f}%)")
    print(f"  Colored pixels: {colored_pixels_total:,}")
    print(f"  Unique colors: {len(all_colors)}")
    
    # Check if we can use smaller dtype for indices
    if len(all_colors) <= 256:
        indices_dtype = np.uint8
    elif len(all_colors) <= 65536:
        indices_dtype = np.uint16
    else:
        indices_dtype = np.uint32
    
    # Convert indices to appropriate dtype
    indices_converted = roi_indices.astype(indices_dtype)
    
    merged_segment = {
        'top_left': (minr, minc),
        'shape': (roi_height, roi_width),
        'palette': all_colors,
        'indices': indices_converted.flatten().tolist(),
        'indices_dtype': str(indices_dtype),
        'method': 'merged',
        'actual_colors': len(all_colors),
        'encoding': 'roi_merged'
    }
    
    return [merged_segment]




def visualize_merged_result(merged_segments, roi_shape, offset_r, offset_c, original_components=None):
    """
    Visualize the merged result for a ROI with side-by-side comparison.
    
    Args:
        merged_segments: List of merged segments (usually 1)
        roi_shape: (height, width) of the ROI
        offset_r, offset_c: ROI offset in original image
        original_components: Original components for comparison (optional)
    """
    import matplotlib.pyplot as plt
    
    h, w = roi_shape
    
    # ==============================================
    # 1. CREATE MERGED CANVAS
    # ==============================================
    merged_canvas = np.zeros((h, w, 3), dtype=np.uint8)
    
    for seg in merged_segments:
        top_left = seg['top_left']
        shape = seg['shape']
        indices = np.array(seg['indices']).reshape(shape)
        palette = seg['palette']
        
        # Calculate position relative to ROI
        rel_r = top_left[0] - offset_r
        rel_c = top_left[1] - offset_c
        
        # Create RGB image from indices
        for r in range(shape[0]):
            for c in range(shape[1]):
                canvas_r = rel_r + r
                canvas_c = rel_c + c
                
                if 0 <= canvas_r < h and 0 <= canvas_c < w:
                    idx = indices[r, c]
                    if idx < len(palette):
                        color = palette[idx]
                        merged_canvas[canvas_r, canvas_c] = color
    
    # ==============================================
    # 2. CREATE BLACK MASK
    # ==============================================
    black_mask = np.all(merged_canvas == [0, 0, 0], axis=2)
    colored_mask = ~black_mask
    
    # Create colored-only image (black pixels become transparent in visualization)
    colored_only = merged_canvas.copy()
    
    # ==============================================
    # 3. CREATE ORIGINAL COMPONENTS CANVAS (if provided)
    # ==============================================
    if original_components:
        original_canvas = np.zeros((h, w, 3), dtype=np.uint8)
        
        for seg in original_components:
            top_left = seg['top_left']
            shape = seg['shape']
            indices = np.array(seg['indices']).reshape(shape)
            palette = seg['palette']
            
            rel_r = top_left[0] - offset_r
            rel_c = top_left[1] - offset_c
            
            for r in range(shape[0]):
                for c in range(shape[1]):
                    canvas_r = rel_r + r
                    canvas_c = rel_c + c
                    
                    if 0 <= canvas_r < h and 0 <= canvas_c < w:
                        idx = indices[r, c]
                        if idx < len(palette):
                            color = palette[idx]
                            original_canvas[canvas_r, canvas_c] = color
    
    # ==============================================
    # 4. CREATE VISUALIZATION
    # ==============================================
    if original_components:
        # Show 4 subplots: original, merged, colored only, black mask
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original components
        axes[0, 0].imshow(original_canvas)
        axes[0, 0].set_title(f'Original: {len(original_components)} segments')
        axes[0, 0].axis('off')
        
        # Merged result
        axes[0, 1].imshow(merged_canvas)
        axes[0, 1].set_title(f'Merged: {len(merged_segments)} segment(s)')
        axes[0, 1].axis('off')
        
        # Colored pixels only (black as white for visibility)
        colored_display = colored_only.copy()
        colored_display[black_mask] = [255, 255, 255]  # White background
        axes[0, 2].imshow(colored_display)
        axes[0, 2].set_title(f'Colored pixels: {np.sum(colored_mask):,}')
        axes[0, 2].axis('off')
        
        # Black pixel mask
        axes[1, 0].imshow(black_mask, cmap='gray')
        axes[1, 0].set_title(f'Black pixels: {np.sum(black_mask):,} ({np.sum(black_mask)/(h*w)*100:.1f}%)')
        axes[1, 0].axis('off')
        
        # Color distribution
        if np.sum(colored_mask) > 0:
            # Get all colored pixels
            colored_pixels = merged_canvas[colored_mask].reshape(-1, 3)
            
            # Create histogram of color intensities
            intensities = np.mean(colored_pixels, axis=1)
            axes[1, 1].hist(intensities, bins=50, color='skyblue', edgecolor='black')
            axes[1, 1].set_title('Color intensity distribution')
            axes[1, 1].set_xlabel('Intensity (0-255)')
            axes[1, 1].set_ylabel('Count')
        else:
            axes[1, 1].text(0.5, 0.5, 'No colored pixels', 
                          ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].axis('off')
        
        # Stats text
        stats_text = f"""
        ROI Size: {h}x{w} = {h*w:,} pixels
        Colored: {np.sum(colored_mask):,} ({np.sum(colored_mask)/(h*w)*100:.1f}%)
        Black: {np.sum(black_mask):,} ({np.sum(black_mask)/(h*w)*100:.1f}%)
        Colors: {len(merged_segments[0]['palette']) if merged_segments else 0}
        """
        axes[1, 2].text(0.1, 0.5, stats_text, fontsize=10, 
                       verticalalignment='center', transform=axes[1, 2].transAxes)
        axes[1, 2].axis('off')
        
    else:
        # Show 2x2 grid for merged only
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Merged result
        axes[0, 0].imshow(merged_canvas)
        axes[0, 0].set_title(f'Merged ROI ({len(merged_segments)} segment(s))')
        axes[0, 0].axis('off')
        
        # Colored pixels only
        colored_display = colored_only.copy()
        colored_display[black_mask] = [255, 255, 255]  # White background
        axes[0, 1].imshow(colored_display)
        axes[0, 1].set_title(f'Colored pixels only ({np.sum(colored_mask):,})')
        axes[0, 1].axis('off')
        
        # Black pixel mask
        axes[1, 0].imshow(black_mask, cmap='gray')
        axes[1, 0].set_title(f'Black pixels: {np.sum(black_mask):,} ({np.sum(black_mask)/(h*w)*100:.1f}%)')
        axes[1, 0].axis('off')
        
        # Color preview (first few colors in palette)
        if merged_segments and len(merged_segments[0]['palette']) > 1:
            palette = merged_segments[0]['palette']
            # Skip black (first color)
            colors_to_show = min(10, len(palette) - 1)
            
            # Create color swatches
            swatch_height = 1
            swatch_width = colors_to_show
            
            color_swatches = np.zeros((swatch_height, swatch_width, 3), dtype=np.uint8)
            for i in range(colors_to_show):
                color_swatches[0, i] = palette[i + 1]  # Skip black
            
            axes[1, 1].imshow(color_swatches)
            axes[1, 1].set_title(f'Top {colors_to_show} colors (excl. black)')
            axes[1, 1].axis('off')
        else:
            axes[1, 1].text(0.5, 0.5, 'No colors to display', 
                          ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed stats
    print(f"\n{'='*60}")
    print(f"VISUALIZATION STATISTICS")
    print(f"{'='*60}")
    print(f"ROI Size: {h}x{w} = {h*w:,} pixels")
    print(f"Colored pixels: {np.sum(colored_mask):,} ({np.sum(colored_mask)/(h*w)*100:.1f}%)")
    print(f"Black pixels: {np.sum(black_mask):,} ({np.sum(black_mask)/(h*w)*100:.1f}%)")
    
    if merged_segments:
        seg = merged_segments[0]
        print(f"Segment shape: {seg['shape']}")
        print(f"Palette size: {len(seg['palette'])} colors")
        print(f"Top-left position: {seg['top_left']}")
        
        # Count how many unique colors actually appear
        indices = np.array(seg['indices'])
        unique_indices = set(indices)
        print(f"Unique color indices used: {len(unique_indices)}")
        
        # Show color frequencies
        if len(seg['palette']) <= 20:  # Only if palette is small
            print("\nColor frequencies:")
            for idx in sorted(unique_indices):
                count = np.sum(indices == idx)
                color = seg['palette'][idx]
                if tuple(color) == (0, 0, 0):
                    color_name = "BLACK"
                else:
                    color_name = f"RGB{tuple(color)}"
                print(f"  Index {idx}: {color_name} - {count:,} pixels ({count/len(indices)*100:.1f}%)")
    
    return merged_canvas
