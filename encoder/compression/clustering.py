import math
from sklearn.cluster import MiniBatchKMeans

def get_all_unique_colors(region_image, top_left_coords):
    """
    Get ALL unique colors from the region without any clustering.
    Returns the same data structure as compression functions.
    """
    if region_image is None or region_image.size == 0:
        return None
    
    h, w, _ = region_image.shape
    total_pixels = h * w
    
    print(f"Getting all unique colors: {h}x{w} region")
    print(f"Top-left: {top_left_coords}")
    
    # ==============================================
    # 1. GET ALL UNIQUE COLORS
    # ==============================================
    pixels = region_image.reshape(-1, 3)
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)
    unique_count = len(unique_colors)
    
    print(f"  Found {unique_count:,} unique colors")
    
    # ==============================================
    # 2. CREATE COLOR TO INDEX MAPPING
    # ==============================================
    # Create a dictionary mapping each color to its index
    color_to_index = {}
    palette = []
    
    for i, color in enumerate(unique_colors):
        color_tuple = tuple(color)
        color_to_index[color_tuple] = i
        palette.append(color.tolist())  # Convert to list for JSON serialization
    
    print(f"  Created palette with {len(palette)} colors")
    
    # ==============================================
    # 3. CREATE INDICES FOR EACH PIXEL
    # ==============================================
    # Map each pixel to its color index
    indices_flat = []
    for pixel in pixels:
        color_tuple = tuple(pixel)
        indices_flat.append(color_to_index[color_tuple])
    
    # ==============================================
    # 4. CALCULATE BASIC STATISTICS
    # ==============================================
    actual_colors = len(palette)
    original_size = total_pixels * 3
    
    # Determine index data type
    if actual_colors <= 256:
        index_dtype = np.uint8
        bytes_per_index = 1
    else:
        index_dtype = np.uint16
        bytes_per_index = 2
    
    # Calculate compressed size
    palette_size = actual_colors * 3
    indices_size = total_pixels * bytes_per_index
    metadata_size = 50  # Approximate metadata size
    
    compressed_size = palette_size + indices_size + metadata_size
    
    # Calculate compression ratio
    compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
    
    # PSNR is perfect since we're using all original colors
    psnr = float('inf')
    
    # ==============================================
    # 5. CREATE OUTPUT DATA STRUCTURE
    # ==============================================
    compressed_data = {
        'method': 'exact_colors',
        'top_left': top_left_coords,
        'shape': (h, w),
        'palette': palette,  # All unique colors
        'indices': indices_flat,
        'max_colors': actual_colors,
        'actual_colors': actual_colors,
        'index_dtype': str(index_dtype),
        'original_size': original_size,
        'compressed_size': compressed_size,
        'compression_ratio': compression_ratio,
        'mse': 0.0,  # Perfect reconstruction
        'psnr': psnr,
        'encoding': 'exact'  # Mark as exact color representation
    }
    
    print(f"  Palette size: {actual_colors} colors")
    print(f"  Original: {original_size:,} bytes")
    print(f"  Compressed: {compressed_size:,} bytes")
    print(f"  Ratio: {compression_ratio:.2f}:1")
    print(f"  PSNR: Perfect (exact colors)")
    
    return compressed_data




def compute_clustering_params(n_colors, quality, color_space='rgb'):
    """
    Compute optimized DBSCAN parameters for color clustering
    
    Args:
        n_colors: Number of distinct colors in palette
        quality: 0-100 quality parameter
        color_space: 'rgb' or 'lab' (LAB is perceptually uniform)
    
    Returns:
        eps, min_samples, max_colors_per_cluster
    """
    # ==============================================
    # 1. EPSILON (maximum distance within cluster)
    # ==============================================
    # Quality 0-100 maps to perceptual thresholds
    # Higher quality = smaller epsilon (tighter clusters)
    
    #eps=256*100/(quality*2)
    eps=128 - 1.28 * quality
    #max_colors_per_cluster=math.ceil(n_colors*100/quality)
    max_colors_per_cluster= math.ceil( ( -(quality / 100) * n_colors + n_colors) / quality )

    if eps==0: eps=1
    if max_colors_per_cluster==0: max_colors_per_cluster=1
    min_samples=1
    
    return eps, min_samples, max_colors_per_cluster



















import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from sklearn.cluster import DBSCAN

def cluster_palette_colors_parallel(quality, compressed_data, eps=10.0, min_samples=2, 
                                   max_colors_per_cluster=5, num_workers=None):
    """
    Parallel version of color clustering.
    Large clusters are split in parallel using multiple threads.
    """
    print(f"\n{'='*60}")
    print(f"CLUSTERING PALETTE COLORS (Parallel - Black preserved)")
    print(f"{'='*60}")
    
    # Extract original data
    palette = np.array(compressed_data['palette'], dtype=np.uint8)
    indices = np.array(compressed_data['indices'])
    h, w = compressed_data['shape']
    top_left = compressed_data['top_left']
    
    # Get the count of colors (scalar, not array)
    original_colors_count = len(palette)  # This is a scalar integer
    num_workers= max(5, math.ceil(original_colors_count/2500) )
    print(f"Original palette: {original_colors_count} colors")
    print(f"Clustering parameters: eps={eps}, min_samples={min_samples}")
    
    # ==============================================
    # 1. SEPARATE BLACK FROM OTHER COLORS
    # ==============================================
    black_indices = []
    non_black_indices = []
    
    for i, color in enumerate(palette):
        if np.array_equal(color, [0, 0, 0]):
            black_indices.append(i)
        else:
            non_black_indices.append(i)
    
    print(f"  Black colors found: {len(black_indices)}")
    print(f"  Non-black colors: {len(non_black_indices)}")
    
    if len(non_black_indices) == 0:
        print("  Only black colors - nothing to cluster!")
        return compressed_data
    
    # ==============================================
    # 2. CLUSTER ONLY NON-BLACK COLORS
    # ==============================================
    non_black_palette = palette[non_black_indices]
    palette_normalized = non_black_palette.astype(float) / 255.0

    if len(non_black_palette)>=10000:

        # Use MiniBatch K-Means for speed
        n_clusters=math.ceil(len(non_black_palette)*(quality/100)/10)
        kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            batch_size=1000,
            random_state=42,
            n_init='auto'
        )
        
        cluster_labels = kmeans.fit_predict(non_black_palette.astype(float))
    
        # Get the actual number of clusters found (K-means always finds n_clusters)
        actual_clusters = len(np.unique(cluster_labels))
        
        # Get the cluster centers (these are your reduced colors)
        cluster_centers = kmeans.cluster_centers_
        
        # Round and convert to uint8
        reduced_colors = np.round(cluster_centers).astype(np.uint8)
        
        print(f"  K-Means reduction: {len(non_black_palette)} → {len(reduced_colors)} colors")
        print(f"  Actual clusters found: {actual_clusters}")
    
    else:
        dbscan = DBSCAN(eps=eps/255.0, min_samples=min_samples, metric='euclidean')
        #dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        cluster_labels = dbscan.fit_predict(palette_normalized)


    
    

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = np.sum(cluster_labels == -1)
    
    print(f"DBSCAN found {n_clusters} clusters + {n_noise} noise colors")
    
    # ==============================================
    # 3. PREPARE CLUSTERS FOR PARALLEL PROCESSING
    # ==============================================
    new_palette = []
    color_mapping = {}
    
    # Add black colors
    for old_black_idx in black_indices:
        new_palette.append(palette[old_black_idx])
        color_mapping[old_black_idx] = len(new_palette) - 1
    
    # Process noise points (not parallelizable - small)
    noise_mask = cluster_labels == -1
    noise_relative_indices = np.where(noise_mask)[0]
    
    for rel_idx in noise_relative_indices:
        old_idx = non_black_indices[rel_idx]
        new_palette.append(palette[old_idx])
        color_mapping[old_idx] = len(new_palette) - 1
    
    # ==============================================
    # 4. PARALLEL PROCESSING OF LARGE CLUSTERS
    # ==============================================
    # Identify clusters that need splitting
    clusters_to_process = []
    small_clusters_info = []  # Store info for small clusters
    
    unique_labels = set(cluster_labels)
    
    for label in unique_labels:
        if label == -1:
            continue  # Already processed noise
        
        cluster_mask = cluster_labels == label
        cluster_relative_indices = np.where(cluster_mask)[0]
        cluster_original_indices = [non_black_indices[idx] for idx in cluster_relative_indices]
        cluster_colors = palette[cluster_original_indices]
        
        if len(cluster_original_indices) > max_colors_per_cluster:
            # Large cluster - needs parallel splitting
            clusters_to_process.append({
                'label': label,
                'colors': cluster_colors,
                'original_indices': cluster_original_indices,
                'size': len(cluster_original_indices)
            })
        else:
            # Small cluster - store for immediate processing
            small_clusters_info.append({
                'label': label,
                'colors': cluster_colors,
                'original_indices': cluster_original_indices
            })
    
    print(f"Found {len(clusters_to_process)} large clusters for parallel processing")
    print(f"Found {len(small_clusters_info)} small clusters for immediate processing")
    
    # Process small clusters immediately
    for cluster_info in small_clusters_info:
        avg_color = np.mean(cluster_info['colors'], axis=0).astype(np.uint8)
        new_idx = len(new_palette)
        new_palette.append(avg_color)
        
        for old_idx in cluster_info['original_indices']:
            color_mapping[old_idx] = new_idx
    
    # ==============================================
    # 5. PARALLEL SPLITTING OF LARGE CLUSTERS
    # ==============================================
    if clusters_to_process:
        # Determine number of workers
        if num_workers is None:
            num_workers = min(multiprocessing.cpu_count(), len(clusters_to_process))
        
        print(f"Using {num_workers} workers for parallel processing")
        
        # Process clusters in parallel
        results = process_clusters_parallel(
            clusters_to_process, 
            max_colors_per_cluster, 
            num_workers
        )
        
        # Process results from parallel workers
        for cluster_info, split_clusters in results:
            label = cluster_info['label']
            original_indices = cluster_info['original_indices']
            original_colors = cluster_info['colors']
            
            if split_clusters is None:
                # Fallback: use simple averaging
                print(f"  Cluster {label}: parallel processing failed, using fallback")
                avg_color = np.mean(original_colors, axis=0).astype(np.uint8)
                new_idx = len(new_palette)
                new_palette.append(avg_color)
                
                for old_idx in original_indices:
                    color_mapping[old_idx] = new_idx
            else:
                # Process each split cluster
                for split_cluster in split_clusters:
                    avg_color = np.mean(split_cluster, axis=0).astype(np.uint8)
                    new_idx = len(new_palette)
                    new_palette.append(avg_color)
                    
                    # Map colors in this split
                    for color in split_cluster:
                        old_idx = find_color_index(palette, color)
                        if old_idx is not None:
                            color_mapping[old_idx] = new_idx
    
    # ==============================================
    # 6. COMPLETE PROCESSING AND CALCULATIONS
    # ==============================================
    new_palette = np.array(new_palette)
    new_color_count = len(new_palette)  # This is a scalar integer
    
    # Verify black is preserved
    black_preserved = any(np.array_equal(color, [0, 0, 0]) for color in new_palette)
    print(f"  Black preserved: {black_preserved}")
    
    # FIXED: Use scalar integers for printing
    print(f"New palette: {new_color_count} colors")
    print(f"Color reduction: {original_colors_count} → {new_color_count} "
          f"({(original_colors_count - new_color_count)/original_colors_count*100:.1f}%)")
    
    # Update indices
    mapping_array = np.zeros(original_colors_count, dtype=np.uint16)
    for old_idx, new_idx in color_mapping.items():
        mapping_array[old_idx] = new_idx
    
    new_indices = mapping_array[indices].tolist()
    
    # ==============================================
    # 7. CALCULATE STATISTICS (MISSING IN ORIGINAL)
    # ==============================================
    total_pixels = h * w
    original_size = compressed_data.get('original_size', total_pixels * 3)
    
    # Calculate new compressed size
    if new_color_count <= 256:
        bytes_per_index = 1
        index_dtype = 'uint8'
    else:
        bytes_per_index = 2
        index_dtype = 'uint16'
    
    new_palette_size = new_color_count * 3
    new_indices_size = total_pixels * bytes_per_index
    metadata_size = 100  # Approximate
    
    new_compressed_size = new_palette_size + new_indices_size + metadata_size
    compression_ratio = original_size / new_compressed_size if new_compressed_size > 0 else 0
    
    # Estimate PSNR - FIXED: pass proper parameters
    mse = 0
    psnr = 10 * np.log10(255*255/mse) if mse > 0 else float('inf')
    
    # ==============================================
    # 8. CREATE NEW COMPRESSED DATA
    # ==============================================
    new_compressed_data = {
        'method': 'clustered_colors',
        'top_left': top_left,
        'shape': (h, w),
        'palette': new_palette.tolist(),
        'indices': new_indices,
        'original_unique_colors': original_colors_count,  # Use scalar
        'compressed_colors': new_color_count,  # Use scalar
        'index_dtype': index_dtype,
        'original_size': original_size,
        'compressed_size': new_compressed_size,
        'compression_ratio': compression_ratio,
        'mse': float(mse),
        'psnr': float(psnr),
        'clustering_params': {
            'eps': eps,
            'min_samples': min_samples,
            'max_colors_per_cluster': max_colors_per_cluster
        },
        'encoding': 'dbscan_clustered',
        'black_preserved': True,
        'parallel_processed': True  # Flag to indicate parallel processing was used
    }
    
    print(f"\nParallel clustering complete:")
    print(f"  Colors: {original_colors_count} → {new_color_count}")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  Compression: {original_size:,} → {new_compressed_size:,} bytes")
    print(f"  Ratio: {compression_ratio:.2f}:1")
    
    return new_compressed_data

def process_clusters_parallel(clusters_to_process, max_colors_per_cluster, num_workers):
    """
    Process multiple clusters in parallel using ThreadPoolExecutor.
    """
    results = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_cluster = {}
        for cluster_info in clusters_to_process:
            future = executor.submit(
                split_cluster_worker,
                cluster_info['colors'],
                max_colors_per_cluster,
                cluster_info['label']
            )
            future_to_cluster[future] = cluster_info
        
        # Process results as they complete
        for future in as_completed(future_to_cluster):
            cluster_info = future_to_cluster[future]
            try:
                split_clusters = future.result(timeout=30)  # 30 second timeout
                results.append((cluster_info, split_clusters))
            except Exception as e:
                print(f"Error processing cluster {cluster_info['label']}: {e}")
                results.append((cluster_info, None))  # Mark as failed
    
    return results

def split_cluster_worker(cluster_colors, max_colors_per_cluster, cluster_label=None):
    """
    Worker function for parallel cluster splitting.
    """
    try:
        return split_large_cluster(cluster_colors, max_colors_per_cluster)
    except Exception as e:
        if cluster_label:
            print(f"Worker error for cluster {cluster_label}: {e}")
        return None


































from sklearn.cluster import DBSCAN
import numpy as np

def cluster_palette_colors(compressed_data, eps=10.0, min_samples=2, max_colors_per_cluster=5):
    """
    Cluster similar colors in palette using DBSCAN.
    Colors within 'eps' distance are clustered together.
    Each cluster is replaced with its average color.
    
    SPECIAL RULE: Black [0, 0, 0] is NEVER clustered - always kept as is.
    """
    print(f"\n{'='*60}")
    print(f"CLUSTERING PALETTE COLORS (Black preserved)")
    print(f"{'='*60}")
    
    # Extract original data
    palette = np.array(compressed_data['palette'], dtype=np.uint8)
    indices = np.array(compressed_data['indices'])
    h, w = compressed_data['shape']
    top_left = compressed_data['top_left']
    
    original_colors = len(palette)
    print(f"Original palette: {original_colors} colors")
    print(f"Clustering parameters: eps={eps}, min_samples={min_samples}")
    
    # ==============================================
    # 1. SEPARATE BLACK FROM OTHER COLORS
    # ==============================================
    # Find which indices correspond to black [0, 0, 0]
    black_indices = []
    non_black_indices = []
    
    for i, color in enumerate(palette):
        if np.array_equal(color, [0, 0, 0]):
            black_indices.append(i)
        else:
            non_black_indices.append(i)
    
    print(f"  Black colors found: {len(black_indices)}")
    print(f"  Non-black colors: {len(non_black_indices)}")
    
    if len(non_black_indices) == 0:
        print("  Only black colors - nothing to cluster!")
        return compressed_data
    
    # ==============================================
    # 2. CLUSTER ONLY NON-BLACK COLORS
    # ==============================================
    non_black_palette = palette[non_black_indices]
    
    # Normalize colors to 0-1 for better distance calculation
    palette_normalized = non_black_palette.astype(float) / 255.0
    
    # Apply DBSCAN (only to non-black colors)
    dbscan = DBSCAN(eps=eps/255.0, min_samples=min_samples, metric='euclidean')
    cluster_labels = dbscan.fit_predict(palette_normalized)
    
    # Count clusters (-1 means noise/outlier colors)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = np.sum(cluster_labels == -1)
    
    print(f"DBSCAN found {n_clusters} clusters + {n_noise} noise colors (excluding black)")
    
    # ==============================================
    # 3. CREATE NEW PALETTE (INCLUDING BLACK)
    # ==============================================
    new_palette = []
    color_mapping = {}  # Old index → new index
    
    # FIRST: Add all black colors exactly as they are
    for old_black_idx in black_indices:
        new_palette.append(palette[old_black_idx])
        color_mapping[old_black_idx] = len(new_palette) - 1
    
    # SECOND: Process non-black clusters
    unique_labels = set(cluster_labels)
    
    for label in unique_labels:
        if label == -1:
            # Noise points - keep them as individual colors
            noise_mask = cluster_labels == -1
            noise_relative_indices = np.where(noise_mask)[0]
            
            for rel_idx in noise_relative_indices:
                # Convert relative index back to original palette index
                old_idx = non_black_indices[rel_idx]
                new_palette.append(palette[old_idx])
                color_mapping[old_idx] = len(new_palette) - 1
        else:
            # Regular cluster
            cluster_mask = cluster_labels == label
            cluster_relative_indices = np.where(cluster_mask)[0]
            
            # Convert relative indices to original palette indices
            cluster_original_indices = [non_black_indices[idx] for idx in cluster_relative_indices]
            
            if len(cluster_original_indices) > max_colors_per_cluster:
                print(f"  Cluster {label} has {len(cluster_original_indices)} colors > {max_colors_per_cluster}, splitting...")
                
                # Get the actual colors for this cluster
                cluster_colors = palette[cluster_original_indices]
                
                # Split large cluster
                split_clusters = split_large_cluster(cluster_colors, int(max_colors_per_cluster))
                
                for split_cluster in split_clusters:
                    avg_color = np.mean(split_cluster, axis=0).astype(np.uint8)
                    new_palette.append(avg_color)
                    
                    # Map all colors in this split to the same new index
                    for color in split_cluster:
                        # Find original index for this color
                        old_idx = find_color_index(palette, color)
                        if old_idx is not None:
                            color_mapping[old_idx] = len(new_palette) - 1
            else:
                # Average colors in cluster
                cluster_colors = palette[cluster_original_indices]
                avg_color = np.mean(cluster_colors, axis=0).astype(np.uint8)
                new_palette.append(avg_color)
                
                # Map all colors in cluster to the same new index
                for old_idx in cluster_original_indices:
                    color_mapping[old_idx] = len(new_palette) - 1
    
    new_palette = np.array(new_palette)
    new_color_count = len(new_palette)
    
    # Verify black is preserved
    black_preserved = any(np.array_equal(color, [0, 0, 0]) for color in new_palette)
    print(f"  Black preserved: {black_preserved}")
    
    print(f"New palette: {new_color_count} colors")
    print(f"Color reduction: {original_colors} → {new_color_count} "
          f"({(original_colors - new_color_count)/original_colors*100:.1f}%)")
    
    # ==============================================
    # 4. UPDATE INDICES WITH NEW COLOR MAPPING
    # ==============================================
    # Create a mapping array for fast lookup
    mapping_array = np.zeros(original_colors, dtype=np.uint16)
    for old_idx, new_idx in color_mapping.items():
        mapping_array[old_idx] = new_idx
    
    # Update indices
    new_indices = mapping_array[indices].tolist()
    
    # ==============================================
    # 5. CALCULATE STATISTICS
    # ==============================================
    total_pixels = h * w
    original_size = compressed_data.get('original_size', total_pixels * 3)
    
    # Calculate new compressed size
    if new_color_count <= 256:
        bytes_per_index = 1
        index_dtype = 'uint8'
    else:
        bytes_per_index = 2
        index_dtype = 'uint16'
    
    new_palette_size = new_color_count * 3
    new_indices_size = total_pixels * bytes_per_index
    metadata_size = 100  # Approximate
    
    new_compressed_size = new_palette_size + new_indices_size + metadata_size
    compression_ratio = original_size / new_compressed_size if new_compressed_size > 0 else 0
    
    # Estimate PSNR
    mse = 0
    psnr = 10 * np.log10(255*255/mse) if mse > 0 else float('inf')
    
    # ==============================================
    # 6. CREATE NEW COMPRESSED DATA
    # ==============================================
    new_compressed_data = {
        'method': 'clustered_colors',
        'top_left': top_left,
        'shape': (h, w),
        'palette': new_palette.tolist(),
        'indices': new_indices,
        'original_unique_colors': original_colors,
        'compressed_colors': new_color_count,
        'index_dtype': index_dtype,
        'original_size': original_size,
        'compressed_size': new_compressed_size,
        'compression_ratio': compression_ratio,
        'mse': float(mse),
        'psnr': float(psnr),
        'clustering_params': {
            'eps': eps,
            'min_samples': min_samples,
            'max_colors_per_cluster': max_colors_per_cluster
        },
        'encoding': 'dbscan_clustered',
        'black_preserved': True  # Flag to indicate black was preserved
    }
    
    print(f"\nClustering complete:")
    print(f"  Colors: {original_colors} → {new_color_count}")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  Compression: {original_size:,} → {new_compressed_size:,} bytes")
    print(f"  Ratio: {compression_ratio:.2f}:1")
    
    return new_compressed_data


def split_large_cluster(cluster_colors, max_colors_per_cluster):
    """
    Split a large cluster into smaller sub-clusters.
    Uses simple K-means to split.
    
    Args:
        cluster_colors: Array of colors in the cluster [n_colors, 3]
        max_colors_per_cluster: Maximum allowed colors per sub-cluster
    
    Returns:
        List of sub-cluster arrays
    """
    n_colors = len(cluster_colors)
    
    # If cluster is already small enough, return it as single cluster
    if n_colors <= max_colors_per_cluster:
        return [cluster_colors]
    
    # Calculate how many sub-clusters we need
    n_splits = max(2, (n_colors + max_colors_per_cluster - 1) // max_colors_per_cluster)
    
    # Ensure we don't ask for more clusters than colors
    n_splits = min(n_splits, n_colors)
    
    # Special case: very small clusters
    if n_colors <= 2 or n_splits < 2:
        return [cluster_colors]
    
    try:
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=n_splits, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(cluster_colors.astype(float))
        
        splits = []
        for i in range(n_splits):
            mask = labels == i
            if np.any(mask):
                splits.append(cluster_colors[mask])
        
        # Double-check that all splits are within size limit
        # If any split is still too large, recursively split it
        final_splits = []
        for split in splits:
            if len(split) > max_colors_per_cluster:
                final_splits.extend(split_large_cluster(split, max_colors_per_cluster))
            else:
                final_splits.append(split)
        
        return final_splits
        
    except Exception as e:
        print(f"Warning: KMeans splitting failed: {e}")
        print(f"  Cluster size: {n_colors}, Requested splits: {n_splits}")
        # Fallback: Simple split by luminance
        return split_by_luminance(cluster_colors, max_colors_per_cluster)


def split_by_luminance(cluster_colors, max_colors_per_cluster):
    """
    Fallback splitting method: sort by luminance and split evenly
    """
    n_colors = len(cluster_colors)
    
    if n_colors <= max_colors_per_cluster:
        return [cluster_colors]
    
    # Calculate luminance (Y = 0.299*R + 0.587*G + 0.114*B)
    luminance = (0.299 * cluster_colors[:, 0] + 
                 0.587 * cluster_colors[:, 1] + 
                 0.114 * cluster_colors[:, 2])
    
    # Sort by luminance
    sorted_indices = np.argsort(luminance)
    sorted_colors = cluster_colors[sorted_indices]
    
    # Split into approximately equal parts
    n_splits = max(2, (n_colors + max_colors_per_cluster - 1) // max_colors_per_cluster)
    split_points = np.array_split(sorted_colors, n_splits)
    
    # Filter out empty splits
    return [split for split in split_points if len(split) > 0]

def find_color_index(palette, color):
    """Find the index of a color in the palette."""
    matches = np.all(palette == color, axis=1)
    if np.any(matches):
        return np.where(matches)[0][0]
    return None















def hierarchical_color_clustering(compressed_data, quality=85):
    """
    Alternative: Hierarchical clustering with quality control.
    """
    palette = np.array(compressed_data['palette'], dtype=np.uint8)
    indices = np.array(compressed_data['indices'])
    
    original_colors = len(palette)
    target_colors = max(2, int(original_colors * quality / 100))
    
    print(f"Hierarchical clustering: {original_colors} → {target_colors} colors")
    
    # Use K-means for hierarchical-like clustering
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(n_clusters=target_colors, random_state=42, n_init=3)
    kmeans.fit(palette.astype(float))
    
    # Get new palette and mapping
    new_palette = kmeans.cluster_centers_.astype(np.uint8)
    new_indices = kmeans.labels_[indices].tolist()
    
    # Create new compressed data
    return create_clustered_result(
        compressed_data, new_palette, new_indices, 'hierarchical_clustered'
    )

def create_clustered_result(original_data, new_palette, new_indices, method_name):
    """Helper to create clustered result structure."""
    h, w = original_data['shape']
    total_pixels = h * w
    
    new_color_count = len(new_palette)
    
    # Calculate sizes
    original_size = original_data.get('original_size', total_pixels * 3)
    
    if new_color_count <= 256:
        bytes_per_index = 1
        index_dtype = 'uint8'
    else:
        bytes_per_index = 2
        index_dtype = 'uint16'
    
    new_compressed_size = new_color_count * 3 + total_pixels * bytes_per_index + 100
    
    return {
        'method': method_name,
        'top_left': original_data['top_left'],
        'shape': (h, w),
        'palette': new_palette.tolist(),
        'indices': new_indices,
        'compressed_colors': new_color_count,
        'index_dtype': index_dtype,
        'original_size': original_size,
        'compressed_size': new_compressed_size,
        'compression_ratio': original_size / new_compressed_size,
        'encoding': method_name
    }


