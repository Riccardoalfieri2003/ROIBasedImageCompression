import numpy as np
import zlib
import struct
import pickle

# ============================================================================
# COMPRESSION
# ============================================================================

def compress_palette(palette):
    """
    Compress palette using zlib (DEFLATE = LZ77 + Huffman).
    
    Args:
        palette: List of RGB tuples [(r,g,b), ...]
    
    Returns:
        bytes: Compressed palette data
    """
    palette_np = np.array(palette, dtype=np.uint8)
    raw_bytes = palette_np.tobytes()
    return zlib.compress(raw_bytes, level=9)


def compress_indices_rle_huffman(indices_list):
    """
    Compress index list using RLE + Huffman (via zlib).
    
    Args:
        indices_list: Flat list or 1D array of indices
    
    Returns:
        bytes: Compressed indices data
    """
    indices = np.array(indices_list, dtype=np.uint16)  # Support up to 65k colors
    
    # ========================================
    # STEP 1: RUN-LENGTH ENCODING
    # ========================================
    rle_data = bytearray()
    
    if len(indices) == 0:
        return zlib.compress(bytes(rle_data), level=9)
    
    current_value = indices[0]
    run_length = 1
    
    for value in indices[1:]:
        if value == current_value and run_length < 65535:
            run_length += 1
        else:
            # Write [value (2 bytes), run_length (2 bytes)]
            rle_data.extend(struct.pack('<HH', current_value, run_length))
            current_value = value
            run_length = 1
    
    # Write last run
    rle_data.extend(struct.pack('<HH', current_value, run_length))
    
    # ========================================
    # STEP 2: HUFFMAN ENCODING (via zlib)
    # ========================================
    # zlib uses DEFLATE which includes Huffman coding
    compressed = zlib.compress(bytes(rle_data), level=9)
    
    return compressed


def compress_indices_simple(indices_list):
    """
    Alternative: Direct zlib compression (already includes LZ77+Huffman).
    Often simpler and just as effective.
    
    Args:
        indices_list: Flat list or 1D array of indices
    
    Returns:
        bytes: Compressed indices data
    """
    indices = np.array(indices_list, dtype=np.uint16)
    raw_bytes = indices.tobytes()
    return zlib.compress(raw_bytes, level=9)


def lossless_compress(palette, indices_list, shape, use_manual_rle=False):
    """
    Main compression function.
    
    Args:
        palette: List of RGB tuples [(r,g,b), ...]
        indices_list: Flat list of palette indices
        shape: Tuple (height, width) of original image
        use_manual_rle: If True, use manual RLE+Huffman. If False, use direct zlib.
    
    Returns:
        dict: Compressed data package
    """
    # Compress palette
    palette_compressed = compress_palette(palette)
    
    # Compress indices (choose method)
    if use_manual_rle:
        indices_compressed = compress_indices_rle_huffman(indices_list)
        method = 'rle_huffman'
    else:
        indices_compressed = compress_indices_simple(indices_list)
        method = 'zlib_direct'
    
    # Package minimal data
    return {
        's': shape,                          # (h, w)
        'ps': len(palette),            # Number of colors
        'p': palette_compressed,       # Compressed palette
        'i': indices_compressed,       # Compressed indices
        #'method': method                          # Compression method used
    }


def save_compressed(compressed_data, filename):
    """
    Save compressed data to file.
    
    Args:
        compressed_data: Output from lossless_compress()
        filename: Path to save file
    
    Returns:
        int: Size of saved file in bytes
    """
    # Serialize the dictionary
    serialized = pickle.dumps(compressed_data, protocol=5)
    
    # Apply final compression layer
    final_compressed = zlib.compress(serialized, level=9)
    
    # Write with magic header
    with open(filename, 'wb') as f:
        f.write(b'RHCCQ')  # Magic number: Region Hierarchical Clustering Color Quantization
        f.write(struct.pack('<I', len(final_compressed)))
        f.write(final_compressed)
    
    return len(final_compressed) + 8  # +8 for header








def lossless_compress_optimized(palette, indices_list, shape, use_manual_rle=False):
    """
    Optimized compression with automatic dtype selection.
    Handles both Python lists and numpy arrays.
    """
    # Compress palette
    palette_compressed = compress_palette(palette)
    
    # Handle different input types
    if isinstance(indices_list, np.ndarray):
        # It's a numpy array
        if indices_list.size == 0:
            max_index = 0
            indices_flat = indices_list.flatten()
        else:
            max_index = indices_list.max()
            indices_flat = indices_list.flatten()
    elif isinstance(indices_list, list):
        # It's a Python list
        if not indices_list:
            max_index = 0
            indices_flat = indices_list
        else:
            max_index = max(indices_list)
            indices_flat = indices_list
    else:
        raise TypeError(f"indices_list must be list or numpy array, got {type(indices_list)}")
    
    # Determine optimal dtype for indices
    if max_index < 256:
        dtype = np.uint8
        dtype_name = 'uint8'
    elif max_index < 65536:
        dtype = np.uint16
        dtype_name = 'uint16'
    else:
        dtype = np.uint32
        dtype_name = 'uint32'
    
    print(f"Indices optimization: max={max_index}, using {dtype_name}")
    
    indices_compressed = compress_indices_simple_optimized(indices_flat, dtype)
    method = 'zlib_direct'
    
    return {
        's': shape,
        'l': len(palette),
        'p': palette_compressed,
        'i': indices_compressed,
        'd': dtype_name,  # Store dtype for decompression
        #'method': method
    }

def compress_indices_simple_optimized(indices_data, dtype=np.uint8):
    """
    Use optimal dtype instead of always uint16.
    Handles both lists and numpy arrays.
    """
    if isinstance(indices_data, np.ndarray):
        # Already numpy array
        if indices_data.dtype != dtype:
            indices = indices_data.astype(dtype)
        else:
            indices = indices_data
    else:
        # Convert list to numpy array
        indices = np.array(indices_data, dtype=dtype)
    
    raw_bytes = indices.tobytes()
    return zlib.compress(raw_bytes, level=9)






import math
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist


def print_compressed_data_types(compressed_data, name="Compressed Data"):
    """
    Print detailed type information about compressed data
    """
    print(f"\n{'='*60}")
    print(f"DATA TYPE ANALYSIS: {name}")
    print(f"{'='*60}")
    
    if not isinstance(compressed_data, dict):
        print(f"ERROR: Expected dict, got {type(compressed_data)}")
        return
    
    for key, value in compressed_data.items():
        # Handle special cases
        if key == 'palette':
            if isinstance(value, list):
                print(f"{key:20}: list of {len(value)} colors")
                if value:
                    first_color = value[0]
                    print(f"  First color type: {type(first_color)}, length: {len(first_color)}")
                    print(f"  Color values: {first_color[:3]}...")
            elif isinstance(value, np.ndarray):
                print(f"{key:20}: numpy array {value.shape}, dtype={value.dtype}")
            else:
                print(f"{key:20}: {type(value)}")
        
        elif key == 'indices':
            if isinstance(value, list):
                print(f"{key:20}: list of {len(value)} indices")
                if value:
                    max_val = max(value)
                    min_val = min(value)
                    print(f"  Value range: {min_val} to {max_val}")
                    
                    # Determine optimal dtype
                    if max_val < 256:
                        optimal_dtype = 'uint8'
                    elif max_val < 65536:
                        optimal_dtype = 'uint16'
                    else:
                        optimal_dtype = 'uint32'
                    print(f"  Optimal dtype: {optimal_dtype}")
                    
                    # Estimate memory usage
                    list_bytes = len(value) * 28  # Approx bytes per Python int
                    optimal_bytes = len(value) * (1 if optimal_dtype == 'uint8' else 2 if optimal_dtype == 'uint16' else 4)
                    print(f"  Memory: {list_bytes:,}B (list) → {optimal_bytes:,}B ({optimal_dtype})")
                    
            elif isinstance(value, np.ndarray):
                print(f"{key:20}: numpy array {value.shape}, dtype={value.dtype}")
                print(f"  Value range: {value.min()} to {value.max()}")
                
                # Check if can be downgraded
                max_val = value.max()
                if max_val < 256 and value.dtype != np.uint8:
                    print(f"  ⚠️  Can be downgraded: {value.dtype} → uint8")
                elif max_val < 65536 and value.dtype not in [np.uint16, np.uint8]:
                    print(f"  ⚠️  Can be downgraded: {value.dtype} → uint16")
            else:
                print(f"{key:20}: {type(value)}")
        
        else:
            # For other keys
            if isinstance(value, (list, tuple)):
                print(f"{key:20}: {type(value).__name__} of {len(value)} items")
            elif isinstance(value, np.ndarray):
                print(f"{key:20}: numpy array {value.shape}, dtype={value.dtype}")
            else:
                print(f"{key:20}: {type(value).__name__}: {repr(value)[:50]}...")
    
    # Calculate total estimated size
    print(f"\n{'='*60}")
    print("MEMORY ESTIMATE:")
    total_size = 0
    
    for key, value in compressed_data.items():
        if key == 'indices' and isinstance(value, list):
            total_size += len(value) * 28  # Approx 28 bytes per Python int
        elif key == 'indices' and isinstance(value, np.ndarray):
            total_size += value.nbytes
        elif key == 'palette' and isinstance(value, list):
            total_size += len(value) * 3  # 3 bytes per RGB color
        elif key == 'palette' and isinstance(value, np.ndarray):
            total_size += value.nbytes
        elif isinstance(value, str):
            total_size += len(value)
        elif isinstance(value, (int, float)):
            total_size += 28  # Approx for Python numbers
    
    print(f"Total estimated size: {total_size:,} bytes")
    print(f"{'='*60}")


def optimize_compressed_dtype(compressed_data):
    """
    Downgrade uint32 indices to uint8 or uint16 when possible.
    
    Args:
        compressed_data: Dictionary with 'indices' and 'palette'
    
    Returns:
        Optimized dictionary with smallest dtype
    """
    if 'indices' not in compressed_data:
        return compressed_data
    
    indices = compressed_data['indices']
    
    # Convert to numpy array if it's a list
    if isinstance(indices, list):
        indices_array = np.array(indices)
    elif isinstance(indices, np.ndarray):
        indices_array = indices.copy()
    else:
        print(f"Warning: Unknown indices type: {type(indices)}")
        return compressed_data
    
    # Get current dtype info
    current_dtype = indices_array.dtype
    max_value = indices_array.max() if len(indices_array) > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"DTYPE OPTIMIZATION")
    print(f"{'='*60}")
    print(f"Current dtype: {current_dtype}")
    print(f"Max index value: {max_value}")
    
    # Determine optimal dtype
    if max_value < 256:
        optimal_dtype = np.uint8
        dtype_name = 'uint8'
        bytes_per_index = 1
    elif max_value < 65536:
        optimal_dtype = np.uint16
        dtype_name = 'uint16'
        bytes_per_index = 2
    else:
        optimal_dtype = np.uint32
        dtype_name = 'uint32'
        bytes_per_index = 4
    
    # Calculate savings
    current_bytes = indices_array.nbytes
    if current_dtype == np.uint8:
        current_bytes_per_index = 1
    elif current_dtype == np.uint16:
        current_bytes_per_index = 2
    elif current_dtype == np.uint32:
        current_bytes_per_index = 4
    else:
        current_bytes_per_index = 4  # Default assumption
    
    optimized_bytes = len(indices_array) * bytes_per_index
    savings = current_bytes - optimized_bytes
    
    print(f"Optimal dtype: {dtype_name}")
    print(f"Current size: {current_bytes:,} bytes ({current_bytes_per_index} bytes/index)")
    print(f"Optimized size: {optimized_bytes:,} bytes ({bytes_per_index} bytes/index)")
    print(f"Savings: {savings:,} bytes ({savings/current_bytes*100:.1f}% smaller)")
    
    # Only convert if there's a benefit
    if optimal_dtype != current_dtype:
        print(f"✅ Converting: {current_dtype} → {optimal_dtype}")
        indices_optimized = indices_array.astype(optimal_dtype)
        
        # Update the compressed data
        optimized_data = compressed_data.copy()
        optimized_data['indices'] = indices_optimized.tolist()  # Or keep as array
        optimized_data['indices_dtype'] = dtype_name
        optimized_data['indices_optimized'] = True
        
        # Update palette count if available
        if 'palette' in optimized_data:
            actual_colors = len(optimized_data['palette'])
            optimized_data['actual_colors'] = actual_colors
            print(f"Palette size: {actual_colors} colors")
        
        return optimized_data
    else:
        print(f"✓ Already optimal dtype: {current_dtype}")
        return compressed_data



