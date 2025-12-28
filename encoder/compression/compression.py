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
        f.write(b'PQIC')  # Magic number: Palette Quantized Image Compressed
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