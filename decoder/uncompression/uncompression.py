import numpy as np
import zlib
import struct
import pickle

# ============================================================================
# DECOMPRESSION
# ============================================================================

def decompress_palette(palette_data, palette_size):
    """
    Decompress palette.
    
    Args:
        palette_data: Compressed palette bytes
        palette_size: Number of colors in palette
    
    Returns:
        list: List of RGB tuples
    """
    decompressed = zlib.decompress(palette_data)
    palette_np = np.frombuffer(decompressed, dtype=np.uint8)
    palette_np = palette_np.reshape(palette_size, 3)
    return palette_np.tolist()


def decompress_indices_rle_huffman(indices_data, total_pixels):
    """
    Decompress indices that were compressed with RLE + Huffman.
    
    Args:
        indices_data: Compressed indices bytes
        total_pixels: Total number of pixels (h * w)
    
    Returns:
        list: Flat list of indices
    """
    # STEP 1: Decompress Huffman (zlib)
    rle_data = zlib.decompress(indices_data)
    
    # STEP 2: Decode RLE
    indices = []
    i = 0
    
    while i < len(rle_data):
        # Read [value, run_length]
        value, run_length = struct.unpack('<HH', rle_data[i:i+4])
        i += 4
        
        # Expand run
        indices.extend([value] * run_length)
    
    return indices




def lossless_decompress(compressed_data):
    """
    Main decompression function.
    
    Args:
        compressed_data: Output from lossless_compress()
    
    Returns:
        tuple: (palette, indices_list, shape)
    """
    # Extract data
    shape = compressed_data['s']
    palette_size = compressed_data['l']
    palette_data = compressed_data['p']
    indices_data = compressed_data['i']
    
    # Get dtype from metadata (added in lossless_compress_optimized)
    dtype_str = compressed_data.get('d', 'uint16')  # Default to uint16 for backward compatibility
    #method = compressed_data.get('method', 'zlib_direct')

    method=""
    
    # Decompress palette
    palette = decompress_palette(palette_data, palette_size)
    
    # Decompress indices
    h, w = shape
    total_pixels = h * w
    
    if method == 'rle_huffman':
        indices_list = decompress_indices_rle_huffman(indices_data, total_pixels)
    else:  # zlib_direct
        indices_list = decompress_indices_simple(indices_data, total_pixels, dtype_str)
    
    return palette, indices_list, shape

def decompress_indices_simple(indices_data, total_pixels, dtype_str='uint16'):
    """
    Decompress indices that were compressed with direct zlib.
    
    Args:
        indices_data: Compressed indices bytes
        total_pixels: Total number of pixels (h * w)
        dtype_str: String indicating dtype ('uint8', 'uint16', 'uint32')
    
    Returns:
        list: Flat list of indices
    """
    decompressed = zlib.decompress(indices_data)
    
    # Use correct dtype based on metadata
    if dtype_str == 'uint8':
        dtype = np.uint8
    elif dtype_str == 'uint16':
        dtype = np.uint16
    elif dtype_str == 'uint32':
        dtype = np.uint32
    else:
        # Fallback: try to infer from data size
        bytes_per_pixel = len(decompressed) / total_pixels if total_pixels > 0 else 2
        
        if bytes_per_pixel <= 1:
            dtype = np.uint8
        elif bytes_per_pixel <= 2:
            dtype = np.uint16
        else:
            dtype = np.uint32
    
    indices_np = np.frombuffer(decompressed, dtype=dtype)
    return indices_np.tolist()

def load_compressed(filename):
    """
    Load compressed data from file.
    
    Args:
        filename: Path to compressed file
    
    Returns:
        dict: Compressed data package
    """
    with open(filename, 'rb') as f:
        # Read header
        magic = f.read(5)
        if magic != b'RHCCQ':
            raise ValueError("Invalid file format")
        
        size = struct.unpack('<I', f.read(4))[0]
        compressed = f.read(size)
    
    # Decompress and deserialize
    serialized = zlib.decompress(compressed)
    return pickle.loads(serialized)





def decompress_color_quantization(compressed_data):
    """
    Decompress region compressed with color quantization.
    Works with the new compression format.
    
    Args:
        compressed_data: Dictionary containing compressed data with keys:
                        - 'top_left': (y, x) position
                        - 'shape': (h, w) dimensions
                        - 'palette_size': number of colors
                        - 'palette_data': compressed palette bytes
                        - 'indices_data': compressed indices bytes
                        - 'method': compression method used
                        - 'quality': (optional) quality parameter
    
    Returns:
        dict: Contains 'image', 'top_left', 'shape', 'method', 'quality'
    """
    # Check if this is already decompressed data (tuple) or compressed dict
    if isinstance(compressed_data, tuple):
        # Already decompressed: (palette, indices_list, shape)
        palette, indices_list, shape = compressed_data
        top_left = (0, 0)  # Default if not available
        quality = 50
        h, w = shape
    else:
        # Dictionary with compressed data
        # Extract metadata
        top_left = compressed_data.get('top_left', (0, 0))
        h, w = compressed_data['shape']
        quality = compressed_data.get('quality', 50)
        
        # Decompress the actual data
        palette, indices_list, shape = lossless_decompress(compressed_data)
    
    # Convert to numpy arrays
    palette_np = np.array(palette, dtype=np.uint8)
    
    # Determine correct dtype based on palette size
    if len(palette_np) > 256:
        dtype = np.uint16
    else:
        dtype = np.uint8
    
    # Convert indices list to array and reshape
    try:
        index_array = np.array(indices_list, dtype=dtype).reshape(h, w)
    except (OverflowError, ValueError):
        # Fallback to uint16 if there's an issue
        print(f"Warning: Issue with {dtype}, falling back to uint16")
        index_array = np.array(indices_list, dtype=np.uint16).reshape(h, w)
    
    # Reconstruct image by mapping indices to palette colors
    reconstructed = palette_np[index_array].reshape(h, w, 3)
    
    # Return with placement info
    return {
        'image': reconstructed,
        'top_left': top_left,
        'shape': (h, w),
        'method': 'color_quantization',
        'quality': quality
    }










def partial_decompress_color_quantization(compressed_data):
    """
    Decompress region compressed with color quantization.
    Works with the new compression format.
    
    Args:
        compressed_data: Dictionary containing compressed data with keys:
                        - 'top_left': (y, x) position
                        - 'shape': (h, w) dimensions
                        - 'palette_size': number of colors
                        - 'palette_data': compressed palette bytes
                        - 'indices_data': compressed indices bytes
                        - 'method': compression method used
                        - 'quality': (optional) quality parameter
    
    Returns:
        dict: Contains 'image', 'top_left', 'shape', 'method', 'quality'
    """
    # Check if this is already decompressed data (tuple) or compressed dict
    if isinstance(compressed_data, tuple):
        # Already decompressed: (palette, indices_list, shape)
        palette, indices_list, shape = compressed_data
        top_left = (0, 0)  # Default if not available
        quality = 50
        h, w = shape
    else:
        # Dictionary with compressed data
        # Extract metadata
        top_left = compressed_data.get('top_left', (0, 0))
        h, w = compressed_data['shape']
        quality = compressed_data.get('quality', 50)

        palette, indices_list, shape = compressed_data["palette"], compressed_data["indices"], compressed_data["shape"]
        
    
    # Convert to numpy arrays
    palette_np = np.array(palette, dtype=np.uint8)
    
    # Determine correct dtype based on palette size
    if len(palette_np) > 256:
        dtype = np.uint16
    else:
        dtype = np.uint8
    
    # Convert indices list to array and reshape
    try:
        index_array = np.array(indices_list, dtype=dtype).reshape(h, w)
    except (OverflowError, ValueError):
        # Fallback to uint16 if there's an issue
        print(f"Warning: Issue with {dtype}, falling back to uint16")
        index_array = np.array(indices_list, dtype=np.uint16).reshape(h, w)
    
    # Reconstruct image by mapping indices to palette colors
    reconstructed = palette_np[index_array].reshape(h, w, 3)
    
    # Return with placement info
    return {
        'image': reconstructed,
        'top_left': top_left,
        'shape': (h, w),
        'method': 'color_quantization',
        'quality': quality
    }

