import cv2
import numpy as np
import os

def compress_with_jpeg(image_path, quality=85, output_path=None):
    """
    Compress an image with JPEG at specified quality.
    
    Args:
        image_path: Path to input image (PNG, JPG, etc.)
        quality: JPEG quality (0-100, higher = better quality)
        output_path: Optional output path (default: add _q{quality}.jpg)
    
    Returns:
        compressed_image, file_size_bytes, compression_ratio
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Generate output path if not provided
    if output_path is None:
        base_name = os.path.splitext(image_path)[0]
        output_path = f"{base_name}_q{quality}.jpg"
    
    # Save with JPEG compression
    cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    
    # Get file sizes
    original_size = os.path.getsize(image_path)
    compressed_size = os.path.getsize(output_path)
    compression_ratio = original_size / compressed_size
    
    # Load back to return image array
    compressed_img = cv2.imread(output_path)
    
    print(f"JPEG Compression Results (Quality: {quality}):")
    print(f"  Original size: {original_size:,} bytes")
    print(f"  JPEG size: {compressed_size:,} bytes")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    print(f"  Saved to: {output_path}")
    
    return compressed_img, compressed_size, compression_ratio

compress_with_jpeg(
    image_path='images/png/adidas.png',
    quality=90,
    output_path='images/jpg/adidas_compressed.jpg'
)