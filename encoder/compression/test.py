import cv2
import math
import matplotlib.pyplot as plt
import numpy as np

from encoder.ROI.roi import get_regions, extract_regions

from encoder.subregions.split_score import calculate_split_score, normalize_result
from encoder.subregions.slic import enhanced_slic_with_texture, extract_slic_segment_boundaries, visualize_split_analysis
from encoder.subregions.visualize import plot_regions


from encoder.compression.subregions import subregion_quantization
from encoder.compression.regions import region_quantization
from encoder.compression.image import quantize_image


from decoder.uncompression.uncompression import decompress_color_quantization


def save_picture(image_seg_compression):

    from PIL import Image
    import numpy as np

    # Assuming reconstruction_result is your decompressed result
    reconstruction_result = decompress_color_quantization(image_seg_compression)

    # Extract the image
    reconstructed_image = reconstruction_result['image']  # This is a numpy array

    # Convert numpy array to PIL Image and save
    pil_image = Image.fromarray(reconstructed_image)
    pil_image.save('reconstructed_image.jpg', quality=25)  # quality 1-100

    print(f"✅ Image saved as 'reconstructed_image.jpg'")
    print(f"   Size: {reconstructed_image.shape[1]}x{reconstructed_image.shape[0]}")

def save_compression(image_seg_compression, filename):
    print(f"\n{'='*60}")
    print(f"FINAL LOSS LESS COMPRESSION")
    print(f"{'='*60}")
    
    # Get your final data
    final_data = image_seg_compression  # or clustered_result
    
    # Extract components
    shape = final_data['shape']
    palette = final_data['palette']
    indices_flat = final_data['indices']
    
    # Convert to matrix
    h, w = shape
    indices_matrix = np.array(indices_flat).reshape(h, w)
    
    print(f"Compressing: {w}x{h} image, {len(palette)} colors")
    
    # Compress
    from encoder.compression.compression import lossless_compress_optimized, save_compressed
    
    compressed_data = lossless_compress_optimized(palette, indices_matrix, shape)
    
    # Save
    file_size = save_compressed(compressed_data, filename)
    
    # Stats
    original_size = h * w * 3
    compression_ratio = original_size / file_size
    
    print(f"✅ Saved: {filename}")
    print(f"   Original: {original_size:,} bytes")
    print(f"   Compressed: {file_size:,} bytes")
    print(f"   Ratio: {compression_ratio:.2f}:1")
    print(f"   Savings: {(1 - file_size/original_size)*100:.1f}%")


if __name__ == "__main__":

    image_name = 'images/png/8.png'
    image = cv2.imread(image_name)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    unified, regions, roi_image, nonroi_image, roi_mask, nonroi_mask = get_regions(image_rgb)
    roi_regions, nonroi_regions = extract_regions(image_rgb, roi_mask, nonroi_mask)

    print(f"Found {len(roi_regions)} ROI regions")
    print(f"Found {len(nonroi_regions)} non-ROI regions")

    # Display some statistics
    print("\nROI Regions (sorted by area):")
    for i, region in enumerate(sorted(roi_regions, key=lambda x: x['area'], reverse=True)[:5]):
        print(f"  Region {i+1}: Area = {region['area']} pixels")

    print("\nNon-ROI Regions (sorted by area):")
    for i, region in enumerate(sorted(nonroi_regions, key=lambda x: x['area'], reverse=True)[:5]):
        print(f"  Region {i+1}: Area = {region['area']} pixels")



    roi_quality=100
    nonroi_quality=100
   


    ROI_subregions_components=subregion_quantization(image_rgb, roi_regions, quality=roi_quality, subregion_type="ROI", debug=False)
    nonROI_subregions_components=subregion_quantization(image_rgb, nonroi_regions, quality=nonroi_quality, subregion_type="nonROI", debug=False)
    


    
    """
    Second phase of hierarchical clustering:
    Clustering colors between each ROI / nonROI
    """

    roi_region_quality=roi_quality*2
    nonroi_region_quality=nonroi_quality*2

    if roi_region_quality>100: roi_region_quality=100
    if nonroi_region_quality>100: nonroi_region_quality=100

    original_image_height, original_image_width, _ = image_rgb.shape

    try: roi_components=region_quantization(ROI_subregions_components, quality=roi_region_quality, original_image_height=original_image_height, original_image_width=original_image_width)
    except: roi_components=[]
    
    try: nonroi_components=region_quantization(nonROI_subregions_components, quality=nonroi_region_quality, original_image_height=original_image_height, original_image_width=original_image_width)
    except: nonroi_components=[]

    image_components=roi_components+nonroi_components

    


    """
    Final section of clustering: whole image
    """

    image_quality=roi_region_quality+nonroi_region_quality
    if image_quality>100: image_quality=100

    image_seg_compression=quantize_image(image_components, quality=image_quality, original_image_height=original_image_height, original_image_width=original_image_width)




    savePicture=False
    if savePicture: save_picture(image_seg_compression)

    saveCompression = True
    if saveCompression: save_compression(image_seg_compression, filename="images/rhccq/compressed_8.rhccq")