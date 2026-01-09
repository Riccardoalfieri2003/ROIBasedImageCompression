import cv2
from decoder.uncompression.uncompression import lossless_decompress, load_compressed, decompress_color_quantization
from decoder.uncompression.comparison import calculate_quality_metrics, create_difference_visualization, print_quality_report, plot_comparison, calculate_adaptive_quality_metrics, print_adaptive_metrics


def apply_edge_preserving_blur(image, blur_strength=3, edge_preservation=10):
    """
    Bilateral filter - blurs while preserving edges.
    Great for compression artifacts!
    
    Args:
        blur_strength: d parameter (larger = more blur)
        edge_preservation: sigmaColor (larger = more edge preservation)
    """
    # Convert to float32 for better precision
    image_float = image.astype(np.float32) / 255.0
    
    # Apply bilateral filter
    blurred = cv2.bilateralFilter(
        image_float, 
        d=blur_strength,
        sigmaColor=edge_preservation/255.0,
        sigmaSpace=edge_preservation
    )
    
    # Convert back to uint8
    return (blurred * 255).astype(np.uint8)


# Tested on Kodak Lossless True Color Image Suite
if __name__ == "__main__":

    # Load images
    original_path = r"C:\Users\rical\OneDrive\Desktop\Wallpaper\Napoli.png"
    reconstructed_path = 'images/rhccq/Napoli_compressed.rhccq'

    # Load original image
    original_bgr = cv2.imread(original_path)
    original = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

    # Load and decompress
    loaded = load_compressed(reconstructed_path)
    compressed=lossless_decompress(loaded)
    palette_restored, indices_restored, shape_restored = lossless_decompress(loaded)
    
    debug=False
    if debug:
        print("\nDecompressed data:")
        print(f"Palette: {palette_restored}")
        print(f"Indices: {indices_restored}")
        print(f"Shape: {shape_restored}")
    

    reconstruction_result = decompress_color_quantization(compressed)
    reconstructed = reconstruction_result['image']  # This is a numpy array

    import cv2
    import numpy as np

    # Assuming reconstructed is a numpy array with shape (h, w, 3)
    #reconstructed = cv2.GaussianBlur(reconstructed, (81, 81), 0)
    #reconstructed = apply_edge_preserving_blur(reconstructed, blur_strength=5, edge_preservation=50)

    import matplotlib.pyplot as plt
    plt.title("reconstructed")
    plt.imshow(reconstructed)
    plt.show()

    #original, reconstructed = load_images(original_path, reconstructed_path)
    
    print(f"\nImage sizes:")
    print(f"  Original: {original.shape[1]}x{original.shape[0]}")
    print(f"  Reconstructed: {reconstructed.shape[1]}x{reconstructed.shape[0]}")
    
    # Calculate metrics
    print("\nCalculating quality metrics...")
    metrics=calculate_adaptive_quality_metrics(original, reconstructed)
    
    # Create difference visualizations
    print("Creating difference visualizations...")
    differences = create_difference_visualization(original, reconstructed)
    
    # Print report
    #print_quality_report(metrics)
    print_adaptive_metrics(metrics, original.shape)
    

    metrics = calculate_quality_metrics(original, reconstructed)
    # Create comprehensive plot
    print("\nGenerating comparison visualization...")
    plot_comparison(original, reconstructed, metrics, differences)