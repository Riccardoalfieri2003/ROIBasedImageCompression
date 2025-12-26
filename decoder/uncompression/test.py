import cv2
from decoder.uncompression.uncompression import lossless_decompress, load_compressed, decompress_color_quantization
from decoder.uncompression.comparison import calculate_quality_metrics, create_difference_visualization, print_quality_report, plot_comparison, calculate_adaptive_quality_metrics, print_adaptive_metrics

if __name__ == "__main__":
    # Example data

    # Load images
    original_path = 'images/komodo.jpg'
    reconstructed_path = 'compressed_komodo.hccq'  # Your saved reconstruction

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