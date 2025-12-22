import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import warnings
warnings.filterwarnings('ignore')

def load_images(original_path, reconstructed_path):
    """Load both images and convert to RGB."""
    # Load original
    original_bgr = cv2.imread(original_path)
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    
    # Load reconstructed
    reconstructed_bgr = cv2.imread(reconstructed_path)
    reconstructed_rgb = cv2.cvtColor(reconstructed_bgr, cv2.COLOR_BGR2RGB)
    
    # Ensure same size (resize reconstructed to match original if needed)
    if original_rgb.shape != reconstructed_rgb.shape:
        print(f"Warning: Image sizes don't match!")
        print(f"Original: {original_rgb.shape}, Reconstructed: {reconstructed_rgb.shape}")
        
        # Resize reconstructed to match original
        reconstructed_rgb = cv2.resize(reconstructed_rgb, 
                                      (original_rgb.shape[1], original_rgb.shape[0]))
        print(f"Resized reconstructed to: {reconstructed_rgb.shape}")
    
    return original_rgb, reconstructed_rgb

def calculate_quality_metrics(original, reconstructed):
    """Calculate various quality metrics."""
    # Convert to float for calculations
    original_f = original.astype(np.float32)
    reconstructed_f = reconstructed.astype(np.float32)
    
    metrics = {}
    
    # 1. PSNR (Peak Signal-to-Noise Ratio)
    try:
        metrics['psnr'] = peak_signal_noise_ratio(original, reconstructed, data_range=255)
    except:
        # Manual calculation
        mse = np.mean((original_f - reconstructed_f) ** 2)
        metrics['psnr'] = 10 * np.log10(255*255/mse) if mse > 0 else float('inf')
    
    # 2. SSIM (Structural Similarity Index)
    try:
        metrics['ssim'] = structural_similarity(original, reconstructed, 
                                               data_range=255, channel_axis=2,
                                               win_size=7)
    except:
        # Calculate per channel and average
        ssim_channels = []
        for i in range(3):
            try:
                ssim_ch = structural_similarity(original[..., i], reconstructed[..., i], 
                                               data_range=255, win_size=7)
                ssim_channels.append(ssim_ch)
            except:
                ssim_channels.append(0)
        metrics['ssim'] = np.mean(ssim_channels)
    
    # 3. MSE (Mean Squared Error)
    metrics['mse'] = np.mean((original_f - reconstructed_f) ** 2)
    
    # 4. RMSE (Root Mean Squared Error)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    
    # 5. MAE (Mean Absolute Error)
    metrics['mae'] = np.mean(np.abs(original_f - reconstructed_f))
    
    # 6. Maximum Absolute Error
    metrics['max_error'] = np.max(np.abs(original_f - reconstructed_f))
    
    # 7. Color channel errors
    for i, channel in enumerate(['R', 'G', 'B']):
        mse_ch = np.mean((original_f[..., i] - reconstructed_f[..., i]) ** 2)
        metrics[f'mse_{channel.lower()}'] = mse_ch
    
    return metrics

def create_difference_visualization(original, reconstructed):
    """Create visualizations of differences."""
    # Absolute difference
    diff_abs = np.abs(original.astype(float) - reconstructed.astype(float))
    diff_abs_normalized = (diff_abs / diff_abs.max() * 255).astype(np.uint8)
    
    # Squared difference (amplified for visibility)
    diff_squared = ((original.astype(float) - reconstructed.astype(float)) ** 2)
    diff_squared_normalized = (diff_squared / diff_squared.max() * 255).astype(np.uint8)
    
    # Perceptual difference (weighted by human sensitivity)
    # Human vision is most sensitive to green, then red, then blue
    weights = np.array([0.299, 0.587, 0.114])  # RGB to grayscale weights
    diff_weighted = np.sum(diff_abs * weights, axis=2)
    diff_weighted_normalized = (diff_weighted / diff_weighted.max() * 255).astype(np.uint8)
    
    # Error heatmap
    error_heatmap = cv2.applyColorMap(diff_weighted_normalized, cv2.COLORMAP_JET)
    
    return {
        'absolute': diff_abs_normalized,
        'squared': diff_squared_normalized,
        'weighted': diff_weighted_normalized,
        'heatmap': error_heatmap
    }

def plot_comparison(original, reconstructed, metrics, differences):
    """Create comprehensive comparison plot."""
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    # 1. Original image
    axes[0].imshow(original)
    axes[0].set_title('Original Image\n' + f'{original.shape[1]}x{original.shape[0]}')
    axes[0].axis('off')
    
    # 2. Reconstructed image
    axes[1].imshow(reconstructed)
    axes[1].set_title('Reconstructed Image')
    axes[1].axis('off')
    
    # 3. Side-by-side (split screen)
    h, w = original.shape[:2]
    side_by_side = np.zeros((h, w*2, 3), dtype=np.uint8)
    side_by_side[:, :w] = original
    side_by_side[:, w:] = reconstructed
    axes[2].imshow(side_by_side)
    axes[2].set_title('Side-by-side Comparison')
    axes[2].axvline(x=w, color='red', linestyle='--', linewidth=2)
    axes[2].axis('off')
    
    # 4. Absolute difference
    axes[3].imshow(differences['absolute'])
    axes[3].set_title('Absolute Difference')
    axes[3].axis('off')
    
    # 5. Squared difference (amplified)
    axes[4].imshow(differences['squared'])
    axes[4].set_title('Squared Difference (Amplified)')
    axes[4].axis('off')
    
    # 6. Weighted difference (perceptual)
    axes[5].imshow(differences['weighted'], cmap='hot')
    axes[5].set_title('Perceptual Difference (Hot)')
    axes[5].axis('off')
    
    # 7. Error heatmap
    axes[6].imshow(differences['heatmap'])
    axes[6].set_title('Error Heatmap')
    axes[6].axis('off')
    
    # 8. Histogram of errors
    error_flat = np.abs(original.astype(float) - reconstructed.astype(float)).flatten()
    axes[7].hist(error_flat, bins=50, color='blue', alpha=0.7, edgecolor='black')
    axes[7].set_title('Error Distribution')
    axes[7].set_xlabel('Absolute Error')
    axes[7].set_ylabel('Frequency')
    axes[7].grid(True, alpha=0.3)
    
    # 9. Per-channel error bars
    channels = ['R', 'G', 'B']
    mse_channels = [metrics[f'mse_{c.lower()}'] for c in channels]
    x_pos = np.arange(len(channels))
    axes[8].bar(x_pos, mse_channels, color=['red', 'green', 'blue'], alpha=0.7)
    axes[8].set_title('MSE per Channel')
    axes[8].set_xlabel('Color Channel')
    axes[8].set_ylabel('MSE')
    axes[8].set_xticks(x_pos)
    axes[8].set_xticklabels(channels)
    axes[8].grid(True, alpha=0.3, axis='y')
    
    # 10. Metrics table
    axes[9].axis('off')
    metrics_text = f"""
    Quality Metrics:
    ----------------
    PSNR: {metrics['psnr']:.2f} dB
    SSIM: {metrics['ssim']:.3f}
    MSE:  {metrics['mse']:.2f}
    RMSE: {metrics['rmse']:.2f}
    MAE:  {metrics['mae']:.2f}
    Max Error: {metrics['max_error']:.2f}
    
    Channel MSE:
      Red:   {metrics['mse_r']:.2f}
      Green: {metrics['mse_g']:.2f}
      Blue:  {metrics['mse_b']:.2f}
    """
    axes[9].text(0.1, 0.5, metrics_text, fontsize=10, 
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 11. SSIM map (if available)
    axes[10].axis('off')
    try:
        from skimage.metrics import structural_similarity
        ssim_map = structural_similarity(original, reconstructed, 
                                        data_range=255, channel_axis=2,
                                        win_size=7, full=True)[1]
        axes[10].imshow(ssim_map, cmap='viridis', vmin=0, vmax=1)
        axes[10].set_title('SSIM Map\n(Structural Similarity)')
        axes[10].axis('off')
    except:
        axes[10].text(0.5, 0.5, 'SSIM Map\nNot Available', 
                     ha='center', va='center', transform=axes[10].transAxes)
    
    # 12. Quality assessment
    axes[11].axis('off')
    
    # Quality rating based on PSNR
    psnr = metrics['psnr']
    if psnr > 40:
        rating = "Excellent"
        color = "green"
    elif psnr > 30:
        rating = "Good"
        color = "lightgreen"
    elif psnr > 20:
        rating = "Fair"
        color = "yellow"
    else:
        rating = "Poor"
        color = "red"
    
    # SSIM rating
    ssim = metrics['ssim']
    if ssim > 0.95:
        ssim_rating = "Excellent"
    elif ssim > 0.85:
        ssim_rating = "Good"
    elif ssim > 0.70:
        ssim_rating = "Fair"
    else:
        ssim_rating = "Poor"
    
    assessment_text = f"""
    Quality Assessment:
    -------------------
    PSNR: {psnr:.1f} dB → {rating}
    SSIM: {ssim:.3f} → {ssim_rating}
    
    Interpretation:
    • PSNR > 40 dB: Excellent
    • 30-40 dB: Good
    • 20-30 dB: Fair
    • < 20 dB: Poor
    
    • SSIM > 0.95: Excellent
    • 0.85-0.95: Good
    • 0.70-0.85: Fair
    • < 0.70: Poor
    """
    axes[11].text(0.1, 0.5, assessment_text, fontsize=9, 
                 verticalalignment='center', color=color,
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.suptitle('Image Quality Comparison: Original vs Reconstructed', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

def print_quality_report(metrics):
    """Print a detailed quality report."""
    print("\n" + "="*60)
    print("QUALITY ANALYSIS REPORT")
    print("="*60)
    
    print(f"\n📊 Objective Metrics:")
    print(f"   PSNR:        {metrics['psnr']:8.2f} dB")
    print(f"   SSIM:        {metrics['ssim']:8.3f}")
    print(f"   MSE:         {metrics['mse']:8.2f}")
    print(f"   RMSE:        {metrics['rmse']:8.2f}")
    print(f"   MAE:         {metrics['mae']:8.2f}")
    print(f"   Max Error:   {metrics['max_error']:8.2f}")
    
    print(f"\n🎨 Channel-wise MSE:")
    print(f"   Red:         {metrics['mse_r']:8.2f}")
    print(f"   Green:       {metrics['mse_g']:8.2f}")
    print(f"   Blue:        {metrics['mse_b']:8.2f}")
    
    # Quality assessment
    print(f"\n⭐ Quality Assessment:")
    
    # PSNR rating
    psnr = metrics['psnr']
    if psnr > 40:
        psnr_rating = "✅ EXCELLENT"
        psnr_desc = "Near-lossless quality"
    elif psnr > 30:
        psnr_rating = "✓ GOOD"
        psnr_desc = "Good quality, minor artifacts"
    elif psnr > 20:
        psnr_rating = "⚠ FAIR"
        psnr_desc = "Visible artifacts"
    else:
        psnr_rating = "❌ POOR"
        psnr_desc = "Strong artifacts, poor quality"
    
    print(f"   PSNR: {psnr_rating} - {psnr_desc}")
    
    # SSIM rating
    ssim = metrics['ssim']
    if ssim > 0.95:
        ssim_rating = "✅ EXCELLENT"
        ssim_desc = "Structurally identical"
    elif ssim > 0.85:
        ssim_rating = "✓ GOOD"
        ssim_desc = "Structurally similar"
    elif ssim > 0.70:
        ssim_rating = "⚠ FAIR"
        ssim_desc = "Structural differences"
    else:
        ssim_rating = "❌ POOR"
        ssim_desc = "Structurally different"
    
    print(f"   SSIM: {ssim_rating} - {ssim_desc}")
    print("="*60)





# Main execution
if __name__ == "__main__":
    # Load images
    original_path = 'images/Lenna.webp'
    reconstructed_path = 'images/reconstructed_image.jpg'  # Your saved reconstruction
    
    print("Loading images...")
    original, reconstructed = load_images(original_path, reconstructed_path)
    
    print(f"\nImage sizes:")
    print(f"  Original: {original.shape[1]}x{original.shape[0]}")
    print(f"  Reconstructed: {reconstructed.shape[1]}x{reconstructed.shape[0]}")
    
    # Calculate metrics
    print("\nCalculating quality metrics...")
    metrics = calculate_quality_metrics(original, reconstructed)
    
    # Create difference visualizations
    print("Creating difference visualizations...")
    differences = create_difference_visualization(original, reconstructed)
    
    # Print report
    print_quality_report(metrics)
    
    # Create comprehensive plot
    print("\nGenerating comparison visualization...")
    plot_comparison(original, reconstructed, metrics, differences)
    