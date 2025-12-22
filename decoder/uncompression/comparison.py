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
























from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def calculate_adaptive_quality_metrics(original, reconstructed):
    """
    Calculate quality metrics with adaptive outlier exclusion.
    Automatically detects and excludes outliers based on error distribution.
    """
    # Convert to float
    original_f = original.astype(np.float32)
    reconstructed_f = reconstructed.astype(np.float32)
    
    # Calculate per-pixel max error (worst channel error)
    abs_error = np.abs(original_f - reconstructed_f)
    max_error_per_pixel = np.max(abs_error, axis=2).flatten()  # Shape: (h*w)
    
    # ==============================================
    # 1. ANALYZE ERROR DISTRIBUTION
    # ==============================================
    error_stats = {
        'min': float(np.min(max_error_per_pixel)),
        'max': float(np.max(max_error_per_pixel)),
        'mean': float(np.mean(max_error_per_pixel)),
        'median': float(np.median(max_error_per_pixel)),
        'std': float(np.std(max_error_per_pixel)),
        'q75': float(np.percentile(max_error_per_pixel, 75)),
        'q90': float(np.percentile(max_error_per_pixel, 90)),
        'q95': float(np.percentile(max_error_per_pixel, 95)),
        'q99': float(np.percentile(max_error_per_pixel, 99))
    }
    
    # ==============================================
    # 2. DETECT OUTLIERS USING MULTIPLE METHODS
    # ==============================================
    outlier_masks = {}
    
    # Method 1: IQR (Interquartile Range) - standard statistical method
    Q1 = np.percentile(max_error_per_pixel, 25)
    Q3 = np.percentile(max_error_per_pixel, 75)
    IQR = Q3 - Q1
    iqr_threshold = Q3 + 2.5 * IQR  # Conservative: 2.5×IQR
    outlier_masks['iqr'] = max_error_per_pixel > iqr_threshold
    
    # Method 2: Z-score (standard deviations from mean)
    z_scores = (max_error_per_pixel - error_stats['mean']) / error_stats['std']
    z_threshold = 3.0  # 3 standard deviations
    outlier_masks['zscore'] = np.abs(z_scores) > z_threshold
    
    # Method 3: Percentile-based (top 1% errors)
    percentile_threshold = np.percentile(max_error_per_pixel, 99)
    outlier_masks['percentile'] = max_error_per_pixel > percentile_threshold
    
    # Method 4: Adaptive threshold based on error distribution
    # If distribution is skewed, use median-based threshold
    if error_stats['mean'] > error_stats['median'] * 1.5:  # Right-skewed
        adaptive_threshold = error_stats['median'] + 3 * error_stats['std']
    else:
        adaptive_threshold = error_stats['mean'] + 2.5 * error_stats['std']
    outlier_masks['adaptive'] = max_error_per_pixel > adaptive_threshold
    
    # ==============================================
    # 3. CHOOSE BEST OUTLIER DETECTION METHOD
    # ==============================================
    # Select method that excludes reasonable amount (1-10% typically)
    best_method = None
    best_mask = None
    
    for method_name, mask in outlier_masks.items():
        outlier_count = np.sum(mask)
        outlier_pct = outlier_count / len(max_error_per_pixel) * 100
        
        # Good outlier detection: excludes 0.1% to 10% of pixels
        if 0.1 <= outlier_pct <= 10.0:
            best_method = method_name
            best_mask = mask
            break
    
    # If no method found in ideal range, use adaptive
    if best_method is None:
        best_method = 'adaptive'
        best_mask = outlier_masks['adaptive']
    
    outlier_count = np.sum(best_mask)
    outlier_pct = outlier_count / len(max_error_per_pixel) * 100
    inlier_mask = ~best_mask
    
    # ==============================================
    # 4. CALCULATE METRICS WITH/WITHOUT OUTLIERS
    # ==============================================
    metrics = {
        'error_distribution': error_stats,
        'outlier_detection': {
            'method': best_method,
            'threshold': float({
                'iqr': iqr_threshold,
                'zscore': error_stats['mean'] + z_threshold * error_stats['std'],
                'percentile': percentile_threshold,
                'adaptive': adaptive_threshold
            }[best_method]),
            'outlier_count': int(outlier_count),
            'outlier_percentage': float(outlier_pct),
            'inlier_count': int(len(max_error_per_pixel) - outlier_count),
            'inlier_percentage': float(100 - outlier_pct)
        }
    }
    
    # ==============================================
    # 5. CALCULATE ALL METRICS
    # ==============================================
    # A) WITH ALL PIXELS
    mse_all = np.mean((original_f - reconstructed_f) ** 2)
    metrics['all_pixels'] = {
        'psnr': 10 * np.log10(255*255/mse_all) if mse_all > 0 else float('inf'),
        'mse': float(mse_all),
        'rmse': float(np.sqrt(mse_all)),
        'mae': float(np.mean(abs_error)),
        'max_error': error_stats['max'],
        'pixel_count': int(len(max_error_per_pixel))
    }
    
    # B) WITHOUT OUTLIERS (if any outliers detected)
    if outlier_count > 0 and outlier_count < len(max_error_per_pixel):
        # Get inlier pixels
        original_inliers = original_f.reshape(-1, 3)[inlier_mask]
        reconstructed_inliers = reconstructed_f.reshape(-1, 3)[inlier_mask]
        
        if len(original_inliers) > 0:
            mse_inliers = np.mean((original_inliers - reconstructed_inliers) ** 2)
            
            metrics['without_outliers'] = {
                'psnr': 10 * np.log10(255*255/mse_inliers) if mse_inliers > 0 else float('inf'),
                'mse': float(mse_inliers),
                'rmse': float(np.sqrt(mse_inliers)),
                'mae': float(np.mean(np.abs(original_inliers - reconstructed_inliers))),
                'max_error': float(np.max(np.abs(original_inliers - reconstructed_inliers))),
                'pixel_count': int(len(original_inliers))
            }
    
    # C) PERCENTILE-BASED METRICS
    for percentile in [99, 95, 90, 75]:
        threshold = np.percentile(max_error_per_pixel, percentile)
        mask = max_error_per_pixel <= threshold
        
        orig_percentile = original_f.reshape(-1, 3)[mask]
        recon_percentile = reconstructed_f.reshape(-1, 3)[mask]
        
        if len(orig_percentile) > 0:
            mse_percentile = np.mean((orig_percentile - recon_percentile) ** 2)
            
            metrics[f'percentile_{percentile}'] = {
                'psnr': 10 * np.log10(255*255/mse_percentile) if mse_percentile > 0 else float('inf'),
                'mse': float(mse_percentile),
                'max_error_included': float(threshold),
                'pixel_count': int(len(orig_percentile)),
                'percentage': float(percentile)
            }
    
    # ==============================================
    # 6. CALCULATE SSIM (with and without outliers)
    # ==============================================
    try:
        metrics['ssim'] = {}
        metrics['ssim']['full'] = float(structural_similarity(
            original, reconstructed, data_range=255, channel_axis=2, win_size=7
        ))
        
        if outlier_count > 0 and outlier_count < len(max_error_per_pixel):
            # Create masked version for SSIM
            h, w = original.shape[:2]
            original_masked = original.copy()
            reconstructed_masked = reconstructed.copy()
            
            # Reshape outlier mask to 2D
            outlier_mask_2d = best_mask.reshape(h, w)
            
            # Set outliers to neutral gray for SSIM
            original_masked[outlier_mask_2d] = [128, 128, 128]
            reconstructed_masked[outlier_mask_2d] = [128, 128, 128]
            
            metrics['ssim']['without_outliers'] = float(structural_similarity(
                original_masked, reconstructed_masked, data_range=255, channel_axis=2, win_size=7
            ))
    except:
        metrics['ssim'] = {'full': 0}
    
    # ==============================================
    # 7. VISUALIZE ERROR DISTRIBUTION
    # ==============================================
    metrics['error_histogram'] = {
        'bins': np.histogram(max_error_per_pixel, bins=50)[0].tolist(),
        'bin_edges': np.histogram(max_error_per_pixel, bins=50)[1].tolist()
    }
    
    return metrics


def print_adaptive_metrics(metrics, original_shape):
    """Print adaptive metrics analysis."""
    h, w = original_shape[:2]
    total_pixels = h * w
    
    print("\n" + "="*70)
    print("ADAPTIVE QUALITY METRICS WITH OUTLIER DETECTION")
    print("="*70)
    
    # Error distribution
    ed = metrics['error_distribution']
    print(f"\n📈 ERROR DISTRIBUTION ANALYSIS:")
    print(f"   Total pixels: {total_pixels:,}")
    print(f"   Min error:    {ed['min']:8.2f}")
    print(f"   Max error:    {ed['max']:8.2f}  ← LIKELY OUTLIERS")
    print(f"   Mean error:   {ed['mean']:8.2f}")
    print(f"   Median error: {ed['median']:8.2f}")
    print(f"   Std dev:      {ed['std']:8.2f}")
    print(f"   75th %ile:    {ed['q75']:8.2f}")
    print(f"   90th %ile:    {ed['q90']:8.2f}")
    print(f"   95th %ile:    {ed['q95']:8.2f}")
    print(f"   99th %ile:    {ed['q99']:8.2f}")
    
    # Outlier detection
    od = metrics['outlier_detection']
    print(f"\n🎯 OUTLIER DETECTION ({od['method'].upper()}):")
    print(f"   Threshold:    {od['threshold']:8.2f}")
    print(f"   Outliers:     {od['outlier_count']:8,} pixels ({od['outlier_percentage']:.2f}%)")
    print(f"   Inliers:      {od['inlier_count']:8,} pixels ({od['inlier_percentage']:.2f}%)")
    
    # Metrics comparison
    print(f"\n📊 METRICS COMPARISON:")
    
    # All pixels
    allp = metrics['all_pixels']
    print(f"   ALL PIXELS ({allp['pixel_count']:,}):")
    print(f"     PSNR:  {allp['psnr']:8.2f} dB")
    print(f"     MSE:   {allp['mse']:8.2f}")
    print(f"     MAE:   {allp['mae']:8.2f}")
    
    # Without outliers
    if 'without_outliers' in metrics:
        wo = metrics['without_outliers']
        improvement = wo['psnr'] - allp['psnr']
        
        print(f"\n   WITHOUT OUTLIERS ({wo['pixel_count']:,}):")
        print(f"     PSNR:  {wo['psnr']:8.2f} dB  (+{improvement:.2f} dB)")
        print(f"     MSE:   {wo['mse']:8.2f}  ({wo['mse']/allp['mse']*100:.1f}% of original)")
        print(f"     MAE:   {wo['mae']:8.2f}  ({wo['mae']/allp['mae']*100:.1f}% of original)")
        print(f"     Max:   {wo['max_error']:8.2f}")
    
    # Percentile metrics
    print(f"\n📐 PERCENTILE METRICS:")
    for percentile in [99, 95, 90, 75]:
        key = f'percentile_{percentile}'
        if key in metrics:
            pm = metrics[key]
            print(f"   Top {100-percentile}% excluded ({pm['pixel_count']:,} pixels):")
            print(f"     PSNR: {pm['psnr']:8.2f} dB")
    
    # SSIM
    if 'ssim' in metrics:
        print(f"\n🖼️  STRUCTURAL SIMILARITY (SSIM):")
        print(f"   Full image:      {metrics['ssim'].get('full', 0):.4f}")
        if 'without_outliers' in metrics['ssim']:
            print(f"   Without outliers: {metrics['ssim']['without_outliers']:.4f}")
    
    print("="*70)


















# Main execution
def main():
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