
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import warnings
warnings.filterwarnings('ignore')

from decoder.uncompression.uncompression import lossless_decompress, load_compressed, decompress_color_quantization
from decoder.uncompression.comparison import calculate_quality_metrics, create_difference_visualization, print_quality_report, plot_comparison, calculate_adaptive_quality_metrics, print_adaptive_metrics

def load_images(original_path, reconstructed_path):
    """Load both images and convert to RGB."""
    # Load original
    original_bgr = cv2.imread(original_path)
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    
    # Load reconstructed
    loaded = load_compressed(reconstructed_path)
    compressed=lossless_decompress(loaded)
    reconstruction_result = decompress_color_quantization(compressed)
    reconstructed_rgb = reconstruction_result['image']  # This is a numpy array
    
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


def main():
    """Loop through all 24 test images and generate quality reports."""
    
    print("="*80)
    print("RHCCQ COMPRESSION QUALITY EVALUATION")
    print("Testing 24 Kodak images")
    print("="*80)
    
    all_metrics = []  # Store metrics for all images
    
    for i in range(1, 25):  # 1 to 24 inclusive
        print(f"\n{'='*60}")
        print(f"IMAGE {i:02d}/24")
        print(f"{'='*60}")
        
        try:
            # Construct file paths
            original_path = f'images/png/{i}.png'
            reconstructed_path = f'images/rhccq_20_10/compressed_{i}.rhccq'  # Adjust if needed
            
            # Load images
            print(f"Loading: {i}.png")
            original, reconstructed = load_images(original_path, reconstructed_path)
            
            if original is None or reconstructed is None:
                print(f"  ⚠️  Skipped: Could not load image {i}")
                continue
            
            # Print image info
            print(f"  Original: {original.shape[1]}x{original.shape[0]}")
            print(f"  Reconstructed: {reconstructed.shape[1]}x{reconstructed.shape[0]}")
            
            # Check dimensions match
            if original.shape != reconstructed.shape:
                print(f"  ⚠️  Warning: Dimension mismatch for image {i}")
                # Optional: resize reconstructed to match original
                reconstructed = cv2.resize(reconstructed, (original.shape[1], original.shape[0]))
            
            # Calculate metrics
            metrics = calculate_quality_metrics(original, reconstructed)
            metrics['image_id'] = i
            metrics['image_name'] = f'image_{i:02d}'
            metrics['dimensions'] = f"{original.shape[1]}x{original.shape[0]}"
            
            all_metrics.append(metrics)
            
            # Print individual report
            print_individual_report(metrics)
            
                
        except Exception as e:
            print(f"  ❌ Error processing image {i}: {str(e)}")
            continue
    
    # Print summary report for all images
    print_summary_report(all_metrics)

def print_individual_report(metrics):
    """Print detailed report for a single image."""
    print(f"\n📊 QUALITY METRICS for Image {metrics['image_id']}:")
    print(f"  PSNR:    {metrics['psnr']:.2f} dB")
    print(f"  SSIM:    {metrics['ssim']:.4f}")
    print(f"  MSE:     {metrics['mse']:.2f}")
    print(f"  MAE:     {metrics['mae']:.2f}")
    
    if 'bpp' in metrics:
        print(f"  BPP:     {metrics['bpp']:.2f} bits/pixel")
    
    # Quality interpretation
    psnr_quality = "Excellent" if metrics['psnr'] > 40 else \
                   "Good" if metrics['psnr'] > 35 else \
                   "Acceptable" if metrics['psnr'] > 30 else \
                   "Poor" if metrics['psnr'] > 25 else "Very Poor"
    
    ssim_quality = "Excellent" if metrics['ssim'] > 0.95 else \
                   "Good" if metrics['ssim'] > 0.90 else \
                   "Acceptable" if metrics['ssim'] > 0.85 else \
                   "Poor" if metrics['ssim'] > 0.80 else "Very Poor"
    
    print(f"  PSNR Quality: {psnr_quality}")
    print(f"  SSIM Quality: {ssim_quality}")

def print_summary_report(all_metrics):
    """Print summary statistics for all images."""
    if not all_metrics:
        print("\n❌ No metrics collected. Check your file paths.")
        return
    
    print("\n" + "="*80)
    print("SUMMARY REPORT - 24 Kodak Images")
    print("="*80)
    
    # Create DataFrame for easy analysis
    import pandas as pd
    
    df = pd.DataFrame(all_metrics)
    
    # Basic statistics
    print("\n📈 STATISTICAL SUMMARY:")
    print(f"  Images processed: {len(df)}")
    print(f"  PSNR Range: [{df['psnr'].min():.2f}, {df['psnr'].max():.2f}] dB")
    print(f"  PSNR Average: {df['psnr'].mean():.2f} ± {df['psnr'].std():.2f} dB")
    print(f"  SSIM Range: [{df['ssim'].min():.4f}, {df['ssim'].max():.4f}]")
    print(f"  SSIM Average: {df['ssim'].mean():.4f} ± {df['ssim'].std():.4f}")
    
    # Quality distribution
    print("\n🏆 QUALITY DISTRIBUTION:")
    
    # PSNR categories
    psnr_counts = {
        'Excellent (>40 dB)': (df['psnr'] > 40).sum(),
        'Good (35-40 dB)': ((df['psnr'] >= 35) & (df['psnr'] <= 40)).sum(),
        'Acceptable (30-35 dB)': ((df['psnr'] >= 30) & (df['psnr'] < 35)).sum(),
        'Poor (25-30 dB)': ((df['psnr'] >= 25) & (df['psnr'] < 30)).sum(),
        'Very Poor (<25 dB)': (df['psnr'] < 25).sum()
    }
    
    for category, count in psnr_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {category}: {count} images ({percentage:.1f}%)")
    
    # Top and bottom performers
    print("\n⭐ TOP PERFORMERS (Highest PSNR):")
    top_3 = df.nlargest(3, 'psnr')[['image_id', 'psnr', 'ssim']]
    for _, row in top_3.iterrows():
        print(f"  Image {row['image_id']:2f}: PSNR={row['psnr']:.2f} dB, SSIM={row['ssim']:.4f}")
    
    print("\n⚠️  LOWEST PERFORMERS (Lowest PSNR):")
    bottom_3 = df.nsmallest(3, 'psnr')[['image_id', 'psnr', 'ssim']]
    for _, row in bottom_3.iterrows():
        print(f"  Image {row['image_id']:2f}: PSNR={row['psnr']:.2f} dB, SSIM={row['ssim']:.4f}")
    
    # Save detailed report to CSV
    output_csv = "compression_metrics_summary.csv"
    df.to_csv(output_csv, index=False)
    print(f"\n💾 Detailed metrics saved to: {output_csv}")
    
    # Create visualization of metrics distribution
    create_metrics_visualization(df)

def create_metrics_visualization(df):
    """Create visualization of metrics distribution."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # PSNR distribution
        axes[0, 0].hist(df['psnr'], bins=15, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(df['psnr'].mean(), color='red', linestyle='--', label=f'Mean: {df["psnr"].mean():.1f} dB')
        axes[0, 0].set_xlabel('PSNR (dB)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('PSNR Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # SSIM distribution
        axes[0, 1].hist(df['ssim'], bins=15, edgecolor='black', alpha=0.7, color='orange')
        axes[0, 1].axvline(df['ssim'].mean(), color='red', linestyle='--', label=f'Mean: {df["ssim"].mean():.3f}')
        axes[0, 1].set_xlabel('SSIM')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('SSIM Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # PSNR vs SSIM scatter
        scatter = axes[1, 0].scatter(df['psnr'], df['ssim'], c=df.index, cmap='viridis', s=50)
        axes[1, 0].set_xlabel('PSNR (dB)')
        axes[1, 0].set_ylabel('SSIM')
        axes[1, 0].set_title('PSNR vs SSIM Correlation')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add image numbers to scatter points
        for i, row in df.iterrows():
            axes[1, 0].text(row['psnr'], row['ssim'], str(row['image_id']), 
                          fontsize=8, ha='center', va='center')
        
        # Ranking bar chart
        axes[1, 1].barh(range(len(df)), df['psnr'].sort_values().values)
        axes[1, 1].set_yticks(range(len(df)))
        axes[1, 1].set_yticklabels(df.sort_values('psnr')['image_id'].values)
        axes[1, 1].set_xlabel('PSNR (dB)')
        axes[1, 1].set_title('Image Ranking by PSNR')
        axes[1, 1].grid(True, alpha=0.3, axis='x')
        
        plt.suptitle('RHCCQ Compression Quality Analysis - 24 Images', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('metrics_distribution.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\n📊 Visualization saved as: metrics_distribution.png")
        
    except ImportError:
        print("\n⚠️  Visualization skipped: matplotlib/seaborn not installed")

def save_visualization(original, reconstructed, image_id, metrics):
    """Save comparison visualization for individual image."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f'Original Image {image_id}')
        axes[0].axis('off')
        
        # Reconstructed
        axes[1].imshow(cv2.cvtColor(reconstructed, cv2.COLOR_BGR2RGB))
        axes[1].set_title(f'RHCCQ Compressed')
        axes[1].axis('off')
        
        # Difference
        diff = cv2.absdiff(original, reconstructed)
        diff_normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        axes[2].imshow(diff_normalized, cmap='hot')
        axes[2].set_title(f'Difference (MSE: {metrics["mse"]:.1f})')
        axes[2].axis('off')
        
        plt.suptitle(f'Image {image_id}: PSNR={metrics["psnr"]:.2f} dB, SSIM={metrics["ssim"]:.4f}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        output_path = f'comparison_image_{image_id:02d}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)  # Close to free memory
        print(f"  💾 Visualization saved: {output_path}")
        
    except Exception as e:
        print(f"  ⚠️  Could not save visualization: {str(e)}")

# Run the main function
if __name__ == "__main__":
    main()