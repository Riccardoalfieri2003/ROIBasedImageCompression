"""
Image Compression Comparison Tool
Compares PNG (original), JPEG, and your custom RHCCQ format
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os
import pandas as pd
from pathlib import Path
import seaborn as sns
from datetime import datetime

# Assuming you have this function to load your RHCCQ format
# from your_module import load_rhccq_image

def load_rhccq_image(rhccq_path):
    """
    Load your custom RHCCQ format.
    Replace this with your actual loading function.
    """
    # This is a placeholder - you need to implement this based on your format
    # Example structure if your format outputs a standard image:
    # return cv2.imread(rhccq_path)  # If saved as standard image
    # Or if you have custom loading:
    from decoder.uncompression.uncompression import lossless_decompress, load_compressed, decompress_color_quantization

    loaded = load_compressed(rhccq_path)
    compressed=lossless_decompress(loaded)

    reconstruction_result = decompress_color_quantization(compressed)
    reconstructed = reconstruction_result['image']  # This is a numpy array

    return reconstructed

def load_and_compare_images(png_path, jpg_path, rhccq_path, jpeg_quality=None):
    """
    Load all three formats and compare them.
    
    Args:
        png_path: Path to original PNG (reference)
        jpg_path: Path to JPEG compressed version
        rhccq_path: Path to your RHCCQ compressed version
        jpeg_quality: JPEG quality if known (for labeling)
    
    Returns:
        Dictionary with all images, metrics, and file sizes
    """
    print(f"\n{'='*80}")
    print(f"COMPARISON: {Path(png_path).name}")
    print(f"{'='*80}")
    
    # Load images
    print("\n📁 Loading images...")
    original = cv2.imread(png_path)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    jpeg = cv2.imread(jpg_path)
    jpeg = cv2.cvtColor(jpeg, cv2.COLOR_BGR2RGB)
    
    # Load your custom format
    rhccq = load_rhccq_image(rhccq_path)
    #if rhccq.shape[2] == 3:  # If BGR, convert to RGB
    #    rhccq = cv2.cvtColor(rhccq, cv2.COLOR_BGR2RGB)
    
    # Verify dimensions match
    assert original.shape == jpeg.shape == rhccq.shape, \
        f"Image dimensions don't match! Original: {original.shape}, " \
        f"JPEG: {jpeg.shape}, RHCCQ: {rhccq.shape}"
    
    print(f"✓ All images loaded: {original.shape}")
    
    # Get file sizes
    print("\n📊 File sizes:")
    png_size = os.path.getsize(png_path) / 1024  # KB
    jpg_size = os.path.getsize(jpg_path) / 1024
    rhccq_size = os.path.getsize(rhccq_path) / 1024
    
    print(f"  PNG (original): {png_size:.1f} KB")
    print(f"  JPEG: {jpg_size:.1f} KB")
    print(f"  RHCCQ: {rhccq_size:.1f} KB")
    
    # Calculate compression ratios (relative to PNG)
    comp_ratio_jpeg = png_size / jpg_size
    comp_ratio_rhccq = png_size / rhccq_size
    
    print(f"\n📈 Compression ratios (vs PNG):")
    print(f"  JPEG: {comp_ratio_jpeg:.2f}x")
    print(f"  RHCCQ: {comp_ratio_rhccq:.2f}x")
    
    # Calculate bits per pixel
    h, w = original.shape[:2]
    total_pixels = h * w
    
    bpp_png = (png_size * 1024 * 8) / total_pixels
    bpp_jpeg = (jpg_size * 1024 * 8) / total_pixels
    bpp_rhccq = (rhccq_size * 1024 * 8) / total_pixels
    
    print(f"\n🎯 Bits per pixel (BPP):")
    print(f"  PNG: {bpp_png:.2f} bpp")
    print(f"  JPEG: {bpp_jpeg:.2f} bpp")
    print(f"  RHCCQ: {bpp_rhccq:.2f} bpp")
    
    # Calculate quality metrics (using PNG as reference)
    print("\n📐 Quality metrics (vs PNG original):")
    
    # PSNR
    psnr_jpeg = psnr(original, jpeg, data_range=255)
    psnr_rhccq = psnr(original, rhccq, data_range=255)
    
    print(f"\n  PSNR (higher is better):")
    print(f"    JPEG: {psnr_jpeg:.2f} dB")
    print(f"    RHCCQ: {psnr_rhccq:.2f} dB")
    print(f"    Difference: {psnr_rhccq - psnr_jpeg:+.2f} dB")
    
    # SSIM
    ssim_jpeg = ssim(original, jpeg, channel_axis=2, data_range=255)
    ssim_rhccq = ssim(original, rhccq, channel_axis=2, data_range=255)
    
    print(f"\n  SSIM (higher is better, range 0-1):")
    print(f"    JPEG: {ssim_jpeg:.4f}")
    print(f"    RHCCQ: {ssim_rhccq:.4f}")
    print(f"    Difference: {ssim_rhccq - ssim_jpeg:+.4f}")
    
    # MSE (Mean Squared Error)
    mse_jpeg = np.mean((original.astype(float) - jpeg.astype(float)) ** 2)
    mse_rhccq = np.mean((original.astype(float) - rhccq.astype(float)) ** 2)
    
    print(f"\n  MSE (lower is better):")
    print(f"    JPEG: {mse_jpeg:.2f}")
    print(f"    RHCCQ: {mse_rhccq:.2f}")
    print(f"    Ratio: {mse_rhccq/mse_jpeg:.2f}x")
    
    # Create comprehensive results dictionary
    results = {
        'image_name': Path(png_path).stem,
        'dimensions': original.shape,
        'original_size_kb': png_size,
        
        'jpeg': {
            'image': jpeg,
            'size_kb': jpg_size,
            'compression_ratio': comp_ratio_jpeg,
            'bpp': bpp_jpeg,
            'psnr': psnr_jpeg,
            'ssim': ssim_jpeg,
            'mse': mse_jpeg,
            'quality': jpeg_quality
        },
        
        'rhccq': {
            'image': rhccq,
            'size_kb': rhccq_size,
            'compression_ratio': comp_ratio_rhccq,
            'bpp': bpp_rhccq,
            'psnr': psnr_rhccq,
            'ssim': ssim_rhccq,
            'mse': mse_rhccq
        },
        
        'comparison': {
            'psnr_diff': psnr_rhccq - psnr_jpeg,
            'ssim_diff': ssim_rhccq - ssim_jpeg,
            'size_diff_pct': (rhccq_size - jpg_size) / jpg_size * 100,
            'bpp_diff': bpp_rhccq - bpp_jpeg
        }
    }
    
    return results

def create_visual_comparison(results, save_path="comparison_results.png"):
    """
    Create a visual comparison plot.
    """
    fig = plt.figure(figsize=(16, 10))
    
    # Layout: 3 rows, 4 columns
    gs = fig.add_gridspec(3, 4, hspace=0.05, wspace=0.05)
    
    # Original (PNG)
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(results['original_image'] if 'original_image' in results else 
                  cv2.cvtColor(cv2.imread('placeholder.png'), cv2.COLOR_BGR2RGB))
    ax_orig.set_title('Original (PNG)', fontsize=12, fontweight='bold')
    ax_orig.text(0.02, 0.98, f"{results['original_size_kb']:.0f} KB", 
                transform=ax_orig.transAxes, fontsize=10,
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax_orig.axis('off')
    
    # JPEG
    ax_jpeg = fig.add_subplot(gs[0, 1])
    ax_jpeg.imshow(results['jpeg']['image'])
    ax_jpeg.set_title(f"JPEG (Q{results['jpeg'].get('quality', '?')})", 
                     fontsize=12, fontweight='bold')
    ax_jpeg.text(0.02, 0.98, f"{results['jpeg']['size_kb']:.0f} KB\n"
                f"PSNR: {results['jpeg']['psnr']:.1f} dB\n"
                f"SSIM: {results['jpeg']['ssim']:.3f}",
                transform=ax_jpeg.transAxes, fontsize=9,
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax_jpeg.axis('off')
    
    # RHCCQ
    ax_rhccq = fig.add_subplot(gs[0, 2])
    ax_rhccq.imshow(results['rhccq']['image'])
    ax_rhccq.set_title('RHCCQ (Your Format)', fontsize=12, fontweight='bold')
    ax_rhccq.text(0.02, 0.98, f"{results['rhccq']['size_kb']:.0f} KB\n"
                f"PSNR: {results['rhccq']['psnr']:.1f} dB\n"
                f"SSIM: {results['rhccq']['ssim']:.3f}",
                transform=ax_rhccq.transAxes, fontsize=9,
                verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax_rhccq.axis('off')
    
    # Difference maps
    # JPEG difference
    ax_diff_jpeg = fig.add_subplot(gs[1, 1])
    diff_jpeg = np.abs(results['original_image'].astype(float) - 
                      results['jpeg']['image'].astype(float))
    diff_jpeg_normalized = (diff_jpeg / diff_jpeg.max() * 255).astype(np.uint8)
    ax_diff_jpeg.imshow(diff_jpeg_normalized, cmap='hot')
    ax_diff_jpeg.set_title('JPEG Error Map', fontsize=11)
    ax_diff_jpeg.text(0.02, 0.98, f"MSE: {results['jpeg']['mse']:.1f}",
                     transform=ax_diff_jpeg.transAxes, fontsize=9,
                     verticalalignment='top', color='white',
                     bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    ax_diff_jpeg.axis('off')
    
    # RHCCQ difference
    ax_diff_rhccq = fig.add_subplot(gs[1, 2])
    diff_rhccq = np.abs(results['original_image'].astype(float) - 
                       results['rhccq']['image'].astype(float))
    diff_rhccq_normalized = (diff_rhccq / diff_rhccq.max() * 255).astype(np.uint8)
    ax_diff_rhccq.imshow(diff_rhccq_normalized, cmap='hot')
    ax_diff_rhccq.set_title('RHCCQ Error Map', fontsize=11)
    ax_diff_rhccq.text(0.02, 0.98, f"MSE: {results['rhccq']['mse']:.1f}",
                      transform=ax_diff_rhccq.transAxes, fontsize=9,
                      verticalalignment='top', color='white',
                      bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    ax_diff_rhccq.axis('off')
    
    # Rate-Distortion plot
    ax_rd = fig.add_subplot(gs[2, :])
    
    # Plot points
    ax_rd.scatter(results['jpeg']['bpp'], results['jpeg']['psnr'], 
                 s=200, c='red', marker='o', label=f"JPEG Q{results['jpeg'].get('quality', '?')}", 
                 edgecolors='black', linewidth=2)
    ax_rd.scatter(results['rhccq']['bpp'], results['rhccq']['psnr'], 
                 s=200, c='blue', marker='s', label='RHCCQ', 
                 edgecolors='black', linewidth=2)
    
    # Add annotations
    ax_rd.annotate(f"PSNR: {results['jpeg']['psnr']:.1f} dB\n"
                  f"BPP: {results['jpeg']['bpp']:.2f}",
                  (results['jpeg']['bpp'], results['jpeg']['psnr']),
                  xytext=(10, 10), textcoords='offset points',
                  fontsize=9, bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    
    ax_rd.annotate(f"PSNR: {results['rhccq']['psnr']:.1f} dB\n"
                  f"BPP: {results['rhccq']['bpp']:.2f}",
                  (results['rhccq']['bpp'], results['rhccq']['psnr']),
                  xytext=(10, -25), textcoords='offset points',
                  fontsize=9, bbox=dict(boxstyle='round', facecolor='blue', alpha=0.3))
    
    ax_rd.set_xlabel('Bits Per Pixel (BPP)', fontsize=12)
    ax_rd.set_ylabel('PSNR (dB)', fontsize=12)
    ax_rd.set_title('Rate-Distortion Comparison', fontsize=14, fontweight='bold')
    ax_rd.grid(True, alpha=0.3)
    ax_rd.legend(loc='best', fontsize=11)
    
    # Add improvement arrow if RHCCQ is better
    if results['comparison']['psnr_diff'] > 0:
        ax_rd.annotate('', 
                      xy=(results['rhccq']['bpp'], results['rhccq']['psnr']),
                      xytext=(results['jpeg']['bpp'], results['jpeg']['psnr']),
                      arrowprops=dict(arrowstyle='->', color='green', lw=2, ls='--'))
        ax_rd.text((results['jpeg']['bpp'] + results['rhccq']['bpp'])/2,
                  (results['jpeg']['psnr'] + results['rhccq']['psnr'])/2,
                  f"+{results['comparison']['psnr_diff']:.1f} dB",
                  fontsize=10, fontweight='bold', color='green',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle(f"Compression Comparison: {results['image_name']} "
                f"({results['dimensions'][1]}×{results['dimensions'][0]})", 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {save_path}")
    plt.show()
    
    return fig

def create_summary_statistics(results_list, output_csv="comparison_summary.csv"):
    """
    Create summary statistics table from multiple comparisons.
    """
    summary_data = []
    
    for results in results_list:
        summary_data.append({
            'Image': results['image_name'],
            'Dimensions': f"{results['dimensions'][1]}×{results['dimensions'][0]}",
            
            'JPEG_Size_KB': results['jpeg']['size_kb'],
            'JPEG_BPP': results['jpeg']['bpp'],
            'JPEG_PSNR': results['jpeg']['psnr'],
            'JPEG_SSIM': results['jpeg']['ssim'],
            
            'RHCCQ_Size_KB': results['rhccq']['size_kb'],
            'RHCCQ_BPP': results['rhccq']['bpp'],
            'RHCCQ_PSNR': results['rhccq']['psnr'],
            'RHCCQ_SSIM': results['rhccq']['ssim'],
            
            'PSNR_Improvement': results['comparison']['psnr_diff'],
            'SSIM_Improvement': results['comparison']['ssim_diff'],
            'Size_Diff_%': results['comparison']['size_diff_pct'],
            'Compression_Ratio_JPEG': results['jpeg']['compression_ratio'],
            'Compression_Ratio_RHCCQ': results['rhccq']['compression_ratio']
        })
    
    df = pd.DataFrame(summary_data)
    
    # Calculate averages
    avg_row = {
        'Image': 'AVERAGE',
        'Dimensions': '',
        'JPEG_Size_KB': df['JPEG_Size_KB'].mean(),
        'JPEG_BPP': df['JPEG_BPP'].mean(),
        'JPEG_PSNR': df['JPEG_PSNR'].mean(),
        'JPEG_SSIM': df['JPEG_SSIM'].mean(),
        'RHCCQ_Size_KB': df['RHCCQ_Size_KB'].mean(),
        'RHCCQ_BPP': df['RHCCQ_BPP'].mean(),
        'RHCCQ_PSNR': df['RHCCQ_PSNR'].mean(),
        'RHCCQ_SSIM': df['RHCCQ_SSIM'].mean(),
        'PSNR_Improvement': df['PSNR_Improvement'].mean(),
        'SSIM_Improvement': df['SSIM_Improvement'].mean(),
        'Size_Diff_%': df['Size_Diff_%'].mean(),
        'Compression_Ratio_JPEG': df['Compression_Ratio_JPEG'].mean(),
        'Compression_Ratio_RHCCQ': df['Compression_Ratio_RHCCQ'].mean()
    }
    
    df = df._append(avg_row, ignore_index=True)
    
    # Format for better readability
    format_dict = {
        'JPEG_Size_KB': '{:.1f}',
        'JPEG_BPP': '{:.2f}',
        'JPEG_PSNR': '{:.2f}',
        'JPEG_SSIM': '{:.4f}',
        'RHCCQ_Size_KB': '{:.1f}',
        'RHCCQ_BPP': '{:.2f}',
        'RHCCQ_PSNR': '{:.2f}',
        'RHCCQ_SSIM': '{:.4f}',
        'PSNR_Improvement': '{:+.2f}',
        'SSIM_Improvement': '{:+.4f}',
        'Size_Diff_%': '{:+.1f}',
        'Compression_Ratio_JPEG': '{:.2f}',
        'Compression_Ratio_RHCCQ': '{:.2f}'
    }
    
    for col, fmt in format_dict.items():
        if col in df.columns:
            df[col] = df[col].apply(lambda x: fmt.format(x) if pd.notnull(x) else '')
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(df.to_string(index=False))
    print(f"\n✅ Summary saved to: {output_csv}")
    
    return df

def generate_report(results_list, output_dir="comparison_report"):
    """
    Generate a comprehensive HTML report.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image Compression Comparison Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; border-bottom: 2px solid #333; }}
            .image-row {{ display: flex; margin: 20px 0; }}
            .image-container {{ flex: 1; margin: 10px; text-align: center; }}
            .image-container img {{ max-width: 100%; border: 1px solid #ddd; }}
            .metrics {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            .improvement {{ color: green; font-weight: bold; }}
            .worse {{ color: red; }}
        </style>
    </head>
    <body>
        <h1>Image Compression Comparison Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    """
    
    for i, results in enumerate(results_list):
        # Save visualization for this image
        vis_path = os.path.join(output_dir, f"comparison_{results['image_name']}.png")
        create_visual_comparison(results, vis_path)
        
        html_content += f"""
        <h2>{i+1}. {results['image_name']} ({results['dimensions'][1]}×{results['dimensions'][0]})</h2>
        
        <div class="image-row">
            <div class="image-container">
                <img src="{vis_path}" alt="Comparison">
            </div>
        </div>
        
        <div class="metrics">
            <h3>Metrics</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>JPEG</th>
                    <th>RHCCQ</th>
                    <th>Difference</th>
                </tr>
                <tr>
                    <td>File Size (KB)</td>
                    <td>{results['jpeg']['size_kb']:.1f}</td>
                    <td>{results['rhccq']['size_kb']:.1f}</td>
                    <td class="{'' if results['comparison']['size_diff_pct'] <= 0 else 'worse'}">
                        {results['comparison']['size_diff_pct']:+.1f}%
                    </td>
                </tr>
                <tr>
                    <td>PSNR (dB)</td>
                    <td>{results['jpeg']['psnr']:.2f}</td>
                    <td>{results['rhccq']['psnr']:.2f}</td>
                    <td class="improvement">
                        {results['comparison']['psnr_diff']:+.2f}
                    </td>
                </tr>
                <tr>
                    <td>SSIM</td>
                    <td>{results['jpeg']['ssim']:.4f}</td>
                    <td>{results['rhccq']['ssim']:.4f}</td>
                    <td class="improvement">
                        {results['comparison']['ssim_diff']:+.4f}
                    </td>
                </tr>
                <tr>
                    <td>Compression Ratio</td>
                    <td>{results['jpeg']['compression_ratio']:.2f}x</td>
                    <td>{results['rhccq']['compression_ratio']:.2f}x</td>
                    <td>—</td>
                </tr>
            </table>
        </div>
        <hr>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    report_path = os.path.join(output_dir, "report.html")
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"\n📊 HTML report generated: {report_path}")
    return report_path

def main():
    """
    Main function to run comparisons.
    Modify this based on your actual file structure.
    """
    # Example file structure - modify as needed
    test_images = [
        {
            'name': '2',
            'png': r"C:\Users\rical\OneDrive\Desktop\Wallpaper\Napoli.png",
            'jpg': 'images/jpg/Napoli_compressed.jpg',  # JPEG at quality 85
            'rhccq': 'images/rhccq/Napoli_compressed.rhccq'   # Your custom format
        }
    ]
    
    all_results = []
    
    for img_info in test_images:
        try:
            print(f"\n{'='*80}")
            print(f"Processing: {img_info['name']}")
            print(f"{'='*80}")
            
            results = load_and_compare_images(
                png_path=img_info['png'],
                jpg_path=img_info['jpg'],
                rhccq_path=img_info['rhccq']
            )
            
            # Store original image in results for visualization
            original_img = cv2.imread(img_info['png'])
            results['original_image'] = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            
            all_results.append(results)
            
            # Create individual visualization
            vis_path = f"comparison_{img_info['name']}.png"
            #create_visual_comparison(results, vis_path)
            
        except Exception as e:
            print(f"❌ Error processing {img_info['name']}: {e}")
            continue
    
    if all_results:
        # Create summary statistics
        df = create_summary_statistics(all_results, "compression_summary.csv")
        
        # Generate HTML report
        #generate_report(all_results, "compression_report")
        
        print(f"\n{'='*80}")
        print("✅ COMPARISON COMPLETE!")
        print(f"{'='*80}")
        print(f"Processed {len(all_results)} images")
        print(f"Summary CSV: compression_summary.csv")
        print(f"HTML Report: compression_report/report.html")
    else:
        print("❌ No images were successfully processed.")

if __name__ == "__main__":
    # Set style for better plots
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Run comparison
    main()