import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt

def print_divided_compression_analysis(result):
    """
    Print detailed analysis of divided compression
    """
    if not result:
        print("❌ No results to analyze")
        return
    
    overall = result['overall_metrics']
    sublist_results = result['sublist_results']
    
    print("\n" + "="*70)
    print("📊 DIVIDED COMPRESSION ANALYSIS")
    print("="*70)
    
    print(f"🎯 Overall Results:")
    print(f"   Number of sublists: {overall['num_sublists']}")
    print(f"   Compression ratio: {overall['compression_ratio']:.1%}")
    print(f"   Total original points: {overall['total_original_points']}")
    print(f"   Total key points: {overall['total_key_points']}")
    print(f"   Overall storage reduction: {overall['storage_reduction']}")
    print(f"   Overall reconstruction error: {overall['mean_error']:.6f}")
    
    print(f"\n📈 Sublist Details:")
    for i, sub_res in enumerate(sublist_results):
        metrics = sub_res['metrics']
        print(f"   Sublist {i+1}:")
        print(f"     - Original points: {metrics['original_points']}")
        print(f"     - Key points used: {metrics['key_points_used']}")
        print(f"     - Sublist error: {metrics['mean_error']:.6f}")
        print(f"     - Storage: {metrics['storage_reduction']}")

def visualize_divided_compression(coordinates, result):
    """
    Visualize the divided compression results
    """
    original_coords = np.array(coordinates)
    if not np.allclose(original_coords[0], original_coords[-1]):
        original_coords = np.vstack([original_coords, original_coords[0]])
    
    combined_reconstructed = result['combined_reconstructed']
    sublist_results = result['sublist_results']
    overall = result['overall_metrics']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original shape
    ax1.plot(original_coords[:, 0], original_coords[:, 1], 'b-', linewidth=2, label='Original')
    ax1.set_title(f'Original Shape\n{overall["total_original_points"]} points')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Combined reconstruction
    ax2.plot(combined_reconstructed[:, 0], combined_reconstructed[:, 1], 'r-', linewidth=2, label='Reconstructed')
    ax2.set_title(f'Combined Reconstruction\n{overall["total_key_points"]} key points\nError: {overall["mean_error"]:.4f}')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Sublist key points
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    ax3.plot(original_coords[:, 0], original_coords[:, 1], 'k-', alpha=0.3, linewidth=1, label='Original')
    """for i, sub_res in enumerate(sublist_results):
        color = colors[i % len(colors)]
        key_points = sub_res['key_points']
        ax3.plot(key_points[:, 0], key_points[:, 1], 'o', color=color, markersize=6,
                markerfacecolor='none', markeredgewidth=2, label=f'Sublist {i+1}')"""
    ax3.set_title(f'Key Points by Sublist\n{overall["num_sublists"]} sublists')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Overlay comparison
    ax4.plot(original_coords[:, 0], original_coords[:, 1], 'b-', linewidth=2, label='Original', alpha=0.7)
    ax4.plot(combined_reconstructed[:, 0], combined_reconstructed[:, 1], 'r--', linewidth=2, label='Reconstructed')
    ax4.set_title(f'Overlay Comparison\nStorage: {overall["storage_reduction"]}')
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.show()

def visualize_minimal_storage_results(original, compressed, reconstructed, storage_info):
    """
    Visualize original vs compressed vs reconstructed boundaries
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    #fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot 1: Original vs Compressed
    axes[0, 0].plot(original[:, 0], original[:, 1], 'b-', alpha=0.7, label='Original', linewidth=2)
    axes[0, 0].plot(compressed[:, 0], compressed[:, 1], 'ro', markersize=4, label='Compressed Key Points')
    axes[0, 0].set_title(f'Original vs Compressed\n({len(compressed)} key points)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_aspect('equal')
    
    # Plot 2: Original vs Reconstructed
    axes[0, 1].plot(original[:, 0], original[:, 1], 'b-', alpha=0.7, label='Original', linewidth=2)
    axes[0, 1].plot(reconstructed[:, 0], reconstructed[:, 1], 'g--', alpha=0.8, label='Reconstructed', linewidth=2)
    axes[0, 1].set_title('Original vs Reconstructed')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_aspect('equal')
    
    # Plot 3: Storage comparison
    sizes = [storage_info['original_size_bytes'], storage_info['rounded_size_bytes']]
    labels = [f'Original\n{storage_info["original_size_bytes"]:,} bytes', 
              f'Compressed\n{storage_info["rounded_size_bytes"]:,} bytes']
    colors = ['lightcoral', 'lightgreen']
    axes[1, 0].bar(labels, sizes, color=colors, alpha=0.8)
    axes[1, 0].set_title('Storage Comparison')
    axes[1, 0].set_ylabel('Bytes')
    
    # Add value labels on bars
    for i, v in enumerate(sizes):
        axes[1, 0].text(i, v + max(sizes)*0.01, f'{v:,}', ha='center', va='bottom')
    
    # Plot 4: Error visualization
    error = np.sqrt(
        (reconstructed[:, 0] - original[:, 0])**2 + 
        (reconstructed[:, 1] - original[:, 1])**2
    )
    axes[1, 1].plot(error, 'r-', alpha=0.7)
    axes[1, 1].set_title('Reconstruction Error per Point')
    axes[1, 1].set_xlabel('Point Index')
    axes[1, 1].set_ylabel('Error')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(bottom=0)
    
    # Add mean error to plot
    mean_error = np.mean(error)
    axes[1, 1].axhline(y=mean_error, color='blue', linestyle='--', label=f'Mean Error: {mean_error:.6f}')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()






















def visualize_reconstruction_comparison(original, compressed, reconstructed):
    """
    Simple side-by-side visualization
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot 1: Original vs Compressed Points
    ax1.plot(original[:, 0], original[:, 1], 'b-', alpha=0.7, linewidth=2, label='Original Boundary')
    ax1.plot(compressed[:, 0], compressed[:, 1], 'ro', markersize=4, label='Compressed Points')
    ax1.set_title(f'Compression: {len(original)} → {len(compressed)} points')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot 2: Original vs Reconstructed
    ax2.plot(original[:, 0], original[:, 1], 'b-', alpha=0.7, linewidth=2, label='Original')
    ax2.plot(reconstructed[:, 0], reconstructed[:, 1], 'g--', alpha=0.8, linewidth=2, label='Reconstructed')
    ax2.set_title('Original vs Reconstructed')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Plot 3: Error Visualization
    error = np.sqrt((reconstructed[:, 0] - original[:, 0])**2 + 
                   (reconstructed[:, 1] - original[:, 1])**2)
    ax3.plot(error, 'r-', alpha=0.7)
    ax3.axhline(y=np.mean(error), color='blue', linestyle='--', 
                label=f'Mean Error: {np.mean(error):.6f}')
    ax3.set_title('Reconstruction Error per Point')
    ax3.set_xlabel('Point Index')
    ax3.set_ylabel('Error')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print error statistics
    print(f"📊 Reconstruction Quality:")
    print(f"   Mean Error: {np.mean(error):.6f}")
    print(f"   Max Error: {np.max(error):.6f}")
    print(f"   Std Error: {np.std(error):.6f}")

def visualize_interactive_comparison(original, compressed, reconstructed):
    """
    Interactive visualization with zoom capability
    """
    from matplotlib.widgets import Slider
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Full view
    ax1.plot(original[:, 0], original[:, 1], 'b-', linewidth=2, label='Original')
    ax1.plot(reconstructed[:, 0], reconstructed[:, 1], 'g--', linewidth=1, label='Reconstructed')
    ax1.plot(compressed[:, 0], compressed[:, 1], 'ro', markersize=3, label='Key Points')
    ax1.set_title('Full Boundary - Original vs Reconstructed')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot 2: Zoomed view
    zoom_factor = 0.1  # Show 10% of the boundary
    zoom_start = int(len(original) * 0.5)  # Start from middle
    zoom_end = int(zoom_start + len(original) * zoom_factor)
    
    ax2.plot(original[zoom_start:zoom_end, 0], original[zoom_start:zoom_end, 1], 
             'b-', linewidth=3, label='Original')
    ax2.plot(reconstructed[zoom_start:zoom_end, 0], reconstructed[zoom_start:zoom_end, 1], 
             'g--', linewidth=2, label='Reconstructed')
    ax2.plot(compressed[:, 0], compressed[:, 1], 'ro', markersize=4, alpha=0.5, label='Key Points')
    ax2.set_title(f'Zoomed View (points {zoom_start}-{zoom_end})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()


def visualize_reconstruction_overlay(original, compressed, reconstructed):
    """
    Overlay visualization showing all three together
    """
    plt.figure(figsize=(12, 10))
    
    # Plot original boundary
    plt.plot(original[:, 0], original[:, 1], 'b-', linewidth=3, alpha=0.5, label='Original Boundary')
    
    # Plot compressed points
    plt.plot(compressed[:, 0], compressed[:, 1], 'ro', markersize=6, label=f'Compressed Points ({len(compressed)})')
    
    # Plot reconstructed boundary
    plt.plot(reconstructed[:, 0], reconstructed[:, 1], 'g--', linewidth=2, alpha=0.8, label='Reconstructed')
    
    plt.title(f'Boundary Reconstruction\n{len(original)} points → {len(compressed)} points → {len(reconstructed)} points', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

def visualize_quality_metrics(original, reconstructed):
    """
    Detailed quality metrics visualization
    """
    # Calculate errors
    errors = np.sqrt((reconstructed[:, 0] - original[:, 0])**2 + 
                    (reconstructed[:, 1] - original[:, 1])**2)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Error distribution
    ax1.hist(errors, bins=50, alpha=0.7, color='red', edgecolor='black')
    ax1.axvline(np.mean(errors), color='blue', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(errors):.6f}')
    ax1.set_xlabel('Error')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Error Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Cumulative error
    cumulative_error = np.cumsum(errors)
    ax2.plot(cumulative_error, 'purple', linewidth=2)
    ax2.set_xlabel('Point Index')
    ax2.set_ylabel('Cumulative Error')
    ax2.set_title('Cumulative Reconstruction Error')
    ax2.grid(True, alpha=0.3)
    
    # 3. Error along boundary
    ax3.plot(errors, 'orange', linewidth=1)
    ax3.axhline(y=np.mean(errors), color='red', linestyle='--', 
                label=f'Mean: {np.mean(errors):.6f}')
    ax3.set_xlabel('Point Index along Boundary')
    ax3.set_ylabel('Error')
    ax3.set_title('Error Along Boundary')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Quality summary
    metrics = {
        'Mean Error': f'{np.mean(errors):.6f}',
        'Max Error': f'{np.max(errors):.6f}',
        'Std Dev': f'{np.std(errors):.6f}',
        '95th Percentile': f'{np.percentile(errors, 95):.6f}',
        'Points > 0.001': f'{np.sum(errors > 0.001):,}',
        'Compression Ratio': f'{len(reconstructed)/len(original):.1%}'
    }
    
    ax4.axis('off')
    text_str = '\n'.join([f'{k}: {v}' for k, v in metrics.items()])
    ax4.text(0.1, 0.9, text_str, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.set_title('Quality Metrics Summary')
    
    plt.tight_layout()
    plt.show()

