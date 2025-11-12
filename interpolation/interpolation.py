import random
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_closed_shape(num_points=8, center_x=500, center_y=500, avg_radius=200, irregularity=0.5, spikiness=0.3, min_size=100):
    """
    Generate coordinates for a closed irregular shape.
    
    Parameters:
    - num_points: Number of points in the shape (default: 8)
    - center_x, center_y: Center of the shape (default: 500, 500)
    - avg_radius: Average distance from center to points (default: 200)
    - irregularity: How irregular the shape is (0-1, default: 0.5)
    - spikiness: How spiky the shape is (0-1, default: 0.3)
    - min_size: Minimum dimension of the bounding box (default: 100)
    
    Returns:
    - List of (x, y) coordinates forming a closed shape
    """
    
    # Parameter validation
    irregularity = max(0, min(1, irregularity))
    spikiness = max(0, min(1, spikiness))
    num_points = max(3, num_points)  # Need at least 3 points for a closed shape
    
    # Ensure minimum size by adjusting radius if needed
    min_radius_required = min_size / 2
    avg_radius = max(avg_radius, min_radius_required)
    
    # Generate random angles around the circle
    angles = []
    angle_step = 2 * math.pi / num_points
    
    for i in range(num_points):
        angle = i * angle_step + random.uniform(-irregularity * 0.5, irregularity * 0.5) * angle_step
        angles.append(angle)
    
    # Sort angles to ensure convex-ish shape
    angles.sort()
    
    # Generate radii with some variation
    points = []
    for i in range(num_points):
        # Add spikiness and irregularity to radius
        radius = avg_radius * (1 + random.uniform(-spikiness, spikiness))
        
        # Calculate point coordinates
        x = center_x + radius * math.cos(angles[i])
        y = center_y + radius * math.sin(angles[i])
        
        # Ensure points stay within 0-1000 bounds
        x = max(0, min(1000, x))
        y = max(0, min(1000, y))
        
        points.append((x, y))
    
    # Ensure the shape meets minimum size requirement
    points = ensure_minimum_size(points, min_size)
    
    # Ensure the shape is closed by repeating the first point at the end
    points.append(points[0])
    
    return points

def ensure_minimum_size(points, min_size):
    """
    Ensure the shape has at least the minimum dimension in both x and y directions.
    """
    if not points:
        return points
    
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    
    # If shape is too small, scale it up
    if width < min_size or height < min_size:
        scale_factor = max(min_size / width, min_size / height)
        
        # Calculate center
        center_x = (max(xs) + min(xs)) / 2
        center_y = (max(ys) + min(ys)) / 2
        
        # Scale points around center
        scaled_points = []
        for x, y in points:
            new_x = center_x + (x - center_x) * scale_factor
            new_y = center_y + (y - center_y) * scale_factor
            # Ensure points stay within bounds
            new_x = max(0, min(1000, new_x))
            new_y = max(0, min(1000, new_y))
            scaled_points.append((new_x, new_y))
        
        return scaled_points
    
    return points

def generate_random_shape(min_size=100):
    """
    Generate a random closed shape with random parameters.
    """
    num_points = random.randint(5, 15)
    center_x = random.randint(200, 800)
    center_y = random.randint(200, 800)
    avg_radius = random.randint(max(50, min_size//2), 300)
    
    return generate_closed_shape(
        num_points=num_points,
        center_x=center_x,
        center_y=center_y,
        avg_radius=avg_radius,
        irregularity=random.uniform(0.2, 0.8),
        spikiness=random.uniform(0.1, 0.5),
        min_size=min_size
    )

def plot_shape(shape_coords, square_size=1000, title="Generated Shape"):
    """
    Plot the shape inside a white square.
    
    Parameters:
    - shape_coords: List of (x, y) coordinates
    - square_size: Size of the square canvas (default: 1000)
    - title: Plot title
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create white square background
    square = patches.Rectangle((0, 0), square_size, square_size, 
                              linewidth=2, edgecolor='black', 
                              facecolor='white', alpha=0.8)
    ax.add_patch(square)
    
    # Extract x and y coordinates for plotting
    x_coords = [p[0] for p in shape_coords]
    y_coords = [p[1] for p in shape_coords]
    
    # Plot the shape
    ax.plot(x_coords, y_coords, 'b-', linewidth=2, label='Shape outline')
    ax.plot(x_coords, y_coords, 'ro', markersize=4, label='Vertices')
    
    # Set plot properties
    ax.set_xlim(-50, square_size + 50)
    ax.set_ylim(-50, square_size + 50)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Calculate and display shape statistics
    xs = [p[0] for p in shape_coords[:-1]]  # Exclude the closing point
    ys = [p[1] for p in shape_coords[:-1]]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    
    stats_text = f"Points: {len(shape_coords)-1}\nWidth: {width:.1f}px\nHeight: {height:.1f}px"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Generate and plot a shape with minimum size 100px
    print("Generating shape with minimum size 100px...")
    shape_coords = generate_closed_shape(min_size=100, num_points=50)
    
    print("Shape coordinates:")
    for i, (x, y) in enumerate(shape_coords):
        print(f"Point {i}: ({x:.1f}, {y:.1f})")
    
    # Verify minimum size
    xs = [p[0] for p in shape_coords[:-1]]
    ys = [p[1] for p in shape_coords[:-1]]
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    print(f"\nShape dimensions: {width:.1f}px × {height:.1f}px")
    print(f"Meets minimum size requirement: {width >= 100 and height >= 100}")
    
    # Plot the shape
    plot_shape(shape_coords, title="Closed Irregular Shape (Min Size: 100px)")
    
    # Generate and plot a random shape
    print("\n" + "="*50)
    print("Generating random shape...")
    random_shape = generate_random_shape(min_size=100)
    plot_shape(random_shape, title="Random Closed Shape (Min Size: 100px)")