import numpy as np
from typing import Optional, List, Union, Dict, Tuple

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def _check_matplotlib():
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization. Install with 'pip install matplotlib'.")

def plot_histogram(
    counts: Dict[str, int], 
    title: str = "Simulation Results", 
    figsize: Tuple[int, int] = (10, 6), 
    filename: Optional[str] = None,
    color: str = "#6495ED"
):
    """Plot a histogram of measurement counts.
    
    Args:
        counts: Dictionary mapping bitstrings to counts
        title: Title of the plot
        figsize: Figure size (width, height)
        filename: If provided, save the plot to this file
        color: Color of the bars
    """
    _check_matplotlib()
        
    # Sort by bitstring
    sorted_keys = sorted(counts.keys())
    sorted_counts = [counts[k] for k in sorted_keys]
    
    plt.figure(figsize=figsize)
    bars = plt.bar(sorted_keys, sorted_counts, color=color)
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}',
                 ha='center', va='bottom')
                 
    plt.title(title)
    plt.xlabel("State")
    plt.ylabel("Counts")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

def plot_bloch_vector(bloch_vector: List[float], title: str = ""):
    """Plot a single Bloch vector [x, y, z].
    
    Args:
        bloch_vector: List of 3 floats representing the Bloch vector
        title: Title of the plot
    """
    _check_matplotlib()
    
    if len(bloch_vector) != 3:
        raise ValueError("Bloch vector must have exactly 3 components")
        
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_surface(x, y, z, color='aliceblue', alpha=0.1, edgecolor='none')
    
    # Draw axes
    ax.plot([-1, 1], [0, 0], [0, 0], 'k-', alpha=0.2)
    ax.plot([0, 0], [-1, 1], [0, 0], 'k-', alpha=0.2)
    ax.plot([0, 0], [0, 0], [-1, 1], 'k-', alpha=0.2)
    
    # Draw equator
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), np.zeros(100), 'k--', alpha=0.2)
    
    # Draw vector
    ax.quiver(0, 0, 0, bloch_vector[0], bloch_vector[1], bloch_vector[2], 
              color='r', arrow_length_ratio=0.1, linewidth=2)
    
    # Add labels
    ax.text(1.1, 0, 0, "x")
    ax.text(0, 1.1, 0, "y")
    ax.text(0, 0, 1.1, "z")
    ax.text(0, 0, -1.2, "|1>")
    ax.text(0, 0, 1.2, "|0>")
    
    if title:
        ax.set_title(title)
        
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    # Set aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    plt.show()
