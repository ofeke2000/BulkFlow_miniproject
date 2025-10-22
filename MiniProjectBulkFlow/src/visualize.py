# visualize.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D scatter
from utils import ensure_dir, timing


# ================================================================
# Basic scatter plots
# ================================================================

@timing
def scatter_overdensity(df, output_dir, title="Overdensity δ₅ distribution", cmap="coolwarm"):
    """
    Create a 3D scatter plot of halos colored by overdensity δ₅.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns ['x', 'y', 'z', 'delta_5'].
    output_dir : str
        Directory to save the plot.
    title : str
        Plot title.
    cmap : str
        Matplotlib colormap name.
    """
    ensure_dir(output_dir)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(
        df['x'], df['y'], df['z'],
        c=df['delta_5'], cmap=cmap,
        s=5, alpha=0.6
    )

    cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
    cbar.set_label(r'$\delta_5$', fontsize=12)

    ax.set_xlabel("x [$h^{-1}$ Mpc]")
    ax.set_ylabel("y [$h^{-1}$ Mpc]")
    ax.set_zlabel("z [$h^{-1}$ Mpc]")
    ax.set_title(title, fontsize=14)

    outpath = os.path.join(output_dir, "scatter_delta5_3d.png")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"[INFO] Saved scatter plot: {outpath}")


# ================================================================
# 2D projection (for heatmap-like visualization)
# ================================================================

@timing
def projection_overdensity(df, output_dir, plane='xy', bins=200, cmap="coolwarm"):
    """
    Create a 2D projection (density map) of halos colored by δ₅.
    Averaged per spatial bin.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns ['x', 'y', 'z', 'delta_5'].
    plane : str
        'xy', 'xz', or 'yz' — the projection plane.
    bins : int
        Number of bins in each dimension.
    cmap : str
        Matplotlib colormap.
    """
    ensure_dir(output_dir)

    if plane == 'xy':
        x, y = df['x'], df['y']
        label_x, label_y = 'x', 'y'
    elif plane == 'xz':
        x, y = df['x'], df['z']
        label_x, label_y = 'x', 'z'
    elif plane == 'yz':
        x, y = df['y'], df['z']
        label_x, label_y = 'y', 'z'
    else:
        raise ValueError("Plane must be one of: 'xy', 'xz', 'yz'")

    # Use 2D histogram weighted by δ₅
    H, xedges, yedges = np.histogram2d(x, y, bins=bins, weights=df['delta_5'])
    counts, _, _ = np.histogram2d(x, y, bins=bins)
    H_avg = np.divide(H, counts, out=np.zeros_like(H), where=counts != 0)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        H_avg.T,
        origin='lower',
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap=cmap,
        aspect='auto'
    )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'$\langle \delta_5 \rangle$', fontsize=12)
    ax.set_xlabel(f"{label_x} [$h^{{-1}}$ Mpc]")
    ax.set_ylabel(f"{label_y} [$h^{{-1}}$ Mpc]")
    ax.set_title(f"Projection on {plane.upper()} plane")

    outpath = os.path.join(output_dir, f"projection_{plane}_delta5.png")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"[INFO] Saved projection: {outpath}")


# ================================================================
# Bulk flow visualization
# ================================================================

@timing
def plot_bulkflow_vectors(centers, vectors, output_dir, title="Bulk Flow Vectors"):
    """
    Plot bulk flow vectors as arrows in 3D space.
    
    Parameters
    ----------
    centers : array-like of shape (N, 3)
        Positions of the spheres.
    vectors : array-like of shape (N, 3)
        Bulk flow vectors.
    output_dir : str
        Output directory.
    """
    ensure_dir(output_dir)

    centers = np.array(centers)
    vectors = np.array(vectors)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.quiver(
        centers[:, 0], centers[:, 1], centers[:, 2],
        vectors[:, 0], vectors[:, 1], vectors[:, 2],
        length=10, normalize=True, color='blue', alpha=0.7
    )

    ax.set_xlabel("x [$h^{-1}$ Mpc]")
    ax.set_ylabel("y [$h^{-1}$ Mpc]")
    ax.set_zlabel("z [$h^{-1}$ Mpc]")
    ax.set_title(title)

    outpath = os.path.join(output_dir, "bulkflow_vectors_3d.png")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"[INFO] Saved bulk flow vector plot: {outpath}")


# ================================================================
# Diagnostic histogram
# ================================================================

@timing
def histogram_delta5(df, output_dir, bins=100):
    """Plot histogram of δ₅ values."""
    ensure_dir(output_dir)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(df['delta_5'], bins=bins, color='steelblue', alpha=0.7)
    ax.set_xlabel(r'$\delta_5$')
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Overdensities")

    outpath = os.path.join(output_dir, "histogram_delta5.png")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"[INFO] Saved δ₅ histogram: {outpath}")
