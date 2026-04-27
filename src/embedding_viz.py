"""
Embedding-space visualization using UMAP and t-SNE.
Projects high-dimensional features into 2D, colored by Depression label.
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FIGURES_DIR = "reports/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


def _save(fig, name: str) -> None:
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved: %s", path)


def plot_umap(X: np.ndarray, y: np.ndarray, n_sample: int = 3000,
              random_state: int = 42) -> None:
    """
    UMAP projection of feature space colored by Depression label.

    Samples n_sample points for speed. Fit on the sample only.

    Parameters
    ----------
    X : preprocessed feature matrix
    y : labels (0/1)
    n_sample : number of samples to visualize
    random_state : reproducibility seed
    """
    try:
        import umap
    except ImportError:
        logger.warning("umap-learn not installed — skipping UMAP plot.")
        return

    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X), size=min(n_sample, len(X)), replace=False)
    X_s, y_s = X[idx], y[idx]

    logger.info("Fitting UMAP on %d samples...", len(X_s))
    reducer = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=30, min_dist=0.1)
    embedding = reducer.fit_transform(X_s)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = {0: "steelblue", 1: "tomato"}
    labels = {0: "No Depression", 1: "Depression"}
    for cls in [0, 1]:
        mask = y_s == cls
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=colors[cls], label=labels[cls],
                   alpha=0.5, s=10, linewidths=0)
    ax.set_title(f"UMAP Projection — Feature Space (n={len(X_s)})", fontsize=13)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(markerscale=3)
    fig.tight_layout()
    _save(fig, "umap_embedding.png")


def plot_tsne(X: np.ndarray, y: np.ndarray, n_sample: int = 2000,
              random_state: int = 42) -> None:
    """
    t-SNE projection of feature space colored by Depression label.

    Parameters
    ----------
    X : preprocessed feature matrix
    y : labels (0/1)
    n_sample : number of samples (t-SNE is O(n²), keep small)
    random_state : reproducibility seed
    """
    from sklearn.manifold import TSNE

    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X), size=min(n_sample, len(X)), replace=False)
    X_s, y_s = X[idx], y[idx]

    logger.info("Fitting t-SNE on %d samples...", len(X_s))
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=40,
                max_iter=1000, learning_rate="auto", init="pca")
    embedding = tsne.fit_transform(X_s)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = {0: "steelblue", 1: "tomato"}
    labels = {0: "No Depression", 1: "Depression"}
    for cls in [0, 1]:
        mask = y_s == cls
        ax.scatter(embedding[mask, 0], embedding[mask, 1],
                   c=colors[cls], label=labels[cls],
                   alpha=0.5, s=10, linewidths=0)
    ax.set_title(f"t-SNE Projection — Feature Space (n={len(X_s)})", fontsize=13)
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")
    ax.legend(markerscale=3)
    fig.tight_layout()
    _save(fig, "tsne_embedding.png")


def run_embedding_viz(X_train: np.ndarray, y_train: np.ndarray) -> None:
    """Run both UMAP and t-SNE visualizations."""
    logger.info("Running embedding visualizations...")
    plot_umap(X_train, y_train)
    plot_tsne(X_train, y_train)
    logger.info("Embedding visualizations complete.")
