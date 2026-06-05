"""
Shared utility functions for DINO feature-based cosine similarity computation.

Extracted from dinov3_notebooks/dino_similarity_refinement.py to be reused
by both the per-patch refinement script and the SimilarityMapStitcher.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, laplace

__all__ = [
    'points_to_token_indices',
    'compute_similarity_maps',
    'aggregate_similarity_raw',
    'aggregate_similarity',
    'upsample_map',
    'locs_to_pixel_coords',
    'power_transform',
    'log_sharpen',
    'blob_score',
]


def points_to_token_indices(points_x, points_y, img_w, img_h, w_p, h_p):
    """Map pixel coordinates to token grid indices (vectorized).

    Args:
        points_x: tensor [K] x pixel coordinates
        points_y: tensor [K] y pixel coordinates
        img_w, img_h: image dimensions in pixels
        w_p, h_p: token grid dimensions

    Returns:
        flat_indices: tensor [K] of flat token indices (py * w_p + px)
    """
    px = (points_x / img_w * w_p).round().long().clamp(0, w_p - 1)
    py = (points_y / img_h * h_p).round().long().clamp(0, h_p - 1)
    return py * w_p + px


def compute_similarity_maps(features_normed, query_indices):
    """Compute cosine similarity for multiple query points (vectorized).

    Args:
        features_normed: [N_tokens, C] L2-normalized features
        query_indices: [K] flat token indices

    Returns:
        sim_maps: [K, N_tokens] cosine similarity scores
    """
    queries = features_normed[query_indices]        # [K, C]
    return (features_normed @ queries.T).T           # [K, N_tokens]


def aggregate_similarity_raw(sim_maps, h_p, w_p, weights=None):
    """Aggregate per-query similarity maps into a single spatial map (unnormalized).

    Unlike aggregate_similarity(), this does NOT apply min-max normalization,
    making it suitable for stitching where normalization must happen globally.

    Args:
        sim_maps: [K, N_tokens] similarity scores
        h_p, w_p: spatial grid dimensions
        weights: optional [K] tensor of dscore weights (for weighted mean)

    Returns:
        agg_map: [h_p, w_p] aggregated map (raw, unnormalized)
    """
    maps = sim_maps.reshape(-1, h_p, w_p)            # [K, h_p, w_p]

    if weights is not None:
        w = weights.reshape(-1, 1, 1)                # [K, 1, 1]
        agg = (maps * w).sum(dim=0) / w.sum()
    else:
        agg = maps.mean(dim=0)

    return agg


def aggregate_similarity(sim_maps, h_p, w_p, weights=None):
    """Aggregate per-query similarity maps with min-max normalization.

    For single-patch use (not stitching). Normalizes output to [0, 1].

    Args:
        sim_maps: [K, N_tokens] similarity scores
        h_p, w_p: spatial grid dimensions
        weights: optional [K] tensor of dscore weights (for weighted mean)

    Returns:
        agg_map: [h_p, w_p] aggregated map normalized to [0, 1]
    """
    agg = aggregate_similarity_raw(sim_maps, h_p, w_p, weights=weights)

    # Normalize to [0, 1]
    agg_min, agg_max = agg.min(), agg.max()
    if agg_max - agg_min > 1e-8:
        agg = (agg - agg_min) / (agg_max - agg_min)
    else:
        agg = torch.zeros_like(agg)

    return agg


def upsample_map(agg_map, factor):
    """Bilinearly upsample a 2-D map by an integer factor."""
    if factor <= 1:
        return agg_map
    return F.interpolate(
        agg_map.unsqueeze(0).unsqueeze(0),
        scale_factor=factor,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)


def power_transform(heatmap: np.ndarray, gamma: float) -> np.ndarray:
    """Apply power transform: min-max normalize to [0, 1] then raise to *gamma*.

    Args:
        heatmap: 2-D array (any range).
        gamma: Exponent (>1 suppresses background, <1 boosts low values).

    Returns:
        Transformed array in [0, 1].
    """
    a_min, a_max = heatmap.min(), heatmap.max()
    if a_max - a_min > 1e-8:
        h = (heatmap - a_min) / (a_max - a_min)
    else:
        return np.zeros_like(heatmap, dtype=np.float64)
    return np.power(h, gamma)


def log_sharpen(heatmap: np.ndarray, sigma: float, alpha: float) -> np.ndarray:
    """Laplacian-of-Gaussian (LoG) sharpening.

    ``h_sharp = h - alpha * laplacian(gaussian(h, sigma))``

    Args:
        heatmap: 2-D array.
        sigma: Gaussian smoothing standard deviation.
        alpha: Weight of the LoG term.

    Returns:
        Sharpened array (may contain negative values).
    """
    smoothed = gaussian_filter(heatmap.astype(np.float64), sigma=sigma)
    lap_of_gauss = laplace(smoothed)
    sharpened = heatmap.astype(np.float64) - alpha * lap_of_gauss
    return sharpened.astype(np.float32)


def blob_score(heatmap: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Mean absolute Laplacian — measures peakiness / blobness.

    Higher values indicate more pronounced blob-like structures.

    Args:
        heatmap: 2-D array.
        mask: Optional 2-D array (same shape as *heatmap*).  When provided,
            only pixels where ``mask > 0`` contribute to the mean.  This
            removes bias from zero-padded border patches whose padding
            would otherwise dilute the score.

    Returns:
        Scalar blob score.  Returns 0.0 if *mask* has no valid pixels.
    """
    lap = np.abs(laplace(heatmap.astype(np.float64)))
    if mask is not None:
        valid = mask.sum()
        if valid < 1:
            return 0.0
        return float((lap * mask).sum() / valid)
    return float(lap.mean())


def locs_to_pixel_coords(locs, img_w, img_h, map_w, map_h):
    """Convert LMDS locations (row, col) to pixel (x, y) coordinates.

    Maps each detected peak to the centre of the corresponding spatial cell.

    Args:
        locs: list of [row, col] or tensor of shape [N, 2]
        img_w, img_h: full image pixel dimensions
        map_w, map_h: map spatial dimensions

    Returns:
        list of (x, y) tuples in pixel coordinates
    """
    pixel_coords = []
    for row, col in locs:
        x = (col + 0.5) * img_w / map_w
        y = (row + 0.5) * img_h / map_h
        pixel_coords.append((x, y))
    return pixel_coords
