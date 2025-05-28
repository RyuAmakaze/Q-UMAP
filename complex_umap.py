"""Utility functions for projecting complex-valued features to 2D using UMAP.

This module provides a helper function that takes a set of features
(possibly containing complex numbers) and maps them to a two-dimensional
embedding space using the UMAP algorithm. Complex numbers are converted
into real-valued feature vectors by concatenating their real and
imaginary parts.

Usage Example
-------------
>>> import numpy as np
>>> from complex_umap import complex_umap_embedding
>>> features = np.array([1+2j, 3+4j])
>>> embedding = complex_umap_embedding(features.reshape(-1, 1))

The returned embedding is a NumPy array of shape (n_samples, 2).

Note: This module requires ``numpy`` and ``umap-learn`` to be installed.
"""

from typing import Iterable

import numpy as np

try:
    import umap
except ImportError as exc:  # pragma: no cover - handled in runtime
    raise ImportError(
        "umap-learn is required for complex_umap_embedding. Install it via 'pip install umap-learn'."
    ) from exc


def _to_real_matrix(features: Iterable[np.ndarray]) -> np.ndarray:
    """Convert an iterable of complex or real feature vectors to a 2D real matrix.

    Parameters
    ----------
    features : Iterable[np.ndarray]
        An iterable yielding arrays of shape (n_features,). Elements may
        contain complex numbers.

    Returns
    -------
    np.ndarray
        A real-valued matrix of shape (n_samples, n_output_features) where
        complex features are expanded into their real and imaginary parts.
    """
    real_rows = []
    for row in features:
        row = np.asarray(row)
        if np.iscomplexobj(row):
            real_rows.append(np.column_stack([row.real, row.imag]).reshape(row.size * 2))
        else:
            real_rows.append(row.astype(float))
    return np.vstack(real_rows)


def complex_umap_embedding(features: Iterable[np.ndarray], **umap_kwargs) -> np.ndarray:
    """Project complex-valued features to a 2D embedding using UMAP.

    Parameters
    ----------
    features : Iterable[np.ndarray]
        Iterable yielding feature vectors. Features may include complex numbers.
    **umap_kwargs : dict
        Additional keyword arguments passed to ``umap.UMAP``.

    Returns
    -------
    np.ndarray
        Array of shape (n_samples, 2) with the 2D embedding.
    """
    real_matrix = _to_real_matrix(features)
    reducer = umap.UMAP(n_components=2, **umap_kwargs)
    return reducer.fit_transform(real_matrix)
