import numpy as np
from scipy.stats import norm

from sktime.transformations.series.sax import SAX


def sax_distance(
    x: np.ndarray,
    y: np.ndarray,
    word_size: int = 8,
    alphabet_size: int = 5,
    frame_size: int = 0,
) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1d arrays.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same length.")

    sax = SAX(word_size=word_size, alphabet_size=alphabet_size, frame_size=frame_size)
    x_symbols = np.asarray(sax.fit_transform(x), dtype=int).ravel()
    y_symbols = np.asarray(sax.fit_transform(y), dtype=int).ravel()

    breakpoints = norm.ppf(np.arange(1, alphabet_size) / alphabet_size, loc=0)
    size = x.shape[0]
    n = x_symbols.shape[0]
    total = 0.0
    for i, j in zip(x_symbols, y_symbols):
        if abs(i - j) <= 1:
            dist = 0.0
        else:
            high = max(i, j)
            low = min(i, j)
            dist = breakpoints[high - 1] - breakpoints[low]
        total += dist * dist
    return float(np.sqrt((size / n) * total))
