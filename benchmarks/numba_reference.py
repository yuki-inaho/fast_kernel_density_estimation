"""Numba reference implementation of the Deriche-filter KDE.

This module mirrors the Rust extension in ``fast_kde/src/lib.rs`` step for
step so that the compiled extension can be checked against an independent
implementation of the *same* algorithm.  It exists purely for verification
and benchmarking; it is not part of the installed ``fast_kde`` package and
must never be imported by library code.

Public API (matching ``fast_kde``):
    kde_deriche(data, bins, sigma) -> (x_coords, pdf_values)
    kde_mode_deriche(data, bins, sigma) -> float
    kde_deriche_2d(data, bins) -> (x_coords, y_coords, pdf)

Reference: "Fast & Accurate Gaussian Kernel Density Estimation" — linear
binning (Section 3) followed by a K=4 Deriche recursive filter (Section 2).

Requires the ``benchmark`` extra:
    uv sync --extra benchmark
"""

from __future__ import annotations

import numpy as np
from numba import njit

# Deriche K=4 coefficients from Heer 2021, equation (2). They are complex
# conjugate pairs; taking only the real parts does not approximate a Gaussian.
# Identical to ALPHA_RE/ALPHA_IM and LAMBDA_RE/LAMBDA_IM in fast_kde/src/lib.rs.
ALPHA = np.array(
    [0.84 + 1.8675j, 0.84 - 1.8675j, -0.34015 - 0.1299j, -0.34015 + 0.1299j]
)
LAMBDA = np.array([1.783 + 0.6318j, 1.783 - 0.6318j, 1.723 + 1.997j, 1.723 - 1.997j])


def deriche_coefficients(sigma):
    """Expand sum_k alpha_k / (1 - exp(-lambda_k / sigma) z^-1) into a rational form.

    Returns ``(b_plus, b_minus, a)``: the causal numerator, the anticausal
    numerator and the shared denominator coefficients for z^-1..z^-4. The
    conjugate pairing cancels the imaginary parts, so the result is real.
    """
    poles = np.exp(-LAMBDA / sigma)

    denominator = np.array([1.0 + 0j])
    for p in poles:
        denominator = np.convolve(denominator, [1.0, -p])

    numerator = np.zeros(4, dtype=complex)
    for k in range(4):
        partial = np.array([1.0 + 0j])
        for j in range(4):
            if j != k:
                partial = np.convolve(partial, [1.0, -poles[j]])
        numerator += ALPHA[k] * partial

    # Fold the 1 / sqrt(2 pi sigma^2) factor of equation (2) into the numerator.
    b_plus = numerator.real * (1.0 / (np.sqrt(2.0 * np.pi) * sigma))
    a = denominator.real[1:]

    # Anticausal numerator (Getreuer, "A Survey of Gaussian Convolution
    # Algorithms", IPOL 2013).
    b_minus = np.zeros(5)
    b_minus[1:4] = b_plus[1:4] - a[:3] * b_plus[0]
    b_minus[4] = -a[3] * b_plus[0]
    return b_plus, b_minus, a


@njit(cache=True, nogil=True)
def _deriche_passes(signal, b_plus, b_minus, a):
    """Run the causal and anticausal 4th-order recursions and sum them."""
    m = len(signal)
    causal = np.zeros(m, dtype=np.float64)
    anticausal = np.zeros(m, dtype=np.float64)

    for i in range(m):
        acc = 0.0
        for k in range(4):
            if i >= k:
                acc += b_plus[k] * signal[i - k]
        for k in range(4):
            if i > k:
                acc -= a[k] * causal[i - k - 1]
        causal[i] = acc

    for i in range(m - 1, -1, -1):
        acc = 0.0
        for k in range(1, 5):
            if i + k < m:
                acc += b_minus[k] * signal[i + k]
        for k in range(4):
            if i + k + 1 < m:
                acc -= a[k] * anticausal[i + k + 1]
        anticausal[i] = acc

    return causal + anticausal


@njit(cache=True, nogil=True)
def linear_binning(data, xmin, xmax, bins):
    """Distribute each sample over its two neighbouring bins by linear weight."""
    if bins == 0:
        return np.zeros(0, dtype=np.float64)

    bin_width = (xmax - xmin) / bins
    if abs(bin_width) < 1e-9:
        return np.zeros(bins, dtype=np.float64)
    if len(data) == 0:
        return np.zeros(bins, dtype=np.float64)

    hist = np.zeros(bins, dtype=np.float64)
    for i in range(len(data)):
        x_val = data[i]
        if xmin <= x_val < xmax:
            pos = (x_val - xmin) / bin_width
            k = int(pos)
            fraction_right = pos - k
            fraction_left = 1.0 - fraction_right
            if k < bins:
                hist[k] += fraction_left
                if k + 1 < bins:
                    hist[k + 1] += fraction_right
        elif abs(x_val - xmax) < 1e-9 and bins > 0:
            # A sample exactly at xmax belongs to the last bin.
            hist[bins - 1] += 1.0
    return hist


def deriche_recursive_filter(signal, sigma):
    """Approximate Gaussian smoothing with Deriche's 4th-order recursive filter.

    A causal pass runs left to right and an anticausal pass right to left; their
    sum approximates convolution with a Gaussian of standard deviation ``sigma``
    (in samples), to better than 0.05% of the peak for sigma from 1 to 50.
    """
    if abs(sigma) < 1e-9 or len(signal) == 0:
        return signal.astype(np.float64).copy()

    b_plus, b_minus, a = deriche_coefficients(sigma)
    return _deriche_passes(
        np.ascontiguousarray(signal, dtype=np.float64), b_plus, b_minus, a
    )


def kde_deriche(data, bins, sigma):
    """Kernel density estimate via linear binning plus a Deriche filter.

    Returns ``(x_coords, pdf_values)`` where ``x_coords`` are bin centres over
    ``[min(data), max(data)]`` and ``pdf_values`` integrate to one.

    Raises:
        ValueError: fewer than four samples, ``bins == 0``, an inverted data
            range, a degenerate bin width, or an inconsistent normalisation.
    """
    data = np.ascontiguousarray(data, dtype=np.float64)

    if len(data) < 4:
        raise ValueError("Need at least 4 samples for KDE.")
    if bins == 0:
        raise ValueError("Number of bins must be greater than 0.")
    # NaN slips past both np.min/np.max propagation and the range comparison in
    # different ways in each implementation, so reject it explicitly (spec S-3).
    non_finite = int((~np.isfinite(data)).sum())
    if non_finite:
        raise ValueError(
            f"Input data must be finite; found {non_finite} non-finite value(s)."
        )
    # A negative sigma flips the sign in exp(-lambda / sigma) and yields a filter
    # that is not a Gaussian approximation (spec S-4). sigma == 0 is allowed and
    # means "no smoothing" (spec S-5).
    if sigma < 0.0:
        raise ValueError("Sigma must be non-negative.")

    xmin = float(np.min(data))
    xmax = float(np.max(data))

    # Degenerate range: every sample sits at (essentially) the same place.
    # Give the grid a 1e-9 width so the spacing is well defined, then put all the
    # mass in the middle bin at a height derived from that spacing, so that
    # sum(pdf) * dx == 1 (spec S-6).
    if abs(xmax - xmin) < 1e-9:
        degenerate_dx = (xmax - xmin + 1e-9) / bins
        x_coords = np.array(
            [xmin + (i + 0.5) * degenerate_dx for i in range(bins)], dtype=np.float64
        )
        pdf_vals = np.zeros(bins, dtype=np.float64)
        pdf_vals[bins // 2] = 1.0 / degenerate_dx
        return x_coords, pdf_vals

    if xmax < xmin:
        raise ValueError(
            f"xmax ({xmax}) must be greater than or equal to xmin ({xmin})"
        )

    hist_counts = linear_binning(data, xmin, xmax, bins)

    dx = (xmax - xmin) / bins
    if np.all(np.abs(hist_counts) < 1e-9):
        x_coords = np.array(
            [xmin + (i + 0.5) * dx for i in range(bins)], dtype=np.float64
        )
        return x_coords, np.zeros(bins, dtype=np.float64)

    # Convert sigma from data units into bin units before filtering.
    sf = bins / (xmax - xmin)
    filtered = deriche_recursive_filter(hist_counts, sigma * sf)

    if abs(dx) < 1e-9:
        raise ValueError("dx (bin width) is too small or zero.")

    normalization_factor = float(np.sum(filtered)) * dx
    if abs(normalization_factor) < 1e-9:
        if np.any(np.abs(filtered) >= 1e-9):
            raise ValueError(
                "Normalization factor is near zero despite non-zero filtered "
                "histogram sum."
            )
    else:
        filtered = filtered / normalization_factor

    x_coords = np.array([xmin + (i + 0.5) * dx for i in range(bins)], dtype=np.float64)
    return x_coords, filtered


def kde_mode_deriche(data, bins, sigma):
    """Return the x coordinate where the Deriche KDE attains its maximum."""
    x_coords, pdf_values = kde_deriche(data, bins, sigma)

    if len(pdf_values) == 0:
        raise ValueError("PDF is empty, cannot find mode.")

    finite = np.isfinite(pdf_values)
    if not finite.any():
        raise ValueError(
            "Could not find a valid mode in the PDF (e.g., all values are NaN/Inf)."
        )

    masked = np.where(finite, pdf_values, -np.inf)
    return float(x_coords[int(np.argmax(masked))])


@njit(cache=True, nogil=True)
def _split_weight(bins, pos):
    """Return the lower node index and the fraction of the upper node.

    ``pos`` is ``(value - xmin) / bin_width``; node centres sit at ``k + 0.5``.
    Positions outside the node range are clamped, so the result always has
    ``index + 1 < bins`` and the two weights sum to one (for ``bins >= 2``).
    """
    pos = pos - 0.5
    last = bins - 1
    if pos <= 0.0:
        return 0, 0.0
    if pos >= last:
        return bins - 2, 1.0
    index = int(np.floor(pos))
    return index, pos - index


@njit(cache=True, nogil=True)
def linear_binning_2d(x, y, xmin, xmax, ymin, ymax, bins):
    """Bilinear linear binning onto the grid of bin centres (Heer 2021, Sec. 3).

    Each sample is distributed over the four surrounding nodes with weights
    proportional to the distance to their centres. Samples outside the node
    range are clamped to the edge nodes so no weight is lost.
    """
    if bins == 0:
        return np.zeros((0, 0), dtype=np.float64)
    if len(x) == 0:
        return np.zeros((bins, bins), dtype=np.float64)
    if bins == 1:
        return np.full((1, 1), float(len(x)), dtype=np.float64)

    width_x = (xmax - xmin) / bins
    width_y = (ymax - ymin) / bins
    if (
        not np.isfinite(width_x)
        or width_x <= 0.0
        or not np.isfinite(width_y)
        or width_y <= 0.0
    ):
        return np.zeros((bins, bins), dtype=np.float64)

    hist = np.zeros((bins, bins), dtype=np.float64)
    for i in range(len(x)):
        ix, fx = _split_weight(bins, (x[i] - xmin) / width_x)
        iy, fy = _split_weight(bins, (y[i] - ymin) / width_y)
        hist[ix, iy] += (1.0 - fx) * (1.0 - fy)
        hist[ix, iy + 1] += (1.0 - fx) * fy
        hist[ix + 1, iy] += fx * (1.0 - fy)
        hist[ix + 1, iy + 1] += fx * fy
    return hist


def kde_deriche_2d(data, bins):
    """2D KDE via bilinear binning plus per-axis Deriche filtering.

    ``data`` must have shape ``(2, N)`` or ``(N, 2)``. The bandwidth is Scott's
    rule for a product kernel, so it is the diagonal of the Scott bandwidth
    matrix and the two axes are smoothed independently. Returns
    ``(x_coords, y_coords, pdf)`` with ``pdf[i, j]`` at ``(x_coords[i],
    y_coords[j])``; the PDF integrates to one over the grid.
    """
    data = np.asarray(data, dtype=np.float64)

    if data.ndim != 2 or (data.shape[0] != 2 and data.shape[1] != 2):
        raise ValueError("Data shape must be either (2, N) or (N, 2).")
    if data.shape[0] == 2:
        x = np.ascontiguousarray(data[0], dtype=np.float64)
        y = np.ascontiguousarray(data[1], dtype=np.float64)
    else:
        x = np.ascontiguousarray(data[:, 0], dtype=np.float64)
        y = np.ascontiguousarray(data[:, 1], dtype=np.float64)

    if len(x) < 4:
        raise ValueError("Need at least 4 samples for KDE.")
    if bins == 0:
        raise ValueError("Number of bins must be greater than 0.")
    non_finite = int((~np.isfinite(x)).sum() + (~np.isfinite(y)).sum())
    if non_finite:
        raise ValueError(
            f"Input data must be finite; found {non_finite} non-finite value(s)."
        )

    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))

    # Degenerate input on both axes: a normalised point mass at the centre.
    if abs(x_max - x_min) < 1e-9 and abs(y_max - y_min) < 1e-9:
        dx = dy = 1e-9 / bins
        x_coords = np.array(
            [x_min + (i + 0.5) * dx for i in range(bins)], dtype=np.float64
        )
        y_coords = np.array(
            [y_min + (i + 0.5) * dy for i in range(bins)], dtype=np.float64
        )
        pdf = np.zeros((bins, bins), dtype=np.float64)
        pdf[bins // 2, bins // 2] = 1.0 / (dx * dy)
        return x_coords, y_coords, pdf

    f = len(x) ** (-1.0 / 6.0)
    sigma_x = f * float(np.std(x, ddof=1))
    sigma_y = f * float(np.std(y, ddof=1))

    # ``max(1e-9)`` keeps the grid width positive when one axis is constant.
    xmin = x_min - 0.5 * max(sigma_x, 1e-9)
    xmax = x_max + 0.5 * max(sigma_x, 1e-9)
    ymin = y_min - 0.5 * max(sigma_y, 1e-9)
    ymax = y_max + 0.5 * max(sigma_y, 1e-9)

    hist = linear_binning_2d(x, y, xmin, xmax, ymin, ymax, bins)

    width_x = (xmax - xmin) / bins
    width_y = (ymax - ymin) / bins
    sigma_x_in_bins = sigma_x * bins / (xmax - xmin)
    sigma_y_in_bins = sigma_y * bins / (ymax - ymin)

    for i in range(bins):
        hist[i, :] = deriche_recursive_filter(hist[i, :], sigma_y_in_bins)
    for j in range(bins):
        hist[:, j] = deriche_recursive_filter(hist[:, j], sigma_x_in_bins)

    total = float(hist.sum())
    if abs(total) < 1e-9:
        if np.any(np.abs(hist) >= 1e-9):
            raise ValueError(
                "Normalization factor is near zero despite non-zero filtered "
                "histogram sum."
            )
    else:
        hist = hist / (total * width_x * width_y)

    x_coords = np.array(
        [xmin + (i + 0.5) * width_x for i in range(bins)], dtype=np.float64
    )
    y_coords = np.array(
        [ymin + (i + 0.5) * width_y for i in range(bins)], dtype=np.float64
    )
    return x_coords, y_coords, hist
