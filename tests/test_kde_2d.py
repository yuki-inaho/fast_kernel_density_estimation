"""Invariants, toy problems and cross-implementation checks for the 2D KDE.

The 2D estimator is a product-kernel KDE: Scott's rule is applied per axis
(the diagonal of the Scott bandwidth matrix) and the Gaussian smoothing is
factorised into a row pass and a column pass, so the density can be compared
exactly against a direct product-kernel oracle.

``numba_reference`` mirrors the Rust extension step for step; the two are
compared at near machine precision. The toy problems pin the behaviours that
were wrong in the first version of the 2D code: the two axis bandwidths were
swapped whenever the axes had different variances, and samples that fell in
the last bin row or column were dropped instead of being clamped to the edge
node (visible as missing mass along the borders for uniform data).
"""

from __future__ import annotations

import numba_reference
import numpy as np
import pytest

import fast_kde

BINS = 96

IMPLEMENTATIONS = pytest.mark.parametrize(
    "kde",
    [pytest.param(fast_kde, id="rust"), pytest.param(numba_reference, id="numba")],
)


@pytest.fixture
def data() -> np.ndarray:
    """A deterministic, well-conditioned sample with different axis scales."""
    rng = np.random.default_rng(0)
    return np.vstack([rng.normal(0.0, 1.0, 1_000), rng.normal(0.0, 0.5, 1_000)])


def _diagonal_product_kernel(
    data: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Direct product-kernel KDE with Scott's per-axis bandwidth.

    This is the estimator the binned Deriche implementation approximates, so
    the two must agree up to the binning and filter discretisation error.
    """
    n = data.shape[1]
    f = n ** (-1.0 / 6.0)
    sigma_x = f * float(np.std(data[0], ddof=1))
    sigma_y = f * float(np.std(data[1], ddof=1))
    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
    pdf = np.zeros(grid_x.shape)
    for i in range(n):
        pdf += np.exp(
            -0.5
            * (
                ((grid_x - data[0, i]) / sigma_x) ** 2
                + ((grid_y - data[1, i]) / sigma_y) ** 2
            )
        )
    return pdf / (n * 2.0 * np.pi * sigma_x * sigma_y)


def _relative_l1(
    actual: np.ndarray, reference: np.ndarray, dx: float, dy: float
) -> float:
    return float(
        np.sum(np.abs(actual - reference)) * dx * dy / (np.sum(reference) * dx * dy)
    )


# --- Shape, grid and normalisation -------------------------------------------


@IMPLEMENTATIONS
def test_outputs_are_well_formed(kde, data: np.ndarray) -> None:
    x, y, pdf = kde.kde_deriche_2d(data, BINS)

    assert x.shape == (BINS,)
    assert y.shape == (BINS,)
    assert pdf.shape == (BINS, BINS)
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(y))
    assert np.all(np.isfinite(pdf))
    assert np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0)
    # The recursive filter rings slightly; keep the undershoot negligible.
    assert pdf.min() > -1e-4 * pdf.max()

    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    assert np.sum(pdf) * dx * dy == pytest.approx(1.0, abs=1e-9)


@IMPLEMENTATIONS
def test_accepts_both_array_layouts(kde, data: np.ndarray) -> None:
    """(2, N) and (N, 2) are the same dataset and must give the same answer."""
    x1, y1, pdf1 = kde.kde_deriche_2d(data, BINS)
    x2, y2, pdf2 = kde.kde_deriche_2d(np.ascontiguousarray(data.T), BINS)

    np.testing.assert_allclose(x1, x2, rtol=0, atol=0)
    np.testing.assert_allclose(y1, y2, rtol=0, atol=0)
    np.testing.assert_allclose(pdf1, pdf2, rtol=1e-12, atol=1e-15)


# --- Toy problems -------------------------------------------------------------


def test_toy_cluster_sits_on_the_correct_pair_of_axes() -> None:
    """A tight cluster must appear at its own (x, y), not transposed."""
    rng = np.random.default_rng(2)
    x0, y0 = 0.7, 0.2
    data = np.vstack(
        [x0 + 0.05 * rng.normal(size=3_000), y0 + 0.02 * rng.normal(size=3_000)]
    )

    x, y, pdf = fast_kde.kde_deriche_2d(data, 64)

    i, j = np.unravel_index(int(np.argmax(pdf)), pdf.shape)
    assert abs(x[i] - x0) <= 3 * (x[1] - x[0])
    assert abs(y[j] - y0) <= 3 * (y[1] - y[0])


def test_toy_uncorrelated_unequal_variances_matches_product_kernel() -> None:
    """Different axis variances must not swap the two bandwidths.

    The first version assigned the smaller eigenvalue of the bandwidth matrix
    to x and the larger one to y: for this data (std(x) = 3, std(y) = 0.3) the
    relative L1 error was 0.67, now it is below 0.01.
    """
    rng = np.random.default_rng(1)
    data = np.vstack([rng.normal(0.0, 3.0, 2_000), rng.normal(0.0, 0.3, 2_000)])

    x, y, pdf = fast_kde.kde_deriche_2d(data, BINS)
    oracle = _diagonal_product_kernel(data, x, y)

    assert _relative_l1(pdf, oracle, x[1] - x[0], y[1] - y[0]) < 0.01


def test_toy_correlated_data_preserves_the_marginals() -> None:
    """Correlation is not modelled, but the marginals must stay correct.

    Because the per-axis bandwidth is the diagonal of the Scott bandwidth
    matrix, the x-marginal of the product-kernel estimate equals the marginal
    of the full-covariance Gaussian KDE (whose kernel (0, 0) entry is the same
    variance). A strongly correlated cloud is the adversarial case for the
    axis-separable filter.
    """
    gaussian_kde = pytest.importorskip("scipy.stats").gaussian_kde

    rng = np.random.default_rng(3)
    z = rng.normal(0.0, 1.0, 2_000)
    data = np.vstack(
        [z + 0.2 * rng.normal(size=2_000), z + 0.2 * rng.normal(size=2_000)]
    )

    x, y, pdf = fast_kde.kde_deriche_2d(data, BINS)
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])

    reference = gaussian_kde(data[0])
    reference.set_bandwidth(bw_method=2_000 ** (-1.0 / 6.0))
    marginal_x = np.sum(pdf, axis=1) * dy
    reference_x = reference.evaluate(x)
    assert _relative_l1(marginal_x, reference_x, dx, 1.0) < 0.05


def test_toy_uniform_edges_conserve_mass() -> None:
    """The outermost bin row and column must keep their samples.

    With small bin counts the samples in the last half bin used to be silently
    dropped (the guard required ``k + 1 < bins`` for all four weights), which
    removed several percent of the mass along the borders of a uniform square.
    """
    rng = np.random.default_rng(4)
    data = np.vstack([rng.uniform(0.0, 1.0, 20_000), rng.uniform(0.0, 1.0, 20_000)])

    x, y, pdf = fast_kde.kde_deriche_2d(data, 16)
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    oracle = _diagonal_product_kernel(data, x, y)

    assert _relative_l1(pdf, oracle, dx, dy) < 0.06
    edge_estimate = float(np.sum(pdf[-1]) * dx * dy)
    edge_reference = float(np.sum(oracle[-1]) * dx * dy)
    assert edge_estimate == pytest.approx(edge_reference, rel=0.15)


# --- Cross-implementation and edge cases -------------------------------------


def test_rust_matches_numba_reference(data: np.ndarray) -> None:
    """The extension agrees with the independent Numba implementation."""
    x_rust, y_rust, pdf_rust = fast_kde.kde_deriche_2d(data, BINS)
    x_ref, y_ref, pdf_ref = numba_reference.kde_deriche_2d(data, BINS)

    np.testing.assert_allclose(x_rust, x_ref, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(y_rust, y_ref, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(pdf_rust, pdf_ref, rtol=1e-9, atol=1e-12)


def test_linear_binning_conserves_mass_and_first_moments() -> None:
    """The bilinear weights must keep both the mass and the mean.

    Samples are placed well inside the node range, where linear binning is
    exact for the first moment: sum(w) == n and sum(w * node) == sum(samples).
    This is the property the original edge guard broke for the border bins.
    """
    rng = np.random.default_rng(6)
    data = rng.uniform(-3.0, 3.0, size=(2, 1_000))
    bins = 64
    xmin, xmax = -4.0, 4.0
    ymin, ymax = -4.0, 4.0

    hist = numba_reference.linear_binning_2d(
        np.ascontiguousarray(data[0]),
        np.ascontiguousarray(data[1]),
        xmin,
        xmax,
        ymin,
        ymax,
        bins,
    )

    nodes_x = xmin + (np.arange(bins) + 0.5) * (xmax - xmin) / bins
    nodes_y = ymin + (np.arange(bins) + 0.5) * (ymax - ymin) / bins
    total = hist.sum()
    mean_x = float(np.sum(hist.sum(axis=1) * nodes_x) / total)
    mean_y = float(np.sum(hist.sum(axis=0) * nodes_y) / total)

    assert total == pytest.approx(1_000.0, rel=1e-12)
    assert mean_x == pytest.approx(float(data[0].mean()), abs=1e-12)
    assert mean_y == pytest.approx(float(data[1].mean()), abs=1e-12)


@IMPLEMENTATIONS
def test_rejects_too_few_samples(kde) -> None:
    with pytest.raises(ValueError, match="at least 4 samples"):
        kde.kde_deriche_2d(np.zeros((2, 3)), 16)


@IMPLEMENTATIONS
def test_rejects_zero_bins(kde, data: np.ndarray) -> None:
    with pytest.raises(ValueError, match="bins must be greater than 0"):
        kde.kde_deriche_2d(data, 0)


@IMPLEMENTATIONS
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "-inf"])
def test_rejects_non_finite_data(kde, bad: float) -> None:
    x = np.arange(10.0)
    y = np.arange(10.0)
    x[5] = bad

    with pytest.raises(ValueError, match="found 1 non-finite"):
        kde.kde_deriche_2d(np.vstack([x, y]), 16)


@IMPLEMENTATIONS
def test_reports_the_number_of_non_finite_values(kde) -> None:
    data = np.zeros((2, 10))
    data[0, 3] = np.nan
    data[1, 7] = np.inf

    with pytest.raises(ValueError, match="found 2 non-finite"):
        kde.kde_deriche_2d(data, 16)


@IMPLEMENTATIONS
@pytest.mark.parametrize(
    "shape", [(1, 10), (0, 10), (3, 4)], ids=["one-row", "no-rows", "four-rows"]
)
def test_rejects_invalid_shapes_without_panicking(kde, shape: tuple[int, int]) -> None:
    """Malformed shapes must raise ValueError, not abort the interpreter."""
    with pytest.raises(ValueError, match="shape must be either"):
        kde.kde_deriche_2d(np.zeros(shape), 16)


@IMPLEMENTATIONS
@pytest.mark.parametrize("bins", [1, 2, 3], ids=["one", "two", "three"])
def test_small_bin_counts_are_finite(kde, data: np.ndarray, bins: int) -> None:
    x, y, pdf = kde.kde_deriche_2d(data, bins)

    assert pdf.shape == (bins, bins)
    assert np.all(np.isfinite(pdf))
    if bins > 1:
        dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
        assert np.sum(pdf) * dx * dy == pytest.approx(1.0, abs=1e-9)


@IMPLEMENTATIONS
def test_constant_data_is_normalized(kde) -> None:
    """A degenerate 2D sample yields a normalised point mass at its centre."""
    data = np.vstack([np.full(10, 1.0), np.full(10, 2.0)])

    x, y, pdf = kde.kde_deriche_2d(data, 32)

    assert np.count_nonzero(pdf) == 1
    assert pdf.max() == pytest.approx(1.0 / (1e-9 / 32) ** 2)
    assert np.sum(pdf) * (x[1] - x[0]) * (y[1] - y[0]) == pytest.approx(1.0, abs=1e-3)


@IMPLEMENTATIONS
def test_one_axis_constant_is_normalized(kde) -> None:
    """A sample on a line is a valid input and still integrates to one."""
    rng = np.random.default_rng(5)
    data = np.vstack([np.full(500, 3.0), rng.normal(0.0, 1.0, 500)])

    x, y, pdf = kde.kde_deriche_2d(data, 64)

    assert np.all(np.isfinite(pdf))
    assert pdf.max() > 0.0
    assert np.sum(pdf) * (x[1] - x[0]) * (y[1] - y[0]) == pytest.approx(1.0, abs=1e-3)
