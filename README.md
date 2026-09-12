# Fast & Accurate Gaussian Kernel Density Estimation (Rust + Python)

One- and two-dimensional Gaussian kernel density estimators implemented in Rust and
exposed to Python through PyO3. They combine **linear binning** with a **K=4 Deriche
recursive filter** to approximate Gaussian smoothing in time independent of the kernel
width, following Jeffrey Heer's *"Fast & Accurate Gaussian Kernel Density Estimation"*.

## Public API

The compiled `fast_kde` module exports three functions:

| Function | Returns |
| :--- | :--- |
| `kde_deriche(data, bins, sigma)` | `(x_coords, pdf_values)` — bin centres over `[min(data), max(data)]` and the PDF, normalised to integrate to 1 |
| `kde_mode_deriche(data, bins, sigma)` | `float` — the `x` coordinate where the PDF is maximal |
| `kde_deriche_2d(data, bins)` | `(x_coords, y_coords, pdf)` — square grid of bin centres and the PDF of shape `(bins, bins)`, where `pdf[i, j]` is the density at `(x_coords[i], y_coords[j])` |

The 1D functions raise `ValueError` for fewer than four samples, `bins == 0`, a
degenerate bin width, or an inconsistent normalisation. `kde_deriche_2d` accepts
`data` shaped `(2, N)` or `(N, 2)` and picks its bandwidth automatically with Scott's
rule (see below); it validates the shape, the sample count, the bin count and
finiteness, and raises `ValueError` otherwise. Constant input is allowed and yields a
normalised point mass, mirroring the 1D behaviour.

## Requirements

* A stable Rust toolchain — install with [rustup](https://rustup.rs), then `source $HOME/.cargo/env`
* [uv](https://docs.astral.sh/uv/getting-started/installation/) for Python environment management

Python packaging uses the **Maturin** build backend (`[tool.maturin]` in
`pyproject.toml`), so `uv sync` compiles the Rust extension and installs it as
`fast_kde` — no separate `maturin develop` step is needed.

## Setup

```bash
uv sync
```

To rebuild the extension after changing Rust sources:

```bash
uv sync --reinstall-package fast-kernel-density-estimation
```

`uv sync` installs exactly the groups you name, so add back any extras you were
using in the same command, e.g.
`uv sync --extra benchmark --reinstall-package fast-kernel-density-estimation`.

Verify the extension is really built (importing `fast_kde` alone is not enough —
the source directory would resolve as an empty namespace package):

```bash
uv run python -c "import fast_kde; print(fast_kde.kde_deriche, fast_kde.kde_mode_deriche, fast_kde.kde_deriche_2d)"
```

### Optional dependency groups

| Group | Install | Contents |
| :--- | :--- | :--- |
| `benchmark` | `uv sync --extra benchmark` | SciPy, Numba, Matplotlib — needed for the test suite and benchmarks |
| `notebook` | `uv sync --extra notebook` | JupyterLab, ipykernel — needed for `kde_comparison.ipynb` |

Combine them when you need both: `uv sync --extra benchmark --extra notebook`.

## Usage

```python
import numpy as np
import fast_kde

data = np.random.default_rng(0).normal(0.0, 1.0, 1_000)

x, pdf = fast_kde.kde_deriche(data, 512, 0.2)
mode = fast_kde.kde_mode_deriche(data, 512, 0.2)

print(x.shape, pdf.shape)          # (512,) (512,)
print(np.trapezoid(pdf, x))        # ~1.0
print(mode)                        # location of the density peak

# 2D: data of shape (2, N) or (N, 2); the bandwidth is chosen automatically
rng = np.random.default_rng(1)
points = np.vstack([rng.normal(0.0, 1.0, 2_000), rng.normal(0.0, 0.5, 2_000)])
gx, gy, gpdf = fast_kde.kde_deriche_2d(points, 128)

print(gx.shape, gy.shape, gpdf.shape)  # (128,) (128,) (128, 128)
```

### 2D estimator

`kde_deriche_2d` uses a product Gaussian kernel, so the K=4 Deriche filter is applied
once along every row and once along every column of the binned grid. The bandwidth is
Scott's rule for a product kernel, `sigma_j = n**(-1/6) * std_j` with `numpy.std(x,
ddof=1)`: this is the diagonal of the Scott bandwidth matrix. The correlation between
the axes is deliberately not modelled — as a product kernel it cannot be represented
by per-axis filtering — but the diagonal choice keeps the *marginals* equal to those
of the full-covariance Gaussian KDE with the same Scott bandwidth. The grid is padded
by `0.5 * sigma` on each side, and the bilinear linear-binning rule follows Heer
(2021, Section 3); samples beyond the outermost node are clamped to that node so no
weight is lost.

## Tests

```bash
uv run --extra benchmark pytest -q
```

The suite checks that the extension is exported at all, that output shapes,
finiteness, grid monotonicity and PDF normalisation hold, that malformed input is
rejected, and that the extension agrees with the Numba reference implementation in
`benchmarks/numba_reference.py` to `rtol=1e-9, atol=1e-12`. The 2D suite adds toy
problems for unequal axis variances, strong correlation, uniform-data edges and
degenerate input, and compares the estimator against a direct product-kernel oracle.

## Benchmark

```bash
uv run --extra benchmark python benchmarks/benchmark_kde.py --quick --output-dir outputs/smoke
```

Compares the Rust extension, the Numba reference and `scipy.stats.gaussian_kde` on
three fixed-seed datasets, reporting wall-clock time, the PDF integral and the
estimated mode. Drop `--quick` for the full size (n=50,000, bins=1,024) and add
`--plot` to also write a timings chart. `--output-dir` is required — nothing is
written outside it.

## Repository layout

```text
fast_kde/            Rust crate (PyO3 extension module)
benchmarks/          verification-only code, not part of the installed package
  numba_reference.py Numba implementation of the same algorithm
  benchmark_kde.py   benchmark CLI
tests/               pytest suite
kde_comparison.ipynb exploratory notebook (needs the notebook extra)
```

## Generated artifacts

Benchmark output (`outputs/`), build artifacts (`dist/`, `*.so`), and Python/Numba/
Ruff/pytest caches are git-ignored and are never committed — regenerate them by
rerunning the commands above.

## Accuracy

The Deriche approximation is checked against an exact Gaussian convolution of the
same binned histogram. The maximum error is below **0.011% of the peak** across
bandwidths from 1.9 to 37.5 bins, and it does not grow with the bandwidth.
`kde_deriche_2d` runs the same filter once per axis. It is pinned against a direct
product-kernel KDE with Scott's per-axis bandwidth (relative L1 below 0.01 on
uncorrelated data) and, for strongly correlated data, its marginals are pinned
against the full-covariance Gaussian KDE with the same Scott bandwidth.

`scipy.stats.gaussian_kde` is used as an independent cross-check in the benchmark.
It evaluates the kernel sum exactly while this library smooths a binned grid, so
the two do not agree pointwise -- expect a few percent, more at very small
bandwidths where the estimate tracks sampling noise. The estimated **mode** does
agree, to within a grid step, wherever the mode is well determined.

Two caveats when reading benchmark output:

- On a uniform distribution the density is flat, so `argmax` is decided by
  sampling noise rather than by a real peak. The reported modes will differ
  between methods and between runs; this is a property of the question, not a
  defect in either implementation.
- Both methods roll off near the data boundary, since neither extends support
  past `min(data)` and `max(data)`.

A large mode disagreement on unimodal data is worth investigating -- it is how the
filter bug described below was found.

### Fixed: saturating filter width

Earlier revisions kept only the real part of the complex coefficient pairs in
equation (2) of the paper and applied the four poles as a cascade. The filter
width then saturated near 11 bins regardless of `sigma`: the error against an
exact Gaussian grew to 43% at large bandwidths and the estimated mode stopped
responding to `sigma` at all. The filter now expands the complex pairs into a
4th-order IIR with a causal and an anticausal pass, as the paper describes.
`tests/test_kde.py` pins both the accuracy and the mode convergence.

## Quality gates

```bash
uv run ruff format --check benchmarks tests
uv run ruff check benchmarks tests
cargo fmt --manifest-path fast_kde/Cargo.toml -- --check
cargo clippy --manifest-path fast_kde/Cargo.toml --all-targets -- -D warnings
uv build   # produces a platform wheel containing the compiled extension
```

## References

* **Jeffrey Heer.** "Fast & Accurate Gaussian Kernel Density Estimation." IEEE VIS Short Papers, 2021.
  * [Paper](http://idl.cs.washington.edu/papers/fast-kde)
* **Rachid Deriche.** "Fast Algorithms for Low-Level Vision." IEEE TPAMI, 1990.
  * The recursive Gaussian approximation whose poles Heer's equation (2) expands.
* **Pascal Getreuer.** "A Survey of Gaussian Convolution Algorithms." IPOL, 2013.
  * Source of the anticausal numerator relation used by the recursive filter.
