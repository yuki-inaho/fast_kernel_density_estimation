# Fast & Accurate Gaussian Kernel Density Estimation (Rust + Python)

## Overview

This project provides a **fast Kernel Density Estimation (KDE) library implemented in Rust** and **Python scripts for benchmarking and comparison**. It specifically focuses on an efficient Gaussian KDE method using the Deriche recursive filter, as proposed in the paper "Fast & Accurate Gaussian Kernel Density Estimation" by Jeffrey Heer.

The `kde_comparison.py` script is designed to compare the performance and accuracy of the Rust `fast_kde` implementation against `gaussian_kde` from Python's SciPy library and a naive KDE implementation JIT-compiled with Numba.

### About `kde_comparison.py`

`kde_comparison.py` is an exported version of the `kde_comparison.ipynb` Jupyter Notebook. It serves as a reference script for demonstration and result verification, and is primarily intended for interactive execution within a Jupyter Notebook environment. This file itself is not strictly necessary for the final distribution and might be removed in the future depending on the project's maturity.

## Installation & Setup

This project leverages PyO3 for Rust-Python interoperability and NumPy for numerical operations. We recommend using `uv` (a next-generation Python package installer) for efficient dependency management and installation.

### 1. Install Rust Toolchain

If you don't have Rust installed, use `rustup` to get it:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

After installation, reload your shell configuration to make the Rust environment available:

```bash
source $HOME/.cargo/env
```

### 2. Install uv

You can find detailed installation instructions for `uv` in its official documentation. This includes various methods like using a standalone installer, `pipx`, `cargo`, Homebrew, and more.

* **[Official uv Installation Guide](https://astral.sh/uv/install)**

### 3. Install Project Dependencies with uv

Navigate to the root directory of this project and use `uv` to install the dependencies. This command will simultaneously resolve and build both the Python dependencies listed in `pyproject.toml` and the Rust dependencies specified in `fast_kde/Cargo.toml`.

```bash
cd fast_kernel_density_estimation # Change to your project's root directory
uv sync
```

This command builds the Rust module and installs it into your Python environment (typically as a shared library). This is the recommended way to get started quickly.

### 4. Alternative: Install Rust Extension with Maturin

If you prefer a more direct control over the Rust extension build process or for development purposes, you can use `maturin` to build and install the `fast_kde` Rust library directly into your Python environment.

First, ensure `maturin` is installed in your Python environment:

```bash
pip install maturin
```

Then, from the root directory of the project, build and install the Rust extension in release mode:

```bash
cd fast_kernel_density_estimation
maturin develop --release
```

This command compiles the Rust code with optimizations and installs the `fast_kde` Python module in "editable" mode, meaning changes to the Rust source will be reflected on the next Python import (after recompilation).

## Benchmarking & Execution

You can perform KDE comparisons and benchmarks by opening and running `kde_comparison.ipynb` in a Jupyter environment.

## References

The Deriche filter-based KDE approximation method used in this project is based on the following research:

* **Jeffrey Heer.** "Fast & Accurate Gaussian Kernel Density Estimation." IEEE VIS Short Papers, 2021.
  * [Paper Link (IDL, University of Washington)](http://idl.cs.washington.edu/papers/fast-kde)
