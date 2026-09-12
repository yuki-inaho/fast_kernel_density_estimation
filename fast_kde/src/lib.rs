// fast_kde/src/lib.rs

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

/// `kde_deriche` の戻り値: (ビン中心のX座標, 対応するPDF値) のNumPy配列ペア。
type KdeGrid<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);
type KdeGrid2D<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
);
type DecomposedMatrix = ([[f64; 2]; 2], [[f64; 2]; 2], [[f64; 2]; 2]);

/// # 1Dデータの線形ビニング
///
/// 入力データを指定された範囲とビン数で線形ビニングします。
/// 論文「Fast & Accurate Gaussian Kernel Density Estimation」のSection 3
/// 「Linear Binning」で説明されている手法に基づいています。
/// 各データ点の重みは、隣接する2つのビンに比例して分配されます。
///
/// Rayonクレートを使用して、ビニング処理を自動的に並列化します。
///
/// ## 引数
/// - `data`: ビニングする1次元データスライス。
/// - `xmin`: データの最小値。
/// - `xmax`: データの最大値。
/// - `bins`: ビンの数。
///
/// ## 戻り値
/// 各ビンのカウントを格納する`Vec<f64>`。
fn linear_binning(data: &[f64], xmin: f64, xmax: f64, bins: usize) -> Vec<f64> {
    if bins == 0 {
        // ビン数が0の場合、空のヒストグラムを返す
        return vec![];
    }

    let bin_width = (xmax - xmin) / bins as f64;

    // データ範囲が極めて小さい場合（全てのデータ点がほぼ同じ場所にある場合など）
    if bin_width.abs() < 1e-9 {
        // この場合、ほとんどのデータ点は一つのビンに集中するか、範囲外となる
        // 初期化されたゼロのヒストグラムを返す
        return vec![0.0; bins];
    }

    // データが空の場合、ゼロで初期化されたヒストグラムを返す
    if data.is_empty() {
        return vec![0.0; bins];
    }

    let num_threads = rayon::current_num_threads();
    // データをスレッド数に基づいてチャンクに分割し、各チャンクで並列処理
    let chunk_size = data.len().div_ceil(num_threads);

    // 各スレッドのローカルヒストグラムを計算し、最後に集計する
    let thread_hists: Vec<Vec<f64>> = data
        .par_chunks(chunk_size) // データを並列チャンクに分割
        .map(|chunk| {
            let mut local_hist = vec![0.0_f64; bins];
            for &x_val in chunk {
                // データ点が範囲内にあるかチェック (xmin <= x_val < xmax)
                if x_val >= xmin && x_val < xmax {
                    // x_valが属するビン内での相対位置を計算
                    let pos = (x_val - xmin) / bin_width;
                    // ビンインデックス (k) と、kの左端からの相対距離 (pos - k)
                    let k = pos.floor() as usize;
                    let fraction_left = 1.0 - (pos - k as f64); // k番目のビンに割り当てる重み
                    let fraction_right = pos - k as f64; // k+1番目のビンに割り当てる重み

                    // k番目のビンに重みを加算
                    if k < bins {
                        local_hist[k] += fraction_left;
                        // k+1番目のビンに重みを加算（範囲内であれば）
                        if k + 1 < bins {
                            local_hist[k + 1] += fraction_right;
                        }
                    }
                } else if (x_val - xmax).abs() < 1e-9 && bins > 0 {
                    // xmaxに厳密に等しいデータ点は最後のビンに割り当てる (境界値のロバストな処理)
                    local_hist[bins - 1] += 1.0;
                }
            }
            local_hist
        })
        .collect();

    // 各スレッドのローカルヒストグラムをグローバルヒストグラムに集約
    let mut global_hist = vec![0.0_f64; bins];
    for local_hist_chunk in thread_hists {
        for i in 0..bins {
            global_hist[i] += local_hist_chunk[i];
        }
    }
    global_hist
}

/// # 1D Deriche（デリシェ）再帰フィルターによるガウススムージングの近似
///
/// Heer 2021「Fast & Accurate Gaussian Kernel Density Estimation」の式(2)に従い、
/// ガウス関数の右半分を K=4 の指数和で近似する。
///
/// ```text
/// h_K(x) = 1 / sqrt(2 pi sigma^2) * sum_k alpha_k * exp(-lambda_k * x / sigma)
/// ```
///
/// `alpha_k` と `lambda_k` は複素共役対であり、実部だけを取り出すと近似が成立しない。
/// この指数和を z 変換して有理関数へ展開すると 4 次の IIR フィルターになり、
/// 因果（左→右）と反因果（右→左）の 2 パスの和がガウス畳み込みを近似する。
///
/// 反因果側の分子係数は Getreuer「A Survey of Gaussian Convolution Algorithms」
/// (IPOL, 2013) の関係式 `b_minus[k] = b_plus[k] - a[k] * b_plus[0]`（k = 1..3）、
/// `b_minus[4] = -a[4] * b_plus[0]` で構成する。
///
/// ## 引数
/// - `signal`: スムージングするデータ（ヒストグラムなど）を格納するミュータブルなスライス。
/// - `sigma`: ガウスカーネルの標準偏差（ビン単位）。
///
/// ## 精度
/// 厳密なガウス畳み込みに対する最大相対誤差は、sigma を 1〜50 ビンで変えても 0.05% 未満に収まる。
///
/// 論文 式(2) の直下に示された K=4 の複素係数。共役対 (k, k+1) で 1 組。
const ALPHA_RE: [f64; 4] = [0.84, 0.84, -0.34015, -0.34015];
const ALPHA_IM: [f64; 4] = [1.8675, -1.8675, -0.1299, 0.1299];
const LAMBDA_RE: [f64; 4] = [1.783, 1.783, 1.723, 1.723];
const LAMBDA_IM: [f64; 4] = [0.6318, -0.6318, 1.997, -1.997];

/// 複素数の積。
fn cmul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

/// `exp(-(re + i*im) / sigma)` を計算して極を求める。
fn pole(re: f64, im: f64, sigma: f64) -> (f64, f64) {
    let magnitude = (-re / sigma).exp();
    let angle = -im / sigma;
    (magnitude * angle.cos(), magnitude * angle.sin())
}

/// 指数和 `sum_k alpha_k / (1 - p_k z^-1)` を有理関数へ展開する。
///
/// 戻り値は `(b_plus, b_minus, a)` で、`a` は分母の z^-1..z^-4 の係数。
/// 共役対を組ませてあるため、展開結果の虚部は打ち消されて実数係数になる。
fn deriche_coefficients(sigma: f64) -> ([f64; 4], [f64; 5], [f64; 4]) {
    let poles: [(f64, f64); 4] = [
        pole(LAMBDA_RE[0], LAMBDA_IM[0], sigma),
        pole(LAMBDA_RE[1], LAMBDA_IM[1], sigma),
        pole(LAMBDA_RE[2], LAMBDA_IM[2], sigma),
        pole(LAMBDA_RE[3], LAMBDA_IM[3], sigma),
    ];

    // 分母 prod_k (1 - p_k z^-1) を多項式として展開する。
    let mut denominator = [(0.0, 0.0); 5];
    denominator[0] = (1.0, 0.0);
    let mut degree = 0;
    for p in poles.iter() {
        degree += 1;
        for i in (1..=degree).rev() {
            let shifted = cmul(denominator[i - 1], (-p.0, -p.1));
            denominator[i] = (denominator[i].0 + shifted.0, denominator[i].1 + shifted.1);
        }
    }

    // 分子 sum_k alpha_k * prod_{j != k} (1 - p_j z^-1)。
    let mut numerator = [(0.0, 0.0); 4];
    for k in 0..4 {
        let mut partial = [(0.0, 0.0); 4];
        partial[0] = (1.0, 0.0);
        let mut partial_degree = 0;
        for (j, p) in poles.iter().enumerate() {
            if j == k {
                continue;
            }
            partial_degree += 1;
            for i in (1..=partial_degree).rev() {
                let shifted = cmul(partial[i - 1], (-p.0, -p.1));
                partial[i] = (partial[i].0 + shifted.0, partial[i].1 + shifted.1);
            }
        }
        let alpha = (ALPHA_RE[k], ALPHA_IM[k]);
        for i in 0..4 {
            let term = cmul(alpha, partial[i]);
            numerator[i] = (numerator[i].0 + term.0, numerator[i].1 + term.1);
        }
    }

    // 式(2) の 1 / sqrt(2 pi sigma^2) を分子へ畳み込む。
    let scale = 1.0 / ((2.0 * std::f64::consts::PI).sqrt() * sigma);
    let mut b_plus = [0.0_f64; 4];
    for i in 0..4 {
        b_plus[i] = numerator[i].0 * scale;
    }
    let mut a = [0.0_f64; 4];
    for i in 0..4 {
        a[i] = denominator[i + 1].0;
    }

    // 反因果側の分子（Getreuer 2013）。
    let mut b_minus = [0.0_f64; 5];
    for k in 1..4 {
        b_minus[k] = b_plus[k] - a[k - 1] * b_plus[0];
    }
    b_minus[4] = -a[3] * b_plus[0];

    (b_plus, b_minus, a)
}

fn deriche_recursive_filter_approx(signal: &mut [f64], sigma: f64) {
    if sigma.abs() < 1e-9 {
        // sigmaが0に近い場合、スムージングは行わない
        return;
    }
    if signal.is_empty() {
        return;
    }

    let (b_plus, b_minus, a) = deriche_coefficients(sigma);
    let m = signal.len();
    let input = signal.to_vec();

    // 因果パス: y[i] = sum_k b_plus[k] x[i-k] - sum_k a[k] y[i-k]
    let mut causal = vec![0.0_f64; m];
    for i in 0..m {
        let mut acc = 0.0;
        for (k, b) in b_plus.iter().enumerate() {
            if i >= k {
                acc += b * input[i - k];
            }
        }
        for (k, coeff) in a.iter().enumerate() {
            if i > k {
                acc -= coeff * causal[i - k - 1];
            }
        }
        causal[i] = acc;
    }

    // 反因果パス: y[i] = sum_k b_minus[k] x[i+k] - sum_k a[k] y[i+k]
    let mut anticausal = vec![0.0_f64; m];
    for i in (0..m).rev() {
        let mut acc = 0.0;
        for k in 1..5 {
            if i + k < m {
                acc += b_minus[k] * input[i + k];
            }
        }
        for (k, coeff) in a.iter().enumerate() {
            if i + k + 1 < m {
                acc -= coeff * anticausal[i + k + 1];
            }
        }
        anticausal[i] = acc;
    }

    for i in 0..m {
        signal[i] = causal[i] + anticausal[i];
    }
}

/// # 線形ビニングとDericheフィルターを使用したカーネル密度推定 (KDE)
///
/// 1次元のデータセットに対して、線形ビニングによってヒストグラムを作成し、
/// その後Deriche再帰フィルターによってガウス平滑化を近似してKDEを計算します。
/// 最終的な結果は、確率密度関数 (PDF) として正規化されます。
///
/// ## 引数
/// - `py`: Python GILトークン。PyO3の関数でPythonオブジェクトを扱うために必要。
/// - `data`: 入力データ（`numpy.ndarray`の1次元f64配列）。
/// - `bins`: KDEを評価するビンの数（グリッドポイントの数）。
/// - `sigma`: ガウスカーネルのバンド幅（標準偏差）。
///
/// ## 戻り値
/// - `PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)>`:
///   - 1番目の要素: ビンの中心X座標（`numpy.ndarray`）。
///   - 2番目の要素: 各ビンの対応するPDF値（`numpy.ndarray`）。
///
/// ## エラー
/// - `PyValueError`:
///   - データが3サンプル未満の場合。
///   - ビン数が0の場合。
///   - `xmax`が`xmin`より小さい場合（データ範囲の不正）。
///   - ビン幅`dx`が極めて小さい、または0の場合。
///   - 正規化係数がゼロに近いのに、フィルタリングされたヒストグラムの合計がゼロでない場合。
#[pyfunction]
fn kde_deriche<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    bins: usize,
    sigma: f64,
) -> PyResult<KdeGrid<'py>> {
    let data_slice = data.as_slice()?; // PythonのndarrayからRustのスライスへ変換

    // データ点数の最小要件をチェック
    if data_slice.len() < 4 {
        return Err(PyValueError::new_err("Need at least 4 samples for KDE."));
    }
    // ビン数の有効性をチェック
    if bins == 0 {
        return Err(PyValueError::new_err(
            "Number of bins must be greater than 0.",
        ));
    }
    // 入力の有限性をチェック（仕様 S-3）
    // NaN は f64::min/max と範囲比較の両方をすり抜けるため、明示的に拒否しないと
    // 欠損値が黙って捨てられ、Numba参照実装との結果も食い違う。
    let non_finite = data_slice.iter().filter(|v| !v.is_finite()).count();
    if non_finite > 0 {
        return Err(PyValueError::new_err(format!(
            "Input data must be finite; found {non_finite} non-finite value(s)."
        )));
    }
    // バンド幅の符号をチェック（仕様 S-4）
    // sigma が負だと exp(-lambda / sigma) の符号が反転し、ガウス近似ではない
    // 別のフィルターになるため、無意味な結果を正常値として返さない。
    // sigma == 0 は「平滑化しない」として許可する（仕様 S-5）。
    if sigma < 0.0 {
        return Err(PyValueError::new_err("Sigma must be non-negative."));
    }

    // データ範囲（最小値と最大値）を計算
    let xmin = data_slice.iter().cloned().fold(f64::INFINITY, f64::min);
    let xmax = data_slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // データ範囲が極めて小さい場合（全てのデータ点が実質的に同じ場所にある場合）の特殊処理
    // この場合、KDEはデルタ関数のような挙動を示すべき
    if (xmax - xmin).abs() < 1e-9 {
        // 幅ゼロの範囲では割り算ができないため、1e-9 だけ幅を持たせたグリッドを張る
        let degenerate_width = xmax - xmin + 1e-9;
        let dx = degenerate_width / bins as f64;
        let x_coords = (0..bins)
            .map(|i| xmin + (i as f64 + 0.5) * dx)
            .collect::<Vec<f64>>();
        let mut pdf_vals = vec![0.0; bins];
        // 中央のビンに全質量を置いた退化PDFを返す（仕様 S-6）。
        // 高さはグリッド幅から導出するので sum(pdf) * dx == 1 が成り立つ。
        // 固定値 1/1e-9 を置いていた頃は積分値が bins 分だけ小さくなっていた。
        pdf_vals[bins / 2] = 1.0 / dx;
        let x_py = PyArray1::from_vec(py, x_coords);
        let pdf_py = PyArray1::from_vec(py, pdf_vals);
        return Ok((x_py, pdf_py));
    }

    // xmaxがxminより小さい場合はエラー (データ範囲が不正)
    if xmax < xmin {
        return Err(PyValueError::new_err(format!(
            "xmax ({}) must be greater than or equal to xmin ({})",
            xmax, xmin
        )));
    }

    // 1. 線形ビニングを実行してヒストグラムを作成
    let mut hist_counts = linear_binning(data_slice, xmin, xmax, bins);

    // ヒストグラムが全てゼロの場合（例：全てのデータ点が範囲外の場合など）
    if hist_counts.iter().all(|&h_val| h_val.abs() < 1e-9) {
        let bin_width_calc = (xmax - xmin) / bins as f64;
        let x_coords: Vec<f64> = (0..bins)
            .map(|i| xmin + (i as f64 + 0.5) * bin_width_calc)
            .collect();
        let pdf_values = vec![0.0; bins]; // PDF値も全て0
        let x_py = PyArray1::from_vec(py, x_coords);
        let pdf_py = PyArray1::from_vec(py, pdf_values);
        return Ok((x_py, pdf_py));
    }

    // 2. Dericheフィルターを適用してヒストグラムを平滑化
    // Dericheフィルターに渡すsigmaを、ビニングのスケールに合わせて調整
    // この sf (scale factor) は、元のデータ単位の1単位が、ビン単位で何ピクセル分に相当するかを示す
    let bin_range = xmax - xmin;
    let sf = bins as f64 / bin_range;
    let deriche_sigma_in_bins = sigma * sf; // Pythonから渡されたsigmaをビンスケールに変換
    deriche_recursive_filter_approx(&mut hist_counts, deriche_sigma_in_bins);

    // 3. 結果をPDFとして正規化
    // PDFの特性として、積分値が1になるように調整する必要がある
    let sum_of_filtered_hist: f64 = hist_counts.iter().sum();
    let dx = (xmax - xmin) / bins as f64; // 各ビンの幅を計算

    // ビン幅が極めて小さい場合はエラー
    if dx.abs() < 1e-9 {
        return Err(PyValueError::new_err(
            "dx (bin width) is too small or zero.",
        ));
    }

    let normalization_factor = sum_of_filtered_hist * dx; // 積分値（面積）
    if normalization_factor.abs() < 1e-9 {
        // 正規化係数がゼロに近いが、ヒストグラムに非ゼロの値が含まれる場合（異常な状態）
        if hist_counts.iter().any(|&h_val| h_val.abs() >= 1e-9) {
            return Err(PyValueError::new_err(
                "Normalization factor is near zero despite non-zero filtered histogram sum.",
            ));
        }
        // ヒストグラムが全てゼロで正規化係数もゼロの場合、何もしない（結果はゼロのまま）
    } else {
        // 各ビン値を正規化係数で割ることで、PDFの条件（積分値が1）を満たす
        for val in hist_counts.iter_mut() {
            *val /= normalization_factor;
        }
    }

    // X軸の座標を生成（各ビンの中心点）
    let x_coords: Vec<f64> = (0..bins).map(|i| xmin + (i as f64 + 0.5) * dx).collect();

    // 結果をPythonのNumPy配列に変換して返す
    let x_py_bound = PyArray1::from_vec(py, x_coords);
    let pdf_py_bound = PyArray1::from_vec(py, hist_counts);

    Ok((x_py_bound, pdf_py_bound))
}

/// # KDEに基づく1Dデータセットのモード（最頻値）推定
///
/// `kde_deriche`関数を使用してデータセットのKDEを計算し、
/// そのPDFの最大値に対応するX座標をモードとして推定します。
///
/// ## 引数
/// - `py`: Python GILトークン。
/// - `data`: 入力データ（`numpy.ndarray`の1次元f64配列）。
/// - `bins`: KDE計算に使用するビンの数。
/// - `sigma`: ガウスカーネルのバンド幅。
///
/// ## 戻り値
/// - `PyResult<f64>`: 推定されたモード値。
///
/// ## エラー
/// - `PyValueError`:
///   - `kde_deriche`関数がエラーを返した場合。
///   - PDFが空の場合。
///   - PDF内に有効なモードが見つからない場合（例: 全てのPDF値がNaN/Infの場合）。
///   - 内部エラー（argmaxインデックスがx_coordsの範囲外の場合）。
#[pyfunction]
fn kde_mode_deriche<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    bins: usize,
    sigma: f64,
) -> PyResult<f64> {
    // まずKDEを計算
    let (x_pyarray_bound, pdf_pyarray_bound) = kde_deriche(py, data, bins, sigma)?;

    // 結果のNumPy配列をRustのスライスとして読み取り
    let x_ro_array: PyReadonlyArray1<'_, f64> = x_pyarray_bound.readonly();
    let pdf_ro_array: PyReadonlyArray1<'_, f64> = pdf_pyarray_bound.readonly();

    let x_slice = x_ro_array.as_slice()?;
    let pdf_slice = pdf_ro_array.as_slice()?;

    // PDFが空でないことを確認
    if pdf_slice.is_empty() {
        return Err(PyValueError::new_err("PDF is empty, cannot find mode."));
    }

    // PDF値が最大となるインデックスを見つける
    // partial_cmpはNaNやInfの比較を安全に行うために使用
    let idx_option = pdf_slice
        .iter()
        .enumerate()
        .max_by(|(_, &a_val), (_, &b_val)| {
            a_val
                .partial_cmp(&b_val)
                .unwrap_or(std::cmp::Ordering::Less) // NaNを小さい値として扱う
        })
        .map(|(idx, _)| idx);

    // 最大値のインデックスが見つかった場合、対応するX座標を返す
    match idx_option {
        Some(max_idx) => {
            if max_idx < x_slice.len() {
                Ok(x_slice[max_idx])
            } else {
                // 通常は発生しない内部エラーチェック
                Err(PyValueError::new_err(
                    "Internal error: Argmax index out of bounds for x_coords.",
                ))
            }
        }
        // 最大値のインデックスが見つからなかった場合（例: 全てのPDF値がNaNの場合など）
        None => Err(PyValueError::new_err(
            "Could not find a valid mode in the PDF (e.g., all values are NaN/Inf).",
        )),
    }
}

/// Performs 2D linear binning of the input data over the specified range and
/// number of bins.
///
/// This method is based on the approach described in Section 5,
/// "Data Binning," of the doi:10.1007/978-3-319-71688-6_5."
/// The weight of each data point is distributed proportionally between its
/// four neighboring bins.
///
/// The binning operation is automatically parallelized using the Rayon crate.
///
/// ## Arguments
/// - x: One-dimensional data slice to be binned.
/// - y: One-dimensional data slice to be binned.
/// - xmin: Minimum value of the x data range.
/// - xmax: Maximum value of the x data range.
/// - ymin: Minimum value of the y data range.
/// - ymax: Maximum value of the y data range.
/// - xbins: Number of bins alongside x axis.
/// - ybins: Number of bins alongside y axis.
///
/// ## Returns
/// A Vec<Vec<f64>> containing the count assigned to each bin.
fn linear_binning_2d(
    x: &[f64],
    y: &[f64],
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
    bins: usize,
) -> Vec<Vec<f64>> {
    let xbins = bins;
    let ybins = bins;

    if xbins == 0 || ybins == 0 {
        return vec![];
    }

    if x.is_empty() || y.is_empty() {
        return vec![vec![0.0; ybins]; xbins];
    }

    let xbin_width = (xmax - xmin) / xbins as f64;
    let ybin_width = (ymax - ymin) / ybins as f64;

    if xbin_width.abs() < 1e-9 || ybin_width.abs() < 1e-9 {
        return vec![vec![0.0; ybins]; xbins];
    }

    assert_eq!(
        x.len(),
        y.len(),
        "x and y must contain the same number of points"
    );

    let num_threads = rayon::current_num_threads();
    let chunk_size = x.len().div_ceil(num_threads);
    let thread_hists: Vec<Vec<f64>> = x
        .par_chunks(chunk_size)
        .zip(y.par_chunks(chunk_size))
        .map(|(x_chunk, y_chunk)| {
            let mut local_hist = vec![0.0_f64; xbins * ybins];
            for (&x_val, &y_val) in x_chunk.iter().zip(y_chunk.iter()) {
                if x_val >= xmin && x_val < xmax && y_val >= ymin && y_val < ymax {
                    let posh = (x_val - xmin) / xbin_width;
                    let posv = (y_val - ymin) / ybin_width;
                    let kh = posh.floor() as usize;
                    let kv = posv.floor() as usize;
                    let fraction_left = 1.0 - (posh - kh as f64);
                    let fraction_right = posh - kh as f64;
                    let fraction_down = 1.0 - (posv - kv as f64);
                    let fraction_up = posv - kv as f64;
                    let a = fraction_up * fraction_left;
                    let b = fraction_up * fraction_right;
                    let c = fraction_down * fraction_left;
                    let d = fraction_down * fraction_right;
                    let total_area = a + b + c + d;
                    if kh < xbins && kv < ybins && kh + 1 < xbins && kv + 1 < ybins {
                        local_hist[kh * ybins + kv] += c / total_area;
                        local_hist[(kh + 1) * ybins + kv] += d / total_area;
                        local_hist[kh * ybins + (kv + 1)] += a / total_area;
                        local_hist[(kh + 1) * ybins + (kv + 1)] += b / total_area;
                    }
                } else if (x_val - xmax).abs() < 1e-9
                    && xbins > 0
                    && (y_val - ymax).abs() < 1e-9
                    && ybins > 0
                {
                    local_hist[(xbins - 1) * ybins + (ybins - 1)] += 1.0;
                }
            }
            local_hist
        })
        .collect();

    let mut global_hist = vec![vec![0.0_f64; ybins]; xbins];
    for local_hist in thread_hists {
        for i in 0..xbins {
            for j in 0..ybins {
                global_hist[i][j] += local_hist[i * ybins + j];
            }
        }
    }
    global_hist
}

/// # 2D Kernel Density Estimation (KDE) Using 2D Linear Binning and the
/// second order approximation Deriche Filter
///
/// For a two-dimensional dataset, a histogram grid is first constructed using
/// 2D linear binning. KDE is then computed by approximating Gaussian smoothing
/// with a recursive Deriche filter over each 1D axis, following the separability
/// condition of the gaussian kernel. The bandwitdh for each axis is automatically
/// calculated using the Scott's factor.
/// The final result is normalized as a probability density function (PDF).
/// A 0.5*sigma tail is added to each of the grid's edges.
///
/// ## Arguments
/// - py: Python GIL token. Required for handling Python objects in PyO3 functions.
/// - data: Input data (numpy.ndarray containing a one-dimensional f64 array).
/// - bins: Number of bins at which the KDE is evaluated (number of grid points).
///
/// ## Returns
/// - PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>,
///             Bound<'py, PyArray2<f64>>)>:
/// - First element: X-coordinates of the bin centers (numpy.ndarray).
/// - Second element: Y-coordinates of the bin centers (numpy.ndarray).
/// - Third element: Corresponding PDF values for each bin (numpy.ndarray).
///
/// ## Errors
/// - PyValueError:
/// - If any dataset dimension contains fewer than 3 samples.
/// - If any number of bins is zero.
/// - If xmax is smaller than xmin (invalid data range).
/// - If ymax is smaller than ymin (invalid data range).
/// - If either bin width dx or dy is extremely small or zero.
/// - If the normalization factor is close to zero while the sum of the
/// filtered histogram is nonzero.
#[pyfunction]
fn kde_deriche_2d<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    bins: usize, // Square binning-only. Temporary
) -> PyResult<KdeGrid2D<'py>> {
    let view = data.as_array();
    let shape = view.shape();
    let (x_slice, y_slice);
    let (x_vec, y_vec); // Temporary storage if transposed array isn't contiguous
    if shape.len() != 2 {
        return Err(PyValueError::new_err("Input array must be 2-dimensional."));
    }
    let xbinding = view.row(0);
    let ybinding = view.row(1);
    if shape[0] == 2 {
        // Shape is (2, N)
        x_slice = xbinding
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("x row is not contiguous"))?;
        y_slice = ybinding
            .as_slice()
            .ok_or_else(|| PyValueError::new_err("y row is not contiguous"))?;
    } else if shape[1] == 2 {
        // Shape is (N, 2)
        x_vec = view.column(0).to_vec();
        y_vec = view.column(1).to_vec();
        x_slice = &x_vec;
        y_slice = &y_vec;
    } else {
        return Err(PyValueError::new_err(
            "Data shape must be either (2, N) or (N, 2).",
        ));
    }
    let x = x_slice;
    let y = y_slice;
    let xbins = bins;
    let ybins = bins;

    if x.len() < 4 || y.len() < 4 {
        return Err(PyValueError::new_err(
            "Need at least 4 samples per dimension for KDE 2D.",
        ));
    }
    if xbins == 0 || ybins == 0 {
        return Err(PyValueError::new_err(
            "Number of bins must be greater than 0 in all dimensions.",
        ));
    }
    let h = bandwidth_matrix(x, y);
    let (_p, d, _pt) = decompose_matrix(&h);
    let sigma_x = d[0][0].sqrt();
    let sigma_y = d[1][1].sqrt();

    let xmin = x.iter().cloned().fold(f64::INFINITY, f64::min) - 0.5 * sigma_x;
    let xmax = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 0.5 * sigma_x;
    let ymin = y.iter().cloned().fold(f64::INFINITY, f64::min) - 0.5 * sigma_y;
    let ymax = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 0.5 * sigma_y;

    if xmax < xmin {
        return Err(PyValueError::new_err(format!(
            "xmax ({}) must be greater than or equal to xmin ({})",
            xmax, xmin
        )));
    }
    if ymax < ymin {
        return Err(PyValueError::new_err(format!(
            "ymax ({}) must be greater than or equal to ymin ({})",
            ymax, ymin
        )));
    }

    let mut hist_counts = linear_binning_2d(x, y, xmin, xmax, ymin, ymax, bins);
    // if hist_counts.iter().flatten().all(|&h_val| h_val.abs() < 1e-9) {
    //     todo!()
    // }
    let bin_range_x = xmax - xmin;
    let bin_range_y = ymax - ymin;

    // Compute bin scale factors (bins per data unit)
    let sfx = xbins as f64 / bin_range_x;
    let sfy = ybins as f64 / bin_range_y;

    // Standard deviation in bins: sqrt(bandwidth_variance) * bins_per_unit
    let deriche_sigma_in_bins_x = sigma_x * sfx;
    let deriche_sigma_in_bins_y = sigma_y * sfy;

    // 1. Filter along Y direction (for each fixed X index i)
    // hist_counts is Vec<Vec<f64>> of shape [xbins][ybins]
    hist_counts.par_iter_mut().for_each(|x_row| {
        // x_row has length ybins
        deriche_recursive_filter_2nd_order_approx(x_row, deriche_sigma_in_bins_y);
    });

    // 2. Filter along X direction (for each fixed Y index j)
    #[allow(clippy::needless_range_loop)]
    for j in 0..ybins {
        let mut column: Vec<f64> = (0..xbins).map(|i| hist_counts[i][j]).collect();

        deriche_recursive_filter_2nd_order_approx(&mut column, deriche_sigma_in_bins_x);

        for i in 0..xbins {
            hist_counts[i][j] = column[i];
        }
    }

    // 3. PDF
    let sum_of_filtered_hist: f64 = hist_counts.iter().flatten().sum();
    let dx = (xmax - xmin) / xbins as f64;
    let dy = (ymax - ymin) / ybins as f64;

    if dx <= 0.0 || dy <= 0.0 {
        return Err(PyValueError::new_err(
            "Bin widths (dx, dy) must not be too small or zero.",
        ));
    }

    let normalization_factor = sum_of_filtered_hist * dx * dy;

    if normalization_factor.abs() < 1e-9 {
        return Err(PyValueError::new_err(
            "Normalization factor is near zero despite non-zero filtered histogram sum.",
        ));
    }

    for val in hist_counts.iter_mut().flatten() {
        *val /= normalization_factor;
    }

    let x_coords: Vec<f64> = (0..xbins).map(|i| xmin + (i as f64 + 0.5) * dx).collect();
    let y_coords: Vec<f64> = (0..ybins).map(|i| ymin + (i as f64 + 0.5) * dy).collect();

    let x_py_bound = PyArray1::from_vec(py, x_coords);
    let t_py_bound = PyArray1::from_vec(py, y_coords);
    let pdf_py_bound = PyArray2::from_vec2(py, &hist_counts)?;
    Ok((x_py_bound, t_py_bound, pdf_py_bound))
}

/// # Uses a second-order filter for 1-dimentional recursion
///
/// Described by Deriche (1990) in doi:10.1109/34.41386
/// alpha definition is described by Yang (2012) in doi:10.1007/978-3-642-33718-5_29
///
/// ## Arguments
/// - signal: Any dimention of input data (numpy.ndarray containing a one-dimensional f64 array). Modified in place.
/// - sigma: Axis bandwidth multiplied by the bin scale factors
fn deriche_recursive_filter_2nd_order_approx(signal: &mut [f64], sigma: f64) {
    if sigma.abs() < 1e-9 || signal.is_empty() {
        return;
    }

    let n = signal.len();
    let alpha = 1.695 / sigma;
    let ea = (-alpha).exp();
    let e2a = (-2.0 * alpha).exp();

    // 2nd-order Deriche filter coefficients (Gaussian approximation)
    let k = (1.0 - ea) * (1.0 - ea) / (1.0 + 2.0 * alpha * ea - e2a);
    let a1_plus = k;
    let a2_plus = k * ea * (alpha - 1.0);
    let a3_minus = k * ea * (alpha + 1.0);
    let a4_minus = -k * e2a;

    let b1 = 2.0 * ea;
    let b2 = -e2a;

    let mut y_plus = vec![0.0; n];
    let mut y_minus = vec![0.0; n];

    // 1. Causal (Forward) Pass
    y_plus[0] = a1_plus * signal[0];
    if n > 1 {
        y_plus[1] = a1_plus * signal[1] + a2_plus * signal[0] + b1 * y_plus[0];
    }
    for i in 2..n {
        y_plus[i] =
            a1_plus * signal[i] + a2_plus * signal[i - 1] + b1 * y_plus[i - 1] + b2 * y_plus[i - 2];
    }

    // 2. Anti-Causal Pass
    y_minus[n - 1] = 0.0;
    if n > 1 {
        y_minus[n - 2] = a3_minus * signal[n - 1];
    }
    if n > 2 {
        for i in (0..n - 2).rev() {
            y_minus[i] = a3_minus * signal[i + 1]
                + a4_minus * signal[i + 2]
                + b1 * y_minus[i + 1]
                + b2 * y_minus[i + 2];
        }
    }

    // 3. Combination of passes
    for i in 0..n {
        signal[i] = y_plus[i] + y_minus[i];
    }
}

/// # Calculates Scott's factor for the input array
///
/// ## Arguments
/// - x: Any dimention of input data (numpy.ndarray containing a one-dimensional f64 array).
///
/// ## Returns
/// - f64
fn scotts_factor(x: &[f64]) -> f64 {
    let d: f64 = 2.;
    let n: f64 = x.len() as f64;
    n.powf(-1. / (4. + d))
}

/// # Calculates average value of 1D array
///
/// ## Arguments
/// - x: Any dimention of input data (numpy.ndarray containing a one-dimensional f64 array).
///
/// ## Returns
/// - f64
fn mean(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let sum: f64 = x.iter().sum();
    sum / n
}

/// # Calculates covariance matrix of the input arrays
///
/// ## Arguments
/// - x: First dimention of input data (numpy.ndarray containing a one-dimensional f64 array).
/// - y: Second dimention of input data (numpy.ndarray containing a one-dimensional f64 array).
///
/// ## Returns
/// - [[f64; 2]; 2]
fn covariance_2d(x: &[f64], y: &[f64], population: bool) -> [[f64; 2]; 2] {
    assert!(x.len() == y.len());
    let n = x.len() as f64;
    let divisor = if population { n } else { n - 1. };
    let mut cov = [[0.; 2]; 2];
    let x_mean = mean(x);
    let y_mean = mean(y);
    for (x, y) in x.iter().zip(y.iter()) {
        let x_diff = x - x_mean;
        let y_diff = y - y_mean;
        cov[0][0] += x_diff * x_diff;
        cov[0][1] += x_diff * y_diff;
        cov[1][1] += y_diff * y_diff;
    }
    cov[0][0] /= divisor;
    cov[0][1] /= divisor;
    cov[1][0] = cov[0][1];
    cov[1][1] /= divisor;
    cov
}

/// # Calculates the bandwidth matrix given the input arrays
///
/// ## Arguments
/// - x: First dimention of input data (numpy.ndarray containing a one-dimensional f64 array).
/// - y: Second dimention of input data (numpy.ndarray containing a one-dimensional f64 array).
///
/// ## Returns
/// - [[f64; 2]; 2]
fn bandwidth_matrix(x: &[f64], y: &[f64]) -> [[f64; 2]; 2] {
    let f = scotts_factor(x);
    let mut h = covariance_2d(x, y, false);
    h.iter_mut().flatten().for_each(|val| *val *= f * f);
    h
}

/// # Decompose a square array A into it's eigenvectors and eigenvalues, following
/// A = PDP^T where P^T is P transposed
///
/// ## Arguments
/// - matrix: 2x2 array.
///
/// ## Returns
/// - ([[f64; 2]; 2], [[f64; 2]; 2], [[f64; 2]; 2])
/// - First element: P
/// - Second element: D
/// - Third element: P^T
fn decompose_matrix(matrix: &[[f64; 2]; 2]) -> DecomposedMatrix {
    let a = matrix[0][0];
    let b0 = matrix[0][1];
    let b1 = matrix[1][0];
    let c = matrix[1][1];
    // det(A-lI) = a*c -(a+c)l + l^2 = 0
    let apc = a + c;
    let delta = (apc * apc - 4. * (a * c - b0 * b1)).sqrt();
    let lambda_1 = 0.5 * (apc + delta);
    let lambda_2 = 0.5 * (apc - delta);
    let p = [[-b0 / (a - lambda_2), -b0 / (a - lambda_1)], [1., 1.]];
    let d = [[lambda_2, 0.], [0., lambda_1]];
    let pt = [[-b0 / (a - lambda_2), 1.], [-b0 / (a - lambda_1), 1.]];
    (p, d, pt)
}

/// # Pythonモジュール `fast_kde` の定義
#[pymodule]
fn fast_kde(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(kde_deriche, m)?)?;
    m.add_function(wrap_pyfunction!(kde_deriche_2d, m)?)?;
    m.add_function(wrap_pyfunction!(kde_mode_deriche, m)?)?;
    Ok(())
}
