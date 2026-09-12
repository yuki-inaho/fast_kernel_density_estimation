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
/// Each sample is distributed bilinearly over the four grid nodes around it,
/// following the linear-binning rule of Heer (2021, Section 3): the weight of a
/// node is the product of the two 1D weights, and each 1D weight is
/// proportional to the distance between the sample and the neighbouring node
/// centres. Samples outside the node range are clamped to the edge nodes, so
/// the total weight is conserved.
///
/// The binning operation is automatically parallelized using the Rayon crate.
///
/// ## Arguments
/// - x: One-dimensional data slice to be binned.
/// - y: One-dimensional data slice to be binned.
/// - xmin: Lower edge of the x grid (node 0 is half a bin above it).
/// - xmax: Upper edge of the x grid.
/// - ymin: Lower edge of the y grid.
/// - ymax: Upper edge of the y grid.
/// - bins: Number of bins along each axis.
///
/// ## Returns
/// A Vec<Vec<f64>> of shape `[bins][bins]` containing the weight of each node.
fn linear_binning_2d(
    x: &[f64],
    y: &[f64],
    xmin: f64,
    xmax: f64,
    ymin: f64,
    ymax: f64,
    bins: usize,
) -> Vec<Vec<f64>> {
    assert_eq!(
        x.len(),
        y.len(),
        "x and y must contain the same number of points"
    );

    if bins == 0 {
        return vec![];
    }

    if x.is_empty() {
        return vec![vec![0.0; bins]; bins];
    }

    if bins == 1 {
        return vec![vec![x.len() as f64]];
    }

    let bin_width_x = (xmax - xmin) / bins as f64;
    let bin_width_y = (ymax - ymin) / bins as f64;

    if !bin_width_x.is_finite()
        || bin_width_x <= 0.0
        || !bin_width_y.is_finite()
        || bin_width_y <= 0.0
    {
        return vec![vec![0.0; bins]; bins];
    }

    let num_threads = rayon::current_num_threads();
    let chunk_size = x.len().div_ceil(num_threads);
    let thread_hists: Vec<Vec<f64>> = x
        .par_chunks(chunk_size)
        .zip(y.par_chunks(chunk_size))
        .map(|(x_chunk, y_chunk)| {
            let mut local_hist = vec![0.0_f64; bins * bins];
            for (&x_val, &y_val) in x_chunk.iter().zip(y_chunk.iter()) {
                let (ix, fx) = split_weight(bins, (x_val - xmin) / bin_width_x);
                let (iy, fy) = split_weight(bins, (y_val - ymin) / bin_width_y);
                local_hist[ix * bins + iy] += (1.0 - fx) * (1.0 - fy);
                local_hist[ix * bins + (iy + 1)] += (1.0 - fx) * fy;
                local_hist[(ix + 1) * bins + iy] += fx * (1.0 - fy);
                local_hist[(ix + 1) * bins + (iy + 1)] += fx * fy;
            }
            local_hist
        })
        .collect();

    let mut global_hist = vec![vec![0.0_f64; bins]; bins];
    for local_hist in thread_hists {
        for i in 0..bins {
            for j in 0..bins {
                global_hist[i][j] += local_hist[i * bins + j];
            }
        }
    }
    global_hist
}

/// Maps a position in bin units to the lower of the two neighbouring nodes and
/// the fraction that belongs to the upper node.
///
/// `pos` is `(value - xmin) / bin_width`, so the node centres sit at
/// `pos = k + 0.5`. Positions outside the node range are clamped, so the
/// returned indices always satisfy `index + 1 < bins` and the two weights sum
/// to one. Requires `bins >= 2`.
fn split_weight(bins: usize, pos: f64) -> (usize, f64) {
    debug_assert!(bins >= 2);
    let pos = pos - 0.5;
    let last = (bins - 1) as f64;
    if pos <= 0.0 {
        (0, 0.0)
    } else if pos >= last {
        (bins - 2, 1.0)
    } else {
        let index = pos.floor() as usize;
        (index, pos - index as f64)
    }
}

/// # 2D Kernel Density Estimation (KDE) Using 2D Linear Binning and the
/// Deriche Filter
///
/// For a two-dimensional dataset, a histogram grid is first constructed using
/// 2D linear binning. KDE is then computed by approximating Gaussian smoothing
/// with the same K=4 Deriche recursive filter used by [`kde_deriche`], applied
/// along every row and every column of the grid. This is valid because the
/// product Gaussian kernel is separable.
///
/// The bandwidth is chosen per axis with Scott's rule for a product kernel,
/// `sigma_j = n**(-1/(4 + d)) * std_j` with `d = 2`, i.e. the diagonal of the
/// Scott bandwidth matrix. Correlation between the axes is not modelled: the
/// density is smoothed along the axes only. A 0.5 * sigma padding is added on
/// each side of the data range.
///
/// The final result is normalised as a probability density function (PDF) on
/// the grid, so `pdf[i][j]` is the density at `(x_coords[i], y_coords[j])`.
///
/// ## Arguments
/// - py: Python GIL token. Required for handling Python objects in PyO3 functions.
/// - data: Input data as a 2D array of shape `(2, N)` or `(N, 2)`.
/// - bins: Number of bins along each axis (the grid is square).
///
/// ## Returns
/// - PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>,
///             Bound<'py, PyArray2<f64>>)>:
/// - First element: X-coordinates of the bin centres (numpy.ndarray).
/// - Second element: Y-coordinates of the bin centres (numpy.ndarray).
/// - Third element: PDF of shape `(bins, bins)` indexed as `[x, y]`.
///
/// ## Errors
/// - PyValueError:
/// - If the array is not shaped `(2, N)` or `(N, 2)`.
/// - If there are fewer than 4 samples.
/// - If the number of bins is zero.
/// - If any input value is non-finite.
/// - If the normalization factor is close to zero while the filtered histogram
///   has non-zero values.
#[pyfunction]
fn kde_deriche_2d<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, f64>,
    bins: usize,
) -> PyResult<KdeGrid2D<'py>> {
    let view = data.as_array();
    let shape = view.shape();

    let (x, y): (Vec<f64>, Vec<f64>) = if shape[0] == 2 {
        (view.row(0).to_vec(), view.row(1).to_vec())
    } else if shape[1] == 2 {
        (view.column(0).to_vec(), view.column(1).to_vec())
    } else {
        return Err(PyValueError::new_err(
            "Data shape must be either (2, N) or (N, 2).",
        ));
    };

    if x.len() < 4 {
        return Err(PyValueError::new_err("Need at least 4 samples for KDE."));
    }
    if bins == 0 {
        return Err(PyValueError::new_err(
            "Number of bins must be greater than 0.",
        ));
    }
    let non_finite = x.iter().chain(y.iter()).filter(|v| !v.is_finite()).count();
    if non_finite > 0 {
        return Err(PyValueError::new_err(format!(
            "Input data must be finite; found {non_finite} non-finite value(s)."
        )));
    }

    let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_min = y.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Degenerate input: every sample sits at (essentially) the same place.
    // Mirror the 1D behaviour with a normalised point mass at the centre.
    if (x_max - x_min).abs() < 1e-9 && (y_max - y_min).abs() < 1e-9 {
        let dx = 1e-9 / bins as f64;
        let dy = 1e-9 / bins as f64;
        let x_coords: Vec<f64> = (0..bins).map(|i| x_min + (i as f64 + 0.5) * dx).collect();
        let y_coords: Vec<f64> = (0..bins).map(|i| y_min + (i as f64 + 0.5) * dy).collect();
        let mut pdf = vec![vec![0.0_f64; bins]; bins];
        pdf[bins / 2][bins / 2] = 1.0 / (dx * dy);
        return Ok((
            PyArray1::from_vec(py, x_coords),
            PyArray1::from_vec(py, y_coords),
            PyArray2::from_vec2(py, &pdf)?,
        ));
    }

    let (sigma_x, sigma_y) = scotts_sigma(&x, &y);

    // `max(1e-9)` keeps the grid width positive when one axis is constant.
    let xmin = x_min - 0.5 * sigma_x.max(1e-9);
    let xmax = x_max + 0.5 * sigma_x.max(1e-9);
    let ymin = y_min - 0.5 * sigma_y.max(1e-9);
    let ymax = y_max + 0.5 * sigma_y.max(1e-9);

    let mut hist_counts = linear_binning_2d(&x, &y, xmin, xmax, ymin, ymax, bins);

    let bin_range_x = xmax - xmin;
    let bin_range_y = ymax - ymin;

    // Convert each bandwidth from data units into bin units.
    let deriche_sigma_in_bins_x = sigma_x * bins as f64 / bin_range_x;
    let deriche_sigma_in_bins_y = sigma_y * bins as f64 / bin_range_y;

    // 1. Filter along Y direction (for each fixed X index i)
    hist_counts.par_iter_mut().for_each(|x_row| {
        deriche_recursive_filter_approx(x_row, deriche_sigma_in_bins_y);
    });

    // 2. Filter along X direction (for each fixed Y index j)
    #[allow(clippy::needless_range_loop)]
    for j in 0..bins {
        let mut column: Vec<f64> = (0..bins).map(|i| hist_counts[i][j]).collect();

        deriche_recursive_filter_approx(&mut column, deriche_sigma_in_bins_x);

        for i in 0..bins {
            hist_counts[i][j] = column[i];
        }
    }

    // 3. PDF
    let dx = bin_range_x / bins as f64;
    let dy = bin_range_y / bins as f64;
    let sum_of_filtered_hist: f64 = hist_counts.iter().flatten().sum();

    if sum_of_filtered_hist.abs() < 1e-9 {
        if hist_counts.iter().flatten().any(|v| v.abs() >= 1e-9) {
            return Err(PyValueError::new_err(
                "Normalization factor is near zero despite non-zero filtered histogram sum.",
            ));
        }
    } else {
        let normalization_factor = sum_of_filtered_hist * dx * dy;
        for val in hist_counts.iter_mut().flatten() {
            *val /= normalization_factor;
        }
    }

    let x_coords: Vec<f64> = (0..bins).map(|i| xmin + (i as f64 + 0.5) * dx).collect();
    let y_coords: Vec<f64> = (0..bins).map(|i| ymin + (i as f64 + 0.5) * dy).collect();

    let x_py_bound = PyArray1::from_vec(py, x_coords);
    let y_py_bound = PyArray1::from_vec(py, y_coords);
    let pdf_py_bound = PyArray2::from_vec2(py, &hist_counts)?;
    Ok((x_py_bound, y_py_bound, pdf_py_bound))
}

/// # Calculates Scott's factor for a 2D product kernel
///
/// `n**(-1 / (4 + d))` with `d = 2`, i.e. `n**(-1/6)`.
///
/// ## Arguments
/// - x: One of the two sample axes; only its length is used.
///
/// ## Returns
/// - f64
fn scotts_factor(x: &[f64]) -> f64 {
    let d: f64 = 2.;
    let n: f64 = x.len() as f64;
    n.powf(-1. / (4. + d))
}

/// # Sample standard deviation (Bessel-corrected, `ddof = 1`)
///
/// Uses the same convention as `numpy.std(x, ddof=1)`, so the bandwidth matches
/// `scipy.stats.gaussian_kde`'s Scott rule.
fn sample_std(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let variance = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0);
    variance.sqrt()
}

/// # Per-axis Scott bandwidth for a 2D product-kernel KDE
///
/// Returns `(f * std(x), f * std(y))` where `f = n**(-1/6)`. These are the
/// square roots of the diagonal of the Scott bandwidth matrix
/// `H = f**2 * cov(x, y)`; the off-diagonal term is intentionally dropped
/// because the Deriche filter smooths each axis independently.
fn scotts_sigma(x: &[f64], y: &[f64]) -> (f64, f64) {
    let f = scotts_factor(x);
    (f * sample_std(x), f * sample_std(y))
}

/// # Pythonモジュール `fast_kde` の定義
#[pymodule]
fn fast_kde(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(kde_deriche, m)?)?;
    m.add_function(wrap_pyfunction!(kde_deriche_2d, m)?)?;
    m.add_function(wrap_pyfunction!(kde_mode_deriche, m)?)?;
    Ok(())
}
