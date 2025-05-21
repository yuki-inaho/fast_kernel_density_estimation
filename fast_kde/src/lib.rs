// fast_kde/src/lib.rs

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

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
    let chunk_size = (data.len() + num_threads - 1) / num_threads;

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
/// 論文「Fast & Accurate Gaussian Kernel Density Estimation」のSection 2および3.4で
/// 説明されているDericheの再帰フィルターを適用し、ガウススムージングを効率的に近似します。
/// このフィルターは、指数的に減衰する関数を結合することでガウス関数を近似します。
/// 4次のフィルター（K=4）を使用し、前方パスと後方パスを実行することで、
/// 両方向からのフィルタリング効果を組み合わせています。
///
/// ## 係数
/// - `ALPHA_COEFFS`: 論文のTable 3（Deriche, K=4）の`alpha`係数に由来。
/// - `LAMBDA_COEFFS`: 論文のTable 3（Deriche, K=4）の`lambda`係数に由来。
///   これらの係数は、再帰フィルターの安定性とガウス関数近似の精度を決定します。
///   Rustコードでは、`alpha`と`lambda`は複素数ではなく実数に分解されています。
///   論文の(37)式における`alpha`と`lambda`の実際の値は、
///   alpha = [0.84, -0.34015], lambda = [1.783, 1.723] となる。
///   これらは2組の複素共役根に対応し、各パスで適用される。
///   ここでは、それを実数係数に展開したものが直接使われている。
///
/// ## 引数
/// - `signal`: スムージングするデータ（ヒストグラムなど）を格納するミュータブルなスライス。
/// - `sigma`: ガウスカーネルの標準偏差（バンド幅）。フィルターの「幅」を決定します。
///
fn deriche_recursive_filter_approx(signal: &mut [f64], sigma: f64) {
    if sigma.abs() < 1e-9 {
        // sigmaが0に近い場合、スムージングは行わない
        return;
    }
    if signal.is_empty() {
        return;
    }

    // 論文のTable 3 (Deriche, K=4) の係数に対応
    // ALPHA_COEFFS: 実数部のalpha係数 (論文37式参照)
    // LAMBDA_COEFFS: 実数部のlambda係数 (論文37式参照)
    // これらの係数は、フィルターの極とゲインを決定します。
    // 実際には、4つのパスはそれぞれ異なる極を持つ。
    // (alpha_1, lambda_1), (alpha_2, lambda_2), (alpha_3, lambda_3), (alpha_4, lambda_4)
    // Rustコードでは、これらを線形に結合した結果の係数が使われているため、
    // ここで定義されているALPHA_COEFFSとLAMBDA_COEFFSは、
    // 論文の(37)式に記載されている複素数係数から導出された実数係数の対に対応している。
    // 具体的には、[0.84, -0.34015, 0.84, -0.34015] は alpha_1, alpha_3 の実部/虚部ではなく、
    // Dericheフィルターの各パスで使われる実数係数と解釈される。
    // これは、Rustコードの `deriche_recursive_filter_approx` 関数の内部実装が、
    // 論文の `dericheConv1d` (JS実装の `causal_coeff` など) とは異なる簡略化された形であるため。
    // この実装は、個々のポールに対するフィルターを順次適用する形式になっている。
    const ALPHA_COEFFS: [f64; 4] = [0.84, -0.34015, 0.84, -0.34015];
    const LAMBDA_COEFFS: [f64; 4] = [1.783, 1.723, 1.783, 1.723];
    let m = signal.len();

    // --- 前方パス (Forward passes) ---
    // 信号の左から右へフィルターを適用
    for k_pass in 0..4 {
        // 各パスのフィルター係数を計算
        // sigmaに対する指数減衰項 (-lambda / sigma).exp()
        let filter_pole = ALPHA_COEFFS[k_pass] * (-LAMBDA_COEFFS[k_pass] / sigma).exp();
        let mut prev_output = 0.0; // 前の出力値を保持（再帰的な計算のため）
        for i in 0..m {
            // y[i] = x[i] + a * y[i-1] の形式の再帰フィルター
            // ここでは、入力は常に元の信号の現在の状態（前のパスからの出力）
            // このため、`signal[i]` を読み込み、計算結果を `prev_output` に格納し、
            // その結果で `signal[i]` を更新している
            prev_output = signal[i] + filter_pole * prev_output;
            signal[i] = prev_output;
        }
    }

    // --- 後方パス (Backward passes) ---
    // 信号の右から左へフィルターを適用。
    // これにより、ガウスフィルターの対称性が近似される。
    // 各後方パスの入力は、直前のフィルター処理の最終状態（前方パスの最終結果、または前の後方パスの結果）
    let mut current_input_for_bwd_pass = signal.to_vec(); // 後方パスの初期入力として現在のsignalの状態をコピー
                                                          // (注意: この毎回ToVec()は、大きなデータでは性能ボトルネックになる可能性がある)
    for k_pass in 0..4 {
        let filter_pole = ALPHA_COEFFS[k_pass] * (-LAMBDA_COEFFS[k_pass] / sigma).exp();
        let mut prev_output = 0.0; // 前の出力値を保持（再帰的な計算のため）
        let input_for_this_specific_pass = current_input_for_bwd_pass.clone(); // このパスの読み込み用に入力をクローン

        // 後方パスは配列を逆順に走査
        for i in (0..m).rev() {
            // y[i] = x[i] + a * y[i+1] の形式の再帰フィルター（後方）
            prev_output = input_for_this_specific_pass[i] + filter_pole * prev_output;
            // フィルタリングされた結果を`signal`バッファに加算して蓄積
            // これにより、前方パスと以前の後方パスの結果が結合される
            signal[i] += prev_output;
        }
        // 次の後方パスのために、現在の`signal`の状態を入力としてコピー
        if k_pass < 3 {
            // 最後のパスの後にはコピーは不要
            current_input_for_bwd_pass = signal.to_vec();
        }
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
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
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

    // データ範囲（最小値と最大値）を計算
    let xmin = data_slice.iter().cloned().fold(f64::INFINITY, f64::min);
    let xmax = data_slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // データ範囲が極めて小さい場合（全てのデータ点が実質的に同じ場所にある場合）の特殊処理
    // この場合、KDEはデルタ関数のような挙動を示すべき
    if (xmax - xmin).abs() < 1e-9 {
        let x_coords = (0..bins)
            .map(|i| xmin + (i as f64 + 0.5) * (xmax - xmin + 1e-9) / bins as f64) // わずかな幅を持たせる
            .collect::<Vec<f64>>();
        let mut pdf_vals = vec![0.0; bins];
        if bins > 0 {
            // 中央のビンに1.0を割り当て、正規化（非常に狭いガウス関数を近似）
            // 論文のimpulsesテストケースとは異なる、よりロバストなエッジケース処理
            if !data_slice.is_empty() {
                pdf_vals[bins / 2] = 1.0 / (1e-9_f64); // 非常に狭い範囲での正規化
            }
        }
        let x_py = PyArray1::from_vec_bound(py, x_coords);
        let pdf_py = PyArray1::from_vec_bound(py, pdf_vals);
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
        let x_py = PyArray1::from_vec_bound(py, x_coords);
        let pdf_py = PyArray1::from_vec_bound(py, pdf_values);
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
    let x_py_bound = PyArray1::from_vec_bound(py, x_coords);
    let pdf_py_bound = PyArray1::from_vec_bound(py, hist_counts);

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
    let x_ro_array: PyReadonlyArray1<f64> = x_pyarray_bound.as_gil_ref().readonly();
    let pdf_ro_array: PyReadonlyArray1<f64> = pdf_pyarray_bound.as_gil_ref().readonly();

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

/// # Pythonモジュール `fast_kde` の定義
#[pymodule]
fn fast_kde(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(kde_deriche, m)?)?;
    m.add_function(wrap_pyfunction!(kde_mode_deriche, m)?)?;
    Ok(())
}
