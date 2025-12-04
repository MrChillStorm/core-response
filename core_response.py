#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

# Optional plotting
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def load_csv_folder(folder_path):
    files = [f for f in os.listdir(folder_path) if f.lower().endswith((".txt", ".csv"))]
    if not files:
        raise FileNotFoundError("No CSV or TXT files found in the folder.")

    all_freqs, all_mags = [], []
    for f in files:
        df = pd.read_csv(os.path.join(folder_path, f))
        if df.shape[1] < 2:
            raise ValueError(f"{f} must have at least two columns (freq, dB).")
        freqs, mags = df.iloc[:, 0].values, df.iloc[:, 1].values
        all_freqs.append(freqs)
        all_mags.append(mags)

    common_freqs = np.unique(np.concatenate(all_freqs))
    measurements = [
        interp1d(f, m, kind="linear", bounds_error=False, fill_value="extrapolate")(common_freqs)
        for f, m in zip(all_freqs, all_mags)
    ]
    return common_freqs, np.array(measurements)


def extract_core_response_svd(measurements, verbose=True, scale_factor=1.0):
    """
    Computes the PCA-based 'core response' using proper scaling:
      - Mean-center the data
      - Perform SVD
      - Use the first principal component scaled by its singular value
      - Add back the mean response for correct dB-level reconstruction
    
    Parameters:
      measurements: np.ndarray, shape (n_samples, n_freqs)
      verbose: bool, print explained variance summary
      scale_factor: float, optional multiplier to emphasize PC1 shape (default 1.0)
    
    Returns:
      core_response: 1D array, same shape as one measurement
      explained_variance: array of explained variance ratios
    """
    # Mean-center
    mean_response = np.mean(measurements, axis=0)
    X_centered = measurements - mean_response  # Shape: (n_samples, n_freqs)

    # SVD decomposition
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    # Explained variance ratios
    explained_variance = (S**2) / np.sum(S**2)

    if verbose:
        print("\nExplained variance by components:")
        for i, var in enumerate(explained_variance[:10]):
            print(f"  PC{i+1}: {var*100:.2f}%")
        print(f"  Total (all PCs): {np.sum(explained_variance)*100:.2f}%\n")

    # Proper reconstruction:
    # The first principal component shape (Vt[0, :]) scaled by its singular value
    # S[0]/sqrt(N-1) gives the correct standard deviation of PC1 projection scores
    n_samples = X_centered.shape[0]
    pc1 = Vt[0, :]
    amplitude = S[0] / np.sqrt(n_samples - 1)
    
    core_response = mean_response + pc1 * amplitude * scale_factor

    return core_response, explained_variance


def extract_medoid(measurements):
    n = len(measurements)
    dists = np.zeros(n)
    for i in range(n):
        dists[i] = np.sum(np.linalg.norm(measurements[i] - measurements, axis=1))
    return measurements[np.argmin(dists)]


def extract_average(measurements):
    return np.mean(measurements, axis=0)


def normalize_to_1khz(freqs, response):
    interp = interp1d(freqs, response, kind="linear", bounds_error=False, fill_value="extrapolate")
    return response - float(interp(1000.0))


def find_diff_peaks(freqs, diff, n_peaks=5):
    pos_idx, _ = find_peaks(diff)
    neg_idx, _ = find_peaks(-diff)

    pos_vals = diff[pos_idx]
    neg_vals = diff[neg_idx]

    top_pos_idx = pos_idx[np.argsort(pos_vals)[-n_peaks:]]
    top_neg_idx = neg_idx[np.argsort(-neg_vals)[-n_peaks:]]

    pos_list = [(freqs[i], diff[i]) for i in top_pos_idx]
    neg_list = [(freqs[i], diff[i]) for i in top_neg_idx]
    return pos_list, neg_list


def plot_results(freqs, measurements, core, medoid, avg, diff, explained_variance):
    if not PLOTLY_AVAILABLE:
        print("Plotly not installed; skipping plot.")
        return

    gold_color = "#e0b060"
    faint_gold = "rgba(224,176,96,0.3)"  # fainter grid color
    dominance_ratio = explained_variance[0] / (explained_variance[0] + explained_variance[1])
    stability_index = explained_variance[:3].sum() * 100

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=False,
        row_heights=[0.75, 0.25],
        subplot_titles=("Frequency Responses", "PCA Explained Variance")
    )

    # Measurement curves
    for m in measurements:
        fig.add_trace(go.Scatter(
            x=freqs, y=m, mode="lines",
            line=dict(color="gray", width=1, dash="dot"),
            opacity=0.3, showlegend=False
        ), row=1, col=1)

    # Main curves
    fig.add_trace(go.Scatter(x=freqs, y=avg, mode="lines",
                             line=dict(color="orange", width=2, dash="dash"),
                             name="Average Response"), row=1, col=1)
    fig.add_trace(go.Scatter(x=freqs, y=medoid, mode="lines",
                             line=dict(color="deepskyblue", width=2),
                             name="Medoid Response"), row=1, col=1)
    fig.add_trace(go.Scatter(x=freqs, y=core, mode="lines",
                             line=dict(color="lime", width=2, dash="dot"),
                             name="Core Response (PCA)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=freqs, y=diff, mode="lines",
                             line=dict(color="red", width=2),
                             name="Core - Average"), row=1, col=1)

    # Annotate peaks
    pos_peaks, neg_peaks = find_diff_peaks(freqs, diff, n_peaks=5)
    for f, v in pos_peaks + neg_peaks:
        fig.add_trace(go.Scatter(
            x=[f], y=[v],
            mode="markers+text",
            marker=dict(color="red", size=8, symbol="circle"),
            text=[f"{v:.1f} dB"],
            textposition="top center",
            showlegend=False,
            textfont=dict(color="white")
        ), row=1, col=1)

    pcs = np.arange(1, len(explained_variance) + 1)
    fig.add_trace(go.Bar(
        x=pcs, y=explained_variance*100,
        marker_color=gold_color,
        name="Explained Variance (%)"
    ), row=2, col=1)

    # Dominance box
    annotation_text = f"Dominance: {dominance_ratio:.2f}\nStability: {stability_index:.1f}%"
    fig.add_annotation(
        text=annotation_text,
        xref="paper", yref="paper",
        x=0.98, y=0.98,
        showarrow=False,
        xanchor="right",
        yanchor="top",
        bgcolor="rgba(0,0,0,0)",
        bordercolor="white",
        borderwidth=1,
        borderpad=4,
        font=dict(color="white", family="Courier New", size=12)
    )

    y_min, y_max = np.min(measurements), np.max(measurements)
    y_margin = 0.1 * (y_max - y_min)

    fig.update_layout(
        title="Core vs Medoid vs Average Frequency Response + PCA Variance",
        title_font=dict(color=gold_color),
        xaxis=dict(title="Frequency (Hz)", type="log",
                   gridcolor=faint_gold, zerolinecolor=faint_gold, linecolor=gold_color,
                   tickfont=dict(color=gold_color)),
        yaxis=dict(title="Magnitude (dB, normalized to 1 kHz)",
                   range=[y_min - y_margin, y_max + y_margin],
                   gridcolor=faint_gold, zerolinecolor=faint_gold, linecolor=gold_color,
                   tickfont=dict(color=gold_color)),
        xaxis2=dict(title="Principal Component",
                    gridcolor=faint_gold, linecolor=gold_color,
                    tickfont=dict(color=gold_color)),
        yaxis2=dict(title="Explained Variance (%)",
                    gridcolor=faint_gold, linecolor=gold_color,
                    tickfont=dict(color=gold_color)),
        template="plotly_dark",
        legend=dict(x=0.02, y=0.98, font=dict(color="white"))
    )

    fig.show()


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Extract core frequency response (PCA, Medoid, Average) with PCA variance plot."
    )
    parser.add_argument("folder", help="Folder containing CSV or TXT sweeps.")
    parser.add_argument("--no-plot", action="store_true", help="Skip interactive plot.")
    args = parser.parse_args()

    freqs, measurements = load_csv_folder(args.folder)

    core, ev = extract_core_response_svd(measurements)
    medoid = extract_medoid(measurements)
    avg = extract_average(measurements)
    diff = core - avg

    core = normalize_to_1khz(freqs, core)
    medoid = normalize_to_1khz(freqs, medoid)
    avg = normalize_to_1khz(freqs, avg)
    diff = normalize_to_1khz(freqs, diff)
    measurements = np.array([normalize_to_1khz(freqs, m) for m in measurements])

    # Save CSVs
    pd.DataFrame({"frequency": freqs, "magnitude_dB": core}).to_csv(
        os.path.join(args.folder, "core_response.csv"), index=False)
    pd.DataFrame({"frequency": freqs, "magnitude_dB": medoid}).to_csv(
        os.path.join(args.folder, "medoid_response.csv"), index=False)
    pd.DataFrame({"frequency": freqs, "magnitude_dB": avg}).to_csv(
        os.path.join(args.folder, "average_response.csv"), index=False)
    pd.DataFrame({"frequency": freqs, "magnitude_dB": diff}).to_csv(
        os.path.join(args.folder, "core_minus_average.csv"), index=False)

    print(f"✅ Saved core_response.csv, medoid_response.csv, average_response.csv, and core_minus_average.csv in {args.folder}")

    # --- RMSE calculations ---
    rmse_avg = np.sqrt(np.mean((core - avg)**2))
    rmse_medoid = np.sqrt(np.mean((core - medoid)**2))

    print(f"\nRMSE comparisons (Core = reference):")
    print(f"  • Average vs Core : {rmse_avg:.3f} dB")
    print(f"  • Medoid  vs Core : {rmse_medoid:.3f} dB\n")

    if not args.no_plot:
        plot_results(freqs, measurements, core, medoid, avg, diff, ev)


if __name__ == "__main__":
    main()
