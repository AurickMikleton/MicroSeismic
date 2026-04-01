import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def downscale_by_averaging(traces, factor):
    n_traces, n_samples = traces.shape
    trimmed_len = (n_samples // factor) * factor
    traces = traces[:, :trimmed_len]
    return traces.reshape(n_traces, trimmed_len // factor, factor).mean(axis=2)

def load_component_matrix(hdf5_path, csv_path, component, max_traces=200, category=None):
    df = pd.read_csv(csv_path)

    if category is not None:
        df = df[df["trace_category"] == category]

    trace_names = df["trace_name"].dropna().tolist()[:max_traces]
    traces = []

    with h5py.File(hdf5_path, "r") as f:
        for trace_name in trace_names:
            key = f"data/{trace_name}"

            data = np.array(f[key], dtype=np.float32)
            if data.ndim != 2 or data.shape[1] < 3:
                continue

            tr = data[:, component]
            m = np.max(np.abs(tr))
            if m > 0:
                tr = tr / m

            traces.append(tr)

    if not traces:
        raise ValueError("No valid traces loaded.")

    min_len = min(len(tr) for tr in traces)
    traces = np.stack([tr[:min_len] for tr in traces], axis=0)
    return traces

def plot_heatmap(e, n, z):
    plt.figure(figsize=(12, 8))

    plt.imshow(e, aspect="auto", origin="lower", cmap="Reds", alpha=0.35)
    plt.imshow(n, aspect="auto", origin="lower", cmap="Greens", alpha=0.35)
    plt.imshow(z, aspect="auto", origin="lower", cmap="Blues", alpha=0.35)

    plt.xlabel("Sample Number")
    plt.ylabel("Trace Number")
    plt.gca().invert_yaxis()
    plt.title(f"Samples vs Trace Number")
    plt.tight_layout()
    plt.show()

def plot_overlay_components(
    hdf5_path,
    csv_path,
    max_traces=200,
    category=None,
    downscale_factor=1,
):
    # e = East - West ground motion -> red
    # n = North - South ground motion -> green
    # z = Up - Down ground motion -> blue
    e = load_component_matrix(hdf5_path, csv_path, component=0, max_traces=max_traces, category=category)
    n = load_component_matrix(hdf5_path, csv_path, component=1, max_traces=max_traces, category=category)
    z = load_component_matrix(hdf5_path, csv_path, component=2, max_traces=max_traces, category=category)

    if downscale_factor > 1:
        e = downscale_by_averaging(e, downscale_factor)
        n = downscale_by_averaging(n, downscale_factor)
        z = downscale_by_averaging(z, downscale_factor)

    e = np.rot90(e)
    n = np.rot90(n)
    z = np.rot90(z)

    plot_heatmap(e, n, z)

if __name__ == "__main__":
    plot_overlay_components(
        hdf5_path="../data/chunk2.hdf5",
        csv_path="../data/chunk2.csv",
        max_traces=300,
        category="earthquake_local", # or "noise" only applicable for merge dataset
        downscale_factor=10
    )
