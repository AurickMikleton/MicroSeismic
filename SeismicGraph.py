import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_stead_samples_vs_trace(
    hdf5_path,
    csv_path,
    component=2,
    max_traces=200,
    category=None,
    normalize_each_trace=True,
    cmap="seismic"
):
    df = pd.read_csv(csv_path)

    if category is not None:
        df = df[df["trace_category"] == category]

    trace_names = df["trace_name"].dropna().tolist()[:max_traces]

    traces = []

    with h5py.File(hdf5_path, "r") as f:
        for trace_name in trace_names:
            ds_path = f"data/{trace_name}"
            data = np.array(f[ds_path])
            tr = data[:, component].astype(np.float32)

            if normalize_each_trace:
                max_abs = np.max(np.abs(tr))
                if max_abs > 0:
                    tr = tr / max_abs

            traces.append(tr)

    min_len = min(len(tr) for tr in traces)
    traces = np.stack([tr[:min_len] for tr in traces], axis=0)
    traces = np.rot90(traces)

    plt.figure(figsize=(12, 8))
    plt.imshow(
        traces,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap=cmap
    )
    plt.colorbar(label="Amplitude")
    plt.xlabel("Trace Number")
    plt.ylabel("Sample Number")
    plt.title(f"Samples vs Trace Number (component={component}, traces={traces.shape[0]})")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_stead_samples_vs_trace(
        hdf5_path="data/chunk2.hdf5",
        csv_path="data/chunk2.csv",
        component=2,
        max_traces=300,
        category="earthquake_local", # or "noise" only applicable for merge dataset
        normalize_each_trace=True
    )
