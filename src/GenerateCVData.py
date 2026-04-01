import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def downscale_by_averaging(traces, factor):
    n_traces, n_samples = traces.shape
    trimmed_len = (n_samples // factor) * factor
    traces = traces[:, :trimmed_len]
    return traces.reshape(n_traces, trimmed_len // factor, factor).mean(axis=2)

def chunk_matrix(
    traces, 
    trace_chunk_size, 
    sample_chunk_size,
    trace_step,
    sample_step
):
    n_traces, n_samples = traces.shape

    if trace_step is None:
        trace_step = trace_chunk_size
    if sample_step is None:
        sample_step = sample_chunk_size

    chunks = []

    for trace_start in range(0, n_traces - trace_chunk_size + 1, trace_step):
        trace_end = trace_start + trace_chunk_size

        for sample_start in range(0, n_samples - sample_chunk_size + 1, sample_step):
            sample_end = sample_start + sample_chunk_size

            chunk = traces[trace_start:trace_end, sample_start:sample_end]

            chunks.append({
                "trace_start": trace_start,
                "trace_end": trace_end,
                "sample_start": sample_start,
                "sample_end": sample_end,
                "data": chunk
            })

    return chunks

def chunk_3_components(
    e,
    n,
    z,
    trace_chunk_size,
    sample_chunk_size,
    trace_step,
    sample_step
):
    e_chunks = chunk_matrix(e, trace_chunk_size, sample_chunk_size, trace_step, sample_step)
    n_chunks = chunk_matrix(n, trace_chunk_size, sample_chunk_size, trace_step, sample_step)
    z_chunks = chunk_matrix(z, trace_chunk_size, sample_chunk_size, trace_step, sample_step)

    merged = []
    for ec, nc, zc in zip(e_chunks, n_chunks, z_chunks):
        merged.append({
            "trace_start": ec["trace_start"],
            "trace_end": ec["trace_end"],
            "sample_start": ec["sample_start"],
            "sample_end": ec["sample_end"],
            "e": ec["data"],
            "n": nc["data"],
            "z": zc["data"],
        })

    return merged

def save_chunks(
    e,
    n,
    z,
    trace_chunk_size,
    sample_chunk_size,
    trace_step=None,
    sample_step=None
):
    out_dir = "../cv_data/"

    chunks = chunk_3_components(
        e,
        n,
        z,
        trace_chunk_size,
        sample_chunk_size,
        trace_step,
        sample_step
    )

    for i, chunk in enumerate(chunks):
        output_dir = "../cv_data/"

        filename = (
            f"chunk_{i:04d}_"
            f"t{chunk['trace_start']}-{chunk['trace_end']}_"
            f"s{chunk['sample_start']}-{chunk['sample_end']}.png"
        )

        h, w = chunk["e"].shape

        fig = plt.figure(figsize=(w / 100, h / 100), dpi=100, frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])

        ax.imshow(chunk["e"], aspect="auto", origin="lower",
                cmap="Reds", alpha=0.35, interpolation="nearest")
        ax.imshow(chunk["n"], aspect="auto", origin="lower",
                cmap="Greens", alpha=0.35, interpolation="nearest")
        ax.imshow(chunk["z"], aspect="auto", origin="lower",
                cmap="Blues", alpha=0.35, interpolation="nearest")

        ax.invert_yaxis()
        ax.set_axis_off()
        fig.savefig(output_dir + filename, dpi=100, pad_inches=0)
        plt.close(fig)

        #plt.figure(figsize=(6, 6), dpi=256)

        #plt.imshow(chunk["e"], aspect="auto", origin="lower", cmap="Reds", alpha=0.35, interpolation="nearest")
        #plt.imshow(chunk["n"], aspect="auto", origin="lower", cmap="Greens", alpha=0.35, interpolation="nearest")
        #plt.imshow(chunk["z"], aspect="auto", origin="lower", cmap="Blues", alpha=0.35, interpolation="nearest")

        #plt.gca().invert_yaxis()
        #plt.axis("off")
        #plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        #plt.savefig(output_dir + filename,
        #        bbox_inches="tight",
        #        pad_inches=0,
        #        transparent=True
        #        )
        #plt.close()

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
            traces.append(tr)

    min_len = min(len(tr) for tr in traces)
    traces = np.stack([tr[:min_len] for tr in traces], axis=0)
    return traces

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

    global_min = min(e.min(), n.min(), z.min())
    global_max = max(e.max(), n.max(), z.max())

    e = (e - global_min) / (global_max - global_min)
    n = (n - global_min) / (global_max - global_min)
    z = (z - global_min) / (global_max - global_min)

    if downscale_factor > 1:
        e = downscale_by_averaging(e, downscale_factor)
        n = downscale_by_averaging(n, downscale_factor)
        z = downscale_by_averaging(z, downscale_factor)

    e = np.rot90(e)
    n = np.rot90(n)
    z = np.rot90(z)

    save_chunks(
        e, n, z,
        trace_chunk_size=256,
        sample_chunk_size=256,
        trace_step=128,
        sample_step=128
    )

if __name__ == "__main__":
    plot_overlay_components(
        hdf5_path="../data/chunk2.hdf5",
        csv_path="../data/chunk2.csv",
        max_traces=6000,
        category="earthquake_local", # or "noise" only applicable for merge dataset
        downscale_factor=10
    )
