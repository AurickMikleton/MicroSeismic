from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import segyio


kernal = np.array(
    [[1.0, 2.0, 1.0],
     [0.0, 0.0, 0.0],
     [-1.0, -2.0, -1.0]],
    dtype=np.float32,
)


def load_segy(path: str | Path, ignore_geometry: bool = True) -> tuple[np.ndarray, np.ndarray, float]:
    with segyio.open(path, "r", ignore_geometry=ignore_geometry) as f:
        f.mmap()
        data = segyio.tools.collect(f.trace[:]).astype(np.float32, copy=False)

        # returns microseconds
        dt_us = float(segyio.tools.dt(f))
        dt_s = dt_us * 1e-6

        n_samples = data.shape[1]
        time_axis_s = np.arange(n_samples, dtype=np.float64) * dt_s

    return data, time_axis_s, dt_s


def apply_time_window(data: np.ndarray, time_axis_s: np.ndarray, tmin: float | None, tmax: float | None):
    mask = np.ones_like(time_axis_s, dtype=bool)
    if tmin is not None:
        mask &= time_axis_s >= tmin
    if tmax is not None:
        mask &= time_axis_s <= tmax
    return data[:, mask], time_axis_s[mask]


def normalize(data: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    scale = np.max(np.abs(data), axis=1, keepdims=True)
    scale = np.maximum(scale, eps)
    return data / scale


def downscale_by_averaging(traces: np.ndarray, factor) -> np.ndarray:
    n_traces, n_samples = traces.shape
    trimmed_len = (n_samples // factor) * factor
    traces = traces[:, :trimmed_len]
    return traces.reshape(n_traces, trimmed_len // factor, factor).mean(axis=2)


def sobel_vertical(data: np.ndarray) -> np.ndarray:
    image = data.T.astype(np.float32, copy=False)
    padded = np.pad(image, ((1, 1), (1, 1)), mode="edge")

    out = np.zeros_like(image, dtype=np.float32)
    for r in range(3):
        for c in range(3):
            out += kernal[r, c] * padded[r:r + image.shape[0], c:c + image.shape[1]]

    return out.T


def print_info(path: str | Path) -> None:
    with segyio.open(path, "r", ignore_geometry=True) as f:
        n_traces = f.tracecount
        n_samples = len(f.samples)
        dt_us = float(segyio.tools.dt(f))
        fmt = f.bin[segyio.BinField.Format]
        print(f"File: {path}")
        print(f"Trace count: {n_traces}")
        print(f"Samples/trace: {n_samples}")
        print(f"Sample interval: {dt_us:.3f} microseconds ({dt_us * 1e-6:.6f} s)")
        print(f"Binary format code: {fmt}")


def plot_gather(
        data: np.ndarray,
        time_axis_s: np.ndarray,
        clip_percentile: float,
        cmap: str, 
        title: str = "SEG-Y Gather"
) -> None:
    vmax = np.percentile(np.abs(data), clip_percentile)
    if vmax == 0:
        vmax = 1.0

    extent = [0, data.shape[0] - 1, time_axis_s[-1], time_axis_s[0]]

    plt.figure(figsize=(12, 8))
    plt.imshow(
        data.T,
        aspect="auto",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        extent=extent,
    )
    plt.xlabel("Trace number")
    plt.ylabel("Time (s)")
    plt.title(title)
    plt.tight_layout()
    plt.show()



def main() -> None:
    file = "../data/1.sgy"

    print_info(file)

    data, time_axis_s, dt_s = load_segy(file, ignore_geometry=True)

    #data = normalize(data)
    #data = downscale_by_averaging(data, 10)

    #data, time_axis_s = apply_time_window(data, time_axis_s, None, None)

    if data.shape[1] == 0:
        raise ValueError("Selected time window is empty.")

    #data = sobel_vertical(data)
    #data = np.abs(data)

    plot_gather(data, time_axis_s, 0.99, "seismic", title="Segy Digest")

if __name__ == "__main__":
    main()
