from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import segyio

import pandas as pd
import re

file_number = 4
filename = f"../data/{file_number}.sgy"

kernal = np.array(
    [[1.0, 2.0, 1.0],
     [0.0, 0.0, 0.0],
     [-1.0, -2.0, -1.0]],
    dtype=np.float32,
)


#def load_segy(path: str | Path, ignore_geometry: bool = True) -> tuple[np.ndarray, np.ndarray, float]:
#    with segyio.open(path, "r", ignore_geometry=ignore_geometry) as f:
#        f.mmap()
#        data = segyio.tools.collect(f.trace[:]).astype(np.float32, copy=False)
#
#        # returns microseconds
#        dt_us = float(segyio.tools.dt(f))
#        dt_s = dt_us * 1e-6
#
#        n_samples = data.shape[1]
#        time_axis_s = np.arange(n_samples, dtype=np.float64) * dt_s
#
#    return data, time_axis_s, dt_s


def load_segy(path: str | Path, ignore_geometry: bool = True) -> tuple[np.ndarray, np.ndarray, int, int, float, float]:
    with segyio.open(path, "r", ignore_geometry=ignore_geometry) as f:
        f.mmap()
        n_samples = f.samples.size
        n_traces = f.tracecount
        samp_interval_us = float(f.bin[segyio.BinField.Interval])
        samp_interval_s = samp_interval_us / 1_000_000.0
        samp_freq = 1.0 / samp_interval_s
        trace_length_sec = n_samples / samp_freq

        dt_us = float(segyio.tools.dt(f))
        dt_s = dt_us * 1e-6
        time_axis_s = np.arange(n_samples, dtype=np.float64) * dt_s

        data = np.zeros((n_samples, n_traces), dtype=np.float32)
        for i, trace in enumerate(f.trace):
            data[:, i] = trace

    return data, time_axis_s, n_samples, n_traces, samp_freq, trace_length_sec


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


def generate_chunks(
    data,
    trace_chunk_size,
    sample_chunk_size,
    trace_step,
    sample_step
):
    data_chunks = chunk_matrix(data, trace_chunk_size, sample_chunk_size, trace_step, sample_step)

    merged = []
    for chunk in data_chunks:
        merged.append({
            "trace_start": chunk["trace_start"],
            "trace_end": chunk["trace_end"],
            "sample_start": chunk["sample_start"],
            "sample_end": chunk["sample_end"],
            "data": chunk["data"],
        })

    return merged


def save_chunks(
    data,
    trace_chunk_size,
    sample_chunk_size,
    trace_step=None,
    sample_step=None
):
    chunks = generate_chunks(
        data,
        trace_chunk_size,
        sample_chunk_size,
        trace_step,
        sample_step
    )

    for i, chunk in enumerate(chunks):
        output_dir = "../cv_data/"

        saved_filename = (
            f"{file_number}_chunk_{i:04d}_"
            f"t{chunk['trace_start']}-{chunk['trace_end']}_"
            f"s{chunk['sample_start']}-{chunk['sample_end']}.png"
        )

        h, w = chunk["data"].shape

        plt.figure(figsize=(w / 100, h / 100), dpi=100, frameon=False)
        plt.axis("off")

        plt.imshow(chunk["data"], aspect="auto", origin="lower",
                cmap="Reds", alpha=0.35, interpolation="nearest")
        plt.savefig(output_dir + saved_filename, dpi=100, pad_inches=0)
        plt.close()


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


def parse_segy_start_time(path: str | Path) -> pd.Timestamp | None:
    """
    Parse timestamps like ..._UTC190419001233.sgy from the SEG-Y filename.
    Interpreted as UTC with format %y%m%d%H%M%S.
    """
    name = Path(path).name
    m = re.search(r"UTC(\d{12})", name)
    if not m:
        return None
    return pd.to_datetime(m.group(1), format="%y%m%d%H%M%S", utc=True)


unit_values = {"s", "m", "m3", "n.m", "pa", "j", "dd:mm:yy:hh:mm:ss"}

def read_event_spreadsheet(path: str | Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    if df.empty: return df

    first_col = df.iloc[:, 0].astype(str).str.strip().str.lower()
    unit_row_mask = first_col.isin(unit_values)
    if unit_row_mask.any():
        df = df.loc[~unit_row_mask].copy()
    else:
        if "MS_EVENT_ID" in df.columns:
            df = df.loc[df["MS_EVENT_ID"].notna()].copy()

    if "JobTime" in df.columns:
        job = df["JobTime"].astype(str).str.strip()
        df["event_time"] = pd.to_datetime(job, format="%d:%m:%y:%H:%M:%S", errors="coerce", utc=True)

    numeric_cols = [c for c in df.columns if c != "JobTime"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    if "MS_EVENT_ID" in df.columns:
        df = df.sort_values(["event_time", "MS_EVENT_ID"], na_position="last")
    elif "event_time" in df.columns:
        df = df.sort_values("event_time", na_position="last")

    return df.reset_index(drop=True)


def find_events_in_segy_window(
    segy_path: str | Path,
    events_df: pd.DataFrame,
    trace_length_sec: float,
    tolerance_sec: float = 0.0,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None, pd.DataFrame]:
    start_time = parse_segy_start_time(segy_path)
    if start_time is None or "event_time" not in events_df.columns:
        return start_time, None, events_df.iloc[0:0].copy()

    end_time = start_time + pd.to_timedelta(trace_length_sec, unit="s")
    lower = start_time - pd.to_timedelta(tolerance_sec, unit="s")
    upper = end_time + pd.to_timedelta(tolerance_sec, unit="s")

    mask = events_df["event_time"].notna() & (events_df["event_time"] >= lower) & (events_df["event_time"] <= upper)
    matches = events_df.loc[mask].copy()
    if not matches.empty:
        matches["offset_sec"] = (matches["event_time"] - start_time).dt.total_seconds()
    return start_time, end_time, matches.reset_index(drop=True)


def sgy_to_png(file: str | Path) -> None:
    data, time_axis_s, dt_s = load_segy(file, ignore_geometry=True)

    data = normalize(data)
    data = downscale_by_averaging(data, 10)
    data = sobel_vertical(data)

    plot_gather(data, time_axis_s, 0.99, "seismic", title="Segy Digest")

    #save_chunks(
    #    data=data,
    #    trace_chunk_size=256,
    #    sample_chunk_size=256,
    #    trace_step=128,
    #    sample_step=128
    #)


def main() -> None:

    data, time_axis_s, n_samples, n_traces, samp_freq, trace_length_sec = load_segy(filename, ignore_geometry=True)

    data = normalize(data)
    data = downscale_by_averaging(data, 10)
    data = sobel_vertical(data)

    plot_gather(data, time_axis_s, 0.99, "seismic", title="Segy Digest")

    event_file = "../data/data.xlsx"
    print_info(filename)
    #sgy_to_png(filename)
    
    events_df = read_event_spreadsheet(event_file)
    start_time, end_time, event_matches = find_events_in_segy_window(
        event_file,
        events_df,
        trace_length_sec=trace_length_sec,
        tolerance_sec=0.0,
    )
    total_events = len(events_df)
    print(f"Spreadsheet events parsed: {total_events}")
    if start_time is None:
        print("Could not infer SEG-Y start time from filename; expected ..._UTCyymmddHHMMSS...")
    else:
        print(f"SEG-Y start: {start_time}")
        print(f"SEG-Y end:   {end_time}")
        print(f"Matching events in window: {len(event_matches)}")
        if len(event_matches):
            cols = [c for c in ["MS_EVENT_ID", "event_time", "offset_sec", "MS_EVENT_TYPE", "MS_LOC_SNR", "QC_LOC_T0", "SP_MAGNITUDE"] if c in event_matches.columns]
            print(event_matches[cols].to_string(index=False))


if __name__ == "__main__":
    main()
