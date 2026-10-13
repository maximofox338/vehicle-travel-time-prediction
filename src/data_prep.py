"""
Data preparation.

The historical travel-time data is delivered as three large text files (~2 GB in total)
inside the gy_link_travel_time_part*.zip archives.  Every row is one 2-minute average for
one link on one day:

    link_ID;date;time_interval;travel_time
    9377906285566510514;2016-05-21;[2016-05-21 23:20:00,2016-05-21 23:22:00);17.6

We only ever need the morning hours, so instead of loading 2 GB into memory we stream each
file in chunks, keep just the rows whose slice starts in the window [06:00, 09:00), and write
a small cache (~a few hundred MB).  In the same pass we also accumulate a per-link average
over the whole day (cheap, and useful as a feature).  Once the cache exists the heavy scan is
skipped automatically.
"""
from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd

from . import config as C

CHUNKSIZE = 3_000_000          # rows per chunk; ~10 chunks over the whole dataset
COL_NAMES = ["link_ID", "date", "time_interval", "travel_time"]


# --------------------------------------------------------------------------------------
# Link static attributes and topology
# --------------------------------------------------------------------------------------
def load_link_info() -> pd.DataFrame:
    """Return the 132 links with their static attributes and a stable integer index."""
    df = pd.read_csv(C.LINK_INFO_FILE, sep=";", dtype={"link_ID": str})
    df = df.sort_values("link_ID").reset_index(drop=True)
    df["link_idx"] = np.arange(len(df), dtype=np.int16)
    return df


def load_topology() -> pd.DataFrame:
    """Return the upstream/downstream relations; in/out links are parsed into lists."""
    df = pd.read_csv(C.LINK_TOP_FILE, sep=";", dtype=str).fillna("")

    def _split(cell: str) -> list[str]:
        return [x for x in cell.split("#") if x] if cell else []

    df["in_links"] = df["in_links"].map(_split)
    df["out_links"] = df["out_links"].map(_split)
    return df


# --------------------------------------------------------------------------------------
# Heavy scan of the travel-time archives
# --------------------------------------------------------------------------------------
def _minute_of_day(time_interval: pd.Series) -> pd.Series:
    """Extract the start minute-of-day from strings like '[2016-05-21 23:20:00,...'."""
    hh = pd.to_numeric(time_interval.str.slice(12, 14), errors="coerce")
    mm = pd.to_numeric(time_interval.str.slice(15, 17), errors="coerce")
    return hh * 60 + mm


def build_cache(force: bool = False, verbose: bool = True) -> None:
    """Stream the three archives once and write the morning cache + per-link daily mean."""
    if C.MORNING_CACHE.exists() and C.LINK_DAILY_MEAN_CACHE.exists() and not force:
        if verbose:
            print(f"Cache already present at {C.MORNING_CACHE} -- skipping scan.")
        return

    link_info = load_link_info()
    id_to_idx = dict(zip(link_info["link_ID"], link_info["link_idx"]))

    morning_parts: list[pd.DataFrame] = []
    daily_sum = np.zeros(len(link_info), dtype=np.float64)   # per-link sum over the whole day
    daily_cnt = np.zeros(len(link_info), dtype=np.float64)   # per-link count over the whole day
    total_rows = 0

    for zip_path in C.TRAVEL_TIME_ZIPS:
        if verbose:
            print(f"\nScanning {zip_path.name} ...")
        with zipfile.ZipFile(zip_path) as z:
            inner = z.namelist()[0]
            with z.open(inner) as fh:
                reader = pd.read_csv(
                    fh, sep=";", names=COL_NAMES, header=0,
                    dtype={"link_ID": str, "date": str, "time_interval": str},
                    chunksize=CHUNKSIZE,
                )
                for chunk in reader:
                    total_rows += len(chunk)
                    chunk["link_idx"] = chunk["link_ID"].map(id_to_idx)
                    chunk = chunk.dropna(subset=["link_idx"])
                    chunk["link_idx"] = chunk["link_idx"].astype(np.int16)
                    chunk["travel_time"] = pd.to_numeric(chunk["travel_time"], errors="coerce")

                    # whole-day per-link accumulation (uses every kept row)
                    valid = chunk.dropna(subset=["travel_time"])
                    grp = valid.groupby("link_idx")["travel_time"].agg(["sum", "count"])
                    daily_sum[grp.index.to_numpy()] += grp["sum"].to_numpy()
                    daily_cnt[grp.index.to_numpy()] += grp["count"].to_numpy()

                    # keep only the morning window [06:00, 09:00)
                    mod = _minute_of_day(chunk["time_interval"])
                    keep = (mod >= C.WINDOW_START_MIN) & (mod < C.WINDOW_END_MIN)
                    sub = chunk.loc[keep, ["link_idx", "date", "travel_time"]].copy()
                    sub["minute_of_day"] = mod[keep].astype(np.int16).to_numpy()
                    morning_parts.append(sub)

                    if verbose:
                        print(f"  processed {total_rows:,} rows "
                              f"(kept {sum(len(p) for p in morning_parts):,} morning rows)")

    morning = pd.concat(morning_parts, ignore_index=True)
    morning["travel_time"] = morning["travel_time"].astype(np.float32)
    morning.to_csv(C.MORNING_CACHE, index=False)

    with np.errstate(invalid="ignore", divide="ignore"):
        link_daily_mean = np.where(daily_cnt > 0, daily_sum / daily_cnt, np.nan)
    pd.DataFrame({
        "link_idx": np.arange(len(link_info), dtype=np.int16),
        "link_ID": link_info["link_ID"].to_numpy(),
        "link_daily_mean": link_daily_mean,
    }).to_csv(C.LINK_DAILY_MEAN_CACHE, index=False)

    if verbose:
        print(f"\nDone. Scanned {total_rows:,} rows, wrote {len(morning):,} morning rows "
              f"to {C.MORNING_CACHE.name}.")


def load_morning() -> pd.DataFrame:
    """Load the cached morning rows (building the cache first if necessary)."""
    if not C.MORNING_CACHE.exists():
        build_cache()
    df = pd.read_csv(
        C.MORNING_CACHE,
        dtype={"link_idx": np.int16, "date": str,
               "travel_time": np.float32, "minute_of_day": np.int16},
    )
    return df


def load_link_daily_mean() -> pd.DataFrame:
    if not C.LINK_DAILY_MEAN_CACHE.exists():
        build_cache()
    return pd.read_csv(C.LINK_DAILY_MEAN_CACHE, dtype={"link_idx": np.int16, "link_ID": str})


if __name__ == "__main__":
    build_cache()
