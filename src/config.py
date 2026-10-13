"""
Project paths and a few global constants.

The raw dataset files (gy_link_info.txt, gy_link_top.txt, gy_link_travel_time_part*.zip)
are NOT bundled with this submission because of their size (~2 GB). By default we look for
them in the parent directory of this package; override with the GY_DATA_DIR environment
variable if you keep them somewhere else.
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent            # .../<submission>/src
PACKAGE_ROOT = HERE.parent                         # .../<submission>
PROJECT_ROOT = PACKAGE_ROOT.parent                 # directory that holds the raw data

# --- Directories -----------------------------------------------------------------
DATA_DIR = Path(os.environ.get("GY_DATA_DIR", PROJECT_ROOT))
CACHE_DIR = Path(os.environ.get("GY_CACHE_DIR", PROJECT_ROOT / "data_cache"))
OUT_DIR = Path(os.environ.get("GY_OUT_DIR", PACKAGE_ROOT))   # figures / predictions / results.json
FIG_DIR = OUT_DIR / "figures"
PRED_DIR = OUT_DIR / "predictions"
for _d in (CACHE_DIR, FIG_DIR, PRED_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Raw files -------------------------------------------------------------------
LINK_INFO_FILE = DATA_DIR / "gy_link_info.txt"
LINK_TOP_FILE = DATA_DIR / "gy_link_top.txt"
TRAVEL_TIME_ZIPS = [
    DATA_DIR / "gy_link_travel_time_part1.zip",
    DATA_DIR / "gy_link_travel_time_part2.zip",
    DATA_DIR / "gy_link_travel_time_part3.zip",
]

# --- Cache files -----------------------------------------------------------------
MORNING_CACHE = CACHE_DIR / "morning_records.csv"     # slim per (link, date, slice) rows 06:00-09:00
LINK_DAILY_MEAN_CACHE = CACHE_DIR / "link_daily_mean.csv"  # per-link mean over the whole day / all history

# --- Problem constants -----------------------------------------------------------
SLICE_MINUTES = 2                  # length of each time slice
WINDOW_START_MIN = 6 * 60          # 06:00  -> first minute we keep in the cache
WINDOW_END_MIN = 9 * 60            # 09:00  -> exclusive upper bound we keep in the cache

FEAT_START_MIN = 6 * 60            # [06:00, 08:00) is the morning window used as input features
FEAT_END_MIN = 8 * 60
TARGET_START_MIN = 8 * 60          # [08:00, 09:00) is what we have to predict
TARGET_END_MIN = 9 * 60

N_TARGET_SLICES = (TARGET_END_MIN - TARGET_START_MIN) // SLICE_MINUTES   # 30 slices
PREDICT_YEAR, PREDICT_MONTH = 2017, 6   # June 2017 is the month we must predict

RANDOM_STATE = 42
