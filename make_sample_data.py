"""
Write a tiny synthetic dataset with the exact layout of the real one, so the pipeline can be
run end to end without the ~2 GB originals (`python make_results.py --demo`).

The numbers are fake: a rush-hour bump around 08:30, a link-specific base time driven by
length / width, a month factor, a 2017 uplift and noise.  Same quirks as the real dump:
June 2016 and July 2017 only contain [06:00, 08:00), so they carry no labels.
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np

N_LINKS, DAYS_PER_MONTH, SEED = 24, 4, 0
MONTHS = [(2016, 3), (2016, 4), (2016, 5), (2016, 6), (2017, 3), (2017, 4), (2017, 5),
          (2017, 6), (2017, 7)]
MORNING_ONLY = {(2016, 6), (2017, 7)}


def write_sample(out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    ids = [str(4377906280000000514 + 10_000_000 * i) for i in range(N_LINKS)]
    length = rng.integers(20, 300, N_LINKS)
    width = rng.choice([3, 6, 9, 12, 15], N_LINKS)
    with open(out_dir / "gy_link_info.txt", "w") as fh:
        fh.write("link_ID;length;width;link_class\n")
        fh.writelines(f"{i};{l};{w};1\n" for i, l, w in zip(ids, length, width))

    with open(out_dir / "gy_link_top.txt", "w") as fh:   # a simple chain: i-1 -> i -> i+1
        fh.write("link_ID;in_links;out_links\n")
        for k, i in enumerate(ids):
            fh.write(f"{i};{ids[k - 1] if k else ''};{ids[k + 1] if k + 1 < N_LINKS else ''}\n")

    base = length / (width + 3) * rng.uniform(0.5, 1.5, N_LINKS)
    lines: dict[int, list[str]] = {0: [], 1: [], 2: []}
    for part, (y, m) in enumerate(MONTHS):
        factor = rng.uniform(0.9, 1.1) * (1.15 if y == 2017 else 1.0)
        end = 8 * 60 if (y, m) in MORNING_ONLY else 9 * 60
        for d in range(1, DAYS_PER_MONTH + 1):
            date = f"{y}-{m:02d}-{d:02d}"
            for mod in range(6 * 60, end, 2):
                rush = 1 + 1.2 * np.exp(-((mod - 510) / 45) ** 2)
                tt = base * factor * rush * rng.lognormal(0, 0.15, N_LINKS)
                t0, t1 = f"{mod // 60:02d}:{mod % 60:02d}:00", f"{(mod + 2) // 60:02d}:{(mod + 2) % 60:02d}:00"
                lines[part % 3] += [f"{i};{date};[{date} {t0},{date} {t1});{v:.1f}\n"
                                    for i, v in zip(ids, tt)]

    for k in range(3):
        with zipfile.ZipFile(out_dir / f"gy_link_travel_time_part{k + 1}.zip", "w",
                             zipfile.ZIP_DEFLATED) as z:
            z.writestr(f"gy_link_travel_time_part{k + 1}.txt",
                       "link_ID;date;time_interval;travel_time\n" + "".join(lines[k]))
    print(f"wrote synthetic dataset ({N_LINKS} links, {len(MONTHS)} months) to {out_dir}")


if __name__ == "__main__":
    write_sample(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "sample_data")
