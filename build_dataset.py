#!/usr/bin/env python3
"""
build_dataset.py - Turn raw Sensor Logger exports into a clean, labelled dataset.

Handles the real folder layout straight from the phone, e.g.:

    <src>/standing-50hz/   *.zip      (each zip = one recording)
    <src>/walking-100hz/   *.zip
    <src>/Jumping-50HZ/    *.zip      (case / separator inconsistencies are fine)
    <src>/still-100hz/     *.zip
    <src>/mixed/           *.zip      (the 2 mixed test sessions)
    <src>/files, files (2), *.pdf     (anything not an activity/mixed folder is ignored)

For every recording it:
  - unzips it (to a temp area; your originals are untouched),
  - merges Accelerometer (linear accel), Gravity and Gyroscope on nearest timestamp,
  - auto-detects the true sampling rate from the timestamps,
  - writes one clean labelled CSV, and
  - prints a validation report: file count, per-activity seconds, rates, NaNs,
    and which sensors each file actually contains.

Usage:
    python build_dataset.py --src ~/Downloads --out .
      -> <out>/data/raw/<activity>_<fs>hz_NN.csv     (training clips)
         <out>/data/test/mixed_<fs>hz_NN.csv         (test sessions)
"""

import argparse
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ACTIVITIES = ("standing", "walking", "jumping", "still")
STD_RATES = [10, 25, 50, 100, 200]
VALUE_PREFIXES = ("acc", "gyr", "grav")
SENSOR_MAP = {"Accelerometer": "acc", "Gyroscope": "gyr", "Gravity": "grav"}
ACT_RE = re.compile(r"^(standing|walking|jumping|still)\b", re.IGNORECASE)


def value_cols(df):
    return [c for c in df.columns if c.split("_")[0] in VALUE_PREFIXES]


def detect_rate(t_ns):
    dt = np.median(np.diff(t_ns.to_numpy())) / 1e9
    actual = 1.0 / dt if dt > 0 else float("nan")
    return min(STD_RATES, key=lambda r: abs(r - actual)), actual


def load_sensor(folder, name):
    df = pd.read_csv(folder / f"{name}.csv")
    cols = {c.lower(): c for c in df.columns}
    for c in ("time", "x", "y", "z"):
        if c not in cols:
            raise ValueError(f"{name}.csv missing '{c}'; has {list(df.columns)}")
    out = df[[cols["time"], cols["x"], cols["y"], cols["z"]]].copy()
    out.columns = ["time", "x", "y", "z"]
    return out.sort_values("time").reset_index(drop=True)


def merge_recording(folder):
    """Merge accel + gyro (required) + gravity (if present). Return (df, actual_hz)."""
    base = load_sensor(folder, "Accelerometer").rename(
        columns={"x": "acc_x", "y": "acc_y", "z": "acc_z"})
    snapped, actual = detect_rate(base["time"])
    tol = int(0.5 * 1e9 / snapped)
    for name, pref in SENSOR_MAP.items():
        if name == "Accelerometer":
            continue
        if not (folder / f"{name}.csv").exists():
            if name == "Gyroscope":
                raise FileNotFoundError("Gyroscope.csv missing")
            continue  # Gravity optional
        s = load_sensor(folder, name).rename(
            columns={"x": f"{pref}_x", "y": f"{pref}_y", "z": f"{pref}_z"})
        base = pd.merge_asof(base, s, on="time", direction="nearest", tolerance=tol)
    base["seconds_elapsed"] = (base["time"] - base["time"].iloc[0]) / 1e9
    base["fs_hz"] = snapped
    return base[["time", "seconds_elapsed", *value_cols(base), "fs_hz"]], actual


def find_accel_dir(root):
    hits = list(root.rglob("Accelerometer.csv"))
    return hits[0].parent if hits else None


def classify(folder_name):
    if folder_name.strip().lower() == "mixed":
        return "test", "mixed"
    m = ACT_RE.match(folder_name.strip())
    return ("train", m.group(1).lower()) if m else (None, None)


def collect_recordings(folder, staging):
    """Return a list of dirs (one per recording) containing the sensor CSVs."""
    recs = []
    zips = sorted(folder.glob("*.zip"))
    for z in zips:
        dest = staging / f"{folder.name}__{z.stem}"
        try:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(dest)
        except zipfile.BadZipFile:
            print(f"  SKIP (bad zip) {z.name}")
            continue
        d = find_accel_dir(dest)
        recs.append(d) if d else print(f"  SKIP (no Accelerometer.csv) {z.name}")
    zip_stems = {z.stem for z in zips}
    for sub in sorted(p for p in folder.iterdir() if p.is_dir()):
        if sub.name in zip_stems:        # already-extracted copy of a zip -> skip
            continue
        d = find_accel_dir(sub)
        if d:
            recs.append(d)
    return recs


def process(src, out):
    raw_dir, test_dir = out / "data" / "raw", out / "data" / "test"
    raw_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    rows, counters = [], {}
    staging = Path(tempfile.mkdtemp(prefix="sl_"))
    try:
        for folder in sorted(p for p in src.iterdir() if p.is_dir()):
            mode, activity = classify(folder.name)
            if mode is None:
                continue
            for d in collect_recordings(folder, staging):
                try:
                    m, actual = merge_recording(d)
                except Exception as e:
                    print(f"  SKIP {folder.name}/{d.name}: {e}")
                    continue
                fs = int(m["fs_hz"].iloc[0])
                vc = value_cols(m)
                nan = int(m[vc].isna().sum().sum())
                m = m.dropna(subset=vc).reset_index(drop=True)
                dur = round(float(m["seconds_elapsed"].iloc[-1]), 1)
                fams = "+".join(p for p in VALUE_PREFIXES
                                if f"{p}_x" in m.columns)
                if mode == "train":
                    m["activity"] = activity
                key = (activity, fs)
                counters[key] = counters.get(key, 0) + 1
                fname = f"{activity}_{fs}hz_{counters[key]:02d}.csv"
                (raw_dir if mode == "train" else test_dir).joinpath(fname)
                m.to_csv((raw_dir if mode == "train" else test_dir) / fname, index=False)
                flag = "DURATION!" if (mode == "train" and not 5 <= dur <= 10) else ""
                if "grav" not in fams and mode == "train":
                    flag = (flag + " NO_GRAVITY").strip()
                rows.append(dict(file=fname, mode=mode, activity=activity, fs=fs,
                                 actual=round(actual, 1), dur=dur, n=len(m),
                                 nan=nan, sensors=fams, flag=flag))
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    if not rows:
        sys.exit(f"No recordings under {src}. Expected folders like 'walking-50hz' and 'mixed'.")

    rep = pd.DataFrame(rows)
    train = rep[rep["mode"] == "train"]
    print("\n=== Per-file ===")
    print(rep.drop(columns=["mode"]).to_string(index=False))
    if len(train):
        print("\n=== Per-activity (train) ===")
        g = train.groupby("activity").agg(files=("file", "size"),
                                          total_s=("dur", "sum")).reset_index()
        g["meets_90s"] = np.where(g["total_s"] >= 90, "OK", "SHORT!")
        print(g.to_string(index=False))
        rates = sorted(int(x) for x in train["fs"].unique())
        print(f"\nTrain files: {len(train)}  (target ~50)   rates present: {rates}")
    print(f"Test files: {len(rep) - len(train)}")
    bad = rep[rep["flag"] != ""]
    if len(bad):
        print(f"\nWARNING: {len(bad)} file(s) flagged - review the 'flag' column above.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build clean HAR dataset from Sensor Logger exports.")
    ap.add_argument("--src", required=True, type=Path,
                    help="folder that contains the activity export folders")
    ap.add_argument("--out", default=Path("."), type=Path,
                    help="project root; writes <out>/data/raw and <out>/data/test")
    a = ap.parse_args()
    process(a.src.expanduser(), a.out.expanduser())