#!/usr/bin/env python3
"""
preprocess.py - Phase 2: trim, harmonize sampling rate, window, label, visualize.

Pipeline (each step maps to a Data Collection / feature rubric clause):
  1. TRIM   - cap every clip at MAX_SECONDS so all clips sit in the 5-10 s band.
  2. HARMONIZE - linearly interpolate every clip (recorded at ~50 or ~100 Hz) onto a
     common uniform TARGET_HZ grid, so a window always spans the same physical time
     regardless of the source device rate. This is the sampling-rate handling clause.
  3. WINDOW - slide a WIN_SECONDS window with OVERLAP fraction. At 50 Hz a 2 s window
     is 100 samples; walking/jumping cycle at ~1-2 Hz, so 2 s captures 2-4 full cycles
     - enough to characterise the rhythm while staying locally stationary.
  4. LABEL  - training clips inherit their activity label on every window; the two mixed
     test sessions are segmented by a fixed activity order (equal segments) and each
     window is majority-labelled, with boundary-straddling windows flagged impure.
  5. PLOT   - save one raw-signal overview figure (accel/gyro/gravity per activity).

Outputs:
  <out>/train.pkl, <out>/test.pkl  - lists of per-clip dicts:
      {name, fs_src, X (n_win, win_len, n_ch), y (n_win,), pure (n_win,)}
  <plots>/sample_signals.png       - report-ready raw-signal figure.

Usage:
  python preprocess.py --raw data/raw --test data/test --out data/processed --plots plots
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless: save figures without a display
import matplotlib.pyplot as plt

# --- configuration -----------------------------------------------------------
TARGET_HZ = 50            # common grid all clips are resampled onto
MAX_SECONDS = 10.0        # trim cap (rubric wants 5-10 s clips)
WIN_SECONDS = 2.0         # window length in seconds
OVERLAP = 0.5             # fraction overlap between consecutive windows
STATES = ["standing", "walking", "jumping", "still"]   # fixed id order
ACT2ID = {a: i for i, a in enumerate(STATES)}
CHANNELS = ["acc_x", "acc_y", "acc_z",
            "gyr_x", "gyr_y", "gyr_z",
            "grav_x", "grav_y", "grav_z"]

WIN_LEN = int(WIN_SECONDS * TARGET_HZ)          # 100 samples
HOP = int(WIN_LEN * (1 - OVERLAP))              # 50 samples


# --- core steps --------------------------------------------------------------
def trim(df, max_s=MAX_SECONDS):
    """Keep only the first max_s seconds of a clip."""
    return df[df["seconds_elapsed"] <= max_s].reset_index(drop=True)


def resample_uniform(df, fs=TARGET_HZ):
    """Linearly interpolate all sensor channels onto a uniform fs grid.
    Returns (t_new, X) where X is (N, n_channels)."""
    t_old = df["seconds_elapsed"].to_numpy()
    dur = t_old[-1]
    n = int(np.floor(dur * fs)) + 1
    t_new = np.arange(n) / fs
    X = np.column_stack([np.interp(t_new, t_old, df[c].to_numpy()) for c in CHANNELS])
    return t_new, X


def make_windows(X, win=WIN_LEN, hop=HOP):
    """Slice (N, C) into (n_win, win, C) via a sliding window."""
    n = X.shape[0]
    if n < win:
        return np.empty((0, win, X.shape[1]))
    starts = range(0, n - win + 1, hop)
    return np.stack([X[s:s + win] for s in starts])


def segment_labels(n_samples, order, fs=TARGET_HZ):
    """Per-sample activity ids for a mixed session split into equal segments."""
    seg = n_samples // len(order)
    lab = np.empty(n_samples, dtype=int)
    for i, act in enumerate(order):
        lo = i * seg
        hi = n_samples if i == len(order) - 1 else (i + 1) * seg
        lab[lo:hi] = ACT2ID[act]
    return lab


def window_labels(sample_labels, win=WIN_LEN, hop=HOP):
    """Majority label per window + a 'pure' flag (window spans a single activity)."""
    y, pure = [], []
    for s in range(0, len(sample_labels) - win + 1, hop):
        seg = sample_labels[s:s + win]
        vals, counts = np.unique(seg, return_counts=True)
        y.append(int(vals[np.argmax(counts)]))
        pure.append(len(vals) == 1)
    return np.array(y), np.array(pure, dtype=bool)


# --- orchestration -----------------------------------------------------------
def process_train(raw_dir):
    clips = []
    for f in sorted(Path(raw_dir).glob("*.csv")):
        df = pd.read_csv(f)
        activity = str(df["activity"].iloc[0])
        _, X = resample_uniform(trim(df))
        W = make_windows(X)
        if len(W) == 0:
            print(f"  WARN {f.name}: too short after trim, skipped")
            continue
        y = np.full(len(W), ACT2ID[activity])
        clips.append(dict(name=f.name, fs_src=int(df["fs_hz"].iloc[0]),
                          X=W, y=y, pure=np.ones(len(W), dtype=bool)))
    return clips


def process_test(test_dir, order):
    sessions = []
    for f in sorted(Path(test_dir).glob("*.csv")):
        df = pd.read_csv(f)
        _, X = resample_uniform(trim(df, max_s=1e9))     # don't trim test sessions
        slab = segment_labels(len(X), order)
        W = make_windows(X)
        y, pure = window_labels(slab)
        sessions.append(dict(name=f.name, fs_src=int(df["fs_hz"].iloc[0]),
                             X=W, y=y, pure=pure))
    return sessions


def plot_samples(raw_dir, plots_dir):
    """4x3 grid of full representative clips: accel magnitude, gyro magnitude,
    gravity axes - one row per activity. Loads one 50 Hz clip per activity."""
    Path(plots_dir).mkdir(parents=True, exist_ok=True)
    rep = {}
    for f in sorted(Path(raw_dir).glob("*.csv")):
        df = pd.read_csv(f)
        act = str(df["activity"].iloc[0])
        if act not in rep and int(df["fs_hz"].iloc[0]) == 50:
            rep[act] = df
    fig, ax = plt.subplots(len(STATES), 3, figsize=(11, 9))
    for r, act in enumerate(STATES):
        t, X = resample_uniform(trim(rep[act]))
        acc = np.linalg.norm(X[:, 0:3], axis=1)
        gyr = np.linalg.norm(X[:, 3:6], axis=1)
        ax[r, 0].plot(t, acc, color="tab:blue")
        ax[r, 1].plot(t, gyr, color="tab:red")
        for k, lab in zip(range(6, 9), ["x", "y", "z"]):
            ax[r, 2].plot(t, X[:, k], label=f"grav_{lab}")
        ax[r, 0].set_ylabel(act, fontsize=11, fontweight="bold")
    ax[0, 0].set_title("Linear accel magnitude (m/s^2)")
    ax[0, 1].set_title("Gyroscope magnitude (rad/s)")
    ax[0, 2].set_title("Gravity (x/y/z)")
    ax[0, 2].legend(fontsize=7, loc="upper right")
    for c in range(3):
        ax[-1, c].set_xlabel("time (s)")
    fig.suptitle("Representative sensor traces per activity (50 Hz, harmonized)",
                 fontsize=13)
    fig.tight_layout()
    out = Path(plots_dir) / "sample_signals.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def summarize(train, test):
    print("\n=== Windowing summary ===")
    print(f"Window: {WIN_LEN} samples = {WIN_SECONDS}s @ {TARGET_HZ}Hz, "
          f"hop {HOP} ({int(OVERLAP*100)}% overlap)")
    per_act = {a: 0 for a in STATES}
    for c in train:
        per_act[STATES[c["y"][0]]] += len(c["X"])
    print("Train windows per activity:", per_act,
          "| total", sum(per_act.values()))
    for s in test:
        print(f"Test {s['name']}: {len(s['X'])} windows "
              f"({int(s['pure'].sum())} pure, {int((~s['pure']).sum())} boundary)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--test", default="data/test")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--plots", default="plots")
    ap.add_argument("--test-order", default="standing,walking,jumping,still")
    a = ap.parse_args()

    order = [s.strip() for s in a.test_order.split(",")]
    assert set(order) == set(STATES), f"test-order must be a permutation of {STATES}"

    train = process_train(a.raw)
    test = process_test(a.test, order)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    with open(Path(a.out) / "train.pkl", "wb") as fh:
        pickle.dump(dict(clips=train, states=STATES, channels=CHANNELS,
                         fs=TARGET_HZ, win_len=WIN_LEN, hop=HOP), fh)
    with open(Path(a.out) / "test.pkl", "wb") as fh:
        pickle.dump(dict(sessions=test, states=STATES, channels=CHANNELS,
                         fs=TARGET_HZ, win_len=WIN_LEN, hop=HOP, order=order), fh)
    fig = plot_samples(a.raw, a.plots)
    summarize(train, test)
    print(f"\nSaved: {a.out}/train.pkl, {a.out}/test.pkl, {fig}")


if __name__ == "__main__":
    main()