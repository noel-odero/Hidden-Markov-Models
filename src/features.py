#!/usr/bin/env python3
"""
features.py - Phase 3: turn each 2 s window into a justified feature vector.

Per window we compute 11 features. Each is chosen to separate specific activities:

TIME-DOMAIN (from linear-accel magnitude unless noted)
  acc_rms      - root-mean-square of accel magnitude. Energy of motion: high for
                 walking/jumping, ~0 for standing/still. Separates dynamic vs static.
  acc_var      - variance of accel magnitude. Movement intensity / burstiness;
                 jumping (large impacts) > walking > static.
  acc_sma      - signal magnitude area, mean(|ax|+|ay|+|az|). Classic overall
                 activity-level feature in HAR.
  gyr_rms      - RMS of gyroscope magnitude. Rotational intensity; walking has arm/
                 body rotation, static has almost none.
  grav_x_mean  } means of the gravity vector. These encode DEVICE ORIENTATION and are
  grav_y_mean  } what separate standing (phone vertical) from still (phone flat) - two
  grav_z_mean  } activities that look identical in motion features.
  corr_acc_xz  - correlation between accel x and z axes. Captures coordinated multi-axis
                 motion patterns (gait) vs uncorrelated noise.

FREQUENCY-DOMAIN (FFT of the mean-removed accel magnitude)
  dom_freq     - dominant frequency (peak of the spectrum). Walking ~1-2 Hz, jumping
                 ~2-3 Hz, static ~0. Directly separates the two rhythmic activities.
  spec_energy  - total spectral power (excluding DC). Oscillatory energy: high for
                 dynamic activities.
  spec_entropy - spectral entropy. Low for a clean periodic signal (jumping), higher
                 for broadband/noisy or static windows. Separates rhythmic vs irregular.

NORMALIZATION
  Features live on wildly different scales (gravity ~9.8, dom_freq ~0-5 Hz, variance
  tiny). The HMM uses diagonal-covariance Gaussian emissions, which are ill-conditioned
  when scales differ by orders of magnitude. We therefore Z-score every feature. The
  scaler mean/std are fit on TRAINING windows only and then applied to the test
  sessions, so no test information leaks into preprocessing.

Input:  data/processed/train.pkl, test.pkl   (from preprocess.py)
Output: data/processed/features.pkl          (feature sequences + scaler + names)

Usage:  python features.py --proc data/processed
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-8
FEATURE_NAMES = ["acc_rms", "acc_var", "acc_sma", "gyr_rms",
                 "grav_x_mean", "grav_y_mean", "grav_z_mean", "corr_acc_xz",
                 "dom_freq", "spec_energy", "spec_entropy"]


def compute_features(win, fs):
    """win: (win_len, 9) [acc xyz, gyr xyz, grav xyz]. Returns (11,) feature vector."""
    acc, gyr, grav = win[:, 0:3], win[:, 3:6], win[:, 6:9]
    accmag = np.linalg.norm(acc, axis=1)
    gyrmag = np.linalg.norm(gyr, axis=1)

    acc_rms = np.sqrt(np.mean(accmag ** 2))
    acc_var = np.var(accmag)
    acc_sma = np.mean(np.sum(np.abs(acc), axis=1))
    gyr_rms = np.sqrt(np.mean(gyrmag ** 2))
    grav_mean = grav.mean(axis=0)  # (3,)

    sx, sz = acc[:, 0], acc[:, 2]
    if sx.std() < EPS or sz.std() < EPS:
        corr_xz = 0.0
    else:
        corr_xz = float(np.corrcoef(sx, sz)[0, 1])
        if not np.isfinite(corr_xz):
            corr_xz = 0.0

    # FFT of the mean-removed accel magnitude
    sig = accmag - accmag.mean()
    spec = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(len(sig), 1.0 / fs)
    power = spec[1:] ** 2                       # drop DC bin
    if power.sum() < EPS:
        dom_freq, spec_energy, spec_entropy = 0.0, 0.0, 0.0
    else:
        dom_freq = float(freqs[1:][np.argmax(power)])
        spec_energy = float(power.sum())
        p = power / power.sum()
        spec_entropy = float(-np.sum(p * np.log(p + EPS)))

    return np.array([acc_rms, acc_var, acc_sma, gyr_rms,
                     grav_mean[0], grav_mean[1], grav_mean[2], corr_xz,
                     dom_freq, spec_energy, spec_entropy])


def featurize(items, fs):
    """Replace each item's raw windows X with a feature matrix F (n_win, n_feat)."""
    out = []
    for it in items:
        F = np.stack([compute_features(w, fs) for w in it["X"]]) if len(it["X"]) \
            else np.empty((0, len(FEATURE_NAMES)))
        rec = dict(name=it["name"], F=F, y=it["y"])
        if "pure" in it:
            rec["pure"] = it["pure"]
        out.append(rec)
    return out


def fit_scaler(train_items):
    """Z-score parameters from pooled TRAINING features only."""
    allF = np.vstack([it["F"] for it in train_items if len(it["F"])])
    mean = allF.mean(axis=0)
    std = allF.std(axis=0)
    std[std < EPS] = 1.0
    return mean, std


def apply_scaler(items, mean, std):
    for it in items:
        if len(it["F"]):
            it["F"] = (it["F"] - mean) / std
    return items


def separability_table(train_items, states):
    """Per-activity mean of each RAW feature - a quick separability sanity check."""
    rows = {a: [] for a in states}
    for it in train_items:
        rows[states[it["y"][0]]].append(it["F"])
    data = {a: np.vstack(v).mean(axis=0) for a, v in rows.items() if v}
    return pd.DataFrame(data, index=FEATURE_NAMES).round(3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc", default="data/processed")
    a = ap.parse_args()
    proc = Path(a.proc)

    train = pickle.load(open(proc / "train.pkl", "rb"))
    test = pickle.load(open(proc / "test.pkl", "rb"))
    fs, states = train["fs"], train["states"]

    train_F = featurize(train["clips"], fs)
    test_F = featurize(test["sessions"], fs)

    print("=== Per-activity mean of each RAW feature (separability check) ===")
    print(separability_table(train_F, states).to_string())

    mean, std = fit_scaler(train_F)
    train_F = apply_scaler(train_F, mean, std)
    test_F = apply_scaler(test_F, mean, std)

    pickle.dump(dict(train=train_F, test=test_F, feat_names=FEATURE_NAMES,
                     mean=mean, std=std, states=states, fs=fs),
                open(proc / "features.pkl", "wb"))

    n_train = sum(len(it["F"]) for it in train_F)
    n_test = sum(len(it["F"]) for it in test_F)
    print(f"\nFeatures: {len(FEATURE_NAMES)}  "
          f"| train windows {n_train}  | test windows {n_test}")
    print(f"Saved: {proc/'features.pkl'}")


if __name__ == "__main__":
    main()