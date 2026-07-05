#!/usr/bin/env python3
"""
diagnose.py - figure out WHY test accuracy is low.

For each mixed test session it prints, as run-length sequences:
  - ASSUMED true labels (from the equal-segment / order assumption in preprocess.py)
  - EMISSION-only prediction (per-window nearest Gaussian, no transitions) - supervised
  - VITERBI prediction - supervised model
  - VITERBI prediction - Baum-Welch model

How to read it:
  * If the VITERBI/emission runs form 4 clean blocks but in a DIFFERENT ORDER or with
    DIFFERENT BOUNDARIES than the assumed labels -> our test labels are wrong (fix the
    order/timing), the model is fine.
  * If the emission-only prediction already matches the assumed labels but Viterbi does
    not -> the transition matrix is the problem.
  * If every prediction is scattered noise -> a genuine signal/feature mismatch in the
    mixed recordings.

Usage: python diagnose.py --proc data/processed
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hmm import GaussianHMM, build_montages  # noqa: E402


def rle(seq, states):
    """Run-length encode a label sequence into 'name xN' chunks."""
    out, i = [], 0
    seq = list(seq)
    while i < len(seq):
        j = i
        while j < len(seq) and seq[j] == seq[i]:
            j += 1
        out.append(f"{states[seq[i]]}x{j - i}")
        i = j
    return "  ".join(out)


def emission_argmax(model, X):
    return np.argmax(model._log_emission(X), axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc", default="data/processed")
    a = ap.parse_args()
    proc = Path(a.proc)

    feats = pickle.load(open(proc / "features.pkl", "rb"))
    saved = pickle.load(open(proc / "hmm_model.pkl", "rb"))
    states, feat_names = feats["states"], feats["feat_names"]
    K, D = len(states), len(feat_names)

    seqs, labs = build_montages(feats["train"], n_montages=5, seed=0)
    sup = GaussianHMM(K, D); sup.fit_supervised(seqs, labs)
    bw = GaussianHMM(K, D)
    bw.pi, bw.A, bw.means, bw.vars = saved["pi"], saved["A"], saved["means"], saved["vars"]

    for s in feats["test"]:
        X, y = s["F"], s["y"]
        print(f"\n===== {s['name']}  ({len(X)} windows) =====")
        print("ASSUMED labels :", rle(y, states))
        print("EMISSION (sup) :", rle(emission_argmax(sup, X), states))
        print("VITERBI  (sup) :", rle(sup.viterbi(X), states))
        print("VITERBI  (bw)  :", rle(bw.viterbi(X), states))

    # sanity: does the supervised model recover TRAINING labels well per-window?
    Xtr = np.vstack([it["F"] for it in feats["train"]])
    ytr = np.concatenate([it["y"] for it in feats["train"]])
    acc = np.mean(emission_argmax(sup, Xtr) == ytr)
    print(f"\n[sanity] supervised emission-only accuracy on TRAINING windows: {acc:.3f}")
    print("        (if this is high but test is low, the test labels/order are the issue)")


if __name__ == "__main__":
    main()