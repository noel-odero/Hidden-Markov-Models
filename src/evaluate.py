#!/usr/bin/env python3
"""
evaluate.py - Phase 5-6: decode the unseen mixed test sessions and report performance.

For BOTH the supervised-init model and the Baum-Welch-refined model it:
  - Viterbi-decodes each test session,
  - builds the confusion matrix (pooled across both sessions),
  - computes per-activity sensitivity, specificity and accuracy (one-vs-rest),
  - reports overall accuracy on pure windows and on all windows (incl. boundaries),
  - reports per-session accuracy (a 50 Hz vs 100 Hz cross-rate check),
  - saves a confusion-matrix figure and a state-conditional emission-means figure.

Metrics are computed on the held-out test sessions only - never on training data.

Usage:
  python evaluate.py --proc data/processed --plots plots
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hmm import GaussianHMM, build_montages  # noqa: E402


def per_class_metrics(cm):
    """From confusion matrix (rows=true, cols=pred) return per-class
    sensitivity, specificity, accuracy, support, and overall accuracy."""
    total = cm.sum()
    K = cm.shape[0]
    rows = []
    for k in range(K):
        tp = cm[k, k]
        fn = cm[k, :].sum() - tp
        fp = cm[:, k].sum() - tp
        tn = total - tp - fn - fp
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        acc = (tp + tn) / total
        rows.append((int(tp + fn), sens, spec, acc))
    overall = np.trace(cm) / total
    return rows, overall


def confusion(y_true, y_pred, K):
    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def decode_sessions(model, sessions):
    """Return pooled (y_true, y_pred, pure_mask) across all test sessions,
    plus per-session accuracy."""
    yt, yp, pm, per_sess = [], [], [], []
    for s in sessions:
        pred = model.viterbi(s["F"])
        yt.append(s["y"]); yp.append(pred); pm.append(s["pure"])
        per_sess.append((s["name"], float(np.mean(pred == s["y"]))))
    return (np.concatenate(yt), np.concatenate(yp),
            np.concatenate(pm), per_sess)


def report(name, model, sessions, states):
    K = len(states)
    yt, yp, pure, per_sess = decode_sessions(model, sessions)

    print(f"\n{name}")
    for sname, acc in per_sess:
        print(f"  {sname}: overall window accuracy = {acc:.3f}")
    all_acc = float(np.mean(yt == yp))
    pure_acc = float(np.mean(yt[pure] == yp[pure]))
    print(f"  Overall accuracy - all windows : {all_acc:.3f}")
    print(f"  Overall accuracy - pure windows: {pure_acc:.3f}")

    cm = confusion(yt[pure], yp[pure], K)          # pure windows for the table
    rows, overall = per_class_metrics(cm)
    print(f"\n  Per-activity metrics (pure windows), overall acc = {overall:.3f}")
    print(f"  {'activity':10s} {'n':>4s} {'sens':>7s} {'spec':>7s} {'acc':>7s}")
    for k, (n, se, sp, ac) in enumerate(rows):
        print(f"  {states[k]:10s} {n:4d} {se:7.3f} {sp:7.3f} {ac:7.3f}")
    return cm, all_acc, pure_acc, rows, overall


def save_confusion(cm, states, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(states))); ax.set_yticks(range(len(states)))
    ax.set_xticklabels(states, rotation=45, ha="right"); ax.set_yticklabels(states)
    thresh = cm.max() / 2 if cm.max() else 0.5
    for i in range(len(states)):
        for j in range(len(states)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    ax.set(xlabel="predicted", ylabel="true", title="Confusion matrix (test, pure windows)")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def save_emissions(model, states, feat_names, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(model.means, cmap="coolwarm", aspect="auto",
                   vmin=-np.abs(model.means).max(), vmax=np.abs(model.means).max())
    ax.set_yticks(range(len(states))); ax.set_yticklabels(states)
    ax.set_xticks(range(len(feat_names)))
    ax.set_xticklabels(feat_names, rotation=45, ha="right", fontsize=8)
    ax.set_title("State-conditional emission means (z-scored features)")
    fig.colorbar(im, fraction=0.025)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def save_decoded_timeline(model, sessions, states, path, hop_seconds=1.0):
    """Per test session: true vs Viterbi-decoded activity over time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(sessions), 1, figsize=(10, 2.6 * len(sessions)), squeeze=False)
    for ax, s in zip(axes[:, 0], sessions):
        pred = model.viterbi(s["F"])
        t = np.arange(len(pred)) * hop_seconds
        ax.step(t, s["y"], where="post", label="true", color="tab:gray", linewidth=2.5)
        ax.step(t, pred, where="post", label="predicted", color="tab:red",
                linewidth=1.3, linestyle="--")
        ax.set_yticks(range(len(states))); ax.set_yticklabels(states)
        ax.set_ylabel(s["name"])
    axes[-1, 0].set_xlabel("time (s)")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Decoded activity sequence: true vs Viterbi-predicted")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc", default="data/processed")
    ap.add_argument("--plots", default="plots")
    a = ap.parse_args()
    proc, plots = Path(a.proc), Path(a.plots)
    plots.mkdir(parents=True, exist_ok=True)

    feats = pickle.load(open(proc / "features.pkl", "rb"))
    saved = pickle.load(open(proc / "hmm_model.pkl", "rb"))
    states, feat_names = feats["states"], feats["feat_names"]
    K, D = len(states), len(feat_names)
    sessions = feats["test"]

    # rebuild supervised-init model (deterministic, same montage seed as training)
    seqs, labs = build_montages(feats["train"], n_montages=5, seed=0)
    sup = GaussianHMM(K, D)
    sup.fit_supervised(seqs, labs)

    # BW-refined model from disk
    bw = GaussianHMM(K, D)
    bw.pi, bw.A = saved["pi"], saved["A"]
    bw.means, bw.vars = saved["means"], saved["vars"]

    cm_sup, *_ = report("SUPERVISED-INIT MODEL", sup, sessions, states)
    cm_bw, bw_all, bw_pure, rows_bw, overall_bw = report(
        "BAUM-WELCH-REFINED MODEL", bw, sessions, states)

    save_confusion(cm_bw, states, plots / "confusion_matrix.png")
    save_emissions(bw, states, feat_names, plots / "emission_means.png")
    save_decoded_timeline(bw, sessions, states, plots / "decoded_timeline.png")
    print(f"\nSaved: {plots/'confusion_matrix.png'}, {plots/'emission_means.png'}, "
          f"{plots/'decoded_timeline.png'}")


if __name__ == "__main__":
    main()