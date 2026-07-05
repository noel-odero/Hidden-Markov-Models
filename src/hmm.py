#!/usr/bin/env python3
"""
hmm.py - Phase 4: a Gaussian (diagonal-covariance) Hidden Markov Model from scratch.

Implements, in numpy and in log-space for numerical stability:
  - Gaussian diagonal emission log-likelihoods
  - forward-backward (alpha/beta) with log-sum-exp
  - Viterbi decoding (most likely state path)
  - Baum-Welch (EM) training with a LOG-LIKELIHOOD convergence check (|dLL| < tol),
    not a fixed iteration count

Training strategy (see report):
  1. SUPERVISED INITIALISATION - because each recorded clip contains a single activity,
     we estimate per-state Gaussians (mean/var) by MLE from labelled windows, the start
     distribution from label frequencies, and an initial transition matrix from the
     transition counts of randomised training "montages" (clips concatenated in random
     activity order, so clip boundaries provide genuine inter-activity transitions).
     Supervised init keeps hidden state k aligned to activity k, which is what lets us
     build a confusion matrix later.
  2. BAUM-WELCH REFINEMENT - EM then refines pi, A and the emissions to maximise the
     data likelihood, monitored to convergence.

The driver (main) loads features.pkl, trains, validates against hmmlearn, and saves the
model plus the log-likelihood curve and transition heatmap.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

VAR_FLOOR = 1e-3
LOG_2PI = np.log(2.0 * np.pi)


def logsumexp(a, axis=None, keepdims=False):
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return out if keepdims else np.squeeze(out, axis=axis)


class GaussianHMM:
    """Diagonal-covariance Gaussian HMM (from scratch)."""

    def __init__(self, n_states, n_features, seed=0):
        self.K, self.D = n_states, n_features
        self.rng = np.random.default_rng(seed)
        self.pi = np.full(n_states, 1.0 / n_states)
        self.A = np.full((n_states, n_states), 1.0 / n_states)
        self.means = np.zeros((n_states, n_features))
        self.vars = np.ones((n_states, n_features))

    # --- emissions -----------------------------------------------------------
    def _log_emission(self, X):
        """Return log p(x_t | state=k) as (T, K)."""
        # (T,1,D) - (1,K,D)
        diff = X[:, None, :] - self.means[None, :, :]
        logdet = np.sum(np.log(self.vars), axis=1)                 # (K,)
        quad = np.sum(diff ** 2 / self.vars[None, :, :], axis=2)   # (T,K)
        return -0.5 * (self.D * LOG_2PI + logdet[None, :] + quad)

    # --- forward / backward --------------------------------------------------
    def _forward(self, logB):
        T = logB.shape[0]
        log_alpha = np.empty((T, self.K))
        logpi, logA = np.log(self.pi + 1e-300), np.log(self.A + 1e-300)
        log_alpha[0] = logpi + logB[0]
        for t in range(1, T):
            log_alpha[t] = logB[t] + logsumexp(
                log_alpha[t - 1][:, None] + logA, axis=0)
        return log_alpha, logsumexp(log_alpha[-1])

    def _backward(self, logB):
        T = logB.shape[0]
        log_beta = np.zeros((T, self.K))
        logA = np.log(self.A + 1e-300)
        for t in range(T - 2, -1, -1):
            log_beta[t] = logsumexp(
                logA + logB[t + 1][None, :] + log_beta[t + 1][None, :], axis=1)
        return log_beta

    # --- Viterbi -------------------------------------------------------------
    def viterbi(self, X):
        logB = self._log_emission(X)
        T = logB.shape[0]
        logpi, logA = np.log(self.pi + 1e-300), np.log(self.A + 1e-300)
        delta = np.empty((T, self.K))
        psi = np.zeros((T, self.K), dtype=int)
        delta[0] = logpi + logB[0]
        for t in range(1, T):
            scores = delta[t - 1][:, None] + logA        # (K_prev, K_cur)
            psi[t] = np.argmax(scores, axis=0)
            delta[t] = logB[t] + np.max(scores, axis=0)
        path = np.empty(T, dtype=int)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return path

    def score(self, sequences):
        return sum(self._forward(self._log_emission(X))[1] for X in sequences)

    # --- supervised initialisation ------------------------------------------
    def fit_supervised(self, sequences, labels):
        X = np.vstack(sequences)
        y = np.concatenate(labels)
        for k in range(self.K):
            xk = X[y == k]
            self.means[k] = xk.mean(axis=0)
            self.vars[k] = xk.var(axis=0) + VAR_FLOOR
        counts = np.bincount(y, minlength=self.K).astype(float)
        self.pi = counts / counts.sum()
        # transition counts from the (montage) label sequences
        A = np.ones((self.K, self.K))  # Laplace smoothing
        for yy in labels:
            for a, b in zip(yy[:-1], yy[1:]):
                A[a, b] += 1
        self.A = A / A.sum(axis=1, keepdims=True)
        return self

    # --- Baum-Welch (EM) -----------------------------------------------------
    def fit_baum_welch(self, sequences, max_iter=100, tol=1e-4, verbose=True):
        history = []
        for it in range(max_iter):
            # accumulators
            pi_acc = np.zeros(self.K)
            A_num = np.zeros((self.K, self.K))
            A_den = np.zeros(self.K)
            m_num = np.zeros((self.K, self.D))
            v_num = np.zeros((self.K, self.D))
            g_sum = np.zeros(self.K)
            total_ll = 0.0

            for X in sequences:
                logB = self._log_emission(X)
                log_alpha, ll = self._forward(logB)
                log_beta = self._backward(logB)
                total_ll += ll
                log_gamma = log_alpha + log_beta - ll
                gamma = np.exp(log_gamma)                       # (T,K)

                pi_acc += gamma[0]
                g_sum += gamma.sum(axis=0)
                m_num += gamma.T @ X
                v_num += gamma.T @ (X ** 2)

                # xi summed over t: (K,K)
                logA = np.log(self.A + 1e-300)
                T = X.shape[0]
                for t in range(T - 1):
                    lx = (log_alpha[t][:, None] + logA
                          + logB[t + 1][None, :] + log_beta[t + 1][None, :] - ll)
                    xi = np.exp(lx)
                    A_num += xi
                    A_den += gamma[t]

            # M-step
            self.pi = pi_acc / pi_acc.sum()
            self.A = A_num / np.maximum(A_den[:, None], 1e-300)
            self.A /= self.A.sum(axis=1, keepdims=True)
            self.means = m_num / g_sum[:, None]
            self.vars = v_num / g_sum[:, None] - self.means ** 2
            self.vars = np.maximum(self.vars, VAR_FLOOR)

            history.append(total_ll)
            if verbose:
                print(f"  BW iter {it:2d}  logL = {total_ll:.3f}")
            if it > 0 and abs(history[-1] - history[-2]) < tol:
                if verbose:
                    print(f"  converged at iter {it} (|dLL| < {tol})")
                break
        return history


# --- montage building --------------------------------------------------------
def build_montages(train_items, n_montages=5, seed=0):
    """Concatenate training clips in randomised activity order into longer sequences,
    so clip boundaries provide inter-activity transitions for A estimation."""
    rng = np.random.default_rng(seed)
    items = [it for it in train_items if len(it["F"])]
    order = rng.permutation(len(items))
    groups = np.array_split(order, n_montages)
    seqs, labs = [], []
    for g in groups:
        F = np.vstack([items[i]["F"] for i in g])
        y = np.concatenate([items[i]["y"] for i in g])
        seqs.append(F)
        labs.append(y)
    return seqs, labs


# --- hmmlearn validation -----------------------------------------------------
def validate_against_hmmlearn(model, sequences):
    """Cross-check our forward log-likelihood and Viterbi path against hmmlearn
    using identical parameters. Returns a dict of comparison metrics."""
    from hmmlearn import hmm as hl
    ref = hl.GaussianHMM(n_components=model.K, covariance_type="diag",
                         init_params="", params="")
    ref.startprob_ = model.pi.copy()
    ref.transmat_ = model.A.copy()
    ref.means_ = model.means.copy()
    ref.covars_ = model.vars.copy()

    X = np.vstack(sequences)
    lengths = [len(s) for s in sequences]
    our_ll = model.score(sequences)
    ref_ll = ref.score(X, lengths)

    our_paths = np.concatenate([model.viterbi(s) for s in sequences])
    ref_paths = ref.predict(X, lengths)
    agree = float(np.mean(our_paths == ref_paths))
    return dict(our_ll=our_ll, ref_ll=ref_ll,
                ll_diff=abs(our_ll - ref_ll), path_agreement=agree)


def save_plots(history, model, states, plots_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(history)), history, "o-")
    ax.set(xlabel="Baum-Welch iteration", ylabel="total log-likelihood",
           title="Baum-Welch convergence")
    fig.tight_layout()
    p1 = Path(plots_dir) / "bw_convergence.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(model.A, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(model.K))
    ax.set_yticks(range(model.K))
    ax.set_xticklabels(states, rotation=45, ha="right")
    ax.set_yticklabels(states)
    for i in range(model.K):
        for j in range(model.K):
            ax.text(j, i, f"{model.A[i, j]:.2f}", ha="center", va="center",
                    color="black" if model.A[i, j] < 0.5 else "white", fontsize=9)
    ax.set(xlabel="to state", ylabel="from state", title="Transition matrix A")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    p2 = Path(plots_dir) / "transition_matrix.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    return p1, p2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc", default="data/processed")
    ap.add_argument("--plots", default="plots")
    ap.add_argument("--out", default="data/processed/hmm_model.pkl")
    a = ap.parse_args()

    data = pickle.load(open(Path(a.proc) / "features.pkl", "rb"))
    train, states = data["train"], data["states"]
    D = len(data["feat_names"])
    K = len(states)

    seqs, labs = build_montages(train, n_montages=5)
    model = GaussianHMM(K, D)
    model.fit_supervised(seqs, labs)

    # training accuracy of the supervised init (sanity check)
    init_paths = np.concatenate([model.viterbi(s) for s in seqs])
    init_true = np.concatenate(labs)
    print(f"Supervised-init training accuracy: {np.mean(init_paths==init_true):.3f}")

    print("Baum-Welch refinement:")
    history = model.fit_baum_welch(seqs, max_iter=100, tol=1e-4)

    bw_paths = np.concatenate([model.viterbi(s) for s in seqs])
    print(f"Post-BW training accuracy:         {np.mean(bw_paths==init_true):.3f}")

    v = validate_against_hmmlearn(model, seqs)
    print("\nValidation vs hmmlearn (same parameters):")
    print(f"our logL = {v['our_ll']:.4f}   hmmlearn logL = {v['ref_ll']:.4f}   "
          f"|diff| = {v['ll_diff']:.2e}")
    print(f"Viterbi path agreement = {v['path_agreement']*100:.2f}%")

    p1, p2 = save_plots(history, model, states, a.plots)
    pickle.dump(dict(pi=model.pi, A=model.A, means=model.means, vars=model.vars,
                     states=states, feat_names=data["feat_names"],
                     mean=data["mean"], std=data["std"], ll_history=history),
                open(a.out, "wb"))
    print(f"\nSaved model -> {a.out}")
    print(f"Saved plots -> {p1}, {p2}")


if __name__ == "__main__":
    main()