# Human Activity Recognition with a Hidden Markov Model

Recognizing four physical activities, standing, walking, jumping, and still, from
smartphone accelerometer and gyroscope data using a Gaussian Hidden Markov Model
implemented from scratch in numpy (Viterbi decoding and Baum-Welch training).

## Project structure

```
data/
  raw/          labelled training clips (one activity per file, 50 Hz and 100 Hz)
  test/         held-out "mixed" sessions used only for evaluation
  processed/    pickled train/test windows, features, and trained model
src/
  preprocess.py   trims, resamples to a common 50 Hz grid, windows, and labels clips
  features.py     extracts time and frequency domain features per window
  hmm.py          Gaussian HMM: Viterbi, Baum-Welch, hmmlearn validation
  evaluate.py     decodes test sessions and reports metrics and plots
  diagnose.py     debugging helper for inspecting decoded label sequences
plots/          generated figures (signals, transition matrix, convergence,
                confusion matrix, emission means, decoded timeline)
har_hmm.ipynb   end to end notebook: data, features, model, evaluation, discussion
```

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pandas matplotlib hmmlearn jupyter
```

## Running the pipeline

The notebook `har_hmm.ipynb` runs the full pipeline and is the primary deliverable.
Each stage can also be run standalone from the command line:

```
python src/preprocess.py --raw data/raw --test data/test --out data/processed --plots plots
python src/features.py --proc data/processed
python src/hmm.py --proc data/processed --plots plots
python src/evaluate.py --proc data/processed --plots plots
```

## Method summary

- **Data**: 54 labelled clips (standing, walking, jumping, still) at ~50 Hz and ~100 Hz,
  plus two continuous "mixed" sessions held out for testing.
- **Preprocessing**: clips are resampled onto a common 50 Hz grid and split into
  2 second, 100 sample windows with 50% overlap.
- **Features**: 11 features per window, 8 time domain (RMS, variance, SMA, gyro RMS,
  gravity means, axis correlation) and 3 frequency domain (dominant frequency, spectral
  energy, spectral entropy), z-scored using training statistics only.
- **Model**: a 4 state diagonal covariance Gaussian HMM, initialised with supervised
  MLE estimates and refined with Baum-Welch to a log-likelihood convergence criterion,
  validated against hmmlearn.
- **Evaluation**: Viterbi decoding on the two unseen mixed sessions, reporting per
  activity sensitivity, specificity, and accuracy, with confusion matrix, transition
  matrix, emission means, and decoded sequence timeline plots.

## Results

The Baum-Welch refined model reaches 96.2% accuracy on pure (non-boundary) windows of
the unseen test sessions, versus 71.2% for the supervised-init model. Full discussion
is in the notebook's final section.
