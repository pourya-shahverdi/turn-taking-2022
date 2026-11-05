# Learning Turn-Taking from Human Demonstrations

This project studies how a social robot can learn when to speak from human
examples. It uses job interviews to learn the interviewer's timing, rather than
only detecting when the other person stops speaking.

**Paper:** [Learning Turn-Taking Behavior from Human Demonstrations for Social Human-Robot Interactions](https://ieeexplore.ieee.org/abstract/document/9981243/)

## How It Works

![How audio, body position, and text become model inputs.](docs/figures/pipeline.svg)

The scripts combine audio features, a summary of body position, speech
recognition confidence, a TurnGPT score, and whether the interviewee is speaking.
An LSTM model uses 7 seconds of history to predict speech activity over the next
3 seconds, at 20 samples per second.

![The model uses 7 seconds of history to predict the next 3 seconds.](docs/figures/forecast-window.svg)

*These diagrams show how the code works, not experiment results.*

## Study Findings

The paper reports results from five students, each interviewed three times,
giving about 28 minutes of data. Voice and text features together gave the
lowest prediction error. Adding gesture features reduced performance in this
small study.

## Getting Started

Use an older TensorFlow/Keras release (before Keras 3), plus NumPy, Matplotlib,
scikit-learn, librosa, PyYAML, and tqdm. Preparing raw recordings also needs
ROS 1 and TurnGPT/PyTorch. Recordings and trained models are not included.

Change the data paths in the scripts and create output folders. Training reads
CSV files without headers: `merged_features/AB_<id>_<run>.csv`, for IDs
`1, 2, 4, 5, 6` and runs `a, b, c`.

The preparation scripts extract audio and body features, process transcripts,
add TurnGPT scores, and combine the inputs with [merge_all.py](merge_all.py).

## Run Experiments

```bash
mkdir -p logs
python train.py --features all --units 15
python train_kfold.py --dir experiment-01 --features all --units 15
python eval_kfold.py logs/experiment-01 --name experiment-01
```

The Python commands train one model, train five models with different recordings
held out, and test those five models. Use `--units 15` to match the test scripts.
Other script defaults may differ from the paper's settings.

Choose inputs with `--features`: `all`, `prosody`, `gesture`, `asr`, or `silence`.
The five-fold scripts also support `nogesture`.

Models, training settings, and error plots are saved under `logs/`. Testing
saves predictions and a precision-recall plot in the results folder set in the
code. Use a new experiment name for each run.

**Note:** [pca.py](pca.py) currently learns from all pose files. For a fair test,
fit PCA using only the training recordings.