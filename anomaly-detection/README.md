# Multivariate Time-Series Anomaly Detection (SMD)

Anomaly detection on server metrics using the **Server Machine Dataset (SMD)**.
This project implements an Isolation Forest baseline and an LSTM Autoencoder,
evaluating both with precision, recall, F1, point-adjusted F1, and PR-AUC.

## Project Structure

```
anomaly-detection/
├── configs/default.yaml      # all hyperparameters and paths
├── data/                     # raw & processed data (gitignored)
├── notebooks/eda.ipynb       # exploratory data analysis
├── results/                  # metrics.json, score plots, checkpoints
├── src/
│   ├── data.py               # download, load, scale, window, DataLoaders
│   ├── models/
│   │   ├── iforest.py        # Isolation Forest baseline
│   │   └── lstm_ae.py        # LSTM Autoencoder
│   ├── train.py              # CLI training loop
│   ├── evaluate.py           # metrics, thresholding, reporting
│   └── utils.py              # seeding, config, device, logging
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Create a virtual environment (Python 3.10+)
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

The SMD dataset is downloaded automatically on first run.  If the download
fails, the script prints manual instructions.

## How to Run

```bash
# Train both models (default config)
python -m src.train --model all --config configs/default.yaml

# Train only the LSTM autoencoder
python -m src.train --model lstm_ae --config configs/default.yaml

# Train only Isolation Forest
python -m src.train --model iforest --config configs/default.yaml

# Quick smoke test (2 epochs)
python -m src.train --model all --config configs/default.yaml --epochs 2
```

Results are saved to `results/metrics.json` and score-vs-label plots are
written to `results/`.

## Configuration

All hyperparameters are in `configs/default.yaml`:

| Parameter             | Default | Description                        |
|-----------------------|---------|------------------------------------|
| `data.machine_id`    | machine-1-1 | SMD machine to evaluate        |
| `window.length`      | 100     | Sliding window size                |
| `lstm_ae.hidden_dim` | 128     | LSTM hidden dimension              |
| `lstm_ae.latent_dim` | 64      | Bottleneck dimension               |
| `training.epochs`    | 50      | Max training epochs                |
| `training.lr`        | 0.001   | Learning rate                      |
| `evaluation.threshold_percentile` | 99 | Val-score percentile for threshold |

## Goal

Build a solid, reproducible baseline for server anomaly detection that can
later be extended with more advanced models (Transformer AE, GNN, etc.) and
a live monitoring dashboard.
