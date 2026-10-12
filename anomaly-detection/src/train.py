"""
train.py — CLI entry point for training and evaluating models.

Usage:
    python -m src.train --model lstm_ae --config configs/default.yaml
    python -m src.train --model iforest --config configs/default.yaml
    python -m src.train --model all     --config configs/default.yaml
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.utils import set_seed, load_config, get_device, setup_logging, ensure_dir
from src.data import prepare_dataloaders, create_windows, download_smd, load_machine, scale_data
from src.models.lstm_ae import LSTMAutoencoder
from src.models.iforest import IForestDetector
from src.evaluate import (
    select_threshold,
    compute_metrics,
    print_comparison_table,
    save_results,
    plot_scores,
)

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────
# LSTM Autoencoder training
# ────────────────────────────────────────────────────────────────────────
def train_lstm_ae(cfg: dict, device: torch.device):
    """Train the LSTM autoencoder and return (model, val_scores, test_scores, test_labels)."""
    train_loader, val_loader, test_loader, test_labels = prepare_dataloaders(cfg)

    mc = cfg["lstm_ae"]
    tc = cfg["training"]

    model = LSTMAutoencoder(
        input_dim=38,
        hidden_dim=mc["hidden_dim"],
        latent_dim=mc["latent_dim"],
        seq_len=cfg["window"]["length"],
        num_layers=mc["num_layers"],
        dropout=mc["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    ckpt_dir = ensure_dir("results/checkpoints")
    best_ckpt = ckpt_dir / "lstm_ae_best.pt"

    logger.info(f"Training LSTM-AE for up to {tc['epochs']} epochs …")

    for epoch in range(1, tc["epochs"] + 1):
        # ── Train ───────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x = batch.to(device) if not isinstance(batch, (list, tuple)) else batch[0].to(device)
            recon = model(x)
            loss = criterion(recon, x)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_loader.dataset)

        # ── Validate ────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch.to(device) if not isinstance(batch, (list, tuple)) else batch[0].to(device)
                recon = model(x)
                val_loss += criterion(recon, x).item() * x.size(0)
        val_loss /= len(val_loader.dataset)

        logger.info(f"Epoch {epoch:>3}/{tc['epochs']}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        # ── Early stopping ──────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_ckpt)
        else:
            patience_counter += 1
            if patience_counter >= tc["early_stop_patience"]:
                logger.info(f"Early stopping at epoch {epoch} (patience={tc['early_stop_patience']})")
                break

    # Load best checkpoint
    model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))
    model.eval()
    logger.info(f"Best checkpoint loaded (val_loss={best_val_loss:.6f})")

    # ── Score computation ───────────────────────────────────────────
    val_scores = _compute_scores(model, val_loader, device)
    test_scores = _compute_scores(model, test_loader, device)

    return model, val_scores, test_scores, test_labels


def _compute_scores(
    model: LSTMAutoencoder,
    loader,
    device: torch.device,
) -> np.ndarray:
    """Collect per-window reconstruction errors from a DataLoader."""
    all_errors = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch.to(device) if not isinstance(batch, (list, tuple)) else batch[0].to(device)
            errors = model.reconstruction_error(x)  # (B,)
            all_errors.append(errors.cpu().numpy())
    return np.concatenate(all_errors)


# ────────────────────────────────────────────────────────────────────────
# Isolation Forest training
# ────────────────────────────────────────────────────────────────────────
def train_iforest(cfg: dict):
    """Train IForest and return (detector, val_scores, test_scores, test_labels)."""
    ic = cfg["iforest"]
    wl = cfg["window"]["length"]
    stride = cfg["window"]["stride"]
    val_split = cfg["training"]["val_split"]

    # Load raw data (reuse download logic)
    dataset_dir = download_smd(cfg["data"]["raw_dir"])
    train_raw, test_raw, test_labels_raw = load_machine(dataset_dir, cfg["data"]["machine_id"])
    train_scaled, test_scaled, _ = scale_data(train_raw, test_raw)

    # Split
    split_idx = int(len(train_scaled) * (1 - val_split))
    train_data = train_scaled[:split_idx]
    val_data = train_scaled[split_idx:]

    # Windows
    train_windows = create_windows(train_data, wl, stride)
    val_windows = create_windows(val_data, wl, stride)
    test_windows = create_windows(test_scaled, wl, stride)

    from src.data import create_window_labels
    test_labels = create_window_labels(test_labels_raw, wl, stride)

    # Fit & score
    detector = IForestDetector(
        n_estimators=ic["n_estimators"],
        contamination=ic["contamination"],
        feature_mode=ic["feature_mode"],
        random_state=cfg["seed"],
    )
    detector.fit(train_windows)
    val_scores = detector.score(val_windows)
    test_scores = detector.score(test_windows)

    return detector, val_scores, test_scores, test_labels


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train anomaly-detection models on SMD")
    parser.add_argument(
        "--model",
        type=str,
        choices=["lstm_ae", "iforest", "all"],
        default="all",
        help="Which model to train (default: all)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs (useful for smoke tests)",
    )
    args = parser.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    set_seed(cfg["seed"])
    device = get_device()

    results_dir = cfg["evaluation"]["results_dir"]
    pct = cfg["evaluation"]["threshold_percentile"]
    all_results = []

    # ── Isolation Forest ────────────────────────────────────────────
    if args.model in ("iforest", "all"):
        logger.info("═══ Isolation Forest ═══")
        t0 = time.time()
        _, val_scores_if, test_scores_if, test_labels_if = train_iforest(cfg)
        thr_if = select_threshold(val_scores_if, pct)
        metrics_if = compute_metrics(test_scores_if, test_labels_if, thr_if, "Isolation Forest")
        metrics_if["train_time_s"] = round(time.time() - t0, 2)
        all_results.append(metrics_if)
        plot_scores(test_scores_if, test_labels_if, thr_if, "Isolation Forest", results_dir)

    # ── LSTM Autoencoder ────────────────────────────────────────────
    if args.model in ("lstm_ae", "all"):
        logger.info("═══ LSTM Autoencoder ═══")
        t0 = time.time()
        _, val_scores_ae, test_scores_ae, test_labels_ae = train_lstm_ae(cfg, device)
        thr_ae = select_threshold(val_scores_ae, pct)
        metrics_ae = compute_metrics(test_scores_ae, test_labels_ae, thr_ae, "LSTM-AE")
        metrics_ae["train_time_s"] = round(time.time() - t0, 2)
        all_results.append(metrics_ae)
        plot_scores(test_scores_ae, test_labels_ae, thr_ae, "LSTM-AE", results_dir)

    # ── Report ──────────────────────────────────────────────────────
    print_comparison_table(all_results)
    save_results(all_results, results_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
