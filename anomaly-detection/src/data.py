"""
data.py — Download SMD, load / scale / window, and build DataLoaders.

The Server Machine Dataset (SMD) lives at
https://github.com/NetManAIOps/OmniAnomaly  (ServerMachineDataset/).
Each machine has 38 metrics; labels are binary per-time-step.
"""

import io
import logging
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# ── GitHub archive URL for the dataset ──────────────────────────────────
_REPO_ZIP_URL = (
    "https://github.com/NetManAIOps/OmniAnomaly/archive/refs/heads/master.zip"
)
_DATASET_SUBDIR = "OmniAnomaly-master/ServerMachineDataset"


# ────────────────────────────────────────────────────────────────────────
# Download helpers
# ────────────────────────────────────────────────────────────────────────
def download_smd(raw_dir: str | Path) -> Path:
    """
    Download and extract the SMD dataset into *raw_dir*.
    Returns the path to the extracted ``ServerMachineDataset/`` folder.

    If the download fails, prints clear manual-download instructions.
    """
    raw_dir = Path(raw_dir)
    dataset_dir = raw_dir / "ServerMachineDataset"

    # Already downloaded?
    if dataset_dir.exists() and (dataset_dir / "train").exists():
        logger.info(f"SMD already present at {dataset_dir}")
        return dataset_dir

    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "OmniAnomaly.zip"

    try:
        logger.info(f"Downloading SMD from {_REPO_ZIP_URL}  (≈ 60 MB) …")
        urlretrieve(_REPO_ZIP_URL, zip_path)
    except Exception as exc:
        _print_manual_instructions(raw_dir, exc)
        raise

    # Extract only the ServerMachineDataset subtree
    logger.info("Extracting ServerMachineDataset …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            if member.startswith(_DATASET_SUBDIR):
                # Strip the repo-prefix so files land under raw_dir/ServerMachineDataset/
                rel = member[len("OmniAnomaly-master/"):]
                target = raw_dir / rel
                if member.endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zf.read(member))

    zip_path.unlink(missing_ok=True)  # clean up zip
    logger.info(f"SMD ready at {dataset_dir}")
    return dataset_dir


def _print_manual_instructions(raw_dir: Path, exc: Exception) -> None:
    msg = f"""
╔══════════════════════════════════════════════════════════════════════╗
║  Automatic download failed: {exc!r}
║
║  Manual steps:
║   1. Go to https://github.com/NetManAIOps/OmniAnomaly
║   2. Click  Code → Download ZIP.
║   3. Unzip  ServerMachineDataset/  into:
║        {raw_dir.resolve()}/ServerMachineDataset/
║      so the layout is:
║        {raw_dir.resolve()}/ServerMachineDataset/train/
║        {raw_dir.resolve()}/ServerMachineDataset/test/
║        {raw_dir.resolve()}/ServerMachineDataset/test_label/
║   4. Re-run the script.
╚══════════════════════════════════════════════════════════════════════╝"""
    print(msg)


# ────────────────────────────────────────────────────────────────────────
# Loading & scaling
# ────────────────────────────────────────────────────────────────────────
def load_machine(
    dataset_dir: str | Path,
    machine_id: str = "machine-1-1",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load raw CSVs for a single machine.

    Returns
    -------
    train : (T_train, 38)
    test  : (T_test, 38)
    labels: (T_test,)   binary
    """
    dataset_dir = Path(dataset_dir)
    train = np.loadtxt(dataset_dir / "train" / f"{machine_id}.txt", delimiter=",")
    test = np.loadtxt(dataset_dir / "test" / f"{machine_id}.txt", delimiter=",")
    labels = np.loadtxt(
        dataset_dir / "test_label" / f"{machine_id}.txt", delimiter=","
    ).astype(int)
    logger.info(
        f"Loaded {machine_id}: train {train.shape}, test {test.shape}, "
        f"anomalies {labels.sum()}/{len(labels)}"
    )
    return train, test, labels


def scale_data(
    train: np.ndarray, test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit StandardScaler on *train*; transform both splits."""
    scaler = StandardScaler(with_mean=True, with_std=True)
    train_s = scaler.fit_transform(train)
    test_s = scaler.transform(test)
    return train_s, test_s, scaler


# ────────────────────────────────────────────────────────────────────────
# Sliding-window utilities
# ────────────────────────────────────────────────────────────────────────
def create_windows(
    data: np.ndarray,
    window_len: int = 100,
    stride: int = 1,
) -> np.ndarray:
    """
    Convert (T, D) array into (N, window_len, D) sliding windows.
    """
    n_windows = (len(data) - window_len) // stride + 1
    indices = np.arange(window_len)[None, :] + stride * np.arange(n_windows)[:, None]
    return data[indices]  # (N, window_len, D)


def create_window_labels(
    labels: np.ndarray,
    window_len: int = 100,
    stride: int = 1,
) -> np.ndarray:
    """Label = 1 if *any* point in the window is anomalous."""
    windows = create_windows(labels.reshape(-1, 1), window_len, stride)
    return (windows.squeeze(-1).max(axis=1) > 0).astype(int)


# ────────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ────────────────────────────────────────────────────────────────────────
class WindowDataset(Dataset):
    """Simple dataset wrapping (N, W, D) windows and optional labels."""

    def __init__(self, windows: np.ndarray, labels: np.ndarray | None = None):
        self.windows = torch.as_tensor(windows, dtype=torch.float32)
        self.labels = (
            torch.as_tensor(labels, dtype=torch.long) if labels is not None else None
        )

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        x = self.windows[idx]
        if self.labels is not None:
            return x, self.labels[idx]
        return x


# ────────────────────────────────────────────────────────────────────────
# High-level data preparation
# ────────────────────────────────────────────────────────────────────────
def prepare_dataloaders(
    cfg: dict,
) -> tuple[DataLoader, DataLoader, DataLoader, np.ndarray]:
    """
    End-to-end pipeline: download → load → scale → window → DataLoaders.

    Returns
    -------
    train_loader : DataLoader  (normal windows for training)
    val_loader   : DataLoader  (normal windows for validation / threshold)
    test_loader  : DataLoader  (test windows with labels)
    test_labels  : np.ndarray  (window-level labels for evaluation)
    """
    raw_dir = cfg["data"]["raw_dir"]
    machine_id = cfg["data"]["machine_id"]
    window_len = cfg["window"]["length"]
    stride = cfg["window"]["stride"]
    val_split = cfg["training"]["val_split"]
    batch_size = cfg["training"]["batch_size"]

    # 1. Download / locate dataset
    dataset_dir = download_smd(raw_dir)

    # 2. Load & scale
    train_raw, test_raw, test_labels_raw = load_machine(dataset_dir, machine_id)
    train_scaled, test_scaled, _ = scale_data(train_raw, test_raw)

    # 3. Train/val split (last val_split fraction of train held out)
    split_idx = int(len(train_scaled) * (1 - val_split))
    train_data = train_scaled[:split_idx]
    val_data = train_scaled[split_idx:]

    # 4. Sliding windows
    train_windows = create_windows(train_data, window_len, stride)
    val_windows = create_windows(val_data, window_len, stride)
    test_windows = create_windows(test_scaled, window_len, stride)
    test_window_labels = create_window_labels(test_labels_raw, window_len, stride)

    logger.info(
        f"Windows — train: {train_windows.shape}, val: {val_windows.shape}, "
        f"test: {test_windows.shape}"
    )

    # 5. DataLoaders
    train_loader = DataLoader(
        WindowDataset(train_windows),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        WindowDataset(val_windows),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        WindowDataset(test_windows, test_window_labels),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader, test_window_labels
