"""Training script for the dual-head SampleCNN.

Trains with a combined loss: cross-entropy for classification + cosine embedding
loss for the embedding head.

Usage:
    python -m samplespace.ml.train
    python -m samplespace.ml.train --epochs 50 --lr 0.0005
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from samplespace.ml.dataset import SAMPLE_TYPES, SampleDataset
from samplespace.ml.model import SampleCNN

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SAMPLES_DIR = Path(__file__).parent.parent.parent.parent.parent / "data" / "samples"
CHECKPOINTS_DIR = Path(__file__).parent.parent.parent.parent.parent / "data" / "checkpoints"


def train(
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 8,
    val_split: float = 0.2,
) -> None:
    """Train the SampleCNN model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on {device}")

    # Load dataset
    dataset = SampleDataset(SAMPLES_DIR, augment=True)
    if len(dataset) == 0:
        logger.error(f"No samples found in {SAMPLES_DIR}")
        return

    # Train/val split
    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"Train: {train_size}, Val: {val_size}, Classes: {len(SAMPLE_TYPES)}")

    # Model, loss, optimizer
    model = SampleCNN().to(device)
    classification_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for spectrograms, labels in train_loader:
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, _embeddings = model(spectrograms)

            loss = classification_loss(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * labels.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for spectrograms, labels in val_loader:
                spectrograms = spectrograms.to(device)
                labels = labels.to(device)

                logits, _embeddings = model(spectrograms)
                loss = classification_loss(logits, labels)

                val_loss_sum += loss.item() * labels.size(0)
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total
        scheduler.step(val_loss)

        logger.info(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.2%} | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.2%}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = CHECKPOINTS_DIR / "sample_cnn_best.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "sample_types": SAMPLE_TYPES,
                },
                checkpoint_path,
            )
            logger.info(f"  Saved best model (val_loss: {val_loss:.4f})")

    # Save final model
    final_path = CHECKPOINTS_DIR / "sample_cnn_final.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epochs,
            "sample_types": SAMPLE_TYPES,
        },
        final_path,
    )
    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    logger.info(f"Checkpoints saved to {CHECKPOINTS_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SampleCNN model")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    train(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
