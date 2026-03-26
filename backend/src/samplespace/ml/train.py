"""Training script for the dual-head SampleCNN.

Trains with a combined loss:
  - Cross-entropy for the classification head
  - Supervised contrastive loss (SupCon) for the embedding head

Usage:
    python -m samplespace.ml.train
    python -m samplespace.ml.train --epochs 50 --lr 0.0005
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
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


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

    Pulls together embeddings of samples from the same class while pushing apart
    embeddings from different classes. Uses all in-batch positives and negatives
    simultaneously, making it more sample-efficient than triplet loss.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute SupCon loss over a batch of L2-normalized embeddings.

        Args:
            embeddings: (batch_size, embed_dim) — must be L2-normalized.
            labels: (batch_size,) — integer class labels.

        Returns:
            Scalar loss averaged over all anchor-positive pairs in the batch.
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]

        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Pairwise cosine similarity (embeddings are already L2-normalized)
        sim_matrix = torch.mm(embeddings, embeddings.T) / self.temperature

        # Mask out self-similarity (diagonal)
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)

        # Positive mask: same class, different sample
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)
        positive_mask = labels_eq & ~self_mask

        # If no positives exist in the batch, return zero loss
        if not positive_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Numerical stability: subtract max from each row before exp
        sim_max, _ = sim_matrix.detach().max(dim=1, keepdim=True)
        sim_matrix = sim_matrix - sim_max

        # Log-sum-exp over all non-self entries (denominator)
        exp_sim = torch.exp(sim_matrix)
        exp_sim = exp_sim.masked_fill(self_mask, 0.0)
        log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # Log-prob for each positive pair
        log_prob = sim_matrix - log_denom

        # Average over positives for each anchor, then average over anchors
        positive_count = positive_mask.sum(dim=1).clamp(min=1)
        mean_log_prob = (log_prob * positive_mask.float()).sum(dim=1) / positive_count

        # Only average over anchors that have at least one positive
        has_positives = positive_mask.any(dim=1)
        loss: torch.Tensor = -mean_log_prob[has_positives].mean()
        return loss


def _compute_per_class_f1(
    all_preds: list[int],
    all_labels: list[int],
    class_names: list[str],
) -> dict[str, float]:
    """Compute per-class F1 scores from prediction and label lists."""
    num_classes = len(class_names)
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes

    for pred, label in zip(all_preds, all_labels):
        if pred == label:
            tp[label] += 1
        else:
            fp[pred] += 1
            fn[label] += 1

    f1_scores: dict[str, float] = {}
    for i, name in enumerate(class_names):
        precision = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0.0
        recall = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores[name] = round(f1, 3)

    return f1_scores


def train(
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 16,
    val_split: float = 0.2,
    lambda_embed: float = 0.5,
) -> None:
    """Train the SampleCNN model with combined classification + SupCon loss."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on {device}")

    # Load dataset
    dataset = SampleDataset(SAMPLES_DIR, augment=True)
    if len(dataset) == 0:
        logger.error(f"No samples found in {SAMPLES_DIR}")
        return

    # Log class distribution
    label_counts = Counter(label for _, label in dataset.samples)
    for idx, count in sorted(label_counts.items()):
        logger.info(f"  {SAMPLE_TYPES[idx]}: {count} samples")

    # Train/val split
    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"Train: {train_size}, Val: {val_size}, Classes: {len(SAMPLE_TYPES)}")

    # Model, losses, optimizer
    model = SampleCNN().to(device)
    classification_loss = nn.CrossEntropyLoss()
    contrastive_loss = SupConLoss(temperature=0.07)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Lambda (embedding loss weight): {lambda_embed}")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss_sum = 0.0
        train_cls_loss_sum = 0.0
        train_con_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for spectrograms, labels in train_loader:
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits, embeddings = model(spectrograms)

            cls_loss = classification_loss(logits, labels)
            con_loss = contrastive_loss(embeddings, labels)
            loss = cls_loss + lambda_embed * con_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item() * labels.size(0)
            train_cls_loss_sum += cls_loss.item() * labels.size(0)
            train_con_loss_sum += con_loss.item() * labels.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        train_loss = train_loss_sum / train_total
        train_cls = train_cls_loss_sum / train_total
        train_con = train_con_loss_sum / train_total
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        val_preds: list[int] = []
        val_labels: list[int] = []

        with torch.no_grad():
            for spectrograms, labels in val_loader:
                spectrograms = spectrograms.to(device)
                labels = labels.to(device)

                logits, embeddings = model(spectrograms)
                cls_loss = classification_loss(logits, labels)
                con_loss = contrastive_loss(embeddings, labels)
                loss = cls_loss + lambda_embed * con_loss

                val_loss_sum += loss.item() * labels.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                val_preds.extend(preds.cpu().tolist())
                val_labels.extend(labels.cpu().tolist())

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total
        scheduler.step(val_loss)

        logger.info(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train loss: {train_loss:.4f} (cls: {train_cls:.4f}, con: {train_con:.4f}), acc: {train_acc:.2%} | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.2%}"
        )

        # Per-class F1 scores
        f1_scores = _compute_per_class_f1(val_preds, val_labels, SAMPLE_TYPES)
        active_f1 = {k: v for k, v in f1_scores.items() if v > 0 or k in [SAMPLE_TYPES[l] for l in val_labels]}
        if active_f1:
            f1_str = ", ".join(f"{k}: {v:.3f}" for k, v in active_f1.items())
            macro_f1 = sum(active_f1.values()) / len(active_f1) if active_f1 else 0.0
            logger.info(f"  Val F1 (macro: {macro_f1:.3f}): {f1_str}")

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
                    "lambda_embed": lambda_embed,
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
            "lambda_embed": lambda_embed,
        },
        final_path,
    )
    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    logger.info(f"Checkpoints saved to {CHECKPOINTS_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the SampleCNN model")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lambda-embed", type=float, default=0.5, help="Weight for embedding (SupCon) loss")
    args = parser.parse_args()

    train(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, lambda_embed=args.lambda_embed)


if __name__ == "__main__":
    main()
