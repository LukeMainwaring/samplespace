"""Training script for the dual-head SampleCNN.

Trains with a combined loss:
  - Cross-entropy for the classification head
  - Supervised contrastive loss (SupCon) for the embedding head

Features:
  - Cosine annealing LR with linear warmup
  - Mixed precision training (CUDA/MPS)
  - Gradient accumulation for larger effective batch sizes
  - Early stopping with configurable patience
  - TensorBoard logging (loss curves, LR, per-class F1, embedding projector)

Usage:
    uv run train-cnn
    uv run train-cnn --epochs 50 --lr 0.0005 --batch-size 32 --grad-accum 2
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from sqlalchemy import select
from torch.utils.data import DataLoader

from samplespace.dependencies.db import get_async_sqlalchemy_session
from samplespace.ml.dataset import LABEL_TO_IDX, SampleDataset, scan_samples
from samplespace.ml.model import SampleCNN
from samplespace.models.sample import Sample
from samplespace.schemas.sample_type import SAMPLE_TYPES
from samplespace.services.sample import find_audio_file

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SAMPLES_DIR = Path(__file__).parent.parent.parent.parent.parent / "data" / "samples"
CHECKPOINTS_DIR = Path(__file__).parent.parent.parent.parent.parent / "data" / "checkpoints"
RUNS_DIR = Path(__file__).parent.parent.parent.parent.parent / "data" / "runs"


async def _fetch_samples_from_db() -> list[tuple[Path, int]] | None:
    """Query the database for all samples with a known sample_type and resolve their audio paths."""
    try:
        async with get_async_sqlalchemy_session() as db:
            stmt = select(Sample).where(Sample.sample_type.in_(SAMPLE_TYPES))
            result = await db.execute(stmt)
            db_samples = result.scalars().all()
    except Exception:
        logger.warning("Could not connect to database, falling back to directory scan", exc_info=True)
        return None

    samples: list[tuple[Path, int]] = []
    for s in db_samples:
        label_idx = LABEL_TO_IDX.get(s.sample_type)  # type: ignore[arg-type]
        if label_idx is None:
            continue
        audio_path = find_audio_file(s)
        if audio_path is None:
            logger.warning(f"Audio file not found for sample {s.id}: {s.filename}")
            continue
        samples.append((audio_path, label_idx))

    return sorted(samples, key=lambda x: (x[1], str(x[0])))


def _load_samples_from_db() -> list[tuple[Path, int]] | None:
    """Sync wrapper for DB sample loading."""
    return asyncio.run(_fetch_samples_from_db())


def _get_device() -> torch.device:
    """Select the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 64,
    val_split: float = 0.2,
    lambda_embed: float = 0.5,
    temperature: float = 0.07,
    grad_accum: int = 1,
    patience: int = 15,
    warmup_epochs: int = 5,
    num_workers: int = 0,
    expensive_augment: bool = False,
    use_tensorboard: bool = True,
) -> None:
    """Train the SampleCNN model with combined classification + SupCon loss."""
    device = _get_device()
    logger.info(f"Training on {device}")

    # Load samples from the database (supports all sources: local, splice, upload).
    # Falls back to directory scan if the database is unavailable.
    # Using separate SampleDataset instances (not random_split) ensures validation
    # data is never augmented — critical for reliable model selection and LR scheduling.
    all_samples = _load_samples_from_db()
    if all_samples:
        logger.info(f"Loaded {len(all_samples)} samples from database")
    else:
        logger.info("Falling back to directory scan")
        all_samples = scan_samples(SAMPLES_DIR)
    if not all_samples:
        logger.error("No samples found")
        return

    # Log class distribution
    label_counts = Counter(label for _, label in all_samples)
    for idx, count in sorted(label_counts.items()):
        logger.info(f"  {SAMPLE_TYPES[idx]}: {count} samples")

    # Deterministic shuffle then split
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(all_samples), generator=generator).tolist()
    val_size = max(1, int(len(all_samples) * val_split))
    train_size = len(all_samples) - val_size

    train_samples = [all_samples[i] for i in indices[:train_size]]
    val_samples = [all_samples[i] for i in indices[train_size:]]

    train_dataset = SampleDataset(augment=True, samples=train_samples, expensive_augment=expensive_augment)
    val_dataset = SampleDataset(augment=False, samples=val_samples)

    # Clamp batch_size to training set size to avoid empty epochs with drop_last=True
    effective_batch_size = min(batch_size, train_size)
    if effective_batch_size < batch_size:
        logger.warning(
            f"Batch size {batch_size} exceeds training set ({train_size}), clamping to {effective_batch_size}"
        )

    # num_workers > 0 parallelizes data loading and augmentation.
    # "forkserver" avoids fork-related deadlocks on macOS/MPS.
    # persistent_workers avoids respawning processes each epoch.
    persist = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        persistent_workers=persist,
        multiprocessing_context="forkserver" if num_workers > 0 else None,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persist,
        multiprocessing_context="forkserver" if num_workers > 0 else None,
        pin_memory=False,
    )

    logger.info(f"Train: {train_size}, Val: {val_size}, Classes: {len(SAMPLE_TYPES)}")

    # Model, losses, optimizer
    model = SampleCNN().to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {param_count:,}")

    classification_loss = nn.CrossEntropyLoss()
    contrastive_loss = SupConLoss(temperature=temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Cosine annealing with linear warmup
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )

    # Mixed precision: CUDA only (MPS autocast has limited dtype coverage and no GradScaler)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)  # type: ignore[attr-defined]

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # TensorBoard
    writer = None
    if use_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter

            RUNS_DIR.mkdir(parents=True, exist_ok=True)
            run_name = f"cnn_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            writer = SummaryWriter(log_dir=str(RUNS_DIR / run_name))
            logger.info(f"TensorBoard logging to {RUNS_DIR / run_name}")
        except ImportError:
            logger.warning("tensorboard not installed, skipping logging (install with: uv sync --directory backend)")

    logger.info(
        f"Hyperparameters: epochs={epochs}, lr={lr}, batch_size={batch_size}, "
        f"grad_accum={grad_accum}, lambda_embed={lambda_embed}, temperature={temperature}, "
        f"warmup={warmup_epochs}, patience={patience}, amp={use_amp}"
    )

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss_sum = 0.0
        train_cls_loss_sum = 0.0
        train_con_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        optimizer.zero_grad()

        for step_idx, (spectrograms, labels) in enumerate(train_loader):
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):  # type: ignore[attr-defined]
                logits, embeddings = model(spectrograms)
                cls_loss = classification_loss(logits, labels)
                con_loss = contrastive_loss(embeddings, labels)
                loss = (cls_loss + lambda_embed * con_loss) / grad_accum

            scaler.scale(loss).backward()

            # Step optimizer every grad_accum iterations (or at end of epoch)
            if (step_idx + 1) % grad_accum == 0 or (step_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Track un-normalized loss for logging
            train_loss_sum += (cls_loss.item() + lambda_embed * con_loss.item()) * labels.size(0)
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

                with torch.amp.autocast(device_type=device.type, enabled=use_amp):  # type: ignore[attr-defined]
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
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        logger.info(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train loss: {train_loss:.4f} (cls: {train_cls:.4f}, con: {train_con:.4f}), acc: {train_acc:.2%} | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.2%} | "
            f"LR: {current_lr:.2e}"
        )

        # Per-class F1 scores
        f1_scores = _compute_per_class_f1(val_preds, val_labels, SAMPLE_TYPES)
        val_label_names = {SAMPLE_TYPES[l] for l in val_labels}
        active_f1 = {k: v for k, v in f1_scores.items() if v > 0 or k in val_label_names}
        macro_f1 = sum(active_f1.values()) / len(active_f1) if active_f1 else 0.0
        if active_f1:
            f1_str = ", ".join(f"{k}: {v:.3f}" for k, v in active_f1.items())
            logger.info(f"  Val F1 (macro: {macro_f1:.3f}): {f1_str}")

        # TensorBoard logging
        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/train_cls", train_cls, epoch)
            writer.add_scalar("Loss/train_con", train_con, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)
            writer.add_scalar("LR", current_lr, epoch)
            writer.add_scalar("F1/macro", macro_f1, epoch)
            for class_name, f1 in f1_scores.items():
                writer.add_scalar(f"F1/{class_name}", f1, epoch)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            checkpoint_path = CHECKPOINTS_DIR / "sample_cnn_best.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "sample_types": SAMPLE_TYPES,
                    "lambda_embed": lambda_embed,
                    "hyperparameters": {
                        "lr": lr,
                        "batch_size": batch_size,
                        "grad_accum": grad_accum,
                        "temperature": temperature,
                        "warmup_epochs": warmup_epochs,
                        "epochs": epochs,
                    },
                    "class_distribution": dict(label_counts),
                    "train_size": train_size,
                    "val_size": val_size,
                },
                checkpoint_path,
            )
            logger.info(f"  Saved best model (val_loss: {val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
                break

    # Embedding visualization in TensorBoard
    if writer:
        model.eval()
        all_embeddings: list[torch.Tensor] = []
        all_label_indices: list[int] = []
        with torch.no_grad():
            for spectrograms, labels in val_loader:
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):  # type: ignore[attr-defined]
                    _, embeddings = model(spectrograms.to(device))
                all_embeddings.append(embeddings.cpu())
                all_label_indices.extend(labels.tolist())
        if all_embeddings:
            embeddings_tensor = torch.cat(all_embeddings)
            label_names = [SAMPLE_TYPES[l] for l in all_label_indices]
            writer.add_embedding(embeddings_tensor, metadata=label_names, tag="val_embeddings")
        writer.close()

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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lambda-embed", type=float, default=0.5, help="Weight for embedding (SupCon) loss")
    parser.add_argument("--temperature", type=float, default=0.07, help="SupCon loss temperature")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience (epochs)")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Linear LR warmup epochs")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes (0 = main thread)")
    parser.add_argument(
        "--expensive-augment", action="store_true", help="Enable pitch shift/time stretch (slow, deadlocks on macOS)"
    )
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard logging")
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        lambda_embed=args.lambda_embed,
        temperature=args.temperature,
        grad_accum=args.grad_accum,
        patience=args.patience,
        warmup_epochs=args.warmup_epochs,
        num_workers=args.num_workers,
        expensive_augment=args.expensive_augment,
        use_tensorboard=not args.no_tensorboard,
    )


if __name__ == "__main__":
    main()
