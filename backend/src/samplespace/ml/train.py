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
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from sqlalchemy import select
from torch.utils.data import DataLoader

from samplespace.core.paths import CHECKPOINTS_DIR, RUNS_DIR, SAMPLES_DIR
from samplespace.dependencies.db import get_async_sqlalchemy_session
from samplespace.ml.dataset import LABEL_TO_IDX, SampleDataset, scan_samples
from samplespace.ml.model import SampleCNN
from samplespace.models.sample import Sample
from samplespace.schemas.sample_type import SAMPLE_TYPES
from samplespace.services.sample import find_audio_file

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def _fetch_samples_from_db() -> list[tuple[Path, int]] | None:
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
    return asyncio.run(_fetch_samples_from_db())


def _get_device() -> torch.device:
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


def _apply_mixup(
    spectrograms: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply mixup augmentation. Returns mixed spectrograms and soft label vectors."""
    batch_size = spectrograms.size(0)

    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    lam = max(lam, 1.0 - lam)  # Ensure primary sample dominates

    perm = torch.randperm(batch_size, device=spectrograms.device)

    mixed_specs = lam * spectrograms + (1.0 - lam) * spectrograms[perm]

    labels_onehot = torch.zeros(batch_size, num_classes, device=labels.device)
    labels_onehot.scatter_(1, labels.unsqueeze(1), 1.0)
    mixed_labels = lam * labels_onehot + (1.0 - lam) * labels_onehot[perm]

    return mixed_specs, mixed_labels


def _soft_cross_entropy(logits: torch.Tensor, soft_labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy with soft (probability vector) targets."""
    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    loss: torch.Tensor = -(soft_labels * log_probs).sum(dim=1).mean()
    return loss


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
    use_tensorboard: bool = True,
    label_smoothing: float = 0.1,
    balance_classes: bool = True,
    mixup_alpha: float = 0.0,
) -> None:
    device = _get_device()
    logger.info(f"Training on {device}")

    # Load samples from the database (supports all sources: library, upload).
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

    label_counts = Counter(label for _, label in all_samples)
    for idx, count in sorted(label_counts.items()):
        logger.info(f"  {SAMPLE_TYPES[idx]}: {count} samples")

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(all_samples), generator=generator).tolist()
    val_size = max(1, int(len(all_samples) * val_split))
    train_size = len(all_samples) - val_size

    train_samples = [all_samples[i] for i in indices[:train_size]]
    val_samples = [all_samples[i] for i in indices[train_size:]]

    train_dataset = SampleDataset(augment=True, samples=train_samples)
    val_dataset = SampleDataset(augment=False, samples=val_samples)

    # Class-weighted sampling: oversample minority classes so each class is seen
    # equally often per epoch. Superior to class weights in the loss (doesn't
    # change gradient magnitudes, only sampling frequency).
    sampler = None
    if balance_classes:
        train_label_counts = Counter(label for _, label in train_samples)
        total_train = len(train_samples)
        class_weights = {
            label: total_train / (len(train_label_counts) * count) for label, count in train_label_counts.items()
        }
        sample_weights = [class_weights[label] for _, label in train_samples]
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=total_train,
            replacement=True,
        )
        min_count = min(train_label_counts.values())
        max_count = max(train_label_counts.values())
        logger.info(
            f"Class balancing enabled (min: {min_count}, max: {max_count}, ratio: {max_count / min_count:.1f}x)"
        )

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
        shuffle=(sampler is None),
        sampler=sampler,
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

    model = SampleCNN().to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {param_count:,}")

    classification_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    contrastive_loss = SupConLoss(temperature=temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

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
        f"warmup={warmup_epochs}, patience={patience}, amp={use_amp}, "
        f"label_smoothing={label_smoothing}, balance={balance_classes}, mixup_alpha={mixup_alpha}"
    )

    training_start = time.monotonic()
    best_epoch = 0
    best_val_acc = 0.0
    best_macro_f1 = 0.0

    for epoch in range(epochs):
        epoch_start = time.monotonic()
        model.train()
        train_loss_sum = 0.0
        train_cls_loss_sum = 0.0
        train_con_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        data_time = 0.0
        forward_time = 0.0
        backward_time = 0.0
        optimizer.zero_grad()

        batch_start = time.monotonic()
        for step_idx, (spectrograms, labels) in enumerate(train_loader):
            data_time += time.monotonic() - batch_start

            spectrograms = spectrograms.to(device)
            labels = labels.to(device)

            fwd_start = time.monotonic()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):  # type: ignore[attr-defined]
                if mixup_alpha > 0.0:
                    # Mixup: blend spectrograms with soft labels for classifier,
                    # but use original spectrograms for SupCon (needs hard labels).
                    mixed_specs, mixed_labels = _apply_mixup(spectrograms, labels, mixup_alpha, len(SAMPLE_TYPES))
                    logits, _ = model(mixed_specs)
                    cls_loss = _soft_cross_entropy(logits, mixed_labels)
                    _, embeddings = model(spectrograms)
                    con_loss = contrastive_loss(embeddings, labels)
                else:
                    logits, embeddings = model(spectrograms)
                    cls_loss = classification_loss(logits, labels)
                    con_loss = contrastive_loss(embeddings, labels)
                loss = (cls_loss + lambda_embed * con_loss) / grad_accum
            forward_time += time.monotonic() - fwd_start

            bwd_start = time.monotonic()
            scaler.scale(loss).backward()

            # Step optimizer every grad_accum iterations (or at end of epoch)
            if (step_idx + 1) % grad_accum == 0 or (step_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            backward_time += time.monotonic() - bwd_start

            # Track un-normalized loss for logging
            train_loss_sum += (cls_loss.item() + lambda_embed * con_loss.item()) * labels.size(0)
            train_cls_loss_sum += cls_loss.item() * labels.size(0)
            train_con_loss_sum += con_loss.item() * labels.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)
            batch_start = time.monotonic()

        train_loss = train_loss_sum / train_total
        train_cls = train_cls_loss_sum / train_total
        train_con = train_con_loss_sum / train_total
        train_acc = train_correct / train_total

        val_start = time.monotonic()
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

        val_time = time.monotonic() - val_start
        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        epoch_time = time.monotonic() - epoch_start
        elapsed = time.monotonic() - training_start
        elapsed_m, elapsed_s = int(elapsed // 60), int(elapsed % 60)

        # F1 scores — show non-zero sorted descending, count zeros
        f1_scores = _compute_per_class_f1(val_preds, val_labels, SAMPLE_TYPES)
        val_label_names = {SAMPLE_TYPES[l] for l in val_labels}
        active_f1 = {k: v for k, v in f1_scores.items() if v > 0 or k in val_label_names}
        macro_f1 = sum(active_f1.values()) / len(active_f1) if active_f1 else 0.0
        nonzero_f1 = {k: v for k, v in f1_scores.items() if v > 0}
        zero_count = len(active_f1) - len(nonzero_f1)

        # Check if this is the best epoch
        is_best = val_loss < best_val_loss
        best_marker = " \u2605 best" if is_best else ""

        # Epoch summary line
        logger.info(
            f"Epoch {epoch + 1:>{len(str(epochs))}}/{epochs} [{elapsed_m}m {elapsed_s:02d}s] | "
            f"train {train_loss:.4f} (CE {train_cls:.4f} + SupCon {train_con:.4f}) acc {train_acc:.1%} | "
            f"val {val_loss:.4f} acc {val_acc:.1%} | "
            f"LR {current_lr:.2e}{best_marker}"
        )

        # Timing with throughput
        data_pct = data_time / epoch_time * 100 if epoch_time > 0 else 0
        samp_per_sec = train_total / data_time if data_time > 0 else 0
        logger.info(
            f"  Timing: {epoch_time:.1f}s | "
            f"data {data_time:.1f}s ({data_pct:.0f}%, {samp_per_sec:.0f} samp/s) | "
            f"fwd {forward_time:.1f}s | bwd {backward_time:.1f}s | val {val_time:.1f}s"
        )

        # F1 — non-zero classes sorted descending, with zero count
        if active_f1:
            sorted_f1 = sorted(nonzero_f1.items(), key=lambda x: x[1], reverse=True)
            f1_parts = [f"{k} {v:.3f}" for k, v in sorted_f1[:5]]
            f1_str = ", ".join(f1_parts)
            zero_suffix = f" ({zero_count} classes at 0)" if zero_count > 0 else ""
            extra = f" +{len(sorted_f1) - 5} more" if len(sorted_f1) > 5 else ""
            logger.info(f"  F1 macro {macro_f1:.3f}: {f1_str}{extra}{zero_suffix}")

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

        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            best_val_acc = val_acc
            best_macro_f1 = macro_f1
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
                        "label_smoothing": label_smoothing,
                        "balance_classes": balance_classes,
                        "mixup_alpha": mixup_alpha,
                    },
                    "class_distribution": dict(label_counts),
                    "train_size": train_size,
                    "val_size": val_size,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)")
                break

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
    total_time = time.monotonic() - training_start
    total_m, total_s = int(total_time // 60), int(total_time % 60)
    completed_epochs = epoch + 1  # noqa: F821 (loop variable from training loop)
    avg_epoch = total_time / completed_epochs
    logger.info(
        f"Training complete in {total_m}m {total_s:02d}s ({completed_epochs} epochs, avg {avg_epoch:.1f}s/epoch)"
    )
    logger.info(
        f"  Best: epoch {best_epoch + 1}, val_loss {best_val_loss:.4f}, "
        f"val_acc {best_val_acc:.1%}, F1 {best_macro_f1:.3f}"
    )
    logger.info(f"  Checkpoints: {CHECKPOINTS_DIR}")


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
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard logging")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing factor (0 = disabled)")
    parser.add_argument("--no-balance", action="store_true", help="Disable class-weighted sampling")
    parser.add_argument("--mixup-alpha", type=float, default=0.0, help="Mixup alpha (0 = disabled, try 0.2)")
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
        use_tensorboard=not args.no_tensorboard,
        label_smoothing=args.label_smoothing,
        balance_classes=not args.no_balance,
        mixup_alpha=args.mixup_alpha,
    )


if __name__ == "__main__":
    main()
