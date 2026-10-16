"""
Training and Evaluation Pipeline for Lottery Ticket Experiments.

Enforces:
1. Fixed training recipe (optimizer, learning rate schedule, weight decay).
2. Continuous mask projection (w <- w * m) after every optimizer step.
3. Strict post-epoch mask assertion.
4. Logging of learning speed (epoch to reach best val acc, AUC-val).
5. Support for late-rewinding checkpoint extraction.
"""

import copy
import time
from typing import Dict, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from prune import apply_masks_, verify_mask_integrity

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str = "cuda"
) -> Tuple[float, float]:
    """Computes average loss and top-1 accuracy on an evaluation dataset."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total_samples += inputs.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, accuracy


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    masks: Dict[str, torch.Tensor],
    device: str = "cuda"
) -> Tuple[float, float]:
    """Runs a single training epoch with continuous mask enforcement."""
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()

        # CRUCIAL: Project weights back to mask space immediately after optimizer step
        apply_masks_(model, masks)

        total_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total_samples += inputs.size(0)

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    accuracy = correct / total_samples if total_samples > 0 else 0.0
    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    masks: Dict[str, torch.Tensor],
    epochs: int = 25,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    optimizer_type: str = "adam",
    scheduler_type: str = "cosine",
    device: str = "cuda",
    verbose: bool = False,
    rewind_epoch: int = 0
) -> Dict[str, Any]:
    """
    Trains a network using the fixed recipe while strictly maintaining the binary mask.

    Returns:
    - History of train/val loss & accuracy per epoch
    - Best validation accuracy
    - Epoch of best validation accuracy
    - AUC of validation accuracy
    - Test accuracy (evaluated using the best model weights)
    - Rewind state dict (at rewind_epoch, if requested)
    - Total training time
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Configure optimizer
    if optimizer_type.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    # Configure scheduler
    if scheduler_type.lower() == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = None

    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": []
    }

    best_val_acc = -1.0
    best_epoch = 0
    best_model_weights = None
    rewind_weights = None

    start_time = time.time()

    # Apply masks before training begins
    apply_masks_(model, masks)
    verify_mask_integrity(model, masks)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_epoch(
            model, dataloaders["train"], optimizer, criterion, masks, device=device
        )
        val_loss, val_acc = evaluate(
            model, dataloaders["val"], criterion, device=device
        )

        if scheduler is not None:
            scheduler.step()

        # Strict assertion at epoch boundary
        verify_mask_integrity(model, masks)

        current_lr = optimizer.param_groups[0]["lr"]
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_weights = copy.deepcopy(model.state_dict())

        # Capture weights for late rewinding if requested
        if epoch == rewind_epoch:
            rewind_weights = copy.deepcopy(model.state_dict())

        epoch_duration = time.time() - epoch_start
        if verbose and (epoch % 5 == 0 or epoch == epochs):
            print(
                f"Epoch {epoch:2d}/{epochs:2d} | "
                f"Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% (Best: {best_val_acc*100:.2f}% @ Ep {best_epoch}) | "
                f"Time: {epoch_duration:.1f}s"
            )

    total_training_time = time.time() - start_time

    # Calculate AUC of validation accuracy (normalized between 0 and 1)
    auc_val_acc = float(np.mean(history["val_acc"])) if history["val_acc"] else 0.0

    # Validation accuracy at fixed early checkpoints (avoiding cosine decay schedule artifacts)
    val_acc_epoch_5 = history["val_acc"][4] if len(history["val_acc"]) >= 5 else (history["val_acc"][-1] if history["val_acc"] else 0.0)
    val_acc_epoch_10 = history["val_acc"][9] if len(history["val_acc"]) >= 10 else (history["val_acc"][-1] if history["val_acc"] else 0.0)

    # Evaluate test accuracy using best weights on validation set
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        apply_masks_(model, masks)
        verify_mask_integrity(model, masks)

    test_loss, test_acc = evaluate(
        model, dataloaders["test"], criterion, device=device
    )

    return {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "auc_val_acc": auc_val_acc,
        "val_acc_epoch_5": val_acc_epoch_5,
        "val_acc_epoch_10": val_acc_epoch_10,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "history": history,
        "training_time": total_training_time,
        "best_model_weights": best_model_weights,
        "rewind_weights": rewind_weights
    }
