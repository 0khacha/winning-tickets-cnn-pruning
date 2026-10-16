"""
Configuration file for Lottery Ticket Hypothesis experiments on Small CNNs.
Centralizes all hyperparameters, dataset settings, pruning schedules, and runtime flags.
"""

from dataclasses import dataclass, field
from typing import List
import random
import numpy as np
import torch

def set_seed(seed: int) -> None:
    """Set random seeds across random, numpy, and torch for strict determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

@dataclass
class ExperimentConfig:
    # --- Reproducibility & Seeds ---
    seeds: List[int] = field(default_factory=lambda: [42, 1337, 2026])
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 2
    
    # --- Datasets ---
    data_dir: str = "./data"
    source_dataset: str = "cifar10"
    target_dataset: str = "svhn"
    val_split_ratio: float = 0.10  # 10% of train set reserved for validation
    batch_size: int = 128
    
    # --- Model Architecture ---
    model_name: str = "conv4"  # Options: 'conv4' (Model A), 'resnet9' (Model B)
    in_channels: int = 3
    num_classes_source: int = 10
    num_classes_target: int = 10
    
    # --- Training Hyperparameters (Fixed Recipe across all variants) ---
    epochs: int = 25
    optimizer: str = "adam"
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    lr_scheduler: str = "cosine"  # 'cosine' or 'none'
    
    # --- Pruning & IMP Protocol ---
    prune_rate: float = 0.20  # Prune lowest 20% of remaining weights each round
    num_prune_rounds: int = 12  # Down to (1 - 0.2)^12 = ~6.87% weights remaining (or 14 for ~4.4%)
    prune_mode: str = "global"  # 'global' or 'layerwise'
    prune_classifier: bool = False  # Keep final classification head unpruned (critical for transfer)
    prune_conv_weights_only: bool = True  # Only prune Conv2d & Linear weights; keep biases & BN intact
    
    # --- Late Rewinding (Model B extension) ---
    rewind_epoch: int = 0  # 0 = standard LTH (Frankle & Carbin 2019); > 0 = late rewinding (Frankle et al., 2020)
    
    # --- Output & Logging ---
    output_dir: str = "./results"
    checkpoints_dir: str = "./results/checkpoints"
    plots_dir: str = "./results/plots"
    save_masks: bool = True
    save_models: bool = True
    resume: bool = True

def get_default_config() -> ExperimentConfig:
    return ExperimentConfig()
