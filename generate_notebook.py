"""
Script to generate the complete, self-contained main.ipynb Jupyter Notebook.
"""

import json
import os

def build_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    def add_md(source):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": source if isinstance(source, list) else [line + "\n" for line in source.split("\n")]
        })

    def add_code(source):
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source if isinstance(source, list) else [line + "\n" for line in source.split("\n")]
        })

    # Cell 1: Header & Introduction
    add_md("""# The Lottery Ticket Hypothesis on Convolutional Neural Networks
## An Empirical Study on Winning Tickets, Convergence Dynamics, and Subnetwork Transferability

**Author:** Mohamed Khacha  
**Repository:** [github.com/0khacha/winning-tickets-cnn-pruning](https://github.com/0khacha/winning-tickets-cnn-pruning)  
**Reference Paper:** *Frankle & Carbin (ICLR 2019), "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks"*

---

### Project Objectives:
1. **Empirical Existence of Winning Tickets:** Demonstrate that dense convolutional networks contain sparse subnetworks that, when trained in isolation from their original initial weights $\\theta_0$, match or outperform the unpruned baseline.
2. **Crucial Control Experiments:** Prove that the winning ticket's performance is **not** purely an inductive bias of the sparse topology by comparing against:
   - Identical mask with random re-initialization ($\\theta_{\\text{rand}} \\odot m$).
   - Random mask with matched layer-wise sparsity.
3. **Convergence Speed Analysis:** Quantify learning speed improvements via early-epoch validation accuracy (Epoch 5) and Area Under the Validation-Accuracy Curve (AUC-val).
4. **Cross-Domain Transfer Study:** Determine whether sparse subnetwork topologies found on CIFAR-10 transfer effectively to a distinct visual domain (SVHN: Street View House Numbers) without retraining masks from scratch.""")

    # Cell 2: Setup & Environment
    add_code("""# Hardware and Environment Verification
import os
import sys
import torch
import torchvision
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print(f"PyTorch Version: {torch.__version__}")
print(f"Torchvision Version: {torchvision.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
else:
    print("Warning: Running on CPU. For faster experiments, use a GPU (local RTX, Colab T4, or Kaggle).")""")

    # Cell 3: Config Markdown
    add_md("""### 1. Configuration & Hyperparameters
All experimental knobs (seeds, learning rate, pruning rate, number of rounds, architecture choice) are defined centrally in `config.py`.
The training recipe is strictly held constant across all variants (dense baseline, winning ticket, and random re-initialization controls).""")

    # Cell 4: Config Code
    add_code("""from config import get_default_config

cfg = get_default_config()
cfg.seeds = [42, 1337, 2026] # 3 random seeds for statistical rigor
cfg.epochs = 25              # Convergence epochs per round
cfg.prune_rate = 0.20        # Bottom 20% magnitude pruning per round
cfg.num_prune_rounds = 12    # Down to (1 - 0.2)^12 = ~6.87% weights
cfg.batch_size = 128
cfg.learning_rate = 3e-4     # Standard Adam lr for Conv-4 (Frankle & Carbin, 2019)
cfg.weight_decay = 1e-4

print("Configuration initialized:")
for k, v in vars(cfg).items():
    if not k.startswith('_'):
        print(f"  {k:25s}: {v}")""")

    # Cell 5: Model Markdown
    add_md("""### 2. Model Architecture: Model A (Conv-4)
Model A is a 4-convolutional layer, 2-fully connected layer network modeled directly after the Conv-4 family from Frankle & Carbin (2019):
- **Conv Block 1:** Conv(3, 64) -> BN -> ReLU -> Conv(64, 64) -> BN -> ReLU -> MaxPool(2,2)
- **Conv Block 2:** Conv(64, 128) -> BN -> ReLU -> Conv(128, 128) -> BN -> ReLU -> MaxPool(2,2)
- **Classifier:** Linear(8192, 256) -> ReLU -> Dropout(0.2) -> Linear(256, 10)

**Pruning Policy:**
- Conv & Linear weights are eligible for magnitude pruning.
- Biases and BatchNorm affine parameters ($\\gamma, \\beta$) are **not** pruned.
- The final classification layer (`fc2`) is kept unpruned (`prune_classifier=False`) to enable clean cross-domain transfer.""")

    # Cell 6: Model Inspection Code
    add_code("""from model import get_model, count_parameters

model_a = get_model("conv4", in_channels=3, num_classes=10)
param_info = count_parameters(model_a, prune_classifier=cfg.prune_classifier)

print("--- Model A (Conv-4) Parameter Breakdown ---")
for k, v in param_info.items():
    print(f"  {k:25s}: {v:,}")

# Model B (ResNet-9) inspection (optional extension)
model_b = get_model("resnet9", in_channels=3, num_classes=10)
param_info_b = count_parameters(model_b, prune_classifier=cfg.prune_classifier)
print("\\n--- Model B (ResNet-9) Parameter Breakdown ---")
for k, v in param_info_b.items():
    print(f"  {k:25s}: {v:,}")""")

    # Cell 7: Dataset Loading Markdown
    add_md("""### 3. Data Pipeline & Deterministic Validation Split
- **Source Dataset:** CIFAR-10 (45,000 train, 5,000 fixed validation, 10,000 test).
- **Target Dataset:** SVHN (Street View House Numbers, 10 classes, 32x32 RGB images).
- Validation accuracy is monitored for early convergence metrics (AUC-val and early validation accuracy).
- Test accuracy is evaluated only once at the conclusion of each round using the best validation checkpoint.""")

    # Cell 8: Dataset Loading Code
    add_code("""from data import get_dataloaders

dataloaders_cifar = get_dataloaders(
    dataset_name="cifar10",
    data_dir=cfg.data_dir,
    batch_size=cfg.batch_size,
    val_split_ratio=cfg.val_split_ratio,
    seed=42
)

print(f"CIFAR-10 Train Samples: {dataloaders_cifar['train_size']}")
print(f"CIFAR-10 Val Samples:   {dataloaders_cifar['val_size']}")
print(f"CIFAR-10 Test Samples:  {dataloaders_cifar['test_size']}")""")

    # Cell 9: Pruning Mechanism Markdown
    add_md("""### 4. Pruning Logic: Iterative Magnitude Pruning (IMP)
The core algorithm:
1. Initialize network $f(x; \\theta_0)$ and record initial weights $\\theta_0$.
2. Train $f(x; \\theta_0)$ for $T$ epochs to obtain trained weights $\\theta_T$.
3. Prune lowest 20% of weights by magnitude: $m = \\text{prune}(|\\theta_T|, 0.20)$.
4. **Winning Ticket Reset:** Restore weights to $\\theta_0$ with mask $m$: $\\theta \\leftarrow \\theta_0 \\odot m$.
5. **Control (Random Re-init):** Re-initialize weights with new random seed $\\theta_{\\text{rand}} \\odot m$.
6. Retrain and repeat for $R$ rounds.

**Continuous Mask Enforcement:**
During training, after every `optimizer.step()`, weights are projected back: $w \\leftarrow w \\odot m$.
At the end of every epoch, an assertion checks that no pruned weight is non-zero.""")

    # Cell 10: Pruning Test Code
    add_code("""from prune import init_masks, prune_magnitude, apply_masks_, verify_mask_integrity, compute_actual_sparsity

# Quick demonstration of mask projection and verification
test_model = get_model("conv4")
masks = init_masks(test_model, prune_classifier=False)
masks = prune_magnitude(test_model, masks, prune_rate=0.20, mode="global")

apply_masks_(test_model, masks)
verify_mask_integrity(test_model, masks)
sparsity_stats = compute_actual_sparsity(masks)

print(f"Global surviving weights: {sparsity_stats['surviving_weights']:,} / {sparsity_stats['total_weights']:,} ({sparsity_stats['fraction_surviving']*100:.2f}%)")""")

    # Cell 11: Execution Markdown
    add_md("""### 5. Running Experiments: Dense Baseline, IMP Winning Tickets, and Controls
Run the complete experimental pipeline across seeds. Checkpoints and metrics are automatically saved to disk after every round to support crash resumption.
*Note: To execute the full suite from command line, run `python main.py --experiment all`.*""")

    # Cell 12: Execution Code
    add_code("""from experiments import run_dense_baseline, run_imp_ticket_and_controls, run_transfer_study

# 1. Run Dense Baseline (CIFAR-10)
dense_cifar = run_dense_baseline(cfg, dataset_name="cifar10")

# 2. Run IMP Winning Ticket + Random Re-init + Random Mask Controls
imp_results = run_imp_ticket_and_controls(cfg, mode="global", include_random_mask_control=True)

# 3. Run Transfer Study to SVHN
dense_svhn = run_dense_baseline(cfg, dataset_name="svhn")
transfer_results = run_transfer_study(cfg, source_mode="global")""")

    # Cell 13: Plotting Markdown
    add_md("""### 6. Results & Visualizations
We generate publication-ready plots with log-scale x-axes and shaded mean $\\pm$ 1 std error bands:
1. **Accuracy vs. Weights Remaining:** Core test of whether winning tickets outperform random re-initialization and survive down to <10% weights.
2. **Learning Speed Dynamics:** Early validation accuracy (Epoch 5) and AUC-val across sparsity levels.
3. **Cross-Domain Transferability:** Transferred CIFAR-10 tickets evaluated on SVHN vs random mask baselines.""")

    # Cell 14: Plotting Code
    add_code("""from plots import (
    plot_accuracy_vs_sparsity,
    plot_learning_speed,
    plot_transfer_learning,
    generate_results_table
)
from IPython.display import Image, display

# 1. Accuracy vs Sparsity
plot_accuracy_vs_sparsity(
    ticket_data=imp_results["ticket"],
    reinit_data=imp_results["reinit"],
    dense_data=dense_cifar,
    randmask_data=imp_results["randmask"],
    save_path="./results/plots/accuracy_vs_sparsity.png"
)
display(Image(filename="./results/plots/accuracy_vs_sparsity.png"))

# 2. Learning Speed Dynamics
plot_learning_speed(
    ticket_data=imp_results["ticket"],
    reinit_data=imp_results["reinit"],
    save_path="./results/plots/learning_speed.png"
)
display(Image(filename="./results/plots/learning_speed.png"))

# 3. Transfer Learning
plot_transfer_learning(
    transfer_data=transfer_results["results"],
    target_dense_data=dense_svhn,
    save_path="./results/plots/transfer_learning.png"
)
display(Image(filename="./results/plots/transfer_learning.png"))""")

    # Cell 15: Results Table Code
    add_code("""# Generate and display summary table
table_df = generate_results_table(
    ticket_data=imp_results["ticket"],
    reinit_data=imp_results["reinit"],
    randmask_data=imp_results["randmask"],
    save_path_md="./results/results_table.md",
    save_path_csv="./results/results_table.csv"
)
table_df""")

    # Cell 16: Research Discussion Markdown
    add_md("""### 7. Scientific Discussion & Critical Caveats

#### A. Core Hypotheses under Investigation:
1. **The 'Winning Ticket' Effect:** Whether subnetworks with $\\sim 20\\%$ to $50\\%$ of original weights match or slightly exceed the dense baseline accuracy (*TBD - run python main.py --experiment all*).
2. **The Re-initialization Gap:** Whether the winning ticket significantly outperforms the randomly re-initialized control ($\\theta_{\\text{rand}} \\odot m$) at moderate to high sparsity, proving that initial weight alignment $\\theta_0$ is essential (*TBD - run python main.py --experiment all*).
3. **Subnetwork Transferability:** Whether masks discovered on CIFAR-10 transfer effectively to SVHN, outperforming random masks at identical sparsity (*TBD - run python main.py --experiment all*).

#### B. Crucial Caveats & Engineering Realities:
1. **Unstructured Sparsity vs. GPU Acceleration:**
   Standard GPUs (e.g. NVIDIA CUDA cores) rely on dense matrix-multiplication (GEMM) units. Unstructured zero weights do **not** provide wall-clock speedups without specialized sparse tensor cores (e.g., 2:4 structured sparsity on NVIDIA Ampere/Ada/Blackwell) or custom sparse engines. The network still consumes standard dense memory bandwidth unless stored in CSR/COO format.
2. **Residual Architectures & Late Rewinding (Literature Context):**
   In deeper networks with residual skip connections and batch normalization, literature notes that resetting to $\\theta_0$ can become unstable at high sparsity (Frankle et al., 2020). Rewinding to an early training checkpoint $\\theta_k$ ($k \\approx 2-5$ epochs) rather than $\\theta_0$ stabilizes training in residual architectures. The experiments module provides hooks for rewinding weights to support this future extension.
3. **Small-Scale Empirical Scope:**
   These experiments are designed for small CNNs on $32 \\times 32$ image benchmarks. While representative of course projects and foundational research, conclusions must be verified before extrapolating to large foundation models.""")

    nb_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print("Successfully built main.ipynb!")

if __name__ == "__main__":
    build_notebook()
