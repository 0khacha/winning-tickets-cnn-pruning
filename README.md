# The Lottery Ticket Hypothesis on Convolutional Neural Networks

**Author:** Mohamed Khacha  
**Repository:** [github.com/0khacha/winning-tickets-cnn-pruning](https://github.com/0khacha/winning-tickets-cnn-pruning)

## 1. Project Summary
This repository contains a modular and reproducible PyTorch implementation of the **Lottery Ticket Hypothesis (LTH)** (Frankle & Carbin, ICLR 2019) on small convolutional neural networks. We investigate whether dense CNNs (Conv-4) trained on CIFAR-10 contain sparse subnetworks that, when trained in isolation from their initial weights ($\theta_0$), match or outperform the unpruned dense model. We rigorously test the hypothesis against synchronous controls—specifically, random re-initialization with the same mask ($\theta_{\text{rand}} \odot m$) and random masks with matched layer-wise sparsity—and explore subnetwork transferability from CIFAR-10 to SVHN.

---

## 2. Repository Structure
```text
winning-tickets-cnn-pruning/
├── config.py              # Central experimental configuration & hyperparameter dataclass
├── data.py                # Deterministic train/val/test data pipelines & transforms (CIFAR-10, SVHN)
├── model.py               # Model A (Conv-4) architecture & parameter counting tools
├── prune.py               # Mask initialization, exact IMP pruning, and theta0/re-init resets
├── train.py               # Training loop with continuous mask projection (w <- w * m)
├── experiments.py         # Experiment orchestration with full checkpointing and crash resumption
├── plots.py               # Plotting routines (log-scale sparsity, learning speed, transfer)
├── main.py                # Command-line entrypoint for running benchmarks
├── test_prune.py          # Unit tests for pruning algebra, mask integrity, and crash resumption
├── main.ipynb             # Interactive Jupyter Notebook walkthrough
├── REPORT.md              # Detailed scientific report
├── requirements.txt       # Pinned environment dependencies
├── LICENSE                # MIT License
└── .gitignore             # Excludes large checkpoints, raw datasets, and caches
```

---

## 3. Installation & Setup
Clone the repository and install the pinned dependencies:
```bash
git clone https://github.com/0khacha/winning-tickets-cnn-pruning.git
cd winning-tickets-cnn-pruning
pip install -r requirements.txt
```

Verify that all unit tests pass:
```bash
pytest
```

---

## 4. Reproducing Experiments

### Quick Sanity Check (< 1 minute on CPU)
Verifies the entire pipeline end-to-end (dense training, IMP pruning, theta0 reset, random re-init, plotting, and table export) on a lightweight synthetic batch setup:
```bash
python main.py --experiment sanity_check
```

### Full Experimental Pipeline
To run the full suite of experiments across 3 seeds (`[42, 1337, 2026]`):
```bash
# 1. Unpruned Dense Baseline (CIFAR-10 and SVHN)
python main.py --experiment dense

# 2. Iterative Magnitude Pruning (Winning Tickets + Controls)
python main.py --experiment imp

# 3. Subnetwork Transfer Study (CIFAR-10 -> SVHN)
python main.py --experiment transfer

# 4. Generate all figures and summary tables from saved results
python main.py --experiment plots

# Or run the entire benchmark pipeline end-to-end:
python main.py --experiment all
```

---

## 5. Hardware and Runtime Notes
- **Device Support:** Automatically utilizes NVIDIA CUDA if available; cleanly falls back to CPU if no GPU is detected.
- **Estimated Runtime:**
  - **Sanity check:** $< 30$ seconds on CPU / $< 5$ seconds on GPU.
  - **Full experiment (Conv-4, 12 rounds, 3 seeds, 25 epochs per round):**
    - RTX 3080/4090: $\sim 1.5 - 2.5$ hours.
    - CPU (8-core): $\sim 18 - 24$ hours.
- **Resumption Support:** All initial weights ($\theta_0$), trained model checkpoints, pruning masks, and per-round metric records are saved to `./results/checkpoints/` after every round. Interrupted runs seamlessly resume from the last completed round without losing progress.

---

## 6. Experimental Results
Results are generated purely from measured empirical runs. Run the experiments to populate the results table:
```bash
python main.py --experiment all
```

| Round | Prunable Weights Remaining (%) | Ticket Test Acc (%) | Re-init Test Acc (%) | Delta (Ticket - Reinit) | Random Mask Acc (%) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0 (Dense) | 100.00% | TBD - run python main.py --experiment all | TBD - run python main.py --experiment all | TBD | TBD - run python main.py --experiment all |
| 1 | 80.00% | TBD - run python main.py --experiment all | TBD - run python main.py --experiment all | TBD | TBD - run python main.py --experiment all |
| 2 | 64.00% | TBD - run python main.py --experiment all | TBD - run python main.py --experiment all | TBD | TBD - run python main.py --experiment all |
| 3 | 51.20% | TBD - run python main.py --experiment all | TBD - run python main.py --experiment all | TBD | TBD - run python main.py --experiment all |
| 4 | 40.96% | TBD - run python main.py --experiment all | TBD - run python main.py --experiment all | TBD | TBD - run python main.py --experiment all |
| 5 | 32.77% | TBD - run python main.py --experiment all | TBD - run python main.py --experiment all | TBD | TBD - run python main.py --experiment all |
| 6 | 26.21% | TBD - run python main.py --experiment all | TBD - run python main.py --experiment all | TBD | TBD - run python main.py --experiment all |
| 7 | 20.97% | TBD - run python main.py --experiment all | TBD - run python main.py --experiment all | TBD | TBD - run python main.py --experiment all |
| 8 | 16.78% | TBD - run python main.py --experiment all | TBD - run python main.py --experiment all | TBD | TBD - run python main.py --experiment all |
| 9 | 13.42% | TBD - run python main.py --experiment all | TBD - run python main.py --experiment all | TBD | TBD - run python main.py --experiment all |
| 10 | 10.74% | TBD - run python main.py --experiment all | TBD - run python main.py --experiment all | TBD | TBD - run python main.py --experiment all |
| 11 | 8.59% | TBD - run python main.py --experiment all | TBD - run python main.py --experiment all | TBD | TBD - run python main.py --experiment all |
| 12 | 6.87% | TBD - run python main.py --experiment all | TBD - run python main.py --experiment all | TBD | TBD - run python main.py --experiment all |

---

## 7. Limitations & Honest Caveats
1. **Unstructured Sparsity vs Wall-Clock Speedup:** Setting individual weight tensor entries to zero does not yield inference speedups on standard dense GPU GEMM cores (cuBLAS). Achieving real latency reductions requires structured (e.g. 2:4) sparsity or custom sparse kernel runtimes.
2. **Computational Overhead of IMP:** Iterative magnitude pruning is discovery-oriented rather than an efficient compression method. Discovering a ticket takes $\sim (R + 1) \times$ the cost of standard dense training.
3. **Small Architectural Scope:** Results are based on Conv-4 architectures on $32 \times 32$ image datasets (CIFAR-10, SVHN); conclusions should not be extrapolated to large-scale Vision Transformers or LLMs without further empirical verification.

---

## 8. License
This project is open-source under the [MIT License](LICENSE).
