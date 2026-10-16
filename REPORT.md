# The Lottery Ticket Hypothesis on Convolutional Neural Networks
## An Empirical Study on Winning Tickets, Convergence Dynamics, and Cross-Domain Transferability

**Author:** Mohamed Khacha  
**Repository:** [github.com/0khacha/winning-tickets-cnn-pruning](https://github.com/0khacha/winning-tickets-cnn-pruning)  
**Target Architecture:** Model A (Conv-4)  
**Datasets:** CIFAR-10 (Source), SVHN (Transfer Target)  
**Primary Reference:** Frankle & Carbin (ICLR 2019), *"The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks"*  
**Secondary Reference:** Frankle, Dziugaite, Roy, Carbin (2020), *"Linear Mode Connectivity and the Lottery Ticket Hypothesis"*; Morcos et al. (NeurIPS 2019), *"One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers"*.

---

## 1. Executive Summary

The **Lottery Ticket Hypothesis (LTH)** (Frankle & Carbin, 2019) asserts that dense, randomly-initialized neural networks contain sparse subnetworks ("winning tickets") that—when trained in isolation from their original initialization—can match or exceed the test accuracy of the original dense network in a comparable number of iterations.

This project delivers a complete, modular, and mathematically rigorous empirical framework in PyTorch to evaluate:
1. **Existence of Winning Tickets:** Whether a small convolutional network (Conv-4, 2.36M weights) pruned iteratively down to $\sim 6.87\%$ remaining prunable parameters can match or outperform the unpruned dense baseline on CIFAR-10.
2. **The Re-initialization Gap (Mandatory Control):** Whether resetting surviving weights to their specific initialization $\theta_0$ outperforms randomly re-initializing the same sparse topology ($\theta_{\text{rand}} \odot m$).
3. **Extra Structural Control:** Whether winning ticket masks outperform randomly generated binary masks with the exact same layer-wise sparsity.
4. **Learning Speed Dynamics:** Whether winning tickets converge faster than randomly re-initialized subnetworks, evaluated via accuracy at fixed early training checkpoints (Epoch 5) and Area Under the Validation-Accuracy Curve ($\text{AUC}_{\text{val}}$).
5. **Cross-Domain Subnetwork Transferability:** Whether winning ticket masks discovered on CIFAR-10 transfer effectively to a visually distinct domain (SVHN: Street View House Numbers, 10 classes) when re-initialized with an adapted classification head.

---

## 2. Theoretical Framework & Methodology

### 2.1 Formal Definition of Winning Tickets
Consider a feed-forward neural network $f(x; \theta)$ parameterized by initial weights $\theta_0 \sim \mathcal{D}_\theta$. When optimized using stochastic gradient descent (or Adam) with training recipe $\mathcal{T}$, the parameters reach a trained state $\theta_T = \mathcal{T}(f(x; \theta_0))$ yielding test accuracy $A$ after $T$ iterations.

A binary mask $m \in \{0, 1\}^{|\theta|}$ defines a subnetwork $f(x; m \odot \theta)$. According to the LTH:
$$ \exists m \text{ such that } \|m\|_0 \ll |\theta|, \quad \mathcal{T}(f(x; m \odot \theta_0)) \text{ achieves accuracy } A' \ge A \text{ in } T' \le T \text{ iterations.} $$

If the surviving weights are instead re-initialized with fresh random weights $\theta_{\text{rand}} \sim \mathcal{D}_\theta$, the resulting trained subnetwork $\mathcal{T}(f(x; m \odot \theta_{\text{rand}}))$ achieves accuracy $A_{\text{reinit}} \ll A'$. This performance delta:
$$ \Delta_{\text{LTH}}(m) = A(\text{Ticket: } m \odot \theta_0) - A(\text{Control: } m \odot \theta_{\text{rand}}) $$
is the empirical signature of a true winning ticket.

### 2.2 Iterative Magnitude Pruning (IMP) Algorithm
Pruning all weights in a single shot (one-shot magnitude pruning) degrades network representation severely at high sparsity ($<20\%$ weights). Iterative Magnitude Pruning (IMP) prunes $p = 20\%$ of the remaining surviving weights per round over $R = 12-15$ rounds:

```
Algorithm 1: Iterative Magnitude Pruning (IMP) with θ0 Reset
Input: Initial model weights θ0, Training recipe T, Prune rate p = 0.20, Prune rounds R
Output: Binary masks {m_r}_{r=0}^R, Checkpoint history

1: Initialize mask m_0 = 1 for all prunable weights.
2: Train θ_0 under mask m_0 using T -> yields trained parameters θ_T^(0).
3: For round r = 1 to R:
4:     Identify lowest-magnitude fraction p of surviving weights in θ_T^(r-1):
           m_r = Prune(θ_T^(r-1), m_(r-1), p)
5:     Winning Ticket: Reset surviving weights to θ_0:
           θ_init^(r) = θ_0 ⊙ m_r
6:     Train θ_init^(r) using identical recipe T with continuous projection w <- w ⊙ m_r:
           θ_T^(r) = T(f(x; θ_init^(r)))
7:     Mandatory Control (Synchronous):
           θ_rand^(r) = θ_rand ⊙ m_r
           Train θ_rand^(r) using identical recipe T -> yields θ_T,reinit^(r)
8:     Record test accuracy, val accuracy, best epoch, AUC_val, and actual sparsity.
```

### 2.3 Strict Engineering Integrity & Mask Algebra
In standard PyTorch training, an optimizer step $w \leftarrow w - \eta \nabla_w \mathcal{L} - \eta \lambda w$ will immediately update and "resurrect" zeroed weights if gradient updates or weight decay are unconstrained. To maintain mathematical validity:
1. **Continuous Post-Step Projection:** Immediately following `optimizer.step()`, the in-place operation `w.data.mul_(mask)` projects the parameters back onto the unpruned subspace.
2. **Post-Epoch Integrity Assertion:** At the end of every epoch, an assertion verifies:
   $$\max_{i \in \{j : m_j = 0\}} |w_i| \le 10^{-7}$$
   guaranteeing zero weight leakage.
3. **Exact Indexing:** Pruning thresholds are computed via `torch.topk(..., largest=False)` on surviving indices to avoid tie-breaking discrepancies or floating-point comparison errors.

### 2.4 Parameter Exclusion Policy
To adhere to standard literature and preserve functional viability:
- **Prunable:** Weight matrices $W \in \mathbb{R}^{C_{\text{out}} \times C_{\text{in}} \times K \times K}$ of `nn.Conv2d` layers and $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ of hidden `nn.Linear` layers.
- **Unpruned:**
  - **Biases ($b$):** Biases constitute $<0.02\%$ of network parameters; pruning biases introduces systematic activation shifts without meaningful sparsity gains.
  - **BatchNorm Parameters ($\gamma, \beta$, running mean/variance):** Pruning batch normalization scaling factors destroys gradient propagation through deep channels.
  - **Classification Head (`fc2`):** Kept unpruned (`prune_classifier=False`) to permit seamless head replacement during transfer learning experiments.

---

## 3. Experimental Setup

### 3.1 Network Architectures

#### Model A: Conv-4 (Frankle & Carbin, 2019)
Modeled directly after the Conv-4 family evaluated in Section 4 of the original LTH paper:
- **Layer 1:** Conv2d(3, 64, kernel=3, pad=1, bias=False) $\rightarrow$ BatchNorm2d(64) $\rightarrow$ ReLU
- **Layer 2:** Conv2d(64, 64, kernel=3, pad=1, bias=False) $\rightarrow$ BatchNorm2d(64) $\rightarrow$ ReLU $\rightarrow$ MaxPool2d(2,2)
- **Layer 3:** Conv2d(64, 128, kernel=3, pad=1, bias=False) $\rightarrow$ BatchNorm2d(128) $\rightarrow$ ReLU
- **Layer 4:** Conv2d(128, 128, kernel=3, pad=1, bias=False) $\rightarrow$ BatchNorm2d(128) $\rightarrow$ ReLU $\rightarrow$ MaxPool2d(2,2)
- **Classifier:** Linear(8192, 256) $\rightarrow$ ReLU $\rightarrow$ Dropout(0.2) $\rightarrow$ Linear(256, 10)

| Parameter Category | Parameter Count | Percentage of Total |
| :--- | :--- | :--- |
| **Total Parameters** | 2,360,522 | 100.00% |
| **Trainable Parameters** | 2,360,522 | 100.00% |
| **Prunable Weights (Conv & FC1)** | 2,356,928 | 99.85% |
| **BatchNorm Parameters ($\gamma, \beta$)** | 768 | 0.03% |
| **FC1 Bias Parameters** | 256 | 0.01% |
| **Classification Head (`fc2` Weights & Bias)** | 2,570 | 0.11% |

#### Model B: ResNet-9 (Residual Architecture Extension)
Compact residual network featuring 3 residual stages (64, 128, 256 channels) with skip connections and global average pooling:
- **Total Parameters:** 2,440,266
- **Prunable Weights:** 2,434,752
- **BatchNorm Parameters:** 2,944
- **Classifier Parameters:** 2,570

### 3.2 Fixed Training Recipe
Identical hyperparameters are applied across all experiments, rounds, and controls:
- **Optimizer:** Adam ($\beta_1=0.9, \beta_2=0.999$, $\epsilon=10^{-8}$)
- **Learning Rate:** $\eta = 3 \times 10^{-4}$
- **Weight Decay:** $\lambda = 1 \times 10^{-4}$
- **Learning Rate Schedule:** Cosine Annealing decay across epochs: $\eta_t = \frac{1}{2}\eta_{\max}\left(1 + \cos\left(\frac{t\pi}{T}\right)\right)$
- **Epochs per Round:** $T = 25$
- **Batch Size:** 128
- **Random Seeds:** 3 distinct seeds (`[42, 1337, 2026]`)
- **Data Augmentation:** Random crop ($32 \times 32$, padding 4) + Random horizontal flip + Standard per-channel normalization.

### 3.3 Datasets & Validation Policy
- **Source Task (CIFAR-10):** 60,000 $32 \times 32$ color images across 10 object classes.
  - Training Split: 45,000 samples (deterministic fixed partition).
  - Validation Split: 5,000 samples (reserved strictly for model selection and learning speed dynamics).
  - Test Split: 10,000 samples (evaluated once for the final selected epoch checkpoint).
- **Target Task (SVHN):** 73,257 training samples and 26,032 test samples of cropped house numbers ($32 \times 32$ RGB, digits 0–9).
  - Validation Split: 7,325 samples (10% deterministic split).
  - Augmentation: Random crop ($32 \times 32$, padding 2) + Normalization (no horizontal flipping, as digit orientation alters semantic identity: e.g. 6 vs 9).

---

## 4. Empirical Results & Findings

### 4.1 Results Summary Table
The table below will report test accuracy (mean ± standard deviation across seeds), early-epoch validation accuracy, and validation AUC for Model A (Conv-4) on CIFAR-10 across 12 IMP pruning rounds once experiments are completed:

| IMP Round | Fraction Surviving ($) | Weights Remaining (%) | Winning Ticket ( \odot \theta_0$) | Random Re-init ( \odot \theta_{\text{rand}}$) | $\Delta_{\text{LTH}}$ (Ticket - Reinit) | Random Mask Control | Val Acc (Epoch 5) | Val AUC |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0 (Dense)** | 1.0000 | 100.00% | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **1** | 0.8000 | 80.00% | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **2** | 0.6400 | 64.00% | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **3** | 0.5120 | 51.20% | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **4** | 0.4096 | 40.96% | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **5** | 0.3277 | 32.77% | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **6** | 0.2621 | 26.21% | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **7** | 0.2097 | 20.97% | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **8** | 0.1678 | 16.78% | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **9** | 0.1342 | 13.42% | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **10** | 0.1074 | 10.74% | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **11** | 0.0859 | 8.59% | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| **12** | 0.0687 | 6.87% | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |

*(Note: Exact quantitative measurements are pending completion of the multi-seed experiment run: python main.py --experiment all).*

### 4.2 Core Hypotheses to Evaluate
Upon running the full experimental suite, the following core questions will be evaluated against the collected metrics:
1. **Winning Tickets vs. Dense Baseline:**
   Whether subnetworks at moderate pruning levels ( \in [0.4, 0.8]$) match or exceed the generalization performance of the unpruned dense model ( - run\ python\ main.py\ --experiment\ all$).
2. **The Re-initialization Advantage ($\Delta_{\text{LTH}} = \text{Acc}(m \odot \theta_0) - \text{Acc}(m \odot \theta_{\text{rand}})$):**
   Evaluating whether the winning ticket systematically outperforms the randomly re-initialized subnetwork across pruning rounds, testing whether sparsity alone provides an inductive bias or if initial weight alignment is necessary ( - run\ python\ main.py\ --experiment\ all$).
3. **Random Mask Control Comparison:**
   Assessing whether the iterative magnitude pruning mask $ provides superior structure compared to an Erdős-Rényi or uniformly randomized mask at matching layer sparsity ( - run\ python\ main.py\ --experiment\ all$).

### 4.3 Learning Speed Dynamics
Rather than relying on peak test epoch (which can be confounded by learning rate decay schedules like cosine annealing), learning speed is quantified via:
- **Early-epoch Validation Accuracy:** Accuracy achieved at fixed early training steps (Epoch 5 and Epoch 10).
- **Validation Area Under the Curve ($\text{AUC}_{\text{val}}$):** Normalized integral of validation accuracy across all training epochs ( - run\ python\ main.py\ --experiment\ all$).

### 4.4 Subnetwork Transferability (CIFAR-10 $\rightarrow$ SVHN)
Evaluating whether winning ticket masks {\text{CIFAR}}$ found on CIFAR-10 can transfer to SVHN (with initial convolutional weights preserved and a freshly initialized linear head):
- Comparing transferred tickets against dense SVHN baselines, native SVHN winning tickets, and random masks at equal sparsity ( - run\ python\ main.py\ --experiment\ all$).

---

## 5. Architectural Considerations: Residual Networks and Rewinding (Future Work)

In larger or residual architectures (e.g., ResNet-9 with skip connections and batch normalization), literature indicates classic lottery ticket pruning with reset to $\theta_0$ can fail due to early training instability (Frankle et al., 2020).
- **Late Rewinding ($\theta_k$):** Resetting weights to an early training checkpoint $\theta_k$ ( \approx 2$ to $ epochs) rather than $\theta_0$ stabilizes subnetworks in residual models.
- While the core empirical pipeline here focuses on Conv-4 on CIFAR-10/SVHN, late rewinding parameter hooks are implemented in experiments.py to support future residual scaling studies.

---

## 6. Engineering Realities & Practical Caveats

1. **Unstructured Sparsity Does NOT Accelerate Standard GPUs:**
   Although an IMP winning ticket has \%$ of its weights set to zero, standard deep learning hardware (e.g. NVIDIA CUDA cores executing dense cuBLAS / cuDNN GEMM operations) still multiplies by zero. Without specialized sparse tensor hardware (e.g., NVIDIA 2:4 Ampere/Ada Sparse Cores) or custom sparse linear algebra runtimes (CSR / SpMM), the wall-clock execution time of an unpruned model and a \%$ sparse model is virtually identical.
2. **Computational Cost of IMP:**
   Finding a winning ticket via IMP is an order of magnitude more expensive than training the dense model once: running 12 pruning rounds requires  \times$ the training time. IMP is primarily a scientific discovery tool for understanding network overparameterization rather than an efficient compression pipeline.
3. **Seed Variance at Extreme Sparsity:**
   Sparse subnetworks can become sensitive to initial random seed assignments and batch stochasticity at extreme pruning levels. All final evaluations should report mean and standard deviation across multiple independent seeds.

---

## 7. Conclusion

This project implements an end-to-end framework for investigating the Lottery Ticket Hypothesis on convolutional networks:
- Rigorous controls including random re-initialization ( \odot \theta_{\text{rand}}$) and random masks at matched sparsity.
- Cross-dataset subnetwork transferability study (CIFAR-10 $\rightarrow$ SVHN).
- Metrics for learning speed (early epoch accuracy, validation AUC) independent of peak epoch artifacts.
- Crash-resilient round-by-round state checkpointing for iterative magnitude pruning.
*(Final quantitative claims will be populated upon executing python main.py --experiment all).*

---

## 8. References *(Verify before citing in formal publications)*
1. Frankle, J., & Carbin, M. (2019). *The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks*. International Conference on Learning Representations (ICLR).
2. Frankle, J., Dziugaite, G. K., Roy, D. M., & Carbin, M. (2020). *Linear Mode Connectivity and the Lottery Ticket Hypothesis*. International Conference on Machine Learning (ICML). *(verify before citing)*
3. Morcos, A., Yu, H., Paganini, M., & Tian, Y. (2019). *One ticket to win them all: generalizing lottery ticket initializations across datasets*. Advances in Neural Information Processing Systems (NeurIPS). *(verify before citing)*
4. He, K., Zhang, X., Ren, S., & Sun, J. (2016). *Deep Residual Learning for Image Recognition*. CVPR.
