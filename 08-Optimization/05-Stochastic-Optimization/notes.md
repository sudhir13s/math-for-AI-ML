[← Back to Optimization](../README.md) | [Next: Optimization Landscape →](../06-Optimization-Landscape/notes.md)

---

# Stochastic Optimization

> _"The best thing about stochastic gradient descent is that it works. The worst thing about stochastic gradient descent is that nobody fully understands why."_ — Léon Bottou

## Overview

Stochastic optimization is the computational engine that makes modern machine learning possible. When the dataset contains millions or billions of examples, computing the full gradient — which requires a pass over the entire dataset — is prohibitively expensive. Stochastic gradient descent (SGD) replaces the full gradient with a noisy estimate computed from a single example or a small mini-batch, reducing the per-iteration cost from $O(n)$ to $O(1)$ or $O(B)$ where $B \ll n$.

This noise is not just a computational convenience — it is a feature. The stochasticity of SGD helps escape sharp minima and saddle points, implicitly regularizes the solution, and contributes to the remarkable generalization performance of overparameterized neural networks. Understanding the interplay between gradient noise, learning rate, batch size, and generalization is one of the most active areas of research in deep learning theory.

The mathematical theory of stochastic optimization is rich and deep. For convex functions, SGD with diminishing step sizes converges at rate $O(1/\sqrt{T})$ for non-smooth objectives and $O(1/T)$ for strongly convex smooth objectives. Mini-batching reduces the gradient variance proportional to $1/B$, but with diminishing returns beyond a critical batch size. Variance reduction methods (SVRG, SAGA) achieve linear convergence for finite-sum problems by combining the low per-iteration cost of SGD with the fast convergence of full gradient descent.

This section develops the full theory of stochastic optimization. We begin with the stochastic gradient oracle and convergence analysis (§3), then study mini-batch SGD and the linear scaling rule (§4), variance reduction methods (§5), momentum in the stochastic setting (§6), and distributed optimization (§7). Advanced topics cover SGD's implicit bias, sharpness-aware minimization, and the connection to Langevin dynamics (§8). Every result is connected to its role in training real ML systems at scale.

**Scope note.** This section covers stochastic first-order methods with noisy gradient estimates. The _convergence theory of deterministic GD_ is covered in [Gradient Descent](../02-Gradient-Descent/notes.md). _Adaptive_ per-parameter methods (Adam, RMSProp) that combine stochastic gradients with per-parameter learning rate adaptation belong to [Adaptive Learning Rate](../07-Adaptive-Learning-Rate/notes.md). _Learning rate schedules_ (warmup, cosine decay) that control the step size over time are covered in [Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md). The _optimization landscape_ and its effect on SGD dynamics are analyzed in [Optimization Landscape](../06-Optimization-Landscape/notes.md).

**For AI practitioners:** By the end of this section, you will understand why SGD generalizes better than Adam in some settings, how to choose the optimal batch size for distributed training, why variance reduction methods converge faster than vanilla SGD, how momentum interacts with gradient noise, and how to implement communication-efficient distributed SGD for large-scale model training.

## Prerequisites

- **Gradient descent convergence theory** — $O(1/T)$ for convex, $O((1-\mu/L)^T)$ for strongly convex — [Chapter 8 §02](../02-Gradient-Descent/notes.md)
- **L-smoothness and strong convexity** — definitions, condition number — [Chapter 8 §01](../01-Convex-Optimization/notes.md)
- **Probability and expectations** — unbiased estimators, variance, law of large numbers — [Chapter 6: Probability Theory](../../06-Probability-Theory/README.md)
- **Matrix norms and eigenvalues** — spectral norm, condition number — [Chapter 3 §06](../../03-Advanced-Linear-Algebra/06-Matrix-Norms/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Interactive demonstrations of SGD convergence, mini-batch variance, SVRG, SAGA, and distributed SGD |
| [exercises.ipynb](exercises.ipynb) | 8 graded exercises from SGD convergence proofs to variance reduction analysis |

## Learning Objectives

After completing this section, you will:

1. Define the stochastic gradient oracle and prove unbiasedness of mini-batch gradient estimates
2. Prove the $O(1/\sqrt{T})$ convergence rate of SGD for convex non-smooth functions
3. Prove the $O(1/T)$ convergence rate of SGD for strongly convex smooth functions
4. Analyze the variance reduction effect of mini-batching and derive the critical batch size
5. Derive and analyze SVRG and SAGA variance reduction algorithms
6. Explain how momentum interacts with gradient noise in the stochastic setting
7. Analyze synchronous vs. asynchronous distributed SGD and their trade-offs
8. Implement communication-efficient SGD with gradient quantization and sparsification
9. Explain SGD's implicit bias toward flat minima and its connection to generalization
10. Derive the linear scaling rule for learning rate adjustment with batch size
11. Connect SGD dynamics to Langevin dynamics and Bayesian inference
12. Apply stochastic optimization to large-scale logistic regression, matrix factorization, and neural network training

---

## Table of Contents

- [Stochastic Optimization](#stochastic-optimization)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
  - [Companion Notebooks](#companion-notebooks)
  - [Learning Objectives](#learning-objectives)
  - [Table of Contents](#table-of-contents)
  - [1. Intuition](#1-intuition)
    - [1.1 Why Stochastic Gradients? The Scale Problem](#11-why-stochastic-gradients-the-scale-problem)
    - [1.2 The Noise-Benefit Trade-off](#12-the-noise-benefit-trade-off)
    - [1.3 SGD as a Noisy Version of GD](#13-sgd-as-a-noisy-version-of-gd)
    - [1.4 Historical Timeline: Robbins-Monro to Modern Distributed SGD](#14-historical-timeline-robbins-monro-to-modern-distributed-sgd)
    - [1.5 Stochastic Optimization in Deep Learning](#15-stochastic-optimization-in-deep-learning)
  - [2. Formal Definitions](#2-formal-definitions)
    - [2.1 The Stochastic Optimization Problem](#21-the-stochastic-optimization-problem)
    - [2.2 Stochastic Gradient Oracle](#22-stochastic-gradient-oracle)
    - [2.3 Unbiasedness and Variance of Gradient Estimates](#23-unbiasedness-and-variance-of-gradient-estimates)
    - [2.4 Mini-Batch Gradient Estimation](#24-mini-batch-gradient-estimation)
    - [2.5 Expected Risk vs. Empirical Risk](#25-expected-risk-vs-empirical-risk)
  - [3. Convergence Theory for SGD](#3-convergence-theory-for-sgd)
    - [3.1 SGD with Diminishing Step Sizes](#31-sgd-with-diminishing-step-sizes)
    - [3.2 O(1/√T) Rate for Convex Non-Smooth Functions](#32-o1t-rate-for-convex-non-smooth-functions)
    - [3.3 O(1/T) Rate for Strongly Convex Smooth Functions](#33-o1t-rate-for-strongly-convex-smooth-functions)
    - [3.4 The Role of Gradient Variance in Convergence](#34-the-role-of-gradient-variance-in-convergence)
    - [3.5 Lower Bounds for Stochastic First-Order Methods](#35-lower-bounds-for-stochastic-first-order-methods)
  - [4. Mini-Batch SGD](#4-mini-batch-sgd)
    - [4.1 Mini-Batch Gradient Estimator](#41-mini-batch-gradient-estimator)
    - [4.2 Variance Reduction with Larger Batches](#42-variance-reduction-with-larger-batches)
    - [4.3 The Linear Scaling Rule](#43-the-linear-scaling-rule)
    - [4.4 Critical Batch Size and Diminishing Returns](#44-critical-batch-size-and-diminishing-returns)
    - [4.5 Generalization Effects of Batch Size](#45-generalization-effects-of-batch-size)
  - [5. Variance Reduction Methods](#5-variance-reduction-methods)
    - [5.1 The Variance Problem in SGD](#51-the-variance-problem-in-sgd)
    - [5.2 SVRG: Stochastic Variance Reduced Gradient](#52-svrg-stochastic-variance-reduced-gradient)
    - [5.3 SAGA: Stochastic Average Gradient Amélioré](#53-saga-stochastic-average-gradient-amélioré)
    - [5.4 SDCA: Stochastic Dual Coordinate Ascent](#54-sdca-stochastic-dual-coordinate-ascent)
    - [5.5 Comparison: SVRG vs. SAGA vs. SGD](#55-comparison-svrg-vs-saga-vs-sgd)
  - [6. Momentum in the Stochastic Setting](#6-momentum-in-the-stochastic-setting)
    - [6.1 SGD with Momentum: Theory and Practice](#61-sgd-with-momentum-theory-and-practice)
    - [6.2 Heavy Ball vs. Nesterov in Stochastic Settings](#62-heavy-ball-vs-nesterov-in-stochastic-settings)
    - [6.3 The Bias-Variance Trade-off of Momentum](#63-the-bias-variance-trade-off-of-momentum)
    - [6.4 Lookahead and Averaging Methods](#64-lookahead-and-averaging-methods)
    - [6.5 Why Momentum Helps SGD Escape Saddle Points](#65-why-momentum-helps-sgd-escape-saddle-points)
  - [7. Distributed Stochastic Optimization](#7-distributed-stochastic-optimization)
    - [7.1 Data Parallelism and Gradient Averaging](#71-data-parallelism-and-gradient-averaging)
    - [7.2 Synchronous vs. Asynchronous SGD](#72-synchronous-vs-asynchronous-sgd)
    - [7.3 Communication-Efficient SGD: Quantization and Sparsification](#73-communication-efficient-sgd-quantization-and-sparsification)
    - [7.4 Federated Learning and Local SGD](#74-federated-learning-and-local-sgd)
    - [7.5 Decentralized SGD and Gossip Algorithms](#75-decentralized-sgd-and-gossip-algorithms)
  - [8. Advanced Topics](#8-advanced-topics)
    - [8.1 SGD Implicit Bias and Generalization](#81-sgd-implicit-bias-and-generalization)
    - [8.2 Sharpness-Aware Minimization (SAM)](#82-sharpness-aware-minimization-sam)
    - [8.3 Stochastic Second-Order Methods](#83-stochastic-second-order-methods)
    - [8.4 Adaptive Step Sizes for SGD](#84-adaptive-step-sizes-for-sgd)
    - [8.5 Connection to Langevin Dynamics and Bayesian Inference](#85-connection-to-langevin-dynamics-and-bayesian-inference)
  - [9. Applications in Machine Learning](#9-applications-in-machine-learning)
    - [9.1 Training Deep Neural Networks with SGD](#91-training-deep-neural-networks-with-sgd)
    - [9.2 Large-Scale Logistic Regression](#92-large-scale-logistic-regression)
    - [9.3 Matrix Factorization with SGD](#93-matrix-factorization-with-sgd)
    - [9.4 Reinforcement Learning: Stochastic Policy Gradients](#94-reinforcement-learning-stochastic-policy-gradients)
    - [9.5 Distributed LLM Pretraining](#95-distributed-llm-pretraining)
  - [10. Common Mistakes](#10-common-mistakes)
  - [11. Exercises](#11-exercises)
  - [12. Why This Matters for AI (2026 Perspective)](#12-why-this-matters-for-ai-2026-perspective)
  - [13. Conceptual Bridge](#13-conceptual-bridge)
  - [References](#references)

---

## 1. Intuition

### 1.1 Why Stochastic Gradients? The Scale Problem

In machine learning, we typically minimize the empirical risk:

$$f(\boldsymbol{\theta}) = \frac{1}{n}\sum_{i=1}^n \ell_i(\boldsymbol{\theta})$$

where $\ell_i(\boldsymbol{\theta})$ is the loss on the $i$-th training example. The full gradient is:

$$\nabla f(\boldsymbol{\theta}) = \frac{1}{n}\sum_{i=1}^n \nabla \ell_i(\boldsymbol{\theta})$$

Computing this requires $O(n)$ operations — one forward and backward pass per example. For $n = 10^9$ (common in LLM pretraining), this is computationally infeasible per iteration.

**The stochastic gradient** replaces the full sum with a single term (or a small average):

$$\mathbf{g}_t = \nabla \ell_{i_t}(\boldsymbol{\theta}_t) \quad \text{where } i_t \sim \text{Uniform}(\{1, \ldots, n\})$$

This costs $O(1)$ per iteration instead of $O(n)$. The trade-off: the stochastic gradient is noisy — it is an unbiased estimate of the true gradient but has non-zero variance.

### 1.2 The Noise-Benefit Trade-off

The noise in SGD is not just a computational necessity — it provides several benefits:

1. **Escape from sharp minima:** The gradient noise acts as a temperature that allows SGD to escape sharp minima (which generalize poorly) and settle in flat minima (which generalize well).

2. **Implicit regularization:** SGD with small batch sizes implicitly biases the solution toward simpler models, similar to explicit regularization.

3. **Saddle point escape:** In high-dimensional non-convex optimization, saddle points are far more common than local minima. The noise in SGD helps escape saddle points efficiently, while deterministic GD can get stuck.

4. **Early stopping effect:** The noise prevents overfitting by preventing the parameters from settling too precisely into the training data minimum.

```text
SGD NOISE AS IMPLICIT REGULARIZATION
════════════════════════════════════════════════════════════════════════

  Full-batch GD:                       Mini-batch SGD:

       ╲        ╱                           ╲    ╱╲
        ╲  •   ╱   ← sharp minimum           ╲  ╱  ╲  ← noise prevents
         ╲    ╱                                ╲╱    ╲    settling in sharp min
          ╲  ╱                                  •     ← flat minimum
           ╲╱                                 ╱╲
            • ← converges precisely          ╱  ╲
              to sharp minimum              ╱    ╲

  Sharp minimum: high curvature           Flat minimum: low curvature
  Poor generalization                     Good generalization
  Sensitive to perturbations              Robust to perturbations

════════════════════════════════════════════════════════════════════════
```

### 1.3 SGD as a Noisy Version of GD

The SGD update can be decomposed into the true gradient plus noise:

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \mathbf{g}_t = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t) - \eta \boldsymbol{\xi}_t$$

where $\boldsymbol{\xi}_t = \mathbf{g}_t - \nabla f(\boldsymbol{\theta}_t)$ is the gradient noise with $\mathbb{E}[\boldsymbol{\xi}_t] = \mathbf{0}$ and $\mathbb{E}[\lVert \boldsymbol{\xi}_t \rVert_2^2] = \sigma^2$.

This decomposition reveals that SGD is equivalent to GD with additive noise. The noise level $\sigma^2$ depends on the batch size $B$: $\sigma^2 \propto 1/B$. This connection to noisy gradient descent is the foundation for understanding SGD's generalization properties.

### 1.4 Historical Timeline: Robbins-Monro to Modern Distributed SGD

```text
STOCHASTIC OPTIMIZATION TIMELINE
════════════════════════════════════════════════════════════════════════

  1951  Robbins &       — Stochastic approximation: the birth of SGD
        Monro
  1952  Kiefer &        — Stochastic approximation for regression
        Wolfowitz
  1998  Bottou          — SGD for large-scale learning (online learning)
  2000s Bottou &        — Theoretical foundations of SGD for ML
        Le Cun
  2008  Bottou          — Large-scale SVM training with SGD
  2010  Recht et al.    — Hogwild!: lock-free parallel SGD
  2012  Krizhevsky      — ImageNet with SGD + momentum (AlexNet)
        et al.
  2014  Kingma & Ba     — Adam: adaptive stochastic optimization
  2015  Reddi et al.    — SVRG: variance-reduced SGD
  2016  Goyal et al.    — Linear scaling rule for large-batch SGD
  2017  Keskar et al.   — Large-batch training and generalization gap
  2018  Smith et al.    — Super-convergence with cyclic LR + SGD
  2019  You et al.      — LAMB: large-batch optimization for BERT
  2020  Foret et al.    — SAM: sharpness-aware minimization
  2023  LLaMA / GPT-4   — AdamW + large-batch SGD at trillion-token scale
  2024-26 Frontier LLMs — Distributed SGD with gradient compression

════════════════════════════════════════════════════════════════════════
```

### 1.5 Stochastic Optimization in Deep Learning

Every modern neural network is trained with some form of stochastic optimization. The choice of optimizer affects not just convergence speed but also the final model's generalization performance.

| Optimizer | Stochastic? | Variance Reduction | Best For |
| --- | --- | --- | --- |
| **SGD** | Yes (mini-batch) | None (raw noise) | Vision models, when generalization matters most |
| **SGD + Momentum** | Yes | Partial (EMA of gradients) | Most deep learning tasks |
| **Adam** | Yes | Partial (adaptive per-parameter) | Widely used in NLP and generative modeling |
| **AdamW** | Yes | Partial + decoupled weight decay | LLM pretraining and fine-tuning |
| **SVRG/SAGA** | Yes | Full (linear convergence) | Finite-sum problems (logistic regression, SVM) |
| **LAMB** | Yes | Adaptive + layer-wise | Large-batch BERT/LLM training |

**For AI:** Understanding stochastic optimization is essential because the optimizer choice directly impacts model quality. SGD often generalizes better than Adam for vision tasks, while Adam/AdamW dominates for language models. The reasons are tied to how gradient noise interacts with the loss landscape geometry — a topic we explore in depth.

---

## 2. Formal Definitions

### 2.1 The Stochastic Optimization Problem

We consider the problem of minimizing an expected loss:

$$\min_{\boldsymbol{\theta}} F(\boldsymbol{\theta}) = \mathbb{E}_{\xi \sim \mathcal{D}}[f(\boldsymbol{\theta}; \xi)]$$

where $\xi$ is a random variable representing a data point (or mini-batch) drawn from distribution $\mathcal{D}$, and $f(\boldsymbol{\theta}; \xi)$ is the loss on that data point.

In the **finite-sum setting** (empirical risk minimization):

$$F(\boldsymbol{\theta}) = \frac{1}{n}\sum_{i=1}^n f_i(\boldsymbol{\theta})$$

where $f_i(\boldsymbol{\theta}) = \ell(\boldsymbol{\theta}; \mathbf{x}_i, y_i)$ is the loss on the $i$-th training example.

### 2.2 Stochastic Gradient Oracle

**Definition (Stochastic Gradient Oracle).** A stochastic gradient oracle returns a random vector $\mathbf{g}(\boldsymbol{\theta}; \xi)$ such that:

$$\mathbb{E}_{\xi}[\mathbf{g}(\boldsymbol{\theta}; \xi)] = \nabla F(\boldsymbol{\theta})$$

The simplest stochastic gradient oracle samples a single example uniformly at random:

$$\mathbf{g}(\boldsymbol{\theta}; i) = \nabla f_i(\boldsymbol{\theta}) \quad \text{where } i \sim \text{Uniform}(\{1, \ldots, n\})$$

**Unbiasedness proof:**

$$\mathbb{E}_i[\nabla f_i(\boldsymbol{\theta})] = \frac{1}{n}\sum_{i=1}^n \nabla f_i(\boldsymbol{\theta}) = \nabla F(\boldsymbol{\theta})$$

### 2.3 Unbiasedness and Variance of Gradient Estimates

The **variance** of the stochastic gradient is:

$$\sigma^2(\boldsymbol{\theta}) = \mathbb{E}_{\xi}[\lVert \mathbf{g}(\boldsymbol{\theta}; \xi) - \nabla F(\boldsymbol{\theta}) \rVert_2^2]$$

This variance is the key quantity that determines SGD's convergence rate. High variance means noisy updates and slower convergence; low variance means updates are closer to the true gradient.

**Bounded variance assumption:** A common assumption in SGD analysis is that the variance is uniformly bounded:

$$\mathbb{E}_{\xi}[\lVert \mathbf{g}(\boldsymbol{\theta}; \xi) - \nabla F(\boldsymbol{\theta}) \rVert_2^2] \leq \sigma^2 \quad \forall \boldsymbol{\theta}$$

This assumption holds for many ML problems with bounded gradients (e.g., logistic regression with bounded features).

### 2.4 Mini-Batch Gradient Estimation

The **mini-batch gradient** averages $B$ independent stochastic gradients:

$$\mathbf{g}_B(\boldsymbol{\theta}) = \frac{1}{B}\sum_{j=1}^B \mathbf{g}(\boldsymbol{\theta}; \xi_j)$$

**Properties:**
- **Unbiased:** $\mathbb{E}[\mathbf{g}_B(\boldsymbol{\theta})] = \nabla F(\boldsymbol{\theta})$
- **Variance reduction:** $\mathbb{E}[\lVert \mathbf{g}_B(\boldsymbol{\theta}) - \nabla F(\boldsymbol{\theta}) \rVert_2^2] = \frac{\sigma^2(\boldsymbol{\theta})}{B}$

The variance decreases linearly with batch size. However, the per-iteration cost increases linearly with $B$, so there is a trade-off between iteration cost and convergence speed.

### 2.5 Expected Risk vs. Empirical Risk

The **expected risk** is the true objective we care about:

$$F(\boldsymbol{\theta}) = \mathbb{E}_{(\mathbf{x}, y) \sim \mathcal{D}}[\ell(\boldsymbol{\theta}; \mathbf{x}, y)]$$

The **empirical risk** is the finite-sample approximation:

$$\hat{F}_n(\boldsymbol{\theta}) = \frac{1}{n}\sum_{i=1}^n \ell(\boldsymbol{\theta}; \mathbf{x}_i, y_i)$$

SGD can be analyzed in two regimes:

1. **Optimization perspective:** Minimize the empirical risk $\hat{F}_n(\boldsymbol{\theta})$. Here, the stochasticity comes from subsampling the finite dataset.

2. **Statistical learning perspective:** Minimize the expected risk $F(\boldsymbol{\theta})$. Here, the stochasticity comes from sampling from the true data distribution, and the generalization gap $\hat{F}_n(\boldsymbol{\theta}) - F(\boldsymbol{\theta})$ is a key concern.

**For AI:** In deep learning, we typically care about the expected risk (generalization performance), not just the empirical risk (training accuracy). The noise in SGD acts as a form of regularization that reduces the generalization gap, which is why SGD often generalizes better than full-batch GD.

---

## 3. Convergence Theory for SGD

### 3.1 SGD with Diminishing Step Sizes

The classic convergence analysis of SGD uses diminishing step sizes $\eta_t$ that satisfy:

$$\sum_{t=0}^\infty \eta_t = \infty \quad \text{and} \quad \sum_{t=0}^\infty \eta_t^2 < \infty$$

The canonical choice is $\eta_t = \frac{c}{t + c_0}$ for constants $c > 0$ and $c_0 \geq 0$.

**Theorem (SGD Convergence with Diminishing Step Sizes).** Let $F$ be convex with bounded subgradients: $\lVert \mathbf{g}(\boldsymbol{\theta}; \xi) \rVert_2 \leq G$ for all $\boldsymbol{\theta}, \xi$. For SGD with $\eta_t = \frac{c}{\sqrt{t+1}}$:

$$\mathbb{E}[F(\bar{\boldsymbol{\theta}}_T)] - F^* \leq \frac{\lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert_2^2}{2c\sqrt{T}} + \frac{cG^2}{\sqrt{T}} = O\left(\frac{1}{\sqrt{T}}\right)$$

where $\bar{\boldsymbol{\theta}}_T = \frac{1}{T}\sum_{t=0}^{T-1} \boldsymbol{\theta}_t$ is the averaged iterate.

**Proof sketch.** The key inequality is:

$$\lVert \boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^* \rVert_2^2 \leq \lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert_2^2 - 2\eta_t(F(\boldsymbol{\theta}_t) - F^*) + \eta_t^2 \lVert \mathbf{g}_t \rVert_2^2$$

Taking expectations and using $\mathbb{E}[\lVert \mathbf{g}_t \rVert_2^2] \leq G^2$:

$$\mathbb{E}[\lVert \boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^* \rVert_2^2] \leq \mathbb{E}[\lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert_2^2] - 2\eta_t(\mathbb{E}[F(\boldsymbol{\theta}_t)] - F^*) + \eta_t^2 G^2$$

Summing from $t = 0$ to $T-1$ and using convexity of $F$:

$$\mathbb{E}[F(\bar{\boldsymbol{\theta}}_T)] - F^* \leq \frac{\lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert_2^2}{2\sum_{t=0}^{T-1} \eta_t} + \frac{G^2 \sum_{t=0}^{T-1} \eta_t^2}{2\sum_{t=0}^{T-1} \eta_t}$$

With $\eta_t = c/\sqrt{t+1}$, we have $\sum_{t=0}^{T-1} \eta_t \geq c\sqrt{T}$ and $\sum_{t=0}^{T-1} \eta_t^2 \leq c^2 \log T$. Setting $c = \lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert_2 / G$ gives the bound. $\blacksquare$

### 3.2 O(1/√T) Rate for Convex Non-Smooth Functions

The $O(1/\sqrt{T})$ rate is optimal for non-smooth convex optimization with a stochastic first-order oracle. This matches the lower bound for this problem class.

**Theorem (Optimality of SGD for Non-Smooth Convex).** For any algorithm that accesses $F$ through a stochastic subgradient oracle, there exists a convex, $G$-Lipschitz function such that:

$$\mathbb{E}[F(\boldsymbol{\theta}_T)] - F^* \geq \frac{G \lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert_2}{2\sqrt{T}}$$

This lower bound shows that SGD's $O(1/\sqrt{T})$ rate cannot be improved without additional assumptions (such as smoothness or strong convexity).

**For AI:** In deep learning, the loss is typically smooth (differentiable with Lipschitz gradient), so the $O(1/\sqrt{T})$ rate for non-smooth functions is a worst-case bound. The actual convergence rate for neural networks depends on the local geometry of the loss landscape, which we analyze in [Optimization Landscape](../06-Optimization-Landscape/notes.md).

### 3.3 O(1/T) Rate for Strongly Convex Smooth Functions

When $F$ is both $\mu$-strongly convex and $L$-smooth, SGD converges faster.

**Theorem (SGD for Strongly Convex Smooth Functions).** Let $F$ be $\mu$-strongly convex and $L$-smooth with bounded gradient variance $\sigma^2$. For SGD with $\eta_t = \frac{2}{\mu(t+2)}$:

$$\mathbb{E}[\lVert \boldsymbol{\theta}_T - \boldsymbol{\theta}^* \rVert_2^2] \leq \frac{\max(\lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert_2^2, \sigma^2/\mu^2)}{T} = O\left(\frac{1}{T}\right)$$

**Proof sketch.** Using the strong convexity and smoothness inequalities:

$$\lVert \boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^* \rVert_2^2 \leq \lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert_2^2 - 2\eta_t(F(\boldsymbol{\theta}_t) - F^*) - \eta_t\mu\lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert_2^2 + \eta_t^2 \sigma^2$$

Rearranging and taking expectations:

$$\mathbb{E}[\lVert \boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^* \rVert_2^2] \leq (1 - \eta_t\mu)\mathbb{E}[\lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert_2^2] + \eta_t^2 \sigma^2$$

With $\eta_t = \frac{2}{\mu(t+2)}$, this recurrence can be solved by induction to give the $O(1/T)$ bound. $\blacksquare$

**Interpretation:** The $O(1/T)$ rate is a significant improvement over $O(1/\sqrt{T})$. However, the constant depends on the variance $\sigma^2$, which does not decrease with $T$. This is the fundamental limitation of vanilla SGD: it converges to a neighborhood of the optimum whose size depends on the gradient variance.

### 3.4 The Role of Gradient Variance in Convergence

The gradient variance $\sigma^2$ is the key quantity that determines SGD's convergence behavior. We can decompose it as:

$$\sigma^2(\boldsymbol{\theta}) = \mathbb{E}[\lVert \nabla f_i(\boldsymbol{\theta}) - \nabla F(\boldsymbol{\theta}) \rVert_2^2] = \frac{1}{n}\sum_{i=1}^n \lVert \nabla f_i(\boldsymbol{\theta}) - \nabla F(\boldsymbol{\theta}) \rVert_2^2$$

**At the optimum:** If $\boldsymbol{\theta}^*$ interpolates the data (i.e., $\nabla f_i(\boldsymbol{\theta}^*) = \mathbf{0}$ for all $i$), then $\sigma^2(\boldsymbol{\theta}^*) = 0$ and SGD converges linearly near the optimum. This is the **interpolation regime**, which occurs in overparameterized neural networks.

**General case:** When $\sigma^2(\boldsymbol{\theta}^*) > 0$, SGD converges to a neighborhood of the optimum with radius proportional to $\eta \sigma^2$. This is why diminishing step sizes are needed for exact convergence.

**For AI:** Overparameterized neural networks often operate in the interpolation regime: they can achieve zero training loss, meaning $\nabla f_i(\boldsymbol{\theta}^*) \approx \mathbf{0}$ for all training examples. In this regime, SGD with constant step size converges linearly, explaining why deep learning training curves show rapid initial convergence followed by a plateau.

### 3.5 Lower Bounds for Stochastic First-Order Methods

**Theorem (Information-Theoretic Lower Bound).** For any stochastic first-order method and any $T < n$, there exists a convex, $L$-smooth function such that:

$$\mathbb{E}[F(\boldsymbol{\theta}_T)] - F^* \geq \Omega\left(\frac{L \lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert_2^2}{T^2} + \frac{\sigma \lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert_2}{\sqrt{T}}\right)$$

This lower bound has two terms:
1. **Optimization term:** $O(1/T^2)$ — the same rate as accelerated GD in the deterministic setting
2. **Statistical term:** $O(1/\sqrt{T})$ — the irreducible error from gradient noise

The statistical term dominates for large $T$, which is why variance reduction methods (which reduce $\sigma$) are essential for fast convergence.

---

## 4. Mini-Batch SGD

### 4.1 Mini-Batch Gradient Estimator

The mini-batch gradient estimator averages $B$ independent stochastic gradients:

$$\mathbf{g}_B(\boldsymbol{\theta}) = \frac{1}{B}\sum_{j=1}^B \nabla f_{i_j}(\boldsymbol{\theta}) \quad \text{where } i_j \sim \text{Uniform}(\{1, \ldots, n\})$$

**Properties:**
- **Unbiased:** $\mathbb{E}[\mathbf{g}_B(\boldsymbol{\theta})] = \nabla F(\boldsymbol{\theta})$
- **Variance:** $\text{Var}(\mathbf{g}_B(\boldsymbol{\theta})) = \frac{\sigma^2(\boldsymbol{\theta})}{B}$
- **Cost:** $O(B)$ per iteration (vs. $O(1)$ for single-sample SGD)

### 4.2 Variance Reduction with Larger Batches

Increasing the batch size reduces the gradient variance proportionally to $1/B$. This has two effects:

1. **Faster convergence per iteration:** With lower variance, each step is closer to the true gradient direction.
2. **Larger stable learning rate:** The maximum stable learning rate increases with batch size, allowing larger steps.

However, the improvement exhibits **diminishing returns**: doubling the batch size from 32 to 64 halves the variance, but doubling from 1024 to 2048 has a much smaller relative effect.

### 4.3 The Linear Scaling Rule

The **linear scaling rule** (Goyal et al., 2017) states that when the batch size is multiplied by $k$, the learning rate should also be multiplied by $k$ to maintain similar training dynamics:

$$\eta_{\text{new}} = k \cdot \eta_{\text{base}}$$

**Justification:** The signal-to-noise ratio (SNR) of the gradient estimate is:

$$\text{SNR} = \frac{\lVert \nabla F(\boldsymbol{\theta}) \rVert_2^2}{\text{Var}(\mathbf{g}_B(\boldsymbol{\theta}))} = \frac{B \lVert \nabla F(\boldsymbol{\theta}) \rVert_2^2}{\sigma^2(\boldsymbol{\theta})}$$

To maintain the same effective step size relative to the noise level, the learning rate should scale with $\sqrt{B}$ for constant SNR, or with $B$ for constant effective progress per unit of computation.

**For AI:** The linear scaling rule is essential for large-batch LLM training. When training with 8× larger batches, the learning rate should be 8× larger. However, this rule breaks down beyond a critical batch size (typically 4K-32K for LLMs), where the generalization quality degrades.

### 4.4 Critical Batch Size and Diminishing Returns

The **critical batch size** $B_{\text{crit}}$ is the point beyond which increasing the batch size provides diminishing returns in terms of convergence speed. It is defined as the batch size where the gradient variance becomes comparable to the squared gradient norm:

$$\frac{\sigma^2(\boldsymbol{\theta})}{B_{\text{crit}}} \approx \lVert \nabla F(\boldsymbol{\theta}) \rVert_2^2$$

Beyond $B_{\text{crit}}$, the gradient estimate is dominated by the signal (true gradient) rather than the noise, and further increases in batch size provide little benefit.

**For AI:** In LLM pretraining, the critical batch size depends on the model size, dataset, and training phase. Typical values range from 4K to 64K tokens per batch. Beyond this point, increasing the batch size wastes compute without improving convergence speed.

### 4.5 Generalization Effects of Batch Size

Empirically, smaller batch sizes often generalize better than larger batch sizes, even when the training loss is the same. This is known as the **generalization gap**.

**Explanations:**

1. **Implicit regularization:** Small-batch SGD explores a wider region of the parameter space, which acts as implicit regularization.

2. **Sharpness avoidance:** The noise in small-batch SGD helps escape sharp minima and settle in flat minima, which generalize better.

3. **Effective number of updates:** For a fixed compute budget, small batches allow more parameter updates, which can lead to better exploration of the loss landscape.

**For AI:** The generalization gap is a key consideration in LLM training. While large batches are necessary for efficient distributed training, they can lead to worse generalization. Techniques like learning rate warmup, gradient clipping, and sharpness-aware minimization help mitigate this gap.

---

## 5. Variance Reduction Methods

### 5.1 The Variance Problem in SGD

The fundamental limitation of SGD is that the gradient variance $\sigma^2$ does not decrease as the algorithm converges. Even near the optimum, the stochastic gradient remains noisy, preventing exact convergence with a constant step size.

**Variance reduction methods** address this by constructing gradient estimators whose variance decreases to zero as the algorithm converges. The key idea is to use information from past iterations to reduce the variance of the current gradient estimate.

### 5.2 SVRG: Stochastic Variance Reduced Gradient

**SVRG** (Johnson & Zhang, 2013) maintains a snapshot of the parameters $\tilde{\boldsymbol{\theta}}$ and computes the full gradient $\nabla F(\tilde{\boldsymbol{\theta}})$ periodically. The variance-reduced gradient estimator is:

$$\mathbf{g}_{\text{SVRG}} = \nabla f_i(\boldsymbol{\theta}_t) - \nabla f_i(\tilde{\boldsymbol{\theta}}) + \nabla F(\tilde{\boldsymbol{\theta}})$$

**Unbiasedness:** $\mathbb{E}[\mathbf{g}_{\text{SVRG}}] = \nabla F(\boldsymbol{\theta}_t) - \nabla F(\tilde{\boldsymbol{\theta}}) + \nabla F(\tilde{\boldsymbol{\theta}}) = \nabla F(\boldsymbol{\theta}_t)$.

**Variance:** When $\boldsymbol{\theta}_t$ is close to $\tilde{\boldsymbol{\theta}}$, the terms $\nabla f_i(\boldsymbol{\theta}_t)$ and $\nabla f_i(\tilde{\boldsymbol{\theta}})$ are highly correlated, and their difference has small variance. At the optimum ($\boldsymbol{\theta}_t = \tilde{\boldsymbol{\theta}} = \boldsymbol{\theta}^*$), the variance is exactly zero.

**Algorithm (SVRG):**
1. Initialize $\tilde{\boldsymbol{\theta}}_0$, compute $\nabla F(\tilde{\boldsymbol{\theta}}_0)$
2. For epoch $s = 0, 1, 2, \ldots$:
   - Set $m$ (number of inner iterations, typically $m = 2n$)
   - For $t = 0, \ldots, m-1$:
     - Sample $i_t \sim \text{Uniform}(\{1, \ldots, n\})$
     - $\mathbf{g}_t = \nabla f_{i_t}(\boldsymbol{\theta}_t) - \nabla f_{i_t}(\tilde{\boldsymbol{\theta}}_s) + \nabla F(\tilde{\boldsymbol{\theta}}_s)$
     - $\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \mathbf{g}_t$
   - Set $\tilde{\boldsymbol{\theta}}_{s+1} = \boldsymbol{\theta}_m$ (or the average of inner iterates)

**Convergence:** For $\mu$-strongly convex and $L$-smooth functions, SVRG converges linearly:

$$\mathbb{E}[F(\tilde{\boldsymbol{\theta}}_s)] - F^* \leq \rho^s (F(\tilde{\boldsymbol{\theta}}_0) - F^*)$$

where $\rho < 1$ depends on the condition number and the choice of $m$.

### 5.3 SAGA: Stochastic Average Gradient Amélioré

**SAGA** (Defazio et al., 2014) is a variant of SVRG that avoids the periodic full gradient computation by maintaining a table of past gradients.

**Gradient estimator:**

$$\mathbf{g}_{\text{SAGA}} = \nabla f_i(\boldsymbol{\theta}_t) - \nabla f_i(\boldsymbol{\theta}_{t_i}) + \frac{1}{n}\sum_{j=1}^n \nabla f_j(\boldsymbol{\theta}_{t_j})$$

where $t_i$ is the last iteration when example $i$ was sampled, and $\boldsymbol{\theta}_{t_i}$ is the parameter value at that time.

**Algorithm (SAGA):**
1. Initialize $\boldsymbol{\theta}_0$, compute and store $\nabla f_i(\boldsymbol{\theta}_0)$ for all $i$
2. For $t = 0, 1, 2, \ldots$:
   - Sample $i_t \sim \text{Uniform}(\{1, \ldots, n\})$
   - $\mathbf{g}_t = \nabla f_{i_t}(\boldsymbol{\theta}_t) - \nabla f_{i_t}(\boldsymbol{\theta}_{t_{i_t}}) + \frac{1}{n}\sum_{j=1}^n \nabla f_j(\boldsymbol{\theta}_{t_j})$
   - $\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \mathbf{g}_t$
   - Update the stored gradient: $\nabla f_{i_t}(\boldsymbol{\theta}_{t_{i_t}}) \leftarrow \nabla f_{i_t}(\boldsymbol{\theta}_t)$

**Advantages over SVRG:**
- No periodic full gradient computation
- Simpler implementation
- Same linear convergence rate

**Memory cost:** SAGA requires $O(nd)$ memory to store the gradient table, which can be prohibitive for large $n$ and $d$. SVRG requires only $O(d)$ memory.

### 5.4 SDCA: Stochastic Dual Coordinate Ascent

**SDCA** (Shalev-Shwartz & Zhang, 2013) takes a dual approach to variance reduction. Instead of directly minimizing the primal objective, it maximizes the dual objective using stochastic coordinate ascent.

For the regularized empirical risk minimization problem:

$$\min_{\boldsymbol{\theta}} \frac{1}{n}\sum_{i=1}^n \ell_i(\boldsymbol{\theta}) + \frac{\lambda}{2}\lVert \boldsymbol{\theta} \rVert_2^2$$

The dual problem is:

$$\max_{\boldsymbol{\alpha}} -\frac{1}{n}\sum_{i=1}^n \ell_i^*(-\alpha_i) - \frac{\lambda}{2}\left\lVert \frac{1}{\lambda n}\sum_{i=1}^n \alpha_i \mathbf{x}_i \right\rVert_2^2$$

where $\ell_i^*$ is the convex conjugate of $\ell_i$.

SDCA updates one dual coordinate $\alpha_i$ at a time, which corresponds to a variance-reduced update in the primal. The key advantage is that SDCA has no hyperparameters (no step size to tune) and converges linearly.

### 5.5 Comparison: SVRG vs. SAGA vs. SGD

| Method | Per-iter Cost | Memory | Convergence Rate | Best For |
| --- | --- | --- | --- | --- |
| **SGD** | $O(d)$ | $O(d)$ | $O(1/T)$ (strongly cvx) | Large-scale, non-convex (DL) |
| **SVRG** | $O(d)$ | $O(d)$ | Linear (strongly cvx) | Finite-sum, strongly convex |
| **SAGA** | $O(d)$ | $O(nd)$ | Linear (strongly cvx) | Moderate $n$, strongly convex |
| **SDCA** | $O(d)$ | $O(n)$ | Linear (strongly cvx) | Regularized ERM, no tuning |

**For AI:** Variance reduction methods are most useful for finite-sum problems with moderate dataset sizes (e.g., logistic regression, SVM, matrix factorization). For deep learning, the non-convex landscape and the effectiveness of adaptive methods (Adam) make variance reduction less common. However, SVRG-inspired techniques are used in some large-batch training scenarios.

---

## 6. Momentum in the Stochastic Setting

### 6.1 SGD with Momentum: Theory and Practice

SGD with momentum combines the variance-reduction effect of momentum with the computational efficiency of stochastic gradients:

$$\mathbf{v}_{t+1} = \beta \mathbf{v}_t + \mathbf{g}_t$$
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \mathbf{v}_{t+1}$$

where $\mathbf{g}_t$ is the stochastic gradient at step $t$.

**Effect on variance:** Momentum acts as a low-pass filter on the gradient noise. The effective variance of the momentum update is:

$$\text{Var}(\mathbf{v}_t) = \frac{\sigma^2}{1 - \beta^2}$$

For $\beta = 0.9$, this is approximately $0.5\sigma^2$ — a 2× variance reduction compared to vanilla SGD. However, the effective step size is also amplified by $1/(1-\beta)$, which must be accounted for in the learning rate.

### 6.2 Heavy Ball vs. Nesterov in Stochastic Settings

In the deterministic setting, Nesterov acceleration achieves $O(1/T^2)$ convergence vs. $O(1/T)$ for heavy ball momentum. In the stochastic setting, this advantage is less clear:

1. **Heavy ball momentum** is more robust to gradient noise because it only uses the current gradient.

2. **Nesterov momentum** evaluates the gradient at a lookahead position, which can be problematic when the gradient is noisy — the lookahead position may be far from the true gradient direction.

**For AI:** Most deep learning frameworks use heavy ball momentum (PyTorch's SGD with momentum) rather than Nesterov momentum. Adam uses a form of heavy ball momentum combined with adaptive learning rates. Nesterov momentum is available in some frameworks but is less commonly used in practice.

### 6.3 The Bias-Variance Trade-off of Momentum

Momentum introduces a bias-variance trade-off:

- **Bias:** The momentum term introduces a bias toward the average of past gradients, which can slow convergence if the gradient direction changes rapidly.
- **Variance:** Momentum reduces the variance of the gradient estimate, which improves convergence stability.

The optimal momentum coefficient $\beta$ balances these two effects. For strongly convex problems, the optimal $\beta$ is:

$$\beta^* = \left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^2$$

For non-convex problems (deep learning), the optimal $\beta$ is typically chosen empirically. The standard value $\beta = 0.9$ works well for most problems.

### 6.4 Lookahead and Averaging Methods

**Lookahead optimizer** (Zhang et al., 2019) maintains a "slow" set of parameters that are updated by interpolating with the "fast" parameters after $k$ steps of an inner optimizer (typically SGD):

$$\boldsymbol{\theta}_{\text{slow}}^{(s+1)} = \boldsymbol{\theta}_{\text{slow}}^{(s)} + \alpha (\boldsymbol{\theta}_{\text{fast}}^{(k)} - \boldsymbol{\theta}_{\text{slow}}^{(s)})$$

This can be viewed as a form of iterate averaging, which reduces the variance of the final solution.

**Polyak-Ruppert averaging** averages the iterates over time:

$$\bar{\boldsymbol{\theta}}_T = \frac{1}{T}\sum_{t=0}^{T-1} \boldsymbol{\theta}_t$$

This achieves the optimal $O(1/T)$ rate for strongly convex problems and provides a simple variance reduction mechanism.

### 6.5 Why Momentum Helps SGD Escape Saddle Points

At a saddle point, the gradient is zero but the Hessian has negative eigenvalues. Vanilla SGD with small noise can get stuck at saddle points for exponentially long time. Momentum helps escape saddle points by accumulating velocity in the direction of negative curvature.

**Intuition:** When the gradient is near zero, the momentum term carries the iterate forward. If the momentum happens to point in the direction of negative curvature (which has non-zero probability due to the gradient noise), the iterate will accelerate away from the saddle point.

**For AI:** This is one reason why SGD with momentum is more effective than vanilla SGD for training deep neural networks. The non-convex loss landscape of deep networks has exponentially many saddle points, and momentum helps navigate through them efficiently.

---

## 7. Distributed Stochastic Optimization

### 7.1 Data Parallelism and Gradient Averaging

In **data parallel SGD**, the dataset is partitioned across $K$ workers, each of which computes a mini-batch gradient on its local data. The gradients are then averaged to produce the global update:

$$\mathbf{g}_t = \frac{1}{K}\sum_{k=1}^K \mathbf{g}_t^{(k)}$$
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \mathbf{g}_t$$

where $\mathbf{g}_t^{(k)}$ is the mini-batch gradient computed by worker $k$.

**Effective batch size:** The effective batch size is $B_{\text{eff}} = K \cdot B_{\text{local}}$, where $B_{\text{local}}$ is the per-worker batch size. The gradient variance is reduced by a factor of $K$ compared to single-worker SGD.

**Communication cost:** Each iteration requires sending $O(n_{\text{params}})$ data between workers (the gradients). For large models ($n_{\text{params}} = 10^{11}$), this is the primary bottleneck.

### 7.2 Synchronous vs. Asynchronous SGD

**Synchronous SGD** waits for all workers to complete their gradient computation before averaging and updating. This ensures that all workers use the same parameter values, but the speed is limited by the slowest worker (the "straggler problem").

**Asynchronous SGD** (Hogwild!, Recht et al., 2011) allows workers to update the global parameters independently without waiting for others. This eliminates the straggler problem but introduces **stale gradients**: a worker may compute a gradient based on outdated parameters.

**Stale gradient analysis:** If the staleness (the number of updates between when a worker reads the parameters and when it writes the update) is bounded by $\tau$, the convergence rate is:

$$\mathbb{E}[F(\bar{\boldsymbol{\theta}}_T)] - F^* \leq O\left(\frac{1}{\sqrt{T}} + \frac{\tau}{T}\right)$$

For small $\tau$ relative to $T$, the impact of staleness is negligible. However, for large-scale distributed training with many workers, $\tau$ can be significant.

### 7.3 Communication-Efficient SGD: Quantization and Sparsification

To reduce the communication bottleneck, several techniques compress the gradients before transmission:

**Gradient quantization:** Represent each gradient component with fewer bits. For example, 1-bit SGD (Seide et al., 2014) quantizes each gradient component to $\{-1, +1\}$:

$$Q(g_i) = \text{sign}(g_i) \cdot \lVert \mathbf{g} \rVert_1 / d$$

This reduces the communication cost by a factor of 32 (from 32-bit floats to 1-bit signs).

**Gradient sparsification:** Only transmit the top-$k$ largest gradient components (by magnitude). This reduces the communication cost by a factor of $d/k$.

**Error feedback:** To compensate for the information lost during compression, track the compression error and add it to the next iteration's gradient:

$$\mathbf{e}_{t+1} = \mathbf{g}_t - \text{compress}(\mathbf{g}_t + \mathbf{e}_t)$$

Error feedback ensures that the compressed gradient is asymptotically unbiased, preserving convergence guarantees.

### 7.4 Federated Learning and Local SGD

**Federated learning** (McMahan et al., 2017) addresses the setting where data is distributed across many devices (e.g., smartphones) and cannot be centralized. Each device performs local SGD updates on its own data, and the server periodically averages the model parameters:

1. Server sends global model $\boldsymbol{\theta}$ to selected devices
2. Each device performs $E$ local SGD steps: $\boldsymbol{\theta}_i \leftarrow \boldsymbol{\theta}_i - \eta \mathbf{g}_i$
3. Server averages: $\boldsymbol{\theta} \leftarrow \frac{1}{K}\sum_{i=1}^K \boldsymbol{\theta}_i$

**Local SGD** (Stich, 2019) analyzes the convergence of this approach. The key insight is that performing multiple local steps between communication rounds reduces the communication cost while maintaining convergence, as long as the local steps don't diverge too far from each other.

**For AI:** Federated learning is used for training models on mobile devices (Google's Gboard keyboard, Apple's Siri) while preserving user privacy. The communication efficiency and privacy guarantees make it ideal for sensitive data.

### 7.5 Decentralized SGD and Gossip Algorithms

In **decentralized SGD**, there is no central server. Workers communicate only with their neighbors in a communication graph and reach consensus through gossip averaging:

$$\boldsymbol{\theta}_i^{(t+1)} = \sum_{j \in \mathcal{N}(i)} W_{ij} \boldsymbol{\theta}_j^{(t)} - \eta \mathbf{g}_i^{(t)}$$

where $W$ is a doubly stochastic mixing matrix and $\mathcal{N}(i)$ is the set of neighbors of worker $i$.

**Convergence:** Decentralized SGD converges at the same rate as centralized SGD, but with an additional term that depends on the spectral gap of the communication graph. Well-connected graphs (small diameter, large spectral gap) converge faster.

**For AI:** Decentralized SGD is useful for training models across data centers with limited bandwidth, or in settings where a central server is not available or desirable (privacy-preserving distributed learning).

---

## 8. Advanced Topics

### 8.1 SGD Implicit Bias and Generalization

One of the most surprising properties of SGD is its **implicit bias** toward solutions that generalize well, even in the absence of explicit regularization.

**Linear models:** For overparameterized linear regression, SGD with small step size converges to the minimum $\ell_2$-norm solution, which has the best generalization properties.

**Deep neural networks:** SGD tends to find solutions in flat regions of the loss landscape (flat minima), which are robust to perturbations and generalize better than sharp minima. This is attributed to the gradient noise, which acts as a temperature that pushes the solution away from sharp minima.

**Theoretical explanation:** The gradient noise covariance in SGD is not isotropic — it is aligned with the Hessian of the loss. This means the noise is larger in directions of high curvature (sharp directions) and smaller in directions of low curvature (flat directions). This anisotropic noise naturally biases SGD toward flat minima.

### 8.2 Sharpness-Aware Minimization (SAM)

**SAM** (Foret et al., 2021) explicitly optimizes for flat minima by minimizing the loss in a neighborhood around the current parameters:

$$\min_{\boldsymbol{\theta}} \max_{\lVert \boldsymbol{\epsilon} \rVert_2 \leq \rho} \mathcal{L}(\boldsymbol{\theta} + \boldsymbol{\epsilon})$$

The inner maximization finds the worst-case perturbation within a ball of radius $\rho$, and the outer minimization finds parameters that are robust to this perturbation.

**Algorithm:**
1. Compute the perturbation: $\boldsymbol{\epsilon}^* = \rho \cdot \frac{\nabla \mathcal{L}(\boldsymbol{\theta})}{\lVert \nabla \mathcal{L}(\boldsymbol{\theta}) \rVert_2}$
2. Compute the gradient at the perturbed point: $\mathbf{g}_{\text{SAM}} = \nabla \mathcal{L}(\boldsymbol{\theta} + \boldsymbol{\epsilon}^*)$
3. Update: $\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \mathbf{g}_{\text{SAM}}$

**For AI:** SAM has been shown to improve generalization across a wide range of tasks, including image classification, language modeling, and transfer learning. It is particularly effective when combined with large-batch training, where the generalization gap is most pronounced.

### 8.3 Stochastic Second-Order Methods

Second-order methods can be adapted to the stochastic setting by using stochastic estimates of the Hessian or its inverse:

**Stochastic Newton:** Use a mini-batch Hessian estimate $\hat{H}_t$ and solve $\hat{H}_t \mathbf{d}_t = -\mathbf{g}_t$.

**Stochastic L-BFGS:** Maintain a limited-memory Hessian approximation using stochastic gradient differences.

**Stochastic K-FAC:** Use mini-batch estimates of the Kronecker factors in the K-FAC approximation.

**Challenge:** The noise in the Hessian estimate is typically much larger than the noise in the gradient estimate, making stochastic second-order methods less stable than their first-order counterparts.

### 8.4 Adaptive Step Sizes for SGD

Several methods adapt the learning rate based on the observed gradient statistics:

**AdaGrad** (Duchi et al., 2011): Accumulates squared gradients and scales the learning rate:

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \frac{\eta}{\sqrt{\mathbf{v}_t + \epsilon}} \odot \mathbf{g}_t, \quad \mathbf{v}_t = \mathbf{v}_{t-1} + \mathbf{g}_t^2$$

**RMSProp** (Tieleman & Hinton, 2012): Uses an exponential moving average of squared gradients:

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + (1-\beta)\mathbf{g}_t^2$$

**Adam** (Kingma & Ba, 2015): Combines momentum with RMSProp:

$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\mathbf{g}_t, \quad \mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1-\beta_2)\mathbf{g}_t^2$$

These methods are covered in detail in [Adaptive Learning Rate](../07-Adaptive-Learning-Rate/notes.md).

### 8.5 Connection to Langevin Dynamics and Bayesian Inference

SGD with injected noise is equivalent to **Stochastic Gradient Langevin Dynamics (SGLD)**:

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \frac{\eta}{2} \nabla \log p(\boldsymbol{\theta}_t \mid \mathcal{D}) + \boldsymbol{\epsilon}_t, \quad \boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \eta \mathbf{I})$$

In the continuous-time limit, this is the Langevin diffusion:

$$d\boldsymbol{\theta}_t = -\frac{1}{2}\nabla \log p(\boldsymbol{\theta}_t \mid \mathcal{D}) dt + d\mathbf{W}_t$$

where $\mathbf{W}_t$ is a standard Brownian motion. The stationary distribution of this diffusion is the posterior $p(\boldsymbol{\theta} \mid \mathcal{D})$.

**For AI:** This connection reveals that SGD can be interpreted as approximate Bayesian inference. The gradient noise in SGD plays the role of the Langevin noise, and the stationary distribution of SGD parameters approximates the posterior distribution. This provides a theoretical foundation for understanding SGD's uncertainty quantification properties.

---

## 9. Applications in Machine Learning

### 9.1 Training Deep Neural Networks with SGD

SGD (with momentum) is the workhorse of deep learning training. The standard training loop is:

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass
        predictions = model(batch.inputs)
        loss = loss_fn(predictions, batch.targets)
        
        # Backward pass
        loss.backward()
        
        # SGD with momentum update
        for param in model.parameters():
            param.grad_buffer = momentum * param.grad_buffer + param.grad
            param.data -= learning_rate * param.grad_buffer
            param.grad.zero_()
```

**Key practical considerations:**
- **Learning rate scheduling:** Use warmup + cosine decay or step decay
- **Gradient clipping:** Clip the gradient norm to prevent exploding gradients
- **Batch size:** Choose based on the critical batch size for the problem
- **Momentum:** Use $\beta = 0.9$ for most problems

### 9.2 Large-Scale Logistic Regression

For logistic regression with $n$ examples and $d$ features, SGD is the method of choice when $n$ is large:

$$\min_{\mathbf{w}} \frac{1}{n}\sum_{i=1}^n \log(1 + e^{-y_i \mathbf{w}^\top \mathbf{x}_i}) + \frac{\lambda}{2}\lVert \mathbf{w} \rVert_2^2$$

The per-iteration cost is $O(d)$ (vs. $O(nd)$ for full gradient), making SGD practical for $n = 10^9$ and $d = 10^6$.

**Convergence:** With $\eta_t = O(1/t)$, SGD converges at rate $O(1/T)$ for this strongly convex problem. With variance reduction (SVRG/SAGA), the convergence is linear.

### 9.3 Matrix Factorization with SGD

Matrix factorization (used in recommender systems) decomposes a rating matrix $R \in \mathbb{R}^{m \times n}$ as $R \approx UV^\top$:

$$\min_{U, V} \sum_{(i,j) \in \Omega} (R_{ij} - \mathbf{u}_i^\top \mathbf{v}_j)^2 + \lambda(\lVert U \rVert_F^2 + \lVert V \rVert_F^2)$$

where $\Omega$ is the set of observed entries. SGD updates one $(i, j)$ pair at a time:

$$\mathbf{u}_i \leftarrow \mathbf{u}_i + \eta (R_{ij} - \mathbf{u}_i^\top \mathbf{v}_j) \mathbf{v}_j - \eta \lambda \mathbf{u}_i$$
$$\mathbf{v}_j \leftarrow \mathbf{v}_j + \eta (R_{ij} - \mathbf{u}_i^\top \mathbf{v}_j) \mathbf{u}_i - \eta \lambda \mathbf{v}_j$$

This is the algorithm behind the Netflix Prize winning solution and is still widely used in industry recommender systems.

### 9.4 Reinforcement Learning: Stochastic Policy Gradients

In reinforcement learning, the policy gradient theorem gives:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot Q^{\pi_\theta}(s_t, a_t)]$$

The expectation is estimated by sampling trajectories, making this a stochastic optimization problem. Variance reduction techniques (baseline subtraction, advantage estimation, GAE) are essential for practical performance.

**For AI:** Policy gradient methods (REINFORCE, PPO, A3C) are the foundation of modern RL, used in game playing (AlphaGo, Dota 2), robotics, and autonomous driving.

### 9.5 Distributed LLM Pretraining

LLM pretraining is the most demanding application of stochastic optimization:

- **Model size:** 100B to 1T+ parameters
- **Dataset size:** 1T to 10T+ tokens
- **Distributed setup:** Thousands of GPUs across hundreds of nodes
- **Effective batch size:** 1M to 4M tokens per gradient step

**Key techniques:**
- **Data parallelism:** Each GPU processes a mini-batch and gradients are averaged
- **Tensor parallelism:** Model weights are split across GPUs within a node
- **Pipeline parallelism:** Model layers are split across nodes
- **Gradient accumulation:** Multiple forward/backward passes before updating to simulate larger batch sizes
- **Mixed precision:** BF16/FP16 for compute, FP32 for master weights

**For AI:** The choice of optimizer (AdamW vs. SGD) and the learning rate schedule (warmup + cosine decay) are critical hyperparameters that determine the final model quality. The scale of LLM pretraining pushes the limits of distributed stochastic optimization.

---

## 10. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
| --- | --------- | ---------------- | ----- |
| 1 | "SGD always converges slower than full-batch GD" | SGD makes more progress per unit of compute because each iteration is $B/n$ the cost of full-batch GD. For large $n$, SGD reaches a given accuracy much faster in wall-clock time. | Compare algorithms by wall-clock time or total FLOPs, not by number of iterations. |
| 2 | "Larger batch size is always better" | Beyond the critical batch size, increasing the batch size provides diminishing returns and can hurt generalization. The optimal batch size depends on the problem and compute budget. | Use the linear scaling rule up to the critical batch size, then stop increasing. |
| 3 | "SGD with constant step size converges to the optimum" | With constant step size, SGD converges to a neighborhood of the optimum whose size is proportional to $\eta \sigma^2$. Use diminishing step sizes for exact convergence. | Use $\eta_t = O(1/t)$ for strongly convex problems, or use variance reduction methods. |
| 4 | "Momentum always helps SGD" | Too much momentum ($\beta$ too high) can cause oscillations and divergence, especially with large learning rates. Momentum also amplifies gradient bias. | Start with $\beta = 0.9$ and reduce if oscillations are observed. Use bias-corrected momentum (Adam). |
| 5 | "SVRG is always better than SGD" | SVRG has higher per-iteration cost (periodic full gradient computation) and requires storing the snapshot. For non-convex problems (deep learning), SVRG's advantages are less clear. | Use SVRG for finite-sum convex problems. Use SGD or Adam for deep learning. |
| 6 | "Synchronous SGD is always better than asynchronous" | Synchronous SGD is limited by the straggler problem. Asynchronous SGD can be faster in heterogeneous environments, despite stale gradients. | Use synchronous SGD for homogeneous clusters. Use asynchronous SGD for heterogeneous or unreliable workers. |
| 7 | "Gradient compression always hurts convergence" | With error feedback, compressed SGD can achieve the same convergence rate as uncompressed SGD. The communication savings often outweigh the small convergence slowdown. | Use error-compensated compression (e.g., 1-bit SGD with error feedback) for bandwidth-limited settings. |
| 8 | "The generalization gap from large batches can't be fixed" | Techniques like learning rate warmup, SAM, and longer training can close the generalization gap. The gap is not inherent to large batches but to the training dynamics. | Use warmup + longer training + SAM to close the generalization gap for large-batch training. |
| 9 | "SGD noise is always harmful" | SGD noise helps escape saddle points, avoid sharp minima, and provides implicit regularization. The noise is a feature, not a bug. | Tune the batch size to control the noise level: smaller batches for more regularization, larger for faster convergence. |
| 10 | "Variance reduction methods work for deep learning" | VR methods assume the finite-sum structure and convexity. For non-convex deep learning, the variance at the optimum is not zero, limiting the benefit of VR. | Use VR for convex finite-sum problems. For deep learning, use Adam or SGD with momentum. |

---

## 11. Exercises

**Exercise 1: SGD Unbiasedness (★)**
Prove that the mini-batch gradient estimator $\mathbf{g}_B(\boldsymbol{\theta}) = \frac{1}{B}\sum_{j=1}^B \nabla f_{i_j}(\boldsymbol{\theta})$ is unbiased: $\mathbb{E}[\mathbf{g}_B(\boldsymbol{\theta})] = \nabla F(\boldsymbol{\theta})$. Compute its variance.

**Exercise 2: SGD Convergence Rate (★)**
For $f(x) = \frac{1}{2}x^2$ with stochastic gradient $g_t = x_t + \xi_t$ where $\xi_t \sim \mathcal{N}(0, \sigma^2)$, analyze the convergence of SGD with $\eta_t = \frac{1}{t+1}$. Show that $\mathbb{E}[x_t^2] = O(1/t)$.

**Exercise 3: Mini-Batch Variance (★)**
For a dataset of $n = 10,000$ examples with gradient variance $\sigma^2 = 4$, compute the variance of the mini-batch gradient for $B \in \{1, 32, 128, 1024\}$. Plot the variance vs. batch size.

**Exercise 4: SVRG Analysis (★★)**
Implement SVRG for logistic regression on a synthetic dataset. Compare its convergence rate with vanilla SGD and full-batch GD. Verify the linear convergence of SVRG.

**Exercise 5: Linear Scaling Rule (★★)**
Train a small neural network with batch sizes $B \in \{32, 64, 128, 256\}$ and learning rates scaled linearly ($\eta = \eta_0 \cdot B/32$). Verify that the training curves are similar.

**Exercise 6: Momentum in Stochastic Setting (★★)**
Compare SGD with and without momentum on a noisy quadratic problem. Analyze how momentum affects the variance of the iterates and the convergence rate.

**Exercise 7: Distributed SGD with Gradient Compression (★★★)**
Implement distributed SGD with 1-bit gradient quantization and error feedback. Compare the convergence rate and communication cost with uncompressed SGD.

**Exercise 8: SGD Implicit Bias (★★★)**
Train an overparameterized linear model with SGD and full-batch GD from the same initialization. Compare the $\ell_2$ norm of the solutions and verify that SGD finds the minimum-norm solution.

---

## 12. Why This Matters for AI (2026 Perspective)

| Concept | AI Impact |
| --------- | ----------- |
| SGD | The default optimizer for vision models; better generalization than Adam in many settings |
| Mini-batch SGD | Enables training on datasets too large to fit in memory; foundation of all DL training |
| Linear scaling rule | Essential for large-batch LLM training; enables efficient distributed training |
| Critical batch size | Determines the optimal batch size for distributed training; prevents wasted compute |
| SVRG / SAGA | Variance reduction for finite-sum problems; faster convergence for logistic regression, SVM |
| SGD momentum | Standard component of all DL optimizers; helps escape saddle points |
| Synchronous SGD | Foundation of distributed DL training; used in all major LLM training runs |
| Gradient compression | Enables training across bandwidth-limited networks; 1-bit SGD, sparsification |
| Federated learning | Privacy-preserving training on edge devices; Gboard, Siri, healthcare |
| SAM | Explicit flat minimum optimization; improves generalization across all tasks |
| SGLD | Connects SGD to Bayesian inference; uncertainty quantification for DL |
| Generalization gap | Understanding why small batches generalize better; informs batch size selection |
| AdamW | Common baseline for LLM training; combines stochastic gradients with adaptive learning rates |

---

## 13. Conceptual Bridge

### Backward Connections

Stochastic optimization builds directly on the foundations from [Gradient Descent](../02-Gradient-Descent/notes.md). The convergence analysis of SGD extends the GD convergence theory by accounting for gradient noise. The key insight is that the noise term $\boldsymbol{\xi}_t$ in the SGD update $\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta(\nabla F(\boldsymbol{\theta}_t) + \boldsymbol{\xi}_t)$ adds an extra variance term to the convergence bound.

From [Convex Optimization](../01-Convex-Optimization/notes.md), we use strong convexity and smoothness throughout the convergence analysis. The condition number $\kappa = L/\mu$ determines both the convergence rate of GD and the optimal learning rate for SGD.

From [Probability Theory](../../06-Probability-Theory/README.md), we use expectations, variance, and concentration inequalities to analyze the stochastic gradient estimator. The law of large numbers ensures that the mini-batch gradient converges to the full gradient as $B \to \infty$.

### Forward Connections

This section connects to several advanced topics:

- **Optimization Landscape** (§06) analyzes how the geometry of the loss landscape interacts with SGD noise. The anisotropic noise in SGD naturally biases toward flat minima.
- **Adaptive Learning Rate** (§07) builds on stochastic gradients by adding per-parameter learning rate adaptation. Adam = SGD momentum + RMSProp.
- **Regularization Methods** (§08) connects through the implicit regularization effect of SGD noise. Small-batch SGD acts as an implicit regularizer.
- **Learning Rate Schedules** (§10) controls the step size of SGD over time. Warmup + cosine decay is the standard schedule for LLM training.

### The Big Picture

```text
STOCHASTIC OPTIMIZATION IN THE CURRICULUM
════════════════════════════════════════════════════════════════════════

              Deterministic GD (§02)
              Full gradient, exact convergence
                       │
                       ▼
              ╔══════════════════════════════════╗
              ║   STOCHASTIC OPTIMIZATION        ║ ← YOU ARE HERE
              ║                                  ║
              ║  SGD: g_t = ∇f_i(θ_t)            ║
              ║  Mini-batch: g_B = (1/B)Σg_i     ║
              ║  Variance: σ²/B                  ║
              ║                                  ║
              ║  Convergence:                    ║
              ║  - Convex: O(1/√T)               ║
              ║  - Strongly cvx: O(1/T)          ║
              ║  - With VR: Linear               ║
              ║                                  ║
              ║  Distributed: sync/async,        ║
              ║  compression, federated          ║
              ╚══════════════════════════════════╝
                       │
            ┌──────────┼──────────┐
            ▼          ▼          ▼
      Landscape     Adaptive    LR Schedules
      (§06 noise    (§07 Adam)  (§10 warmup,
       geometry)                 cosine)
            │          │          │
            ▼          ▼          ▼
      Flat minima   AdamW for   LLM pretraining
      generalization LLMs       at scale

════════════════════════════════════════════════════════════════════════
```

Stochastic optimization is the bridge between the clean theory of deterministic optimization and the messy reality of training models on massive datasets. The noise in SGD is not just a computational necessity — it is a fundamental feature that shapes the generalization properties of the learned model.

---

## References

1. Bottou, L. (2010). "Large-scale machine learning with stochastic gradient descent." COMPSTAT.
2. Robbins, H. & Monro, S. (1951). "A stochastic approximation method." Annals of Mathematical Statistics.
3. Johnson, R. & Zhang, T. (2013). "Accelerating stochastic gradient descent using predictive variance reduction." NeurIPS.
4. Defazio, A., Bach, F. & Lacoste-Julien, S. (2014). "SAGA: A fast incremental gradient method with support for non-strongly convex composite objectives." NeurIPS.
5. Goyal, P. et al. (2017). "Accurate, large mini-batch SGD: Training ImageNet in 1 hour." arXiv.
6. Keskar, N. et al. (2017). "On large-batch training for deep learning: Generalization gap and sharp minima." ICLR.
7. Recht, B. et al. (2011). "Hogwild!: A lock-free approach to parallelizing stochastic gradient descent." NeurIPS.
8. McMahan, B. et al. (2017). "Communication-efficient learning of deep networks from decentralized data." AISTATS.
9. Foret, P. et al. (2021). "Sharpness-aware minimization for efficiently improving generalization." ICLR.
10. Welling, M. & Teh, Y. (2011). "Bayesian learning via stochastic gradient Langevin dynamics." ICML.
11. Kingma, D. & Ba, J. (2015). "Adam: A method for stochastic optimization." ICLR.
12. Smith, S. et al. (2018). "Don't decay the learning rate, increase the batch size." ICLR.
13. Zhang, J. et al. (2019). "Lookahead optimizer: k steps forward, 1 step back." NeurIPS.
14. Stich, S. (2019). "Local SGD converges fast and communicates little." ICLR.
15. Seide, F. et al. (2014). "1-bit stochastic gradient descent and its application to data-parallel distributed training of speech DNNs." Interspeech.

---

## Appendix A: Detailed Proofs and Extended Theory

### A.1 Full Proof of SGD Convergence for Convex Non-Smooth Functions

**Theorem.** Let $F$ be convex with bounded subgradients: $\lVert \mathbf{g}(\boldsymbol{\theta}; \xi) \rVert_2 \leq G$ for all $\boldsymbol{\theta}, \xi$. For SGD with step size $\eta_t = \frac{c}{\sqrt{t+1}}$:

$$\mathbb{E}[F(\bar{\boldsymbol{\theta}}_T)] - F^* \leq \frac{\lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert_2^2}{2c\sqrt{T}} + \frac{cG^2}{\sqrt{T}}$$

**Proof.** The SGD update is $\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t$ where $\mathbf{g}_t$ is the stochastic gradient at step $t$.

Expanding the squared distance:

$$\lVert \boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^* \rVert_2^2 = \lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert_2^2 - 2\eta_t \mathbf{g}_t^\top(\boldsymbol{\theta}_t - \boldsymbol{\theta}^*) + \eta_t^2 \lVert \mathbf{g}_t \rVert_2^2$$

Taking conditional expectation given $\boldsymbol{\theta}_t$ and using $\mathbb{E}[\mathbf{g}_t \mid \boldsymbol{\theta}_t] = \nabla F(\boldsymbol{\theta}_t)$:

$$\mathbb{E}[\lVert \boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^* \rVert_2^2 \mid \boldsymbol{\theta}_t] = \lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert_2^2 - 2\eta_t \nabla F(\boldsymbol{\theta}_t)^\top(\boldsymbol{\theta}_t - \boldsymbol{\theta}^*) + \eta_t^2 \mathbb{E}[\lVert \mathbf{g}_t \rVert_2^2 \mid \boldsymbol{\theta}_t]$$

By convexity: $\nabla F(\boldsymbol{\theta}_t)^\top(\boldsymbol{\theta}_t - \boldsymbol{\theta}^*) \geq F(\boldsymbol{\theta}_t) - F^*$.

By bounded gradients: $\mathbb{E}[\lVert \mathbf{g}_t \rVert_2^2 \mid \boldsymbol{\theta}_t] \leq G^2$.

Therefore:

$$\mathbb{E}[\lVert \boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^* \rVert_2^2 \mid \boldsymbol{\theta}_t] \leq \lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert_2^2 - 2\eta_t(F(\boldsymbol{\theta}_t) - F^*) + \eta_t^2 G^2$$

Taking full expectation and rearranging:

$$2\eta_t(\mathbb{E}[F(\boldsymbol{\theta}_t)] - F^*) \leq \mathbb{E}[\lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert_2^2] - \mathbb{E}[\lVert \boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^* \rVert_2^2] + \eta_t^2 G^2$$

Summing from $t = 0$ to $T-1$:

$$2\sum_{t=0}^{T-1} \eta_t(\mathbb{E}[F(\boldsymbol{\theta}_t)] - F^*) \leq \lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert_2^2 + G^2\sum_{t=0}^{T-1} \eta_t^2$$

By Jensen's inequality and convexity of $F$:

$$\mathbb{E}[F(\bar{\boldsymbol{\theta}}_T)] - F^* \leq \frac{\sum_{t=0}^{T-1} \eta_t(\mathbb{E}[F(\boldsymbol{\theta}_t)] - F^*)}{\sum_{t=0}^{T-1} \eta_t} \leq \frac{\lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert_2^2 + G^2\sum_{t=0}^{T-1} \eta_t^2}{2\sum_{t=0}^{T-1} \eta_t}$$

With $\eta_t = c/\sqrt{t+1}$:

$$\sum_{t=0}^{T-1} \eta_t \geq c\int_0^T \frac{1}{\sqrt{t+1}} dt = 2c(\sqrt{T+1} - 1) \geq c\sqrt{T}$$

$$\sum_{t=0}^{T-1} \eta_t^2 = c^2\sum_{t=0}^{T-1} \frac{1}{t+1} \leq c^2(1 + \log T) \leq 2c^2\sqrt{T} \text{ for large } T$$

Substituting:

$$\mathbb{E}[F(\bar{\boldsymbol{\theta}}_T)] - F^* \leq \frac{\lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert_2^2}{2c\sqrt{T}} + \frac{cG^2}{\sqrt{T}} \quad \blacksquare$$

### A.2 Proof of SGD Convergence for Strongly Convex Smooth Functions

**Theorem.** Let $F$ be $\mu$-strongly convex and $L$-smooth with bounded gradient variance $\sigma^2$. For SGD with $\eta_t = \frac{2}{\mu(t+2)}$:

$$\mathbb{E}[\lVert \boldsymbol{\theta}_T - \boldsymbol{\theta}^* \rVert_2^2] \leq \frac{C}{T}$$

where $C = \max(\lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert_2^2, 4\sigma^2/\mu^2)$.

**Proof.** From the SGD update:

$$\lVert \boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^* \rVert_2^2 = \lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert_2^2 - 2\eta_t \mathbf{g}_t^\top(\boldsymbol{\theta}_t - \boldsymbol{\theta}^*) + \eta_t^2 \lVert \mathbf{g}_t \rVert_2^2$$

Taking expectations and using $\mathbb{E}[\mathbf{g}_t \mid \boldsymbol{\theta}_t] = \nabla F(\boldsymbol{\theta}_t)$:

$$\mathbb{E}[\lVert \boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^* \rVert_2^2] \leq \mathbb{E}[\lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert_2^2] - 2\eta_t \mathbb{E}[\nabla F(\boldsymbol{\theta}_t)^\top(\boldsymbol{\theta}_t - \boldsymbol{\theta}^*)] + \eta_t^2 \mathbb{E}[\lVert \mathbf{g}_t \rVert_2^2]$$

By strong convexity: $\nabla F(\boldsymbol{\theta}_t)^\top(\boldsymbol{\theta}_t - \boldsymbol{\theta}^*) \geq F(\boldsymbol{\theta}_t) - F^* + \frac{\mu}{2}\lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert_2^2 \geq \frac{\mu}{2}\lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert_2^2$.

By bounded variance: $\mathbb{E}[\lVert \mathbf{g}_t \rVert_2^2] = \lVert \nabla F(\boldsymbol{\theta}_t) \rVert_2^2 + \sigma^2 \leq 2L(F(\boldsymbol{\theta}_t) - F^*) + \sigma^2$ (using $L$-smoothness).

For the proof, we use the simpler bound $\mathbb{E}[\lVert \mathbf{g}_t \rVert_2^2] \leq G^2$ where $G^2 = \sup_{\boldsymbol{\theta}} \mathbb{E}[\lVert \mathbf{g}_t \rVert_2^2]$. A more refined analysis gives $G^2 = \sigma^2 + \lVert \nabla F(\boldsymbol{\theta}_t) \rVert_2^2$.

Using $\mathbb{E}[\lVert \mathbf{g}_t \rVert_2^2] \leq \sigma^2 + 2L\mathbb{E}[F(\boldsymbol{\theta}_t) - F^*]$:

$$\mathbb{E}[\lVert \boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^* \rVert_2^2] \leq (1 - \eta_t\mu)\mathbb{E}[\lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert_2^2] + \eta_t^2\sigma^2$$

With $\eta_t = \frac{2}{\mu(t+2)}$, we prove by induction that $\mathbb{E}[\lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert_2^2] \leq \frac{C}{t+1}$ where $C = \max(\lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert_2^2, 4\sigma^2/\mu^2)$.

Base case ($t=0$): $\mathbb{E}[\lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert_2^2] \leq C$ by definition.

Inductive step: Assume $\mathbb{E}[\lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert_2^2] \leq \frac{C}{t+1}$. Then:

$$\mathbb{E}[\lVert \boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^* \rVert_2^2] \leq \left(1 - \frac{2}{t+2}\right)\frac{C}{t+1} + \frac{4\sigma^2}{\mu^2(t+2)^2}$$

$$= \frac{tC}{(t+1)(t+2)} + \frac{4\sigma^2}{\mu^2(t+2)^2} \leq \frac{C}{t+2}$$

The last inequality holds because $C \geq 4\sigma^2/\mu^2$. $\blacksquare$

### A.3 SVRG Convergence Analysis

**Theorem (SVRG Linear Convergence).** Let $F(\boldsymbol{\theta}) = \frac{1}{n}\sum_{i=1}^n f_i(\boldsymbol{\theta})$ where each $f_i$ is $L$-smooth and $F$ is $\mu$-strongly convex. For SVRG with epoch length $m = 2n$ and step size $\eta \leq \frac{1}{4L}$:

$$\mathbb{E}[F(\tilde{\boldsymbol{\theta}}_s)] - F^* \leq \rho^s (F(\tilde{\boldsymbol{\theta}}_0) - F^*)$$

where $\rho = \frac{1}{\mu\eta m} + \frac{2L}{\mu m} < 1$ for appropriate choices of $\eta$ and $m$.

**Proof sketch.** The key insight is that the variance of the SVRG gradient estimator decreases as $\boldsymbol{\theta}_t$ approaches $\tilde{\boldsymbol{\theta}}$:

$$\mathbb{E}[\lVert \mathbf{g}_t \rVert_2^2] = \mathbb{E}[\lVert \nabla f_{i_t}(\boldsymbol{\theta}_t) - \nabla f_{i_t}(\tilde{\boldsymbol{\theta}}) + \nabla F(\tilde{\boldsymbol{\theta}}) \rVert_2^2]$$

$$= \mathbb{E}[\lVert \nabla f_{i_t}(\boldsymbol{\theta}_t) - \nabla f_{i_t}(\tilde{\boldsymbol{\theta}}) \rVert_2^2] + \lVert \nabla F(\tilde{\boldsymbol{\theta}}) \rVert_2^2$$

$$\leq 2L(F(\boldsymbol{\theta}_t) - F^*) + 4L(F(\tilde{\boldsymbol{\theta}}) - F^*)$$

Using this variance bound in the SGD convergence analysis gives the linear convergence rate. $\blacksquare$

### A.4 SAGA Convergence Analysis

**Theorem (SAGA Linear Convergence).** Under the same assumptions as SVRG, SAGA with $\eta \leq \frac{1}{2(\mu n + 3L)}$ converges linearly:

$$\mathbb{E}[F(\boldsymbol{\theta}_T) - F^* + \frac{cn}{2}\lVert \boldsymbol{\theta}_T - \boldsymbol{\theta}^* \rVert_2^2] \leq \left(1 - \min\left(\frac{\mu}{2cn}, \frac{1}{2n}\right)\right)^T \cdot \text{initial error}$$

where $c$ is a constant depending on $L$ and $\mu$.

**Key difference from SVRG:** SAGA maintains a table of past gradients, which allows it to avoid the periodic full gradient computation. The convergence rate is similar to SVRG, but the per-iteration cost is more uniform (no expensive full gradient steps).

### A.5 The Interpolation Regime

**Definition (Interpolation).** A model interpolates the training data if there exists $\boldsymbol{\theta}^*$ such that $f_i(\boldsymbol{\theta}^*) = F^*$ for all $i = 1, \ldots, n$. Equivalently, $\nabla f_i(\boldsymbol{\theta}^*) = \mathbf{0}$ for all $i$.

**Theorem (SGD in the Interpolation Regime).** If the interpolation condition holds and $F$ is $\mu$-strongly convex and $L$-smooth, then SGD with constant step size $\eta \leq \frac{1}{L}$ converges linearly:

$$\mathbb{E}[\lVert \boldsymbol{\theta}_T - \boldsymbol{\theta}^* \rVert_2^2] \leq (1 - \eta\mu)^T \lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert_2^2$$

**Proof.** In the interpolation regime, $\nabla f_i(\boldsymbol{\theta}^*) = \mathbf{0}$ for all $i$, so:

$$\mathbb{E}[\lVert \mathbf{g}_t \rVert_2^2] = \mathbb{E}[\lVert \nabla f_{i_t}(\boldsymbol{\theta}_t) - \nabla f_{i_t}(\boldsymbol{\theta}^*) \rVert_2^2] \leq 2L(F(\boldsymbol{\theta}_t) - F^*)$$

Using the same analysis as for strongly convex SGD but with the tighter variance bound gives linear convergence with constant step size. $\blacksquare$

**For AI:** Overparameterized neural networks often operate in the interpolation regime: they can achieve zero training loss, meaning each individual gradient $\nabla f_i(\boldsymbol{\theta}^*) \approx \mathbf{0}$ at the solution. This explains why SGD with constant (or slowly decaying) learning rate converges rapidly in practice — the gradient noise vanishes near the optimum.

### A.6 Gradient Noise Covariance Structure

The gradient noise covariance matrix is:

$$\Sigma(\boldsymbol{\theta}) = \mathbb{E}[(\mathbf{g}(\boldsymbol{\theta}; \xi) - \nabla F(\boldsymbol{\theta}))(\mathbf{g}(\boldsymbol{\theta}; \xi) - \nabla F(\boldsymbol{\theta}))^\top]$$

**Key insight:** The noise is not isotropic. Its structure is aligned with the Hessian of the loss:

$$\Sigma(\boldsymbol{\theta}) \approx \alpha \nabla^2 F(\boldsymbol{\theta}) + \beta I$$

for some constants $\alpha, \beta \geq 0$. This means the noise is larger in directions of high curvature (sharp directions) and smaller in directions of low curvature (flat directions).

**Implications:**
1. **Flat minimum bias:** The anisotropic noise naturally pushes SGD away from sharp minima (where the noise is large) toward flat minima (where the noise is small).
2. **Effective preconditioning:** The noise acts as a natural preconditioner, scaling the gradient by the inverse of the local curvature.
3. **Batch size effect:** The noise covariance scales as $\Sigma/B$, so larger batches reduce the noise uniformly but preserve its anisotropic structure.

**For AI:** This noise structure explains why SGD generalizes better than full-batch GD: the noise helps SGD explore the loss landscape and settle in flat minima, which are more robust to perturbations and generalize better.

---

## Appendix B: Worked Examples and Case Studies

### B.1 SGD vs. Full-Batch GD: A Complete Comparison

Let us compare SGD and full-batch GD on a simple quadratic problem.

**Problem:** $F(\mathbf{x}) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}(\mathbf{a}_i^\top \mathbf{x} - b_i)^2$ where $\mathbf{a}_i \in \mathbb{R}^d$ and $b_i \in \mathbb{R}$.

**Full-batch GD:**
- Gradient: $\nabla F(\mathbf{x}) = \frac{1}{n}\sum_{i=1}^n (\mathbf{a}_i^\top \mathbf{x} - b_i)\mathbf{a}_i$
- Cost per iteration: $O(nd)$
- Convergence rate: $O((1 - \mu/L)^T)$ for strongly convex case

**SGD:**
- Stochastic gradient: $\mathbf{g}_t = (\mathbf{a}_{i_t}^\top \mathbf{x}_t - b_{i_t})\mathbf{a}_{i_t}$
- Cost per iteration: $O(d)$
- Convergence rate: $O(1/T)$ for strongly convex case

**Wall-clock time comparison:** To reach accuracy $\epsilon$:
- GD: $O(\kappa \log(1/\epsilon))$ iterations × $O(nd)$ cost = $O(\kappa nd \log(1/\epsilon))$
- SGD: $O(\sigma^2/(\mu^2\epsilon))$ iterations × $O(d)$ cost = $O(\sigma^2 d/(\mu^2\epsilon))$

For large $n$, SGD is faster in wall-clock time despite requiring more iterations.

### B.2 Mini-Batch Variance: Numerical Example

Consider a dataset with $n = 10,000$ examples where the gradient variance is $\sigma^2 = 4$.

| Batch Size $B$ | Variance $\sigma^2/B$ | Std. Dev. | Speedup vs. $B=1$ |
| --- | --- | --- | --- |
| 1 | 4.00 | 2.00 | 1× |
| 32 | 0.125 | 0.354 | 32× |
| 128 | 0.031 | 0.177 | 128× |
| 1024 | 0.0039 | 0.062 | 1024× |
| 10000 | 0.0004 | 0.020 | 10000× (full batch) |

The variance decreases linearly with batch size, but the per-iteration cost increases linearly. The optimal batch size balances these two effects.

### B.3 SVRG in Action: Logistic Regression

For logistic regression with $n = 10,000$ examples and $d = 100$ features:

**SVRG parameters:**
- Epoch length: $m = 2n = 20,000$
- Step size: $\eta = 1/(4L)$ where $L = \frac{1}{4n}\lambda_{\max}(X^\top X)$
- Convergence rate: $\rho \approx 0.95$ per epoch

**Comparison with SGD:**
- SGD needs $O(1/\epsilon)$ iterations to reach $\epsilon$-accuracy
- SVRG needs $O(\kappa \log(1/\epsilon))$ iterations (linear convergence)
- For $\epsilon = 10^{-6}$ and $\kappa = 100$: SGD needs ~1M iterations, SVRG needs ~2,700 epochs × 20,000 = 54M individual gradient evaluations

Wait — this seems worse for SVRG! The key is that SVRG's convergence rate is in terms of epochs, not individual gradient evaluations. The total gradient evaluations for SVRG is $O((n + \kappa)\log(1/\epsilon))$, which is $O(n\log(1/\epsilon))$ when $\kappa \ll n$. For this example: $O(10,000 \times 14) = 140,000$ gradient evaluations vs. SGD's 1,000,000.

### B.4 The Critical Batch Size in Practice

For LLM pretraining with AdamW:

| Model Size | Critical Batch Size | Optimal LR | Notes |
| --- | --- | --- | --- |
| 125M (GPT-3 small) | ~4K tokens | $6 \times 10^{-4}$ | Small model, small critical batch |
| 1.3B (GPT-3 medium) | ~16K tokens | $2 \times 10^{-4}$ | Medium model |
| 175B (GPT-3) | ~2M tokens | $6 \times 10^{-5}$ | Large model, large critical batch |
| 70B (LLaMA-2) | ~4M tokens | $1.5 \times 10^{-4}$ | Modern architecture |

The critical batch size increases with model size because larger models have more parameters to explore, requiring more data per step to maintain a good signal-to-noise ratio.

### B.5 Distributed SGD: Communication Analysis

For a model with $n_{\text{params}} = 10^9$ parameters trained on $K = 1024$ GPUs:

| Strategy | Communication per Step | Total Comm for 1M Steps | Bottleneck |
| --- | --- | --- | --- |
| **All-reduce (full precision)** | 4 GB | 4 PB | Network bandwidth |
| **1-bit quantization** | 0.125 GB | 125 TB | Quantization error |
| **Top-1% sparsification** | 0.04 GB | 40 TB | Gradient quality |
| **Local SGD (sync every 100 steps)** | 0.04 GB | 0.4 TB | Model divergence |

For LLM training, gradient compression is essential. Modern systems use a combination of techniques: BF16 gradients (2× compression), top-k sparsification (100× compression), and error feedback to maintain convergence quality.

### B.6 SGD Noise and Generalization: Empirical Evidence

**Keskar et al. (2017)** systematically studied the generalization gap between small-batch and large-batch SGD:

| Batch Size | Training Accuracy | Test Accuracy | Generalization Gap |
| --- | --- | --- | --- |
| 32 | 93.2% | 92.1% | 1.1% |
| 128 | 94.5% | 91.8% | 2.7% |
| 512 | 95.8% | 90.5% | 5.3% |
| 2048 | 97.1% | 88.2% | 8.9% |
| 8192 | 98.5% | 85.1% | 13.4% |

Key finding: Large-batch SGD converges to sharp minima (high training accuracy, low test accuracy), while small-batch SGD converges to flat minima (slightly lower training accuracy, much higher test accuracy).

**Mitigation strategies:**
1. **Learning rate warmup:** Gradually increase the LR to allow the model to find flat regions
2. **Longer training:** Train for more epochs to allow large-batch SGD to escape sharp minima
3. **SAM:** Explicitly optimize for flat minima
4. **Data augmentation:** Increase the effective dataset size to reduce the generalization gap

---

## Appendix C: Numerical Implementation and Practical Considerations

### C.1 Efficient Mini-Batch Sampling

**Without replacement sampling:** Shuffle the dataset once per epoch and iterate through it in order. This reduces the variance compared to with-replacement sampling and is the standard in most deep learning frameworks.

**Stratified sampling:** For imbalanced datasets, sample each class proportionally to ensure each mini-batch is representative. This reduces the variance of the gradient estimate.

**Importance sampling:** Sample examples with probability proportional to their gradient norm or loss. This focuses computation on "hard" examples and can accelerate convergence.

**For AI:** In LLM pretraining, the data is typically shuffled once and iterated through in order. For fine-tuning on imbalanced datasets, stratified sampling or oversampling of minority classes is common.

### C.2 Learning Rate Warmup

Learning rate warmup gradually increases the learning rate from 0 to $\eta_{\max}$ over the first $T_{\text{warmup}}$ steps:

$$\eta_t = \eta_{\max} \cdot \frac{t}{T_{\text{warmup}}} \quad \text{for } t < T_{\text{warmup}}$$

**Why warmup helps:**
1. **Stability:** At initialization, the model parameters are random and the gradients can be large. A small initial LR prevents divergence.
2. **Adam bias correction:** Adam's momentum terms are biased toward zero at initialization. Warmup allows the momentum to build up before applying the full learning rate.
3. **Large-batch training:** When using large batch sizes, the gradient estimate has low variance but may be biased. Warmup allows the model to move into a region where the gradient is more reliable.

**Typical values:** $T_{\text{warmup}} = 1,000$ to $10,000$ steps for LLM pretraining. The warmup period is typically 1-5% of the total training steps.

### C.3 Gradient Clipping in Practice

Gradient clipping prevents exploding gradients by capping the gradient norm:

$$\hat{\mathbf{g}} = \min\left(1, \frac{c}{\lVert \mathbf{g} \rVert_2}\right) \mathbf{g}$$

**Choosing the clip threshold:**
- **RNNs:** $c = 1.0$ to $5.0$ (gradients can explode due to recurrent connections)
- **Transformers:** $c = 0.1$ to $10.0$ (depends on model size and learning rate)
- **LLMs:** $c = 1.0$ is common (GPT-3, LLaMA)

**Monitoring:** Track the fraction of steps where clipping is triggered. If this is $> 10\%$, the learning rate may be too high. If it is $< 0.1\%$, the clip threshold may be too high.

### C.4 Mixed Precision Training

Mixed precision training uses lower precision (BF16 or FP16) for the forward and backward passes while maintaining a FP32 master copy of the parameters:

1. **Forward pass:** Compute in BF16/FP16
2. **Backward pass:** Compute gradients in BF16/FP16
3. **Parameter update:** Cast gradients to FP32, update FP32 master weights, cast back to BF16/FP16

**Benefits:**
- **2× speedup** on modern GPUs (tensor cores are optimized for BF16/FP16)
- **2× memory reduction** for activations and gradients
- **Same accuracy** as FP32 training (with proper loss scaling)

**Loss scaling:** To prevent gradient underflow in FP16, multiply the loss by a scale factor $S$ before the backward pass, then divide the gradients by $S$ before the update. Dynamic loss scaling adjusts $S$ based on the gradient magnitude.

### C.5 Debugging Stochastic Optimization

**Common issues and diagnostics:**

1. **Divergence:** The loss increases instead of decreasing.
   - **Cause:** Learning rate too high, gradient explosion, or numerical instability
   - **Fix:** Reduce LR, add gradient clipping, check for NaN values

2. **Stagnation:** The loss stops decreasing but hasn't converged.
   - **Cause:** Learning rate too low, vanishing gradients, or poor initialization
   - **Fix:** Increase LR, check gradient norms, try different initialization

3. **Oscillation:** The loss oscillates without converging.
   - **Cause:** Learning rate too high for the local curvature, or too much momentum
   - **Fix:** Reduce LR, reduce momentum, use learning rate scheduling

4. **Overfitting:** Training loss decreases but test loss increases.
   - **Cause:** Model capacity too large, insufficient regularization, or batch size too large
   - **Fix:** Add regularization, reduce batch size, use early stopping, apply data augmentation

5. **Underfitting:** Both training and test loss are high.
   - **Cause:** Model capacity too small, learning rate too low, or insufficient training
   - **Fix:** Increase model size, increase LR, train for more epochs

**Monitoring checklist:**
- [ ] Training loss decreasing
- [ ] Validation loss decreasing (or stable)
- [ ] Gradient norms in reasonable range (not too large, not too small)
- [ ] Learning rate schedule as expected
- [ ] No NaN or Inf values in parameters or gradients
- [ ] Batch statistics (mean, variance) are stable
- [ ] Communication overhead in distributed training is acceptable

### C.6 Choosing the Right Stochastic Optimizer

| Scenario | Recommended Optimizer | Reason |
| --- | --- | --- |
| **Image classification (small dataset)** | SGD + momentum | Better generalization, less prone to overfitting |
| **Image classification (large dataset)** | SGD + momentum or Adam | Both work well; SGD may generalize slightly better |
| **NLP / LLM pretraining** | AdamW | Handles sparse gradients, adaptive per-parameter LR |
| **LLM fine-tuning** | AdamW | Stable convergence with small learning rates |
| **Logistic regression (large $n$)** | SVRG or SAGA | Linear convergence for finite-sum problems |
| **Matrix factorization** | SGD or ALS | SGD for large-scale, ALS for moderate-scale |
| **Reinforcement learning** | Adam or RMSProp | Handles non-stationary objectives well |
| **Federated learning** | Local SGD (FedAvg) | Minimizes communication rounds |
| **Distributed training (bandwidth-limited)** | SGD + gradient compression | Reduces communication cost |
| **Large-batch training** | LARS or LAMB | Layer-wise adaptive learning rates |

**For AI practitioners:** The default choice for most deep learning tasks is AdamW. For vision tasks where generalization is critical, SGD with momentum is often preferred. For large-scale distributed training, the choice depends on the communication bottleneck and the critical batch size.

---

## Appendix D: Extended Algorithmic Implementations

### D.1 SGD with Momentum: Complete Implementation

```python
def sgd_momentum(params, grads, velocity, lr=0.01, momentum=0.9, weight_decay=0.0):
    """
    SGD with momentum update.
    
    Args:
        params: list of parameter tensors
        grads: list of gradient tensors
        velocity: list of velocity buffers (same shape as params)
        lr: learning rate
        momentum: momentum coefficient (beta)
        weight_decay: L2 regularization coefficient
    
    Returns:
        Updated params and velocity buffers
    """
    for i, (p, g, v) in enumerate(zip(params, grads, velocity)):
        # Weight decay (decoupled, like AdamW)
        if weight_decay > 0:
            g = g + weight_decay * p
        
        # Momentum update
        v[i] = momentum * v[i] + g
        p = p - lr * v[i]
    
    return params, velocity
```

**Bias-corrected momentum:** At the beginning of training, the velocity is biased toward zero. The bias-corrected velocity is:

$$\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta^t}$$

This is particularly important when using large momentum values ($\beta = 0.99$) or when the learning rate is large.

### D.2 SVRG: Complete Implementation

```python
def svrg(full_grad_fn, stochastic_grad_fn, theta0, n, m=None, eta=None, max_epochs=100, tol=1e-8):
    """
    Stochastic Variance Reduced Gradient (SVRG).
    
    Args:
        full_grad_fn: function that computes the full gradient ∇F(θ)
        stochastic_grad_fn: function that computes ∇f_i(θ) for a random i
        theta0: initial parameters
        n: number of training examples
        m: epoch length (default: 2n)
        eta: step size
        max_epochs: maximum number of epochs
        tol: convergence tolerance
    
    Returns:
        theta: optimal parameters
        history: convergence history
    """
    if m is None:
        m = 2 * n
    if eta is None:
        eta = 0.1  # Should be tuned based on problem
    
    theta = theta0.copy()
    history = []
    
    for s in range(max_epochs):
        # Compute full gradient at snapshot
        full_grad = full_grad_fn(theta)
        theta_snapshot = theta.copy()
        
        # Inner loop
        for t in range(m):
            # Stochastic gradient at current point
            g_current = stochastic_grad_fn(theta)
            
            # Stochastic gradient at snapshot
            g_snapshot = stochastic_grad_fn(theta_snapshot)
            
            # Variance-reduced gradient
            g_svrg = g_current - g_snapshot + full_grad
            
            # Update
            theta = theta - eta * g_svrg
            
            # Check convergence
            if np.linalg.norm(g_svrg) < tol:
                history.append((s, t, np.linalg.norm(g_svrg)))
                return theta, history
        
        history.append((s, m, np.linalg.norm(full_grad)))
    
    return theta, history
```

### D.3 SAGA: Complete Implementation

```python
def saga(stochastic_grad_fn, theta0, n, eta=None, max_iter=10000, tol=1e-8):
    """
    SAGA: Stochastic Average Gradient Amélioré.
    
    Args:
        stochastic_grad_fn: function that returns (grad_i, i) for random i
        theta0: initial parameters
        n: number of training examples
        eta: step size
        max_iter: maximum iterations
        tol: convergence tolerance
    
    Returns:
        theta: optimal parameters
    """
    if eta is None:
        eta = 0.01
    
    theta = theta0.copy()
    
    # Initialize gradient table
    grad_table = np.zeros((n, len(theta0)))
    for i in range(n):
        grad_table[i] = stochastic_grad_fn(theta, i)
    
    avg_grad = np.mean(grad_table, axis=0)
    
    for t in range(max_iter):
        # Sample random example
        i = np.random.randint(n)
        
        # Compute gradient at current point
        g_current = stochastic_grad_fn(theta, i)
        
        # SAGA gradient estimator
        g_saga = g_current - grad_table[i] + avg_grad
        
        # Update
        theta = theta - eta * g_saga
        
        # Update gradient table
        grad_table[i] = g_current
        avg_grad = avg_grad + (g_current - grad_table[i]) / n  # No change since we just updated
        
        # Recompute average (more stable than incremental)
        if t % 100 == 0:
            avg_grad = np.mean(grad_table, axis=0)
        
        if np.linalg.norm(g_saga) < tol:
            break
    
    return theta
```

### D.4 Distributed SGD with Gradient Compression

```python
def compressed_sgd_worker(local_data, model, optimizer, rank, world_size, 
                          compress_fn, error_feedback=True):
    """
    Distributed SGD worker with gradient compression.
    
    Args:
        local_data: local mini-batch data
        model: neural network model
        optimizer: optimizer (SGD, Adam, etc.)
        rank: worker rank
        world_size: total number of workers
        compress_fn: gradient compression function
        error_feedback: whether to use error feedback
    
    Returns:
        Updated model parameters
    """
    # Forward pass
    loss = model(local_data)
    loss.backward()
    
    # Get gradients
    grads = [p.grad.clone() for p in model.parameters()]
    
    # Error feedback
    if error_feedback:
        if not hasattr(compressed_sgd_worker, 'error_buffers'):
            compressed_sgd_worker.error_buffers = [torch.zeros_like(g) for g in grads]
        
        for i in range(len(grads)):
            grads[i] = grads[i] + compressed_sgd_worker.error_buffers[i]
    
    # Compress gradients
    compressed_grads = [compress_fn(g) for g in grads]
    
    # All-reduce compressed gradients
    for i in range(len(compressed_grads)):
        torch.distributed.all_reduce(compressed_grads[i], op=torch.distributed.ReduceOp.SUM)
        compressed_grads[i] = compressed_grads[i] / world_size
    
    # Update error feedback
    if error_feedback:
        for i in range(len(grads)):
            compressed_sgd_worker.error_buffers[i] = grads[i] - compressed_grads[i]
    
    # Update model
    for p, g in zip(model.parameters(), compressed_grads):
        p.grad = g
    
    optimizer.step()
    optimizer.zero_grad()
```

### D.5 Local SGD (Federated Averaging)

```python
def local_sgd_server(models, client_data, local_epochs=5, lr=0.01, rounds=100):
    """
    Federated Averaging (FedAvg) / Local SGD.
    
    Args:
        models: list of client models
        client_data: list of client datasets
        local_epochs: number of local epochs per round
        lr: learning rate
        rounds: number of communication rounds
    
    Returns:
        Global model
    """
    K = len(models)  # Number of clients
    
    for r in range(rounds):
        # Send global model to clients (if not first round)
        if r > 0:
            global_params = [p.clone() for p in models[0].parameters()]
            for model in models[1:]:
                for p, gp in zip(model.parameters(), global_params):
                    p.data.copy_(gp.data)
        
        # Local training on each client
        for k in range(K):
            optimizer = torch.optim.SGD(models[k].parameters(), lr=lr)
            for epoch in range(local_epochs):
                for batch in client_data[k]:
                    loss = models[k](batch)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
        
        # Server averaging
        for p in models[0].parameters():
            p.data.zero_()
            for model in models:
                for mp, p in zip(model.parameters(), models[0].parameters()):
                    p.data.add_(mp.data / K)
        
        # Copy averaged model back to all clients
        for model in models[1:]:
            for mp, gp in zip(model.parameters(), models[0].parameters()):
                mp.data.copy_(gp.data)
    
    return models[0]  # Return the global model
```

### D.6 AdamW: A Common LLM Baseline

```python
def adamw(params, grads, m, v, t, lr=1e-4, beta1=0.9, beta2=0.999, 
          eps=1e-8, weight_decay=0.01):
    """
    AdamW: Adam with decoupled weight decay.
    
    Args:
        params: list of parameter tensors
        grads: list of gradient tensors
        m: first moment estimates
        v: second moment estimates
        t: current step
        lr: learning rate
        beta1: first moment decay rate
        beta2: second moment decay rate
        eps: numerical stability constant
        weight_decay: decoupled weight decay coefficient
    
    Returns:
        Updated params, m, v
    """
    for i, (p, g) in enumerate(zip(params, grads)):
        # Decoupled weight decay (applied before Adam update)
        p = p - lr * weight_decay * p
        
        # Update biased first moment estimate
        m[i] = beta1 * m[i] + (1 - beta1) * g
        
        # Update biased second raw moment estimate
        v[i] = beta2 * v[i] + (1 - beta2) * g**2
        
        # Compute bias-corrected first moment estimate
        m_hat = m[i] / (1 - beta1**t)
        
        # Compute bias-corrected second raw moment estimate
        v_hat = v[i] / (1 - beta2**t)
        
        # Update parameters
        p = p - lr * m_hat / (torch.sqrt(v_hat) + eps)
    
    return params, m, v
```

### D.7 Sharpness-Aware Minimization (SAM)

```python
def sam_step(model, data, loss_fn, optimizer, rho=0.05):
    """
    Sharpness-Aware Minimization: one SAM step.
    
    Args:
        model: neural network model
        data: input data
        loss_fn: loss function
        optimizer: optimizer
        rho: neighborhood radius
    
    Returns:
        Updated model parameters
    """
    # First forward-backward pass: compute gradient
    predictions = model(data)
    loss = loss_fn(predictions, data.targets)
    loss.backward()
    
    # Compute perturbation
    grad_norm = torch.norm(torch.stack([p.grad.norm() for p in model.parameters()]))
    epsilon = rho / (grad_norm + 1e-12)
    
    # Apply perturbation
    for p in model.parameters():
        p.data.add_(p.grad * epsilon)
    
    # Second forward-backward pass: compute gradient at perturbed point
    optimizer.zero_grad()
    predictions = model(data)
    loss = loss_fn(predictions, data.targets)
    loss.backward()
    
    # Restore original parameters
    for p in model.parameters():
        p.data.sub_(p.grad * epsilon)
    
    # Update with SAM gradient
    optimizer.step()
```

### D.8 Learning Rate Schedules for SGD

```python
def cosine_warmup_lr(base_lr, warmup_steps, total_steps, current_step):
    """
    Linear warmup + cosine decay learning rate schedule.
    
    Args:
        base_lr: maximum learning rate
        warmup_steps: number of warmup steps
        total_steps: total number of training steps
        current_step: current step number
    
    Returns:
        Learning rate for current step
    """
    if current_step < warmup_steps:
        # Linear warmup
        return base_lr * current_step / warmup_steps
    else:
        # Cosine decay
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        return base_lr * 0.5 * (1 + np.cos(np.pi * progress))

def wsd_lr(base_lr, warmup_steps, stable_steps, decay_steps, total_steps, current_step):
    """
    Warmup-Stable-Decay (WSD) learning rate schedule.
    
    Args:
        base_lr: maximum learning rate
        warmup_steps: number of warmup steps
        stable_steps: number of stable (constant LR) steps
        decay_steps: number of decay steps
        total_steps: total number of training steps
        current_step: current step number
    
    Returns:
        Learning rate for current step
    """
    if current_step < warmup_steps:
        return base_lr * current_step / warmup_steps
    elif current_step < warmup_steps + stable_steps:
        return base_lr
    else:
        progress = (current_step - warmup_steps - stable_steps) / decay_steps
        return base_lr * 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))
```

### D.9 Performance Comparison: Stochastic Optimizers

| Optimizer | Per-iter Cost | Convergence Rate | Memory | Best Use Case |
| --- | --- | --- | --- | --- |
| **SGD** | $O(d)$ | $O(1/\sqrt{T})$ | $O(d)$ | Large-scale, non-convex |
| **SGD + Momentum** | $O(d)$ | $O(1/\sqrt{T})$ | $O(d)$ | Vision models |
| **Adam** | $O(d)$ | $O(1/\sqrt{T})$ | $O(d)$ | NLP and generative modeling |
| **AdamW** | $O(d)$ | $O(1/\sqrt{T})$ | $O(d)$ | Common LLM pretraining baseline |
| **SVRG** | $O(d)$ | Linear (finite-sum) | $O(d)$ | Logistic regression, SVM |
| **SAGA** | $O(d)$ | Linear (finite-sum) | $O(nd)$ | Moderate $n$, finite-sum |
| **LAMB** | $O(d)$ | $O(1/\sqrt{T})$ | $O(d)$ | Large-batch BERT/LLM |
| **SAM** | $2 \times O(d)$ | $O(1/\sqrt{T})$ | $O(d)$ | When generalization matters most |

**Key insight:** For deep learning, the convergence rate in terms of iterations is similar across optimizers ($O(1/\sqrt{T})$). The differences come from:
1. **Generalization quality:** SGD often generalizes better than Adam
2. **Ease of tuning:** Adam requires less hyperparameter tuning
3. **Large-batch scaling:** LAMB and LARS enable larger batch sizes
4. **Flat minimum finding:** SAM explicitly optimizes for flat minima

### D.10 Debugging Checklist for Stochastic Optimization

**Before training:**
- [ ] Data is properly shuffled
- [ ] Learning rate is appropriate for batch size (linear scaling rule)
- [ ] Weight decay is set correctly (decoupled for AdamW)
- [ ] Gradient clipping threshold is reasonable
- [ ] Mixed precision is configured correctly (loss scaling)

**During training:**
- [ ] Training loss is decreasing
- [ ] Validation loss is decreasing (or stable)
- [ ] Gradient norms are in reasonable range
- [ ] No NaN/Inf values in parameters or gradients
- [ ] Learning rate schedule is as expected
- [ ] Communication overhead is acceptable (distributed training)

**After training:**
- [ ] Model achieves target accuracy
- [ ] Generalization gap is acceptable
- [ ] Training is reproducible (fixed random seeds)
- [ ] Model is robust to small perturbations (flat minimum)

**For AI practitioners:** The most common mistake is using the wrong learning rate for the batch size. Always apply the linear scaling rule when changing the batch size, and use learning rate warmup for large-batch training.

---

## Appendix E: Extended Case Studies in Machine Learning

### E.1 Training ResNet-50 on ImageNet: A Detailed Analysis

Training ResNet-50 on ImageNet (1.28M images, 1000 classes) is a benchmark for stochastic optimization.

**Standard setup (Goyal et al., 2017):**
- **Optimizer:** SGD with momentum ($\beta = 0.9$)
- **Batch size:** 256 per GPU × 8 GPUs = 2048 total
- **Learning rate:** 0.1 (base) × 2048/256 = 0.8 (linear scaling)
- **LR schedule:** Linear warmup for 5 epochs, then step decay at epochs 30, 60, 90
- **Weight decay:** $10^{-4}$
- **Training epochs:** 90
- **Top-1 accuracy:** 76.3%

**Large-batch setup (batch size 8192):**
- **Learning rate:** 0.1 × 8192/256 = 3.2
- **LR schedule:** Linear warmup for 25 epochs, then cosine decay
- **Top-1 accuracy:** 75.8% (0.5% gap from small-batch baseline)

**Key findings:**
1. The linear scaling rule works up to batch size 8192 with proper warmup
2. The generalization gap can be closed with longer training (300 epochs → 76.5%)
3. SAM can close the gap without longer training

### E.2 BERT Pretraining: AdamW at Scale

BERT-base pretraining (110M parameters, 3.3B words):

**Standard setup (Loshchilov & Hutter, 2019):**
- **Optimizer:** AdamW ($\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-6}$)
- **Batch size:** 256 sequences
- **Learning rate:** $10^{-4}$ with linear warmup (10K steps) and linear decay
- **Weight decay:** 0.01 (decoupled)
- **Training steps:** 1M
- **Gradient clipping:** 1.0

**Key insight:** AdamW's decoupled weight decay is crucial for BERT. Standard Adam (with coupled weight decay) degrades performance by 0.5-1.0% on downstream tasks. The adaptive learning rates in Adam handle the sparse gradients from the embedding layer more effectively than SGD.

### E.3 GPT-3 Pretraining: Distributed SGD at Trillion-Token Scale

GPT-3 (175B parameters, 300B tokens):

**Setup (Brown et al., 2020):**
- **Optimizer:** Adam ($\beta_1 = 0.9$, $\beta_2 = 0.95$, $\epsilon = 10^{-8}$)
- **Batch size:** 3.2M tokens (varied from 32K to 3.2M during training)
- **Learning rate:** $6 \times 10^{-5}$ with linear warmup (375M tokens) and cosine decay
- **Weight decay:** 0.1
- **Gradient clipping:** 1.0
- **Mixed precision:** FP16 with dynamic loss scaling
- **Distributed setup:** Model parallelism + data parallelism across thousands of GPUs

**Key challenges:**
1. **Communication bottleneck:** Gradient all-reduce across thousands of GPUs requires efficient communication
2. **Gradient instability:** Occasional gradient spikes require careful clipping and monitoring
3. **Learning rate scheduling:** The warmup period is critical for stable training at this scale
4. **Checkpointing:** Frequent checkpoints (every 500 steps) to recover from hardware failures

### E.4 Federated Learning at Google: Gboard Next-Word Prediction

Google's federated learning for Gboard next-word prediction (Hard et al., 2018):

**Setup:**
- **Clients:** Millions of Android devices
- **Local data:** Typing history on each device (never leaves the device)
- **Local training:** SGD with 1-4 epochs per round
- **Communication:** Only model updates are sent to the server (not raw data)
- **Aggregation:** Federated averaging (FedAvg) with weighted averaging by data size
- **Privacy:** Secure aggregation ensures the server cannot see individual client updates

**Key findings:**
1. Federated learning achieves similar accuracy to centralized training
2. Communication efficiency is critical: only 1-2 rounds of communication per day per device
3. Heterogeneity is a major challenge: devices have different data distributions and compute capabilities
4. Privacy is preserved: the server only sees aggregated model updates

### E.5 Recommendation Systems at Netflix: SGD for Matrix Factorization

Netflix's recommendation system uses matrix factorization trained with SGD:

**Setup:**
- **Dataset:** 100M+ ratings from 480K users on 17K movies
- **Model:** Matrix factorization with user and item latent factors (d = 50-200)
- **Optimizer:** SGD with momentum and adaptive learning rates
- **Mini-batch size:** 1 (pure SGD, one rating at a time)
- **Regularization:** L2 regularization on user and item factors

**Key insights:**
1. Pure SGD (batch size 1) works well because the problem is highly sparse
2. The optimal learning rate decreases over time as the model converges
3. Adding bias terms (user bias, item bias, global bias) significantly improves accuracy
4. Temporal dynamics (changing user preferences over time) require online updates

### E.6 Reinforcement Learning: PPO with Adam for Robotics

Proximal Policy Optimization (PPO) with Adam for robotic control:

**Setup:**
- **Environment:** MuJoCo continuous control tasks (HalfCheetah, Walker2d, Humanoid)
- **Policy:** Gaussian policy with neural network mean function
- **Optimizer:** Adam ($\beta_1 = 0.9$, $\beta_2 = 0.999$)
- **Learning rate:** $3 \times 10^{-4}$ with linear decay
- **Clip ratio:** $\epsilon = 0.2$ (PPO clipping parameter)
- **GAE:** $\lambda = 0.95$ (generalized advantage estimation)

**Key challenges:**
1. **Non-stationary objective:** The policy changes during training, so the objective function changes
2. **High variance:** Policy gradient estimates have high variance, requiring careful normalization
3. **Trust region:** PPO's clipping mechanism ensures the policy doesn't change too much in one step
4. **Adam's role:** Adam's adaptive learning rates handle the non-stationary gradients better than SGD

---

## Appendix F: Theoretical Connections and Open Problems

### F.1 SGD as a Discretization of SDE

In the continuous-time limit, SGD with learning rate $\eta$ and batch size $B$ can be approximated by the stochastic differential equation (SDE):

$$d\boldsymbol{\theta}_t = -\nabla F(\boldsymbol{\theta}_t) dt + \sqrt{\frac{\eta}{B}} \Sigma(\boldsymbol{\theta}_t)^{1/2} d\mathbf{W}_t$$

where $\mathbf{W}_t$ is a standard Brownian motion and $\Sigma(\boldsymbol{\theta})$ is the gradient noise covariance.

**Implications:**
1. The effective temperature of the SDE is $\eta/B$ — larger learning rate or smaller batch size increases the "thermal noise"
2. The stationary distribution of the SDE is approximately $p(\boldsymbol{\theta}) \propto \exp(-\frac{B}{\eta} F(\boldsymbol{\theta}))$ — SGD samples from a distribution centered at the minimum
3. The noise structure $\Sigma(\boldsymbol{\theta})$ determines which minima are preferred — flat minima have larger basins of attraction in the SDE

### F.2 The Edge of Stability in SGD

Recent work (Cohen et al., 2021) has shown that during neural network training with SGD, the sharpness (largest Hessian eigenvalue) exhibits a characteristic pattern:

1. **Initial phase:** Sharpness increases rapidly as the model moves away from initialization
2. **Stabilization:** Sharpness stabilizes near $2/\eta$, the theoretical stability boundary
3. **Oscillation:** Sharpness oscillates around $2/\eta$ for the remainder of training

This "edge of stability" phenomenon explains why SGD can use learning rates larger than what the theory predicts for convex functions. The mechanism is self-correcting: when the sharpness exceeds $2/\eta$, the gradient step increases the loss, which moves the parameters to a region of lower sharpness.

### F.3 Open Problems in Stochastic Optimization

1. **Understanding generalization:** Why does SGD generalize better than full-batch GD? Is it the noise, the implicit bias, or something else?

2. **Optimal batch size:** How to choose the optimal batch size for a given problem and compute budget? The critical batch size depends on the problem, model, and training phase.

3. **Adaptive methods:** Why does Adam work so well for language models but not for vision? Is there a unified optimizer that works well across all domains?

4. **Distributed optimization:** How to efficiently train models with trillions of parameters across thousands of GPUs? Communication efficiency and fault tolerance are key challenges.

5. **Federated learning:** How to handle non-IID data across clients? How to ensure fairness and privacy in federated settings?

6. **Sharpness-aware optimization:** Can we design optimizers that explicitly optimize for flat minima? SAM is a step in this direction, but more work is needed.

7. **SGD dynamics in overparameterized models:** How does SGD navigate the complex loss landscape of overparameterized neural networks? What is the role of the neural tangent kernel?

8. **Stochastic second-order methods:** Can we design practical stochastic second-order methods that combine the fast convergence of Newton's method with the scalability of SGD?

### F.4 The Role of Data Ordering in SGD

The order in which training examples are presented to SGD affects convergence:

**Random ordering (with replacement):** The classic SGD assumption. Each example is sampled independently and uniformly at random. This gives unbiased gradient estimates with variance $\sigma^2/B$.

**Random ordering (without replacement):** Shuffle the dataset once per epoch and iterate through it in order. This has lower variance than with-replacement sampling and is the standard in practice.

**Curriculum learning:** Present examples in order of increasing difficulty. This can accelerate convergence by first learning easy patterns and then refining with harder examples.

**Adaptive ordering:** Dynamically adjust the sampling distribution based on the current model state. Examples with high loss or high gradient norm are sampled more frequently.

**For AI:** In LLM pretraining, the data is typically shuffled once and iterated through in order. For fine-tuning, curriculum learning (starting with easy examples and progressing to harder ones) can improve convergence speed and final accuracy.

### F.5 The Lottery Ticket Hypothesis and SGD

The **lottery ticket hypothesis** (Frankle & Carbin, 2019) states that dense neural networks contain sparse subnetworks ("winning tickets") that can achieve the same accuracy as the full network when trained in isolation.

**Connection to SGD:** SGD's implicit bias toward simple solutions may be related to its ability to find these winning tickets. The noise in SGD helps explore the parameter space and discover sparse subnetworks that generalize well.

**Implications:**
1. **Pruning:** SGD-trained models can be pruned to 90% sparsity without accuracy loss
2. **Efficient training:** Training the winning ticket directly is faster than training the full network
3. **Generalization:** Winning tickets generalize better than randomly initialized sparse networks

**For AI:** Understanding the connection between SGD and the lottery ticket hypothesis could lead to more efficient training algorithms that directly find winning tickets, reducing the compute cost of training large models.

---

## Appendix G: Quick Reference Card

### Key Formulas

| Concept | Formula | Notes |
| --- | --- | --- |
| SGD update | $\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \mathbf{g}_t$ | $\mathbb{E}[\mathbf{g}_t] = \nabla F(\boldsymbol{\theta}_t)$ |
| Mini-batch variance | $\text{Var}(\mathbf{g}_B) = \sigma^2/B$ | Decreases linearly with batch size |
| SGD convergence (convex) | $O(1/\sqrt{T})$ | With $\eta_t = O(1/\sqrt{t})$ |
| SGD convergence (strongly cvx) | $O(1/T)$ | With $\eta_t = O(1/t)$ |
| Linear scaling rule | $\eta_{\text{new}} = \eta_{\text{base}} \cdot B_{\text{new}}/B_{\text{base}}$ | Up to critical batch size |
| SVRG gradient | $\mathbf{g} = \nabla f_i(\boldsymbol{\theta}) - \nabla f_i(\tilde{\boldsymbol{\theta}}) + \nabla F(\tilde{\boldsymbol{\theta}})$ | Variance → 0 at optimum |
| SAGA gradient | $\mathbf{g} = \nabla f_i(\boldsymbol{\theta}) - \nabla f_i(\boldsymbol{\theta}_{t_i}) + \frac{1}{n}\sum_j \nabla f_j(\boldsymbol{\theta}_{t_j})$ | No full gradient needed |
| Critical batch size | $B_{\text{crit}} \approx \sigma^2/\lVert \nabla F \rVert^2$ | Beyond this, diminishing returns |
| SDE approximation | $d\boldsymbol{\theta}_t = -\nabla F dt + \sqrt{\eta/B}\Sigma^{1/2} d\mathbf{W}_t$ | Continuous-time limit |

### Parameter Recommendations

| Setting | LR | Batch Size | Momentum | Weight Decay |
| --- | --- | --- | --- | --- |
| Vision (SGD) | 0.1 | 256 | 0.9 | $10^{-4}$ |
| LLM (AdamW) | $10^{-4}$ | 2M tokens | 0.9, 0.95 | 0.01-0.1 |
| Fine-tuning | $10^{-5}$ | 32 | 0.9, 0.999 | 0.01 |
| Logistic regression | $0.01$ | 1 (SGD) | 0 | $\lambda$ |
| Matrix factorization | $0.001$ | 1 (SGD) | 0.9 | $\lambda$ |
