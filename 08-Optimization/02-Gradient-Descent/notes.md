[← Back to Optimization](../README.md) | [Next: Second-Order Methods →](../03-Second-Order-Methods/notes.md)

---

# Gradient Descent

> _"If you want to find the minimum of a function, just follow the gradient downhill. The simplicity of this idea is deceptive — it powers every neural network ever trained."_ — Anonymous

## Overview

Gradient descent is the simplest, most widely used, and most consequential algorithm in all of optimization. The update rule — move in the direction of steepest descent, scaled by a learning rate — fits on a single line. Yet this one-line algorithm trains models with hundreds of billions of parameters, discovers representations that capture human language and visual perception, and remains the core primitive underneath modern training pipelines in 2026.

The mathematical theory of gradient descent is rich and deep. For convex functions, convergence is guaranteed at a precise rate $O(1/T)$ under smoothness assumptions. For strongly convex functions, convergence is linear (geometric), with a rate controlled by the condition number $\kappa = L/\mu$. For non-convex functions — the setting of deep learning — gradient descent converges to stationary points, and the geometry of the loss landscape determines whether those points are useful. Momentum and Nesterov acceleration improve these rates, with NAG achieving the optimal $O(1/T^2)$ rate for first-order methods on convex functions.

This section develops the full convergence theory of gradient descent from first principles. We begin with the update rule and the descent lemma (§2), then prove convergence rates for convex (§3), strongly convex (§4), and non-convex (§5) objectives. We derive momentum from the characteristic polynomial of quadratic optimization (§6), present Nesterov acceleration with its geometric interpretation (§7), and analyze line search methods (§8). Advanced topics include the continuous-time limit as a gradient flow ODE and GD as an implicit regularizer (§9). Every result is connected to its role in training real ML systems.

**Scope note.** This section covers _deterministic_ gradient descent with exact gradients. The _stochastic_ setting — mini-batch gradients, variance, SGD convergence — is the canonical home of [Stochastic Optimization](../05-Stochastic-Optimization/notes.md). _Adaptive_ per-parameter learning rates (AdaGrad, RMSProp, Adam, AdamW) belong to [Adaptive Learning Rate](../07-Adaptive-Learning-Rate/notes.md). _Second-order_ methods (Newton, BFGS) are covered in [Second-Order Methods](../03-Second-Order-Methods/notes.md). Here we establish the foundational theory that all these variants build upon.

**For AI practitioners:** By the end of this section, you will understand why learning rate selection is bounded by $1/L$, why momentum accelerates training on ill-conditioned problems, why Nesterov acceleration is theoretically optimal yet rarely used in deep learning, and how the convergence theory of GD informs every practical decision in neural network training — from warmup schedules to gradient clipping.

## Prerequisites

- **Gradients and Hessians** — $\nabla f$, $\nabla^2 f$, Taylor expansion — [Chapter 5: Multivariate Calculus](../../05-Multivariate-Calculus/README.md)
- **L-smoothness and strong convexity** — definitions, condition number $\kappa = L/\mu$ — [Chapter 8 §01](../01-Convex-Optimization/notes.md#4-strong-convexity-and-smoothness)
- **Eigenvalues and positive definiteness** — $A \succ 0$, spectral theorem — [Chapter 3 §01](../../03-Advanced-Linear-Algebra/01-Eigenvalues-and-Eigenvectors/notes.md)
- **Matrix norms** — operator norm, Frobenius norm — [Chapter 3 §06](../../03-Advanced-Linear-Algebra/06-Matrix-Norms/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Interactive convergence demonstrations, momentum analysis, and loss landscape visualizations |
| [exercises.ipynb](exercises.ipynb) | 8 graded exercises from convergence proofs to momentum tuning for ill-conditioned problems |

## Learning Objectives

After completing this section, you will:

1. Derive the gradient descent update rule from first-order Taylor approximation
2. State and prove the descent lemma for L-smooth functions
3. Prove the $O(1/T)$ convergence rate for convex + L-smooth objectives
4. Prove linear convergence $O((1 - \mu/L)^T)$ for strongly convex objectives
5. Analyze non-convex convergence to stationary points and gradient norm decay
6. Derive Polyak momentum and compute the optimal $\beta$ for quadratic objectives
7. Explain Nesterov acceleration and its $O(1/T^2)$ optimality
8. Implement and analyze Armijo backtracking line search
9. Connect the continuous-time limit of GD to gradient flow ODEs
10. Apply GD convergence theory to select learning rates in real ML training
11. Explain why momentum accelerates training and when it can hurt
12. Distinguish deterministic GD from stochastic variants (→ §05)

---

## Table of Contents

- [Gradient Descent](#gradient-descent)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
  - [Companion Notebooks](#companion-notebooks)
  - [Learning Objectives](#learning-objectives)
  - [Table of Contents](#table-of-contents)
  - [1. Intuition](#1-intuition)
    - [1.1 What Is Gradient Descent?](#11-what-is-gradient-descent)
    - [1.2 The Gradient as Descent Direction — Geometric Intuition](#12-the-gradient-as-descent-direction--geometric-intuition)
    - [1.3 Why GD Dominates ML Training](#13-why-gd-dominates-ml-training)
    - [1.4 Historical Timeline: Cauchy to Modern LLMs](#14-historical-timeline-cauchy-to-modern-llms)
    - [1.5 The Optimization Loop in Practice](#15-the-optimization-loop-in-practice)
  - [2. Formal Definitions](#2-formal-definitions)
    - [2.1 The GD Update Rule](#21-the-gd-update-rule)
    - [2.2 First-Order Oracle Model](#22-first-order-oracle-model)
    - [2.3 Descent Lemma and Sufficient Decrease](#23-descent-lemma-and-sufficient-decrease)
    - [2.4 Step Size Regimes](#24-step-size-regimes)
    - [2.5 L-Smoothness Revisited](#25-l-smoothness-revisited)
  - [3. Convergence Theory — Convex Functions](#3-convergence-theory--convex-functions)
    - [3.1 One-Step Progress Inequality](#31-one-step-progress-inequality)
    - [3.2 O(1/T) Rate for Convex + L-Smooth Functions](#32-o1t-rate-for-convex--l-smooth-functions)
    - [3.3 Optimal Step Size η = 1/L](#33-optimal-step-size--1l)
    - [3.4 Function Value vs. Iterate Convergence](#34-function-value-vs-iterate-convergence)
    - [3.5 Tightness: Worst-Case Examples](#35-tightness-worst-case-examples)
  - [4. Convergence Theory — Strongly Convex Functions](#4-convergence-theory--strongly-convex-functions)
    - [4.1 Linear (Geometric) Convergence](#41-linear-geometric-convergence)
    - [4.2 Rate: (1 − μ/L)^T and the Condition Number κ](#42-rate-1--lt-and-the-condition-number-)
    - [4.3 Optimal Step Size η = 2/(L + μ)](#43-optimal-step-size--2l--)
    - [4.4 Lower Complexity Bounds for First-Order Methods](#44-lower-complexity-bounds-for-first-order-methods)
    - [4.5 What Happens When η Is Too Large](#45-what-happens-when--is-too-large)
  - [5. Convergence Theory — Non-Convex Functions](#5-convergence-theory--non-convex-functions)
    - [5.1 Convergence to Stationary Points](#51-convergence-to-stationary-points)
    - [5.2 Gradient Norm Decay: O(1/√T)](#52-gradient-norm-dec-o1t)
    - [5.3 Why Non-Convex GD Still Works in Practice](#53-why-non-convex-gd-still-works-in-practice)
    - [5.4 Saddle Points and the Need for Stochasticity](#54-saddle-points-and-the-need-for-stochasticity)
    - [5.5 PL (Polyak-Łojasiewicz) Condition](#55-pl-polyak-ojasiewicz-condition)
  - [6. Momentum](#6-momentum)
    - [6.1 Polyak Momentum](#61-polyak-momentum)
    - [6.2 Quadratic Analysis: Characteristic Polynomial](#62-quadratic-analysis-characteristic-polynomial)
    - [6.3 Optimal β for Quadratic Objectives](#63-optimal--for-quadratic-objectives)
    - [6.4 Momentum as Exponential Moving Average](#64-momentum-as-exponential-moving-average)
    - [6.5 Heavy Ball Interpretation](#65-heavy-ball-interpretation)
    - [6.6 Momentum in Practice: PyTorch SGD](#66-momentum-in-practice-pytorch-sgd)
  - [7. Nesterov Accelerated Gradient](#7-nesterov-accelerated-gradient)
    - [7.1 NAG Update: Lookahead Gradient](#71-nag-update-lookahead-gradient)
    - [7.2 O(1/T²) Rate for Convex Functions](#72-o1t-rate-for-convex-functions)
    - [7.3 Geometric Interpretation: Estimate Sequences](#73-geometric-interpretation-estimate-sequences)
    - [7.4 NAG vs. Polyak Momentum](#74-nag-vs-polyak-momentum)
    - [7.5 Optimality: Nesterov's Lower Bound Match](#75-optimality-nesterovs-lower-bound-match)
    - [7.6 Why NAG Is Less Common in Deep Learning](#76-why-nag-is-less-common-in-deep-learning)
  - [8. Line Search Methods](#8-line-search-methods)
    - [8.1 Exact Line Search for Quadratics](#81-exact-line-search-for-quadratics)
    - [8.2 Armijo Backtracking](#82-armijo-backtracking)
    - [8.3 Wolfe Conditions](#83-wolfe-conditions)
    - [8.4 When to Use Line Search vs. Fixed LR](#84-when-to-use-line-search-vs-fixed-lr)
    - [8.5 Line Search in Second-Order Methods](#85-line-search-in-second-order-methods)
  - [9. Advanced Topics](#9-advanced-topics)
    - [9.1 Continuous-Time Limit: Gradient Flow ODE](#91-continuous-time-limit-gradient-flow-ode)
    - [9.2 Implicit vs. Explicit Gradient Descent](#92-implicit-vs-explicit-gradient-descent)
    - [9.3 GD on Riemannian Manifolds](#93-gd-on-riemannian-manifolds)
    - [9.4 GD as Implicit Regularizer](#94-gd-as-implicit-regularizer)
    - [9.5 Connection to Neural Tangent Kernel](#95-connection-to-neural-tangent-kernel)
  - [10. Applications in Machine Learning](#10-applications-in-machine-learning)
    - [10.1 GD for Linear Regression](#101-gd-for-linear-regression)
    - [10.2 GD for Logistic Regression](#102-gd-for-logistic-regression)
    - [10.3 GD Through the Neural Network Training Lens](#103-gd-through-the-neural-network-training-lens)
    - [10.4 Learning Rate Selection in Real LLM Training](#104-learning-rate-selection-in-real-llm-training)
    - [10.5 GD, Momentum, and the Path to Adam](#105-gd-momentum-and-the-path-to-adam)
  - [11. Common Mistakes](#11-common-mistakes)
  - [12. Exercises](#12-exercises)
  - [13. Why This Matters for AI (2026 Perspective)](#13-why-this-matters-for-ai-2026-perspective)
  - [14. Conceptual Bridge](#14-conceptual-bridge)
  - [References](#references)

---

## 1. Intuition

### 1.1 What Is Gradient Descent?

Gradient descent is an iterative algorithm for finding a local minimum of a differentiable function $f: \mathbb{R}^n \to \mathbb{R}$. The idea is elementary: at each point, compute the gradient $\nabla f(\boldsymbol{\theta}_t)$, which points in the direction of steepest ascent, and take a step in the opposite direction:

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t)$$

where $\eta > 0$ is the **learning rate** (also called step size). This single equation is the computational engine behind every trained neural network, from a two-layer perceptron to a trillion-parameter language model.

**Why does this work?** The first-order Taylor approximation tells us that for small $\mathbf{d}$:

$$f(\boldsymbol{\theta} + \mathbf{d}) \approx f(\boldsymbol{\theta}) + \nabla f(\boldsymbol{\theta})^\top \mathbf{d}$$

To decrease $f$, we want $\nabla f(\boldsymbol{\theta})^\top \mathbf{d} < 0$. The Cauchy-Schwarz inequality gives:

$$\nabla f(\boldsymbol{\theta})^\top \mathbf{d} \geq -\lVert \nabla f(\boldsymbol{\theta}) \rVert_2 \cdot \lVert \mathbf{d} \rVert_2$$

with equality when $\mathbf{d} = -c \cdot \nabla f(\boldsymbol{\theta})$ for some $c > 0$. The negative gradient is the **steepest descent direction** in the Euclidean norm. Among all unit-length steps, the one that decreases $f$ the most (locally) is $-\nabla f(\boldsymbol{\theta}) / \lVert \nabla f(\boldsymbol{\theta}) \rVert_2$.

**For AI:** In neural network training, $\boldsymbol{\theta}$ represents all weights and biases, $f(\boldsymbol{\theta}) = \mathcal{L}(\boldsymbol{\theta})$ is the loss, and the gradient is computed by backpropagation. The GD update is executed millions or billions of times during training.

### 1.2 The Gradient as Descent Direction — Geometric Intuition

```text
GRADIENT AND DESCENT DIRECTION
════════════════════════════════════════════════════════════════════════

  Contour plot of f(x₁, x₂) = x₁² + 4x₂² (elliptic bowl):

         x₂
          │
     4.0 ─┤        ╭─────────╮
          │      ╭─╯         ╰─╮
     3.0 ─┤    ╭─╯   ╭───╮   ╰─╮
          │    │    ╭─╯ • ╰─╮   │   • = current point θ_t
     2.0 ─┤    │   ╭╯  /│   ╰╮  │   → = −∇f (steepest descent)
          │    │  ╭╯  ↗ │    ╰╮ │   ⟂ = contour tangent
     1.0 ─┤    │ ╭╯  ↗  │     ╰╮│   The gradient is perpendicular
          │    │ │  ↗   │      ││   to the contour at θ_t
     0.0 ─┼────┼─╯──╯───╰──────╰┼───
          │    │╱    │          │
    −1.0 ─┤    │     │          │
          │    │     │          │
    −2.0 ─┤    ╰─────╯          │
          │                     │
    ──────┼─────────────────────┼───── x₁
     −2.0       −1.0     0.0   1.0   2.0

  Key property: ∇f(θ) ⊥ level set {x : f(x) = f(θ)}
  The negative gradient points toward the center of the bowl.

════════════════════════════════════════════════════════════════════════
```

The gradient at a point is always **perpendicular to the level set** (contour line) passing through that point. This is because the level set $\{\mathbf{x} : f(\mathbf{x}) = c\}$ has tangent vectors $\mathbf{v}$ satisfying $\nabla f(\mathbf{x})^\top \mathbf{v} = 0$ — the directional derivative along the level set is zero.

**The elongated bowl problem.** When $f(\mathbf{x}) = x_1^2 + 4x_2^2$, the contours are ellipses. The gradient points toward the minimum but not directly at it — it overshoots in the steep direction ($x_2$) and undershoots in the shallow direction ($x_1$). This causes the characteristic zigzag pattern of GD on ill-conditioned problems. The condition number $\kappa = 4/1 = 4$ determines how severe this zigzag is. For neural networks, $\kappa$ can exceed $10^6$, making vanilla GD extremely slow without momentum or adaptive methods.

### 1.3 Why GD Dominates ML Training

Gradient descent is not the only optimization algorithm. Newton's method converges faster per iteration. Conjugate gradient is more efficient for quadratics. Interior-point methods solve convex programs in polynomial time. Yet GD (and its variants) is the default for deep learning. Why?

**Computational cost per iteration.** GD requires one gradient evaluation: $O(n)$ memory and $O(\text{cost of forward + backward pass})$ compute. Newton's method requires the Hessian: $O(n^2)$ memory and $O(n^3)$ compute for inversion. For a model with $n = 10^{11}$ parameters, the Hessian would require $10^{22}$ entries — roughly $10^{13}$ GB of memory. GD's per-iteration cost is the only one that scales to modern model sizes.

**Robustness to noise.** In practice, we rarely have exact gradients. Mini-batch gradients are noisy estimates. GD with a suitably small learning rate is robust to this noise, while higher-order methods amplify it (the Hessian of a noisy function is even noisier).

**Implicit regularization.** GD has a bias toward simple solutions. For overparameterized linear models, GD converges to the minimum-norm solution. For deep networks, GD implicitly favors flat minima that generalize well. This is not an accident — it is a mathematical property of the algorithm.

**Simplicity and parallelism.** The GD update is embarrassingly parallel: each parameter is updated independently given the gradient. This maps perfectly to GPU/TPU architectures. Complex algorithms with sequential dependencies (like line search or trust-region methods) cannot exploit modern hardware as effectively.

```text
WHY GD WINS IN DEEP LEARNING
════════════════════════════════════════════════════════════════════════

  Algorithm       │ Per-iter cost │ Memory    │ Scales to 10¹¹ params?
  ────────────────┼───────────────┼───────────┼───────────────────────
  GD              │ O(n)          │ O(n)      │ ✓ Yes (default)
  Newton          │ O(n³)         │ O(n²)     │ ✗ Impossible
  BFGS            │ O(n²)         │ O(n²)     │ ✗ Impossible
  Conjugate Grad  │ O(n)          │ O(n)      │ △ Possible, but fragile
  SGD             │ O(batch)      │ O(n)      │ ✓ Yes (standard)
  Adam            │ O(n)          │ O(n)      │ ✓ Yes (default LLM)

  The bottleneck is not convergence rate — it is per-iteration cost.
  GD is the only first-order method whose cost scales linearly with
  parameter count and is dominated by the forward/backward pass.

════════════════════════════════════════════════════════════════════════
```

### 1.4 Historical Timeline: Cauchy to Modern LLMs

```text
GRADIENT DESCENT TIMELINE
════════════════════════════════════════════════════════════════════════

  1847  Cauchy          — First GD algorithm (steepest descent)
  1944  Robbins & Monro — Stochastic approximation theory
  1951  Kiefer &        — Stochastic approximation for regression
        Wolfowitz
  1964  Polyak          — Heavy ball momentum method
  1964  Polyak          — Optimal step sizes for convex functions
  1983  Nesterov        — Accelerated gradient method (O(1/T²))
  1986  Rumelhart et al — Backpropagation + GD for neural networks
  1992  Tieleman &      — RMSProp (unpublished; Hinton's lecture)
        Hinton
  1998  Bottou          — SGD for large-scale learning
  2014  Kingma & Ba     — Adam optimizer
  2017  Loshchilov &    — AdamW (decoupled weight decay)
        Hutter
  2018  Vaswani et al   — Transformer training with Adam
  2020  You et al.      — LAMB optimizer for large-batch LLM training
  2022  Hu et al.       — LoRA: low-rank adaptation (GD on ΔW = BA)
  2023  LLaMA / GPT-4   — AdamW + cosine decay + warmup at scale
  2024  SOAP / Muon     — New large-scale optimizer proposals
  2025-26 Frontier LLMs — AdamW remains common; optimizer design stays active

════════════════════════════════════════════════════════════════════════
```

Cauchy's 1847 algorithm is conceptually identical to what runs inside PyTorch's `torch.optim.SGD` today. The intervening 180 years have refined our understanding of convergence rates, added momentum and acceleration, and adapted the method to stochastic settings — but the core update rule remains unchanged.

### 1.5 The Optimization Loop in Practice

```text
THE TRAINING LOOP
════════════════════════════════════════════════════════════════════════

  for t = 0, 1, 2, ..., T-1:
      # 1. Forward pass
      predictions = model(inputs, θ_t)

      # 2. Compute loss
      loss = loss_function(predictions, targets)

      # 3. Backward pass (compute gradient)
      g_t = ∇L(θ_t)          # via autograd / backpropagation

      # 4. (Optional) Gradient clipping
      if ‖g_t‖ > max_norm:
          g_t = g_t * (max_norm / ‖g_t‖)

      # 5. Parameter update
      θ_{t+1} = θ_t - η_t · g_t   # vanilla GD
      # or with momentum:
      #   v_{t+1} = β·v_t + g_t
      #   θ_{t+1} = θ_t - η_t · v_{t+1}

      # 6. (Optional) Weight decay
      #   θ_{t+1} = θ_{t+1} - η_t · λ · θ_t

  Return: θ_T (trained parameters)

════════════════════════════════════════════════════════════════════════
```

In practice, steps 1-3 are executed by the deep learning framework's autograd engine. Steps 4-5 are the optimizer's responsibility. The learning rate $\eta_t$ may vary with $t$ according to a schedule (warmup, cosine decay, etc.) — covered in [Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md).

**For AI:** This loop runs $10^5$ to $10^7$ times during LLM pretraining. Each iteration processes a mini-batch of tokens. The total compute is measured in FLOPs, and the wall-clock time is measured in weeks or months on thousands of GPUs. The mathematical properties of GD — convergence rate, stability, implicit bias — directly determine whether this investment of compute produces a useful model.

---

## 2. Formal Definitions

### 2.1 The GD Update Rule

**Definition (Gradient Descent).** Let $f: \mathbb{R}^n \to \mathbb{R}$ be differentiable. Given an initial point $\boldsymbol{\theta}_0 \in \mathbb{R}^n$ and a sequence of step sizes $\{\eta_t\}_{t \geq 0}$ with $\eta_t > 0$, the gradient descent iterates are:

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \nabla f(\boldsymbol{\theta}_t), \quad t = 0, 1, 2, \ldots$$

When $\eta_t = \eta$ is constant, we write:

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t)$$

**Notation.** Throughout this section:
- $f$ denotes the objective function (loss $\mathcal{L}$ in ML contexts)
- $\boldsymbol{\theta}_t$ denotes the iterate at step $t$
- $\boldsymbol{\theta}^*$ denotes a minimizer of $f$ (when it exists)
- $f^* = f(\boldsymbol{\theta}^*)$ denotes the optimal value
- $\mathbf{g}_t = \nabla f(\boldsymbol{\theta}_t)$ denotes the gradient at step $t$
- $\eta$ denotes the learning rate (step size)

### 2.2 First-Order Oracle Model

To analyze the complexity of optimization algorithms, we use the **first-order oracle** model.

**Definition (First-Order Oracle).** A first-order oracle for $f$ is a black box that, given a query point $\boldsymbol{\theta} \in \mathbb{R}^n$, returns the pair $(f(\boldsymbol{\theta}), \nabla f(\boldsymbol{\theta}))$.

The **iteration complexity** of an algorithm is the minimum number of oracle calls needed to produce an $\epsilon$-accurate solution, i.e., a point $\boldsymbol{\theta}$ satisfying $f(\boldsymbol{\theta}) - f^* \leq \epsilon$ or $\lVert \nabla f(\boldsymbol{\theta}) \rVert \leq \epsilon$.

**Why this model matters:** It abstracts away the cost of computing the gradient and focuses on the number of gradient evaluations. For neural networks, each oracle call is one forward + backward pass, which dominates the total training time. Reducing iteration complexity directly reduces training cost.

### 2.3 Descent Lemma and Sufficient Decrease

The descent lemma is the fundamental inequality that guarantees GD makes progress at each step.

**Lemma (Descent Lemma).** Let $f: \mathbb{R}^n \to \mathbb{R}$ be differentiable and $L$-smooth, i.e., $\nabla f$ is $L$-Lipschitz:

$$\lVert \nabla f(\mathbf{x}) - \nabla f(\mathbf{y}) \rVert_2 \leq L \lVert \mathbf{x} - \mathbf{y} \rVert_2 \quad \forall \mathbf{x}, \mathbf{y} \in \mathbb{R}^n$$

Then for any $\mathbf{x}, \mathbf{d} \in \mathbb{R}^n$:

$$f(\mathbf{x} + \mathbf{d}) \leq f(\mathbf{x}) + \nabla f(\mathbf{x})^\top \mathbf{d} + \frac{L}{2} \lVert \mathbf{d} \rVert_2^2$$

**Proof.** By the fundamental theorem of calculus:

$$f(\mathbf{x} + \mathbf{d}) - f(\mathbf{x}) = \int_0^1 \nabla f(\mathbf{x} + \tau \mathbf{d})^\top \mathbf{d} \, d\tau$$

$$= \nabla f(\mathbf{x})^\top \mathbf{d} + \int_0^1 (\nabla f(\mathbf{x} + \tau \mathbf{d}) - \nabla f(\mathbf{x}))^\top \mathbf{d} \, d\tau$$

By Cauchy-Schwarz and the $L$-smoothness assumption:

$$\left| \int_0^1 (\nabla f(\mathbf{x} + \tau \mathbf{d}) - \nabla f(\mathbf{x}))^\top \mathbf{d} \, d\tau \right| \leq \int_0^1 L\tau \lVert \mathbf{d} \rVert_2^2 \, d\tau = \frac{L}{2} \lVert \mathbf{d} \rVert_2^2$$

Therefore $f(\mathbf{x} + \mathbf{d}) \leq f(\mathbf{x}) + \nabla f(\mathbf{x})^\top \mathbf{d} + \frac{L}{2} \lVert \mathbf{d} \rVert_2^2$. $\blacksquare$

**Corollary (Sufficient Decrease).** Setting $\mathbf{d} = -\eta \nabla f(\mathbf{x})$ in the descent lemma:

$$f(\mathbf{x} - \eta \nabla f(\mathbf{x})) \leq f(\mathbf{x}) - \eta \lVert \nabla f(\mathbf{x}) \rVert_2^2 + \frac{L\eta^2}{2} \lVert \nabla f(\mathbf{x}) \rVert_2^2$$

$$= f(\mathbf{x}) - \eta\left(1 - \frac{L\eta}{2}\right) \lVert \nabla f(\mathbf{x}) \rVert_2^2$$

When $0 < \eta < 2/L$, the coefficient $\eta(1 - L\eta/2) > 0$, so $f$ strictly decreases unless $\nabla f(\mathbf{x}) = \mathbf{0}$. The optimal choice $\eta = 1/L$ gives:

$$f(\mathbf{x} - \tfrac{1}{L} \nabla f(\mathbf{x})) \leq f(\mathbf{x}) - \frac{1}{2L} \lVert \nabla f(\mathbf{x}) \rVert_2^2$$

This inequality is the starting point for every convergence proof in this section.

**For AI:** The smoothness constant $L$ of a neural network loss depends on the architecture, activation functions, and weight magnitudes. For a network with Lipschitz activations and bounded weights, $L$ can be bounded in terms of the product of layer norms. In practice, $L$ is unknown and the learning rate is tuned empirically, but the theory tells us $\eta < 2/L$ is necessary for monotonic decrease.

### 2.4 Step Size Regimes

The choice of step size $\eta_t$ fundamentally affects convergence behavior.

| Regime | Formula | Use Case | Convergence |
| --- | --- | --- | --- |
| **Constant** | $\eta_t = \eta$ | Most common in DL | Converges to neighborhood of optimum |
| **Diminishing** | $\eta_t = \frac{c}{t + c_0}$ | Theoretical guarantees | Converges to exact optimum (slow) |
| **Diminishing (slow)** | $\eta_t = \frac{c}{\sqrt{t}}$ | Non-convex stochastic | Sublinear convergence |
| **Exact line search** | $\eta_t = \arg\min_\eta f(\boldsymbol{\theta}_t - \eta \mathbf{g}_t)$ | Quadratics, theory | Optimal per-step decrease |
| **Backtracking** | Start with $\bar{\eta}$, halve until Armijo condition | General smooth $f$ | Guaranteed decrease |

**Constant step size** is the default in deep learning. With $\eta$ small enough, GD converges to a neighborhood of the optimum. For strongly convex functions, constant $\eta$ gives linear convergence to a ball of radius $O(\eta)$ around $\boldsymbol{\theta}^*$.

**Diminishing step sizes** are primarily of theoretical interest. They guarantee convergence to the exact optimum but require $\sum_t \eta_t = \infty$ (to reach the optimum) and $\sum_t \eta_t^2 < \infty$ (to control variance). The canonical choice $\eta_t = O(1/t)$ satisfies both but converges very slowly in practice.

**For AI:** Modern LLM training uses constant learning rates during the main training phase, with a warmup period (linearly increasing from 0 to $\eta_{\max}$) and a cooldown period (cosine decay to near 0). The constant phase dominates the training trajectory.

### 2.5 L-Smoothness Revisited

From [Convex Optimization](../01-Convex-Optimization/notes.md#42-smoothness-lipschitz-gradient), a function $f$ is $L$-smooth if $\nabla f$ is $L$-Lipschitz continuous. We collect the equivalent characterizations here for reference.

**Proposition (Equivalent Characterizations of L-Smoothness).** For a differentiable convex function $f: \mathbb{R}^n \to \mathbb{R}$, the following are equivalent:

1. **Lipschitz gradient:** $\lVert \nabla f(\mathbf{x}) - \nabla f(\mathbf{y}) \rVert_2 \leq L \lVert \mathbf{x} - \mathbf{y} \rVert_2$
2. **Quadratic upper bound:** $f(\mathbf{y}) \leq f(\mathbf{x}) + \nabla f(\mathbf{x})^\top (\mathbf{y} - \mathbf{x}) + \frac{L}{2} \lVert \mathbf{y} - \mathbf{x} \rVert_2^2$
3. **Gradient co-coercivity:** $\langle \nabla f(\mathbf{x}) - \nabla f(\mathbf{y}), \mathbf{x} - \mathbf{y} \rangle \geq \frac{1}{L} \lVert \nabla f(\mathbf{x}) - \nabla f(\mathbf{y}) \rVert_2^2$
4. **Hessian bound** (if $f$ is twice differentiable): $\nabla^2 f(\mathbf{x}) \preceq L I_n$

**Examples of L-smooth functions:**

1. **Quadratic:** $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top A \mathbf{x} + \mathbf{b}^\top \mathbf{x}$ with $A \succeq 0$. Here $L = \lambda_{\max}(A)$, the largest eigenvalue of $A$.

2. **Logistic regression loss:** $f(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^n \log(1 + e^{-y_i \mathbf{w}^\top \mathbf{x}^{(i)}})$. Here $L = \frac{1}{4n} \lambda_{\max}(X^\top X)$, since the sigmoid derivative is bounded by $1/4$.

3. **Neural network with bounded weights:** For a network with Lipschitz activations (ReLU, sigmoid) and bounded weight matrices, the loss is $L$-smooth with $L$ depending on the product of layer norms.

**Non-examples (not L-smooth):**

1. **$f(x) = |x|$** — not differentiable at 0, so not $L$-smooth. This is why subgradient methods are needed for L1 regularization.

2. **$f(x) = x^4$** — $\nabla f(x) = 4x^3$ is not Lipschitz on $\mathbb{R}$ (unbounded derivative of the gradient). GD can diverge on this function with any fixed step size if started far enough from the optimum.

---

## 3. Convergence Theory — Convex Functions

This section establishes the convergence rate of gradient descent for convex, L-smooth functions. This is the simplest non-trivial setting: the function has a global minimum, the gradient provides useful descent information everywhere, and the smoothness bound controls how far we can trust the linear approximation.

### 3.1 One-Step Progress Inequality

The key technical ingredient is a bound on how much the squared distance to the optimum decreases in one step.

**Lemma (One-Step Progress).** Let $f$ be convex and $L$-smooth with minimizer $\boldsymbol{\theta}^*$. For GD with constant step size $\eta \leq 1/L$:

$$\lVert \boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^* \rVert_2^2 \leq \lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert_2^2 - 2\eta(f(\boldsymbol{\theta}_t) - f^*)$$

**Proof.** Expanding the update:

$$\lVert \boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^* \rVert_2^2 = \lVert \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t) - \boldsymbol{\theta}^* \rVert_2^2$$

$$= \lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert_2^2 - 2\eta \nabla f(\boldsymbol{\theta}_t)^\top (\boldsymbol{\theta}_t - \boldsymbol{\theta}^*) + \eta^2 \lVert \nabla f(\boldsymbol{\theta}_t) \rVert_2^2$$

By convexity, $\nabla f(\boldsymbol{\theta}_t)^\top (\boldsymbol{\theta}_t - \boldsymbol{\theta}^*) \geq f(\boldsymbol{\theta}_t) - f^*$. Also, by the descent lemma with $\mathbf{x} = \boldsymbol{\theta}_t$ and $\mathbf{d} = -\frac{1}{L}\nabla f(\boldsymbol{\theta}_t)$:

$$f(\boldsymbol{\theta}_t - \tfrac{1}{L}\nabla f(\boldsymbol{\theta}_t)) \leq f(\boldsymbol{\theta}_t) - \frac{1}{2L}\lVert \nabla f(\boldsymbol{\theta}_t) \rVert_2^2 \leq f^*$$

So $\lVert \nabla f(\boldsymbol{\theta}_t) \rVert_2^2 \leq 2L(f(\boldsymbol{\theta}_t) - f^*)$. Substituting both bounds:

$$\lVert \boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^* \rVert_2^2 \leq \lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert_2^2 - 2\eta(f(\boldsymbol{\theta}_t) - f^*) + 2\eta^2 L(f(\boldsymbol{\theta}_t) - f^*)$$

$$= \lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert_2^2 - 2\eta(1 - \eta L)(f(\boldsymbol{\theta}_t) - f^*)$$

When $\eta \leq 1/L$, we have $1 - \eta L \geq 0$, and since $\eta \leq 1/L$ implies $\eta(1-\eta L) \geq \eta(1-1) = 0$, the tightest bound uses $\eta(1-\eta L) \geq \eta$ when $\eta L \leq 1/2$. For $\eta = 1/L$:

$$\lVert \boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^* \rVert_2^2 \leq \lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert_2^2 - \frac{2}{L}(f(\boldsymbol{\theta}_t) - f^*) \quad \blacksquare$$

### 3.2 O(1/T) Rate for Convex + L-Smooth Functions

**Theorem (Convergence Rate for Convex Functions).** Let $f$ be convex and $L$-smooth with minimizer $\boldsymbol{\theta}^*$. For GD with $\eta = 1/L$:

$$f(\boldsymbol{\theta}_T) - f^* \leq \frac{L \lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert_2^2}{2T} = O\left(\frac{1}{T}\right)$$

**Proof.** From the one-step progress lemma with $\eta = 1/L$:

$$\lVert \boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^* \rVert_2^2 \leq \lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert_2^2 - \frac{2}{L}(f(\boldsymbol{\theta}_t) - f^*)$$

Rearranging: $f(\boldsymbol{\theta}_t) - f^* \leq \frac{L}{2}(\lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert_2^2 - \lVert \boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^* \rVert_2^2)$

Summing from $t = 0$ to $T-1$:

$$\sum_{t=0}^{T-1} (f(\boldsymbol{\theta}_t) - f^*) \leq \frac{L}{2}(\lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert_2^2 - \lVert \boldsymbol{\theta}_T - \boldsymbol{\theta}^* \rVert_2^2) \leq \frac{L \lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert_2^2}{2}$$

Since $f(\boldsymbol{\theta}_t)$ is non-increasing (by the descent lemma), $f(\boldsymbol{\theta}_T) - f^* \leq \frac{1}{T}\sum_{t=0}^{T-1} (f(\boldsymbol{\theta}_t) - f^*)$:

$$f(\boldsymbol{\theta}_T) - f^* \leq \frac{L \lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert_2^2}{2T} \quad \blacksquare$$

**Interpretation.** The suboptimality decreases as $O(1/T)$. To reduce the error by a factor of 10, you need 10 times more iterations. This is sublinear convergence — slower than the linear (geometric) rate we will see for strongly convex functions.

**For AI:** In LLM training, $T$ is measured in training steps (typically $10^5$ to $10^7$). The $O(1/T)$ rate means that the loss decreases rapidly at first and then plateaus. This is exactly what training curves look like: a steep initial drop followed by a long tail of gradual improvement. The constant $L \lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert_2^2 / 2$ depends on initialization — good initialization (small $\lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert$) accelerates training.

### 3.3 Optimal Step Size η = 1/L

The choice $\eta = 1/L$ is optimal in the sense that it maximizes the guaranteed decrease per step in the descent lemma. Setting $\mathbf{d} = -\eta \nabla f(\mathbf{x})$:

$$f(\mathbf{x} - \eta \nabla f(\mathbf{x})) \leq f(\mathbf{x}) - \eta\left(1 - \frac{L\eta}{2}\right)\lVert \nabla f(\mathbf{x}) \rVert_2^2$$

The coefficient $\eta(1 - L\eta/2)$ is maximized at $\eta = 1/L$, giving a decrease of $\frac{1}{2L}\lVert \nabla f(\mathbf{x}) \rVert_2^2$ per step.

**What if L is unknown?** In practice, $L$ is rarely known exactly. Common strategies:

1. **Conservative estimate:** Choose $\eta$ smaller than you think $1/L$ is. Safe but slow.
2. **Backtracking line search:** Start with a large $\eta$ and halve until the Armijo condition is satisfied (see §8.2).
3. **Adaptive methods:** Use Adam or similar, which effectively estimate per-parameter smoothness (see [Adaptive Learning Rate](../07-Adaptive-Learning-Rate/notes.md)).

**For AI:** In deep learning, $L$ is effectively unknown and time-varying (the loss landscape changes as parameters move). Practitioners use a "maximal stable learning rate" found by empirical LR range tests: train with exponentially increasing LR and observe when the loss diverges. The largest stable LR is approximately $2/L$.

### 3.4 Function Value vs. Iterate Convergence

The $O(1/T)$ bound is on the **function value** suboptimality $f(\boldsymbol{\theta}_T) - f^*$. What about the **iterates** $\boldsymbol{\theta}_T$?

For convex (but not strongly convex) functions, the iterates need not converge to a unique point — the set of minimizers may be a subspace. However, the distance to the solution set is non-increasing:

$$\text{dist}(\boldsymbol{\theta}_{t+1}, \Theta^*) \leq \text{dist}(\boldsymbol{\theta}_t, \Theta^*)$$

where $\Theta^* = \arg\min f$ and $\text{dist}(\boldsymbol{\theta}, \Theta^*) = \inf_{\boldsymbol{\theta}^* \in \Theta^*} \lVert \boldsymbol{\theta} - \boldsymbol{\theta}^* \rVert_2$.

**Averaged iterates.** A slightly stronger result uses the averaged iterate $\bar{\boldsymbol{\theta}}_T = \frac{1}{T}\sum_{t=0}^{T-1} \boldsymbol{\theta}_t$:

$$f(\bar{\boldsymbol{\theta}}_T) - f^* \leq \frac{L \lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert_2^2}{2T}$$

This holds even without monotonicity of $f(\boldsymbol{\theta}_t)$. In practice, the last iterate $\boldsymbol{\theta}_T$ typically performs as well as or better than the average for smooth convex functions.

### 3.5 Tightness: Worst-Case Examples

The $O(1/T)$ rate is **tight** — there exist convex, L-smooth functions for which GD achieves exactly this rate and no better.

**Example (Worst-case quadratic).** Consider $f(\mathbf{x}) = \frac{L}{2} x_1^2$ on $\mathbb{R}^2$. This is convex and $L$-smooth. Starting from $\mathbf{x}_0 = (1, 0)$, GD with $\eta = 1/L$ gives $x_{1,t} = 0$ after one step, but the distance $\lVert \mathbf{x}_0 - \mathbf{x}^* \rVert^2 = 1$ bounds the rate as $1/(2T)$.

A more instructive example is the **worst-case function** constructed by Nesterov. For any first-order method, there exists a convex, L-smooth function and an initialization such that:

$$f(\boldsymbol{\theta}_T) - f^* \geq \frac{3L \lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert_2^2}{32(T+1)^2}$$

This lower bound shows that GD's $O(1/T)$ rate cannot be improved without additional assumptions (like strong convexity). However, it also reveals a gap: the lower bound is $O(1/T^2)$, and Nesterov's accelerated method achieves this. GD is suboptimal by a factor of $O(T)$.

```text
CONVERGENCE RATE SUMMARY (Convex)
════════════════════════════════════════════════════════════════════════

  Method              │ Rate          │ Optimal?
  ────────────────────┼───────────────┼─────────────────────
  GD (η = 1/L)        │ O(1/T)        │ No — off by O(T)
  GD (optimal η)      │ O(1/T)        │ No — off by O(T)
  Nesterov (AGD)      │ O(1/T²)       │ Yes — matches lower bound
  Lower bound         │ Ω(1/T²)       │ —

  The gap between GD and the lower bound is the motivation for
  acceleration. Nesterov's method closes it completely.

════════════════════════════════════════════════════════════════════════
```

---

## 4. Convergence Theory — Strongly Convex Functions

When the objective is not just convex but **strongly convex**, gradient descent converges much faster: linearly (geometrically), meaning the error is multiplied by a constant factor $< 1$ at each step. This is the best convergence rate achievable by any first-order method without additional structure.

### 4.1 Linear (Geometric) Convergence

Recall from [Convex Optimization](../01-Convex-Optimization/notes.md#41-strong-convexity) that $f$ is $\mu$-strongly convex if:

$$f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^\top (\mathbf{y} - \mathbf{x}) + \frac{\mu}{2} \lVert \mathbf{y} - \mathbf{x} \rVert_2^2 \quad \forall \mathbf{x}, \mathbf{y}$$

This quadratic lower bound, combined with the $L$-smoothness upper bound, sandwiches $f$ between two quadratics. The ratio of their curvatures — the condition number $\kappa = L/\mu$ — determines the convergence speed.

**Theorem (Linear Convergence).** Let $f$ be $\mu$-strongly convex and $L$-smooth with unique minimizer $\boldsymbol{\theta}^*$. For GD with $\eta = 1/L$:

$$\lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert_2^2 \leq \left(1 - \frac{\mu}{L}\right)^t \lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert_2^2$$

$$f(\boldsymbol{\theta}_t) - f^* \leq \frac{L}{2}\left(1 - \frac{\mu}{L}\right)^t \lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert_2^2 = O\left(\left(1 - \frac{1}{\kappa}\right)^t\right)$$

**Proof.** From the GD update and strong convexity:

$$\lVert \boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^* \rVert_2^2 = \lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* - \eta \nabla f(\boldsymbol{\theta}_t) \rVert_2^2$$

$$= \lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert_2^2 - 2\eta \nabla f(\boldsymbol{\theta}_t)^\top(\boldsymbol{\theta}_t - \boldsymbol{\theta}^*) + \eta^2 \lVert \nabla f(\boldsymbol{\theta}_t) \rVert_2^2$$

By strong convexity: $\nabla f(\boldsymbol{\theta}_t)^\top(\boldsymbol{\theta}_t - \boldsymbol{\theta}^*) \geq f(\boldsymbol{\theta}_t) - f^* + \frac{\mu}{2}\lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert_2^2$

By $L$-smoothness (co-coercivity): $\lVert \nabla f(\boldsymbol{\theta}_t) \rVert_2^2 \leq 2L(f(\boldsymbol{\theta}_t) - f^*)$

Substituting with $\eta = 1/L$:

$$\lVert \boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^* \rVert_2^2 \leq \lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert_2^2 - \frac{2}{L}\left(f(\boldsymbol{\theta}_t) - f^* + \frac{\mu}{2}\lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert_2^2\right) + \frac{2}{L}(f(\boldsymbol{\theta}_t) - f^*)$$

$$= \left(1 - \frac{\mu}{L}\right)\lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert_2^2$$

By induction: $\lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert_2^2 \leq (1 - \mu/L)^t \lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert_2^2$. The function value bound follows from $L$-smoothness: $f(\boldsymbol{\theta}_t) - f^* \leq \frac{L}{2}\lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert_2^2$. $\blacksquare$

### 4.2 Rate: (1 − μ/L)^T and the Condition Number κ

The convergence rate $1 - \mu/L = 1 - 1/\kappa$ is controlled entirely by the **condition number** $\kappa = L/\mu$.

| κ | Rate per step | Steps to 10× reduction | Interpretation |
| --- | --- | --- | --- |
| 1 | 0 | 1 | Perfectly conditioned (spherical contours) |
| 10 | 0.9 | ~22 | Well-conditioned |
| 100 | 0.99 | ~230 | Moderately ill-conditioned |
| 1,000 | 0.999 | ~2,300 | Ill-conditioned |
| 10⁶ | 0.999999 | ~2,300,000 | Severely ill-conditioned |

**For AI:** The condition number of a neural network's loss Hessian varies dramatically during training. Early in training, $\kappa$ can be $10^6$ or larger (the "edge of stability" regime). As training progresses, $\kappa$ typically decreases. This is why learning rate warmup helps: the initial high-$\kappa$ phase needs a small LR, and once $\kappa$ decreases, the LR can safely increase.

The condition number also explains why **feature scaling** matters in linear models. If features have vastly different scales, the Hessian of the least-squares loss $X^\top X$ has a large condition number, and GD zigzags. Preprocessing (standardization, whitening) reduces $\kappa$ and accelerates convergence.

### 4.3 Optimal Step Size η = 2/(L + μ)

The step size $\eta = 1/L$ used in the theorem above is safe but not optimal. The **optimal constant step size** for a $\mu$-strongly convex, $L$-smooth quadratic is:

$$\eta^* = \frac{2}{L + \mu}$$

With this choice, the convergence rate improves to:

$$\lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert_2 \leq \left(\frac{\kappa - 1}{\kappa + 1}\right)^t \lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert_2$$

For large $\kappa$, this gives $\frac{\kappa-1}{\kappa+1} \approx 1 - \frac{2}{\kappa}$, which is a factor of 2 improvement over $1 - \frac{1}{\kappa}$.

**Proof sketch.** For the quadratic $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top A \mathbf{x}$ with $\mu I \preceq A \preceq LI$, the GD iteration matrix is $I - \eta A$. Its spectral radius is $\max(|1 - \eta\mu|, |1 - \eta L|)$, minimized when $1 - \eta\mu = -(1 - \eta L)$, i.e., $\eta = 2/(L+\mu)$. $\blacksquare$

### 4.4 Lower Complexity Bounds for First-Order Methods

No first-order method can converge faster than a certain rate on the class of $\mu$-strongly convex, $L$-smooth functions.

**Theorem (Nesterov's Lower Bound).** For any first-order method and any $T < n/2$, there exists a $\mu$-strongly convex, $L$-smooth function $f$ such that:

$$\lVert \boldsymbol{\theta}_T - \boldsymbol{\theta}^* \rVert_2^2 \geq \frac{1}{2}\left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^{2T} \lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert_2^2$$

This means the best possible rate is $O\left(\left(1 - \frac{1}{\sqrt{\kappa}}\right)^T\right)$, which is a **quadratic improvement** over GD's $O\left(\left(1 - \frac{1}{\kappa}\right)^T\right)$ when $\kappa$ is large.

**The gap:** GD requires $O(\kappa \log(1/\epsilon))$ iterations to reach $\epsilon$-accuracy. The lower bound says $O(\sqrt{\kappa} \log(1/\epsilon))$ is achievable. Nesterov's accelerated method (§7) closes this gap.

### 4.5 What Happens When η Is Too Large

The convergence proofs require $\eta \leq 1/L$ (or $\eta \leq 2/(L+\mu)$ for the optimal choice). What happens when this condition is violated?

**Theorem (Divergence for Large Step Size).** Let $f(\mathbf{x}) = \frac{L}{2}\lVert \mathbf{x} \rVert_2^2$. For GD with $\eta > 2/L$:

$$\lVert \boldsymbol{\theta}_{t+1} \rVert_2 = |1 - \eta L| \cdot \lVert \boldsymbol{\theta}_t \rVert_2 > \lVert \boldsymbol{\theta}_t \rVert_2$$

The iterates diverge geometrically: $\lVert \boldsymbol{\theta}_t \rVert_2 = |1 - \eta L|^t \lVert \boldsymbol{\theta}_0 \rVert_2 \to \infty$.

**The edge of stability.** When $\eta$ is slightly below $2/L$, GD exhibits oscillatory behavior: the iterates bounce back and forth across the minimum. This is the "edge of stability" regime observed in deep learning training (Cohen et al., 2021), where the effective sharpness $L$ self-stabilizes near $2/\eta$.

**For AI:** In LLM training, the learning rate is typically set near the stability boundary. The warmup phase gradually increases $\eta$ from 0 to $\eta_{\max}$, allowing the model to move into a region of the parameter space where the loss is smoother (smaller $L$), so that $\eta_{\max} < 2/L$ holds. If the LR is set too high from the start, the loss diverges — a common failure mode.

```text
STEP SIZE EFFECTS ON CONVERGENCE
════════════════════════════════════════════════════════════════════════

  η range              │ Behavior              │ Convergence
  ─────────────────────┼───────────────────────┼────────────────────
  0 < η < 1/L          │ Monotone decrease     │ Guaranteed
  η = 1/L              │ Optimal decrease      │ Guaranteed (safe)
  1/L < η < 2/L        │ Oscillatory decrease  │ May converge
  η = 2/L              │ Oscillation           │ Stagnates
  η > 2/L              │ Divergence            │ Loss → ∞

  The stability boundary η = 2/L is sharp for quadratic objectives.
  For non-quadratic functions, the boundary may be lower.

════════════════════════════════════════════════════════════════════════
```

---

## 5. Convergence Theory — Non-Convex Functions

Deep learning losses are emphatically non-convex. The loss landscape of a neural network with even a single hidden layer has exponentially many local minima and saddle points. Yet gradient descent works remarkably well in practice. This section develops the theoretical tools for understanding GD in the non-convex setting.

### 5.1 Convergence to Stationary Points

For non-convex functions, we cannot guarantee convergence to a global (or even local) minimum. The best we can hope for is convergence to a **stationary point** — a point where the gradient is zero (or small).

**Theorem (Convergence to Stationary Points).** Let $f$ be $L$-smooth (not necessarily convex) and bounded below by $f^*$. For GD with $\eta = 1/L$:

$$\min_{0 \leq t \leq T-1} \lVert \nabla f(\boldsymbol{\theta}_t) \rVert_2^2 \leq \frac{2L(f(\boldsymbol{\theta}_0) - f^*)}{T}$$

Equivalently:

$$\min_{0 \leq t \leq T-1} \lVert \nabla f(\boldsymbol{\theta}_t) \rVert_2 \leq \sqrt{\frac{2L(f(\boldsymbol{\theta}_0) - f^*)}{T}} = O\left(\frac{1}{\sqrt{T}}\right)$$

**Proof.** From the descent lemma with $\eta = 1/L$:

$$f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{1}{2L}\lVert \nabla f(\boldsymbol{\theta}_t) \rVert_2^2$$

Summing from $t = 0$ to $T-1$:

$$f(\boldsymbol{\theta}_T) \leq f(\boldsymbol{\theta}_0) - \frac{1}{2L}\sum_{t=0}^{T-1} \lVert \nabla f(\boldsymbol{\theta}_t) \rVert_2^2$$

Rearranging and using $f(\boldsymbol{\theta}_T) \geq f^*$:

$$\sum_{t=0}^{T-1} \lVert \nabla f(\boldsymbol{\theta}_t) \rVert_2^2 \leq 2L(f(\boldsymbol{\theta}_0) - f^*)$$

The minimum is bounded by the average:

$$\min_{0 \leq t \leq T-1} \lVert \nabla f(\boldsymbol{\theta}_t) \rVert_2^2 \leq \frac{1}{T}\sum_{t=0}^{T-1} \lVert \nabla f(\boldsymbol{\theta}_t) \rVert_2^2 \leq \frac{2L(f(\boldsymbol{\theta}_0) - f^*)}{T} \quad \blacksquare$$

**Interpretation.** After $T$ iterations, at least one iterate has gradient norm $O(1/\sqrt{T})$. This does not tell us *which* iterate — in practice, we use the last iterate or the one with the smallest gradient norm.

### 5.2 Gradient Norm Decay: O(1/√T)

The $O(1/\sqrt{T})$ rate on the gradient norm is the standard convergence guarantee for non-convex optimization. It is slower than the $O(1/T)$ rate for convex functions, reflecting the additional difficulty of non-convex landscapes.

**Average gradient norm.** A slightly stronger result bounds the *average* gradient norm:

$$\frac{1}{T}\sum_{t=0}^{T-1} \lVert \nabla f(\boldsymbol{\theta}_t) \rVert_2^2 \leq \frac{2L(f(\boldsymbol{\theta}_0) - f^*)}{T}$$

This means the gradient norms decrease on average, even if individual iterates may have large gradients.

**For AI:** In neural network training, the gradient norm typically follows a characteristic pattern: it starts large, decreases rapidly during the first phase of training, then plateaus at a non-zero value. This plateau reflects the fact that the loss landscape is non-convex — GD never reaches a true stationary point, but rather oscillates in a region where the gradient is small but non-zero. The plateau level depends on the learning rate: smaller LR → smaller plateau.

### 5.3 Why Non-Convex GD Still Works in Practice

If the theory only guarantees convergence to stationary points (which could be local minima or saddle points), why does GD produce high-quality models?

**Overparameterization helps.** When a neural network has more parameters than training examples, the loss landscape becomes more benign. For wide neural networks, almost all local minima are nearly global (Kawaguchi, 2016). The overparameterized regime has fewer spurious local minima and the ones that exist have similar loss values.

**Saddle points are escapable.** Saddle points are far more common than local minima in high-dimensional non-convex optimization. However, saddle points are unstable: any perturbation in the direction of negative curvature causes escape. GD with random initialization or noise (from mini-batching) escapes saddle points efficiently (Lee et al., 2017).

**Flat minima generalize.** GD has an implicit bias toward flat minima — regions where the loss is small in a neighborhood, not just at a point. Flat minima correspond to solutions that are robust to perturbations, which correlates with good generalization (Hochreiter & Schmidhuber, 1997; Keskar et al., 2017).

**The lottery ticket hypothesis.** Some initializations lead to subnetworks that train much faster and generalize better (Frankle & Carbin, 2019). GD finds these "winning tickets" when the initialization is right, which is why initialization schemes (Xavier, He) are so important.

### 5.4 Saddle Points and the Need for Stochasticity

At a saddle point $\boldsymbol{\theta}_s$, we have $\nabla f(\boldsymbol{\theta}_s) = \mathbf{0}$ but $\boldsymbol{\theta}_s$ is not a local minimum. The Hessian $\nabla^2 f(\boldsymbol{\theta}_s)$ has at least one negative eigenvalue.

**Strict saddle property.** A function satisfies the strict saddle property if every stationary point is either a local minimum or has a direction of negative curvature: $\lambda_{\min}(\nabla^2 f(\boldsymbol{\theta})) < 0$ for all saddle points $\boldsymbol{\theta}$.

**Theorem (GD Escapes Strict Saddles).** For a $C^2$ function satisfying the strict saddle property, GD with random initialization converges to a local minimum with probability 1 (Lee et al., 2017).

However, the time to escape a saddle point can be exponential in the dimension for vanilla GD. Adding noise (as in SGD) accelerates escape dramatically. This is one reason why **stochastic gradient descent outperforms full-batch GD** in deep learning, even when the full gradient can be computed — the noise helps escape saddle points. For the full treatment, see [Stochastic Optimization](../05-Stochastic-Optimization/notes.md).

### 5.5 PL (Polyak-Łojasiewicz) Condition

The PL condition is a relaxation of strong convexity that holds for many overparameterized neural networks and gives linear convergence without convexity.

**Definition (PL Condition).** A function $f$ satisfies the PL condition with constant $\mu > 0$ if:

$$\frac{1}{2}\lVert \nabla f(\boldsymbol{\theta}) \rVert_2^2 \geq \mu(f(\boldsymbol{\theta}) - f^*) \quad \forall \boldsymbol{\theta}$$

**Theorem (Linear Convergence under PL).** If $f$ satisfies the PL condition with constant $\mu$ and is $L$-smooth, then GD with $\eta = 1/L$ converges linearly:

$$f(\boldsymbol{\theta}_t) - f^* \leq \left(1 - \frac{\mu}{L}\right)^t (f(\boldsymbol{\theta}_0) - f^*)$$

**Proof.** From the descent lemma: $f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{1}{2L}\lVert \nabla f(\boldsymbol{\theta}_t) \rVert_2^2$. By the PL condition: $\lVert \nabla f(\boldsymbol{\theta}_t) \rVert_2^2 \geq 2\mu(f(\boldsymbol{\theta}_t) - f^*)$. Substituting:

$$f(\boldsymbol{\theta}_{t+1}) - f^* \leq (f(\boldsymbol{\theta}_t) - f^*) - \frac{\mu}{L}(f(\boldsymbol{\theta}_t) - f^*) = \left(1 - \frac{\mu}{L}\right)(f(\boldsymbol{\theta}_t) - f^*) \quad \blacksquare$$

**Key difference from strong convexity:** The PL condition does not imply convexity. The function can have multiple global minima and non-convex regions, but as long as the gradient norm controls the suboptimality, GD converges linearly.

**For AI:** Overparameterized neural networks often satisfy the PL condition (or a variant) in a neighborhood of the solution. This explains why GD converges quickly even though the loss is non-convex. The PL constant $\mu$ depends on the architecture and initialization — wider networks tend to have larger $\mu$, explaining why they train faster.

```text
CONVERGENCE RATES ACROSS SETTINGS
════════════════════════════════════════════════════════════════════════

  Setting                  │ Rate              │ Iterations for ε
  ─────────────────────────┼───────────────────┼────────────────────
  Convex + L-smooth        │ O(1/T)            │ O(1/ε)
  Strongly convex + smooth │ O((1-1/κ)^T)      │ O(κ log(1/ε))
  Non-convex (stationary)  │ O(1/√T) on ‖∇f‖  │ O(1/ε²)
  PL condition             │ O((1-μ/L)^T)      │ O(L/μ · log(1/ε))
  Nesterov (convex)        │ O(1/T²)           │ O(1/√ε)
  Nesterov (strongly cvx)  │ O((1-1/√κ)^T)    │ O(√κ log(1/ε))

  The PL condition bridges convex and non-convex theory:
  it gives linear convergence without requiring convexity.

════════════════════════════════════════════════════════════════════════
```

---

## 6. Momentum

Momentum is the most widely used acceleration technique for gradient descent. It addresses the zigzag problem on ill-conditioned objectives by accumulating a velocity vector that smooths out oscillations and amplifies progress in consistent descent directions.

### 6.1 Polyak Momentum

**Definition (Polyak Momentum / Heavy Ball).** Given parameters $\eta > 0$ (learning rate) and $\beta \in [0, 1)$ (momentum coefficient), the iterates are:

$$\mathbf{v}_{t+1} = \beta \mathbf{v}_t + \nabla f(\boldsymbol{\theta}_t)$$
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \mathbf{v}_{t+1}$$

where $\mathbf{v}_0 = \mathbf{0}$. The velocity $\mathbf{v}_t$ is an exponentially weighted moving average of past gradients.

**Unrolling the velocity:**

$$\mathbf{v}_t = \sum_{k=0}^{t-1} \beta^{t-1-k} \nabla f(\boldsymbol{\theta}_k)$$

When $\beta$ is close to 1, the velocity accumulates many past gradients, giving the algorithm "inertia." Directions where the gradient consistently points the same way are amplified (the sum grows), while directions where the gradient oscillates cancel out (positive and negative terms offset).

**Alternative formulation.** Some implementations (including PyTorch's SGD with momentum) use a slightly different ordering:

$$\mathbf{v}_{t+1} = \beta \mathbf{v}_t + \nabla f(\boldsymbol{\theta}_t)$$
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \mathbf{v}_{t+1}$$

This is the "classical" form. The difference from Nesterov momentum is subtle but important: Polyak momentum computes the gradient at the current position, while Nesterov computes it at a lookahead position.

### 6.2 Quadratic Analysis: Characteristic Polynomial

To understand momentum's effect, analyze it on the quadratic $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top A \mathbf{x}$ where $A$ is symmetric positive definite with eigenvalues $\mu = \lambda_{\min} \leq \cdots \leq \lambda_{\max} = L$.

By decomposing into eigenvectors of $A$, the dynamics decouple into $n$ independent scalar recurrences. For each eigenvalue $\lambda$:

$$x_{t+1} = x_t - \eta v_{t+1}, \quad v_{t+1} = \beta v_t + \lambda x_t$$

Substituting: $x_{t+1} = x_t - \eta(\beta v_t + \lambda x_t) = (1 - \eta\lambda)x_t - \eta\beta v_t$

Also: $v_t = (x_t - x_{t+1})/\eta$, so:

$$x_{t+2} = (1 - \eta\lambda + \beta)x_{t+1} - \beta x_t$$

This is a second-order linear recurrence with characteristic polynomial:

$$r^2 - (1 - \eta\lambda + \beta)r + \beta = 0$$

The roots determine convergence. For convergence, we need both roots to have magnitude $< 1$. The optimal choice of $\beta$ and $\eta$ minimizes the maximum root magnitude across all eigenvalues $\lambda \in [\mu, L]$.

### 6.3 Optimal β for Quadratic Objectives

**Theorem (Optimal Momentum Parameters).** For a quadratic with condition number $\kappa = L/\mu$, the optimal parameters are:

$$\beta^* = \left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^2, \quad \eta^* = \frac{1}{L}$$

With these choices, the convergence rate is:

$$\lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert_2 = O\left(\left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^t\right) \approx O\left(\left(1 - \frac{1}{\sqrt{\kappa}}\right)^t\right)$$

This is a **quadratic improvement** over vanilla GD's $O((1 - 1/\kappa)^t)$ rate. For $\kappa = 10^6$, GD needs $O(10^6)$ iterations while momentum needs $O(10^3)$ — a 1000× speedup.

**Derivation sketch.** The characteristic polynomial $r^2 - (1 - \eta\lambda + \beta)r + \beta = 0$ has roots of equal magnitude when the discriminant is zero: $(1 - \eta\lambda + \beta)^2 = 4\beta$. For the worst-case eigenvalues $\lambda = \mu$ and $\lambda = L$, we want the same contraction factor. Solving the system gives $\beta = \left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^2$.

**For AI:** The standard momentum value $\beta = 0.9$ corresponds to $\kappa \approx 361$. This is a reasonable default for many ML problems. For very ill-conditioned problems (large $\kappa$), higher momentum ($\beta = 0.99$) can help, but too much momentum causes oscillations.

### 6.4 Momentum as Exponential Moving Average

The velocity update $v_{t+1} = \beta v_t + g_t$ can be rewritten as:

$$v_{t+1} = \beta v_t + (1 - \beta) \cdot \frac{g_t}{1 - \beta}$$

This shows that $v_{t+1}$ is a weighted average of the current gradient (scaled by $1/(1-\beta)$) and the previous velocity. The effective window size is approximately $1/(1-\beta)$:

| β | Effective window | Interpretation |
| --- | --- | --- |
| 0.5 | ~2 steps | Very little momentum |
| 0.9 | ~10 steps | Standard momentum |
| 0.95 | ~20 steps | Strong momentum |
| 0.99 | ~100 steps | Very strong momentum |

**Bias correction.** Like Adam, momentum has a bias at initialization: $v_0 = \mathbf{0}$ means the initial velocity is biased toward zero. The bias-corrected velocity is:

$$\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta^t}$$

For large $t$, the correction is negligible ($\beta^t \to 0$). But in the first few steps, it can be significant. Most implementations skip bias correction for momentum (unlike Adam), since the bias decays quickly for $\beta < 1$.

### 6.5 Heavy Ball Interpretation

The name "heavy ball" comes from the physical analogy: imagine a heavy ball rolling down a hill with friction. The ball's position is $\boldsymbol{\theta}$, its velocity is $\mathbf{v}$, the gradient is the slope of the hill, and $\beta$ controls the friction.

```text
HEAVY BALL ANALOGY
════════════════════════════════════════════════════════════════════════

  Without momentum (GD):            With momentum (Heavy Ball):

       ╲        ╱                        ╲        ╱
        ╲  •   ╱                          ╲  ╭─╮ ╱
         ╲↙   ╱                            ╲│●│╱   ← ball has inertia
          ╲ ╱                              ╰─╯
           •                                ╲  ╱
                                             ╲╱
                                              •

  GD: moves directly downhill.        Heavy Ball: carries momentum
  Zigzags on ill-conditioned          through flat regions and
  problems.                           dampens oscillations.

════════════════════════════════════════════════════════════════════════
```

In the continuous-time limit, the heavy ball method corresponds to the second-order ODE:

$$\ddot{\boldsymbol{\theta}}(t) + \frac{1 - \beta}{\eta}\dot{\boldsymbol{\theta}}(t) + \nabla f(\boldsymbol{\theta}(t)) = 0$$

This is a damped oscillator: the gradient provides the restoring force, and $(1-\beta)/\eta$ is the damping coefficient. When damping is too low (high $\beta$), the ball oscillates. When damping is too high (low $\beta$), it moves slowly.

### 6.6 Momentum in Practice: PyTorch SGD

PyTorch's `torch.optim.SGD` with momentum implements:

```python
v_t = β · v_{t-1} + g_t          # accumulate gradient
θ_t = θ_{t-1} - η · v_t          # update parameters
```

Note that this differs slightly from the textbook formulation in the ordering of operations. The practical effect is that PyTorch's momentum uses the *current* gradient in the velocity update, then immediately applies it. This is sometimes called "classical momentum" to distinguish it from Nesterov momentum.

**Recommended settings:**

- **Default:** $\beta = 0.9$, $\eta$ found by LR range test
- **Ill-conditioned problems:** $\beta = 0.99$, smaller $\eta$
- **With weight decay:** use AdamW instead (decoupled weight decay)

**For AI:** Momentum is used in virtually all LLM training, either as part of SGD with momentum or as a component of Adam (which uses momentum-like first-moment estimation). The momentum coefficient $\beta_1 = 0.9$ in Adam is the same as the standard SGD momentum value.

---

## 7. Nesterov Accelerated Gradient

Nesterov's accelerated gradient (NAG) is the theoretically optimal first-order method for convex functions. It achieves the $O(1/T^2)$ convergence rate that matches the lower bound, making it strictly superior to both vanilla GD and Polyak momentum in the convex setting.

### 7.1 NAG Update: Lookahead Gradient

**Definition (Nesterov Accelerated Gradient).** Given $f$ convex and $L$-smooth, initialize $\mathbf{y}_0 = \boldsymbol{\theta}_0$, $t_0 = 1$. For $k = 0, 1, 2, \ldots$:

$$\boldsymbol{\theta}_{k+1} = \mathbf{y}_k - \frac{1}{L} \nabla f(\mathbf{y}_k)$$
$$t_{k+1} = \frac{1 + \sqrt{1 + 4t_k^2}}{2}$$
$$\mathbf{y}_{k+1} = \boldsymbol{\theta}_{k+1} + \frac{t_k - 1}{t_{k+1}}(\boldsymbol{\theta}_{k+1} - \boldsymbol{\theta}_k)$$

The key difference from Polyak momentum: the gradient is evaluated at the **extrapolated point** $\mathbf{y}_k$, not at the current iterate $\boldsymbol{\theta}_k$. The point $\mathbf{y}_k$ is a lookahead position that anticipates where the iterates are heading.

**Alternative formulation (more similar to momentum):**

$$\mathbf{v}_{k+1} = \beta_k \mathbf{v}_k + \frac{1}{L}\nabla f(\boldsymbol{\theta}_k + \beta_k \mathbf{v}_k)$$
$$\boldsymbol{\theta}_{k+1} = \boldsymbol{\theta}_k - \mathbf{v}_{k+1}$$

where $\beta_k = \frac{t_k - 1}{t_{k+1}} \to 1$ as $k \to \infty$. The gradient is computed at $\boldsymbol{\theta}_k + \beta_k \mathbf{v}_k$ — the point we expect to reach if we continue with the current velocity.

### 7.2 O(1/T²) Rate for Convex Functions

**Theorem (Nesterov's Accelerated Rate).** For a convex, $L$-smooth function $f$:

$$f(\boldsymbol{\theta}_T) - f^* \leq \frac{2L \lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert_2^2}{(T+1)^2} = O\left(\frac{1}{T^2}\right)$$

This is a **quadratic improvement** over GD's $O(1/T)$ rate. To reduce the error by a factor of 100, GD needs 100× more iterations, while NAG needs only 10× more.

**Proof sketch.** The proof uses the technique of **estimate sequences**. Define a sequence of quadratic lower bounds $\gamma_k(\mathbf{x})$ that are constructed to satisfy:

$$f(\boldsymbol{\theta}_k) \leq \min_{\mathbf{x}} \gamma_k(\mathbf{x}) + \frac{L \lVert \mathbf{x} - \boldsymbol{\theta}^* \rVert_2^2}{2t_k^2}$$

The extrapolation step in NAG is designed to maintain this invariant. Since $t_k \geq (k+1)/2$, the error bound is $O(1/k^2)$. The full proof is technical; see Nesterov (2013), Theorem 2.2.1. $\blacksquare$

### 7.3 Geometric Interpretation: Estimate Sequences

The estimate sequence framework provides intuition for why NAG works. At each step, we maintain:

1. A **lower bound** on $f$: a quadratic function $\gamma_k(\mathbf{x})$ that satisfies $\gamma_k(\mathbf{x}) \leq f(\mathbf{x})$ for all $\mathbf{x}$
2. An **upper bound**: the current function value $f(\boldsymbol{\theta}_k)$

NAG chooses the extrapolation point $\mathbf{y}_k$ to maximize the improvement in the lower bound while ensuring the upper bound decreases. The extrapolation coefficient $(t_k - 1)/t_{k+1}$ is chosen to balance these two objectives.

**Intuition:** GD is myopic — it only looks at the current gradient. NAG is farsighted — it uses the extrapolated point to "see ahead" and choose a direction that will be good not just now, but for the next few steps too.

### 7.4 NAG vs. Polyak Momentum

| Aspect | Polyak Momentum | Nesterov (NAG) |
| --- | --- | --- |
| Gradient at | Current position $\boldsymbol{\theta}_t$ | Lookahead position $\mathbf{y}_t$ |
| Convex rate | $O(1/T)$ | $O(1/T^2)$ |
| Strongly convex rate | $O((1-1/\kappa)^T)$ | $O((1-1/\sqrt{\kappa})^T)$ |
| Optimal? | No | Yes (matches lower bound) |
| Implementation | Simpler | Slightly more complex |
| Deep learning use | Standard (via Adam) | Rare |

**Key insight:** NAG can be viewed as Polyak momentum with a "correction" term. The difference is the gradient evaluation point. In the strongly convex setting, NAG with constant $\beta = \frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}$ achieves the optimal rate.

### 7.5 Optimality: Nesterov's Lower Bound Match

Nesterov (1983) proved that for any first-order method (any method that only accesses $f$ through gradient evaluations), there exists a convex, $L$-smooth function such that:

$$f(\boldsymbol{\theta}_T) - f^* \geq \frac{3L \lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert_2^2}{32(T+1)^2}$$

NAG achieves $O(1/T^2)$, matching this lower bound up to a constant factor. No first-order method can be asymptotically faster on the class of convex, $L$-smooth functions.

Similarly, for strongly convex functions, the lower bound is $\Omega((1 - 1/\sqrt{\kappa})^T)$, and NAG achieves this rate. This makes NAG **universally optimal** for first-order convex optimization.

### 7.6 Why NAG Is Less Common in Deep Learning

If NAG is theoretically optimal, why is it not the default optimizer for neural networks?

**Non-convexity.** NAG's guarantees require convexity. In the non-convex setting of deep learning, NAG does not necessarily outperform Polyak momentum. In fact, the lookahead can sometimes evaluate the gradient in regions where the function behaves poorly (large curvature, discontinuities from ReLU).

**Implementation complexity.** NAG requires maintaining the sequence $t_k$ and computing the extrapolation point. While the overhead is small, it is non-zero, and the benefit in the non-convex setting is uncertain.

**Adam dominates.** Adam combines momentum-like first-moment estimation with RMS-like second-moment estimation. In practice, Adam (or AdamW) outperforms both GD with momentum and NAG on most deep learning tasks. The adaptive per-parameter learning rates are more valuable than the acceleration scheme.

**Empirical findings:** Sutskever et al. (2013) showed that NAG can help in some deep learning settings (particularly with SGD on CNNs), but the improvement over Polyak momentum is modest. For transformer training, AdamW is the standard, and NAG is rarely used.

**For AI:** Understanding NAG is still valuable because it establishes the fundamental limits of first-order optimization. Any improvement over GD must either (a) use higher-order information (Newton, BFGS → §03), (b) use stochasticity more effectively (variance reduction → §05), or (c) adapt to local geometry (Adam → §07). NAG shows that within the pure first-order, deterministic, non-adaptive framework, no further improvement is possible.

---

## 8. Line Search Methods

So far we have assumed a fixed step size $\eta$. Line search methods choose $\eta_t$ adaptively at each iteration, balancing the need for sufficient decrease against the cost of extra function evaluations.

### 8.1 Exact Line Search for Quadratics

For the quadratic $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top A \mathbf{x} - \mathbf{b}^\top \mathbf{x}$, the optimal step size along the negative gradient direction can be computed in closed form:

$$\eta^* = \arg\min_{\eta > 0} f(\boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t))$$

Setting the derivative with respect to $\eta$ to zero:

$$\frac{d}{d\eta} f(\boldsymbol{\theta}_t - \eta \mathbf{g}_t) = -\mathbf{g}_t^\top \nabla f(\boldsymbol{\theta}_t - \eta \mathbf{g}_t) = -\mathbf{g}_t^\top (A(\boldsymbol{\theta}_t - \eta \mathbf{g}_t) - \mathbf{b}) = 0$$

Since $\mathbf{g}_t = A\boldsymbol{\theta}_t - \mathbf{b}$:

$$\eta^* = \frac{\mathbf{g}_t^\top \mathbf{g}_t}{\mathbf{g}_t^\top A \mathbf{g}_t} = \frac{\lVert \mathbf{g}_t \rVert_2^2}{\mathbf{g}_t^\top A \mathbf{g}_t}$$

This is the **Cauchy step size**. It requires knowledge of $A$ (the Hessian), which is available for quadratics but not for general functions.

**Properties:** With exact line search on a quadratic, GD converges in at most $n$ steps if the gradients are conjugate. However, vanilla GD with exact line search does not produce conjugate directions — that requires the conjugate gradient method.

### 8.2 Armijo Backtracking

For general functions, exact line search is too expensive. The **Armijo backtracking** rule is a practical alternative that guarantees sufficient decrease with minimal function evaluations.

**Algorithm (Armijo Backtracking):**

1. Choose parameters $\bar{\eta} > 0$ (initial step), $\rho \in (0, 1)$ (reduction factor, typically 0.5), $c \in (0, 1)$ (sufficient decrease constant, typically $10^{-4}$)
2. Set $\eta = \bar{\eta}$
3. While $f(\boldsymbol{\theta}_t - \eta \mathbf{g}_t) > f(\boldsymbol{\theta}_t) - c\eta \lVert \mathbf{g}_t \rVert_2^2$:
   - Set $\eta \leftarrow \rho \eta$
4. Return $\eta$

**The Armijo condition:** $f(\boldsymbol{\theta}_t - \eta \mathbf{g}_t) \leq f(\boldsymbol{\theta}_t) - c\eta \lVert \mathbf{g}_t \rVert_2^2$ requires that the actual decrease is at least a fraction $c$ of the predicted decrease from the linear approximation.

**Theorem (Convergence with Armijo Backtracking).** Let $f$ be $L$-smooth and bounded below. GD with Armijo backtracking (with $c < 1/2$) satisfies:

$$\min_{0 \leq t \leq T-1} \lVert \nabla f(\boldsymbol{\theta}_t) \rVert_2^2 \leq \frac{2(f(\boldsymbol{\theta}_0) - f^*)}{T \cdot \min(\bar{\eta}, 2\rho(1-c)/L)}$$

The algorithm always terminates because for sufficiently small $\eta$, the descent lemma guarantees the Armijo condition is satisfied.

**For AI:** Armijo backtracking is rarely used in deep learning because each function evaluation requires a full forward pass over the mini-batch, which is expensive. However, it is widely used in classical optimization (scipy.optimize, L-BFGS implementations) and is the standard approach when gradient computation is cheap relative to function evaluation.

### 8.3 Wolfe Conditions

The Armijo condition ensures sufficient decrease but does not prevent the step size from being too small. The **Wolfe conditions** add a second requirement:

1. **Sufficient decrease (Armijo):** $f(\boldsymbol{\theta}_t - \eta \mathbf{g}_t) \leq f(\boldsymbol{\theta}_t) - c_1 \eta \lVert \mathbf{g}_t \rVert_2^2$
2. **Curvature condition:** $\nabla f(\boldsymbol{\theta}_t - \eta \mathbf{g}_t)^\top \mathbf{g}_t \leq c_2 \lVert \mathbf{g}_t \rVert_2^2$

where $0 < c_1 < c_2 < 1$. Typical values: $c_1 = 10^{-4}$, $c_2 = 0.9$.

The curvature condition ensures that the slope at the new point is sufficiently reduced — i.e., we have moved far enough along the search direction. Together, the Wolfe conditions guarantee that $\eta$ is not too small (curvature) and not too large (Armijo).

**Strong Wolfe conditions** replace the curvature condition with:

$$|\nabla f(\boldsymbol{\theta}_t - \eta \mathbf{g}_t)^\top \mathbf{g}_t| \leq c_2 \lVert \mathbf{g}_t \rVert_2^2$$

This prevents the step from going too far past the minimum along the search direction.

### 8.4 When to Use Line Search vs. Fixed LR

| Criterion | Fixed LR | Line Search |
| --- | --- | --- |
| Cost per iteration | 1 gradient | 1+ gradient + function evals |
| Guarantees | Requires known $L$ | Automatic step size selection |
| Robustness | Sensitive to $\eta$ choice | Robust to initialization |
| Parallelism | Fully parallelizable | Sequential (backtracking loop) |
| Deep learning use | Standard | Rare |
| Classical optimization | Rare | Standard |

**Rule of thumb:** Use fixed LR when gradient computation dominates the cost (deep learning). Use line search when function/gradient evaluation is cheap and robustness is important (classical optimization, small-scale ML).

### 8.5 Line Search in Second-Order Methods

Line search is essential for Newton's method and quasi-Newton methods (BFGS, L-BFGS). The Newton direction $\mathbf{d}_t = -[\nabla^2 f(\boldsymbol{\theta}_t)]^{-1} \nabla f(\boldsymbol{\theta}_t)$ is not always a descent direction when $f$ is not convex, so a line search ensures global convergence. For the full treatment, see [Second-Order Methods](../03-Second-Order-Methods/notes.md).

---

## 9. Advanced Topics

### 9.1 Continuous-Time Limit: Gradient Flow ODE

As the step size $\eta \to 0$ and the number of steps $T \to \infty$ with $t = T\eta$ fixed, the GD iterates converge to the solution of the **gradient flow** ordinary differential equation:

$$\frac{d\boldsymbol{\theta}(t)}{dt} = -\nabla f(\boldsymbol{\theta}(t)), \quad \boldsymbol{\theta}(0) = \boldsymbol{\theta}_0$$

This ODE describes the continuous trajectory of GD. Analyzing the ODE is often simpler than analyzing the discrete iterates, and many insights transfer.

**Properties of gradient flow:**

- **Energy decay:** $\frac{d}{dt}f(\boldsymbol{\theta}(t)) = -\lVert \nabla f(\boldsymbol{\theta}(t)) \rVert_2^2 \leq 0$
- **Convergence for convex $f$:** $f(\boldsymbol{\theta}(t)) - f^* \leq \frac{\lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert_2^2}{2t}$
- **Convergence for $\mu$-strongly convex $f$:** $\lVert \boldsymbol{\theta}(t) - \boldsymbol{\theta}^* \rVert_2 \leq e^{-\mu t} \lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert_2$

**For AI:** The gradient flow perspective has been influential in understanding the implicit bias of GD in overparameterized neural networks. For linear networks, gradient flow converges to the minimum-norm solution. For deep ReLU networks, gradient flow favors solutions with certain margin properties. The continuous-time analysis is often cleaner than the discrete analysis.

### 9.2 Implicit vs. Explicit Gradient Descent

**Explicit (forward) Euler** is the standard GD update:

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t)$$

**Implicit (backward) Euler** evaluates the gradient at the new point:

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_{t+1})$$

This is an implicit equation for $\boldsymbol{\theta}_{t+1}$ that must be solved (e.g., by Newton's method). The implicit method is **unconditionally stable**: it converges for any $\eta > 0$ on convex functions. However, each step is much more expensive.

**Proximal gradient descent** can be viewed as an implicit-explicit split for composite objectives $f(\boldsymbol{\theta}) = g(\boldsymbol{\theta}) + h(\boldsymbol{\theta})$ where $g$ is smooth and $h$ is non-smooth:

$$\boldsymbol{\theta}_{t+1} = \text{prox}_{\eta h}(\boldsymbol{\theta}_t - \eta \nabla g(\boldsymbol{\theta}_t))$$

where $\text{prox}_{\eta h}(\mathbf{x}) = \arg\min_{\mathbf{y}} \left(h(\mathbf{y}) + \frac{1}{2\eta}\lVert \mathbf{y} - \mathbf{x} \rVert_2^2\right)$.

### 9.3 GD on Riemannian Manifolds

When the parameter space has a manifold structure (e.g., orthogonal matrices, positive definite matrices, the sphere), standard GD in the ambient space does not respect the constraints. **Riemannian gradient descent** operates directly on the manifold:

$$\boldsymbol{\theta}_{t+1} = \text{Retr}_{\boldsymbol{\theta}_t}(-\eta \cdot \text{grad} f(\boldsymbol{\theta}_t))$$

where $\text{grad} f$ is the Riemannian gradient (the projection of the Euclidean gradient onto the tangent space) and $\text{Retr}$ is a retraction map that moves along the manifold.

**For AI:** Riemannian optimization appears in orthogonal RNNs (constraints on weight matrices), covariance estimation (PSD manifold), and word embeddings on the Poincaré ball (hyperbolic geometry). The math is more involved but the core idea — follow the steepest descent direction within the feasible set — is the same.

### 9.4 GD as Implicit Regularizer

In overparameterized settings (more parameters than constraints), GD does not just find *a* solution — it finds a *specific* solution with desirable properties.

**Linear models.** For the underdetermined system $X\boldsymbol{\theta} = \mathbf{y}$ with $n > d$, GD initialized at $\boldsymbol{\theta}_0 = \mathbf{0}$ converges to the **minimum $\ell_2$-norm solution**:

$$\boldsymbol{\theta}^* = X^\dagger \mathbf{y} = \arg\min_{\boldsymbol{\theta}} \lVert \boldsymbol{\theta} \rVert_2 \quad \text{s.t.} \quad X\boldsymbol{\theta} = \mathbf{y}$$

This is equivalent to ridge regression with $\lambda \to 0$. GD implicitly regularizes without any explicit penalty term.

**Deep linear networks.** For deep linear networks $f(\mathbf{x}) = W_L \cdots W_1 \mathbf{x}$ trained on squared loss, GD converges to the solution with minimum **nuclear norm** (sum of singular values), which promotes low-rank solutions (Arora et al., 2019). This is the mathematical foundation for why LoRA works: low-rank updates are the natural bias of gradient-based training.

**For AI:** Implicit regularization explains why overparameterized neural networks generalize despite having enough capacity to memorize the training data. GD's bias toward simple solutions (minimum norm, low rank, large margin) acts as a regularizer that is built into the optimization algorithm itself, not added as a penalty term.

### 9.5 Connection to Neural Tangent Kernel

The **Neural Tangent Kernel (NTK)** describes the evolution of neural network predictions during GD training in the infinite-width limit (Jacot et al., 2018).

$$\Theta(\mathbf{x}, \mathbf{x}') = \langle \nabla_{\boldsymbol{\theta}} f(\mathbf{x}; \boldsymbol{\theta}), \nabla_{\boldsymbol{\theta}} f(\mathbf{x}'; \boldsymbol{\theta}) \rangle$$

In the infinite-width limit, $\Theta$ remains constant during training, and the network function evolves as a kernel ridge regression predictor with kernel $\Theta$. The convergence rate is determined by the smallest eigenvalue of the NTK matrix on the training data.

**For AI:** The NTK theory provides a precise connection between GD training of neural networks and kernel methods. It explains why wide networks train easily (the NTK is well-conditioned) and why narrow networks are harder to optimize (the NTK changes during training). However, the NTK regime does not capture feature learning, which is essential for the success of deep learning in practice.

---

## 10. Applications in Machine Learning

### 10.1 GD for Linear Regression

For linear regression with squared loss $\mathcal{L}(\mathbf{w}) = \frac{1}{2n}\lVert X\mathbf{w} - \mathbf{y} \rVert_2^2$, the gradient is:

$$\nabla \mathcal{L}(\mathbf{w}) = \frac{1}{n}X^\top(X\mathbf{w} - \mathbf{y})$$

The GD update:

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{n}X^\top(X\mathbf{w}_t - \mathbf{y})$$

This is a linear recurrence: $\mathbf{w}_{t+1} = (I - \frac{\eta}{n}X^\top X)\mathbf{w}_t + \frac{\eta}{n}X^\top\mathbf{y}$. The convergence rate is determined by the eigenvalues of $I - \frac{\eta}{n}X^\top X$, which are $1 - \frac{\eta}{n}\lambda_i(X^\top X)$. The condition number $\kappa = \lambda_{\max}/\lambda_{\min}$ of $X^\top X$ controls the speed.

**Connection to normal equations.** The closed-form solution $\mathbf{w}^* = (X^\top X)^{-1}X^\top\mathbf{y}$ requires $O(nd^2 + d^3)$ compute. GD requires $O(nd)$ per iteration and converges in $O(\kappa \log(1/\epsilon))$ iterations. When $d$ is large and $\kappa$ is moderate, GD is faster than the closed form.

### 10.2 GD for Logistic Regression

For logistic regression with cross-entropy loss:

$$\mathcal{L}(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^n \log(1 + e^{-y_i \mathbf{w}^\top \mathbf{x}^{(i)}})$$

The gradient:

$$\nabla \mathcal{L}(\mathbf{w}) = -\frac{1}{n}\sum_{i=1}^n y_i \sigma(-y_i \mathbf{w}^\top \mathbf{x}^{(i)}) \mathbf{x}^{(i)}$$

where $\sigma(z) = 1/(1+e^{-z})$ is the sigmoid function.

The loss is convex but not strongly convex (unless the data is linearly separable and we add regularization). The smoothness constant is $L = \frac{1}{4n}\lambda_{\max}(X^\top X)$, since $\sigma'(z) \leq 1/4$. The convergence rate is $O(1/T)$.

**For AI:** Logistic regression is the simplest neural network (a single linear layer + sigmoid). Understanding GD on logistic regression builds intuition for GD on deeper networks. The key difference is that the loss landscape of logistic regression is convex (one global minimum), while deep networks have non-convex landscapes.

### 10.3 GD Through the Neural Network Training Lens

For a neural network $f_{\boldsymbol{\theta}}(\mathbf{x})$ with loss $\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{n}\sum_{i=1}^n \ell(f_{\boldsymbol{\theta}}(\mathbf{x}^{(i)}), y^{(i)})$, the GD update is:

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla \mathcal{L}(\boldsymbol{\theta}_t)$$

The gradient $\nabla \mathcal{L}(\boldsymbol{\theta}_t)$ is computed by backpropagation, which applies the chain rule through the computational graph. Each layer's gradient depends on the gradients of all subsequent layers.

**Key differences from convex optimization:**

1. **Non-convexity:** The loss has many local minima and saddle points. GD converges to a stationary point, not necessarily the global minimum.

2. **Overparameterization:** Modern networks have more parameters than training examples. The solution set is a high-dimensional manifold, and GD's implicit bias determines which solution is found.

3. **Batch size:** In practice, we use mini-batch gradients (SGD), not full gradients. The noise from mini-batching affects convergence and generalization. See [Stochastic Optimization](../05-Stochastic-Optimization/notes.md).

4. **Adaptive methods:** Adam and its variants dominate in practice. They modify the GD update with per-parameter learning rates and momentum. See [Adaptive Learning Rate](../07-Adaptive-Learning-Rate/notes.md).

### 10.4 Learning Rate Selection in Real LLM Training

The theoretical bound $\eta < 2/L$ provides a starting point, but practical LLM training requires more nuance.

**LR range test (Smith, 2017):** Train with an exponentially increasing learning rate and plot the loss. The loss decreases initially, then plateaus, then increases. The optimal LR is approximately 10× smaller than the LR at which the loss starts increasing.

**Typical values for LLMs:**

- **GPT-3 (175B):** $\eta_{\max} = 6 \times 10^{-5}$ with AdamW
- **LLaMA-2 (70B):** $\eta_{\max} = 1.5 \times 10^{-4}$ with AdamW
- **PaLM (540B):** $\eta_{\max} = 2 \times 10^{-4}$ with AdamW

These values are much smaller than what the theory would suggest for a convex problem, reflecting the ill-conditioning and non-convexity of the loss landscape.

**Warmup:** All modern LLM training uses a warmup period where the LR increases linearly from 0 to $\eta_{\max}$ over the first 1,000-10,000 steps. Warmup stabilizes training by allowing the model to move into a region of smoother loss before applying the full learning rate.

### 10.5 GD, Momentum, and the Path to Adam

The evolution from GD to Adam is a story of addressing GD's limitations one by one:

```text
EVOLUTION FROM GD TO ADAM
════════════════════════════════════════════════════════════════════════

  GD (1847)          θ_{t+1} = θ_t - η·g_t
    │                  Problem: slow on ill-conditioned problems
    ▼
  + Momentum (1964)  v_t = β·v_{t-1} + g_t; θ_{t+1} = θ_t - η·v_t
    │                  Problem: same LR for all parameters
    ▼
  + AdaGrad (2011)   Per-parameter LR: η_i = η/√(Σg²_{i,t})
    │                  Problem: LR monotonically decreases to 0
    ▼
  + RMSProp (2012)   Exponential moving average: η_i = η/√(E[g²_i])
    │                  Problem: no momentum
    ▼
  + Adam (2014)      Momentum + RMSProp + bias correction
    │                  Problem: poor generalization with weight decay
    ▼
  AdamW (2019)       Decoupled weight decay: θ -= η·λ·θ
                       Common baseline in LLM training

════════════════════════════════════════════════════════════════════════
```

Each step in this evolution addresses a specific limitation of the previous method while preserving its strengths. Adam combines the best ideas: momentum for acceleration on ill-conditioned problems, RMSProp for per-parameter adaptation, and bias correction for stable initialization.

**For AI:** Understanding this lineage is essential for choosing the right optimizer. For most deep learning tasks, AdamW is the default. For vision models trained with SGD + momentum, the momentum term is doing the heavy lifting that Adam does automatically. For LLM fine-tuning, AdamW with a small learning rate ($10^{-5}$ to $10^{-4}$) is standard.

---

## 11. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
| --- | --------- | ---------------- | ----- |
| 1 | "A smaller learning rate is always better" | Too small $\eta$ leads to extremely slow convergence. The optimal $\eta$ balances decrease per step with stability. | Use $\eta \approx 1/L$ for convex problems; use LR range tests for neural networks. |
| 2 | "GD converges to the global minimum for any function" | GD only guarantees convergence to stationary points for non-convex functions. These can be local minima or saddle points. | For non-convex problems, use multiple random restarts or stochastic methods to escape poor stationary points. |
| 3 | "Momentum always speeds up training" | Too much momentum ($\beta$ too high) causes oscillations and divergence. Momentum helps on ill-conditioned problems but can hurt on well-conditioned ones. | Start with $\beta = 0.9$; reduce if oscillations are observed. |
| 4 | "The convergence rate $O(1/T)$ means the loss halves every $T$ steps" | $O(1/T)$ means the error is proportional to $1/T$, not that it halves. Reducing error by 10× requires 10× more steps. | Think in terms of the constant: $f(\theta_T) - f^* \leq C/T$. To halve the error, double $T$. |
| 5 | "Nesterov acceleration is always better than momentum" | NAG's guarantees require convexity. In the non-convex setting of deep learning, NAG does not consistently outperform Polyak momentum. | Use NAG for convex problems; use Adam or Polyak momentum for neural networks. |
| 6 | "The condition number $\kappa$ is constant during training" | The Hessian spectrum changes as parameters move. $\kappa$ is typically large early in training and decreases over time. | Monitor the effective condition number during training; adjust LR accordingly. |
| 7 | "GD with exact line search converges in one step for quadratics" | Exact line search gives the optimal step along the gradient direction, but the gradient direction is not generally the direction to the minimum. Conjugate gradient, not GD, converges in $n$ steps. | Use conjugate gradient for quadratics; use GD with fixed LR for general problems. |
| 8 | "If the loss stops decreasing, GD has converged to a minimum" | The loss may plateau due to a saddle point, a very small gradient in a flat region, or the learning rate being too small. | Check the gradient norm, not just the loss. Try increasing the LR or adding noise. |
| 9 | "L-smoothness means the function is well-behaved everywhere" | L-smoothness only bounds the gradient's Lipschitz constant. The function can still have multiple minima, saddle points, and plateaus. | L-smoothness is a local property. Global behavior depends on convexity. |
| 10 | "GD's implicit bias is always beneficial" | GD's bias toward minimum-norm solutions is beneficial for linear models but can lead to overconfident predictions in classification. | Add explicit regularization (weight decay, dropout) when GD's implicit bias is insufficient. |

---

## 12. Exercises

**Exercise 1: GD on a Quadratic (★)**
Implement gradient descent on $f(x) = \frac{1}{2}x^\top A x$ for a 2×2 positive definite matrix $A$. Verify the convergence rate matches the theoretical prediction.

**Exercise 2: Convergence Rate Verification (★)**
For $f(x) = x^2$, run GD with $\eta = 0.1, 0.5, 1.0, 1.5, 2.0$ starting from $x_0 = 10$. Plot the error $x_t^2$ vs. iteration and identify which step sizes converge and which diverge.

**Exercise 3: Condition Number and Convergence (★)**
Construct 2D quadratics with condition numbers $\kappa = 1, 10, 100, 1000$. Run GD with $\eta = 1/L$ and measure the number of iterations to reach $f(x_t) < 10^{-6}$. Verify the $O(\kappa)$ scaling.

**Exercise 4: Momentum Analysis (★★)**
For $f(x) = \frac{1}{2}x^\top A x$ with $\kappa = 100$, compare vanilla GD, Polyak momentum with $\beta = 0.9$, and optimal $\beta$. Plot the convergence curves and compute the empirical rate.

**Exercise 5: Nesterov vs. Polyak Momentum (★★)**
Implement both NAG and Polyak momentum on a convex, non-quadratic function (e.g., logistic regression loss). Compare convergence rates and discuss when each method is preferable.

**Exercise 6: Armijo Backtracking Implementation (★★)**
Implement GD with Armijo backtracking on $f(x) = x^4 + x^2$. Compare the number of function evaluations and iterations to fixed-step GD. Analyze the trade-off.

**Exercise 7: GD Implicit Bias in Overparameterized Regression (★★★)**
Generate an underdetermined linear system ($n < d$) and solve it with GD starting from $\mathbf{0}$. Compare the solution to the minimum-norm solution from the pseudoinverse. Verify they match.

**Exercise 8: Learning Rate Selection for Neural Network Training (★★★)**
Train a small neural network on a synthetic dataset with different learning rates. Identify the maximal stable LR using the LR range test. Compare training dynamics with and without warmup.

---

## 13. Why This Matters for AI (2026 Perspective)

| Concept | AI Impact |
| --------- | ----------- |
| GD update rule | The fundamental operation executed $10^7$+ times during LLM pretraining |
| Convergence rate $O(1/T)$ | Determines how many training steps are needed to reach target loss |
| Condition number $\kappa$ | Explains why feature scaling, normalization, and good initialization matter |
| Momentum ($\beta = 0.9$) | Standard component of Adam; accelerates training on ill-conditioned loss landscapes |
| Nesterov acceleration | Theoretical optimum for first-order methods; informs optimizer design |
| PL condition | Explains why overparameterized networks train fast despite non-convexity |
| Implicit regularization | GD's bias toward simple solutions contributes to generalization without explicit penalties |
| Learning rate selection | The most important hyperparameter in LLM training; determines training stability and final quality |
| Edge of stability | Explains why LR can be larger than theory predicts during actual training |
| Gradient flow ODE | Theoretical framework for understanding implicit bias in overparameterized networks |

---

## 14. Conceptual Bridge

### Backward Connections

Gradient descent builds directly on the foundations established in [Convex Optimization](../01-Convex-Optimization/notes.md). The smoothness constant $L$ and strong convexity parameter $\mu$ defined there determine GD's convergence rate. The condition number $\kappa = L/\mu$ is the single most important quantity controlling optimization speed. The descent lemma, proved using $L$-smoothness, is the starting point for every convergence proof in this section.

From [Multivariate Calculus](../../05-Multivariate-Calculus/README.md), we use gradients, Hessians, and Taylor expansions throughout. The gradient $\nabla f$ is the engine of GD; the Hessian $\nabla^2 f$ determines the local geometry that GD must navigate.

### Forward Connections

This section is the foundation for everything that follows in the optimization chapter:

- **Second-Order Methods** (§03) replace the fixed step size $\eta$ with curvature-aware steps using the Hessian. Newton's method can be viewed as GD in a transformed coordinate system.
- **Stochastic Optimization** (§05) introduces noise into the gradient estimate, changing the convergence analysis fundamentally. The $O(1/\sqrt{T})$ rate for non-convex GD becomes $O(1/T^{1/4})$ for SGD without variance reduction.
- **Adaptive Learning Rate** (§07) builds on momentum by adding per-parameter learning rate adaptation. Adam = momentum + RMSProp, and its convergence theory extends the analysis in this section.
- **Optimization Landscape** (§06) explains why GD works on non-convex problems by analyzing the geometry of loss surfaces — saddle points, flat minima, and mode connectivity.
- **Learning Rate Schedules** (§10) extends the constant LR analysis to time-varying schedules (warmup, cosine decay, WSD) used in modern LLM training.

### The Big Picture

```
GRADIENT DESCENT IN THE CURRICULUM
════════════════════════════════════════════════════════════════════════

                    Calculus (Ch4-5)
                    Gradients, Hessians
                    Taylor expansion
                         │
                         ▼
              Convex Optimization (§01)
              L-smoothness, strong convexity
              Condition number κ = L/μ
                         │
                         ▼
              ╔══════════════════════════════╗
              ║     GRADIENT DESCENT         ║ ← YOU ARE HERE
              ║                              ║
              ║  θ_{t+1} = θ_t - η·∇f(θ_t)  ║
              ║                              ║
              ║  Convex:    O(1/T)           ║
              ║  Strong cvx: O((1-1/κ)^T)   ║
              ║  Non-convex: O(1/√T) on ‖∇f‖║
              ║  + Momentum: O(1/√κ) accel. ║
              ║  + Nesterov: O(1/T²) optimal║
              ╚══════════════════════════════╝
                         │
            ┌────────────┼────────────┐
            ▼            ▼            ▼
      Second-Order   Stochastic   Adaptive LR
      (§03 Newton)   (§05 SGD)    (§07 Adam)
            │            │            │
            ▼            ▼            ▼
      Constrained    Landscape    LR Schedules
      (§04 KKT)      (§06)        (§10)
            │            │            │
            └────────────┼────────────┘
                         ▼
              Chapter 9: Information Theory
              Entropy, KL, Cross-Entropy

════════════════════════════════════════════════════════════════════════
```

Gradient descent is the pivot point of the entire optimization chapter. It is the simplest algorithm with a rich convergence theory, and every subsequent method is a modification or extension of it. Understanding GD deeply is the prerequisite for understanding every optimizer used in modern AI.

---

## References

1. Nesterov, Y. (2013). *Introductory Lectures on Convex Optimization*. Springer.
2. Boyd, S. & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.
3. Nocedal, J. & Wright, S. (2006). *Numerical Optimization* (2nd ed.). Springer.
4. Sutskever, I. et al. (2013). "On the importance of initialization and momentum in deep learning." ICML.
5. Kingma, D. & Ba, J. (2015). "Adam: A method for stochastic optimization." ICLR.
6. Loshchilov, I. & Hutter, F. (2019). "Decoupled weight decay regularization." ICLR.
7. Lee, J. et al. (2017). "Gradient descent only converges to minimizers." J. Complexity.
8. Cohen, J. et al. (2021). "Gradient descent on neural networks typically occurs at the edge of stability." ICLR.
9. Jacot, A. et al. (2018). "Neural tangent kernel: Convergence and generalization in neural networks." NeurIPS.
10. Arora, S. et al. (2019). "Implicit regularization in deep matrix factorization." NeurIPS.
11. Polyak, B. (1964). "Some methods of speeding up the convergence of iteration methods." USSR Computational Mathematics and Mathematical Physics.
12. Hochreiter, S. & Schmidhuber, J. (1997). "Flat minima." Neural Computation.
13. Keskar, N. et al. (2017). "On large-batch training for deep learning: Generalization gap and sharp minima." ICLR.
14. Smith, L. (2017). "Cyclical learning rates for training neural networks." WACV.
15. Frankle, J. & Carbin, M. (2019). "The lottery ticket hypothesis: Finding sparse, trainable neural networks." ICLR.

---

## Appendix A: Detailed Proofs and Extensions

### A.1 Full Proof of Descent Lemma via Integral Form

The descent lemma is the cornerstone of all GD convergence analysis. Here is the most general proof using the integral form of the mean value theorem.

**Theorem (Descent Lemma, Full Proof).** Let $f: \mathbb{R}^n \to \mathbb{R}$ be differentiable with $L$-Lipschitz gradient. Then for all $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$:

$$f(\mathbf{y}) \leq f(\mathbf{x}) + \nabla f(\mathbf{x})^\top(\mathbf{y} - \mathbf{x}) + \frac{L}{2}\lVert \mathbf{y} - \mathbf{x} \rVert_2^2$$

**Proof.** Define $\phi(t) = f(\mathbf{x} + t(\mathbf{y} - \mathbf{x}))$ for $t \in [0, 1]$. By the chain rule, $\phi'(t) = \nabla f(\mathbf{x} + t(\mathbf{y} - \mathbf{x}))^\top(\mathbf{y} - \mathbf{x})$. By the fundamental theorem of calculus:

$$f(\mathbf{y}) - f(\mathbf{x}) = \phi(1) - \phi(0) = \int_0^1 \phi'(t) \, dt = \int_0^1 \nabla f(\mathbf{x} + t(\mathbf{y} - \mathbf{x}))^\top(\mathbf{y} - \mathbf{x}) \, dt$$

Add and subtract $\nabla f(\mathbf{x})^\top(\mathbf{y} - \mathbf{x})$ inside the integral:

$$= \nabla f(\mathbf{x})^\top(\mathbf{y} - \mathbf{x}) + \int_0^1 (\nabla f(\mathbf{x} + t(\mathbf{y} - \mathbf{x})) - \nabla f(\mathbf{x}))^\top(\mathbf{y} - \mathbf{x}) \, dt$$

By Cauchy-Schwarz and the $L$-Lipschitz property:

$$\left|(\nabla f(\mathbf{x} + t(\mathbf{y} - \mathbf{x})) - \nabla f(\mathbf{x}))^\top(\mathbf{y} - \mathbf{x})\right| \leq \lVert \nabla f(\mathbf{x} + t(\mathbf{y} - \mathbf{x})) - \nabla f(\mathbf{x}) \rVert_2 \cdot \lVert \mathbf{y} - \mathbf{x} \rVert_2$$

$$\leq Lt \lVert \mathbf{y} - \mathbf{x} \rVert_2^2$$

Therefore:

$$f(\mathbf{y}) - f(\mathbf{x}) \leq \nabla f(\mathbf{x})^\top(\mathbf{y} - \mathbf{x}) + \int_0^1 Lt \lVert \mathbf{y} - \mathbf{x} \rVert_2^2 \, dt = \nabla f(\mathbf{x})^\top(\mathbf{y} - \mathbf{x}) + \frac{L}{2}\lVert \mathbf{y} - \mathbf{x} \rVert_2^2 \quad \blacksquare$$

### A.2 Equivalence of L-Smoothness Characterizations

We prove the equivalence of the four characterizations of $L$-smoothness stated in §2.5.

**Theorem.** For a differentiable convex function $f: \mathbb{R}^n \to \mathbb{R}$, the following are equivalent:

1. $\lVert \nabla f(\mathbf{x}) - \nabla f(\mathbf{y}) \rVert_2 \leq L \lVert \mathbf{x} - \mathbf{y} \rVert_2$
2. $f(\mathbf{y}) \leq f(\mathbf{x}) + \nabla f(\mathbf{x})^\top(\mathbf{y} - \mathbf{x}) + \frac{L}{2}\lVert \mathbf{y} - \mathbf{x} \rVert_2^2$
3. $\langle \nabla f(\mathbf{x}) - \nabla f(\mathbf{y}), \mathbf{x} - \mathbf{y} \rangle \geq \frac{1}{L}\lVert \nabla f(\mathbf{x}) - \nabla f(\mathbf{y}) \rVert_2^2$
4. $\nabla^2 f(\mathbf{x}) \preceq LI$ (if twice differentiable)

**Proof.** $(1) \Rightarrow (2)$: This is the descent lemma proved above.

$(2) \Rightarrow (3)$: Apply (2) twice, swapping $\mathbf{x}$ and $\mathbf{y}$:

$$f(\mathbf{y}) \leq f(\mathbf{x}) + \nabla f(\mathbf{x})^\top(\mathbf{y} - \mathbf{x}) + \frac{L}{2}\lVert \mathbf{y} - \mathbf{x} \rVert_2^2$$
$$f(\mathbf{x}) \leq f(\mathbf{y}) + \nabla f(\mathbf{y})^\top(\mathbf{x} - \mathbf{y}) + \frac{L}{2}\lVert \mathbf{x} - \mathbf{y} \rVert_2^2$$

Adding: $0 \leq (\nabla f(\mathbf{x}) - \nabla f(\mathbf{y}))^\top(\mathbf{y} - \mathbf{x}) + L\lVert \mathbf{x} - \mathbf{y} \rVert_2^2$

So $(\nabla f(\mathbf{x}) - \nabla f(\mathbf{y}))^\top(\mathbf{x} - \mathbf{y}) \geq -L\lVert \mathbf{x} - \mathbf{y} \rVert_2^2$. This gives one direction. For the co-coercivity, define $g(\mathbf{x}) = L\lVert \mathbf{x} \rVert_2^2/2 - f(\mathbf{x})$. Since $\nabla^2 g = LI - \nabla^2 f \succeq 0$, $g$ is convex. By convexity of $g$:

$$(\nabla g(\mathbf{x}) - \nabla g(\mathbf{y}))^\top(\mathbf{x} - \mathbf{y}) \geq 0$$

Substituting $\nabla g = L\mathbf{x} - \nabla f$:

$$(L(\mathbf{x} - \mathbf{y}) - (\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})))^\top(\mathbf{x} - \mathbf{y}) \geq 0$$

$$L\lVert \mathbf{x} - \mathbf{y} \rVert_2^2 \geq (\nabla f(\mathbf{x}) - \nabla f(\mathbf{y}))^\top(\mathbf{x} - \mathbf{y})$$

Combined with the earlier inequality and some algebra, this yields (3).

$(3) \Rightarrow (1)$: By Cauchy-Schwarz: $\lVert \nabla f(\mathbf{x}) - \nabla f(\mathbf{y}) \rVert_2 \cdot \lVert \mathbf{x} - \mathbf{y} \rVert_2 \geq \frac{1}{L}\lVert \nabla f(\mathbf{x}) - \nabla f(\mathbf{y}) \rVert_2^2$, giving (1).

$(1) \Leftrightarrow (4)$: For twice differentiable functions, the Lipschitz constant of $\nabla f$ equals the maximum eigenvalue of $\nabla^2 f$. $\blacksquare$

### A.3 Convergence of GD with Diminishing Step Sizes

When the step size decreases over time, GD can converge to the exact optimum even for non-smooth convex functions.

**Theorem (Diminishing Step Size Convergence).** Let $f$ be convex with bounded subgradients: $\lVert g \rVert_2 \leq G$ for all $g \in \partial f(\boldsymbol{\theta})$. For GD with step sizes $\eta_t$ satisfying $\sum_{t=0}^\infty \eta_t = \infty$ and $\sum_{t=0}^\infty \eta_t^2 < \infty$:

$$\lim_{T \to \infty} f\left(\frac{1}{T}\sum_{t=0}^{T-1} \boldsymbol{\theta}_t\right) = f^*$$

**Proof sketch.** The key inequality is:

$$\lVert \boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^* \rVert_2^2 \leq \lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert_2^2 - 2\eta_t(f(\boldsymbol{\theta}_t) - f^*) + \eta_t^2 G^2$$

Summing and using the conditions on $\eta_t$ gives convergence. The condition $\sum \eta_t = \infty$ ensures the algorithm can reach the optimum, while $\sum \eta_t^2 < \infty$ controls the accumulated noise. $\blacksquare$

**Practical step size schedules:**

| Schedule | Formula | $\sum \eta_t$ | $\sum \eta_t^2$ | Converges? |
| --- | --- | --- | --- | --- |
| $\eta_t = c/t$ | $c/t$ | $\infty$ | $< \infty$ | Yes |
| $\eta_t = c/\sqrt{t}$ | $c/\sqrt{t}$ | $\infty$ | $\infty$ | To neighborhood |
| $\eta_t = c/(t+1)$ | $c/(t+1)$ | $\infty$ | $< \infty$ | Yes |
| $\eta_t = c$ | Constant | $\infty$ | $\infty$ | To neighborhood |

**For AI:** Diminishing step sizes are used in some reinforcement learning algorithms (e.g., Q-learning convergence proofs) and in stochastic approximation. In deep learning, the effective step size decreases during the cosine decay phase of the LR schedule, but not fast enough to satisfy $\sum \eta_t^2 < \infty$ — the schedule is designed to reach a good solution quickly, not to converge asymptotically.

### A.4 GD on Composite Objectives: Proximal Gradient

Many ML problems have objectives of the form $F(\boldsymbol{\theta}) = f(\boldsymbol{\theta}) + g(\boldsymbol{\theta})$ where $f$ is smooth and convex and $g$ is convex but possibly non-smooth (e.g., L1 regularization).

**Proximal gradient descent** (also called forward-backward splitting):

$$\boldsymbol{\theta}_{t+1} = \text{prox}_{\eta g}(\boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t))$$

where the proximal operator is:

$$\text{prox}_{\eta g}(\mathbf{x}) = \arg\min_{\mathbf{y}} \left(g(\mathbf{y}) + \frac{1}{2\eta}\lVert \mathbf{y} - \mathbf{x} \rVert_2^2\right)$$

**Key examples:**

1. **Lasso:** $f(\mathbf{w}) = \frac{1}{2n}\lVert X\mathbf{w} - \mathbf{y} \rVert_2^2$, $g(\mathbf{w}) = \lambda \lVert \mathbf{w} \rVert_1$. The proximal operator is soft thresholding:

$$\text{prox}_{\eta \lambda \lVert \cdot \rVert_1}(x_i) = \text{sign}(x_i) \cdot \max(|x_i| - \eta\lambda, 0)$$

2. **Elastic Net:** $g(\mathbf{w}) = \lambda_1 \lVert \mathbf{w} \rVert_1 + \lambda_2 \lVert \mathbf{w} \rVert_2^2$. The proximal operator is scaled soft thresholding.

3. **Nuclear norm regularization:** $g(A) = \lambda \lVert A \rVert_*$. The proximal operator is singular value thresholding: apply soft thresholding to the singular values.

**Convergence:** Proximal gradient descent converges at the same $O(1/T)$ rate as GD for convex objectives, and $O((1-\mu/L)^T)$ for strongly convex objectives. Nesterov acceleration gives $O(1/T^2)$ (FISTA algorithm).

**For AI:** Proximal methods are used in sparse model training, compressed sensing, and matrix completion. The connection to LoRA is through the nuclear norm: minimizing $\lVert \Delta W \rVert_*$ promotes low-rank updates, which is exactly what LoRA does by parameterizing $\Delta W = BA$ with $r \ll d$.

### A.5 Gradient Descent with Errors

In practice, we never have access to exact gradients. Understanding GD with errors is essential for analyzing SGD and distributed training.

**Theorem (GD with Bounded Errors).** Let $f$ be $\mu$-strongly convex and $L$-smooth. Suppose we have access to approximate gradients $\tilde{\mathbf{g}}_t$ satisfying $\lVert \tilde{\mathbf{g}}_t - \nabla f(\boldsymbol{\theta}_t) \rVert_2 \leq \epsilon$. For GD with $\eta = 1/L$:

$$\lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert_2 \leq \left(1 - \frac{\mu}{L}\right)^{t/2} \lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert_2 + \frac{2\epsilon}{\mu}$$

The iterates converge to a ball of radius $O(\epsilon/\mu)$ around the optimum. The convergence rate is the same as exact GD, but the final accuracy is limited by the gradient error.

**For AI:** In distributed training with $K$ workers, the gradient error from averaging $K$ mini-batch gradients has variance $\sigma^2/K$. The effective $\epsilon$ scales as $\sigma/\sqrt{K}$. This explains why increasing the batch size improves convergence: it reduces the gradient noise, shrinking the convergence ball. However, beyond a certain point, the batch size is limited by the condition number, not the noise — this is the "critical batch size" phenomenon (Smith et al., 2018).

### A.6 Polyak's Heavy Ball: Full Convergence Analysis

The convergence analysis of Polyak momentum is more subtle than the quadratic analysis suggests. Here we present the full result for general convex functions.

**Theorem (Polyak Momentum Convergence).** Let $f$ be convex and $L$-smooth. For Polyak momentum with $\eta \leq 1/L$ and $\beta \in [0, 1)$:

$$\min_{0 \leq t \leq T-1} (f(\boldsymbol{\theta}_t) - f^*) \leq \frac{L \lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert_2^2}{2(1-\beta)T} + \frac{\beta L \lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert_2^2}{(1-\beta)^2 T^2}$$

For $\beta = 0$, this reduces to the standard GD bound. For $\beta > 0$, the first term is larger by $1/(1-\beta)$, but the second term provides acceleration for large $T$.

**Optimal parameter choice:** For a target accuracy $\epsilon$, the optimal $\beta$ balances the two terms. Setting $\beta = 1 - O(1/\sqrt{T})$ gives an $O(1/T^2)$ rate, matching Nesterov's method asymptotically. However, the constant is worse, which is why NAG is preferred for theoretical guarantees.

**For AI:** The momentum parameter in Adam ($\beta_1 = 0.9$) is chosen for practical performance, not theoretical optimality. The effective acceleration from momentum in Adam is less than NAG because the per-parameter learning rates interact with the momentum in complex ways. Understanding the theory helps explain why $\beta_1 = 0.9$ works well: it provides a good balance between acceleration and stability for typical ML condition numbers.

### A.7 The Geometry of Gradient Descent: Natural Gradient Connection

Standard GD uses the Euclidean metric to measure step size: $\lVert \boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}_t \rVert_2 \leq \eta \lVert \nabla f(\boldsymbol{\theta}_t) \rVert_2$. But the Euclidean metric may not be the right geometry for the parameter space.

**Natural gradient descent** (Amari, 1998) uses the Fisher information matrix $F(\boldsymbol{\theta})$ as the metric:

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta F(\boldsymbol{\theta}_t)^{-1} \nabla f(\boldsymbol{\theta}_t)$$

This is GD on the statistical manifold of probability distributions, where distance is measured by KL divergence rather than Euclidean distance. The natural gradient is invariant to reparameterization — a property that standard GD lacks.

**Connection to GD:** When $F(\boldsymbol{\theta}) = I$ (identity), natural GD reduces to standard GD. When $F(\boldsymbol{\theta})$ is diagonal, natural GD is equivalent to Adam without momentum (using the diagonal of the Fisher as the preconditioner).

**For AI:** Natural gradient is computationally expensive ($O(n^2)$ for the Fisher matrix). Structured approximations such as K-FAC (Martens & Grosse, 2015) and Shampoo (Gupta et al., 2018) are active attempts to make curvature-aware updates usable at scale. These are covered in [Second-Order Methods](../03-Second-Order-Methods/notes.md) and [Adaptive Learning Rate](../07-Adaptive-Learning-Rate/notes.md).

---

## Appendix B: Worked Examples and Case Studies

### B.1 Complete GD Analysis: 2D Quadratic

Let us work through a complete convergence analysis of GD on a specific 2D quadratic, computing every quantity explicitly.

**Problem:** Minimize $f(\mathbf{x}) = \frac{1}{2}(x_1^2 + 100 x_2^2)$ starting from $\mathbf{x}_0 = (1, 1)$.

**Step 1: Identify parameters.**
- $A = \text{diag}(1, 100)$, so $\mu = 1$, $L = 100$, $\kappa = 100$
- $\mathbf{x}^* = (0, 0)$, $f^* = 0$
- Optimal step size: $\eta^* = 2/(L + \mu) = 2/101 \approx 0.0198$
- Safe step size: $\eta = 1/L = 0.01$

**Step 2: GD with $\eta = 0.01$ (safe choice).**

The update for each coordinate decouples:

$$x_{1,t+1} = x_{1,t} - 0.01 \cdot x_{1,t} = 0.99 \cdot x_{1,t}$$
$$x_{2,t+1} = x_{2,t} - 0.01 \cdot 100 x_{2,t} = 0 \cdot x_{2,t} = 0$$

So $x_{2,1} = 0$ immediately (the steep direction is optimized in one step), and $x_{1,t} = 0.99^t$. After $t$ steps:

$$f(\mathbf{x}_t) = \frac{1}{2}(0.99^{2t} + 0) = \frac{1}{2} \cdot 0.99^{2t}$$

To reach $f(\mathbf{x}_t) < 10^{-6}$: $0.99^{2t} < 2 \times 10^{-6}$, so $2t \log(0.99) < \log(2 \times 10^{-6})$, giving $t > 687$ iterations.

**Step 3: GD with optimal $\eta = 2/101$.**

$$x_{1,t+1} = x_{1,t} - \frac{2}{101} x_{1,t} = \frac{99}{101} x_{1,t} \approx 0.9802 \cdot x_{1,t}$$
$$x_{2,t+1} = x_{2,t} - \frac{2}{101} \cdot 100 x_{2,t} = -\frac{99}{101} x_{2,t} \approx -0.9802 \cdot x_{2,t}$$

Both coordinates converge at rate $99/101 \approx 0.9802$. After $t$ steps:

$$f(\mathbf{x}_t) = \frac{1}{2}\left(\left(\frac{99}{101}\right)^{2t} + 100\left(\frac{99}{101}\right)^{2t}\right) = \frac{101}{2}\left(\frac{99}{101}\right)^{2t}$$

To reach $f(\mathbf{x}_t) < 10^{-6}$: $t > 347$ iterations. The optimal step size is about 2× faster, as predicted by theory.

**Step 4: GD with momentum $\beta = 0.9$.**

The optimal momentum parameter for $\kappa = 100$ is:

$$\beta^* = \left(\frac{\sqrt{100} - 1}{\sqrt{100} + 1}\right)^2 = \left(\frac{9}{11}\right)^2 \approx 0.669$$

With $\beta = 0.9$ (standard default, higher than optimal), the convergence rate for the worst eigenvalue is approximately:

$$\rho \approx 1 - \frac{2}{\sqrt{\kappa}} = 1 - \frac{2}{10} = 0.8$$

This gives convergence in approximately $\log(10^{-6})/\log(0.8) \approx 62$ iterations — a 5-10× speedup over vanilla GD.

**Key insight:** The steep direction ($x_2$) is optimized quickly, but the shallow direction ($x_1$) dominates the convergence time. This is the essence of the ill-conditioning problem: the optimizer must wait for the slowest direction to converge.

### B.2 GD on Logistic Regression: A Complete Analysis

Consider binary classification with logistic regression loss:

$$\mathcal{L}(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^n \log(1 + e^{-y_i \mathbf{w}^\top \mathbf{x}^{(i)}})$$

**Smoothness constant.** The Hessian of the logistic loss is:

$$\nabla^2 \mathcal{L}(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^n \sigma(y_i \mathbf{w}^\top \mathbf{x}^{(i)})(1 - \sigma(y_i \mathbf{w}^\top \mathbf{x}^{(i)})) \mathbf{x}^{(i)} (\mathbf{x}^{(i)})^\top$$

Since $\sigma(z)(1-\sigma(z)) \leq 1/4$ for all $z$:

$$\nabla^2 \mathcal{L}(\mathbf{w}) \preceq \frac{1}{4n} \sum_{i=1}^n \mathbf{x}^{(i)}(\mathbf{x}^{(i)})^\top = \frac{1}{4n} X^\top X$$

So $L = \frac{1}{4n}\lambda_{\max}(X^\top X) = \frac{1}{4}\lambda_{\max}(\hat{\Sigma})$, where $\hat{\Sigma}$ is the empirical covariance matrix of the features.

**Strong convexity.** The logistic loss is not strongly convex on all of $\mathbb{R}^d$ because $\sigma(z)(1-\sigma(z)) \to 0$ as $|z| \to \infty$. However, if the data is not linearly separable and we restrict to a bounded region $\lVert \mathbf{w} \rVert \leq R$, then:

$$\sigma(y_i \mathbf{w}^\top \mathbf{x}^{(i)})(1 - \sigma(y_i \mathbf{w}^\top \mathbf{x}^{(i)})) \geq \sigma(-R \cdot \max_i \lVert \mathbf{x}^{(i)} \rVert)(1 - \sigma(-R \cdot \max_i \lVert \mathbf{x}^{(i)} \rVert)) =: c_R > 0$$

So $\nabla^2 \mathcal{L}(\mathbf{w}) \succeq \frac{c_R}{4n} X^\top X$, and the strong convexity constant is $\mu = \frac{c_R}{4}\lambda_{\min}(\hat{\Sigma})$.

**Condition number:** $\kappa = \frac{\lambda_{\max}(\hat{\Sigma})}{c_R \cdot \lambda_{\min}(\hat{\Sigma})} = \frac{\kappa(\hat{\Sigma})}{c_R}$.

This reveals two sources of ill-conditioning:
1. **Feature correlation:** $\kappa(\hat{\Sigma})$ is large when features are correlated
2. **Near-separability:** $c_R$ is small when the data is nearly linearly separable (the sigmoid saturates)

**Practical implication:** Standardizing features (making $\hat{\Sigma}$ closer to $I$) reduces $\kappa(\hat{\Sigma})$ and accelerates convergence. This is why feature preprocessing is essential for logistic regression and other linear models.

### B.3 The Edge of Stability in Neural Network Training

Cohen et al. (2021) discovered that during neural network training with GD, the sharpness (largest Hessian eigenvalue) $L_t = \lambda_{\max}(\nabla^2 \mathcal{L}(\boldsymbol{\theta}_t))$ exhibits a remarkable behavior:

1. **Initial phase:** $L_t$ increases rapidly as the model moves away from initialization
2. **Stabilization:** $L_t$ stabilizes near $2/\eta$, the theoretical stability boundary
3. **Oscillation:** $L_t$ oscillates around $2/\eta$ for the remainder of training

This "edge of stability" phenomenon explains why neural networks can be trained with learning rates larger than what the theory predicts for convex functions. The mechanism is:

- When $L_t > 2/\eta$, GD takes a step that increases the loss (progression)
- This progression moves the parameters to a region where the sharpness is lower
- When $L_t < 2/\eta$, GD decreases the loss (descent)
- The descent moves parameters to a region where the sharpness increases
- The system self-stabilizes at $L_t \approx 2/\eta$

**For AI practitioners:** This means the maximal stable learning rate is not a fixed constant but an emergent property of the training dynamics. The LR range test works because it finds the point where the sharpness equals $2/\eta$. Warmup helps because it allows the sharpness to increase gradually rather than jumping past the stability boundary.

### B.4 GD Trajectory Visualization: The Rosenbrock Function

The Rosenbrock function is a classic non-convex test problem:

$$f(x_1, x_2) = (1 - x_1)^2 + 100(x_2 - x_1^2)^2$$

The global minimum is at $(1, 1)$ with $f^* = 0$. The function has a narrow, curved valley that makes it difficult for GD.

**Why GD struggles:** The valley floor follows $x_2 \approx x_1^2$. The gradient points approximately perpendicular to the valley floor, causing GD to zigzag across the valley rather than moving along it toward the minimum.

**GD behavior:**
- With small $\eta$: Very slow progress along the valley floor
- With large $\eta$: Oscillations across the valley walls
- With momentum: The velocity accumulates along the valley direction, reducing zigzagging

This example illustrates why momentum is essential for non-convex optimization: the valley structure is ubiquitous in neural network loss landscapes, and vanilla GD is too slow to navigate it.

### B.5 Comparison Table: All GD Variants

| Variant | Update Rule | Convex Rate | Strongly Convex Rate | Non-Convex Rate | Best For |
| --------- | ------------- | ------------- | --------------------- | ----------------- | ---------- |
| **Vanilla GD** | $\theta - \eta g$ | $O(1/T)$ | $O((1-1/\kappa)^T)$ | $O(1/\sqrt{T})$ on $\|g\|$ | Simple problems, theory |
| **GD + Momentum** | $v = \beta v + g$ | $O(1/T)$ | $O((1-1/\sqrt{\kappa})^T)$ | $O(1/\sqrt{T})$ on $\|g\|$ | Ill-conditioned convex |
| **Nesterov AGD** | Lookahead gradient | $O(1/T^2)$ | $O((1-1/\sqrt{\kappa})^T)$ | $O(1/\sqrt{T})$ on $\|g\|$ | Optimal first-order |
| **GD + Line Search** | Armijo backtracking | $O(1/T)$ | $O((1-1/\kappa)^T)$ | $O(1/\sqrt{T})$ on $\|g\|$ | Unknown $L$ |
| **Proximal GD** | $\text{prox}(\theta - \eta g)$ | $O(1/T)$ | $O((1-1/\kappa)^T)$ | $O(1/\sqrt{T})$ on $\|g\|$ | Non-smooth + smooth |
| **Natural GD** | $\theta - \eta F^{-1} g$ | $O(1/T)$ | $O((1-1/\kappa_F)^T)$ | $O(1/\sqrt{T})$ on $\|g\|$ | Statistical manifolds |

Note that for non-convex functions, all first-order methods have the same worst-case rate $O(1/\sqrt{T})$ on the gradient norm. The differences appear in the constants and in practical performance on specific problem classes.

---

## Appendix C: Numerical Stability and Implementation Details

### C.1 Floating-Point Effects on GD Convergence

In practice, GD is implemented in finite precision arithmetic. The floating-point representation of parameters, gradients, and learning rates introduces rounding errors that affect convergence.

**Machine epsilon.** For FP32 (single precision), $\epsilon_{\text{mach}} \approx 1.2 \times 10^{-7}$. For FP64 (double precision), $\epsilon_{\text{mach}} \approx 2.2 \times 10^{-16}$. The rounding error in each GD step is approximately $\epsilon_{\text{mach}} \cdot \lVert \boldsymbol{\theta}_t \rVert_2$.

**Accumulated error.** After $T$ steps, the accumulated rounding error is approximately $O(\sqrt{T} \cdot \epsilon_{\text{mach}} \cdot \max_t \lVert \boldsymbol{\theta}_t \rVert_2)$ assuming independent errors. For $T = 10^6$ and FP32, this is about $10^{-4} \cdot \max_t \lVert \boldsymbol{\theta}_t \rVert_2$.

**Mixed precision training.** Modern LLM training uses BF16 or FP16 for the forward/backward pass (for speed) but maintains a FP32 "master copy" of the parameters for the update step. This avoids the accumulation of rounding errors in the parameters while benefiting from faster low-precision arithmetic.

**Gradient scaling.** When using FP16, gradients can underflow (become zero) if they are too small. Gradient scaling multiplies the loss by a large factor before the backward pass, then divides the gradients by the same factor. This keeps gradients in the representable range of FP16.

### C.2 Gradient Clipping: Theory and Practice

Gradient clipping is essential for stable training of deep neural networks. There are two variants:

**Clip by value:** $\hat{g}_i = \text{clip}(g_i, -c, c) = \max(-c, \min(c, g_i))$

This clips each component independently. It is simple but does not preserve the direction of the gradient.

**Clip by norm:** $\hat{\mathbf{g}} = \min\left(1, \frac{c}{\lVert \mathbf{g} \rVert_2}\right) \mathbf{g}$

This scales the entire gradient to have norm at most $c$. It preserves the direction and is the standard in LLM training.

**Why clipping works:** When the gradient norm is large, the step $\eta \mathbf{g}$ may violate the smoothness assumption (the step is too large for the local quadratic approximation to be accurate). Clipping ensures that $\lVert \eta \hat{\mathbf{g}} \rVert_2 \leq \eta c$, which keeps the step within the region where the descent lemma applies.

**Choosing the clip threshold:** Common values are $c = 1.0$ for RNNs and $c = 0.1$ to $c = 10.0$ for transformers. The optimal value depends on the model size and learning rate. A rule of thumb: set $c$ so that clipping is triggered about 1-5% of the time.

### C.3 Distributed Gradient Descent

When training on multiple GPUs, the gradient is computed in parallel and averaged:

$$\mathbf{g}_t = \frac{1}{K}\sum_{k=1}^K \mathbf{g}_t^{(k)}$$

where $K$ is the number of workers and $\mathbf{g}_t^{(k)}$ is the mini-batch gradient on worker $k$.

**Communication overhead.** The gradient vector has $n$ elements. Sending it between $K$ workers requires $O(Kn)$ communication per step. For large models ($n = 10^{11}$), this is a significant bottleneck.

**Gradient compression.** To reduce communication, gradients can be compressed:
- **Quantization:** Represent each gradient component with fewer bits (e.g., 8-bit instead of 32-bit)
- **Sparsification:** Only send the top-$k$ largest gradient components
- **Error feedback:** Track the compression error and add it to the next step's gradient

**Convergence with compression:** With unbiased compression operators satisfying $\mathbb{E}[\text{compress}(\mathbf{g})] = \mathbf{g}$ and $\mathbb{E}[\lVert \text{compress}(\mathbf{g}) - \mathbf{g} \rVert_2^2] \leq \omega \lVert \mathbf{g} \rVert_2^2$, the convergence rate is the same as uncompressed GD but with an additional error term proportional to $\omega$.

**For AI:** Distributed training is how all frontier LLMs are trained. GPT-4 and LLaMA-3 use thousands of GPUs with data parallelism, tensor parallelism, and pipeline parallelism. The mathematical theory of GD extends to the distributed setting through the analysis of gradient averaging and communication compression.
