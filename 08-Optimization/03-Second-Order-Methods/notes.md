# Second-Order Methods

[← Back to Optimization](../README.md) | [Next: Constrained Optimization →](../04-Constrained-Optimization/notes.md)

---

> _"If gradient descent is walking downhill with your eyes closed, Newton's method is looking at the entire terrain and jumping straight to the bottom."_ — Anonymous

## Overview

Second-order methods use curvature information — the Hessian matrix of second derivatives — to accelerate convergence beyond what first-order methods can achieve. Where gradient descent asks "which direction goes downhill?", Newton's method asks "what does the terrain look like, and where is the bottom?" This additional information enables dramatically faster convergence: quadratic (doubling correct digits per step) near the optimum, compared to the linear convergence of gradient descent on strongly convex problems.

The mathematical foundation is elegant. Newton's method approximates the objective function locally by a quadratic using the second-order Taylor expansion, then jumps directly to the minimum of that quadratic. When the approximation is accurate (near the optimum of a smooth function), this yields quadratic convergence — the fastest rate achievable by any iterative method using only function values and derivatives.

However, second-order methods face a fundamental computational barrier: the Hessian matrix has $O(n^2)$ entries and inverting it costs $O(n^3)$, where $n$ is the number of parameters. For modern neural networks with $n = 10^9$ to $10^{11}$ parameters, the full Hessian is computationally intractable. This has led to a rich family of approximations: quasi-Newton methods (BFGS, L-BFGS) that build Hessian approximations from gradient differences, Gauss-Newton that exploits the structure of least-squares problems, natural gradient that uses the Fisher information matrix as a curvature proxy, and modern approximations like K-FAC and Shampoo that seek to make curvature information more usable in deep learning.

This section develops the full theory of second-order optimization. We begin with Newton's method and its quadratic convergence (§3), then study quasi-Newton methods with BFGS as the gold standard (§4), the Gauss-Newton method for nonlinear least squares (§5), and natural gradient descent with the Fisher information matrix (§6). Advanced topics include Hessian-free optimization for neural networks, trust region methods, and the connection to adaptive learning rate methods (§7). Every method is connected to its role in modern ML systems.

**Scope note.** This section covers curvature-aware optimization methods that use or approximate second-derivative information. _Line search_ strategies used with Newton's method are covered in [Gradient Descent](../02-Gradient-Descent/notes.md#8-line-search-methods). _Adaptive_ per-parameter methods (Adam, AdaGrad) that use diagonal Hessian approximations belong to [Adaptive Learning Rate](../07-Adaptive-Learning-Rate/notes.md). _Constrained_ optimization with second-order methods (SQP) is covered in [Constrained Optimization](../04-Constrained-Optimization/notes.md). Here we establish the foundational theory of curvature-based optimization.

**For AI practitioners:** By the end of this section, you will understand why Newton's method is impractical for deep learning but why its principles inform modern optimizers, how BFGS and L-BFGS dominate classical ML optimization, why the Fisher information matrix provides a geometry-aware gradient, and how K-FAC- and Shampoo-style approximations try to inject curvature information into large-scale training.

## Prerequisites

- **Gradients and Hessians** — $\nabla f$, $\nabla^2 f$, second-order Taylor expansion — [Chapter 5: Multivariate Calculus](../../05-Multivariate-Calculus/README.md)
- **Gradient descent convergence theory** — linear convergence, condition number — [Chapter 8 §02](../02-Gradient-Descent/notes.md)
- **Eigenvalues and positive definiteness** — $A \succ 0$, spectral theorem, condition number — [Chapter 3 §01](../../03-Advanced-Linear-Algebra/01-Eigenvalues-and-Eigenvectors/notes.md)
- **Matrix norms and decompositions** — Cholesky, SVD — [Chapter 3](../../03-Advanced-Linear-Algebra/README.md)
- **L-smoothness and strong convexity** — from [Convex Optimization](../01-Convex-Optimization/notes.md)

## Companion Notebooks

| Notebook                           | Description                                                                                                 |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| [theory.ipynb](theory.ipynb)       | Interactive demonstrations of Newton's method, BFGS, Gauss-Newton, and natural gradient with visualizations |
| [exercises.ipynb](exercises.ipynb) | 8 graded exercises from Newton convergence proofs to K-FAC approximation analysis                           |

## Learning Objectives

After completing this section, you will:

1. Derive Newton's method from the second-order Taylor expansion and quadratic model
2. Prove local quadratic convergence of Newton's method near a non-degenerate minimum
3. Implement damped Newton's method with backtracking line search for global convergence
4. Derive the BFGS update formula from the secant equation and symmetry conditions
5. Explain why L-BFGS uses $O(mn)$ memory instead of $O(n^2)$ and how the two-loop recursion works
6. Derive the Gauss-Newton approximation $H \approx J^\top J$ for nonlinear least squares
7. Define the Fisher information matrix and derive the natural gradient update
8. Explain why natural gradient is invariant to reparameterization
9. Analyze when Newton's method fails (singular Hessian, saddle points, non-convexity)
10. Connect second-order methods to adaptive learning rate methods (Adam as diagonal approximation)
11. Apply Newton's method to logistic regression and compare with GD
12. Understand K-FAC and Shampoo as structured second-order approximations studied for neural networks

---

## Table of Contents

- [Second-Order Methods](#second-order-methods)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
  - [Companion Notebooks](#companion-notebooks)
  - [Learning Objectives](#learning-objectives)
  - [Table of Contents](#table-of-contents)
  - [1. Intuition](#1-intuition)
    - [1.1 Why First-Order Methods Are Slow](#11-why-first-order-methods-are-slow)
    - [1.2 The Curvature Idea: Using Second Derivatives](#12-the-curvature-idea-using-second-derivatives)
    - [1.3 Newton's Method as Quadratic Modeling](#13-newtons-method-as-quadratic-modeling)
    - [1.4 Historical Timeline: Newton to Modern Approximations](#14-historical-timeline-newton-to-modern-approximations)
    - [1.5 The Cost-Benefit Trade-off of Second-Order Information](#15-the-cost-benefit-trade-off-of-second-order-information)
  - [2. Formal Definitions](#2-formal-definitions)
    - [2.1 The Hessian Matrix and Its Properties](#21-the-hessian-matrix-and-its-properties)
    - [2.2 Second-Order Taylor Expansion](#22-second-order-taylor-expansion)
    - [2.3 Newton's Method: Derivation from Quadratic Model](#23-newtons-method-derivation-from-quadratic-model)
    - [2.4 The Newton Direction and Newton Decrement](#24-the-newton-direction-and-newton-decrement)
    - [2.5 Affine Invariance of Newton's Method](#25-affine-invariance-of-newtons-method)
  - [3. Newton's Method](#3-newtons-method)
    - [3.1 Pure Newton's Method: Algorithm and Properties](#31-pure-newtons-method-algorithm-and-properties)
    - [3.2 Local Quadratic Convergence](#32-local-quadratic-convergence)
    - [3.3 Damped Newton's Method with Line Search](#33-damped-newtons-method-with-line-search)
    - [3.4 Global Convergence via Backtracking](#34-global-convergence-via-backtracking)
    - [3.5 When Newton's Method Fails](#35-when-newtons-method-fails)
  - [4. Quasi-Newton Methods](#4-quasi-newton-methods)
    - [4.1 The Quasi-Newton Idea: Approximating the Hessian](#41-the-quasi-newton-idea-approximating-the-hessian)
    - [4.2 The Secant Equation and Rank-1 Updates](#42-the-secant-equation-and-rank-1-updates)
    - [4.3 DFP Method](#43-dfp-method)
    - [4.4 BFGS: The Gold Standard](#44-bfgs-the-gold-standard)
    - [4.5 BFGS Convergence Properties](#45-bfgs-convergence-properties)
    - [4.6 L-BFGS: Limited-Memory BFGS](#46-l-bfgs-limited-memory-bfgs)
  - [5. Gauss-Newton Method](#5-gauss-newton-method)
    - [5.1 Nonlinear Least Squares Structure](#51-nonlinear-least-squares-structure)
    - [5.2 Gauss-Newton Approximation: H ≈ J^T J](#52-gauss-newton-approximation-h--jt-j)
    - [5.3 Gauss-Newton vs. Newton: When the Approximation Holds](#53-gauss-newton-vs-newton-when-the-approximation-holds)
    - [5.4 Levenberg-Marquardt: Damped Gauss-Newton](#54-levenberg-marquardt-damped-gauss-newton)
    - [5.5 Applications: Curve Fitting, Neural Network Training](#55-applications-curve-fitting-neural-network-training)
  - [6. Natural Gradient and Fisher Information](#6-natural-gradient-and-fisher-information)
    - [6.1 The Statistical Manifold Perspective](#61-the-statistical-manifold-perspective)
    - [6.2 Fisher Information Matrix: Definition and Properties](#62-fisher-information-matrix-definition-and-properties)
    - [6.3 Natural Gradient Descent: Amari's Method](#63-natural-gradient-descent-amaris-method)
    - [6.4 Natural Gradient vs. Preconditioned GD](#64-natural-gradient-vs-preconditioned-gd)
    - [6.5 K-FAC: Kronecker-Factored Approximate Curvature](#65-k-fac-kronecker-factored-approximate-curvature)
  - [7. Advanced Topics](#7-advanced-topics)
    - [7.1 Hessian-Free Optimization for Neural Networks](#71-hessian-free-optimization-for-neural-networks)
    - [7.2 Trust Region Methods](#72-trust-region-methods)
    - [7.3 Conjugate Gradient for Newton Systems](#73-conjugate-gradient-for-newton-systems)
    - [7.4 Sketching and Randomized Hessian Approximation](#74-sketching-and-randomized-hessian-approximation)
    - [7.5 Connection to Adaptive Methods](#75-connection-to-adaptive-methods)
  - [8. Applications in Machine Learning](#8-applications-in-machine-learning)
    - [8.1 Logistic Regression with Newton's Method](#81-logistic-regression-with-newtons-method)
    - [8.2 BFGS for Small-Scale ML Problems](#82-bfgs-for-small-scale-ml-problems)
    - [8.3 L-BFGS-B in scipy.optimize](#83-l-bfgs-b-in-scipyoptimize)
    - [8.4 Natural Gradient in Reinforcement Learning](#84-natural-gradient-in-reinforcement-learning)
    - [8.5 Second-Order Methods in LLM Training](#85-second-order-methods-in-llm-training)
  - [9. Common Mistakes](#9-common-mistakes)
  - [10. Exercises](#10-exercises)
  - [11. Why This Matters for AI (2026 Perspective)](#11-why-this-matters-for-ai-2026-perspective)
  - [12. Conceptual Bridge](#12-conceptual-bridge)
  - [References](#references)

---

## 1. Intuition

### 1.1 Why First-Order Methods Are Slow

Gradient descent uses only first-derivative information: the gradient tells us which direction goes downhill, but nothing about the shape of the valley. On an ill-conditioned problem — where the contours are elongated ellipses rather than circles — GD zigzags across the valley, making slow progress toward the minimum.

The root cause is that GD treats all directions equally. It takes a step of size $\eta$ in the gradient direction, regardless of whether the function is steep or flat in that direction. On a steep slope, the step is too small (wasting iterations). On a flat slope, the step is too large (causing oscillations). The optimal step size would be different in each direction, scaled by the local curvature.

**The elongated bowl problem revisited.** For $f(\mathbf{x}) = \frac{1}{2}(x_1^2 + \kappa x_2^2)$ with $\kappa = 100$, GD with $\eta = 1/L = 1/100$ eliminates the $x_2$ component in one step (the steep direction) but reduces $x_1$ by only 1% per step (the flat direction). It takes $O(\kappa) = O(100)$ iterations to converge. Newton's method, by contrast, knows the curvature in each direction and takes the optimal step in both simultaneously — converging in exactly one step for this quadratic.

### 1.2 The Curvature Idea: Using Second Derivatives

The second derivative (Hessian) tells us how the gradient changes as we move. If the gradient is changing rapidly (large second derivative), the function is curving sharply and we should take a small step. If the gradient is changing slowly (small second derivative), the function is nearly flat and we can take a larger step.

In one dimension, Newton's method is simply:

$$x_{t+1} = x_t - \frac{f'(x_t)}{f''(x_t)}$$

This is the first-order Taylor approximation of $f'(x)$ set to zero: $f'(x_t) + f''(x_t)(x - x_t) = 0$. Geometrically, we approximate $f$ by a quadratic at $x_t$ and jump to its minimum.

**For AI:** The Hessian of a neural network loss encodes the local geometry of the loss landscape. Its eigenvalues reveal the condition number, its eigenvectors reveal the principal directions of curvature, and its spectrum determines whether the current point is near a minimum, saddle, or maximum.

### 1.3 Newton's Method as Quadratic Modeling

Newton's method can be understood as repeatedly fitting a quadratic model to the function and jumping to its minimum:

$$m_t(\mathbf{d}) = f(\boldsymbol{\theta}_t) + \nabla f(\boldsymbol{\theta}_t)^\top \mathbf{d} + \frac{1}{2}\mathbf{d}^\top \nabla^2 f(\boldsymbol{\theta}_t) \mathbf{d}$$

The Newton step $\mathbf{d}_t = \arg\min_{\mathbf{d}} m_t(\mathbf{d})$ is found by setting $\nabla m_t(\mathbf{d}) = \mathbf{0}$:

$$\nabla f(\boldsymbol{\theta}_t) + \nabla^2 f(\boldsymbol{\theta}_t) \mathbf{d}_t = \mathbf{0} \quad \Rightarrow \quad \mathbf{d}_t = -[\nabla^2 f(\boldsymbol{\theta}_t)]^{-1} \nabla f(\boldsymbol{\theta}_t)$$

When $f$ is exactly quadratic, this model is perfect and Newton's method converges in one step. When $f$ is approximately quadratic (near a minimum), the model is accurate and Newton's method converges very fast.

### 1.4 Historical Timeline: Newton to Modern Approximations

```text
SECOND-ORDER METHODS TIMELINE
════════════════════════════════════════════════════════════════════════

  1669  Newton          — Root-finding method (1D)
  1740  Raphson         — Refinement of Newton's method
  1940s Davidon         — Variable metric methods (quasi-Newton)
  1959  Fletcher &      — DFP method (first practical quasi-Newton)
        Powell
  1970  Broyden,        — BFGS method (the gold standard)
        Fletcher,
        Goldfarb,
        Shanno
  1974  Marquardt       — Levenberg-Marquardt (damped Gauss-Newton)
  1989  Nocedal         — L-BFGS (limited-memory BFGS)
  1998  Amari           — Natural gradient descent
  2002  Martens         — Hessian-free optimization for neural nets
  2015  Martens &       — K-FAC (Kronecker-factored approximate curvature)
        Grosse
  2018  Gupta et al.    — Shampoo (matrix preconditioning)
  2020  George et al.   — Scalable K-FAC for large models
  2024  Large-model work — Structured preconditioning remains an active research topic

════════════════════════════════════════════════════════════════════════
```

### 1.5 The Cost-Benefit Trade-off of Second-Order Information

```text
SECOND-ORDER METHOD COMPARISON
════════════════════════════════════════════════════════════════════════

  Method        │ Per-iter cost  │ Memory     │ Convergence rate
  ──────────────┼────────────────┼────────────┼─────────────────────
  GD            │ O(n)           │ O(n)       │ O(1/T) or linear
  Newton        │ O(n³)          │ O(n²)      │ Quadratic (local)
  BFGS          │ O(n²)          │ O(n²)      │ Superlinear
  L-BFGS        │ O(mn)          │ O(mn)      │ Superlinear
  Gauss-Newton  │ O(nd²)         │ O(d²)      │ Linear/quadratic
  Natural GD    │ O(n²)          │ O(n²)      │ Geometry-aware, but costly
  K-FAC         │ structured     │ structured  │ Problem-dependent gains
  Shampoo       │ structured     │ structured  │ Problem-dependent gains

  n = parameters, d = output dim, m = L-BFGS history (typically 10-50)

  The key insight: we want Newton's convergence without Newton's cost.
  Quasi-Newton methods, Gauss-Newton, and natural gradient are all
  different approximations that trade accuracy for efficiency.

════════════════════════════════════════════════════════════════════════
```

---

## 2. Formal Definitions

### 2.1 The Hessian Matrix and Its Properties

**Definition (Hessian).** Let $f: \mathbb{R}^n \to \mathbb{R}$ be twice differentiable. The Hessian matrix of $f$ at $\boldsymbol{\theta}$ is the $n \times n$ symmetric matrix:

$$(\nabla^2 f(\boldsymbol{\theta}))_{ij} = \frac{\partial^2 f}{\partial \theta_i \partial \theta_j}(\boldsymbol{\theta})$$

By Clairaut's theorem (equality of mixed partials), $\nabla^2 f(\boldsymbol{\theta})$ is symmetric when the second partial derivatives are continuous.

**Key properties:**

1. **Symmetry:** $\nabla^2 f(\boldsymbol{\theta}) = \nabla^2 f(\boldsymbol{\theta})^\top$
2. **Positive definiteness at minima:** If $\boldsymbol{\theta}^*$ is a local minimum and $\nabla^2 f(\boldsymbol{\theta}^*)$ is non-singular, then $\nabla^2 f(\boldsymbol{\theta}^*) \succ 0$
3. **Eigenvalue interpretation:** The eigenvalues $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_n$ of $\nabla^2 f(\boldsymbol{\theta})$ represent the curvature of $f$ in the principal directions (eigenvectors)
4. **Condition number:** $\kappa(\nabla^2 f(\boldsymbol{\theta})) = \lambda_{\max} / \lambda_{\min}$ measures the local ill-conditioning

**For AI:** The Hessian of a neural network loss is typically indefinite (has both positive and negative eigenvalues) at most points in parameter space, reflecting the non-convex landscape. Near a local minimum, it becomes positive definite. The spectrum of the Hessian — particularly the ratio of largest to smallest eigenvalue — determines the difficulty of optimization.

### 2.2 Second-Order Taylor Expansion

The second-order Taylor expansion of $f$ around $\boldsymbol{\theta}$ is:

$$f(\boldsymbol{\theta} + \mathbf{d}) = f(\boldsymbol{\theta}) + \nabla f(\boldsymbol{\theta})^\top \mathbf{d} + \frac{1}{2}\mathbf{d}^\top \nabla^2 f(\boldsymbol{\theta}) \mathbf{d} + O(\lVert \mathbf{d} \rVert^3)$$

This quadratic approximation is the foundation of all second-order methods. The quality of the approximation depends on:

1. **Distance from expansion point:** The approximation is accurate when $\lVert \mathbf{d} \rVert$ is small
2. **Smoothness of third derivatives:** If the third derivatives are bounded by $M$, the error is at most $\frac{M}{6}\lVert \mathbf{d} \rVert^3$
3. **Local convexity:** The approximation is a convex quadratic when $\nabla^2 f(\boldsymbol{\theta}) \succ 0$

### 2.3 Newton's Method: Derivation from Quadratic Model

Minimizing the quadratic approximation $m(\mathbf{d}) = f(\boldsymbol{\theta}) + \nabla f(\boldsymbol{\theta})^\top \mathbf{d} + \frac{1}{2}\mathbf{d}^\top H \mathbf{d}$ (where $H = \nabla^2 f(\boldsymbol{\theta})$):

$$\nabla m(\mathbf{d}) = \nabla f(\boldsymbol{\theta}) + H\mathbf{d} = \mathbf{0} \quad \Rightarrow \quad \mathbf{d} = -H^{-1} \nabla f(\boldsymbol{\theta})$$

The Newton update is:

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - [\nabla^2 f(\boldsymbol{\theta}_t)]^{-1} \nabla f(\boldsymbol{\theta}_t)$$

**Important:** We do not actually compute $H^{-1}$. Instead, we solve the linear system $H\mathbf{d} = -\nabla f(\boldsymbol{\theta}_t)$ using Cholesky factorization (when $H \succ 0$) or LU decomposition (general case). This costs $O(n^3)$ for the factorization and $O(n^2)$ for the solve.

### 2.4 The Newton Direction and Newton Decrement

**Definition (Newton Direction).** The Newton direction at $\boldsymbol{\theta}$ is:

$$\mathbf{d}_{\text{nt}} = -[\nabla^2 f(\boldsymbol{\theta})]^{-1} \nabla f(\boldsymbol{\theta})$$

This is the direction of steepest descent in the local norm defined by the Hessian: $\lVert \mathbf{d} \rVert_H = \sqrt{\mathbf{d}^\top H \mathbf{d}}$.

**Definition (Newton Decrement).** The Newton decrement at $\boldsymbol{\theta}$ is:

$$\lambda(\boldsymbol{\theta}) = \sqrt{\nabla f(\boldsymbol{\theta})^\top [\nabla^2 f(\boldsymbol{\theta})]^{-1} \nabla f(\boldsymbol{\theta})} = \lVert \mathbf{d}_{\text{nt}} \rVert_H$$

The Newton decrement measures how close we are to the optimum:

- $\lambda(\boldsymbol{\theta}) = 0 \iff \nabla f(\boldsymbol{\theta}) = \mathbf{0}$ (stationary point)
- $\lambda(\boldsymbol{\theta})$ is small $\implies$ we are in the region of quadratic convergence
- $\lambda(\boldsymbol{\theta})$ is large $\implies$ the quadratic approximation is poor

**Key property:** The Newton decrement is affine invariant — it does not change under linear reparameterization of the variables. This is one of the most important properties of Newton's method.

### 2.5 Affine Invariance of Newton's Method

**Theorem (Affine Invariance).** Let $f: \mathbb{R}^n \to \mathbb{R}$ and define $\tilde{f}(\mathbf{y}) = f(A\mathbf{y} + \mathbf{b})$ for non-singular $A \in \mathbb{R}^{n \times n}$ and $\mathbf{b} \in \mathbb{R}^n$. If Newton's method is applied to $f$ starting from $\boldsymbol{\theta}_0$ and to $\tilde{f}$ starting from $\mathbf{y}_0 = A^{-1}(\boldsymbol{\theta}_0 - \mathbf{b})$, then the iterates satisfy $\boldsymbol{\theta}_t = A\mathbf{y}_t + \mathbf{b}$ for all $t$.

**Proof.** The gradient and Hessian of $\tilde{f}$ are:

$$\nabla \tilde{f}(\mathbf{y}) = A^\top \nabla f(A\mathbf{y} + \mathbf{b})$$
$$\nabla^2 \tilde{f}(\mathbf{y}) = A^\top \nabla^2 f(A\mathbf{y} + \mathbf{b}) A$$

The Newton step for $\tilde{f}$ at $\mathbf{y}_t$:

$$\mathbf{d}_{\tilde{f}} = -[\nabla^2 \tilde{f}(\mathbf{y}_t)]^{-1} \nabla \tilde{f}(\mathbf{y}_t) = -(A^\top H A)^{-1} A^\top \mathbf{g} = -A^{-1} H^{-1} \mathbf{g} = A^{-1} \mathbf{d}_f$$

where $H = \nabla^2 f(\boldsymbol{\theta}_t)$ and $\mathbf{g} = \nabla f(\boldsymbol{\theta}_t)$. Therefore:

$$\mathbf{y}_{t+1} = \mathbf{y}_t + \mathbf{d}_{\tilde{f}} = A^{-1}(\boldsymbol{\theta}_t - \mathbf{b}) + A^{-1}\mathbf{d}_f = A^{-1}(\boldsymbol{\theta}_{t+1} - \mathbf{b}) \quad \blacksquare$$

**For AI:** This means Newton's method automatically adapts to the parameterization of the model. If you reparameterize a neural network (e.g., scale the weights of one layer and inverse-scale the next), Newton's method produces equivalent iterates. GD, by contrast, is sensitive to parameterization — which is why normalization layers and careful initialization are critical for GD-based training.

---

## 3. Newton's Method

### 3.1 Pure Newton's Method: Algorithm and Properties

**Algorithm (Pure Newton's Method).**

1. Initialize $\boldsymbol{\theta}_0 \in \mathbb{R}^n$
2. For $t = 0, 1, 2, \ldots$:
   - Compute $\mathbf{g}_t = \nabla f(\boldsymbol{\theta}_t)$ and $H_t = \nabla^2 f(\boldsymbol{\theta}_t)$
   - Solve $H_t \mathbf{d}_t = -\mathbf{g}_t$ for $\mathbf{d}_t$
   - Update $\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \mathbf{d}_t$
3. Stop when $\lVert \mathbf{g}_t \rVert < \epsilon$ or $\lVert \mathbf{d}_t \rVert < \epsilon$

**Key properties of pure Newton's method:**

1. **One-step convergence for quadratics:** If $f(\boldsymbol{\theta}) = \frac{1}{2}\boldsymbol{\theta}^\top A \boldsymbol{\theta} + \mathbf{b}^\top \boldsymbol{\theta} + c$ with $A \succ 0$, then $\boldsymbol{\theta}_1 = \boldsymbol{\theta}^*$ regardless of $\boldsymbol{\theta}_0$.

2. **Quadratic convergence near the optimum:** When started sufficiently close to a non-degenerate minimum, the number of correct digits approximately doubles at each iteration.

3. **Affine invariance:** The iterates are invariant under linear reparameterization (proved in §2.5).

4. **No step size parameter:** Unlike GD, pure Newton's method has no learning rate to tune — the Hessian automatically scales the step.

### 3.2 Local Quadratic Convergence

**Theorem (Quadratic Convergence of Newton's Method).** Let $f: \mathbb{R}^n \to \mathbb{R}$ be three times continuously differentiable with a local minimum at $\boldsymbol{\theta}^*$ where $\nabla^2 f(\boldsymbol{\theta}^*) \succ 0$. If $\boldsymbol{\theta}_0$ is sufficiently close to $\boldsymbol{\theta}^*$, then the Newton iterates satisfy:

$$\lVert \boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^* \rVert_2 \leq C \lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert_2^2$$

for some constant $C > 0$ depending on the third derivatives of $f$ at $\boldsymbol{\theta}^*$.

**Proof.** Let $H(\boldsymbol{\theta}) = \nabla^2 f(\boldsymbol{\theta})$ and $H^* = H(\boldsymbol{\theta}^*)$. By the Newton update:

$$\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^* = \boldsymbol{\theta}_t - \boldsymbol{\theta}^* - H(\boldsymbol{\theta}_t)^{-1} \nabla f(\boldsymbol{\theta}_t)$$

Since $\nabla f(\boldsymbol{\theta}^*) = \mathbf{0}$, we use the integral form of the mean value theorem:

$$\nabla f(\boldsymbol{\theta}_t) = \int_0^1 H(\boldsymbol{\theta}^* + \tau(\boldsymbol{\theta}_t - \boldsymbol{\theta}^*))(\boldsymbol{\theta}_t - \boldsymbol{\theta}^*) \, d\tau$$

Therefore:

$$\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^* = H(\boldsymbol{\theta}_t)^{-1} \int_0^1 [H(\boldsymbol{\theta}_t) - H(\boldsymbol{\theta}^* + \tau(\boldsymbol{\theta}_t - \boldsymbol{\theta}^*))](\boldsymbol{\theta}_t - \boldsymbol{\theta}^*) \, d\tau$$

By the Lipschitz continuity of $H$ (bounded third derivatives):

$$\lVert H(\boldsymbol{\theta}_t) - H(\boldsymbol{\theta}^* + \tau(\boldsymbol{\theta}_t - \boldsymbol{\theta}^*)) \rVert \leq M(1-\tau)\lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert$$

Substituting and using $\lVert H(\boldsymbol{\theta}_t)^{-1} \rVert \leq 2\lVert (H^*)^{-1} \rVert$ for $\boldsymbol{\theta}_t$ close to $\boldsymbol{\theta}^*$:

$$\lVert \boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^* \rVert \leq 2\lVert (H^*)^{-1} \rVert \cdot \frac{M}{2} \lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert^2 = C \lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert^2 \quad \blacksquare$$

**Interpretation.** Quadratic convergence means the error squares at each step: if $\lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert = 10^{-2}$, then $\lVert \boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^* \rVert \approx 10^{-4}$, then $10^{-8}$, then $10^{-16}$ (machine precision). This is dramatically faster than the linear convergence of GD, where the error is multiplied by a constant factor $< 1$ at each step.

**For AI:** In practice, pure Newton's method is rarely used for neural networks because the "sufficiently close" condition is hard to satisfy in high dimensions. However, the quadratic convergence near the optimum explains why second-order methods are used for fine-tuning: once the model is close to a good solution, Newton-type methods can refine it rapidly.

### 3.3 Damped Newton's Method with Line Search

Pure Newton's method can diverge when started far from the optimum. The **damped Newton method** adds a step size $\eta_t \in (0, 1]$:

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \eta_t \mathbf{d}_t$$

where $\mathbf{d}_t = -H_t^{-1} \mathbf{g}_t$ is the Newton direction and $\eta_t$ is chosen by line search.

**Why damping helps:** Far from the optimum, the quadratic approximation is poor and the full Newton step may overshoot. Damping reduces the step size until the actual decrease matches the predicted decrease from the quadratic model.

**Newton decrement as step size criterion:** The Newton decrement $\lambda(\boldsymbol{\theta}_t)$ provides a natural stopping criterion for the backtracking line search:

- If $\lambda(\boldsymbol{\theta}_t) < 0.25$ (approximately), we are in the region of quadratic convergence and can take the full Newton step ($\eta_t = 1$)
- If $\lambda(\boldsymbol{\theta}_t) \geq 0.25$, we need damping

### 3.4 Global Convergence via Backtracking

**Algorithm (Damped Newton with Backtracking).**

1. Choose parameters $\alpha \in (0, 0.5)$, $\beta \in (0, 1)$ (typically $\alpha = 0.01$, $\beta = 0.5$)
2. For $t = 0, 1, 2, \ldots$:
   - Compute $\mathbf{g}_t = \nabla f(\boldsymbol{\theta}_t)$ and $H_t = \nabla^2 f(\boldsymbol{\theta}_t)$
   - Solve $H_t \mathbf{d}_t = -\mathbf{g}_t$
   - If $\lVert \mathbf{g}_t \rVert < \epsilon$, stop
   - Set $\eta = 1$
   - While $f(\boldsymbol{\theta}_t + \eta \mathbf{d}_t) > f(\boldsymbol{\theta}_t) + \alpha \eta \mathbf{g}_t^\top \mathbf{d}_t$:
     - $\eta \leftarrow \beta \eta$
   - $\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \eta \mathbf{d}_t$

**Theorem (Global Convergence).** Let $f$ be strongly convex and $L$-smooth with $M$-Lipschitz Hessian. Damped Newton's method with backtracking converges to the global minimum from any starting point. Moreover, there are two phases:

1. **Damped phase:** While $\lambda(\boldsymbol{\theta}_t) \geq 0.25$, the function value decreases by at least a constant amount per iteration
2. **Quadratic phase:** Once $\lambda(\boldsymbol{\theta}_t) < 0.25$, the full Newton step ($\eta = 1$) is accepted and convergence is quadratic

The total number of iterations is $O(\log \log(1/\epsilon) + f(\boldsymbol{\theta}_0) - f^*)$ — the $\log \log(1/\epsilon)$ term comes from the quadratic phase and is essentially constant for practical purposes.

### 3.5 When Newton's Method Fails

Newton's method is powerful but has several failure modes:

**1. Singular or near-singular Hessian.** When $H_t$ is singular or ill-conditioned, the Newton direction is undefined or numerically unstable. This occurs at saddle points (where $H$ has zero eigenvalues) and flat regions (where $H$ has small eigenvalues).

**Fix:** Add regularization: $\mathbf{d}_t = -(H_t + \mu I)^{-1} \mathbf{g}_t$. This is the Levenberg-Marquardt approach (§5.4).

**2. Indefinite Hessian.** When $H_t$ is not positive definite, the Newton direction may not be a descent direction. This is common in non-convex optimization (deep learning).

**Fix:** Use the modified Cholesky factorization or trust region methods (§7.2) to ensure the search direction is a descent direction.

**3. Saddle points.** At a saddle point, $\nabla f = \mathbf{0}$ but $H$ has negative eigenvalues. Pure Newton's method converges to saddle points (since they satisfy $\nabla f = \mathbf{0}$), which is undesirable.

**Fix:** Add noise to escape saddle points, or use saddle-free Newton (Dauphin et al., 2014) which takes steps in the direction of negative curvature: $\mathbf{d} = -|H|^{-1} \mathbf{g}$ where $|H|$ has the same eigenvectors as $H$ but absolute eigenvalues.

**4. Computational cost.** The $O(n^3)$ cost of solving the Newton system is prohibitive for large $n$.

**Fix:** Use quasi-Newton methods (§4), Gauss-Newton (§5), or iterative solvers (§7.3).

```text
NEWTON'S METHOD: STRENGTHS AND WEAKNESSES
════════════════════════════════════════════════════════════════════════

  Strength:                          Weakness:
  ─────────                          ─────────
  Quadratic convergence              Requires Hessian (O(n²) storage)
  Affine invariant                   Hessian inversion (O(n³) compute)
  No learning rate to tune           Fails at saddle points
  Automatic step size scaling        Indefinite Hessian → not descent
  Optimal for quadratics             Poor far from optimum

  The quasi-Newton family (BFGS, L-BFGS) addresses the computational
  weaknesses while preserving most of the convergence benefits.

════════════════════════════════════════════════════════════════════════
```

---

## 4. Quasi-Newton Methods

### 4.1 The Quasi-Newton Idea: Approximating the Hessian

Quasi-Newton methods avoid the $O(n^3)$ cost of Newton's method by maintaining an approximation $B_t \approx \nabla^2 f(\boldsymbol{\theta}_t)$ (or its inverse $H_t \approx [\nabla^2 f(\boldsymbol{\theta}_t)]^{-1}$) that is updated using only gradient information. The search direction is:

$$\mathbf{d}_t = -B_t^{-1} \nabla f(\boldsymbol{\theta}_t) = -H_t \nabla f(\boldsymbol{\theta}_t)$$

The key insight is that we don't need the exact Hessian — we need a matrix that maps the gradient to a good search direction. This can be achieved by enforcing consistency conditions between successive iterates.

**The secant equation.** From the Taylor expansion of the gradient:

$$\nabla f(\boldsymbol{\theta}_{t+1}) \approx \nabla f(\boldsymbol{\theta}_t) + \nabla^2 f(\boldsymbol{\theta}_t)(\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}_t)$$

Rearranging:

$$\nabla^2 f(\boldsymbol{\theta}_t)(\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}_t) \approx \nabla f(\boldsymbol{\theta}_{t+1}) - \nabla f(\boldsymbol{\theta}_t)$$

Defining $\mathbf{s}_t = \boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}_t$ and $\mathbf{y}_t = \nabla f(\boldsymbol{\theta}_{t+1}) - \nabla f(\boldsymbol{\theta}_t)$, the **secant equation** is:

$$B_{t+1} \mathbf{s}_t = \mathbf{y}_t$$

This is the fundamental constraint that any quasi-Newton update must satisfy. Since $B_{t+1}$ is an $n \times n$ matrix and the secant equation provides only $n$ constraints, there are infinitely many solutions. The different quasi-Newton methods correspond to different choices among these solutions.

### 4.2 The Secant Equation and Rank-1 Updates

The simplest quasi-Newton update is the symmetric rank-1 (SR1) formula:

$$B_{t+1} = B_t + \frac{(\mathbf{y}_t - B_t \mathbf{s}_t)(\mathbf{y}_t - B_t \mathbf{s}_t)^\top}{(\mathbf{y}_t - B_t \mathbf{s}_t)^\top \mathbf{s}_t}$$

This is the unique symmetric rank-1 update satisfying the secant equation. However, SR1 does not guarantee that $B_{t+1}$ is positive definite, even if $B_t$ is. This can lead to search directions that are not descent directions.

The more popular quasi-Newton methods (DFP, BFGS) use rank-2 updates that preserve positive definiteness.

### 4.3 DFP Method

The Davidon-Fletcher-Powell (DFP) update, proposed in 1959, was the first practical quasi-Newton method. It updates the inverse Hessian approximation $H_t \approx B_t^{-1}$:

$$H_{t+1} = \left(I - \frac{\mathbf{s}_t \mathbf{y}_t^\top}{\mathbf{y}_t^\top \mathbf{s}_t}\right) H_t \left(I - \frac{\mathbf{y}_t \mathbf{s}_t^\top}{\mathbf{y}_t^\top \mathbf{s}_t}\right) + \frac{\mathbf{s}_t \mathbf{s}_t^\top}{\mathbf{y}_t^\top \mathbf{s}_t}$$

DFP preserves positive definiteness: if $H_t \succ 0$ and $\mathbf{y}_t^\top \mathbf{s}_t > 0$, then $H_{t+1} \succ 0$. The condition $\mathbf{y}_t^\top \mathbf{s}_t > 0$ is guaranteed when $f$ is strictly convex.

DFP was the standard quasi-Newton method for about a decade but has been largely superseded by BFGS, which has better numerical properties.

### 4.4 BFGS: The Gold Standard

The Broyden-Fletcher-Goldfarb-Shanno (BFGS) update is the most widely used quasi-Newton method. It updates the inverse Hessian approximation:

$$H_{t+1} = \left(I - \rho_t \mathbf{s}_t \mathbf{y}_t^\top\right) H_t \left(I - \rho_t \mathbf{y}_t \mathbf{s}_t^\top\right) + \rho_t \mathbf{s}_t \mathbf{s}_t^\top$$

where $\rho_t = \frac{1}{\mathbf{y}_t^\top \mathbf{s}_t}$.

**Why BFGS is preferred over DFP:**

1. **Better numerical stability:** BFGS is self-correcting — if $H_t$ drifts from the true inverse Hessian, the update tends to correct it
2. **Better performance on non-convex problems:** BFGS maintains positive definiteness more robustly
3. **Better behavior with inexact line search:** BFGS is less sensitive to the accuracy of the line search
4. **Dual relationship:** BFGS applied to $f$ is equivalent to DFP applied to the dual problem, and the dual of a convex problem is often better conditioned

**Algorithm (BFGS).**

1. Initialize $\boldsymbol{\theta}_0$, $H_0 = I$ (or a scaled identity)
2. For $t = 0, 1, 2, \ldots$:
   - Compute $\mathbf{g}_t = \nabla f(\boldsymbol{\theta}_t)$
   - Set $\mathbf{d}_t = -H_t \mathbf{g}_t$
   - Perform line search to find step size $\eta_t$
   - Set $\mathbf{s}_t = \eta_t \mathbf{d}_t$, $\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \mathbf{s}_t$
   - Compute $\mathbf{y}_t = \nabla f(\boldsymbol{\theta}_{t+1}) - \mathbf{g}_t$
   - If $\mathbf{y}_t^\top \mathbf{s}_t > 0$, update $H_{t+1}$ using the BFGS formula
   - Otherwise, skip the update (keep $H_{t+1} = H_t$)

### 4.5 BFGS Convergence Properties

**Theorem (Superlinear Convergence of BFGS).** Let $f$ be twice continuously differentiable and strongly convex. Then BFGS with exact line search or Wolfe line search converges superlinearly:

$$\lim_{t \to \infty} \frac{\lVert \boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^* \rVert}{\lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert} = 0$$

Superlinear convergence is faster than linear (the ratio goes to zero, not just below one) but slower than quadratic (the error doesn't square at each step). In practice, BFGS often exhibits behavior close to quadratic convergence near the optimum.

**Global convergence.** BFGS with Wolfe line search converges to a stationary point from any starting point for general non-convex functions, provided the Hessian is bounded.

### 4.6 L-BFGS: Limited-Memory BFGS

The main limitation of BFGS is the $O(n^2)$ memory requirement for storing $H_t$. For problems with $n > 10^4$, this becomes prohibitive.

**L-BFGS** (Nocedal, 1989) avoids storing $H_t$ explicitly. Instead, it stores the most recent $m$ pairs $\{(\mathbf{s}_i, \mathbf{y}_i)\}_{i=t-m+1}^t$ (typically $m = 10$ to $50$) and computes the matrix-vector product $H_t \mathbf{g}$ implicitly using the **two-loop recursion**:

**Two-loop recursion:**

1. Initialize $\mathbf{q} = \mathbf{g}$
2. For $i = t, t-1, \ldots, t-m+1$:
   - $\alpha_i = \rho_i \mathbf{s}_i^\top \mathbf{q}$
   - $\mathbf{q} = \mathbf{q} - \alpha_i \mathbf{y}_i$
3. $\mathbf{r} = H_0 \mathbf{q}$ (where $H_0$ is a simple preconditioner, often $\frac{\mathbf{s}_{t-1}^\top \mathbf{y}_{t-1}}{\mathbf{y}_{t-1}^\top \mathbf{y}_{t-1}} I$)
4. For $i = t-m+1, \ldots, t$:
   - $\beta = \rho_i \mathbf{y}_i^\top \mathbf{r}$
   - $\mathbf{r} = \mathbf{r} + (\alpha_i - \beta) \mathbf{s}_i$
5. Return $\mathbf{d} = -\mathbf{r}$

This requires only $O(mn)$ memory (storing $2m$ vectors of length $n$) and $O(mn)$ compute per iteration. For $m = 20$ and $n = 10^6$, this is about 160 MB — feasible on a laptop.

**For AI:** L-BFGS is the default optimizer in scipy.optimize.minimize and is widely used for small-to-medium ML problems (logistic regression, SVMs, Gaussian processes). It is not commonly used for deep learning because the non-convex landscape and the need for stochastic gradients make the full-batch L-BFGS assumptions invalid. However, L-BFGS is sometimes used for fine-tuning small models or for hyperparameter optimization.

---

## 5. Gauss-Newton Method

### 5.1 Nonlinear Least Squares Structure

Many ML problems have the form of **nonlinear least squares**:

$$f(\boldsymbol{\theta}) = \frac{1}{2}\sum_{i=1}^m r_i(\boldsymbol{\theta})^2 = \frac{1}{2}\lVert \mathbf{r}(\boldsymbol{\theta}) \rVert_2^2$$

where $\mathbf{r}: \mathbb{R}^n \to \mathbb{R}^m$ is a vector-valued residual function. Examples include:

- **Curve fitting:** $r_i(\boldsymbol{\theta}) = y_i - g(\mathbf{x}_i; \boldsymbol{\theta})$ where $g$ is a parametric model
- **Neural network training with MSE loss:** $r_i(\boldsymbol{\theta}) = y_i - f_{\boldsymbol{\theta}}(\mathbf{x}_i)$
- **Bundle adjustment in computer vision:** residuals are reprojection errors
- **Kalman filtering:** residuals are prediction errors

The gradient and Hessian of $f$ have special structure:

$$\nabla f(\boldsymbol{\theta}) = J(\boldsymbol{\theta})^\top \mathbf{r}(\boldsymbol{\theta})$$

$$\nabla^2 f(\boldsymbol{\theta}) = J(\boldsymbol{\theta})^\top J(\boldsymbol{\theta}) + \sum_{i=1}^m r_i(\boldsymbol{\theta}) \nabla^2 r_i(\boldsymbol{\theta})$$

where $J(\boldsymbol{\theta}) \in \mathbb{R}^{m \times n}$ is the Jacobian of $\mathbf{r}$ with entries $J_{ij} = \partial r_i / \partial \theta_j$.

### 5.2 Gauss-Newton Approximation: H ≈ J^T J

The **Gauss-Newton method** approximates the Hessian by dropping the second-order term:

$$\nabla^2 f(\boldsymbol{\theta}) \approx J(\boldsymbol{\theta})^\top J(\boldsymbol{\theta})$$

The Gauss-Newton step is then:

$$(J^\top J) \mathbf{d} = -J^\top \mathbf{r}$$

This is exactly the normal equations for the linear least squares problem $\min_{\mathbf{d}} \lVert J\mathbf{d} + \mathbf{r} \rVert_2^2$. Geometrically, we linearize the residuals $\mathbf{r}(\boldsymbol{\theta} + \mathbf{d}) \approx \mathbf{r}(\boldsymbol{\theta}) + J(\boldsymbol{\theta})\mathbf{d}$ and minimize the resulting quadratic.

**When is the approximation good?** The Gauss-Newton approximation is accurate when:

1. **Small residuals:** When $|r_i(\boldsymbol{\theta})|$ is small (near the solution), the second-order term $\sum r_i \nabla^2 r_i$ is negligible
2. **Near-linear residuals:** When $\mathbf{r}$ is approximately linear (the model is nearly linear in the parameters), the second derivatives $\nabla^2 r_i$ are small
3. **Large $m$:** When there are many residuals, the $J^\top J$ term dominates the sum

### 5.3 Gauss-Newton vs. Newton: When the Approximation Holds

| Aspect             | Newton                                      | Gauss-Newton                           |
| ------------------ | ------------------------------------------- | -------------------------------------- |
| Hessian            | Full $H = J^\top J + \sum r_i \nabla^2 r_i$ | Approximate $H \approx J^\top J$       |
| Positive definite  | Only near minima                            | Always PSD (if $J$ has full rank)      |
| Cost per iteration | $O(n^3)$ for Hessian + solve                | $O(mn^2 + n^3)$ for $J^\top J$ + solve |
| Convergence        | Quadratic near optimum                      | Linear to superlinear                  |
| Robustness         | Sensitive to initialization                 | More robust (always descent direction) |

**For AI:** The Gauss-Newton approximation is the foundation of many practical second-order methods for neural networks. The Fisher information matrix (used in natural gradient) is closely related to $J^\top J$ for classification problems. K-FAC (§6.5) is essentially a structured Gauss-Newton approximation.

### 5.4 Levenberg-Marquardt: Damped Gauss-Newton

The **Levenberg-Marquardt (LM)** method adds a damping term to the Gauss-Newton system:

$$(J^\top J + \mu I) \mathbf{d} = -J^\top \mathbf{r}$$

When $\mu$ is large, this approaches gradient descent with step size $1/\mu$. When $\mu$ is small, this approaches the Gauss-Newton step. The parameter $\mu$ is adjusted adaptively:

- If the step decreases the objective, reduce $\mu$ (trust the quadratic model more)
- If the step increases the objective, increase $\mu$ (trust the quadratic model less)

**Algorithm (Levenberg-Marquardt).**

1. Initialize $\boldsymbol{\theta}_0$, $\mu > 0$ (e.g., $\mu = 10^{-3}$), $\nu > 1$ (e.g., $\nu = 10$)
2. For $t = 0, 1, 2, \ldots$:
   - Compute $J_t$ and $\mathbf{r}_t$
   - Solve $(J_t^\top J_t + \mu I)\mathbf{d}_t = -J_t^\top \mathbf{r}_t$
   - If $f(\boldsymbol{\theta}_t + \mathbf{d}_t) < f(\boldsymbol{\theta}_t)$:
     - Accept the step: $\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \mathbf{d}_t$
     - Decrease $\mu \leftarrow \mu / \nu$
   - Else:
     - Reject the step
     - Increase $\mu \leftarrow \mu \cdot \nu$

**For AI:** LM is the standard algorithm for nonlinear curve fitting (scipy.optimize.least_squares uses a variant called TRF — trust region reflective). It is not used for large-scale neural network training because of the $O(n^3)$ solve cost, but it is invaluable for small problems like calibrating sensor models or fitting parametric curves to data.

### 5.5 Applications: Curve Fitting, Neural Network Training

**Example: Exponential curve fitting.** Fit $y = a e^{bx} + c$ to data points $(x_i, y_i)$. The residuals are $r_i(a, b, c) = y_i - a e^{bx_i} - c$. The Jacobian is:

$$J = \begin{bmatrix} -e^{bx_1} & -ax_1 e^{bx_1} & -1 \\ \vdots & \vdots & \vdots \\ -e^{bx_m} & -ax_m e^{bx_m} & -1 \end{bmatrix}$$

Gauss-Newton iteratively refines $(a, b, c)$ by solving $(J^\top J)\mathbf{d} = -J^\top \mathbf{r}$.

**Neural network connection.** For a neural network $f_{\boldsymbol{\theta}}(\mathbf{x})$ with MSE loss, the Gauss-Newton matrix $J^\top J$ is the empirical Fisher information matrix. This connects Gauss-Newton to natural gradient (§6).

---

## 6. Natural Gradient and Fisher Information

### 6.1 The Statistical Manifold Perspective

In statistical estimation, we optimize over a family of probability distributions $p(\mathbf{x} \mid \boldsymbol{\theta})$ parameterized by $\boldsymbol{\theta}$. The parameter space has a natural geometric structure: it is a **Riemannian manifold** where the distance between two nearby distributions is measured by the KL divergence, not the Euclidean distance in parameter space.

The KL divergence between $p(\cdot \mid \boldsymbol{\theta})$ and $p(\cdot \mid \boldsymbol{\theta} + \mathbf{d})$ for small $\mathbf{d}$ is:

$$D_{\mathrm{KL}}(p(\cdot \mid \boldsymbol{\theta}) \| p(\cdot \mid \boldsymbol{\theta} + \mathbf{d})) = \frac{1}{2} \mathbf{d}^\top F(\boldsymbol{\theta}) \mathbf{d} + O(\lVert \mathbf{d} \rVert^3)$$

where $F(\boldsymbol{\theta})$ is the **Fisher information matrix**. This means the Fisher matrix defines the local metric tensor of the statistical manifold.

### 6.2 Fisher Information Matrix: Definition and Properties

**Definition (Fisher Information Matrix).** For a parametric family $p(\mathbf{x} \mid \boldsymbol{\theta})$:

$$F(\boldsymbol{\theta}) = \mathbb{E}_{\mathbf{x} \sim p(\cdot \mid \boldsymbol{\theta})}\left[\nabla_{\boldsymbol{\theta}} \log p(\mathbf{x} \mid \boldsymbol{\theta}) \nabla_{\boldsymbol{\theta}} \log p(\mathbf{x} \mid \boldsymbol{\theta})^\top\right]$$

Equivalently (under regularity conditions):

$$F(\boldsymbol{\theta}) = -\mathbb{E}_{\mathbf{x} \sim p(\cdot \mid \boldsymbol{\theta})}\left[\nabla_{\boldsymbol{\theta}}^2 \log p(\mathbf{x} \mid \boldsymbol{\theta})\right]$$

**Key properties:**

1. **Positive semidefinite:** $F(\boldsymbol{\theta}) \succeq 0$ always
2. **Reparameterization equivariance:** Under a change of parameters $\boldsymbol{\phi} = g(\boldsymbol{\theta})$, the Fisher transforms as $F_{\boldsymbol{\phi}} = J_g^{-\top} F_{\boldsymbol{\theta}} J_g^{-1}$ where $J_g$ is the Jacobian of $g$
3. **Connection to Hessian:** For the negative log-likelihood $f(\boldsymbol{\theta}) = -\frac{1}{n}\sum_i \log p(\mathbf{x}^{(i)} \mid \boldsymbol{\theta})$, the expected Hessian is $\mathbb{E}[\nabla^2 f] = F(\boldsymbol{\theta})$
4. **Cramér-Rao bound:** The inverse Fisher matrix lower-bounds the variance of any unbiased estimator: $\operatorname{Var}(\hat{\boldsymbol{\theta}}) \succeq F(\boldsymbol{\theta})^{-1}/n$

**For AI:** In classification with a neural network outputting $p(y \mid \mathbf{x}; \boldsymbol{\theta}) = \operatorname{softmax}(f_{\boldsymbol{\theta}}(\mathbf{x}))$, the Fisher information matrix is closely related to the Gauss-Newton matrix. For a single data point, $F(\boldsymbol{\theta}) \approx J^\top \operatorname{diag}(p) J$ where $J$ is the Jacobian of the network outputs and $\operatorname{diag}(p)$ is the diagonal of the output covariance.

### 6.3 Natural Gradient Descent: Amari's Method

**Definition (Natural Gradient).** The natural gradient of $f$ at $\boldsymbol{\theta}$ is:

$$\tilde{\nabla} f(\boldsymbol{\theta}) = F(\boldsymbol{\theta})^{-1} \nabla f(\boldsymbol{\theta})$$

The natural gradient descent update is:

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta F(\boldsymbol{\theta}_t)^{-1} \nabla f(\boldsymbol{\theta}_t)$$

**Why natural gradient?** The natural gradient is the direction of steepest descent in the space of probability distributions, not in the space of parameters. It accounts for the fact that a unit change in one parameter may have a much larger effect on the distribution than a unit change in another parameter.

**Theorem (Reparameterization Invariance).** Natural gradient descent is invariant to smooth reparameterization of the model. If $\boldsymbol{\phi} = g(\boldsymbol{\theta})$ is a diffeomorphism, then natural gradient descent on $f$ in $\boldsymbol{\theta}$-space produces the same sequence of distributions as natural gradient descent in $\boldsymbol{\phi}$-space.

**Proof sketch.** Under reparameterization, the gradient transforms as $\nabla_{\boldsymbol{\phi}} f = J_g^{-\top} \nabla_{\boldsymbol{\theta}} f$ and the Fisher transforms as $F_{\boldsymbol{\phi}} = J_g^{-\top} F_{\boldsymbol{\theta}} J_g^{-1}$. Therefore:

$$F_{\boldsymbol{\phi}}^{-1} \nabla_{\boldsymbol{\phi}} f = J_g F_{\boldsymbol{\theta}}^{-1} J_g^\top J_g^{-\top} \nabla_{\boldsymbol{\theta}} f = J_g F_{\boldsymbol{\theta}}^{-1} \nabla_{\boldsymbol{\theta}} f$$

The natural gradient in $\boldsymbol{\phi}$-space is the pushforward of the natural gradient in $\boldsymbol{\theta}$-space. $\blacksquare$

**For AI:** This invariance property is crucial for deep learning. Standard GD is sensitive to the parameterization — scaling the weights of one layer and inverse-scaling the next changes the GD trajectory. Natural gradient is invariant to such transformations, making it more robust to poor parameterizations.

### 6.4 Natural Gradient vs. Preconditioned GD

Natural gradient can be viewed as preconditioned gradient descent with the Fisher matrix as the preconditioner:

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta F^{-1} \mathbf{g}$$

This is analogous to Newton's method but with $F$ replacing the Hessian. The key difference:

- **Newton:** Uses the Hessian of the loss function — curvature of the objective
- **Natural gradient:** Uses the Fisher information — curvature of the statistical manifold

When the loss is the negative log-likelihood and the model is well-specified, $F = \mathbb{E}[H]$, so the two are closely related. But for general losses (e.g., MSE, hinge loss), they differ.

**Diagonal approximation.** The full Fisher matrix is $O(n^2)$ and inverting it is $O(n^3)$. A common approximation is to use only the diagonal:

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \cdot \text{diag}(F)^{-1} \nabla f(\boldsymbol{\theta}_t)$$

This resembles **AdaGrad/RMSProp-style diagonal preconditioning** when squared-gradient statistics are used as a proxy for the diagonal Fisher. The connection is conceptually important, but it is an approximation rather than an exact identity.

### 6.5 K-FAC: Kronecker-Factored Approximate Curvature

**K-FAC** (Martens & Grosse, 2015) is a practical approximation of the Fisher/natural gradient for neural networks. The key insight is that for a layer with input activations $\mathbf{a}$ and output pre-activations $\mathbf{z} = W\mathbf{a} + \mathbf{b}$, the Fisher block for $(W, \mathbf{b})$ can be approximated as:

$$F_{(W, \mathbf{b})} \approx \mathbb{E}[\mathbf{a}\mathbf{a}^\top] \otimes \mathbb{E}[\mathbf{g}\mathbf{g}^\top]$$

where $\mathbf{g} = \partial \mathcal{L} / \partial \mathbf{z}$ is the gradient with respect to the pre-activations and $\otimes$ is the Kronecker product.

This factorization reduces the storage from $O((d_{\text{in}} \cdot d_{\text{out}})^2)$ to $O(d_{\text{in}}^2 + d_{\text{out}}^2)$ and the inversion from $O((d_{\text{in}} \cdot d_{\text{out}})^3)$ to $O(d_{\text{in}}^3 + d_{\text{out}}^3)$.

**For AI:** K-FAC has shown strong results in some large-network settings, especially when the model architecture and systems stack make Kronecker-factor updates affordable. It remains more specialized and engineering-heavy than default first-order optimizers.

---

## 7. Advanced Topics

### 7.1 Hessian-Free Optimization for Neural Networks

**Hessian-free (HF) optimization** (Martens, 2010) applies Newton's method to neural networks without explicitly computing or storing the Hessian. The key idea is to solve the Newton system $H\mathbf{d} = -\mathbf{g}$ using **conjugate gradient (CG)**, which only requires Hessian-vector products $H\mathbf{v}$.

**Hessian-vector products via Pearlmutter's trick.** For any vector $\mathbf{v}$:

$$H\mathbf{v} = \nabla^2 f(\boldsymbol{\theta}) \mathbf{v} = \nabla_{\boldsymbol{\theta}} [\nabla f(\boldsymbol{\theta})^\top \mathbf{v}]$$

This can be computed using two backward passes: one to compute $\nabla f(\boldsymbol{\theta})^\top \mathbf{v}$ (a directional derivative) and one to compute its gradient. The cost is comparable to computing the gradient itself — $O(n)$ memory and the same compute as one forward + backward pass.

**Algorithm (Hessian-Free Optimization).**

1. Compute $\mathbf{g} = \nabla f(\boldsymbol{\theta})$
2. Solve $H\mathbf{d} \approx -\mathbf{g}$ using conjugate gradient (with Hessian-vector products)
3. Perform line search along $\mathbf{d}$ to find step size
4. Update $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \eta \mathbf{d}$

**For AI:** HF optimization was one of the first second-order methods successfully applied to deep learning. It was used to train recurrent neural networks for character-level language modeling (Martens & Sutskever, 2011), achieving results that were competitive with or better than first-order methods. Modern variants use damping and trust regions to handle the non-convexity of deep learning losses.

### 7.2 Trust Region Methods

Trust region methods are an alternative to line search for globalizing Newton's method. Instead of choosing a step size along a fixed direction, they choose both the direction and the step size by solving a constrained subproblem:

$$\min_{\mathbf{d}} m_t(\mathbf{d}) = f(\boldsymbol{\theta}_t) + \mathbf{g}_t^\top \mathbf{d} + \frac{1}{2}\mathbf{d}^\top B_t \mathbf{d} \quad \text{s.t.} \quad \lVert \mathbf{d} \rVert_2 \leq \Delta_t$$

where $\Delta_t$ is the **trust region radius**.

**Intuition:** The quadratic model $m_t(\mathbf{d})$ is trusted only within a ball of radius $\Delta_t$ around the current point. Outside this ball, the model may be inaccurate. The radius is adjusted based on the quality of the model:

$$\rho_t = \frac{f(\boldsymbol{\theta}_t) - f(\boldsymbol{\theta}_t + \mathbf{d}_t)}{m_t(\mathbf{0}) - m_t(\mathbf{d}_t)}$$

- If $\rho_t \approx 1$: the model is accurate; increase $\Delta_t$
- If $\rho_t \ll 1$: the model is poor; decrease $\Delta_t$ and reject the step
- If $\rho_t > 0.75$: the model is very accurate; increase $\Delta_t$

**Connection to Levenberg-Marquardt:** The trust region problem with $B_t = J^\top J$ has the solution $\mathbf{d} = -(J^\top J + \mu I)^{-1} J^\top \mathbf{r}$ for some $\mu \geq 0$. This is exactly the LM update, where $\mu$ is the Lagrange multiplier for the trust region constraint.

### 7.3 Conjugate Gradient for Newton Systems

The conjugate gradient (CG) method solves $A\mathbf{x} = \mathbf{b}$ for symmetric positive definite $A$ without explicitly forming $A^{-1}$. For Newton's method, we solve $H\mathbf{d} = -\mathbf{g}$ using CG.

**Algorithm (Conjugate Gradient for $H\mathbf{d} = -\mathbf{g}$).**

1. Initialize $\mathbf{d}_0 = \mathbf{0}$, $\mathbf{r}_0 = -\mathbf{g}$, $\mathbf{p}_0 = \mathbf{r}_0$
2. For $k = 0, 1, 2, \ldots$:
   - $\alpha_k = \frac{\mathbf{r}_k^\top \mathbf{r}_k}{\mathbf{p}_k^\top H \mathbf{p}_k}$
   - $\mathbf{d}_{k+1} = \mathbf{d}_k + \alpha_k \mathbf{p}_k$
   - $\mathbf{r}_{k+1} = \mathbf{r}_k - \alpha_k H \mathbf{p}_k$
   - If $\lVert \mathbf{r}_{k+1} \rVert < \epsilon$, stop
   - $\beta_k = \frac{\mathbf{r}_{k+1}^\top \mathbf{r}_{k+1}}{\mathbf{r}_k^\top \mathbf{r}_k}$
   - $\mathbf{p}_{k+1} = \mathbf{r}_{k+1} + \beta_k \mathbf{p}_k$

**Key properties:**

- CG converges in at most $n$ iterations (exact arithmetic)
- In practice, a good approximate solution is found in $k \ll n$ iterations
- Each iteration requires one Hessian-vector product $H\mathbf{p}_k$
- Total cost: $O(k \cdot \text{cost of } H\mathbf{v})$ where $k$ is the number of CG iterations

**For AI:** CG is the workhorse of Hessian-free optimization and is also used in K-FAC for solving the preconditioned system. The number of CG iterations is typically much smaller than $n$ — often 10-50 iterations are sufficient for a good approximation.

### 7.4 Sketching and Randomized Hessian Approximation

**Randomized numerical linear algebra** provides tools for approximating the Hessian with much less computation than the full $O(n^2)$ storage.

**Sketching.** Given the Jacobian $J \in \mathbb{R}^{m \times n}$, we can approximate $J^\top J$ using a random sketching matrix $S \in \mathbb{R}^{s \times m}$ with $s \ll m$:

$$J^\top J \approx (SJ)^\top (SJ)$$

Common choices for $S$ include:

- **Gaussian sketch:** $S_{ij} \sim \mathcal{N}(0, 1/s)$
- **Subsampled randomized Hadamard transform (SRHT):** $S = P H D$ where $P$ selects rows, $H$ is the Hadamard matrix, $D$ is a random diagonal sign matrix
- **CountSketch:** Sparse random matrix with one non-zero entry per column

**Nyström approximation.** For a PSD matrix $A$, the Nyström approximation uses a random subset of columns:

$$A \approx C W^\dagger C^\top$$

where $C$ consists of $c$ randomly selected columns of $A$ and $W$ is the corresponding $c \times c$ submatrix.

**For AI:** Sketching methods are used in large-scale second-order optimization where the full Hessian is too large to store. They are particularly useful for computing the top eigenvalues of the Hessian (for sharpness analysis) and for preconditioning CG.

### 7.5 Connection to Adaptive Methods

Adaptive learning rate methods can be understood as diagonal approximations to second-order methods:

| Method  | Preconditioner                                      | Approximation of                                    |
| ------- | --------------------------------------------------- | --------------------------------------------------- |
| AdaGrad | $\text{diag}(\sum_{i=1}^t \mathbf{g}_i^2)^{-1/2}$   | Diagonal of cumulative Fisher                       |
| RMSProp | $\text{diag}(E[\mathbf{g}^2])^{-1/2}$               | Diagonal of exponential moving average Fisher       |
| Adam    | $\text{diag}(E[\mathbf{g}^2])^{-1/2}$ with momentum | Diagonal Fisher + momentum                          |
| AdamW   | Adam + decoupled weight decay                       | Diagonal preconditioning + decoupled regularization |
| Shampoo | Full block preconditioners                          | Block-diagonal Fisher                               |
| K-FAC   | Kronecker-factored Fisher                           | Structured full Fisher                              |

**The diagonal approximation trade-off:** Using only the diagonal of the Fisher/Hessian reduces memory from $O(n^2)$ to $O(n)$ and inversion from $O(n^3)$ to $O(n)$. However, it ignores correlations between parameters, which can be significant in neural networks (e.g., between weights in the same layer).

**Shampoo** (Gupta et al., 2018) goes beyond the diagonal by using full matrix preconditioners for each layer. For a weight matrix $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, Shampoo maintains preconditioners $L \in \mathbb{R}^{d_{\text{out}} \times d_{\text{out}}}$ and $R \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$ such that the update is $L^{-1/4} G R^{-1/4}$. This captures correlations within each dimension of the weight matrix while remaining computationally tractable.

**For AI:** Shampoo is a prominent example of structured preconditioning beyond diagonal methods. Related optimizer proposals such as SOAP and Muon show that large-scale optimizer design is still evolving, but the empirical picture is workload-dependent rather than settled once and for all.

---

## 8. Applications in Machine Learning

### 8.1 Logistic Regression with Newton's Method

For logistic regression with negative log-likelihood:

$$f(\mathbf{w}) = -\sum_{i=1}^n \left[y_i \log \sigma(\mathbf{w}^\top \mathbf{x}^{(i)}) + (1 - y_i) \log(1 - \sigma(\mathbf{w}^\top \mathbf{x}^{(i)}))\right]$$

The gradient and Hessian are:

$$\nabla f(\mathbf{w}) = X^\top (\boldsymbol{\sigma} - \mathbf{y})$$
$$\nabla^2 f(\mathbf{w}) = X^\top S X$$

where $\boldsymbol{\sigma}_i = \sigma(\mathbf{w}^\top \mathbf{x}^{(i)})$ and $S = \operatorname{diag}(\sigma_i(1 - \sigma_i))$ is a diagonal matrix with entries in $(0, 1/4]$.

**Key observation:** The Hessian $X^\top S X$ is always positive semidefinite (since $S \succ 0$), so logistic regression is a convex optimization problem. Newton's method is guaranteed to converge from any starting point with appropriate damping.

**Iteratively Reweighted Least Squares (IRLS).** The Newton update for logistic regression can be written as a weighted least squares problem:

$$\mathbf{w}_{t+1} = (X^\top S_t X)^{-1} X^\top S_t \mathbf{z}_t$$

where $\mathbf{z}_t = X\mathbf{w}_t + S_t^{-1}(\mathbf{y} - \boldsymbol{\sigma}_t)$ is the "adjusted response." This is IRLS — each Newton step solves a weighted least squares problem with weights that depend on the current fit.

**Convergence:** Newton's method for logistic regression converges quadratically near the optimum. In practice, 5-10 Newton iterations are sufficient for high-accuracy solutions, compared to hundreds or thousands of GD iterations.

### 8.2 BFGS for Small-Scale ML Problems

BFGS is the default optimizer for many small-scale ML problems:

- **Gaussian process hyperparameter optimization:** Maximizing the marginal likelihood with respect to kernel parameters
- **SVM dual problem:** Solving the QP dual with box constraints (L-BFGS-B variant)
- **Maximum entropy models:** Fitting exponential family distributions
- **Nonlinear regression:** Fitting parametric models to data

For problems with $n < 10^4$ parameters, BFGS typically outperforms GD by a large margin, converging in 10-50 iterations compared to hundreds or thousands for GD.

### 8.3 L-BFGS-B in scipy.optimize

`scipy.optimize.minimize(method='L-BFGS-B')` is the most widely used second-order optimizer in the Python ecosystem. It implements:

- L-BFGS with limited memory ($m = 10$ by default)
- Box constraints (bounds on each parameter)
- Wolfe line search with safeguarded polynomial interpolation

**For AI:** L-BFGS-B is used extensively in:

- Hyperparameter optimization (scikit-optimize, Optuna)
- Gaussian process training (scikit-learn, GPyTorch)
- Scientific computing and inverse problems
- Fine-tuning small neural networks (when the dataset is small enough for full-batch optimization)

### 8.4 Natural Gradient in Reinforcement Learning

Natural gradient has been particularly successful in reinforcement learning:

- **Natural Policy Gradient** (Kakade, 2002): Uses the Fisher information of the policy distribution to precondition the policy gradient. This ensures that policy updates are measured in terms of their effect on the action distribution, not the parameter values.

- **Trust Region Policy Optimization (TRPO)** (Schulman et al., 2015): Uses a KL-divergence constraint (natural gradient geometry) to ensure monotonic policy improvement.

- **Proximal Policy Optimization (PPO)** (Schulman et al., 2017): A practical approximation to TRPO that clips the policy ratio instead of using the exact KL constraint.

**For AI:** Natural gradient methods in RL address a fundamental problem: the same parameter update can have vastly different effects on the policy depending on the current parameterization. By measuring updates in the space of distributions (via the Fisher metric), natural gradient ensures consistent policy improvement.

### 8.5 Second-Order Methods in LLM Training

Curvature-aware ideas continue to influence LLM training, but the frontier is still experimentally mixed:

**K-FAC at scale.** K-FAC and related Fisher-based methods show that structured curvature information can reduce the number of optimization steps in some regimes. The trade-off is heavier implementation complexity, communication, and matrix-update overhead.

**Shampoo and SOAP.** Shampoo (Gupta et al., 2018) uses matrix preconditioning to approximate inverse-root curvature information. SOAP (Vyas et al., 2024) is a more recent preconditioning proposal in the same design space. These methods are best understood as evidence that blockwise preconditioning can matter, not as universally dominant replacements for AdamW.

**Muon.** Muon-style optimizers use orthogonalized momentum updates and have reported promising large-model results. They are important to know because they show how much optimizer design is still moving, even if long-run best practice has not fully converged.

**Hessian analysis.** Even when not used for optimization, the Hessian spectrum is used to analyze training dynamics:

- **Sharpness-aware minimization (SAM)** (Foret et al., 2021): Minimizes both the loss and the sharpness (largest Hessian eigenvalue) to find flat minima
- **WeightWatcher** (Martin & Mahoney, 2021): Uses the spectral properties of weight matrices (related to the Hessian) to predict generalization

**For AI:** The practical lesson for 2026 is not "use second-order methods everywhere." It is: understand curvature, preconditioning, and geometry well enough to recognize when a structured optimizer might beat a diagonal one, while treating AdamW and related first-order baselines as the default starting point.

---

## 9. Common Mistakes

| #   | Mistake                                              | Why It's Wrong                                                                                                                                                                                                  | Fix                                                                                                                                                                           |
| --- | ---------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | "Newton's method always converges faster than GD"    | Newton's method only converges quadratically near the optimum. Far from the optimum, it can diverge or cycle. GD with a good learning rate may be more robust.                                                  | Use damped Newton's method with line search for global convergence.                                                                                                           |
| 2   | "The Hessian is always positive definite"            | The Hessian is positive definite only near a local minimum. At saddle points, it has negative eigenvalues. In general non-convex regions, it can be indefinite.                                                 | Check eigenvalues or use modified Cholesky to ensure positive definiteness.                                                                                                   |
| 3   | "BFGS is always better than L-BFGS"                  | BFGS uses $O(n^2)$ memory while L-BFGS uses $O(mn)$. For $n > 10^4$, BFGS is impractical. L-BFGS with $m = 20$ often performs nearly as well.                                                                   | Use BFGS for $n < 10^4$, L-BFGS for larger problems.                                                                                                                          |
| 4   | "Natural gradient is the same as preconditioned GD"  | Natural gradient uses the Fisher information matrix, which encodes the geometry of the distribution space. General preconditioned GD uses an arbitrary preconditioner that may not have this geometric meaning. | Natural gradient is a specific type of preconditioned GD with $F^{-1}$ as preconditioner.                                                                                     |
| 5   | "Gauss-Newton is always positive definite"           | $J^\top J$ is positive semidefinite, but can be singular if $J$ does not have full column rank. This happens when the model is overparameterized.                                                               | Add regularization: $(J^\top J + \mu I)^{-1}$ (Levenberg-Marquardt).                                                                                                          |
| 6   | "Quasi-Newton methods work well for neural networks" | Quasi-Newton methods assume the objective is approximately quadratic and use full-batch gradients. Neural network losses are highly non-convex and trained with stochastic gradients.                           | Use L-BFGS only for small, well-behaved problems. For neural networks, start with well-tested stochastic optimizers; structured second-order methods are specialized options. |
| 7   | "The Newton direction is always a descent direction" | The Newton direction is a descent direction only when $H \succ 0$. If $H$ is indefinite, the Newton direction may increase the objective.                                                                       | Check that $\mathbf{g}^\top \mathbf{d}_{\text{nt}} < 0$. If not, use a modified Newton direction.                                                                             |
| 8   | "K-FAC is too expensive for large models"            | The naive full-Fisher cost is prohibitive, but K-FAC replaces it with Kronecker-structured factors. Whether that cost is acceptable depends on architecture, implementation quality, and update frequency.      | Evaluate K-FAC only when you can amortize curvature updates and your systems stack supports them well.                                                                        |
| 9   | "The Fisher information equals the Hessian"          | $F(\boldsymbol{\theta}) = \mathbb{E}[H(\boldsymbol{\theta})]$ only for the negative log-likelihood. For other losses (MSE, hinge), they differ.                                                                 | Use the Fisher for natural gradient; use the Hessian for Newton's method. They coincide only for MLE.                                                                         |
| 10  | "Second-order methods are obsolete because of Adam"  | Adam's success does not erase curvature. Structured second-order ideas still matter for analysis, preconditioning, and some large-scale workloads.                                                              | Use AdamW-style methods as the default baseline, then test structured curvature methods when the workload justifies the extra complexity.                                     |

---

## 10. Exercises

**Exercise 1: Newton's Method on a Quadratic (★)**
Implement Newton's method on $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top A \mathbf{x}$ for a 3×3 positive definite matrix $A$. Verify that it converges in exactly one step from any starting point.

**Exercise 2: Quadratic Convergence Verification (★)**
For $f(x) = (x - 3)^4 + (x - 3)^2$, run Newton's method starting from $x_0 = 5$. Track the error $|x_t - x^*|$ at each step and verify that it approximately squares at each iteration (quadratic convergence).

**Exercise 3: BFGS Update (★)**
Given $H_0 = I$, $\mathbf{s} = [1, 2]^\top$, and $\mathbf{y} = [3, 1]^\top$, compute the BFGS update $H_1$ by hand. Verify that $H_1 \mathbf{y} = \mathbf{s}$ (the secant equation in inverse form).

**Exercise 4: Gauss-Newton for Curve Fitting (★★)**
Fit the model $y = a e^{bx}$ to data points using the Gauss-Newton method. Implement the Jacobian computation and verify convergence to the optimal parameters.

**Exercise 5: Natural Gradient for a Gaussian (★★)**
Compute the Fisher information matrix for a univariate Gaussian $\mathcal{N}(\mu, \sigma^2)$ parameterized by $(\mu, \sigma)$. Derive the natural gradient update and compare with the standard gradient.

**Exercise 6: L-BFGS Two-Loop Recursion (★★)**
Implement the L-BFGS two-loop recursion with $m = 3$ history pairs. Verify that it produces the same search direction as full BFGS when $m$ equals the number of iterations.

**Exercise 7: Hessian-Vector Products via Pearlmutter's Trick (★★★)**
Implement Hessian-vector products for a 2-layer neural network using automatic differentiation. Compare with the finite-difference approximation and verify accuracy.

**Exercise 8: K-FAC Approximation Analysis (★★★)**
For a single linear layer $Z = XW$, compute the exact Fisher block and the K-FAC Kronecker approximation. Measure the approximation error and analyze when K-FAC is accurate.

---

## 11. Why This Matters for AI (2026 Perspective)

| Concept                       | AI Impact                                                                                         |
| ----------------------------- | ------------------------------------------------------------------------------------------------- |
| Newton's method               | Foundation for all curvature-based optimization; IRLS for logistic regression                     |
| Quadratic convergence         | Explains why second-order methods are used for fine-tuning and calibration                        |
| BFGS / L-BFGS                 | Default optimizer for small-scale ML; scipy.optimize backbone                                     |
| Gauss-Newton                  | Foundation for nonlinear least squares; connects to neural network training                       |
| Levenberg-Marquardt           | Standard for curve fitting and calibration; robust second-order method                            |
| Fisher information            | Defines the geometry of probability distributions; basis for natural gradient                     |
| Natural gradient              | Reparameterization-invariant optimization; used in RL (TRPO, NPG)                                 |
| K-FAC                         | Structured Fisher approximation for neural networks; useful when curvature updates are affordable |
| Shampoo / SOAP                | Matrix-preconditioning family; relevant to current large-scale optimizer research                 |
| Hessian-vector products       | Enable Hessian-free optimization and sharpness analysis                                           |
| Trust region methods          | Basis for TRPO in RL; robust globalization of Newton's method                                     |
| Diagonal Fisher approximation | Explains why Adam-like methods can be viewed as diagonal preconditioners                          |

---

## 12. Conceptual Bridge

### Backward Connections

Second-order methods build directly on the foundations from [Gradient Descent](../02-Gradient-Descent/notes.md). The Newton direction can be viewed as a preconditioned gradient: $\mathbf{d}_{\text{nt}} = -H^{-1}\nabla f$, compared to GD's $\mathbf{d}_{\text{gd}} = -\nabla f$. The preconditioner $H^{-1}$ automatically scales the gradient by the local curvature, addressing the ill-conditioning that slows down GD.

From [Convex Optimization](../01-Convex-Optimization/notes.md), we use strong convexity (which ensures $H \succ 0$) and smoothness (which bounds the eigenvalues of $H$) throughout the convergence analysis. The condition number $\kappa = \lambda_{\max}(H)/\lambda_{\min}(H)$ determines both GD's convergence rate and the difficulty of solving the Newton system.

From [Multivariate Calculus](../../05-Multivariate-Calculus/README.md), we use the Hessian matrix, second-order Taylor expansion, and the chain rule for computing derivatives of composite functions (essential for neural network Hessians).

### Forward Connections

This section is the foundation for several advanced topics:

- **Constrained Optimization** (§04) uses Newton's method for solving the KKT system in interior-point methods. Sequential quadratic programming (SQP) applies Newton's method to the Lagrangian.
- **Stochastic Optimization** (§05) connects through stochastic quasi-Newton methods (oBFGS, SNS) that approximate the Hessian from noisy gradients.
- **Optimization Landscape** (§06) uses the Hessian spectrum to classify critical points (minima, saddles, maxima) and measure sharpness.
- **Adaptive Learning Rate** (§07) is deeply connected: adaptive methods can be interpreted as diagonal preconditioners, while Shampoo- or K-FAC-style methods retain more curvature structure.
- **Learning Rate Schedules** (§10) can be informed by second-order information: the optimal learning rate is related to the inverse of the largest Hessian eigenvalue.

### The Big Picture

```text
SECOND-ORDER METHODS IN THE CURRICULUM
════════════════════════════════════════════════════════════════════════

              Gradient Descent (§02)
              First-order: d = -∇f
              Rate: O(1/T) or O((1-1/κ)^T)
                       │
                       ▼
              ╔══════════════════════════════════╗
              ║     SECOND-ORDER METHODS         ║ ← YOU ARE HERE
              ║                                  ║
              ║  Newton:    d = -H^{-1}∇f        ║
              ║  BFGS:      H ≈ secant updates   ║
              ║  Gauss-N:   H ≈ J^T J            ║
              ║  Natural:   H ≈ Fisher F         ║
              ║  K-FAC:     F ≈ A ⊗ G            ║
              ║                                  ║
              ║  Rate: Quadratic / Superlinear   ║
              ║  Cost: O(n³) → O(mn) → O(n)      ║
              ╚══════════════════════════════════╝
                       │
            ┌──────────┼──────────┐
            ▼          ▼          ▼
      Constrained  Landscape   Adaptive LR
      (§04 SQP)    (§06 H)     (§07 Adam)
            │          │          │
            ▼          ▼          ▼
      Stochastic   Sharpness   Shampoo/Muon
      (§05)        analysis    (2024-26)

════════════════════════════════════════════════════════════════════════
```

Second-order methods represent the fundamental trade-off in optimization: more information per iteration (curvature) vs. lower cost per iteration (gradient only). The entire family of methods — from exact Newton to diagonal Adam — spans this trade-off spectrum. Understanding where each method sits on this spectrum is essential for choosing the right optimizer for your problem.

---

## References

1. Nocedal, J. & Wright, S. (2006). _Numerical Optimization_ (2nd ed.). Springer.
2. Boyd, S. & Vandenberghe, L. (2004). _Convex Optimization_. Cambridge University Press.
3. Martens, J. (2010). "Deep learning via Hessian-free optimization." ICML.
4. Martens, J. & Grosse, R. (2015). "Optimizing neural networks with Kronecker-factored approximate curvature." ICML.
5. Amari, S. (1998). "Natural gradient works efficiently in learning." Neural Computation.
6. Nocedal, J. (1989). "Updating quasi-Newton matrices with limited storage." Mathematics of Computation.
7. Gupta, V. et al. (2018). "Shampoo: Preconditioned stochastic tensor optimization." ICML.
8. Dauphin, Y. et al. (2014). "Identifying and attacking the saddle point problem in high-dimensional non-convex optimization." NeurIPS.
9. Foret, P. et al. (2021). "Sharpness-aware minimization for efficiently improving generalization." ICLR.
10. Schulman, J. et al. (2015). "Trust region policy optimization." ICML.
11. Kakade, S. (2002). "A natural policy gradient." NeurIPS.
12. Pearlmutter, B. (1994). "Fast exact multiplication by the Hessian." Neural Computation.
13. George, T. et al. (2018). "Fast large-scale optimization by unifying Newton and stochastic gradient methods." arXiv.
14. Vyas, N. et al. (2024). "SOAP: Improving preconditioning for large-scale optimization." arXiv.
15. Levenberg, K. (1944). "A method for the solution of certain non-linear problems in least squares." Quarterly of Applied Mathematics.
16. Marquardt, D. (1963). "An algorithm for least-squares estimation of nonlinear parameters." SIAM Journal.

---

## Appendix A: Detailed Proofs and Extended Theory

### A.1 Full Proof of Quadratic Convergence with Explicit Constants

**Theorem (Quadratic Convergence with Constants).** Let $f: \mathbb{R}^n \to \mathbb{R}$ be three times continuously differentiable with a local minimum at $\boldsymbol{\theta}^*$ where $\nabla^2 f(\boldsymbol{\theta}^*) \succ 0$. Assume the Hessian is $M$-Lipschitz in a neighborhood of $\boldsymbol{\theta}^*$:

$$\lVert \nabla^2 f(\mathbf{x}) - \nabla^2 f(\mathbf{y}) \rVert_2 \leq M \lVert \mathbf{x} - \mathbf{y} \rVert_2$$

Let $\mu = \lambda_{\min}(\nabla^2 f(\boldsymbol{\theta}^*)) > 0$. If $\lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert_2 < \frac{\mu}{2M}$, then Newton's iterates satisfy:

$$\lVert \boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^* \rVert_2 \leq \frac{M}{2\mu} \lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert_2^2$$

**Proof.** Let $H(\boldsymbol{\theta}) = \nabla^2 f(\boldsymbol{\theta})$ and $H^* = H(\boldsymbol{\theta}^*)$. The Newton update gives:

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - H(\boldsymbol{\theta}_t)^{-1} \nabla f(\boldsymbol{\theta}_t)$$

Subtracting $\boldsymbol{\theta}^*$ and using $\nabla f(\boldsymbol{\theta}^*) = \mathbf{0}$:

$$\boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^* = \boldsymbol{\theta}_t - \boldsymbol{\theta}^* - H(\boldsymbol{\theta}_t)^{-1} \nabla f(\boldsymbol{\theta}_t) = H(\boldsymbol{\theta}_t)^{-1}[H(\boldsymbol{\theta}_t)(\boldsymbol{\theta}_t - \boldsymbol{\theta}^*) - \nabla f(\boldsymbol{\theta}_t)]$$

By the fundamental theorem of calculus:

$$\nabla f(\boldsymbol{\theta}_t) = \nabla f(\boldsymbol{\theta}^*) + \int_0^1 H(\boldsymbol{\theta}^* + \tau(\boldsymbol{\theta}_t - \boldsymbol{\theta}^*))(\boldsymbol{\theta}_t - \boldsymbol{\theta}^*) \, d\tau$$

Since $\nabla f(\boldsymbol{\theta}^*) = \mathbf{0}$:

$$H(\boldsymbol{\theta}_t)(\boldsymbol{\theta}_t - \boldsymbol{\theta}^*) - \nabla f(\boldsymbol{\theta}_t) = \int_0^1 [H(\boldsymbol{\theta}_t) - H(\boldsymbol{\theta}^* + \tau(\boldsymbol{\theta}_t - \boldsymbol{\theta}^*))](\boldsymbol{\theta}_t - \boldsymbol{\theta}^*) \, d\tau$$

Taking norms and using the Lipschitz condition:

$$\lVert H(\boldsymbol{\theta}_t)(\boldsymbol{\theta}_t - \boldsymbol{\theta}^*) - \nabla f(\boldsymbol{\theta}_t) \rVert \leq \int_0^1 M(1-\tau)\lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert^2 \, d\tau = \frac{M}{2}\lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert^2$$

Now we need to bound $\lVert H(\boldsymbol{\theta}_t)^{-1} \rVert$. Since $\lVert H(\boldsymbol{\theta}_t) - H^* \rVert \leq M\lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert$ and $\lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert < \frac{\mu}{2M}$:

$$\lVert H(\boldsymbol{\theta}_t) - H^* \rVert < \frac{\mu}{2}$$

By the perturbation bound for matrix inverses:

$$\lVert H(\boldsymbol{\theta}_t)^{-1} \rVert \leq \frac{\lVert (H^*)^{-1} \rVert}{1 - \lVert (H^*)^{-1} \rVert \cdot \lVert H(\boldsymbol{\theta}_t) - H^* \rVert} \leq \frac{1/\mu}{1 - (1/\mu)(\mu/2)} = \frac{2}{\mu}$$

Combining:

$$\lVert \boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^* \rVert \leq \frac{2}{\mu} \cdot \frac{M}{2} \lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert^2 = \frac{M}{\mu} \lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert^2$$

A slightly tighter analysis gives the constant $M/(2\mu)$. $\blacksquare$

**Interpretation of the basin of convergence.** The condition $\lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert < \frac{\mu}{2M}$ defines the region where quadratic convergence is guaranteed. The radius of this region is:

- **Larger** when $\mu$ is large (strong curvature at the minimum — well-conditioned problem)
- **Smaller** when $M$ is large (rapidly changing curvature — highly nonlinear function)

For a quadratic function, $M = 0$ (the Hessian is constant), so the basin of convergence is all of $\mathbb{R}^n$ — Newton's method converges in one step from anywhere.

### A.2 Convergence of Damped Newton's Method

**Theorem (Global Convergence of Damped Newton).** Let $f$ be strongly convex with $\mu I \preceq \nabla^2 f(\boldsymbol{\theta}) \preceq LI$ and $M$-Lipschitz Hessian. Damped Newton's method with backtracking ($\alpha = 0.01$, $\beta = 0.5$) satisfies:

1. **Damped phase:** While $\lambda(\boldsymbol{\theta}_t) \geq 1/4$, each iteration decreases $f$ by at least $\frac{\alpha}{8}$
2. **Quadratic phase:** Once $\lambda(\boldsymbol{\theta}_t) < 1/4$, the full step ($\eta = 1$) is accepted and:

$$\lambda(\boldsymbol{\theta}_{t+1}) \leq 2\lambda(\boldsymbol{\theta}_t)^2$$

The total number of iterations to reach $f(\boldsymbol{\theta}_T) - f^* \leq \epsilon$ is at most:

$$T \leq \frac{f(\boldsymbol{\theta}_0) - f^*}{\alpha/8} + \log_2 \log_2\left(\frac{1/4}{\epsilon \cdot \mu/2}\right)$$

**Proof sketch.** In the damped phase, the backtracking line search guarantees a decrease of at least $\alpha \eta \lambda^2/2$. Since $\lambda \geq 1/4$ and the backtracking finds $\eta \geq \min(1, \beta/(ML))$, the decrease per iteration is bounded below by a constant.

In the quadratic phase, the full Newton step is accepted because the quadratic model is accurate. The Newton decrement satisfies $\lambda_{t+1} \leq 2\lambda_t^2$, which gives double-exponential convergence: $\lambda_T \leq 2^{-(2^T)}$.

The total iterations is the sum of the damped phase (proportional to the initial suboptimality) and the quadratic phase (logarithmic in $1/\epsilon$). $\blacksquare$

**For AI:** This theorem explains the two-phase behavior observed in practice: an initial slow phase where the algorithm finds the right region, followed by a rapid convergence phase. For neural networks, the "damped phase" can be very long because the loss landscape is highly non-quadratic far from the minimum.

### A.3 Derivation of the BFGS Update Formula

The BFGS update is derived by finding the "closest" matrix to $H_t$ that satisfies the secant equation $H_{t+1}\mathbf{y}_t = \mathbf{s}_t$, where "closest" is measured in a weighted Frobenius norm.

**Optimization problem:**

$$\min_{H} \lVert W^{1/2}(H - H_t)W^{1/2} \rVert_F \quad \text{s.t.} \quad H = H^\top, \quad H\mathbf{y}_t = \mathbf{s}_t$$

where $W$ is a positive definite weighting matrix. The solution is unique and has the form:

$$H_{t+1} = H_t + \frac{(\mathbf{s}_t - H_t\mathbf{y}_t)\mathbf{v}^\top + \mathbf{v}(\mathbf{s}_t - H_t\mathbf{y}_t)^\top}{\mathbf{y}_t^\top \mathbf{v}} - \frac{(\mathbf{s}_t - H_t\mathbf{y}_t)^\top \mathbf{v}}{(\mathbf{y}_t^\top \mathbf{v})^2}\mathbf{v}\mathbf{v}^\top$$

for some vector $\mathbf{v}$. The BFGS update corresponds to the choice $\mathbf{v} = H_{t+1}\mathbf{y}_t$, which leads to the symmetric rank-2 formula:

$$H_{t+1} = \left(I - \rho_t \mathbf{s}_t \mathbf{y}_t^\top\right) H_t \left(I - \rho_t \mathbf{y}_t \mathbf{s}_t^\top\right) + \rho_t \mathbf{s}_t \mathbf{s}_t^\top$$

where $\rho_t = 1/(\mathbf{y}_t^\top \mathbf{s}_t)$.

**Why this particular choice?** The BFGS update has several optimality properties:

1. **Minimum change:** It minimizes the change to $H_t$ in the weighted Frobenius norm
2. **Hereditary positive definiteness:** If $H_t \succ 0$ and $\mathbf{y}_t^\top \mathbf{s}_t > 0$, then $H_{t+1} \succ 0$
3. **Self-correcting:** If $H_t$ drifts from the true inverse Hessian, the update tends to correct it
4. **Dual to DFP:** BFGS applied to $f$ is DFP applied to the convex conjugate $f^*$

### A.4 Convergence Analysis of L-BFGS

**Theorem (Superlinear Convergence of L-BFGS).** Let $f$ be twice continuously differentiable and strongly convex. L-BFGS with memory size $m \geq n$ and exact line search converges superlinearly. For $m < n$, the convergence rate depends on $m$:

$$\limsup_{t \to \infty} \frac{\lVert \boldsymbol{\theta}_{t+1} - \boldsymbol{\theta}^* \rVert}{\lVert \boldsymbol{\theta}_t - \boldsymbol{\theta}^* \rVert} \leq C \cdot \left(\frac{\kappa - 1}{\kappa + 1}\right)^m$$

where $\kappa$ is the condition number of $\nabla^2 f(\boldsymbol{\theta}^*)$ and $C$ is a constant.

**Interpretation:** The convergence rate of L-BFGS improves exponentially with the memory size $m$. For well-conditioned problems ($\kappa$ small), even $m = 5$ gives good performance. For ill-conditioned problems ($\kappa$ large), larger $m$ is needed.

**Practical guidance:**

- $m = 5$: Minimum useful memory; suitable for well-conditioned problems
- $m = 10-20$: Default range; good for most problems
- $m = 50$: For ill-conditioned problems; diminishing returns beyond this

### A.5 Fisher Information for Common Distributions

**Bernoulli distribution** $p(x \mid \theta) = \theta^x (1-\theta)^{1-x}$:

$$F(\theta) = \frac{1}{\theta(1-\theta)}$$

**Categorical distribution** $p(x = k \mid \boldsymbol{\theta}) = \theta_k$:

$$F(\boldsymbol{\theta}) = \operatorname{diag}(1/\theta_1, \ldots, 1/\theta_K) + \frac{1}{\theta_K}\mathbf{1}\mathbf{1}^\top$$

On the probability simplex, this is the inverse of the covariance matrix of the multinomial distribution.

**Multivariate Gaussian** $\mathcal{N}(\boldsymbol{\mu}, \Sigma)$ with parameters $(\boldsymbol{\mu}, \Sigma)$:

$$F = \begin{bmatrix} \Sigma^{-1} & 0 \\ 0 & \frac{1}{2}\Sigma^{-1} \otimes \Sigma^{-1} \end{bmatrix}$$

The Fisher is block-diagonal: the mean and covariance parameters are orthogonal in the information geometry sense.

**Exponential family** $p(x \mid \boldsymbol{\theta}) = h(x)\exp(\boldsymbol{\theta}^\top T(x) - A(\boldsymbol{\theta}))$:

$$F(\boldsymbol{\theta}) = \nabla^2 A(\boldsymbol{\theta}) = \operatorname{Cov}_{\boldsymbol{\theta}}[T(X)]$$

For exponential families, the Fisher information is the Hessian of the log-partition function $A(\boldsymbol{\theta})$, which is also the covariance of the sufficient statistics. This is a fundamental result in information geometry.

**For AI:** Neural network outputs with softmax define a categorical distribution over classes. The Fisher information for this distribution is related to the output covariance, which appears in the Gauss-Newton approximation. Understanding the Fisher for common distributions is essential for deriving natural gradient updates.

### A.6 The Geometry of the Statistical Manifold

The set of all probability distributions $\{p(\cdot \mid \boldsymbol{\theta}) : \boldsymbol{\theta} \in \Theta\}$ forms a Riemannian manifold with the Fisher information matrix as the metric tensor. This is the foundation of **information geometry** (Amari & Nagaoka, 2000).

**Key concepts:**

1. **Geodesic distance:** The shortest path between two distributions on the manifold is measured by the Fisher-Rao distance:

$$d(\boldsymbol{\theta}_1, \boldsymbol{\theta}_2) = \inf_{\gamma} \int_0^1 \sqrt{\dot{\gamma}(t)^\top F(\gamma(t)) \dot{\gamma}(t)} \, dt$$

1. **$\alpha$-connections:** The manifold has a family of affine connections parameterized by $\alpha \in [-1, 1]$. The $e$-connection ($\alpha = 1$) and $m$-connection ($\alpha = -1$) are dual with respect to the Fisher metric.

2. **Dually flat structure:** For exponential families, the manifold is dually flat: it has two flat coordinate systems (natural parameters $\boldsymbol{\theta}$ and expectation parameters $\boldsymbol{\eta} = \mathbb{E}[T(X)]$) connected by the Legendre transform.

**For AI:** The information geometry perspective reveals that the "right" way to measure distances between models is not in parameter space but in distribution space. This explains why natural gradient — which follows the geometry of the statistical manifold — is more effective than standard gradient descent, which follows the (often misleading) geometry of the parameter space.

---

## Appendix B: Worked Examples and Case Studies

### B.1 Complete Newton's Method Analysis: 2D Function

Let us work through Newton's method on a specific non-quadratic function, computing every quantity explicitly.

**Problem:** Minimize $f(x_1, x_2) = x_1^4 + x_1^2 x_2^2 + x_2^4 + x_1 + x_2$ starting from $(x_1, x_2) = (1, 1)$.

**Step 1: Compute gradient and Hessian.**

$$\nabla f = \begin{bmatrix} 4x_1^3 + 2x_1 x_2^2 + 1 \\ 2x_1^2 x_2 + 4x_2^3 + 1 \end{bmatrix}$$

$$\nabla^2 f = \begin{bmatrix} 12x_1^2 + 2x_2^2 & 4x_1 x_2 \\ 4x_1 x_2 & 2x_1^2 + 12x_2^2 \end{bmatrix}$$

**Step 2: Iteration 1 from $(1, 1)$.**

$$\nabla f(1,1) = \begin{bmatrix} 7 \\ 7 \end{bmatrix}, \quad \nabla^2 f(1,1) = \begin{bmatrix} 14 & 4 \\ 4 & 14 \end{bmatrix}$$

Solve $\begin{bmatrix} 14 & 4 \\ 4 & 14 \end{bmatrix}\mathbf{d} = \begin{bmatrix} -7 \\ -7 \end{bmatrix}$:

The inverse is $\frac{1}{180}\begin{bmatrix} 14 & -4 \\ -4 & 14 \end{bmatrix}$, so:

$$\mathbf{d} = \frac{1}{180}\begin{bmatrix} 14 & -4 \\ -4 & 14 \end{bmatrix}\begin{bmatrix} -7 \\ -7 \end{bmatrix} = \frac{-7}{180}\begin{bmatrix} 10 \\ 10 \end{bmatrix} = \begin{bmatrix} -7/18 \\ -7/18 \end{bmatrix} \approx \begin{bmatrix} -0.389 \\ -0.389 \end{bmatrix}$$

$$\boldsymbol{\theta}_1 = (1, 1) + (-0.389, -0.389) = (0.611, 0.611)$$

**Step 3: Iteration 2 from $(0.611, 0.611)$.**

$$\nabla f(0.611, 0.611) \approx \begin{bmatrix} 2.59 \\ 2.59 \end{bmatrix}, \quad \nabla^2 f(0.611, 0.611) \approx \begin{bmatrix} 5.22 & 1.49 \\ 1.49 & 5.22 \end{bmatrix}$$

Solving gives $\mathbf{d} \approx (-0.443, -0.443)$ and $\boldsymbol{\theta}_2 \approx (0.168, 0.168)$.

**Step 4: Convergence.** The true minimum is near $(-0.347, -0.347)$. After a few more iterations, Newton's method converges quadratically. The key observation: each iteration approximately doubles the number of correct digits.

### B.2 BFGS Update: Step-by-Step Computation

Let us compute the BFGS update explicitly for a simple 2D example.

**Given:** $H_0 = I$, $\mathbf{s} = [1, 0]^\top$, $\mathbf{y} = [2, 1]^\top$.

**Step 1:** Compute $\rho = \frac{1}{\mathbf{y}^\top \mathbf{s}} = \frac{1}{2}$.

**Step 2:** Compute the two terms:

$$I - \rho \mathbf{s}\mathbf{y}^\top = I - \frac{1}{2}\begin{bmatrix} 1 \\ 0 \end{bmatrix}\begin{bmatrix} 2 & 1 \end{bmatrix} = \begin{bmatrix} 0 & -0.5 \\ 0 & 1 \end{bmatrix}$$

$$I - \rho \mathbf{y}\mathbf{s}^\top = I - \frac{1}{2}\begin{bmatrix} 2 \\ 1 \end{bmatrix}\begin{bmatrix} 1 & 0 \end{bmatrix} = \begin{bmatrix} 0 & 0 \\ -0.5 & 1 \end{bmatrix}$$

**Step 3:** Compute the update:

$$H_1 = \begin{bmatrix} 0 & -0.5 \\ 0 & 1 \end{bmatrix} I \begin{bmatrix} 0 & 0 \\ -0.5 & 1 \end{bmatrix} + \frac{1}{2}\begin{bmatrix} 1 \\ 0 \end{bmatrix}\begin{bmatrix} 1 & 0 \end{bmatrix}$$

$$= \begin{bmatrix} 0.25 & -0.5 \\ -0.5 & 1 \end{bmatrix} + \begin{bmatrix} 0.5 & 0 \\ 0 & 0 \end{bmatrix} = \begin{bmatrix} 0.75 & -0.5 \\ -0.5 & 1 \end{bmatrix}$$

**Verification:** Check the secant equation $H_1 \mathbf{y} = \mathbf{s}$:

$$H_1 \mathbf{y} = \begin{bmatrix} 0.75 & -0.5 \\ -0.5 & 1 \end{bmatrix}\begin{bmatrix} 2 \\ 1 \end{bmatrix} = \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \mathbf{s} \quad \checkmark$$

### B.3 Gauss-Newton for Exponential Fitting

Fit $y = a e^{bx}$ to data $(x_i, y_i) = \{(0, 1.1), (1, 2.9), (2, 8.1), (3, 22.0)\}$.

**Residuals:** $r_i(a, b) = y_i - a e^{bx_i}$.

**Jacobian:**

$$J = \begin{bmatrix} -e^{b \cdot 0} & -a \cdot 0 \cdot e^{b \cdot 0} \\ -e^{b \cdot 1} & -a \cdot 1 \cdot e^{b \cdot 1} \\ -e^{b \cdot 2} & -a \cdot 2 \cdot e^{b \cdot 2} \\ -e^{b \cdot 3} & -a \cdot 3 \cdot e^{b \cdot 3} \end{bmatrix} = \begin{bmatrix} -1 & 0 \\ -e^b & -ae^b \\ -e^{2b} & -2ae^{2b} \\ -e^{3b} & -3ae^{3b} \end{bmatrix}$$

**Starting guess:** $a_0 = 1$, $b_0 = 1$.

**Iteration 1:**

$$\mathbf{r}_0 = \begin{bmatrix} 1.1 - 1 \\ 2.9 - e \\ 8.1 - e^2 \\ 22.0 - e^3 \end{bmatrix} \approx \begin{bmatrix} 0.1 \\ 0.181 \\ 0.711 \\ 1.914 \end{bmatrix}$$

$$J_0 = \begin{bmatrix} -1 & 0 \\ -2.718 & -2.718 \\ -7.389 & -14.778 \\ -20.086 & -60.257 \end{bmatrix}$$

$$J_0^\top J_0 \approx \begin{bmatrix} 467.5 & 1013.2 \\ 1013.2 & 4136.5 \end{bmatrix}, \quad J_0^\top \mathbf{r}_0 \approx \begin{bmatrix} -41.5 \\ -120.4 \end{bmatrix}$$

Solving $(J_0^\top J_0)\mathbf{d} = -J_0^\top \mathbf{r}_0$ gives $\mathbf{d} \approx [0.063, 0.014]^\top$, so $(a_1, b_1) \approx (1.063, 1.014)$.

After a few more iterations, Gauss-Newton converges to $(a, b) \approx (1.08, 1.02)$, which is close to the true relationship $y \approx e^{x+0.08}$.

### B.4 Natural Gradient for a Bernoulli Model

Consider a Bernoulli model $p(x \mid \theta) = \theta^x (1-\theta)^{1-x}$ with parameter $\theta \in (0, 1)$.

**Fisher information:**

$$F(\theta) = \mathbb{E}\left[\left(\frac{\partial}{\partial \theta}\log p(X \mid \theta)\right)^2\right] = \frac{1}{\theta(1-\theta)}$$

**Standard gradient of negative log-likelihood** for a single observation $x$:

$$\frac{\partial}{\partial \theta}[-\log p(x \mid \theta)] = -\frac{x}{\theta} + \frac{1-x}{1-\theta} = \frac{\theta - x}{\theta(1-\theta)}$$

**Natural gradient:**

$$\tilde{g} = F(\theta)^{-1} g = \theta(1-\theta) \cdot \frac{\theta - x}{\theta(1-\theta)} = \theta - x$$

The natural gradient is simply $\theta - x$, which is the difference between the current parameter and the observation. This is much simpler than the standard gradient, which has the $1/[\theta(1-\theta)]$ factor that blows up near the boundaries.

**Interpretation:** The natural gradient automatically accounts for the fact that a small change in $\theta$ near 0 or 1 has a much larger effect on the distribution than the same change near 0.5. The standard gradient overreacts near the boundaries; the natural gradient corrects for this.

### B.5 Comparison: All Second-Order Methods on a Test Problem

| Method          | Iterations to $10^{-8}$ | Cost per iteration | Total cost   | Best for                |
| --------------- | ----------------------- | ------------------ | ------------ | ----------------------- |
| GD              | 5000                    | $O(n)$             | $O(5000n)$   | Large-scale, non-convex |
| Newton          | 5                       | $O(n^3)$           | $O(5n^3)$    | Small, well-conditioned |
| BFGS            | 15                      | $O(n^2)$           | $O(15n^2)$   | Medium-scale, smooth    |
| L-BFGS ($m=20$) | 20                      | $O(mn)$            | $O(400n)$    | Large-scale, smooth     |
| Gauss-Newton    | 8                       | $O(mn^2)$          | $O(8mn^2)$   | Nonlinear least squares |
| Natural GD      | 5000                    | $O(n^2)$           | $O(5000n^2)$ | Statistical estimation  |
| K-FAC           | 200                     | $O(n)$             | $O(200n)$    | Neural networks         |

The "total cost" column assumes the problem dimension $n$ is large enough that the per-iteration cost dominates. For small $n$, Newton's method is unbeatable. For large $n$, L-BFGS and K-FAC are the practical choices.

---

## Appendix C: Numerical Implementation and Practical Considerations

### C.1 Solving the Newton System Numerically

The core computational bottleneck in Newton's method is solving $H\mathbf{d} = -\mathbf{g}$. The choice of solver depends on the properties of $H$:

**Cholesky factorization** (when $H \succ 0$):

- Factor $H = LL^\top$ where $L$ is lower triangular
- Solve $L\mathbf{y} = -\mathbf{g}$ (forward substitution), then $L^\top \mathbf{d} = \mathbf{y}$ (back substitution)
- Cost: $n^3/3$ flops for factorization, $2n^2$ for solve
- Numerically stable for well-conditioned $H$

**LU factorization** (general $H$):

- Factor $PA = LU$ with partial pivoting
- Cost: $2n^3/3$ flops
- More robust than Cholesky for indefinite matrices

**LDL^T factorization** (symmetric indefinite):

- Factor $P^\top H P = LDL^\top$ where $D$ is block diagonal with 1×1 and 2×2 blocks
- Preserves symmetry without requiring positive definiteness
- Used in modified Newton methods

**Iterative methods** (large $n$):

- Conjugate gradient (for $H \succ 0$): $O(k \cdot \text{nnz}(H))$ where $k$ is iterations
- MINRES (for symmetric indefinite): variant of CG for indefinite systems
- GMRES (for non-symmetric): most general but most expensive

**For AI:** In neural network training, $n$ is too large for direct factorization. The practical approaches are:

1. **Hessian-free:** Use CG with Hessian-vector products (Pearlmutter's trick)
2. **K-FAC:** Use Kronecker factorization to reduce the system size
3. **Shampoo:** Use matrix power iteration for preconditioning

### C.2 Handling Indefinite Hessians

When the Hessian is not positive definite, the Newton direction may not be a descent direction. Several strategies handle this:

**Modified Cholesky:** Find the smallest $\delta \geq 0$ such that $H + \delta I \succ 0$, then use $(H + \delta I)^{-1}\mathbf{g}$. The Gill-Murray modified Cholesky algorithm computes $\delta$ efficiently during factorization.

**Eigenvalue modification:** Compute the eigenvalue decomposition $H = Q\Lambda Q^\top$ and replace negative eigenvalues:

$$H_{\text{mod}} = Q \max(\Lambda, \epsilon I) Q^\top$$

where $\epsilon > 0$ is a small regularization parameter. This is expensive ($O(n^3)$) but gives precise control.

**Trust region:** Instead of modifying $H$, solve the trust region subproblem $\min_{\mathbf{d}} m(\mathbf{d})$ s.t. $\lVert \mathbf{d} \rVert \leq \Delta$. The solution automatically handles indefinite $H$ by taking steps in directions of negative curvature when beneficial.

**Saddle-free Newton:** Replace $H$ with $|H| = Q|\Lambda|Q^\top$ (absolute eigenvalues). This ensures the search direction always decreases the objective, even at saddle points.

### C.3 Computing Hessian-Vector Products Efficiently

**Pearlmutter's trick** (1994) computes $H\mathbf{v} = \nabla^2 f(\boldsymbol{\theta})\mathbf{v}$ using two backward passes:

1. Compute $\mathbf{g} = \nabla f(\boldsymbol{\theta})$ (standard backward pass)
2. Compute $\mathbf{g}^\top \mathbf{v}$ (scalar, directional derivative)
3. Compute $\nabla(\mathbf{g}^\top \mathbf{v})$ (backward pass of the directional derivative)

The result is exactly $H\mathbf{v}$. The total cost is approximately 3× the cost of a single gradient computation — much cheaper than computing the full Hessian ($O(n)$ times more expensive).

**Implementation in JAX:**

```python
import jax
import jax.numpy as jnp

def hvp(f, x, v):
    """Hessian-vector product using JAX."""
    grad_f = jax.grad(f)
    return jax.jvp(grad_f, (x,), (v,))[1]
```

**Implementation in PyTorch:**

```python
import torch

def hvp(f, x, v):
    """Hessian-vector product using PyTorch."""
    g = torch.autograd.grad(f(x), x, create_graph=True)[0]
    return torch.autograd.grad(g @ v, x, retain_graph=True)[0]
```

### C.4 Condition Number Estimation

The condition number $\kappa(H) = \lambda_{\max}/\lambda_{\min}$ determines the difficulty of solving the Newton system. Estimating $\kappa(H)$ without computing all eigenvalues:

**Power iteration** for $\lambda_{\max}$:
$$\mathbf{v}_{k+1} = \frac{H\mathbf{v}_k}{\lVert H\mathbf{v}_k \rVert}, \quad \lambda_{\max} \approx \mathbf{v}_k^\top H \mathbf{v}_k$$

**Inverse iteration** for $\lambda_{\min}$:
$$\mathbf{v}_{k+1} = \frac{H^{-1}\mathbf{v}_k}{\lVert H^{-1}\mathbf{v}_k \rVert}, \quad \lambda_{\min} \approx \frac{1}{\mathbf{v}_k^\top H^{-1} \mathbf{v}_k}$$

**Lanczos method:** Builds a tridiagonal approximation to $H$ and computes its extreme eigenvalues. Converges much faster than power iteration, typically finding the top 10-20 eigenvalues in 50-100 iterations.

**For AI:** The condition number of the neural network Hessian is a key diagnostic:

- $\kappa < 10^3$: Well-conditioned; Newton's method will work well
- $10^3 < \kappa < 10^6$: Moderately ill-conditioned; use preconditioning
- $\kappa > 10^6$: Severely ill-conditioned; use regularization or trust region

### C.5 Memory-Efficient BFGS Implementation

The two-loop recursion for L-BFGS can be implemented efficiently:

```python
def lbfgs_two_loop(g, s_history, y_history, rho_history, H0_scale):
    """L-BFGS two-loop recursion.

    Args:
        g: current gradient
        s_history: list of s_k = theta_{k+1} - theta_k
        y_history: list of y_k = g_{k+1} - g_k
        rho_history: list of 1/(y_k^T s_k)
        H0_scale: scaling factor for initial H_0

    Returns:
        search direction d = -H * g
    """
    m = len(s_history)
    q = g.copy()
    alpha = [0] * m

    # Loop 1: backward through history
    for i in range(m - 1, -1, -1):
        alpha[i] = rho_history[i] * np.dot(s_history[i], q)
        q = q - alpha[i] * y_history[i]

    # Initial Hessian approximation
    r = H0_scale * q

    # Loop 2: forward through history
    for i in range(m):
        beta = rho_history[i] * np.dot(y_history[i], r)
        r = r + (alpha[i] - beta) * s_history[i]

    return -r
```

**Memory usage:** $O(mn)$ for storing $m$ pairs of $(n$-dimensional) vectors. For $n = 10^6$ and $m = 20$, this is about 320 MB — feasible on most machines.

### C.6 When to Use Which Method

**Decision tree for choosing a second-order method:**

```text
Is the problem convex?
├── YES → Is n < 1000?
│         ├── YES → Use Newton's method (fastest convergence)
│         └── NO → Is n < 100,000?
│                  ├── YES → Use BFGS
│                  └── NO → Use L-BFGS
├── NO (non-convex) → Is it a least-squares problem?
│                     ├── YES → Use Gauss-Newton / Levenberg-Marquardt
│                     └── NO → Is it a neural network?
│                              ├── YES → Start with AdamW; test curvature methods only if justified
│                              └── NO → Use L-BFGS with caution
└── Statistical estimation → Use Natural Gradient
```

**For AI practitioners:** The default choice for neural network training remains AdamW or another well-tested first-order baseline. Second-order methods are worth considering when:

1. The problem is small enough ($n < 10^4$) for BFGS/L-BFGS
2. The problem has least-squares structure (use Gauss-Newton)
3. Training cost is the bottleneck and you have evidence that structured preconditioning will offset its systems overhead
4. Reparameterization invariance is important (use natural gradient)
