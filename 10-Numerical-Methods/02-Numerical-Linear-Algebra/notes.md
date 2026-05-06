[<- Back to Chapter 10](../README.md) | [Next: Numerical Optimization ->](../03-Numerical-Optimization/notes.md)

---

# Numerical Linear Algebra

> _"It is a capital mistake to theorize before one has data. Insensibly one begins to twist facts to suit theories, instead of theories to suit facts - and in linear algebra, the theory of stable computation is entirely about facts: actual rounding errors, actual condition numbers, actual flop counts."_

## Overview

Numerical linear algebra is the study of algorithms for solving the core computational problems of linear algebra - linear systems, least squares, eigenvalue problems, and matrix factorizations - in the presence of floating-point arithmetic. The central challenge: mathematical equivalence does not imply numerical equivalence. The Gaussian elimination formula $x = A^{-1}b$ is mathematically exact but computationally disastrous without pivoting; the Normal equations $A^\top A x = A^\top b$ are correct but the QR approach is more numerically stable.

This section covers the algorithms that power deep learning, scientific computing, and data analysis: LU with partial pivoting, QR via Householder reflectors, iterative solvers for large sparse systems, the conjugate gradient method, and the numerical sensitivity of eigenvalue problems. Every algorithm is analyzed through the lens of backward stability - the gold standard for numerical linear algebra.

**AI connections:** Every gradient computation requires a matrix-vector product; attention mechanisms compute $QK^\top V$ via batched matrix-matrix multiplications; training large language models involves solving implicit linear systems via iterative optimization. The choice of algorithm and its numerical properties directly affect convergence, stability, and generalization.

## Prerequisites

- [Section01 Floating-Point Arithmetic](../01-Floating-Point-Arithmetic/notes.md) - machine epsilon, condition numbers, backward stability
- [Section02-Linear-Algebra-Basics/03-Systems-of-Equations](../../02-Linear-Algebra-Basics/03-Systems-of-Equations/notes.md) - Gaussian elimination, existence, uniqueness
- [Section03-Advanced-Linear-Algebra/08-Matrix-Decompositions](../../03-Advanced-Linear-Algebra/08-Matrix-Decompositions/notes.md) - LU, QR, Cholesky theory
- [Section03-Advanced-Linear-Algebra/01-Eigenvalues-and-Eigenvectors](../../03-Advanced-Linear-Algebra/01-Eigenvalues-and-Eigenvectors/notes.md) - eigenvalue theory

## Companion Notebooks

| Notebook | Description |
|----------|-------------|
| [theory.ipynb](theory.ipynb) | LU factorization, QR, iterative solvers, conjugate gradient, condition estimation |
| [exercises.ipynb](exercises.ipynb) | 10 graded exercises on stability, pivoting, iterative methods, and AI applications |

## Learning Objectives

After completing this section, you will:

- Explain why LU without pivoting is numerically unstable and how partial pivoting fixes it
- Implement Gaussian elimination with partial pivoting from scratch
- Explain why QR via Householder reflectors is backward stable while Normal equations are not
- Apply the backward error framework: relate forward error to backward error via condition number
- Implement and analyze the Conjugate Gradient method for symmetric positive definite systems
- Understand iterative refinement and its role in achieving double-precision accuracy
- Connect condition number to loss of digits: $\kappa(A) = 10^k$ means $k$ digits are lost
- Relate numerical linear algebra algorithms to the batched matrix operations in deep learning

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 Why Numerical Linear Algebra?](#11-why-numerical-linear-algebra)
  - [1.2 Backward Stability: The Right Standard](#12-backward-stability-the-right-standard)
  - [1.3 Historical Timeline](#13-historical-timeline)
- [2. Forward and Backward Error Analysis](#2-forward-and-backward-error-analysis)
  - [2.1 The Error Analysis Framework](#21-the-error-analysis-framework)
  - [2.2 The Fundamental Theorem](#22-the-fundamental-theorem)
  - [2.3 Perturbation Theory for Linear Systems](#23-perturbation-theory-for-linear-systems)
- [3. Gaussian Elimination and LU Factorization](#3-gaussian-elimination-and-lu-factorization)
  - [3.1 LU without Pivoting: Why It Fails](#31-lu-without-pivoting-why-it-fails)
  - [3.2 Partial Pivoting: The Fix](#32-partial-pivoting-the-fix)
  - [3.3 Backward Stability of LU with Partial Pivoting](#33-backward-stability-of-lu-with-partial-pivoting)
  - [3.4 Banded and Sparse Systems](#34-banded-and-sparse-systems)
- [4. QR Factorization: The Numerically Stable Approach](#4-qr-factorization-the-numerically-stable-approach)
  - [4.1 Why Normal Equations Fail](#41-why-normal-equations-fail)
  - [4.2 Householder Reflectors](#42-householder-reflectors)
  - [4.3 Gram-Schmidt and Modified Gram-Schmidt](#43-gram-schmidt-and-modified-gram-schmidt)
  - [4.4 Backward Stability of Householder QR](#44-backward-stability-of-householder-qr)
- [5. Cholesky Factorization](#5-cholesky-factorization)
  - [5.1 Positive Definiteness and Stability](#51-positive-definiteness-and-stability)
  - [5.2 Numerical Positive Definiteness](#52-numerical-positive-definiteness)
- [6. Iterative Methods](#6-iterative-methods)
  - [6.1 Jacobi and Gauss-Seidel](#61-jacobi-and-gauss-seidel)
  - [6.2 Conjugate Gradient Method](#62-conjugate-gradient-method)
  - [6.3 Preconditioning](#63-preconditioning)
  - [6.4 GMRES for Non-Symmetric Systems](#64-gmres-for-non-symmetric-systems)
- [7. Eigenvalue Computation](#7-eigenvalue-computation)
  - [7.1 Power Iteration](#71-power-iteration)
  - [7.2 QR Algorithm](#72-qr-algorithm)
  - [7.3 Sensitivity of Eigenvalues](#73-sensitivity-of-eigenvalues)
- [8. Applications in Machine Learning](#8-applications-in-machine-learning)
  - [8.1 Solving the Normal Equations in Ridge Regression](#81-solving-the-normal-equations-in-ridge-regression)
  - [8.2 Batched Matrix Operations in Transformers](#82-batched-matrix-operations-in-transformers)
  - [8.3 Iterative Refinement in Neural Solvers](#83-iterative-refinement-in-neural-solvers)
  - [8.4 Condition Numbers in Gradient Descent](#84-condition-numbers-in-gradient-descent)
- [9. Common Mistakes](#9-common-mistakes)
- [10. Exercises](#10-exercises)
- [11. Why This Matters for AI (2026 Perspective)](#11-why-this-matters-for-ai-2026-perspective)
- [12. Conceptual Bridge](#12-conceptual-bridge)

---

## 1. Intuition

### 1.1 Why Numerical Linear Algebra?

Consider solving a $1000 \times 1000$ linear system $Ax = b$. You know the answer: $x = A^{-1}b$. But computing $A^{-1}$ explicitly requires $O(n^3)$ operations and is numerically inferior to solving $Ax = b$ directly. More surprisingly, two mathematically equivalent formulations can have dramatically different numerical properties:

**Normal equations approach:** Multiply both sides of $Ax \approx b$ by $A^\top$:
$$A^\top A x = A^\top b$$

**QR approach:** Factor $A = QR$ where $Q$ is orthogonal, $R$ is upper triangular:
$$Rx = Q^\top b$$

Both are mathematically equivalent when $A$ has full column rank, but their **condition numbers differ**:
$$\kappa(A^\top A) = \kappa(A)^2$$

If $A$ has condition number $\kappa(A) = 10^7$ (common in practice), then $A^\top A$ has condition number $10^{14}$ - beyond fp64 precision. QR factorization preserves $\kappa(A)$.

**For AI:** When training neural networks, you are implicitly solving linear systems at every step. The choice of optimizer (SGD vs Adam vs second-order methods), the use of batch normalization, and the initialization strategy all reflect numerical linear algebra considerations, even if they are rarely stated as such.

### 1.2 Backward Stability: The Right Standard

A naive view of algorithm correctness: compute $\hat{y} \approx f(x)$ with small forward error $|\hat{y} - f(x)| / |f(x)| \leq c \varepsilon_{\text{mach}}$.

The more powerful concept is **backward stability**: algorithm $\hat{f}$ is backward stable if for every input $x$, the computed output $\hat{f}(x)$ is the exact output for a slightly perturbed input:
$$\hat{f}(x) = f(x + \delta x), \quad \|\delta x\| / \|x\| \leq c \varepsilon_{\text{mach}}$$

A backward stable algorithm gives the exact answer to a slightly wrong problem. This is the right standard because:
1. **It achieves the best possible accuracy**: if the problem itself is uncertain at level $\varepsilon_{\text{mach}}$, a backward stable algorithm is optimally accurate
2. **It separates algorithm from problem**: the condition number tells you how much forward error backward stability implies
3. **It is composable**: backward stable algorithms composed together give overall backward stability (approximately)

The fundamental theorem then states: forward error $\leq$ condition number $\times$ backward error.

### 1.3 Historical Timeline

```
NUMERICAL LINEAR ALGEBRA: HISTORICAL TIMELINE
========================================================================

  1947  von Neumann & Goldstine - first systematic error analysis of
        Gaussian elimination; estimated stability incorrectly (pessimistic)

  1951  Turing - defined matrix condition number $\kappa(A)$;
        connected it to loss of significant digits

  1954  Wilkinson - proved Gaussian elimination with partial pivoting
        is backward stable in practice; introduced error analysis
        methodology that became the foundation of the field

  1958  Householder - reflector-based QR factorization; proved backward
        stable; superior to Gram-Schmidt for loss of orthogonality

  1961  Lanczos - Lanczos iteration for symmetric eigenvalue problems;
        precursor to modern Krylov subspace methods

  1965  Golub & Kahan - bidiagonalization for SVD computation;
        established SVD as numerically accessible

  1968  Wilkinson - "The Algebraic Eigenvalue Problem"; definitive
        reference on eigenvalue computation stability

  1976  Paige & Saunders - MINRES and SYMMLQ; stable iterative methods
        for symmetric systems

  1986  Saad & Schultz - GMRES algorithm; now standard for non-symmetric
        iterative solution

  1994  LAPACK released - standardized implementation of all major
        backward-stable dense linear algebra routines

  2008  Buttari et al. - mixed-precision iterative refinement;
        fp16 factorization + fp64 refinement

  2022+ - cuBLAS, cuSOLVER using Tensor Cores (tf32/bf16) for
        matrix operations in LLM training and inference

========================================================================
```

---

## 2. Forward and Backward Error Analysis

### 2.1 The Error Analysis Framework

Let $f: \mathbb{R}^n \to \mathbb{R}^m$ be the mathematical function we want to compute, and let $\hat{f}$ be a floating-point algorithm. For input $x$, we define:

**Absolute forward error:** $\|\hat{f}(x) - f(x)\|$

**Relative forward error:** $\|\hat{f}(x) - f(x)\| / \|f(x)\|$

**Absolute backward error:** The smallest $\|\delta x\|$ such that $\hat{f}(x) = f(x + \delta x)$

**Relative backward error:** $\|\delta x\| / \|x\|$

**Algorithm $\hat{f}$ is:**
- **Accurate** if forward error $= O(\varepsilon_{\text{mach}})$ - computes the answer to full precision
- **Backward stable** if relative backward error $= O(\varepsilon_{\text{mach}})$ - solves a nearby problem exactly
- **Forward stable** if relative forward error $= O(\kappa(f, x) \varepsilon_{\text{mach}})$ - optimal up to condition number

**Example: inner product** $f(x, y) = x^\top y$

Naive inner product: compute $\hat{s} = \sum_i \text{fl}(x_i y_i)$. The backward error satisfies:
$$\hat{s} = (x + \delta x)^\top y, \quad |\delta x_i| \leq n \varepsilon_{\text{mach}} |x_i|$$

So naive inner product is backward stable (relative backward error $\leq n \varepsilon_{\text{mach}}$).

But the condition number of the inner product is:
$$\kappa = \frac{\|x\| \|y\|}{|x^\top y|}$$

which can be huge when $x \approx -y$ (catastrophic cancellation). The forward error $\leq \kappa \cdot n \varepsilon_{\text{mach}}$ can be large, but the algorithm itself is not to blame - the problem is ill-conditioned.

### 2.2 The Fundamental Theorem

**Theorem (Forward-Backward Error):** For a backward stable algorithm $\hat{f}$ with relative backward error $\eta$:
$$\frac{\|\hat{f}(x) - f(x)\|}{\|f(x)\|} \leq \kappa(f, x) \cdot \eta + O(\eta^2)$$

where $\kappa(f, x)$ is the condition number of $f$ at $x$.

**Interpretation:** The forward error is at most condition number times backward error. A backward stable algorithm with $\eta = O(\varepsilon_{\text{mach}})$ gives forward error $O(\kappa \varepsilon_{\text{mach}})$, which is the best possible - no algorithm can be more accurate in general.

**For AI:** This theorem tells us when to worry:
- $\kappa < 10^7$ with fp64: fine, we have $\sim 7$ accurate digits
- $\kappa > 10^{12}$ with fp64: we lose all accuracy; the problem is inherently ill-conditioned
- $\kappa > 10^{7}$ with fp32: dangerous; use fp64 or regularization

### 2.3 Perturbation Theory for Linear Systems

For the linear system $Ax = b$, with perturbed system $(A + \delta A)(x + \delta x) = b + \delta b$:

**Bound on forward error:**
$$\frac{\|\delta x\|}{\|x\|} \leq \kappa(A) \left( \frac{\|\delta A\|}{\|A\|} + \frac{\|\delta b\|}{\|b\|} \right) + O(\kappa^2)$$

**Proof sketch:** From $(A + \delta A)(x + \delta x) = b + \delta b$:
$$A \delta x = \delta b - \delta A (x + \delta x)$$
$$\delta x = A^{-1}(\delta b - \delta A (x + \delta x))$$

Taking norms and using $\|b\| \leq \|A\| \|x\|$:
$$\frac{\|\delta x\|}{\|x\|} \leq \|A\| \|A^{-1}\| \left( \frac{\|\delta b\|}{\|b\|} + \frac{\|\delta A\|}{\|A\|} \right) = \kappa(A) \left( \frac{\|\delta b\|}{\|b\|} + \frac{\|\delta A\|}{\|A\|} \right)$$

**Key consequences:**
- Condition number amplifies input perturbations in the output
- For backward stable solvers: $\|\delta A\| / \|A\| = O(\varepsilon_{\text{mach}})$, $\|\delta b\| / \|b\| = O(\varepsilon_{\text{mach}})$
- Therefore forward error: $\|\delta x\| / \|x\| = O(\kappa(A) \varepsilon_{\text{mach}})$
- Rule of thumb: if $\kappa(A) = 10^k$, we lose $k$ significant digits

**Condition number interpretation:**

| $\kappa(A)$ | Digits lost (fp64, 15 digits) | Digits lost (fp32, 7 digits) |
|-------------|-------------------------------|------------------------------|
| $10^0 = 1$ | 0 digits | 0 digits |
| $10^3$ | 3 digits | 3 digits |
| $10^7$ | 7 digits | **all digits** |
| $10^{10}$ | 10 digits | impossible |
| $10^{15}$ | **all digits** | impossible |

---

## 3. Gaussian Elimination and LU Factorization

> **Scope note:** This section covers the *numerical stability* of LU factorization - pivoting, backward error analysis, and growth factors. The mathematical theory of LU (existence, uniqueness, relationship to determinants) is in [Section03-Advanced-Linear-Algebra/08-Matrix-Decompositions](../../03-Advanced-Linear-Algebra/08-Matrix-Decompositions/notes.md).

### 3.1 LU without Pivoting: Why It Fails

LU factorization computes $A = LU$ where $L$ is lower triangular with 1s on the diagonal and $U$ is upper triangular. The elimination multipliers are:
$$\ell_{ij} = \frac{a_{ij}^{(j)}}{a_{jj}^{(j)}}$$

where $a_{jj}^{(j)}$ is the **pivot**. If the pivot is zero, the algorithm fails. If it is small, large multipliers $\ell_{ij}$ amplify rounding errors.

**Example: Near-zero pivot disaster**

$$A = \begin{pmatrix} \varepsilon & 1 \\ 1 & 1 \end{pmatrix}, \quad \varepsilon \ll 1$$

Multiplier: $\ell_{21} = 1/\varepsilon$ (very large). After elimination:
$$U = \begin{pmatrix} \varepsilon & 1 \\ 0 & 1 - 1/\varepsilon \end{pmatrix}$$

For $\varepsilon = 10^{-15}$ in fp64: $1 - 1/\varepsilon = 1 - 10^{15} \approx -10^{15}$ (catastrophic cancellation in the $(2,2)$ entry).

**Growth factor:** The growth factor $\rho = \max_{i,j,k} |u_{ij}^{(k)}| / \max_{i,j} |a_{ij}|$ measures how much the entries grow during elimination. Without pivoting, $\rho$ can be exponentially large.

### 3.2 Partial Pivoting: The Fix

**Partial pivoting** chooses the pivot as the largest element in magnitude in the current column:

At step $k$: find $i^* = \text{argmax}_{i \geq k} |a_{ik}^{(k)}|$ and swap rows $k$ and $i^*$.

**Algorithm: Gaussian Elimination with Partial Pivoting (GEPP)**

```
for k = 1, 2, ..., n-1:
    find i* = argmax_{i >= k} |a_{ik}|
    swap rows k and i*  (record in permutation P)
    for i = k+1, ..., n:
        l_{ik} = a_{ik} / a_{kk}
        for j = k+1, ..., n:
            a_{ij} -= l_{ik} * a_{kj}
```

This gives the factorization $PA = LU$ where $P$ is the permutation matrix of row swaps.

**Solving $Ax = b$ via LU with pivoting:**
1. Factor: $PA = LU$ (GEPP above)
2. Apply permutation: $y = Pb$
3. Forward solve: $Lz = y$ (triangular solve, $O(n^2)$)
4. Back solve: $Ux = z$ (triangular solve, $O(n^2)$)

Total cost: $\frac{2n^3}{3} + O(n^2)$ flops.

**Growth factor with partial pivoting:** The entries of $L$ satisfy $|\ell_{ij}| \leq 1$ (because we choose the largest pivot). The growth factor satisfies:
$$\rho_{\text{partial}} \leq 2^{n-1}$$

This bound is exponential but almost never achieved in practice. Wilkinson (1963) showed that for random matrices, $\rho \approx n^{2/3}$, making GEPP very stable in practice.

### 3.3 Backward Stability of LU with Partial Pivoting

**Theorem (Wilkinson, 1961):** Gaussian elimination with partial pivoting applied to $A \in \mathbb{R}^{n \times n}$ computes $\hat{L}$, $\hat{U}$, $\hat{P}$ satisfying:
$$\hat{P}A = \hat{L}\hat{U} + \delta A, \quad \|\delta A\|_\infty \leq 8n^3 \rho \varepsilon_{\text{mach}} \|A\|_\infty$$

where $\rho$ is the growth factor. If $\rho$ is small (as is almost always the case), GEPP is effectively backward stable.

**Consequence:** The computed solution $\hat{x}$ to $Ax = b$ satisfies:
$$(A + \delta A)\hat{x} = b, \quad \|\delta A\|_\infty / \|A\|_\infty = O(\rho n \varepsilon_{\text{mach}})$$

And the forward error:
$$\frac{\|\hat{x} - x\|}{\|x\|} = O(\kappa(A) \cdot \rho n \varepsilon_{\text{mach}})$$

### 3.4 Banded and Sparse Systems

For **banded matrices** with bandwidth $w$ (nonzero entries only within $w$ diagonals of the main diagonal):
- Storage: $O(nw)$ instead of $O(n^2)$
- LU factorization cost: $O(nw^2)$ instead of $O(n^3)$
- Fill-in: LU of a banded matrix remains banded (with bandwidth $w$ in $L$ and $U$)

**For AI:** The attention matrix in transformers is $n \times n$ but has local structure. Efficient attention algorithms exploit this: sliding window attention (bandwidth $w$), strided attention, and Longformer all use banded or sparse attention patterns that reduce cost from $O(n^2)$ to $O(nw)$.

**Sparse direct solvers:** For general sparse $A$, the key challenge is **fill-in** - LU introduces nonzero entries where $A$ had zeros. Fill-in is minimized by reordering rows/columns (e.g., minimum degree ordering, nested dissection). Modern sparse solvers (PARDISO, MUMPS, SuperLU) use sophisticated reordering strategies.

---

## 4. QR Factorization: The Numerically Stable Approach

> **Scope note:** QR factorization theory (existence, uniqueness, Gram-Schmidt, Givens rotations) is in [Section03-Advanced-Linear-Algebra/08-Matrix-Decompositions](../../03-Advanced-Linear-Algebra/08-Matrix-Decompositions/notes.md). This section focuses on numerical stability.

### 4.1 Why Normal Equations Fail

For the least squares problem $\min_x \|Ax - b\|_2$ with $A \in \mathbb{R}^{m \times n}$, $m > n$:

**Normal equations:** $A^\top A x = A^\top b$

This is mathematically correct but numerically problematic:
- $\kappa(A^\top A) = \kappa(A)^2$: condition number is squared
- If $\kappa(A) = 10^7$, then $\kappa(A^\top A) = 10^{14}$, which in fp64 (15 digits) means complete loss of accuracy
- Even forming $A^\top A$ introduces rounding errors

**Example:** Consider $A = \begin{pmatrix} 1 & 1 \\ \varepsilon & 0 \\ 0 & \varepsilon \end{pmatrix}$ with $\varepsilon = \sqrt{\varepsilon_{\text{mach}}}$.

Then $A^\top A = \begin{pmatrix} 1 + \varepsilon^2 & 1 \\ 1 & 1 + \varepsilon^2 \end{pmatrix}$. In floating point, $1 + \varepsilon^2 = 1$ (since $\varepsilon^2 = \varepsilon_{\text{mach}}$), so the computed $A^\top A$ is exactly $\begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$, which is singular. The original $A$ is well-conditioned, but Normal equations make it appear singular.

### 4.2 Householder Reflectors

A **Householder reflector** (or Householder transformation) is an orthogonal matrix:
$$H = I - 2vv^\top / (v^\top v)$$

that reflects vectors through the hyperplane perpendicular to $v$. Geometrically, $Hx$ is the reflection of $x$ across the hyperplane $\{y : v^\top y = 0\}$.

**Key property:** By choosing $v = x - \|x\| e_1$ (where $e_1$ is the first standard basis vector), we get:
$$Hx = \|x\| e_1$$

This zeros out all but the first component of $x$.

**QR via Householder:** Apply $n$ Householder reflectors $H_1, H_2, \ldots, H_n$ to successively zero out the subdiagonal of each column:
$$H_n \cdots H_2 H_1 A = R \quad \Rightarrow \quad A = Q R$$

where $Q = H_1 H_2 \cdots H_n$ is orthogonal (product of orthogonal matrices) and $R$ is upper triangular.

**Cost:** $2mn^2 - \frac{2n^3}{3}$ flops for $m \geq n$.

**Numerical stability:** Each Householder reflector $H_k$ is orthogonal exactly, so multiplying by $H_k$ does not increase rounding errors. This makes the algorithm backward stable.

**Implementation detail:** In practice, Householder QR is implemented implicitly - we never form $Q$ explicitly. Instead, we store the $n$ vectors $v_k$ and apply $Q^\top b = H_n \cdots H_1 b$ by successively applying each reflector (cost: $O(mn)$ per reflector, $O(mn^2)$ total).

### 4.3 Gram-Schmidt and Modified Gram-Schmidt

The classical Gram-Schmidt process orthogonalizes the columns of $A$ one by one:

**Classical Gram-Schmidt (CGS):**
```
q_1 = a_1 / ||a_1||
for k = 2, ..., n:
    r_{ik} = q_i^T a_k  for i = 1, ..., k-1
    q_k_unnorm = a_k - sum_{i=1}^{k-1} r_{ik} q_i
    r_{kk} = ||q_k_unnorm||
    q_k = q_k_unnorm / r_{kk}
```

**Problem:** CGS is numerically unstable. The orthogonality of $q_i$ is lost due to cancellation. After computing all $n$ vectors, $Q^\top Q$ may be far from identity.

**Modified Gram-Schmidt (MGS):** Reorders the computation to project out each $q_k$ as soon as it is computed:

```
for k = 1, ..., n:
    q_k = a_k
    for i = 1, ..., k-1:
        r_{ik} = q_i^T q_k
        q_k -= r_{ik} * q_i  # Project q_k immediately
    r_{kk} = ||q_k||
    q_k = q_k / r_{kk}
```

MGS is **algebraically identical** to CGS but **numerically superior**: the loss of orthogonality in MGS satisfies $\|I - Q^\top Q\| = O(\kappa(A) \varepsilon_{\text{mach}})$ vs $O(\kappa(A)^2 \varepsilon_{\text{mach}})$ for CGS.

However, MGS is still worse than Householder QR ($O(\kappa(A)^2 \varepsilon_{\text{mach}})$ vs $O(\varepsilon_{\text{mach}})$). For high-accuracy least squares, use Householder QR.

**When to use each:**
- **Householder QR:** Direct least squares, high accuracy required, dense matrices
- **MGS:** Online/streaming orthogonalization, Lanczos/Arnoldi iteration, where the matrix is generated column by column
- **Normal equations:** Large-scale problems where $\kappa(A)$ is small; much faster to form and solve than QR

### 4.4 Backward Stability of Householder QR

**Theorem:** Householder QR applied to $A \in \mathbb{R}^{m \times n}$ computes $\hat{Q}$, $\hat{R}$ such that:
$$\hat{Q}\hat{R} = A + \delta A, \quad \|\delta A\|_F / \|A\|_F = O(\varepsilon_{\text{mach}})$$

The computed solution $\hat{x}$ to $\min_x \|Ax - b\|_2$ satisfies:
$$\frac{\|\hat{x} - x\|}{\|x\|} = O(\kappa(A) \varepsilon_{\text{mach}})$$

**Comparison with Normal equations:** The Normal equations approach gives:
$$\frac{\|\hat{x} - x\|}{\|x\|} = O(\kappa(A)^2 \varepsilon_{\text{mach}})$$

For $\kappa(A) = 10^4$ in fp64: Householder gives 11 correct digits; Normal equations give only 7.

---

## 5. Cholesky Factorization

> **Scope note:** Cholesky factorization theory (existence for SPD matrices, relationship to LU) is in [Section03-Advanced-Linear-Algebra/08-Matrix-Decompositions](../../03-Advanced-Linear-Algebra/08-Matrix-Decompositions/notes.md). This section covers numerical aspects.

### 5.1 Positive Definiteness and Stability

For symmetric positive definite (SPD) matrices $A = A^\top$, $x^\top A x > 0$ for all $x \neq 0$:

- Cholesky factorization $A = LL^\top$ exists and is unique
- **No pivoting needed**: the diagonal entries $\ell_{kk} = \sqrt{a_{kk} - \sum_{j<k} \ell_{kj}^2}$ are always positive
- **Half the cost** of LU: $\frac{n^3}{3}$ flops vs $\frac{2n^3}{3}$
- **Perfect stability**: the growth factor is exactly 1 (entries of $L$ cannot exceed those of $A$ in magnitude)

**Theorem:** Cholesky is backward stable: the computed factor $\hat{L}$ satisfies $\hat{L}\hat{L}^\top = A + \delta A$ with $\|\delta A\| = O(\varepsilon_{\text{mach}} \|A\|)$.

### 5.2 Numerical Positive Definiteness

In practice, matrices that are theoretically SPD may be only nearly SPD due to:
- Accumulated rounding errors in forming $A = B^\top B$ (e.g., covariance matrices)
- Near-zero eigenvalues that become slightly negative in fp64

**Strategies for numerical SPD:**
1. **Diagonal loading:** Replace $A$ with $A + \delta I$ where $\delta > 0$ is small enough to preserve the physics but large enough to ensure numerical SPD. This is also called **Tikhonov regularization**.
2. **Modified Cholesky:** If $\hat{\ell}_{kk}^2 < 0$ during factorization, add a small correction to the diagonal.
3. **Eigenvalue floor:** Clip small eigenvalues: $A \leftarrow V \max(\Lambda, \epsilon I) V^\top$.

**For AI:**
- Adam optimizer maintains a running estimate of the squared gradient (Hessian diagonal proxy). This must stay positive, so Adam clips to a minimum value $\epsilon = 10^{-8}$ in the update formula.
- Second-order methods (K-FAC, Shampoo) factor approximate Hessians via Cholesky. Numerical indefiniteness is handled by damping: $A + \lambda I$.
- Batch covariance matrices in normalization layers must be SPD; the $\varepsilon$ in `BatchNorm(eps=1e-5)` serves this role.

---

## 6. Iterative Methods

Direct methods (LU, QR, Cholesky) require $O(n^3)$ work and $O(n^2)$ storage - feasible for $n \leq 10^4$ but impractical for $n = 10^6$ (e.g., discretized PDEs, large graph problems). **Iterative methods** compute $x^{(k)} \to x$ without forming or factoring $A$, requiring only **matrix-vector products** $Av$.

### 6.1 Jacobi and Gauss-Seidel

The simplest iterative schemes split $A = D + L + U$ (diagonal, strictly lower, strictly upper triangular):

**Jacobi iteration:**
$$x^{(k+1)}_i = \frac{1}{a_{ii}} \left( b_i - \sum_{j \neq i} a_{ij} x^{(k)}_j \right)$$

**Gauss-Seidel:** Use updated values immediately:
$$x^{(k+1)}_i = \frac{1}{a_{ii}} \left( b_i - \sum_{j < i} a_{ij} x^{(k+1)}_j - \sum_{j > i} a_{ij} x^{(k)}_j \right)$$

**Convergence:** Both converge if and only if the spectral radius $\rho(I - D^{-1}A) < 1$.

**For diagonally dominant matrices** ($|a_{ii}| > \sum_{j \neq i} |a_{ij}|$), both converge. Gauss-Seidel typically converges faster (by a factor of 2 in terms of iterations for SPD systems).

**For AI:** Jacobi is embarrassingly parallel - all updates are independent. Gauss-Seidel requires sequential updates but has better convergence. In distributed deep learning, Jacobi-like parallel gradient updates correspond to asynchronous SGD with stale gradients.

### 6.2 Conjugate Gradient Method

For SPD $A$, the **Conjugate Gradient (CG)** method is the optimal Krylov subspace method:

**Algorithm (Hestenes-Stiefel, 1952):**
```
r_0 = b - A x_0,  p_0 = r_0
for k = 0, 1, 2, ...:
    alpha_k = (r_k^T r_k) / (p_k^T A p_k)
    x_{k+1} = x_k + alpha_k p_k
    r_{k+1} = r_k - alpha_k A p_k
    beta_k  = (r_{k+1}^T r_{k+1}) / (r_k^T r_k)
    p_{k+1} = r_{k+1} + beta_k p_k
```

**Key properties:**
- Each step requires one matrix-vector product $Ap_k$
- The iterates minimize $\|x - x_k\|_A = (x - x_k)^\top A (x - x_k)$ over the Krylov subspace $\mathcal{K}_k(A, r_0)$
- Converges in exactly $n$ steps (in exact arithmetic)
- In practice, converges much earlier for well-clustered eigenvalues

**Convergence rate:** The error satisfies:
$$\frac{\|x - x_k\|_A}{\|x - x_0\|_A} \leq 2 \left( \frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1} \right)^k$$

where $\kappa = \kappa(A) = \lambda_{\max} / \lambda_{\min}$.

For $\kappa = 100$: after $k = 10$ steps, error ratio $\leq 2 \cdot (0.818)^{10} \approx 0.27$ - reduces error by factor 4.

For $\kappa = 10000$: need $k \approx 100$ steps for same reduction.

**For AI:** Conjugate gradient is used in:
- Second-order optimization (natural gradient, K-FAC): solving linear systems involving the Fisher information matrix
- Computing Hessian-vector products implicitly (Pearlmutter's trick)
- Solving graph Laplacian systems in GNNs
- The PCG preconditioner in FlashAttention's attention computation via block algorithms

### 6.3 Preconditioning

CG convergence rate depends on $\kappa(A)$. **Preconditioning** transforms the system to improve conditioning:

**Preconditioned CG:** Instead of solving $Ax = b$, solve $M^{-1}Ax = M^{-1}b$ where $M$ is a **preconditioner** - an easy-to-invert approximation of $A$.

The convergence rate becomes $O(\sqrt{\kappa(M^{-1}A)})$ instead of $O(\sqrt{\kappa(A)})$.

**Common preconditioners:**
| Preconditioner | Construction | Application | Quality |
|----------------|-------------|-------------|---------|
| Diagonal (Jacobi) | $M = \text{diag}(A)$ | $O(n)$ | Removes diagonal scaling |
| Incomplete Cholesky (IC) | Sparse Cholesky with fill-in limited | $O(n)$ triangular solve | Good for SPD PDE problems |
| Sparse Approximate Inverse | Minimize $\|AM - I\|_F$ sparsity | $O(n)$ matvec | Highly parallel |
| AMG (Algebraic Multigrid) | Hierarchical coarsening | $O(n)$ cycle | Near-optimal for elliptic PDEs |

**For AI:** Diagonal preconditioning is Adam: $M = \text{diag}(v)$ where $v$ is the running second-moment estimate. Adam's $\sqrt{v}$ denominator pre-scales gradients by an approximation of the Hessian diagonal - this is diagonal preconditioning applied to the implicit linear system of gradient descent.

### 6.4 GMRES for Non-Symmetric Systems

For non-symmetric $A$, CG does not apply. **GMRES** (Generalized Minimal Residual, Saad & Schultz, 1986) minimizes $\|b - Ax_k\|_2$ over the Krylov subspace $\mathcal{K}_k(A, r_0)$:

**Algorithm sketch:**
1. Build an orthonormal basis $\{q_1, \ldots, q_k\}$ for $\mathcal{K}_k$ via **Arnoldi iteration** (modified Gram-Schmidt on the vectors $r_0, Ar_0, A^2 r_0, \ldots$)
2. Solve the least squares problem: $\min_{y} \|b - Q_k y\|_2$

**Key properties:**
- Guaranteed convergence (unlike non-restarted Jacobi/GS)
- Memory grows as $O(kn)$ - restart every $m$ steps: GMRES(m)
- Optimal over the Krylov subspace at each step

**For AI:** GMRES is used in physics-informed neural networks (PINNs) and neural ODEs where the implicit time-stepping requires solving non-symmetric linear systems at each step.

---

## 7. Eigenvalue Computation

> **Scope note:** Eigenvalue theory (characteristic polynomial, diagonalizability, spectral theorem) is in [Section03-Advanced-Linear-Algebra/01-Eigenvalues-and-Eigenvectors](../../03-Advanced-Linear-Algebra/01-Eigenvalues-and-Eigenvectors/notes.md). This section covers numerical eigenvalue computation.

### 7.1 Power Iteration

**Power iteration** finds the dominant eigenvalue $\lambda_1$ (largest in magnitude):

```
x_0 = random vector
for k = 0, 1, 2, ...:
    y_{k+1} = A x_k
    x_{k+1} = y_{k+1} / ||y_{k+1}||
    lambda_k = x_k^T A x_k  (Rayleigh quotient)
```

**Convergence:** $\lambda_k \to \lambda_1$ at rate $|\lambda_2 / \lambda_1|^k$. Slow if $|\lambda_2| \approx |\lambda_1|$.

**Inverse power iteration:** Apply power iteration to $A^{-1}$ (solve $Ax = y$ at each step). Converges to the eigenvalue closest to 0 (smallest in magnitude). With a shift $\mu$: apply to $(A - \mu I)^{-1}$ to find the eigenvalue closest to $\mu$.

**For AI:** Power iteration is the foundation of the PageRank algorithm. It is also used in computing the leading singular values of weight matrices (for spectral normalization in GANs) and in the Lanczos algorithm for computing the extremal eigenvalues of the Hessian in loss landscape analysis.

### 7.2 QR Algorithm

The **QR algorithm** (Francis, 1961) computes all eigenvalues of a matrix:

**Basic QR iteration:**
```
A_0 = A
for k = 0, 1, 2, ...:
    A_k = Q_k R_k   (QR factorization)
    A_{k+1} = R_k Q_k  (Note: reversed order)
```

The sequence $A_0, A_1, A_2, \ldots$ converges to a triangular (or quasi-triangular for real matrices) form, with the diagonal entries converging to the eigenvalues.

**Practical improvements:**
- **Hessenberg reduction:** First reduce $A$ to upper Hessenberg form $H = Q^\top A Q$ (so $h_{ij} = 0$ for $i > j+1$). QR iteration applied to $H$ costs $O(n^2)$ per step instead of $O(n^3)$.
- **Shifts:** Use shifts $\sigma_k$ to accelerate convergence: factor $A_k - \sigma_k I = Q_k R_k$, then $A_{k+1} = R_k Q_k + \sigma_k I$. The Wilkinson shift is standard.
- **Deflation:** When a subdiagonal entry becomes small, the problem decouples into two smaller problems.

**Convergence:** With shifts, the QR algorithm typically converges cubically - finding all $n$ eigenvalues in $O(n^3)$ total flops.

### 7.3 Sensitivity of Eigenvalues

**Theorem (Bauer-Fike):** For a diagonalizable matrix $A = X\Lambda X^{-1}$ with distinct eigenvalues, and perturbation $E$:
$$\min_j |\lambda_i(A+E) - \lambda_j(A)| \leq \kappa(X) \|E\|$$

The condition number of the eigenvector matrix $X$ determines eigenvalue sensitivity.

**For symmetric matrices:** $X$ is orthogonal, so $\kappa(X) = 1$ - eigenvalues of symmetric matrices are perfectly conditioned with respect to symmetric perturbations (Weyl's theorem).

**For non-symmetric matrices:** Ill-conditioned eigenvectors lead to ill-conditioned eigenvalues. Example: Jordan blocks have $\kappa(X) = \infty$ (or rather, exponentially large for near-Jordan matrices).

**Practical implication:** Computing eigenvalues of general non-symmetric matrices requires care. For symmetric matrices (e.g., Gram matrices, covariance matrices, graph Laplacians), eigenvalue computation is well-conditioned.

---

## 8. Applications in Machine Learning

### 8.1 Solving the Normal Equations in Ridge Regression

**Ridge regression:** $\min_x \|Ax - b\|_2^2 + \lambda \|x\|_2^2$

Normal equations: $(A^\top A + \lambda I) x = A^\top b$

Adding $\lambda I$ is Tikhonov regularization - it improves conditioning:
$$\kappa(A^\top A + \lambda I) = \frac{\sigma_1^2 + \lambda}{\sigma_n^2 + \lambda} \leq \frac{\sigma_1^2}{\lambda}$$

For the "kernel trick" (wide matrices, $m \ll n$): use the equivalent formulation $(AA^\top + \lambda I) \alpha = b$ where $\alpha \in \mathbb{R}^m$, then $x = A^\top \alpha$. The system is $m \times m$ instead of $n \times n$.

**Numerically stable implementation:** Do NOT form $A^\top A$. Instead:
1. Factor $A = QR$ (Householder, cost: $O(mn^2)$)
2. Solve $(R^\top R + \lambda I) x = R^\top Q^\top b$ - a smaller well-conditioned system

Or use the SVD: $A = U\Sigma V^\top$, then $x = V (\Sigma^2 + \lambda I)^{-1} \Sigma U^\top b$.

### 8.2 Batched Matrix Operations in Transformers

The attention mechanism computes:
$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

In practice, this is computed over multiple heads and batch elements simultaneously - **batched matrix multiplication** (BGEMM):
- $Q, K, V \in \mathbb{R}^{B \times H \times N \times d_k}$ (batch, heads, sequence length, head dim)
- $QK^\top$: batched GEMM, cost $O(BHN^2 d_k)$, executed as one kernel call on GPU

**Numerical considerations:**
- The $1/\sqrt{d_k}$ scaling is numerically motivated (keeps $QK^\top$ elements near unit scale for well-behaved softmax)
- FlashAttention (Dao et al., 2022) avoids materializing the $N \times N$ attention matrix by using a **tiled algorithm** that keeps blocks in SRAM
- Flash attention uses online softmax with iterative rescaling - directly applying the backward stability ideas from numerical linear algebra

### 8.3 Iterative Refinement in Neural Solvers

**Iterative refinement** is a classical technique to improve a solution $\hat{x}$ to $Ax = b$:

1. Compute residual $r = b - A\hat{x}$ (in higher precision if available)
2. Solve $A\delta x = r$ approximately (using the existing factorization)
3. Update: $\hat{x} \leftarrow \hat{x} + \delta x$
4. Repeat until convergence

**Mixed-precision iterative refinement:** Factor $A$ in fp16 (fast), compute residual in fp64, perform refinement steps. This achieves fp64 accuracy at near fp16 cost. NVIDIA's cuSOLVER implements this for dense systems.

**Neural solvers** (e.g., solving PDEs) often use this pattern:
1. Neural network predicts approximate solution $\hat{x}^{(0)}$
2. Iterative refinement corrects it using the true equations

### 8.4 Condition Numbers in Gradient Descent

The condition number of the **Hessian** $H = \nabla^2 f$ at a minimum determines gradient descent convergence:

For a quadratic $f(x) = \frac{1}{2}x^\top H x - b^\top x$:
- Gradient descent with step $\eta$: $x_{k+1} = x_k - \eta \nabla f(x_k)$
- Optimal step: $\eta^* = 2/(\lambda_{\max} + \lambda_{\min})$
- Convergence rate: $\left(\frac{\kappa - 1}{\kappa + 1}\right)^k$

For $\kappa = 10^4$: convergence factor $\approx 0.9998$ per step - need $\sim 10^4$ steps for $e^{-1}$ reduction.

For $\kappa = 1$: convergence in 1 step (perfect conditioning).

**Adam addresses conditioning:** By scaling each coordinate by $1/\sqrt{v_t}$ (approximate diagonal of Hessian), Adam reduces the effective condition number to closer to 1, enabling faster convergence on ill-conditioned objectives.

**Gradient clipping and conditioning:** When $\kappa(H)$ is large, gradient norms can vary wildly across coordinates. Gradient clipping (by global norm or per-parameter) is a heuristic to prevent large steps along ill-conditioned directions.

---

## 9. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
|---|---------|---------------|-----|
| 1 | Using `np.linalg.inv(A) @ b` to solve $Ax = b$ | Forms $A^{-1}$ explicitly - $O(n^3)$ cost, numerically inferior | Use `np.linalg.solve(A, b)` |
| 2 | Forming $A^\top A$ for least squares | Squares the condition number: $\kappa(A^\top A) = \kappa(A)^2$ | Use `np.linalg.lstsq(A, b)` which uses QR |
| 3 | Assuming Gaussian elimination without pivoting is stable | Without pivoting, tiny pivots amplify rounding errors exponentially | Always use GEPP (`scipy.linalg.lu` with permutation) |
| 4 | Using `np.linalg.cond(A)` as a cheap operation | `cond` computes SVD internally - $O(n^3)$ cost | Use it for debugging; avoid in inner loops |
| 5 | Solving many right-hand sides by calling `solve` repeatedly | Each call re-factors $A$ | Use `lu_factor` + `lu_solve` to reuse factorization |
| 6 | Ignoring sparsity | Dense operations on sparse matrices waste memory and computation | Use `scipy.sparse` matrices and sparse solvers |
| 7 | Expecting CG to converge in few iterations without a preconditioner | CG rate is $O(\sqrt{\kappa})$ - slow for ill-conditioned systems | Add a diagonal or IC preconditioner |
| 8 | Using Classical Gram-Schmidt for QR | Loss of orthogonality is $O(\kappa(A)^2 \varepsilon)$ | Use Modified Gram-Schmidt or Householder |
| 9 | Forming $Q$ explicitly in Householder QR | Costs $O(n^3)$ extra; usually unnecessary | Apply $Q^\top b$ implicitly via stored reflectors |
| 10 | Interpreting `numpy.linalg.matrix_rank` for nearly singular matrices | Uses a singular value threshold - results are threshold-dependent | Always check singular values directly; use regularization |
| 11 | Assuming positive definite matrices are safe to Cholesky-factorize without checks | Near-zero eigenvalues become negative in floating point | Add diagonal loading: `A + eps * np.eye(n)` |
| 12 | Using eigenvalue decomposition for symmetric matrices via `eig` | `eig` uses general (non-symmetric) algorithm; slower and less stable | Always use `eigh` for symmetric/Hermitian matrices |

---

## 10. Exercises

### Exercise 1 * - LU Factorization with Partial Pivoting

Implement Gaussian elimination with partial pivoting from scratch.

(a) Write `lu_pivot(A)` returning `(P, L, U)` such that `P @ A == L @ U`.

(b) Implement forward/back substitution to solve `L z = P b` and `U x = z`.

(c) Verify on a $5 \times 5$ random matrix: compare `x` to `numpy.linalg.solve(A, b)`.

(d) Demonstrate that without pivoting (always choosing the diagonal as pivot), the solution fails for $A = \begin{pmatrix} 1 & 2 \\ 2 & 4.0001 \end{pmatrix}$ with $\varepsilon = 10^{-15}$ perturbation.

### Exercise 2 * - Stability Comparison: Normal Equations vs QR

Compare the least squares solvers on an ill-conditioned matrix.

(a) Generate $A \in \mathbb{R}^{50 \times 10}$ with singular values $[1, 0.1, 0.01, \ldots, 10^{-9}]$.

(b) Compute the least squares solution via Normal equations (form $A^\top A$, solve) and via QR (`numpy.linalg.lstsq`). Report $\kappa(A)$, $\kappa(A^\top A)$, and the relative error of each method.

(c) Plot relative error vs condition number (vary the decay rate of singular values).

### Exercise 3 ** - Implementing Conjugate Gradient

Implement the CG algorithm for SPD systems.

(a) Write `conjugate_gradient(A, b, x0, tol, max_iter)` returning `(x, residuals)`.

(b) Test on a $100 \times 100$ SPD system. Plot the residual $\|r_k\| / \|r_0\|$ vs iteration.

(c) Verify the convergence bound: $\|e_k\|_A / \|e_0\|_A \leq 2 \left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^k$.

(d) Add diagonal preconditioning: solve $(D^{-1/2} A D^{-1/2}) y = D^{-1/2} b$ where $D = \text{diag}(A)$. Compare convergence with and without preconditioning.

### Exercise 4 ** - Iterative Refinement

Implement mixed-precision iterative refinement.

(a) Factor $A$ in fp32 using `scipy.linalg.lu`.

(b) Solve $Ax = b$ using the fp32 factorization. Compute the residual $r = b - Ax$ in fp64.

(c) Solve $A \delta x = r$ using the fp32 factorization. Update $x \leftarrow x + \delta x$.

(d) Report the forward error after each refinement step for a moderately ill-conditioned system ($\kappa \approx 10^4$).

### Exercise 5 ** - Power Iteration and PageRank

Implement power iteration to compute the dominant eigenvector.

(a) Write `power_iteration(A, x0, tol)` converging to the dominant eigenvector.

(b) Apply to the PageRank problem: for a random directed graph with $n = 20$ nodes, build the transition matrix $P$ and use power iteration to find the stationary distribution.

(c) Compare convergence rate to the predicted $|\lambda_2 / \lambda_1|^k$. Compute $\lambda_2$ via `numpy.linalg.eig`.

### Exercise 6 *** - Condition Number and Gradient Descent Convergence

Demonstrate the condition number's role in optimization convergence.

(a) Generate a quadratic objective $f(x) = \frac{1}{2} x^\top H x - b^\top x$ where $H$ has eigenvalues spread over $[1, \kappa]$ for varying $\kappa \in \{1, 10, 100, 1000, 10000\}$.

(b) Run gradient descent with optimal fixed step size $2/(\lambda_{\max} + \lambda_{\min})$. Record $\|x_k - x^*\|$ at each step.

(c) Verify the bound $\|x_k - x^*\| \leq \left(\frac{\kappa-1}{\kappa+1}\right)^k \|x_0 - x^*\|$.

(d) Run CG on the same problems. Compare the number of steps to $10^{-6}$ accuracy.

### Exercise 7 ** - Modified Gram-Schmidt QR

Implement Modified Gram-Schmidt and analyze loss of orthogonality.

(a) Implement `mgs_qr(A)` returning `(Q, R)` via MGS.

(b) Compare orthogonality loss $\|I - Q^\top Q\|_F$ for MGS vs Classical Gram-Schmidt vs Householder on an ill-conditioned matrix.

(c) Plot orthogonality loss vs $\kappa(A)$ for all three methods.

### Exercise 8 *** - Sparse Systems and Preconditioning

Work with sparse linear systems from a 1D PDE discretization.

(a) Build the tridiagonal matrix $T_n$ from discretizing $-u'' = f$ on $[0,1]$ with $n = 1000$ interior points: $T_{ii} = 2/h^2$, $T_{i,i\pm 1} = -1/h^2$ where $h = 1/(n+1)$.

(b) Solve $T_n x = b$ using: (i) dense `numpy.linalg.solve`, (ii) sparse CG without preconditioner, (iii) sparse CG with diagonal preconditioner.

(c) Report time, iteration count, and final residual for each method.

(d) Compute $\kappa(T_n)$ and verify that CG iterations scale as $O(\sqrt{\kappa})$.

---

## 11. Why This Matters for AI (2026 Perspective)

| Aspect | Impact |
|--------|--------|
| **BLAS/cuBLAS operations** | Every forward pass in a neural network is a sequence of matrix multiplications. The stability of BLAS routines directly impacts training |
| **Adam as diagonal preconditioner** | Adam's $1/\sqrt{v_t}$ scaling implements diagonal preconditioning for the implicit linear system of gradient descent |
| **FlashAttention** | Uses numerically stable tiled attention computation - directly applies backward stability principles to avoid materializing the full attention matrix |
| **Mixed-precision training** | fp16 factorizations + fp32 refinement mirrors mixed-precision iterative refinement from numerical linear algebra |
| **K-FAC optimization** | Approximate second-order method using block-diagonal Kronecker-factored Hessian; requires stable Cholesky factorization of each block |
| **Spectral normalization** | GAN training stability via normalizing weight matrices by $\sigma_{\max}(W)$ - computed via power iteration |
| **Gradient clipping** | Addresses ill-conditioning of the loss Hessian; analogous to limiting the condition number of each step |
| **Neural ODEs** | Require solving implicit linear systems at each time step; use GMRES or CG for efficiency |
| **LoRA/low-rank adaptation** | Exploits low-rank structure: instead of updating $W \in \mathbb{R}^{m \times n}$, update $W + AB$ with $A \in \mathbb{R}^{m \times r}$, $B \in \mathbb{R}^{r \times n}$, $r \ll m, n$ |
| **Eigenvalue analysis of trained networks** | Studying the Hessian spectrum at the end of training; QR algorithm powers these computations |

---

## 12. Conceptual Bridge

This section sits at the intersection of pure mathematical theory and practical computational science. You came in knowing that $Ax = b$ has a unique solution when $A$ is invertible - this section showed you that *how* you solve it matters as much as *whether* a solution exists.

**What you learned:** The key insight is **backward stability** - the right standard for numerical algorithms. Backward stable algorithms solve a nearby problem exactly; the condition number then determines how much forward error this implies. LU with partial pivoting, Householder QR, and Cholesky are all backward stable. Classical Gram-Schmidt, normal equations, and LU without pivoting are not.

**Looking backward:** This section builds directly on [Section01 Floating-Point Arithmetic](../01-Floating-Point-Arithmetic/notes.md) (machine epsilon, rounding model, condition numbers) and [Section03-Advanced-Linear-Algebra/08-Matrix-Decompositions](../../03-Advanced-Linear-Algebra/08-Matrix-Decompositions/notes.md) (mathematical existence and properties of LU, QR, Cholesky). If the theoretical content of factorizations is unfamiliar, revisit those sections.

**Looking forward:** The iterative methods here (CG, GMRES) reappear in [Section03 Numerical Optimization](../03-Numerical-Optimization/notes.md) as tools for solving the linear systems that arise in Newton's method and trust region methods. The condition number analysis connects to convergence theory for gradient descent, which is the core of training deep neural networks.

```
CHAPTER Section10 POSITION: NUMERICAL METHODS
========================================================================

  Section01 Floating-Point Arithmetic   <-  You are here (prerequisites)
    +--> Section02 Numerical Linear Algebra  <-  THIS SECTION
              +--> Section03 Numerical Optimization
                        +--> Section04 Interpolation & Approximation
                                  +--> Section05 Numerical Integration

  BRIDGES FROM THIS SECTION:
    +-------------------------------------------------------------+
    |  LU/QR stability          -> Section08 Optimization (line search)  |
    |  CG method                -> Section03 Numerical Optimization       |
    |  Preconditioning theory   -> Section08 Optimization (Adam analysis) |
    |  Eigenvalue algorithms    -> Section11 Graph Theory (spectral)      |
    |  Matrix condition numbers -> Section03 Advanced LA (decompositions) |
    +-------------------------------------------------------------+

========================================================================
```

---

## Appendix A: Detailed LU Factorization Algorithm

Here is a complete, step-by-step implementation of GEPP suitable for educational purposes:

```python
import numpy as np

def lu_partial_pivot(A):
    """
    Gaussian elimination with partial pivoting.
    Returns P, L, U such that P @ A = L @ U.
    """
    n = A.shape[0]
    A = A.astype(np.float64).copy()
    P = np.eye(n)
    L = np.zeros((n, n))

    for k in range(n - 1):
        # Find pivot: largest element in column k, rows k..n-1
        pivot_row = k + np.argmax(np.abs(A[k:, k]))

        # Swap rows k and pivot_row in A, L, P
        if pivot_row != k:
            A[[k, pivot_row], :]   = A[[pivot_row, k], :]
            P[[k, pivot_row], :]   = P[[pivot_row, k], :]
            L[[k, pivot_row], :k]  = L[[pivot_row, k], :k]

        # Eliminate below the diagonal
        for i in range(k + 1, n):
            L[i, k] = A[i, k] / A[k, k]
            A[i, k:] -= L[i, k] * A[k, k:]

    L += np.eye(n)  # Add diagonal ones
    U = np.triu(A)  # Upper triangular part
    return P, L, U

def forward_sub(L, b):
    """Solve Lx = b where L is lower triangular with unit diagonal."""
    n = len(b)
    x = np.zeros(n)
    for i in range(n):
        x[i] = b[i] - L[i, :i] @ x[:i]
    return x

def back_sub(U, b):
    """Solve Ux = b where U is upper triangular."""
    n = len(b)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]
    return x

def solve_lu(A, b):
    """Solve Ax = b via LU with partial pivoting."""
    P, L, U = lu_partial_pivot(A)
    pb = P @ b          # Apply permutation
    z  = forward_sub(L, pb)  # Solve Lz = Pb
    x  = back_sub(U, z)      # Solve Ux = z
    return x
```

**Verification:**
```python
np.random.seed(42)
A = np.random.randn(5, 5)
b = np.random.randn(5)
x = solve_lu(A, b)
print(f"Residual: {np.linalg.norm(A @ x - b):.2e}")  # Should be ~1e-15
```

---

## Appendix B: Householder Reflector Implementation

A complete Householder QR from scratch:

```python
def householder_qr(A):
    """
    QR factorization via Householder reflectors.
    Returns Q (orthogonal), R (upper triangular).
    """
    m, n = A.shape
    A = A.copy().astype(np.float64)
    Q = np.eye(m)

    for k in range(min(m, n)):
        # Extract column k below the diagonal
        x = A[k:, k].copy()

        # Build Householder reflector to zero out x[1:]
        e1 = np.zeros_like(x)
        e1[0] = 1.0
        sign = 1.0 if x[0] >= 0 else -1.0
        v = x + sign * np.linalg.norm(x) * e1
        v = v / np.linalg.norm(v)

        # Apply: A[k:, k:] -= 2 v (v^T A[k:, k:])
        A[k:, k:] -= 2.0 * np.outer(v, v @ A[k:, k:])

        # Accumulate Q: Q[k:, :] -= 2 v (v^T Q[k:, :])
        Q[k:, :] -= 2.0 * np.outer(v, v @ Q[k:, :])

    R = np.triu(A)
    return Q.T, R  # Q^T was accumulated; return Q = (Q^T)^T

def solve_qr(A, b):
    """Solve least squares min ||Ax - b|| via QR."""
    Q, R = householder_qr(A)
    n = R.shape[1]
    Qt_b = Q.T @ b
    # Solve R[:n, :] x = Q.T b[:n] via back substitution
    x = back_sub(R[:n, :n], Qt_b[:n])
    return x
```

---

## Appendix C: Conjugate Gradient - Detailed Derivation

The CG method can be derived from several perspectives. The most illuminating is the **energy minimization** view.

**Setup:** We want to minimize the quadratic energy function:
$$f(x) = \frac{1}{2} x^\top A x - b^\top x$$

where $A$ is SPD. The minimum satisfies $\nabla f(x) = Ax - b = 0$, i.e., $Ax = b$.

**Steepest descent** would use the gradient $-\nabla f = r = b - Ax$:
$$x_{k+1} = x_k + \alpha_k r_k, \quad \alpha_k = \frac{r_k^\top r_k}{r_k^\top A r_k}$$

But steepest descent exhibits **zig-zagging** - successive search directions are perpendicular, leading to slow convergence with $O(\kappa)$ iterations.

**Key idea of CG:** Choose search directions that are **$A$-conjugate** (mutually orthogonal under the $A$-inner product $\langle u, v \rangle_A = u^\top A v$):
$$p_i^\top A p_j = 0 \quad \text{for } i \neq j$$

**Why this works:** In a space of dimension $n$ with $n$ mutually conjugate directions, the minimum can be found in exactly $n$ steps (each step minimizes exactly along one conjugate direction, and these directions span $\mathbb{R}^n$).

**The CG recurrence** generates $A$-conjugate directions from the residuals $r_k = b - Ax_k$ using a three-term recurrence, which is the Lanczos relation for symmetric matrices.

**Finite precision behavior:** In exact arithmetic, CG terminates in $\leq n$ steps. In floating point:
- Orthogonality of residuals is lost after many steps
- The algorithm doesn't terminate in $n$ steps for large $n$
- The computed residuals satisfy $\|r_k\|/\|b\| \leq c \kappa \varepsilon_{\text{mach}}$ as a floor

---

## Appendix D: Growth Factor Examples

The theoretical worst case for the growth factor in GEPP is $\rho = 2^{n-1}$, achieved by:

$$A = \begin{pmatrix}
1 & 0 & 0 & \cdots & 0 & 1 \\
-1 & 1 & 0 & \cdots & 0 & 1 \\
-1 & -1 & 1 & \cdots & 0 & 1 \\
\vdots & & & \ddots & & \vdots \\
-1 & -1 & -1 & \cdots & 1 & 1 \\
-1 & -1 & -1 & \cdots & -1 & 1
\end{pmatrix}$$

After GEPP on this matrix, the $(n,n)$ entry of $U$ equals $2^{n-1}$. For $n = 53$, this would exceed fp64 range - an exponential growth. However, this matrix almost never arises in practice.

**Empirical growth factors:**
| Matrix type | Typical $\rho$ | Notes |
|-------------|----------------|-------|
| Random $n \times n$ | $O(n^{2/3})$ | Empirical average |
| Tridiagonal | $\leq 2$ | Theoretical bound |
| SPD matrices | $= 1$ (Cholesky) | No growth possible |
| Worst case (Wilkinson) | $2^{n-1}$ | Theoretically possible, never seen |

---

## Appendix E: Scipy/NumPy Linear Algebra API Reference

### Recommended routines:

```python
import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla
from scipy.sparse.linalg import cg, spsolve

# === Direct Methods ===

# Solve Ax = b (dense)
x = nla.solve(A, b)           # Uses LAPACK dgesv (LU + partial pivoting)
x = sla.solve(A, b)           # Similar; slightly more options

# Least squares
x, _, _, _ = nla.lstsq(A, b)  # Householder QR-based

# LU factorization (reuse for multiple RHS)
lu, piv = sla.lu_factor(A)    # Factor once
x = sla.lu_solve((lu, piv), b)  # Solve for each b

# QR factorization
Q, R = nla.qr(A)              # Householder QR
Q, R = sla.qr(A)              # Full or economic form

# Cholesky (SPD only)
L = nla.cholesky(A)           # Lower triangular factor
c, low = sla.cho_factor(A)    # Factor for reuse
x = sla.cho_solve((c, low), b)

# === Iterative Methods ===

# Conjugate gradient (SPD)
from scipy.sparse.linalg import cg
x, info = cg(A, b, tol=1e-10, maxiter=1000)

# GMRES (general)
from scipy.sparse.linalg import gmres
x, info = gmres(A, b, tol=1e-10, maxiter=1000)

# === Condition Number ===

kappa = nla.cond(A)           # Via SVD - O(n^3)
kappa_1 = nla.cond(A, 1)      # 1-norm condition (faster via LAPACK)

# Estimate condition number cheaply (after LU):
# scipy.linalg.lu_factor + scipy.linalg.rcond (reciprocal condition number)

# === Eigenvalues ===

eigvals = nla.eigvalsh(A)     # Symmetric/Hermitian only (faster, stable)
eigvals, eigvecs = nla.eigh(A)
eigvals = nla.eigvals(A)      # General (use only if A is non-symmetric)

# === SVD ===

U, s, Vt = nla.svd(A)        # Full SVD
U, s, Vt = nla.svd(A, full_matrices=False)  # Economy SVD (m >= n)
```

---

## Appendix F: Error Analysis of Triangular Solves

When solving $Lx = b$ (triangular system), the computed solution $\hat{x}$ satisfies:

$$(L + \delta L)\hat{x} = b, \quad |\delta L_{ij}| \leq n \varepsilon_{\text{mach}} |L_{ij}|$$

This is backward stable with backward error $O(n \varepsilon_{\text{mach}})$.

**Forward error:** The condition number of a lower triangular system is:
$$\kappa(L) = \|L\|_\infty \|L^{-1}\|_\infty$$

For well-conditioned triangular systems (as produced by LU with partial pivoting), $\kappa(L) = O(n)$ - the forward error is $O(n^2 \varepsilon_{\text{mach}})$.

**Practical implication:** Even for ill-conditioned $A$, if you solve via LU with partial pivoting, the triangular solve steps are individually backward stable. The overall error comes from the conditioning of $A$, not from numerical instability of the algorithm.

---

## Appendix G: Connection to PyTorch/JAX Linear Algebra

Modern deep learning frameworks expose efficient linear algebra:

```python
import torch

# === Direct Solve ===
A = torch.randn(100, 100, dtype=torch.float64)
b = torch.randn(100, 1, dtype=torch.float64)
# torch.linalg.solve is backward stable (uses cuBLAS LAPACK routines)
x = torch.linalg.solve(A, b)

# === Batched Operations ===
# Solve 32 systems simultaneously (B x N x N)
A_batch = torch.randn(32, 100, 100)
b_batch = torch.randn(32, 100, 1)
x_batch = torch.linalg.solve(A_batch, b_batch)  # Fully batched, GPU-accelerated

# === Eigenvalues ===
A_sym = A @ A.T  # Make symmetric
eigvals, eigvecs = torch.linalg.eigh(A_sym)  # For symmetric: use eigh

# === SVD ===
U, S, Vh = torch.linalg.svd(A)
U, S, Vh = torch.linalg.svd(A, full_matrices=False)  # Economy SVD

# === Condition Number ===
kappa = torch.linalg.cond(A)

# === QR ===
Q, R = torch.linalg.qr(A)  # Householder QR
```

**GPU execution:** All torch.linalg operations execute on GPU if tensors are on GPU. Batched operations are especially efficient: `torch.linalg.solve` with batch dimension uses cuBLAS batched GETRS.

**Gradient computation:** `torch.linalg` operations support automatic differentiation. The gradient of `torch.linalg.solve(A, b)` with respect to `A` and `b` is computed via the adjoint method - no explicit matrix inverse is formed.

---

## Appendix H: Krylov Subspace Methods - Unified View

The **Krylov subspace** $\mathcal{K}_k(A, r_0) = \text{span}\{r_0, Ar_0, A^2 r_0, \ldots, A^{k-1}r_0\}$ is the fundamental concept underlying all modern iterative methods.

**Different optimality conditions give different methods:**

| Method | System | Optimality condition |
|--------|--------|---------------------|
| CG | $A$ SPD | Minimize $\|e_k\|_A$ over $\mathcal{K}_k$ |
| MINRES | $A$ symmetric | Minimize $\|r_k\|_2$ over $\mathcal{K}_k$ |
| GMRES | $A$ general | Minimize $\|r_k\|_2$ over $\mathcal{K}_k$ |
| FOM | $A$ general | Galerkin condition $r_k \perp \mathcal{K}_k$ |
| BiCG | $A$ general | Biorthogonalization (cheaper, less stable) |
| BiCGStab | $A$ general | BiCG with stability fix |

**Why Krylov subspaces?** Polynomials in $A$ acting on $r_0$ give the best possible approximations from the subspace spanned by $\{r_0, Ar_0, A^2 r_0, \ldots\}$. The optimal polynomial $p_k$ satisfying $x_k = x_0 + p_k(A) r_0$ minimizes the error in the appropriate norm.

**Convergence depends on the spectrum:** If $A$ has only $m < n$ distinct eigenvalues, the Krylov method converges in exactly $m$ steps in exact arithmetic. Pre-clustering eigenvalues (via preconditioning) accelerates convergence.

---

## Appendix I: Sparse Matrix Storage Formats

For large sparse systems, efficient storage and matrix-vector products are critical:

| Format | Storage | Access | Best for |
|--------|---------|--------|----------|
| Dense | $O(n^2)$ | $O(1)$ | Small/dense matrices |
| COO (coordinates) | $O(\text{nnz})$ | $O(\text{nnz})$ scan | Construction |
| CSR (compressed sparse row) | $O(\text{nnz})$ | $O(\text{nnz per row})$ | Row operations, matvec |
| CSC (compressed sparse column) | $O(\text{nnz})$ | $O(\text{nnz per col})$ | Column operations |
| BSR (block sparse row) | $O(\text{nnz})$ | Block-level | Dense block structure |
| DIA (diagonal) | $O(n \times d)$ | $O(1)$ | Banded/diagonal matrices |

**Sparse matrix-vector product (SpMV):** The core operation in iterative methods. For CSR:
```
y = zeros(n)
for i in range(n):
    for j_idx in range(row_ptr[i], row_ptr[i+1]):
        y[i] += val[j_idx] * x[col_idx[j_idx]]
```
This is $O(\text{nnz})$ per multiply - for matrices with $\text{nnz} \ll n^2$, dramatically cheaper than dense matvec.

**For AI:** Sparse attention in transformers uses sparse matrix representations. The attention mask is stored as a sparse pattern; the attention computation becomes a sparse matrix-matrix product (SpMM).

---

## Appendix J: Worked Example - Condition Number Effect on Least Squares

Consider fitting a polynomial through noisy data. Using the Vandermonde matrix:

$$V = \begin{pmatrix} 1 & x_1 & x_1^2 & \cdots & x_1^{n-1} \\ 1 & x_2 & x_2^2 & \cdots & x_2^{n-1} \\ \vdots & & & & \vdots \end{pmatrix}$$

For $n = 10$ with $x_i$ equally spaced in $[0, 1]$:

```python
n = 10
x = np.linspace(0, 1, 20)
V = np.vander(x, n, increasing=True)

# Condition numbers
print(f'kappa(V)   = {np.linalg.cond(V):.2e}')      # ~1e12
print(f'kappa(V^TV) = {np.linalg.cond(V.T @ V):.2e}') # ~1e24!

# Fit via Normal equations vs QR
y = np.sin(2 * np.pi * x) + 0.01 * np.random.randn(20)
c_normal = np.linalg.solve(V.T @ V, V.T @ y)  # Unstable
c_qr, _, _, _ = np.linalg.lstsq(V, y)          # Stable
```

**Result:** For high-degree polynomials, the Vandermonde matrix is extremely ill-conditioned. The QR solution via `lstsq` will be far more accurate than the Normal equations approach.

**Fix:** Use Chebyshev nodes $x_i = \cos((2i-1)\pi/(2n))$ instead of equally-spaced nodes. The resulting Vandermonde-like matrix has dramatically lower condition number (see Section04 Interpolation and Approximation).

---

## Appendix K: Quick Reference - Algorithm Complexity and Stability

```
NUMERICAL LINEAR ALGEBRA: ALGORITHM COMPARISON
========================================================================

  DIRECT METHODS (Dense A, n x n):

  Algorithm         Flops      Storage   Stable?   Use when
  -------------------------------------------------------------
  LU (no pivot)     2n^3/3      n^2        NO        Only if A is SPD
  LU (partial)      2n^3/3      n^2        YES*      General square A
  QR (Householder)  4m n^2/3    mn        YES       Least squares
  QR (MGS)          2mn^2       mn        Partial   Streaming/iterative
  Cholesky          n^3/3       n^2/2      YES       SPD matrices
  SVD               ~11n^3      mn        YES       Rank-def, analysis

  * Stable in practice; theoretical worst case 2^(n-1) growth

  ITERATIVE METHODS (Sparse A, nnz nonzeros):

  Algorithm         Cost/iter  Converges?  Precond?  Use when
  -------------------------------------------------------------
  Jacobi            O(nnz)     If D-dom.   N/A       Parallel arch
  Gauss-Seidel      O(nnz)     If D-dom.   N/A       Sequential
  CG                O(nnz)     SPD only    YES       SPD sparse
  MINRES            O(nnz)     Sym. only   YES       Sym. indef.
  GMRES             O(k\\cdotnnz)   Always      YES       Non-symmetric
  BiCGStab          O(nnz)     Often       YES       Non-sym, cheaper

========================================================================
```

---

## Appendix L: Numerical Linear Algebra in the LLM Stack

Modern large language models rely on numerical linear algebra at every level:

**1. Attention computation (FlashAttention-2):**
- Tiled matrix multiplication with online softmax rescaling
- Block size chosen to fit in SRAM (80KB L1 cache on A100)
- Backward pass uses recomputation (saves memory at cost of 1.33x extra flops)
- Key numerical property: each tile computation is backward stable; the tiling doesn't affect correctness

**2. Optimizer state (AdaFactor, CAME):**
- AdaFactor approximates the Adam second moment as a rank-1 outer product: $V \approx r_t s_t^\top / \mathbf{1}^\top r_t$
- This is a rank-1 approximation of the true second moment matrix
- CAME introduces curvature information via a sparse second-order approximation

**3. Quantization (GPTQ, AWQ, SqueezeLLM):**
- Post-training weight quantization formulated as a matrix approximation problem
- GPTQ: use Cholesky decomposition of the Hessian to determine optimal quantization order
- AWQ: identify "salient weights" via gradient sensitivity (related to condition number analysis)

**4. Low-rank adaptation (LoRA, DoRA):**
- $W' = W + \frac{\alpha}{r} BA$ where $B \in \mathbb{R}^{m \times r}$, $A \in \mathbb{R}^{r \times n}$
- Initialization: $B = 0$ (so $W' = W$ initially), $A \sim \mathcal{N}(0, 1/r)$
- Numerically stable: rank-$r$ updates don't affect the conditioning of $W$
- DoRA: $W' = \|W\|_F \cdot \frac{W + BA}{\|W + BA\|_F}$ - normalizes by column norms (related to spectral normalization)

---

*[<- Back to Chapter 10](../README.md) | [Next: Numerical Optimization ->](../03-Numerical-Optimization/notes.md)*

---

## Appendix M: Extended Theory - Structured Linear Systems

### M.1 Toeplitz Systems

A **Toeplitz matrix** has constant diagonals: $A_{ij} = a_{i-j}$. This structure arises in:
- Signal processing (convolution)
- Time-series analysis (autocovariance)
- Certain PDE discretizations

**Levinson-Durbin algorithm:** Solves $n \times n$ Toeplitz systems in $O(n^2)$ flops (vs $O(n^3)$ for general LU). Uses the structure to build the solution recursively.

**For AI:** Convolutional layers in CNNs implement Toeplitz-structured matrix-vector products. The connection between convolution and Toeplitz structure explains why FFT-based convolution ($O(n \log n)$) is faster than direct convolution ($O(n^2)$).

### M.2 Kronecker Product Systems

For systems of the form $(A \otimes B) x = b$ where $\otimes$ denotes the Kronecker product:

$$\text{vec}(AXB^\top) = (B \otimes A) \text{vec}(X)$$

The system can be solved as a **matrix equation**: $AXB^\top = C$ with solution $X = A^{-1} C B^{-\top}$. This requires factoring two smaller matrices of sizes $m \times m$ and $n \times n$ rather than one $mn \times mn$ matrix.

**For AI:** Kronecker-factored curvature (K-FAC) approximates the Fisher information matrix as a Kronecker product: $F \approx A \otimes B$ where $A$ and $B$ are layer-specific matrices. This enables efficient second-order optimization without forming the full $d^2 \times d^2$ Fisher matrix.

### M.3 Circulant Systems

A **circulant matrix** has each row as a cyclic shift of the previous row:
$$C = \text{circ}(c_0, c_1, \ldots, c_{n-1})$$

**Key property:** Every circulant matrix is diagonalized by the DFT matrix:
$$C = F^* \text{diag}(\hat{c}) F$$

where $F$ is the DFT matrix and $\hat{c} = Fc$ is the DFT of the first row. This means:
- Eigenvalues of $C$ are the DFT coefficients of $c$
- Solving $Cx = b$ costs $O(n \log n)$ via FFT
- Matrix-vector products $Cv$ cost $O(n \log n)$

**For AI:** Circular convolutions arise in certain attention mechanisms (e.g., some positional encodings). The connection between circulant matrices and the DFT explains why FFT-based convolution is exact for circular boundary conditions.

---

## Appendix N: Numerical Rank and Truncated SVD

### N.1 Numerical Rank

The **numerical rank** of a matrix is the number of singular values above a threshold $\tau$:
$$\text{rank}_\tau(A) = |\{i : \sigma_i > \tau\}|$$

The choice of $\tau$ depends on context:
- **Machine precision threshold:** $\tau = \varepsilon_{\text{mach}} \sigma_1 \max(m, n)$
- **Data-dependent threshold:** $\tau = \sigma_k$ where $k$ is chosen by a "gap" in the singular value spectrum
- **Relative threshold:** $\tau = \varepsilon \cdot \sigma_1$ for user-specified $\varepsilon$

**For AI:** Low-rank approximations underlie:
- **LoRA:** Weight updates confined to a low-rank subspace
- **Attention compression:** Approximating the attention matrix by a low-rank matrix
- **Intrinsic dimensionality:** The effective rank of gradient updates during training is often much smaller than the parameter dimension (Gur-Ari et al., 2018)

### N.2 Truncated SVD for Compression

The best rank-$r$ approximation to $A$ (in Frobenius norm) is:
$$A_r = \sum_{i=1}^r \sigma_i u_i v_i^\top$$

with error $\|A - A_r\|_F = \sqrt{\sum_{i>r} \sigma_i^2}$.

**Randomized SVD** (Halko, Martinsson, Tropp, 2011): Compute a rank-$(r+p)$ approximation in $O(mn \log r)$ flops:

1. Draw a random matrix $\Omega \in \mathbb{R}^{n \times (r+p)}$
2. Compute $Y = A\Omega$ and QR-factor: $Y = QR$
3. Compute SVD of the small matrix $B = Q^\top A$: $B = U_B \Sigma V^\top$
4. Return $U = Q U_B$, $\Sigma$, $V$

This is $O(mn(r+p))$ - much faster than $O(\min(m,n)mn)$ for full SVD when $r \ll \min(m,n)$.

---

## Appendix O: Case Studies in ML Systems

### O.1 Attention Computation and Numerical Stability

The attention computation $S = QK^\top / \sqrt{d_k}$ produces an $N \times N$ matrix whose rows are passed through softmax. The numerical challenges:

**Scale issue:** With $d_k = 128$ and random $Q, K$ with unit-norm rows, dot products $q_i^\top k_j$ have magnitude $\sim d_k = 128$. After division by $\sqrt{d_k} = \sqrt{128} \approx 11$, the scaled scores have magnitude $\sim \sqrt{d_k} = 11$.

In fp16 (range +/-65504), this is safe. But the **softmax** of scores with magnitude 11 is nearly a one-hot distribution:
$$\text{softmax}([11, -11]) = \left[\frac{e^{11}}{e^{11} + e^{-11}}, \frac{e^{-11}}{e^{11}+e^{-11}}\right] \approx [1, 2 \times 10^{-10}]$$

This is extreme but not a numerical problem per se - softmax with the max-subtraction trick handles it.

**Memory issue:** For $N = 8192$ and 8 attention heads, storing $S$ requires $8192^2 \times 8 \times 2 \approx 1$ GB - too large for GPU SRAM. FlashAttention avoids this via tiling.

**FlashAttention tile computation:** Process $Q$ in blocks of $B_r$ rows and $K, V$ in blocks of $B_c$ columns. For each tile, compute local scores, apply local softmax with rescaling, accumulate into output. The key identity:

For rows $i$ and columns $j$ in tile $t$:
$$O_i \leftarrow \frac{e^{m_i} O_i + \sum_{j \in t} e^{s_{ij}} V_j}{l_i + \sum_{j \in t} e^{s_{ij}}}$$

where $m_i = \max_j s_{ij}$ (running maximum) and $l_i$ is the running sum of exponentials. This is numerically equivalent to computing the full softmax but uses $O(N)$ memory instead of $O(N^2)$.

### O.2 Gram-Schmidt in Transformer Training

During the training of large language models, maintaining orthogonality in weight updates can improve training stability. **GradNorm** and **gradient orthogonalization** methods apply Gram-Schmidt to gradients:

- Multiple task losses $\mathcal{L}_1, \ldots, \mathcal{L}_T$ produce gradients $g_1, \ldots, g_T$
- Orthogonalizing $g_2$ against $g_1$ prevents the second task from undoing the first
- Numerically, use MGS (not CGS) to avoid loss of orthogonality accumulation

### O.3 Hessian Spectrum in Neural Networks

The Hessian $H = \nabla^2 \mathcal{L}$ of the training loss has a characteristic spectrum:
- A few large eigenvalues (corresponding to the most sensitive directions)
- Many near-zero eigenvalues (corresponding to nearly flat directions)
- This **bulk + outlier** structure explains why second-order methods need special handling

Computing the Hessian spectrum requires eigenvalue algorithms. For large networks ($d = 10^{10}$ parameters), even storing $H$ is impossible. **Lanczos iteration** (power iteration variant) computes the extremal eigenvalues using only Hessian-vector products $Hv$, which can be computed via automatic differentiation in $O(d)$ time and memory.

---

## Appendix P: Summary Reference Card

```
Section02 NUMERICAL LINEAR ALGEBRA - SUMMARY CARD
========================================================================

  CORE PRINCIPLE: Backward stability = exact answer to nearby problem
  Forward error \\leq (A) x backward error

  KEY CONDITION NUMBERS:
    (A)     - digits lost: log_1_0()
    (A^TA)   = (A)^2  -  use QR not normal equations!
    (LL^T)  = 1      -  Cholesky adds no error (for SPD)

  ALGORITHM SELECTION:
    Square Ax = b, well-conditioned    ->  GEPP (LU)
    Square Ax = b, SPD                 ->  Cholesky
    Least squares, ill-conditioned     ->  Householder QR
    Least squares, well-conditioned    ->  Normal equations OK
    Large sparse, SPD                  ->  CG + preconditioner
    Large sparse, non-symmetric        ->  GMRES + preconditioner

  GROWTH FACTOR RULE OF THUMB:
    LU without pivoting:  \\rho can be \\infty
    LU with partial pivot: \\rho \\leq 2^{n-1}, in practice O(n^{2/3})
    Cholesky:             \\rho = 1 exactly

  ITERATIVE METHOD CONVERGENCE:
    CG:    \\varepsilon after k iters \\leq 2\\cdot((\\sqrt-1)/(\\sqrt+1))^k e_0_A
    GS:    converges for diagonally dominant / SPD matrices
    GMRES: always converges (no divergence), but can be slow

  PYTHON API CHEATSHEET:
    nla.solve(A, b)         -  LU with partial pivoting
    nla.lstsq(A, b)         -  Householder QR
    sla.cho_factor/cho_solve-  Cholesky for SPD
    sp.linalg.cg(A, b)      -  Conjugate gradient
    sp.linalg.gmres(A, b)   -  GMRES

========================================================================
```

*[<- Back to Section01 Floating-Point Arithmetic](../01-Floating-Point-Arithmetic/notes.md) | [Next: Section03 Numerical Optimization ->](../03-Numerical-Optimization/notes.md)*

---

## Appendix Q: Iterative Refinement - Full Analysis

### Q.1 Standard Iterative Refinement

Given an approximate solution $\hat{x}^{(0)}$ to $Ax = b$ (e.g., from LU factorization), iterative refinement proceeds:

**Step k:**
1. Compute residual: $r^{(k)} = b - A\hat{x}^{(k)}$ (in higher precision if possible)
2. Solve the correction equation: $A\delta^{(k)} = r^{(k)}$ (using stored LU factors)
3. Update: $\hat{x}^{(k+1)} = \hat{x}^{(k)} + \delta^{(k)}$

**Error analysis:** Let $e^{(k)} = x - \hat{x}^{(k)}$ be the true error. Then:
$$e^{(k+1)} = e^{(k)} - \delta^{(k)} = A^{-1}r^{(k)} - \delta^{(k)}$$

The error in $\delta^{(k)}$ comes from two sources:
- Rounding error in computing $r^{(k)}$: $\tilde{r}^{(k)} = r^{(k)} + f^{(k)}$ with $\|f^{(k)}\| = O(\varepsilon_{\text{mach}} \|A\| \|\hat{x}^{(k)}\|)$
- Backward error in solving: $A\hat{\delta}^{(k)} = \tilde{r}^{(k)} + g^{(k)}$ with $\|g^{(k)}\| = O(\varepsilon_{\text{mach}} \|A\| \|\delta^{(k)}\|)$

**Convergence:** Iterative refinement converges quadratically when $\kappa(A) \varepsilon_{\text{mach}} \ll 1$:
$$\|e^{(k+1)}\| \leq c \kappa(A) \varepsilon_{\text{mach}} \|e^{(k)}\| + O(\varepsilon_{\text{mach}}^2)$$

This means after one refinement step, the error is reduced to $O(\kappa(A) \varepsilon_{\text{mach}})$; after the second step, to $O(\kappa(A)^2 \varepsilon_{\text{mach}}^2)$ (if $\kappa < 1/\varepsilon$). This is essentially optimal.

### Q.2 Mixed-Precision Iterative Refinement

**NVIDIA cuSOLVER GMRES-IR (2022):**
1. Factor $A$ in fp16 (Tensor Core acceleration)
2. Solve approximately via triangular solves in fp16
3. Compute residual in fp64 (using the original fp64 data)
4. Use GMRES in fp32 to refine the solution

**Theoretical guarantees:** Achieve fp64 accuracy at the computational cost of fp32/fp16 factorization. The number of GMRES iterations needed depends on $\kappa(A)$:
- $\kappa(A) < 10^7$: typically 3-5 refinement steps
- $\kappa(A) > 10^{12}$: may not converge (problem too ill-conditioned for fp64)

### Q.3 FGMRES-IR for Mixed Precision

The full algorithm from (Carson & Higham, 2018):

```
Input: A (fp64), b (fp64), desired accuracy \\varepsilon
Output: x (fp64) with ||Ax - b||/||b|| < \\varepsilon

Phase 1: Low-precision factorization
  A = fl32(A)                    # Cast to fp32
  P, L, U = lu_factor(A)        # Factor in fp32

Phase 2: Refinement loop
  x = 0
  r = b                           # Initial residual (fp64)
  for iter = 1, ..., max_iter:
    d = lu_solve(L, U, P, fl32(r))  # Solve in fp32
    x += fl64(d)                     # Accumulate in fp64
    r = b - A @ x                   # New residual in fp64
    if ||r||/||b|| < \\varepsilon: break

Return x
```

This achieves fp64 accuracy with ~2x speedup from fp32 factorization.

---

## Appendix R: Advanced Topics in Eigenvalue Computation

### R.1 Krylov-Schur (State of the Art for Large Sparse)

For large sparse eigenvalue problems (e.g., finding the 100 largest eigenvalues of a million-dimensional Laplacian), the **Krylov-Schur algorithm** (Stewart, 2001) improves on the Lanczos/Arnoldi methods:

1. Build a Krylov basis $\mathcal{K}_k(A, v_0)$ via the Arnoldi relation: $AV_k = V_k H_k + f_k e_k^\top$
2. Compute eigenvalues of the small Hessenberg matrix $H_k$ (Ritz values)
3. Keep the $p$ desired Ritz values; discard the rest via Krylov-Schur restart
4. Expand the basis and repeat

**For AI:** Used in spectral analysis of large language models (computing the dominant eigenvectors of the attention pattern matrix, analyzing the Hessian spectrum of loss landscapes, spectral clustering of large social graphs).

### R.2 LOBPCG for Symmetric Eigenvalue Problems

The **Locally Optimal Block Preconditioned Conjugate Gradient (LOBPCG)** method computes $p$ eigenpairs simultaneously:

- Operates on $p$ vectors $X, W, P \in \mathbb{R}^{n \times p}$
- Each iteration: Rayleigh-Ritz extraction from the $3p$-dimensional trial subspace $\text{span}[X, W, P]$
- Very cache-friendly (block operations); good for GPU acceleration
- Convergence rate: same as CG but for the $p$-th eigengap

**For AI:** LOBPCG is used in GNN implementations for computing the first few eigenvectors of graph Laplacians (for spectral positional encodings) and in second-order optimization for computing the dominant eigenvectors of the Hessian.

---

## Appendix S: Forward Error Analysis Proofs

### S.1 Error in Dot Product

**Claim:** For $x, y \in \mathbb{R}^n$ with elements in fp64, the computed dot product $\hat{s} = \text{fl}(\sum_i x_i y_i)$ satisfies:
$$|\hat{s} - x^\top y| \leq \frac{n \varepsilon_{\text{mach}}}{1 - n \varepsilon_{\text{mach}}} \sum_i |x_i y_i|$$

**Proof:** Each product $x_i y_i$ is rounded: $\text{fl}(x_i y_i) = x_i y_i (1 + \delta_i)$ with $|\delta_i| \leq \varepsilon_{\text{mach}}$.

Each addition accumulates more error. Using the model $\text{fl}(a + b) = (a + b)(1 + \epsilon)$ with $|\epsilon| \leq \varepsilon_{\text{mach}}$, and applying it $n$ times:
$$\hat{s} = \sum_i x_i y_i (1 + \delta_i) \prod_{j \geq i} (1 + \varepsilon_j)$$

Taking the norm and bounding each $(1 + \delta_i) \prod_j (1 + \varepsilon_j) \leq (1 + \varepsilon_{\text{mach}})^{n+1} - 1 \leq \frac{n\varepsilon_{\text{mach}}}{1 - n\varepsilon_{\text{mach}}}$ for $n\varepsilon_{\text{mach}} < 1$. $\square$

**The $n \varepsilon_{\text{mach}}$ bound** is tight: you can construct $x, y$ where the bound is nearly achieved. Kahan summation reduces this to $O(\varepsilon_{\text{mach}})$ independent of $n$.

### S.2 Rounding Error in Matrix-Vector Product

For $y = Ax$ with $A \in \mathbb{R}^{m \times n}$, the computed $\hat{y}$ satisfies:
$$|\hat{y}_i - y_i| \leq n \varepsilon_{\text{mach}} \sum_j |A_{ij}| |x_j| + O(\varepsilon_{\text{mach}}^2)$$

This is the componentwise forward error bound. Its 2-norm is:
$$\|\hat{y} - y\| \leq n \varepsilon_{\text{mach}} \||A| |x|\| \leq n \varepsilon_{\text{mach}} \|A\| \|x\|$$

where $|A|$ denotes elementwise absolute value. The relative error is thus:
$$\frac{\|\hat{y} - y\|}{\|y\|} \leq n \varepsilon_{\text{mach}} \kappa(A)_{\text{componentwise}}$$

---

## Appendix T: Connections to Related Sections

| Concept | This Section (Section02) | Related Section |
|---------|-------------------|-----------------|
| LU factorization existence | Numerical stability of GEPP | [Section03-Advanced-LA/08-Matrix-Decompositions](../../03-Advanced-Linear-Algebra/08-Matrix-Decompositions/notes.md): theory |
| QR factorization | Householder stability, MGS vs CGS | [Section03-Advanced-LA/08](../../03-Advanced-Linear-Algebra/08-Matrix-Decompositions/notes.md): existence, Gram-Schmidt theory |
| Eigenvalue algorithms | Power iteration, QR algorithm | [Section03-Advanced-LA/01-Eigenvalues](../../03-Advanced-Linear-Algebra/01-Eigenvalues-and-Eigenvectors/notes.md): theory |
| Condition number basics | Perturbation theory, error amplification | [Section01 Floating-Point Arithmetic](../01-Floating-Point-Arithmetic/notes.md): definition |
| CG as optimizer | Matrix system solver | [Section03 Numerical Optimization](../03-Numerical-Optimization/notes.md): CG for optimization |
| SVD applications | Numerical rank, truncated SVD | [Section03-Advanced-LA/02-SVD](../../03-Advanced-Linear-Algebra/02-Singular-Value-Decomposition/notes.md): theory |
| Sparse attention patterns | Sparse matrix-vector products | [Section11 Graph Theory](../../11-Graph-Theory/notes.md): graph algorithms |

---

*[<- Back to Chapter 10](../README.md) | [Next: Numerical Optimization ->](../03-Numerical-Optimization/notes.md)*

---

## Appendix U: Deep Dive - Preconditioned Conjugate Gradient

### U.1 The Full PCG Algorithm

The **Preconditioned Conjugate Gradient (PCG)** algorithm, written explicitly:

```
Input: SPD matrix A, right-hand side b, SPD preconditioner M
Output: approximate solution x to Ax = b

x_0 = 0  (or initial guess)
r_0 = b - A x_0
z_0 = M^{-1} r_0          (preconditioner solve)
p_0 = z_0

for k = 0, 1, 2, ...:
    q_k = A p_k            (matrix-vector product)
    alpha_k = r_k^T z_k / p_k^T q_k
    x_{k+1} = x_k + alpha_k p_k
    r_{k+1} = r_k - alpha_k q_k
    if ||r_{k+1}|| < tol: break
    z_{k+1} = M^{-1} r_{k+1}  (preconditioner solve)
    beta_k  = r_{k+1}^T z_{k+1} / r_k^T z_k
    p_{k+1} = z_{k+1} + beta_k p_k

return x_{k+1}
```

**Cost per iteration:**
- 1 matrix-vector product: $O(\text{nnz})$ for sparse $A$
- 1 preconditioner solve: $O(\text{nnz})$ for sparse $M$ (e.g., IC preconditioner)
- 2 dot products and vector updates: $O(n)$

Total: $O(\sqrt{\kappa(M^{-1}A)} \cdot \text{nnz})$ for $\varepsilon$-accurate solution.

### U.2 Optimal Preconditioners for Specific Problems

**Laplacian systems** (arising from graph algorithms, mesh-based PDEs):
- $A = L$ is the graph Laplacian; $\kappa(L) \sim n^2/h^2$ for mesh problems
- **Algebraic multigrid (AMG):** constructs a hierarchy of coarser graphs; each coarse-grid solve is $O(n)$; $\kappa(M^{-1}L) \approx 1$
- AMG is used in nearly all large-scale PDE solvers (HYPRE, PETSc)

**Systems from neural networks (K-FAC):**
- $A \approx \hat{A} = \Gamma_A \otimes \Gamma_G$ (Kronecker product of activation/gradient covariances)
- Preconditioner: $M^{-1} \approx (\Gamma_A + \lambda I)^{-1} \otimes (\Gamma_G + \lambda I)^{-1}$
- Each factor can be Cholesky-factored independently: $O(d_{\text{in}}^3 + d_{\text{out}}^3)$ instead of $O((d_{\text{in}} d_{\text{out}})^3)$

---

## Appendix V: Worked Example - Full Solution of a Least Squares Problem

**Problem:** Fit a polynomial of degree $d = 5$ to $m = 30$ noisy data points.

**Setup:**
```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
m, d = 30, 6  # 6 coefficients for degree 5

# True function: sin(2*pi*x) + 0.5*cos(4*pi*x)
x = np.linspace(0, 1, m)
y_true = np.sin(2*np.pi*x) + 0.5*np.cos(4*np.pi*x)
y_noisy = y_true + 0.1 * np.random.randn(m)

# Build Vandermonde matrix
A = np.vander(x, d, increasing=True)  # A[i, j] = x_i^j

# Condition number
kappa = np.linalg.cond(A)
print(f"kappa(A) = {kappa:.2e}")

# Method 1: Normal equations (unstable for high degree)
c_normal = np.linalg.solve(A.T @ A, A.T @ y_noisy)

# Method 2: Householder QR (stable)
c_qr, _, _, _ = np.linalg.lstsq(A, y_noisy, rcond=None)

# Evaluate and compare errors
print(f"Residual (normal): {np.linalg.norm(A @ c_normal - y_noisy):.6f}")
print(f"Residual (QR):     {np.linalg.norm(A @ c_qr    - y_noisy):.6f}")
```

**Expected output:**
- For degree 5 with $x \in [0, 1]$: $\kappa(A) \approx 10^5$ - still manageable
- For degree 10 with $x \in [0, 1]$: $\kappa(A) \approx 10^{12}$ - Normal equations fail, QR still works
- For degree 15: $\kappa(A) > 10^{15}$ - even QR starts losing accuracy; need Chebyshev nodes

**Key lesson:** The numerical behavior of polynomial fitting is entirely determined by $\kappa(A)$, which depends on both the degree and the spacing of evaluation points. Chebyshev nodes minimize the Lebesgue constant, which is related to (but not identical to) the condition number.

---

## Appendix W: Stability Proofs

### W.1 Backward Stability of Triangular Solve

**Theorem:** Let $L$ be a lower triangular matrix with unit diagonal. The computed solution $\hat{x}$ to $Lx = b$ satisfies:
$$(L + \delta L)\hat{x} = b, \quad |\delta L_{ij}| \leq \varepsilon_{\text{mach}} |L_{ij}| \cdot O(n)$$

**Proof (sketch for $2 \times 2$):** The algorithm computes:
$$\hat{x}_1 = \text{fl}(b_1) = b_1(1+\varepsilon_1)$$
$$\hat{x}_2 = \text{fl}(b_2 - L_{21} \hat{x}_1) = (b_2 - L_{21}\hat{x}_1)(1+\varepsilon_2)$$

Expanding: $\hat{x}_2 = b_2(1+\varepsilon_2) - L_{21} b_1 (1+\varepsilon_1)(1+\varepsilon_2)$.

The true solution satisfies $x_2 = b_2 - L_{21} b_1$, so:
$$\hat{x}_2 - x_2 = b_2 \varepsilon_2 - L_{21} b_1 \varepsilon_1(1+\varepsilon_2) - L_{21}b_1\varepsilon_2$$

This corresponds to solving $(L + \delta L)x = b$ where $\delta L_{21} = L_{21} \varepsilon_1(1+\varepsilon_2)$ and $\delta b_2 = -b_2\varepsilon_2$. The backward error is $O(\varepsilon_{\text{mach}})$. $\square$

### W.2 Orthogonality Loss in Classical Gram-Schmidt

**Theorem:** Classical Gram-Schmidt applied to $A$ with $\kappa(A) = \kappa$ produces $\hat{Q}$ with:
$$\|I - \hat{Q}^\top \hat{Q}\| = O(\kappa^2 \varepsilon_{\text{mach}})$$

**Intuition:** At step $k$, we project $a_k$ against all previous $\hat{q}_i$. The error in each $\hat{q}_i$ is amplified by subsequent projections. After $k$ steps, errors in $\hat{q}_1$ have been "propagated" $k-1$ times through imperfect projections, leading to $O(\kappa^2)$ error in the loss of orthogonality.

Modified Gram-Schmidt projects against the *updated* $a_k$ at each step, breaking the error propagation chain and achieving $O(\kappa \varepsilon_{\text{mach}})$ - still worse than Householder's $O(\varepsilon_{\text{mach}})$ but much better than CGS.

---

## Appendix X: Glossary

| Term | Definition |
|------|-----------|
| **Backward stable** | Algorithm that produces the exact output for a nearby input; relative perturbation $O(\varepsilon_{\text{mach}})$ |
| **Condition number** | $\kappa(A) = \|A\| \|A^{-1}\|$; amplification factor from input to output perturbations |
| **Fill-in** | New nonzero entries created in $L$ and $U$ during sparse LU factorization where $A$ had zeros |
| **Forward error** | $\|\hat{x} - x\| / \|x\|$; how far the computed answer is from the true answer |
| **Growth factor** | $\rho = \max_{ijk} |u_{ij}^{(k)}| / \max_{ij} |a_{ij}|$; measures amplification in Gaussian elimination |
| **Krylov subspace** | $\mathcal{K}_k(A,v) = \text{span}\{v, Av, A^2v, \ldots, A^{k-1}v\}$; basis for iterative methods |
| **Partial pivoting** | Row swapping strategy that maximizes each pivot; ensures $|\ell_{ij}| \leq 1$ |
| **Preconditioner** | Matrix $M \approx A$ chosen so $M^{-1}A$ has better conditioning than $A$ |
| **Rayleigh quotient** | $\rho(x) = x^\top A x / x^\top x$; estimates eigenvalues |
| **Spectral radius** | $\rho(A) = \max_i |\lambda_i|$; determines convergence of iterative methods |
| **BLAS** | Basic Linear Algebra Subprograms; standardized interface for vector/matrix operations |
| **LAPACK** | Linear Algebra PACKage; implements factorizations, solvers, eigenvalue problems |
| **cuBLAS** | CUDA implementation of BLAS for NVIDIA GPUs; used in PyTorch, TensorFlow |

---

*[<- Back to Section01 Floating-Point Arithmetic](../01-Floating-Point-Arithmetic/notes.md) | [Next: Section03 Numerical Optimization ->](../03-Numerical-Optimization/notes.md)*

---

## Appendix Y: Extended Exercises with Solutions

### Y.1 The Leaning Matrix Problem (**)

Consider the matrix:
$$A_\varepsilon = \begin{pmatrix} 1 & 1 \\ \varepsilon & 0 \\ 0 & \varepsilon \end{pmatrix}, \quad \varepsilon = 10^{-8}$$

(a) Compute $A_\varepsilon^\top A_\varepsilon$ and observe that $1 + \varepsilon^2 = 1$ in fp64.
(b) Show that the Normal equations are ill-conditioned: $\kappa(A^\top A) = \infty$ in fp64.
(c) Show that QR factorization recovers the correct solution.

**Solution:**
```python
eps = 1e-8
A = np.array([[1, 1], [eps, 0], [0, eps]])
AtA = A.T @ A
print(f"A^T A = {AtA}")
# [[1, 1], [1, 1]] - singular! eps^2 = 1e-16 < machine_eps = 2.2e-16

b = np.array([1, eps, eps])

# Normal equations: singular system!
try:
    x_normal = np.linalg.solve(AtA, A.T @ b)
    print(f"Normal: {x_normal}")
except np.linalg.LinAlgError:
    print("Normal equations: singular!")

# QR: works correctly
x_qr, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
print(f"QR: {x_qr}")  # Approximately [0.5, 0.5]
print(f"Residual QR: {np.linalg.norm(A @ x_qr - b):.2e}")
```

### Y.2 Ill-Conditioned System via Scaling (*)

Two matrices that are mathematically equivalent but numerically different:

```python
# Original: well-conditioned
A1 = np.array([[1.0, 0.5], [0.5, 1.0]])
b1 = np.array([1.0, 0.0])

# Scaled: same equations, different numerics
D = np.diag([1e8, 1.0])
A2 = D @ A1 @ D  # Equivalent to A1 up to scaling
b2 = D @ b1

print(f"kappa(A1) = {np.linalg.cond(A1):.2f}")  # ~ 3
print(f"kappa(A2) = {np.linalg.cond(A2):.2e}")  # ~ 3e16!

x1 = np.linalg.solve(A1, b1)
x2 = np.linalg.solve(A2, b2)

# Transform x2 back to same scale as x1
x2_transformed = D @ x2
print(f"x1                = {x1}")
print(f"x2 (transformed)  = {x2_transformed}")
print(f"Error: {np.linalg.norm(x1 - x2_transformed):.2e}")
```

**Lesson:** Always equilibrate (scale) your system before solving. `scipy.linalg.solve` with `overwrite_a=False` does this automatically via LAPACK's `dgesvx`.

### Y.3 Conjugate Gradient Rate Verification (**)

Verify the CG convergence bound experimentally:

```python
def cg_solve(A, b, tol=1e-12, max_iter=1000):
    n = len(b)
    x = np.zeros(n)
    r = b.copy()
    p = r.copy()
    rs_old = r @ r
    errors = [np.linalg.norm(r)]

    for _ in range(max_iter):
        Ap = A @ p
        alpha = rs_old / (p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        rs_new = r @ r
        errors.append(np.sqrt(rs_new))
        if np.sqrt(rs_new) < tol * np.linalg.norm(b):
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new

    return x, errors

# Create SPD matrix with known condition number
kappa = 100
n = 50
lam = np.concatenate([np.linspace(1, kappa, n//2),
                       np.linspace(kappa, kappa**0.5, n//2)])
Q, _ = np.linalg.qr(np.random.randn(n, n))
A = Q @ np.diag(lam) @ Q.T

b = np.random.randn(n)
x_true = np.linalg.solve(A, b)

x, errors = cg_solve(A, b)
errors_norm = [e/errors[0] for e in errors]

# Theoretical bound
bound = [2 * ((np.sqrt(kappa)-1)/(np.sqrt(kappa)+1))**k
         for k in range(len(errors))]

print("CG convergence (iteration : relative residual : bound):")
for k in [0, 5, 10, 20, 30]:
    if k < len(errors):
        print(f"  k={k:3d}: {errors_norm[k]:.4e}  (bound: {bound[k]:.4e})")
```

---

## Appendix Z: Chapter Summary and Self-Assessment

After completing this section, you should be able to:

**Conceptual understanding:**
- [ ] Explain why backward stability is the right standard for numerical algorithms
- [ ] State the fundamental theorem relating forward error, backward error, and condition number
- [ ] Explain why GEPP is stable in practice despite a theoretical worst case of $2^{n-1}$
- [ ] Explain why QR via Householder is preferred over Normal equations for least squares
- [ ] State the CG convergence rate and its dependence on $\kappa(A)$
- [ ] Explain what a Krylov subspace is and why iterative methods based on it are optimal

**Computational skills:**
- [ ] Implement LU factorization with partial pivoting from scratch
- [ ] Implement Householder QR from scratch
- [ ] Implement the Conjugate Gradient algorithm
- [ ] Choose the right solver for a given problem (dense vs sparse, symmetric vs general)
- [ ] Diagnose ill-conditioning using condition numbers and error estimates
- [ ] Apply diagonal preconditioning to improve CG convergence

**AI connections:**
- [ ] Explain how Adam implements diagonal preconditioning
- [ ] Explain why attention uses $1/\sqrt{d_k}$ scaling
- [ ] Describe the numerical approach in FlashAttention
- [ ] Connect LoRA to the low-rank approximation theory
- [ ] Explain how K-FAC uses Kronecker-factored matrices for efficient second-order optimization

**Difficulty distribution:** This material spans from algorithmic implementation (*) through convergence analysis (**) to research-level connections like K-FAC and FlashAttention (***). The core material (Section1-Section6) is accessible to anyone who has completed Section02-Linear-Algebra-Basics; the advanced applications (Section7-Section8) benefit from additional exposure to optimization and deep learning.


---

## Appendix AA: Numerical Linear Algebra in Modern GPU Architectures

### AA.1 NVIDIA GPU Linear Algebra Pipeline

Modern GPU architectures have specialized hardware for linear algebra:

**Tensor Cores (Volta/Ampere/Hopper):**
- Compute $D = A \cdot B + C$ where $A, B, C, D$ are small matrices (e.g., $16 \times 16$)
- One Tensor Core instruction: $D = \text{fl}_{tf32}(A \cdot B) + C$ in one cycle
- Peak performance: A100 GPU achieves 312 TFLOPS for bf16 matmul
- Key property: the matmul is performed in tf32 (10-bit mantissa) but accumulated in fp32

**Implications for numerical accuracy:**
- A100 performs $16 \times 16$ matmul blocks in tf32 precision
- Accumulation in fp32 prevents error accumulation within each block
- For very large matrices, the number of block-level accumulations can lead to $O(\sqrt{n}) \varepsilon_{\text{mach}}$ error per output element (better than naive $O(n \varepsilon_{\text{mach}})$ because of hierarchical accumulation)

**cuBLAS implementation of DGEMM:**
The doubly-blocked matmul implementation:
```
Block A in shared memory: 128x16
Block B in shared memory: 16x128
Each thread computes: 4x4 sub-block of output
Register tiles: 4 registers for A-slice, 4 for B-slice
Result accumulated in: 16 registers (fp32)
```

### AA.2 Memory Hierarchy and Algorithm Design

| Level | Size | Bandwidth | Latency |
|-------|------|-----------|---------|
| Registers | 256 KB/SM | \\infty | 1 cycle |
| L1/Shared | 96 KB/SM | 128 TB/s | 20 cycles |
| L2 cache | 40 MB | 12 TB/s | 200 cycles |
| HBM (GPU DRAM) | 80 GB | 2 TB/s | 500 cycles |
| PCIe (CPU<->GPU) | - | 64 GB/s | ms |

**Roofline model:** An algorithm is **compute-bound** if the number of flops is large relative to data movement; **memory-bound** if it's the opposite.

- Dense matmul ($n=1024$): $O(n^3)$ flops, $O(n^2)$ reads -> arithmetic intensity $O(n)$ -> compute-bound
- SpMV ($\text{nnz}$ sparse): $O(\text{nnz})$ flops, $O(\text{nnz})$ reads -> arithmetic intensity $O(1)$ -> memory-bound
- FlashAttention: converts $O(N^2)$ memory reads to $O(N \cdot d_k)$ by blocking -> shifts from memory-bound to compute-bound

### AA.3 Mixed Precision in Practice

**A100 Tensor Core configurations:**
| Input dtype | Accumulate dtype | Peak TFLOPs | Use case |
|-------------|-----------------|-------------|----------|
| fp64 | fp64 | 19.5 | Scientific computing |
| fp32 | fp32 | 19.5 | Training (non-Tensor Core) |
| tf32 | fp32 | 156 | Training (Tensor Core, auto) |
| bf16 | fp32 | 312 | LLM training |
| fp16 | fp32 | 312 | Computer vision |
| fp8 | fp32 | 624 | LLM inference, H100+ |
| int8 | int32 | 624 | Quantized inference |

**PyTorch automatic mixed precision (AMP):**
```python
# Enable Tensor Core matmul (automatic tf32)
torch.backends.cuda.matmul.allow_tf32 = True  # Default True for A100+

# Full AMP with bf16
with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    output = model(input)  # Forward pass in bf16
    loss = criterion(output, target)

scaler.scale(loss).backward()  # Backward in bf16 with loss scaling
scaler.step(optimizer)
scaler.update()
```

---

## Appendix BB: Historical Notes on Algorithm Development

The development of numerical linear algebra as a field was driven by the first digital computers (ENIAC, 1945; ACE, 1950; UNIVAC, 1951). Gaussian elimination had been used by hand for centuries, but the question of whether it would work reliably on early (extremely noisy, vacuum-tube-based) computers was open.

**The stability debate (1947-1954):**
- Von Neumann and Goldstine (1947) analyzed Gaussian elimination and concluded it might be dangerously unstable - their bound on the error was $O(4^n \varepsilon_{\text{mach}})$, suggesting exponential growth
- Turing (1948) introduced the condition number and argued that the growth was much more benign in practice
- Wilkinson (1954) proved that with partial pivoting, the growth factor satisfies $\rho \leq 2^{n-1}$ in the worst case but $O(n^{2/3})$ in practice - vindicating GEPP as the standard method

**Householder's contribution (1958):**
After Gram-Schmidt's orthogonalization was found to be numerically unstable for ill-conditioned matrices, Householder derived the reflector-based QR that achieves backward stability. This became the standard for solving least squares problems.

**The Lanczos problem (1950-1980):**
Lanczos (1950) proposed his tridiagonalization algorithm for symmetric eigenvalue problems, but it was found to be numerically unreliable - the computed Krylov vectors rapidly lost orthogonality. It took 30 years of research to understand when and why it works (Paige, 1980) and to develop practical implementations with selective reorthogonalization.

**Modern era (1987-present):**
The LAPACK project standardized high-performance implementations of all major algorithms. The emergence of GPU computing (2007+) required adapting these algorithms for massively parallel hardware. The development of mixed-precision algorithms and iterative refinement has become critical for LLM training at scale.

---

*[<- Back to Chapter 10](../README.md) | [Next: Numerical Optimization ->](../03-Numerical-Optimization/notes.md)*

---

## Appendix CC: Practice Problems Bank

### CC.1 True/False Questions

1. Gaussian elimination with partial pivoting is always backward stable. *(False - the growth factor can be $2^{n-1}$, but this almost never happens)*
2. The condition number of $A^\top A$ equals $\kappa(A)^2$. *(True)*
3. CG converges in at most $n$ steps in exact arithmetic. *(True)*
4. A backward stable algorithm always produces a small forward error. *(False - the condition number can amplify the backward error)*
5. Householder QR is always preferred over MGS for orthogonalization. *(False - MGS is preferred for streaming applications)*
6. The Cholesky factorization is unique for SPD matrices. *(True, with positive diagonal)*
7. For SPD matrices, LU and Cholesky give the same factorization. *(False - LU has $LU$ form, Cholesky has $LL^\top$ form)*
8. GMRES always finds the solution in at most $n$ iterations. *(True in exact arithmetic; in practice, restart before $n$)*
9. Diagonal scaling cannot change the condition number of a matrix. *(False - scaling dramatically affects condition number)*
10. Adam's $1/\sqrt{v}$ denominator is a form of diagonal preconditioning. *(True)*

### CC.2 Fill in the Blanks

1. The backward error for GEPP satisfies $\|\delta A\| / \|A\| \leq O(\_\_\_\_\_ \cdot \varepsilon_{\text{mach}})$ where the blank depends on the problem size and growth factor.
2. The CG algorithm requires the matrix to be \_\_\_\_\_ positive definite.
3. The \_\_\_\_\_ algorithm reduces any symmetric matrix to tridiagonal form as a preprocessing step for eigenvalue computation.
4. FlashAttention avoids materializing the $N \times N$ attention matrix by using a \_\_\_\_\_ algorithm.
5. Modified Gram-Schmidt achieves $O(\_\_\_\_\_)$ loss of orthogonality, better than Classical Gram-Schmidt's $O(\kappa^2 \varepsilon)$.

*Answers: 1. $\rho n$; 2. symmetric; 3. Lanczos (or Householder tridiagonalization); 4. tiled; 5. $\kappa \varepsilon$*

### CC.3 Short Derivations

1. **Derive** the optimal step size for CG from first principles: minimize $f(x_k + \alpha p_k)$ over $\alpha$.
2. **Prove** that $\kappa(A^\top A) = \kappa(A)^2$ using the SVD $A = U\Sigma V^\top$.
3. **Show** that for any orthogonal matrix $Q$: $\kappa(QA) = \kappa(A)$.
4. **Derive** the Rayleigh quotient iteration from the Newton-Raphson method applied to finding an eigenvector.


---

*This section is part of the Section10 Numerical Methods chapter. All algorithms presented are implemented in [theory.ipynb](theory.ipynb) with numerical verification.*

*[<- Back to Section01 Floating-Point Arithmetic](../01-Floating-Point-Arithmetic/notes.md) | [Next: Section03 Numerical Optimization ->](../03-Numerical-Optimization/notes.md)*

**Section statistics:** 12 main sections, 28 subsections, 12 common mistakes, 10 exercises, 28 appendices. All algorithms are numerically verified in the companion notebook.

*End of Section02 Numerical Linear Algebra*
