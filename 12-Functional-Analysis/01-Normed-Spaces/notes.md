[Back to Functional Analysis](../README.md) | [Next: Hilbert Spaces](../02-Hilbert-Spaces/notes.md)

---

# Normed Spaces

> _"A norm is not just a way to measure length. It is a choice of geometry, topology, stability, and convergence."_

## Overview

Normed spaces are vector spaces equipped with a rule for measuring size. That small addition turns algebra into analysis: once vectors have a norm, we can talk about distance, convergence, continuity, boundedness, approximation, stability, and fixed points. Functional analysis begins here because machine learning often studies not just finite vectors, but functions, operators, probability distributions, embeddings, and limits of iterative algorithms.

For AI, norms are everywhere. Regularization uses norms to control model complexity. Adversarial robustness uses norms to define allowable perturbations. Gradient clipping uses norms to control update size. Spectral normalization uses operator norms to control Lipschitz constants. Reinforcement learning relies on contraction mappings in normed spaces. Kernel methods and Hilbert spaces build on this foundation by adding inner products later.

This section is the canonical home for normed-space foundations in the Functional Analysis chapter. We focus on norm axioms, induced metrics, common finite and infinite-dimensional examples, norm equivalence, convergence, completeness, Banach spaces, bounded linear operators, operator norms, dual norms, Lipschitz maps, and the Banach fixed-point theorem. Inner products, orthogonality, projections, and RKHS theory are intentionally left to [Hilbert Spaces](../02-Hilbert-Spaces/notes.md) and [Kernel Methods](../03-Kernel-Methods/notes.md).

## Prerequisites

- **Vector spaces, span, linear maps, and subspaces** - [Linear Algebra Basics](../../02-Linear-Algebra-Basics/README.md)
- **Matrix norms and singular values** - [Matrix Norms](../../03-Advanced-Linear-Algebra/06-Matrix-Norms/notes.md)
- **Sequences and limits** - [Calculus Fundamentals](../../04-Calculus-Fundamentals/README.md)
- **Convexity and regularization language** - [Convex Optimization](../../08-Optimization/01-Convex-Optimization/notes.md)
- **Probability and expectations** - useful for $L^p$ spaces - [Probability Theory](../../06-Probability-Theory/README.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Interactive norm geometry, unit balls, convergence, operator norms, dual norms, contractions, and ML perturbation examples |
| [exercises.ipynb](exercises.ipynb) | 8 graded exercises on norm axioms, counterexamples, equivalence, completeness, operator norms, duality, and fixed points |

## Learning Objectives

After completing this section, you will:

1. State the norm axioms and distinguish norms, seminorms, metrics, and pseudometrics
2. Explain how a norm induces a metric and a topology
3. Compute common $\ell^p$, matrix, operator, and function norms
4. Prove that $\ell^0$ and $\ell^p$ for $0<p<1$ are not norms
5. Interpret unit balls geometrically and connect their shape to regularization and robustness
6. Use norm equivalence in finite-dimensional spaces and explain why it fails in infinite dimensions
7. Define convergence, Cauchy sequences, completeness, and Banach spaces
8. Analyze bounded linear operators and compute operator norms in simple cases
9. Derive dual norms and apply Holder's inequality
10. Use contraction mappings and the Banach fixed-point theorem to reason about iterative algorithms
11. Connect normed-space ideas to adversarial perturbations, gradient clipping, spectral normalization, and reinforcement learning

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 Norms as Geometry of Size and Perturbation](#11-norms-as-geometry-of-size-and-perturbation)
  - [1.2 Why Normed Spaces Matter for ML Stability](#12-why-normed-spaces-matter-for-ml-stability)
  - [1.3 Unit Balls and Sparsity](#13-unit-balls-and-sparsity)
  - [1.4 Finite-Dimensional vs Infinite-Dimensional Behavior](#14-finite-dimensional-vs-infinite-dimensional-behavior)
  - [1.5 Historical Timeline](#15-historical-timeline)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Vector Spaces and Norm Axioms](#21-vector-spaces-and-norm-axioms)
  - [2.2 Normed Spaces and Induced Metrics](#22-normed-spaces-and-induced-metrics)
  - [2.3 Seminorms, Pseudometrics, and Quotients](#23-seminorms-pseudometrics-and-quotients)
  - [2.4 Examples and Non-Examples](#24-examples-and-non-examples)
  - [2.5 Open Balls, Closed Balls, Neighborhoods, and Topology](#25-open-balls-closed-balls-neighborhoods-and-topology)
- [3. Core Theory I: Common Norms and Geometry](#3-core-theory-i-common-norms-and-geometry)
  - [3.1 $\ell^p$ Norms on $\mathbb{R}^n$](#31-ellp-norms-on-mathbbrn)
  - [3.2 Why $\ell^0$ Is Not a Norm](#32-why-ell0-is-not-a-norm)
  - [3.3 Matrix Norms](#33-matrix-norms)
  - [3.4 Norm Equivalence in Finite Dimensions](#34-norm-equivalence-in-finite-dimensions)
  - [3.5 Unit Ball Geometry and Convexity](#35-unit-ball-geometry-and-convexity)
- [4. Core Theory II: Convergence and Completeness](#4-core-theory-ii-convergence-and-completeness)
  - [4.1 Convergence in a Normed Space](#41-convergence-in-a-normed-space)
  - [4.2 Cauchy Sequences](#42-cauchy-sequences)
  - [4.3 Banach Spaces](#43-banach-spaces)
  - [4.4 Examples: $\mathbb{R}^n$, $C[0,1]$, $\ell^p$, and $L^p$](#44-examples-mathbbrn-c01-ellp-and-lp)
  - [4.5 Completion of a Normed Space](#45-completion-of-a-normed-space)
- [5. Core Theory III: Operators, Lipschitz Maps, and Dual Norms](#5-core-theory-iii-operators-lipschitz-maps-and-dual-norms)
  - [5.1 Bounded Linear Operators](#51-bounded-linear-operators)
  - [5.2 Bounded if and only if Continuous for Linear Maps](#52-bounded-if-and-only-if-continuous-for-linear-maps)
  - [5.3 Operator Norms and Neural-Network Lipschitz Constants](#53-operator-norms-and-neural-network-lipschitz-constants)
  - [5.4 Dual Spaces and Dual Norms](#54-dual-spaces-and-dual-norms)
  - [5.5 Holder Inequality and Regularization Geometry](#55-holder-inequality-and-regularization-geometry)
- [6. Advanced Topics](#6-advanced-topics)
  - [6.1 Banach Fixed-Point Theorem and Contractions](#61-banach-fixed-point-theorem-and-contractions)
  - [6.2 Compactness: Finite vs Infinite Dimension](#62-compactness-finite-vs-infinite-dimension)
  - [6.3 Uniform, Pointwise, and Norm Convergence](#63-uniform-pointwise-and-norm-convergence)
  - [6.4 Banach-Space Geometry: Strict Convexity and Smoothness](#64-banach-space-geometry-strict-convexity-and-smoothness)
  - [6.5 Bridge to Hilbert Spaces](#65-bridge-to-hilbert-spaces)
- [7. Applications in Machine Learning](#7-applications-in-machine-learning)
  - [7.1 $L^1$, $L^2$, and Elastic-Net Regularization](#71-l1-l2-and-elastic-net-regularization)
  - [7.2 Adversarial Perturbations in $\ell^\infty$ and $\ell^2$ Balls](#72-adversarial-perturbations-in-ellinfty-and-ell2-balls)
  - [7.3 Gradient Clipping and Update Control](#73-gradient-clipping-and-update-control)
  - [7.4 Spectral Normalization and Lipschitz Control](#74-spectral-normalization-and-lipschitz-control)
  - [7.5 Fixed-Point Iterations in RL, Diffusion, and Implicit Layers](#75-fixed-point-iterations-in-rl-diffusion-and-implicit-layers)
- [8. Common Mistakes](#8-common-mistakes)
- [9. Exercises](#9-exercises)
- [10. Why This Matters for AI](#10-why-this-matters-for-ai)
- [11. Conceptual Bridge](#11-conceptual-bridge)
- [References](#references)

---

## 1. Intuition

### 1.1 Norms as Geometry of Size and Perturbation

A vector space only tells us how to add vectors and multiply by scalars. It does not tell us whether a vector is small, whether two vectors are close, whether a sequence converges, or whether an update is stable. A norm adds that missing analytic structure.

A norm is a function

$$
\lVert \cdot \rVert : V \to \mathbb{R}
$$

that measures vector size in a way compatible with the vector-space operations. Once the norm is chosen, distance follows:

$$
d(\mathbf{u}, \mathbf{v}) = \lVert \mathbf{u} - \mathbf{v} \rVert.
$$

The important phrase is "once the norm is chosen." Different norms create different geometries. In $\mathbb{R}^2$, the $\ell^2$ unit ball is round, the $\ell^1$ unit ball is diamond-shaped, and the $\ell^\infty$ unit ball is square. Those shapes are not decorative; they control which perturbations are considered small and which regularized solutions are encouraged.

```text
NORM CHOICE = GEOMETRY CHOICE
============================================================

  l1 ball: diamond       l2 ball: disk        linf ball: square

       /\                   *****              +-------+
      /  \                **     **            |       |
     /    \              *         *           |       |
     \    /              *         *           |       |
      \  /                **     **            |       |
       \/                   *****              +-------+

  sparse corners       rotation invariant     worst-case coordinate

============================================================
```

**For AI:** The norm defines the perturbation model. If adversarial examples are constrained in $\ell^\infty$, then every pixel may change a little. If constrained in $\ell^2$, then total energy is bounded. If regularization uses $\ell^1$, sparse parameters are favored. Geometry becomes behavior.

### 1.2 Why Normed Spaces Matter for ML Stability

Machine learning algorithms are iterative. Training repeatedly updates parameters. Inference repeatedly applies layers. Reinforcement learning repeatedly applies Bellman operators. Diffusion samplers repeatedly denoise. If these repeated transformations are unstable, small errors grow.

Norms let us state stability precisely. A function $f : V \to W$ is $L$-Lipschitz if

$$
\lVert f(\mathbf{x}) - f(\mathbf{y}) \rVert_W
\leq L \lVert \mathbf{x} - \mathbf{y} \rVert_V.
$$

If $L$ is small, outputs cannot change too much when inputs change slightly. If $L$ is huge, small perturbations can be amplified. This is why operator norms and Lipschitz constants are load-bearing concepts in robust ML.

Examples:

- Gradient clipping bounds $\lVert \mathbf{g}_t \rVert$ before applying an update.
- Spectral normalization controls $\lVert W \rVert_2$ for weight matrices.
- Wasserstein GANs constrain discriminator Lipschitzness.
- Bellman operators are contractions in the sup norm under discounting.
- Fixed-point implicit layers require contraction or monotonicity-style stability.

### 1.3 Unit Balls and Sparsity

The unit ball of a norm is

$$
B = \{\mathbf{x} \in V : \lVert \mathbf{x} \rVert \leq 1\}.
$$

For a norm, $B$ is convex, symmetric, absorbing, and balanced. Conversely, many norms can be understood through their unit balls. The sharp corners of the $\ell^1$ ball explain why $\ell^1$ regularization promotes sparse solutions: optimizing a linear objective over a diamond tends to hit a corner, and the corners lie on coordinate axes.

Compare:

$$
\lVert \mathbf{x} \rVert_1 = \sum_i \lvert x_i \rvert,
\qquad
\lVert \mathbf{x} \rVert_2 = \sqrt{\sum_i x_i^2},
\qquad
\lVert \mathbf{x} \rVert_\infty = \max_i \lvert x_i \rvert.
$$

Each norm says "small" differently. $\ell^1$ counts total absolute mass, $\ell^2$ measures Euclidean energy, and $\ell^\infty$ controls the largest coordinate.

### 1.4 Finite-Dimensional vs Infinite-Dimensional Behavior

In finite-dimensional spaces, all norms are equivalent. That means they define the same notion of convergence, even if they assign different numerical lengths. For $\mathbb{R}^n$,

$$
\lVert \mathbf{x} \rVert_\infty
\leq \lVert \mathbf{x} \rVert_2
\leq \lVert \mathbf{x} \rVert_1
\leq \sqrt{n}\lVert \mathbf{x} \rVert_2
\leq n\lVert \mathbf{x} \rVert_\infty.
$$

In infinite-dimensional spaces this safety disappears. Pointwise convergence, uniform convergence, $L^1$ convergence, and $L^2$ convergence can disagree. A sequence of functions can converge under one norm and fail under another. This is one reason functional analysis is not just "linear algebra with bigger vectors."

**For AI:** Finite-dimensional implementations may hide infinite-dimensional ideas. Kernel methods, Gaussian processes, neural tangent kernels, and function-space generalization all require care about which normed function space is actually being used.

### 1.5 Historical Timeline

```text
NORMED SPACES TIMELINE
============================================================

  1900s   Frechet, Riesz       metric and function-space foundations
  1920s   Banach               complete normed spaces and operator theory
  1930s   Hahn-Banach era      duality and extension theorems
  1940s   Fixed point theory   contractions and iterative methods
  1960s   Convex analysis      dual norms and optimization geometry
  1990s   Sparse learning      l1 methods and compressed sensing roots
  2010s   Deep learning        spectral norms, clipping, adversarial norms
  2020s   Large models         operator stability, function-space views

============================================================
```

---

## 2. Formal Definitions

### 2.1 Vector Spaces and Norm Axioms

Let $V$ be a vector space over $\mathbb{R}$ or $\mathbb{C}$. A **norm** on $V$ is a function $\lVert \cdot \rVert : V \to \mathbb{R}$ satisfying, for all $\mathbf{u},\mathbf{v} \in V$ and scalars $\alpha$:

1. **Positive definiteness**

$$
\lVert \mathbf{v} \rVert \geq 0,
\qquad
\lVert \mathbf{v} \rVert = 0
\iff
\mathbf{v}=\mathbf{0}.
$$

2. **Absolute homogeneity**

$$
\lVert \alpha \mathbf{v} \rVert
= \lvert \alpha \rvert \lVert \mathbf{v} \rVert.
$$

3. **Triangle inequality**

$$
\lVert \mathbf{u}+\mathbf{v} \rVert
\leq
\lVert \mathbf{u} \rVert+\lVert \mathbf{v} \rVert.
$$

The triangle inequality is the analytic core. It ensures that direct movement is never longer than moving through an intermediate point.

### 2.2 Normed Spaces and Induced Metrics

A **normed space** is a pair $(V,\lVert \cdot \rVert)$ where $V$ is a vector space and $\lVert \cdot \rVert$ is a norm on $V$.

Every norm induces a metric:

$$
d(\mathbf{u},\mathbf{v})
= \lVert \mathbf{u}-\mathbf{v} \rVert.
$$

This metric satisfies:

- $d(\mathbf{u},\mathbf{v}) \geq 0$
- $d(\mathbf{u},\mathbf{v})=0$ iff $\mathbf{u}=\mathbf{v}$
- $d(\mathbf{u},\mathbf{v})=d(\mathbf{v},\mathbf{u})$
- $d(\mathbf{u},\mathbf{w}) \leq d(\mathbf{u},\mathbf{v})+d(\mathbf{v},\mathbf{w})$

Not every metric comes from a norm. A norm-induced metric is translation-invariant:

$$
d(\mathbf{u}+\mathbf{a},\mathbf{v}+\mathbf{a})
= d(\mathbf{u},\mathbf{v}).
$$

It is also homogeneous:

$$
d(\alpha\mathbf{u},\alpha\mathbf{v})
= \lvert \alpha \rvert d(\mathbf{u},\mathbf{v}).
$$

### 2.3 Seminorms, Pseudometrics, and Quotients

A **seminorm** satisfies homogeneity and the triangle inequality, but may assign zero size to nonzero vectors:

$$
p(\mathbf{v}) = 0
\quad \text{does not necessarily imply} \quad
\mathbf{v}=\mathbf{0}.
$$

Example: on differentiable functions, define

$$
p(f)=\sup_{x\in[0,1]}\lvert f'(x)\rvert.
$$

Constant nonzero functions have $p(f)=0$, so this is not a norm on the whole space.

Seminorms induce pseudometrics:

$$
d_p(f,g)=p(f-g),
$$

where distinct objects can have distance zero. The usual fix is a quotient: identify objects whose difference has seminorm zero. This is exactly the kind of idea behind $L^p$ spaces, where functions equal almost everywhere are treated as the same element.

### 2.4 Examples and Non-Examples

**Examples of norms.**

1. $\mathbb{R}^n$ with $\ell^1$, $\ell^2$, or $\ell^\infty$ norm.
2. $C[0,1]$ with the sup norm

$$
\lVert f \rVert_\infty = \sup_{x\in[0,1]}\lvert f(x)\rvert.
$$

3. Matrix Frobenius norm

$$
\lVert A \rVert_F = \sqrt{\sum_{i,j} A_{ij}^2}.
$$

4. Operator spectral norm

$$
\lVert A \rVert_2
= \sup_{\mathbf{x}\neq \mathbf{0}}
\frac{\lVert A\mathbf{x}\rVert_2}{\lVert \mathbf{x}\rVert_2}.
$$

**Non-examples.**

1. $\lVert \mathbf{x} \rVert_0$, the number of nonzero coordinates, is not homogeneous.
2. $\left(\sum_i \lvert x_i\rvert^p\right)^{1/p}$ for $0<p<1$ violates the triangle inequality.
3. $p(f)=\lvert f(0)\rvert$ is a seminorm on function space, not a norm.
4. $d(x,y)=\min(\lvert x-y\rvert,1)$ is a metric on $\mathbb{R}$ but not induced by a norm because it is bounded.

### 2.5 Open Balls, Closed Balls, Neighborhoods, and Topology

In a normed space, the open ball centered at $\mathbf{x}$ with radius $r>0$ is

$$
B(\mathbf{x},r)
=
\{\mathbf{y}\in V : \lVert \mathbf{y}-\mathbf{x}\rVert < r\}.
$$

The closed ball is

$$
\overline{B}(\mathbf{x},r)
=
\{\mathbf{y}\in V : \lVert \mathbf{y}-\mathbf{x}\rVert \leq r\}.
$$

These balls define the topology: a set $U$ is open if every point in $U$ has a small open ball contained in $U$. This is how a norm turns vector algebra into a space where limits, continuity, and compactness can be studied.

---

## 3. Core Theory I: Common Norms and Geometry

### 3.1 $\ell^p$ Norms on $\mathbb{R}^n$

For $1\leq p<\infty$, define

$$
\lVert \mathbf{x} \rVert_p
=
\left(\sum_{i=1}^n \lvert x_i\rvert^p\right)^{1/p}.
$$

For $p=\infty$,

$$
\lVert \mathbf{x} \rVert_\infty
=
\max_{1\leq i\leq n}\lvert x_i\rvert.
$$

Important cases:

| Norm | Formula | ML interpretation |
| --- | --- | --- |
| $\ell^1$ | $\sum_i \lvert x_i\rvert$ | sparsity and feature selection |
| $\ell^2$ | $\sqrt{\sum_i x_i^2}$ | Euclidean energy and weight decay |
| $\ell^\infty$ | $\max_i \lvert x_i\rvert$ | worst-coordinate perturbation |

The $\ell^2$ norm is special because it comes from the Euclidean inner product:

$$
\lVert \mathbf{x} \rVert_2
=
\sqrt{\langle \mathbf{x},\mathbf{x}\rangle}.
$$

But not every norm comes from an inner product. That distinction is the doorway to Hilbert spaces.

### 3.2 Why $\ell^0$ Is Not a Norm

The quantity

$$
\lVert \mathbf{x} \rVert_0
=
\#\{i : x_i\neq 0\}
$$

counts nonzero entries. It is useful in sparse modeling, but it is not a norm because homogeneity fails.

Take $\mathbf{x}=(1,0,0)$ and $\alpha=2$. Then

$$
\lVert \alpha\mathbf{x} \rVert_0 = 1,
\qquad
\lvert \alpha\rvert \lVert \mathbf{x} \rVert_0 = 2.
$$

So

$$
\lVert \alpha\mathbf{x} \rVert_0
\neq
\lvert \alpha\rvert \lVert \mathbf{x} \rVert_0.
$$

This is why $\ell^0$ optimization is combinatorial and nonconvex. The $\ell^1$ norm is often used as a convex surrogate.

### 3.3 Matrix Norms

For a matrix $A\in\mathbb{R}^{m\times n}$, the Frobenius norm is

$$
\lVert A\rVert_F
=
\sqrt{\sum_{i=1}^m\sum_{j=1}^n A_{ij}^2}
=
\sqrt{\operatorname{tr}(A^\top A)}.
$$

The spectral norm is the induced $\ell^2$ operator norm:

$$
\lVert A\rVert_2
=
\sup_{\lVert \mathbf{x}\rVert_2=1}
\lVert A\mathbf{x}\rVert_2
=
\sigma_{\max}(A).
$$

The nuclear norm is

$$
\lVert A\rVert_*
=
\sum_i \sigma_i(A).
$$

These norms measure different properties:

- Frobenius norm: total energy of entries
- Spectral norm: largest possible stretch
- Nuclear norm: sum of singular values, a convex proxy for rank

**For AI:** Spectral norm controls worst-case amplification by a linear layer. Frobenius norm appears in weight decay. Nuclear norm appears in low-rank regularization and matrix completion.

### 3.4 Norm Equivalence in Finite Dimensions

On a finite-dimensional vector space, any two norms are equivalent. For norms $\lVert\cdot\rVert_a$ and $\lVert\cdot\rVert_b$ on $\mathbb{R}^n$, there exist constants $c,C>0$ such that

$$
c\lVert \mathbf{x}\rVert_a
\leq
\lVert \mathbf{x}\rVert_b
\leq
C\lVert \mathbf{x}\rVert_a
\quad
\text{for all } \mathbf{x}\in\mathbb{R}^n.
$$

For common norms:

$$
\lVert \mathbf{x}\rVert_\infty
\leq
\lVert \mathbf{x}\rVert_2
\leq
\sqrt{n}\lVert \mathbf{x}\rVert_\infty,
$$

and

$$
\lVert \mathbf{x}\rVert_2
\leq
\lVert \mathbf{x}\rVert_1
\leq
\sqrt{n}\lVert \mathbf{x}\rVert_2.
$$

The constants depend on dimension. This dependence matters in high-dimensional ML: a bound that is harmless for $n=10$ can be loose for $n=10^6$.

### 3.5 Unit Ball Geometry and Convexity

The unit ball of every norm is convex:

$$
\mathbf{x},\mathbf{y}\in B,
\quad
0\leq \lambda\leq 1
\quad\Rightarrow\quad
\lambda \mathbf{x}+(1-\lambda)\mathbf{y}\in B.
$$

Proof:

$$
\lVert \lambda \mathbf{x}+(1-\lambda)\mathbf{y}\rVert
\leq
\lambda\lVert \mathbf{x}\rVert
+(1-\lambda)\lVert \mathbf{y}\rVert
\leq 1.
$$

For $0<p<1$, the would-be $\ell^p$ "ball" is nonconvex. This is another way to see why it is not a norm.

---

## 4. Core Theory II: Convergence and Completeness

### 4.1 Convergence in a Normed Space

A sequence $(\mathbf{x}_k)$ in a normed space $V$ converges to $\mathbf{x}\in V$ if

$$
\lVert \mathbf{x}_k-\mathbf{x}\rVert \to 0.
$$

This means:

$$
\forall \epsilon>0,\ \exists N
\text{ such that }
k\geq N
\Rightarrow
\lVert \mathbf{x}_k-\mathbf{x}\rVert < \epsilon.
$$

The norm defines what it means to be close. In function spaces, this choice is critical. Pointwise convergence may ignore large local spikes; uniform convergence does not; $L^2$ convergence averages squared error.

### 4.2 Cauchy Sequences

A sequence $(\mathbf{x}_k)$ is **Cauchy** if

$$
\forall \epsilon>0,\ \exists N
\text{ such that }
m,n\geq N
\Rightarrow
\lVert \mathbf{x}_m-\mathbf{x}_n\rVert<\epsilon.
$$

Every convergent sequence is Cauchy. The converse depends on the space. A Cauchy sequence contains enough internal evidence that it should converge, but the limit may live outside the space.

Example: rational numbers $\mathbb{Q}$ with the usual absolute value are not complete. A rational sequence can converge to $\sqrt{2}$ in $\mathbb{R}$, but $\sqrt{2}\notin\mathbb{Q}$.

### 4.3 Banach Spaces

A **Banach space** is a complete normed space: every Cauchy sequence converges to an element of the space.

Completeness is not a decorative property. It is what makes many existence theorems work. If an iterative algorithm produces a Cauchy sequence, completeness guarantees the limit is actually in the model space.

Examples of Banach spaces:

- $\mathbb{R}^n$ with any norm
- $\ell^p$ for $1\leq p\leq\infty$
- $C[0,1]$ with $\lVert f\rVert_\infty$
- $L^p(\Omega)$ for $1\leq p\leq\infty$

Non-example:

- $C[0,1]$ with the $L^1$ norm is not complete; its completion is related to $L^1[0,1]$.

### 4.4 Examples: $\mathbb{R}^n$, $C[0,1]$, $\ell^p$, and $L^p$

For sequences, define

$$
\ell^p
=
\left\{
\mathbf{x}=(x_1,x_2,\dots)
:
\sum_{i=1}^{\infty}\lvert x_i\rvert^p < \infty
\right\}
$$

with norm

$$
\lVert \mathbf{x}\rVert_p
=
\left(\sum_{i=1}^{\infty}\lvert x_i\rvert^p\right)^{1/p}.
$$

For functions,

$$
L^p(\Omega)
=
\left\{
f:
\int_\Omega \lvert f(x)\rvert^p\,dx < \infty
\right\},
$$

with

$$
\lVert f\rVert_p
=
\left(\int_\Omega \lvert f(x)\rvert^p\,dx\right)^{1/p}.
$$

Strictly speaking, $L^p$ elements are equivalence classes of functions equal almost everywhere. This quotient viewpoint prevents a function that differs only on a measure-zero set from being a different element at distance zero.

### 4.5 Completion of a Normed Space

Every normed space has a completion. Informally, we add in all missing limits of Cauchy sequences. The real numbers complete the rationals. $L^p$ spaces complete suitable spaces of simple or continuous functions under $L^p$ norms.

The completion matters in ML because practical models often approximate ideal objects. A finite network may approximate a function in a larger function space. A discretized algorithm may approximate a continuous operator. Norms specify what kind of approximation is being made.

---

## 5. Core Theory III: Operators, Lipschitz Maps, and Dual Norms

### 5.1 Bounded Linear Operators

Let $V$ and $W$ be normed spaces. A linear map $T:V\to W$ is **bounded** if there exists $C\geq 0$ such that

$$
\lVert T\mathbf{v}\rVert_W
\leq
C\lVert \mathbf{v}\rVert_V
\quad
\text{for all } \mathbf{v}\in V.
$$

The smallest such $C$ is the operator norm:

$$
\lVert T\rVert
=
\sup_{\mathbf{v}\neq \mathbf{0}}
\frac{\lVert T\mathbf{v}\rVert_W}{\lVert \mathbf{v}\rVert_V}
=
\sup_{\lVert \mathbf{v}\rVert_V=1}
\lVert T\mathbf{v}\rVert_W.
$$

For a matrix $A$ acting on Euclidean space, this recovers the spectral norm when both domain and codomain use $\ell^2$.

### 5.2 Bounded if and only if Continuous for Linear Maps

For linear maps between normed spaces:

$$
T \text{ is bounded}
\iff
T \text{ is continuous}
\iff
T \text{ is continuous at } \mathbf{0}.
$$

Proof sketch:

If $T$ is bounded, then

$$
\lVert T\mathbf{x}-T\mathbf{y}\rVert
=
\lVert T(\mathbf{x}-\mathbf{y})\rVert
\leq
\lVert T\rVert \lVert \mathbf{x}-\mathbf{y}\rVert,
$$

so $T$ is Lipschitz and therefore continuous.

Conversely, if $T$ is continuous at $\mathbf{0}$, there exists $\delta>0$ such that $\lVert \mathbf{v}\rVert<\delta$ implies $\lVert T\mathbf{v}\rVert<1$. For arbitrary nonzero $\mathbf{x}$, scale it into the $\delta$ ball and rearrange to get a global bound.

### 5.3 Operator Norms and Neural-Network Lipschitz Constants

For a neural network layer

$$
\mathbf{h}_{l+1} = \sigma(W_l\mathbf{h}_l+\mathbf{b}_l),
$$

if $\sigma$ is $L_\sigma$-Lipschitz and $W_l$ has operator norm $\lVert W_l\rVert_2$, then the layer is at most

$$
L_\sigma \lVert W_l\rVert_2
$$

Lipschitz under $\ell^2$. A composition of layers has Lipschitz constant bounded by the product:

$$
L_{\text{net}}
\leq
\prod_l L_{\sigma_l}\lVert W_l\rVert_2.
$$

This bound can be loose, but it explains why spectral normalization and operator-norm control are central to robustness.

### 5.4 Dual Spaces and Dual Norms

The **dual space** $V^*$ is the space of bounded linear functionals $f:V\to\mathbb{R}$ or $f:V\to\mathbb{C}$.

The dual norm is

$$
\lVert f\rVert_{V^*}
=
\sup_{\lVert \mathbf{x}\rVert_V\leq 1}
\lvert f(\mathbf{x})\rvert.
$$

For finite-dimensional $\mathbb{R}^n$, a vector $\mathbf{y}$ defines a functional

$$
f_{\mathbf{y}}(\mathbf{x})=\langle \mathbf{y},\mathbf{x}\rangle.
$$

The dual norm is

$$
\lVert \mathbf{y}\rVert_*
=
\sup_{\lVert \mathbf{x}\rVert\leq 1}
\lvert \langle \mathbf{y},\mathbf{x}\rangle\rvert.
$$

Dual pairs:

| Primal norm | Dual norm |
| --- | --- |
| $\ell^1$ | $\ell^\infty$ |
| $\ell^2$ | $\ell^2$ |
| $\ell^\infty$ | $\ell^1$ |
| $\ell^p$ | $\ell^q$ with $\frac{1}{p}+\frac{1}{q}=1$ |

### 5.5 Holder Inequality and Regularization Geometry

Holder's inequality states:

$$
\lvert \langle \mathbf{x},\mathbf{y}\rangle\rvert
\leq
\lVert \mathbf{x}\rVert_p
\lVert \mathbf{y}\rVert_q,
\qquad
\frac{1}{p}+\frac{1}{q}=1.
$$

This is the central inequality behind dual norms. It also explains regularization geometry. If a model constrains $\lVert \boldsymbol{\theta}\rVert_p$, then the maximum linear response to a feature vector depends on the dual norm of that feature vector.

For example:

$$
\sup_{\lVert \boldsymbol{\theta}\rVert_1\leq R}
\langle \boldsymbol{\theta},\mathbf{x}\rangle
=
R\lVert \mathbf{x}\rVert_\infty.
$$

That is why $\ell^1$ constraints interact naturally with max-coordinate behavior.

---

## 6. Advanced Topics

### 6.1 Banach Fixed-Point Theorem and Contractions

A map $T:X\to X$ on a normed space is a contraction if there exists $0\leq \gamma<1$ such that

$$
\lVert T\mathbf{x}-T\mathbf{y}\rVert
\leq
\gamma \lVert \mathbf{x}-\mathbf{y}\rVert
$$

for all $\mathbf{x},\mathbf{y}\in X$.

**Banach fixed-point theorem.** If $X$ is a nonempty complete metric space and $T:X\to X$ is a contraction, then:

1. $T$ has a unique fixed point $\mathbf{x}^*$ satisfying $T\mathbf{x}^*=\mathbf{x}^*$.
2. For any starting point $\mathbf{x}_0$, the iteration

$$
\mathbf{x}_{k+1}=T\mathbf{x}_k
$$

converges to $\mathbf{x}^*$.

This theorem is one of the reasons completeness matters.

### 6.2 Compactness: Finite vs Infinite Dimension

In $\mathbb{R}^n$, closed and bounded sets are compact. In infinite-dimensional normed spaces, this fails. The closed unit ball in an infinite-dimensional Banach space is generally not compact in the norm topology.

This failure changes analysis. In finite dimensions, many optimization arguments rely on compactness. In function spaces, existence of minimizers can require extra structure: weak compactness, coercivity, lower semicontinuity, or regularization.

### 6.3 Uniform, Pointwise, and Norm Convergence

For functions $f_n:[0,1]\to\mathbb{R}$, pointwise convergence means

$$
f_n(x)\to f(x)
\quad
\text{for each fixed } x.
$$

Uniform convergence means

$$
\lVert f_n-f\rVert_\infty
=
\sup_{x\in[0,1]}\lvert f_n(x)-f(x)\rvert
\to 0.
$$

$L^2$ convergence means

$$
\lVert f_n-f\rVert_2
=
\left(\int_0^1 \lvert f_n(x)-f(x)\rvert^2\,dx\right)^{1/2}
\to 0.
$$

These are different. A spike can vanish in $L^2$ while failing to vanish uniformly. This distinction matters for function approximation, density estimation, and PDE-inspired ML.

### 6.4 Banach-Space Geometry: Strict Convexity and Smoothness

A normed space is **strictly convex** if the boundary of its unit ball contains no line segments. In such spaces, midpoint equality on the unit sphere is rigid.

$\ell^2$ is strictly convex. $\ell^1$ and $\ell^\infty$ are not strictly convex in dimension at least $2$ because their unit balls have flat edges.

Smoothness of the norm is another geometric property. It affects uniqueness of supporting hyperplanes and the behavior of duality maps. These ideas become important in advanced optimization and Banach-space learning theory.

### 6.5 Bridge to Hilbert Spaces

Every inner product induces a norm:

$$
\lVert \mathbf{x}\rVert
=
\sqrt{\langle \mathbf{x},\mathbf{x}\rangle}.
$$

But not every norm comes from an inner product. A norm comes from an inner product if and only if it satisfies the parallelogram law:

$$
\lVert \mathbf{x}+\mathbf{y}\rVert^2
+
\lVert \mathbf{x}-\mathbf{y}\rVert^2
=
2\lVert \mathbf{x}\rVert^2
+
2\lVert \mathbf{y}\rVert^2.
$$

This is the conceptual bridge to [Hilbert Spaces](../02-Hilbert-Spaces/notes.md), where angles, orthogonality, projections, Fourier expansions, and RKHS theory become available.

---

## 7. Applications in Machine Learning

### 7.1 $L^1$, $L^2$, and Elastic-Net Regularization

Norm-based regularization adds a penalty to a training objective:

$$
\min_{\boldsymbol{\theta}}
\mathcal{L}(\boldsymbol{\theta})
+
\lambda R(\boldsymbol{\theta}).
$$

Common choices:

| Regularizer | Penalty | Effect |
| --- | --- | --- |
| Ridge | $\lVert \boldsymbol{\theta}\rVert_2^2$ | smooth shrinkage |
| Lasso | $\lVert \boldsymbol{\theta}\rVert_1$ | sparse solutions |
| Elastic net | $\alpha\lVert \boldsymbol{\theta}\rVert_1+(1-\alpha)\lVert \boldsymbol{\theta}\rVert_2^2$ | sparsity plus stability |

The geometry of the norm ball explains the optimizer's bias. Sharp corners encourage sparse coordinates; round balls shrink smoothly.

### 7.2 Adversarial Perturbations in $\ell^\infty$ and $\ell^2$ Balls

An adversarial perturbation is often constrained by a norm:

$$
\lVert \boldsymbol{\delta}\rVert_p \leq \epsilon.
$$

The adversarial example is

$$
\mathbf{x}_{\mathrm{adv}}
=
\mathbf{x}+\boldsymbol{\delta}.
$$

Different $p$ define different threat models:

- $\ell^\infty$: every coordinate can change by at most $\epsilon$
- $\ell^2$: total perturbation energy is bounded
- $\ell^1$: sparse perturbation budget

Dual norms explain first-order attacks. If the loss gradient is $\mathbf{g}$, then the largest first-order loss increase over an $\ell^p$ ball is controlled by $\lVert \mathbf{g}\rVert_q$, where $q$ is dual to $p$.

### 7.3 Gradient Clipping and Update Control

Gradient clipping enforces

$$
\widetilde{\mathbf{g}}
=
\mathbf{g}\cdot
\min\left(1,\frac{C}{\lVert \mathbf{g}\rVert_2}\right).
$$

Then

$$
\lVert \widetilde{\mathbf{g}}\rVert_2 \leq C.
$$

This is norm control on the update direction. It prevents a single mini-batch from producing a catastrophic parameter jump. In RNNs, transformers, reinforcement learning, and mixed-precision training, clipping is often a stability guard.

### 7.4 Spectral Normalization and Lipschitz Control

For a linear layer $W$, spectral normalization rescales

$$
\overline{W}
=
\frac{W}{\lVert W\rVert_2}.
$$

Then $\lVert \overline{W}\rVert_2=1$ when the norm estimate is exact. Since

$$
\lVert W\mathbf{x}-W\mathbf{y}\rVert_2
\leq
\lVert W\rVert_2\lVert \mathbf{x}-\mathbf{y}\rVert_2,
$$

controlling $\lVert W\rVert_2$ controls worst-case linear amplification.

### 7.5 Fixed-Point Iterations in RL, Diffusion, and Implicit Layers

In discounted reinforcement learning, the Bellman operator $T$ is a contraction in the sup norm:

$$
\lVert TV-TU\rVert_\infty
\leq
\gamma \lVert V-U\rVert_\infty,
\qquad
0\leq \gamma<1.
$$

Therefore value iteration converges to a unique fixed point.

Implicit layers and equilibrium models similarly search for a state $\mathbf{z}^*$ satisfying

$$
\mathbf{z}^* = F(\mathbf{z}^*,\mathbf{x}).
$$

If $F$ is a contraction in the state variable, fixed-point iteration is stable and convergent.

---

## 8. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
| --- | --- | --- | --- |
| 1 | "A norm is just any distance." | A norm is a vector-size function; it induces a special translation-invariant metric. | Distinguish norms from general metrics. |
| 2 | "$\ell^0$ is a norm because people call it one." | It violates homogeneity. | Call it the $\ell^0$ count or pseudo-norm. |
| 3 | "All norms behave the same." | Norms may be equivalent in finite dimension, but constants and geometry matter. | Track dimension and unit-ball shape. |
| 4 | "Infinite-dimensional norms are all equivalent too." | Norm equivalence fails in infinite-dimensional spaces. | Specify the function-space norm. |
| 5 | "Every norm has angles." | Angles require an inner product, not just a norm. | Use Hilbert-space structure when angles matter. |
| 6 | "Bounded operator means bounded output set." | Bounded linear operator means $\lVert T\mathbf{x}\rVert \leq C\lVert \mathbf{x}\rVert$. | Use the operator inequality. |
| 7 | "Pointwise convergence is enough for learning theory." | It may ignore spikes and does not imply norm convergence. | Choose convergence mode deliberately. |
| 8 | "Closed and bounded means compact in any normed space." | This is finite-dimensional behavior. | In infinite dimensions, compactness needs extra care. |
| 9 | "Gradient clipping solves all instability." | It bounds update norm but may hide bad scaling or poor objectives. | Log norms and diagnose root causes. |
| 10 | "Spectral norm bounds are exact network robustness certificates." | Product bounds can be loose. | Treat them as useful upper bounds. |
| 11 | "Dual norms are abstract and optional." | Dual norms control worst-case linear response and adversarial directions. | Use duality for perturbation and regularization arguments. |
| 12 | "Completeness is technical bookkeeping." | Completeness guarantees limits of Cauchy iterations remain in the space. | Check Banach assumptions before applying fixed-point theorems. |

---

## 9. Exercises

1. **Exercise 1 (Easy): Verify Norm Axioms**
   Check the norm axioms for $\ell^1$, $\ell^2$, and $\ell^\infty$ on $\mathbb{R}^3$ using both symbolic reasoning and numerical tests.

2. **Exercise 2 (Easy): Find a Non-Norm Counterexample**
   Show that $\ell^0$ and $\ell^p$ with $0<p<1$ fail to be norms.

3. **Exercise 3 (Easy): Unit Ball Geometry**
   Plot $\ell^1$, $\ell^2$, and $\ell^\infty$ unit balls and explain how their geometry affects sparse regularization.

4. **Exercise 4 (Medium): Norm Equivalence Constants**
   Derive and numerically verify inequalities between $\ell^1$, $\ell^2$, and $\ell^\infty$ norms.

5. **Exercise 5 (Medium): Cauchy and Completeness**
   Construct a Cauchy sequence of rational numbers converging to an irrational limit and explain why $\mathbb{Q}$ is not Banach.

6. **Exercise 6 (Medium): Operator Norm**
   Compute the spectral norm of a matrix using singular values and verify the induced-norm definition by sampling.

7. **Exercise 7 (Hard): Dual Norms and Holder**
   Compute dual norms for $\ell^1$, $\ell^2$, and $\ell^\infty$ and verify Holder's inequality numerically.

8. **Exercise 8 (Hard): Contraction Mapping**
   Implement a contraction iteration and verify convergence to the unique fixed point.

---

## 10. Why This Matters for AI

| Normed-space concept | AI impact |
| --- | --- |
| Norm axioms | Define valid size measures for parameters, activations, and functions |
| Unit balls | Explain regularization geometry and adversarial threat models |
| Norm equivalence | Shows finite-dimensional convergence can be robust to norm choice, but constants matter |
| Banach spaces | Provide complete settings for iterative algorithms |
| Operator norms | Bound layer Lipschitz constants and perturbation amplification |
| Dual norms | Describe worst-case linear response and adversarial directions |
| Fixed-point theorem | Guarantees convergence for contractions in RL and implicit models |
| Function norms | Clarify what it means for learned functions to approximate targets |
| Completeness | Ensures limits of learning or approximation sequences remain inside the space |
| Bridge to Hilbert spaces | Prepares for kernels, projections, and RKHS methods |

Normed spaces matter because AI is full of "small change should not cause disaster" claims. Norms turn those claims into mathematics.

---

## 11. Conceptual Bridge

Normed spaces sit between linear algebra and Hilbert-space geometry. From linear algebra we inherit vector spaces, linear maps, matrices, and subspaces. By adding a norm, we gain convergence, continuity, boundedness, completeness, duality, and fixed-point reasoning.

The next section, [Hilbert Spaces](../02-Hilbert-Spaces/notes.md), adds inner products. This gives angles, orthogonality, projections, Fourier expansions, and the geometry behind kernels and attention. Every Hilbert space is a normed space, but not every normed space is Hilbert.

Kernel methods then use Hilbert-space structure to turn nonlinear learning into linear learning in feature space. Without normed-space foundations, the RKHS norm, representer theorem, and kernel regularization are just formulas. With normed spaces, they become part of a larger story about geometry, stability, and approximation.

```text
FUNCTIONAL ANALYSIS PATH
============================================================

  Vector spaces
      algebra: add vectors, scale vectors
          |
          v
  Normed spaces
      size, distance, convergence, bounded operators
          |
          v
  Banach spaces
      completeness and fixed-point theorems
          |
          v
  Hilbert spaces
      inner products, angles, projections
          |
          v
  RKHS / Kernel methods
      function-space learning with computable inner products

============================================================
```

## References

1. Rodriguez, C. (2021). *18.102 Introduction to Functional Analysis*. MIT OpenCourseWare. https://ocw.mit.edu/courses/18-102-introduction-to-functional-analysis-spring-2021/
2. MIT OpenCourseWare. *Functional Analysis Lecture Notes, Spring 2020*. https://ocw.mit.edu/courses/18-102-introduction-to-functional-analysis-spring-2021/3d4cc88026d44a01f936cd6a0aa995cb_MIT18_102s20_lec_FA.pdf
3. Ye, X. *Lecture Notes on Functional Analysis*. Georgia State University. https://math.gsu.edu/xye/course/fa_handout/fa_notes.pdf
4. Kreyszig, E. (1989). *Introductory Functional Analysis with Applications*. Wiley.
5. Conway, J. B. (1990). *A Course in Functional Analysis*. Springer.
6. Rudin, W. (1991). *Functional Analysis*. McGraw-Hill.
7. Boyd, S., and Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.
8. Goodfellow, I., Bengio, Y., and Courville, A. (2016). *Deep Learning*. MIT Press.
