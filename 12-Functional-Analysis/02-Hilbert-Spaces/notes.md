[Back to Functional Analysis](../README.md) | [Previous: Normed Spaces](../01-Normed-Spaces/notes.md) | [Next: Kernel Methods](../03-Kernel-Methods/notes.md)

---

# Hilbert Spaces

> Inner products turn size into geometry. Completeness makes that geometry stable under limits.

## Overview

Hilbert spaces are complete inner product spaces. They keep the geometry of Euclidean vectors while allowing infinite-dimensional objects such as functions, signals, square-summable sequences, random variables, and feature maps. Normed spaces let us measure size. Hilbert spaces add angle, orthogonality, projection, Fourier coordinates, and self-duality.

This section is the bridge between the normed-space foundations of the previous section and the kernel methods of the next section. The central message is practical: whenever a learning algorithm uses dot products, cosine similarity, least squares, PCA, attention scores, Fourier features, Gaussian processes, or kernelized optimization, it is leaning on Hilbert-space structure.

We focus on the core Hilbert toolkit:

- inner products and induced norms
- Cauchy-Schwarz, Pythagoras, and orthogonality
- projection theorem and least squares
- orthonormal systems, Bessel inequality, Parseval identity, and Fourier-Bessel expansion
- Riesz representation and gradients as vectors
- adjoints, self-adjoint operators, positive operators, compact operators, and spectral decomposition
- careful bridges to RKHS, kernels, Fourier analysis, PCA, and neural tangent kernels

Kernel methods, positive definite kernels, support vector machines, Gaussian processes, Mercer expansions, and scalable kernel approximations are only previewed here. They are developed in detail in [Kernel Methods](../03-Kernel-Methods/notes.md).

## Prerequisites

- **Normed spaces and completeness** - [Normed Spaces](../01-Normed-Spaces/notes.md)
- **Vector spaces, bases, and linear maps** - [Linear Algebra Basics](../../02-Linear-Algebra-Basics/README.md)
- **Orthogonality in finite dimensions** - [Orthogonality and Orthonormality](../../03-Advanced-Linear-Algebra/05-Orthogonality-and-Orthonormality/notes.md)
- **Matrix norms and singular values** - [Matrix Norms](../../03-Advanced-Linear-Algebra/06-Matrix-Norms/notes.md)
- **Least squares and convex optimization** - [Convex Optimization](../../08-Optimization/01-Convex-Optimization/notes.md)
- **Random variables and expectations** - useful for $L^2$ spaces and Gaussian-process intuition

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Interactive Hilbert geometry, projections, Gram-Schmidt, Parseval checks, Riesz gradients, adjoints, PCA, and kernel previews |
| [exercises.ipynb](exercises.ipynb) | 8 graded exercises covering inner products, projections, bases, Riesz representation, operators, PCA, and RKHS previews |

## Learning Objectives

After completing this section, you will be able to:

1. Define real and complex inner products and the norm they induce
2. Distinguish pre-Hilbert spaces from Hilbert spaces
3. Prove and apply Cauchy-Schwarz, Pythagoras, and the parallelogram law
4. Use orthogonal complements to decompose Hilbert spaces
5. Apply the projection theorem to closed subspaces and least-squares problems
6. Build orthonormal systems with Gram-Schmidt
7. Use Bessel inequality, Parseval identity, and Fourier-Bessel coordinates
8. State and apply the Riesz representation theorem
9. Interpret gradients as Riesz representatives of differentials
10. Work with adjoints, self-adjoint operators, positive operators, and compact operators
11. Explain how Hilbert geometry supports attention, PCA, ridge regression, Gaussian processes, RKHS theory, and infinite-width neural-network limits

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 From Norms to Angles](#11-from-norms-to-angles)
  - [1.2 Why Completeness Matters](#12-why-completeness-matters)
  - [1.3 Why Hilbert Spaces Matter for AI](#13-why-hilbert-spaces-matter-for-ai)
  - [1.4 Historical Timeline](#14-historical-timeline)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Real and Complex Inner Products](#21-real-and-complex-inner-products)
  - [2.2 Induced Norm and Metric](#22-induced-norm-and-metric)
  - [2.3 Hilbert and Pre-Hilbert Spaces](#23-hilbert-and-pre-hilbert-spaces)
  - [2.4 Examples and Non-Examples](#24-examples-and-non-examples)
  - [2.5 Parallelogram Law and Polarization Identity](#25-parallelogram-law-and-polarization-identity)
- [3. Core Theory I: Inner Product Geometry](#3-core-theory-i-inner-product-geometry)
  - [3.1 Cauchy-Schwarz Inequality](#31-cauchy-schwarz-inequality)
  - [3.2 Angles, Cosine Similarity, and Orthogonality](#32-angles-cosine-similarity-and-orthogonality)
  - [3.3 Pythagorean Theorem](#33-pythagorean-theorem)
  - [3.4 Orthogonal Complements and Closed Subspaces](#34-orthogonal-complements-and-closed-subspaces)
  - [3.5 Best Approximation Intuition](#35-best-approximation-intuition)
- [4. Core Theory II: Projection Theorem](#4-core-theory-ii-projection-theorem)
  - [4.1 Projection onto Closed Convex Sets](#41-projection-onto-closed-convex-sets)
  - [4.2 Orthogonal Projection onto Closed Subspaces](#42-orthogonal-projection-onto-closed-subspaces)
  - [4.3 Least Squares as Projection](#43-least-squares-as-projection)
  - [4.4 Projection Operators](#44-projection-operators)
  - [4.5 Projection Algorithms in ML](#45-projection-algorithms-in-ml)
- [5. Core Theory III: Orthonormal Systems and Bases](#5-core-theory-iii-orthonormal-systems-and-bases)
  - [5.1 Orthonormal Sets and Gram-Schmidt](#51-orthonormal-sets-and-gram-schmidt)
  - [5.2 Bessel Inequality](#52-bessel-inequality)
  - [5.3 Complete Orthonormal Systems and Hilbert Bases](#53-complete-orthonormal-systems-and-hilbert-bases)
  - [5.4 Parseval Identity and Fourier-Bessel Expansion](#54-parseval-identity-and-fourier-bessel-expansion)
  - [5.5 Separability and $\ell^2$ Coordinates](#55-separability-and-ell2-coordinates)
- [6. Core Theory IV: Riesz Representation and Duality](#6-core-theory-iv-riesz-representation-and-duality)
  - [6.1 Bounded Linear Functionals](#61-bounded-linear-functionals)
  - [6.2 Riesz Representation Theorem](#62-riesz-representation-theorem)
  - [6.3 Hilbert Spaces Are Self-Dual](#63-hilbert-spaces-are-self-dual)
  - [6.4 Gradients as Riesz Representatives](#64-gradients-as-riesz-representatives)
  - [6.5 Why This Matters for Optimization and Backpropagation](#65-why-this-matters-for-optimization-and-backpropagation)
- [7. Core Theory V: Operators on Hilbert Spaces](#7-core-theory-v-operators-on-hilbert-spaces)
  - [7.1 Bounded Operators and Operator Norms](#71-bounded-operators-and-operator-norms)
  - [7.2 Adjoints and Self-Adjoint Operators](#72-adjoints-and-self-adjoint-operators)
  - [7.3 Positive Operators and Quadratic Forms](#73-positive-operators-and-quadratic-forms)
  - [7.4 Compact Operators](#74-compact-operators)
  - [7.5 Spectral Theorem for Compact Self-Adjoint Operators](#75-spectral-theorem-for-compact-self-adjoint-operators)
- [8. Advanced Topics and Bridges](#8-advanced-topics-and-bridges)
  - [8.1 Weak Convergence](#81-weak-convergence)
  - [8.2 Fourier Analysis as Hilbert-Space Coordinates](#82-fourier-analysis-as-hilbert-space-coordinates)
  - [8.3 RKHS and Reproducing Kernels](#83-rkhs-and-reproducing-kernels)
  - [8.4 Kernel Gradient Flow and Neural Tangent Kernels](#84-kernel-gradient-flow-and-neural-tangent-kernels)
  - [8.5 Continuous vs Discrete Spectra](#85-continuous-vs-discrete-spectra)
- [9. Applications in Machine Learning](#9-applications-in-machine-learning)
- [10. Common Mistakes](#10-common-mistakes)
- [11. Exercises](#11-exercises)
- [12. Why This Matters for AI](#12-why-this-matters-for-ai)
- [13. Conceptual Bridge](#13-conceptual-bridge)
- [References](#references)

---

## 1. Intuition

### 1.1 From Norms to Angles

A normed space tells us how large a vector is. A Hilbert space tells us not only how large vectors are, but also how they face each other.

The extra structure is the inner product:

$$
\langle \mathbf{x}, \mathbf{y} \rangle.
$$

In $\mathbb{R}^n$, the standard inner product is

$$
\langle \mathbf{x}, \mathbf{y} \rangle
= \mathbf{x}^\top \mathbf{y}
= \sum_{i=1}^n x_i y_i.
$$

This single operation gives length,

$$
\lVert \mathbf{x} \rVert = \sqrt{\langle \mathbf{x}, \mathbf{x} \rangle},
$$

angle,

$$
\cos \theta
=
\frac{\langle \mathbf{x}, \mathbf{y} \rangle}
{\lVert \mathbf{x} \rVert \lVert \mathbf{y} \rVert},
$$

and orthogonality,

$$
\mathbf{x} \perp \mathbf{y}
\quad \Longleftrightarrow \quad
\langle \mathbf{x}, \mathbf{y} \rangle = 0.
$$

The philosophical shift is small but powerful:

```text
normed space:
  vector + size + convergence

Hilbert space:
  vector + size + convergence + angle + projection + coordinates
```

In machine learning, this is why dot products can act as similarity scores, why least squares has a clean geometric solution, why PCA is an orthogonal coordinate system, and why Fourier analysis can decompose functions into energy-preserving frequency components.

### 1.2 Why Completeness Matters

A Hilbert space is not just an inner product space. It is a complete inner product space. Completeness means every Cauchy sequence converges to a point inside the same space.

This matters because optimization, approximation, and learning algorithms often produce limits. If the space is incomplete, an algorithm can converge toward an object that does not live in the space being studied.

Let $\mathcal{P}[0,1]$ be the polynomials on $[0,1]$ with inner product

$$
\langle f,g \rangle = \int_0^1 f(t)g(t)\,dt.
$$

This is an inner product space, but it is not complete. A sequence of polynomials can be Cauchy in the $L^2$ norm and converge to a square-integrable function that is not a polynomial. The completion is $L^2[0,1]$.

For AI, incompleteness shows up whenever we approximate functions by finite models but reason about limiting function classes. A Hilbert space is the stable mathematical container for those limits.

### 1.3 Why Hilbert Spaces Matter for AI

Hilbert-space ideas are everywhere in modern ML:

| ML concept | Hilbert-space idea |
| --- | --- |
| attention score $\mathbf{q}^\top \mathbf{k}$ | inner product as alignment |
| cosine similarity | normalized Hilbert angle |
| least squares | orthogonal projection |
| ridge regression | projected or regularized Hilbert problem |
| PCA | spectral theorem for self-adjoint covariance operators |
| Fourier features | coordinates in an orthonormal system |
| Gaussian processes | covariance kernels and function-space geometry |
| kernel methods | implicit inner products in feature Hilbert spaces |
| gradient descent in function space | Riesz representation of differentials |
| neural tangent kernel | kernel gradient flow in a Hilbert-like feature geometry |

The key operational pattern is:

$$
\text{learning problem}
\quad \leadsto \quad
\text{approximation in a Hilbert space}
\quad \leadsto \quad
\text{projection, coordinates, or spectral analysis}.
$$

### 1.4 Historical Timeline

- **1900s:** Hilbert and Schmidt formalize infinite systems of equations and spectral methods.
- **1920s-1930s:** Hilbert spaces become the language of quantum mechanics.
- **1940s-1950s:** Functional analysis develops projection, duality, and operator theory.
- **1950s-1970s:** RKHS theory connects kernels with Hilbert spaces of functions.
- **1990s-2000s:** Support vector machines and Gaussian processes bring kernels into mainstream ML.
- **2010s-2020s:** Neural tangent kernels and infinite-width limits reconnect deep learning with Hilbert-space and kernel viewpoints.

---

## 2. Formal Definitions

### 2.1 Real and Complex Inner Products

Let $\mathcal{H}$ be a vector space over $\mathbb{R}$ or $\mathbb{C}$. An inner product is a map

$$
\langle \cdot,\cdot \rangle : \mathcal{H} \times \mathcal{H} \to \mathbb{F},
\qquad \mathbb{F} \in \{\mathbb{R}, \mathbb{C}\},
$$

such that for all $\mathbf{x},\mathbf{y},\mathbf{z} \in \mathcal{H}$ and scalars $a,b \in \mathbb{F}$:

1. **Linearity in the first argument**

   $$
   \langle a\mathbf{x} + b\mathbf{y}, \mathbf{z} \rangle
   =
   a\langle \mathbf{x},\mathbf{z} \rangle
   +
   b\langle \mathbf{y},\mathbf{z} \rangle.
   $$

2. **Conjugate symmetry**

   $$
   \langle \mathbf{x},\mathbf{y} \rangle
   =
   \overline{\langle \mathbf{y},\mathbf{x} \rangle}.
   $$

3. **Positive definiteness**

   $$
   \langle \mathbf{x},\mathbf{x} \rangle \geq 0,
   \qquad
   \langle \mathbf{x},\mathbf{x} \rangle = 0
   \Longleftrightarrow
   \mathbf{x}=\mathbf{0}.
   $$

Some books choose linearity in the second argument for complex spaces. This repository uses linearity in the first argument in this section. The formulas involving adjoints and Riesz representatives should be read consistently with that convention.

### 2.2 Induced Norm and Metric

Every inner product induces a norm:

$$
\lVert \mathbf{x} \rVert
=
\sqrt{\langle \mathbf{x},\mathbf{x} \rangle}.
$$

The induced metric is

$$
d(\mathbf{x},\mathbf{y})
=
\lVert \mathbf{x}-\mathbf{y} \rVert.
$$

So every inner product space is a normed space and every inner product space is a metric space. The reverse is false. Many norms do not come from any inner product.

### 2.3 Hilbert and Pre-Hilbert Spaces

An **inner product space** is also called a **pre-Hilbert space** when we want to emphasize that it may not be complete.

A **Hilbert space** is a complete inner product space. Equivalently, $\mathcal{H}$ is Hilbert if every sequence $(\mathbf{x}_n)$ satisfying

$$
\forall \epsilon > 0,\ \exists N,\ \forall m,n \geq N:
\lVert \mathbf{x}_n-\mathbf{x}_m \rVert < \epsilon
$$

has a limit $\mathbf{x} \in \mathcal{H}$ with

$$
\lVert \mathbf{x}_n-\mathbf{x} \rVert \to 0.
$$

Completeness is a statement about the norm induced by the inner product.

### 2.4 Examples and Non-Examples

**Finite-dimensional Euclidean spaces.** $\mathbb{R}^n$ and $\mathbb{C}^n$ are Hilbert spaces with the usual inner product. Every finite-dimensional inner product space is complete.

**Weighted Euclidean spaces.** If $W \in \mathbb{S}^n_{++}$, then

$$
\langle \mathbf{x},\mathbf{y} \rangle_W
=
\mathbf{x}^\top W\mathbf{y}
$$

is an inner product on $\mathbb{R}^n$. The induced norm is a Mahalanobis-style norm:

$$
\lVert \mathbf{x} \rVert_W
=
\sqrt{\mathbf{x}^\top W\mathbf{x}}.
$$

**Square-summable sequences.** The space

$$
\ell^2
=
\left\{
\mathbf{x}=(x_1,x_2,\ldots):
\sum_{i=1}^\infty \lvert x_i \rvert^2 < \infty
\right\}
$$

is Hilbert under

$$
\langle \mathbf{x},\mathbf{y} \rangle
=
\sum_{i=1}^\infty x_i\overline{y_i}.
$$

**Square-integrable functions.** The space $L^2(\Omega)$ is Hilbert under

$$
\langle f,g \rangle
=
\int_{\Omega} f(t)\overline{g(t)}\,dt.
$$

Technically, elements of $L^2$ are equivalence classes of functions equal almost everywhere. That detail prevents zero-norm nonzero representatives from breaking positive definiteness.

**Polynomials are not complete.** The polynomial space $\mathcal{P}[0,1]$ with the $L^2$ inner product is a pre-Hilbert space, not a Hilbert space.

**$C[0,1]$ with $L^2$ inner product is not complete.** Continuous functions are dense in $L^2[0,1]$, but an $L^2$ limit of continuous functions need not be continuous.

**$\ell^1$ is not a Hilbert space with its usual norm.** The norm $\lVert \mathbf{x} \rVert_1$ does not satisfy the parallelogram law, so it cannot be induced by any inner product.

### 2.5 Parallelogram Law and Polarization Identity

Every inner-product norm satisfies the parallelogram law:

$$
\lVert \mathbf{x}+\mathbf{y} \rVert^2
+
\lVert \mathbf{x}-\mathbf{y} \rVert^2
=
2\lVert \mathbf{x} \rVert^2
+
2\lVert \mathbf{y} \rVert^2.
$$

This law says the two diagonals of a parallelogram carry exactly the same total squared length as twice the sum of squared side lengths.

For real inner product spaces, the inner product can be recovered from the norm:

$$
\langle \mathbf{x},\mathbf{y} \rangle
=
\frac{1}{4}
\left(
\lVert \mathbf{x}+\mathbf{y} \rVert^2
-
\lVert \mathbf{x}-\mathbf{y} \rVert^2
\right).
$$

For complex spaces, the polarization identity is

$$
\langle \mathbf{x},\mathbf{y} \rangle
=
\frac{1}{4}
\sum_{k=0}^{3}
i^k
\lVert \mathbf{x}+i^k\mathbf{y} \rVert^2.
$$

Thus a Hilbert norm is not just any norm. It is exactly a norm whose geometry secretly contains an inner product.

---

## 3. Core Theory I: Inner Product Geometry

### 3.1 Cauchy-Schwarz Inequality

For any $\mathbf{x},\mathbf{y}$ in an inner product space,

$$
\lvert \langle \mathbf{x},\mathbf{y} \rangle \rvert
\leq
\lVert \mathbf{x} \rVert \lVert \mathbf{y} \rVert.
$$

Equality holds if and only if $\mathbf{x}$ and $\mathbf{y}$ are linearly dependent.

**Proof sketch.** If $\mathbf{y}=\mathbf{0}$, the claim is immediate. Otherwise, consider

$$
\phi(a)
=
\lVert \mathbf{x}-a\mathbf{y} \rVert^2
\geq 0.
$$

Choose

$$
a
=
\frac{\langle \mathbf{x},\mathbf{y} \rangle}{\lVert \mathbf{y} \rVert^2}.
$$

Expanding $\phi(a)$ gives

$$
0
\leq
\lVert \mathbf{x} \rVert^2
-
\frac{\lvert \langle \mathbf{x},\mathbf{y} \rangle \rvert^2}
{\lVert \mathbf{y} \rVert^2},
$$

which rearranges to Cauchy-Schwarz.

In ML, Cauchy-Schwarz bounds the maximum possible alignment between embeddings, gradients, features, and residuals. It justifies cosine similarity and many norm-based generalization bounds.

### 3.2 Angles, Cosine Similarity, and Orthogonality

For nonzero $\mathbf{x}$ and $\mathbf{y}$ in a real Hilbert space, define the angle $\theta$ by

$$
\cos \theta
=
\frac{\langle \mathbf{x},\mathbf{y} \rangle}
{\lVert \mathbf{x} \rVert\lVert \mathbf{y} \rVert}.
$$

Cauchy-Schwarz guarantees the right side lies in $[-1,1]$.

Cosine similarity is the same expression. It removes magnitude and keeps directional alignment:

$$
\operatorname{cosim}(\mathbf{x},\mathbf{y})
=
\left\langle
\frac{\mathbf{x}}{\lVert \mathbf{x} \rVert},
\frac{\mathbf{y}}{\lVert \mathbf{y} \rVert}
\right\rangle.
$$

Two vectors are orthogonal when

$$
\langle \mathbf{x},\mathbf{y} \rangle = 0.
$$

Attention scores, embedding retrieval, contrastive learning, and spectral algorithms all use this notion of alignment.

### 3.3 Pythagorean Theorem

If $\mathbf{x}\perp\mathbf{y}$, then

$$
\lVert \mathbf{x}+\mathbf{y} \rVert^2
=
\lVert \mathbf{x} \rVert^2
+
\lVert \mathbf{y} \rVert^2.
$$

Proof:

$$
\begin{aligned}
\lVert \mathbf{x}+\mathbf{y} \rVert^2
&=
\langle \mathbf{x}+\mathbf{y},\mathbf{x}+\mathbf{y} \rangle \\
&=
\lVert \mathbf{x} \rVert^2
+ \langle \mathbf{x},\mathbf{y} \rangle
+ \langle \mathbf{y},\mathbf{x} \rangle
+ \lVert \mathbf{y} \rVert^2 \\
&=
\lVert \mathbf{x} \rVert^2
+ \lVert \mathbf{y} \rVert^2.
\end{aligned}
$$

This identity is the energy bookkeeping behind least squares, Fourier analysis, PCA, and variance decomposition.

### 3.4 Orthogonal Complements and Closed Subspaces

For a subset $\mathcal{M}\subseteq\mathcal{H}$, its orthogonal complement is

$$
\mathcal{M}^{\perp}
=
\{\mathbf{x}\in\mathcal{H}:
\langle \mathbf{x},\mathbf{m} \rangle = 0
\text{ for all } \mathbf{m}\in\mathcal{M}\}.
$$

$\mathcal{M}^{\perp}$ is always a closed subspace, even if $\mathcal{M}$ is not closed.

If $\mathcal{M}$ is a closed subspace of a Hilbert space, then

$$
\mathcal{H}
=
\mathcal{M}
\oplus
\mathcal{M}^{\perp}.
$$

Every vector $\mathbf{x}\in\mathcal{H}$ has a unique decomposition

$$
\mathbf{x}
=
\mathbf{m}
+
\mathbf{r},
\qquad
\mathbf{m}\in\mathcal{M},
\quad
\mathbf{r}\in\mathcal{M}^{\perp}.
$$

In least squares, $\mathbf{m}$ is the fitted value and $\mathbf{r}$ is the residual.

### 3.5 Best Approximation Intuition

Hilbert spaces make approximation geometric. Suppose $\mathcal{M}$ is a model class that is a closed subspace and $\mathbf{x}$ is a target. The best approximation problem is

$$
\min_{\mathbf{m}\in\mathcal{M}}
\lVert \mathbf{x}-\mathbf{m} \rVert.
$$

The solution is characterized not by a complicated search condition but by orthogonality:

$$
\mathbf{x}-\mathbf{m}^{\star}
\perp
\mathcal{M}.
$$

That is, the error left after the best approximation has no component in the model subspace.

```text
target x
   *
   |\
   | \
   |  \ residual x - P_M x
   |   \
   |    *
   |   P_M x
---+---------------- model subspace M
```

---

## 4. Core Theory II: Projection Theorem

### 4.1 Projection onto Closed Convex Sets

Let $\mathcal{C}$ be a nonempty closed convex subset of a Hilbert space $\mathcal{H}$. For every $\mathbf{x}\in\mathcal{H}$, there exists a unique point $P_{\mathcal{C}}\mathbf{x}\in\mathcal{C}$ such that

$$
\lVert \mathbf{x}-P_{\mathcal{C}}\mathbf{x} \rVert
=
\inf_{\mathbf{c}\in\mathcal{C}}
\lVert \mathbf{x}-\mathbf{c} \rVert.
$$

The point $P_{\mathcal{C}}\mathbf{x}$ is the projection of $\mathbf{x}$ onto $\mathcal{C}$.

For convex sets, the variational characterization is

$$
\operatorname{Re}
\langle \mathbf{x}-P_{\mathcal{C}}\mathbf{x},
\mathbf{c}-P_{\mathcal{C}}\mathbf{x} \rangle
\leq 0
\quad
\text{for all } \mathbf{c}\in\mathcal{C}.
$$

This inequality says every feasible direction from the projection point makes an obtuse angle with the residual.

### 4.2 Orthogonal Projection onto Closed Subspaces

When $\mathcal{M}$ is a closed subspace, the projection condition simplifies:

$$
\mathbf{x}-P_{\mathcal{M}}\mathbf{x}
\in
\mathcal{M}^{\perp}.
$$

Equivalently,

$$
\langle \mathbf{x}-P_{\mathcal{M}}\mathbf{x},\mathbf{m} \rangle = 0
\quad
\text{for all } \mathbf{m}\in\mathcal{M}.
$$

If $\{ \mathbf{e}_1,\ldots,\mathbf{e}_k \}$ is an orthonormal basis for a finite-dimensional subspace $\mathcal{M}$, then

$$
P_{\mathcal{M}}\mathbf{x}
=
\sum_{j=1}^k
\langle \mathbf{x},\mathbf{e}_j \rangle
\mathbf{e}_j.
$$

This formula is the finite-dimensional ancestor of Fourier expansion.

### 4.3 Least Squares as Projection

Given $A\in\mathbb{R}^{m\times n}$ and $\mathbf{b}\in\mathbb{R}^m$, least squares solves

$$
\min_{\mathbf{x}\in\mathbb{R}^n}
\lVert A\mathbf{x}-\mathbf{b} \rVert_2^2.
$$

The fitted vector $A\mathbf{x}^{\star}$ is the projection of $\mathbf{b}$ onto the column space $\operatorname{col}(A)$. The residual is orthogonal to every column of $A$:

$$
A^\top(A\mathbf{x}^{\star}-\mathbf{b})
=
\mathbf{0}.
$$

Thus the normal equations are not arbitrary algebra. They are the orthogonality condition for projection:

$$
A^\top A\mathbf{x}^{\star}
=
A^\top\mathbf{b}.
$$

If $A^\top A$ is invertible,

$$
\mathbf{x}^{\star}
=
(A^\top A)^{-1}A^\top\mathbf{b}.
$$

If not, the minimum-norm solution is

$$
\mathbf{x}^{\star}
=
A^\dagger\mathbf{b}.
$$

### 4.4 Projection Operators

The orthogonal projection $P_{\mathcal{M}}:\mathcal{H}\to\mathcal{M}$ satisfies:

$$
P_{\mathcal{M}}^2=P_{\mathcal{M}},
\qquad
P_{\mathcal{M}}^*=P_{\mathcal{M}},
\qquad
\lVert P_{\mathcal{M}} \rVert \leq 1.
$$

If $\mathcal{M}\neq\{\mathbf{0}\}$, then $\lVert P_{\mathcal{M}} \rVert=1$.

For a matrix $A$ with full column rank, the projection matrix onto $\operatorname{col}(A)$ is

$$
P_A
=
A(A^\top A)^{-1}A^\top.
$$

It satisfies

$$
P_A^2=P_A,
\qquad
P_A^\top=P_A.
$$

### 4.5 Projection Algorithms in ML

Projection appears in many algorithms:

- projected gradient descent enforces constraints by applying $P_{\mathcal{C}}$
- alternating projections solve feasibility problems
- least-squares layers project labels onto feature spans
- PCA projects data onto top eigenspaces
- denoising methods often estimate a projection onto a data manifold or low-dimensional signal set
- constrained decoding can be viewed as repeated projection onto feasible token or structure sets, though usually in non-Hilbert geometries

The exact Hilbert projection theorem applies to closed convex sets in Hilbert spaces. ML practice often borrows the intuition outside this perfect setting, so the theorem-level statement and engineering heuristic should not be confused.

---

## 5. Core Theory III: Orthonormal Systems and Bases

### 5.1 Orthonormal Sets and Gram-Schmidt

A set $\{\mathbf{e}_j\}_{j\in J}$ is orthonormal if

$$
\langle \mathbf{e}_i,\mathbf{e}_j \rangle
=
\begin{cases}
1, & i=j, \\
0, & i\neq j.
\end{cases}
$$

Given linearly independent vectors $\mathbf{v}_1,\ldots,\mathbf{v}_k$, Gram-Schmidt constructs an orthonormal set:

$$
\mathbf{u}_j
=
\mathbf{v}_j
-
\sum_{i=1}^{j-1}
\langle \mathbf{v}_j,\mathbf{e}_i \rangle
\mathbf{e}_i,
\qquad
\mathbf{e}_j
=
\frac{\mathbf{u}_j}{\lVert \mathbf{u}_j \rVert}.
$$

Numerically, classical Gram-Schmidt can be unstable. Modified Gram-Schmidt or QR factorization is preferred in computation.

### 5.2 Bessel Inequality

If $\{\mathbf{e}_j\}_{j\in J}$ is an orthonormal set, then for every $\mathbf{x}\in\mathcal{H}$,

$$
\sum_{j\in J}
\lvert \langle \mathbf{x},\mathbf{e}_j \rangle \rvert^2
\leq
\lVert \mathbf{x} \rVert^2.
$$

For finite $J$, define

$$
\mathbf{p}
=
\sum_{j\in J}
\langle \mathbf{x},\mathbf{e}_j \rangle
\mathbf{e}_j.
$$

Then $\mathbf{x}-\mathbf{p}$ is orthogonal to $\mathbf{p}$, so

$$
\lVert \mathbf{x} \rVert^2
=
\lVert \mathbf{p} \rVert^2
+
\lVert \mathbf{x}-\mathbf{p} \rVert^2
\geq
\lVert \mathbf{p} \rVert^2
=
\sum_{j\in J}
\lvert \langle \mathbf{x},\mathbf{e}_j \rangle \rvert^2.
$$

Bessel inequality says orthonormal coordinates cannot contain more energy than the vector itself.

### 5.3 Complete Orthonormal Systems and Hilbert Bases

An orthonormal set $\{\mathbf{e}_j\}_{j\in J}$ is complete if the only vector orthogonal to every $\mathbf{e}_j$ is $\mathbf{0}$:

$$
\left(
\langle \mathbf{x},\mathbf{e}_j \rangle = 0
\text{ for all } j\in J
\right)
\Longrightarrow
\mathbf{x}=\mathbf{0}.
$$

For separable Hilbert spaces, a countable complete orthonormal system is often called a Hilbert basis. This is not a Hamel basis. Infinite Hilbert expansions converge in norm, not as finite algebraic sums.

### 5.4 Parseval Identity and Fourier-Bessel Expansion

If $\{\mathbf{e}_j\}_{j\in J}$ is a complete orthonormal system, then

$$
\mathbf{x}
=
\sum_{j\in J}
\langle \mathbf{x},\mathbf{e}_j \rangle
\mathbf{e}_j
$$

with convergence in the Hilbert norm, and

$$
\lVert \mathbf{x} \rVert^2
=
\sum_{j\in J}
\lvert \langle \mathbf{x},\mathbf{e}_j \rangle \rvert^2.
$$

The first identity is Fourier-Bessel expansion. The second is Parseval identity.

For $L^2[-\pi,\pi]$, the trigonometric functions form an orthonormal coordinate system after normalization. A square-integrable function can be represented by its Fourier coefficients in $L^2$ norm, even when pointwise convergence needs additional hypotheses.

### 5.5 Separability and $\ell^2$ Coordinates

A Hilbert space is separable if it has a countable dense subset. Most Hilbert spaces used in ML and signal processing are separable.

If $\mathcal{H}$ is separable with orthonormal basis $(\mathbf{e}_j)_{j=1}^{\infty}$, the map

$$
\mathbf{x}
\mapsto
\left(
\langle \mathbf{x},\mathbf{e}_1 \rangle,
\langle \mathbf{x},\mathbf{e}_2 \rangle,
\ldots
\right)
$$

is an isometric isomorphism from $\mathcal{H}$ to a closed subspace of $\ell^2$. If the basis is complete, it is an isometric isomorphism onto $\ell^2$.

This is the reason $\ell^2$ is the prototype of separable infinite-dimensional Hilbert spaces.

---

## 6. Core Theory IV: Riesz Representation and Duality

### 6.1 Bounded Linear Functionals

A linear functional is a linear map

$$
L:\mathcal{H}\to\mathbb{F}.
$$

It is bounded if there exists $C\geq 0$ such that

$$
\lvert L(\mathbf{x}) \rvert
\leq
C\lVert \mathbf{x} \rVert
\quad
\text{for all } \mathbf{x}\in\mathcal{H}.
$$

The operator norm is

$$
\lVert L \rVert
=
\sup_{\lVert \mathbf{x} \rVert\leq 1}
\lvert L(\mathbf{x}) \rvert.
$$

Bounded linear functionals are exactly continuous linear functionals.

### 6.2 Riesz Representation Theorem

For every bounded linear functional $L$ on a Hilbert space $\mathcal{H}$, there exists a unique vector $\mathbf{h}_L\in\mathcal{H}$ such that

$$
L(\mathbf{x})
=
\langle \mathbf{x},\mathbf{h}_L \rangle
\quad
\text{for all } \mathbf{x}\in\mathcal{H}.
$$

Moreover,

$$
\lVert L \rVert
=
\lVert \mathbf{h}_L \rVert.
$$

The vector $\mathbf{h}_L$ is the Riesz representative of $L$.

### 6.3 Hilbert Spaces Are Self-Dual

The Riesz theorem identifies $\mathcal{H}^*$ with $\mathcal{H}$. In a general Banach space, the dual can be quite different from the original space. In a Hilbert space, continuous linear measurements are inner products with vectors inside the same space.

This is the clean mathematical reason gradients can often be represented as vectors. A derivative is naturally a linear functional. To turn it into a gradient vector, we choose an inner product and apply Riesz representation.

### 6.4 Gradients as Riesz Representatives

Let $F:\mathcal{H}\to\mathbb{R}$ be differentiable. Its differential at $\mathbf{x}$ is a bounded linear functional

$$
DF(\mathbf{x})[\mathbf{v}].
$$

The gradient $\nabla F(\mathbf{x})$ is the Riesz representative satisfying

$$
DF(\mathbf{x})[\mathbf{v}]
=
\langle \mathbf{v},\nabla F(\mathbf{x}) \rangle
\quad
\text{for all } \mathbf{v}\in\mathcal{H}.
$$

Changing the inner product changes the gradient vector while preserving the same differential. This is why natural gradient, preconditioning, and mirror-descent-like methods can be interpreted as changing the geometry in which steepest descent is measured.

### 6.5 Why This Matters for Optimization and Backpropagation

Backpropagation computes derivatives. Optimizers apply gradient vectors. The bridge from derivative-as-functional to gradient-as-vector is Riesz representation plus an inner product choice.

For the squared loss

$$
F(\mathbf{w})
=
\frac{1}{2}
\lVert X\mathbf{w}-\mathbf{y} \rVert_2^2,
$$

the differential is

$$
DF(\mathbf{w})[\mathbf{v}]
=
\langle X\mathbf{w}-\mathbf{y},X\mathbf{v} \rangle
=
\langle X^\top(X\mathbf{w}-\mathbf{y}),\mathbf{v} \rangle.
$$

Thus the Euclidean gradient is

$$
\nabla F(\mathbf{w})
=
X^\top(X\mathbf{w}-\mathbf{y}).
$$

With a different inner product $\langle \mathbf{u},\mathbf{v} \rangle_G=\mathbf{u}^\top G\mathbf{v}$, the gradient vector becomes

$$
\nabla_G F(\mathbf{w})
=
G^{-1}
\nabla F(\mathbf{w}).
$$

This is preconditioning in Hilbert geometry.

---

## 7. Core Theory V: Operators on Hilbert Spaces

### 7.1 Bounded Operators and Operator Norms

A linear operator $T:\mathcal{H}\to\mathcal{K}$ is bounded if

$$
\lVert T\mathbf{x} \rVert_{\mathcal{K}}
\leq
C\lVert \mathbf{x} \rVert_{\mathcal{H}}
\quad
\text{for all } \mathbf{x}\in\mathcal{H}.
$$

Its operator norm is

$$
\lVert T \rVert
=
\sup_{\lVert \mathbf{x} \rVert_{\mathcal{H}}\leq 1}
\lVert T\mathbf{x} \rVert_{\mathcal{K}}.
$$

For matrices with Euclidean geometry, this is the spectral norm:

$$
\lVert A \rVert_2
=
\sigma_{\max}(A).
$$

### 7.2 Adjoints and Self-Adjoint Operators

The adjoint $T^*:\mathcal{K}\to\mathcal{H}$ is defined by

$$
\langle T\mathbf{x},\mathbf{y} \rangle_{\mathcal{K}}
=
\langle \mathbf{x},T^*\mathbf{y} \rangle_{\mathcal{H}}.
$$

In finite-dimensional real Euclidean space, the adjoint is the transpose. In complex Euclidean space, it is the conjugate transpose.

An operator is self-adjoint if

$$
T=T^*.
$$

Self-adjoint operators are the Hilbert-space analogue of symmetric matrices. Covariance operators, kernel integral operators, Hessians of quadratic losses, and graph Laplacians are central ML examples.

### 7.3 Positive Operators and Quadratic Forms

A self-adjoint operator $T$ is positive if

$$
\langle T\mathbf{x},\mathbf{x} \rangle \geq 0
\quad
\text{for all } \mathbf{x}\in\mathcal{H}.
$$

In finite dimensions, this is the positive semidefinite condition $A\succeq0$.

Positive operators define energy functionals:

$$
E(\mathbf{x})
=
\langle T\mathbf{x},\mathbf{x} \rangle.
$$

Examples:

- covariance matrix: variance in direction $\mathbf{x}$
- graph Laplacian: smoothness energy over graph nodes
- kernel Gram matrix: squared norm of a finite feature combination
- Hessian: local curvature of a twice-differentiable loss

### 7.4 Compact Operators

A bounded operator $T:\mathcal{H}\to\mathcal{K}$ is compact if it maps bounded sets to relatively compact sets. Equivalently, every bounded sequence $(\mathbf{x}_n)$ has a subsequence such that $(T\mathbf{x}_{n_k})$ converges.

In finite dimensions, every bounded operator is compact. In infinite dimensions, compactness is special. Compact operators often behave like infinite matrices whose singular values decay to zero.

Integral operators with smooth kernels are typical examples:

$$
(Tf)(s)
=
\int_{\Omega} k(s,t)f(t)\,dt.
$$

Kernel methods and Gaussian-process covariance operators often lead to compact positive self-adjoint operators under suitable domain and regularity assumptions.

### 7.5 Spectral Theorem for Compact Self-Adjoint Operators

If $T$ is compact and self-adjoint on a Hilbert space, then there is an orthonormal set of eigenvectors associated with nonzero real eigenvalues, and the nonzero spectrum can accumulate only at $0$.

In favorable cases,

$$
T\mathbf{x}
=
\sum_{j=1}^{\infty}
\lambda_j
\langle \mathbf{x},\mathbf{e}_j \rangle
\mathbf{e}_j.
$$

This is the infinite-dimensional analogue of diagonalizing a symmetric matrix:

$$
A
=
Q\Lambda Q^\top.
$$

PCA is the finite-sample version of this story. Kernel PCA and Gaussian-process covariance analysis use the same idea in feature or function spaces.

---

## 8. Advanced Topics and Bridges

### 8.1 Weak Convergence

Norm convergence is strong:

$$
\lVert \mathbf{x}_n-\mathbf{x} \rVert \to 0.
$$

Weak convergence asks only that all bounded linear measurements converge:

$$
\mathbf{x}_n \rightharpoonup \mathbf{x}
\quad
\Longleftrightarrow
\quad
\langle \mathbf{x}_n,\mathbf{h} \rangle
\to
\langle \mathbf{x},\mathbf{h} \rangle
\quad
\text{for all } \mathbf{h}\in\mathcal{H}.
$$

Strong convergence implies weak convergence. Weak convergence does not generally imply strong convergence.

In infinite-dimensional optimization, weak compactness and lower semicontinuity are often enough to prove existence of minimizers even when norm-compactness fails.

### 8.2 Fourier Analysis as Hilbert-Space Coordinates

Fourier analysis is Hilbert-space coordinate expansion in $L^2$. The functions

$$
\frac{1}{\sqrt{2\pi}},
\qquad
\frac{\cos(nt)}{\sqrt{\pi}},
\qquad
\frac{\sin(nt)}{\sqrt{\pi}},
\quad n\geq1,
$$

form an orthonormal system in $L^2[-\pi,\pi]$.

Fourier coefficients are inner products:

$$
a_n
=
\langle f,\mathbf{e}_n \rangle.
$$

Parseval says signal energy equals coefficient energy. This is the basis of spectral signal processing, convolution analysis, and random Fourier feature intuition.

### 8.3 RKHS and Reproducing Kernels

A reproducing kernel Hilbert space is a Hilbert space $\mathcal{H}_K$ of functions on a set $\mathcal{X}$ such that point evaluation is continuous:

$$
L_x(f)=f(x).
$$

By Riesz representation, for each $x\in\mathcal{X}$ there exists $K_x\in\mathcal{H}_K$ such that

$$
f(x)
=
\langle f,K_x \rangle_{\mathcal{H}_K}.
$$

Define

$$
K(x,x')
=
K_{x'}(x)
=
\langle K_{x'},K_x \rangle_{\mathcal{H}_K}.
$$

This is the reproducing kernel. The next section develops positive definite kernels, the Moore-Aronszajn theorem, SVMs, kernel ridge regression, Gaussian processes, and kernel approximations in detail.

### 8.4 Kernel Gradient Flow and Neural Tangent Kernels

Many overparameterized models can be studied by tracking how predictions change under gradient descent. In certain infinite-width limits, the dynamics of the prediction function are governed by a kernel:

$$
\frac{d f_t(\mathbf{x})}{dt}
=
-
\sum_{i=1}^n
K(\mathbf{x},\mathbf{x}^{(i)})
\frac{\partial \mathcal{L}}{\partial f_t(\mathbf{x}^{(i)})}.
$$

The neural tangent kernel perspective says that, under specific assumptions, very wide neural networks behave like kernel machines during training. This is a modern bridge between deep learning and Hilbert-space geometry, but the details depend on architecture, scaling, initialization, and limiting arguments.

### 8.5 Continuous vs Discrete Spectra

Compact self-adjoint operators resemble diagonal matrices with eigenvalues tending to zero. General self-adjoint operators can have continuous spectrum. This matters in quantum mechanics, PDEs, and some infinite-dimensional learning problems.

For this course, the main working intuition is:

- finite-dimensional symmetric matrices have orthonormal eigenvectors
- compact self-adjoint operators retain a similar discrete spectral structure
- general self-adjoint operators require spectral measures, which are beyond this section

---

## 9. Applications in Machine Learning

### 9.1 Embedding Similarity and Attention Scores

The transformer attention score between query $\mathbf{q}$ and key $\mathbf{k}$ is

$$
\frac{\langle \mathbf{q},\mathbf{k} \rangle}{\sqrt{d_k}}.
$$

The inner product measures alignment. Scaling by $\sqrt{d_k}$ controls variance when coordinates have roughly unit scale. Cosine similarity normalizes away vector length; dot-product attention keeps both angle and magnitude.

### 9.2 Least Squares, Ridge Regression, and Orthogonal Projection

Least squares is projection onto the feature span. Ridge regression solves

$$
\min_{\mathbf{w}}
\lVert X\mathbf{w}-\mathbf{y} \rVert_2^2
+
\lambda
\lVert \mathbf{w} \rVert_2^2.
$$

The regularization term adds Hilbert norm control. In kernel ridge regression, the same idea is lifted to an RKHS norm:

$$
\min_{f\in\mathcal{H}_K}
\sum_{i=1}^n
\left(f(\mathbf{x}^{(i)})-y^{(i)}\right)^2
+
\lambda
\lVert f \rVert_{\mathcal{H}_K}^2.
$$

The representer theorem then says the minimizer lies in the finite span of kernel sections $K(\mathbf{x}^{(i)},\cdot)$.

### 9.3 PCA and Spectral Decomposition

Given centered data matrix $X$, the sample covariance is

$$
C
=
\frac{1}{n}X^\top X.
$$

It is positive self-adjoint. PCA finds orthonormal directions $\mathbf{v}_j$ that solve

$$
C\mathbf{v}_j
=
\lambda_j\mathbf{v}_j.
$$

Projecting onto the top $k$ eigenvectors gives the best rank-$k$ approximation in squared reconstruction error. This is Hilbert projection plus spectral theory.

### 9.4 Gaussian Processes and RKHS Preview

A Gaussian process is determined by a mean function and covariance kernel. The covariance kernel is positive definite, so it defines a Hilbert geometry. The RKHS associated with the kernel is not the same as the random sample-path space in general, but it captures the deterministic smoothness geometry encoded by the kernel.

### 9.5 Infinite-Width Neural Networks and NTK Preview

In neural tangent kernel theory, an infinite-width network can converge to dynamics described by a kernel operator. The Hilbert-space lesson is that inner products between parameter gradients induce a geometry on functions:

$$
K(\mathbf{x},\mathbf{x}')
=
\left\langle
\nabla_{\boldsymbol{\theta}} f(\mathbf{x};\boldsymbol{\theta}),
\nabla_{\boldsymbol{\theta}} f(\mathbf{x}';\boldsymbol{\theta})
\right\rangle.
$$

This kernel measures how a parameter update that changes the output at $\mathbf{x}$ also changes the output at $\mathbf{x}'$.

---

## 10. Common Mistakes

1. **Confusing every normed space with a Hilbert space.** A Hilbert space needs an inner product whose induced norm is complete. $\ell^1$ with its usual norm is Banach but not Hilbert.

2. **Forgetting completeness.** Polynomials with the $L^2$ inner product have angles and lengths but are not complete.

3. **Treating a Hilbert basis like a Hamel basis.** Hilbert expansions can be infinite and converge in norm.

4. **Assuming orthogonal projection exists onto every subspace.** The subspace must be closed. Dense proper subspaces do not have nontrivial orthogonal complements.

5. **Using pointwise convergence when $L^2$ convergence is meant.** Fourier series often converge in mean-square even when pointwise behavior is delicate.

6. **Assuming all ML feature spaces are finite-dimensional.** Kernels often represent implicit Hilbert spaces that may be infinite-dimensional.

7. **Ignoring the chosen inner product when discussing gradients.** Gradients depend on geometry. Differentials do not.

8. **Confusing PSD matrices with positive operators in all contexts.** PSD matrices are finite-dimensional examples. Infinite-dimensional positive operators require domain and boundedness care.

9. **Assuming RKHS sample paths are typical Gaussian-process samples.** The RKHS is the deterministic geometry of the kernel; GP sample paths may lie outside it with probability one in many common settings.

10. **Overstating NTK conclusions.** NTK limits are mathematically specific and do not automatically explain all finite-width training behavior.

---

## 11. Exercises

1. Prove Cauchy-Schwarz using minimization of $\lVert \mathbf{x}-a\mathbf{y} \rVert^2$.
2. Check whether a weighted bilinear form $\mathbf{x}^\top W\mathbf{y}$ is an inner product for several matrices $W$.
3. Show that $\ell^1$ does not satisfy the parallelogram law.
4. Compute the orthogonal projection of a vector onto a line and onto a column space.
5. Derive the normal equations for least squares from residual orthogonality.
6. Implement Gram-Schmidt and compare it with QR factorization.
7. Verify Bessel inequality and Parseval identity for a finite orthonormal basis.
8. Find the Riesz representative of $L(\mathbf{x})=\mathbf{a}^\top\mathbf{x}$ under a weighted inner product.
9. Show that a symmetric PSD matrix defines a positive self-adjoint operator.
10. Explain why kernel evaluation is point evaluation represented as an inner product in an RKHS.

The companion exercise notebook contains scaffolded versions with computational checks and full solutions.

---

## 12. Why This Matters for AI

Hilbert spaces are one of the quiet foundations of AI mathematics. They explain why dot products are meaningful, why squared error is geometrically special, why projections solve approximation problems, why PCA diagonalizes variance, why Fourier features preserve energy, why kernels let nonlinear learning become linear in a feature space, and why gradients become vectors once an inner product is chosen.

Three ideas are especially important:

**1. Similarity is geometry.** Attention, retrieval, contrastive learning, and recommendation systems all compare representations. Inner products and normalized inner products are Hilbert-space measurements of alignment.

**2. Approximation is projection.** Least squares, PCA, denoising, and many function-approximation problems become clearer when the learned object is viewed as a projection onto a subspace or a regularized approximation inside a Hilbert space.

**3. Learning dynamics depend on inner products.** A gradient is not just a list of partial derivatives. It is a Riesz representative of a differential under a chosen geometry. Changing the geometry changes the optimization path.

---

## 13. Conceptual Bridge

The next section, [Kernel Methods](../03-Kernel-Methods/notes.md), starts from the Hilbert idea that inner products measure similarity. A kernel is a function

$$
K(\mathbf{x},\mathbf{x}')
=
\langle \phi(\mathbf{x}),\phi(\mathbf{x}') \rangle_{\mathcal{H}}
$$

that computes an inner product in a feature Hilbert space without explicitly constructing $\phi(\mathbf{x})$.

The key transition is:

```text
Hilbert spaces:
  inner products, projections, bases, Riesz representation

Kernel methods:
  compute Hilbert-space inner products through K(x, x')
  learn nonlinear functions with linear Hilbert-space geometry
```

Once you understand Hilbert spaces, the kernel trick is no longer a trick. It is just inner-product geometry moved into a richer feature space.

---

## References

- R. B. Melrose, *Hilbert Spaces*, MIT 18.155 lecture notes, 2016. <https://math.mit.edu/~rbm/18.155-F16/L10.pdf>
- S. New, *Hilbert Spaces*, University of Waterloo PMATH 453 lecture notes. <https://www.math.uwaterloo.ca/~snew/PMATH453/Chap2HilbertSpaces.pdf>
- J. B. Conway, *A Course in Functional Analysis*, Springer, 1990.
- W. Rudin, *Functional Analysis*, McGraw-Hill, 1991.
- E. Kreyszig, *Introductory Functional Analysis with Applications*, Wiley, 1978.
- G. Strang, *Introduction to Linear Algebra*, Wellesley-Cambridge Press, 2016.
- T. Hastie, R. Tibshirani, and J. Friedman, *The Elements of Statistical Learning*, Springer, 2009.
- F. Cucker and S. Smale, "On the Mathematical Foundations of Learning," *Bulletin of the AMS*, 2002.
- N. Aronszajn, "Theory of Reproducing Kernels," *Transactions of the American Mathematical Society*, 1950.
- A. Jacot, F. Gabriel, and C. Hongler, "Neural Tangent Kernel: Convergence and Generalization in Neural Networks," NeurIPS, 2018. <https://arxiv.org/abs/1806.07572>
- Berkeley STAT 154, *Reproducing Kernel Hilbert Spaces* lecture material, 2025. <https://stat154.berkeley.edu/spring-2025/lectures/unit5/unit5_rkhs.html>
- Stanford CS229T, *Kernel Basics* lecture notes, 2017. <https://web.stanford.edu/class/cs229t/2017/Lectures/kernel-basics.pdf>
