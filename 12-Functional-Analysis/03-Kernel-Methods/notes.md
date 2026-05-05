[Home](../../README.md) | [Previous: Hilbert Spaces](../02-Hilbert-Spaces/notes.md) | [Next: Loss Functions](../../13-ML-Specific-Math/01-Loss-Functions/notes.md)

---

# Kernel Methods

> _"A kernel is a promise: compute similarity in the input space, and behave as if learning happened in a richer feature space."_

## Overview

Kernel methods are a family of learning methods built around one central observation:
many algorithms only need inner products between examples.
If an algorithm can be written using $\langle \mathbf{x}, \mathbf{x}' \rangle$, then we can often replace that inner product with a kernel value $k(\mathbf{x}, \mathbf{x}')$.
The replacement makes the algorithm act as though each input has been mapped into a high-dimensional, sometimes infinite-dimensional, feature space.

This section studies the kernel trick, positive semidefinite kernels, Reproducing Kernel Hilbert Spaces, common kernel families, kernelized algorithms, and scalable approximations.
The goal is not to treat kernels as a historical detour before deep learning.
The goal is to understand kernels as one of the cleanest mathematical bridges between Hilbert space geometry and modern AI systems.

Kernel methods explain support vector machines, kernel ridge regression, kernel PCA, Gaussian processes, random Fourier features, and the neural tangent kernel.
They also sharpen everyday ML instincts about similarity, regularization, smoothness, uncertainty, and data-dependent representation.
When you ask whether two embeddings are similar, whether a function should vary smoothly, or whether a model should express uncertainty away from data, you are already asking kernel-shaped questions.

## Prerequisites

- Vector spaces, span, and linear maps from [Vector Spaces and Subspaces](../../02-Linear-Algebra-Basics/06-Vector-Spaces-Subspaces/notes.md).
- Inner products, orthogonality, and projections from [Hilbert Spaces](../02-Hilbert-Spaces/notes.md).
- Norms, completeness, and operator language from [Normed Spaces](../01-Normed-Spaces/notes.md).
- Positive semidefinite matrices from [Positive Definite Matrices](../../03-Advanced-Linear-Algebra/07-Positive-Definite-Matrices/notes.md).
- Eigenvalues, SVD, and matrix rank from [Advanced Linear Algebra](../../03-Advanced-Linear-Algebra/README.md) and its sections.
- Constrained optimization and Lagrange multipliers from [Constrained Optimization](../../08-Optimization/04-Constrained-Optimization/notes.md).
- Basic supervised learning vocabulary: dataset, labels, regression, classification, loss, and regularization.

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Interactive derivations for Gram matrices, kernels, RKHS intuition, kernel ridge regression, kernel PCA, Gaussian process prediction, random Fourier features, and NTK-style behavior. |
| [exercises.ipynb](exercises.ipynb) | Eight graded exercises covering kernel validity, Gram matrices, centering, kernel ridge regression, kernel PCA, Gaussian processes, random Fourier features, and neural tangent kernels. |

## Learning Objectives

After completing this section, you will be able to:

- Explain the kernel trick as an inner-product replacement, not as magic.
- Construct explicit feature maps for linear and polynomial kernels.
- Test a finite Gram matrix for positive semidefiniteness.
- Distinguish valid kernels from arbitrary similarity functions.
- Explain how every positive semidefinite kernel defines an RKHS.
- Use the reproducing property to interpret function evaluation as an inner product.
- Apply the representer theorem to regularized learning in RKHS.
- Implement kernel ridge regression and interpret the role of $\lambda$.
- Center a kernel matrix and run kernel PCA.
- Interpret Gaussian process regression as Bayesian learning with a kernel covariance.
- Use Nystrom and random Fourier approximations to reduce kernel cost.
- Connect kernel eigenvalues to generalization, smoothness, and neural tangent kernel intuition.

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 Why Kernels Matter for AI](#11-why-kernels-matter-for-ai)
  - [1.2 Linear Learning in Nonlinear Feature Spaces](#12-linear-learning-in-nonlinear-feature-spaces)
  - [1.3 The Kernel Trick as Inner-Product Substitution](#13-the-kernel-trick-as-inner-product-substitution)
  - [1.4 Historical Path](#14-historical-path)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Feature Maps and Feature Spaces](#21-feature-maps-and-feature-spaces)
  - [2.2 Kernel Functions and Gram Matrices](#22-kernel-functions-and-gram-matrices)
  - [2.3 Positive Semidefinite Kernels](#23-positive-semidefinite-kernels)
  - [2.4 Valid Kernels and Counterexamples](#24-valid-kernels-and-counterexamples)
  - [2.5 Kernel Normalization and Centering](#25-kernel-normalization-and-centering)
- [3. RKHS Foundations](#3-rkhs-foundations)
  - [3.1 Reproducing Kernel Hilbert Spaces](#31-reproducing-kernel-hilbert-spaces)
  - [3.2 The Reproducing Property](#32-the-reproducing-property)
  - [3.3 Moore-Aronszajn Theorem](#33-moore-aronszajn-theorem)
  - [3.4 Mercer Expansion and Eigenfunctions](#34-mercer-expansion-and-eigenfunctions)
  - [3.5 RKHS Norm as Smoothness Control](#35-rkhs-norm-as-smoothness-control)
- [4. Common Kernels](#4-common-kernels)
  - [4.1 Linear and Polynomial Kernels](#41-linear-and-polynomial-kernels)
  - [4.2 RBF Kernel](#42-rbf-kernel)
  - [4.3 Laplacian and Matern Kernels](#43-laplacian-and-matern-kernels)
  - [4.4 Periodic, Rational Quadratic, String, and Graph Kernels](#44-periodic-rational-quadratic-string-and-graph-kernels)
  - [4.5 Kernel Composition Rules](#45-kernel-composition-rules)
- [5. Core Kernel Algorithms](#5-core-kernel-algorithms)
  - [5.1 Kernel Perceptron](#51-kernel-perceptron)
  - [5.2 Support Vector Machines](#52-support-vector-machines)
  - [5.3 Kernel Ridge Regression and Representer Theorem](#53-kernel-ridge-regression-and-representer-theorem)
  - [5.4 Kernel PCA](#54-kernel-pca)
  - [5.5 Gaussian Processes](#55-gaussian-processes)
- [6. Computation and Scalability](#6-computation-and-scalability)
  - [6.1 Kernel Matrix Cost](#61-kernel-matrix-cost)
  - [6.2 Conditioning and Regularization](#62-conditioning-and-regularization)
  - [6.3 Nystrom Approximation](#63-nystrom-approximation)
  - [6.4 Random Fourier Features](#64-random-fourier-features)
  - [6.5 When Kernels Are Practical](#65-when-kernels-are-practical)
- [7. Kernel Methods in Modern AI](#7-kernel-methods-in-modern-ai)
  - [7.1 Similarity Search and Embeddings](#71-similarity-search-and-embeddings)
  - [7.2 Kernels and Attention](#72-kernels-and-attention)
  - [7.3 Neural Tangent Kernel](#73-neural-tangent-kernel)
  - [7.4 Deep Kernel Learning](#74-deep-kernel-learning)
  - [7.5 Uncertainty and Small-Data Learning](#75-uncertainty-and-small-data-learning)
- [8. Common Mistakes](#8-common-mistakes)
- [9. Exercises](#9-exercises)
- [10. Why This Matters for AI](#10-why-this-matters-for-ai)
- [11. Conceptual Bridge](#11-conceptual-bridge)
- [12. References](#12-references)

---

## 1. Intuition

Kernel methods begin with a deceptively simple question.
What if the original coordinates are the wrong coordinates for learning?

Suppose a binary classification dataset in $\mathbb{R}^2$ is not linearly separable.
No line can separate the classes in the original plane.
But after adding a feature such as $x_1^2 + x_2^2$, the same points may become separable by a hyperplane in the transformed feature space.

The direct feature-map story is clear:
map $\mathbf{x}$ to $\phi(\mathbf{x})$.
Run a linear algorithm in the $\phi$-space.
Return a nonlinear predictor in the original input space.

The kernel story is more subtle:
do not compute $\phi(\mathbf{x})$.
Compute only the inner product that would have appeared after the feature map.
That inner product is the kernel.

$$
k(\mathbf{x}, \mathbf{x}') =
\langle \phi(\mathbf{x}), \phi(\mathbf{x}') \rangle_{\mathcal{H}}.
$$

This identity is the whole game.
The feature space can be very large.
It can be infinite-dimensional.
It can even be a function space.
But if the learning algorithm only needs dot products, the explicit coordinates are unnecessary.

### 1.1 Why Kernels Matter for AI

Kernels matter because they convert assumptions about similarity into mathematically controlled learning algorithms.
In neural networks, similarity is often learned implicitly.
In kernel methods, similarity is specified by $k$.

A kernel can encode smoothness.
A kernel can encode periodicity.
A kernel can encode sequence similarity.
A kernel can encode graph structure.
A kernel can encode invariances.
A kernel can encode prior uncertainty.

This makes kernels especially useful when:

- data is limited;
- uncertainty estimates matter;
- interpretability of inductive bias matters;
- training a large neural model is unnecessary or too expensive;
- a strong handcrafted similarity is available;
- we want a clean mathematical model of generalization.

Kernels also matter in deep learning theory.
The neural tangent kernel studies how wide neural networks behave during gradient descent.
In the infinite-width limit, a neural network can behave like a kernel machine whose kernel is determined by architecture and initialization.
This does not mean all useful neural networks are just kernels.
It means kernels provide a precise baseline for understanding when representation learning does, or does not, happen.

The AI story has three levels:

| Level | Kernel role | Example |
| --- | --- | --- |
| Algorithmic | Kernel gives a nonlinear model with convex or closed-form training. | SVM, kernel ridge regression, kernel PCA. |
| Probabilistic | Kernel is a covariance function over functions. | Gaussian process regression. |
| Theoretical | Kernel describes training dynamics of very wide networks. | Neural tangent kernel. |

### 1.2 Linear Learning in Nonlinear Feature Spaces

Let $\mathcal{X}$ be an input space.
For tabular data, $\mathcal{X}$ might be $\mathbb{R}^d$.
For text, $\mathcal{X}$ might be strings.
For graphs, $\mathcal{X}$ might be graphs.
For functions, $\mathcal{X}$ might already be infinite-dimensional.

A feature map is a function

$$
\phi : \mathcal{X} \to \mathcal{H},
$$

where $\mathcal{H}$ is a Hilbert space.
Once inputs are mapped into $\mathcal{H}$, a linear predictor has the form

$$
f(\mathbf{x}) =
\langle \mathbf{w}, \phi(\mathbf{x}) \rangle_{\mathcal{H}} + b.
$$

This predictor is linear in feature space.
It can be nonlinear in the original input coordinates.

Example:
let $\mathbf{x} = (x_1, x_2)^\top$.
Define

$$
\phi(\mathbf{x}) =
\begin{bmatrix}
x_1^2 \\
\sqrt{2}x_1x_2 \\
x_2^2
\end{bmatrix}.
$$

Then

$$
\langle \phi(\mathbf{x}), \phi(\mathbf{z}) \rangle
=
x_1^2z_1^2 + 2x_1x_2z_1z_2 + x_2^2z_2^2
=
(\mathbf{x}^\top \mathbf{z})^2.
$$

So the degree-2 homogeneous polynomial kernel is

$$
k(\mathbf{x}, \mathbf{z}) =
(\mathbf{x}^\top \mathbf{z})^2.
$$

The feature map has dimension 3 for $\mathbb{R}^2$.
For degree $p$ in dimension $d$, the explicit feature dimension grows combinatorially.
The kernel evaluation remains cheap.

For the inhomogeneous polynomial kernel,

$$
k(\mathbf{x}, \mathbf{z}) =
(\mathbf{x}^\top \mathbf{z} + c)^p,
$$

the implicit feature space contains monomials of degree up to $p$.
This is a nonlinear feature expansion without manually writing the expanded feature vector.

Feature-space picture:

```text
input space                 feature space

    x                              phi(x)
  points       ---- phi ---->      high-dimensional points

linear separator impossible        linear separator possible
in original coordinates            after nonlinear lifting
```

The crucial warning:
the kernel trick does not make impossible algorithms possible.
It makes inner-product algorithms operate in richer spaces without explicit coordinates.

### 1.3 The Kernel Trick as Inner-Product Substitution

An algorithm is kernelizable when its input dependence appears only through inner products.

For a linear model trained by certain dual methods, the weight vector can often be written as

$$
\mathbf{w} =
\sum_{i=1}^n \alpha_i y^{(i)} \phi(\mathbf{x}^{(i)}).
$$

Then prediction becomes

$$
f(\mathbf{x})
=
\left\langle
\sum_{i=1}^n \alpha_i y^{(i)} \phi(\mathbf{x}^{(i)}),
\phi(\mathbf{x})
\right\rangle_{\mathcal{H}}
+ b.
$$

By linearity,

$$
f(\mathbf{x})
=
\sum_{i=1}^n \alpha_i y^{(i)}
\langle \phi(\mathbf{x}^{(i)}), \phi(\mathbf{x}) \rangle_{\mathcal{H}}
+ b.
$$

Replace the inner product with a kernel:

$$
f(\mathbf{x})
=
\sum_{i=1}^n \alpha_i y^{(i)}
k(\mathbf{x}^{(i)}, \mathbf{x})
+ b.
$$

The model now needs only kernel evaluations between training examples and test examples.
It does not need explicit $\phi(\mathbf{x})$ coordinates.

This is why the kernel matrix becomes the central data structure:

$$
K_{ij} =
k(\mathbf{x}^{(i)}, \mathbf{x}^{(j)}).
$$

The kernel matrix is an $n \times n$ matrix of pairwise similarities.
For many algorithms, the entire training set enters only through $K$.

Kernel trick checklist:

1. Write the algorithm in terms of inner products.
2. Replace $\langle \mathbf{x}^{(i)}, \mathbf{x}^{(j)} \rangle$ with $k(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})$.
3. Ensure $k$ is a valid positive semidefinite kernel.
4. Solve the resulting finite-dimensional problem over coefficients.
5. Predict using kernel evaluations against training points.

The phrase "trick" can be misleading.
The kernel trick is not a shortcut around mathematics.
It is a theorem-supported representation choice.

### 1.4 Historical Path

Kernel methods have a long history.
They did not begin with modern SVMs.
They grew out of approximation theory, potential functions, integral operators, and Hilbert-space methods.

Some milestones:

| Period | Development | Why it matters |
| --- | --- | --- |
| 1909 | Mercer's theorem | Connects positive kernels with eigenfunction expansions. |
| 1950 | Aronszajn's RKHS theory | Provides the Hilbert-space foundation for reproducing kernels. |
| 1960s | Potential function methods | Early pattern-recognition use of kernel-like similarity functions. |
| 1990s | Support vector machines | Convex margin classifiers made kernels central in ML. |
| 2000s | Gaussian processes and kernel machines | Kernels became standard tools for Bayesian nonparametrics and nonlinear learning. |
| 2007 | Random Fourier features | Scalable approximations made large kernel methods practical. |
| 2018 | Neural tangent kernel | Kernel theory became a tool for analyzing wide neural networks. |

The deep-learning era did not erase kernels.
It changed where kernels appear.
Sometimes they appear as explicit algorithms.
Sometimes they appear as covariance functions.
Sometimes they appear as theory for training dynamics.
Sometimes they appear as similarity functions inside attention, retrieval, and embedding systems.

---

## 2. Formal Definitions

This section gives the exact mathematical objects.
The definitions are intentionally finite-first.
Most practical kernel work begins with a finite dataset and a Gram matrix.
The infinite-dimensional RKHS story comes after the finite matrix tests.

### 2.1 Feature Maps and Feature Spaces

Let $\mathcal{X}$ be an input set.
Let $\mathcal{H}$ be a Hilbert space with inner product $\langle \cdot, \cdot \rangle_{\mathcal{H}}$.
A feature map is a function

$$
\phi : \mathcal{X} \to \mathcal{H}.
$$

The feature space is the target space $\mathcal{H}$.
The coordinates of $\phi(\mathbf{x})$ may be explicit, implicit, finite, or infinite.

Examples:

1. Identity feature map:

$$
\phi(\mathbf{x}) = \mathbf{x},
$$

with $\mathcal{H} = \mathbb{R}^d$.

2. Polynomial feature map:

$$
\phi(\mathbf{x})
=
\text{all monomials in } \mathbf{x} \text{ up to degree } p.
$$

3. Fourier feature map for a shift-invariant kernel:

$$
\phi_{\boldsymbol{\omega}}(\mathbf{x})
=
\sqrt{2}\cos(\boldsymbol{\omega}^\top \mathbf{x} + b).
$$

4. Function-space feature map:

$$
\phi(\mathbf{x}) = k(\mathbf{x}, \cdot),
$$

which maps each input to a function of another input.

Non-examples:

- A mapping into a set with no inner product is not enough for the kernel trick.
- A similarity score with no positive semidefinite property may fail to define any Hilbert-space feature map.
- A learned embedding model is not automatically a kernel unless the induced similarity satisfies the kernel conditions.

The feature-space viewpoint explains why kernels are geometric.
Learning becomes linear geometry in $\mathcal{H}$.
Classification becomes hyperplane separation.
Regression becomes regularized function fitting.
PCA becomes variance decomposition in feature space.
Gaussian process covariance becomes inner-product structure over functions.

### 2.2 Kernel Functions and Gram Matrices

A kernel function is a function

$$
k : \mathcal{X} \times \mathcal{X} \to \mathbb{R}.
$$

When a feature map exists, the kernel has the form

$$
k(\mathbf{x}, \mathbf{z})
=
\langle \phi(\mathbf{x}), \phi(\mathbf{z}) \rangle_{\mathcal{H}}.
$$

Given data $\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(n)}$, the Gram matrix is

$$
K =
\begin{bmatrix}
k(\mathbf{x}^{(1)}, \mathbf{x}^{(1)}) & \cdots & k(\mathbf{x}^{(1)}, \mathbf{x}^{(n)}) \\
\vdots & \ddots & \vdots \\
k(\mathbf{x}^{(n)}, \mathbf{x}^{(1)}) & \cdots & k(\mathbf{x}^{(n)}, \mathbf{x}^{(n)})
\end{bmatrix}.
$$

In compact notation:

$$
K_{ij} = k(\mathbf{x}^{(i)}, \mathbf{x}^{(j)}).
$$

If the kernel comes from a feature map, then

$$
K_{ij}
=
\langle \phi(\mathbf{x}^{(i)}), \phi(\mathbf{x}^{(j)}) \rangle_{\mathcal{H}}.
$$

For any coefficient vector $\mathbf{c} \in \mathbb{R}^n$,

$$
\mathbf{c}^\top K \mathbf{c}
=
\sum_{i=1}^n \sum_{j=1}^n c_i c_j k(\mathbf{x}^{(i)}, \mathbf{x}^{(j)}).
$$

If $k$ comes from an inner product, then

$$
\mathbf{c}^\top K \mathbf{c}
=
\left\lVert
\sum_{i=1}^n c_i \phi(\mathbf{x}^{(i)})
\right\rVert_{\mathcal{H}}^2
\ge 0.
$$

This equation is the most important finite test:
a valid kernel produces positive semidefinite Gram matrices for every finite dataset.

### 2.3 Positive Semidefinite Kernels

A symmetric function $k : \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ is a positive semidefinite kernel if for every finite set $\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(n)} \in \mathcal{X}$ and every $\mathbf{c} \in \mathbb{R}^n$,

$$
\sum_{i=1}^n \sum_{j=1}^n c_i c_j k(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})
\ge 0.
$$

Equivalently, the Gram matrix $K$ satisfies

$$
K \succeq 0.
$$

There are two conditions:

1. Symmetry:

$$
k(\mathbf{x}, \mathbf{z}) = k(\mathbf{z}, \mathbf{x}).
$$

2. Positive semidefiniteness:

$$
\mathbf{c}^\top K \mathbf{c} \ge 0
\quad
\text{for all finite datasets and all } \mathbf{c}.
$$

Do not confuse these with pairwise positivity.
A kernel can take negative values and still be valid.
The linear kernel $k(\mathbf{x}, \mathbf{z}) = \mathbf{x}^\top \mathbf{z}$ can be negative.
It is still a valid kernel.

Do not confuse positive semidefinite with positive definite.
Positive semidefinite allows zero eigenvalues.
This matters because duplicate points, redundant features, and finite-dimensional feature maps often produce singular Gram matrices.

Finite check:
for a proposed kernel and a finite dataset, compute eigenvalues of $K$.
If any eigenvalue is significantly negative, the kernel failed on that dataset.
If all eigenvalues are nonnegative on one dataset, that is evidence, not proof, that the kernel is valid.

### 2.4 Valid Kernels and Counterexamples

Valid kernels:

| Kernel | Formula | Valid because |
| --- | --- | --- |
| Linear | $k(\mathbf{x}, \mathbf{z}) = \mathbf{x}^\top \mathbf{z}$ | Direct inner product. |
| Polynomial | $k(\mathbf{x}, \mathbf{z}) = (\mathbf{x}^\top \mathbf{z} + c)^p$ with $c \ge 0$ and integer $p \ge 1$ | Explicit finite feature expansion. |
| RBF | $k(\mathbf{x}, \mathbf{z}) = \exp(-\gamma \lVert \mathbf{x} - \mathbf{z} \rVert_2^2)$ with $\gamma > 0$ | Positive Fourier transform by Bochner's theorem. |
| Laplacian | $k(\mathbf{x}, \mathbf{z}) = \exp(-\gamma \lVert \mathbf{x} - \mathbf{z} \rVert_1)$ | Product of valid one-dimensional Laplace kernels. |
| Sum | $k_1 + k_2$ | Sum of PSD matrices is PSD. |
| Product | $k_1 k_2$ | Schur product theorem. |

Invalid or suspicious similarities:

| Function | Problem |
| --- | --- |
| $k(\mathbf{x}, \mathbf{z}) = -\lVert \mathbf{x} - \mathbf{z} \rVert_2^2$ | Not generally PSD as a kernel, though related to conditionally negative definite distances. |
| $k(\mathbf{x}, \mathbf{z}) = \sin(\mathbf{x}^\top \mathbf{z})$ | Not generally PSD. |
| $k(\mathbf{x}, \mathbf{z}) = \mathbb{1}[\mathbf{x} \ne \mathbf{z}]$ | Can produce indefinite Gram matrices. |
| Arbitrary cosine of learned embeddings | May be useful similarity but not automatically a kernel over the original input set. |

Counterexample method:

1. Choose a small dataset.
2. Form the Gram matrix.
3. Compute eigenvalues.
4. Find a vector $\mathbf{c}$ such that $\mathbf{c}^\top K \mathbf{c} < 0$.

This method is not just a homework trick.
Indefinite similarity matrices appear in practice when people combine scores, distances, edit penalties, or learned similarities without checking PSD structure.
Some algorithms can work with indefinite similarities, but the RKHS theory no longer applies directly.

### 2.5 Kernel Normalization and Centering

Kernel values can be sensitive to scale.
A normalized kernel is often defined as

$$
\tilde{k}(\mathbf{x}, \mathbf{z})
=
\frac{k(\mathbf{x}, \mathbf{z})}
{\sqrt{k(\mathbf{x}, \mathbf{x})k(\mathbf{z}, \mathbf{z})}}.
$$

If $k(\mathbf{x}, \mathbf{x}) > 0$ and $k(\mathbf{z}, \mathbf{z}) > 0$, then

$$
\tilde{k}(\mathbf{x}, \mathbf{x}) = 1.
$$

In feature-space language,

$$
\tilde{k}(\mathbf{x}, \mathbf{z})
=
\frac{
\langle \phi(\mathbf{x}), \phi(\mathbf{z}) \rangle_{\mathcal{H}}
}{
\lVert \phi(\mathbf{x}) \rVert_{\mathcal{H}}
\lVert \phi(\mathbf{z}) \rVert_{\mathcal{H}}
}.
$$

So normalized kernels are cosine similarities in feature space.

Kernel centering is different.
Kernel PCA and related methods require centered feature vectors:

$$
\tilde{\phi}(\mathbf{x}^{(i)})
=
\phi(\mathbf{x}^{(i)}) -
\frac{1}{n}\sum_{j=1}^n \phi(\mathbf{x}^{(j)}).
$$

The centered Gram matrix is

$$
K_c =
HKH,
$$

where

$$
H =
I_n - \frac{1}{n}\mathbf{1}\mathbf{1}^\top.
$$

Centering removes the feature-space mean.
Without centering, kernel PCA can waste its first component on the mean offset rather than variance structure.

Normalization controls diagonal scale.
Centering controls feature-space mean.
They solve different problems.

---

## 3. RKHS Foundations

The finite Gram matrix story explains how kernels are used computationally.
The RKHS story explains what functions are being learned.

A Reproducing Kernel Hilbert Space is a Hilbert space of functions where point evaluation is continuous and represented by inner product with a kernel section.
That sentence is dense.
This section unpacks it carefully.

### 3.1 Reproducing Kernel Hilbert Spaces

Let $\mathcal{X}$ be an input set.
Let $\mathcal{H}$ be a Hilbert space whose elements are functions

$$
f : \mathcal{X} \to \mathbb{R}.
$$

For each $\mathbf{x} \in \mathcal{X}$, define the evaluation functional

$$
L_{\mathbf{x}}(f) = f(\mathbf{x}).
$$

The space $\mathcal{H}$ is a Reproducing Kernel Hilbert Space if every evaluation functional is continuous.
By the Riesz representation theorem for Hilbert spaces, continuity implies that for every $\mathbf{x}$ there exists a unique element $k(\mathbf{x}, \cdot) \in \mathcal{H}$ such that

$$
f(\mathbf{x})
=
\langle f, k(\mathbf{x}, \cdot) \rangle_{\mathcal{H}}
$$

for every $f \in \mathcal{H}$.

The function $k$ is the reproducing kernel.

This definition has three important consequences:

1. Each input $\mathbf{x}$ corresponds to a function $k(\mathbf{x}, \cdot)$.
2. Evaluating $f$ at $\mathbf{x}$ is an inner product in $\mathcal{H}$.
3. The kernel is not merely a similarity score; it is the object that represents point evaluation.

The canonical feature map of an RKHS is

$$
\phi(\mathbf{x}) = k(\mathbf{x}, \cdot).
$$

Then

$$
k(\mathbf{x}, \mathbf{z})
=
\langle k(\mathbf{x}, \cdot), k(\mathbf{z}, \cdot) \rangle_{\mathcal{H}}.
$$

This means every RKHS kernel is automatically a feature-map kernel.

### 3.2 The Reproducing Property

The reproducing property is

$$
f(\mathbf{x})
=
\langle f, k(\mathbf{x}, \cdot) \rangle_{\mathcal{H}}.
$$

It says that function evaluation is inner product geometry.

If $f$ is represented as

$$
f =
\sum_{i=1}^n \alpha_i k(\mathbf{x}^{(i)}, \cdot),
$$

then

$$
f(\mathbf{x})
=
\left\langle
\sum_{i=1}^n \alpha_i k(\mathbf{x}^{(i)}, \cdot),
k(\mathbf{x}, \cdot)
\right\rangle_{\mathcal{H}}.
$$

By linearity,

$$
f(\mathbf{x})
=
\sum_{i=1}^n \alpha_i
\langle k(\mathbf{x}^{(i)}, \cdot), k(\mathbf{x}, \cdot) \rangle_{\mathcal{H}}.
$$

Using the kernel identity,

$$
f(\mathbf{x})
=
\sum_{i=1}^n \alpha_i k(\mathbf{x}^{(i)}, \mathbf{x}).
$$

This is exactly the prediction form used by kernel machines.

The reproducing property explains why coefficients over training points are enough.
The learned function lives in a Hilbert space, but prediction reduces to kernel evaluations.

### 3.3 Moore-Aronszajn Theorem

The Moore-Aronszajn theorem states:
every positive semidefinite kernel $k$ on $\mathcal{X}$ uniquely determines an RKHS $\mathcal{H}_k$ for which $k$ is the reproducing kernel.

This theorem closes the loop:

```text
positive semidefinite kernel
        |
        v
unique RKHS of functions
        |
        v
feature map x -> k(x, .)
        |
        v
inner product recovers k(x, z)
```

The theorem is important because it means we do not need to explicitly construct $\phi$.
If we prove that $k$ is positive semidefinite, then an RKHS exists.
The feature space is guaranteed by the theorem.

This is why kernel validity matters.
An invalid similarity may still produce useful numbers.
But it does not automatically give a Hilbert space of functions, a reproducing property, or a standard regularization theory.

### 3.4 Mercer Expansion and Eigenfunctions

Mercer's theorem is one of the classical links between kernels and spectral theory.
Under suitable conditions, such as a continuous positive semidefinite kernel on a compact domain with respect to a measure, the kernel has an expansion

$$
k(\mathbf{x}, \mathbf{z})
=
\sum_{m=1}^{\infty}
\lambda_m \psi_m(\mathbf{x})\psi_m(\mathbf{z}),
$$

where $\lambda_m \ge 0$ and $\{\psi_m\}$ are orthonormal eigenfunctions of the associated integral operator.

The integral operator is

$$
(T_k f)(\mathbf{x})
=
\int_{\mathcal{X}} k(\mathbf{x}, \mathbf{z})f(\mathbf{z})\,d\mu(\mathbf{z}).
$$

The eigenfunction equation is

$$
T_k \psi_m =
\lambda_m \psi_m.
$$

The expansion resembles matrix eigendecomposition.
The kernel acts like an infinite matrix.
The eigenfunctions play the role of eigenvectors.
The eigenvalues tell which directions in function space are favored.

Practical interpretation:

- large eigenvalues correspond to functions the kernel represents easily;
- small eigenvalues correspond to functions strongly penalized by the RKHS norm;
- kernel ridge regression shrinks more strongly along low-eigenvalue directions;
- NTK eigenvalues can predict which target functions are learned quickly by wide networks.

Mercer is not required every time we use kernels.
The finite Gram matrix definition is more general in many ML settings.
But Mercer gives spectral intuition for smoothness and generalization.

### 3.5 RKHS Norm as Smoothness Control

The RKHS norm $\lVert f \rVert_{\mathcal{H}_k}$ measures function complexity relative to the kernel.
Different kernels create different notions of complexity.

For a regularized learning problem,

$$
\min_{f \in \mathcal{H}_k}
\sum_{i=1}^n
\ell(f(\mathbf{x}^{(i)}), y^{(i)})
+
\lambda \lVert f \rVert_{\mathcal{H}_k}^2,
$$

the first term fits the data.
The second term penalizes functions that are complex in the RKHS.

The same function can be simple under one kernel and complex under another.
For example:

- an RBF kernel favors very smooth functions;
- a periodic kernel favors periodic functions;
- a linear kernel favors linear functions;
- a string kernel favors functions expressible through substring features;
- a graph kernel favors functions aligned with graph similarity.

This is why choosing a kernel is choosing an inductive bias.

The RKHS norm is not merely a generic "size" penalty.
It encodes which functions are cheap and which are expensive.
In a Gaussian process, this same kernel encodes covariance.
In kernel ridge regression, it encodes regularization.
In SVMs, it encodes margin geometry.

---

## 4. Common Kernels

Kernel choice determines the geometry of learning.
This section catalogs the most common kernels and the assumptions they encode.

### 4.1 Linear and Polynomial Kernels

The linear kernel is

$$
k(\mathbf{x}, \mathbf{z})
=
\mathbf{x}^\top \mathbf{z}.
$$

It corresponds to the identity feature map.
Its Gram matrix is

$$
K = XX^\top
$$

when rows of $X$ are examples.

Use the linear kernel when:

- the input features are already expressive;
- dimension is high and nonlinear expansion is unnecessary;
- interpretability and speed matter;
- the model should behave like a regularized linear method.

The polynomial kernel is

$$
k(\mathbf{x}, \mathbf{z})
=
(\mathbf{x}^\top \mathbf{z} + c)^p,
$$

where $c \ge 0$ and $p$ is a positive integer.

It represents interactions among input coordinates.
For $p=2$, it includes pairwise products.
For $p=3$, it includes cubic interactions.
Higher degree increases expressivity but can overfit quickly.

Polynomial kernels are useful for:

- controlled feature interactions;
- small or medium-dimensional tabular data;
- settings where multiplicative feature relationships matter;
- teaching the kernel trick because the feature map can be written explicitly.

AI warning:
large polynomial degree can behave like an unstable feature explosion.
Regularization and scaling are not optional.

### 4.2 RBF Kernel

The radial basis function kernel, also called the Gaussian kernel, is

$$
k(\mathbf{x}, \mathbf{z})
=
\exp(-\gamma \lVert \mathbf{x} - \mathbf{z} \rVert_2^2),
$$

where $\gamma > 0$.
Equivalently,

$$
k(\mathbf{x}, \mathbf{z})
=
\exp\left(
-\frac{\lVert \mathbf{x} - \mathbf{z} \rVert_2^2}{2\sigma^2}
\right),
$$

with $\gamma = 1/(2\sigma^2)$.

The RBF kernel is shift-invariant:

$$
k(\mathbf{x}, \mathbf{z}) = \kappa(\mathbf{x} - \mathbf{z}).
$$

It is also radial:
the value depends only on distance.

Bandwidth interpretation:

- small $\sigma$ means very local similarity;
- large $\sigma$ means broad similarity;
- too small $\sigma$ can make $K$ close to identity;
- too large $\sigma$ can make $K$ close to a constant matrix.

Both extremes are bad.
If $K \approx I_n$, the model can memorize but generalize poorly.
If $K \approx \mathbf{1}\mathbf{1}^\top$, the model has too little resolution.

The RBF kernel is universal on many compact domains.
Informally, its RKHS is rich enough to approximate many continuous functions.
But universality does not remove the need for regularization.
Richness without control is overfitting.

### 4.3 Laplacian and Matern Kernels

The Laplacian kernel is

$$
k(\mathbf{x}, \mathbf{z})
=
\exp(-\gamma \lVert \mathbf{x} - \mathbf{z} \rVert_1).
$$

Compared with RBF, it produces rougher functions.
It can be useful when the target function has sharper changes.

The Matern family is a common Gaussian-process kernel family.
For distance $r = \lVert \mathbf{x} - \mathbf{z} \rVert_2$, it is

$$
k_{\nu}(r)
=
\frac{2^{1-\nu}}{\Gamma(\nu)}
\left(
\frac{\sqrt{2\nu}r}{\ell}
\right)^\nu
K_{\nu}^{\mathrm{Bessel}}
\left(
\frac{\sqrt{2\nu}r}{\ell}
\right),
$$

where $\nu > 0$ controls smoothness, $\ell > 0$ is lengthscale, and $K_{\nu}^{\mathrm{Bessel}}$ is the modified Bessel function of the second kind.

Common special cases avoid Bessel functions:

$$
k_{\nu=1/2}(r)
=
\exp(-r/\ell).
$$

$$
k_{\nu=3/2}(r)
=
\left(1 + \frac{\sqrt{3}r}{\ell}\right)
\exp\left(-\frac{\sqrt{3}r}{\ell}\right).
$$

$$
k_{\nu=5/2}(r)
=
\left(1 + \frac{\sqrt{5}r}{\ell} + \frac{5r^2}{3\ell^2}\right)
\exp\left(-\frac{\sqrt{5}r}{\ell}\right).
$$

Matern smoothness interpretation:

| Kernel | Smoothness assumption |
| --- | --- |
| $\nu = 1/2$ | Rough, continuous but not very smooth. |
| $\nu = 3/2$ | Once-differentiable style behavior. |
| $\nu = 5/2$ | Smoother functions. |
| $\nu \to \infty$ | Approaches RBF smoothness. |

In scientific ML and Bayesian optimization, Matern kernels are often more realistic than RBF kernels.
Many physical functions are smooth but not infinitely smooth.

### 4.4 Periodic, Rational Quadratic, String, and Graph Kernels

The periodic kernel is

$$
k(x,z)
=
\exp\left(
-\frac{2\sin^2(\pi \lvert x-z \rvert / p)}{\ell^2}
\right).
$$

It encodes periodic structure with period $p$.
It is useful for seasonal signals and cyclic processes.

The rational quadratic kernel is

$$
k(\mathbf{x}, \mathbf{z})
=
\left(
1 +
\frac{\lVert \mathbf{x} - \mathbf{z} \rVert_2^2}{2\alpha\ell^2}
\right)^{-\alpha}.
$$

It can be interpreted as a scale mixture of RBF kernels.
It is useful when multiple lengthscales are present.

String kernels compare sequences by shared subsequences or substrings.
They were important in bioinformatics and text classification.
The general form is

$$
k(s,t)
=
\sum_{u \in \mathcal{A}^*}
\phi_u(s)\phi_u(t),
$$

where $\phi_u(s)$ counts or weights occurrences of subsequence $u$ in string $s$.

Graph kernels compare graphs through walks, paths, subtrees, graphlets, or spectral features.
They are related to graph representation learning but are not the same as graph neural networks.
For full graph neural network treatment, see [Graph Neural Networks](../../11-Graph-Theory/05-Graph-Neural-Networks/notes.md).

These structured kernels show that kernels are not limited to vectors.
A kernel needs a valid PSD similarity over an input set.
The input set can be strings, trees, graphs, distributions, molecules, or programs.

### 4.5 Kernel Composition Rules

Kernel design often uses closure rules.
If $k_1$ and $k_2$ are valid kernels, then the following are valid under standard conditions:

| Construction | Formula |
| --- | --- |
| Nonnegative scaling | $ak_1$ for $a \ge 0$ |
| Sum | $k_1 + k_2$ |
| Product | $k_1 k_2$ |
| Pointwise limit | $\lim_m k_m$ when the limit exists |
| Pullback by a map | $k(g(\mathbf{x}), g(\mathbf{z}))$ |
| Tensor-style product | $k_X(\mathbf{x}, \mathbf{z})k_Y(\mathbf{u}, \mathbf{v})$ |
| Direct-sum kernel | $k_X + k_Y$ |

Why sums work:
for any finite dataset,

$$
\mathbf{c}^\top (K_1 + K_2)\mathbf{c}
=
\mathbf{c}^\top K_1\mathbf{c}
+
\mathbf{c}^\top K_2\mathbf{c}
\ge 0.
$$

Why products work:
the Gram matrix of $k_1k_2$ is the Hadamard product

$$
K_1 \odot K_2.
$$

The Schur product theorem says the Hadamard product of PSD matrices is PSD.

Composition examples:

$$
k(\mathbf{x}, \mathbf{z})
=
k_{\text{RBF}}(\mathbf{x}, \mathbf{z})
+
k_{\text{periodic}}(\mathbf{x}, \mathbf{z})
$$

models smooth trend plus periodic structure.

$$
k(\mathbf{x}, \mathbf{z})
=
k_{\text{RBF}}(\mathbf{x}, \mathbf{z})
k_{\text{linear}}(\mathbf{x}, \mathbf{z})
$$

models smooth local similarity modulated by linear alignment.

Kernel composition is inductive-bias engineering.
It is the kernel-method analogue of architecture design.

---

## 5. Core Kernel Algorithms

This section shows how kernels enter algorithms.
The common pattern is finite coefficient representation.
The learned object may live in a function space, but training solves for a finite vector.

### 5.1 Kernel Perceptron

The ordinary perceptron learns a linear classifier

$$
f(\mathbf{x}) =
\operatorname{sign}(\mathbf{w}^\top \mathbf{x}).
$$

With labels $y^{(i)} \in \{-1, +1\}$, a mistake update is

$$
\mathbf{w}
\leftarrow
\mathbf{w} + y^{(i)}\mathbf{x}^{(i)}.
$$

After many updates, the weight vector is a linear combination of training examples:

$$
\mathbf{w}
=
\sum_{i=1}^n \alpha_i y^{(i)} \mathbf{x}^{(i)}.
$$

In feature space,

$$
\mathbf{w}
=
\sum_{i=1}^n \alpha_i y^{(i)} \phi(\mathbf{x}^{(i)}).
$$

Prediction becomes

$$
f(\mathbf{x})
=
\operatorname{sign}
\left(
\sum_{i=1}^n
\alpha_i y^{(i)} k(\mathbf{x}^{(i)}, \mathbf{x})
\right).
$$

The kernel perceptron update increments $\alpha_i$ when example $i$ is misclassified.

Algorithm:

1. Initialize $\boldsymbol{\alpha} = \mathbf{0}$.
2. For each example $i$, compute score

$$
s_i =
\sum_{j=1}^n
\alpha_j y^{(j)} k(\mathbf{x}^{(j)}, \mathbf{x}^{(i)}).
$$

3. If $y^{(i)}s_i \le 0$, set $\alpha_i \leftarrow \alpha_i + 1$.
4. Repeat for multiple passes.

The kernel perceptron is not usually the best modern kernel classifier.
But it is the cleanest algorithmic example of kernelization.

### 5.2 Support Vector Machines

Support vector machines use kernels to build maximum-margin classifiers.
For a feature map $\phi$, the hard-margin primal problem is

$$
\min_{\mathbf{w}, b}
\frac{1}{2}\lVert \mathbf{w} \rVert_{\mathcal{H}}^2
$$

subject to

$$
y^{(i)}
\left(
\langle \mathbf{w}, \phi(\mathbf{x}^{(i)}) \rangle_{\mathcal{H}} + b
\right)
\ge 1
$$

for all $i$.

The dual problem is

$$
\max_{\boldsymbol{\alpha}}
\sum_{i=1}^n \alpha_i
-
\frac{1}{2}
\sum_{i=1}^n\sum_{j=1}^n
\alpha_i\alpha_j y^{(i)}y^{(j)}
k(\mathbf{x}^{(i)}, \mathbf{x}^{(j)}),
$$

subject to

$$
\alpha_i \ge 0
\quad
\text{and}
\quad
\sum_{i=1}^n \alpha_i y^{(i)} = 0.
$$

The soft-margin SVM adds slack variables and a penalty parameter $C$.
In the dual, this gives box constraints:

$$
0 \le \alpha_i \le C.
$$

Prediction is

$$
f(\mathbf{x})
=
\operatorname{sign}
\left(
\sum_{i=1}^n
\alpha_i y^{(i)}
k(\mathbf{x}^{(i)}, \mathbf{x})
+ b
\right).
$$

Only points with $\alpha_i > 0$ contribute.
These are support vectors.

SVM intuition:

- the margin is measured in feature-space norm;
- the kernel determines the feature-space geometry;
- support vectors are the examples that define the boundary;
- $C$ controls the tradeoff between margin and training errors;
- kernel bandwidth controls the flexibility of the boundary.

Important scoping note:
this section does not reteach constrained optimization or KKT conditions in full.
For the full optimization machinery, see [Constrained Optimization](../../08-Optimization/04-Constrained-Optimization/notes.md).

### 5.3 Kernel Ridge Regression and Representer Theorem

Kernel ridge regression solves

$$
\min_{f \in \mathcal{H}_k}
\sum_{i=1}^n
\left(
y^{(i)} - f(\mathbf{x}^{(i)})
\right)^2
+
\lambda \lVert f \rVert_{\mathcal{H}_k}^2.
$$

The representer theorem says the minimizer has the finite form

$$
f^*(\mathbf{x})
=
\sum_{i=1}^n
\alpha_i k(\mathbf{x}^{(i)}, \mathbf{x}).
$$

This is remarkable.
The optimization is over an infinite-dimensional function space.
The solution lies in the span of $n$ kernel sections centered at training inputs.

Let $\mathbf{y} \in \mathbb{R}^n$ collect labels.
Let $K$ be the Gram matrix.
Predictions on training data are

$$
\hat{\mathbf{y}} = K\boldsymbol{\alpha}.
$$

The objective becomes

$$
\lVert \mathbf{y} - K\boldsymbol{\alpha} \rVert_2^2
+
\lambda \boldsymbol{\alpha}^\top K\boldsymbol{\alpha}.
$$

The closed-form coefficient vector is

$$
\boldsymbol{\alpha}
=
(K + \lambda I_n)^{-1}\mathbf{y}.
$$

Some conventions use $n\lambda$ instead of $\lambda$.
Always check the objective scaling.

Prediction for a test point $\mathbf{x}_*$ is

$$
f^*(\mathbf{x}_*)
=
\mathbf{k}_*^\top \boldsymbol{\alpha},
$$

where

$$
\mathbf{k}_*
=
\begin{bmatrix}
k(\mathbf{x}^{(1)}, \mathbf{x}_*) \\
\vdots \\
k(\mathbf{x}^{(n)}, \mathbf{x}_*)
\end{bmatrix}.
$$

Regularization interpretation:

- large $\lambda$ shrinks coefficients and smooths the function;
- small $\lambda$ fits training data more closely;
- $\lambda$ also improves conditioning of $K + \lambda I_n$;
- when $K$ is nearly singular, the ridge term is numerically essential.

### 5.4 Kernel PCA

PCA finds directions of high variance.
Kernel PCA finds high-variance directions in feature space.

Let $\phi(\mathbf{x}^{(i)})$ be centered in $\mathcal{H}$.
The feature-space covariance operator is

$$
C =
\frac{1}{n}
\sum_{i=1}^n
\phi(\mathbf{x}^{(i)}) \otimes \phi(\mathbf{x}^{(i)}).
$$

Instead of diagonalizing $C$ directly, kernel PCA diagonalizes the centered Gram matrix

$$
K_c = HKH.
$$

If

$$
K_c \mathbf{v}_m = \lambda_m \mathbf{v}_m,
$$

then $\mathbf{v}_m$ gives coefficients for the $m$-th feature-space principal direction.

The embedding coordinate for a point $\mathbf{x}$ is computed through kernel evaluations against training points.

Kernel PCA is useful for:

- nonlinear dimensionality reduction;
- visualizing curved manifolds;
- denoising in feature space;
- teaching how kernel methods extend linear algorithms.

Important scoping note:
this section uses PCA only as a kernelized algorithm.
For full PCA theory, see [Principal Component Analysis](../../03-Advanced-Linear-Algebra/03-Principal-Component-Analysis/notes.md).

Kernel PCA common failure:
forgetting to center the kernel matrix.
PCA assumes centered data.
In kernel PCA, centering must happen in feature space, which is why $K_c = HKH$ appears.

### 5.5 Gaussian Processes

A Gaussian process is a distribution over functions.
It is written

$$
f \sim \mathcal{GP}(m, k),
$$

where $m$ is the mean function and $k$ is the covariance kernel.

For any finite set of inputs, the function values are jointly Gaussian:

$$
\begin{bmatrix}
f(\mathbf{x}^{(1)}) \\
\vdots \\
f(\mathbf{x}^{(n)})
\end{bmatrix}
\sim
\mathcal{N}(\mathbf{m}, K).
$$

Here

$$
m_i = m(\mathbf{x}^{(i)})
$$

and

$$
K_{ij} = k(\mathbf{x}^{(i)}, \mathbf{x}^{(j)}).
$$

For noisy observations

$$
y^{(i)} = f(\mathbf{x}^{(i)}) + \epsilon_i,
$$

with

$$
\epsilon_i \sim \mathcal{N}(0, \sigma_n^2),
$$

the predictive mean at a test input $\mathbf{x}_*$ is

$$
\mu_*
=
\mathbf{k}_*^\top
(K + \sigma_n^2 I_n)^{-1}
\mathbf{y}.
$$

The predictive variance is

$$
\sigma_*^2
=
k(\mathbf{x}_*, \mathbf{x}_*)
-
\mathbf{k}_*^\top
(K + \sigma_n^2 I_n)^{-1}
\mathbf{k}_*.
$$

Gaussian process regression has the same algebraic core as kernel ridge regression.
The GP adds probabilistic interpretation and uncertainty.

Important scoping note:
this section gives the kernel-method view of Gaussian processes.
For full Bayesian inference background, see [Bayesian Inference](../../07-Statistics/04-Bayesian-Inference/notes.md).

GP kernel interpretation:

- the kernel is a covariance function;
- nearby points under the kernel have correlated function values;
- the lengthscale controls how quickly correlation decays;
- the posterior mean interpolates or smooths observations;
- the posterior variance grows away from observed data.

This uncertainty is one reason Gaussian processes remain useful in Bayesian optimization, active learning, calibration, and scientific modeling.

---

## 6. Computation and Scalability

Kernel methods are elegant, but the kernel matrix is expensive.
This section treats computation as part of the mathematics, not an implementation afterthought.

### 6.1 Kernel Matrix Cost

For $n$ training examples, the Gram matrix has $n^2$ entries.
Memory cost is

$$
O(n^2).
$$

Naive construction cost is

$$
O(n^2 d)
$$

for vector inputs in $\mathbb{R}^d$ when each kernel evaluation costs $O(d)$.

Solving exact kernel ridge regression by dense linear algebra costs

$$
O(n^3)
$$

in general.

Prediction for one test point costs

$$
O(n)
$$

kernel evaluations unless the model is sparse or approximated.

This is the core scalability problem.
The feature space may be infinite-dimensional, but the training set creates an $n \times n$ object.

Practical thresholds:

- $n = 10^3$ is usually easy;
- $n = 10^4$ is possible with care;
- $n = 10^5$ is expensive for exact kernels;
- $n = 10^6$ usually requires approximation, structure, or distributed methods.

The bottleneck is often memory before arithmetic.
A float64 Gram matrix for $n=100{,}000$ would need roughly 80 GB just for entries.

### 6.2 Conditioning and Regularization

Kernel matrices can be ill-conditioned.
This happens when:

- points are nearly duplicated;
- the bandwidth is too large;
- the bandwidth is too small;
- features are poorly scaled;
- the kernel has rapidly decaying eigenvalues.

Condition number is

$$
\kappa(K)
=
\frac{\lambda_{\max}(K)}{\lambda_{\min}(K)}
$$

when $K$ is positive definite.

For kernel ridge regression, the system is

$$
(K + \lambda I_n)\boldsymbol{\alpha}
=
\mathbf{y}.
$$

The ridge shift changes eigenvalues:

$$
\lambda_m(K + \lambda I_n)
=
\lambda_m(K) + \lambda.
$$

So regularization improves conditioning.
This is both statistical and numerical.

Numerical practices:

- standardize input features before distance-based kernels;
- tune bandwidth and regularization jointly;
- add jitter when doing Gaussian process Cholesky factorization;
- prefer Cholesky solves over explicit matrix inverse;
- monitor eigenvalues of $K$ or $K + \lambda I_n$.

In GP code, a small jitter term is often written as

$$
K_{\text{stable}} =
K + (\sigma_n^2 + \epsilon)I_n.
$$

The $\epsilon$ term is not a probabilistic modeling claim.
It is numerical stabilization.

### 6.3 Nystrom Approximation

The Nystrom approximation uses a subset of $m \ll n$ landmark points.
Partition the kernel matrix conceptually into

$$
K_{nm}
\in
\mathbb{R}^{n \times m}
$$

and

$$
K_{mm}
\in
\mathbb{R}^{m \times m}.
$$

The approximate Gram matrix is

$$
K
\approx
K_{nm}K_{mm}^{\dagger}K_{mn}.
$$

If $K_{mm}$ is regularized,

$$
K
\approx
K_{nm}(K_{mm} + \epsilon I_m)^{-1}K_{mn}.
$$

The approximation has rank at most $m$.
It reduces storage from $O(n^2)$ to roughly $O(nm)$.

Landmark choices:

- uniform random sampling;
- k-means centers;
- leverage-score sampling;
- inducing points optimized by a model;
- domain-specific prototypes.

Nystrom intuition:
represent all points by their similarity to landmarks.
Then reconstruct the full similarity structure through the landmark-landmark matrix.

This is a low-rank approximation to kernel geometry.
It is related to matrix approximation, inducing-point Gaussian processes, and scalable spectral methods.

### 6.4 Random Fourier Features

Random Fourier features approximate shift-invariant kernels.
Bochner's theorem says a continuous shift-invariant kernel

$$
k(\mathbf{x}, \mathbf{z}) = \kappa(\mathbf{x} - \mathbf{z})
$$

is positive semidefinite if and only if $\kappa$ is the Fourier transform of a nonnegative measure.

For the RBF kernel, sample

$$
\boldsymbol{\omega}_r \sim \mathcal{N}(\mathbf{0}, 2\gamma I_d)
$$

and

$$
b_r \sim \mathcal{U}(0, 2\pi).
$$

Define random features

$$
z_r(\mathbf{x})
=
\sqrt{\frac{2}{D}}
\cos(\boldsymbol{\omega}_r^\top \mathbf{x} + b_r).
$$

Then

$$
z(\mathbf{x})^\top z(\mathbf{z})
\approx
k(\mathbf{x}, \mathbf{z}).
$$

Now a kernel method can be approximated by a linear method in $D$ random features.
Training cost shifts from Gram-matrix algorithms to linear-model algorithms.

Benefits:

- storage $O(nD)$ instead of $O(n^2)$;
- prediction $O(D)$ instead of $O(n)$;
- compatibility with SGD;
- easy integration into neural networks.

Tradeoffs:

- approximation variance decreases as $D$ grows;
- poor random features need many dimensions;
- input scaling still matters;
- structured random features may be faster but more complex.

Random Fourier features are a key example of turning an implicit kernel feature map into an explicit approximate feature map.

### 6.5 When Kernels Are Practical

Kernels are practical when the data scale and inductive bias fit the method.

Good kernel settings:

- small to medium datasets;
- expensive labels;
- tabular or scientific data;
- uncertainty matters;
- smoothness or periodicity assumptions are strong;
- a domain-specific similarity is available;
- convex training is preferred;
- interpretability of inductive bias matters.

Poor kernel settings:

- massive datasets without approximation;
- high-throughput online prediction with many support vectors;
- raw image/text tasks where representation learning dominates;
- settings where the right similarity must be learned from huge data;
- datasets with severe feature scaling problems and no preprocessing.

The practical question is not "kernels or deep learning?"
The practical question is "where should similarity be specified, where should it be learned, and what scale must the algorithm handle?"

Modern hybrids answer this in mixed ways:

- deep kernel learning uses neural features inside a kernel;
- Gaussian processes can sit on top of learned embeddings;
- random features can act as a fixed feature layer;
- NTK uses kernels to analyze neural training;
- retrieval systems use learned embeddings but often kernel-like similarity.

---

## 7. Kernel Methods in Modern AI

Kernels remain relevant because similarity remains central.
Deep learning changed how similarity is learned, but it did not remove the need to understand similarity geometry.

### 7.1 Similarity Search and Embeddings

Embedding systems map objects into vectors.
Semantic search, recommendation, contrastive learning, and retrieval-augmented generation often use similarity scores such as dot product or cosine similarity.

A normalized dot product is a linear kernel on normalized embeddings:

$$
k(\mathbf{u}, \mathbf{v})
=
\frac{\mathbf{u}^\top \mathbf{v}}
{\lVert \mathbf{u} \rVert_2 \lVert \mathbf{v} \rVert_2}.
$$

This does not mean every retrieval system is a kernel machine.
The embeddings are learned by a neural model, and approximate nearest-neighbor search is an engineering system.
But the similarity score is still a geometric object.

Kernel thinking asks:

- what invariances should similarity respect?
- what scale makes two items close?
- is the similarity symmetric?
- is the implied Gram matrix PSD?
- does normalization help or hurt?
- does the similarity align with downstream loss?

These questions are useful even when the final system is not a classical kernel method.

### 7.2 Kernels and Attention

Transformer attention uses scores

$$
S =
\frac{QK^\top}{\sqrt{d_k}}.
$$

Each score is an inner product between a query vector and a key vector.
Softmax then converts scores into weights.

This is not an RKHS kernel method in the classical sense because:

- queries and keys are learned and input-dependent;
- the attention matrix is usually not symmetric;
- softmax normalization is row-wise;
- attention produces weighted values, not only scalar function prediction.

Still, kernel ideas are nearby.
Linear attention methods often replace softmax attention with feature maps satisfying

$$
\exp(\mathbf{q}^\top \mathbf{k})
\approx
\phi(\mathbf{q})^\top \phi(\mathbf{k}).
$$

This is a kernel approximation idea.
It converts attention into associative linear algebra:

$$
\operatorname{Attn}(Q,K,V)
\approx
\phi(Q)(\phi(K)^\top V).
$$

The kernel lesson:
attention is not "just a kernel," but kernel feature maps help explain and approximate some attention mechanisms.

### 7.3 Neural Tangent Kernel

Consider a neural network $f(\mathbf{x}; \boldsymbol{\theta})$.
The neural tangent kernel is

$$
\Theta(\mathbf{x}, \mathbf{z})
=
\nabla_{\boldsymbol{\theta}} f(\mathbf{x}; \boldsymbol{\theta})^\top
\nabla_{\boldsymbol{\theta}} f(\mathbf{z}; \boldsymbol{\theta}).
$$

This is a kernel over inputs induced by parameter gradients.
It measures how changing parameters to affect $\mathbf{x}$ also affects $\mathbf{z}$.

In the infinite-width limit for certain architectures and initializations, the NTK remains nearly constant during training.
Gradient descent then behaves like kernel regression with kernel $\Theta$.

NTK interpretation:

- if $\Theta(\mathbf{x}, \mathbf{z})$ is large, learning on $\mathbf{x}$ strongly changes prediction at $\mathbf{z}$;
- eigenvalues of the NTK affect learning speed along target-function directions;
- wide-network training can be analyzed through kernel dynamics;
- feature learning is limited in the lazy-training regime.

Important warning:
finite neural networks can learn representations in ways the fixed NTK does not capture.
The NTK is a baseline and an analysis tool, not a complete theory of all deep learning.

### 7.4 Deep Kernel Learning

Deep kernel learning composes a neural feature extractor with a kernel.
Let

$$
g_{\boldsymbol{\theta}} : \mathcal{X} \to \mathbb{R}^m
$$

be a neural network.
Define

$$
k_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z})
=
k_0(g_{\boldsymbol{\theta}}(\mathbf{x}), g_{\boldsymbol{\theta}}(\mathbf{z})).
$$

If $k_0$ is a valid kernel, then $k_{\boldsymbol{\theta}}$ is a valid kernel for fixed $\boldsymbol{\theta}$.

Deep kernel learning separates two roles:

- the neural network learns a representation;
- the kernel provides smoothness, uncertainty, or nonparametric structure in representation space.

This hybrid is useful when:

- raw inputs need learned features;
- uncertainty estimates are still desired;
- data is too structured for a hand-designed kernel;
- the final task benefits from GP-style posterior behavior.

The same idea appears in practical systems that train embeddings and then run kernel-like methods or GP heads on top.

### 7.5 Uncertainty and Small-Data Learning

Deep networks often provide strong predictions but weak calibrated uncertainty without extra machinery.
Gaussian processes and Bayesian kernel methods provide uncertainty by construction.

In small-data settings, kernel priors are valuable because assumptions matter more than scale.
A well-chosen kernel can encode:

- smoothness in physical space;
- periodicity in time;
- conservation-inspired structure;
- similarity between molecules;
- correlation across tasks;
- graph structure;
- known invariances.

This is why kernels remain common in:

- Bayesian optimization;
- active learning;
- scientific machine learning;
- spatial statistics;
- molecular property prediction;
- calibration and uncertainty modeling;
- low-data regression.

Kernels are less dominant when raw representation learning is the bottleneck.
But when the representation is known or learned elsewhere, kernel methods can be excellent final-layer learners.

---

## 8. Common Mistakes

| # | Mistake | Why it is wrong | Fix |
| --- | --- | --- | --- |
| 1 | Treating any similarity as a kernel. | RKHS theory requires positive semidefinite Gram matrices, not just intuitive similarity. | Check PSD conditions or use known valid kernel constructions. |
| 2 | Thinking pairwise positive values imply PSD. | A matrix can have all positive entries and still have a negative eigenvalue. | Test $\mathbf{c}^\top K\mathbf{c}$ or eigenvalues for finite examples. |
| 3 | Forgetting feature scaling before RBF kernels. | Distance-based kernels are dominated by large-scale coordinates. | Standardize inputs or use anisotropic lengthscales. |
| 4 | Choosing RBF bandwidth blindly. | Too small gives memorization; too large gives nearly constant similarity. | Tune bandwidth with regularization. |
| 5 | Forgetting to center $K$ for kernel PCA. | PCA requires centered features; uncentered kernels distort components. | Use $K_c = HKH$. |
| 6 | Computing explicit inverses. | Explicit inverse is slower and less stable than solving a linear system. | Use Cholesky or linear solves for $K+\lambda I_n$. |
| 7 | Assuming kernel methods avoid overfitting. | Rich kernels can interpolate noise. | Use validation, regularization, and bandwidth control. |
| 8 | Confusing SVM sparsity with KRR density. | SVM predictions use support vectors; KRR usually uses all training points. | Choose algorithm based on prediction cost and objective. |
| 9 | Treating GP variance as generic model confidence. | GP uncertainty is conditional on the kernel, likelihood, and training assumptions. | Interpret uncertainty relative to the model assumptions. |
| 10 | Ignoring $O(n^2)$ memory. | Exact kernels become impossible at large $n$. | Use Nystrom, random features, inducing points, or linear models. |
| 11 | Reteaching PCA inside kernel PCA. | PCA has its own canonical section. | Use kernel PCA as an application and point to the PCA section. |
| 12 | Thinking NTK explains all deep learning. | NTK mainly describes lazy or infinite-width regimes. | Use NTK as a baseline, not a complete representation-learning theory. |

---

## 9. Exercises

1. **Exercise 1 (Easy): Build and Inspect Gram Matrices**
   Given a small dataset in $\mathbb{R}^2$, compute linear, polynomial, and RBF Gram matrices.
   Verify symmetry.
   Compute eigenvalues.
   Explain which matrices are PSD and why.

2. **Exercise 2 (Easy): Explicit Polynomial Feature Map**
   For $\mathbf{x} = (x_1, x_2)^\top$, construct a feature map whose inner product equals $(\mathbf{x}^\top \mathbf{z} + 1)^2$.
   Verify the identity numerically for random points.

3. **Exercise 3 (Easy): Kernel Normalization and Centering**
   Normalize an RBF Gram matrix.
   Center it using $H = I_n - \mathbf{1}\mathbf{1}^\top/n$.
   Verify that the centered matrix has approximately zero row and column sums.

4. **Exercise 4 (Medium): Kernel Ridge Regression**
   Implement kernel ridge regression from scratch.
   Fit noisy samples of a nonlinear one-dimensional function.
   Vary $\lambda$ and the RBF bandwidth.
   Explain underfitting, good fit, and overfitting.

5. **Exercise 5 (Medium): Kernel PCA**
   Generate a two-dimensional nonlinear dataset such as concentric circles.
   Compute an RBF kernel.
   Center the kernel.
   Extract the first two kernel principal components.
   Compare with linear PCA intuition.

6. **Exercise 6 (Medium): Gaussian Process Posterior**
   Implement the GP posterior mean and variance for one-dimensional regression.
   Use an RBF kernel.
   Plot uncertainty away from observed points.
   Explain the role of noise variance.

7. **Exercise 7 (Hard): Random Fourier Features**
   Approximate the RBF kernel with random Fourier features.
   Measure approximation error as the number of random features grows.
   Train a linear ridge model on random features and compare it with exact KRR.

8. **Exercise 8 (Hard): Neural Tangent Kernel**
   Define a small two-layer neural network.
   Compute the empirical NTK for a finite dataset using parameter gradients or finite differences.
   Verify that the NTK Gram matrix is PSD up to numerical tolerance.
   Interpret what large NTK entries mean.

---

## 10. Why This Matters for AI

| Kernel concept | AI impact |
| --- | --- |
| Feature map | Explains how nonlinear representation can make linear learning powerful. |
| Gram matrix | Central object for pairwise similarity, spectral methods, and kernel algorithms. |
| PSD condition | Separates valid Hilbert-space kernels from arbitrary similarity scores. |
| RKHS | Provides the function-space setting for regularized learning. |
| Reproducing property | Turns function evaluation into inner-product geometry. |
| Representer theorem | Explains why infinite-dimensional optimization reduces to finite coefficients. |
| RBF bandwidth | Controls locality, smoothness, memorization, and generalization. |
| Kernel ridge regression | Gives closed-form nonlinear regression with regularization. |
| SVM dual | Shows how margins can be learned through kernel evaluations. |
| Kernel PCA | Extends linear spectral methods to nonlinear feature spaces. |
| Gaussian process kernels | Encode prior covariance and provide uncertainty estimates. |
| Nystrom approximation | Makes large kernel matrices manageable through landmarks. |
| Random Fourier features | Converts kernel learning into scalable linear learning. |
| Neural tangent kernel | Gives a mathematical baseline for wide-network training dynamics. |
| Deep kernel learning | Combines learned representations with kernel uncertainty or smoothness. |

The durable lesson is:
kernels turn similarity assumptions into learnable geometry.

In deep learning, the representation is often learned.
In kernel methods, the representation geometry is often specified.
Hybrid methods do both.

For LLMs and large AI systems, kernels are most visible in theory, retrieval similarity, approximation methods, uncertainty heads, and small-data or scientific layers around larger learned systems.
The concepts remain load-bearing even when the production model is not a classical SVM.

---

## 11. Conceptual Bridge

This section completes the functional-analysis arc from normed spaces to Hilbert spaces to kernel methods.
Normed spaces introduced size and convergence.
Hilbert spaces added inner products, orthogonality, and projection.
Kernel methods use Hilbert-space geometry without explicitly constructing the Hilbert-space coordinates.

Backward connection:
from [Normed Spaces](../01-Normed-Spaces/notes.md), we use norms and regularization.
From [Hilbert Spaces](../02-Hilbert-Spaces/notes.md), we use inner products and RKHS language.
From [Positive Definite Matrices](../../03-Advanced-Linear-Algebra/07-Positive-Definite-Matrices/notes.md), we use PSD Gram matrices.
From [Constrained Optimization](../../08-Optimization/04-Constrained-Optimization/notes.md), we use duality in SVMs.

Forward connection:
kernel methods prepare you for ML-specific losses, activation functions, normalization, and sampling.
They also prepare you to read Gaussian-process models, spectral generalization analyses, kernelized attention approximations, and NTK papers.

Curriculum position:

```text
Functional Analysis

01 Normed Spaces
    |
    v
02 Hilbert Spaces
    |
    v
03 Kernel Methods
    |
    v
ML-Specific Math
    |
    +-- loss functions
    +-- activations
    +-- normalization
    +-- sampling
```

The conceptual shift is important:
we began with spaces of vectors.
We moved to spaces of functions.
Kernels let us compute with those function spaces through finite matrices.

That is the deepest practical idea in this section.
Infinite-dimensional thinking can produce finite algorithms.

---

## 12. References

- Aronszajn, N. (1950). Theory of Reproducing Kernels. *Transactions of the American Mathematical Society*.
- Cortes, C. and Vapnik, V. (1995). Support-Vector Networks. *Machine Learning*.
- Scholkopf, B., Herbrich, R., and Smola, A. J. (2001). A Generalized Representer Theorem.
- Scholkopf, B. and Smola, A. J. (2002). *Learning with Kernels*. MIT Press.
- Rasmussen, C. E. and Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*.
- Rahimi, A. and Recht, B. (2007). Random Features for Large-Scale Kernel Machines.
- Jacot, A., Gabriel, F., and Hongler, C. (2018). Neural Tangent Kernel: Convergence and Generalization in Neural Networks.
- Stanford CS229 notes on kernels and support vector machines.
- Cornell CS4780 lecture notes on kernel methods.
- scikit-learn documentation on kernel approximation and kernel methods.
