[Back to Curriculum](../../README.md) | [Previous: Geodesics](../03-Geodesics/notes.md) | [Next: Curriculum Home](../../README.md)

---

# Optimization on Manifolds

> _"Manifold optimization moves in tangent spaces and returns to the curved constraint without pretending the constraint is flat."_

## Overview

Optimization on manifolds turns curved constraints into geometry-aware algorithms for spheres, Stiefel and Grassmann manifolds, SPD matrices, low-rank models, and natural gradients.

Differential geometry is the part of mathematics that makes calculus work on curved spaces. Earlier chapters treated vectors, matrices, probabilities, optimization, and measures mostly in flat coordinate systems. This chapter explains what changes when the object being modeled is a sphere, a space of subspaces, a positive-definite covariance matrix, a statistical model, or a learned latent manifold.

This section uses LaTeX Markdown throughout. Inline mathematics uses `$...$`, and display mathematics uses `$$...$$`. The AI focus is practical: local coordinates, tangent spaces, metrics, geodesics, retractions, natural gradients, and matrix-manifold constraints.

## Prerequisites

- [Geodesics](../03-Geodesics/notes.md)
- [Constrained Optimization](../../08-Optimization/04-Constrained-Optimization/notes.md)
- [Numerical Optimization](../../10-Numerical-Methods/03-Numerical-Optimization/notes.md)
- [PCA](../../03-Advanced-Linear-Algebra/03-Principal-Component-Analysis/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Executable demonstrations for optimization on manifolds |
| [exercises.ipynb](exercises.ipynb) | Graded practice for optimization on manifolds |

## Learning Objectives

After completing this section, you will be able to:

- Explain manifold optimization as unconstrained optimization on a curved feasible set
- Write the Riemannian gradient descent update
- Define retractions and explain why they replace exact exponential maps
- Project Euclidean gradients onto tangent spaces for embedded manifolds
- Run sphere optimization with a retraction
- Describe Stiefel, Grassmann, and SPD manifold examples
- Explain first-order optimality on manifolds
- Preview vector transport, Hessians, line search, and trust regions
- Connect PCA and low-rank learning to Grassmann optimization
- Relate Fisher natural gradient to Riemannian steepest descent

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 Constrained optimization as unconstrained manifold optimization](#11-constrained-optimization-as-unconstrained-manifold-optimization)
  - [1.2 Why projection alone is not always geometry-aware](#12-why-projection-alone-is-not-always-geometryaware)
  - [1.3 Tangent-space updates and retractions](#13-tangentspace-updates-and-retractions)
  - [1.4 Matrix manifolds in ML](#14-matrix-manifolds-in-ml)
  - [1.5 Optimization pipeline: gradient step retract transport](#15-optimization-pipeline-gradient-step-retract-transport)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Riemannian gradient descent update](#21-riemannian-gradient-descent-update)
  - [2.2 Retraction $R_p(\mathbf{v})$](#22-retraction)
  - [2.3 Vector transport preview](#23-vector-transport-preview)
  - [2.4 Riemannian Hessian preview](#24-riemannian-hessian-preview)
  - [2.5 First-order optimality on manifolds](#25-firstorder-optimality-on-manifolds)
- [3. Core Theory](#3-core-theory)
  - [3.1 Sphere optimization](#31-sphere-optimization)
  - [3.2 Stiefel manifold and orthogonality constraints](#32-stiefel-manifold-and-orthogonality-constraints)
  - [3.3 Grassmann manifold and subspace learning](#33-grassmann-manifold-and-subspace-learning)
  - [3.4 SPD manifold and covariance learning](#34-spd-manifold-and-covariance-learning)
  - [3.5 Riemannian line search and trust-region preview](#35-riemannian-line-search-and-trustregion-preview)
- [4. AI Applications](#4-ai-applications)
  - [4.1 Orthogonal weight constraints](#41-orthogonal-weight-constraints)
  - [4.2 Low-rank matrix factorization](#42-lowrank-matrix-factorization)
  - [4.3 PCA as Grassmann optimization](#43-pca-as-grassmann-optimization)
  - [4.4 Natural-gradient learning and Fisher geometry](#44-naturalgradient-learning-and-fisher-geometry)
  - [4.5 Hyperbolic and SPD representation learning](#45-hyperbolic-and-spd-representation-learning)
- [5. Common Mistakes](#5-common-mistakes)
- [6. Exercises](#6-exercises)
- [7. Why This Matters for AI](#7-why-this-matters-for-ai)
- [8. Conceptual Bridge](#8-conceptual-bridge)
- [References](#references)

---

## 1. Intuition

Intuition develops the part of optimization on manifolds specified by the approved Chapter 25 table of contents. The treatment is geometry-first and AI-facing.

### 1.1 Constrained optimization as unconstrained manifold optimization

Constrained optimization as unconstrained manifold optimization belongs to the canonical scope of Optimization on Manifolds. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: Riemannian gradient descent, retractions, vector transport, Riemannian Hessian preview, first-order optimality, matrix manifolds, and ML optimization examples. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$\mathbf{x}_{k+1}=R_{\mathbf{x}_k}\left(-\eta_k\operatorname{grad} f(\mathbf{x}_k)\right).$$

**Operational definition.**

Manifold optimization updates in a tangent space and maps the step back to the manifold with an exponential map or retraction.

**Worked reading.**

On the sphere, take a tangent gradient step and normalize. Normalization is a simple retraction because it returns to the sphere and agrees with the tangent direction to first order.

| Geometric object | Meaning | AI interpretation |
| --- | --- | --- |
| Manifold $M$ | Curved space with local coordinates | Data manifold, latent space, constraint set, parameter space |
| Chart $\varphi$ | Local coordinate map | Local representation or embedding coordinates |
| Tangent space $T_pM$ | Linearized directions at $p$ | Local perturbations, gradients, velocities |
| Metric $g_p$ | Inner product on $T_pM$ | Geometry-aware length, angle, steepest descent |
| Geodesic | Straightest curved-space path | Latent interpolation, shortest motion, curved optimization path |
| Retraction | Practical map from tangent step back to $M$ | Efficient constrained update in training loops |

Three examples of constrained optimization as unconstrained manifold optimization:

1. PCA on Grassmann manifolds.
2. Orthogonal weights on Stiefel manifolds.
3. Covariance learning on SPD manifolds.

Two non-examples clarify the boundary:

1. Euclidean gradient descent followed by arbitrary clipping.
2. A projection step that destroys the first-order update direction.

Proof or verification habit for constrained optimization as unconstrained manifold optimization:

Check tangent feasibility, descent direction under the metric, and retraction properties.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, constrained optimization as unconstrained manifold optimization matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

This turns constraints such as orthogonality, low rank, and positive definiteness into native geometry instead of penalties.

**Mini derivation lens.**

1. Choose a point $p$ on the manifold $M$ and name the local representation used near $p$.
2. Move the question into a chart, tangent space, or embedded constraint where first-order calculus is available.
3. Compute the local object: derivative, tangent projection, metric-weighted gradient, path velocity, or retraction step.
4. Translate the result back into coordinate-free language so the answer is not tied to one chart by accident.
5. Check the invariant: the point remains on $M$, the direction remains in $T_pM$, or the distance/gradient uses the stated metric.

**Implementation lens.**

A practical ML implementation should store both the ambient array representation and the geometric contract attached to it. For example, a normalized embedding is not just a vector; it is a point on a sphere. An orthogonal weight matrix is not just a matrix; it is a point on a Stiefel-type constraint. A covariance matrix is not just a symmetric array; it must stay positive definite.

The clean computational pattern is: encode the state, compute an ambient derivative if needed, convert it into a tangent or metric-aware object, take a small local step, and then return to the manifold with a geodesic formula or retraction. This is the same pattern used in the companion notebooks, just scaled down to visible two- and three-dimensional examples.

The important warning is that coordinate code can pass shape checks while still violating geometry. Differential geometry adds checks that are semantic: tangentness, smooth compatibility, metric choice, path validity, and constraint preservation.

Practical checklist:

- State the manifold and whether it is abstract, embedded, or quotient-like.
- State the local coordinates or tangent representation being used.
- Separate ambient vectors from tangent vectors.
- Name the metric before computing distances, angles, or gradients.
- Use geodesics or retractions when moving on the manifold.
- For ML claims, identify whether geometry is data geometry, parameter geometry, or statistical geometry.

Local diagnostic: Write gradient, tangent projection, retraction, and stopping criterion.

The companion notebook uses low-dimensional synthetic examples: circles, spheres, tangent projections, spherical interpolation, SPD matrices, and orthogonality constraints. These examples keep geometry visible while preserving the same update logic used in higher-dimensional ML systems.

| Compact ML phrase | Differential-geometric reading |
| --- | --- |
| local linearization | tangent-space approximation at a point |
| normalized embedding | point on a sphere with tangent constraints |
| natural gradient | Riemannian gradient under Fisher metric |
| orthogonal weights | point on a Stiefel-type manifold |
| latent interpolation | path that may need geodesic structure |
| covariance geometry | SPD manifold rather than arbitrary matrices |

A useful learning move is to compute everything first on a sphere. The sphere has visible curvature, simple tangent spaces, closed-form geodesics, and practical retractions. Once those are clear, Stiefel, Grassmann, SPD, and information-geometric examples become less mysterious.

For implementation, the main discipline is to avoid leaving the manifold silently. If a gradient step violates a constraint, either project the gradient into the tangent space before stepping or use a method whose update is intrinsic by design.

The final question for this subsection is whether a Euclidean formula is being used as an approximation, a coordinate expression, or a mistaken replacement for geometry. Differential geometry is the habit of telling those cases apart.

### 1.2 Why projection alone is not always geometry-aware

Why projection alone is not always geometry-aware belongs to the canonical scope of Optimization on Manifolds. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: Riemannian gradient descent, retractions, vector transport, Riemannian Hessian preview, first-order optimality, matrix manifolds, and ML optimization examples. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$R_p(\mathbf{0}_p)=p,\qquad dR_p(\mathbf{0}_p)=\operatorname{id}_{T_pM}.$$

**Operational definition.**

Why projection alone is not always geometry-aware belongs to the canonical scope of Optimization on Manifolds: Riemannian gradient descent, retractions, vector transport, Riemannian Hessian preview, first-order optimality, matrix manifolds, and ML optimization examples.

**Worked reading.**

Start from a concrete embedded example, compute the local tangent or metric object, then translate back to intrinsic notation.

| Geometric object | Meaning | AI interpretation |
| --- | --- | --- |
| Manifold $M$ | Curved space with local coordinates | Data manifold, latent space, constraint set, parameter space |
| Chart $\varphi$ | Local coordinate map | Local representation or embedding coordinates |
| Tangent space $T_pM$ | Linearized directions at $p$ | Local perturbations, gradients, velocities |
| Metric $g_p$ | Inner product on $T_pM$ | Geometry-aware length, angle, steepest descent |
| Geodesic | Straightest curved-space path | Latent interpolation, shortest motion, curved optimization path |
| Retraction | Practical map from tangent step back to $M$ | Efficient constrained update in training loops |

Three examples of why projection alone is not always geometry-aware:

1. Sphere geometry.
2. Embedding-space local coordinates.
3. Matrix-manifold parameter constraints.

Two non-examples clarify the boundary:

1. A flat Euclidean approximation used globally.
2. A geometric claim made without metric or tangent space.

Proof or verification habit for why projection alone is not always geometry-aware:

The proof habit is to compute locally and verify coordinate-independent meaning.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, why projection alone is not always geometry-aware matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

The AI relevance is that model spaces are often curved even when implemented as arrays.

**Mini derivation lens.**

1. Choose a point $p$ on the manifold $M$ and name the local representation used near $p$.
2. Move the question into a chart, tangent space, or embedded constraint where first-order calculus is available.
3. Compute the local object: derivative, tangent projection, metric-weighted gradient, path velocity, or retraction step.
4. Translate the result back into coordinate-free language so the answer is not tied to one chart by accident.
5. Check the invariant: the point remains on $M$, the direction remains in $T_pM$, or the distance/gradient uses the stated metric.

**Implementation lens.**

A practical ML implementation should store both the ambient array representation and the geometric contract attached to it. For example, a normalized embedding is not just a vector; it is a point on a sphere. An orthogonal weight matrix is not just a matrix; it is a point on a Stiefel-type constraint. A covariance matrix is not just a symmetric array; it must stay positive definite.

The clean computational pattern is: encode the state, compute an ambient derivative if needed, convert it into a tangent or metric-aware object, take a small local step, and then return to the manifold with a geodesic formula or retraction. This is the same pattern used in the companion notebooks, just scaled down to visible two- and three-dimensional examples.

The important warning is that coordinate code can pass shape checks while still violating geometry. Differential geometry adds checks that are semantic: tangentness, smooth compatibility, metric choice, path validity, and constraint preservation.

Practical checklist:

- State the manifold and whether it is abstract, embedded, or quotient-like.
- State the local coordinates or tangent representation being used.
- Separate ambient vectors from tangent vectors.
- Name the metric before computing distances, angles, or gradients.
- Use geodesics or retractions when moving on the manifold.
- For ML claims, identify whether geometry is data geometry, parameter geometry, or statistical geometry.

Local diagnostic: Name the manifold, tangent space, metric, and map being used.

The companion notebook uses low-dimensional synthetic examples: circles, spheres, tangent projections, spherical interpolation, SPD matrices, and orthogonality constraints. These examples keep geometry visible while preserving the same update logic used in higher-dimensional ML systems.

| Compact ML phrase | Differential-geometric reading |
| --- | --- |
| local linearization | tangent-space approximation at a point |
| normalized embedding | point on a sphere with tangent constraints |
| natural gradient | Riemannian gradient under Fisher metric |
| orthogonal weights | point on a Stiefel-type manifold |
| latent interpolation | path that may need geodesic structure |
| covariance geometry | SPD manifold rather than arbitrary matrices |

A useful learning move is to compute everything first on a sphere. The sphere has visible curvature, simple tangent spaces, closed-form geodesics, and practical retractions. Once those are clear, Stiefel, Grassmann, SPD, and information-geometric examples become less mysterious.

For implementation, the main discipline is to avoid leaving the manifold silently. If a gradient step violates a constraint, either project the gradient into the tangent space before stepping or use a method whose update is intrinsic by design.

The final question for this subsection is whether a Euclidean formula is being used as an approximation, a coordinate expression, or a mistaken replacement for geometry. Differential geometry is the habit of telling those cases apart.

### 1.3 Tangent-space updates and retractions

Tangent-space updates and retractions belongs to the canonical scope of Optimization on Manifolds. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: Riemannian gradient descent, retractions, vector transport, Riemannian Hessian preview, first-order optimality, matrix manifolds, and ML optimization examples. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$\operatorname{grad} f(p)=\Pi_{T_pM}\left(\nabla \bar{f}(p)\right)\quad\text{for embedded submanifolds with Euclidean metric}.$$

**Operational definition.**

A tangent space is the vector space of allowable first-order velocities through a point on a manifold.

**Worked reading.**

For the unit sphere, tangent vectors at $\mathbf{x}$ are exactly vectors $\mathbf{v}$ satisfying $\mathbf{x}^\top\mathbf{v}=0$.

| Geometric object | Meaning | AI interpretation |
| --- | --- | --- |
| Manifold $M$ | Curved space with local coordinates | Data manifold, latent space, constraint set, parameter space |
| Chart $\varphi$ | Local coordinate map | Local representation or embedding coordinates |
| Tangent space $T_pM$ | Linearized directions at $p$ | Local perturbations, gradients, velocities |
| Metric $g_p$ | Inner product on $T_pM$ | Geometry-aware length, angle, steepest descent |
| Geodesic | Straightest curved-space path | Latent interpolation, shortest motion, curved optimization path |
| Retraction | Practical map from tangent step back to $M$ | Efficient constrained update in training loops |

Three examples of tangent-space updates and retractions:

1. Velocity of a curve on a sphere.
2. Jacobian pushing embedding perturbations forward.
3. A vector field assigning one tangent direction per point.

Two non-examples clarify the boundary:

1. An arbitrary ambient vector not tangent to the constraint.
2. A finite difference step that leaves the manifold without retraction.

Proof or verification habit for tangent-space updates and retractions:

For embedded manifolds, differentiate the constraint; for abstract manifolds, use curves or derivations.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, tangent-space updates and retractions matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

Tangent spaces are where local sensitivity, Jacobians, and first-order optimization live.

**Mini derivation lens.**

1. Choose a point $p$ on the manifold $M$ and name the local representation used near $p$.
2. Move the question into a chart, tangent space, or embedded constraint where first-order calculus is available.
3. Compute the local object: derivative, tangent projection, metric-weighted gradient, path velocity, or retraction step.
4. Translate the result back into coordinate-free language so the answer is not tied to one chart by accident.
5. Check the invariant: the point remains on $M$, the direction remains in $T_pM$, or the distance/gradient uses the stated metric.

**Implementation lens.**

A practical ML implementation should store both the ambient array representation and the geometric contract attached to it. For example, a normalized embedding is not just a vector; it is a point on a sphere. An orthogonal weight matrix is not just a matrix; it is a point on a Stiefel-type constraint. A covariance matrix is not just a symmetric array; it must stay positive definite.

The clean computational pattern is: encode the state, compute an ambient derivative if needed, convert it into a tangent or metric-aware object, take a small local step, and then return to the manifold with a geodesic formula or retraction. This is the same pattern used in the companion notebooks, just scaled down to visible two- and three-dimensional examples.

The important warning is that coordinate code can pass shape checks while still violating geometry. Differential geometry adds checks that are semantic: tangentness, smooth compatibility, metric choice, path validity, and constraint preservation.

Practical checklist:

- State the manifold and whether it is abstract, embedded, or quotient-like.
- State the local coordinates or tangent representation being used.
- Separate ambient vectors from tangent vectors.
- Name the metric before computing distances, angles, or gradients.
- Use geodesics or retractions when moving on the manifold.
- For ML claims, identify whether geometry is data geometry, parameter geometry, or statistical geometry.

Local diagnostic: Verify the proposed direction satisfies the tangent constraint.

The companion notebook uses low-dimensional synthetic examples: circles, spheres, tangent projections, spherical interpolation, SPD matrices, and orthogonality constraints. These examples keep geometry visible while preserving the same update logic used in higher-dimensional ML systems.

| Compact ML phrase | Differential-geometric reading |
| --- | --- |
| local linearization | tangent-space approximation at a point |
| normalized embedding | point on a sphere with tangent constraints |
| natural gradient | Riemannian gradient under Fisher metric |
| orthogonal weights | point on a Stiefel-type manifold |
| latent interpolation | path that may need geodesic structure |
| covariance geometry | SPD manifold rather than arbitrary matrices |

A useful learning move is to compute everything first on a sphere. The sphere has visible curvature, simple tangent spaces, closed-form geodesics, and practical retractions. Once those are clear, Stiefel, Grassmann, SPD, and information-geometric examples become less mysterious.

For implementation, the main discipline is to avoid leaving the manifold silently. If a gradient step violates a constraint, either project the gradient into the tangent space before stepping or use a method whose update is intrinsic by design.

The final question for this subsection is whether a Euclidean formula is being used as an approximation, a coordinate expression, or a mistaken replacement for geometry. Differential geometry is the habit of telling those cases apart.

### 1.4 Matrix manifolds in ML

Matrix manifolds in ML belongs to the canonical scope of Optimization on Manifolds. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: Riemannian gradient descent, retractions, vector transport, Riemannian Hessian preview, first-order optimality, matrix manifolds, and ML optimization examples. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$\operatorname{grad} f(p)=0\quad\text{is the first-order criticality condition on }M.$$

**Operational definition.**

Matrix manifolds in ML belongs to the canonical scope of Optimization on Manifolds: Riemannian gradient descent, retractions, vector transport, Riemannian Hessian preview, first-order optimality, matrix manifolds, and ML optimization examples.

**Worked reading.**

Start from a concrete embedded example, compute the local tangent or metric object, then translate back to intrinsic notation.

| Geometric object | Meaning | AI interpretation |
| --- | --- | --- |
| Manifold $M$ | Curved space with local coordinates | Data manifold, latent space, constraint set, parameter space |
| Chart $\varphi$ | Local coordinate map | Local representation or embedding coordinates |
| Tangent space $T_pM$ | Linearized directions at $p$ | Local perturbations, gradients, velocities |
| Metric $g_p$ | Inner product on $T_pM$ | Geometry-aware length, angle, steepest descent |
| Geodesic | Straightest curved-space path | Latent interpolation, shortest motion, curved optimization path |
| Retraction | Practical map from tangent step back to $M$ | Efficient constrained update in training loops |

Three examples of matrix manifolds in ml:

1. Sphere geometry.
2. Embedding-space local coordinates.
3. Matrix-manifold parameter constraints.

Two non-examples clarify the boundary:

1. A flat Euclidean approximation used globally.
2. A geometric claim made without metric or tangent space.

Proof or verification habit for matrix manifolds in ml:

The proof habit is to compute locally and verify coordinate-independent meaning.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, matrix manifolds in ml matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

The AI relevance is that model spaces are often curved even when implemented as arrays.

**Mini derivation lens.**

1. Choose a point $p$ on the manifold $M$ and name the local representation used near $p$.
2. Move the question into a chart, tangent space, or embedded constraint where first-order calculus is available.
3. Compute the local object: derivative, tangent projection, metric-weighted gradient, path velocity, or retraction step.
4. Translate the result back into coordinate-free language so the answer is not tied to one chart by accident.
5. Check the invariant: the point remains on $M$, the direction remains in $T_pM$, or the distance/gradient uses the stated metric.

**Implementation lens.**

A practical ML implementation should store both the ambient array representation and the geometric contract attached to it. For example, a normalized embedding is not just a vector; it is a point on a sphere. An orthogonal weight matrix is not just a matrix; it is a point on a Stiefel-type constraint. A covariance matrix is not just a symmetric array; it must stay positive definite.

The clean computational pattern is: encode the state, compute an ambient derivative if needed, convert it into a tangent or metric-aware object, take a small local step, and then return to the manifold with a geodesic formula or retraction. This is the same pattern used in the companion notebooks, just scaled down to visible two- and three-dimensional examples.

The important warning is that coordinate code can pass shape checks while still violating geometry. Differential geometry adds checks that are semantic: tangentness, smooth compatibility, metric choice, path validity, and constraint preservation.

Practical checklist:

- State the manifold and whether it is abstract, embedded, or quotient-like.
- State the local coordinates or tangent representation being used.
- Separate ambient vectors from tangent vectors.
- Name the metric before computing distances, angles, or gradients.
- Use geodesics or retractions when moving on the manifold.
- For ML claims, identify whether geometry is data geometry, parameter geometry, or statistical geometry.

Local diagnostic: Name the manifold, tangent space, metric, and map being used.

The companion notebook uses low-dimensional synthetic examples: circles, spheres, tangent projections, spherical interpolation, SPD matrices, and orthogonality constraints. These examples keep geometry visible while preserving the same update logic used in higher-dimensional ML systems.

| Compact ML phrase | Differential-geometric reading |
| --- | --- |
| local linearization | tangent-space approximation at a point |
| normalized embedding | point on a sphere with tangent constraints |
| natural gradient | Riemannian gradient under Fisher metric |
| orthogonal weights | point on a Stiefel-type manifold |
| latent interpolation | path that may need geodesic structure |
| covariance geometry | SPD manifold rather than arbitrary matrices |

A useful learning move is to compute everything first on a sphere. The sphere has visible curvature, simple tangent spaces, closed-form geodesics, and practical retractions. Once those are clear, Stiefel, Grassmann, SPD, and information-geometric examples become less mysterious.

For implementation, the main discipline is to avoid leaving the manifold silently. If a gradient step violates a constraint, either project the gradient into the tangent space before stepping or use a method whose update is intrinsic by design.

The final question for this subsection is whether a Euclidean formula is being used as an approximation, a coordinate expression, or a mistaken replacement for geometry. Differential geometry is the habit of telling those cases apart.

### 1.5 Optimization pipeline: gradient step retract transport

Optimization pipeline: gradient step retract transport belongs to the canonical scope of Optimization on Manifolds. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: Riemannian gradient descent, retractions, vector transport, Riemannian Hessian preview, first-order optimality, matrix manifolds, and ML optimization examples. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$\mathbf{x}_{k+1}=R_{\mathbf{x}_k}\left(-\eta_k\operatorname{grad} f(\mathbf{x}_k)\right).$$

**Operational definition.**

The Riemannian gradient is the tangent vector whose inner product with any direction equals the directional derivative.

**Worked reading.**

In coordinates with metric matrix $G$, the Riemannian gradient is $G^{-1}
abla f$, not usually the raw Euclidean gradient.

| Geometric object | Meaning | AI interpretation |
| --- | --- | --- |
| Manifold $M$ | Curved space with local coordinates | Data manifold, latent space, constraint set, parameter space |
| Chart $\varphi$ | Local coordinate map | Local representation or embedding coordinates |
| Tangent space $T_pM$ | Linearized directions at $p$ | Local perturbations, gradients, velocities |
| Metric $g_p$ | Inner product on $T_pM$ | Geometry-aware length, angle, steepest descent |
| Geodesic | Straightest curved-space path | Latent interpolation, shortest motion, curved optimization path |
| Retraction | Practical map from tangent step back to $M$ | Efficient constrained update in training loops |

Three examples of optimization pipeline: gradient step retract transport:

1. Natural gradient using Fisher information.
2. Projected gradient on the sphere.
3. Geometry-aware update for SPD covariance matrices.

Two non-examples clarify the boundary:

1. Raw parameter gradient treated as invariant under reparameterization.
2. A direction off the tangent space called a manifold gradient.

Proof or verification habit for optimization pipeline: gradient step retract transport:

Use the defining identity $g_p(\operatorname{grad} f,\mathbf{v})=df_p[\mathbf{v}]$ for all tangent directions.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, optimization pipeline: gradient step retract transport matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

Natural gradient and second-order preconditioning are geometry choices, not only optimizer tricks.

**Mini derivation lens.**

1. Choose a point $p$ on the manifold $M$ and name the local representation used near $p$.
2. Move the question into a chart, tangent space, or embedded constraint where first-order calculus is available.
3. Compute the local object: derivative, tangent projection, metric-weighted gradient, path velocity, or retraction step.
4. Translate the result back into coordinate-free language so the answer is not tied to one chart by accident.
5. Check the invariant: the point remains on $M$, the direction remains in $T_pM$, or the distance/gradient uses the stated metric.

**Implementation lens.**

A practical ML implementation should store both the ambient array representation and the geometric contract attached to it. For example, a normalized embedding is not just a vector; it is a point on a sphere. An orthogonal weight matrix is not just a matrix; it is a point on a Stiefel-type constraint. A covariance matrix is not just a symmetric array; it must stay positive definite.

The clean computational pattern is: encode the state, compute an ambient derivative if needed, convert it into a tangent or metric-aware object, take a small local step, and then return to the manifold with a geodesic formula or retraction. This is the same pattern used in the companion notebooks, just scaled down to visible two- and three-dimensional examples.

The important warning is that coordinate code can pass shape checks while still violating geometry. Differential geometry adds checks that are semantic: tangentness, smooth compatibility, metric choice, path validity, and constraint preservation.

Practical checklist:

- State the manifold and whether it is abstract, embedded, or quotient-like.
- State the local coordinates or tangent representation being used.
- Separate ambient vectors from tangent vectors.
- Name the metric before computing distances, angles, or gradients.
- Use geodesics or retractions when moving on the manifold.
- For ML claims, identify whether geometry is data geometry, parameter geometry, or statistical geometry.

Local diagnostic: Ask which metric converts covectors into update vectors.

The companion notebook uses low-dimensional synthetic examples: circles, spheres, tangent projections, spherical interpolation, SPD matrices, and orthogonality constraints. These examples keep geometry visible while preserving the same update logic used in higher-dimensional ML systems.

| Compact ML phrase | Differential-geometric reading |
| --- | --- |
| local linearization | tangent-space approximation at a point |
| normalized embedding | point on a sphere with tangent constraints |
| natural gradient | Riemannian gradient under Fisher metric |
| orthogonal weights | point on a Stiefel-type manifold |
| latent interpolation | path that may need geodesic structure |
| covariance geometry | SPD manifold rather than arbitrary matrices |

A useful learning move is to compute everything first on a sphere. The sphere has visible curvature, simple tangent spaces, closed-form geodesics, and practical retractions. Once those are clear, Stiefel, Grassmann, SPD, and information-geometric examples become less mysterious.

For implementation, the main discipline is to avoid leaving the manifold silently. If a gradient step violates a constraint, either project the gradient into the tangent space before stepping or use a method whose update is intrinsic by design.

The final question for this subsection is whether a Euclidean formula is being used as an approximation, a coordinate expression, or a mistaken replacement for geometry. Differential geometry is the habit of telling those cases apart.

## 2. Formal Definitions

Formal Definitions develops the part of optimization on manifolds specified by the approved Chapter 25 table of contents. The treatment is geometry-first and AI-facing.

### 2.1 Riemannian gradient descent update

Riemannian gradient descent update belongs to the canonical scope of Optimization on Manifolds. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: Riemannian gradient descent, retractions, vector transport, Riemannian Hessian preview, first-order optimality, matrix manifolds, and ML optimization examples. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$R_p(\mathbf{0}_p)=p,\qquad dR_p(\mathbf{0}_p)=\operatorname{id}_{T_pM}.$$

**Operational definition.**

The Riemannian gradient is the tangent vector whose inner product with any direction equals the directional derivative.

**Worked reading.**

In coordinates with metric matrix $G$, the Riemannian gradient is $G^{-1}
abla f$, not usually the raw Euclidean gradient.

| Geometric object | Meaning | AI interpretation |
| --- | --- | --- |
| Manifold $M$ | Curved space with local coordinates | Data manifold, latent space, constraint set, parameter space |
| Chart $\varphi$ | Local coordinate map | Local representation or embedding coordinates |
| Tangent space $T_pM$ | Linearized directions at $p$ | Local perturbations, gradients, velocities |
| Metric $g_p$ | Inner product on $T_pM$ | Geometry-aware length, angle, steepest descent |
| Geodesic | Straightest curved-space path | Latent interpolation, shortest motion, curved optimization path |
| Retraction | Practical map from tangent step back to $M$ | Efficient constrained update in training loops |

Three examples of riemannian gradient descent update:

1. Natural gradient using Fisher information.
2. Projected gradient on the sphere.
3. Geometry-aware update for SPD covariance matrices.

Two non-examples clarify the boundary:

1. Raw parameter gradient treated as invariant under reparameterization.
2. A direction off the tangent space called a manifold gradient.

Proof or verification habit for riemannian gradient descent update:

Use the defining identity $g_p(\operatorname{grad} f,\mathbf{v})=df_p[\mathbf{v}]$ for all tangent directions.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, riemannian gradient descent update matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

Natural gradient and second-order preconditioning are geometry choices, not only optimizer tricks.

**Mini derivation lens.**

1. Choose a point $p$ on the manifold $M$ and name the local representation used near $p$.
2. Move the question into a chart, tangent space, or embedded constraint where first-order calculus is available.
3. Compute the local object: derivative, tangent projection, metric-weighted gradient, path velocity, or retraction step.
4. Translate the result back into coordinate-free language so the answer is not tied to one chart by accident.
5. Check the invariant: the point remains on $M$, the direction remains in $T_pM$, or the distance/gradient uses the stated metric.

**Implementation lens.**

A practical ML implementation should store both the ambient array representation and the geometric contract attached to it. For example, a normalized embedding is not just a vector; it is a point on a sphere. An orthogonal weight matrix is not just a matrix; it is a point on a Stiefel-type constraint. A covariance matrix is not just a symmetric array; it must stay positive definite.

The clean computational pattern is: encode the state, compute an ambient derivative if needed, convert it into a tangent or metric-aware object, take a small local step, and then return to the manifold with a geodesic formula or retraction. This is the same pattern used in the companion notebooks, just scaled down to visible two- and three-dimensional examples.

The important warning is that coordinate code can pass shape checks while still violating geometry. Differential geometry adds checks that are semantic: tangentness, smooth compatibility, metric choice, path validity, and constraint preservation.

Practical checklist:

- State the manifold and whether it is abstract, embedded, or quotient-like.
- State the local coordinates or tangent representation being used.
- Separate ambient vectors from tangent vectors.
- Name the metric before computing distances, angles, or gradients.
- Use geodesics or retractions when moving on the manifold.
- For ML claims, identify whether geometry is data geometry, parameter geometry, or statistical geometry.

Local diagnostic: Ask which metric converts covectors into update vectors.

The companion notebook uses low-dimensional synthetic examples: circles, spheres, tangent projections, spherical interpolation, SPD matrices, and orthogonality constraints. These examples keep geometry visible while preserving the same update logic used in higher-dimensional ML systems.

| Compact ML phrase | Differential-geometric reading |
| --- | --- |
| local linearization | tangent-space approximation at a point |
| normalized embedding | point on a sphere with tangent constraints |
| natural gradient | Riemannian gradient under Fisher metric |
| orthogonal weights | point on a Stiefel-type manifold |
| latent interpolation | path that may need geodesic structure |
| covariance geometry | SPD manifold rather than arbitrary matrices |

A useful learning move is to compute everything first on a sphere. The sphere has visible curvature, simple tangent spaces, closed-form geodesics, and practical retractions. Once those are clear, Stiefel, Grassmann, SPD, and information-geometric examples become less mysterious.

For implementation, the main discipline is to avoid leaving the manifold silently. If a gradient step violates a constraint, either project the gradient into the tangent space before stepping or use a method whose update is intrinsic by design.

The final question for this subsection is whether a Euclidean formula is being used as an approximation, a coordinate expression, or a mistaken replacement for geometry. Differential geometry is the habit of telling those cases apart.

### 2.2 Retraction $R_p(\mathbf{v})$

Retraction $R_p(\mathbf{v})$ belongs to the canonical scope of Optimization on Manifolds. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: Riemannian gradient descent, retractions, vector transport, Riemannian Hessian preview, first-order optimality, matrix manifolds, and ML optimization examples. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$\operatorname{grad} f(p)=\Pi_{T_pM}\left(\nabla \bar{f}(p)\right)\quad\text{for embedded submanifolds with Euclidean metric}.$$

**Operational definition.**

Manifold optimization updates in a tangent space and maps the step back to the manifold with an exponential map or retraction.

**Worked reading.**

On the sphere, take a tangent gradient step and normalize. Normalization is a simple retraction because it returns to the sphere and agrees with the tangent direction to first order.

| Geometric object | Meaning | AI interpretation |
| --- | --- | --- |
| Manifold $M$ | Curved space with local coordinates | Data manifold, latent space, constraint set, parameter space |
| Chart $\varphi$ | Local coordinate map | Local representation or embedding coordinates |
| Tangent space $T_pM$ | Linearized directions at $p$ | Local perturbations, gradients, velocities |
| Metric $g_p$ | Inner product on $T_pM$ | Geometry-aware length, angle, steepest descent |
| Geodesic | Straightest curved-space path | Latent interpolation, shortest motion, curved optimization path |
| Retraction | Practical map from tangent step back to $M$ | Efficient constrained update in training loops |

Three examples of retraction $r_p(\mathbf{v})$:

1. PCA on Grassmann manifolds.
2. Orthogonal weights on Stiefel manifolds.
3. Covariance learning on SPD manifolds.

Two non-examples clarify the boundary:

1. Euclidean gradient descent followed by arbitrary clipping.
2. A projection step that destroys the first-order update direction.

Proof or verification habit for retraction $r_p(\mathbf{v})$:

Check tangent feasibility, descent direction under the metric, and retraction properties.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, retraction $r_p(\mathbf{v})$ matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

This turns constraints such as orthogonality, low rank, and positive definiteness into native geometry instead of penalties.

**Mini derivation lens.**

1. Choose a point $p$ on the manifold $M$ and name the local representation used near $p$.
2. Move the question into a chart, tangent space, or embedded constraint where first-order calculus is available.
3. Compute the local object: derivative, tangent projection, metric-weighted gradient, path velocity, or retraction step.
4. Translate the result back into coordinate-free language so the answer is not tied to one chart by accident.
5. Check the invariant: the point remains on $M$, the direction remains in $T_pM$, or the distance/gradient uses the stated metric.

**Implementation lens.**

A practical ML implementation should store both the ambient array representation and the geometric contract attached to it. For example, a normalized embedding is not just a vector; it is a point on a sphere. An orthogonal weight matrix is not just a matrix; it is a point on a Stiefel-type constraint. A covariance matrix is not just a symmetric array; it must stay positive definite.

The clean computational pattern is: encode the state, compute an ambient derivative if needed, convert it into a tangent or metric-aware object, take a small local step, and then return to the manifold with a geodesic formula or retraction. This is the same pattern used in the companion notebooks, just scaled down to visible two- and three-dimensional examples.

The important warning is that coordinate code can pass shape checks while still violating geometry. Differential geometry adds checks that are semantic: tangentness, smooth compatibility, metric choice, path validity, and constraint preservation.

Practical checklist:

- State the manifold and whether it is abstract, embedded, or quotient-like.
- State the local coordinates or tangent representation being used.
- Separate ambient vectors from tangent vectors.
- Name the metric before computing distances, angles, or gradients.
- Use geodesics or retractions when moving on the manifold.
- For ML claims, identify whether geometry is data geometry, parameter geometry, or statistical geometry.

Local diagnostic: Write gradient, tangent projection, retraction, and stopping criterion.

The companion notebook uses low-dimensional synthetic examples: circles, spheres, tangent projections, spherical interpolation, SPD matrices, and orthogonality constraints. These examples keep geometry visible while preserving the same update logic used in higher-dimensional ML systems.

| Compact ML phrase | Differential-geometric reading |
| --- | --- |
| local linearization | tangent-space approximation at a point |
| normalized embedding | point on a sphere with tangent constraints |
| natural gradient | Riemannian gradient under Fisher metric |
| orthogonal weights | point on a Stiefel-type manifold |
| latent interpolation | path that may need geodesic structure |
| covariance geometry | SPD manifold rather than arbitrary matrices |

A useful learning move is to compute everything first on a sphere. The sphere has visible curvature, simple tangent spaces, closed-form geodesics, and practical retractions. Once those are clear, Stiefel, Grassmann, SPD, and information-geometric examples become less mysterious.

For implementation, the main discipline is to avoid leaving the manifold silently. If a gradient step violates a constraint, either project the gradient into the tangent space before stepping or use a method whose update is intrinsic by design.

The final question for this subsection is whether a Euclidean formula is being used as an approximation, a coordinate expression, or a mistaken replacement for geometry. Differential geometry is the habit of telling those cases apart.

### 2.3 Vector transport preview

Vector transport preview belongs to the canonical scope of Optimization on Manifolds. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: Riemannian gradient descent, retractions, vector transport, Riemannian Hessian preview, first-order optimality, matrix manifolds, and ML optimization examples. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$\operatorname{grad} f(p)=0\quad\text{is the first-order criticality condition on }M.$$

**Operational definition.**

Vector transport preview belongs to the canonical scope of Optimization on Manifolds: Riemannian gradient descent, retractions, vector transport, Riemannian Hessian preview, first-order optimality, matrix manifolds, and ML optimization examples.

**Worked reading.**

Start from a concrete embedded example, compute the local tangent or metric object, then translate back to intrinsic notation.

| Geometric object | Meaning | AI interpretation |
| --- | --- | --- |
| Manifold $M$ | Curved space with local coordinates | Data manifold, latent space, constraint set, parameter space |
| Chart $\varphi$ | Local coordinate map | Local representation or embedding coordinates |
| Tangent space $T_pM$ | Linearized directions at $p$ | Local perturbations, gradients, velocities |
| Metric $g_p$ | Inner product on $T_pM$ | Geometry-aware length, angle, steepest descent |
| Geodesic | Straightest curved-space path | Latent interpolation, shortest motion, curved optimization path |
| Retraction | Practical map from tangent step back to $M$ | Efficient constrained update in training loops |

Three examples of vector transport preview:

1. Sphere geometry.
2. Embedding-space local coordinates.
3. Matrix-manifold parameter constraints.

Two non-examples clarify the boundary:

1. A flat Euclidean approximation used globally.
2. A geometric claim made without metric or tangent space.

Proof or verification habit for vector transport preview:

The proof habit is to compute locally and verify coordinate-independent meaning.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, vector transport preview matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

The AI relevance is that model spaces are often curved even when implemented as arrays.

**Mini derivation lens.**

1. Choose a point $p$ on the manifold $M$ and name the local representation used near $p$.
2. Move the question into a chart, tangent space, or embedded constraint where first-order calculus is available.
3. Compute the local object: derivative, tangent projection, metric-weighted gradient, path velocity, or retraction step.
4. Translate the result back into coordinate-free language so the answer is not tied to one chart by accident.
5. Check the invariant: the point remains on $M$, the direction remains in $T_pM$, or the distance/gradient uses the stated metric.

**Implementation lens.**

A practical ML implementation should store both the ambient array representation and the geometric contract attached to it. For example, a normalized embedding is not just a vector; it is a point on a sphere. An orthogonal weight matrix is not just a matrix; it is a point on a Stiefel-type constraint. A covariance matrix is not just a symmetric array; it must stay positive definite.

The clean computational pattern is: encode the state, compute an ambient derivative if needed, convert it into a tangent or metric-aware object, take a small local step, and then return to the manifold with a geodesic formula or retraction. This is the same pattern used in the companion notebooks, just scaled down to visible two- and three-dimensional examples.

The important warning is that coordinate code can pass shape checks while still violating geometry. Differential geometry adds checks that are semantic: tangentness, smooth compatibility, metric choice, path validity, and constraint preservation.

Practical checklist:

- State the manifold and whether it is abstract, embedded, or quotient-like.
- State the local coordinates or tangent representation being used.
- Separate ambient vectors from tangent vectors.
- Name the metric before computing distances, angles, or gradients.
- Use geodesics or retractions when moving on the manifold.
- For ML claims, identify whether geometry is data geometry, parameter geometry, or statistical geometry.

Local diagnostic: Name the manifold, tangent space, metric, and map being used.

The companion notebook uses low-dimensional synthetic examples: circles, spheres, tangent projections, spherical interpolation, SPD matrices, and orthogonality constraints. These examples keep geometry visible while preserving the same update logic used in higher-dimensional ML systems.

| Compact ML phrase | Differential-geometric reading |
| --- | --- |
| local linearization | tangent-space approximation at a point |
| normalized embedding | point on a sphere with tangent constraints |
| natural gradient | Riemannian gradient under Fisher metric |
| orthogonal weights | point on a Stiefel-type manifold |
| latent interpolation | path that may need geodesic structure |
| covariance geometry | SPD manifold rather than arbitrary matrices |

A useful learning move is to compute everything first on a sphere. The sphere has visible curvature, simple tangent spaces, closed-form geodesics, and practical retractions. Once those are clear, Stiefel, Grassmann, SPD, and information-geometric examples become less mysterious.

For implementation, the main discipline is to avoid leaving the manifold silently. If a gradient step violates a constraint, either project the gradient into the tangent space before stepping or use a method whose update is intrinsic by design.

The final question for this subsection is whether a Euclidean formula is being used as an approximation, a coordinate expression, or a mistaken replacement for geometry. Differential geometry is the habit of telling those cases apart.

### 2.4 Riemannian Hessian preview

Riemannian Hessian preview belongs to the canonical scope of Optimization on Manifolds. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: Riemannian gradient descent, retractions, vector transport, Riemannian Hessian preview, first-order optimality, matrix manifolds, and ML optimization examples. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$\mathbf{x}_{k+1}=R_{\mathbf{x}_k}\left(-\eta_k\operatorname{grad} f(\mathbf{x}_k)\right).$$

**Operational definition.**

Riemannian Hessian preview belongs to the canonical scope of Optimization on Manifolds: Riemannian gradient descent, retractions, vector transport, Riemannian Hessian preview, first-order optimality, matrix manifolds, and ML optimization examples.

**Worked reading.**

Start from a concrete embedded example, compute the local tangent or metric object, then translate back to intrinsic notation.

| Geometric object | Meaning | AI interpretation |
| --- | --- | --- |
| Manifold $M$ | Curved space with local coordinates | Data manifold, latent space, constraint set, parameter space |
| Chart $\varphi$ | Local coordinate map | Local representation or embedding coordinates |
| Tangent space $T_pM$ | Linearized directions at $p$ | Local perturbations, gradients, velocities |
| Metric $g_p$ | Inner product on $T_pM$ | Geometry-aware length, angle, steepest descent |
| Geodesic | Straightest curved-space path | Latent interpolation, shortest motion, curved optimization path |
| Retraction | Practical map from tangent step back to $M$ | Efficient constrained update in training loops |

Three examples of riemannian hessian preview:

1. Sphere geometry.
2. Embedding-space local coordinates.
3. Matrix-manifold parameter constraints.

Two non-examples clarify the boundary:

1. A flat Euclidean approximation used globally.
2. A geometric claim made without metric or tangent space.

Proof or verification habit for riemannian hessian preview:

The proof habit is to compute locally and verify coordinate-independent meaning.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, riemannian hessian preview matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

The AI relevance is that model spaces are often curved even when implemented as arrays.

**Mini derivation lens.**

1. Choose a point $p$ on the manifold $M$ and name the local representation used near $p$.
2. Move the question into a chart, tangent space, or embedded constraint where first-order calculus is available.
3. Compute the local object: derivative, tangent projection, metric-weighted gradient, path velocity, or retraction step.
4. Translate the result back into coordinate-free language so the answer is not tied to one chart by accident.
5. Check the invariant: the point remains on $M$, the direction remains in $T_pM$, or the distance/gradient uses the stated metric.

**Implementation lens.**

A practical ML implementation should store both the ambient array representation and the geometric contract attached to it. For example, a normalized embedding is not just a vector; it is a point on a sphere. An orthogonal weight matrix is not just a matrix; it is a point on a Stiefel-type constraint. A covariance matrix is not just a symmetric array; it must stay positive definite.

The clean computational pattern is: encode the state, compute an ambient derivative if needed, convert it into a tangent or metric-aware object, take a small local step, and then return to the manifold with a geodesic formula or retraction. This is the same pattern used in the companion notebooks, just scaled down to visible two- and three-dimensional examples.

The important warning is that coordinate code can pass shape checks while still violating geometry. Differential geometry adds checks that are semantic: tangentness, smooth compatibility, metric choice, path validity, and constraint preservation.

Practical checklist:

- State the manifold and whether it is abstract, embedded, or quotient-like.
- State the local coordinates or tangent representation being used.
- Separate ambient vectors from tangent vectors.
- Name the metric before computing distances, angles, or gradients.
- Use geodesics or retractions when moving on the manifold.
- For ML claims, identify whether geometry is data geometry, parameter geometry, or statistical geometry.

Local diagnostic: Name the manifold, tangent space, metric, and map being used.

The companion notebook uses low-dimensional synthetic examples: circles, spheres, tangent projections, spherical interpolation, SPD matrices, and orthogonality constraints. These examples keep geometry visible while preserving the same update logic used in higher-dimensional ML systems.

| Compact ML phrase | Differential-geometric reading |
| --- | --- |
| local linearization | tangent-space approximation at a point |
| normalized embedding | point on a sphere with tangent constraints |
| natural gradient | Riemannian gradient under Fisher metric |
| orthogonal weights | point on a Stiefel-type manifold |
| latent interpolation | path that may need geodesic structure |
| covariance geometry | SPD manifold rather than arbitrary matrices |

A useful learning move is to compute everything first on a sphere. The sphere has visible curvature, simple tangent spaces, closed-form geodesics, and practical retractions. Once those are clear, Stiefel, Grassmann, SPD, and information-geometric examples become less mysterious.

For implementation, the main discipline is to avoid leaving the manifold silently. If a gradient step violates a constraint, either project the gradient into the tangent space before stepping or use a method whose update is intrinsic by design.

The final question for this subsection is whether a Euclidean formula is being used as an approximation, a coordinate expression, or a mistaken replacement for geometry. Differential geometry is the habit of telling those cases apart.

### 2.5 First-order optimality on manifolds

First-order optimality on manifolds belongs to the canonical scope of Optimization on Manifolds. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: Riemannian gradient descent, retractions, vector transport, Riemannian Hessian preview, first-order optimality, matrix manifolds, and ML optimization examples. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$R_p(\mathbf{0}_p)=p,\qquad dR_p(\mathbf{0}_p)=\operatorname{id}_{T_pM}.$$

**Operational definition.**

Manifold optimization updates in a tangent space and maps the step back to the manifold with an exponential map or retraction.

**Worked reading.**

On the sphere, take a tangent gradient step and normalize. Normalization is a simple retraction because it returns to the sphere and agrees with the tangent direction to first order.

| Geometric object | Meaning | AI interpretation |
| --- | --- | --- |
| Manifold $M$ | Curved space with local coordinates | Data manifold, latent space, constraint set, parameter space |
| Chart $\varphi$ | Local coordinate map | Local representation or embedding coordinates |
| Tangent space $T_pM$ | Linearized directions at $p$ | Local perturbations, gradients, velocities |
| Metric $g_p$ | Inner product on $T_pM$ | Geometry-aware length, angle, steepest descent |
| Geodesic | Straightest curved-space path | Latent interpolation, shortest motion, curved optimization path |
| Retraction | Practical map from tangent step back to $M$ | Efficient constrained update in training loops |

Three examples of first-order optimality on manifolds:

1. PCA on Grassmann manifolds.
2. Orthogonal weights on Stiefel manifolds.
3. Covariance learning on SPD manifolds.

Two non-examples clarify the boundary:

1. Euclidean gradient descent followed by arbitrary clipping.
2. A projection step that destroys the first-order update direction.

Proof or verification habit for first-order optimality on manifolds:

Check tangent feasibility, descent direction under the metric, and retraction properties.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, first-order optimality on manifolds matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

This turns constraints such as orthogonality, low rank, and positive definiteness into native geometry instead of penalties.

**Mini derivation lens.**

1. Choose a point $p$ on the manifold $M$ and name the local representation used near $p$.
2. Move the question into a chart, tangent space, or embedded constraint where first-order calculus is available.
3. Compute the local object: derivative, tangent projection, metric-weighted gradient, path velocity, or retraction step.
4. Translate the result back into coordinate-free language so the answer is not tied to one chart by accident.
5. Check the invariant: the point remains on $M$, the direction remains in $T_pM$, or the distance/gradient uses the stated metric.

**Implementation lens.**

A practical ML implementation should store both the ambient array representation and the geometric contract attached to it. For example, a normalized embedding is not just a vector; it is a point on a sphere. An orthogonal weight matrix is not just a matrix; it is a point on a Stiefel-type constraint. A covariance matrix is not just a symmetric array; it must stay positive definite.

The clean computational pattern is: encode the state, compute an ambient derivative if needed, convert it into a tangent or metric-aware object, take a small local step, and then return to the manifold with a geodesic formula or retraction. This is the same pattern used in the companion notebooks, just scaled down to visible two- and three-dimensional examples.

The important warning is that coordinate code can pass shape checks while still violating geometry. Differential geometry adds checks that are semantic: tangentness, smooth compatibility, metric choice, path validity, and constraint preservation.

Practical checklist:

- State the manifold and whether it is abstract, embedded, or quotient-like.
- State the local coordinates or tangent representation being used.
- Separate ambient vectors from tangent vectors.
- Name the metric before computing distances, angles, or gradients.
- Use geodesics or retractions when moving on the manifold.
- For ML claims, identify whether geometry is data geometry, parameter geometry, or statistical geometry.

Local diagnostic: Write gradient, tangent projection, retraction, and stopping criterion.

The companion notebook uses low-dimensional synthetic examples: circles, spheres, tangent projections, spherical interpolation, SPD matrices, and orthogonality constraints. These examples keep geometry visible while preserving the same update logic used in higher-dimensional ML systems.

| Compact ML phrase | Differential-geometric reading |
| --- | --- |
| local linearization | tangent-space approximation at a point |
| normalized embedding | point on a sphere with tangent constraints |
| natural gradient | Riemannian gradient under Fisher metric |
| orthogonal weights | point on a Stiefel-type manifold |
| latent interpolation | path that may need geodesic structure |
| covariance geometry | SPD manifold rather than arbitrary matrices |

A useful learning move is to compute everything first on a sphere. The sphere has visible curvature, simple tangent spaces, closed-form geodesics, and practical retractions. Once those are clear, Stiefel, Grassmann, SPD, and information-geometric examples become less mysterious.

For implementation, the main discipline is to avoid leaving the manifold silently. If a gradient step violates a constraint, either project the gradient into the tangent space before stepping or use a method whose update is intrinsic by design.

The final question for this subsection is whether a Euclidean formula is being used as an approximation, a coordinate expression, or a mistaken replacement for geometry. Differential geometry is the habit of telling those cases apart.

## 3. Core Theory

Core Theory develops the part of optimization on manifolds specified by the approved Chapter 25 table of contents. The treatment is geometry-first and AI-facing.

### 3.1 Sphere optimization

Sphere optimization belongs to the canonical scope of Optimization on Manifolds. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: Riemannian gradient descent, retractions, vector transport, Riemannian Hessian preview, first-order optimality, matrix manifolds, and ML optimization examples. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$\operatorname{grad} f(p)=\Pi_{T_pM}\left(\nabla \bar{f}(p)\right)\quad\text{for embedded submanifolds with Euclidean metric}.$$

**Operational definition.**

Manifold optimization updates in a tangent space and maps the step back to the manifold with an exponential map or retraction.

**Worked reading.**

On the sphere, take a tangent gradient step and normalize. Normalization is a simple retraction because it returns to the sphere and agrees with the tangent direction to first order.

| Geometric object | Meaning | AI interpretation |
| --- | --- | --- |
| Manifold $M$ | Curved space with local coordinates | Data manifold, latent space, constraint set, parameter space |
| Chart $\varphi$ | Local coordinate map | Local representation or embedding coordinates |
| Tangent space $T_pM$ | Linearized directions at $p$ | Local perturbations, gradients, velocities |
| Metric $g_p$ | Inner product on $T_pM$ | Geometry-aware length, angle, steepest descent |
| Geodesic | Straightest curved-space path | Latent interpolation, shortest motion, curved optimization path |
| Retraction | Practical map from tangent step back to $M$ | Efficient constrained update in training loops |

Three examples of sphere optimization:

1. PCA on Grassmann manifolds.
2. Orthogonal weights on Stiefel manifolds.
3. Covariance learning on SPD manifolds.

Two non-examples clarify the boundary:

1. Euclidean gradient descent followed by arbitrary clipping.
2. A projection step that destroys the first-order update direction.

Proof or verification habit for sphere optimization:

Check tangent feasibility, descent direction under the metric, and retraction properties.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, sphere optimization matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

This turns constraints such as orthogonality, low rank, and positive definiteness into native geometry instead of penalties.

**Mini derivation lens.**

1. Choose a point $p$ on the manifold $M$ and name the local representation used near $p$.
2. Move the question into a chart, tangent space, or embedded constraint where first-order calculus is available.
3. Compute the local object: derivative, tangent projection, metric-weighted gradient, path velocity, or retraction step.
4. Translate the result back into coordinate-free language so the answer is not tied to one chart by accident.
5. Check the invariant: the point remains on $M$, the direction remains in $T_pM$, or the distance/gradient uses the stated metric.

**Implementation lens.**

A practical ML implementation should store both the ambient array representation and the geometric contract attached to it. For example, a normalized embedding is not just a vector; it is a point on a sphere. An orthogonal weight matrix is not just a matrix; it is a point on a Stiefel-type constraint. A covariance matrix is not just a symmetric array; it must stay positive definite.

The clean computational pattern is: encode the state, compute an ambient derivative if needed, convert it into a tangent or metric-aware object, take a small local step, and then return to the manifold with a geodesic formula or retraction. This is the same pattern used in the companion notebooks, just scaled down to visible two- and three-dimensional examples.

The important warning is that coordinate code can pass shape checks while still violating geometry. Differential geometry adds checks that are semantic: tangentness, smooth compatibility, metric choice, path validity, and constraint preservation.

Practical checklist:

- State the manifold and whether it is abstract, embedded, or quotient-like.
- State the local coordinates or tangent representation being used.
- Separate ambient vectors from tangent vectors.
- Name the metric before computing distances, angles, or gradients.
- Use geodesics or retractions when moving on the manifold.
- For ML claims, identify whether geometry is data geometry, parameter geometry, or statistical geometry.

Local diagnostic: Write gradient, tangent projection, retraction, and stopping criterion.

The companion notebook uses low-dimensional synthetic examples: circles, spheres, tangent projections, spherical interpolation, SPD matrices, and orthogonality constraints. These examples keep geometry visible while preserving the same update logic used in higher-dimensional ML systems.

| Compact ML phrase | Differential-geometric reading |
| --- | --- |
| local linearization | tangent-space approximation at a point |
| normalized embedding | point on a sphere with tangent constraints |
| natural gradient | Riemannian gradient under Fisher metric |
| orthogonal weights | point on a Stiefel-type manifold |
| latent interpolation | path that may need geodesic structure |
| covariance geometry | SPD manifold rather than arbitrary matrices |

A useful learning move is to compute everything first on a sphere. The sphere has visible curvature, simple tangent spaces, closed-form geodesics, and practical retractions. Once those are clear, Stiefel, Grassmann, SPD, and information-geometric examples become less mysterious.

For implementation, the main discipline is to avoid leaving the manifold silently. If a gradient step violates a constraint, either project the gradient into the tangent space before stepping or use a method whose update is intrinsic by design.

The final question for this subsection is whether a Euclidean formula is being used as an approximation, a coordinate expression, or a mistaken replacement for geometry. Differential geometry is the habit of telling those cases apart.

### 3.2 Stiefel manifold and orthogonality constraints

Stiefel manifold and orthogonality constraints belongs to the canonical scope of Optimization on Manifolds. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: Riemannian gradient descent, retractions, vector transport, Riemannian Hessian preview, first-order optimality, matrix manifolds, and ML optimization examples. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$\operatorname{grad} f(p)=0\quad\text{is the first-order criticality condition on }M.$$

**Operational definition.**

Manifold optimization updates in a tangent space and maps the step back to the manifold with an exponential map or retraction.

**Worked reading.**

On the sphere, take a tangent gradient step and normalize. Normalization is a simple retraction because it returns to the sphere and agrees with the tangent direction to first order.

| Geometric object | Meaning | AI interpretation |
| --- | --- | --- |
| Manifold $M$ | Curved space with local coordinates | Data manifold, latent space, constraint set, parameter space |
| Chart $\varphi$ | Local coordinate map | Local representation or embedding coordinates |
| Tangent space $T_pM$ | Linearized directions at $p$ | Local perturbations, gradients, velocities |
| Metric $g_p$ | Inner product on $T_pM$ | Geometry-aware length, angle, steepest descent |
| Geodesic | Straightest curved-space path | Latent interpolation, shortest motion, curved optimization path |
| Retraction | Practical map from tangent step back to $M$ | Efficient constrained update in training loops |

Three examples of stiefel manifold and orthogonality constraints:

1. PCA on Grassmann manifolds.
2. Orthogonal weights on Stiefel manifolds.
3. Covariance learning on SPD manifolds.

Two non-examples clarify the boundary:

1. Euclidean gradient descent followed by arbitrary clipping.
2. A projection step that destroys the first-order update direction.

Proof or verification habit for stiefel manifold and orthogonality constraints:

Check tangent feasibility, descent direction under the metric, and retraction properties.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, stiefel manifold and orthogonality constraints matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

This turns constraints such as orthogonality, low rank, and positive definiteness into native geometry instead of penalties.

**Mini derivation lens.**

1. Choose a point $p$ on the manifold $M$ and name the local representation used near $p$.
2. Move the question into a chart, tangent space, or embedded constraint where first-order calculus is available.
3. Compute the local object: derivative, tangent projection, metric-weighted gradient, path velocity, or retraction step.
4. Translate the result back into coordinate-free language so the answer is not tied to one chart by accident.
5. Check the invariant: the point remains on $M$, the direction remains in $T_pM$, or the distance/gradient uses the stated metric.

**Implementation lens.**

A practical ML implementation should store both the ambient array representation and the geometric contract attached to it. For example, a normalized embedding is not just a vector; it is a point on a sphere. An orthogonal weight matrix is not just a matrix; it is a point on a Stiefel-type constraint. A covariance matrix is not just a symmetric array; it must stay positive definite.

The clean computational pattern is: encode the state, compute an ambient derivative if needed, convert it into a tangent or metric-aware object, take a small local step, and then return to the manifold with a geodesic formula or retraction. This is the same pattern used in the companion notebooks, just scaled down to visible two- and three-dimensional examples.

The important warning is that coordinate code can pass shape checks while still violating geometry. Differential geometry adds checks that are semantic: tangentness, smooth compatibility, metric choice, path validity, and constraint preservation.

Practical checklist:

- State the manifold and whether it is abstract, embedded, or quotient-like.
- State the local coordinates or tangent representation being used.
- Separate ambient vectors from tangent vectors.
- Name the metric before computing distances, angles, or gradients.
- Use geodesics or retractions when moving on the manifold.
- For ML claims, identify whether geometry is data geometry, parameter geometry, or statistical geometry.

Local diagnostic: Write gradient, tangent projection, retraction, and stopping criterion.

The companion notebook uses low-dimensional synthetic examples: circles, spheres, tangent projections, spherical interpolation, SPD matrices, and orthogonality constraints. These examples keep geometry visible while preserving the same update logic used in higher-dimensional ML systems.

| Compact ML phrase | Differential-geometric reading |
| --- | --- |
| local linearization | tangent-space approximation at a point |
| normalized embedding | point on a sphere with tangent constraints |
| natural gradient | Riemannian gradient under Fisher metric |
| orthogonal weights | point on a Stiefel-type manifold |
| latent interpolation | path that may need geodesic structure |
| covariance geometry | SPD manifold rather than arbitrary matrices |

A useful learning move is to compute everything first on a sphere. The sphere has visible curvature, simple tangent spaces, closed-form geodesics, and practical retractions. Once those are clear, Stiefel, Grassmann, SPD, and information-geometric examples become less mysterious.

For implementation, the main discipline is to avoid leaving the manifold silently. If a gradient step violates a constraint, either project the gradient into the tangent space before stepping or use a method whose update is intrinsic by design.

The final question for this subsection is whether a Euclidean formula is being used as an approximation, a coordinate expression, or a mistaken replacement for geometry. Differential geometry is the habit of telling those cases apart.

### 3.3 Grassmann manifold and subspace learning

Grassmann manifold and subspace learning belongs to the canonical scope of Optimization on Manifolds. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: Riemannian gradient descent, retractions, vector transport, Riemannian Hessian preview, first-order optimality, matrix manifolds, and ML optimization examples. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$\mathbf{x}_{k+1}=R_{\mathbf{x}_k}\left(-\eta_k\operatorname{grad} f(\mathbf{x}_k)\right).$$

**Operational definition.**

Manifold optimization updates in a tangent space and maps the step back to the manifold with an exponential map or retraction.

**Worked reading.**

On the sphere, take a tangent gradient step and normalize. Normalization is a simple retraction because it returns to the sphere and agrees with the tangent direction to first order.

| Geometric object | Meaning | AI interpretation |
| --- | --- | --- |
| Manifold $M$ | Curved space with local coordinates | Data manifold, latent space, constraint set, parameter space |
| Chart $\varphi$ | Local coordinate map | Local representation or embedding coordinates |
| Tangent space $T_pM$ | Linearized directions at $p$ | Local perturbations, gradients, velocities |
| Metric $g_p$ | Inner product on $T_pM$ | Geometry-aware length, angle, steepest descent |
| Geodesic | Straightest curved-space path | Latent interpolation, shortest motion, curved optimization path |
| Retraction | Practical map from tangent step back to $M$ | Efficient constrained update in training loops |

Three examples of grassmann manifold and subspace learning:

1. PCA on Grassmann manifolds.
2. Orthogonal weights on Stiefel manifolds.
3. Covariance learning on SPD manifolds.

Two non-examples clarify the boundary:

1. Euclidean gradient descent followed by arbitrary clipping.
2. A projection step that destroys the first-order update direction.

Proof or verification habit for grassmann manifold and subspace learning:

Check tangent feasibility, descent direction under the metric, and retraction properties.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, grassmann manifold and subspace learning matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

This turns constraints such as orthogonality, low rank, and positive definiteness into native geometry instead of penalties.

**Mini derivation lens.**

1. Choose a point $p$ on the manifold $M$ and name the local representation used near $p$.
2. Move the question into a chart, tangent space, or embedded constraint where first-order calculus is available.
3. Compute the local object: derivative, tangent projection, metric-weighted gradient, path velocity, or retraction step.
4. Translate the result back into coordinate-free language so the answer is not tied to one chart by accident.
5. Check the invariant: the point remains on $M$, the direction remains in $T_pM$, or the distance/gradient uses the stated metric.

**Implementation lens.**

A practical ML implementation should store both the ambient array representation and the geometric contract attached to it. For example, a normalized embedding is not just a vector; it is a point on a sphere. An orthogonal weight matrix is not just a matrix; it is a point on a Stiefel-type constraint. A covariance matrix is not just a symmetric array; it must stay positive definite.

The clean computational pattern is: encode the state, compute an ambient derivative if needed, convert it into a tangent or metric-aware object, take a small local step, and then return to the manifold with a geodesic formula or retraction. This is the same pattern used in the companion notebooks, just scaled down to visible two- and three-dimensional examples.

The important warning is that coordinate code can pass shape checks while still violating geometry. Differential geometry adds checks that are semantic: tangentness, smooth compatibility, metric choice, path validity, and constraint preservation.

Practical checklist:

- State the manifold and whether it is abstract, embedded, or quotient-like.
- State the local coordinates or tangent representation being used.
- Separate ambient vectors from tangent vectors.
- Name the metric before computing distances, angles, or gradients.
- Use geodesics or retractions when moving on the manifold.
- For ML claims, identify whether geometry is data geometry, parameter geometry, or statistical geometry.

Local diagnostic: Write gradient, tangent projection, retraction, and stopping criterion.

The companion notebook uses low-dimensional synthetic examples: circles, spheres, tangent projections, spherical interpolation, SPD matrices, and orthogonality constraints. These examples keep geometry visible while preserving the same update logic used in higher-dimensional ML systems.

| Compact ML phrase | Differential-geometric reading |
| --- | --- |
| local linearization | tangent-space approximation at a point |
| normalized embedding | point on a sphere with tangent constraints |
| natural gradient | Riemannian gradient under Fisher metric |
| orthogonal weights | point on a Stiefel-type manifold |
| latent interpolation | path that may need geodesic structure |
| covariance geometry | SPD manifold rather than arbitrary matrices |

A useful learning move is to compute everything first on a sphere. The sphere has visible curvature, simple tangent spaces, closed-form geodesics, and practical retractions. Once those are clear, Stiefel, Grassmann, SPD, and information-geometric examples become less mysterious.

For implementation, the main discipline is to avoid leaving the manifold silently. If a gradient step violates a constraint, either project the gradient into the tangent space before stepping or use a method whose update is intrinsic by design.

The final question for this subsection is whether a Euclidean formula is being used as an approximation, a coordinate expression, or a mistaken replacement for geometry. Differential geometry is the habit of telling those cases apart.

### 3.4 SPD manifold and covariance learning

SPD manifold and covariance learning belongs to the canonical scope of Optimization on Manifolds. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: Riemannian gradient descent, retractions, vector transport, Riemannian Hessian preview, first-order optimality, matrix manifolds, and ML optimization examples. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$R_p(\mathbf{0}_p)=p,\qquad dR_p(\mathbf{0}_p)=\operatorname{id}_{T_pM}.$$

**Operational definition.**

Manifold optimization updates in a tangent space and maps the step back to the manifold with an exponential map or retraction.

**Worked reading.**

On the sphere, take a tangent gradient step and normalize. Normalization is a simple retraction because it returns to the sphere and agrees with the tangent direction to first order.

| Geometric object | Meaning | AI interpretation |
| --- | --- | --- |
| Manifold $M$ | Curved space with local coordinates | Data manifold, latent space, constraint set, parameter space |
| Chart $\varphi$ | Local coordinate map | Local representation or embedding coordinates |
| Tangent space $T_pM$ | Linearized directions at $p$ | Local perturbations, gradients, velocities |
| Metric $g_p$ | Inner product on $T_pM$ | Geometry-aware length, angle, steepest descent |
| Geodesic | Straightest curved-space path | Latent interpolation, shortest motion, curved optimization path |
| Retraction | Practical map from tangent step back to $M$ | Efficient constrained update in training loops |

Three examples of spd manifold and covariance learning:

1. PCA on Grassmann manifolds.
2. Orthogonal weights on Stiefel manifolds.
3. Covariance learning on SPD manifolds.

Two non-examples clarify the boundary:

1. Euclidean gradient descent followed by arbitrary clipping.
2. A projection step that destroys the first-order update direction.

Proof or verification habit for spd manifold and covariance learning:

Check tangent feasibility, descent direction under the metric, and retraction properties.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, spd manifold and covariance learning matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

This turns constraints such as orthogonality, low rank, and positive definiteness into native geometry instead of penalties.

**Mini derivation lens.**

1. Choose a point $p$ on the manifold $M$ and name the local representation used near $p$.
2. Move the question into a chart, tangent space, or embedded constraint where first-order calculus is available.
3. Compute the local object: derivative, tangent projection, metric-weighted gradient, path velocity, or retraction step.
4. Translate the result back into coordinate-free language so the answer is not tied to one chart by accident.
5. Check the invariant: the point remains on $M$, the direction remains in $T_pM$, or the distance/gradient uses the stated metric.

**Implementation lens.**

A practical ML implementation should store both the ambient array representation and the geometric contract attached to it. For example, a normalized embedding is not just a vector; it is a point on a sphere. An orthogonal weight matrix is not just a matrix; it is a point on a Stiefel-type constraint. A covariance matrix is not just a symmetric array; it must stay positive definite.

The clean computational pattern is: encode the state, compute an ambient derivative if needed, convert it into a tangent or metric-aware object, take a small local step, and then return to the manifold with a geodesic formula or retraction. This is the same pattern used in the companion notebooks, just scaled down to visible two- and three-dimensional examples.

The important warning is that coordinate code can pass shape checks while still violating geometry. Differential geometry adds checks that are semantic: tangentness, smooth compatibility, metric choice, path validity, and constraint preservation.

Practical checklist:

- State the manifold and whether it is abstract, embedded, or quotient-like.
- State the local coordinates or tangent representation being used.
- Separate ambient vectors from tangent vectors.
- Name the metric before computing distances, angles, or gradients.
- Use geodesics or retractions when moving on the manifold.
- For ML claims, identify whether geometry is data geometry, parameter geometry, or statistical geometry.

Local diagnostic: Write gradient, tangent projection, retraction, and stopping criterion.

The companion notebook uses low-dimensional synthetic examples: circles, spheres, tangent projections, spherical interpolation, SPD matrices, and orthogonality constraints. These examples keep geometry visible while preserving the same update logic used in higher-dimensional ML systems.

| Compact ML phrase | Differential-geometric reading |
| --- | --- |
| local linearization | tangent-space approximation at a point |
| normalized embedding | point on a sphere with tangent constraints |
| natural gradient | Riemannian gradient under Fisher metric |
| orthogonal weights | point on a Stiefel-type manifold |
| latent interpolation | path that may need geodesic structure |
| covariance geometry | SPD manifold rather than arbitrary matrices |

A useful learning move is to compute everything first on a sphere. The sphere has visible curvature, simple tangent spaces, closed-form geodesics, and practical retractions. Once those are clear, Stiefel, Grassmann, SPD, and information-geometric examples become less mysterious.

For implementation, the main discipline is to avoid leaving the manifold silently. If a gradient step violates a constraint, either project the gradient into the tangent space before stepping or use a method whose update is intrinsic by design.

The final question for this subsection is whether a Euclidean formula is being used as an approximation, a coordinate expression, or a mistaken replacement for geometry. Differential geometry is the habit of telling those cases apart.

### 3.5 Riemannian line search and trust-region preview

Riemannian line search and trust-region preview belongs to the canonical scope of Optimization on Manifolds. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: Riemannian gradient descent, retractions, vector transport, Riemannian Hessian preview, first-order optimality, matrix manifolds, and ML optimization examples. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$\operatorname{grad} f(p)=\Pi_{T_pM}\left(\nabla \bar{f}(p)\right)\quad\text{for embedded submanifolds with Euclidean metric}.$$

**Operational definition.**

Manifold optimization updates in a tangent space and maps the step back to the manifold with an exponential map or retraction.

**Worked reading.**

On the sphere, take a tangent gradient step and normalize. Normalization is a simple retraction because it returns to the sphere and agrees with the tangent direction to first order.

| Geometric object | Meaning | AI interpretation |
| --- | --- | --- |
| Manifold $M$ | Curved space with local coordinates | Data manifold, latent space, constraint set, parameter space |
| Chart $\varphi$ | Local coordinate map | Local representation or embedding coordinates |
| Tangent space $T_pM$ | Linearized directions at $p$ | Local perturbations, gradients, velocities |
| Metric $g_p$ | Inner product on $T_pM$ | Geometry-aware length, angle, steepest descent |
| Geodesic | Straightest curved-space path | Latent interpolation, shortest motion, curved optimization path |
| Retraction | Practical map from tangent step back to $M$ | Efficient constrained update in training loops |

Three examples of riemannian line search and trust-region preview:

1. PCA on Grassmann manifolds.
2. Orthogonal weights on Stiefel manifolds.
3. Covariance learning on SPD manifolds.

Two non-examples clarify the boundary:

1. Euclidean gradient descent followed by arbitrary clipping.
2. A projection step that destroys the first-order update direction.

Proof or verification habit for riemannian line search and trust-region preview:

Check tangent feasibility, descent direction under the metric, and retraction properties.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, riemannian line search and trust-region preview matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

This turns constraints such as orthogonality, low rank, and positive definiteness into native geometry instead of penalties.

**Mini derivation lens.**

1. Choose a point $p$ on the manifold $M$ and name the local representation used near $p$.
2. Move the question into a chart, tangent space, or embedded constraint where first-order calculus is available.
3. Compute the local object: derivative, tangent projection, metric-weighted gradient, path velocity, or retraction step.
4. Translate the result back into coordinate-free language so the answer is not tied to one chart by accident.
5. Check the invariant: the point remains on $M$, the direction remains in $T_pM$, or the distance/gradient uses the stated metric.

**Implementation lens.**

A practical ML implementation should store both the ambient array representation and the geometric contract attached to it. For example, a normalized embedding is not just a vector; it is a point on a sphere. An orthogonal weight matrix is not just a matrix; it is a point on a Stiefel-type constraint. A covariance matrix is not just a symmetric array; it must stay positive definite.

The clean computational pattern is: encode the state, compute an ambient derivative if needed, convert it into a tangent or metric-aware object, take a small local step, and then return to the manifold with a geodesic formula or retraction. This is the same pattern used in the companion notebooks, just scaled down to visible two- and three-dimensional examples.

The important warning is that coordinate code can pass shape checks while still violating geometry. Differential geometry adds checks that are semantic: tangentness, smooth compatibility, metric choice, path validity, and constraint preservation.

Practical checklist:

- State the manifold and whether it is abstract, embedded, or quotient-like.
- State the local coordinates or tangent representation being used.
- Separate ambient vectors from tangent vectors.
- Name the metric before computing distances, angles, or gradients.
- Use geodesics or retractions when moving on the manifold.
- For ML claims, identify whether geometry is data geometry, parameter geometry, or statistical geometry.

Local diagnostic: Write gradient, tangent projection, retraction, and stopping criterion.

The companion notebook uses low-dimensional synthetic examples: circles, spheres, tangent projections, spherical interpolation, SPD matrices, and orthogonality constraints. These examples keep geometry visible while preserving the same update logic used in higher-dimensional ML systems.

| Compact ML phrase | Differential-geometric reading |
| --- | --- |
| local linearization | tangent-space approximation at a point |
| normalized embedding | point on a sphere with tangent constraints |
| natural gradient | Riemannian gradient under Fisher metric |
| orthogonal weights | point on a Stiefel-type manifold |
| latent interpolation | path that may need geodesic structure |
| covariance geometry | SPD manifold rather than arbitrary matrices |

A useful learning move is to compute everything first on a sphere. The sphere has visible curvature, simple tangent spaces, closed-form geodesics, and practical retractions. Once those are clear, Stiefel, Grassmann, SPD, and information-geometric examples become less mysterious.

For implementation, the main discipline is to avoid leaving the manifold silently. If a gradient step violates a constraint, either project the gradient into the tangent space before stepping or use a method whose update is intrinsic by design.

The final question for this subsection is whether a Euclidean formula is being used as an approximation, a coordinate expression, or a mistaken replacement for geometry. Differential geometry is the habit of telling those cases apart.

## 4. AI Applications

AI Applications develops the part of optimization on manifolds specified by the approved Chapter 25 table of contents. The treatment is geometry-first and AI-facing.

### 4.1 Orthogonal weight constraints

Orthogonal weight constraints belongs to the canonical scope of Optimization on Manifolds. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: Riemannian gradient descent, retractions, vector transport, Riemannian Hessian preview, first-order optimality, matrix manifolds, and ML optimization examples. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$\operatorname{grad} f(p)=0\quad\text{is the first-order criticality condition on }M.$$

**Operational definition.**

Orthogonal weight constraints belongs to the canonical scope of Optimization on Manifolds: Riemannian gradient descent, retractions, vector transport, Riemannian Hessian preview, first-order optimality, matrix manifolds, and ML optimization examples.

**Worked reading.**

Start from a concrete embedded example, compute the local tangent or metric object, then translate back to intrinsic notation.

| Geometric object | Meaning | AI interpretation |
| --- | --- | --- |
| Manifold $M$ | Curved space with local coordinates | Data manifold, latent space, constraint set, parameter space |
| Chart $\varphi$ | Local coordinate map | Local representation or embedding coordinates |
| Tangent space $T_pM$ | Linearized directions at $p$ | Local perturbations, gradients, velocities |
| Metric $g_p$ | Inner product on $T_pM$ | Geometry-aware length, angle, steepest descent |
| Geodesic | Straightest curved-space path | Latent interpolation, shortest motion, curved optimization path |
| Retraction | Practical map from tangent step back to $M$ | Efficient constrained update in training loops |

Three examples of orthogonal weight constraints:

1. Sphere geometry.
2. Embedding-space local coordinates.
3. Matrix-manifold parameter constraints.

Two non-examples clarify the boundary:

1. A flat Euclidean approximation used globally.
2. A geometric claim made without metric or tangent space.

Proof or verification habit for orthogonal weight constraints:

The proof habit is to compute locally and verify coordinate-independent meaning.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, orthogonal weight constraints matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

The AI relevance is that model spaces are often curved even when implemented as arrays.

**Mini derivation lens.**

1. Choose a point $p$ on the manifold $M$ and name the local representation used near $p$.
2. Move the question into a chart, tangent space, or embedded constraint where first-order calculus is available.
3. Compute the local object: derivative, tangent projection, metric-weighted gradient, path velocity, or retraction step.
4. Translate the result back into coordinate-free language so the answer is not tied to one chart by accident.
5. Check the invariant: the point remains on $M$, the direction remains in $T_pM$, or the distance/gradient uses the stated metric.

**Implementation lens.**

A practical ML implementation should store both the ambient array representation and the geometric contract attached to it. For example, a normalized embedding is not just a vector; it is a point on a sphere. An orthogonal weight matrix is not just a matrix; it is a point on a Stiefel-type constraint. A covariance matrix is not just a symmetric array; it must stay positive definite.

The clean computational pattern is: encode the state, compute an ambient derivative if needed, convert it into a tangent or metric-aware object, take a small local step, and then return to the manifold with a geodesic formula or retraction. This is the same pattern used in the companion notebooks, just scaled down to visible two- and three-dimensional examples.

The important warning is that coordinate code can pass shape checks while still violating geometry. Differential geometry adds checks that are semantic: tangentness, smooth compatibility, metric choice, path validity, and constraint preservation.

Practical checklist:

- State the manifold and whether it is abstract, embedded, or quotient-like.
- State the local coordinates or tangent representation being used.
- Separate ambient vectors from tangent vectors.
- Name the metric before computing distances, angles, or gradients.
- Use geodesics or retractions when moving on the manifold.
- For ML claims, identify whether geometry is data geometry, parameter geometry, or statistical geometry.

Local diagnostic: Name the manifold, tangent space, metric, and map being used.

The companion notebook uses low-dimensional synthetic examples: circles, spheres, tangent projections, spherical interpolation, SPD matrices, and orthogonality constraints. These examples keep geometry visible while preserving the same update logic used in higher-dimensional ML systems.

| Compact ML phrase | Differential-geometric reading |
| --- | --- |
| local linearization | tangent-space approximation at a point |
| normalized embedding | point on a sphere with tangent constraints |
| natural gradient | Riemannian gradient under Fisher metric |
| orthogonal weights | point on a Stiefel-type manifold |
| latent interpolation | path that may need geodesic structure |
| covariance geometry | SPD manifold rather than arbitrary matrices |

A useful learning move is to compute everything first on a sphere. The sphere has visible curvature, simple tangent spaces, closed-form geodesics, and practical retractions. Once those are clear, Stiefel, Grassmann, SPD, and information-geometric examples become less mysterious.

For implementation, the main discipline is to avoid leaving the manifold silently. If a gradient step violates a constraint, either project the gradient into the tangent space before stepping or use a method whose update is intrinsic by design.

The final question for this subsection is whether a Euclidean formula is being used as an approximation, a coordinate expression, or a mistaken replacement for geometry. Differential geometry is the habit of telling those cases apart.

### 4.2 Low-rank matrix factorization

Low-rank matrix factorization belongs to the canonical scope of Optimization on Manifolds. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: Riemannian gradient descent, retractions, vector transport, Riemannian Hessian preview, first-order optimality, matrix manifolds, and ML optimization examples. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$\mathbf{x}_{k+1}=R_{\mathbf{x}_k}\left(-\eta_k\operatorname{grad} f(\mathbf{x}_k)\right).$$

**Operational definition.**

Low-rank matrix factorization belongs to the canonical scope of Optimization on Manifolds: Riemannian gradient descent, retractions, vector transport, Riemannian Hessian preview, first-order optimality, matrix manifolds, and ML optimization examples.

**Worked reading.**

Start from a concrete embedded example, compute the local tangent or metric object, then translate back to intrinsic notation.

| Geometric object | Meaning | AI interpretation |
| --- | --- | --- |
| Manifold $M$ | Curved space with local coordinates | Data manifold, latent space, constraint set, parameter space |
| Chart $\varphi$ | Local coordinate map | Local representation or embedding coordinates |
| Tangent space $T_pM$ | Linearized directions at $p$ | Local perturbations, gradients, velocities |
| Metric $g_p$ | Inner product on $T_pM$ | Geometry-aware length, angle, steepest descent |
| Geodesic | Straightest curved-space path | Latent interpolation, shortest motion, curved optimization path |
| Retraction | Practical map from tangent step back to $M$ | Efficient constrained update in training loops |

Three examples of low-rank matrix factorization:

1. Sphere geometry.
2. Embedding-space local coordinates.
3. Matrix-manifold parameter constraints.

Two non-examples clarify the boundary:

1. A flat Euclidean approximation used globally.
2. A geometric claim made without metric or tangent space.

Proof or verification habit for low-rank matrix factorization:

The proof habit is to compute locally and verify coordinate-independent meaning.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, low-rank matrix factorization matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

The AI relevance is that model spaces are often curved even when implemented as arrays.

**Mini derivation lens.**

1. Choose a point $p$ on the manifold $M$ and name the local representation used near $p$.
2. Move the question into a chart, tangent space, or embedded constraint where first-order calculus is available.
3. Compute the local object: derivative, tangent projection, metric-weighted gradient, path velocity, or retraction step.
4. Translate the result back into coordinate-free language so the answer is not tied to one chart by accident.
5. Check the invariant: the point remains on $M$, the direction remains in $T_pM$, or the distance/gradient uses the stated metric.

**Implementation lens.**

A practical ML implementation should store both the ambient array representation and the geometric contract attached to it. For example, a normalized embedding is not just a vector; it is a point on a sphere. An orthogonal weight matrix is not just a matrix; it is a point on a Stiefel-type constraint. A covariance matrix is not just a symmetric array; it must stay positive definite.

The clean computational pattern is: encode the state, compute an ambient derivative if needed, convert it into a tangent or metric-aware object, take a small local step, and then return to the manifold with a geodesic formula or retraction. This is the same pattern used in the companion notebooks, just scaled down to visible two- and three-dimensional examples.

The important warning is that coordinate code can pass shape checks while still violating geometry. Differential geometry adds checks that are semantic: tangentness, smooth compatibility, metric choice, path validity, and constraint preservation.

Practical checklist:

- State the manifold and whether it is abstract, embedded, or quotient-like.
- State the local coordinates or tangent representation being used.
- Separate ambient vectors from tangent vectors.
- Name the metric before computing distances, angles, or gradients.
- Use geodesics or retractions when moving on the manifold.
- For ML claims, identify whether geometry is data geometry, parameter geometry, or statistical geometry.

Local diagnostic: Name the manifold, tangent space, metric, and map being used.

The companion notebook uses low-dimensional synthetic examples: circles, spheres, tangent projections, spherical interpolation, SPD matrices, and orthogonality constraints. These examples keep geometry visible while preserving the same update logic used in higher-dimensional ML systems.

| Compact ML phrase | Differential-geometric reading |
| --- | --- |
| local linearization | tangent-space approximation at a point |
| normalized embedding | point on a sphere with tangent constraints |
| natural gradient | Riemannian gradient under Fisher metric |
| orthogonal weights | point on a Stiefel-type manifold |
| latent interpolation | path that may need geodesic structure |
| covariance geometry | SPD manifold rather than arbitrary matrices |

A useful learning move is to compute everything first on a sphere. The sphere has visible curvature, simple tangent spaces, closed-form geodesics, and practical retractions. Once those are clear, Stiefel, Grassmann, SPD, and information-geometric examples become less mysterious.

For implementation, the main discipline is to avoid leaving the manifold silently. If a gradient step violates a constraint, either project the gradient into the tangent space before stepping or use a method whose update is intrinsic by design.

The final question for this subsection is whether a Euclidean formula is being used as an approximation, a coordinate expression, or a mistaken replacement for geometry. Differential geometry is the habit of telling those cases apart.

### 4.3 PCA as Grassmann optimization

PCA as Grassmann optimization belongs to the canonical scope of Optimization on Manifolds. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: Riemannian gradient descent, retractions, vector transport, Riemannian Hessian preview, first-order optimality, matrix manifolds, and ML optimization examples. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$R_p(\mathbf{0}_p)=p,\qquad dR_p(\mathbf{0}_p)=\operatorname{id}_{T_pM}.$$

**Operational definition.**

Manifold optimization updates in a tangent space and maps the step back to the manifold with an exponential map or retraction.

**Worked reading.**

On the sphere, take a tangent gradient step and normalize. Normalization is a simple retraction because it returns to the sphere and agrees with the tangent direction to first order.

| Geometric object | Meaning | AI interpretation |
| --- | --- | --- |
| Manifold $M$ | Curved space with local coordinates | Data manifold, latent space, constraint set, parameter space |
| Chart $\varphi$ | Local coordinate map | Local representation or embedding coordinates |
| Tangent space $T_pM$ | Linearized directions at $p$ | Local perturbations, gradients, velocities |
| Metric $g_p$ | Inner product on $T_pM$ | Geometry-aware length, angle, steepest descent |
| Geodesic | Straightest curved-space path | Latent interpolation, shortest motion, curved optimization path |
| Retraction | Practical map from tangent step back to $M$ | Efficient constrained update in training loops |

Three examples of pca as grassmann optimization:

1. PCA on Grassmann manifolds.
2. Orthogonal weights on Stiefel manifolds.
3. Covariance learning on SPD manifolds.

Two non-examples clarify the boundary:

1. Euclidean gradient descent followed by arbitrary clipping.
2. A projection step that destroys the first-order update direction.

Proof or verification habit for pca as grassmann optimization:

Check tangent feasibility, descent direction under the metric, and retraction properties.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, pca as grassmann optimization matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

This turns constraints such as orthogonality, low rank, and positive definiteness into native geometry instead of penalties.

**Mini derivation lens.**

1. Choose a point $p$ on the manifold $M$ and name the local representation used near $p$.
2. Move the question into a chart, tangent space, or embedded constraint where first-order calculus is available.
3. Compute the local object: derivative, tangent projection, metric-weighted gradient, path velocity, or retraction step.
4. Translate the result back into coordinate-free language so the answer is not tied to one chart by accident.
5. Check the invariant: the point remains on $M$, the direction remains in $T_pM$, or the distance/gradient uses the stated metric.

**Implementation lens.**

A practical ML implementation should store both the ambient array representation and the geometric contract attached to it. For example, a normalized embedding is not just a vector; it is a point on a sphere. An orthogonal weight matrix is not just a matrix; it is a point on a Stiefel-type constraint. A covariance matrix is not just a symmetric array; it must stay positive definite.

The clean computational pattern is: encode the state, compute an ambient derivative if needed, convert it into a tangent or metric-aware object, take a small local step, and then return to the manifold with a geodesic formula or retraction. This is the same pattern used in the companion notebooks, just scaled down to visible two- and three-dimensional examples.

The important warning is that coordinate code can pass shape checks while still violating geometry. Differential geometry adds checks that are semantic: tangentness, smooth compatibility, metric choice, path validity, and constraint preservation.

Practical checklist:

- State the manifold and whether it is abstract, embedded, or quotient-like.
- State the local coordinates or tangent representation being used.
- Separate ambient vectors from tangent vectors.
- Name the metric before computing distances, angles, or gradients.
- Use geodesics or retractions when moving on the manifold.
- For ML claims, identify whether geometry is data geometry, parameter geometry, or statistical geometry.

Local diagnostic: Write gradient, tangent projection, retraction, and stopping criterion.

The companion notebook uses low-dimensional synthetic examples: circles, spheres, tangent projections, spherical interpolation, SPD matrices, and orthogonality constraints. These examples keep geometry visible while preserving the same update logic used in higher-dimensional ML systems.

| Compact ML phrase | Differential-geometric reading |
| --- | --- |
| local linearization | tangent-space approximation at a point |
| normalized embedding | point on a sphere with tangent constraints |
| natural gradient | Riemannian gradient under Fisher metric |
| orthogonal weights | point on a Stiefel-type manifold |
| latent interpolation | path that may need geodesic structure |
| covariance geometry | SPD manifold rather than arbitrary matrices |

A useful learning move is to compute everything first on a sphere. The sphere has visible curvature, simple tangent spaces, closed-form geodesics, and practical retractions. Once those are clear, Stiefel, Grassmann, SPD, and information-geometric examples become less mysterious.

For implementation, the main discipline is to avoid leaving the manifold silently. If a gradient step violates a constraint, either project the gradient into the tangent space before stepping or use a method whose update is intrinsic by design.

The final question for this subsection is whether a Euclidean formula is being used as an approximation, a coordinate expression, or a mistaken replacement for geometry. Differential geometry is the habit of telling those cases apart.

### 4.4 Natural-gradient learning and Fisher geometry

Natural-gradient learning and Fisher geometry belongs to the canonical scope of Optimization on Manifolds. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: Riemannian gradient descent, retractions, vector transport, Riemannian Hessian preview, first-order optimality, matrix manifolds, and ML optimization examples. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$\operatorname{grad} f(p)=\Pi_{T_pM}\left(\nabla \bar{f}(p)\right)\quad\text{for embedded submanifolds with Euclidean metric}.$$

**Operational definition.**

The Riemannian gradient is the tangent vector whose inner product with any direction equals the directional derivative.

**Worked reading.**

In coordinates with metric matrix $G$, the Riemannian gradient is $G^{-1}
abla f$, not usually the raw Euclidean gradient.

| Geometric object | Meaning | AI interpretation |
| --- | --- | --- |
| Manifold $M$ | Curved space with local coordinates | Data manifold, latent space, constraint set, parameter space |
| Chart $\varphi$ | Local coordinate map | Local representation or embedding coordinates |
| Tangent space $T_pM$ | Linearized directions at $p$ | Local perturbations, gradients, velocities |
| Metric $g_p$ | Inner product on $T_pM$ | Geometry-aware length, angle, steepest descent |
| Geodesic | Straightest curved-space path | Latent interpolation, shortest motion, curved optimization path |
| Retraction | Practical map from tangent step back to $M$ | Efficient constrained update in training loops |

Three examples of natural-gradient learning and fisher geometry:

1. Natural gradient using Fisher information.
2. Projected gradient on the sphere.
3. Geometry-aware update for SPD covariance matrices.

Two non-examples clarify the boundary:

1. Raw parameter gradient treated as invariant under reparameterization.
2. A direction off the tangent space called a manifold gradient.

Proof or verification habit for natural-gradient learning and fisher geometry:

Use the defining identity $g_p(\operatorname{grad} f,\mathbf{v})=df_p[\mathbf{v}]$ for all tangent directions.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, natural-gradient learning and fisher geometry matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

Natural gradient and second-order preconditioning are geometry choices, not only optimizer tricks.

**Mini derivation lens.**

1. Choose a point $p$ on the manifold $M$ and name the local representation used near $p$.
2. Move the question into a chart, tangent space, or embedded constraint where first-order calculus is available.
3. Compute the local object: derivative, tangent projection, metric-weighted gradient, path velocity, or retraction step.
4. Translate the result back into coordinate-free language so the answer is not tied to one chart by accident.
5. Check the invariant: the point remains on $M$, the direction remains in $T_pM$, or the distance/gradient uses the stated metric.

**Implementation lens.**

A practical ML implementation should store both the ambient array representation and the geometric contract attached to it. For example, a normalized embedding is not just a vector; it is a point on a sphere. An orthogonal weight matrix is not just a matrix; it is a point on a Stiefel-type constraint. A covariance matrix is not just a symmetric array; it must stay positive definite.

The clean computational pattern is: encode the state, compute an ambient derivative if needed, convert it into a tangent or metric-aware object, take a small local step, and then return to the manifold with a geodesic formula or retraction. This is the same pattern used in the companion notebooks, just scaled down to visible two- and three-dimensional examples.

The important warning is that coordinate code can pass shape checks while still violating geometry. Differential geometry adds checks that are semantic: tangentness, smooth compatibility, metric choice, path validity, and constraint preservation.

Practical checklist:

- State the manifold and whether it is abstract, embedded, or quotient-like.
- State the local coordinates or tangent representation being used.
- Separate ambient vectors from tangent vectors.
- Name the metric before computing distances, angles, or gradients.
- Use geodesics or retractions when moving on the manifold.
- For ML claims, identify whether geometry is data geometry, parameter geometry, or statistical geometry.

Local diagnostic: Ask which metric converts covectors into update vectors.

The companion notebook uses low-dimensional synthetic examples: circles, spheres, tangent projections, spherical interpolation, SPD matrices, and orthogonality constraints. These examples keep geometry visible while preserving the same update logic used in higher-dimensional ML systems.

| Compact ML phrase | Differential-geometric reading |
| --- | --- |
| local linearization | tangent-space approximation at a point |
| normalized embedding | point on a sphere with tangent constraints |
| natural gradient | Riemannian gradient under Fisher metric |
| orthogonal weights | point on a Stiefel-type manifold |
| latent interpolation | path that may need geodesic structure |
| covariance geometry | SPD manifold rather than arbitrary matrices |

A useful learning move is to compute everything first on a sphere. The sphere has visible curvature, simple tangent spaces, closed-form geodesics, and practical retractions. Once those are clear, Stiefel, Grassmann, SPD, and information-geometric examples become less mysterious.

For implementation, the main discipline is to avoid leaving the manifold silently. If a gradient step violates a constraint, either project the gradient into the tangent space before stepping or use a method whose update is intrinsic by design.

The final question for this subsection is whether a Euclidean formula is being used as an approximation, a coordinate expression, or a mistaken replacement for geometry. Differential geometry is the habit of telling those cases apart.

### 4.5 Hyperbolic and SPD representation learning

Hyperbolic and SPD representation learning belongs to the canonical scope of Optimization on Manifolds. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: Riemannian gradient descent, retractions, vector transport, Riemannian Hessian preview, first-order optimality, matrix manifolds, and ML optimization examples. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$\operatorname{grad} f(p)=0\quad\text{is the first-order criticality condition on }M.$$

**Operational definition.**

The manifold hypothesis says high-dimensional observations often concentrate near a lower-dimensional structure.

**Worked reading.**

Images may live in pixel space, but small semantic changes such as pose or lighting often vary along far fewer directions than the number of pixels.

| Geometric object | Meaning | AI interpretation |
| --- | --- | --- |
| Manifold $M$ | Curved space with local coordinates | Data manifold, latent space, constraint set, parameter space |
| Chart $\varphi$ | Local coordinate map | Local representation or embedding coordinates |
| Tangent space $T_pM$ | Linearized directions at $p$ | Local perturbations, gradients, velocities |
| Metric $g_p$ | Inner product on $T_pM$ | Geometry-aware length, angle, steepest descent |
| Geodesic | Straightest curved-space path | Latent interpolation, shortest motion, curved optimization path |
| Retraction | Practical map from tangent step back to $M$ | Efficient constrained update in training loops |

Three examples of hyperbolic and spd representation learning:

1. Autoencoder latent spaces.
2. Embedding neighborhoods with low local rank.
3. Diffusion trajectories following learned score geometry.

Two non-examples clarify the boundary:

1. Uniform noise in every ambient direction.
2. A dataset whose classes occupy disconnected structures but are forced into one manifold.

Proof or verification habit for hyperbolic and spd representation learning:

Evidence is empirical, not theorem-level: estimate local dimension, reconstruction error, neighborhood stability, and tangent consistency.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, hyperbolic and spd representation learning matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

This hypothesis motivates representation learning, dimensionality reduction, and geometry-aware generative modeling.

**Mini derivation lens.**

1. Choose a point $p$ on the manifold $M$ and name the local representation used near $p$.
2. Move the question into a chart, tangent space, or embedded constraint where first-order calculus is available.
3. Compute the local object: derivative, tangent projection, metric-weighted gradient, path velocity, or retraction step.
4. Translate the result back into coordinate-free language so the answer is not tied to one chart by accident.
5. Check the invariant: the point remains on $M$, the direction remains in $T_pM$, or the distance/gradient uses the stated metric.

**Implementation lens.**

A practical ML implementation should store both the ambient array representation and the geometric contract attached to it. For example, a normalized embedding is not just a vector; it is a point on a sphere. An orthogonal weight matrix is not just a matrix; it is a point on a Stiefel-type constraint. A covariance matrix is not just a symmetric array; it must stay positive definite.

The clean computational pattern is: encode the state, compute an ambient derivative if needed, convert it into a tangent or metric-aware object, take a small local step, and then return to the manifold with a geodesic formula or retraction. This is the same pattern used in the companion notebooks, just scaled down to visible two- and three-dimensional examples.

The important warning is that coordinate code can pass shape checks while still violating geometry. Differential geometry adds checks that are semantic: tangentness, smooth compatibility, metric choice, path validity, and constraint preservation.

Practical checklist:

- State the manifold and whether it is abstract, embedded, or quotient-like.
- State the local coordinates or tangent representation being used.
- Separate ambient vectors from tangent vectors.
- Name the metric before computing distances, angles, or gradients.
- Use geodesics or retractions when moving on the manifold.
- For ML claims, identify whether geometry is data geometry, parameter geometry, or statistical geometry.

Local diagnostic: Ask whether the data are on, near, or only metaphorically described by a manifold.

The companion notebook uses low-dimensional synthetic examples: circles, spheres, tangent projections, spherical interpolation, SPD matrices, and orthogonality constraints. These examples keep geometry visible while preserving the same update logic used in higher-dimensional ML systems.

| Compact ML phrase | Differential-geometric reading |
| --- | --- |
| local linearization | tangent-space approximation at a point |
| normalized embedding | point on a sphere with tangent constraints |
| natural gradient | Riemannian gradient under Fisher metric |
| orthogonal weights | point on a Stiefel-type manifold |
| latent interpolation | path that may need geodesic structure |
| covariance geometry | SPD manifold rather than arbitrary matrices |

A useful learning move is to compute everything first on a sphere. The sphere has visible curvature, simple tangent spaces, closed-form geodesics, and practical retractions. Once those are clear, Stiefel, Grassmann, SPD, and information-geometric examples become less mysterious.

For implementation, the main discipline is to avoid leaving the manifold silently. If a gradient step violates a constraint, either project the gradient into the tangent space before stepping or use a method whose update is intrinsic by design.

The final question for this subsection is whether a Euclidean formula is being used as an approximation, a coordinate expression, or a mistaken replacement for geometry. Differential geometry is the habit of telling those cases apart.

## 5. Common Mistakes

| # | Mistake | Why It Is Wrong | Fix |
| --- | --- | --- | --- |
| 1 | Treating a manifold as just a nonlinear set | A manifold includes compatible local coordinates and smooth structure. | State charts, tangent spaces, or the embedding structure being used. |
| 2 | Confusing intrinsic dimension with ambient dimension | A sphere in $\mathbb{R}^3$ is two-dimensional. | Separate coordinates on the manifold from coordinates in the ambient space. |
| 3 | Using Euclidean gradients without projection | Euclidean gradients may point off the manifold. | Project to $T_pM$ or compute the Riemannian gradient. |
| 4 | Assuming shortest and straightest always coincide globally | Geodesics are locally shortest under conditions, not always globally minimizing. | Check cut loci, endpoints, and global topology. |
| 5 | Calling any interpolation a geodesic | Linear interpolation in ambient space may leave the manifold. | Use geodesic formulas or retractions. |
| 6 | Forgetting the metric | Angles, distances, gradients, and geodesics depend on the metric. | Name $g$ before making geometric claims. |
| 7 | Using projection as a retraction without checking local behavior | A retraction must match the exponential map to first order. | Verify $R_p(0)=p$ and $dR_p(0)=\operatorname{id}$. |
| 8 | Flattening SPD matrices as ordinary vectors | SPD matrices have positivity and natural metrics that flattening can destroy. | Use SPD-aware geometry when covariance structure matters. |
| 9 | Treating quotient spaces as ordinary parameter spaces | Symmetry creates equivalence classes. | Identify whether points represent states or equivalence classes. |
| 10 | Overclaiming the manifold hypothesis | Real data may lie near noisy, stratified, or mixed-dimensional structures. | Use diagnostics and local dimension estimates. |

## 6. Exercises

1. (*) Run one Riemannian gradient step on the sphere and verify that the new point stays on the sphere.
   - (a) State the manifold and local representation.
   - (b) Identify the tangent space, metric, path, or retraction involved.
   - (c) Compute the finite or low-dimensional example.
   - (d) Interpret the result for an ML, LLM, or representation-learning setting.

2. (*) Check the two defining properties of a retraction for the normalization map on $S^{n-1}$.
   - (a) State the manifold and local representation.
   - (b) Identify the tangent space, metric, path, or retraction involved.
   - (c) Compute the finite or low-dimensional example.
   - (d) Interpret the result for an ML, LLM, or representation-learning setting.

3. (*) Project an ambient matrix direction onto the tangent space of the Stiefel manifold.
   - (a) State the manifold and local representation.
   - (b) Identify the tangent space, metric, path, or retraction involved.
   - (c) Compute the finite or low-dimensional example.
   - (d) Interpret the result for an ML, LLM, or representation-learning setting.

4. (**) Explain why Grassmann optimization treats bases that span the same subspace as equivalent.
   - (a) State the manifold and local representation.
   - (b) Identify the tangent space, metric, path, or retraction involved.
   - (c) Compute the finite or low-dimensional example.
   - (d) Interpret the result for an ML, LLM, or representation-learning setting.

5. (**) Compare the Euclidean and affine-invariant views of SPD covariance updates.
   - (a) State the manifold and local representation.
   - (b) Identify the tangent space, metric, path, or retraction involved.
   - (c) Compute the finite or low-dimensional example.
   - (d) Interpret the result for an ML, LLM, or representation-learning setting.

6. (**) State the first-order optimality condition for a smooth function on a manifold.
   - (a) State the manifold and local representation.
   - (b) Identify the tangent space, metric, path, or retraction involved.
   - (c) Compute the finite or low-dimensional example.
   - (d) Interpret the result for an ML, LLM, or representation-learning setting.

7. (***) Describe when vector transport is needed in a practical optimizer.
   - (a) State the manifold and local representation.
   - (b) Identify the tangent space, metric, path, or retraction involved.
   - (c) Compute the finite or low-dimensional example.
   - (d) Interpret the result for an ML, LLM, or representation-learning setting.

8. (***) Frame PCA as an optimization problem over subspaces rather than over raw coordinate matrices.
   - (a) State the manifold and local representation.
   - (b) Identify the tangent space, metric, path, or retraction involved.
   - (c) Compute the finite or low-dimensional example.
   - (d) Interpret the result for an ML, LLM, or representation-learning setting.

9. (***) Explain how Fisher natural gradient is a Riemannian steepest-descent method.
   - (a) State the manifold and local representation.
   - (b) Identify the tangent space, metric, path, or retraction involved.
   - (c) Compute the finite or low-dimensional example.
   - (d) Interpret the result for an ML, LLM, or representation-learning setting.

10. (***) Design a training-loop checklist for using manifold constraints in an ML model.
   - (a) State the manifold and local representation.
   - (b) Identify the tangent space, metric, path, or retraction involved.
   - (c) Compute the finite or low-dimensional example.
   - (d) Interpret the result for an ML, LLM, or representation-learning setting.

## 7. Why This Matters for AI

| Concept | AI Impact |
| --- | --- |
| Manifold hypothesis | Explains why high-dimensional data can have low-dimensional local structure. |
| Tangent spaces | Provide local linear approximations used in embeddings, Jacobians, and sensitivity analysis. |
| Riemannian metric | Defines geometry-aware gradients, distances, and regularization. |
| Natural gradient | Uses Fisher geometry to make parameter updates less coordinate-dependent. |
| Geodesics | Support curved interpolation, distance, and representation-path analysis. |
| Retractions | Make manifold optimization computationally practical. |
| Stiefel and Grassmann manifolds | Model orthogonality and subspace constraints in PCA and representation learning. |
| SPD manifolds | Respect covariance and positive-definite structure in probabilistic models. |

## 8. Conceptual Bridge

Optimization on Manifolds follows measure theory because probability and density statements become most useful in AI once they live on structured spaces. Chapter 24 made distributions rigorous. Chapter 25 asks what happens when the spaces that carry data, parameters, or distributions are curved.

The backward bridge is local linearization. Linear algebra gave vector spaces, calculus gave derivatives, functional analysis gave inner-product geometry, and measure theory gave rigorous probability. Differential geometry combines these ideas point-by-point on curved domains.

The forward bridge is practice: modern ML often uses normalized embeddings, orthogonal constraints, low-rank subspaces, covariance matrices, hyperbolic representations, and natural-gradient updates. Those are not exotic decorations; they are geometric objects in training systems.

```text
+------------------------------------------------------------------+
| Flat math: vectors, matrices, gradients, probability measures     |
| Differential geometry: local linear math on curved spaces         |
| ML use: embeddings, latent paths, natural gradients, constraints  |
+------------------------------------------------------------------+
```

## References

- Boumal. An Introduction to Optimization on Smooth Manifolds. https://www.nicolasboumal.net/book/IntroOptimManifolds_Boumal_2023.pdf
- Absil, Mahony, Sepulchre. Optimization Algorithms on Matrix Manifolds. https://press.princeton.edu/books/hardcover/9780691132983/optimization-algorithms-on-matrix-manifolds
- Amari. Natural Gradient Works Efficiently in Learning. https://doi.org/10.1162/089976698300017746
- Han et al.. Riemannian Coordinate Descent Algorithms on Matrix Manifolds. https://proceedings.mlr.press/v235/han24c.html
