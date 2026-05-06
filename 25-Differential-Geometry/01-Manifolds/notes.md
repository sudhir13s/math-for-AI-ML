[Back to Curriculum](../../README.md) | [Previous: Radon-Nikodym Theorem](../../24-Measure-Theory/04-Radon-Nikodym-Theorem/notes.md) | [Next: Riemannian Geometry](../02-Riemannian-Geometry/notes.md)

---

# Manifolds

> _"A manifold is a space that looks flat when you stand close enough."_

## Overview

Manifolds give the language for curved state spaces, latent spaces, symmetry spaces, and locally linear representations in AI.

Differential geometry is the part of mathematics that makes calculus work on curved spaces. Earlier chapters treated vectors, matrices, probabilities, optimization, and measures mostly in flat coordinate systems. This chapter explains what changes when the object being modeled is a sphere, a space of subspaces, a positive-definite covariance matrix, a statistical model, or a learned latent manifold.

This section uses LaTeX Markdown throughout. Inline mathematics uses `$...$`, and display mathematics uses `$$...$$`. The AI focus is practical: local coordinates, tangent spaces, metrics, geodesics, retractions, natural gradients, and matrix-manifold constraints.

## Prerequisites

- [Functions and Mappings](../../01-Mathematical-Foundations/03-Functions-and-Mappings/notes.md)
- [Vector Spaces and Subspaces](../../02-Linear-Algebra-Basics/06-Vector-Spaces-Subspaces/notes.md)
- [Jacobian and Hessian](../../05-Multivariate-Calculus/02-Jacobians-and-Hessians/notes.md)
- [Radon-Nikodym Theorem](../../24-Measure-Theory/04-Radon-Nikodym-Theorem/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Executable demonstrations for manifolds |
| [exercises.ipynb](exercises.ipynb) | Graded practice for manifolds |

## Learning Objectives

After completing this section, you will be able to:

- Define charts, atlases, smooth compatibility, and smooth manifolds
- Explain why local coordinates are needed for curved spaces
- Distinguish local Euclidean structure from global topology
- Compute tangent spaces for embedded examples such as spheres
- Interpret tangent vectors as velocities of curves
- Use differentials to push tangent vectors through smooth maps
- Identify embedded and immersed submanifolds
- Connect the manifold hypothesis to representation learning
- Recognize common ML manifolds: spheres, Stiefel, Grassmann, and SPD matrices
- Prepare for Riemannian metrics by separating topology, smoothness, and geometry

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 Why curved spaces need local coordinates](#11-why-curved-spaces-need-local-coordinates)
  - [1.2 The manifold hypothesis in ML](#12-the-manifold-hypothesis-in-ml)
  - [1.3 Local Euclidean behavior vs global curvature](#13-local-euclidean-behavior-vs-global-curvature)
  - [1.4 Charts atlases and coordinate patches](#14-charts-atlases-and-coordinate-patches)
  - [1.5 Examples: sphere torus Stiefel Grassmann SPD matrices](#15-examples-sphere-torus-stiefel-grassmann-spd-matrices)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Topological manifold](#21-topological-manifold)
  - [2.2 Smooth atlas and smooth compatibility](#22-smooth-atlas-and-smooth-compatibility)
  - [2.3 Smooth manifold $M$](#23-smooth-manifold)
  - [2.4 Smooth maps between manifolds](#24-smooth-maps-between-manifolds)
  - [2.5 Embedded and immersed submanifolds](#25-embedded-and-immersed-submanifolds)
- [3. Core Theory](#3-core-theory)
  - [3.1 Tangent vectors as velocities of curves](#31-tangent-vectors-as-velocities-of-curves)
  - [3.2 Tangent spaces $T_pM$](#32-tangent-spaces)
  - [3.3 Differentials and pushforwards $dF_p:T_pM\to T_{F(p)}N$](#33-differentials-and-pushforwards)
  - [3.4 Tangent bundle $TM$](#34-tangent-bundle)
  - [3.5 Vector fields and flows preview](#35-vector-fields-and-flows-preview)
- [4. AI Applications](#4-ai-applications)
  - [4.1 Data manifolds and representation learning](#41-data-manifolds-and-representation-learning)
  - [4.2 Latent spaces in VAEs and diffusion models](#42-latent-spaces-in-vaes-and-diffusion-models)
  - [4.3 Embedding manifolds and local linearization](#43-embedding-manifolds-and-local-linearization)
  - [4.4 Symmetry and quotient spaces preview](#44-symmetry-and-quotient-spaces-preview)
  - [4.5 Manifold learning diagnostics](#45-manifold-learning-diagnostics)
- [5. Common Mistakes](#5-common-mistakes)
- [6. Exercises](#6-exercises)
- [7. Why This Matters for AI](#7-why-this-matters-for-ai)
- [8. Conceptual Bridge](#8-conceptual-bridge)
- [References](#references)

---

## 1. Intuition

Intuition develops the part of manifolds specified by the approved Chapter 25 table of contents. The treatment is geometry-first and AI-facing.

### 1.1 Why curved spaces need local coordinates

Why curved spaces need local coordinates belongs to the canonical scope of Manifolds. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: smooth manifolds, charts, atlases, tangent spaces, differentials, tangent bundles, embedded submanifolds, and ML manifold intuition. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$\varphi:U\subseteq M\to \varphi(U)\subseteq \mathbb{R}^d.$$

**Operational definition.**

A chart turns a neighborhood of a curved space into coordinates in Euclidean space; an atlas is a compatible collection of such charts.

**Worked reading.**

On the unit circle, angle $t$ gives local coordinates except where a single global angle chart breaks. Multiple patches avoid artificial singularities.

| Geometric object | Meaning | AI interpretation |
| --- | --- | --- |
| Manifold $M$ | Curved space with local coordinates | Data manifold, latent space, constraint set, parameter space |
| Chart $\varphi$ | Local coordinate map | Local representation or embedding coordinates |
| Tangent space $T_pM$ | Linearized directions at $p$ | Local perturbations, gradients, velocities |
| Metric $g_p$ | Inner product on $T_pM$ | Geometry-aware length, angle, steepest descent |
| Geodesic | Straightest curved-space path | Latent interpolation, shortest motion, curved optimization path |
| Retraction | Practical map from tangent step back to $M$ | Efficient constrained update in training loops |

Three examples of why curved spaces need local coordinates:

1. Latitude-longitude patches on a sphere.
2. Angle coordinates on a circle.
3. Local PCA coordinates on a data cloud.

Two non-examples clarify the boundary:

1. One global coordinate map with a seam treated as smooth.
2. A point cloud with no neighborhood structure.

Proof or verification habit for why curved spaces need local coordinates:

The proof habit is to check transition maps between overlapping charts, not only individual parameterizations.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, why curved spaces need local coordinates matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

Local coordinates are the mathematical version of local representation learning: simple coordinates may exist nearby even when no global linear model exists.

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

Local diagnostic: Name the patch, coordinates, and transition map.

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

### 1.2 The manifold hypothesis in ML

The manifold hypothesis in ML belongs to the canonical scope of Manifolds. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: smooth manifolds, charts, atlases, tangent spaces, differentials, tangent bundles, embedded submanifolds, and ML manifold intuition. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$\varphi_\beta\circ\varphi_\alpha^{-1}:\varphi_\alpha(U_\alpha\cap U_\beta)\to\varphi_\beta(U_\alpha\cap U_\beta).$$

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

Three examples of the manifold hypothesis in ml:

1. Autoencoder latent spaces.
2. Embedding neighborhoods with low local rank.
3. Diffusion trajectories following learned score geometry.

Two non-examples clarify the boundary:

1. Uniform noise in every ambient direction.
2. A dataset whose classes occupy disconnected structures but are forced into one manifold.

Proof or verification habit for the manifold hypothesis in ml:

Evidence is empirical, not theorem-level: estimate local dimension, reconstruction error, neighborhood stability, and tangent consistency.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, the manifold hypothesis in ml matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

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

### 1.3 Local Euclidean behavior vs global curvature

Local Euclidean behavior vs global curvature belongs to the canonical scope of Manifolds. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: smooth manifolds, charts, atlases, tangent spaces, differentials, tangent bundles, embedded submanifolds, and ML manifold intuition. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$T_pM=\{\dot{\gamma}(0):\gamma(0)=p,\ \gamma \text{ smooth curve in }M\}.$$

**Operational definition.**

A connection differentiates vector fields along curves while keeping the result tangent to the manifold; curvature measures how tangent spaces twist around loops.

**Worked reading.**

On a sphere, a tangent vector transported around a loop can rotate relative to its starting direction. That mismatch is curvature made visible.

| Geometric object | Meaning | AI interpretation |
| --- | --- | --- |
| Manifold $M$ | Curved space with local coordinates | Data manifold, latent space, constraint set, parameter space |
| Chart $\varphi$ | Local coordinate map | Local representation or embedding coordinates |
| Tangent space $T_pM$ | Linearized directions at $p$ | Local perturbations, gradients, velocities |
| Metric $g_p$ | Inner product on $T_pM$ | Geometry-aware length, angle, steepest descent |
| Geodesic | Straightest curved-space path | Latent interpolation, shortest motion, curved optimization path |
| Retraction | Practical map from tangent step back to $M$ | Efficient constrained update in training loops |

Three examples of local euclidean behavior vs global curvature:

1. Levi-Civita connection.
2. Covariant derivative of a velocity field.
3. Curvature affecting geodesic spread.

Two non-examples clarify the boundary:

1. Ordinary derivative of a tangent vector that leaves the tangent space.
2. Curvature treated as only a visualization artifact.

Proof or verification habit for local euclidean behavior vs global curvature:

For a first course, focus on compatibility and projection intuition; full curvature tensors are preview material here.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, local euclidean behavior vs global curvature matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

Curvature affects interpolation, optimization stability, and how local neighborhoods scale.

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

Local diagnostic: Distinguish differentiating coordinates from differentiating geometric vector fields.

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

### 1.4 Charts atlases and coordinate patches

Charts atlases and coordinate patches belongs to the canonical scope of Manifolds. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: smooth manifolds, charts, atlases, tangent spaces, differentials, tangent bundles, embedded submanifolds, and ML manifold intuition. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$dF_p(\mathbf{v})=\frac{d}{dt}\bigg|_{t=0}F(\gamma(t)),\qquad \dot{\gamma}(0)=\mathbf{v}.$$

**Operational definition.**

A chart turns a neighborhood of a curved space into coordinates in Euclidean space; an atlas is a compatible collection of such charts.

**Worked reading.**

On the unit circle, angle $t$ gives local coordinates except where a single global angle chart breaks. Multiple patches avoid artificial singularities.

| Geometric object | Meaning | AI interpretation |
| --- | --- | --- |
| Manifold $M$ | Curved space with local coordinates | Data manifold, latent space, constraint set, parameter space |
| Chart $\varphi$ | Local coordinate map | Local representation or embedding coordinates |
| Tangent space $T_pM$ | Linearized directions at $p$ | Local perturbations, gradients, velocities |
| Metric $g_p$ | Inner product on $T_pM$ | Geometry-aware length, angle, steepest descent |
| Geodesic | Straightest curved-space path | Latent interpolation, shortest motion, curved optimization path |
| Retraction | Practical map from tangent step back to $M$ | Efficient constrained update in training loops |

Three examples of charts atlases and coordinate patches:

1. Latitude-longitude patches on a sphere.
2. Angle coordinates on a circle.
3. Local PCA coordinates on a data cloud.

Two non-examples clarify the boundary:

1. One global coordinate map with a seam treated as smooth.
2. A point cloud with no neighborhood structure.

Proof or verification habit for charts atlases and coordinate patches:

The proof habit is to check transition maps between overlapping charts, not only individual parameterizations.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, charts atlases and coordinate patches matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

Local coordinates are the mathematical version of local representation learning: simple coordinates may exist nearby even when no global linear model exists.

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

Local diagnostic: Name the patch, coordinates, and transition map.

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

### 1.5 Examples: sphere torus Stiefel Grassmann SPD matrices

Examples: sphere torus Stiefel Grassmann SPD matrices belongs to the canonical scope of Manifolds. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: smooth manifolds, charts, atlases, tangent spaces, differentials, tangent bundles, embedded submanifolds, and ML manifold intuition. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$\varphi:U\subseteq M\to \varphi(U)\subseteq \mathbb{R}^d.$$

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

Three examples of examples: sphere torus stiefel grassmann spd matrices:

1. PCA on Grassmann manifolds.
2. Orthogonal weights on Stiefel manifolds.
3. Covariance learning on SPD manifolds.

Two non-examples clarify the boundary:

1. Euclidean gradient descent followed by arbitrary clipping.
2. A projection step that destroys the first-order update direction.

Proof or verification habit for examples: sphere torus stiefel grassmann spd matrices:

Check tangent feasibility, descent direction under the metric, and retraction properties.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, examples: sphere torus stiefel grassmann spd matrices matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

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

## 2. Formal Definitions

Formal Definitions develops the part of manifolds specified by the approved Chapter 25 table of contents. The treatment is geometry-first and AI-facing.

### 2.1 Topological manifold

Topological manifold belongs to the canonical scope of Manifolds. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: smooth manifolds, charts, atlases, tangent spaces, differentials, tangent bundles, embedded submanifolds, and ML manifold intuition. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$\varphi_\beta\circ\varphi_\alpha^{-1}:\varphi_\alpha(U_\alpha\cap U_\beta)\to\varphi_\beta(U_\alpha\cap U_\beta).$$

**Operational definition.**

A chart turns a neighborhood of a curved space into coordinates in Euclidean space; an atlas is a compatible collection of such charts.

**Worked reading.**

On the unit circle, angle $t$ gives local coordinates except where a single global angle chart breaks. Multiple patches avoid artificial singularities.

| Geometric object | Meaning | AI interpretation |
| --- | --- | --- |
| Manifold $M$ | Curved space with local coordinates | Data manifold, latent space, constraint set, parameter space |
| Chart $\varphi$ | Local coordinate map | Local representation or embedding coordinates |
| Tangent space $T_pM$ | Linearized directions at $p$ | Local perturbations, gradients, velocities |
| Metric $g_p$ | Inner product on $T_pM$ | Geometry-aware length, angle, steepest descent |
| Geodesic | Straightest curved-space path | Latent interpolation, shortest motion, curved optimization path |
| Retraction | Practical map from tangent step back to $M$ | Efficient constrained update in training loops |

Three examples of topological manifold:

1. Latitude-longitude patches on a sphere.
2. Angle coordinates on a circle.
3. Local PCA coordinates on a data cloud.

Two non-examples clarify the boundary:

1. One global coordinate map with a seam treated as smooth.
2. A point cloud with no neighborhood structure.

Proof or verification habit for topological manifold:

The proof habit is to check transition maps between overlapping charts, not only individual parameterizations.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, topological manifold matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

Local coordinates are the mathematical version of local representation learning: simple coordinates may exist nearby even when no global linear model exists.

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

Local diagnostic: Name the patch, coordinates, and transition map.

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

### 2.2 Smooth atlas and smooth compatibility

Smooth atlas and smooth compatibility belongs to the canonical scope of Manifolds. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: smooth manifolds, charts, atlases, tangent spaces, differentials, tangent bundles, embedded submanifolds, and ML manifold intuition. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$T_pM=\{\dot{\gamma}(0):\gamma(0)=p,\ \gamma \text{ smooth curve in }M\}.$$

**Operational definition.**

A chart turns a neighborhood of a curved space into coordinates in Euclidean space; an atlas is a compatible collection of such charts.

**Worked reading.**

On the unit circle, angle $t$ gives local coordinates except where a single global angle chart breaks. Multiple patches avoid artificial singularities.

| Geometric object | Meaning | AI interpretation |
| --- | --- | --- |
| Manifold $M$ | Curved space with local coordinates | Data manifold, latent space, constraint set, parameter space |
| Chart $\varphi$ | Local coordinate map | Local representation or embedding coordinates |
| Tangent space $T_pM$ | Linearized directions at $p$ | Local perturbations, gradients, velocities |
| Metric $g_p$ | Inner product on $T_pM$ | Geometry-aware length, angle, steepest descent |
| Geodesic | Straightest curved-space path | Latent interpolation, shortest motion, curved optimization path |
| Retraction | Practical map from tangent step back to $M$ | Efficient constrained update in training loops |

Three examples of smooth atlas and smooth compatibility:

1. Latitude-longitude patches on a sphere.
2. Angle coordinates on a circle.
3. Local PCA coordinates on a data cloud.

Two non-examples clarify the boundary:

1. One global coordinate map with a seam treated as smooth.
2. A point cloud with no neighborhood structure.

Proof or verification habit for smooth atlas and smooth compatibility:

The proof habit is to check transition maps between overlapping charts, not only individual parameterizations.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, smooth atlas and smooth compatibility matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

Local coordinates are the mathematical version of local representation learning: simple coordinates may exist nearby even when no global linear model exists.

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

Local diagnostic: Name the patch, coordinates, and transition map.

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

### 2.3 Smooth manifold $M$

Smooth manifold $M$ belongs to the canonical scope of Manifolds. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: smooth manifolds, charts, atlases, tangent spaces, differentials, tangent bundles, embedded submanifolds, and ML manifold intuition. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$dF_p(\mathbf{v})=\frac{d}{dt}\bigg|_{t=0}F(\gamma(t)),\qquad \dot{\gamma}(0)=\mathbf{v}.$$

**Operational definition.**

A chart turns a neighborhood of a curved space into coordinates in Euclidean space; an atlas is a compatible collection of such charts.

**Worked reading.**

On the unit circle, angle $t$ gives local coordinates except where a single global angle chart breaks. Multiple patches avoid artificial singularities.

| Geometric object | Meaning | AI interpretation |
| --- | --- | --- |
| Manifold $M$ | Curved space with local coordinates | Data manifold, latent space, constraint set, parameter space |
| Chart $\varphi$ | Local coordinate map | Local representation or embedding coordinates |
| Tangent space $T_pM$ | Linearized directions at $p$ | Local perturbations, gradients, velocities |
| Metric $g_p$ | Inner product on $T_pM$ | Geometry-aware length, angle, steepest descent |
| Geodesic | Straightest curved-space path | Latent interpolation, shortest motion, curved optimization path |
| Retraction | Practical map from tangent step back to $M$ | Efficient constrained update in training loops |

Three examples of smooth manifold $m$:

1. Latitude-longitude patches on a sphere.
2. Angle coordinates on a circle.
3. Local PCA coordinates on a data cloud.

Two non-examples clarify the boundary:

1. One global coordinate map with a seam treated as smooth.
2. A point cloud with no neighborhood structure.

Proof or verification habit for smooth manifold $m$:

The proof habit is to check transition maps between overlapping charts, not only individual parameterizations.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, smooth manifold $m$ matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

Local coordinates are the mathematical version of local representation learning: simple coordinates may exist nearby even when no global linear model exists.

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

Local diagnostic: Name the patch, coordinates, and transition map.

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

### 2.4 Smooth maps between manifolds

Smooth maps between manifolds belongs to the canonical scope of Manifolds. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: smooth manifolds, charts, atlases, tangent spaces, differentials, tangent bundles, embedded submanifolds, and ML manifold intuition. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$\varphi:U\subseteq M\to \varphi(U)\subseteq \mathbb{R}^d.$$

**Operational definition.**

Smooth maps between manifolds belongs to the canonical scope of Manifolds: smooth manifolds, charts, atlases, tangent spaces, differentials, tangent bundles, embedded submanifolds, and ML manifold intuition.

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

Three examples of smooth maps between manifolds:

1. Sphere geometry.
2. Embedding-space local coordinates.
3. Matrix-manifold parameter constraints.

Two non-examples clarify the boundary:

1. A flat Euclidean approximation used globally.
2. A geometric claim made without metric or tangent space.

Proof or verification habit for smooth maps between manifolds:

The proof habit is to compute locally and verify coordinate-independent meaning.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, smooth maps between manifolds matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

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

### 2.5 Embedded and immersed submanifolds

Embedded and immersed submanifolds belongs to the canonical scope of Manifolds. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: smooth manifolds, charts, atlases, tangent spaces, differentials, tangent bundles, embedded submanifolds, and ML manifold intuition. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$\varphi_\beta\circ\varphi_\alpha^{-1}:\varphi_\alpha(U_\alpha\cap U_\beta)\to\varphi_\beta(U_\alpha\cap U_\beta).$$

**Operational definition.**

Embedded and immersed submanifolds belongs to the canonical scope of Manifolds: smooth manifolds, charts, atlases, tangent spaces, differentials, tangent bundles, embedded submanifolds, and ML manifold intuition.

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

Three examples of embedded and immersed submanifolds:

1. Sphere geometry.
2. Embedding-space local coordinates.
3. Matrix-manifold parameter constraints.

Two non-examples clarify the boundary:

1. A flat Euclidean approximation used globally.
2. A geometric claim made without metric or tangent space.

Proof or verification habit for embedded and immersed submanifolds:

The proof habit is to compute locally and verify coordinate-independent meaning.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, embedded and immersed submanifolds matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

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

## 3. Core Theory

Core Theory develops the part of manifolds specified by the approved Chapter 25 table of contents. The treatment is geometry-first and AI-facing.

### 3.1 Tangent vectors as velocities of curves

Tangent vectors as velocities of curves belongs to the canonical scope of Manifolds. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: smooth manifolds, charts, atlases, tangent spaces, differentials, tangent bundles, embedded submanifolds, and ML manifold intuition. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$T_pM=\{\dot{\gamma}(0):\gamma(0)=p,\ \gamma \text{ smooth curve in }M\}.$$

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

Three examples of tangent vectors as velocities of curves:

1. Velocity of a curve on a sphere.
2. Jacobian pushing embedding perturbations forward.
3. A vector field assigning one tangent direction per point.

Two non-examples clarify the boundary:

1. An arbitrary ambient vector not tangent to the constraint.
2. A finite difference step that leaves the manifold without retraction.

Proof or verification habit for tangent vectors as velocities of curves:

For embedded manifolds, differentiate the constraint; for abstract manifolds, use curves or derivations.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, tangent vectors as velocities of curves matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

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

### 3.2 Tangent spaces $T_pM$

Tangent spaces $T_pM$ belongs to the canonical scope of Manifolds. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: smooth manifolds, charts, atlases, tangent spaces, differentials, tangent bundles, embedded submanifolds, and ML manifold intuition. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$dF_p(\mathbf{v})=\frac{d}{dt}\bigg|_{t=0}F(\gamma(t)),\qquad \dot{\gamma}(0)=\mathbf{v}.$$

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

Three examples of tangent spaces $t_pm$:

1. Velocity of a curve on a sphere.
2. Jacobian pushing embedding perturbations forward.
3. A vector field assigning one tangent direction per point.

Two non-examples clarify the boundary:

1. An arbitrary ambient vector not tangent to the constraint.
2. A finite difference step that leaves the manifold without retraction.

Proof or verification habit for tangent spaces $t_pm$:

For embedded manifolds, differentiate the constraint; for abstract manifolds, use curves or derivations.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, tangent spaces $t_pm$ matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

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

### 3.3 Differentials and pushforwards $dF_p:T_pM\to T_{F(p)}N$

Differentials and pushforwards $dF_p:T_pM\to T_{F(p)}N$ belongs to the canonical scope of Manifolds. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: smooth manifolds, charts, atlases, tangent spaces, differentials, tangent bundles, embedded submanifolds, and ML manifold intuition. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$\varphi:U\subseteq M\to \varphi(U)\subseteq \mathbb{R}^d.$$

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

Three examples of differentials and pushforwards $df_p:t_pm\to t_{f(p)}n$:

1. Velocity of a curve on a sphere.
2. Jacobian pushing embedding perturbations forward.
3. A vector field assigning one tangent direction per point.

Two non-examples clarify the boundary:

1. An arbitrary ambient vector not tangent to the constraint.
2. A finite difference step that leaves the manifold without retraction.

Proof or verification habit for differentials and pushforwards $df_p:t_pm\to t_{f(p)}n$:

For embedded manifolds, differentiate the constraint; for abstract manifolds, use curves or derivations.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, differentials and pushforwards $df_p:t_pm\to t_{f(p)}n$ matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

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

### 3.4 Tangent bundle $TM$

Tangent bundle $TM$ belongs to the canonical scope of Manifolds. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: smooth manifolds, charts, atlases, tangent spaces, differentials, tangent bundles, embedded submanifolds, and ML manifold intuition. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$\varphi_\beta\circ\varphi_\alpha^{-1}:\varphi_\alpha(U_\alpha\cap U_\beta)\to\varphi_\beta(U_\alpha\cap U_\beta).$$

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

Three examples of tangent bundle $tm$:

1. Velocity of a curve on a sphere.
2. Jacobian pushing embedding perturbations forward.
3. A vector field assigning one tangent direction per point.

Two non-examples clarify the boundary:

1. An arbitrary ambient vector not tangent to the constraint.
2. A finite difference step that leaves the manifold without retraction.

Proof or verification habit for tangent bundle $tm$:

For embedded manifolds, differentiate the constraint; for abstract manifolds, use curves or derivations.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, tangent bundle $tm$ matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

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

### 3.5 Vector fields and flows preview

Vector fields and flows preview belongs to the canonical scope of Manifolds. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: smooth manifolds, charts, atlases, tangent spaces, differentials, tangent bundles, embedded submanifolds, and ML manifold intuition. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$T_pM=\{\dot{\gamma}(0):\gamma(0)=p,\ \gamma \text{ smooth curve in }M\}.$$

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

Three examples of vector fields and flows preview:

1. Velocity of a curve on a sphere.
2. Jacobian pushing embedding perturbations forward.
3. A vector field assigning one tangent direction per point.

Two non-examples clarify the boundary:

1. An arbitrary ambient vector not tangent to the constraint.
2. A finite difference step that leaves the manifold without retraction.

Proof or verification habit for vector fields and flows preview:

For embedded manifolds, differentiate the constraint; for abstract manifolds, use curves or derivations.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, vector fields and flows preview matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

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

## 4. AI Applications

AI Applications develops the part of manifolds specified by the approved Chapter 25 table of contents. The treatment is geometry-first and AI-facing.

### 4.1 Data manifolds and representation learning

Data manifolds and representation learning belongs to the canonical scope of Manifolds. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: smooth manifolds, charts, atlases, tangent spaces, differentials, tangent bundles, embedded submanifolds, and ML manifold intuition. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$dF_p(\mathbf{v})=\frac{d}{dt}\bigg|_{t=0}F(\gamma(t)),\qquad \dot{\gamma}(0)=\mathbf{v}.$$

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

Three examples of data manifolds and representation learning:

1. Autoencoder latent spaces.
2. Embedding neighborhoods with low local rank.
3. Diffusion trajectories following learned score geometry.

Two non-examples clarify the boundary:

1. Uniform noise in every ambient direction.
2. A dataset whose classes occupy disconnected structures but are forced into one manifold.

Proof or verification habit for data manifolds and representation learning:

Evidence is empirical, not theorem-level: estimate local dimension, reconstruction error, neighborhood stability, and tangent consistency.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, data manifolds and representation learning matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

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

### 4.2 Latent spaces in VAEs and diffusion models

Latent spaces in VAEs and diffusion models belongs to the canonical scope of Manifolds. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: smooth manifolds, charts, atlases, tangent spaces, differentials, tangent bundles, embedded submanifolds, and ML manifold intuition. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$\varphi:U\subseteq M\to \varphi(U)\subseteq \mathbb{R}^d.$$

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

Three examples of latent spaces in vaes and diffusion models:

1. Autoencoder latent spaces.
2. Embedding neighborhoods with low local rank.
3. Diffusion trajectories following learned score geometry.

Two non-examples clarify the boundary:

1. Uniform noise in every ambient direction.
2. A dataset whose classes occupy disconnected structures but are forced into one manifold.

Proof or verification habit for latent spaces in vaes and diffusion models:

Evidence is empirical, not theorem-level: estimate local dimension, reconstruction error, neighborhood stability, and tangent consistency.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, latent spaces in vaes and diffusion models matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

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

### 4.3 Embedding manifolds and local linearization

Embedding manifolds and local linearization belongs to the canonical scope of Manifolds. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: smooth manifolds, charts, atlases, tangent spaces, differentials, tangent bundles, embedded submanifolds, and ML manifold intuition. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$\varphi_\beta\circ\varphi_\alpha^{-1}:\varphi_\alpha(U_\alpha\cap U_\beta)\to\varphi_\beta(U_\alpha\cap U_\beta).$$

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

Three examples of embedding manifolds and local linearization:

1. Autoencoder latent spaces.
2. Embedding neighborhoods with low local rank.
3. Diffusion trajectories following learned score geometry.

Two non-examples clarify the boundary:

1. Uniform noise in every ambient direction.
2. A dataset whose classes occupy disconnected structures but are forced into one manifold.

Proof or verification habit for embedding manifolds and local linearization:

Evidence is empirical, not theorem-level: estimate local dimension, reconstruction error, neighborhood stability, and tangent consistency.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, embedding manifolds and local linearization matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

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

### 4.4 Symmetry and quotient spaces preview

Symmetry and quotient spaces preview belongs to the canonical scope of Manifolds. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: smooth manifolds, charts, atlases, tangent spaces, differentials, tangent bundles, embedded submanifolds, and ML manifold intuition. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$T_pM=\{\dot{\gamma}(0):\gamma(0)=p,\ \gamma \text{ smooth curve in }M\}.$$

**Operational definition.**

Symmetry and quotient spaces preview belongs to the canonical scope of Manifolds: smooth manifolds, charts, atlases, tangent spaces, differentials, tangent bundles, embedded submanifolds, and ML manifold intuition.

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

Three examples of symmetry and quotient spaces preview:

1. Sphere geometry.
2. Embedding-space local coordinates.
3. Matrix-manifold parameter constraints.

Two non-examples clarify the boundary:

1. A flat Euclidean approximation used globally.
2. A geometric claim made without metric or tangent space.

Proof or verification habit for symmetry and quotient spaces preview:

The proof habit is to compute locally and verify coordinate-independent meaning.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, symmetry and quotient spaces preview matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

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

### 4.5 Manifold learning diagnostics

Manifold learning diagnostics belongs to the canonical scope of Manifolds. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: smooth manifolds, charts, atlases, tangent spaces, differentials, tangent bundles, embedded submanifolds, and ML manifold intuition. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$dF_p(\mathbf{v})=\frac{d}{dt}\bigg|_{t=0}F(\gamma(t)),\qquad \dot{\gamma}(0)=\mathbf{v}.$$

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

Three examples of manifold learning diagnostics:

1. Autoencoder latent spaces.
2. Embedding neighborhoods with low local rank.
3. Diffusion trajectories following learned score geometry.

Two non-examples clarify the boundary:

1. Uniform noise in every ambient direction.
2. A dataset whose classes occupy disconnected structures but are forced into one manifold.

Proof or verification habit for manifold learning diagnostics:

Evidence is empirical, not theorem-level: estimate local dimension, reconstruction error, neighborhood stability, and tangent consistency.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, manifold learning diagnostics matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

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

1. (*) Build two overlapping charts for $S^1$ and write the transition map on the overlap.
   - (a) State the manifold and local representation.
   - (b) Identify the tangent space, metric, path, or retraction involved.
   - (c) Compute the finite or low-dimensional example.
   - (d) Interpret the result for an ML, LLM, or representation-learning setting.

2. (*) For the sphere $S^2$, compute the tangent constraint at a point $\mathbf{x}$ using $h(\mathbf{x})=\mathbf{x}^\top\mathbf{x}-1$.
   - (a) State the manifold and local representation.
   - (b) Identify the tangent space, metric, path, or retraction involved.
   - (c) Compute the finite or low-dimensional example.
   - (d) Interpret the result for an ML, LLM, or representation-learning setting.

3. (*) Given a smooth map $F:M\to N$, describe how a curve-based tangent vector is pushed forward by $dF_p$.
   - (a) State the manifold and local representation.
   - (b) Identify the tangent space, metric, path, or retraction involved.
   - (c) Compute the finite or low-dimensional example.
   - (d) Interpret the result for an ML, LLM, or representation-learning setting.

4. (**) Explain why a single latitude-longitude coordinate chart cannot cover the entire sphere smoothly.
   - (a) State the manifold and local representation.
   - (b) Identify the tangent space, metric, path, or retraction involved.
   - (c) Compute the finite or low-dimensional example.
   - (d) Interpret the result for an ML, LLM, or representation-learning setting.

5. (**) Compare an embedded submanifold and an immersed submanifold using one concrete example of each.
   - (a) State the manifold and local representation.
   - (b) Identify the tangent space, metric, path, or retraction involved.
   - (c) Compute the finite or low-dimensional example.
   - (d) Interpret the result for an ML, LLM, or representation-learning setting.

6. (**) Diagnose whether a synthetic point cloud is plausibly one-dimensional, two-dimensional, or mixed-dimensional.
   - (a) State the manifold and local representation.
   - (b) Identify the tangent space, metric, path, or retraction involved.
   - (c) Compute the finite or low-dimensional example.
   - (d) Interpret the result for an ML, LLM, or representation-learning setting.

7. (***) Explain what can go wrong when a latent space is treated as globally Euclidean after a nonlinear decoder.
   - (a) State the manifold and local representation.
   - (b) Identify the tangent space, metric, path, or retraction involved.
   - (c) Compute the finite or low-dimensional example.
   - (d) Interpret the result for an ML, LLM, or representation-learning setting.

8. (***) Write the tangent bundle $TM$ for a simple manifold and interpret a vector field as a section.
   - (a) State the manifold and local representation.
   - (b) Identify the tangent space, metric, path, or retraction involved.
   - (c) Compute the finite or low-dimensional example.
   - (d) Interpret the result for an ML, LLM, or representation-learning setting.

9. (***) Identify a symmetry in an ML representation and explain why it suggests a quotient-space viewpoint.
   - (a) State the manifold and local representation.
   - (b) Identify the tangent space, metric, path, or retraction involved.
   - (c) Compute the finite or low-dimensional example.
   - (d) Interpret the result for an ML, LLM, or representation-learning setting.

10. (***) Summarize how charts, tangent spaces, and differentials prepare the ground for Riemannian metrics.
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

Manifolds follows measure theory because probability and density statements become most useful in AI once they live on structured spaces. Chapter 24 made distributions rigorous. Chapter 25 asks what happens when the spaces that carry data, parameters, or distributions are curved.

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

- MIT OCW. 18.950 Differential Geometry. https://ocw.mit.edu/courses/18-950-differential-geometry-fall-2008/
- Lee. Introduction to Smooth Manifolds. https://math.berkeley.edu/~jchaidez/materials/reu/lee_smooth_manifolds.pdf
- Boumal. An Introduction to Optimization on Smooth Manifolds. https://www.nicolasboumal.net/book/IntroOptimManifolds_Boumal_2023.pdf
- Bengio, Courville, Vincent. Representation Learning: A Review and New Perspectives. https://www.cs.columbia.edu/~blei/fogm/2020F/readings/BengioCourvilleVincent2013.pdf
