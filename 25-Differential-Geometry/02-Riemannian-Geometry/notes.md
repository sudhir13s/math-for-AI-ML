[Back to Curriculum](../../README.md) | [Previous: Manifolds](../01-Manifolds/notes.md) | [Next: Geodesics](../03-Geodesics/notes.md)

---

# Riemannian Geometry

> _"A Riemannian metric tells each tangent space how to measure length, angle, and steepest descent."_

## Overview

Riemannian geometry adds smoothly varying inner products to manifolds, making gradients, distances, curvature, and natural-gradient learning coordinate-aware.

Differential geometry is the part of mathematics that makes calculus work on curved spaces. Earlier chapters treated vectors, matrices, probabilities, optimization, and measures mostly in flat coordinate systems. This chapter explains what changes when the object being modeled is a sphere, a space of subspaces, a positive-definite covariance matrix, a statistical model, or a learned latent manifold.

This section uses LaTeX Markdown throughout. Inline mathematics uses `$...$`, and display mathematics uses `$$...$$`. The AI focus is practical: local coordinates, tangent spaces, metrics, geodesics, retractions, natural gradients, and matrix-manifold constraints.

## Prerequisites

- [Manifolds](../01-Manifolds/notes.md)
- [Hilbert Spaces](../../12-Functional-Analysis/02-Hilbert-Spaces/notes.md)
- [Fisher Information](../../09-Information-Theory/05-Fisher-Information/notes.md)
- [Second-Order Methods](../../08-Optimization/03-Second-Order-Methods/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Executable demonstrations for riemannian geometry |
| [exercises.ipynb](exercises.ipynb) | Graded practice for riemannian geometry |

## Learning Objectives

After completing this section, you will be able to:

- Define Riemannian metrics and Riemannian manifolds
- Compute curve length from a metric
- Distinguish Euclidean gradients from Riemannian gradients
- Use the metric tensor to raise and lower gradient representations
- Explain covariant derivative and Levi-Civita connection intuition
- Preview curvature without overloading the first course
- Connect Fisher information to natural-gradient learning
- Describe hyperbolic and SPD geometry in ML settings
- Recognize coordinate-dependence as a geometry problem
- Prepare for geodesics as metric-respecting paths

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 Adding inner products to tangent spaces](#11-adding-inner-products-to-tangent-spaces)
  - [1.2 Length angle and distance on curved spaces](#12-length-angle-and-distance-on-curved-spaces)
  - [1.3 Why Euclidean gradients are coordinate-dependent](#13-why-euclidean-gradients-are-coordinatedependent)
  - [1.4 Curvature as changing geometry](#14-curvature-as-changing-geometry)
  - [1.5 Information geometry and natural gradients](#15-information-geometry-and-natural-gradients)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Riemannian metric $g_p(\cdot,\cdot)$](#21-riemannian-metric)
  - [2.2 Riemannian manifold $(M,g)$](#22-riemannian-manifold)
  - [2.3 Length of curves and induced distance](#23-length-of-curves-and-induced-distance)
  - [2.4 Riemannian gradient $\operatorname{grad} f$](#24-riemannian-gradient)
  - [2.5 Volume forms and integration preview](#25-volume-forms-and-integration-preview)
- [3. Core Theory](#3-core-theory)
  - [3.1 Metric tensor in coordinates](#31-metric-tensor-in-coordinates)
  - [3.2 Musical isomorphisms and gradient representation](#32-musical-isomorphisms-and-gradient-representation)
  - [3.3 Levi-Civita connection preview](#33-levicivita-connection-preview)
  - [3.4 Covariant derivative intuition](#34-covariant-derivative-intuition)
  - [3.5 Curvature: sectional Ricci scalar preview](#35-curvature-sectional-ricci-scalar-preview)
- [4. AI Applications](#4-ai-applications)
  - [4.1 Fisher metric and natural gradient](#41-fisher-metric-and-natural-gradient)
  - [4.2 Hyperbolic embeddings for hierarchies](#42-hyperbolic-embeddings-for-hierarchies)
  - [4.3 SPD covariance manifolds](#43-spd-covariance-manifolds)
  - [4.4 Wasserstein and information-geometric intuition](#44-wasserstein-and-informationgeometric-intuition)
  - [4.5 Geometry-aware regularization](#45-geometryaware-regularization)
- [5. Common Mistakes](#5-common-mistakes)
- [6. Exercises](#6-exercises)
- [7. Why This Matters for AI](#7-why-this-matters-for-ai)
- [8. Conceptual Bridge](#8-conceptual-bridge)
- [References](#references)

---

## 1. Intuition

Intuition develops the part of riemannian geometry specified by the approved Chapter 25 table of contents. The treatment is geometry-first and AI-facing.

### 1.1 Adding inner products to tangent spaces

Adding inner products to tangent spaces belongs to the canonical scope of Riemannian Geometry. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: Riemannian metrics, curve length, induced distance, Riemannian gradients, metric tensors, connections, curvature previews, and information geometry. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$g_p:T_pM\times T_pM\to \mathbb{R},\qquad g_p(\mathbf{u},\mathbf{v})=\langle \mathbf{u},\mathbf{v}\rangle_p.$$

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

Three examples of adding inner products to tangent spaces:

1. Velocity of a curve on a sphere.
2. Jacobian pushing embedding perturbations forward.
3. A vector field assigning one tangent direction per point.

Two non-examples clarify the boundary:

1. An arbitrary ambient vector not tangent to the constraint.
2. A finite difference step that leaves the manifold without retraction.

Proof or verification habit for adding inner products to tangent spaces:

For embedded manifolds, differentiate the constraint; for abstract manifolds, use curves or derivations.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, adding inner products to tangent spaces matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

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

### 1.2 Length angle and distance on curved spaces

Length angle and distance on curved spaces belongs to the canonical scope of Riemannian Geometry. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: Riemannian metrics, curve length, induced distance, Riemannian gradients, metric tensors, connections, curvature previews, and information geometry. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$L(\gamma)=\int_a^b \sqrt{g_{\gamma(t)}(\dot{\gamma}(t),\dot{\gamma}(t))}\,dt.$$

**Operational definition.**

A Riemannian metric assigns an inner product to every tangent space smoothly.

**Worked reading.**

If a coordinate metric is $G(\mathbf{x})$, then length of a velocity $\dot{\mathbf{x}}$ is $\sqrt{\dot{\mathbf{x}}^\top G(\mathbf{x})\dot{\mathbf{x}}}$.

| Geometric object | Meaning | AI interpretation |
| --- | --- | --- |
| Manifold $M$ | Curved space with local coordinates | Data manifold, latent space, constraint set, parameter space |
| Chart $\varphi$ | Local coordinate map | Local representation or embedding coordinates |
| Tangent space $T_pM$ | Linearized directions at $p$ | Local perturbations, gradients, velocities |
| Metric $g_p$ | Inner product on $T_pM$ | Geometry-aware length, angle, steepest descent |
| Geodesic | Straightest curved-space path | Latent interpolation, shortest motion, curved optimization path |
| Retraction | Practical map from tangent step back to $M$ | Efficient constrained update in training loops |

Three examples of length angle and distance on curved spaces:

1. Euclidean metric on a sphere inherited from ambient space.
2. Fisher metric on statistical models.
3. Affine-invariant metric on SPD matrices.

Two non-examples clarify the boundary:

1. A distance formula with no tangent-space inner product.
2. A fixed Euclidean metric used after nonlinear reparameterization without checking geometry.

Proof or verification habit for length angle and distance on curved spaces:

Check symmetry, bilinearity, positive definiteness, and smooth variation with the base point.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, length angle and distance on curved spaces matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

The metric determines what steepest descent, distance, and regularization mean for a representation or parameter space.

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

Local diagnostic: State the metric before computing lengths or gradients.

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

### 1.3 Why Euclidean gradients are coordinate-dependent

Why Euclidean gradients are coordinate-dependent belongs to the canonical scope of Riemannian Geometry. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: Riemannian metrics, curve length, induced distance, Riemannian gradients, metric tensors, connections, curvature previews, and information geometry. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$g_p(\operatorname{grad} f,\mathbf{v})=df_p[\mathbf{v}]\quad \forall \mathbf{v}\in T_pM.$$

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

Three examples of why euclidean gradients are coordinate-dependent:

1. Natural gradient using Fisher information.
2. Projected gradient on the sphere.
3. Geometry-aware update for SPD covariance matrices.

Two non-examples clarify the boundary:

1. Raw parameter gradient treated as invariant under reparameterization.
2. A direction off the tangent space called a manifold gradient.

Proof or verification habit for why euclidean gradients are coordinate-dependent:

Use the defining identity $g_p(\operatorname{grad} f,\mathbf{v})=df_p[\mathbf{v}]$ for all tangent directions.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, why euclidean gradients are coordinate-dependent matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

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

### 1.4 Curvature as changing geometry

Curvature as changing geometry belongs to the canonical scope of Riemannian Geometry. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: Riemannian metrics, curve length, induced distance, Riemannian gradients, metric tensors, connections, curvature previews, and information geometry. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$\operatorname{grad} f=G^{-1}\nabla_{\mathbf{x}} f\quad\text{in local coordinates with metric matrix }G.$$

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

Three examples of curvature as changing geometry:

1. Levi-Civita connection.
2. Covariant derivative of a velocity field.
3. Curvature affecting geodesic spread.

Two non-examples clarify the boundary:

1. Ordinary derivative of a tangent vector that leaves the tangent space.
2. Curvature treated as only a visualization artifact.

Proof or verification habit for curvature as changing geometry:

For a first course, focus on compatibility and projection intuition; full curvature tensors are preview material here.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, curvature as changing geometry matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

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

### 1.5 Information geometry and natural gradients

Information geometry and natural gradients belongs to the canonical scope of Riemannian Geometry. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: Riemannian metrics, curve length, induced distance, Riemannian gradients, metric tensors, connections, curvature previews, and information geometry. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$g_p:T_pM\times T_pM\to \mathbb{R},\qquad g_p(\mathbf{u},\mathbf{v})=\langle \mathbf{u},\mathbf{v}\rangle_p.$$

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

Three examples of information geometry and natural gradients:

1. Natural gradient using Fisher information.
2. Projected gradient on the sphere.
3. Geometry-aware update for SPD covariance matrices.

Two non-examples clarify the boundary:

1. Raw parameter gradient treated as invariant under reparameterization.
2. A direction off the tangent space called a manifold gradient.

Proof or verification habit for information geometry and natural gradients:

Use the defining identity $g_p(\operatorname{grad} f,\mathbf{v})=df_p[\mathbf{v}]$ for all tangent directions.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, information geometry and natural gradients matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

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

Formal Definitions develops the part of riemannian geometry specified by the approved Chapter 25 table of contents. The treatment is geometry-first and AI-facing.

### 2.1 Riemannian metric $g_p(\cdot,\cdot)$

Riemannian metric $g_p(\cdot,\cdot)$ belongs to the canonical scope of Riemannian Geometry. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: Riemannian metrics, curve length, induced distance, Riemannian gradients, metric tensors, connections, curvature previews, and information geometry. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$L(\gamma)=\int_a^b \sqrt{g_{\gamma(t)}(\dot{\gamma}(t),\dot{\gamma}(t))}\,dt.$$

**Operational definition.**

A Riemannian metric assigns an inner product to every tangent space smoothly.

**Worked reading.**

If a coordinate metric is $G(\mathbf{x})$, then length of a velocity $\dot{\mathbf{x}}$ is $\sqrt{\dot{\mathbf{x}}^\top G(\mathbf{x})\dot{\mathbf{x}}}$.

| Geometric object | Meaning | AI interpretation |
| --- | --- | --- |
| Manifold $M$ | Curved space with local coordinates | Data manifold, latent space, constraint set, parameter space |
| Chart $\varphi$ | Local coordinate map | Local representation or embedding coordinates |
| Tangent space $T_pM$ | Linearized directions at $p$ | Local perturbations, gradients, velocities |
| Metric $g_p$ | Inner product on $T_pM$ | Geometry-aware length, angle, steepest descent |
| Geodesic | Straightest curved-space path | Latent interpolation, shortest motion, curved optimization path |
| Retraction | Practical map from tangent step back to $M$ | Efficient constrained update in training loops |

Three examples of riemannian metric $g_p(\cdot,\cdot)$:

1. Euclidean metric on a sphere inherited from ambient space.
2. Fisher metric on statistical models.
3. Affine-invariant metric on SPD matrices.

Two non-examples clarify the boundary:

1. A distance formula with no tangent-space inner product.
2. A fixed Euclidean metric used after nonlinear reparameterization without checking geometry.

Proof or verification habit for riemannian metric $g_p(\cdot,\cdot)$:

Check symmetry, bilinearity, positive definiteness, and smooth variation with the base point.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, riemannian metric $g_p(\cdot,\cdot)$ matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

The metric determines what steepest descent, distance, and regularization mean for a representation or parameter space.

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

Local diagnostic: State the metric before computing lengths or gradients.

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

### 2.2 Riemannian manifold $(M,g)$

Riemannian manifold $(M,g)$ belongs to the canonical scope of Riemannian Geometry. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: Riemannian metrics, curve length, induced distance, Riemannian gradients, metric tensors, connections, curvature previews, and information geometry. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$g_p(\operatorname{grad} f,\mathbf{v})=df_p[\mathbf{v}]\quad \forall \mathbf{v}\in T_pM.$$

**Operational definition.**

Riemannian manifold $(M,g)$ belongs to the canonical scope of Riemannian Geometry: Riemannian metrics, curve length, induced distance, Riemannian gradients, metric tensors, connections, curvature previews, and information geometry.

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

Three examples of riemannian manifold $(m,g)$:

1. Sphere geometry.
2. Embedding-space local coordinates.
3. Matrix-manifold parameter constraints.

Two non-examples clarify the boundary:

1. A flat Euclidean approximation used globally.
2. A geometric claim made without metric or tangent space.

Proof or verification habit for riemannian manifold $(m,g)$:

The proof habit is to compute locally and verify coordinate-independent meaning.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, riemannian manifold $(m,g)$ matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

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

### 2.3 Length of curves and induced distance

Length of curves and induced distance belongs to the canonical scope of Riemannian Geometry. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: Riemannian metrics, curve length, induced distance, Riemannian gradients, metric tensors, connections, curvature previews, and information geometry. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$\operatorname{grad} f=G^{-1}\nabla_{\mathbf{x}} f\quad\text{in local coordinates with metric matrix }G.$$

**Operational definition.**

A Riemannian metric assigns an inner product to every tangent space smoothly.

**Worked reading.**

If a coordinate metric is $G(\mathbf{x})$, then length of a velocity $\dot{\mathbf{x}}$ is $\sqrt{\dot{\mathbf{x}}^\top G(\mathbf{x})\dot{\mathbf{x}}}$.

| Geometric object | Meaning | AI interpretation |
| --- | --- | --- |
| Manifold $M$ | Curved space with local coordinates | Data manifold, latent space, constraint set, parameter space |
| Chart $\varphi$ | Local coordinate map | Local representation or embedding coordinates |
| Tangent space $T_pM$ | Linearized directions at $p$ | Local perturbations, gradients, velocities |
| Metric $g_p$ | Inner product on $T_pM$ | Geometry-aware length, angle, steepest descent |
| Geodesic | Straightest curved-space path | Latent interpolation, shortest motion, curved optimization path |
| Retraction | Practical map from tangent step back to $M$ | Efficient constrained update in training loops |

Three examples of length of curves and induced distance:

1. Euclidean metric on a sphere inherited from ambient space.
2. Fisher metric on statistical models.
3. Affine-invariant metric on SPD matrices.

Two non-examples clarify the boundary:

1. A distance formula with no tangent-space inner product.
2. A fixed Euclidean metric used after nonlinear reparameterization without checking geometry.

Proof or verification habit for length of curves and induced distance:

Check symmetry, bilinearity, positive definiteness, and smooth variation with the base point.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, length of curves and induced distance matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

The metric determines what steepest descent, distance, and regularization mean for a representation or parameter space.

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

Local diagnostic: State the metric before computing lengths or gradients.

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

### 2.4 Riemannian gradient $\operatorname{grad} f$

Riemannian gradient $\operatorname{grad} f$ belongs to the canonical scope of Riemannian Geometry. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: Riemannian metrics, curve length, induced distance, Riemannian gradients, metric tensors, connections, curvature previews, and information geometry. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$g_p:T_pM\times T_pM\to \mathbb{R},\qquad g_p(\mathbf{u},\mathbf{v})=\langle \mathbf{u},\mathbf{v}\rangle_p.$$

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

Three examples of riemannian gradient $\operatorname{grad} f$:

1. Natural gradient using Fisher information.
2. Projected gradient on the sphere.
3. Geometry-aware update for SPD covariance matrices.

Two non-examples clarify the boundary:

1. Raw parameter gradient treated as invariant under reparameterization.
2. A direction off the tangent space called a manifold gradient.

Proof or verification habit for riemannian gradient $\operatorname{grad} f$:

Use the defining identity $g_p(\operatorname{grad} f,\mathbf{v})=df_p[\mathbf{v}]$ for all tangent directions.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, riemannian gradient $\operatorname{grad} f$ matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

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

### 2.5 Volume forms and integration preview

Volume forms and integration preview belongs to the canonical scope of Riemannian Geometry. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: Riemannian metrics, curve length, induced distance, Riemannian gradients, metric tensors, connections, curvature previews, and information geometry. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$L(\gamma)=\int_a^b \sqrt{g_{\gamma(t)}(\dot{\gamma}(t),\dot{\gamma}(t))}\,dt.$$

**Operational definition.**

A Riemannian metric assigns an inner product to every tangent space smoothly.

**Worked reading.**

If a coordinate metric is $G(\mathbf{x})$, then length of a velocity $\dot{\mathbf{x}}$ is $\sqrt{\dot{\mathbf{x}}^\top G(\mathbf{x})\dot{\mathbf{x}}}$.

| Geometric object | Meaning | AI interpretation |
| --- | --- | --- |
| Manifold $M$ | Curved space with local coordinates | Data manifold, latent space, constraint set, parameter space |
| Chart $\varphi$ | Local coordinate map | Local representation or embedding coordinates |
| Tangent space $T_pM$ | Linearized directions at $p$ | Local perturbations, gradients, velocities |
| Metric $g_p$ | Inner product on $T_pM$ | Geometry-aware length, angle, steepest descent |
| Geodesic | Straightest curved-space path | Latent interpolation, shortest motion, curved optimization path |
| Retraction | Practical map from tangent step back to $M$ | Efficient constrained update in training loops |

Three examples of volume forms and integration preview:

1. Euclidean metric on a sphere inherited from ambient space.
2. Fisher metric on statistical models.
3. Affine-invariant metric on SPD matrices.

Two non-examples clarify the boundary:

1. A distance formula with no tangent-space inner product.
2. A fixed Euclidean metric used after nonlinear reparameterization without checking geometry.

Proof or verification habit for volume forms and integration preview:

Check symmetry, bilinearity, positive definiteness, and smooth variation with the base point.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, volume forms and integration preview matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

The metric determines what steepest descent, distance, and regularization mean for a representation or parameter space.

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

Local diagnostic: State the metric before computing lengths or gradients.

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

Core Theory develops the part of riemannian geometry specified by the approved Chapter 25 table of contents. The treatment is geometry-first and AI-facing.

### 3.1 Metric tensor in coordinates

Metric tensor in coordinates belongs to the canonical scope of Riemannian Geometry. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: Riemannian metrics, curve length, induced distance, Riemannian gradients, metric tensors, connections, curvature previews, and information geometry. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$g_p(\operatorname{grad} f,\mathbf{v})=df_p[\mathbf{v}]\quad \forall \mathbf{v}\in T_pM.$$

**Operational definition.**

A Riemannian metric assigns an inner product to every tangent space smoothly.

**Worked reading.**

If a coordinate metric is $G(\mathbf{x})$, then length of a velocity $\dot{\mathbf{x}}$ is $\sqrt{\dot{\mathbf{x}}^\top G(\mathbf{x})\dot{\mathbf{x}}}$.

| Geometric object | Meaning | AI interpretation |
| --- | --- | --- |
| Manifold $M$ | Curved space with local coordinates | Data manifold, latent space, constraint set, parameter space |
| Chart $\varphi$ | Local coordinate map | Local representation or embedding coordinates |
| Tangent space $T_pM$ | Linearized directions at $p$ | Local perturbations, gradients, velocities |
| Metric $g_p$ | Inner product on $T_pM$ | Geometry-aware length, angle, steepest descent |
| Geodesic | Straightest curved-space path | Latent interpolation, shortest motion, curved optimization path |
| Retraction | Practical map from tangent step back to $M$ | Efficient constrained update in training loops |

Three examples of metric tensor in coordinates:

1. Euclidean metric on a sphere inherited from ambient space.
2. Fisher metric on statistical models.
3. Affine-invariant metric on SPD matrices.

Two non-examples clarify the boundary:

1. A distance formula with no tangent-space inner product.
2. A fixed Euclidean metric used after nonlinear reparameterization without checking geometry.

Proof or verification habit for metric tensor in coordinates:

Check symmetry, bilinearity, positive definiteness, and smooth variation with the base point.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, metric tensor in coordinates matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

The metric determines what steepest descent, distance, and regularization mean for a representation or parameter space.

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

Local diagnostic: State the metric before computing lengths or gradients.

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

### 3.2 Musical isomorphisms and gradient representation

Musical isomorphisms and gradient representation belongs to the canonical scope of Riemannian Geometry. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: Riemannian metrics, curve length, induced distance, Riemannian gradients, metric tensors, connections, curvature previews, and information geometry. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$\operatorname{grad} f=G^{-1}\nabla_{\mathbf{x}} f\quad\text{in local coordinates with metric matrix }G.$$

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

Three examples of musical isomorphisms and gradient representation:

1. Autoencoder latent spaces.
2. Embedding neighborhoods with low local rank.
3. Diffusion trajectories following learned score geometry.

Two non-examples clarify the boundary:

1. Uniform noise in every ambient direction.
2. A dataset whose classes occupy disconnected structures but are forced into one manifold.

Proof or verification habit for musical isomorphisms and gradient representation:

Evidence is empirical, not theorem-level: estimate local dimension, reconstruction error, neighborhood stability, and tangent consistency.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, musical isomorphisms and gradient representation matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

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

### 3.3 Levi-Civita connection preview

Levi-Civita connection preview belongs to the canonical scope of Riemannian Geometry. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: Riemannian metrics, curve length, induced distance, Riemannian gradients, metric tensors, connections, curvature previews, and information geometry. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$g_p:T_pM\times T_pM\to \mathbb{R},\qquad g_p(\mathbf{u},\mathbf{v})=\langle \mathbf{u},\mathbf{v}\rangle_p.$$

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

Three examples of levi-civita connection preview:

1. Levi-Civita connection.
2. Covariant derivative of a velocity field.
3. Curvature affecting geodesic spread.

Two non-examples clarify the boundary:

1. Ordinary derivative of a tangent vector that leaves the tangent space.
2. Curvature treated as only a visualization artifact.

Proof or verification habit for levi-civita connection preview:

For a first course, focus on compatibility and projection intuition; full curvature tensors are preview material here.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, levi-civita connection preview matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

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

### 3.4 Covariant derivative intuition

Covariant derivative intuition belongs to the canonical scope of Riemannian Geometry. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: Riemannian metrics, curve length, induced distance, Riemannian gradients, metric tensors, connections, curvature previews, and information geometry. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$L(\gamma)=\int_a^b \sqrt{g_{\gamma(t)}(\dot{\gamma}(t),\dot{\gamma}(t))}\,dt.$$

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

Three examples of covariant derivative intuition:

1. Levi-Civita connection.
2. Covariant derivative of a velocity field.
3. Curvature affecting geodesic spread.

Two non-examples clarify the boundary:

1. Ordinary derivative of a tangent vector that leaves the tangent space.
2. Curvature treated as only a visualization artifact.

Proof or verification habit for covariant derivative intuition:

For a first course, focus on compatibility and projection intuition; full curvature tensors are preview material here.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, covariant derivative intuition matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

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

### 3.5 Curvature: sectional Ricci scalar preview

Curvature: sectional Ricci scalar preview belongs to the canonical scope of Riemannian Geometry. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: Riemannian metrics, curve length, induced distance, Riemannian gradients, metric tensors, connections, curvature previews, and information geometry. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$g_p(\operatorname{grad} f,\mathbf{v})=df_p[\mathbf{v}]\quad \forall \mathbf{v}\in T_pM.$$

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

Three examples of curvature: sectional ricci scalar preview:

1. Levi-Civita connection.
2. Covariant derivative of a velocity field.
3. Curvature affecting geodesic spread.

Two non-examples clarify the boundary:

1. Ordinary derivative of a tangent vector that leaves the tangent space.
2. Curvature treated as only a visualization artifact.

Proof or verification habit for curvature: sectional ricci scalar preview:

For a first course, focus on compatibility and projection intuition; full curvature tensors are preview material here.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, curvature: sectional ricci scalar preview matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

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

## 4. AI Applications

AI Applications develops the part of riemannian geometry specified by the approved Chapter 25 table of contents. The treatment is geometry-first and AI-facing.

### 4.1 Fisher metric and natural gradient

Fisher metric and natural gradient belongs to the canonical scope of Riemannian Geometry. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: Riemannian metrics, curve length, induced distance, Riemannian gradients, metric tensors, connections, curvature previews, and information geometry. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$\operatorname{grad} f=G^{-1}\nabla_{\mathbf{x}} f\quad\text{in local coordinates with metric matrix }G.$$

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

Three examples of fisher metric and natural gradient:

1. Natural gradient using Fisher information.
2. Projected gradient on the sphere.
3. Geometry-aware update for SPD covariance matrices.

Two non-examples clarify the boundary:

1. Raw parameter gradient treated as invariant under reparameterization.
2. A direction off the tangent space called a manifold gradient.

Proof or verification habit for fisher metric and natural gradient:

Use the defining identity $g_p(\operatorname{grad} f,\mathbf{v})=df_p[\mathbf{v}]$ for all tangent directions.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, fisher metric and natural gradient matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

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

### 4.2 Hyperbolic embeddings for hierarchies

Hyperbolic embeddings for hierarchies belongs to the canonical scope of Riemannian Geometry. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: Riemannian metrics, curve length, induced distance, Riemannian gradients, metric tensors, connections, curvature previews, and information geometry. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$g_p:T_pM\times T_pM\to \mathbb{R},\qquad g_p(\mathbf{u},\mathbf{v})=\langle \mathbf{u},\mathbf{v}\rangle_p.$$

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

Three examples of hyperbolic embeddings for hierarchies:

1. Autoencoder latent spaces.
2. Embedding neighborhoods with low local rank.
3. Diffusion trajectories following learned score geometry.

Two non-examples clarify the boundary:

1. Uniform noise in every ambient direction.
2. A dataset whose classes occupy disconnected structures but are forced into one manifold.

Proof or verification habit for hyperbolic embeddings for hierarchies:

Evidence is empirical, not theorem-level: estimate local dimension, reconstruction error, neighborhood stability, and tangent consistency.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, hyperbolic embeddings for hierarchies matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

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

### 4.3 SPD covariance manifolds

SPD covariance manifolds belongs to the canonical scope of Riemannian Geometry. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: Riemannian metrics, curve length, induced distance, Riemannian gradients, metric tensors, connections, curvature previews, and information geometry. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$L(\gamma)=\int_a^b \sqrt{g_{\gamma(t)}(\dot{\gamma}(t),\dot{\gamma}(t))}\,dt.$$

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

Three examples of spd covariance manifolds:

1. PCA on Grassmann manifolds.
2. Orthogonal weights on Stiefel manifolds.
3. Covariance learning on SPD manifolds.

Two non-examples clarify the boundary:

1. Euclidean gradient descent followed by arbitrary clipping.
2. A projection step that destroys the first-order update direction.

Proof or verification habit for spd covariance manifolds:

Check tangent feasibility, descent direction under the metric, and retraction properties.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, spd covariance manifolds matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

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

### 4.4 Wasserstein and information-geometric intuition

Wasserstein and information-geometric intuition belongs to the canonical scope of Riemannian Geometry. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: Riemannian metrics, curve length, induced distance, Riemannian gradients, metric tensors, connections, curvature previews, and information geometry. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$g_p(\operatorname{grad} f,\mathbf{v})=df_p[\mathbf{v}]\quad \forall \mathbf{v}\in T_pM.$$

**Operational definition.**

Wasserstein and information-geometric intuition belongs to the canonical scope of Riemannian Geometry: Riemannian metrics, curve length, induced distance, Riemannian gradients, metric tensors, connections, curvature previews, and information geometry.

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

Three examples of wasserstein and information-geometric intuition:

1. Sphere geometry.
2. Embedding-space local coordinates.
3. Matrix-manifold parameter constraints.

Two non-examples clarify the boundary:

1. A flat Euclidean approximation used globally.
2. A geometric claim made without metric or tangent space.

Proof or verification habit for wasserstein and information-geometric intuition:

The proof habit is to compute locally and verify coordinate-independent meaning.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, wasserstein and information-geometric intuition matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

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

### 4.5 Geometry-aware regularization

Geometry-aware regularization belongs to the canonical scope of Riemannian Geometry. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: Riemannian metrics, curve length, induced distance, Riemannian gradients, metric tensors, connections, curvature previews, and information geometry. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$\operatorname{grad} f=G^{-1}\nabla_{\mathbf{x}} f\quad\text{in local coordinates with metric matrix }G.$$

**Operational definition.**

Geometry-aware regularization belongs to the canonical scope of Riemannian Geometry: Riemannian metrics, curve length, induced distance, Riemannian gradients, metric tensors, connections, curvature previews, and information geometry.

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

Three examples of geometry-aware regularization:

1. Sphere geometry.
2. Embedding-space local coordinates.
3. Matrix-manifold parameter constraints.

Two non-examples clarify the boundary:

1. A flat Euclidean approximation used globally.
2. A geometric claim made without metric or tangent space.

Proof or verification habit for geometry-aware regularization:

The proof habit is to compute locally and verify coordinate-independent meaning.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, geometry-aware regularization matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

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

1. (*) Given a coordinate metric matrix $G(x)$, compute the length of a short velocity vector.
   - (a) State the manifold and local representation.
   - (b) Identify the tangent space, metric, path, or retraction involved.
   - (c) Compute the finite or low-dimensional example.
   - (d) Interpret the result for an ML, LLM, or representation-learning setting.

2. (*) Show why the Riemannian gradient is $G^{-1}\nabla f$ in coordinates under a metric tensor $G$.
   - (a) State the manifold and local representation.
   - (b) Identify the tangent space, metric, path, or retraction involved.
   - (c) Compute the finite or low-dimensional example.
   - (d) Interpret the result for an ML, LLM, or representation-learning setting.

3. (*) Compare Euclidean distance and geodesic distance on a sphere for nearby and far-apart points.
   - (a) State the manifold and local representation.
   - (b) Identify the tangent space, metric, path, or retraction involved.
   - (c) Compute the finite or low-dimensional example.
   - (d) Interpret the result for an ML, LLM, or representation-learning setting.

4. (**) Explain how the Fisher information matrix defines a metric on a statistical model.
   - (a) State the manifold and local representation.
   - (b) Identify the tangent space, metric, path, or retraction involved.
   - (c) Compute the finite or low-dimensional example.
   - (d) Interpret the result for an ML, LLM, or representation-learning setting.

5. (**) Compute a projected gradient on the unit sphere and check that it is tangent.
   - (a) State the manifold and local representation.
   - (b) Identify the tangent space, metric, path, or retraction involved.
   - (c) Compute the finite or low-dimensional example.
   - (d) Interpret the result for an ML, LLM, or representation-learning setting.

6. (**) Describe how a change of parameterization can alter raw Euclidean gradients but not the intrinsic directional derivative.
   - (a) State the manifold and local representation.
   - (b) Identify the tangent space, metric, path, or retraction involved.
   - (c) Compute the finite or low-dimensional example.
   - (d) Interpret the result for an ML, LLM, or representation-learning setting.

7. (***) Explain, without heavy tensor algebra, what the Levi-Civita connection is designed to preserve.
   - (a) State the manifold and local representation.
   - (b) Identify the tangent space, metric, path, or retraction involved.
   - (c) Compute the finite or low-dimensional example.
   - (d) Interpret the result for an ML, LLM, or representation-learning setting.

8. (***) Give one ML example where volume, density, or integration on a curved space matters.
   - (a) State the manifold and local representation.
   - (b) Identify the tangent space, metric, path, or retraction involved.
   - (c) Compute the finite or low-dimensional example.
   - (d) Interpret the result for an ML, LLM, or representation-learning setting.

9. (***) Compare hyperbolic, spherical, and Euclidean metrics as choices for representation geometry.
   - (a) State the manifold and local representation.
   - (b) Identify the tangent space, metric, path, or retraction involved.
   - (c) Compute the finite or low-dimensional example.
   - (d) Interpret the result for an ML, LLM, or representation-learning setting.

10. (***) Summarize the metric decisions needed before claiming an optimizer is geometry-aware.
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

Riemannian Geometry follows measure theory because probability and density statements become most useful in AI once they live on structured spaces. Chapter 24 made distributions rigorous. Chapter 25 asks what happens when the spaces that carry data, parameters, or distributions are curved.

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
- Amari. Natural Gradient Works Efficiently in Learning. https://doi.org/10.1162/089976698300017746
- Boumal. An Introduction to Optimization on Smooth Manifolds. https://www.nicolasboumal.net/book/IntroOptimManifolds_Boumal_2023.pdf
