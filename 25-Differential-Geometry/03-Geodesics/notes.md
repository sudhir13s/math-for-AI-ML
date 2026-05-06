[Back to Curriculum](../../README.md) | [Previous: Riemannian Geometry](../02-Riemannian-Geometry/notes.md) | [Next: Optimization on Manifolds](../04-Optimization-on-Manifolds/notes.md)

---

# Geodesics

> _"A geodesic is what a straight line becomes when the space itself is curved."_

## Overview

Geodesics define geometry-aware motion, interpolation, distance, and convexity on curved spaces used by latent models, embeddings, robotics, and optimization.

Differential geometry is the part of mathematics that makes calculus work on curved spaces. Earlier chapters treated vectors, matrices, probabilities, optimization, and measures mostly in flat coordinate systems. This chapter explains what changes when the object being modeled is a sphere, a space of subspaces, a positive-definite covariance matrix, a statistical model, or a learned latent manifold.

This section uses LaTeX Markdown throughout. Inline mathematics uses `$...$`, and display mathematics uses `$$...$$`. The AI focus is practical: local coordinates, tangent spaces, metrics, geodesics, retractions, natural gradients, and matrix-manifold constraints.

## Prerequisites

- [Riemannian Geometry](../02-Riemannian-Geometry/notes.md)
- [Differential Equations and Dynamical Systems](../../06-Probability-Theory/06-Stochastic-Processes/notes.md)
- [Numerical Integration](../../10-Numerical-Methods/05-Numerical-Integration/notes.md)
- [Representation Learning](../../15-Math-for-LLMs/02-Embedding-Space-Math/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Executable demonstrations for geodesics |
| [exercises.ipynb](exercises.ipynb) | Graded practice for geodesics |

## Learning Objectives

After completing this section, you will be able to:

- Define curves, velocities, and geodesics on manifolds
- Explain straightest path and shortest path interpretations
- Write the geodesic equation using a connection
- Compute great-circle geodesics on the sphere
- Use exponential and logarithm maps in simple cases
- Explain geodesic distance and metric completeness
- Preview parallel transport and geodesic convexity
- Connect curved interpolation to latent representations
- Describe hyperbolic geodesics for hierarchical embeddings
- Prepare for manifold optimization updates and retractions

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 Straightest vs shortest paths](#11-straightest-vs-shortest-paths)
  - [1.2 Great circles on spheres](#12-great-circles-on-spheres)
  - [1.3 Why interpolation in latent space may be curved](#13-why-interpolation-in-latent-space-may-be-curved)
  - [1.4 Energy minimization and path length](#14-energy-minimization-and-path-length)
  - [1.5 Geodesics as geometry-aware motion](#15-geodesics-as-geometryaware-motion)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Curves $\gamma(t)$ on manifolds](#21-curves-on-manifolds)
  - [2.2 Velocity field $\dot{\gamma}(t)\in T_{\gamma(t)}M$](#22-velocity-field)
  - [2.3 Geodesic equation $\nabla_{\dot{\gamma}}\dot{\gamma}=0$](#23-geodesic-equation)
  - [2.4 Exponential map $\operatorname{Exp}_p(\mathbf{v})$](#24-exponential-map)
  - [2.5 Logarithm map and injectivity-radius preview](#25-logarithm-map-and-injectivityradius-preview)
- [3. Core Theory](#3-core-theory)
  - [3.1 Christoffel symbols in coordinates](#31-christoffel-symbols-in-coordinates)
  - [3.2 Geodesics on the sphere](#32-geodesics-on-the-sphere)
  - [3.3 Geodesic distance and metric completeness](#33-geodesic-distance-and-metric-completeness)
  - [3.4 Parallel transport preview](#34-parallel-transport-preview)
  - [3.5 Geodesic convexity preview](#35-geodesic-convexity-preview)
- [4. AI Applications](#4-ai-applications)
  - [4.1 Latent interpolation and representation paths](#41-latent-interpolation-and-representation-paths)
  - [4.2 Hyperbolic geodesics for tree-like data](#42-hyperbolic-geodesics-for-treelike-data)
  - [4.3 Shortest paths on learned manifolds](#43-shortest-paths-on-learned-manifolds)
  - [4.4 Geodesic loss functions](#44-geodesic-loss-functions)
  - [4.5 Trajectory planning and robotics](#45-trajectory-planning-and-robotics)
- [5. Common Mistakes](#5-common-mistakes)
- [6. Exercises](#6-exercises)
- [7. Why This Matters for AI](#7-why-this-matters-for-ai)
- [8. Conceptual Bridge](#8-conceptual-bridge)
- [References](#references)

---

## 1. Intuition

Intuition develops the part of geodesics specified by the approved Chapter 25 table of contents. The treatment is geometry-first and AI-facing.

### 1.1 Straightest vs shortest paths

Straightest vs shortest paths belongs to the canonical scope of Geodesics. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: curves, velocities, geodesic equation, exponential and logarithm maps, Christoffel symbols, sphere geodesics, parallel transport, and geodesic convexity preview. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$\dot{\gamma}(t)\in T_{\gamma(t)}M.$$

**Operational definition.**

A geodesic is a curve whose acceleration vanishes under the connection; locally, it is the curved-space analogue of a straight line.

**Worked reading.**

On a unit sphere, geodesics are great circles. The spherical interpolation formula stays on the sphere while linear interpolation cuts through the ambient ball.

| Geometric object | Meaning | AI interpretation |
| --- | --- | --- |
| Manifold $M$ | Curved space with local coordinates | Data manifold, latent space, constraint set, parameter space |
| Chart $\varphi$ | Local coordinate map | Local representation or embedding coordinates |
| Tangent space $T_pM$ | Linearized directions at $p$ | Local perturbations, gradients, velocities |
| Metric $g_p$ | Inner product on $T_pM$ | Geometry-aware length, angle, steepest descent |
| Geodesic | Straightest curved-space path | Latent interpolation, shortest motion, curved optimization path |
| Retraction | Practical map from tangent step back to $M$ | Efficient constrained update in training loops |

Three examples of straightest vs shortest paths:

1. Great-circle path between normalized embeddings.
2. Hyperbolic path through hierarchy embeddings.
3. Exponential-map step from a tangent vector.

Two non-examples clarify the boundary:

1. Ambient straight-line interpolation between two sphere points.
2. A shortest path across a discontinuous graph called a smooth geodesic.

Proof or verification habit for straightest vs shortest paths:

Check the geodesic equation or use known symmetry of the manifold to characterize the path.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, straightest vs shortest paths matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

Geodesics make latent interpolation, representation distances, and motion planning respect the actual geometry.

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

Local diagnostic: Verify the path stays on the manifold and has the right initial velocity.

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

### 1.2 Great circles on spheres

Great circles on spheres belongs to the canonical scope of Geodesics. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: curves, velocities, geodesic equation, exponential and logarithm maps, Christoffel symbols, sphere geodesics, parallel transport, and geodesic convexity preview. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$\nabla_{\dot{\gamma}}\dot{\gamma}=0.$$

**Operational definition.**

A geodesic is a curve whose acceleration vanishes under the connection; locally, it is the curved-space analogue of a straight line.

**Worked reading.**

On a unit sphere, geodesics are great circles. The spherical interpolation formula stays on the sphere while linear interpolation cuts through the ambient ball.

| Geometric object | Meaning | AI interpretation |
| --- | --- | --- |
| Manifold $M$ | Curved space with local coordinates | Data manifold, latent space, constraint set, parameter space |
| Chart $\varphi$ | Local coordinate map | Local representation or embedding coordinates |
| Tangent space $T_pM$ | Linearized directions at $p$ | Local perturbations, gradients, velocities |
| Metric $g_p$ | Inner product on $T_pM$ | Geometry-aware length, angle, steepest descent |
| Geodesic | Straightest curved-space path | Latent interpolation, shortest motion, curved optimization path |
| Retraction | Practical map from tangent step back to $M$ | Efficient constrained update in training loops |

Three examples of great circles on spheres:

1. Great-circle path between normalized embeddings.
2. Hyperbolic path through hierarchy embeddings.
3. Exponential-map step from a tangent vector.

Two non-examples clarify the boundary:

1. Ambient straight-line interpolation between two sphere points.
2. A shortest path across a discontinuous graph called a smooth geodesic.

Proof or verification habit for great circles on spheres:

Check the geodesic equation or use known symmetry of the manifold to characterize the path.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, great circles on spheres matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

Geodesics make latent interpolation, representation distances, and motion planning respect the actual geometry.

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

Local diagnostic: Verify the path stays on the manifold and has the right initial velocity.

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

### 1.3 Why interpolation in latent space may be curved

Why interpolation in latent space may be curved belongs to the canonical scope of Geodesics. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: curves, velocities, geodesic equation, exponential and logarithm maps, Christoffel symbols, sphere geodesics, parallel transport, and geodesic convexity preview. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$\frac{d^2 x^k}{dt^2}+\sum_{i,j}\Gamma^k_{ij}\frac{dx^i}{dt}\frac{dx^j}{dt}=0.$$

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

Three examples of why interpolation in latent space may be curved:

1. Autoencoder latent spaces.
2. Embedding neighborhoods with low local rank.
3. Diffusion trajectories following learned score geometry.

Two non-examples clarify the boundary:

1. Uniform noise in every ambient direction.
2. A dataset whose classes occupy disconnected structures but are forced into one manifold.

Proof or verification habit for why interpolation in latent space may be curved:

Evidence is empirical, not theorem-level: estimate local dimension, reconstruction error, neighborhood stability, and tangent consistency.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, why interpolation in latent space may be curved matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

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

### 1.4 Energy minimization and path length

Energy minimization and path length belongs to the canonical scope of Geodesics. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: curves, velocities, geodesic equation, exponential and logarithm maps, Christoffel symbols, sphere geodesics, parallel transport, and geodesic convexity preview. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$d_M(p,q)=\inf_{\gamma(0)=p,\gamma(1)=q} L(\gamma).$$

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

Three examples of energy minimization and path length:

1. Euclidean metric on a sphere inherited from ambient space.
2. Fisher metric on statistical models.
3. Affine-invariant metric on SPD matrices.

Two non-examples clarify the boundary:

1. A distance formula with no tangent-space inner product.
2. A fixed Euclidean metric used after nonlinear reparameterization without checking geometry.

Proof or verification habit for energy minimization and path length:

Check symmetry, bilinearity, positive definiteness, and smooth variation with the base point.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, energy minimization and path length matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

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

### 1.5 Geodesics as geometry-aware motion

Geodesics as geometry-aware motion belongs to the canonical scope of Geodesics. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: curves, velocities, geodesic equation, exponential and logarithm maps, Christoffel symbols, sphere geodesics, parallel transport, and geodesic convexity preview. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$\dot{\gamma}(t)\in T_{\gamma(t)}M.$$

**Operational definition.**

A geodesic is a curve whose acceleration vanishes under the connection; locally, it is the curved-space analogue of a straight line.

**Worked reading.**

On a unit sphere, geodesics are great circles. The spherical interpolation formula stays on the sphere while linear interpolation cuts through the ambient ball.

| Geometric object | Meaning | AI interpretation |
| --- | --- | --- |
| Manifold $M$ | Curved space with local coordinates | Data manifold, latent space, constraint set, parameter space |
| Chart $\varphi$ | Local coordinate map | Local representation or embedding coordinates |
| Tangent space $T_pM$ | Linearized directions at $p$ | Local perturbations, gradients, velocities |
| Metric $g_p$ | Inner product on $T_pM$ | Geometry-aware length, angle, steepest descent |
| Geodesic | Straightest curved-space path | Latent interpolation, shortest motion, curved optimization path |
| Retraction | Practical map from tangent step back to $M$ | Efficient constrained update in training loops |

Three examples of geodesics as geometry-aware motion:

1. Great-circle path between normalized embeddings.
2. Hyperbolic path through hierarchy embeddings.
3. Exponential-map step from a tangent vector.

Two non-examples clarify the boundary:

1. Ambient straight-line interpolation between two sphere points.
2. A shortest path across a discontinuous graph called a smooth geodesic.

Proof or verification habit for geodesics as geometry-aware motion:

Check the geodesic equation or use known symmetry of the manifold to characterize the path.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, geodesics as geometry-aware motion matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

Geodesics make latent interpolation, representation distances, and motion planning respect the actual geometry.

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

Local diagnostic: Verify the path stays on the manifold and has the right initial velocity.

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

Formal Definitions develops the part of geodesics specified by the approved Chapter 25 table of contents. The treatment is geometry-first and AI-facing.

### 2.1 Curves $\gamma(t)$ on manifolds

Curves $\gamma(t)$ on manifolds belongs to the canonical scope of Geodesics. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: curves, velocities, geodesic equation, exponential and logarithm maps, Christoffel symbols, sphere geodesics, parallel transport, and geodesic convexity preview. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$\nabla_{\dot{\gamma}}\dot{\gamma}=0.$$

**Operational definition.**

Curves $\gamma(t)$ on manifolds belongs to the canonical scope of Geodesics: curves, velocities, geodesic equation, exponential and logarithm maps, Christoffel symbols, sphere geodesics, parallel transport, and geodesic convexity preview.

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

Three examples of curves $\gamma(t)$ on manifolds:

1. Sphere geometry.
2. Embedding-space local coordinates.
3. Matrix-manifold parameter constraints.

Two non-examples clarify the boundary:

1. A flat Euclidean approximation used globally.
2. A geometric claim made without metric or tangent space.

Proof or verification habit for curves $\gamma(t)$ on manifolds:

The proof habit is to compute locally and verify coordinate-independent meaning.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, curves $\gamma(t)$ on manifolds matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

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

### 2.2 Velocity field $\dot{\gamma}(t)\in T_{\gamma(t)}M$

Velocity field $\dot{\gamma}(t)\in T_{\gamma(t)}M$ belongs to the canonical scope of Geodesics. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: curves, velocities, geodesic equation, exponential and logarithm maps, Christoffel symbols, sphere geodesics, parallel transport, and geodesic convexity preview. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$\frac{d^2 x^k}{dt^2}+\sum_{i,j}\Gamma^k_{ij}\frac{dx^i}{dt}\frac{dx^j}{dt}=0.$$

**Operational definition.**

Velocity field $\dot{\gamma}(t)\in T_{\gamma(t)}M$ belongs to the canonical scope of Geodesics: curves, velocities, geodesic equation, exponential and logarithm maps, Christoffel symbols, sphere geodesics, parallel transport, and geodesic convexity preview.

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

Three examples of velocity field $\dot{\gamma}(t)\in t_{\gamma(t)}m$:

1. Sphere geometry.
2. Embedding-space local coordinates.
3. Matrix-manifold parameter constraints.

Two non-examples clarify the boundary:

1. A flat Euclidean approximation used globally.
2. A geometric claim made without metric or tangent space.

Proof or verification habit for velocity field $\dot{\gamma}(t)\in t_{\gamma(t)}m$:

The proof habit is to compute locally and verify coordinate-independent meaning.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, velocity field $\dot{\gamma}(t)\in t_{\gamma(t)}m$ matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

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

### 2.3 Geodesic equation $\nabla_{\dot{\gamma}}\dot{\gamma}=0$

Geodesic equation $\nabla_{\dot{\gamma}}\dot{\gamma}=0$ belongs to the canonical scope of Geodesics. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: curves, velocities, geodesic equation, exponential and logarithm maps, Christoffel symbols, sphere geodesics, parallel transport, and geodesic convexity preview. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$d_M(p,q)=\inf_{\gamma(0)=p,\gamma(1)=q} L(\gamma).$$

**Operational definition.**

A geodesic is a curve whose acceleration vanishes under the connection; locally, it is the curved-space analogue of a straight line.

**Worked reading.**

On a unit sphere, geodesics are great circles. The spherical interpolation formula stays on the sphere while linear interpolation cuts through the ambient ball.

| Geometric object | Meaning | AI interpretation |
| --- | --- | --- |
| Manifold $M$ | Curved space with local coordinates | Data manifold, latent space, constraint set, parameter space |
| Chart $\varphi$ | Local coordinate map | Local representation or embedding coordinates |
| Tangent space $T_pM$ | Linearized directions at $p$ | Local perturbations, gradients, velocities |
| Metric $g_p$ | Inner product on $T_pM$ | Geometry-aware length, angle, steepest descent |
| Geodesic | Straightest curved-space path | Latent interpolation, shortest motion, curved optimization path |
| Retraction | Practical map from tangent step back to $M$ | Efficient constrained update in training loops |

Three examples of geodesic equation $\nabla_{\dot{\gamma}}\dot{\gamma}=0$:

1. Great-circle path between normalized embeddings.
2. Hyperbolic path through hierarchy embeddings.
3. Exponential-map step from a tangent vector.

Two non-examples clarify the boundary:

1. Ambient straight-line interpolation between two sphere points.
2. A shortest path across a discontinuous graph called a smooth geodesic.

Proof or verification habit for geodesic equation $\nabla_{\dot{\gamma}}\dot{\gamma}=0$:

Check the geodesic equation or use known symmetry of the manifold to characterize the path.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, geodesic equation $\nabla_{\dot{\gamma}}\dot{\gamma}=0$ matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

Geodesics make latent interpolation, representation distances, and motion planning respect the actual geometry.

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

Local diagnostic: Verify the path stays on the manifold and has the right initial velocity.

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

### 2.4 Exponential map $\operatorname{Exp}_p(\mathbf{v})$

Exponential map $\operatorname{Exp}_p(\mathbf{v})$ belongs to the canonical scope of Geodesics. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: curves, velocities, geodesic equation, exponential and logarithm maps, Christoffel symbols, sphere geodesics, parallel transport, and geodesic convexity preview. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$\dot{\gamma}(t)\in T_{\gamma(t)}M.$$

**Operational definition.**

A geodesic is a curve whose acceleration vanishes under the connection; locally, it is the curved-space analogue of a straight line.

**Worked reading.**

On a unit sphere, geodesics are great circles. The spherical interpolation formula stays on the sphere while linear interpolation cuts through the ambient ball.

| Geometric object | Meaning | AI interpretation |
| --- | --- | --- |
| Manifold $M$ | Curved space with local coordinates | Data manifold, latent space, constraint set, parameter space |
| Chart $\varphi$ | Local coordinate map | Local representation or embedding coordinates |
| Tangent space $T_pM$ | Linearized directions at $p$ | Local perturbations, gradients, velocities |
| Metric $g_p$ | Inner product on $T_pM$ | Geometry-aware length, angle, steepest descent |
| Geodesic | Straightest curved-space path | Latent interpolation, shortest motion, curved optimization path |
| Retraction | Practical map from tangent step back to $M$ | Efficient constrained update in training loops |

Three examples of exponential map $\operatorname{exp}_p(\mathbf{v})$:

1. Great-circle path between normalized embeddings.
2. Hyperbolic path through hierarchy embeddings.
3. Exponential-map step from a tangent vector.

Two non-examples clarify the boundary:

1. Ambient straight-line interpolation between two sphere points.
2. A shortest path across a discontinuous graph called a smooth geodesic.

Proof or verification habit for exponential map $\operatorname{exp}_p(\mathbf{v})$:

Check the geodesic equation or use known symmetry of the manifold to characterize the path.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, exponential map $\operatorname{exp}_p(\mathbf{v})$ matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

Geodesics make latent interpolation, representation distances, and motion planning respect the actual geometry.

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

Local diagnostic: Verify the path stays on the manifold and has the right initial velocity.

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

### 2.5 Logarithm map and injectivity-radius preview

Logarithm map and injectivity-radius preview belongs to the canonical scope of Geodesics. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: curves, velocities, geodesic equation, exponential and logarithm maps, Christoffel symbols, sphere geodesics, parallel transport, and geodesic convexity preview. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$\nabla_{\dot{\gamma}}\dot{\gamma}=0.$$

**Operational definition.**

A geodesic is a curve whose acceleration vanishes under the connection; locally, it is the curved-space analogue of a straight line.

**Worked reading.**

On a unit sphere, geodesics are great circles. The spherical interpolation formula stays on the sphere while linear interpolation cuts through the ambient ball.

| Geometric object | Meaning | AI interpretation |
| --- | --- | --- |
| Manifold $M$ | Curved space with local coordinates | Data manifold, latent space, constraint set, parameter space |
| Chart $\varphi$ | Local coordinate map | Local representation or embedding coordinates |
| Tangent space $T_pM$ | Linearized directions at $p$ | Local perturbations, gradients, velocities |
| Metric $g_p$ | Inner product on $T_pM$ | Geometry-aware length, angle, steepest descent |
| Geodesic | Straightest curved-space path | Latent interpolation, shortest motion, curved optimization path |
| Retraction | Practical map from tangent step back to $M$ | Efficient constrained update in training loops |

Three examples of logarithm map and injectivity-radius preview:

1. Great-circle path between normalized embeddings.
2. Hyperbolic path through hierarchy embeddings.
3. Exponential-map step from a tangent vector.

Two non-examples clarify the boundary:

1. Ambient straight-line interpolation between two sphere points.
2. A shortest path across a discontinuous graph called a smooth geodesic.

Proof or verification habit for logarithm map and injectivity-radius preview:

Check the geodesic equation or use known symmetry of the manifold to characterize the path.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, logarithm map and injectivity-radius preview matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

Geodesics make latent interpolation, representation distances, and motion planning respect the actual geometry.

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

Local diagnostic: Verify the path stays on the manifold and has the right initial velocity.

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

Core Theory develops the part of geodesics specified by the approved Chapter 25 table of contents. The treatment is geometry-first and AI-facing.

### 3.1 Christoffel symbols in coordinates

Christoffel symbols in coordinates belongs to the canonical scope of Geodesics. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: curves, velocities, geodesic equation, exponential and logarithm maps, Christoffel symbols, sphere geodesics, parallel transport, and geodesic convexity preview. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$\frac{d^2 x^k}{dt^2}+\sum_{i,j}\Gamma^k_{ij}\frac{dx^i}{dt}\frac{dx^j}{dt}=0.$$

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

Three examples of christoffel symbols in coordinates:

1. Natural gradient using Fisher information.
2. Projected gradient on the sphere.
3. Geometry-aware update for SPD covariance matrices.

Two non-examples clarify the boundary:

1. Raw parameter gradient treated as invariant under reparameterization.
2. A direction off the tangent space called a manifold gradient.

Proof or verification habit for christoffel symbols in coordinates:

Use the defining identity $g_p(\operatorname{grad} f,\mathbf{v})=df_p[\mathbf{v}]$ for all tangent directions.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, christoffel symbols in coordinates matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

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

### 3.2 Geodesics on the sphere

Geodesics on the sphere belongs to the canonical scope of Geodesics. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: curves, velocities, geodesic equation, exponential and logarithm maps, Christoffel symbols, sphere geodesics, parallel transport, and geodesic convexity preview. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$d_M(p,q)=\inf_{\gamma(0)=p,\gamma(1)=q} L(\gamma).$$

**Operational definition.**

A geodesic is a curve whose acceleration vanishes under the connection; locally, it is the curved-space analogue of a straight line.

**Worked reading.**

On a unit sphere, geodesics are great circles. The spherical interpolation formula stays on the sphere while linear interpolation cuts through the ambient ball.

| Geometric object | Meaning | AI interpretation |
| --- | --- | --- |
| Manifold $M$ | Curved space with local coordinates | Data manifold, latent space, constraint set, parameter space |
| Chart $\varphi$ | Local coordinate map | Local representation or embedding coordinates |
| Tangent space $T_pM$ | Linearized directions at $p$ | Local perturbations, gradients, velocities |
| Metric $g_p$ | Inner product on $T_pM$ | Geometry-aware length, angle, steepest descent |
| Geodesic | Straightest curved-space path | Latent interpolation, shortest motion, curved optimization path |
| Retraction | Practical map from tangent step back to $M$ | Efficient constrained update in training loops |

Three examples of geodesics on the sphere:

1. Great-circle path between normalized embeddings.
2. Hyperbolic path through hierarchy embeddings.
3. Exponential-map step from a tangent vector.

Two non-examples clarify the boundary:

1. Ambient straight-line interpolation between two sphere points.
2. A shortest path across a discontinuous graph called a smooth geodesic.

Proof or verification habit for geodesics on the sphere:

Check the geodesic equation or use known symmetry of the manifold to characterize the path.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, geodesics on the sphere matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

Geodesics make latent interpolation, representation distances, and motion planning respect the actual geometry.

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

Local diagnostic: Verify the path stays on the manifold and has the right initial velocity.

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

### 3.3 Geodesic distance and metric completeness

Geodesic distance and metric completeness belongs to the canonical scope of Geodesics. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: curves, velocities, geodesic equation, exponential and logarithm maps, Christoffel symbols, sphere geodesics, parallel transport, and geodesic convexity preview. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$\dot{\gamma}(t)\in T_{\gamma(t)}M.$$

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

Three examples of geodesic distance and metric completeness:

1. Euclidean metric on a sphere inherited from ambient space.
2. Fisher metric on statistical models.
3. Affine-invariant metric on SPD matrices.

Two non-examples clarify the boundary:

1. A distance formula with no tangent-space inner product.
2. A fixed Euclidean metric used after nonlinear reparameterization without checking geometry.

Proof or verification habit for geodesic distance and metric completeness:

Check symmetry, bilinearity, positive definiteness, and smooth variation with the base point.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, geodesic distance and metric completeness matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

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

### 3.4 Parallel transport preview

Parallel transport preview belongs to the canonical scope of Geodesics. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: curves, velocities, geodesic equation, exponential and logarithm maps, Christoffel symbols, sphere geodesics, parallel transport, and geodesic convexity preview. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$\nabla_{\dot{\gamma}}\dot{\gamma}=0.$$

**Operational definition.**

Energy, length, transport, and geodesic convexity are tools for comparing paths and moving tangent information across points.

**Worked reading.**

A tangent vector at one point cannot be subtracted from a tangent vector at another point without a transport rule.

| Geometric object | Meaning | AI interpretation |
| --- | --- | --- |
| Manifold $M$ | Curved space with local coordinates | Data manifold, latent space, constraint set, parameter space |
| Chart $\varphi$ | Local coordinate map | Local representation or embedding coordinates |
| Tangent space $T_pM$ | Linearized directions at $p$ | Local perturbations, gradients, velocities |
| Metric $g_p$ | Inner product on $T_pM$ | Geometry-aware length, angle, steepest descent |
| Geodesic | Straightest curved-space path | Latent interpolation, shortest motion, curved optimization path |
| Retraction | Practical map from tangent step back to $M$ | Efficient constrained update in training loops |

Three examples of parallel transport preview:

1. Transporting momentum in Riemannian optimization.
2. Geodesically convex SPD objectives.
3. Shortest-path loss for robotics trajectories.

Two non-examples clarify the boundary:

1. Subtracting tangent vectors at different points as if all tangent spaces were identical.
2. Euclidean convexity assumed on a curved domain.

Proof or verification habit for parallel transport preview:

The proof habit is to phrase inequalities along geodesics, not ambient line segments.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, parallel transport preview matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

This is where geometry changes training dynamics: momentum, convexity, and interpolation all need curved-space versions.

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

Local diagnostic: Ask whether the comparison happens inside one tangent space or across different tangent spaces.

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

### 3.5 Geodesic convexity preview

Geodesic convexity preview belongs to the canonical scope of Geodesics. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: curves, velocities, geodesic equation, exponential and logarithm maps, Christoffel symbols, sphere geodesics, parallel transport, and geodesic convexity preview. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$\frac{d^2 x^k}{dt^2}+\sum_{i,j}\Gamma^k_{ij}\frac{dx^i}{dt}\frac{dx^j}{dt}=0.$$

**Operational definition.**

A geodesic is a curve whose acceleration vanishes under the connection; locally, it is the curved-space analogue of a straight line.

**Worked reading.**

On a unit sphere, geodesics are great circles. The spherical interpolation formula stays on the sphere while linear interpolation cuts through the ambient ball.

| Geometric object | Meaning | AI interpretation |
| --- | --- | --- |
| Manifold $M$ | Curved space with local coordinates | Data manifold, latent space, constraint set, parameter space |
| Chart $\varphi$ | Local coordinate map | Local representation or embedding coordinates |
| Tangent space $T_pM$ | Linearized directions at $p$ | Local perturbations, gradients, velocities |
| Metric $g_p$ | Inner product on $T_pM$ | Geometry-aware length, angle, steepest descent |
| Geodesic | Straightest curved-space path | Latent interpolation, shortest motion, curved optimization path |
| Retraction | Practical map from tangent step back to $M$ | Efficient constrained update in training loops |

Three examples of geodesic convexity preview:

1. Great-circle path between normalized embeddings.
2. Hyperbolic path through hierarchy embeddings.
3. Exponential-map step from a tangent vector.

Two non-examples clarify the boundary:

1. Ambient straight-line interpolation between two sphere points.
2. A shortest path across a discontinuous graph called a smooth geodesic.

Proof or verification habit for geodesic convexity preview:

Check the geodesic equation or use known symmetry of the manifold to characterize the path.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, geodesic convexity preview matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

Geodesics make latent interpolation, representation distances, and motion planning respect the actual geometry.

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

Local diagnostic: Verify the path stays on the manifold and has the right initial velocity.

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

AI Applications develops the part of geodesics specified by the approved Chapter 25 table of contents. The treatment is geometry-first and AI-facing.

### 4.1 Latent interpolation and representation paths

Latent interpolation and representation paths belongs to the canonical scope of Geodesics. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: curves, velocities, geodesic equation, exponential and logarithm maps, Christoffel symbols, sphere geodesics, parallel transport, and geodesic convexity preview. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$d_M(p,q)=\inf_{\gamma(0)=p,\gamma(1)=q} L(\gamma).$$

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

Three examples of latent interpolation and representation paths:

1. Autoencoder latent spaces.
2. Embedding neighborhoods with low local rank.
3. Diffusion trajectories following learned score geometry.

Two non-examples clarify the boundary:

1. Uniform noise in every ambient direction.
2. A dataset whose classes occupy disconnected structures but are forced into one manifold.

Proof or verification habit for latent interpolation and representation paths:

Evidence is empirical, not theorem-level: estimate local dimension, reconstruction error, neighborhood stability, and tangent consistency.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, latent interpolation and representation paths matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

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

### 4.2 Hyperbolic geodesics for tree-like data

Hyperbolic geodesics for tree-like data belongs to the canonical scope of Geodesics. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: curves, velocities, geodesic equation, exponential and logarithm maps, Christoffel symbols, sphere geodesics, parallel transport, and geodesic convexity preview. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$\dot{\gamma}(t)\in T_{\gamma(t)}M.$$

**Operational definition.**

A geodesic is a curve whose acceleration vanishes under the connection; locally, it is the curved-space analogue of a straight line.

**Worked reading.**

On a unit sphere, geodesics are great circles. The spherical interpolation formula stays on the sphere while linear interpolation cuts through the ambient ball.

| Geometric object | Meaning | AI interpretation |
| --- | --- | --- |
| Manifold $M$ | Curved space with local coordinates | Data manifold, latent space, constraint set, parameter space |
| Chart $\varphi$ | Local coordinate map | Local representation or embedding coordinates |
| Tangent space $T_pM$ | Linearized directions at $p$ | Local perturbations, gradients, velocities |
| Metric $g_p$ | Inner product on $T_pM$ | Geometry-aware length, angle, steepest descent |
| Geodesic | Straightest curved-space path | Latent interpolation, shortest motion, curved optimization path |
| Retraction | Practical map from tangent step back to $M$ | Efficient constrained update in training loops |

Three examples of hyperbolic geodesics for tree-like data:

1. Great-circle path between normalized embeddings.
2. Hyperbolic path through hierarchy embeddings.
3. Exponential-map step from a tangent vector.

Two non-examples clarify the boundary:

1. Ambient straight-line interpolation between two sphere points.
2. A shortest path across a discontinuous graph called a smooth geodesic.

Proof or verification habit for hyperbolic geodesics for tree-like data:

Check the geodesic equation or use known symmetry of the manifold to characterize the path.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, hyperbolic geodesics for tree-like data matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

Geodesics make latent interpolation, representation distances, and motion planning respect the actual geometry.

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

Local diagnostic: Verify the path stays on the manifold and has the right initial velocity.

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

### 4.3 Shortest paths on learned manifolds

Shortest paths on learned manifolds belongs to the canonical scope of Geodesics. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: curves, velocities, geodesic equation, exponential and logarithm maps, Christoffel symbols, sphere geodesics, parallel transport, and geodesic convexity preview. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$\nabla_{\dot{\gamma}}\dot{\gamma}=0.$$

**Operational definition.**

A geodesic is a curve whose acceleration vanishes under the connection; locally, it is the curved-space analogue of a straight line.

**Worked reading.**

On a unit sphere, geodesics are great circles. The spherical interpolation formula stays on the sphere while linear interpolation cuts through the ambient ball.

| Geometric object | Meaning | AI interpretation |
| --- | --- | --- |
| Manifold $M$ | Curved space with local coordinates | Data manifold, latent space, constraint set, parameter space |
| Chart $\varphi$ | Local coordinate map | Local representation or embedding coordinates |
| Tangent space $T_pM$ | Linearized directions at $p$ | Local perturbations, gradients, velocities |
| Metric $g_p$ | Inner product on $T_pM$ | Geometry-aware length, angle, steepest descent |
| Geodesic | Straightest curved-space path | Latent interpolation, shortest motion, curved optimization path |
| Retraction | Practical map from tangent step back to $M$ | Efficient constrained update in training loops |

Three examples of shortest paths on learned manifolds:

1. Great-circle path between normalized embeddings.
2. Hyperbolic path through hierarchy embeddings.
3. Exponential-map step from a tangent vector.

Two non-examples clarify the boundary:

1. Ambient straight-line interpolation between two sphere points.
2. A shortest path across a discontinuous graph called a smooth geodesic.

Proof or verification habit for shortest paths on learned manifolds:

Check the geodesic equation or use known symmetry of the manifold to characterize the path.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, shortest paths on learned manifolds matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

Geodesics make latent interpolation, representation distances, and motion planning respect the actual geometry.

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

Local diagnostic: Verify the path stays on the manifold and has the right initial velocity.

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

### 4.4 Geodesic loss functions

Geodesic loss functions belongs to the canonical scope of Geodesics. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: curves, velocities, geodesic equation, exponential and logarithm maps, Christoffel symbols, sphere geodesics, parallel transport, and geodesic convexity preview. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$\frac{d^2 x^k}{dt^2}+\sum_{i,j}\Gamma^k_{ij}\frac{dx^i}{dt}\frac{dx^j}{dt}=0.$$

**Operational definition.**

A geodesic is a curve whose acceleration vanishes under the connection; locally, it is the curved-space analogue of a straight line.

**Worked reading.**

On a unit sphere, geodesics are great circles. The spherical interpolation formula stays on the sphere while linear interpolation cuts through the ambient ball.

| Geometric object | Meaning | AI interpretation |
| --- | --- | --- |
| Manifold $M$ | Curved space with local coordinates | Data manifold, latent space, constraint set, parameter space |
| Chart $\varphi$ | Local coordinate map | Local representation or embedding coordinates |
| Tangent space $T_pM$ | Linearized directions at $p$ | Local perturbations, gradients, velocities |
| Metric $g_p$ | Inner product on $T_pM$ | Geometry-aware length, angle, steepest descent |
| Geodesic | Straightest curved-space path | Latent interpolation, shortest motion, curved optimization path |
| Retraction | Practical map from tangent step back to $M$ | Efficient constrained update in training loops |

Three examples of geodesic loss functions:

1. Great-circle path between normalized embeddings.
2. Hyperbolic path through hierarchy embeddings.
3. Exponential-map step from a tangent vector.

Two non-examples clarify the boundary:

1. Ambient straight-line interpolation between two sphere points.
2. A shortest path across a discontinuous graph called a smooth geodesic.

Proof or verification habit for geodesic loss functions:

Check the geodesic equation or use known symmetry of the manifold to characterize the path.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, geodesic loss functions matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

Geodesics make latent interpolation, representation distances, and motion planning respect the actual geometry.

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

Local diagnostic: Verify the path stays on the manifold and has the right initial velocity.

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

### 4.5 Trajectory planning and robotics

Trajectory planning and robotics belongs to the canonical scope of Geodesics. The goal is to make curved-space reasoning concrete enough for ML practice without turning the section into a pure topology course.

Working scope for this subsection: curves, velocities, geodesic equation, exponential and logarithm maps, Christoffel symbols, sphere geodesics, parallel transport, and geodesic convexity preview. The recurring pattern is localize, linearize, measure, move, and return to the manifold.

$$d_M(p,q)=\inf_{\gamma(0)=p,\gamma(1)=q} L(\gamma).$$

**Operational definition.**

Energy, length, transport, and geodesic convexity are tools for comparing paths and moving tangent information across points.

**Worked reading.**

A tangent vector at one point cannot be subtracted from a tangent vector at another point without a transport rule.

| Geometric object | Meaning | AI interpretation |
| --- | --- | --- |
| Manifold $M$ | Curved space with local coordinates | Data manifold, latent space, constraint set, parameter space |
| Chart $\varphi$ | Local coordinate map | Local representation or embedding coordinates |
| Tangent space $T_pM$ | Linearized directions at $p$ | Local perturbations, gradients, velocities |
| Metric $g_p$ | Inner product on $T_pM$ | Geometry-aware length, angle, steepest descent |
| Geodesic | Straightest curved-space path | Latent interpolation, shortest motion, curved optimization path |
| Retraction | Practical map from tangent step back to $M$ | Efficient constrained update in training loops |

Three examples of trajectory planning and robotics:

1. Transporting momentum in Riemannian optimization.
2. Geodesically convex SPD objectives.
3. Shortest-path loss for robotics trajectories.

Two non-examples clarify the boundary:

1. Subtracting tangent vectors at different points as if all tangent spaces were identical.
2. Euclidean convexity assumed on a curved domain.

Proof or verification habit for trajectory planning and robotics:

The proof habit is to phrase inequalities along geodesics, not ambient line segments.

```text
global object      -> curved manifold or constraint set
local object       -> chart, tangent space, or coordinate patch
linear operation   -> derivative, gradient, velocity, Hessian approximation
geometric measure  -> metric, length, distance, curvature
algorithmic move   -> tangent step followed by geodesic or retraction
```

In AI systems, trajectory planning and robotics matters because learned representations and constrained parameter spaces are rarely globally flat. A local linear approximation may be useful, but it must be attached to the point where it is valid.

This is where geometry changes training dynamics: momentum, convexity, and interpolation all need curved-space versions.

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

Local diagnostic: Ask whether the comparison happens inside one tangent space or across different tangent spaces.

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

1. (*) Derive the great-circle interpolation formula for two non-antipodal points on $S^2$.
   - (a) State the manifold and local representation.
   - (b) Identify the tangent space, metric, path, or retraction involved.
   - (c) Compute the finite or low-dimensional example.
   - (d) Interpret the result for an ML, LLM, or representation-learning setting.

2. (*) Explain the difference between locally shortest, globally shortest, and straightest paths.
   - (a) State the manifold and local representation.
   - (b) Identify the tangent space, metric, path, or retraction involved.
   - (c) Compute the finite or low-dimensional example.
   - (d) Interpret the result for an ML, LLM, or representation-learning setting.

3. (*) Write the coordinate geodesic equation and identify the role of Christoffel symbols.
   - (a) State the manifold and local representation.
   - (b) Identify the tangent space, metric, path, or retraction involved.
   - (c) Compute the finite or low-dimensional example.
   - (d) Interpret the result for an ML, LLM, or representation-learning setting.

4. (**) Compute the exponential map on the unit sphere for a small tangent vector.
   - (a) State the manifold and local representation.
   - (b) Identify the tangent space, metric, path, or retraction involved.
   - (c) Compute the finite or low-dimensional example.
   - (d) Interpret the result for an ML, LLM, or representation-learning setting.

5. (**) Explain why the logarithm map can fail to be unique at or beyond a cut locus.
   - (a) State the manifold and local representation.
   - (b) Identify the tangent space, metric, path, or retraction involved.
   - (c) Compute the finite or low-dimensional example.
   - (d) Interpret the result for an ML, LLM, or representation-learning setting.

6. (**) Compare ambient linear interpolation and spherical geodesic interpolation for normalized embeddings.
   - (a) State the manifold and local representation.
   - (b) Identify the tangent space, metric, path, or retraction involved.
   - (c) Compute the finite or low-dimensional example.
   - (d) Interpret the result for an ML, LLM, or representation-learning setting.

7. (***) Describe a parallel-transport task and what quantity should remain constant along the path.
   - (a) State the manifold and local representation.
   - (b) Identify the tangent space, metric, path, or retraction involved.
   - (c) Compute the finite or low-dimensional example.
   - (d) Interpret the result for an ML, LLM, or representation-learning setting.

8. (***) Give an example where geodesic convexity is more appropriate than Euclidean convexity.
   - (a) State the manifold and local representation.
   - (b) Identify the tangent space, metric, path, or retraction involved.
   - (c) Compute the finite or low-dimensional example.
   - (d) Interpret the result for an ML, LLM, or representation-learning setting.

9. (***) Explain how hyperbolic geodesics support tree-like embedding geometry.
   - (a) State the manifold and local representation.
   - (b) Identify the tangent space, metric, path, or retraction involved.
   - (c) Compute the finite or low-dimensional example.
   - (d) Interpret the result for an ML, LLM, or representation-learning setting.

10. (***) Summarize how geodesics become the conceptual bridge to retractions and manifold optimization.
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

Geodesics follows measure theory because probability and density statements become most useful in AI once they live on structured spaces. Chapter 24 made distributions rigorous. Chapter 25 asks what happens when the spaces that carry data, parameters, or distributions are curved.

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
- Nickel and Kiela. Poincare Embeddings for Learning Hierarchical Representations. https://arxiv.org/abs/1705.08039
