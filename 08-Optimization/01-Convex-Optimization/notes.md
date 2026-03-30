[← Back to Optimization](../README.md) | [Next: Gradient Descent →](../02-Gradient-Descent/notes.md)

---

# Convex Optimization

> _"The great watershed in optimization isn't between linear and nonlinear — it's between convex and nonconvex."_ — R. Tyrrell Rockafellar

## Overview

Convex optimization is the mathematical foundation upon which all of modern optimization — and therefore all of machine learning training — rests. A convex optimization problem has a remarkable structural property: every local minimum is a global minimum, and efficient algorithms can find it in polynomial time. This stands in stark contrast to the general nonconvex problems encountered in deep learning, where the loss landscape is riddled with saddle points, local minima, and plateaus.

Why does a chapter on convex optimization open the optimization sequence in a curriculum about AI? Three reasons. First, the _language_ of convex optimization — feasible sets, duality, optimality conditions, condition numbers — is the language in which all optimization algorithms are analyzed, even when applied to nonconvex problems. Second, many core ML components _are_ convex: logistic regression, SVMs, ridge and lasso regression, and the cross-entropy loss as a function of logits. Third, convex relaxations provide principled approximations to intractable nonconvex problems — from nuclear-norm minimization in matrix completion to semidefinite relaxations in clustering.

This section develops convexity from first principles: convex sets (§2), convex functions (§3), the key regularity conditions of strong convexity and smoothness (§4), the taxonomy of convex programs (§5), optimality conditions (§6), and duality theory (§7). Every concept is tied to its role in modern AI systems.

**Scope note.** This section covers the mathematical theory of convexity and convex optimization problems. The _algorithms_ for solving these problems — gradient descent, Newton's method, interior-point methods — are covered in subsequent sections (§02–§03). The treatment of _constraints_ in depth — KKT conditions, penalty and barrier methods, projected gradient descent — belongs to [Constrained Optimization](../04-Constrained-Optimization/notes.md). Here we establish the foundations that those algorithmic sections build upon.

**What you will gain from this section.** By the end, you will have a precise vocabulary for optimization: you will know what it means for a problem to be "well-conditioned" ($\kappa$ is small), "tractable" (convex with known structure), or "relaxable" (nonconvex but approximable by a convex problem). You will understand why weight decay works (adds strong convexity), why learning rate matters (bounded by $1/L$), and why the kernel trick is possible (duality). These are not abstract concepts — they are the daily tools of anyone training or analyzing ML models.

**A note on computation.** Throughout this section, we focus on the _structure_ of convex problems rather than the _algorithms_ for solving them. The key insight is that recognizing structure (convexity, strong convexity, smoothness, duality) is more important than any particular algorithm — once you know the structure, the algorithm follows. A convex QP should be solved with a QP solver; a non-smooth convex problem needs proximal methods; a very large convex problem needs first-order methods. The structure dictates the approach.

> _"The great watershed in optimization isn't between problems we can solve and problems we can't — it's between problems where we can certify optimality and problems where we can't. Convexity provides that certificate."_ — Stephen Boyd

## Prerequisites

- **Gradients and Hessians** — $\nabla f$, $\nabla^2 f$, directional derivatives — [Chapter 5: Multivariate Calculus](../../05-Multivariate-Calculus/README.md)
- **Eigenvalues and positive definiteness** — $A \succ 0$ iff all eigenvalues positive — [Chapter 3 §01](../../03-Advanced-Linear-Algebra/01-Eigenvalues-and-Eigenvectors/notes.md) and [§07](../../03-Advanced-Linear-Algebra/07-Positive-Definite-Matrices/notes.md)
- **Norms** — $\lVert \mathbf{x} \rVert_1$, $\lVert \mathbf{x} \rVert_2$, $\lVert A \rVert_F$ — [Chapter 3 §06](../../03-Advanced-Linear-Algebra/06-Matrix-Norms/notes.md)
- **Taylor expansion** — first- and second-order approximations of multivariate functions — [Chapter 4](../../04-Calculus-Fundamentals/README.md)

## Companion Notebooks

| Notebook | Description |
|---|---|
| [theory.ipynb](theory.ipynb) | Interactive demonstrations of convexity, duality, and problem classes with visualizations |
| [exercises.ipynb](exercises.ipynb) | 8 graded exercises from verifying convexity to convex relaxations in ML |

## Learning Objectives

After completing this section, you will:

1. Define convex sets and convex functions and verify convexity using first- and second-order conditions
2. Prove that operations like intersection, nonnegative combination, and composition preserve convexity
3. Distinguish strong convexity from ordinary convexity and compute the condition number $\kappa = L/\mu$
4. Classify optimization problems into the LP → QP → SOCP → SDP hierarchy
5. State and apply first-order and second-order optimality conditions for unconstrained convex problems
6. Formulate the Lagrangian dual and apply weak and strong duality
7. Verify Slater's condition and explain when strong duality holds
8. Identify which ML loss functions are convex and explain why this matters for training
9. Explain why deep learning is nonconvex and what convex structure survives
10. Apply convex relaxation techniques (nuclear norm, SDP) to ML problems

---

## Table of Contents

- [Convex Optimization](#convex-optimization)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
  - [Companion Notebooks](#companion-notebooks)
  - [Learning Objectives](#learning-objectives)
  - [Table of Contents](#table-of-contents)
  - [1. Intuition](#1-intuition)
    - [1.1 What Is Convex Optimization?](#11-what-is-convex-optimization)
    - [1.2 Why Convexity Matters for AI](#12-why-convexity-matters-for-ai)
    - [1.3 Historical Timeline](#13-historical-timeline)
    - [1.4 The Landscape Metaphor](#14-the-landscape-metaphor)
  - [2. Convex Sets](#2-convex-sets)
    - [2.1 Definition and Examples](#21-definition-and-examples)
    - [2.2 Operations Preserving Convexity of Sets](#22-operations-preserving-convexity-of-sets)
    - [2.3 Supporting Hyperplanes and Separation Theorems](#23-supporting-hyperplanes-and-separation-theorems)
    - [2.4 Cones and the PSD Cone](#24-cones-and-the-psd-cone)
  - [3. Convex Functions](#3-convex-functions)
    - [3.1 Definition and First-Order Characterization](#31-definition-and-first-order-characterization)
    - [3.2 Second-Order Characterization](#32-second-order-characterization)
    - [3.3 Examples and Non-Examples](#33-examples-and-non-examples)
    - [3.4 Operations Preserving Convexity of Functions](#34-operations-preserving-convexity-of-functions)
    - [3.5 Epigraph and Sublevel Sets](#35-epigraph-and-sublevel-sets)
  - [4. Strong Convexity and Smoothness](#4-strong-convexity-and-smoothness)
    - [4.1 Strong Convexity](#41-strong-convexity)
    - [4.2 Smoothness (Lipschitz Gradient)](#42-smoothness-lipschitz-gradient)
    - [4.3 Condition Number](#43-condition-number)
    - [4.4 Implications for Convergence Rates](#44-implications-for-convergence-rates)
  - [5. Convex Optimization Problems](#5-convex-optimization-problems)
    - [5.1 Standard Form and Terminology](#51-standard-form-and-terminology)
    - [5.2 Linear Programming](#52-linear-programming)
    - [5.3 Quadratic Programming](#53-quadratic-programming)
    - [5.4 Second-Order Cone Programming](#54-second-order-cone-programming)
    - [5.5 Semidefinite Programming](#55-semidefinite-programming)
    - [5.6 The Hierarchy of Convex Programs](#56-the-hierarchy-of-convex-programs)
  - [6. Optimality Conditions](#6-optimality-conditions)
    - [6.1 Unconstrained Optimality](#61-unconstrained-optimality)
    - [6.2 Constrained Preview: Lagrangian and Dual Problem](#62-constrained-preview-lagrangian-and-dual-problem)
    - [6.3 Slater's Condition and Strong Duality](#63-slaters-condition-and-strong-duality)
  - [7. Duality](#7-duality)
    - [7.1 Lagrangian Dual Problem](#71-lagrangian-dual-problem)
    - [7.2 Weak and Strong Duality](#72-weak-and-strong-duality)
    - [7.3 Duality in ML](#73-duality-in-ml)
  - [8. Applications in Machine Learning](#8-applications-in-machine-learning)
    - [8.1 Convexity of Common Loss Functions](#81-convexity-of-common-loss-functions)
    - [8.2 Why Deep Learning Is Non-Convex (and What Survives)](#82-why-deep-learning-is-non-convex-and-what-survives)
    - [8.3 Convex Relaxations in Practice](#83-convex-relaxations-in-practice)
    - [8.4 LoRA, Weight Decay, and Convex Penalties](#84-lora-weight-decay-and-convex-penalties)
  - [9. Advanced Topics](#9-advanced-topics)
    - [9.1 Proximal Operators and Non-Smooth Convex Optimization](#91-proximal-operators-and-non-smooth-convex-optimization)
    - [9.2 Fenchel Conjugates](#92-fenchel-conjugates)
    - [9.3 Mirror Descent Preview](#93-mirror-descent-preview)
  - [10. Common Mistakes](#10-common-mistakes)
  - [Exercises](#exercises)
    - [Exercise 1: Verifying Convexity (★)](#exercise-1-verifying-convexity-)
    - [Exercise 2: Convex Set Operations (★)](#exercise-2-convex-set-operations-)
    - [Exercise 3: Computing Condition Numbers (★)](#exercise-3-computing-condition-numbers-)
    - [Exercise 4: Preservation Rules (★★)](#exercise-4-preservation-rules-)
    - [Exercise 5: Lagrangian Dual (★★)](#exercise-5-lagrangian-dual-)
    - [Exercise 6: Strong Convexity Analysis (★★)](#exercise-6-strong-convexity-analysis-)
    - [Exercise 7: Convex Relaxation for Matrix Completion (★★★)](#exercise-7-convex-relaxation-for-matrix-completion-)
    - [Exercise 8: Proximal Gradient for Lasso (★★★)](#exercise-8-proximal-gradient-for-lasso-)
  - [11. Why This Matters for AI](#11-why-this-matters-for-ai)
  - [12. Conceptual Bridge](#12-conceptual-bridge)
    - [Backward Connections](#backward-connections)
    - [Forward Connections](#forward-connections)
    - [The Big Picture](#the-big-picture)
  - [References](#references)

---

## 1. Intuition

### 1.1 What Is Convex Optimization?

At its core, a convex optimization problem asks: find the point that minimizes a convex function over a convex set. Written formally:

$$\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})$$

where $f$ is a convex function and $\mathcal{C}$ is a convex set. The defining property of this problem class is the **no-local-traps guarantee**: if you find any point where the function cannot be decreased by local moves, that point is the global minimum. No other class of optimization problems offers this guarantee.

To appreciate why this matters, consider the alternative. A general (nonconvex) optimization problem can have exponentially many local minima, saddle points, and plateaus. Finding the global minimum is NP-hard in the worst case. Convex optimization, by contrast, admits polynomial-time algorithms that find the global minimum to arbitrary precision.

```
THE CONVEX GUARANTEE
════════════════════════════════════════════════════════════════════════

  Nonconvex landscape:              Convex landscape:

       ╱╲    ╱╲                          ╲         ╱
      ╱  ╲  ╱  ╲   ╱╲                    ╲       ╱
     ╱    ╲╱    ╲ ╱  ╲                    ╲     ╱
    ╱              ╲   ╲                    ╲   ╱
   ╱                ╲                        ╲ ╱
                                              •  ← global min
   Multiple local     Saddle              UNIQUE minimum
   minima             points              (every local min = global)

════════════════════════════════════════════════════════════════════════
```

**The formal definition** combines two ingredients:

1. **Convex set** $\mathcal{C}$: for any two points $\mathbf{x}, \mathbf{y} \in \mathcal{C}$ and any $\theta \in [0, 1]$, the line segment connecting them lies entirely within $\mathcal{C}$:
$$\theta \mathbf{x} + (1 - \theta)\mathbf{y} \in \mathcal{C}$$

2. **Convex function** $f$: the function lies below (or on) the chord connecting any two points on its graph:
$$f(\theta \mathbf{x} + (1 - \theta)\mathbf{y}) \leq \theta f(\mathbf{x}) + (1 - \theta)f(\mathbf{y})$$

These two conditions together ensure the optimization landscape has no "pockets" where an algorithm can get trapped.

### 1.2 Why Convexity Matters for AI

Convexity appears throughout machine learning at three levels:

**Level 1: Directly convex problems.** Many classical ML models solve convex optimization problems:

| Model | Objective | Why Convex |
|---|---|---|
| Linear regression (OLS) | $\min_{\mathbf{w}} \lVert X\mathbf{w} - \mathbf{y} \rVert_2^2$ | Quadratic in $\mathbf{w}$, Hessian $= 2X^\top X \succeq 0$ |
| Ridge regression | $\min_{\mathbf{w}} \lVert X\mathbf{w} - \mathbf{y} \rVert_2^2 + \lambda\lVert \mathbf{w} \rVert_2^2$ | Sum of convex functions |
| Logistic regression | $\min_{\mathbf{w}} \sum_i \log(1 + e^{-y_i \mathbf{w}^\top \mathbf{x}^{(i)}})$ | Composition of convex log-sum-exp with affine |
| SVM (primal) | $\min_{\mathbf{w}} \frac{1}{2}\lVert \mathbf{w} \rVert^2 + C\sum_i \max(0, 1 - y_i \mathbf{w}^\top \mathbf{x}^{(i)})$ | Sum of convex quadratic and convex hinge loss |
| Lasso | $\min_{\mathbf{w}} \lVert X\mathbf{w} - \mathbf{y} \rVert_2^2 + \lambda\lVert \mathbf{w} \rVert_1$ | Sum of convex functions (non-smooth) |

**Level 2: Convex components in nonconvex systems.** Deep learning training is globally nonconvex, but individual components are convex:
- The cross-entropy loss $\mathcal{L}(\mathbf{z}) = -\sum_k y_k \log(\text{softmax}(\mathbf{z})_k)$ is convex in the logits $\mathbf{z}$
- Weight decay $\lambda \lVert \boldsymbol{\theta} \rVert_2^2$ is strongly convex
- Layer normalization solves a convex subproblem at each layer

**Level 3: Convex relaxations of nonconvex problems.** When the true problem is intractable, we solve a convex approximation:
- **Nuclear norm minimization** relaxes rank constraints for matrix completion (Netflix Prize)
- **SDP relaxations** provide bounds for clustering (Mixon et al., 2017) and community detection
- **LoRA** exploits the low-rank structure that nuclear norm penalization promotes (Hu et al., 2022)

**For AI practitioners:** Understanding convexity tells you _when_ you can trust your optimizer (convex losses), _why_ certain tricks work (weight decay adds strong convexity), and _how_ to design tractable formulations (convex relaxation).

### 1.3 Historical Timeline

```
CONVEX OPTIMIZATION TIMELINE
════════════════════════════════════════════════════════════════════════

  1847  Cauchy         — First gradient descent algorithm
  1939  Kantorovich    — Linear programming for economic planning
  1947  Dantzig        — Simplex method for LP
  1951  Kuhn & Tucker  — KKT conditions for constrained optimization
  1960s Rockafellar    — Convex analysis: duality, subdifferentials
  1970s Khachiyan      — Ellipsoid method (first poly-time LP algorithm)
  1984  Karmarkar      — Interior-point methods (practical poly-time)
  1990s Nesterov       — Self-concordance, optimal methods
  1995  Vapnik         — SVM: convex optimization meets ML
  2004  Boyd &         — "Convex Optimization" textbook unifies the
        Vandenberghe     field and makes it accessible
  2013  Parikh & Boyd  — Proximal algorithms for ML
  2017  CVXPY          — Domain-specific language for convex programs
  2022  Hu et al.      — LoRA: low-rank (convex penalty) for LLMs
  2024  Modern LLMs    — AdamW = GD on strongly convex subproblem

════════════════════════════════════════════════════════════════════════
```

The historical arc reveals a pattern: convex optimization theory matured in the 1960s–1990s, then became a computational workhorse for ML in the 2000s–2020s. The field's key insight — that _structure_ (convexity, sparsity, low rank) enables efficient computation — drives modern ML system design.

### 1.4 The Landscape Metaphor

The best way to build intuition for convex optimization is to think about terrain. Imagine standing on a hilly landscape and trying to find the lowest point.

**Convex landscape** = a single bowl. No matter where you stand, walking downhill always leads to the bottom. There is exactly one valley, and it is the global minimum. A ball released anywhere on the surface will roll to the same point.

**Nonconvex landscape** = mountain terrain with multiple valleys, ridges, and saddle passes. Walking downhill leads to _a_ valley, but not necessarily the _deepest_ one. A ball released from different positions may settle in different valleys.

```
LANDSCAPE COMPARISON
════════════════════════════════════════════════════════════════════════

  Convex (bowl):                    Nonconvex (terrain):

  ╲                        ╱        ╲    ╱╲        ╱╲    ╱
   ╲                      ╱          ╲  ╱  ╲  ╱╲  ╱  ╲  ╱
    ╲                    ╱            ╲╱    ╲╱  ╲╱    ╲╱
     ╲                  ╱              local   saddle   local
      ╲      •        ╱                min     point    min
       ╲  global    ╱
        ╲  min    ╱                   • Which is deepest?
         ╲______╱                       Unknown without
                                        exhaustive search
  Property: ∇f(x*) = 0 ⟹ global min
  Algorithm: gradient descent converges

════════════════════════════════════════════════════════════════════════
```

**For deep learning:** neural network loss surfaces are emphatically nonconvex — they have astronomical numbers of saddle points (far more than local minima). Yet SGD with momentum often finds good solutions. Understanding _why_ requires first understanding what makes convex problems easy, so that we can identify which convex properties partially survive in the nonconvex setting. This is covered in [Optimization Landscape](../06-Optimization-Landscape/notes.md).

**The "convexity toolkit" for the rest of this chapter:**

We develop four layers of theory, each building on the last:

1. **Convex sets** (§2) — the geometry of feasible regions
2. **Convex functions** (§3) — the structure of objectives; when "downhill = globally optimal"
3. **Strong convexity and smoothness** (§4) — quantitative curvature bounds that control convergence speed
4. **Problem classes and duality** (§5–§7) — the taxonomy of tractable problems and the power of the dual

Each layer provides tools used daily in ML: sets define feasible regions (simplex for softmax, PSD cone for kernel matrices), functions define losses (convexity guarantees training converges), curvature bounds set learning rates, and duality enables the kernel trick and connects regularization to constraints.

---

## 2. Convex Sets

### 2.1 Definition and Examples

**Definition (Convex Set).** A set $\mathcal{C} \subseteq \mathbb{R}^n$ is **convex** if for every $\mathbf{x}, \mathbf{y} \in \mathcal{C}$ and every $\theta \in [0, 1]$:

$$\theta \mathbf{x} + (1 - \theta)\mathbf{y} \in \mathcal{C}$$

Geometrically: the line segment between any two points in the set lies entirely within the set.

**Examples of convex sets:**

1. **The empty set $\emptyset$ and all of $\mathbb{R}^n$** — vacuously and trivially convex.

2. **Hyperplanes.** $\mathcal{H} = \{\mathbf{x} \in \mathbb{R}^n : \mathbf{a}^\top \mathbf{x} = b\}$ for some $\mathbf{a} \neq \mathbf{0}$, $b \in \mathbb{R}$. A hyperplane is convex because any linear combination of points satisfying $\mathbf{a}^\top \mathbf{x} = b$ also satisfies it: $\mathbf{a}^\top (\theta\mathbf{x} + (1-\theta)\mathbf{y}) = \theta b + (1-\theta)b = b$.

3. **Halfspaces.** $\mathcal{H}_+ = \{\mathbf{x} : \mathbf{a}^\top \mathbf{x} \leq b\}$. Convex by the same argument with $\leq$.

4. **Euclidean balls.** $\mathcal{B}(\mathbf{c}, r) = \{\mathbf{x} : \lVert \mathbf{x} - \mathbf{c} \rVert_2 \leq r\}$. By the triangle inequality: $\lVert \theta(\mathbf{x} - \mathbf{c}) + (1-\theta)(\mathbf{y} - \mathbf{c}) \rVert \leq \theta r + (1-\theta) r = r$.

5. **Polyhedra.** $\mathcal{P} = \{\mathbf{x} : A\mathbf{x} \leq \mathbf{b}\}$ — intersection of finitely many halfspaces. Every LP feasible region is a polyhedron.

6. **Ellipsoids.** $\mathcal{E} = \{\mathbf{x} : (\mathbf{x} - \mathbf{c})^\top P^{-1}(\mathbf{x} - \mathbf{c}) \leq 1\}$ where $P \succ 0$. The confidence ellipsoid of a Gaussian distribution is an ellipsoid.

7. **The probability simplex.** $\Delta_n = \{\mathbf{p} \in \mathbb{R}^n : \mathbf{p} \geq \mathbf{0},\; \mathbf{1}^\top \mathbf{p} = 1\}$. Every softmax output lives in this set.

**For AI:** The probability simplex $\Delta_n$ is ubiquitous — it is the output space of every classifier using softmax. The feasible region of an SVM dual is the intersection of a simplex with box constraints, which is a polyhedron. Confidence ellipsoids from Gaussian posteriors are ellipsoids.

**Non-examples (not convex):**

1. **The set $\{0, 1\}^n$** (binary vectors) — the midpoint $(\frac{1}{2}, \ldots, \frac{1}{2})$ is not in the set. This is why integer programming is NP-hard while LP is polynomial.

2. **The unit sphere** $\{\mathbf{x} : \lVert \mathbf{x} \rVert = 1\}$ — the midpoint of two antipodal points is the origin, which has norm 0, not 1. The unit _ball_ is convex; the unit _sphere_ (boundary only) is not.

3. **The set of rank-$k$ matrices** $\{A \in \mathbb{R}^{m \times n} : \operatorname{rank}(A) = k\}$ — the sum of two rank-1 matrices can have rank 2. This is why rank-constrained optimization is hard, motivating the nuclear norm convex relaxation.

### 2.2 Operations Preserving Convexity of Sets

A powerful feature of convex sets is that many natural operations preserve convexity. This allows building complex convex sets from simple ones.

**Theorem (Intersection).** If $\mathcal{C}_1, \mathcal{C}_2, \ldots$ are convex, then $\bigcap_i \mathcal{C}_i$ is convex.

_Proof._ Let $\mathbf{x}, \mathbf{y} \in \bigcap_i \mathcal{C}_i$ and $\theta \in [0,1]$. Then $\mathbf{x}, \mathbf{y} \in \mathcal{C}_i$ for every $i$, so $\theta\mathbf{x} + (1-\theta)\mathbf{y} \in \mathcal{C}_i$ for every $i$ (each $\mathcal{C}_i$ is convex), hence $\theta\mathbf{x} + (1-\theta)\mathbf{y} \in \bigcap_i \mathcal{C}_i$. $\square$

**Corollary.** A polyhedron $\{\mathbf{x} : A\mathbf{x} \leq \mathbf{b}\}$ is convex (intersection of halfspaces).

**For AI:** The feasible region of any constrained ML problem (SVM, constrained fine-tuning, fairness constraints) is typically an intersection of convex sets — hence convex.

**Theorem (Affine Image and Preimage).** If $\mathcal{C}$ is convex and $T(\mathbf{x}) = A\mathbf{x} + \mathbf{b}$ is an affine map, then:
- $T(\mathcal{C}) = \{A\mathbf{x} + \mathbf{b} : \mathbf{x} \in \mathcal{C}\}$ is convex (image)
- $T^{-1}(\mathcal{C}) = \{\mathbf{x} : A\mathbf{x} + \mathbf{b} \in \mathcal{C}\}$ is convex (preimage)

_Proof sketch._ For the image: if $\mathbf{u} = A\mathbf{x} + \mathbf{b}$ and $\mathbf{v} = A\mathbf{y} + \mathbf{b}$ with $\mathbf{x}, \mathbf{y} \in \mathcal{C}$, then $\theta\mathbf{u} + (1-\theta)\mathbf{v} = A(\theta\mathbf{x} + (1-\theta)\mathbf{y}) + \mathbf{b}$, and $\theta\mathbf{x} + (1-\theta)\mathbf{y} \in \mathcal{C}$ by convexity. The preimage argument is analogous. $\square$

**Other operations preserving convexity:**

| Operation | Result | Example |
|---|---|---|
| Intersection | $\mathcal{C}_1 \cap \mathcal{C}_2$ | Polyhedra, feasible regions |
| Affine image | $\{A\mathbf{x} + \mathbf{b} : \mathbf{x} \in \mathcal{C}\}$ | Linear projections of convex sets |
| Affine preimage | $\{\mathbf{x} : A\mathbf{x} + \mathbf{b} \in \mathcal{C}\}$ | Constraint transformation |
| Cartesian product | $\mathcal{C}_1 \times \mathcal{C}_2$ | Independent parameter blocks |
| Minkowski sum | $\mathcal{C}_1 + \mathcal{C}_2 = \{\mathbf{x} + \mathbf{y} : \mathbf{x} \in \mathcal{C}_1, \mathbf{y} \in \mathcal{C}_2\}$ | Error tolerance regions |
| Perspective | $\{(\mathbf{x}/t, 1/t) : (\mathbf{x}, t) \in \mathcal{C}, t > 0\}$ | Perspective projections |

**Warning:** Union does NOT preserve convexity. $\mathcal{C}_1 \cup \mathcal{C}_2$ is generally nonconvex even when both $\mathcal{C}_1$ and $\mathcal{C}_2$ are convex. This is why "solve problem A _or_ problem B" is harder than solving each one separately.

### 2.3 Supporting Hyperplanes and Separation Theorems

The geometry of convex sets is characterized by their interaction with hyperplanes. Two fundamental results formalize this.

**Definition (Supporting Hyperplane).** A hyperplane $\{\mathbf{x} : \mathbf{a}^\top \mathbf{x} = b\}$ is a _supporting hyperplane_ to a convex set $\mathcal{C}$ at a boundary point $\mathbf{x}_0 \in \partial \mathcal{C}$ if:
- $\mathbf{a}^\top \mathbf{x}_0 = b$
- $\mathbf{a}^\top \mathbf{x} \leq b$ for all $\mathbf{x} \in \mathcal{C}$

The hyperplane "touches" the set at $\mathbf{x}_0$ and the entire set lies on one side.

**Theorem (Supporting Hyperplane Theorem).** For any convex set $\mathcal{C}$ with nonempty interior and any boundary point $\mathbf{x}_0 \in \partial \mathcal{C}$, there exists a supporting hyperplane at $\mathbf{x}_0$.

**Theorem (Separating Hyperplane Theorem).** If $\mathcal{C}_1$ and $\mathcal{C}_2$ are disjoint convex sets, then there exists a hyperplane $\mathbf{a}^\top \mathbf{x} = b$ separating them:

$$\mathbf{a}^\top \mathbf{x} \leq b \text{ for all } \mathbf{x} \in \mathcal{C}_1, \quad \mathbf{a}^\top \mathbf{x} \geq b \text{ for all } \mathbf{x} \in \mathcal{C}_2$$

**For AI:** The separating hyperplane theorem is the theoretical foundation of SVMs. The SVM seeks the hyperplane with maximum margin separating two classes — it finds the _best_ separating hyperplane between the convex hulls of the two classes. The SVM dual exploits this geometric structure to find the optimal separator efficiently.

**Strict separation.** If $\mathcal{C}_1$ and $\mathcal{C}_2$ are disjoint, _closed_, and at least one is _compact_ (bounded), then there exists a _strictly_ separating hyperplane with a gap $\delta > 0$:

$$\mathbf{a}^\top\mathbf{x} \leq b - \delta \text{ for all } \mathbf{x} \in \mathcal{C}_1, \quad \mathbf{a}^\top\mathbf{x} \geq b + \delta \text{ for all } \mathbf{x} \in \mathcal{C}_2$$

This gap $2\delta$ is the _margin_ in SVM terminology. The SVM maximizes this margin, which by Vapnik's theory (1995) minimizes the generalization bound.

**Farkas' Lemma.** A related result that underpins LP duality: for a matrix $A$ and vector $\mathbf{b}$, exactly one of the following is true:

- $\exists \mathbf{x} \geq \mathbf{0}$ such that $A\mathbf{x} = \mathbf{b}$
- $\exists \mathbf{y}$ such that $A^\top\mathbf{y} \geq \mathbf{0}$ and $\mathbf{b}^\top\mathbf{y} < 0$

Farkas' lemma is a "theorem of alternatives" — it says a system of linear inequalities either has a solution, or there is a certificate (a separating hyperplane in dual space) proving infeasibility. This is the foundation of LP duality and sensitivity analysis.

```
SEPARATING HYPERPLANE
════════════════════════════════════════════════════════════════════════

         C₁                               C₂
        ●  ●                           ▲  ▲
       ●  ●  ●         |            ▲  ▲  ▲
        ●  ●            |             ▲  ▲
         ●              |              ▲
                        |
                   separating
                   hyperplane
                   aᵀx = b

  All of C₁ on left (aᵀx ≤ b)    All of C₂ on right (aᵀx ≥ b)

════════════════════════════════════════════════════════════════════════
```

### 2.4 Cones and the PSD Cone

**Definition (Cone).** A set $\mathcal{K} \subseteq \mathbb{R}^n$ is a _cone_ if for every $\mathbf{x} \in \mathcal{K}$ and $\alpha \geq 0$, we have $\alpha\mathbf{x} \in \mathcal{K}$. A **convex cone** is a cone that is also convex, which means it is closed under nonnegative linear combinations: $\alpha_1\mathbf{x}_1 + \alpha_2\mathbf{x}_2 \in \mathcal{K}$ for all $\mathbf{x}_1, \mathbf{x}_2 \in \mathcal{K}$ and $\alpha_1, \alpha_2 \geq 0$.

**Examples of convex cones:**

1. **The nonnegative orthant** $\mathbb{R}^n_+ = \{\mathbf{x} \in \mathbb{R}^n : x_i \geq 0,\; \forall i\}$. This is the cone associated with linear programming.

2. **The second-order cone (SOC)** $\mathcal{Q}^{n+1} = \{(\mathbf{x}, t) \in \mathbb{R}^{n+1} : \lVert \mathbf{x} \rVert_2 \leq t\}$. Also called the "ice cream cone" or Lorentz cone.

3. **The positive semidefinite (PSD) cone** $\mathbb{S}^n_+ = \{A \in \mathbb{S}^n : A \succeq 0\}$. This is the cone of $n \times n$ symmetric matrices with all nonnegative eigenvalues.

The PSD cone is the most important cone in machine learning optimization.

**Why the PSD cone matters for AI:**

- **Covariance matrices** are PSD: $\Sigma = \mathbb{E}[(\mathbf{x} - \boldsymbol{\mu})(\mathbf{x} - \boldsymbol{\mu})^\top] \succeq 0$
- **Kernel matrices** (Gram matrices) must be PSD for valid kernels: $K_{ij} = k(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})$ with $K \succeq 0$
- **Hessian PSD** characterizes convex functions: $\nabla^2 f(\mathbf{x}) \succeq 0$ for all $\mathbf{x}$ iff $f$ is convex
- **Fisher information matrix** is PSD: $I(\boldsymbol{\theta}) \succeq 0$, used in natural gradient methods

The hierarchy of cones — nonneg. orthant $\subset$ SOC $\subset$ PSD cone — mirrors the hierarchy of convex programs: LP $\subset$ SOCP $\subset$ SDP. Each level of the hierarchy is more expressive but computationally more expensive.

**Conic programs.** A _conic program_ optimizes a linear objective over the intersection of an affine subspace with a cone:

$$\min_{\mathbf{x}} \quad \mathbf{c}^\top\mathbf{x} \quad \text{s.t.} \quad A\mathbf{x} = \mathbf{b}, \quad \mathbf{x} \in \mathcal{K}$$

where $\mathcal{K}$ is a convex cone. Every problem in the LP→QP→SOCP→SDP hierarchy can be written as a conic program over the appropriate cone. This unified view enables general-purpose conic solvers (SCS, ECOS, MOSEK) that handle all problem types.

**Dual cones.** The dual cone $\mathcal{K}^* = \{\mathbf{y} : \mathbf{y}^\top\mathbf{x} \geq 0 \text{ for all } \mathbf{x} \in \mathcal{K}\}$ plays a key role in conic duality. Self-dual cones satisfy $\mathcal{K}^* = \mathcal{K}$. All three of our key cones are self-dual:

- $(\mathbb{R}^n_+)^* = \mathbb{R}^n_+$ (the nonneg. orthant is self-dual)
- $(\mathcal{Q}^n)^* = \mathcal{Q}^n$ (the SOC is self-dual)
- $(\mathbb{S}^n_+)^* = \mathbb{S}^n_+$ (the PSD cone is self-dual)

Self-duality is what makes LP, SOCP, and SDP dualities especially clean — the dual problem has the same structure as the primal.

**The PSD cone in detail.** To verify $A \succeq 0$:

1. **Eigenvalue test**: compute $\lambda_1 \geq \cdots \geq \lambda_n$ and check $\lambda_n \geq 0$. Cost: $O(n^3)$.
2. **Cholesky test**: attempt $A = LL^\top$. Succeeds iff $A \succeq 0$. Cost: $O(n^3/3)$ — fastest.
3. **Definition test**: check $\mathbf{v}^\top A\mathbf{v} \geq 0$ for all $\mathbf{v}$. Not constructive, but useful for proofs.
4. **Schur complement test**: if $A = \begin{pmatrix} B & C \\ C^\top & D \end{pmatrix}$ with $D \succ 0$, then $A \succeq 0$ iff $B - CD^{-1}C^\top \succeq 0$.

**For AI:** The Schur complement test is used in Gaussian process inference (block-structured kernel matrices), in Fisher information matrix computations (block structure from layer-wise parameters), and in the analysis of attention matrices.

**Proper cones and generalized inequalities.** A _proper cone_ $\mathcal{K}$ is a convex cone that is closed, solid (has nonempty interior), and pointed ($\mathcal{K} \cap (-\mathcal{K}) = \{\mathbf{0}\}$). A proper cone induces a _generalized inequality_ $\preceq_{\mathcal{K}}$:

$$\mathbf{x} \preceq_{\mathcal{K}} \mathbf{y} \iff \mathbf{y} - \mathbf{x} \in \mathcal{K}$$

For $\mathcal{K} = \mathbb{R}^n_+$: this is componentwise $\leq$. For $\mathcal{K} = \mathbb{S}^n_+$: this is the Löwner order $A \preceq B$ iff $B - A \succeq 0$. The Löwner order is used throughout ML when comparing covariance matrices, Hessians, or kernel matrices.

**Generalized inequalities in ML applications:**

- **Comparing Hessians**: $\mu I \preceq \nabla^2 f(\mathbf{x}) \preceq LI$ expresses strong convexity and smoothness simultaneously
- **Covariance comparison**: $\Sigma_1 \preceq \Sigma_2$ means $\Sigma_2 - \Sigma_1$ is PSD — distribution 2 has "more variance" in every direction
- **Kernel dominance**: $K_1 \preceq K_2$ implies the RKHS of kernel 1 is contained in the RKHS of kernel 2
- **Information ordering**: $I(\boldsymbol{\theta}_1) \preceq I(\boldsymbol{\theta}_2)$ means parameter $\boldsymbol{\theta}_2$ is more "informative" in the Fisher sense

These generalized inequalities extend the familiar $\leq$ on real numbers to the rich setting of matrices and cones, providing a unified language for optimization theory.

**Minimum and infimum with respect to generalized inequalities.** A point $\mathbf{x}$ is the _minimum_ of a set $\mathcal{S}$ w.r.t. $\preceq_{\mathcal{K}}$ if $\mathbf{x} \preceq_{\mathcal{K}} \mathbf{y}$ for all $\mathbf{y} \in \mathcal{S}$. A minimum need not exist (unlike for scalar-valued functions). For example, the set $\{A \in \mathbb{S}^2_+ : \operatorname{tr}(A) = 1\}$ has no minimum element in the Löwner order — there is no single PSD matrix that is $\preceq$ all others in the set. This non-existence of a matrix minimum is related to the fact that covariance estimation has no single "best" estimator in all directions simultaneously — different estimators (shrinkage, thresholding, factor models) are optimal in different senses.

**Vector optimization.** When the objective itself is vector-valued, we optimize w.r.t. a cone ordering, leading to Pareto-optimal solutions. In multi-objective ML (accuracy vs. fairness, loss vs. latency), the Pareto frontier is the set of solutions that cannot be improved in one objective without degrading another. Scalarization — combining objectives into $\min \sum_i w_i f_i(\mathbf{x})$ — converts to standard convex optimization when all $f_i$ are convex. The weights $w_i$ trace out the Pareto frontier as they vary, connecting multi-objective optimization to the duality framework developed in §7.

```
CONE HIERARCHY
════════════════════════════════════════════════════════════════════════

  LP cone (ℝ₊ⁿ)  ⊂  SOCP cone (𝒬)  ⊂  SDP cone (𝕊₊ⁿ)

  Nonneg. orthant     Second-order         PSD matrices
  x ≥ 0               ‖x‖ ≤ t             A ≽ 0

  Cheapest to solve    Medium cost          Most expensive
  Simplex / IPM        IPM                  IPM / SDP solvers

  Example:             Example:             Example:
  Resource allocation  Robust optimization  Kernel learning
  Network flow         Portfolio opt.       Matrix completion

════════════════════════════════════════════════════════════════════════
```

---

## 3. Convex Functions

### 3.1 Definition and First-Order Characterization

**Definition (Convex Function).** A function $f: \mathbb{R}^n \to \mathbb{R}$ is **convex** if its domain $\operatorname{dom} f$ is a convex set and for all $\mathbf{x}, \mathbf{y} \in \operatorname{dom} f$ and $\theta \in [0, 1]$:

$$f(\theta \mathbf{x} + (1 - \theta)\mathbf{y}) \leq \theta f(\mathbf{x}) + (1 - \theta) f(\mathbf{y})$$

Geometrically: the chord connecting any two points on the graph lies above (or on) the graph. A function is **strictly convex** if the inequality is strict for $\theta \in (0, 1)$ and $\mathbf{x} \neq \mathbf{y}$.

**Theorem (First-Order Condition).** A differentiable function $f$ is convex if and only if $\operatorname{dom} f$ is convex and:

$$f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^\top (\mathbf{y} - \mathbf{x}) \quad \text{for all } \mathbf{x}, \mathbf{y} \in \operatorname{dom} f$$

_Interpretation:_ the tangent plane at any point is a global underestimator of the function. This is the most practically useful characterization — it says the first-order Taylor approximation never overestimates a convex function.

_Proof sketch (necessity)._ Suppose $f$ is convex. For $\theta \in (0, 1]$:

$$f(\mathbf{x} + \theta(\mathbf{y} - \mathbf{x})) \leq (1-\theta)f(\mathbf{x}) + \theta f(\mathbf{y})$$

Rearranging: $f(\mathbf{x} + \theta(\mathbf{y} - \mathbf{x})) - f(\mathbf{x}) \leq \theta(f(\mathbf{y}) - f(\mathbf{x}))$. Dividing by $\theta$ and taking $\theta \to 0^+$:

$$\nabla f(\mathbf{x})^\top (\mathbf{y} - \mathbf{x}) \leq f(\mathbf{y}) - f(\mathbf{x})$$

which gives the first-order condition. $\square$

**Key consequence.** If $f$ is convex and $\nabla f(\mathbf{x}^*) = \mathbf{0}$, then for all $\mathbf{y}$:
$$f(\mathbf{y}) \geq f(\mathbf{x}^*) + \mathbf{0}^\top(\mathbf{y} - \mathbf{x}^*) = f(\mathbf{x}^*)$$

So $\mathbf{x}^*$ is a **global minimum**. This is _the_ fundamental theorem of convex optimization: zero gradient implies global optimality.

```text
FIRST-ORDER CHARACTERIZATION
════════════════════════════════════════════════════════════════════════

  Convex function:                  Non-convex function:

  f(y)                              f(y)
   │                                 │       ╱╲
   │     ╱                           │      ╱  ╲
   │    ╱                            │     ╱    ╲    ╱
   │   ╱  ← function                 │  ╲╱      ╲  ╱
   │  ╱                              │            ╲╱
   │ ╱ ← tangent (global             │   tangent is NOT a
   │╱    underestimator)              │   global underestimator
   └──────────── x                   └──────────── x

  f(y) ≥ f(x) + ∇f(x)ᵀ(y-x)       f(y) can be BELOW the tangent
  for ALL y                          → first-order condition fails

════════════════════════════════════════════════════════════════════════
```

**Monotone gradient characterization.** Another useful equivalent: $f$ is convex if and only if the gradient is monotone:

$$(\nabla f(\mathbf{x}) - \nabla f(\mathbf{y}))^\top (\mathbf{x} - \mathbf{y}) \geq 0 \quad \text{for all } \mathbf{x}, \mathbf{y}$$

Intuitively: moving in the direction of increasing $\mathbf{x}$ always increases the gradient. For strongly convex $f$, this strengthens to $(\nabla f(\mathbf{x}) - \nabla f(\mathbf{y}))^\top (\mathbf{x} - \mathbf{y}) \geq \mu\lVert\mathbf{x} - \mathbf{y}\rVert^2$.

**For AI:** This is why gradient-based training works reliably for convex losses. When you train logistic regression and the gradient approaches zero, you know you are approaching the global optimum — not stuck in a local minimum. For deep learning, the loss is nonconvex in the parameters, so $\nabla \mathcal{L} = \mathbf{0}$ might be a saddle point. But the loss _is_ convex in the final-layer logits, which is why the output layer learns reliably.

**Convexity and gradient descent convergence (preview).** The first-order condition gives us a bound on suboptimality using just the gradient norm. For $L$-smooth convex $f$:

$$f(\mathbf{x}) - f(\mathbf{x}^*) \leq \frac{L}{2}\lVert\mathbf{x} - \mathbf{x}^*\rVert^2$$

And for $\mu$-strongly convex $f$ (the **gradient dominance** or **Polyak-Łojasiewicz condition**):

$$f(\mathbf{x}) - f(\mathbf{x}^*) \leq \frac{1}{2\mu}\lVert\nabla f(\mathbf{x})\rVert^2$$

This last inequality is extraordinarily useful: it says that the suboptimality gap is controlled by the gradient norm. If $\lVert\nabla f\rVert$ is small, we are close to optimal. This is the theoretical basis for using gradient norm as a convergence diagnostic during training.

### 3.2 Second-Order Characterization

**Theorem (Second-Order Condition).** A twice-differentiable function $f$ is convex if and only if $\operatorname{dom} f$ is convex and:

$$\nabla^2 f(\mathbf{x}) \succeq 0 \quad \text{for all } \mathbf{x} \in \operatorname{dom} f$$

That is, the Hessian is positive semidefinite everywhere. If $\nabla^2 f(\mathbf{x}) \succ 0$ for all $\mathbf{x}$, then $f$ is strictly convex (but the converse does not hold — consider $f(x) = x^4$, which is strictly convex but has $f''(0) = 0$).

_Proof sketch._ The second-order Taylor expansion is:

$$f(\mathbf{y}) = f(\mathbf{x}) + \nabla f(\mathbf{x})^\top(\mathbf{y} - \mathbf{x}) + \frac{1}{2}(\mathbf{y} - \mathbf{x})^\top \nabla^2 f(\mathbf{z})(\mathbf{y} - \mathbf{x})$$

for some $\mathbf{z}$ on the line segment $[\mathbf{x}, \mathbf{y}]$. If $\nabla^2 f \succeq 0$ everywhere, the quadratic term is nonneg., giving $f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^\top(\mathbf{y} - \mathbf{x})$, which is the first-order condition. $\square$

**Practical recipe for checking convexity:**

1. Compute $\nabla^2 f(\mathbf{x})$
2. Check whether $\nabla^2 f(\mathbf{x}) \succeq 0$ for all $\mathbf{x}$ in the domain
3. Common methods: (a) show all eigenvalues $\geq 0$, (b) show $\mathbf{v}^\top \nabla^2 f \, \mathbf{v} \geq 0$ for all $\mathbf{v}$, (c) show leading principal minors are nonneg. (Sylvester's criterion)

**Example: Quadratic function.** $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top Q \mathbf{x} + \mathbf{b}^\top \mathbf{x} + c$ where $Q \in \mathbb{S}^n$.

- $\nabla f(\mathbf{x}) = Q\mathbf{x} + \mathbf{b}$
- $\nabla^2 f(\mathbf{x}) = Q$
- $f$ is convex iff $Q \succeq 0$; strictly convex iff $Q \succ 0$

**For AI:** The loss function of linear regression $\mathcal{L}(\mathbf{w}) = \frac{1}{2}\lVert X\mathbf{w} - \mathbf{y} \rVert^2$ has Hessian $\nabla^2 \mathcal{L} = X^\top X \succeq 0$ (always PSD), confirming convexity. Adding Ridge regularization gives Hessian $X^\top X + \lambda I \succ 0$ (strictly PD for $\lambda > 0$), making the problem _strictly_ convex with a unique minimizer. This is exactly the mathematical mechanism behind weight decay improving optimization stability.

**Example: Verifying convexity of logistic loss.** The binary logistic loss is $f(z) = \log(1 + e^{-z})$. Computing derivatives:

$$f'(z) = \frac{-e^{-z}}{1 + e^{-z}} = -(1 - \sigma(z)) = \sigma(z) - 1$$

$$f''(z) = \sigma(z)(1 - \sigma(z))$$

Since $\sigma(z) \in (0, 1)$, we have $f''(z) > 0$ for all $z$ — the function is strictly convex. The maximum of $f''$ is $1/4$ (at $z = 0$), so the logistic loss is $1/4$-smooth.

For the multivariate logistic regression loss $F(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^n \log(1 + e^{-y_i\mathbf{w}^\top\mathbf{x}^{(i)}})$, the Hessian is:

$$\nabla^2 F(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^n \sigma_i(1 - \sigma_i)\mathbf{x}^{(i)}\mathbf{x}^{(i)\top} \preceq \frac{1}{4n}X^\top X$$

where $\sigma_i = \sigma(y_i\mathbf{w}^\top\mathbf{x}^{(i)})$. This gives $L = \frac{1}{4n}\lambda_{\max}(X^\top X)$, which determines the maximum safe learning rate for gradient descent on logistic regression.

### 3.3 Examples and Non-Examples

**Convex functions (with proofs):**

| Function | Domain | Why Convex |
|---|---|---|
| $f(x) = ax + b$ | $\mathbb{R}$ | Affine; $f'' = 0 \succeq 0$ (also concave) |
| $f(x) = x^2$ | $\mathbb{R}$ | $f'' = 2 > 0$ (strictly convex) |
| $f(x) = e^{ax}$ | $\mathbb{R}$ | $f'' = a^2 e^{ax} > 0$ |
| $f(x) = -\log x$ | $\mathbb{R}_{++}$ | $f'' = 1/x^2 > 0$ (self-concordant barrier) |
| $f(\mathbf{x}) = \lVert \mathbf{x} \rVert_p$ | $\mathbb{R}^n$ | Triangle inequality (for $p \geq 1$) |
| $f(\mathbf{x}) = \max_i x_i$ | $\mathbb{R}^n$ | Pointwise max of affine functions |
| $f(\mathbf{x}) = \log\sum_i e^{x_i}$ | $\mathbb{R}^n$ | Log-sum-exp; smooth approx. to max |
| $f(X) = -\log\det(X)$ | $\mathbb{S}^n_{++}$ | Hessian is PD on PD matrices |

**Concave functions** — a function $f$ is concave if $-f$ is convex. Key examples:

| Function | Domain | Why Concave |
| --- | --- | --- |
| $f(x) = \log x$ | $\mathbb{R}_{++}$ | $f'' = -1/x^2 < 0$ |
| $f(x) = \sqrt{x}$ | $\mathbb{R}_+$ | $f'' = -1/(4x^{3/2}) < 0$ |
| $f(\mathbf{p}) = -\sum_i p_i\log p_i$ | $\Delta_n$ | Entropy; Hessian $= -\operatorname{diag}(1/\mathbf{p}) \prec 0$ |
| $f(X) = \log\det(X)$ | $\mathbb{S}^n_{++}$ | Negative of the convex $-\log\det$ |

Concave functions are important because maximizing a concave function is equivalent to minimizing a convex function. The entropy $H(\mathbf{p}) = -\sum_i p_i \log p_i$ is concave on the simplex — this is why maximum entropy distributions (e.g., the Gaussian for fixed mean and variance) have a well-defined unique solution.

**The log-sum-exp function** deserves special attention — it is the _key_ function connecting convexity to deep learning.

$$\text{lse}(\mathbf{z}) = \log\left(\sum_{i=1}^n e^{z_i}\right)$$

This function is convex, and the cross-entropy loss can be written as:

$$\mathcal{L}(\mathbf{z}) = -z_k + \text{lse}(\mathbf{z})$$

where $k$ is the correct class. Since $-z_k$ is affine (hence convex) and $\text{lse}$ is convex, the sum is convex in the logits $\mathbf{z}$. This is why the output layer of a classifier has a well-behaved loss landscape even when the full network is nonconvex.

**Non-examples (not convex):**

1. **$f(x) = x^3$** on $\mathbb{R}$: $f'' = 6x$, which is negative for $x < 0$. (Convex on $[0, \infty)$ only.)

2. **$f(x) = \cos(x)$**: $f'' = -\cos(x)$, which alternates sign.

3. **$f(\mathbf{x}) = \lVert \mathbf{x} \rVert_0$** (number of nonzeros): not even continuous, let alone convex. Its convex relaxation is $\lVert \mathbf{x} \rVert_1$ — this is the mathematical basis of Lasso sparsity.

4. **$f(W) = \operatorname{rank}(W)$**: not convex (rank is integer-valued). Its convex relaxation is the nuclear norm $\lVert W \rVert_* = \sum_i \sigma_i(W)$ — this is the mathematical basis of LoRA and low-rank matrix completion.

5. **$f(\boldsymbol{\theta}) = \mathcal{L}_{\text{neural net}}(\boldsymbol{\theta})$**: the loss of a deep network as a function of _all_ parameters is nonconvex due to the composition of nonlinear activations and the symmetry of hidden units.

### 3.4 Operations Preserving Convexity of Functions

Just as for sets, functions have a rich algebra of convexity-preserving operations. These rules are the primary tools for establishing convexity in practice — you rarely need to compute the Hessian directly.

**Rule 1: Nonnegative weighted sum.** If $f_1, \ldots, f_k$ are convex and $w_1, \ldots, w_k \geq 0$, then $\sum_i w_i f_i$ is convex.

_Application:_ The total loss $\mathcal{L} = \frac{1}{n}\sum_{i=1}^n \ell(\mathbf{w}; \mathbf{x}^{(i)}, y^{(i)})$ is convex if each per-sample loss $\ell$ is convex in $\mathbf{w}$. Adding a convex regularizer $\lambda R(\mathbf{w})$ preserves convexity.

**Rule 2: Composition with affine.** If $f$ is convex, then $g(\mathbf{x}) = f(A\mathbf{x} + \mathbf{b})$ is convex.

_Application:_ $\lVert A\mathbf{x} - \mathbf{b} \rVert^2$ is convex because $\lVert \cdot \rVert^2$ is convex and $A\mathbf{x} - \mathbf{b}$ is affine. This is why least-squares is convex regardless of $A$.

**Rule 3: Pointwise maximum.** If $f_1, \ldots, f_k$ are convex, then $f(\mathbf{x}) = \max_i f_i(\mathbf{x})$ is convex.

_Application:_ The hinge loss $\max(0, 1 - y \mathbf{w}^\top \mathbf{x})$ is the max of a convex function and zero (affine, hence convex). The ReLU activation $\max(0, x)$ is convex, though this does not make neural network loss convex (composition breaks convexity).

**Rule 4: Composition rules.** For $h: \mathbb{R} \to \mathbb{R}$ and convex $g: \mathbb{R}^n \to \mathbb{R}$:
- $h(g(\mathbf{x}))$ is convex if $h$ is convex and nondecreasing
- $h(g(\mathbf{x}))$ is convex if $h$ is convex and nonincreasing, and $g$ is _concave_

_Application:_ $e^{g(\mathbf{x})}$ is convex when $g$ is convex (since $e^x$ is convex and increasing). $\log(g(\mathbf{x}))$ is concave when $g$ is concave and positive (since $\log$ is concave and increasing).

**Rule 5: Infimum (partial minimization).** If $f(\mathbf{x}, \mathbf{y})$ is convex in $(\mathbf{x}, \mathbf{y})$ jointly, then $g(\mathbf{x}) = \inf_{\mathbf{y}} f(\mathbf{x}, \mathbf{y})$ is convex in $\mathbf{x}$.

_Application:_ The Lagrangian dual function $g(\boldsymbol{\lambda}) = \inf_{\mathbf{x}} \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda})$ is always concave (infimum of affine functions in $\boldsymbol{\lambda}$), regardless of whether the original problem is convex. This is the foundation of duality theory.

**Summary of preservation rules:**

```
CONVEXITY PRESERVATION TOOLKIT
════════════════════════════════════════════════════════════════════════

  Nonneg. combination:  w₁f₁ + w₂f₂           (wᵢ ≥ 0)
  Affine composition:   f(Ax + b)
  Pointwise maximum:    max(f₁, f₂, ..., fₖ)
  Scalar composition:   h(g(x)) with h convex ↑, g convex
  Partial minimization: inf_y f(x, y)

  NOT preserving:
  × General composition f(g(x))    — ReLU(Wx) doesn't help
  × Pointwise minimum min(f₁, f₂)  — concave, not convex
  × Product f₁ · f₂                — not convex in general

════════════════════════════════════════════════════════════════════════
```

### 3.5 Epigraph and Sublevel Sets

**Definition (Epigraph).** The _epigraph_ of a function $f: \mathbb{R}^n \to \mathbb{R}$ is the set of points lying on or above the graph:

$$\operatorname{epi} f = \{(\mathbf{x}, t) \in \mathbb{R}^{n+1} : f(\mathbf{x}) \leq t\}$$

**Theorem.** $f$ is convex if and only if $\operatorname{epi} f$ is a convex set (in $\mathbb{R}^{n+1}$).

This equivalence is the bridge between convex functions and convex sets — it means we can reduce function convexity questions to set convexity questions and vice versa.

**Definition (Sublevel Set).** The $\alpha$-sublevel set of $f$ is:

$$\mathcal{S}_\alpha = \{\mathbf{x} \in \operatorname{dom} f : f(\mathbf{x}) \leq \alpha\}$$

**Theorem.** If $f$ is convex, then all sublevel sets $\mathcal{S}_\alpha$ are convex.

_Proof._ Let $\mathbf{x}, \mathbf{y} \in \mathcal{S}_\alpha$, so $f(\mathbf{x}) \leq \alpha$ and $f(\mathbf{y}) \leq \alpha$. Then:
$$f(\theta\mathbf{x} + (1-\theta)\mathbf{y}) \leq \theta f(\mathbf{x}) + (1-\theta)f(\mathbf{y}) \leq \theta\alpha + (1-\theta)\alpha = \alpha$$

So $\theta\mathbf{x} + (1-\theta)\mathbf{y} \in \mathcal{S}_\alpha$. $\square$

**Warning:** The converse is false — a function with convex sublevel sets need not be convex. Such functions are called **quasiconvex**. Example: $f(x) = \sqrt{|x|}$ has convex sublevel sets (intervals) but is not convex ($f''(x) < 0$ for $x > 0$).

**Quasiconvexity.** A function $f$ is _quasiconvex_ if all its sublevel sets are convex. Equivalently:

$$f(\theta\mathbf{x} + (1-\theta)\mathbf{y}) \leq \max(f(\mathbf{x}), f(\mathbf{y}))$$

Quasiconvex functions are important because:

- The ratio of two positive affine functions is quasiconvex (useful in fractional programming)
- Quasiconvex optimization can still be solved efficiently via bisection on the optimal value
- Several ML-relevant functions (like the generalization error as a function of model complexity) are believed to be quasiconvex (U-shaped)

```text
CONVEX vs QUASICONVEX vs NONCONVEX
════════════════════════════════════════════════════════════════════════

  Convex:               Quasiconvex:            Nonconvex:

      ╲    ╱                 ╱╲                    ╱╲    ╱╲
       ╲  ╱                 ╱  ╲                  ╱  ╲  ╱  ╲
        ╲╱                 ╱    ╲  ╱             ╱    ╲╱    ╲
                          ╱      ╲╱             ╱

  Sublevel sets:        Sublevel sets:          Sublevel sets:
  convex ✓              convex ✓                NOT convex ✗
  Chord above ✓         Chord above ✗           Chord above ✗

════════════════════════════════════════════════════════════════════════
```

**For AI:** Loss contours (the curves where $\mathcal{L}(\boldsymbol{\theta}) = c$) are sublevel set boundaries. For convex losses, these contours are _nested convex sets_ — nice ellipsoidal shapes that gradient descent navigates efficiently. For nonconvex neural network losses, the contours can be highly irregular, explaining why optimization is harder. The condition number $\kappa$ (§4.3) measures how elongated the sublevel set ellipsoids are — large $\kappa$ means elongated ellipsoids and slow convergence.

---

## 4. Strong Convexity and Smoothness

Strong convexity and smoothness are the two regularity conditions that determine _how fast_ optimization algorithms converge. They quantify the curvature of a function: strong convexity provides a lower bound on curvature, smoothness provides an upper bound.

### 4.1 Strong Convexity

**Definition.** A function $f$ is **$\mu$-strongly convex** (with $\mu > 0$) if for all $\mathbf{x}, \mathbf{y} \in \operatorname{dom} f$ and $\theta \in [0, 1]$:

$$f(\theta\mathbf{x} + (1-\theta)\mathbf{y}) \leq \theta f(\mathbf{x}) + (1-\theta)f(\mathbf{y}) - \frac{\mu}{2}\theta(1-\theta)\lVert \mathbf{x} - \mathbf{y} \rVert^2$$

Equivalently, $f(\mathbf{x}) - \frac{\mu}{2}\lVert \mathbf{x} \rVert^2$ is convex. Or: $\nabla^2 f(\mathbf{x}) \succeq \mu I$ for all $\mathbf{x}$ (the Hessian eigenvalues are bounded below by $\mu$).

**What strong convexity buys you:**

1. **Unique minimizer.** A $\mu$-strongly convex function has exactly one minimizer $\mathbf{x}^*$.

2. **Quadratic growth.** $f(\mathbf{x}) \geq f(\mathbf{x}^*) + \frac{\mu}{2}\lVert \mathbf{x} - \mathbf{x}^* \rVert^2$. The function grows at least quadratically away from its minimum.

3. **Linear convergence of GD.** Gradient descent converges at rate $(1 - \mu/L)^t$ — exponentially fast (covered in [Gradient Descent](../02-Gradient-Descent/notes.md)).

**Equivalent characterizations of $\mu$-strong convexity** (for twice-differentiable $f$):

| Characterization | Formula |
| --- | --- |
| Jensen inequality gap | $f(\theta\mathbf{x} + (1-\theta)\mathbf{y}) \leq \theta f(\mathbf{x}) + (1-\theta)f(\mathbf{y}) - \frac{\mu}{2}\theta(1-\theta)\lVert\mathbf{x} - \mathbf{y}\rVert^2$ |
| First-order lower bound | $f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^\top(\mathbf{y} - \mathbf{x}) + \frac{\mu}{2}\lVert\mathbf{y} - \mathbf{x}\rVert^2$ |
| Hessian lower bound | $\nabla^2 f(\mathbf{x}) \succeq \mu I$ for all $\mathbf{x}$ |
| Subtracted quadratic | $f(\mathbf{x}) - \frac{\mu}{2}\lVert\mathbf{x}\rVert^2$ is convex |
| Strong monotonicity | $(\nabla f(\mathbf{x}) - \nabla f(\mathbf{y}))^\top(\mathbf{x} - \mathbf{y}) \geq \mu\lVert\mathbf{x} - \mathbf{y}\rVert^2$ |

The first-order lower bound is particularly useful: it says the function is sandwiched between two quadratics — a paraboloid of curvature $\mu$ from below and a paraboloid of curvature $L$ from above (by smoothness). This "sandwich" controls convergence tightly.

```text
STRONG CONVEXITY: QUADRATIC SANDWICH
════════════════════════════════════════════════════════════════════════

              Upper bound (L-smooth):
              f(x) + ∇f(x)ᵀ(y-x) + (L/2)‖y-x‖²
                    ╱  ╲
                   ╱    ╲
                  ╱ f(y) ╲  ← actual function
                 ╱  ╱  ╲  ╲
                ╱  ╱    ╲  ╲
              ╱  ╱      ╲  ╲
             Lower bound (μ-strongly convex):
             f(x) + ∇f(x)ᵀ(y-x) + (μ/2)‖y-x‖²

  Gap between bounds at optimum:
  f(x) - f(x*) ≤ (1/(2μ))‖∇f(x)‖²    (gradient dominance)

════════════════════════════════════════════════════════════════════════
```

**Examples:**
- $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top Q\mathbf{x}$ with $Q \succ 0$ is $\lambda_{\min}(Q)$-strongly convex
- $f(\mathbf{w}) = \lVert X\mathbf{w} - \mathbf{y} \rVert^2 + \lambda\lVert \mathbf{w} \rVert^2$ is $2\lambda$-strongly convex (the regularization term contributes $\mu = 2\lambda$)
- $f(\mathbf{x}) = \lVert \mathbf{x} \rVert_1$ is convex but **not** strongly convex (flat along coordinate axes)
- $f(x) = e^x$ is strictly convex but NOT strongly convex on $\mathbb{R}$ (the Hessian $e^x \to 0$ as $x \to -\infty$)

**For AI:** Weight decay adds $\lambda\lVert \boldsymbol{\theta} \rVert^2$ to the loss, making it $2\lambda$-strongly convex _in the parameters_. This is the optimization-theoretic reason weight decay improves training: it guarantees a unique optimum and faster convergence. In AdamW (Loshchilov & Hutter, 2019), weight decay is applied directly to the parameters (decoupled from the adaptive step), preserving this strong convexity benefit.

**Strong convexity as regularization.** The relationship between strong convexity and generalization is deep. By adding $\lambda\lVert\mathbf{w}\rVert^2$ to any convex loss:

1. **Optimization**: The loss becomes $\lambda$-strongly convex, guaranteeing linear convergence with rate $(1 - \lambda/L)^t$
2. **Stability**: The solution changes by at most $O(1/(\lambda n))$ when one training example is modified (algorithmic stability)
3. **Generalization**: Stability implies a generalization bound of $O(1/(\lambda n))$ (Bousquet & Elisseeff, 2002)

This triple connection — optimization speed, stability, generalization — is why weight decay is essentially universal in ML training. The strength $\lambda$ trades off training accuracy against generalization: too small and the model overfits (slow convergence, poor generalization); too large and the model underfits (fast convergence to a bad solution).

**Non-examples of strong convexity (important for understanding limitations):**

| Function | Convex? | Strongly Convex? | Why Not Strongly Convex |
| --- | --- | --- | --- |
| $f(\mathbf{x}) = \lVert\mathbf{x}\rVert_1$ | Yes | No | Flat along coordinate axes; $\nabla^2 f$ is zero where it exists |
| $f(x) = \lvert x\rvert$ | Yes | No | Non-differentiable at 0; subgradient set $[-1, 1]$ is bounded |
| $f(\mathbf{w}) = \lVert X\mathbf{w} - \mathbf{y}\rVert^2$ with $\operatorname{rank}(X) < d$ | Yes | No | $\nabla^2 f = 2X^\top X$ has zero eigenvalues; null space of $X$ |
| $f(x) = e^x$ | Yes | No (on $\mathbb{R}$) | $f''(x) = e^x \to 0$ as $x \to -\infty$; no uniform lower bound |

### 4.2 Smoothness (Lipschitz Gradient)

**Definition.** A differentiable function $f$ is **$L$-smooth** (or has $L$-Lipschitz gradient) if:

$$\lVert \nabla f(\mathbf{x}) - \nabla f(\mathbf{y}) \rVert \leq L \lVert \mathbf{x} - \mathbf{y} \rVert \quad \text{for all } \mathbf{x}, \mathbf{y}$$

Equivalently: $\nabla^2 f(\mathbf{x}) \preceq LI$ for all $\mathbf{x}$ (the Hessian eigenvalues are bounded above by $L$). Or: $f$ is upper-bounded by a quadratic:

$$f(\mathbf{y}) \leq f(\mathbf{x}) + \nabla f(\mathbf{x})^\top(\mathbf{y} - \mathbf{x}) + \frac{L}{2}\lVert \mathbf{y} - \mathbf{x} \rVert^2$$

**What smoothness buys you:**

1. **Guaranteed descent.** With step size $\eta = 1/L$, each gradient step is guaranteed to decrease the function value (the descent lemma).

2. **Convergence rate.** GD converges at rate $O(1/t)$ for convex functions and $O((1-\mu/L)^t)$ for strongly convex functions.

3. **Step size selection.** The smoothness constant $L$ directly determines the maximum safe learning rate: $\eta \leq 1/L$.

**The Descent Lemma.** If $f$ is $L$-smooth and we take a gradient step $\mathbf{x}_{+} = \mathbf{x} - \eta\nabla f(\mathbf{x})$ with $\eta \leq 1/L$, then:

$$f(\mathbf{x}_{+}) \leq f(\mathbf{x}) - \frac{\eta}{2}\lVert\nabla f(\mathbf{x})\rVert^2$$

_Proof._ By $L$-smoothness:

$$f(\mathbf{x}_{+}) \leq f(\mathbf{x}) + \nabla f(\mathbf{x})^\top(\mathbf{x}_{+} - \mathbf{x}) + \frac{L}{2}\lVert\mathbf{x}_{+} - \mathbf{x}\rVert^2$$

Substituting $\mathbf{x}_{+} - \mathbf{x} = -\eta\nabla f(\mathbf{x})$:

$$f(\mathbf{x}_{+}) \leq f(\mathbf{x}) - \eta\lVert\nabla f(\mathbf{x})\rVert^2 + \frac{L\eta^2}{2}\lVert\nabla f(\mathbf{x})\rVert^2 = f(\mathbf{x}) - \eta\left(1 - \frac{L\eta}{2}\right)\lVert\nabla f(\mathbf{x})\rVert^2$$

For $\eta \leq 1/L$, we have $1 - L\eta/2 \geq 1/2$, giving the result. $\square$

This is the single most important inequality in optimization theory — it guarantees that each gradient step makes progress proportional to $\lVert\nabla f\rVert^2$. The step size $\eta = 1/L$ maximizes the guaranteed improvement.

**Examples:**
- $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top Q\mathbf{x}$ is $\lambda_{\max}(Q)$-smooth
- Logistic loss $f(z) = \log(1 + e^{-z})$ is $1/4$-smooth (the Hessian $\sigma(z)(1-\sigma(z)) \leq 1/4$)
- $f(\mathbf{w}) = \frac{1}{n}\sum_i \log(1 + e^{-y_i\mathbf{w}^\top\mathbf{x}^{(i)}})$ is $\frac{1}{4n}\lambda_{\max}(X^\top X)$-smooth
- $f(x) = x^4$ is NOT globally smooth on $\mathbb{R}$ (the Hessian $12x^2$ is unbounded) — smoothness requires bounded curvature

**For AI:** The smoothness constant $L$ is the reason learning rate matters so much. If you set $\eta > 2/L$, gradient descent _diverges_. If you set $\eta = 1/L$, you get the optimal convergence rate for a first-order method. In practice, the effective smoothness constant varies across the loss landscape, which is why learning rate warmup (gradually increasing $\eta$) helps — it avoids divergence in the early, high-curvature phase of training. See [Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md).

### 4.3 Condition Number

**Definition.** The **condition number** of an $L$-smooth, $\mu$-strongly convex function is:

$$\kappa = \frac{L}{\mu}$$

It measures the ratio of maximum to minimum curvature. For a quadratic $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top Q\mathbf{x}$, this is exactly the matrix condition number $\kappa(Q) = \lambda_{\max}(Q)/\lambda_{\min}(Q)$.

**Why condition number matters:**

| Condition Number | Sublevel Set Shape | GD Convergence | Intuition |
|---|---|---|---|
| $\kappa = 1$ | Spherical | 1 step | All directions have same curvature |
| $\kappa = 10$ | Mildly elliptical | Fast | GD navigates efficiently |
| $\kappa = 10^3$ | Very elongated | Slow | GD zigzags along narrow valleys |
| $\kappa = 10^6$ | Extremely elongated | Impractical | Need preconditioning or 2nd-order |
| $\kappa = \infty$ | Degenerate (flat dir.) | No convergence | Not strongly convex |

```
CONDITION NUMBER AND CONVERGENCE
════════════════════════════════════════════════════════════════════════

  κ = 1 (well-conditioned):        κ = 100 (ill-conditioned):

      ╭────╮                           ╭──────────────────────╮
      │    │                           │                      │
      │  • │ ← GD goes straight        │  ←←←←←←←←←←←←←←←  │
      │    │   to minimum               │  →→→→→→→→→→→→→→→  │
      ╰────╯                           │  ←←←←←←←←←←← •   │
                                       ╰──────────────────────╯
  GD steps: ~1                        GD steps: ~κ = 100
  (direct path)                       (zigzag along narrow valley)

════════════════════════════════════════════════════════════════════════
```

**Geometric interpretation.** The sublevel sets of a quadratic $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top Q\mathbf{x}$ are ellipsoids. The condition number $\kappa$ determines the _eccentricity_ of these ellipsoids — the ratio of the longest to shortest axis. When $\kappa = 1$, the ellipsoids are spheres and GD goes straight to the minimum. When $\kappa \gg 1$, the ellipsoids are extremely elongated and GD zigzags, bouncing between the narrow walls of the ellipsoid while making slow progress along its long axis.

The eigenvectors of $Q$ define the principal axes of the ellipsoid. The eigenvalue $\lambda_i$ determines the curvature along each axis. GD converges fastest along high-curvature (large $\lambda$) directions and slowest along low-curvature (small $\lambda$) directions. The convergence rate is dominated by the worst axis — i.e., by $\kappa = \lambda_{\max}/\lambda_{\min}$.

**For AI:** The condition number of the Hessian of the loss function determines how hard the optimization problem is. Ill-conditioned problems (high $\kappa$) are the primary motivation for:
- **Preconditioning** — second-order methods like Newton's method effectively reduce $\kappa$ to 1 (see [Second-Order Methods](../03-Second-Order-Methods/notes.md))
- **Adaptive methods** — Adam and AdaGrad approximate per-coordinate preconditioning (see [Adaptive Learning Rate](../07-Adaptive-Learning-Rate/notes.md))
- **Batch normalization** — empirically reduces the effective condition number of the loss landscape (Santurkar et al., 2018)
- **Weight initialization** — schemes like He/Xavier aim to keep the Hessian well-conditioned at initialization

### 4.4 Implications for Convergence Rates

The interplay of $\mu$ and $L$ determines the convergence rate of gradient descent (proven in [Gradient Descent](../02-Gradient-Descent/notes.md)):

| Function Class | Convergence Rate | Steps to $\epsilon$-optimality |
|---|---|---|
| Convex, $L$-smooth | $f(\mathbf{x}_t) - f^* \leq O(L/t)$ | $O(L/\epsilon)$ |
| $\mu$-strongly convex, $L$-smooth | $f(\mathbf{x}_t) - f^* \leq O((1 - \mu/L)^t)$ | $O(\kappa \log(1/\epsilon))$ |
| Convex + Nesterov acceleration | $f(\mathbf{x}_t) - f^* \leq O(L/t^2)$ | $O(\sqrt{L/\epsilon})$ |
| Strongly convex + Nesterov | $f(\mathbf{x}_t) - f^* \leq O((1 - \sqrt{\mu/L})^t)$ | $O(\sqrt{\kappa} \log(1/\epsilon))$ |

Key observations:
- Strong convexity converts sublinear ($O(1/t)$) to linear (exponential) convergence
- Nesterov acceleration reduces the dependence from $\kappa$ to $\sqrt{\kappa}$ — a quadratic speedup
- The condition number $\kappa = L/\mu$ appears in every convergence bound

**For AI:** These rates explain why learning rate tuning is critical. The learning rate $\eta = 1/L$ is optimal for GD; using $\eta > 2/L$ causes divergence. In practice, $L$ is unknown, so we use learning rate schedules or adaptive methods as proxies for estimating and adapting to the local smoothness.

**Lower bounds (optimality of rates).** Nesterov (2004) proved that no first-order method can do better than:

- $O(1/t^2)$ for convex, $L$-smooth functions (matching Nesterov acceleration)
- $O((1 - \sqrt{\mu/L})^t)$ for $\mu$-strongly convex, $L$-smooth functions (matching Nesterov acceleration)

These are _information-theoretic_ lower bounds: they hold for any algorithm that only accesses $f$ through gradient evaluations. The accelerated rates are therefore _optimal_ — you cannot converge faster without using second-order information (Hessians).

**Practical implications for ML training:**

| Scenario | $\kappa$ | Optimal Method | Real-World Example |
| --- | --- | --- | --- |
| Well-conditioned convex | $\kappa \leq 10$ | GD with $\eta = 1/L$ | Logistic regression on centered data |
| Moderate conditioning | $10 < \kappa \leq 10^3$ | Nesterov-accelerated GD | Ridge regression with moderate $\lambda$ |
| Ill-conditioned | $10^3 < \kappa \leq 10^6$ | L-BFGS or Adam | Deep network with poor initialization |
| Extremely ill-conditioned | $\kappa > 10^6$ | Preconditioned methods | LLM training (varying eigenvalues) |
| Not strongly convex | $\kappa = \infty$ | SGD with diminishing $\eta$ | Unregularized neural network |

**Co-coercivity.** For $L$-smooth convex functions, the gradient satisfies the _co-coercivity_ inequality:

$$(\nabla f(\mathbf{x}) - \nabla f(\mathbf{y}))^\top(\mathbf{x} - \mathbf{y}) \geq \frac{1}{L}\lVert\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\rVert^2$$

This says the gradient is not just Lipschitz ($\lVert\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\rVert \leq L\lVert\mathbf{x} - \mathbf{y}\rVert$) but also "co-coercive" — the inner product of gradient difference and position difference is at least $1/L$ times the squared gradient difference. Co-coercivity is a stronger condition that combines smoothness and convexity, and it is the key tool in tight convergence proofs.

**Relationship between $\mu$-strong convexity and $L$-smoothness.** A function $f$ is $\mu$-strongly convex and $L$-smooth if and only if:

$$\frac{\mu L}{\mu + L}\lVert\mathbf{x} - \mathbf{y}\rVert^2 + \frac{1}{\mu + L}\lVert\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\rVert^2 \leq (\nabla f(\mathbf{x}) - \nabla f(\mathbf{y}))^\top(\mathbf{x} - \mathbf{y})$$

This interpolation inequality, due to Nesterov (2004), unifies strong convexity and smoothness into a single condition and is the sharpest characterization of the function class $\mathcal{F}_{\mu,L}$.

---

## 5. Convex Optimization Problems

### 5.1 Standard Form and Terminology

A convex optimization problem has the standard form:

$$\min_{\mathbf{x}} \quad f_0(\mathbf{x})$$
$$\text{s.t.} \quad f_i(\mathbf{x}) \leq 0, \quad i = 1, \ldots, m$$
$$\phantom{\text{s.t.}} \quad \mathbf{a}_j^\top \mathbf{x} = b_j, \quad j = 1, \ldots, p$$

where $f_0, f_1, \ldots, f_m$ are convex functions and the equality constraints are affine.

| Term | Definition |
|---|---|
| **Objective function** $f_0$ | The convex function being minimized |
| **Inequality constraints** $f_i(\mathbf{x}) \leq 0$ | Must be convex functions |
| **Equality constraints** $\mathbf{a}_j^\top\mathbf{x} = b_j$ | Must be affine (linear) |
| **Feasible set** $\mathcal{F}$ | $\{\mathbf{x} : f_i(\mathbf{x}) \leq 0,\; \mathbf{a}_j^\top\mathbf{x} = b_j\}$ |
| **Optimal value** $f^*$ | $\inf\{f_0(\mathbf{x}) : \mathbf{x} \in \mathcal{F}\}$ |
| **Optimal point** $\mathbf{x}^*$ | Any $\mathbf{x} \in \mathcal{F}$ with $f_0(\mathbf{x}) = f^*$ |

**Why convex constraints must be of this form:** Inequality constraints $f_i(\mathbf{x}) \leq 0$ with convex $f_i$ yield convex sublevel sets, so the feasible set is an intersection of convex sets — hence convex. Equality constraints must be affine because a nonlinear equality $h(\mathbf{x}) = 0$ defines a non-convex set in general (the zero set of a nonlinear function is typically a curved surface, not a convex region).

### 5.2 Linear Programming

**Standard form:**

$$\min_{\mathbf{x}} \quad \mathbf{c}^\top \mathbf{x} \quad \text{s.t.} \quad A\mathbf{x} \leq \mathbf{b},\; \mathbf{x} \geq \mathbf{0}$$

The objective is affine, the feasible set is a polyhedron. The optimal solution (if it exists and is finite) occurs at a vertex of the polyhedron.

**Fundamental theorem of LP.** If an LP has a finite optimal value, then at least one optimal solution is a vertex (extreme point) of the feasible polyhedron. This is why the simplex method works: it only needs to check vertices, and it walks from vertex to vertex along edges, improving the objective at each step.

**Duality of LP.** The dual of $\min \mathbf{c}^\top\mathbf{x}$ s.t. $A\mathbf{x} \geq \mathbf{b}$, $\mathbf{x} \geq \mathbf{0}$ is:

$$\max \mathbf{b}^\top\mathbf{y} \quad \text{s.t.} \quad A^\top\mathbf{y} \leq \mathbf{c}, \quad \mathbf{y} \geq \mathbf{0}$$

Strong duality always holds for LP (no Slater's condition needed — LP duality is unconditional for feasible problems). The dual variables $\mathbf{y}$ have an economic interpretation as _shadow prices_: $y_i$ measures how much the optimal value would improve if constraint $i$ were relaxed by one unit.

**For AI:** LPs appear in:
- **Sparse optimization**: the $\ell_1$ minimization problem $\min \lVert \mathbf{x} \rVert_1$ s.t. $A\mathbf{x} = \mathbf{b}$ can be reformulated as an LP
- **Network flow**: data routing in distributed training systems
- **Linear classification**: the original linear SVM (before the kernel trick) is an LP in certain formulations

**Algorithms:** Simplex method (Dantzig, 1947) — walks along vertices; interior-point methods (Karmarkar, 1984) — traverses the interior. Both solve in practice in time nearly linear in the problem size.

**Example: L1 minimization as LP.** The basis pursuit problem $\min \lVert\mathbf{x}\rVert_1$ s.t. $A\mathbf{x} = \mathbf{b}$ can be rewritten as an LP by introducing variables $\mathbf{u}, \mathbf{v} \geq \mathbf{0}$ with $\mathbf{x} = \mathbf{u} - \mathbf{v}$:

$$\min_{\mathbf{u}, \mathbf{v}} \mathbf{1}^\top\mathbf{u} + \mathbf{1}^\top\mathbf{v} \quad \text{s.t.} \quad A(\mathbf{u} - \mathbf{v}) = \mathbf{b}, \quad \mathbf{u} \geq \mathbf{0}, \quad \mathbf{v} \geq \mathbf{0}$$

This reformulation is the foundation of compressed sensing (Donoho, 2006; Candès & Tao, 2005), which shows that sparse signals can be recovered from surprisingly few measurements by solving an LP. The implication for AI: model pruning via L1 penalization is computationally tractable.

### 5.3 Quadratic Programming

**Standard form:**

$$\min_{\mathbf{x}} \quad \frac{1}{2}\mathbf{x}^\top Q\mathbf{x} + \mathbf{c}^\top\mathbf{x} \quad \text{s.t.} \quad A\mathbf{x} \leq \mathbf{b}$$

where $Q \succeq 0$ (convex QP). If $Q \succ 0$, the objective is strictly convex and the solution is unique.

**For AI:** QPs are the core optimization subproblems in:
- **SVM (hard-margin primal)**: $\min_{\mathbf{w}} \frac{1}{2}\lVert\mathbf{w}\rVert^2$ s.t. $y_i(\mathbf{w}^\top\mathbf{x}^{(i)} + b) \geq 1$ — a QP
- **Ridge regression**: $\min_{\mathbf{w}} \frac{1}{2}\lVert X\mathbf{w} - \mathbf{y}\rVert^2 + \frac{\lambda}{2}\lVert\mathbf{w}\rVert^2$ — an unconstrained QP
- **Model predictive control**: used in robotics and autonomous systems
- **Portfolio optimization**: Markowitz mean-variance framework

**Solving unconstrained QPs.** When $Q \succ 0$, the unconstrained QP $\min \frac{1}{2}\mathbf{x}^\top Q\mathbf{x} + \mathbf{c}^\top\mathbf{x}$ has the closed-form solution:

$$\mathbf{x}^* = -Q^{-1}\mathbf{c}$$

For Ridge regression, $Q = X^\top X + \lambda I$ and $\mathbf{c} = -X^\top\mathbf{y}$, giving $\mathbf{w}^* = (X^\top X + \lambda I)^{-1}X^\top\mathbf{y}$ — the classical Ridge formula. The regularization $\lambda I$ ensures $Q \succ 0$ even when $X^\top X$ is singular (more features than samples), which is the mathematical reason Ridge regression works in high-dimensional settings.

**Constrained QPs** require iterative methods. The active set method and interior-point methods are the standard approaches. For the SVM QP, the SMO algorithm (Platt, 1998) decomposes the problem into a sequence of minimal 2-variable QP subproblems, each with a closed-form solution.

### 5.4 Second-Order Cone Programming

**Standard form:**

$$\min_{\mathbf{x}} \quad \mathbf{c}^\top\mathbf{x} \quad \text{s.t.} \quad \lVert A_i\mathbf{x} + \mathbf{b}_i \rVert_2 \leq \mathbf{c}_i^\top\mathbf{x} + d_i, \quad i = 1, \ldots, m$$

The constraints require points to lie in second-order cones. SOCP is strictly more general than LP and QP.

**For AI:** SOCPs appear in:

- **Robust optimization**: handling worst-case perturbations (adversarial robustness). If each data point $\mathbf{x}^{(i)}$ has uncertainty $\lVert\boldsymbol{\delta}\rVert \leq \epsilon$, robust linear classification becomes an SOCP
- **Norm-constrained problems**: $\min f(\mathbf{x})$ s.t. $\lVert\mathbf{x}\rVert_2 \leq r$ — trust region methods used in natural gradient and TRPO (Schulman et al., 2015) for reinforcement learning
- **Portfolio optimization with risk constraints**: $\min -\boldsymbol{\mu}^\top\mathbf{w}$ s.t. $\lVert\Sigma^{1/2}\mathbf{w}\rVert_2 \leq \gamma$ (Lobo et al., 1998)

**SOCP subsumes QP.** Any QP constraint $\mathbf{x}^\top Q\mathbf{x} \leq c$ with $Q \succeq 0$ can be written as $\lVert Q^{1/2}\mathbf{x}\rVert_2 \leq \sqrt{c}$, which is an SOC constraint.

### 5.5 Semidefinite Programming

**Standard form:**

$$\min_{X} \quad \operatorname{tr}(CX) \quad \text{s.t.} \quad \operatorname{tr}(A_iX) = b_i,\; i = 1,\ldots,m, \quad X \succeq 0$$

where $C, A_1, \ldots, A_m \in \mathbb{S}^n$ and the variable $X$ is a symmetric positive semidefinite matrix.

**For AI:** SDPs appear in:

- **Kernel learning**: learning the optimal kernel matrix $K \succeq 0$
- **Relaxations for clustering**: the SDP relaxation of $k$-means and spectral clustering provides provable approximation guarantees (Peng & Wei, 2007)
- **Matrix completion**: the nuclear norm minimization for recommender systems can be cast as an SDP
- **Sum-of-squares optimization**: verifying polynomial inequalities for robustness certificates in neural networks
- **Optimal transport**: the Kantorovich relaxation of optimal transport is an LP; structured variants become SDPs

**Example: SDP relaxation of Max-Cut.** The Max-Cut problem asks to partition a graph into two groups to maximize the number of edges between them. The integer program:

$$\max \frac{1}{4}\sum_{(i,j) \in E} (1 - x_i x_j), \quad x_i \in \{-1, +1\}$$

is NP-hard. The SDP relaxation replaces $x_i x_j$ with $X_{ij}$ where $X \succeq 0$ and $X_{ii} = 1$:

$$\max \frac{1}{4}\operatorname{tr}(LX), \quad X \succeq 0, \quad X_{ii} = 1$$

where $L$ is the graph Laplacian. Goemans & Williamson (1995) proved this SDP achieves at least $0.878$ of the optimal cut — a remarkable approximation ratio from a polynomial-time algorithm. The rounding scheme (hyperplane rounding) extracts a binary partition from the PSD matrix.

### 5.6 The Hierarchy of Convex Programs

The problem classes form a strict hierarchy of increasing expressiveness and computational cost:

$$\text{LP} \subset \text{QP} \subset \text{QCQP} \subset \text{SOCP} \subset \text{SDP}$$

```
HIERARCHY OF CONVEX PROGRAMS
════════════════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────────────┐
  │  SDP  (Semidefinite Programming)                                │
  │  Variable: PSD matrix X ≽ 0                                    │
  │  ┌─────────────────────────────────────────────────────────┐    │
  │  │  SOCP  (Second-Order Cone Programming)                   │    │
  │  │  Constraint: ‖Ax + b‖ ≤ cᵀx + d                        │    │
  │  │  ┌─────────────────────────────────────────────────┐     │    │
  │  │  │  QP  (Quadratic Programming)                     │     │    │
  │  │  │  Objective: ½xᵀQx + cᵀx, Q ≽ 0                 │     │    │
  │  │  │  ┌─────────────────────────────────────────┐     │     │    │
  │  │  │  │  LP  (Linear Programming)                │     │     │    │
  │  │  │  │  Objective: cᵀx                          │     │     │    │
  │  │  │  │  Constraints: Ax ≤ b                     │     │     │    │
  │  │  │  └─────────────────────────────────────────┘     │     │    │
  │  │  └─────────────────────────────────────────────────┘     │    │
  │  └─────────────────────────────────────────────────────────┘    │
  └─────────────────────────────────────────────────────────────────┘

  More expressive →  More expensive to solve →

════════════════════════════════════════════════════════════════════════
```

Every LP is a QP (with $Q = 0$). Every QP is an SOCP. Every SOCP is an SDP. The practical rule: always use the simplest problem class that can express your problem, because solvers for simpler classes are faster.

**Computational complexity of convex programs:**

| Problem Class | Typical Solver | Time Complexity | Variables $n$, Constraints $m$ |
| --- | --- | --- | --- |
| LP | Simplex / IPM | $O(n^{2.5})$ (IPM) | Can solve $n \sim 10^6$ |
| QP | Active set / IPM | $O(n^3)$ (dense) | Can solve $n \sim 10^4$ |
| SOCP | IPM | $O(n^3)$ per iteration | Can solve $n \sim 10^3$ |
| SDP | IPM | $O(n^{6.5})$ (worst case) | Can solve $n \sim 10^2$ (matrix size) |
| General convex | First-order methods | $O(1/\epsilon)$ or $O(\log(1/\epsilon))$ | Can solve $n \sim 10^9$ |

The last row is critical for ML: when the problem has millions or billions of parameters (as in LLMs), we cannot use interior-point methods — they require $O(n^3)$ per iteration for factoring the Newton system. Instead, we use first-order methods (gradient descent, SGD, Adam) that only need $O(n)$ per iteration. This is why understanding convergence rates in terms of $\mu$, $L$, and $\kappa$ is so important for ML — first-order methods are all we have at scale.

**Software for convex optimization:**

| Tool | Language | Strengths |
| --- | --- | --- |
| CVXPY | Python | Domain-specific language; automatic problem classification |
| scipy.optimize.linprog | Python | LP solver; integrated with NumPy |
| Gurobi | C/Python/Java | Industrial-grade LP/QP/SOCP solver |
| MOSEK | C/Python | Best SDP solver; used in academic research |
| PyTorch/JAX | Python | First-order methods via autograd; for ML-scale problems |

**For AI:** CVXPY (Diamond & Boyd, 2016) is particularly useful because it automatically classifies your problem into the hierarchy (LP/QP/SOCP/SDP) and selects the most efficient solver. This is called _disciplined convex programming_ (DCP) — you express the problem using convexity-preserving operations, and the system verifies convexity and dispatches to the right algorithm.

**Disciplined convex programming** enforces the preservation rules from §3.4 at the language level. In CVXPY:

```python
import cvxpy as cp

x = cp.Variable(n)
objective = cp.Minimize(cp.sum_squares(A @ x - b) + lam * cp.norm(x, 1))
constraints = [x >= 0]
prob = cp.Problem(objective, constraints)
prob.solve()  # CVXPY verifies convexity and selects solver
```

CVXPY checks that `sum_squares` (convex) + `norm` (convex) = convex objective, and `x >= 0` is an affine (hence convex) constraint. If you write a non-DCP expression (e.g., `cp.sqrt(cp.sum_squares(x))`), CVXPY raises an error rather than silently solving a potentially nonconvex problem. This "convexity by construction" paradigm prevents many optimization bugs.

**When to use CVXPY vs. PyTorch/JAX:**

| Criterion | CVXPY | PyTorch/JAX |
| --- | --- | --- |
| Problem size | $n \leq 10^4$ (dense), $10^6$ (sparse LP) | $n$ up to $10^{11}$ |
| Problem type | LP, QP, SOCP, SDP, general convex | Any differentiable (and some non-diff.) |
| Guarantees | Certifiable optimality with duality gap | No convergence guarantees for nonconvex |
| Use case | SVM, portfolio opt., signal processing | Neural network training |
| Method | Interior-point, ADMM | SGD, Adam, first-order |

---

## 6. Optimality Conditions

### 6.1 Unconstrained Optimality

For an unconstrained convex optimization problem $\min_{\mathbf{x}} f(\mathbf{x})$, the optimality conditions are remarkably simple.

**Theorem (First-Order Necessary and Sufficient Condition).** If $f$ is convex and differentiable, then $\mathbf{x}^*$ is a global minimizer of $f$ if and only if:

$$\nabla f(\mathbf{x}^*) = \mathbf{0}$$

_Proof._ **Necessity:** Standard calculus — if $\mathbf{x}^*$ is a local (hence global) min of a differentiable function, then $\nabla f(\mathbf{x}^*) = \mathbf{0}$. **Sufficiency:** By the first-order condition for convexity, for all $\mathbf{y}$:

$$f(\mathbf{y}) \geq f(\mathbf{x}^*) + \nabla f(\mathbf{x}^*)^\top(\mathbf{y} - \mathbf{x}^*) = f(\mathbf{x}^*) + \mathbf{0} = f(\mathbf{x}^*)$$

So $\mathbf{x}^*$ is a global minimizer. $\square$

For **non-differentiable** convex functions (like $\lVert \mathbf{x} \rVert_1$), the condition generalizes to $\mathbf{0} \in \partial f(\mathbf{x}^*)$, where $\partial f$ is the _subdifferential_ — the set of all subgradients. A vector $\mathbf{g}$ is a subgradient of $f$ at $\mathbf{x}$ if:

$$f(\mathbf{y}) \geq f(\mathbf{x}) + \mathbf{g}^\top(\mathbf{y} - \mathbf{x}) \quad \text{for all } \mathbf{y}$$

**Example: Optimality for Lasso.** The Lasso objective is $F(\mathbf{w}) = \frac{1}{2}\lVert X\mathbf{w} - \mathbf{y} \rVert^2 + \lambda\lVert \mathbf{w} \rVert_1$. The optimality condition $\mathbf{0} \in \partial F(\mathbf{w}^*)$ gives, for each coordinate $j$:

$$\begin{cases} X_j^\top(X\mathbf{w}^* - \mathbf{y}) + \lambda \operatorname{sign}(w_j^*) = 0 & \text{if } w_j^* \neq 0 \\ |X_j^\top(X\mathbf{w}^* - \mathbf{y})| \leq \lambda & \text{if } w_j^* = 0 \end{cases}$$

This shows why Lasso produces sparse solutions: coordinates where the gradient magnitude is below $\lambda$ are set exactly to zero. The threshold $\lambda$ acts as a sparsity gate.

**For AI:** The Lasso optimality condition is the mathematical mechanism behind L1-based pruning in neural networks. By adding an $\ell_1$ penalty $\lambda\lVert\boldsymbol{\theta}\rVert_1$ to the loss, weights with small gradients are driven to exactly zero, producing a sparse model. This is used in structured pruning for model compression (Zhu & Gupta, 2017).

**Second-order sufficient conditions.** For twice-differentiable $f$, if $\nabla f(\mathbf{x}^*) = \mathbf{0}$ and $\nabla^2 f(\mathbf{x}^*) \succ 0$, then $\mathbf{x}^*$ is a strict local (and for convex $f$, global) minimizer. The sufficient condition $\nabla^2 f \succ 0$ is stronger than needed — $\nabla^2 f \succeq 0$ (PSD) at a critical point of a convex function already guarantees global optimality.

**Optimality for constrained convex minimization over a convex set.** For $\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})$ where $\mathcal{C}$ is a convex set and $f$ is convex and differentiable, the optimality condition is the **variational inequality**:

$$\nabla f(\mathbf{x}^*)^\top(\mathbf{y} - \mathbf{x}^*) \geq 0 \quad \text{for all } \mathbf{y} \in \mathcal{C}$$

This says the gradient at the optimum makes an obtuse angle with every feasible direction. When $\mathcal{C} = \mathbb{R}^n$, this reduces to $\nabla f(\mathbf{x}^*) = \mathbf{0}$ (since $\mathbf{y} - \mathbf{x}^*$ can be any direction).

**For AI:** The variational inequality is the optimality condition for projected gradient descent. When training a model with parameter constraints (e.g., weight clipping in WGANs, norm constraints in adversarial training), the optimizer projects the updated parameters onto the constraint set $\mathcal{C}$, and the solution satisfies this variational inequality rather than the simple zero-gradient condition.

### 6.2 Constrained Preview: Lagrangian and Dual Problem

For the constrained problem $\min f_0(\mathbf{x})$ s.t. $f_i(\mathbf{x}) \leq 0$, $\mathbf{a}_j^\top\mathbf{x} = b_j$, we define the **Lagrangian**:

$$\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = f_0(\mathbf{x}) + \sum_{i=1}^m \lambda_i f_i(\mathbf{x}) + \sum_{j=1}^p \nu_j(\mathbf{a}_j^\top\mathbf{x} - b_j)$$

where $\boldsymbol{\lambda} \in \mathbb{R}^m_+$ are the dual variables for inequality constraints and $\boldsymbol{\nu} \in \mathbb{R}^p$ for equality constraints.

The **Lagrangian dual function** is:

$$g(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \inf_{\mathbf{x}} \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})$$

This function is always concave (as an infimum of affine functions in $(\boldsymbol{\lambda}, \boldsymbol{\nu})$), regardless of whether the original problem is convex.

> _Full treatment of KKT conditions, complementary slackness, and constrained optimization algorithms: [Constrained Optimization](../04-Constrained-Optimization/notes.md)_

### 6.3 Slater's Condition and Strong Duality

**Weak duality** always holds: $g(\boldsymbol{\lambda}, \boldsymbol{\nu}) \leq f^*$ for all $\boldsymbol{\lambda} \geq \mathbf{0}$, $\boldsymbol{\nu}$. The dual provides a lower bound on the primal optimum.

**Strong duality** means $g^* = f^*$ — the dual optimum equals the primal optimum. This is not automatic, but holds under regularity conditions.

**Slater's Condition.** If the primal problem is convex and there exists a strictly feasible point $\hat{\mathbf{x}}$ with $f_i(\hat{\mathbf{x}}) < 0$ for all $i$ (strict inequality), then strong duality holds.

_Intuition:_ Slater's condition rules out pathological cases where the constraints are "barely feasible" in a way that creates a duality gap. For most practical ML problems, Slater's condition holds easily — the feasible region has nonempty interior.

**Why strong duality matters for ML:**

1. **SVM dual.** The SVM primal is a QP in $(\mathbf{w}, b)$ with dimension equal to the number of features. The SVM dual is a QP in $\boldsymbol{\alpha}$ with dimension equal to the number of training samples. Strong duality guarantees that both give the same answer, and we can solve whichever is smaller. The kernel trick operates on the dual, enabling nonlinear classification without computing the feature map explicitly.

2. **Regularization-constraint equivalence.** For any $\lambda > 0$, the regularized problem:
$$\min_{\mathbf{w}} \mathcal{L}(\mathbf{w}) + \lambda\lVert\mathbf{w}\rVert^2$$
is equivalent (via duality) to the constrained problem:
$$\min_{\mathbf{w}} \mathcal{L}(\mathbf{w}) \quad \text{s.t.} \quad \lVert\mathbf{w}\rVert^2 \leq r$$
for some $r = r(\lambda)$. This is the formal justification for viewing weight decay as a constraint on parameter norm.

---

## 7. Duality

### 7.1 Lagrangian Dual Problem

Given a convex optimization problem (primal), the **dual problem** is:

$$\max_{\boldsymbol{\lambda}, \boldsymbol{\nu}} \quad g(\boldsymbol{\lambda}, \boldsymbol{\nu}) \quad \text{s.t.} \quad \boldsymbol{\lambda} \geq \mathbf{0}$$

where $g(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \inf_{\mathbf{x}} \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})$ is the dual function.

**Key properties of the dual:**

| Property | Statement |
|---|---|
| Always concave | $g$ is concave regardless of the primal |
| Provides lower bound | $g(\boldsymbol{\lambda}, \boldsymbol{\nu}) \leq f^*$ for all feasible $(\boldsymbol{\lambda}, \boldsymbol{\nu})$ |
| Dual is always convex | Maximizing a concave function = convex optimization |
| Duality gap | $f^* - g^* \geq 0$ (weak duality); $= 0$ under strong duality |

The dual problem is always a convex optimization problem, even when the primal is nonconvex. This makes the dual a powerful tool — we can always solve it efficiently, and it provides bounds on the intractable primal.

**Example: Dual of a QP.** Consider the primal:
$$\min_{\mathbf{x}} \quad \frac{1}{2}\mathbf{x}^\top Q\mathbf{x} + \mathbf{c}^\top\mathbf{x} \quad \text{s.t.} \quad A\mathbf{x} = \mathbf{b}$$

The Lagrangian is $\mathcal{L}(\mathbf{x}, \boldsymbol{\nu}) = \frac{1}{2}\mathbf{x}^\top Q\mathbf{x} + \mathbf{c}^\top\mathbf{x} + \boldsymbol{\nu}^\top(A\mathbf{x} - \mathbf{b})$.

Setting $\nabla_{\mathbf{x}} \mathcal{L} = Q\mathbf{x} + \mathbf{c} + A^\top\boldsymbol{\nu} = \mathbf{0}$ gives $\mathbf{x} = -Q^{-1}(\mathbf{c} + A^\top\boldsymbol{\nu})$ (assuming $Q \succ 0$). Substituting back:

$$g(\boldsymbol{\nu}) = -\frac{1}{2}(\mathbf{c} + A^\top\boldsymbol{\nu})^\top Q^{-1}(\mathbf{c} + A^\top\boldsymbol{\nu}) - \boldsymbol{\nu}^\top\mathbf{b}$$

This is a concave quadratic in $\boldsymbol{\nu}$ — an unconstrained concave maximization, solvable in closed form.

### 7.2 Weak and Strong Duality

**Weak duality** (always holds):

$$g^* = \max_{\boldsymbol{\lambda} \geq 0, \boldsymbol{\nu}} g(\boldsymbol{\lambda}, \boldsymbol{\nu}) \leq f^* = \min_{\mathbf{x} \in \mathcal{F}} f_0(\mathbf{x})$$

_Proof._ For any feasible $\mathbf{x}$ and any $\boldsymbol{\lambda} \geq \mathbf{0}$:
$$g(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \inf_{\mathbf{z}} \mathcal{L}(\mathbf{z}, \boldsymbol{\lambda}, \boldsymbol{\nu}) \leq \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = f_0(\mathbf{x}) + \sum_i \lambda_i \underbrace{f_i(\mathbf{x})}_{\leq 0} + \sum_j \nu_j \underbrace{(\mathbf{a}_j^\top\mathbf{x} - b_j)}_{= 0} \leq f_0(\mathbf{x})$$

Taking $\max$ over $(\boldsymbol{\lambda}, \boldsymbol{\nu})$ on the left and $\min$ over $\mathbf{x}$ on the right: $g^* \leq f^*$. $\square$

**Strong duality** ($g^* = f^*$) holds for convex problems under Slater's condition (§6.3). When strong duality holds, the primal and dual problems are equivalent — solving either one gives the answer to both.

**The duality gap** $f^* - g^*$ is a certificate of suboptimality. If we have primal-feasible $\mathbf{x}$ and dual-feasible $(\boldsymbol{\lambda}, \boldsymbol{\nu})$ with $f_0(\mathbf{x}) - g(\boldsymbol{\lambda}, \boldsymbol{\nu}) \leq \epsilon$, then $\mathbf{x}$ is $\epsilon$-optimal. Interior-point methods use this duality gap as their stopping criterion.

```text
WEAK vs STRONG DUALITY
════════════════════════════════════════════════════════════════════════

  Weak duality (always holds):

       Primal optimal ──── f* ────────────────
                                              │
                                              │ duality gap ≥ 0
                                              │
       Dual optimal   ──── g* ────────────────

  Strong duality (under Slater's condition):

       Primal optimal ──── f* = g* ──── Dual optimal
                                        │
                                        │ gap = 0
                                        │

  Practical use:
  • If you can compute both f(x) and g(λ), the gap f(x) - g(λ)
    is a certified bound on suboptimality
  • Interior-point methods stop when gap < ε

════════════════════════════════════════════════════════════════════════
```

**When does strong duality fail?** Strong duality can fail for nonconvex problems where the Lagrangian relaxation is too loose. For convex problems, Slater's condition (existence of a strictly feasible point) is almost always satisfied in ML applications — the feasible regions typically have nonempty interior.

### 7.3 Duality in ML

**SVM Dual.** The hard-margin SVM primal:

$$\min_{\mathbf{w}, b} \frac{1}{2}\lVert\mathbf{w}\rVert^2 \quad \text{s.t.} \quad y_i(\mathbf{w}^\top\mathbf{x}^{(i)} + b) \geq 1, \quad i = 1, \ldots, n$$

has the dual:

$$\max_{\boldsymbol{\alpha}} \sum_i \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j \mathbf{x}^{(i)\top}\mathbf{x}^{(j)} \quad \text{s.t.} \quad \boldsymbol{\alpha} \geq \mathbf{0},\; \sum_i \alpha_i y_i = 0$$

Notice that the dual depends on the data only through inner products $\mathbf{x}^{(i)\top}\mathbf{x}^{(j)}$. This enables the **kernel trick**: replace $\mathbf{x}^{(i)\top}\mathbf{x}^{(j)}$ with $k(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})$ to implicitly map to a high-dimensional feature space. Strong duality guarantees the kernel SVM dual gives the same optimum as the primal.

**Regularization as Duality.** For any convex loss $\mathcal{L}$ and convex regularizer $R$:

| Formulation | Mathematical Form | Connection |
|---|---|---|
| Regularized (Lagrangian) | $\min \mathcal{L}(\mathbf{w}) + \lambda R(\mathbf{w})$ | Unconstrained; $\lambda$ controls tradeoff |
| Constrained (Ivanov) | $\min \mathcal{L}(\mathbf{w})$ s.t. $R(\mathbf{w}) \leq r$ | Constrained; $r$ bounds complexity |
| Penalized (Morozov) | $\min R(\mathbf{w})$ s.t. $\mathcal{L}(\mathbf{w}) \leq \epsilon$ | Simplest model fitting data to $\epsilon$ |

These three formulations are equivalent via duality (for appropriate correspondences between $\lambda$, $r$, and $\epsilon$). Weight decay ($R = \lVert\mathbf{w}\rVert^2$) and weight clipping ($R(\mathbf{w}) \leq r$) are dual views of the same regularization.

**For AI:** This duality between regularization and constraints is why:

- **Weight decay** (Lagrangian form) and **weight clipping** (constraint form) achieve similar effects
- **LoRA** constrains the rank of weight updates (constraint) rather than penalizing it (regularization), though nuclear norm penalty (the convex relaxation of rank) connects the two
- **Gradient clipping** can be viewed as a constraint on the dual (gradient) space

**The SVM dual derivation (detailed).** Starting from the primal $\min_{\mathbf{w},b} \frac{1}{2}\lVert\mathbf{w}\rVert^2$ s.t. $y_i(\mathbf{w}^\top\mathbf{x}^{(i)} + b) \geq 1$:

_Step 1._ Rewrite constraints as $1 - y_i(\mathbf{w}^\top\mathbf{x}^{(i)} + b) \leq 0$ and form the Lagrangian:

$$\mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2}\lVert\mathbf{w}\rVert^2 - \sum_{i=1}^n \alpha_i\left[y_i(\mathbf{w}^\top\mathbf{x}^{(i)} + b) - 1\right]$$

_Step 2._ Minimize over primal variables. Setting $\nabla_{\mathbf{w}}\mathcal{L} = \mathbf{0}$:

$$\mathbf{w} = \sum_{i=1}^n \alpha_i y_i \mathbf{x}^{(i)}$$

Setting $\partial\mathcal{L}/\partial b = 0$:

$$\sum_{i=1}^n \alpha_i y_i = 0$$

_Step 3._ Substitute back to get the dual:

$$g(\boldsymbol{\alpha}) = \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i\alpha_j y_i y_j \mathbf{x}^{(i)\top}\mathbf{x}^{(j)}$$

_Step 4._ The dual problem is $\max_{\boldsymbol{\alpha}} g(\boldsymbol{\alpha})$ s.t. $\boldsymbol{\alpha} \geq \mathbf{0}$, $\sum_i \alpha_i y_i = 0$.

The key observation: the dual depends on data only through inner products $\mathbf{x}^{(i)\top}\mathbf{x}^{(j)}$. Replacing these with a kernel $k(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})$ gives the kernel SVM — an infinite-dimensional feature space at finite cost. This is duality's greatest triumph in ML.

---

## 8. Applications in Machine Learning

### 8.1 Convexity of Common Loss Functions

Understanding which loss functions are convex — and in which variables — is essential for reasoning about ML training.

| Loss Function | Formula | Convex in | Proof Sketch |
|---|---|---|---|
| Mean Squared Error | $\frac{1}{n}\sum_i(\hat{y}_i - y_i)^2$ | $\hat{y}$ | $f'' = 2/n > 0$ per coordinate |
| Cross-entropy | $-\sum_k y_k \log p_k$ | $p$ (on simplex) | $-\log$ is convex; nonneg. sum |
| Cross-entropy (logits) | $-z_k + \log\sum_j e^{z_j}$ | $\mathbf{z}$ (logits) | Affine + log-sum-exp (convex) |
| Hinge loss | $\max(0, 1 - y\hat{y})$ | $\hat{y}$ | Max of affine functions |
| Huber loss | Quadratic near 0, linear far | $\hat{y}$ | Piecewise convex, continuous |
| KL divergence | $\sum_k p_k \log(p_k / q_k)$ | $q$ | $-\log$ composition; convex |
| Focal loss | $-\alpha(1-p)^\gamma \log p$ | $p$ (for $\gamma \leq 1$) | Verify Hessian PSD |

**Critical distinction:** The cross-entropy loss is convex in the logits $\mathbf{z}$ but nonconvex in the network parameters $\boldsymbol{\theta}$ (because $\mathbf{z} = f_{\boldsymbol{\theta}}(\mathbf{x})$ is a nonlinear function of $\boldsymbol{\theta}$). This is why logistic regression (direct map from features to logits) is convex, while deep network training is not.

**Detailed proof: cross-entropy is convex in logits.** The cross-entropy loss for a $K$-class classification with true label $k$ is:

$$\mathcal{L}(\mathbf{z}) = -z_k + \log\left(\sum_{j=1}^K e^{z_j}\right)$$

The first term $-z_k$ is affine (hence convex). The second term is the log-sum-exp function. To show log-sum-exp is convex, compute its Hessian:

$$\frac{\partial^2}{\partial z_i \partial z_j}\text{lse}(\mathbf{z}) = \begin{cases} p_i(1 - p_i) & \text{if } i = j \\ -p_i p_j & \text{if } i \neq j \end{cases}$$

where $p_i = e^{z_i}/\sum_j e^{z_j}$ is the softmax output. In matrix form: $\nabla^2 \text{lse} = \operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$. For any vector $\mathbf{v}$:

$$\mathbf{v}^\top(\operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top)\mathbf{v} = \sum_i p_i v_i^2 - \left(\sum_i p_i v_i\right)^2 = \operatorname{Var}_p[v] \geq 0$$

by Jensen's inequality ($\mathbb{E}[X^2] \geq (\mathbb{E}[X])^2$). So the Hessian is PSD, confirming convexity. $\square$

**For AI:** When designing custom loss functions, ensure convexity in the model's output. This guarantees that the last layer always has a well-behaved gradient landscape, even if the full loss over all parameters is nonconvex.

**Fisher consistency and convex surrogates.** In binary classification, the 0-1 loss $\ell_{01}(\hat{y}, y) = \mathbb{1}[\hat{y} \neq y]$ is the quantity we actually care about, but it is nonconvex and non-smooth. Convex surrogates (logistic loss, hinge loss, exponential loss) are convex upper bounds on the 0-1 loss that are tractable to optimize:

$$\ell_{01}(z, y) \leq \ell_{\text{surrogate}}(z, y)$$

A surrogate is _Fisher consistent_ if minimizing it over all measurable functions yields the Bayes-optimal classifier. The logistic loss and hinge loss are Fisher consistent; the squared loss for classification is not (for some distributions). This mathematical framework — optimizing a convex surrogate as a proxy for the true objective — is the foundation of empirical risk minimization in ML.

**The softmax bottleneck.** Yang et al. (2018) showed that the softmax output layer imposes a low-rank constraint on the log-probability matrix, limiting the expressiveness of language models. This is a case where the convexity of the output layer (cross-entropy in logits) comes with a hidden cost — the output distribution cannot be fully general. Mixture-of-softmax and other approaches address this limitation.

### 8.2 Why Deep Learning Is Non-Convex (and What Survives)

The loss function of a neural network $\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{n}\sum_i \ell(f_{\boldsymbol{\theta}}(\mathbf{x}^{(i)}), y^{(i)})$ is nonconvex in $\boldsymbol{\theta}$ for three fundamental reasons:

**1. Composition of nonlinear activations.** Even $f(w) = \text{ReLU}(w \cdot x)$ composed with a convex loss gives a nonconvex overall loss in $w$ (the composition of a convex function with a non-affine function is not convex in general).

**2. Hidden unit symmetry.** Permuting the neurons in a hidden layer gives a different $\boldsymbol{\theta}$ with the same function. This creates $k!$ equivalent minima for a layer with $k$ neurons — the loss surface has inherent symmetry.

**3. Scaling ambiguity.** For ReLU networks, scaling a weight up in one layer and down in the next preserves the function: $f(\alpha w_1, w_2/\alpha) = f(w_1, w_2)$ for $\alpha > 0$. This creates continuous manifolds of equivalent solutions.

**What convex structure survives:**

| Property | Status | Practical Impact |
|---|---|---|
| No local minima (for convex) | ✗ Lost | But local minima are rare in practice (mostly saddle points) |
| Loss convex in output layer | ✓ Preserved | Output layer trains reliably |
| Convex regularizers | ✓ Preserved | Weight decay, L1 work as expected |
| Sublevel sets structure | Partially | Mode connectivity: good minima form connected regions |
| Duality theory | Partially | SVM dual works; NN dual is intractable |

**For AI:** Despite nonconvexity, deep learning works because: (a) overparameterization creates many good minima, (b) SGD noise helps escape saddle points, and (c) the loss surface has benign structure (connected level sets, few bad local minima). See [Optimization Landscape](../06-Optimization-Landscape/notes.md) for the full analysis.

```text
CONVEX vs NONCONVEX IN NEURAL NETWORKS
════════════════════════════════════════════════════════════════════════

  Layer structure:  input → [W₁] → ReLU → [W₂] → softmax → loss

  Convexity analysis at each stage:

  1. z₁ = W₁x + b₁         ← affine in W₁ (convex AND concave)
  2. a₁ = ReLU(z₁)          ← convex in z₁, but composition
                                W₁ → z₁ → a₁ is NOT convex in W₁
  3. z₂ = W₂a₁ + b₂        ← affine in W₂, but depends on W₁
                                through a₁ → NONCONVEX in (W₁, W₂)
  4. p = softmax(z₂)        ← convex mapping
  5. L = -log(pₖ)           ← convex in z₂ (proven above)

  Result: L is convex in z₂ (last-layer logits)
          L is NONCONVEX in (W₁, W₂) (all parameters)
          But L is well-behaved enough for SGD to find good solutions

════════════════════════════════════════════════════════════════════════
```

**The convexity spectrum in modern ML:**

| Method | Convexity Status | Implication |
| --- | --- | --- |
| Linear regression | Convex (QP) | Closed-form solution $\mathbf{w}^* = (X^\top X)^{-1}X^\top\mathbf{y}$ |
| Logistic regression | Convex | GD converges to global optimum |
| SVM | Convex (QP) | Dual enables kernel trick |
| 1-layer neural net | Nonconvex | Overparameterization helps |
| Deep network | Highly nonconvex | SGD + momentum + tricks |
| LLM training | Extremely nonconvex | AdamW + warmup + WSD schedule |
| Linear probe on frozen features | Convex | Used for model evaluation |
| LoRA fine-tuning | Nonconvex (but low-rank) | Better conditioned than full fine-tuning |

### 8.3 Convex Relaxations in Practice

When the true optimization problem is nonconvex, we can sometimes solve a convex _relaxation_ — a convex problem whose solution approximates the original.

**Nuclear norm relaxation (for low-rank problems).** The problem of finding a low-rank matrix:

$$\min_{W} \mathcal{L}(W) \quad \text{s.t.} \quad \operatorname{rank}(W) \leq k$$

is NP-hard (rank constraint is nonconvex). The convex relaxation replaces the rank constraint with a nuclear norm penalty:

$$\min_{W} \mathcal{L}(W) + \lambda\lVert W \rVert_*$$

where $\lVert W \rVert_* = \sum_i \sigma_i(W)$ is the sum of singular values. The nuclear norm is the tightest convex relaxation of rank (Fazel, 2002), and under conditions like the RIP (restricted isometry property), the relaxation recovers the exact solution.

**For AI:** This is directly relevant to:
- **Matrix completion** (Netflix Prize): recovering a ratings matrix from sparse observations by minimizing nuclear norm
- **LoRA intuition** (Hu et al., 2022): LoRA constrains weight updates $\Delta W = BA$ to have rank $\leq r$. The nuclear norm is the convex proxy; LoRA uses the explicit low-rank parametrization instead
- **Sparse + low-rank decomposition**: separating a matrix into sparse noise and low-rank signal (Candès et al., 2011)

**SDP relaxation (for combinatorial problems).** The Max-Cut problem:

$$\max \sum_{(i,j) \in E} \frac{1 - x_i x_j}{2}, \quad x_i \in \{-1, +1\}$$

is NP-hard. Relaxing $x_i \in \{-1, +1\}$ to $X \succeq 0$, $X_{ii} = 1$ gives an SDP that achieves at least 0.878 of the optimum (Goemans & Williamson, 1995). This technique applies to graph-based clustering in ML pipelines.

### 8.4 LoRA, Weight Decay, and Convex Penalties

Modern LLM training relies heavily on convex penalty terms, even though the overall problem is nonconvex.

**Weight decay as strong convexity.** Adding $\frac{\lambda}{2}\lVert\boldsymbol{\theta}\rVert^2$ to the loss:
- Makes the loss $\lambda$-strongly convex in $\boldsymbol{\theta}$, locally
- Guarantees a unique optimum for the regularized problem (for fixed data)
- Improves the condition number: $\kappa_{\text{new}} = (L + \lambda)/\lambda \leq L/\lambda$

In AdamW (Loshchilov & Hutter, 2019), weight decay is _decoupled_ from the adaptive gradient step, preserving its regularization effect. This is the default optimizer for training GPT, LLaMA, and most modern LLMs.

**L1 penalty and sparsity.** The $\ell_1$ penalty $\lambda\lVert\boldsymbol{\theta}\rVert_1$ is convex but non-smooth. Its proximal operator is the soft-thresholding function:

$$\operatorname{prox}_{\lambda\lVert\cdot\rVert_1}(\mathbf{x})_i = \operatorname{sign}(x_i)\max(|x_i| - \lambda, 0)$$

This is the mathematical basis of L1-based model pruning and sparse fine-tuning.

**Nuclear norm and low-rank structure.** For matrix-valued parameters, the nuclear norm penalty $\lambda\lVert W \rVert_*$ promotes low-rank solutions. Its proximal operator is singular value soft-thresholding:

$$\operatorname{prox}_{\lambda\lVert\cdot\rVert_*}(W) = U \operatorname{diag}(\max(\sigma_i - \lambda, 0)) V^\top$$

This connects to LoRA: instead of penalizing the nuclear norm (convex relaxation of rank), LoRA directly parametrizes $\Delta W = BA$ with $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$, enforcing rank $\leq r$ by construction.

**Elastic net: combining L1 and L2.** The elastic net penalty $\lambda_1\lVert\mathbf{w}\rVert_1 + \lambda_2\lVert\mathbf{w}\rVert^2$ combines the sparsity of L1 with the strong convexity of L2. The L2 term ensures the problem is strongly convex (unique solution, stable), while L1 induces sparsity. The elastic net is convex because it is the sum of two convex functions.

**For AI:** Elastic net is used in:

- **Sparse fine-tuning**: combining L1 sparsity with L2 stability for pruning (Sanh et al., 2020)
- **Feature selection in genomics/NLP**: selecting relevant features from high-dimensional inputs
- **GLMNET** (Friedman et al., 2010): the standard implementation uses coordinate descent with proximal updates, running in $O(ndK)$ time for $K$ regularization parameters

**Convex penalties as Bayesian priors.** Each convex penalty corresponds to a Bayesian prior through the MAP (maximum a posteriori) connection:

| Penalty $R(\mathbf{w})$ | Prior $p(\mathbf{w})$ | Effect on Posterior |
| --- | --- | --- |
| $\frac{\lambda}{2}\lVert\mathbf{w}\rVert^2$ | $\mathcal{N}(\mathbf{0}, \lambda^{-1}I)$ | Gaussian — shrinks all weights |
| $\lambda\lVert\mathbf{w}\rVert_1$ | $\prod_i \frac{\lambda}{2}e^{-\lambda\lvert w_i\rvert}$ | Laplace — promotes exact zeros |
| $\lambda\lVert W\rVert_*$ | Prior favoring low singular values | Matrix Laplace — promotes low rank |

This connection means that choosing a regularizer is equivalent to choosing a prior belief about the parameter structure. Weight decay assumes weights are Gaussian (no sparsity); L1 assumes weights are Laplace (sparse); nuclear norm assumes weight matrices are low-rank. See [Bayesian Inference](../../07-Statistics/04-Bayesian-Inference/notes.md) for the full treatment.

**DoRA and structured convex penalties.** Weight-Decomposed Low-Rank Adaptation (DoRA, Liu et al., 2024) decomposes weight updates into magnitude and direction components: $W' = m \cdot \frac{W + \Delta W}{\lVert W + \Delta W\rVert}$. The direction normalization is a projection onto the unit sphere (a nonconvex set), but the magnitude $m$ is learned with a standard convex loss. This hybrid approach uses convex optimization for the easy part (magnitude) and handles the nonconvex part (direction) separately — a pattern that recurs throughout modern efficient fine-tuning methods.

**Summary of convex penalties in modern LLM training:**

| Method | Penalty / Constraint | Convex? | Purpose |
| --- | --- | --- | --- |
| Weight decay (AdamW) | $\frac{\lambda}{2}\lVert\boldsymbol{\theta}\rVert^2$ | Yes (strongly convex) | Prevents weight explosion; improves generalization |
| L1 pruning | $\lambda\lVert\boldsymbol{\theta}\rVert_1$ | Yes (non-smooth) | Induces sparsity; model compression |
| LoRA | $\operatorname{rank}(\Delta W) \leq r$ | No (rank constraint) | Efficient fine-tuning; low-rank updates |
| Nuclear norm | $\lambda\lVert W\rVert_*$ | Yes (convex relaxation of rank) | Promotes low-rank structure |
| Gradient clipping | $\lVert\nabla\mathcal{L}\rVert \leq c$ | Yes (ball constraint) | Stabilizes training; prevents explosions |
| KL penalty (DPO/PPO) | $D_{\text{KL}}(\pi \| \pi_{\text{ref}}) \leq \epsilon$ | Yes (convex in $\pi$) | Keeps policy close to reference |

---

## 9. Advanced Topics

### 9.1 Proximal Operators and Non-Smooth Convex Optimization

Many important convex functions in ML are non-smooth — they have "kinks" where the gradient does not exist. The $\ell_1$ norm $\lVert\mathbf{x}\rVert_1$ has a kink at every coordinate where $x_i = 0$. The hinge loss $\max(0, 1 - z)$ has a kink at $z = 1$. Standard gradient descent cannot handle these functions directly.

**Definition (Proximal Operator).** For a convex function $h$, the proximal operator is:

$$\operatorname{prox}_{\eta h}(\mathbf{v}) = \arg\min_{\mathbf{x}} \left\{h(\mathbf{x}) + \frac{1}{2\eta}\lVert\mathbf{x} - \mathbf{v}\rVert^2\right\}$$

The proximal operator finds the point that balances minimizing $h$ with staying close to $\mathbf{v}$. It generalizes the gradient step: for smooth $h$, $\operatorname{prox}_{\eta h}(\mathbf{v}) \approx \mathbf{v} - \eta\nabla h(\mathbf{v})$.

**Proximal gradient descent** handles composite objectives $F(\mathbf{x}) = f(\mathbf{x}) + h(\mathbf{x})$ where $f$ is smooth and $h$ is convex but possibly non-smooth:

$$\mathbf{x}_{t+1} = \operatorname{prox}_{\eta h}\left(\mathbf{x}_t - \eta\nabla f(\mathbf{x}_t)\right)$$

This alternates a gradient step on the smooth part with a proximal step on the non-smooth part.

**Key proximal operators for ML:**

| Function $h$ | $\operatorname{prox}_{\eta h}(\mathbf{v})$ | ML Use |
|---|---|---|
| $\lambda\lVert\mathbf{x}\rVert_1$ | Soft-thresholding: $\operatorname{sign}(v_i)\max(\lvert v_i\rvert - \eta\lambda, 0)$ | Lasso, sparse models |
| $\lambda\lVert X\rVert_*$ | SVD soft-thresholding: $U\operatorname{diag}(\max(\sigma_i - \eta\lambda, 0))V^\top$ | Low-rank matrix recovery |
| $\delta_{\mathcal{C}}$ (indicator of $\mathcal{C}$) | Projection $\Pi_{\mathcal{C}}(\mathbf{v})$ | Constrained optimization |
| $\lambda\lVert\mathbf{x}\rVert_2$ (group lasso) | Block soft-thresholding | Group sparsity |

**For AI:** Proximal gradient descent is the algorithm behind ISTA (Iterative Shrinkage-Thresholding Algorithm) and its accelerated version FISTA (Beck & Teboulle, 2009), widely used for sparse signal recovery and L1-regularized models. The connection to neural network pruning is direct: applying the $\ell_1$ proximal operator zeroes out small weights.

**Convergence of proximal gradient descent.** For $F = f + h$ with $f$ being $L$-smooth and $h$ convex (possibly non-smooth):

| Method | Rate | Steps to $\epsilon$-optimal |
| --- | --- | --- |
| Proximal GD (ISTA) | $O(L/t)$ | $O(L/\epsilon)$ |
| Accelerated Proximal GD (FISTA) | $O(L/t^2)$ | $O(\sqrt{L/\epsilon})$ |
| Proximal GD + strong convexity | $O((1 - \mu/L)^t)$ | $O(\kappa\log(1/\epsilon))$ |

FISTA achieves the same quadratic speedup as Nesterov acceleration for smooth problems. The key insight: acceleration works for non-smooth optimization too, as long as the non-smooth part has a cheap proximal operator.

```text
PROXIMAL GRADIENT DESCENT GEOMETRY
════════��═══════════════════════════════════════════════════════════════

  Objective: F(x) = f(x) + h(x)
                    smooth   non-smooth (e.g., λ‖x‖₁)

  Step 1: Gradient step on smooth part
          v = x_t - η∇f(x_t)

  Step 2: Proximal step on non-smooth part
          x_{t+1} = prox_{ηh}(v)

  For h = λ‖·‖₁ (Lasso):

  Before prox:  v = [0.3, -0.05, 1.2, 0.02, -0.8]
  After  prox:  x = [0.2,  0.00, 1.1, 0.00, -0.7]  (ηλ = 0.1)
                      ↑     ↑           ↑
                   shrunk  zeroed!     zeroed!

  The proximal operator simultaneously shrinks and sparsifies

════════���═════════════════════════════════���═════════════════════════════
```

### 9.2 Fenchel Conjugates

**Definition (Fenchel Conjugate).** The _convex conjugate_ (or Fenchel conjugate) of $f: \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$ is:

$$f^*(\mathbf{y}) = \sup_{\mathbf{x}} \left\{\mathbf{y}^\top\mathbf{x} - f(\mathbf{x})\right\}$$

The conjugate $f^*$ is always convex (as a supremum of affine functions), even if $f$ is not.

**Key conjugate pairs:**

| $f(\mathbf{x})$ | $f^*(\mathbf{y})$ | Notes |
|---|---|---|
| $\frac{1}{2}\lVert\mathbf{x}\rVert^2$ | $\frac{1}{2}\lVert\mathbf{y}\rVert^2$ | Self-conjugate |
| $\lVert\mathbf{x}\rVert$ (any norm) | $\delta_{\{\lVert\mathbf{y}\rVert_* \leq 1\}}$ | Dual norm unit ball indicator |
| $\lVert\mathbf{x}\rVert_1$ | $\delta_{\{\lVert\mathbf{y}\rVert_\infty \leq 1\}}$ | L1 ↔ L∞ duality |
| $e^x$ | $y\log y - y$ (for $y > 0$) | Entropy connection |
| $\delta_{\mathcal{C}}$ (indicator) | $\sigma_{\mathcal{C}}(\mathbf{y}) = \sup_{\mathbf{x} \in \mathcal{C}} \mathbf{y}^\top\mathbf{x}$ | Support function |

**Fenchel-Young inequality:** $f(\mathbf{x}) + f^*(\mathbf{y}) \geq \mathbf{y}^\top\mathbf{x}$ for all $\mathbf{x}, \mathbf{y}$, with equality iff $\mathbf{y} \in \partial f(\mathbf{x})$.

**Biconjugate theorem:** If $f$ is closed and convex, then $f^{**} = f$ — the conjugate of the conjugate recovers the original function. This is the deepest expression of convex duality.

**Computing conjugates — a worked example.** Let $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top Q\mathbf{x}$ with $Q \succ 0$. Then:

$$f^*(\mathbf{y}) = \sup_{\mathbf{x}} \left\{\mathbf{y}^\top\mathbf{x} - \frac{1}{2}\mathbf{x}^\top Q\mathbf{x}\right\}$$

Taking the gradient and setting to zero: $\mathbf{y} - Q\mathbf{x} = \mathbf{0}$, so $\mathbf{x}^* = Q^{-1}\mathbf{y}$. Substituting:

$$f^*(\mathbf{y}) = \mathbf{y}^\top Q^{-1}\mathbf{y} - \frac{1}{2}\mathbf{y}^\top Q^{-1}Q\,Q^{-1}\mathbf{y} = \frac{1}{2}\mathbf{y}^\top Q^{-1}\mathbf{y}$$

So the conjugate of a quadratic with Hessian $Q$ is a quadratic with Hessian $Q^{-1}$. This is the convex-analytic reason that preconditioning (multiplying by $Q^{-1}$) transforms the optimization problem to have condition number 1.

**For AI:** Fenchel conjugates appear in:
- **Variational inference**: the evidence lower bound (ELBO) involves the conjugate of the KL divergence
- **GANs**: the Wasserstein GAN objective is derived from the Kantorovich-Rubinstein duality, which uses Fenchel conjugates
- **f-divergence estimation**: estimating divergences between distributions (Nguyen et al., 2010) uses the variational form $D_f(p\|q) = \sup_T \{\mathbb{E}_p[T] - f^*(\mathbb{E}_q[T])\}$

### 9.3 Mirror Descent Preview

Mirror descent generalizes gradient descent by replacing the Euclidean distance $\frac{1}{2}\lVert\mathbf{x} - \mathbf{y}\rVert^2$ with a **Bregman divergence** $D_\phi(\mathbf{x}, \mathbf{y})$ defined by a strictly convex function $\phi$:

$$D_\phi(\mathbf{x}, \mathbf{y}) = \phi(\mathbf{x}) - \phi(\mathbf{y}) - \nabla\phi(\mathbf{y})^\top(\mathbf{x} - \mathbf{y})$$

The mirror descent update is:

$$\mathbf{x}_{t+1} = \arg\min_{\mathbf{x} \in \mathcal{C}} \left\{\eta \nabla f(\mathbf{x}_t)^\top \mathbf{x} + D_\phi(\mathbf{x}, \mathbf{x}_t)\right\}$$

When $\phi(\mathbf{x}) = \frac{1}{2}\lVert\mathbf{x}\rVert^2$, we recover standard projected gradient descent. When $\phi(\mathbf{x}) = \sum_i x_i \log x_i$ (negative entropy), we get the **exponentiated gradient** algorithm, which is natural for optimization over the probability simplex.

**Key Bregman divergences and their induced algorithms:**

| Mirror map $\phi$ | Bregman divergence $D_\phi$ | Algorithm | Natural domain |
| --- | --- | --- | --- |
| $\frac{1}{2}\lVert\mathbf{x}\rVert^2$ | $\frac{1}{2}\lVert\mathbf{x} - \mathbf{y}\rVert^2$ | Projected GD | $\mathbb{R}^n$ |
| $\sum_i x_i \log x_i$ | $\sum_i x_i \log(x_i/y_i)$ (KL) | Exponentiated gradient | Simplex $\Delta_n$ |
| $-\sum_i \log x_i$ | $\sum_i (x_i/y_i - \log(x_i/y_i) - 1)$ | Interior-point | Positive orthant |
| $\frac{1}{2}\mathbf{x}^\top Q\mathbf{x}$ | $\frac{1}{2}(\mathbf{x}-\mathbf{y})^\top Q(\mathbf{x}-\mathbf{y})$ | Preconditioned GD | $\mathbb{R}^n$ |

**For AI:** Mirror descent is the theoretical framework behind:

- **Natural gradient descent** (Amari, 1998): uses the Fisher information as the Bregman divergence, adapting to the geometry of the parameter space
- **Exponentiated gradient** for online learning on the simplex (language model token probabilities)
- **Follow-the-regularized-leader** (FTRL): the unifying framework for AdaGrad and related adaptive methods
- **RLHF with KL penalty**: the PPO/DPO objective includes a KL divergence from the reference policy, which is a Bregman divergence on the simplex

> _Full treatment of gradient descent and its variants: [Gradient Descent](../02-Gradient-Descent/notes.md)_

---

## 10. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
| --- | --- | --- | --- |
| 1 | Assuming convex sublevel sets imply convexity | Quasiconvex functions have convex sublevel sets but are not convex (e.g., $\sqrt{\lvert x\rvert}$) | Check the definition or Hessian PSD condition directly |
| 2 | Claiming $f(g(\mathbf{x}))$ is convex because $f$ and $g$ are both convex | Composition of convex functions is not convex in general; need $f$ nondecreasing | Apply the composition rules: $f$ convex nondecreasing + $g$ convex → convex |
| 3 | Using $\nabla^2 f \succ 0$ as the definition of strict convexity | $\nabla^2 f \succ 0$ is sufficient but not necessary; $f(x) = x^4$ is strictly convex with $f''(0) = 0$ | Strictly convex means strict inequality in Jensen's inequality for $\mathbf{x} \neq \mathbf{y}$ |
| 4 | Forgetting that equality constraints must be affine in convex problems | $h(\mathbf{x}) = 0$ with nonlinear $h$ gives a nonconvex feasible set | Only linear equalities $\mathbf{a}^\top\mathbf{x} = b$ are allowed |
| 5 | Confusing convexity in logits vs. convexity in parameters | Cross-entropy is convex in logits $\mathbf{z}$ but nonconvex in network weights $\boldsymbol{\theta}$ | Specify which variable: convex in $\mathbf{z}$, not in $\boldsymbol{\theta}$ |
| 6 | Assuming strong duality always holds | Weak duality always holds; strong duality requires Slater's condition or similar | Verify a strictly feasible point exists |
| 7 | Setting learning rate $\eta = 1/\mu$ instead of $\eta = 1/L$ | The safe step size is governed by smoothness $L$ (upper curvature), not strong convexity $\mu$ (lower curvature) | Use $\eta \leq 1/L$; $\mu$ affects convergence _rate_, not step size |
| 8 | Treating $\lVert\mathbf{x}\rVert_0$ as a norm or convex function | The $\ell_0$ "norm" counts nonzeros; it is not a norm (violates triangle inequality) and not convex | Use $\lVert\mathbf{x}\rVert_1$ as the convex relaxation of $\lVert\mathbf{x}\rVert_0$ |
| 9 | Assuming union of convex sets is convex | $\mathcal{C}_1 \cup \mathcal{C}_2$ is generally nonconvex; only intersection preserves convexity | Use intersection; take convex hull if union is needed |
| 10 | Confusing PSD ($\succeq 0$) with entrywise nonneg. | $A \succeq 0$ means $\mathbf{v}^\top A\mathbf{v} \geq 0$ for all $\mathbf{v}$; a matrix with all positive entries can have negative eigenvalues | Check eigenvalues or use Sylvester's criterion |
| 11 | Thinking ReLU makes neural network loss convex | ReLU is convex, but composing convex functions is NOT convex in general | Convexity of activation $\neq$ convexity of loss in parameters |
| 12 | Using too large a learning rate and blaming the algorithm | If $\eta > 2/L$, GD diverges — this is not a bug in GD but a violation of the smoothness-based step size bound | Compute or estimate $L$; use $\eta \leq 1/L$ |

---

## Exercises

### Exercise 1: Verifying Convexity (★)

Determine whether each function is convex, concave, or neither. Justify using the Hessian (second-order condition) or the preservation rules.

**(a)** $f(x, y) = x^2 + 3xy + 4y^2$ on $\mathbb{R}^2$. Compute the Hessian $\nabla^2 f$ and check if it is PSD by examining its eigenvalues.

**(b)** $f(\mathbf{x}) = \log\left(\sum_{i=1}^n e^{x_i}\right)$ (log-sum-exp). Show this is convex by computing the Hessian $\nabla^2 f = \operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$ and proving it is PSD using Jensen's inequality.

**(c)** $f(x) = x \log x$ on $\mathbb{R}_{++}$. Compute $f''(x)$ and determine the sign.

**(d)** $f(X) = \operatorname{tr}(X^\top A X)$ for fixed symmetric $A$. This is a quadratic form in the entries of $X$. Determine the Hessian (as a linear operator on matrices) and state the condition on $A$ for convexity.

**(e)** $f(\mathbf{x}) = \max(x_1, x_2, \ldots, x_n)$. Use the pointwise maximum preservation rule.

### Exercise 2: Convex Set Operations (★)

**(a)** Prove that the probability simplex $\Delta_n = \{\mathbf{p} \geq \mathbf{0} : \mathbf{1}^\top\mathbf{p} = 1\}$ is convex.

**(b)** Give an example showing that the union of two convex sets need not be convex.

**(c)** Prove that the image of a convex set under an affine map is convex.

### Exercise 3: Computing Condition Numbers (★)

For each quadratic $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top Q\mathbf{x}$, compute the strong convexity constant $\mu$, smoothness constant $L$, and condition number $\kappa$. Recall that for a quadratic, $\mu = \lambda_{\min}(Q)$ and $L = \lambda_{\max}(Q)$.

**(a)** $Q = \begin{pmatrix} 4 & 0 \\ 0 & 1 \end{pmatrix}$. This is diagonal, so the eigenvalues are on the diagonal.

**(b)** $Q = \begin{pmatrix} 10 & 3 \\ 3 & 2 \end{pmatrix}$. Compute eigenvalues using the characteristic polynomial.

**(c)** $Q = X^\top X + \lambda I$ where $X \in \mathbb{R}^{100 \times 5}$ with entries from $\mathcal{N}(0, 1)$, $\lambda = 0.1$. Generate the matrix, compute eigenvalues, and report $\kappa$. How does $\lambda$ affect the condition number?

**(d)** For part (c), plot $\kappa$ as a function of $\lambda \in [0.001, 10]$ on a log-log scale. Explain the relationship.

### Exercise 4: Preservation Rules (★★)

**(a)** Prove that the pointwise maximum of convex functions is convex: if $f_1, \ldots, f_k$ are convex, then $g(\mathbf{x}) = \max_i f_i(\mathbf{x})$ is convex.

**(b)** Prove that the infimum over a convex variable preserves convexity: if $f(\mathbf{x}, \mathbf{y})$ is convex in $(\mathbf{x}, \mathbf{y})$ jointly, then $g(\mathbf{x}) = \inf_{\mathbf{y}} f(\mathbf{x}, \mathbf{y})$ is convex in $\mathbf{x}$.

**(c)** Give a counterexample showing that the composition $f(g(\mathbf{x}))$ of two convex functions is not convex in general.

### Exercise 5: Lagrangian Dual (★★)

Derive the dual of the Ridge regression problem:

$$\min_{\mathbf{w}} \frac{1}{2}\lVert X\mathbf{w} - \mathbf{y}\rVert^2 + \frac{\lambda}{2}\lVert\mathbf{w}\rVert^2$$

**(a)** Write the equivalent constrained form: $\min_{\mathbf{w}, \mathbf{r}} \frac{1}{2}\lVert\mathbf{r}\rVert^2 + \frac{\lambda}{2}\lVert\mathbf{w}\rVert^2$ s.t. $X\mathbf{w} - \mathbf{y} = \mathbf{r}$

**(b)** Form the Lagrangian and minimize over primal variables to get the dual function.

**(c)** Show that the dual solution relates to the kernel Ridge regression formula $(XX^\top + \lambda I)^{-1}\mathbf{y}$.

**(d)** Explain why the dual is useful when the number of features $d$ is much larger than the number of samples $n$. What is the computational advantage?

### Exercise 6: Strong Convexity Analysis (★★)

**(a)** Prove that if $f$ is $\mu$-strongly convex, then $f$ has a unique minimizer.

**(b)** Show that adding $\ell_2$ regularization $\frac{\lambda}{2}\lVert\mathbf{x}\rVert^2$ to any convex function makes it $\lambda$-strongly convex.

**(c)** Compute the condition number of logistic regression loss $\frac{1}{n}\sum_i \log(1 + e^{-y_i\mathbf{w}^\top\mathbf{x}^{(i)}}) + \frac{\lambda}{2}\lVert\mathbf{w}\rVert^2$ in terms of $\lambda$ and the data matrix $X$. _Hint:_ the Hessian is bounded above by $\frac{1}{4n}X^\top X + \lambda I$ and below by $\lambda I$.

**(d)** Numerically verify your answer in (c) for synthetic data with $n = 200$, $d = 10$, and $\lambda = 0.01$. How does increasing $\lambda$ affect $\kappa$?

### Exercise 7: Convex Relaxation for Matrix Completion (★★★)

Consider the matrix completion problem: given partial observations $\{(i,j, M_{ij}) : (i,j) \in \Omega\}$ of a low-rank matrix $M$, recover $M$. This is the mathematical problem behind recommender systems (Netflix Prize).

**(a)** Explain why $\min_{X} \sum_{(i,j) \in \Omega}(X_{ij} - M_{ij})^2$ s.t. $\operatorname{rank}(X) \leq r$ is nonconvex. What property of the rank function causes this?

**(b)** Formulate the nuclear norm relaxation: $\min_{X} \sum_{(i,j) \in \Omega}(X_{ij} - M_{ij})^2 + \lambda\lVert X\rVert_*$. Explain why $\lVert X\rVert_*$ is the tightest convex relaxation of $\operatorname{rank}(X)$ on the unit ball.

**(c)** Implement the proximal gradient method: compute the gradient of the data-fidelity term, then apply SVD soft-thresholding (the proximal operator of the nuclear norm).

**(d)** Test on a synthetic $20 \times 20$ rank-3 matrix with 50% observed entries. Report the relative recovery error $\lVert X_{\text{recovered}} - M_{\text{true}}\rVert_F / \lVert M_{\text{true}}\rVert_F$.

**(e)** Plot the singular values of the recovered matrix. Do they match the true rank?

### Exercise 8: Proximal Gradient for Lasso (★★★)

Implement the ISTA algorithm for the Lasso problem: $\min_{\mathbf{w}} \frac{1}{2}\lVert X\mathbf{w} - \mathbf{y}\rVert^2 + \lambda\lVert\mathbf{w}\rVert_1$.

**(a)** Derive the gradient of the smooth part and the proximal operator of the $\ell_1$ part.

**(b)** Implement ISTA with step size $\eta = 1/L$ where $L = \lambda_{\max}(X^\top X)$.

**(c)** Run on synthetic data with $n = 100$ samples, $d = 50$ features, true weight vector with only 5 nonzero entries. Plot the convergence and the sparsity of the solution.

**(d)** Compare the sparsity pattern of the ISTA solution with the true support.

---

## 11. Why This Matters for AI

| Concept | AI Impact |
|---|---|
| Convex functions | Cross-entropy, MSE, hinge loss are convex in model outputs — guarantees reliable output-layer training |
| Strong convexity | Weight decay adds $\mu$-strong convexity to the loss, ensuring unique optima and faster convergence in AdamW |
| Condition number $\kappa$ | Determines learning rate sensitivity and convergence speed; motivates preconditioning (Adam, K-FAC, Shampoo) |
| Lagrangian duality | Enables the SVM kernel trick; connects regularization (Lagrangian) with norm constraints (Ivanov) |
| LP/QP/SDP hierarchy | Determines which ML subproblems are tractable; SDP relaxations provide certificates for adversarial robustness |
| Convex relaxation | Nuclear norm relaxes rank for matrix completion (recommenders) and low-rank adaptation (LoRA) |
| Proximal operators | Power ISTA/FISTA for sparse models; proximal policy optimization (PPO) in reinforcement learning uses proximal trust regions |
| Fenchel conjugates | Foundation of Wasserstein GANs, f-divergence estimation, and variational inference bounds |
| Non-smooth optimization | L1 sparsity, hinge loss (SVM), and group lasso for structured pruning all require non-smooth convex tools |
| Sublevel sets / contours | Shape of loss contours determines GD trajectory; batch normalization improves contour geometry (Santurkar et al., 2018) |

---

## 12. Conceptual Bridge

### The Three Pillars

Convex optimization rests on three pillars, each essential for ML:

1. **Structure** — Convex sets and functions have no hidden traps. This structural property is what makes convex problems tractable and enables the global guarantees that underpin reliable ML training for convex models.

2. **Duality** — Every convex problem has a dual that provides bounds, enables the kernel trick, and connects regularization with constraints. Duality is the theoretical glue between the Lagrangian, KKT conditions, and the regularization-constraint equivalence that governs modern training.

3. **Regularity** — Strong convexity ($\mu$) and smoothness ($L$) quantify the curvature, determining convergence rates and optimal step sizes. The condition number $\kappa = L/\mu$ is the single number that predicts how hard an optimization problem is.

### Backward Connections

This section builds on:

- **Linear algebra** (Chapter 3): positive semidefiniteness characterizes convex functions via the Hessian; eigenvalues give the strong convexity and smoothness constants; SVD appears in nuclear norm and proximal operators
- **Calculus** (Chapters 4–5): gradients define the first-order optimality condition; Hessians define the second-order condition; Taylor expansion motivates the descent lemma and smoothness
- **Probability** (Chapter 6): the probability simplex is a fundamental convex set; KL divergence is convex; log-likelihood of exponential families is concave

### Forward Connections

This section enables:

- **Gradient descent** (§02): convergence proofs depend on smoothness ($L$) and strong convexity ($\mu$); the condition number $\kappa = L/\mu$ appears in every convergence bound
- **Second-order methods** (§03): Newton's method exploits the Hessian to reduce the effective condition number to 1; quasi-Newton methods approximate this
- **Constrained optimization** (§04): KKT conditions build on the Lagrangian and duality theory introduced here; Slater's condition ensures strong duality
- **Optimization landscape** (§06): the gap between convex guarantees and nonconvex reality drives the study of loss surface geometry, saddle points, and flat minima
- **Regularization** (§08): L1, L2, nuclear norm penalties are convex; their proximal operators from §9.1 are the computational tools for non-smooth regularization
- **Information theory** (Chapter 9): cross-entropy and KL divergence are convex, connecting optimization to information-theoretic quantities

### What Convexity Gives You vs. What It Doesn't

| What Convexity Gives You | What It Doesn't Give You |
| --- | --- |
| Every local min is global | A tractable representation of the global landscape |
| Polynomial-time algorithms exist | Fast algorithms for all problem sizes |
| Duality provides certificates | Duality for nonconvex problems |
| Condition number predicts difficulty | Guarantees for neural network training |
| Strong convexity → unique solution | Uniqueness for merely convex problems |
| Proximal operators for non-smooth terms | Proximal operators for arbitrary functions |

The honest assessment: convex optimization is the foundation, but modern deep learning transcends it. The concepts (gradient, condition number, duality, regularization) carry over, even though the guarantees do not. Understanding convexity tells you what you _should_ expect in the ideal case — and helps you diagnose what goes wrong when the nonconvex reality deviates from it.

### How This Section Connects to the Rest of Chapter 8

Each subsequent section in the Optimization chapter builds directly on specific results from this section:

| Subsequent Section | What It Uses from Here |
| --- | --- |
| §02 Gradient Descent | Smoothness ($L$), strong convexity ($\mu$), descent lemma, convergence rate framework |
| §03 Second-Order Methods | Hessian PSD condition, condition number $\kappa$, Newton step as Hessian preconditioning |
| §04 Constrained Optimization | Lagrangian, dual function, Slater's condition, weak/strong duality |
| §05 Stochastic Optimization | Smoothness for bounding variance, strong convexity for convergence guarantees |
| §06 Optimization Landscape | Convex vs. nonconvex contrast; Hessian spectrum analysis; saddle points as non-PSD Hessian |
| §07 Adaptive Learning Rate | Condition number motivation for per-parameter adaptation; preconditioning theory |
| §08 Regularization | L1/L2 as convex penalties; proximal operators; Bayesian-regularization duality |
| §09 Hyperparameter Optimization | Bayesian optimization uses GP posterior (convex in kernel parameters) |
| §10 Learning Rate Schedules | Smoothness determines max safe $\eta$; warmup addresses initial high curvature |

### The Big Picture
```
CONVEX OPTIMIZATION IN THE CURRICULUM
════════════════════════════════════════════════════════════════════════

  Chapter 3                Chapter 4–5              Chapter 6
  Linear Algebra           Calculus                 Probability
  ┌─────────────┐         ┌─────────────┐         ┌───────────────┐
  │ Eigenvalues  │         │ Gradients    │         │ Simplex       │
  │ PSD matrices │         │ Hessians     │         │ KL divergence │
  │ SVD, norms   │         │ Taylor exp.  │         │ Log-lik.      │
  └──────┬───────┘         └──────┬───────┘         └──────┬────────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │  §01 CONVEX OPTIMIZATION      │
                   │  Sets, functions, duality,     │
                   │  problem classes, conditions   │
                   └──────────────┬────────────────┘
                                  │
          ┌───────────┬───────────┼───────────┬──────────────┐
          ▼           ▼           ▼           ▼              ▼
     ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────────┐
     │ §02 GD  │ │ §03 2nd │ │ §04 Con-│ │ §06 Land│ │ §08 Reg. │
     │ Conv.   │ │ Order   │ │ strained│ │ -scape  │ │ Methods  │
     │ rates   │ │ Newton  │ │ KKT     │ │ Saddles │ │ L1/L2    │
     └─────────┘ └─────────┘ └─────────┘ └─────────┘ └──────────┘

════════════════════════════════════════════════════════════════════════
```

---

## Key Takeaways

Before moving to [Gradient Descent](../02-Gradient-Descent/notes.md), ensure you can answer these questions:

1. **Can you verify convexity?** Given a function, can you check the Hessian PSD condition or apply the preservation rules to determine whether it is convex?

2. **Can you compute the condition number?** Given a quadratic or regularized loss, can you find $\mu$, $L$, and $\kappa = L/\mu$?

3. **Can you set up the Lagrangian?** Given a constrained problem, can you form $\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda})$ and derive the dual?

4. **Can you classify the problem?** Is a given optimization problem an LP, QP, SOCP, or SDP?

5. **Can you identify convex relaxations?** Given a nonconvex problem (rank constraint, integer constraint), can you propose the appropriate convex relaxation?

6. **Can you connect to ML?** For a given ML model (logistic regression, SVM, Ridge, Lasso), can you identify why it is convex and what the condition number depends on?

These are the questions that the [exercises](exercises.ipynb) will test. The [theory notebook](theory.ipynb) provides interactive verification of all major results.

**Self-assessment checklist:**

- [ ] I can state the definition of a convex set and give 5 examples / 3 non-examples
- [ ] I can verify convexity of a function using first- and second-order conditions
- [ ] I know which operations preserve convexity (nonneg. sum, affine composition, pointwise max, partial min) and which do not (general composition, union, product)
- [ ] I can compute $\mu$, $L$, and $\kappa$ for a quadratic and for regularized logistic regression
- [ ] I understand the descent lemma and why $\eta \leq 1/L$ is required
- [ ] I can classify a problem as LP, QP, SOCP, or SDP
- [ ] I can form a Lagrangian, derive the dual, and apply Slater's condition
- [ ] I can explain the SVM dual and why the kernel trick works
- [ ] I can identify which ML loss functions are convex (and in which variables)
- [ ] I can explain what a proximal operator does and compute it for $\ell_1$ and nuclear norm
- [ ] I understand the Fenchel conjugate and the biconjugate theorem

---

## Summary of Key Formulas

| Concept | Formula | Section |
| --- | --- | --- |
| Convex set definition | $\theta\mathbf{x} + (1-\theta)\mathbf{y} \in \mathcal{C}$ | §2.1 |
| Convex function definition | $f(\theta\mathbf{x} + (1-\theta)\mathbf{y}) \leq \theta f(\mathbf{x}) + (1-\theta)f(\mathbf{y})$ | §3.1 |
| First-order condition | $f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^\top(\mathbf{y} - \mathbf{x})$ | §3.1 |
| Second-order condition | $\nabla^2 f(\mathbf{x}) \succeq 0$ | §3.2 |
| Strong convexity | $\nabla^2 f(\mathbf{x}) \succeq \mu I$ | §4.1 |
| Smoothness | $\nabla^2 f(\mathbf{x}) \preceq LI$ | §4.2 |
| Condition number | $\kappa = L/\mu$ | §4.3 |
| Descent lemma | $f(\mathbf{x} - \eta\nabla f) \leq f(\mathbf{x}) - \frac{\eta}{2}\lVert\nabla f\rVert^2$ | §4.2 |
| Lagrangian | $\mathcal{L} = f_0 + \sum_i \lambda_i f_i + \sum_j \nu_j h_j$ | §6.2 |
| Weak duality | $g^* \leq f^*$ | §7.2 |
| Proximal operator | $\operatorname{prox}_{\eta h}(\mathbf{v}) = \arg\min \{h(\mathbf{x}) + \frac{1}{2\eta}\lVert\mathbf{x} - \mathbf{v}\rVert^2\}$ | §9.1 |
| Fenchel conjugate | $f^*(\mathbf{y}) = \sup_{\mathbf{x}}\{\mathbf{y}^\top\mathbf{x} - f(\mathbf{x})\}$ | §9.2 |

---

## Notation Summary

| Symbol | Meaning | First Appears |
| --- | --- | --- |
| $\mathcal{C}$ | Convex set | §2.1 |
| $\mathcal{F}$ | Feasible set | §5.1 |
| $f^*$ | Optimal value | §5.1 |
| $\mathbf{x}^*$ | Optimal solution | §5.1 |
| $\mu$ | Strong convexity constant | §4.1 |
| $L$ | Smoothness (Lipschitz gradient) constant | §4.2 |
| $\kappa$ | Condition number $L/\mu$ | §4.3 |
| $\boldsymbol{\lambda}$ | Lagrange multipliers (dual variables) | §6.2 |
| $\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda})$ | Lagrangian function | §6.2 |
| $g(\boldsymbol{\lambda})$ | Dual function | §7.1 |
| $\mathbb{S}^n_+$ | PSD cone ($n \times n$ symmetric PSD matrices) | §2.4 |
| $\lVert W\rVert_*$ | Nuclear norm (sum of singular values) | §8.3 |
| $\partial f(\mathbf{x})$ | Subdifferential (set of subgradients) | §6.1 |
| $\operatorname{prox}_h$ | Proximal operator of $h$ | §9.1 |
| $f^*$ | Fenchel conjugate (context-dependent) | §9.2 |
| $D_\phi$ | Bregman divergence with mirror map $\phi$ | §9.3 |

---

## References

1. Boyd, S. & Vandenberghe, L. (2004). _Convex Optimization_. Cambridge University Press.
2. Rockafellar, R. T. (1970). _Convex Analysis_. Princeton University Press.
3. Nesterov, Y. (2004). _Introductory Lectures on Convex Optimization_. Springer.
4. Nocedal, J. & Wright, S. J. (2006). _Numerical Optimization_. Springer.
5. Bubeck, S. (2015). "Convex Optimization: Algorithms and Complexity." _Foundations and Trends in Machine Learning_.
6. Parikh, N. & Boyd, S. (2014). "Proximal Algorithms." _Foundations and Trends in Optimization_.
7. Beck, A. & Teboulle, M. (2009). "A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems." _SIAM J. Imaging Sciences_.
8. Hu, E. J. et al. (2022). "LoRA: Low-Rank Adaptation of Large Language Models." _ICLR_.
9. Loshchilov, I. & Hutter, F. (2019). "Decoupled Weight Decay Regularization." _ICLR_.
10. Fazel, M. (2002). "Matrix Rank Minimization with Applications." PhD Thesis, Stanford.
11. Goemans, M. X. & Williamson, D. P. (1995). "Improved Approximation Algorithms for Maximum Cut." _JACM_.
12. Diamond, S. & Boyd, S. (2016). "CVXPY: A Python-Embedded Modeling Language for Convex Optimization." _JMLR_.
13. Santurkar, S. et al. (2018). "How Does Batch Normalization Help Optimization?" _NeurIPS_.
14. Liu, S. et al. (2024). "DoRA: Weight-Decomposed Low-Rank Adaptation." _ICML_.
15. Bousquet, O. & Elisseeff, A. (2002). "Stability and Generalization." _JMLR_.
16. Donoho, D. (2006). "Compressed Sensing." _IEEE Trans. Information Theory_.
17. Candès, E. & Tao, T. (2005). "Decoding by Linear Programming." _IEEE Trans. Information Theory_.
18. Vapnik, V. (1995). _The Nature of Statistical Learning Theory_. Springer.
19. Platt, J. (1998). "Sequential Minimal Optimization." _Microsoft Research Technical Report_.
20. Yang, Z. et al. (2018). "Breaking the Softmax Bottleneck." _ICLR_.
21. Friedman, J., Hastie, T. & Tibshirani, R. (2010). "Regularization Paths for Generalized Linear Models via Coordinate Descent." _J. Statistical Software_.
22. Schulman, J. et al. (2015). "Trust Region Policy Optimization." _ICML_.
23. Sanh, V. et al. (2020). "Movement Pruning: Adaptive Sparsity during Fine-Tuning." _NeurIPS_.
24. Amari, S. (1998). "Natural Gradient Works Efficiently in Learning." _Neural Computation_.
25. Nguyen, X. et al. (2010). "Estimating Divergence Functionals and the Likelihood Ratio by Convex Risk Minimization." _IEEE Trans. Information Theory_.
26. Candès, E. et al. (2011). "Robust Principal Component Analysis?" _JACM_.

---

[← Back to Optimization](../README.md) | [Next: Gradient Descent →](../02-Gradient-Descent/notes.md)
