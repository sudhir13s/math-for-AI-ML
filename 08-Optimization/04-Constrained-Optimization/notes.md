[← Back to Optimization](../README.md) | [Next: Stochastic Optimization →](../05-Stochastic-Optimization/notes.md)

---

# Constrained Optimization

> _"The world is full of constraints. The art of optimization is not just finding the lowest point — it's finding the lowest point that you're actually allowed to visit."_ — Stephen Boyd

## Overview

Constrained optimization is the mathematical framework for finding the minimum of an objective function subject to restrictions on the feasible set of solutions. Unlike unconstrained optimization, where the algorithm can move freely in any direction, constrained optimization must navigate the boundary between feasible and infeasible regions — a fundamentally more complex geometric challenge.

Constraints appear everywhere in machine learning and AI. Support vector machines maximize the margin subject to classification constraints. Portfolio optimization maximizes return subject to risk and budget constraints. Non-negative matrix factorization requires all factors to be non-negative. Reinforcement learning with safety constraints must satisfy expected cost bounds. Even the simplex constraint on probability distributions ($\sum p_i = 1, p_i \geq 0$) is a constrained optimization problem at its core.

The mathematical theory of constrained optimization is elegant and powerful. The **Lagrangian** transforms a constrained problem into an unconstrained saddle-point problem. The **KKT conditions** provide necessary (and under convexity, sufficient) conditions for optimality. **Duality theory** reveals that every constrained optimization problem has a dual problem whose solution provides bounds on the primal and interprets the Lagrange multipliers as shadow prices — the marginal value of relaxing each constraint.

This section develops the full theory from equality constraints (§3) through inequality constraints and the KKT conditions (§4), duality theory (§5), and practical algorithms including projected gradient descent, penalty methods, barrier methods, and ADMM (§6). Advanced topics cover SQP, primal-dual interior-point methods, and conic programming (§7). Every concept is connected to its role in modern ML systems, with a detailed treatment of the SVM dual problem as the canonical example (§8).

**Scope note.** This section covers the mathematical theory and algorithms for constrained optimization. The _Lagrangian duality_ is previewed in [Convex Optimization](../01-Convex-Optimization/notes.md) as part of the convex analysis toolkit; here we develop the full KKT machinery and algorithmic framework. _Projected gradient descent_ connects to [Stochastic Optimization](../05-Stochastic-Optimization/notes.md) when combined with noisy gradients. The _SVM dual problem_ connects to [Regularization Methods](../08-Regularization-Methods/notes.md) through the hinge loss interpretation.

**For AI practitioners:** By the end of this section, you will understand how to formulate ML problems with constraints, derive and interpret the KKT conditions, solve the SVM dual problem, implement projected gradient descent for constrained neural network training, and use ADMM for distributed optimization with consensus constraints.

## Prerequisites

- **Gradients and Hessians** — $\nabla f$, $\nabla^2 f$, Taylor expansion — [Chapter 5: Multivariate Calculus](../../05-Multivariate-Calculus/README.md)
- **Convexity and strong convexity** — convex sets, convex functions, strong convexity, smoothness — [Chapter 8 §01](../01-Convex-Optimization/notes.md)
- **Gradient descent convergence theory** — convergence rates, condition number — [Chapter 8 §02](../02-Gradient-Descent/notes.md)
- **Lagrangian duality preview** — weak/strong duality, Slater's condition — [Chapter 8 §01](../01-Convex-Optimization/notes.md#7-duality)
- **Eigenvalues and positive definiteness** — $A \succ 0$, spectral theorem — [Chapter 3 §01](../../03-Advanced-Linear-Algebra/01-Eigenvalues-and-Eigenvectors/notes.md)

## Companion Notebooks

| Notebook | Description |
|---|---|
| [theory.ipynb](theory.ipynb) | Interactive demonstrations of KKT conditions, projected GD, penalty methods, barrier methods, and SVM dual |
| [exercises.ipynb](exercises.ipynb) | 8 graded exercises from Lagrange multipliers to ADMM for distributed optimization |

## Learning Objectives

After completing this section, you will:

1. Formulate constrained optimization problems with equality and inequality constraints
2. Derive the Lagrangian and interpret Lagrange multipliers as shadow prices
3. State and apply the KKT conditions for optimality
4. Verify constraint qualifications (LICQ, Slater's condition) for specific problems
5. Derive and solve the dual problem for convex optimization
6. Implement projected gradient descent and compute projections onto common sets
7. Apply penalty and barrier methods to transform constrained problems into unconstrained ones
8. Derive the SVM dual problem from the primal and interpret the support vectors
9. Implement ADMM for distributed optimization with consensus constraints
10. Analyze sequential quadratic programming and primal-dual interior-point methods
11. Formulate conic programs (SOCP, SDP) and recognize their structure in ML problems
12. Connect constrained optimization to regularization, proximal methods, and constrained RL

---

## Table of Contents

- [Constrained Optimization](#constrained-optimization)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
  - [Companion Notebooks](#companion-notebooks)
  - [Learning Objectives](#learning-objectives)
  - [Table of Contents](#table-of-contents)
  - [1. Intuition](#1-intuition)
    - [1.1 Why Constraints Matter in Optimization](#11-why-constraints-matter-in-optimization)
    - [1.2 The Geometry of Feasible Sets](#12-the-geometry-of-feasible-sets)
    - [1.3 From Unconstrained to Constrained: The Lagrangian Idea](#13-from-unconstrained-to-constrained-the-lagrangian-idea)
    - [1.4 Historical Timeline: Lagrange to Modern Interior-Point Methods](#14-historical-timeline-lagrange-to-modern-interior-point-methods)
    - [1.5 Constraints in Machine Learning](#15-constraints-in-machine-learning)
  - [2. Formal Definitions](#2-formal-definitions)
    - [2.1 Constrained Optimization Problem Statement](#21-constrained-optimization-problem-statement)
    - [2.2 Feasible Sets and Active Constraints](#22-feasible-sets-and-active-constraints)
    - [2.3 The Lagrangian Function](#23-the-lagrangian-function)
    - [2.4 Lagrange Multipliers: Geometric Interpretation](#24-lagrange-multipliers-geometric-interpretation)
    - [2.5 Regularity Conditions (Constraint Qualifications)](#25-regularity-conditions-constraint-qualifications)
  - [3. Equality-Constrained Optimization](#3-equality-constrained-optimization)
    - [3.1 Lagrange Multiplier Method](#31-lagrange-multiplier-method)
    - [3.2 First-Order Necessary Conditions](#32-first-order-necessary-conditions)
    - [3.3 Second-Order Conditions for Equality Constraints](#33-second-order-conditions-for-equality-constraints)
    - [3.4 Newton's Method for Equality-Constrained Problems](#34-newtons-method-for-equality-constrained-problems)
    - [3.5 Applications: Minimum-Norm Solutions, Portfolio Optimization](#35-applications-minimum-norm-solutions-portfolio-optimization)
  - [4. Inequality-Constrained Optimization: KKT Conditions](#4-inequality-constrained-optimization-kkt-conditions)
    - [4.1 Karush-Kuhn-Tucker Conditions](#41-karush-kuhn-tucker-conditions)
    - [4.2 Primal and Dual Feasibility](#42-primal-and-dual-feasibility)
    - [4.3 Complementary Slackness](#43-complementary-slackness)
    - [4.4 Second-Order Sufficient Conditions](#44-second-order-sufficient-conditions)
    - [4.5 Constraint Qualifications: LICQ, Slater's Condition](#45-constraint-qualifications-licq-slaters-condition)
  - [5. Duality Theory](#5-duality-theory)
    - [5.1 The Dual Problem](#51-the-dual-problem)
    - [5.2 Weak Duality and the Duality Gap](#52-weak-duality-and-the-duality-gap)
    - [5.3 Strong Duality and Slater's Condition](#53-strong-duality-and-slaters-condition)
    - [5.4 Interpreting Dual Variables as Shadow Prices](#54-interpreting-dual-variables-as-shadow-prices)
    - [5.5 Duality in SVM: From Primal to Dual](#55-duality-in-svm-from-primal-to-dual)
  - [6. Algorithms for Constrained Optimization](#6-algorithms-for-constrained-optimization)
    - [6.1 Projected Gradient Descent](#61-projected-gradient-descent)
    - [6.2 Projection onto Common Sets](#62-projection-onto-common-sets)
    - [6.3 Penalty Methods](#63-penalty-methods)
    - [6.4 Barrier (Interior-Point) Methods](#64-barrier-interior-point-methods)
    - [6.5 Augmented Lagrangian Method](#65-augmented-lagrangian-method)
  - [7. Advanced Topics](#7-advanced-topics)
    - [7.1 Sequential Quadratic Programming (SQP)](#71-sequential-quadratic-programming-sqp)
    - [7.2 Primal-Dual Interior-Point Methods](#72-primal-dual-interior-point-methods)
    - [7.3 ADMM: Alternating Direction Method of Multipliers](#73-admm-alternating-direction-method-of-multipliers)
    - [7.4 Conic Programming: SOCP and SDP](#74-conic-programming-socp-and-sdp)
    - [7.5 Connection to Proximal Methods](#75-connection-to-proximal-methods)
  - [8. Applications in Machine Learning](#8-applications-in-machine-learning)
    - [8.1 Support Vector Machines: Primal and Dual](#81-support-vector-machines-primal-and-dual)
    - [8.2 Constrained Logistic Regression](#82-constrained-logistic-regression)
    - [8.3 Non-Negative Matrix Factorization](#83-non-negative-matrix-factorization)
    - [8.4 Trust Region Subproblems](#84-trust-region-subproblems)
    - [8.5 Constrained Reinforcement Learning](#85-constrained-reinforcement-learning)
  - [9. Common Mistakes](#9-common-mistakes)
  - [10. Exercises](#10-exercises)
  - [11. Why This Matters for AI (2026 Perspective)](#11-why-this-matters-for-ai-2026-perspective)
  - [12. Conceptual Bridge](#12-conceptual-bridge)
  - [References](#references)

---

## 1. Intuition

### 1.1 Why Constraints Matter in Optimization

In unconstrained optimization, the algorithm searches the entire space $\mathbb{R}^n$ for the minimum. In constrained optimization, the search is restricted to a **feasible set** $\mathcal{C} \subseteq \mathbb{R}^n$ defined by the constraints. This restriction fundamentally changes the nature of the problem.

**The key insight:** At the optimal point, the gradient of the objective function must be "balanced" by the gradients of the active constraints. If the unconstrained minimum lies inside the feasible set, the constraints are inactive and the solution is the same as the unconstrained problem. But if the unconstrained minimum lies outside the feasible set, the constrained optimum lies on the boundary, and the gradient of the objective points outward — it is exactly counterbalanced by a linear combination of the constraint gradients.

This balancing act is captured by the **Lagrange multipliers**: each constraint $g_i(\mathbf{x}) \leq 0$ is assigned a multiplier $\lambda_i \geq 0$ that measures how much the optimal value would improve if the constraint were relaxed. A large multiplier means the constraint is "tight" — relaxing it would significantly improve the objective. A zero multiplier means the constraint is inactive — it doesn't affect the solution.

### 1.2 The Geometry of Feasible Sets

```
CONSTRAINED OPTIMIZATION GEOMETRY
════════════════════════════════════════════════════════════════════════

  Case 1: Unconstrained minimum is feasible    Case 2: Constrained optimum on boundary

       ╱╲    ╱╲                                     ╲         ╱
      ╱  ╲  ╱  ╲   Feasible set (circle)             ╲       ╱
     ╱    ╲●    ╲  ╱────╲                             ╲  ●  ╱  ← constrained min
    ╱      ╲    ╲╱      ╲                             ╲ ╱╲ ╱
   ╲        ╲    ● unconstrained min                  ╲╱  ╲╱
    ╲        ╲  ╱ feasible                            ╲   ╱
     ╲________╲╱                                       ╲ ╱
                                                        ╲╱
   ∇f(x*) = 0                                         ∇f(x*) = -Σ λ_i ∇g_i(x*)
   No constraints active                              Constraints balance the gradient

════════════════════════════════════════════════════════════════════════
```

The geometric picture reveals why constrained optimization is harder: the solution can be either in the interior (where $\nabla f = \mathbf{0}$) or on the boundary (where $\nabla f$ is a linear combination of constraint gradients). The algorithm must determine which case applies, and if on the boundary, which constraints are active.

### 1.3 From Unconstrained to Constrained: The Lagrangian Idea

The Lagrangian transforms a constrained problem into an unconstrained saddle-point problem. For the problem:

$$\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{s.t.} \quad g_i(\mathbf{x}) \leq 0, \; i = 1, \ldots, m$$

The Lagrangian is:

$$\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}) = f(\mathbf{x}) + \sum_{i=1}^m \lambda_i g_i(\mathbf{x})$$

where $\boldsymbol{\lambda} \geq \mathbf{0}$ are the Lagrange multipliers.

**The intuition:** The Lagrangian penalizes constraint violations. If $g_i(\mathbf{x}) > 0$ (constraint violated), the term $\lambda_i g_i(\mathbf{x})$ increases the Lagrangian, making the point less attractive. If $g_i(\mathbf{x}) \leq 0$ (constraint satisfied), the term is non-positive (since $\lambda_i \geq 0$), so it doesn't penalize the point.

The optimal solution $(\mathbf{x}^*, \boldsymbol{\lambda}^*)$ is a **saddle point** of the Lagrangian: it minimizes $\mathcal{L}$ with respect to $\mathbf{x}$ and maximizes $\mathcal{L}$ with respect to $\boldsymbol{\lambda}$. This saddle-point structure is the foundation of duality theory and many constrained optimization algorithms.

### 1.4 Historical Timeline: Lagrange to Modern Interior-Point Methods

```
CONSTRAINED OPTIMIZATION TIMELINE
════════════════════════════════════════════════════════════════════════

  1760  Lagrange        — Method of Lagrange multipliers
  1847  Gauss           — Least squares with constraints
  1939  Karush          — KKT conditions (Master's thesis, overlooked)
  1948  Kuhn & Tucker   — KKT conditions (rediscovered, popularized)
  1951  Fritz John      — Fritz John conditions (weaker constraint qual.)
  1950s Dantzig         — Simplex method for linear programming
  1960s Fiacco &        — Sequential unconstrained minimization
        McCormick         (SUMT: penalty and barrier methods)
  1984  Karmarkar       — Interior-point method for LP (polynomial time)
  1980s-90s Nesterov &  — Self-concordant barriers, primal-dual
        Nemirovski          interior-point methods
  1990s Vapnik          — SVM: constrained optimization meets ML
  2000s Boyd &          — Convex optimization textbook; CVX modeling
        Vandenberghe
  2008  Boyd et al.     — ADMM: distributed convex optimization
  2010s Modern ML       — Projected GD, constrained RL, fair ML

════════════════════════════════════════════════════════════════════════
```

### 1.5 Constraints in Machine Learning

Constraints appear throughout ML at every level:

| Constraint Type | ML Application | Mathematical Form |
|---|---|---|
| **Equality** | Probability normalization ($\sum p_i = 1$) | $\mathbf{1}^\top \mathbf{p} = 1$ |
| **Equality** | Portfolio budget ($\sum w_i = 1$) | $\mathbf{1}^\top \mathbf{w} = 1$ |
| **Inequality** | SVM margin constraints | $y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1$ |
| **Inequality** | Non-negativity (NMF, topic models) | $W \geq 0, H \geq 0$ |
| **Inequality** | Risk constraints (portfolio, safety) | $\mathbf{w}^\top \Sigma \mathbf{w} \leq \sigma_{\max}^2$ |
| **Inequality** | Fairness constraints (demographic parity) | $|P(\hat{y}=1 \mid A=0) - P(\hat{y}=1 \mid A=1)| \leq \epsilon$ |
| **Norm ball** | Trust regions, robust optimization | $\lVert \boldsymbol{\theta} - \boldsymbol{\theta}_0 \rVert \leq \Delta$ |
| **Simplex** | Mixture models, attention weights | $\boldsymbol{\alpha} \in \Delta_n$ |

**For AI:** Understanding constrained optimization is essential for formulating ML problems correctly. Many "unconstrained" ML problems are actually constrained in disguise — the softmax function implicitly enforces the simplex constraint, and weight decay can be viewed as a trust region constraint via duality.

---

## 2. Formal Definitions

### 2.1 Constrained Optimization Problem Statement

**Standard form:**

$$\begin{aligned}
\min_{\mathbf{x} \in \mathbb{R}^n} \quad & f(\mathbf{x}) \\
\text{s.t.} \quad & g_i(\mathbf{x}) \leq 0, \quad i = 1, \ldots, m \\
& h_j(\mathbf{x}) = 0, \quad j = 1, \ldots, p
\end{aligned}$$

where:
- $f: \mathbb{R}^n \to \mathbb{R}$ is the **objective function**
- $g_i: \mathbb{R}^n \to \mathbb{R}$ are the **inequality constraint functions**
- $h_j: \mathbb{R}^n \to \mathbb{R}$ are the **equality constraint functions**
- $\mathbf{x} \in \mathbb{R}^n$ is the **optimization variable**

**Feasible set:** The set of all points satisfying the constraints:

$$\mathcal{F} = \{\mathbf{x} \in \mathbb{R}^n : g_i(\mathbf{x}) \leq 0 \; \forall i, \; h_j(\mathbf{x}) = 0 \; \forall j\}$$

**Optimal value:** $p^* = \inf\{f(\mathbf{x}) : \mathbf{x} \in \mathcal{F}\}$, with $p^* = +\infty$ if $\mathcal{F} = \emptyset$ (infeasible).

**Convex optimization problem:** The problem is convex if $f$ and all $g_i$ are convex functions and all $h_j$ are affine functions ($h_j(\mathbf{x}) = \mathbf{a}_j^\top \mathbf{x} - b_j$). In this case, the feasible set $\mathcal{F}$ is convex.

### 2.2 Feasible Sets and Active Constraints

**Definition (Active Constraint).** An inequality constraint $g_i$ is **active** at $\mathbf{x}$ if $g_i(\mathbf{x}) = 0$. It is **inactive** if $g_i(\mathbf{x}) < 0$.

The **active set** at $\mathbf{x}$ is $\mathcal{A}(\mathbf{x}) = \{i : g_i(\mathbf{x}) = 0\}$.

**Why active constraints matter:** At the optimum, only the active constraints influence the solution. The inactive constraints can be ignored locally — they don't affect the optimality conditions. This is the basis of **active set methods**, which iteratively guess the active set and solve the resulting equality-constrained subproblem.

**Definition (Strict Feasibility).** A point $\mathbf{x}$ is **strictly feasible** if $g_i(\mathbf{x}) < 0$ for all $i$ (all inequality constraints are strictly satisfied) and $h_j(\mathbf{x}) = 0$ for all $j$.

Strict feasibility is important because it ensures that Slater's condition holds (for convex problems), which guarantees strong duality.

### 2.3 The Lagrangian Function

**Definition (Lagrangian).** For the standard form problem, the Lagrangian is:

$$\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = f(\mathbf{x}) + \sum_{i=1}^m \lambda_i g_i(\mathbf{x}) + \sum_{j=1}^p \nu_j h_j(\mathbf{x})$$

where $\boldsymbol{\lambda} \in \mathbb{R}^m$ are the **inequality multipliers** (dual variables) and $\boldsymbol{\nu} \in \mathbb{R}^p$ are the **equality multipliers**.

**Key properties:**

1. **Lower bound property:** For any $\boldsymbol{\lambda} \geq \mathbf{0}$ and any $\boldsymbol{\nu}$:

$$\inf_{\mathbf{x}} \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) \leq p^*$$

This is because for any feasible $\mathbf{x}$, we have $g_i(\mathbf{x}) \leq 0$ and $h_j(\mathbf{x}) = 0$, so $\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) \leq f(\mathbf{x})$.

2. **Saddle-point property:** $(\mathbf{x}^*, \boldsymbol{\lambda}^*, \boldsymbol{\nu}^*)$ is optimal if and only if:

$$\mathcal{L}(\mathbf{x}^*, \boldsymbol{\lambda}, \boldsymbol{\nu}) \leq \mathcal{L}(\mathbf{x}^*, \boldsymbol{\lambda}^*, \boldsymbol{\nu}^*) \leq \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}^*, \boldsymbol{\nu}^*)$$

for all $\mathbf{x}$, $\boldsymbol{\lambda} \geq \mathbf{0}$, and $\boldsymbol{\nu}$.

### 2.4 Lagrange Multipliers: Geometric Interpretation

At the optimal point $\mathbf{x}^*$, the gradient of the objective is a linear combination of the gradients of the active constraints:

$$\nabla f(\mathbf{x}^*) + \sum_{i=1}^m \lambda_i^* \nabla g_i(\mathbf{x}^*) + \sum_{j=1}^p \nu_j^* \nabla h_j(\mathbf{x}^*) = \mathbf{0}$$

This means $\nabla f(\mathbf{x}^*)$ lies in the **normal cone** of the feasible set at $\mathbf{x}^*$ — the cone spanned by the outward normals of the active constraints.

**Shadow price interpretation:** The multiplier $\lambda_i^*$ measures the sensitivity of the optimal value to the $i$-th constraint:

$$\lambda_i^* = -\frac{\partial p^*}{\partial b_i}$$

where $b_i$ is the right-hand side of the constraint $g_i(\mathbf{x}) \leq b_i$. A large $\lambda_i^*$ means the constraint is "expensive" — relaxing it would significantly improve the objective.

**For AI:** In SVM, the Lagrange multipliers $\alpha_i$ are exactly the shadow prices of the margin constraints. Points with $\alpha_i > 0$ are the **support vectors** — they are the "expensive" constraints that determine the decision boundary. Points with $\alpha_i = 0$ are not support vectors — their constraints are inactive and they don't affect the solution.

### 2.5 Regularity Conditions (Constraint Qualifications)

The KKT conditions are necessary for optimality only under certain **constraint qualifications** that ensure the constraints are "well-behaved" at the solution.

**Linear Independence Constraint Qualification (LICQ):** The gradients of the active constraints are linearly independent:

$$\{\nabla g_i(\mathbf{x}^*) : i \in \mathcal{A}(\mathbf{x}^*)\} \cup \{\nabla h_j(\mathbf{x}^*) : j = 1, \ldots, p\} \text{ are linearly independent}$$

**Slater's Condition (for convex problems):** There exists a strictly feasible point:

$$\exists \mathbf{x} \text{ such that } g_i(\mathbf{x}) < 0 \; \forall i, \; h_j(\mathbf{x}) = 0 \; \forall j$$

Slater's condition is the most important constraint qualification for convex optimization. It guarantees strong duality and ensures that the KKT conditions are necessary and sufficient for optimality.

**Mangasarian-Fromovitz Constraint Qualification (MFCQ):** A weaker condition than LICQ that requires the existence of a direction that strictly decreases all active inequality constraints while maintaining equality constraints.

**For AI:** In most ML applications, the constraints are simple (non-negativity, norm balls, simplex) and Slater's condition is easily verified. For example, the SVM primal always satisfies Slater's condition as long as the data is not perfectly separable with zero margin.


---

## 3. Equality-Constrained Optimization

### 3.1 Lagrange Multiplier Method

Consider the equality-constrained problem:

$$\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{s.t.} \quad A\mathbf{x} = \mathbf{b}$$

where $A \in \mathbb{R}^{p \times n}$ has full row rank ($p \leq n$).

The Lagrangian is:

$$\mathcal{L}(\mathbf{x}, \boldsymbol{\nu}) = f(\mathbf{x}) + \boldsymbol{\nu}^\top(A\mathbf{x} - \mathbf{b})$$

The optimality conditions are:

$$\nabla_{\mathbf{x}} \mathcal{L} = \nabla f(\mathbf{x}) + A^\top \boldsymbol{\nu} = \mathbf{0}$$
$$\nabla_{\boldsymbol{\nu}} \mathcal{L} = A\mathbf{x} - \mathbf{b} = \mathbf{0}$$

These are $n + p$ equations in $n + p$ unknowns $(\mathbf{x}, \boldsymbol{\nu})$.

**Example: Minimum-norm solution.** For $f(\mathbf{x}) = \frac{1}{2}\lVert \mathbf{x} \rVert_2^2$:

$$\nabla f(\mathbf{x}) = \mathbf{x} \quad \Rightarrow \quad \mathbf{x} + A^\top \boldsymbol{\nu} = \mathbf{0} \quad \Rightarrow \quad \mathbf{x} = -A^\top \boldsymbol{\nu}$$

Substituting into the constraint: $A(-A^\top \boldsymbol{\nu}) = \mathbf{b}$, so $\boldsymbol{\nu} = -(AA^\top)^{-1}\mathbf{b}$ and:

$$\mathbf{x}^* = A^\top(AA^\top)^{-1}\mathbf{b} = A^\dagger \mathbf{b}$$

This is the minimum-norm solution to the underdetermined system $A\mathbf{x} = \mathbf{b}$.

### 3.2 First-Order Necessary Conditions

**Theorem (First-Order Necessary Conditions for Equality Constraints).** Let $\mathbf{x}^*$ be a local minimum of $f$ subject to $h_j(\mathbf{x}) = 0$, $j = 1, \ldots, p$. If the gradients $\nabla h_j(\mathbf{x}^*)$ are linearly independent (LICQ), then there exist unique multipliers $\boldsymbol{\nu}^*$ such that:

$$\nabla f(\mathbf{x}^*) + \sum_{j=1}^p \nu_j^* \nabla h_j(\mathbf{x}^*) = \mathbf{0}$$
$$h_j(\mathbf{x}^*) = 0, \quad j = 1, \ldots, p$$

**Proof sketch.** The feasible set near $\mathbf{x}^*$ is a manifold of dimension $n - p$. The tangent space at $\mathbf{x}^*$ is the null space of the constraint Jacobian: $T = \{\mathbf{d} : \nabla h_j(\mathbf{x}^*)^\top \mathbf{d} = 0 \; \forall j\}$. For $\mathbf{x}^*$ to be a local minimum, the directional derivative of $f$ must be zero in all feasible directions: $\nabla f(\mathbf{x}^*)^\top \mathbf{d} = 0$ for all $\mathbf{d} \in T$. This means $\nabla f(\mathbf{x}^*)$ is orthogonal to the tangent space, i.e., it lies in the span of the constraint gradients. $\blacksquare$

### 3.3 Second-Order Conditions for Equality Constraints

**Theorem (Second-Order Necessary Condition).** If $\mathbf{x}^*$ is a local minimum satisfying the first-order conditions, then:

$$\mathbf{d}^\top \nabla_{\mathbf{xx}}^2 \mathcal{L}(\mathbf{x}^*, \boldsymbol{\nu}^*) \mathbf{d} \geq 0 \quad \forall \mathbf{d} \in T$$

where $T = \{\mathbf{d} : \nabla h_j(\mathbf{x}^*)^\top \mathbf{d} = 0 \; \forall j\}$ is the tangent space.

**Theorem (Second-Order Sufficient Condition).** If the first-order conditions hold and:

$$\mathbf{d}^\top \nabla_{\mathbf{xx}}^2 \mathcal{L}(\mathbf{x}^*, \boldsymbol{\nu}^*) \mathbf{d} > 0 \quad \forall \mathbf{d} \in T \setminus \{\mathbf{0}\}$$

then $\mathbf{x}^*$ is a strict local minimum.

**Key insight:** The Hessian of the Lagrangian need only be positive definite on the tangent space, not on all of $\mathbb{R}^n$. This is weaker than requiring $\nabla^2 f(\mathbf{x}^*) \succ 0$ because the constraints restrict the feasible directions.

### 3.4 Newton's Method for Equality-Constrained Problems

For the equality-constrained problem, Newton's method solves the **KKT system**:

$$\begin{bmatrix} \nabla_{\mathbf{xx}}^2 \mathcal{L}(\mathbf{x}_t, \boldsymbol{\nu}_t) & A^\top \\ A & \mathbf{0} \end{bmatrix} \begin{bmatrix} \Delta \mathbf{x} \\ \Delta \boldsymbol{\nu} \end{bmatrix} = \begin{bmatrix} -\nabla f(\mathbf{x}_t) - A^\top \boldsymbol{\nu}_t \\ -(A\mathbf{x}_t - \mathbf{b}) \end{bmatrix}$$

This is a symmetric indefinite system of size $(n+p) \times (n+p)$. It can be solved efficiently using:

1. **Direct factorization:** LDL^T factorization of the KKT matrix (cost: $O((n+p)^3)$)
2. **Schur complement:** Eliminate $\Delta \mathbf{x}$ to get a $p \times p$ system for $\Delta \boldsymbol{\nu}$ (cost: $O(n^3 + p^3)$ when $p \ll n$)
3. **Iterative methods:** MINRES or SYMMLQ for large-scale problems

**Convergence:** Newton's method for equality-constrained problems converges quadratically near the solution, just like unconstrained Newton's method.

### 3.5 Applications: Minimum-Norm Solutions, Portfolio Optimization

**Minimum-norm solution.** Given an underdetermined system $A\mathbf{x} = \mathbf{b}$ with $A \in \mathbb{R}^{p \times n}$, $p < n$, find the solution with minimum $\ell_2$ norm:

$$\min_{\mathbf{x}} \frac{1}{2}\lVert \mathbf{x} \rVert_2^2 \quad \text{s.t.} \quad A\mathbf{x} = \mathbf{b}$$

Solution: $\mathbf{x}^* = A^\top(AA^\top)^{-1}\mathbf{b} = A^\dagger \mathbf{b}$.

**Mean-variance portfolio optimization.** Given expected returns $\boldsymbol{\mu}$ and covariance $\Sigma$, find the portfolio $\mathbf{w}$ that minimizes risk subject to a target return:

$$\min_{\mathbf{w}} \frac{1}{2}\mathbf{w}^\top \Sigma \mathbf{w} \quad \text{s.t.} \quad \boldsymbol{\mu}^\top \mathbf{w} = r_{\text{target}}, \quad \mathbf{1}^\top \mathbf{w} = 1$$

The Lagrangian is $\mathcal{L}(\mathbf{w}, \nu_1, \nu_2) = \frac{1}{2}\mathbf{w}^\top \Sigma \mathbf{w} + \nu_1(\boldsymbol{\mu}^\top \mathbf{w} - r_{\text{target}}) + \nu_2(\mathbf{1}^\top \mathbf{w} - 1)$.

Setting $\nabla_{\mathbf{w}} \mathcal{L} = \Sigma \mathbf{w} + \nu_1 \boldsymbol{\mu} + \nu_2 \mathbf{1} = \mathbf{0}$ gives $\mathbf{w} = -\Sigma^{-1}(\nu_1 \boldsymbol{\mu} + \nu_2 \mathbf{1})$. Substituting into the constraints yields a $2 \times 2$ system for $(\nu_1, \nu_2)$.

---

## 4. Inequality-Constrained Optimization: KKT Conditions

### 4.1 Karush-Kuhn-Tucker Conditions

The KKT conditions are the fundamental optimality conditions for constrained optimization with inequality constraints.

**Theorem (KKT Conditions).** Consider the problem $\min f(\mathbf{x})$ s.t. $g_i(\mathbf{x}) \leq 0$, $h_j(\mathbf{x}) = 0$. If $\mathbf{x}^*$ is a local minimum and a constraint qualification holds (e.g., LICQ or Slater's condition), then there exist multipliers $\boldsymbol{\lambda}^* \geq \mathbf{0}$ and $\boldsymbol{\nu}^*$ such that:

1. **Stationarity:** $\nabla f(\mathbf{x}^*) + \sum_{i=1}^m \lambda_i^* \nabla g_i(\mathbf{x}^*) + \sum_{j=1}^p \nu_j^* \nabla h_j(\mathbf{x}^*) = \mathbf{0}$
2. **Primal feasibility:** $g_i(\mathbf{x}^*) \leq 0$, $h_j(\mathbf{x}^*) = 0$
3. **Dual feasibility:** $\lambda_i^* \geq 0$
4. **Complementary slackness:** $\lambda_i^* g_i(\mathbf{x}^*) = 0$ for all $i$

**For convex problems,** the KKT conditions are not just necessary but also **sufficient** for global optimality. If $f$ and all $g_i$ are convex, all $h_j$ are affine, and Slater's condition holds, then any point satisfying the KKT conditions is the global minimum.

### 4.2 Primal and Dual Feasibility

**Primal feasibility** means the solution satisfies all constraints. This is the most basic requirement — an infeasible point cannot be optimal.

**Dual feasibility** means the inequality multipliers are non-negative: $\lambda_i^* \geq 0$. This has a clear interpretation: if relaxing a constraint (making $g_i(\mathbf{x}) \leq 0$ less restrictive) would improve the objective, the multiplier is positive. If tightening the constraint would improve the objective (which is impossible at the optimum), the multiplier would be negative — but this can't happen.

### 4.3 Complementary Slackness

Complementary slackness is the most insightful KKT condition:

$$\lambda_i^* g_i(\mathbf{x}^*) = 0 \quad \text{for all } i$$

This means for each constraint, either:
- $\lambda_i^* = 0$ (the constraint is inactive — it doesn't affect the solution), or
- $g_i(\mathbf{x}^*) = 0$ (the constraint is active — it binds at the optimum)

Both can be zero simultaneously (degenerate case), but they can't both be non-zero.

**For AI:** In SVM, complementary slackness means $\alpha_i^*(1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b)) = 0$. Points with $\alpha_i^* > 0$ lie exactly on the margin boundary ($y_i(\mathbf{w}^\top \mathbf{x}_i + b) = 1$) — these are the support vectors. Points with $\alpha_i^* = 0$ are strictly on the correct side of the margin and don't affect the decision boundary.

### 4.4 Second-Order Sufficient Conditions

For inequality-constrained problems, the second-order condition involves the Hessian of the Lagrangian restricted to the **critical cone**:

$$\mathcal{C} = \{\mathbf{d} : \nabla g_i(\mathbf{x}^*)^\top \mathbf{d} \leq 0 \text{ for active } i, \; \nabla h_j(\mathbf{x}^*)^\top \mathbf{d} = 0 \text{ for all } j\}$$

**Theorem (Second-Order Sufficient Condition).** If the KKT conditions hold and:

$$\mathbf{d}^\top \nabla_{\mathbf{xx}}^2 \mathcal{L}(\mathbf{x}^*, \boldsymbol{\lambda}^*, \boldsymbol{\nu}^*) \mathbf{d} > 0 \quad \forall \mathbf{d} \in \mathcal{C} \setminus \{\mathbf{0}\}$$

then $\mathbf{x}^*$ is a strict local minimum.

### 4.5 Constraint Qualifications: LICQ, Slater's Condition

**LICQ (Linear Independence Constraint Qualification):** The gradients of all active constraints are linearly independent. This is the strongest and most commonly used constraint qualification.

**Slater's Condition:** For convex problems, there exists a strictly feasible point. This is the most important constraint qualification in convex optimization because it guarantees strong duality.

**MFCQ (Mangasarian-Fromovitz):** Weaker than LICQ, requires the existence of a feasible descent direction.

**For AI:** In practice, Slater's condition is easy to verify for most ML problems. For the SVM, any point with sufficiently large margin satisfies Slater's condition. For non-negative matrix factorization, any strictly positive factorization satisfies it.


---

## 5. Duality Theory

### 5.1 The Dual Problem

The **Lagrange dual function** is the minimum value of the Lagrangian over $\mathbf{x}$:

$$g(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \inf_{\mathbf{x}} \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = \inf_{\mathbf{x}} \left(f(\mathbf{x}) + \sum_{i=1}^m \lambda_i g_i(\mathbf{x}) + \sum_{j=1}^p \nu_j h_j(\mathbf{x})\right)$$

The **dual problem** is:

$$\max_{\boldsymbol{\lambda}, \boldsymbol{\nu}} g(\boldsymbol{\lambda}, \boldsymbol{\nu}) \quad \text{s.t.} \quad \boldsymbol{\lambda} \geq \mathbf{0}$$

**Key property:** The dual function is always **concave** (it's the pointwise infimum of affine functions in $(\boldsymbol{\lambda}, \boldsymbol{\nu})$), regardless of whether the primal problem is convex. This means the dual problem is always a convex optimization problem (maximizing a concave function is convex).

### 5.2 Weak Duality and the Duality Gap

**Theorem (Weak Duality).** For any feasible primal point $\tilde{\mathbf{x}}$ and any dual feasible point $(\tilde{\boldsymbol{\lambda}}, \tilde{\boldsymbol{\nu}})$ with $\tilde{\boldsymbol{\lambda}} \geq \mathbf{0}$:

$$g(\tilde{\boldsymbol{\lambda}}, \tilde{\boldsymbol{\nu}}) \leq p^* \leq f(\tilde{\mathbf{x}})$$

where $p^*$ is the optimal primal value and $d^* = \max g(\boldsymbol{\lambda}, \boldsymbol{\nu})$ is the optimal dual value.

The **duality gap** is $f(\tilde{\mathbf{x}}) - g(\tilde{\boldsymbol{\lambda}}, \tilde{\boldsymbol{\nu}}) \geq 0$. Any feasible primal-dual pair provides bounds on the optimal value.

**For AI:** The duality gap is used as a stopping criterion in many constrained optimization algorithms. When the gap is small, we know we're close to optimal. In SVM solvers (like LIBSVM), the duality gap monitors convergence.

### 5.3 Strong Duality and Slater's Condition

**Strong duality** holds when $d^* = p^*$ — the optimal dual value equals the optimal primal value. This is not always true, but it holds for convex problems under mild conditions.

**Theorem (Strong Duality via Slater's Condition).** If the primal problem is convex and Slater's condition holds (there exists a strictly feasible point), then strong duality holds. Moreover, the dual optimum is attained.

**For AI:** Strong duality is the foundation of the SVM dual formulation. It guarantees that solving the dual problem gives the same solution as solving the primal. This is crucial because the dual problem is often easier to solve (it has simple box constraints and a quadratic objective).

### 5.4 Interpreting Dual Variables as Shadow Prices

The optimal dual variables $\boldsymbol{\lambda}^*$ have a concrete economic interpretation: they are the **shadow prices** of the constraints. Consider perturbing the right-hand side of the $i$-th constraint:

$$\min f(\mathbf{x}) \quad \text{s.t.} \quad g_i(\mathbf{x}) \leq u_i$$

Let $p^*(\mathbf{u})$ be the optimal value as a function of the perturbations $\mathbf{u}$. Then:

$$\lambda_i^* = -\frac{\partial p^*}{\partial u_i}(\mathbf{0})$$

This means $\lambda_i^*$ measures how much the optimal value improves (decreases) per unit relaxation of the $i$-th constraint.

**For AI:** In constrained reinforcement learning, the dual variables associated with safety constraints represent the "cost" of safety — how much reward must be sacrificed to satisfy each constraint. In portfolio optimization, the dual variables represent the marginal value of additional budget or risk tolerance.

### 5.5 Duality in SVM: From Primal to Dual

The SVM primal problem is:

$$\min_{\mathbf{w}, b} \frac{1}{2}\lVert \mathbf{w} \rVert_2^2 \quad \text{s.t.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1, \; i = 1, \ldots, n$$

The Lagrangian is:

$$\mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2}\lVert \mathbf{w} \rVert_2^2 - \sum_{i=1}^n \alpha_i [y_i(\mathbf{w}^\top \mathbf{x}_i + b) - 1]$$

Setting derivatives to zero:

$$\nabla_{\mathbf{w}} \mathcal{L} = \mathbf{w} - \sum_{i=1}^n \alpha_i y_i \mathbf{x}_i = \mathbf{0} \quad \Rightarrow \quad \mathbf{w} = \sum_{i=1}^n \alpha_i y_i \mathbf{x}_i$$

$$\frac{\partial \mathcal{L}}{\partial b} = -\sum_{i=1}^n \alpha_i y_i = 0$$

Substituting back into the Lagrangian gives the **dual problem**:

$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j \mathbf{x}_i^\top \mathbf{x}_j$$
$$\text{s.t.} \quad \alpha_i \geq 0, \quad \sum_{i=1}^n \alpha_i y_i = 0$$

**Key observations:**
1. The dual depends only on inner products $\mathbf{x}_i^\top \mathbf{x}_j$ — this enables the **kernel trick**
2. The dual has simple constraints: box constraints $\alpha_i \geq 0$ and one linear equality
3. The solution is **sparse**: most $\alpha_i = 0$, and only the support vectors ($\alpha_i > 0$) matter

---

## 6. Algorithms for Constrained Optimization

### 6.1 Projected Gradient Descent

**Projected gradient descent** applies GD followed by a projection onto the feasible set:

$$\mathbf{x}_{t+1} = \Pi_{\mathcal{C}}(\mathbf{x}_t - \eta \nabla f(\mathbf{x}_t))$$

where $\Pi_{\mathcal{C}}(\mathbf{y}) = \arg\min_{\mathbf{x} \in \mathcal{C}} \lVert \mathbf{x} - \mathbf{y} \rVert_2$ is the Euclidean projection onto $\mathcal{C}$.

**Convergence:** For convex $f$ and convex $\mathcal{C}$, projected GD with $\eta = 1/L$ converges at rate $O(1/T)$, the same as unconstrained GD. For strongly convex $f$, the rate is $O((1 - \mu/L)^T)$.

**Why it works:** The projection step ensures feasibility while making the smallest possible change to the GD step. When the unconstrained GD step is feasible, the projection is the identity and projected GD reduces to standard GD.

### 6.2 Projection onto Common Sets

**Euclidean ball** $\mathcal{B}(\mathbf{c}, r) = \{\mathbf{x} : \lVert \mathbf{x} - \mathbf{c} \rVert_2 \leq r\}$:

$$\Pi_{\mathcal{B}}(\mathbf{y}) = \mathbf{c} + r \cdot \frac{\mathbf{y} - \mathbf{c}}{\max(\lVert \mathbf{y} - \mathbf{c} \rVert_2, r)}$$

**Simplex** $\Delta_n = \{\mathbf{x} : \mathbf{x} \geq \mathbf{0}, \mathbf{1}^\top \mathbf{x} = 1\}$:

The projection can be computed in $O(n \log n)$ time using a sorting-based algorithm (Duchi et al., 2008):

1. Sort $\mathbf{y}$ in descending order
2. Find $\rho = \max\{j : y_{(j)} - \frac{1}{j}(\sum_{r=1}^j y_{(r)} - 1) > 0\}$
3. Set $\theta = \frac{1}{\rho}(\sum_{r=1}^\rho y_{(r)} - 1)$
4. Return $\Pi_{\Delta_n}(\mathbf{y}) = \max(\mathbf{y} - \theta, \mathbf{0})$

**Non-negative orthant** $\mathbb{R}^n_+$: $\Pi_{\mathbb{R}^n_+}(\mathbf{y}) = \max(\mathbf{y}, \mathbf{0})$ (element-wise).

**Box constraints** $[\mathbf{l}, \mathbf{u}]$: $\Pi_{[\mathbf{l}, \mathbf{u}]}(\mathbf{y}) = \min(\max(\mathbf{y}, \mathbf{l}), \mathbf{u})$ (element-wise).

**For AI:** Projection onto the simplex is essential for optimization over probability distributions (e.g., policy optimization in RL, attention weights). Projection onto the non-negative orthant is used in NMF and non-negative constrained neural networks.

### 6.3 Penalty Methods

**Penalty methods** transform a constrained problem into a sequence of unconstrained problems by adding a penalty for constraint violations:

$$\min_{\mathbf{x}} f(\mathbf{x}) + \mu \sum_{i=1}^m \max(0, g_i(\mathbf{x}))^2 + \mu \sum_{j=1}^p h_j(\mathbf{x})^2$$

where $\mu > 0$ is the penalty parameter. As $\mu \to \infty$, the solution of the penalized problem converges to the solution of the constrained problem.

**Quadratic penalty:** $P(\mathbf{x}) = \sum_i \max(0, g_i(\mathbf{x}))^2 + \sum_j h_j(\mathbf{x})^2$. Smooth but requires $\mu \to \infty$ for exact feasibility.

**Exact penalty:** $P(\mathbf{x}) = \sum_i \max(0, g_i(\mathbf{x})) + \sum_j |h_j(\mathbf{x})|$. Non-smooth but gives the exact solution for sufficiently large $\mu$.

**For AI:** Penalty methods are used in constrained reinforcement learning, where the constraint violation is added as a penalty term to the reward function. The penalty coefficient is gradually increased during training.

### 6.4 Barrier (Interior-Point) Methods

**Barrier methods** add a barrier function that goes to $+\infty$ as the boundary of the feasible set is approached, keeping the iterates strictly inside the feasible region:

$$\min_{\mathbf{x}} f(\mathbf{x}) - \mu \sum_{i=1}^m \log(-g_i(\mathbf{x}))$$

The **logarithmic barrier** $-\log(-g_i(\mathbf{x}))$ is the most common choice. As $\mu \to 0$, the solution of the barrier problem converges to the solution of the constrained problem.

**Central path:** The set of solutions $\mathbf{x}^*(\mu)$ for all $\mu > 0$ forms the central path. Interior-point methods follow this path by gradually decreasing $\mu$.

**Convergence:** Primal-dual interior-point methods converge in $O(\sqrt{m} \log(1/\epsilon))$ iterations for convex problems, where $m$ is the number of inequality constraints. Each iteration requires solving a Newton system.

**For AI:** Interior-point methods are used for small-to-medium scale convex problems (SVMs with moderate data sizes, portfolio optimization). They are not practical for large-scale deep learning due to the $O(n^3)$ cost per iteration.

### 6.5 Augmented Lagrangian Method

The **augmented Lagrangian** (method of multipliers) combines the Lagrangian with a quadratic penalty:

$$\mathcal{L}_\mu(\mathbf{x}, \boldsymbol{\lambda}) = f(\mathbf{x}) + \sum_{i=1}^m \lambda_i g_i(\mathbf{x}) + \frac{\mu}{2}\sum_{i=1}^m \max(0, g_i(\mathbf{x}))^2$$

**Algorithm:**
1. Minimize $\mathcal{L}_{\mu_k}(\mathbf{x}, \boldsymbol{\lambda}_k)$ with respect to $\mathbf{x}$
2. Update multipliers: $\lambda_i^{k+1} = \max(0, \lambda_i^k + \mu_k g_i(\mathbf{x}_k))$
3. Increase $\mu_k$ if needed

**Advantage over pure penalty methods:** The augmented Lagrangian converges for finite $\mu$ — the multipliers adjust to enforce the constraints without requiring $\mu \to \infty$. This avoids the ill-conditioning that plagues pure penalty methods.

**For AI:** The augmented Lagrangian is the foundation of ADMM (§7.3) and is used in distributed optimization, where each worker solves a local augmented Lagrangian subproblem and the multipliers enforce consensus.


---

## 7. Advanced Topics

### 7.1 Sequential Quadratic Programming (SQP)

SQP is the most widely used method for general non-convex constrained optimization. At each iteration, it solves a quadratic programming (QP) subproblem:

$$\min_{\mathbf{d}} \quad \nabla f(\mathbf{x}_k)^\top \mathbf{d} + \frac{1}{2}\mathbf{d}^\top B_k \mathbf{d}$$
$$\text{s.t.} \quad g_i(\mathbf{x}_k) + \nabla g_i(\mathbf{x}_k)^\top \mathbf{d} \leq 0$$
$$\quad \quad h_j(\mathbf{x}_k) + \nabla h_j(\mathbf{x}_k)^\top \mathbf{d} = 0$$

where $B_k$ approximates the Hessian of the Lagrangian (often updated via BFGS).

**Why SQP works:** The QP subproblem is the Newton step for the KKT conditions. When $B_k = \nabla_{\mathbf{xx}}^2 \mathcal{L}$, SQP reduces to Newton's method for the KKT system and converges quadratically.

**For AI:** SQP is used in robotics (trajectory optimization with dynamics constraints), optimal control, and engineering design. It is not commonly used in deep learning due to the cost of solving the QP subproblem, but it is the method of choice for small-to-medium constrained problems with smooth objectives.

### 7.2 Primal-Dual Interior-Point Methods

Primal-dual interior-point methods solve the KKT conditions directly using Newton's method, applied to the perturbed KKT system:

$$\nabla f(\mathbf{x}) + \sum_{i=1}^m \lambda_i \nabla g_i(\mathbf{x}) + \sum_{j=1}^p \nu_j \nabla h_j(\mathbf{x}) = \mathbf{0}$$
$$g_i(\mathbf{x}) \leq 0, \quad h_j(\mathbf{x}) = 0, \quad \lambda_i \geq 0$$
$$\lambda_i g_i(\mathbf{x}) = -\mu \quad \text{(perturbed complementary slackness)}$$

The parameter $\mu > 0$ is gradually decreased to zero. At each iteration, Newton's method is applied to the nonlinear system, and a line search ensures that $\lambda_i > 0$ and $g_i(\mathbf{x}) < 0$ (strict feasibility).

**Complexity:** For convex problems with $m$ inequality constraints and $n$ variables, each iteration costs $O(n^3 + mn^2)$ and the number of iterations is $O(\sqrt{m} \log(1/\epsilon))$.

### 7.3 ADMM: Alternating Direction Method of Multipliers

**ADMM** solves problems of the form:

$$\min_{\mathbf{x}, \mathbf{z}} f(\mathbf{x}) + g(\mathbf{z}) \quad \text{s.t.} \quad A\mathbf{x} + B\mathbf{z} = \mathbf{c}$$

**Algorithm:**
1. $\mathbf{x}_{k+1} = \arg\min_{\mathbf{x}} \mathcal{L}_\rho(\mathbf{x}, \mathbf{z}_k, \mathbf{u}_k)$
2. $\mathbf{z}_{k+1} = \arg\min_{\mathbf{z}} \mathcal{L}_\rho(\mathbf{x}_{k+1}, \mathbf{z}, \mathbf{u}_k)$
3. $\mathbf{u}_{k+1} = \mathbf{u}_k + A\mathbf{x}_{k+1} + B\mathbf{z}_{k+1} - \mathbf{c}$

where $\mathcal{L}_\rho$ is the augmented Lagrangian with penalty parameter $\rho > 0$.

**Why ADMM is powerful:** It splits the problem into two subproblems that can often be solved efficiently (e.g., one is a proximal operator, the other is a linear system). The alternating structure makes it ideal for distributed computing.

**For AI:** ADMM is used for:
- **Distributed training:** Each worker optimizes a local model, and ADMM enforces consensus
- **Sparse optimization:** Splitting the objective into a smooth loss and a non-smooth regularizer (L1, nuclear norm)
- **Federated learning:** ADMM naturally handles the communication constraints of federated settings

### 7.4 Conic Programming: SOCP and SDP

**Second-Order Cone Programming (SOCP):**

$$\min \mathbf{c}^\top \mathbf{x} \quad \text{s.t.} \quad \lVert A_i \mathbf{x} + \mathbf{b}_i \rVert_2 \leq \mathbf{c}_i^\top \mathbf{x} + d_i$$

SOCP generalizes LP, QP, and QCQP. It can model robust optimization problems, norm minimization, and many engineering design problems.

**Semidefinite Programming (SDP):**

$$\min \operatorname{tr}(C X) \quad \text{s.t.} \quad \operatorname{tr}(A_i X) = b_i, \quad X \succeq 0$$

SDP is the most general tractable convex optimization problem. It can model eigenvalue optimization, matrix completion, and relaxations of combinatorial problems.

**For AI:** SDP relaxations are used for:
- **Community detection:** The Goemans-Williamson relaxation for Max-Cut
- **Matrix completion:** Nuclear norm minimization as an SDP
- **Neural network verification:** SDP relaxations for robustness certification
- **Clustering:** SDP relaxations for k-means and spectral clustering

### 7.5 Connection to Proximal Methods

The **proximal operator** of a function $g$ is:

$$\text{prox}_g(\mathbf{v}) = \arg\min_{\mathbf{x}} \left(g(\mathbf{x}) + \frac{1}{2}\lVert \mathbf{x} - \mathbf{v} \rVert_2^2\right)$$

This is exactly the projection onto the feasible set when $g$ is the indicator function of a convex set: $g(\mathbf{x}) = 0$ if $\mathbf{x} \in \mathcal{C}$, $+\infty$ otherwise.

**Proximal gradient descent** for $f(\mathbf{x}) + g(\mathbf{x})$ where $f$ is smooth and $g$ is possibly non-smooth:

$$\mathbf{x}_{t+1} = \text{prox}_{\eta g}(\mathbf{x}_t - \eta \nabla f(\mathbf{x}_t))$$

This generalizes projected gradient descent (where $g$ is the indicator function) and includes the soft-thresholding operator for L1 regularization.

**For AI:** Proximal methods are the foundation of many ML algorithms:
- **Lasso:** Proximal gradient with $g(\mathbf{x}) = \lambda \lVert \mathbf{x} \rVert_1$ (soft thresholding)
- **Nuclear norm regularization:** Proximal operator is singular value thresholding
- **FISTA:** Accelerated proximal gradient with $O(1/T^2)$ rate

---

## 8. Applications in Machine Learning

### 8.1 Support Vector Machines: Primal and Dual

The soft-margin SVM primal problem:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}\lVert \mathbf{w} \rVert_2^2 + C\sum_{i=1}^n \xi_i \quad \text{s.t.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

The dual problem:

$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)$$
$$\text{s.t.} \quad 0 \leq \alpha_i \leq C, \quad \sum_{i=1}^n \alpha_i y_i = 0$$

where $K(\mathbf{x}_i, \mathbf{x}_j) = \phi(\mathbf{x}_i)^\top \phi(\mathbf{x}_j)$ is the kernel function.

**The kernel trick:** The dual depends only on inner products, so we can replace $\mathbf{x}_i^\top \mathbf{x}_j$ with any valid kernel $K(\mathbf{x}_i, \mathbf{x}_j)$. This implicitly maps the data to a high-dimensional (possibly infinite-dimensional) feature space without explicitly computing the mapping.

**Common kernels:**
- **Linear:** $K(\mathbf{x}, \mathbf{x}') = \mathbf{x}^\top \mathbf{x}'$
- **Polynomial:** $K(\mathbf{x}, \mathbf{x}') = (\mathbf{x}^\top \mathbf{x}' + c)^d$
- **RBF (Gaussian):** $K(\mathbf{x}, \mathbf{x}') = \exp(-\gamma \lVert \mathbf{x} - \mathbf{x}' \rVert_2^2)$

### 8.2 Constrained Logistic Regression

Standard logistic regression is unconstrained. But we may want to add constraints:

**Non-negative logistic regression:** $\mathbf{w} \geq \mathbf{0}$. This ensures that all features have non-negative influence, which is useful for interpretability.

**Sparse logistic regression:** $\lVert \mathbf{w} \rVert_1 \leq t$. This can be formulated as a constrained problem or equivalently as the L1-regularized problem.

**Group-sparse logistic regression:** $\sum_{g} \lVert \mathbf{w}_g \rVert_2 \leq t$. This encourages entire groups of features to be zero, useful for feature selection with structured features.

### 8.3 Non-Negative Matrix Factorization

NMF factorizes a non-negative matrix $V \in \mathbb{R}^{m \times n}_+$ as $V \approx WH$ where $W \in \mathbb{R}^{m \times k}_+$ and $H \in \mathbb{R}^{k \times n}_+$:

$$\min_{W, H \geq 0} \frac{1}{2}\lVert V - WH \rVert_F^2$$

This is a non-convex constrained optimization problem (non-convex objective, convex constraints). Standard algorithms use alternating minimization: fix $W$ and optimize $H$ (non-negative least squares), then fix $H$ and optimize $W$.

**For AI:** NMF is used for topic modeling (documents as non-negative combinations of topics), image processing (parts-based representation), and recommender systems (non-negative user-item factors).

### 8.4 Trust Region Subproblems

The trust region subproblem arises in trust region methods for unconstrained optimization:

$$\min_{\mathbf{d}} \quad \mathbf{g}^\top \mathbf{d} + \frac{1}{2}\mathbf{d}^\top B \mathbf{d} \quad \text{s.t.} \quad \lVert \mathbf{d} \rVert_2 \leq \Delta$$

This is a convex QP when $B \succeq 0$ and a non-convex QP when $B$ is indefinite. Despite the non-convexity, the trust region subproblem has no spurious local minima — every local minimum is global. This is a rare and valuable property.

**Solution:** The optimal solution satisfies $(B + \lambda I)\mathbf{d} = -\mathbf{g}$ with $\lambda \geq 0$ chosen so that $\lVert \mathbf{d} \rVert_2 = \Delta$ (or $\lambda = 0$ if the unconstrained minimum is inside the trust region).

### 8.5 Constrained Reinforcement Learning

Constrained Markov Decision Processes (CMDPs) extend standard MDPs with cost constraints:

$$\max_{\pi} \mathbb{E}\left[\sum_{t=0}^\infty \gamma^t r(s_t, a_t)\right] \quad \text{s.t.} \quad \mathbb{E}\left[\sum_{t=0}^\infty \gamma^t c_j(s_t, a_t)\right] \leq d_j, \quad j = 1, \ldots, m$$

The Lagrangian for a CMDP is:

$$\mathcal{L}(\pi, \boldsymbol{\lambda}) = \mathbb{E}\left[\sum_{t=0}^\infty \gamma^t \left(r(s_t, a_t) - \sum_{j=1}^m \lambda_j c_j(s_t, a_t)\right)\right] + \sum_{j=1}^m \lambda_j d_j$$

This transforms the constrained problem into an unconstrained problem with a modified reward function $r'(s, a) = r(s, a) - \sum_j \lambda_j c_j(s, a)$. The dual variables $\lambda_j$ are updated using gradient ascent on the dual function.

**For AI:** Constrained RL is essential for safety-critical applications (autonomous driving, medical treatment, robotics) where the agent must satisfy safety constraints while maximizing performance. Methods like Constrained Policy Optimization (CPO) and Lagrangian-based PPO use the dual formulation to handle constraints.


---

## 9. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
|---|---------|----------------|-----|
| 1 | "The KKT conditions are always necessary for optimality" | KKT conditions require a constraint qualification (LICQ, Slater's, etc.). Without it, the optimum may not satisfy KKT. | Verify a constraint qualification holds before applying KKT. For convex problems, check Slater's condition. |
| 2 | "Lagrange multipliers are always positive" | Equality constraint multipliers $\nu_j$ can be positive or negative. Only inequality multipliers $\lambda_i$ are non-negative. | Remember: $\lambda_i \geq 0$ for inequalities, $\nu_j \in \mathbb{R}$ for equalities. |
| 3 | "Strong duality always holds" | Strong duality requires convexity and a constraint qualification. For non-convex problems, there can be a duality gap. | Check convexity and Slater's condition. For non-convex problems, use the duality gap as a bound, not an equality. |
| 4 | "Projected GD is the same as GD with penalty" | Projected GD maintains feasibility at every step. Penalty methods allow infeasible iterates and only converge to feasibility as $\mu \to \infty$. | Use projected GD when projection is cheap. Use penalty methods when the projection is expensive or unavailable. |
| 5 | "The dual problem is always easier to solve" | The dual can be harder if the dual function is difficult to evaluate (requires solving an unconstrained optimization at each dual evaluation). | Analyze the structure of both primal and dual. The dual is easier when it has simple constraints and a closed-form dual function. |
| 6 | "Complementary slackness means all constraints are active" | Complementary slackness means each constraint is either active OR has zero multiplier — not both. Most constraints are typically inactive. | Use complementary slackness to identify which constraints are active: if $\lambda_i > 0$, then $g_i(\mathbf{x}^*) = 0$. |
| 7 | "ADMM always converges for non-convex problems" | ADMM convergence is guaranteed for convex problems. For non-convex problems, convergence depends on the specific structure. | For non-convex problems, verify convergence conditions or use ADMM as a heuristic with careful monitoring. |
| 8 | "The SVM dual always has a unique solution" | The primal SVM solution $\mathbf{w}^*$ is unique, but the dual solution $\boldsymbol{\alpha}^*$ may not be unique if the kernel matrix is singular. | The primal solution is always unique for strictly convex objectives. The dual may have multiple solutions that map to the same primal. |
| 9 | "Barrier methods work for any constraints" | Barrier methods require strictly feasible starting points and can only handle inequality constraints (equality constraints must be handled separately). | Use barrier methods for inequality constraints with a feasible starting point. Use augmented Lagrangian for mixed constraints. |
| 10 | "Adding more constraints always makes the problem harder" | Adding redundant constraints (that are already satisfied) doesn't change the problem. Adding constraints that cut off infeasible regions can sometimes make the problem easier to solve. | Analyze which constraints are binding. Redundant constraints can be removed to simplify the problem. |

---

## 10. Exercises

**Exercise 1: Lagrange Multipliers for Equality Constraints (★)**
Solve $\min \frac{1}{2}(x_1^2 + x_2^2)$ subject to $x_1 + x_2 = 1$ using the method of Lagrange multipliers. Verify the solution geometrically.

**Exercise 2: KKT Conditions for a Simple Problem (★)**
For $\min x_1^2 + x_2^2$ subject to $x_1 + x_2 \geq 1$, write down the KKT conditions and solve them. Identify which constraints are active.

**Exercise 3: Projection onto the Simplex (★)**
Implement the $O(n \log n)$ algorithm for projecting onto the probability simplex. Verify that the result satisfies $\mathbf{1}^\top \mathbf{x} = 1$ and $\mathbf{x} \geq \mathbf{0}$.

**Exercise 4: SVM Dual Derivation (★★)**
Derive the dual of the soft-margin SVM from the primal. Show that the dual variables satisfy $0 \leq \alpha_i \leq C$ and $\sum \alpha_i y_i = 0$.

**Exercise 5: Penalty Method Convergence (★★)**
Apply the quadratic penalty method to $\min x^2$ subject to $x \geq 1$. Solve the penalized problem analytically and show that the solution converges to $x^* = 1$ as $\mu \to \infty$.

**Exercise 6: ADMM for Lasso (★★)**
Formulate the Lasso problem $\min \frac{1}{2}\lVert A\mathbf{x} - \mathbf{b} \rVert_2^2 + \lambda \lVert \mathbf{x} \rVert_1$ in ADMM form. Derive the update steps for $\mathbf{x}$, $\mathbf{z}$, and $\mathbf{u}$.

**Exercise 7: Constrained Portfolio Optimization (★★★)**
Solve the mean-variance portfolio optimization problem with non-negativity constraints: $\min \frac{1}{2}\mathbf{w}^\top \Sigma \mathbf{w}$ subject to $\boldsymbol{\mu}^\top \mathbf{w} = r_{\text{target}}$, $\mathbf{1}^\top \mathbf{w} = 1$, $\mathbf{w} \geq \mathbf{0}$. Use projected gradient descent.

**Exercise 8: CMDP Lagrangian Analysis (★★★)**
For a two-state, two-action CMDP with one cost constraint, derive the Lagrangian and show that the optimal policy can be found by solving an unconstrained MDP with modified rewards. Analyze how the dual variable $\lambda$ trades off reward vs. cost.

---

## 11. Why This Matters for AI (2026 Perspective)

| Concept | AI Impact |
|---------|-----------|
| KKT conditions | Foundation for SVM, constrained RL, fair ML, and all constrained optimization in ML |
| Lagrange multipliers | Shadow prices interpret constraint costs; dual variables in CMDPs |
| Duality theory | Kernel trick via SVM dual; distributed optimization via dual decomposition |
| Projected gradient descent | Constrained neural network training; policy optimization with safety constraints |
| Simplex projection | Attention weight normalization; policy optimization in RL |
| Penalty methods | Constrained RL reward shaping; fairness constraints in classification |
| Barrier methods | Interior-point SVM solvers; log-barrier for non-negativity constraints |
| Augmented Lagrangian | ADMM for distributed training; federated learning with consensus |
| ADMM | Distributed optimization across GPUs; federated learning; sparse model training |
| SVM dual + kernel trick | Non-linear classification; kernel methods; Gaussian processes |
| Trust region subproblems | Trust region policy optimization (TRPO); robust optimization |
| Conic programming (SDP) | Neural network verification; matrix completion; clustering relaxations |
| Proximal methods | Lasso, nuclear norm regularization; FISTA for large-scale sparse learning |
| Constrained RL | Safety-critical AI; autonomous systems; medical treatment optimization |

---

## 12. Conceptual Bridge

### Backward Connections

Constrained optimization builds directly on the foundations from [Convex Optimization](../01-Convex-Optimization/notes.md) and [Gradient Descent](../02-Gradient-Descent/notes.md). The Lagrangian duality theory was previewed in §01 as part of the convex analysis toolkit; here we develop the full KKT machinery that makes duality computationally useful. Projected gradient descent extends the GD convergence theory from §02 to constrained settings — the convergence rates are identical when the projection is non-expansive (which it is for convex sets).

From [Multivariate Calculus](../../05-Multivariate-Calculus/README.md), we use gradients, Hessians, and the chain rule throughout. The second-order conditions for constrained problems involve the Hessian of the Lagrangian, which combines the objective Hessian with the constraint Hessians weighted by the multipliers.

### Forward Connections

This section connects to several advanced topics:

- **Stochastic Optimization** (§05) combines projected gradient descent with noisy gradients for large-scale constrained problems. Stochastic projected GD has the same convergence rates as stochastic GD when the projection is onto a convex set.
- **Optimization Landscape** (§06) analyzes the geometry of constrained critical points. The KKT conditions define the stationary points of constrained problems, and the second-order conditions determine their stability.
- **Adaptive Learning Rate** (§07) connects through the augmented Lagrangian: ADMM can be viewed as an adaptive method that adjusts the penalty parameter based on constraint violation.
- **Regularization Methods** (§08) connects through the SVM: the hinge loss is the convex surrogate for the 0-1 loss, and the SVM dual reveals the connection between regularization and constraint satisfaction.
- **Learning Rate Schedules** (§10) applies to projected GD and penalty methods — the step size affects both the objective decrease and the constraint satisfaction.

### The Big Picture

```
CONSTRAINED OPTIMIZATION IN THE CURRICULUM
════════════════════════════════════════════════════════════════════════

              Unconstrained Optimization (§01-§03)
              GD, Newton, BFGS on R^n
                       │
                       ▼
              ╔══════════════════════════════════╗
              ║    CONSTRAINED OPTIMIZATION      ║ ← YOU ARE HERE
              ║                                  ║
              ║  Lagrangian: L = f + λg + νh     ║
              ║  KKT: stationarity + feasibility ║
              ║       + complementarity          ║
              ║  Dual: max g(λ,ν) s.t. λ ≥ 0     ║
              ║                                  ║
              ║  Algorithms:                     ║
              ║  - Projected GD (simple sets)    ║
              ║  - Penalty/Barrier (general)     ║
              ║  - ADMM (distributed)            ║
              ║  - SQP (non-convex)              ║
              ╚══════════════════════════════════╝
                       │
            ┌──────────┼──────────┐
            ▼          ▼          ▼
      Stochastic   Landscape   Adaptive LR
      (§05 proj.   (§06 KKT    (§07 augmented
       GD)          points)     Lagrangian)
            │          │          │
            ▼          ▼          ▼
      Regularization  Fair ML   Federated
      (§08 SVM)       (constraints) Learning

════════════════════════════════════════════════════════════════════════
```

Constrained optimization is the bridge between the clean theory of unconstrained optimization and the messy reality of ML applications. Every practical ML problem has constraints — whether explicit (non-negativity, budget, safety) or implicit (probability normalization, norm bounds). Understanding constrained optimization is essential for formulating, analyzing, and solving real ML problems.

---

## References

1. Boyd, S. & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.
2. Nocedal, J. & Wright, S. (2006). *Numerical Optimization* (2nd ed.). Springer.
3. Bertsekas, D. (2014). *Constrained Optimization and Lagrange Multiplier Methods*. Athena Scientific.
4. Vapnik, V. (1998). *Statistical Learning Theory*. Wiley.
5. Boyd, S. et al. (2011). "Distributed optimization and statistical learning via the alternating direction method of multipliers." Foundations and Trends in Machine Learning.
6. Duchi, J. et al. (2008). "Efficient projections onto the l1-ball for learning in high dimensions." ICML.
7. Achiam, J. et al. (2017). "Constrained policy optimization." ICML.
8. Martens, J. & Grosse, R. (2015). "Optimizing neural networks with Kronecker-factored approximate curvature." ICML.
9. Nesterov, Y. & Nemirovski, A. (1994). *Interior-Point Polynomial Algorithms in Convex Programming*. SIAM.
10. Karmarkar, N. (1984). "A new polynomial-time algorithm for linear programming." Combinatorica.
11. Karush, W. (1939). "Minima of functions of several variables with inequalities as side conditions." Master's thesis, University of Chicago.
12. Kuhn, H. & Tucker, A. (1951). "Nonlinear programming." Proceedings of the Second Berkeley Symposium.
13. Goemans, M. & Williamson, D. (1995). "Improved approximation algorithms for maximum cut and satisfiability problems." JACM.
14. Candès, E. & Recht, B. (2009). "Exact matrix completion via convex optimization." Foundations of Computational Mathematics.
15. Beck, A. & Teboulle, M. (2009). "A fast iterative shrinkage-thresholding algorithm for linear inverse problems." SIAM Journal on Imaging Sciences.


---

## Appendix A: Detailed Proofs and Extended Theory

### A.1 Full Proof of the KKT Conditions

**Theorem (KKT Necessary Conditions).** Let $\mathbf{x}^*$ be a local minimum of $\min f(\mathbf{x})$ subject to $g_i(\mathbf{x}) \leq 0$ ($i = 1, \ldots, m$) and $h_j(\mathbf{x}) = 0$ ($j = 1, \ldots, p$). Assume LICQ holds at $\mathbf{x}^*$. Then there exist $\boldsymbol{\lambda}^* \geq \mathbf{0}$ and $\boldsymbol{\nu}^*$ such that:

1. $\nabla f(\mathbf{x}^*) + \sum_{i=1}^m \lambda_i^* \nabla g_i(\mathbf{x}^*) + \sum_{j=1}^p \nu_j^* \nabla h_j(\mathbf{x}^*) = \mathbf{0}$
2. $g_i(\mathbf{x}^*) \leq 0$, $h_j(\mathbf{x}^*) = 0$
3. $\lambda_i^* \geq 0$
4. $\lambda_i^* g_i(\mathbf{x}^*) = 0$

**Proof.** Let $\mathcal{A} = \{i : g_i(\mathbf{x}^*) = 0\}$ be the active set. Consider the tangent cone at $\mathbf{x}^*$:

$$T = \{\mathbf{d} : \nabla g_i(\mathbf{x}^*)^\top \mathbf{d} \leq 0 \; \forall i \in \mathcal{A}, \; \nabla h_j(\mathbf{x}^*)^\top \mathbf{d} = 0 \; \forall j\}$$

For $\mathbf{x}^*$ to be a local minimum, we must have $\nabla f(\mathbf{x}^*)^\top \mathbf{d} \geq 0$ for all $\mathbf{d} \in T$. Otherwise, there would be a feasible descent direction.

By Farkas' lemma (a fundamental result in linear programming), the condition $\nabla f(\mathbf{x}^*)^\top \mathbf{d} \geq 0$ for all $\mathbf{d} \in T$ is equivalent to the existence of $\lambda_i^* \geq 0$ ($i \in \mathcal{A}$) and $\nu_j^*$ ($j = 1, \ldots, p$) such that:

$$\nabla f(\mathbf{x}^*) + \sum_{i \in \mathcal{A}} \lambda_i^* \nabla g_i(\mathbf{x}^*) + \sum_{j=1}^p \nu_j^* \nabla h_j(\mathbf{x}^*) = \mathbf{0}$$

Setting $\lambda_i^* = 0$ for inactive constraints ($i \notin \mathcal{A}$) gives the full stationarity condition. The complementary slackness $\lambda_i^* g_i(\mathbf{x}^*) = 0$ follows because either $i \in \mathcal{A}$ (so $g_i(\mathbf{x}^*) = 0$) or $i \notin \mathcal{A}$ (so $\lambda_i^* = 0$). $\blacksquare$

### A.2 Proof of Strong Duality for Convex Problems

**Theorem (Strong Duality via Slater's Condition).** Let the primal problem be convex with $f$ and $g_i$ convex, $h_j$ affine. If Slater's condition holds (there exists $\mathbf{x}$ with $g_i(\mathbf{x}) < 0$ for all $i$ and $h_j(\mathbf{x}) = 0$ for all $j$), then strong duality holds: $d^* = p^*$.

**Proof.** Define the set:

$$\mathcal{G} = \{(\mathbf{u}, \mathbf{v}, t) \in \mathbb{R}^m \times \mathbb{R}^p \times \mathbb{R} : \exists \mathbf{x} \text{ with } g_i(\mathbf{x}) \leq u_i, h_j(\mathbf{x}) = v_j, f(\mathbf{x}) \leq t\}$$

This set is convex because $f$ and $g_i$ are convex and $h_j$ are affine. The optimal value is $p^* = \inf\{t : (\mathbf{0}, \mathbf{0}, t) \in \mathcal{G}\}$.

By Slater's condition, there exists $\tilde{\mathbf{x}}$ with $g_i(\tilde{\mathbf{x}}) < 0$ and $h_j(\tilde{\mathbf{x}}) = 0$. So $(\mathbf{g}(\tilde{\mathbf{x}}), \mathbf{0}, f(\tilde{\mathbf{x}})) \in \mathcal{G}$ with strictly negative inequality components.

By the separating hyperplane theorem, there exists a non-zero vector $(\boldsymbol{\lambda}, \boldsymbol{\nu}, \mu)$ and scalar $\alpha$ such that:

$$\boldsymbol{\lambda}^\top \mathbf{u} + \boldsymbol{\nu}^\top \mathbf{v} + \mu t \geq \alpha \quad \forall (\mathbf{u}, \mathbf{v}, t) \in \mathcal{G}$$

and $\boldsymbol{\lambda}^\top \mathbf{0} + \boldsymbol{\nu}^\top \mathbf{0} + \mu p^* \leq \alpha$.

It can be shown that $\mu > 0$ (using Slater's condition) and $\boldsymbol{\lambda} \geq \mathbf{0}$. Normalizing so that $\mu = 1$, we get:

$$g(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \inf_{\mathbf{x}} \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) \geq p^*$$

Combined with weak duality ($g(\boldsymbol{\lambda}, \boldsymbol{\nu}) \leq p^*$), this gives $g(\boldsymbol{\lambda}, \boldsymbol{\nu}) = p^*$, so strong duality holds. $\blacksquare$

### A.3 Convergence Analysis of Projected Gradient Descent

**Theorem (Convergence of Projected GD).** Let $f$ be convex and $L$-smooth, and let $\mathcal{C}$ be a closed convex set. For projected GD with $\eta = 1/L$:

$$f(\bar{\mathbf{x}}_T) - f^* \leq \frac{L \lVert \mathbf{x}_0 - \mathbf{x}^* \rVert_2^2}{2T}$$

where $\bar{\mathbf{x}}_T = \frac{1}{T}\sum_{t=0}^{T-1} \mathbf{x}_t$ and $\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})$.

**Proof.** The key property of the projection operator is **non-expansiveness**:

$$\lVert \Pi_{\mathcal{C}}(\mathbf{x}) - \Pi_{\mathcal{C}}(\mathbf{y}) \rVert_2 \leq \lVert \mathbf{x} - \mathbf{y} \rVert_2$$

This follows from the variational characterization of the projection: $\Pi_{\mathcal{C}}(\mathbf{y})$ is the unique point satisfying $(\mathbf{x} - \Pi_{\mathcal{C}}(\mathbf{y}))^\top(\mathbf{y} - \Pi_{\mathcal{C}}(\mathbf{y})) \leq 0$ for all $\mathbf{x} \in \mathcal{C}$.

Using this and the same analysis as unconstrained GD:

$$\lVert \mathbf{x}_{t+1} - \mathbf{x}^* \rVert_2^2 = \lVert \Pi_{\mathcal{C}}(\mathbf{x}_t - \eta \nabla f(\mathbf{x}_t)) - \Pi_{\mathcal{C}}(\mathbf{x}^*) \rVert_2^2 \leq \lVert \mathbf{x}_t - \eta \nabla f(\mathbf{x}_t) - \mathbf{x}^* \rVert_2^2$$

The rest of the proof follows the same steps as unconstrained GD, using the fact that $\mathbf{x}^* \in \mathcal{C}$ and the first-order optimality condition for constrained optimization: $\nabla f(\mathbf{x}^*)^\top(\mathbf{x} - \mathbf{x}^*) \geq 0$ for all $\mathbf{x} \in \mathcal{C}$. $\blacksquare$

### A.4 ADMM Convergence Analysis

**Theorem (ADMM Convergence).** Consider the problem $\min f(\mathbf{x}) + g(\mathbf{z})$ s.t. $A\mathbf{x} + B\mathbf{z} = \mathbf{c}$ where $f$ and $g$ are closed, proper, convex. Assume the augmented Lagrangian has a saddle point. Then ADMM with any $\rho > 0$ satisfies:

1. **Residual convergence:** $A\mathbf{x}_k + B\mathbf{z}_k - \mathbf{c} \to \mathbf{0}$
2. **Objective convergence:** $f(\mathbf{x}_k) + g(\mathbf{z}_k) \to p^*$
3. **Dual variable convergence:** $\mathbf{u}_k \to \mathbf{u}^*$

**Proof sketch.** The proof uses the fact that ADMM is equivalent to applying Douglas-Rachford splitting to the dual problem. The key inequality is:

$$\lVert \mathbf{u}_{k+1} - \mathbf{u}^* \rVert_2^2 + \rho \lVert B(\mathbf{z}_{k+1} - \mathbf{z}^*) \rVert_2^2 \leq \lVert \mathbf{u}_k - \mathbf{u}^* \rVert_2^2 + \rho \lVert B(\mathbf{z}_k - \mathbf{z}^*) \rVert_2^2$$

This shows that the sequence is non-increasing in a weighted norm, which implies convergence. The rate is $O(1/k)$ in an ergodic sense. $\blacksquare$

**For AI:** The $O(1/k)$ rate of ADMM is slower than the $O(1/k^2)$ rate of accelerated methods, but ADMM's advantage is its ability to split complex problems into simple subproblems. In distributed training, the communication cost per iteration is low (only the consensus variables are exchanged), making ADMM attractive despite the slower convergence rate.

### A.5 The Geometry of the KKT System

The KKT system can be viewed as finding a zero of the **KKT operator**:

$$F(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu}) = \begin{bmatrix} \nabla f(\mathbf{x}) + \sum_i \lambda_i \nabla g_i(\mathbf{x}) + \sum_j \nu_j \nabla h_j(\mathbf{x}) \\ \text{diag}(\boldsymbol{\lambda}) \mathbf{g}(\mathbf{x}) \\ \mathbf{h}(\mathbf{x}) \end{bmatrix} = \mathbf{0}$$

The Jacobian of $F$ at the solution is:

$$J = \begin{bmatrix} \nabla_{\mathbf{xx}}^2 \mathcal{L} & \nabla \mathbf{g}(\mathbf{x}^*) & \nabla \mathbf{h}(\mathbf{x}^*) \\ \text{diag}(\boldsymbol{\lambda}^*) \nabla \mathbf{g}(\mathbf{x}^*)^\top & \text{diag}(\mathbf{g}(\mathbf{x}^*)) & \mathbf{0} \\ \nabla \mathbf{h}(\mathbf{x}^*)^\top & \mathbf{0} & \mathbf{0} \end{bmatrix}$$

This matrix is non-singular under the second-order sufficient conditions and LICQ, which ensures that Newton's method for the KKT system converges quadratically.

**Active-set identification:** Near the solution, the active set can be identified from the signs of the constraint functions and multipliers. Once the active set is known, the problem reduces to an equality-constrained problem, which can be solved efficiently.

### A.6 Sensitivity Analysis

**Theorem (Sensitivity of the Optimal Solution).** Let $\mathbf{x}^*(\mathbf{u})$ be the optimal solution of the perturbed problem:

$$\min f(\mathbf{x}) \quad \text{s.t.} \quad g_i(\mathbf{x}) \leq u_i, \quad h_j(\mathbf{x}) = v_j$$

Under appropriate regularity conditions, the sensitivity of the optimal value is:

$$\frac{\partial p^*}{\partial u_i} = -\lambda_i^*, \quad \frac{\partial p^*}{\partial v_j} = -\nu_j^*$$

And the sensitivity of the optimal solution is:

$$\frac{\partial \mathbf{x}^*}{\partial u_i} = -[\nabla_{\mathbf{xx}}^2 \mathcal{L}]^{-1} \nabla g_i(\mathbf{x}^*)$$

**For AI:** Sensitivity analysis is crucial for understanding how changes in constraints affect the solution. In portfolio optimization, it tells us how the optimal portfolio changes when we adjust the target return or risk budget. In constrained RL, it reveals the trade-off curve between reward and safety cost.


---

## Appendix B: Worked Examples and Case Studies

### B.1 Complete KKT Analysis: Quadratic Program

Let us work through a complete KKT analysis of a simple QP.

**Problem:** $\min \frac{1}{2}(x_1^2 + x_2^2)$ subject to $x_1 + x_2 \geq 1$ and $x_1 \geq 0$.

Rewriting in standard form: $g_1(\mathbf{x}) = 1 - x_1 - x_2 \leq 0$, $g_2(\mathbf{x}) = -x_1 \leq 0$.

**Lagrangian:** $\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}) = \frac{1}{2}(x_1^2 + x_2^2) + \lambda_1(1 - x_1 - x_2) + \lambda_2(-x_1)$

**KKT conditions:**

1. **Stationarity:** $x_1 - \lambda_1 - \lambda_2 = 0$, $x_2 - \lambda_1 = 0$
2. **Primal feasibility:** $x_1 + x_2 \geq 1$, $x_1 \geq 0$
3. **Dual feasibility:** $\lambda_1 \geq 0$, $\lambda_2 \geq 0$
4. **Complementary slackness:** $\lambda_1(1 - x_1 - x_2) = 0$, $\lambda_2(-x_1) = 0$

**Case analysis:**

- **Case 1:** Both constraints active ($x_1 + x_2 = 1$, $x_1 = 0$). Then $x_2 = 1$. From stationarity: $0 - \lambda_1 - \lambda_2 = 0$ and $1 - \lambda_1 = 0$. So $\lambda_1 = 1$, $\lambda_2 = -1$. But $\lambda_2 < 0$ violates dual feasibility. **Infeasible.**

- **Case 2:** Only $g_1$ active ($x_1 + x_2 = 1$, $x_1 > 0$). Then $\lambda_2 = 0$. From stationarity: $x_1 = \lambda_1$, $x_2 = \lambda_1$. So $x_1 = x_2 = \lambda_1$. From $x_1 + x_2 = 1$: $2\lambda_1 = 1$, so $\lambda_1 = 1/2$, $x_1 = x_2 = 1/2$. Check: $\lambda_1 = 1/2 \geq 0$ ✓, $x_1 = 1/2 > 0$ ✓. **Valid KKT point.**

- **Case 3:** Only $g_2$ active ($x_1 = 0$, $x_1 + x_2 > 1$). Then $\lambda_1 = 0$. From stationarity: $x_1 = \lambda_2$, $x_2 = 0$. So $x_1 = \lambda_2$, $x_2 = 0$. But $x_1 + x_2 = \lambda_2 > 1$ requires $\lambda_2 > 1$, and $x_1 = \lambda_2 > 1 > 0$ contradicts $x_1 = 0$. **Infeasible.**

- **Case 4:** Neither constraint active. Then $\lambda_1 = \lambda_2 = 0$. From stationarity: $x_1 = x_2 = 0$. But $x_1 + x_2 = 0 < 1$ violates primal feasibility. **Infeasible.**

**Solution:** $\mathbf{x}^* = (1/2, 1/2)$, $\boldsymbol{\lambda}^* = (1/2, 0)$, $f^* = 1/4$.

**Geometric interpretation:** The unconstrained minimum $(0, 0)$ is outside the feasible set. The constrained minimum lies on the boundary $x_1 + x_2 = 1$, at the point closest to the origin. The gradient $\nabla f = (1/2, 1/2)$ is balanced by the constraint gradient $\nabla g_1 = (-1, -1)$ with multiplier $\lambda_1 = 1/2$.

### B.2 SVM Dual: Complete Derivation with Soft Margin

The soft-margin SVM primal:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}\lVert \mathbf{w} \rVert_2^2 + C\sum_{i=1}^n \xi_i \quad \text{s.t.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

**Lagrangian:**

$$\mathcal{L} = \frac{1}{2}\lVert \mathbf{w} \rVert_2^2 + C\sum_i \xi_i - \sum_i \alpha_i[y_i(\mathbf{w}^\top \mathbf{x}_i + b) - 1 + \xi_i] - \sum_i \mu_i \xi_i$$

**Stationarity conditions:**

$$\nabla_{\mathbf{w}} \mathcal{L} = \mathbf{w} - \sum_i \alpha_i y_i \mathbf{x}_i = \mathbf{0} \quad \Rightarrow \quad \mathbf{w} = \sum_i \alpha_i y_i \mathbf{x}_i$$

$$\frac{\partial \mathcal{L}}{\partial b} = -\sum_i \alpha_i y_i = 0$$

$$\frac{\partial \mathcal{L}}{\partial \xi_i} = C - \alpha_i - \mu_i = 0 \quad \Rightarrow \quad \alpha_i = C - \mu_i$$

Since $\mu_i \geq 0$, we have $\alpha_i \leq C$. Combined with $\alpha_i \geq 0$: $0 \leq \alpha_i \leq C$.

**Substituting into the Lagrangian:**

$$\mathcal{L} = \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^\top \mathbf{x}_j + C\sum_i \xi_i - \sum_{i,j} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^\top \mathbf{x}_j - b\sum_i \alpha_i y_i + \sum_i \alpha_i - \sum_i \alpha_i \xi_i - \sum_i \mu_i \xi_i$$

Using $\sum_i \alpha_i y_i = 0$ and $C - \alpha_i - \mu_i = 0$:

$$\mathcal{L} = -\frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^\top \mathbf{x}_j + \sum_i \alpha_i + \sum_i (C - \alpha_i - \mu_i)\xi_i$$

The last term vanishes. The dual is:

$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j K(\mathbf{x}_i, \mathbf{x}_j)$$
$$\text{s.t.} \quad 0 \leq \alpha_i \leq C, \quad \sum_{i=1}^n \alpha_i y_i = 0$$

**Complementary slackness interpretation:**
- $\alpha_i > 0 \implies y_i(\mathbf{w}^\top \mathbf{x}_i + b) = 1 - \xi_i$ (on or inside margin)
- $\alpha_i < C \implies \mu_i > 0 \implies \xi_i = 0$ (no slack)
- $0 < \alpha_i < C \implies$ point is exactly on the margin ($y_i(\mathbf{w}^\top \mathbf{x}_i + b) = 1$)
- $\alpha_i = C \implies$ point is inside the margin or misclassified ($\xi_i > 0$)

### B.3 ADMM for Distributed Consensus

Consider the problem of fitting a model to data distributed across $K$ workers:

$$\min_{\mathbf{w}} \sum_{k=1}^K f_k(\mathbf{w})$$

Reformulate with local copies $\mathbf{w}_k$ and a global variable $\mathbf{z}$:

$$\min_{\mathbf{w}_1, \ldots, \mathbf{w}_K, \mathbf{z}} \sum_{k=1}^K f_k(\mathbf{w}_k) \quad \text{s.t.} \quad \mathbf{w}_k = \mathbf{z}, \; k = 1, \ldots, K$$

**ADMM updates:**

1. **Local update (parallel):** $\mathbf{w}_k^{t+1} = \arg\min_{\mathbf{w}_k} f_k(\mathbf{w}_k) + \frac{\rho}{2}\lVert \mathbf{w}_k - \mathbf{z}^t + \mathbf{u}_k^t \rVert_2^2$
2. **Global averaging:** $\mathbf{z}^{t+1} = \frac{1}{K}\sum_{k=1}^K (\mathbf{w}_k^{t+1} + \mathbf{u}_k^t)$
3. **Dual update (parallel):** $\mathbf{u}_k^{t+1} = \mathbf{u}_k^t + \mathbf{w}_k^{t+1} - \mathbf{z}^{t+1}$

**Communication pattern:** Each worker sends its local model $\mathbf{w}_k$ to the server, which computes the average and broadcasts $\mathbf{z}$ back. The dual variables $\mathbf{u}_k$ are local to each worker.

**For AI:** This is the foundation of federated learning with ADMM. The local update step can use any optimizer (SGD, Adam), and the global averaging step ensures consensus across workers. The method is robust to communication delays and can handle partial participation.

### B.4 Comparison: All Constrained Optimization Methods

| Method | Problem Type | Convergence Rate | Per-iter Cost | Best For |
|---|---|---|---|---|
| Projected GD | Convex + simple constraints | $O(1/T)$ | $O(n)$ + projection | Large-scale, simple constraints |
| Penalty methods | General | Depends on $\mu$ | Same as unconstrained | When projection is hard |
| Barrier methods | Convex inequality | $O(\sqrt{m}\log(1/\epsilon))$ iters | $O(n^3)$ per Newton step | Medium-scale convex |
| Augmented Lagrangian | General | Superlinear | Depends on subproblem | When penalty methods fail |
| ADMM | Separable convex | $O(1/k)$ | Parallel subproblems | Distributed optimization |
| SQP | Non-convex | Quadratic (local) | QP solve per iter | Small-scale non-convex |
| Interior-point | Convex | $O(\sqrt{m}\log(1/\epsilon))$ iters | $O(n^3)$ per step | Medium-scale convex |

### B.5 Numerical Stability in KKT Systems

The KKT matrix for equality-constrained problems:

$$K = \begin{bmatrix} H & A^\top \\ A & \mathbf{0} \end{bmatrix}$$

is symmetric indefinite, which makes it challenging to solve numerically.

**Condition number:** The condition number of $K$ depends on both the conditioning of $H$ and the conditioning of $A$. If $A$ has nearly linearly dependent rows (nearly redundant constraints), $K$ becomes ill-conditioned.

**Regularization strategies:**
1. **Add small diagonal to $H$:** $H \leftarrow H + \delta I$ with $\delta > 0$ small
2. **Regularize the Schur complement:** Solve $(AH^{-1}A^\top)\boldsymbol{\nu} = \mathbf{b}$ with regularization
3. **Use iterative refinement:** Solve, compute residual, and correct

**For AI:** In neural network training with constraints, the KKT matrix can be very large and ill-conditioned. Iterative methods (MINRES, GMRES) with preconditioning are preferred over direct factorization.


---

## Appendix C: Numerical Implementation and Practical Considerations

### C.1 Computing Projections Efficiently

The efficiency of projected gradient descent depends entirely on how fast we can compute projections. Here are the key projections used in ML:

**Projection onto the $\ell_2$ ball** $\mathcal{B}_2(r) = \{\mathbf{x} : \lVert \mathbf{x} \rVert_2 \leq r\}$:

$$\Pi_{\mathcal{B}_2(r)}(\mathbf{y}) = \begin{cases} \mathbf{y} & \text{if } \lVert \mathbf{y} \rVert_2 \leq r \\ r \cdot \frac{\mathbf{y}}{\lVert \mathbf{y} \rVert_2} & \text{otherwise} \end{cases}$$

Cost: $O(n)$ for the norm computation.

**Projection onto the $\ell_1$ ball** $\mathcal{B}_1(r) = \{\mathbf{x} : \lVert \mathbf{x} \rVert_1 \leq r\}$:

This can be computed in $O(n \log n)$ time using the same algorithm as simplex projection (sort and threshold). The connection: projecting onto the $\ell_1$ ball is equivalent to projecting $|\mathbf{y}|$ onto the simplex and then restoring signs.

**Projection onto the PSD cone** $\mathbb{S}^n_+$:

$$\Pi_{\mathbb{S}^n_+}(X) = Q \max(\Lambda, 0) Q^\top$$

where $X = Q\Lambda Q^\top$ is the eigendecomposition. Cost: $O(n^3)$ for the eigendecomposition.

**Projection onto the nuclear norm ball** $\{X : \lVert X \rVert_* \leq r\}$:

$$\Pi(X) = U \cdot \Pi_{\ell_1(r)}(\boldsymbol{\sigma}) \cdot V^\top$$

where $X = U\Sigma V^\top$ is the SVD and $\Pi_{\ell_1(r)}$ projects the singular values onto the $\ell_1$ ball. Cost: $O(\min(mn^2, m^2n))$ for the SVD.

### C.2 Solving the KKT System Efficiently

For the equality-constrained QP:

$$\min \frac{1}{2}\mathbf{x}^\top P \mathbf{x} + \mathbf{q}^\top \mathbf{x} \quad \text{s.t.} \quad A\mathbf{x} = \mathbf{b}$$

The KKT system is:

$$\begin{bmatrix} P & A^\top \\ A & \mathbf{0} \end{bmatrix} \begin{bmatrix} \mathbf{x} \\ \boldsymbol{\nu} \end{bmatrix} = \begin{bmatrix} -\mathbf{q} \\ \mathbf{b} \end{bmatrix}$$

**Direct methods:**
- **LDL^T factorization:** Exploits symmetry. Cost: $O((n+p)^3/3)$.
- **Schur complement:** Eliminate $\mathbf{x}$ to get $AP^{-1}A^\top \boldsymbol{\nu} = A P^{-1}\mathbf{q} + \mathbf{b}$. Cost: $O(n^3 + p^3 + p^2n)$ when $p \ll n$.

**Iterative methods:**
- **MINRES:** For symmetric indefinite systems. Requires only matrix-vector products.
- **GMRES:** For general non-symmetric systems (when $P$ is not symmetric).
- **Preconditioning:** Use block-diagonal preconditioner $\text{diag}(P, AP^{-1}A^\top)$.

**For AI:** In large-scale constrained neural network training, the KKT system is too large for direct methods. Iterative methods with preconditioning are essential. The preconditioner can be approximated using diagonal or block-diagonal approximations of the Hessian.

### C.3 Handling Non-Convex Constraints

When the constraint functions $g_i$ are non-convex, the feasible set may be non-convex and the KKT conditions are only necessary (not sufficient) for local optimality.

**Sequential convex programming:** Approximate the non-convex constraints by convex ones at each iteration:

$$\tilde{g}_i(\mathbf{x}) = g_i(\mathbf{x}_k) + \nabla g_i(\mathbf{x}_k)^\top(\mathbf{x} - \mathbf{x}_k) \leq 0$$

This is the linearization used in SQP. The resulting convex subproblem is solved, and the process repeats.

**Trust region for non-convex constraints:** Add a trust region constraint $\lVert \mathbf{x} - \mathbf{x}_k \rVert \leq \Delta_k$ to ensure the linearization is accurate:

$$\min \nabla f(\mathbf{x}_k)^\top \mathbf{d} + \frac{1}{2}\mathbf{d}^\top B_k \mathbf{d} \quad \text{s.t.} \quad g_i(\mathbf{x}_k) + \nabla g_i(\mathbf{x}_k)^\top \mathbf{d} \leq 0, \quad \lVert \mathbf{d} \rVert \leq \Delta_k$$

**For AI:** Non-convex constraints appear in neural network verification (the network's output must satisfy certain properties for all inputs in a region), robust optimization (the model must be robust to adversarial perturbations), and constrained generative models (the generated samples must satisfy physical constraints).

### C.4 Warm-Starting Constrained Solvers

Warm-starting is crucial for efficiency when solving a sequence of related constrained problems:

**For projected GD:** Use the previous solution as the starting point. If the problem changes slowly, the previous solution is already close to the new optimum.

**For interior-point methods:** Use the previous central path point as the starting point for the new barrier problem. This is particularly effective when the barrier parameter $\mu$ decreases gradually.

**For active-set methods:** Use the previous active set as the initial guess. If the active set doesn't change much between iterations, this saves significant computation.

**For ADMM:** Use the previous primal and dual variables as the starting point. This is essential for online and streaming settings where the problem changes over time.

**For AI:** In online learning and streaming data, warm-starting is essential. Each new data point slightly modifies the optimization problem, and warm-starting from the previous solution allows for fast updates. This is used in online SVM, online portfolio optimization, and adaptive control.

### C.5 Constraint Handling in Deep Learning

Constrained deep learning is an active area of research. Common approaches include:

**1. Projection after each gradient step:**

$$\boldsymbol{\theta}_{t+1} = \Pi_{\mathcal{C}}(\boldsymbol{\theta}_t - \eta \nabla \mathcal{L}(\boldsymbol{\theta}_t))$$

This is simple but requires an efficient projection operator. For norm constraints, the projection is cheap. For complex constraints (e.g., fairness constraints), the projection may require solving a subproblem.

**2. Lagrangian relaxation:**

$$\mathcal{L}(\boldsymbol{\theta}, \boldsymbol{\lambda}) = \mathcal{L}_{\text{task}}(\boldsymbol{\theta}) + \sum_i \lambda_i g_i(\boldsymbol{\theta})$$

Alternate between gradient descent on $\boldsymbol{\theta}$ and gradient ascent on $\boldsymbol{\lambda}$. This is the approach used in constrained RL and fair ML.

**3. Barrier functions:**

$$\mathcal{L}_{\text{barrier}}(\boldsymbol{\theta}) = \mathcal{L}_{\text{task}}(\boldsymbol{\theta}) - \mu \sum_i \log(-g_i(\boldsymbol{\theta}))$$

This keeps the parameters strictly feasible but requires careful tuning of $\mu$.

**4. Penalty methods:**

$$\mathcal{L}_{\text{penalty}}(\boldsymbol{\theta}) = \mathcal{L}_{\text{task}}(\boldsymbol{\theta}) + \mu \sum_i \max(0, g_i(\boldsymbol{\theta}))^2$$

Simple to implement but can cause ill-conditioning for large $\mu$.

**For AI:** The choice of method depends on the constraint structure:
- **Simple constraints** (non-negativity, norm bounds): Use projection
- **Complex constraints** (fairness, safety): Use Lagrangian relaxation
- **Soft constraints** (approximately satisfied): Use penalty methods
- **Strict feasibility required**: Use barrier methods

### C.6 Debugging Constrained Optimization

Common issues and how to diagnose them:

**1. Infeasible problem:** The constraints are contradictory. Check by solving a feasibility problem: $\min 0$ s.t. constraints. If infeasible, relax constraints or check for errors.

**2. Ill-conditioned KKT system:** The constraint gradients are nearly linearly dependent. Check the condition number of the KKT matrix. Remove redundant constraints or add regularization.

**3. Slow convergence of projected GD:** The projection is expensive or the feasible set has a bad geometry. Check the condition number of the projection operator. Use a better preconditioner or switch to ADMM.

**4. Dual variables diverging:** The problem is infeasible or the step size for dual updates is too large. Reduce the dual step size or check feasibility.

**5. Complementary slackness violation:** The algorithm hasn't converged. Check the duality gap. Increase the number of iterations or tighten the convergence tolerance.

**For AI:** In constrained neural network training, the most common issue is the trade-off between constraint satisfaction and task performance. Monitoring both the training loss and the constraint violation during training is essential. If the constraint violation doesn't decrease, the constraint may be too tight or the learning rate may be too large.


---

## Appendix D: Extended Case Studies in Machine Learning

### D.1 Fair Classification with Demographic Parity Constraints

Consider binary classification with a sensitive attribute $A \in \{0, 1\}$. The demographic parity constraint requires:

$$|P(\hat{Y} = 1 \mid A = 0) - P(\hat{Y} = 1 \mid A = 1)| \leq \epsilon$$

For a linear classifier $\hat{Y} = \text{sign}(\mathbf{w}^\top \mathbf{x} + b)$, this can be approximated as:

$$\left|\frac{1}{n_0}\sum_{i: A_i = 0} \sigma(\mathbf{w}^\top \mathbf{x}_i + b) - \frac{1}{n_1}\sum_{i: A_i = 1} \sigma(\mathbf{w}^\top \mathbf{x}_i + b)\right| \leq \epsilon$$

where $\sigma$ is the sigmoid function and $n_a$ is the number of samples with $A = a$.

This is a non-convex constraint (due to the sigmoid), but it can be handled using the augmented Lagrangian method:

$$\mathcal{L}_\mu(\mathbf{w}, b, \lambda) = \mathcal{L}_{\text{CE}}(\mathbf{w}, b) + \lambda \cdot h(\mathbf{w}, b) + \frac{\mu}{2} h(\mathbf{w}, b)^2$$

where $h(\mathbf{w}, b)$ is the fairness constraint violation.

**Training procedure:**
1. Initialize $\mathbf{w}, b, \lambda = 0, \mu = 1$
2. For each epoch:
   - Minimize $\mathcal{L}_\mu$ with respect to $\mathbf{w}, b$ using SGD
   - Update $\lambda \leftarrow \lambda + \mu \cdot h(\mathbf{w}, b)$
   - If constraint violation is not decreasing, increase $\mu \leftarrow 2\mu$
3. Return the model with the smallest constraint violation

**Trade-off analysis:** By varying $\epsilon$, we can trace the Pareto frontier between accuracy and fairness. This reveals the "price of fairness" — how much accuracy must be sacrificed to achieve a given level of demographic parity.

### D.2 Robust Optimization with Uncertainty Sets

Robust optimization addresses the problem of optimizing under uncertainty in the problem data:

$$\min_{\mathbf{x}} \max_{\mathbf{u} \in \mathcal{U}} f(\mathbf{x}, \mathbf{u})$$

where $\mathcal{U}$ is an uncertainty set. For example, in robust linear regression:

$$\min_{\mathbf{w}} \max_{\Delta X : \lVert \Delta X \rVert_F \leq \rho} \lVert (X + \Delta X)\mathbf{w} - \mathbf{y} \rVert_2^2$$

The inner maximization can be solved analytically:

$$\max_{\lVert \Delta X \rVert_F \leq \rho} \lVert (X + \Delta X)\mathbf{w} - \mathbf{y} \rVert_2^2 = \lVert X\mathbf{w} - \mathbf{y} \rVert_2^2 + 2\rho \lVert \mathbf{w} \rVert_2 \lVert X\mathbf{w} - \mathbf{y} \rVert_2 + \rho^2 \lVert \mathbf{w} \rVert_2^2$$

This is equivalent to a regularized least squares problem, revealing the connection between robustness and regularization.

**For AI:** Robust optimization is used for:
- **Adversarial training:** The uncertainty set contains adversarial perturbations
- **Distributionally robust optimization:** The uncertainty set contains probability distributions close to the empirical distribution
- **Robust portfolio optimization:** The uncertainty set contains possible returns and covariances

### D.3 Constrained Neural Architecture Search

Neural architecture search (NAS) with constraints:

$$\min_{\alpha} \mathcal{L}_{\text{val}}(\mathbf{w}^*(\alpha), \alpha) \quad \text{s.t.} \quad \text{FLOPs}(\alpha) \leq B, \quad \text{latency}(\alpha) \leq L$$

where $\alpha$ are the architecture parameters and $\mathbf{w}^*(\alpha)$ are the optimal weights for architecture $\alpha$.

This is a bilevel optimization problem with constraints. The standard approach uses:
1. **Relaxation:** Replace discrete architecture choices with continuous parameters
2. **Penalty method:** Add FLOPs and latency as penalty terms to the objective
3. **Projected gradient:** After each gradient step, project onto the feasible set of architectures

**For AI:** Constrained NAS produces models that are both accurate and deployable on resource-constrained devices (mobile phones, edge devices). The constraints ensure that the searched architecture meets the latency and memory requirements of the target platform.

### D.4 Optimal Transport with Marginal Constraints

The optimal transport problem finds the minimum-cost way to transport mass from one distribution to another:

$$\min_{P \in \mathbb{R}^{n \times m}} \sum_{i,j} C_{ij} P_{ij} \quad \text{s.t.} \quad P\mathbf{1} = \mathbf{a}, \quad P^\top \mathbf{1} = \mathbf{b}, \quad P \geq 0$$

where $\mathbf{a} \in \Delta_n$ and $\mathbf{b} \in \Delta_m$ are the source and target distributions, and $C$ is the cost matrix.

This is a linear program with $nm$ variables and $n + m$ equality constraints. The Sinkhorn algorithm solves it efficiently using entropy regularization:

$$\min_P \sum_{i,j} C_{ij} P_{ij} - \epsilon H(P) \quad \text{s.t.} \quad P\mathbf{1} = \mathbf{a}, \quad P^\top \mathbf{1} = \mathbf{b}$$

where $H(P) = -\sum_{i,j} P_{ij} \log P_{ij}$ is the entropy. The solution has the form $P_{ij} = u_i K_{ij} v_j$ where $K_{ij} = \exp(-C_{ij}/\epsilon)$, and the scaling vectors $\mathbf{u}, \mathbf{v}$ are found by iterative matrix scaling (Sinkhorn iterations).

**For AI:** Optimal transport is used for:
- **Domain adaptation:** Aligning source and target feature distributions
- **Generative modeling:** Wasserstein GANs use the Wasserstein distance (optimal transport cost) as the training objective
- **Attention mechanisms:** Optimal transport provides a structured alternative to softmax attention

### D.5 Constrained Policy Optimization in Robotics

In robotics, safety constraints are critical. A robot arm must reach its target while avoiding obstacles and respecting joint limits:

$$\max_{\pi} \mathbb{E}\left[\sum_{t=0}^T r(s_t, a_t)\right] \quad \text{s.t.} \quad \mathbb{E}\left[\sum_{t=0}^T c_j(s_t, a_t)\right] \leq d_j, \quad j = 1, \ldots, m$$

where $c_j$ are cost functions (e.g., distance to obstacles, joint velocity limits).

**Constrained Policy Optimization (CPO)** (Achiam et al., 2017) solves this using a trust region approach:

$$\max_{\pi} \mathbb{E}_{s \sim d_\pi, a \sim \pi}[A_\pi(s, a)] \quad \text{s.t.} \quad D_{\text{KL}}(\pi_{\text{old}} \| \pi) \leq \delta, \quad \mathbb{E}[A_{c_j, \pi}(s, a)] \leq \frac{d_j - V_{c_j, \pi_{\text{old}}}}{1 - \gamma}$$

This is a constrained optimization problem over the policy space. CPO solves it approximately by linearizing the objective and constraints and solving the resulting quadratic program.

**For AI:** Constrained policy optimization is essential for deploying RL in the real world. Applications include autonomous driving (safety constraints), medical treatment (dosage constraints), and industrial control (operating range constraints).

### D.6 Comparison: Constrained Optimization Methods for Deep Learning

| Method | Constraint Types | Scalability | Constraint Satisfaction | Best Use Case |
|---|---|---|---|---|
| Projected GD | Convex sets | Excellent (O(n)) | Exact (if projection exact) | Simple constraints (norm, non-negativity) |
| Lagrangian relaxation | Any differentiable | Excellent (O(n)) | Approximate (depends on dual convergence) | Complex constraints (fairness, safety) |
| Penalty methods | Any differentiable | Excellent (O(n)) | Approximate (improves with μ) | Soft constraints |
| Barrier methods | Inequality | Good (O(n)) | Exact (interior) | Strict inequality constraints |
| ADMM | Separable | Excellent (parallel) | Exact (at convergence) | Distributed training |
| SQP | Any smooth | Poor (O(n³)) | Exact | Small-scale non-convex |
| Interior-point | Convex | Poor (O(n³)) | Exact (interior) | Medium-scale convex |

**Recommendations for deep learning:**
- **Norm constraints on weights:** Projected GD (cheap projection)
- **Fairness constraints:** Lagrangian relaxation (flexible, easy to implement)
- **Safety constraints in RL:** CPO or Lagrangian PPO (proven safety guarantees)
- **Distributed training with consensus:** ADMM (natural parallelism)
- **Architecture search with FLOPs constraints:** Penalty method (easy to differentiate)


---

## Appendix E: Numerical Methods and Algorithmic Details

### E.1 The Alternating Direction Method of Multipliers: Full Derivation

ADMM solves problems of the form:

$$\min_{\mathbf{x}, \mathbf{z}} f(\mathbf{x}) + g(\mathbf{z}) \quad \text{s.t.} \quad A\mathbf{x} + B\mathbf{z} = \mathbf{c}$$

The augmented Lagrangian is:

$$\mathcal{L}_\rho(\mathbf{x}, \mathbf{z}, \mathbf{u}) = f(\mathbf{x}) + g(\mathbf{z}) + \mathbf{u}^\top(A\mathbf{x} + B\mathbf{z} - \mathbf{c}) + \frac{\rho}{2}\lVert A\mathbf{x} + B\mathbf{z} - \mathbf{c} \rVert_2^2$$

**ADMM iterations:**

1. **x-update:** $\mathbf{x}_{k+1} = \arg\min_{\mathbf{x}} \mathcal{L}_\rho(\mathbf{x}, \mathbf{z}_k, \mathbf{u}_k)$
2. **z-update:** $\mathbf{z}_{k+1} = \arg\min_{\mathbf{z}} \mathcal{L}_\rho(\mathbf{x}_{k+1}, \mathbf{z}, \mathbf{u}_k)$
3. **u-update:** $\mathbf{u}_{k+1} = \mathbf{u}_k + \rho(A\mathbf{x}_{k+1} + B\mathbf{z}_{k+1} - \mathbf{c})$

**Scaled form:** Define the scaled dual variable $\mathbf{u} = \frac{1}{\rho}\boldsymbol{\lambda}$. Then:

$$\mathcal{L}_\rho(\mathbf{x}, \mathbf{z}, \mathbf{u}) = f(\mathbf{x}) + g(\mathbf{z}) + \frac{\rho}{2}\lVert A\mathbf{x} + B\mathbf{z} - \mathbf{c} + \mathbf{u} \rVert_2^2 - \frac{\rho}{2}\lVert \mathbf{u} \rVert_2^2$$

The scaled ADMM iterations are:

1. $\mathbf{x}_{k+1} = \arg\min_{\mathbf{x}} f(\mathbf{x}) + \frac{\rho}{2}\lVert A\mathbf{x} + B\mathbf{z}_k - \mathbf{c} + \mathbf{u}_k \rVert_2^2$
2. $\mathbf{z}_{k+1} = \arg\min_{\mathbf{z}} g(\mathbf{z}) + \frac{\rho}{2}\lVert A\mathbf{x}_{k+1} + B\mathbf{z} - \mathbf{c} + \mathbf{u}_k \rVert_2^2$
3. $\mathbf{u}_{k+1} = \mathbf{u}_k + A\mathbf{x}_{k+1} + B\mathbf{z}_{k+1} - \mathbf{c}$

**Consensus form:** For the problem $\min \sum_{i=1}^N f_i(\mathbf{x}_i)$ s.t. $\mathbf{x}_i = \mathbf{z}$:

1. $\mathbf{x}_i^{k+1} = \arg\min_{\mathbf{x}_i} f_i(\mathbf{x}_i) + \frac{\rho}{2}\lVert \mathbf{x}_i - \mathbf{z}^k + \mathbf{u}_i^k \rVert_2^2$
2. $\mathbf{z}^{k+1} = \frac{1}{N}\sum_{i=1}^N (\mathbf{x}_i^{k+1} + \mathbf{u}_i^k)$
3. $\mathbf{u}_i^{k+1} = \mathbf{u}_i^k + \mathbf{x}_i^{k+1} - \mathbf{z}^{k+1}$

**Choosing $\rho$:** The penalty parameter $\rho$ affects convergence speed but not the final solution. Common strategies:
- Fixed $\rho$: Simple but may be slow
- Adaptive $\rho$: Increase $\rho$ when primal residual is large, decrease when dual residual is large
- Balancing: Choose $\rho$ to balance primal and dual residuals

### E.2 The Interior-Point Method: Barrier Function Analysis

The logarithmic barrier function for inequality constraints $g_i(\mathbf{x}) \leq 0$ is:

$$\phi(\mathbf{x}) = -\sum_{i=1}^m \log(-g_i(\mathbf{x}))$$

The barrier problem is:

$$\min_{\mathbf{x}} tf(\mathbf{x}) + \phi(\mathbf{x}) \quad \text{s.t.} \quad h_j(\mathbf{x}) = 0$$

where $t > 0$ is the barrier parameter. As $t \to \infty$, the solution converges to the optimal solution of the original problem.

**Central path:** The set of solutions $\mathbf{x}^*(t)$ for all $t > 0$ forms the central path. The duality gap at $\mathbf{x}^*(t)$ is approximately $m/t$.

**Newton step for the barrier problem:** The Newton step $\Delta \mathbf{x}_{\text{nt}}$ solves:

$$(t\nabla^2 f(\mathbf{x}) + \nabla^2 \phi(\mathbf{x}))\Delta \mathbf{x}_{\text{nt}} = -(t\nabla f(\mathbf{x}) + \nabla \phi(\mathbf{x}))$$

The Hessian of the barrier is:

$$\nabla^2 \phi(\mathbf{x}) = \sum_{i=1}^m \frac{1}{g_i(\mathbf{x})^2}\nabla g_i(\mathbf{x})\nabla g_i(\mathbf{x})^\top - \sum_{i=1}^m \frac{1}{g_i(\mathbf{x})}\nabla^2 g_i(\mathbf{x})$$

For linear constraints $g_i(\mathbf{x}) = \mathbf{a}_i^\top \mathbf{x} - b_i$, the second term vanishes and:

$$\nabla^2 \phi(\mathbf{x}) = \sum_{i=1}^m \frac{1}{(\mathbf{a}_i^\top \mathbf{x} - b_i)^2}\mathbf{a}_i\mathbf{a}_i^\top$$

**Backtracking line search:** The Newton step is scaled to ensure strict feasibility:

$$\mathbf{x}_{\text{new}} = \mathbf{x} + s \cdot \Delta \mathbf{x}_{\text{nt}}$$

where $s \in (0, 1]$ is chosen by backtracking to ensure $g_i(\mathbf{x}_{\text{new}}) < 0$ for all $i$.

**Complexity analysis:** The number of Newton steps per centering step is bounded by a constant (typically 10-20). The number of centering steps is $O(\sqrt{m} \log(m/\epsilon))$. Each Newton step costs $O(n^3)$ for the factorization. Total complexity: $O(n^3 \sqrt{m} \log(1/\epsilon))$.

### E.3 The Projected Newton Method

For problems with simple constraints (non-negativity, box constraints), the projected Newton method combines Newton's method with projection:

$$\mathbf{x}_{k+1} = \Pi_{\mathcal{C}}(\mathbf{x}_k - [\nabla^2 f(\mathbf{x}_k)]^{-1} \nabla f(\mathbf{x}_k))$$

**Active set identification:** At each iteration, identify the active constraints:

$$\mathcal{A}_k = \{i : x_{k,i} = 0 \text{ and } [\nabla f(\mathbf{x}_k)]_i > 0\}$$

Then solve the reduced Newton system on the free variables:

$$[\nabla^2 f(\mathbf{x}_k)]_{\mathcal{F}\mathcal{F}} \Delta \mathbf{x}_{\mathcal{F}} = -[\nabla f(\mathbf{x}_k)]_{\mathcal{F}}$$

where $\mathcal{F} = \{1, \ldots, n\} \setminus \mathcal{A}_k$ is the set of free variables.

**Convergence:** The projected Newton method converges quadratically near the solution if the active set is correctly identified. The active set identification is exact near the solution under the strict complementarity condition ($\lambda_i^* > 0$ for all active constraints).

**For AI:** The projected Newton method is used for non-negative least squares (NNLS), non-negative matrix factorization (NMF), and L1-regularized problems (via the connection to box constraints on the positive and negative parts).

### E.4 The Method of Multipliers: Convergence Analysis

The method of multipliers (augmented Lagrangian method) for equality constraints:

$$\min f(\mathbf{x}) \quad \text{s.t.} \quad h(\mathbf{x}) = \mathbf{0}$$

**Algorithm:**
1. $\mathbf{x}_{k+1} = \arg\min_{\mathbf{x}} f(\mathbf{x}) + \frac{\mu_k}{2}\lVert h(\mathbf{x}) \rVert_2^2 + \boldsymbol{\lambda}_k^\top h(\mathbf{x})$
2. $\boldsymbol{\lambda}_{k+1} = \boldsymbol{\lambda}_k + \mu_k h(\mathbf{x}_{k+1})$

**Convergence theorem:** If $f$ is convex and $h$ is affine, and the problem has a solution, then:
- $\mathbf{x}_k \to \mathbf{x}^*$ (primal convergence)
- $\boldsymbol{\lambda}_k \to \boldsymbol{\lambda}^*$ (dual convergence)
- The convergence rate is linear with factor $O(1/\mu_k)$

**Key insight:** Unlike pure penalty methods, the method of multipliers converges for finite $\mu_k$. The multipliers $\boldsymbol{\lambda}_k$ adjust to enforce the constraints without requiring $\mu_k \to \infty$. This avoids the ill-conditioning that plagues pure penalty methods.

**Practical considerations:**
- **Initial $\mu_0$:** Start with a moderate value (e.g., $\mu_0 = 1$)
- **Update rule:** Increase $\mu_k$ if the constraint violation doesn't decrease sufficiently
- **Subproblem accuracy:** The subproblem doesn't need to be solved exactly — an approximate solution is sufficient

### E.5 Constraint Qualifications: Detailed Analysis

**LICQ (Linear Independence Constraint Qualification):**

The gradients of active constraints are linearly independent. This is the strongest and most commonly used constraint qualification.

**Verification:** Form the matrix $J = [\nabla g_i(\mathbf{x}^*)]_{i \in \mathcal{A}}$ and check that $\text{rank}(J) = |\mathcal{A}|$.

**Slater's Condition:**

For convex problems, there exists a strictly feasible point. This is the most important constraint qualification for convex optimization.

**Verification:** Find a point $\mathbf{x}$ such that $g_i(\mathbf{x}) < 0$ for all $i$ and $h_j(\mathbf{x}) = 0$ for all $j$. For many ML problems, this is easy: for the SVM, any point with sufficiently large margin satisfies Slater's condition.

**MFCQ (Mangasarian-Fromovitz Constraint Qualification):**

There exists a direction $\mathbf{d}$ such that $\nabla g_i(\mathbf{x}^*)^\top \mathbf{d} < 0$ for all active $i$ and $\nabla h_j(\mathbf{x}^*)^\top \mathbf{d} = 0$ for all $j$.

**Relationship between constraint qualifications:**

$$\text{LICQ} \implies \text{MFCQ} \implies \text{Abadie CQ}$$

For convex problems, Slater's condition implies MFCQ, which implies the KKT conditions are necessary.

**For AI:** In most ML applications, Slater's condition is easy to verify. For the SVM, any point with large enough weights satisfies it. For non-negative matrix factorization, any strictly positive factorization satisfies it. For constrained neural networks, if there exists a feasible network (which is usually the case), Slater's condition holds.

### E.6 The Dual Problem in Practice: Computing the Dual Function

The dual function is $g(\boldsymbol{\lambda}, \boldsymbol{\nu}) = \inf_{\mathbf{x}} \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\nu})$. Computing this requires solving an unconstrained optimization problem for each $(\boldsymbol{\lambda}, \boldsymbol{\nu})$.

**Closed-form dual function:** For some problems, the infimum can be computed analytically:

- **Quadratic objective:** $\min \frac{1}{2}\mathbf{x}^\top P \mathbf{x} + \mathbf{q}^\top \mathbf{x}$ s.t. $A\mathbf{x} = \mathbf{b}$
  - Dual: $\max -\frac{1}{2}\boldsymbol{\nu}^\top A P^{-1} A^\top \boldsymbol{\nu} - \mathbf{b}^\top \boldsymbol{\nu} - \frac{1}{2}\mathbf{q}^\top P^{-1} \mathbf{q}$

- **Entropy objective:** $\min \sum_i x_i \log x_i$ s.t. $A\mathbf{x} = \mathbf{b}$, $\mathbf{x} \geq \mathbf{0}$
  - Dual: $\max -\mathbf{b}^\top \boldsymbol{\nu} - \sum_i \exp(-\mathbf{a}_i^\top \boldsymbol{\nu} - 1)$

**Numerical dual function:** When no closed form exists, the dual function is evaluated numerically by solving the inner minimization problem. This is expensive but necessary for general problems.

**Subgradient of the dual:** If $\mathbf{x}^*(\boldsymbol{\lambda}, \boldsymbol{\nu})$ achieves the infimum, then:

$$\frac{\partial g}{\partial \lambda_i} = g_i(\mathbf{x}^*), \quad \frac{\partial g}{\partial \nu_j} = h_j(\mathbf{x}^*)$$

This allows subgradient ascent on the dual problem, which is the basis of dual decomposition methods.

**For AI:** Dual decomposition is used for distributed optimization, where each worker solves a local subproblem and the dual variables coordinate the solutions. This is the foundation of federated learning with dual averaging and distributed SVM training.


---

## Appendix F: Extended Algorithmic Implementations

### F.1 Projected Gradient Descent: Complete Implementation

```python
def projected_gradient_descent(f, grad_f, proj_C, x0, eta=0.01, T=1000, tol=1e-8):
    """
    Projected Gradient Descent for min f(x) s.t. x in C.
    
    Args:
        f: objective function
        grad_f: gradient of objective
        proj_C: projection operator onto feasible set C
        x0: initial point
        eta: learning rate
        T: maximum iterations
        tol: convergence tolerance
    
    Returns:
        x: optimal solution
        history: list of (f(x), constraint_violation) tuples
    """
    x = x0.copy()
    history = []
    
    for t in range(T):
        g = grad_f(x)
        x_unproj = x - eta * g
        x_new = proj_C(x_unproj)
        
        # Check convergence
        step_size = np.linalg.norm(x_new - x)
        grad_norm = np.linalg.norm(g)
        
        history.append((f(x_new), step_size, grad_norm))
        
        if step_size < tol and grad_norm < tol:
            break
        
        x = x_new
    
    return x, history
```

**Projection operators:**

```python
def proj_nonnegative(x):
    """Project onto non-negative orthant."""
    return np.maximum(x, 0)

def proj_l2_ball(x, radius=1.0):
    """Project onto L2 ball of given radius."""
    norm = np.linalg.norm(x)
    if norm <= radius:
        return x
    return radius * x / norm

def proj_simplex(x):
    """Project onto probability simplex (O(n log n))."""
    n = len(x)
    u = np.sort(x)[::-1]
    cssv = np.cumsum(u) - 1
    ind = np.arange(n) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    return np.maximum(x - theta, 0)

def proj_box(x, lower, upper):
    """Project onto box constraints [lower, upper]."""
    return np.clip(x, lower, upper)
```

### F.2 ADMM: Complete Implementation for Lasso

```python
def admm_lasso(A, b, lam, rho=1.0, max_iter=1000, tol=1e-6):
    """
    Solve Lasso: min 0.5*||Ax - b||^2 + lam*||z||_1 s.t. x = z
    
    Args:
        A: design matrix (n x p)
        b: response vector (n,)
        lam: regularization parameter
        rho: ADMM penalty parameter
        max_iter: maximum iterations
        tol: convergence tolerance
    
    Returns:
        x: solution
        history: convergence history
    """
    n, p = A.shape
    x = np.zeros(p)
    z = np.zeros(p)
    u = np.zeros(p)
    
    # Precompute: (A^T A + rho I)^{-1}
    AtA = A.T @ A
    L = np.linalg.cholesky(AtA + rho * np.eye(p))
    
    history = {'primal_res': [], 'dual_res': [], 'objective': []}
    
    for k in range(max_iter):
        # x-update: solve (A^T A + rho I) x = A^T b + rho (z - u)
        rhs = A.T @ b + rho * (z - u)
        x = np.linalg.solve(L.T, np.linalg.solve(L, rhs))
        
        # z-update: soft thresholding
        z_old = z.copy()
        z = soft_threshold(x + u, lam / rho)
        
        # u-update: dual ascent
        u = u + x - z
        
        # Compute residuals
        primal_res = np.linalg.norm(x - z)
        dual_res = np.linalg.norm(rho * (z - z_old))
        
        obj = 0.5 * np.linalg.norm(A @ x - b)**2 + lam * np.linalg.norm(z, 1)
        history['primal_res'].append(primal_res)
        history['dual_res'].append(dual_res)
        history['objective'].append(obj)
        
        # Check convergence
        if primal_res < tol and dual_res < tol:
            break
    
    return x, history

def soft_threshold(x, threshold):
    """Soft thresholding operator (proximal operator of L1 norm)."""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
```

### F.3 Interior-Point Method for Linear Programming

```python
def interior_point_lp(c, A, b, max_iter=100, tol=1e-8):
    """
    Primal-dual interior-point method for LP:
    min c^T x s.t. Ax = b, x >= 0
    
    Args:
        c: objective coefficients
        A: constraint matrix
        b: right-hand side
        max_iter: maximum iterations
        tol: convergence tolerance
    
    Returns:
        x: optimal solution
        y: dual variables for equality constraints
        s: dual slack variables
    """
    m, n = A.shape
    
    # Initialize strictly feasible point
    x = np.ones(n)
    y = np.zeros(m)
    s = np.ones(n)
    
    mu = x @ s / n  # Initial duality gap
    
    for k in range(max_iter):
        # Check convergence
        primal_res = np.linalg.norm(A @ x - b)
        dual_res = np.linalg.norm(A.T @ y + s - c)
        
        if primal_res < tol and dual_res < tol and mu < tol:
            break
        
        # Compute search direction
        X = np.diag(x)
        S = np.diag(s)
        
        # Solve for dy: (A X S^{-1} A^T) dy = A X S^{-1} (c - A^T y - s) - (Ax - b)
        # Simplified: use the standard primal-dual system
        rhs1 = c - A.T @ y - s
        rhs2 = -A @ x + b
        
        # Form the augmented system
        M = np.block([
            [np.zeros((m, m)), A],
            [A.T, np.diag(s / x)]
        ])
        rhs = np.concatenate([rhs2, -rhs1 + mu * np.ones(n) / x])
        
        delta = np.linalg.solve(M, rhs)
        dy = delta[:m]
        dx_ds = delta[m:]
        dx = dx_ds[:n]
        ds = dx_ds[n:]
        
        # Line search: ensure x + alpha*dx > 0 and s + alpha*ds > 0
        alpha_x = 1.0
        alpha_s = 1.0
        
        for i in range(n):
            if dx[i] < 0:
                alpha_x = min(alpha_x, -0.99 * x[i] / dx[i])
            if ds[i] < 0:
                alpha_s = min(alpha_s, -0.99 * s[i] / ds[i])
        
        # Update
        x = x + alpha_x * dx
        y = y + alpha_s * dy
        s = s + alpha_s * ds
        
        mu = x @ s / n
    
    return x, y, s
```

### F.4 Sequential Quadratic Programming: Implementation

```python
def sqp(f, grad_f, hess_f, g, grad_g, h, grad_h, x0, max_iter=100, tol=1e-8):
    """
    Sequential Quadratic Programming for:
    min f(x) s.t. g(x) <= 0, h(x) = 0
    
    Args:
        f, grad_f, hess_f: objective and its derivatives
        g, grad_g: inequality constraints and their gradients
        h, grad_h: equality constraints and their gradients
        x0: initial point
        max_iter: maximum iterations
        tol: convergence tolerance
    
    Returns:
        x: optimal solution
    """
    x = x0.copy()
    n = len(x)
    m = len(g(x)) if callable(g) else 0
    p = len(h(x)) if callable(h) else 0
    
    for k in range(max_iter):
        # Evaluate functions at current point
        fk = f(x)
        gk = grad_f(x)
        Hk = hess_f(x)
        
        g_vals = g(x) if m > 0 else np.array([])
        Gk = grad_g(x) if m > 0 else np.array([]).reshape(0, n)
        
        h_vals = h(x) if p > 0 else np.array([])
        Hk_eq = grad_h(x) if p > 0 else np.array([]).reshape(0, n)
        
        # Check KKT conditions
        # (simplified: check gradient norm and constraint violation)
        kkt_violation = np.linalg.norm(gk)
        constraint_violation = 0
        if m > 0:
            constraint_violation += np.sum(np.maximum(g_vals, 0)**2)
        if p > 0:
            constraint_violation += np.sum(h_vals**2)
        
        if kkt_violation < tol and constraint_violation < tol:
            break
        
        # Solve QP subproblem (using scipy for simplicity)
        # min 0.5 * d^T Hk d + gk^T d
        # s.t. Gk * d + g_vals <= 0
        #      Hk_eq * d + h_vals = 0
        
        from scipy.optimize import minimize
        
        def qp_obj(d):
            return 0.5 * d @ Hk @ d + gk @ d
        
        def qp_grad(d):
            return Hk @ d + gk
        
        constraints = []
        if p > 0:
            constraints.append({'type': 'eq', 'fun': lambda d: Hk_eq @ d + h_vals,
                               'jac': lambda d: Hk_eq})
        if m > 0:
            constraints.append({'type': 'ineq', 'fun': lambda d: -(Gk @ d + g_vals),
                               'jac': lambda d: -Gk})
        
        d0 = np.zeros(n)
        res = minimize(qp_obj, d0, jac=qp_grad, constraints=constraints, method='SLSQP')
        d = res.x
        
        # Line search (Armijo)
        alpha = 1.0
        merit = lambda x: f(x) + 100 * (np.sum(np.maximum(g(x), 0)**2) + np.sum(h(x)**2))
        
        while merit(x + alpha * d) > merit(x) + 0.01 * alpha * gk @ d:
            alpha *= 0.5
            if alpha < 1e-10:
                break
        
        x = x + alpha * d
    
    return x
```

### F.5 Constrained Optimization in PyTorch

```python
import torch

class ConstrainedOptimizer:
    """Wrapper for constrained optimization in PyTorch."""
    
    def __init__(self, model, base_optimizer, proj_fn=None, constraints=None):
        self.model = model
        self.base_optimizer = base_optimizer
        self.proj_fn = proj_fn  # Projection function for parameters
        self.constraints = constraints or []  # List of constraint functions
    
    def step(self):
        # Standard optimizer step
        self.base_optimizer.step()
        
        # Project parameters onto feasible set
        if self.proj_fn is not None:
            with torch.no_grad():
                for param in self.model.parameters():
                    param.data = self.proj_fn(param.data)
        
        # Check constraint violations
        violations = []
        for constraint in self.constraints:
            violation = constraint(self.model)
            violations.append(violation)
        
        return violations
    
    def lagrangian_step(self, lagrange_multipliers):
        """Step using Lagrangian relaxation of constraints."""
        # Compute Lagrangian gradient
        total_loss = self.model.loss()
        for i, constraint in enumerate(self.constraints):
            total_loss = total_loss + lagrange_multipliers[i] * constraint(self.model)
        
        total_loss.backward()
        self.base_optimizer.step()
        self.base_optimizer.zero_grad()

# Example: Non-negative constrained neural network
def proj_nonnegative(tensor):
    return torch.clamp(tensor, min=0)

model = MyNeuralNetwork()
base_optim = torch.optim.Adam(model.parameters(), lr=1e-3)
constrained_optim = ConstrainedOptimizer(model, base_optim, proj_fn=proj_nonnegative)

for epoch in range(100):
    loss = model.compute_loss(data)
    loss.backward()
    violations = constrained_optim.step()
```

### F.6 Debugging and Diagnostics for Constrained Optimization

```python
class ConstrainedOptimizationDiagnostics:
    """Diagnostic tools for constrained optimization."""
    
    def __init__(self):
        self.history = {
            'objective': [],
            'primal_residual': [],
            'dual_residual': [],
            'duality_gap': [],
            'constraint_violations': [],
            'lagrange_multipliers': [],
        }
    
    def record(self, x, lam, nu, f_val, g_vals, h_vals):
        """Record diagnostics at current iteration."""
        self.history['objective'].append(f_val)
        
        # Primal residual: constraint violations
        primal_res = np.sum(np.maximum(g_vals, 0)**2) + np.sum(h_vals**2)
        self.history['primal_residual'].append(primal_res)
        
        # Dual residual: stationarity violation
        # (requires gradient computation)
        
        # Duality gap: f(x) - g(lam, nu)
        # (requires dual function evaluation)
        
        self.history['constraint_violations'].append(g_vals.copy())
        self.history['lagrange_multipliers'].append(lam.copy())
    
    def plot_convergence(self):
        """Plot convergence diagnostics."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Objective value
        axes[0, 0].plot(self.history['objective'])
        axes[0, 0].set_title('Objective Value')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('f(x)')
        
        # Primal residual
        axes[0, 1].plot(self.history['primal_residual'])
        axes[0, 1].set_title('Primal Residual')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('||constraints||^2')
        axes[0, 1].set_yscale('log')
        
        # Constraint violations over time
        violations = np.array(self.history['constraint_violations'])
        if violations.shape[1] > 0:
            for i in range(violations.shape[1]):
                axes[1, 0].plot(violations[:, i], label=f'g_{i+1}')
            axes[1, 0].set_title('Individual Constraint Violations')
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('g_i(x)')
            axes[1, 0].legend()
        
        # Lagrange multipliers over time
        multipliers = np.array(self.history['lagrange_multipliers'])
        if multipliers.shape[1] > 0:
            for i in range(multipliers.shape[1]):
                axes[1, 1].plot(multipliers[:, i], label=f'lambda_{i+1}')
            axes[1, 1].set_title('Lagrange Multipliers')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('lambda_i')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def check_kkt_conditions(self, x, lam, nu, grad_f, grad_g, grad_h, tol=1e-6):
        """Check if KKT conditions are satisfied."""
        g_vals = grad_g(x) if grad_g is not None else np.array([])
        h_vals = grad_h(x) if grad_h is not None else np.array([])
        
        # Stationarity
        stationarity = grad_f(x)
        if g_vals.size > 0:
            stationarity += lam @ g_vals
        if h_vals.size > 0:
            stationarity += nu @ h_vals
        stat_violation = np.linalg.norm(stationarity)
        
        # Primal feasibility
        primal_violation = 0
        if g_vals.size > 0:
            primal_violation += np.sum(np.maximum(g_vals, 0)**2)
        if h_vals.size > 0:
            primal_violation += np.sum(h_vals**2)
        
        # Dual feasibility
        dual_violation = np.sum(np.maximum(-lam, 0)**2) if lam.size > 0 else 0
        
        # Complementary slackness
        comp_slack = np.sum(lam * g_vals**2) if (lam.size > 0 and g_vals.size > 0) else 0
        
        results = {
            'stationarity': stat_violation,
            'primal_feasibility': primal_violation,
            'dual_feasibility': dual_violation,
            'complementary_slackness': comp_slack,
            'all_satisfied': (stat_violation < tol and primal_violation < tol and
                            dual_violation < tol and comp_slack < tol)
        }
        
        return results
```

### F.7 Performance Comparison: Constrained Optimization Methods

```python
def benchmark_constrained_methods():
    """Compare different constrained optimization methods on a test problem."""
    import time
    from scipy.optimize import minimize, linprog
    
    # Test problem: minimize ||x||^2 subject to x >= 1 (element-wise)
    n = 1000
    x_true = np.ones(n)
    
    def f(x): return 0.5 * np.sum(x**2)
    def grad_f(x): return x
    
    methods = {
        'Projected GD': lambda: projected_gd_benchmark(f, grad_f, n),
        'Penalty Method': lambda: penalty_method_benchmark(f, grad_f, n),
        'scipy SLSQP': lambda: scipy_slsqp_benchmark(f, grad_f, n),
        'scipy trust-constr': lambda: scipy_trust_benchmark(f, grad_f, n),
    }
    
    results = {}
    for name, method in methods.items():
        start = time.time()
        x_opt, n_iter = method()
        elapsed = time.time() - start
        error = np.linalg.norm(x_opt - x_true)
        results[name] = {
            'time': elapsed,
            'iterations': n_iter,
            'error': error,
            'objective': f(x_opt)
        }
    
    print(f'{"Method":<25} | {"Time (s)":>10} | {"Iters":>6} | {"Error":>10} | {"Objective":>12}')
    print('-' * 70)
    for name, res in results.items():
        print(f'{name:<25} | {res["time"]:>10.4f} | {res["iterations"]:>6} | '
              f'{res["error"]:>10.2e} | {res["objective"]:>12.6f}')
    
    return results
```

### F.8 Common Pitfalls and How to Avoid Them

**1. Ill-conditioned projection:** When the feasible set has a very small or very large scale, the projection can be numerically unstable.

**Fix:** Scale the problem so that the feasible set has reasonable size (e.g., unit ball). Use high-precision arithmetic if necessary.

**2. Constraint qualification failure:** When LICQ or Slater's condition fails, the KKT conditions may not hold at the optimum, and algorithms may fail to converge.

**Fix:** Check constraint qualifications before solving. For convex problems, verify Slater's condition by finding a strictly feasible point. For non-convex problems, use regularization to ensure LICQ.

**3. Dual variable explosion:** When the problem is infeasible or the step size is too large, the dual variables can grow without bound.

**Fix:** Clip dual variables to a reasonable range. Use adaptive step sizes for dual updates. Check feasibility of the primal problem.

**4. Slow convergence of ADMM:** ADMM can converge slowly when the penalty parameter $\rho$ is poorly chosen.

**Fix:** Use adaptive $\rho$ update rules. Balance primal and dual residuals. Consider using over-relaxation ($\alpha \in (1, 2)$) to accelerate convergence.

**5. Numerical issues in interior-point methods:** The barrier function can cause numerical issues when iterates approach the boundary.

**Fix:** Use a reasonable starting point strictly inside the feasible set. Use backtracking line search to ensure strict feasibility. Regularize the Hessian if needed.

**6. Active set cycling:** In active set methods, the active set can cycle between iterations, leading to slow convergence.

**Fix:** Use a memory of past active sets to avoid cycling. Use a different strategy for adding/removing constraints from the active set.

**For AI practitioners:** The most common issue in constrained deep learning is the trade-off between constraint satisfaction and task performance. Monitor both metrics during training. If the constraint violation doesn't decrease, consider:
- Increasing the penalty parameter or Lagrange multiplier step size
- Using a different constraint handling method (e.g., switch from penalty to Lagrangian)
- Relaxing the constraint if it's too tight
- Using a warm start from a feasible point

