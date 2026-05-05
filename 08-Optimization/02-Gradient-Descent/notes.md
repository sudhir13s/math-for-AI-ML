[Previous: Convex Optimization](../01-Convex-Optimization/notes.md) | [Back to Chapter 8: Optimization](../README.md) | [Next: Second-Order Methods](../03-Second-Order-Methods/notes.md)

---

# Gradient Descent

> _"A gradient is local information; an optimizer is the discipline of spending it wisely."_

## Overview

Gradient Descent is part of the optimization spine of this curriculum. It explains how
mathematical assumptions become training behavior, and how training behavior becomes measurable
engineering evidence. The section is the canonical home for first-order deterministic
optimization, descent lemmas, step-size regimes, convergence rates, momentum, Nesterov
acceleration, and line-search theory.

The rewrite is deliberately AI-facing: every definition is connected to a loss, an update rule,
a notebook experiment, or a concrete model-training failure mode. Classical guarantees remain
important, but they are used as instruments for reasoning about neural networks, transformers,
large-batch runs, fine-tuning, and optimizer diagnostics.

A recurring principle runs through the entire chapter: do not memorize optimizer names. Instead,
identify the objective, the geometry, the stochasticity, the state carried by the method, and
the quantities that must be logged. That habit transfers from convex baselines to frontier-scale
LLM training.

## Prerequisites

- Gradients $\nabla f(\boldsymbol{\theta})$, Hessians $H_f(\boldsymbol{\theta})$, Jacobians $J_f$, and Taylor expansions from Chapter 5.
- Eigenvalues $\lambda_i$, positive definite matrices $A \succ 0$, matrix norms $\lVert A\rVert$, and condition numbers $\kappa(A)$ from Chapters 2-3.
- Expectation $\mathbb{E}[X]$, variance $\operatorname{Var}(X)$, concentration, and empirical risk from Chapters 6-7.
- Loss functions $\ell(\boldsymbol{\theta}; \mathbf{x}, y)$, cross-entropy, and negative log-likelihood from Statistics and Information Theory.
- Basic Python, NumPy arrays, and matplotlib plotting for the companion notebooks.
- The previous optimization section, [Convex Optimization](../01-Convex-Optimization/notes.md), is assumed as local context.

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Interactive derivations, numerical checks, and visual diagnostics for Gradient Descent. |
| [exercises.ipynb](exercises.ipynb) | Graded implementation and proof exercises for Gradient Descent. |

## Learning Objectives

- Define the canonical objects used in Gradient Descent with repository notation.
- Derive the main update rule and state the assumptions under which it is valid.
- Explain at least three examples and two non-examples for every major definition.
- Prove or sketch the core inequality that controls convergence or stability.
- Connect the theory to at least four modern AI or LLM training practices.
- Implement a minimal NumPy experiment that checks the mathematical claim numerically.
- Diagnose divergence, stagnation, overfitting, or instability using logged quantities.
- Identify which neighboring section owns related but non-canonical material.
- Translate formulas into practical framework-level implementation decisions.
- Explain why the topic still matters in a 2026 AI training stack.

## Notation and LaTeX Markdown Conventions

This section is written in LaTeX-in-Markdown style. Inline mathematical expressions are
delimited with single dollar signs, while central identities and updates are displayed in
double-dollar equation blocks. Vectors are bold lowercase, matrices are uppercase, sets and
spaces are calligraphic, and norms use $\lVert \cdot \rVert$ rather than bare vertical bars.

| Object | Convention | Example |
| --- | --- | --- |
| Parameter vector | bold lowercase | $\boldsymbol{\theta} \in \mathbb{R}^d$ |
| Data vector | bold lowercase | $\mathbf{x}^{(i)} \in \mathbb{R}^d$ |
| Objective | scalar function | $f : \mathbb{R}^d \to \mathbb{R}$ |
| Loss | calligraphic or script-style scalar | $\mathcal{L}(\boldsymbol{\theta})$ |
| Gradient | column vector | $\nabla f(\boldsymbol{\theta})$ |
| Hessian | matrix | $H_f(\boldsymbol{\theta}) = \nabla^2 f(\boldsymbol{\theta})$ |
| Learning rate | scalar schedule | $\eta_t > 0$ |
| Constraint set | calligraphic set | $\mathcal{C} \subseteq \mathbb{R}^d$ |

The canonical update for this section is:

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 Why Gradient Descent matters for training systems](#11-why-gradient-descent-matters-for-training-systems)
  - [1.2 The optimization object: parameters, objective, algorithm, and diagnostic](#12-the-optimization-object-parameters-objective-algorithm-and-diagnostic)
  - [1.3 Historical arc from classical optimization to modern AI](#13-historical-arc-from-classical-optimization-to-modern-ai)
  - [1.4 What this section treats as canonical scope](#14-what-this-section-treats-as-canonical-scope)
  - [1.5 A first mental model for LLM training](#15-a-first-mental-model-for-llm-training)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Primary definition: gradient direction](#21-primary-definition-gradient-direction)
  - [2.2 Secondary definition: descent lemma](#22-secondary-definition-descent-lemma)
  - [2.3 Algorithmic object: constant step size](#23-algorithmic-object-constant-step-size)
  - [2.4 Examples, non-examples, and boundary cases](#24-examples-nonexamples-and-boundary-cases)
  - [2.5 Notation, dimensions, and assumptions](#25-notation-dimensions-and-assumptions)
- [3. Core Theory I: Geometry and Guarantees](#3-core-theory-i-geometry-and-guarantees)
  - [3.1 Geometry of exact line search](#31-geometry-of-exact-line-search)
  - [3.2 Key inequality for backtracking line search](#32-key-inequality-for-backtracking-line-search)
  - [3.3 Role of Armijo condition](#33-role-of-armijo-condition)
  - [3.4 Proof template and what the proof actually buys](#34-proof-template-and-what-the-proof-actually-buys)
  - [3.5 Failure modes when assumptions are removed](#35-failure-modes-when-assumptions-are-removed)
- [4. Core Theory II: Algorithms and Dynamics](#4-core-theory-ii-algorithms-and-dynamics)
  - [4.1 Algorithmic update for Wolfe conditions](#41-algorithmic-update-for-wolfe-conditions)
  - [4.2 Stability role of convex convergence](#42-stability-role-of-convex-convergence)
  - [4.3 Rate or complexity controlled by strongly convex convergence](#43-rate-or-complexity-controlled-by-strongly-convex-convergence)
  - [4.4 Diagnostic interpretation of the update path](#44-diagnostic-interpretation-of-the-update-path)
  - [4.5 Connection to the next section in the chapter](#45-connection-to-the-next-section-in-the-chapter)
- [5. Core Theory III: Practical Variants](#5-core-theory-iii-practical-variants)
  - [5.1 Variant built around nonconvex stationarity](#51-variant-built-around-nonconvex-stationarity)
  - [5.2 Variant built around PL condition](#52-variant-built-around-pl-condition)
  - [5.3 Variant built around condition number](#53-variant-built-around-condition-number)
  - [5.4 Implementation constraints and numerical stability](#54-implementation-constraints-and-numerical-stability)
  - [5.5 What belongs here versus neighboring sections](#55-what-belongs-here-versus-neighboring-sections)
- [6. Advanced Topics](#6-advanced-topics)
  - [6.1 Advanced view of Polyak momentum](#61-advanced-view-of-polyak-momentum)
  - [6.2 Advanced view of Nesterov acceleration](#62-advanced-view-of-nesterov-acceleration)
  - [6.3 Advanced view of gradient flow](#63-advanced-view-of-gradient-flow)
  - [6.4 Infinite-dimensional or large-scale interpretation](#64-infinitedimensional-or-largescale-interpretation)
  - [6.5 Open questions for frontier model training](#65-open-questions-for-frontier-model-training)
- [7. Applications in Machine Learning](#7-applications-in-machine-learning)
  - [7.1 the basic training loop used by every neural-network optimizer](#71-the-basic-training-loop-used-by-every-neuralnetwork-optimizer)
  - [7.2 step-size stability for cross-entropy and mean-squared-error objectives](#72-stepsize-stability-for-crossentropy-and-meansquarederror-objectives)
  - [7.3 momentum as the ancestor of Adam's first-moment accumulator](#73-momentum-as-the-ancestor-of-adams-firstmoment-accumulator)
  - [7.4 line-search logic as a debugging model for divergence and oscillation](#74-linesearch-logic-as-a-debugging-model-for-divergence-and-oscillation)
  - [7.5 Diagnostic checklist for real experiments](#75-diagnostic-checklist-for-real-experiments)
- [8. Implementation and Diagnostics](#8-implementation-and-diagnostics)
  - [8.1 Minimal NumPy experiment for oscillation](#81-minimal-numpy-experiment-for-oscillation)
  - [8.2 Monitoring signal for edge of stability preview](#82-monitoring-signal-for-edge-of-stability-preview)
  - [8.3 Failure signature for gradient clipping preview](#83-failure-signature-for-gradient-clipping-preview)
  - [8.4 Framework-level implementation pattern](#84-frameworklevel-implementation-pattern)
  - [8.5 Reproducibility and logging checklist](#85-reproducibility-and-logging-checklist)
- [9. Common Mistakes](#9-common-mistakes)
- [10. Exercises](#10-exercises)
- [11. Why This Matters for AI (2026 Perspective)](#11-why-this-matters-for-ai-2026-perspective)
- [12. Conceptual Bridge](#12-conceptual-bridge)
- [Appendix A. Extended Derivation and Diagnostic Cards](#appendix-a-extended-derivation-and-diagnostic-cards)
- [References](#references)

---

## 1. Intuition

This block develops intuition for Gradient Descent. It keeps the scope local to this section
while pointing forward when a neighboring topic owns the full treatment.

### 1.1 Why Gradient Descent matters for training systems

In this section, backtracking line search is treated as a concrete optimization object rather
than a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Gradient Descent, the phrase "Why Gradient Descent matters for training systems"
means a precise mathematical habit: state the assumptions, write the update, identify what can
be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **backtracking line search** is the part of Gradient Descent that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where backtracking line search can be computed directly and compared with theory.
- A logistic-regression or softmax objective where backtracking line search affects optimization but the model remains interpretable.
- A transformer training diagnostic where backtracking line search appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating backtracking line search as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
backtracking line search, and use the section assumptions to bound the change in objective
value. If the assumption is geometric, the proof turns a picture into an inequality. If the
assumption is stochastic, the proof takes conditional expectation before applying the bound. If
the assumption is algorithmic, the proof checks that the proposed update is a descent,
projection, or preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes backtracking line search visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about backtracking line search is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- the basic training loop used by every neural-network optimizer.
- step-size stability for cross-entropy and mean-squared-error objectives.
- momentum as the ancestor of Adam's first-moment accumulator.
- line-search logic as a debugging model for divergence and oscillation.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 1.2 The optimization object: parameters, objective, algorithm, and diagnostic

In this section, Armijo condition is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Gradient Descent, the phrase "The optimization object: parameters, objective, algorithm, and
diagnostic" means a precise mathematical habit: state the assumptions, write the update,
identify what can be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **Armijo condition** is the part of Gradient Descent that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Armijo condition can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Armijo condition affects optimization but the model remains interpretable.
- A transformer training diagnostic where Armijo condition appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Armijo condition as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving Armijo
condition, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Armijo condition visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Armijo condition is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- the basic training loop used by every neural-network optimizer.
- step-size stability for cross-entropy and mean-squared-error objectives.
- momentum as the ancestor of Adam's first-moment accumulator.
- line-search logic as a debugging model for divergence and oscillation.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 1.3 Historical arc from classical optimization to modern AI

In this section, Wolfe conditions is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Gradient Descent, the phrase "Historical arc from classical optimization to modern AI" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **Wolfe conditions** is the part of Gradient Descent that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Wolfe conditions can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Wolfe conditions affects optimization but the model remains interpretable.
- A transformer training diagnostic where Wolfe conditions appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Wolfe conditions as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving Wolfe
conditions, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Wolfe conditions visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Wolfe conditions is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- the basic training loop used by every neural-network optimizer.
- step-size stability for cross-entropy and mean-squared-error objectives.
- momentum as the ancestor of Adam's first-moment accumulator.
- line-search logic as a debugging model for divergence and oscillation.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 1.4 What this section treats as canonical scope

In this section, convex convergence is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Gradient Descent, the phrase "What this section treats as canonical scope" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **convex convergence** is the part of Gradient Descent that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where convex convergence can be computed directly and compared with theory.
- A logistic-regression or softmax objective where convex convergence affects optimization but the model remains interpretable.
- A transformer training diagnostic where convex convergence appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating convex convergence as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving convex
convergence, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes convex convergence visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about convex convergence is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- the basic training loop used by every neural-network optimizer.
- step-size stability for cross-entropy and mean-squared-error objectives.
- momentum as the ancestor of Adam's first-moment accumulator.
- line-search logic as a debugging model for divergence and oscillation.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 1.5 A first mental model for LLM training

In this section, strongly convex convergence is treated as a concrete optimization object rather
than a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Gradient Descent, the phrase "A first mental model for LLM training" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **strongly convex convergence** is the part of Gradient Descent that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where strongly convex convergence can be computed directly and compared with theory.
- A logistic-regression or softmax objective where strongly convex convergence affects optimization but the model remains interpretable.
- A transformer training diagnostic where strongly convex convergence appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating strongly convex convergence as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving strongly
convex convergence, and use the section assumptions to bound the change in objective value. If
the assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes strongly convex convergence visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about strongly convex convergence is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- the basic training loop used by every neural-network optimizer.
- step-size stability for cross-entropy and mean-squared-error objectives.
- momentum as the ancestor of Adam's first-moment accumulator.
- line-search logic as a debugging model for divergence and oscillation.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 2. Formal Definitions

This block develops formal definitions for Gradient Descent. It keeps the scope local to this
section while pointing forward when a neighboring topic owns the full treatment.

### 2.1 Primary definition: gradient direction

In this section, convex convergence is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Gradient Descent, the phrase "Primary definition: gradient direction" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **convex convergence** is the part of Gradient Descent that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where convex convergence can be computed directly and compared with theory.
- A logistic-regression or softmax objective where convex convergence affects optimization but the model remains interpretable.
- A transformer training diagnostic where convex convergence appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating convex convergence as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving convex
convergence, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes convex convergence visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about convex convergence is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- the basic training loop used by every neural-network optimizer.
- step-size stability for cross-entropy and mean-squared-error objectives.
- momentum as the ancestor of Adam's first-moment accumulator.
- line-search logic as a debugging model for divergence and oscillation.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 2.2 Secondary definition: descent lemma

In this section, strongly convex convergence is treated as a concrete optimization object rather
than a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Gradient Descent, the phrase "Secondary definition: descent lemma" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **strongly convex convergence** is the part of Gradient Descent that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where strongly convex convergence can be computed directly and compared with theory.
- A logistic-regression or softmax objective where strongly convex convergence affects optimization but the model remains interpretable.
- A transformer training diagnostic where strongly convex convergence appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating strongly convex convergence as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving strongly
convex convergence, and use the section assumptions to bound the change in objective value. If
the assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes strongly convex convergence visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about strongly convex convergence is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- the basic training loop used by every neural-network optimizer.
- step-size stability for cross-entropy and mean-squared-error objectives.
- momentum as the ancestor of Adam's first-moment accumulator.
- line-search logic as a debugging model for divergence and oscillation.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 2.3 Algorithmic object: constant step size

In this section, nonconvex stationarity is treated as a concrete optimization object rather than
a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Gradient Descent, the phrase "Algorithmic object: constant step size" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **nonconvex stationarity** is the part of Gradient Descent that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where nonconvex stationarity can be computed directly and compared with theory.
- A logistic-regression or softmax objective where nonconvex stationarity affects optimization but the model remains interpretable.
- A transformer training diagnostic where nonconvex stationarity appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating nonconvex stationarity as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving nonconvex
stationarity, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes nonconvex stationarity visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about nonconvex stationarity is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- the basic training loop used by every neural-network optimizer.
- step-size stability for cross-entropy and mean-squared-error objectives.
- momentum as the ancestor of Adam's first-moment accumulator.
- line-search logic as a debugging model for divergence and oscillation.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 2.4 Examples, non-examples, and boundary cases

In this section, PL condition is treated as a concrete optimization object rather than a slogan.
The goal is to understand how it changes the objective, the update rule, the convergence story,
and the diagnostics a practitioner should inspect when training a modern model. For Gradient
Descent, the phrase "Examples, non-examples, and boundary cases" means a precise mathematical
habit: state the assumptions, write the update, identify what can be measured, and connect the
result to a real AI training decision.

> **Definition.**
>
> For this section, **PL condition** is the part of Gradient Descent that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where PL condition can be computed directly and compared with theory.
- A logistic-regression or softmax objective where PL condition affects optimization but the model remains interpretable.
- A transformer training diagnostic where PL condition appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating PL condition as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving PL
condition, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes PL condition visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about PL condition is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- the basic training loop used by every neural-network optimizer.
- step-size stability for cross-entropy and mean-squared-error objectives.
- momentum as the ancestor of Adam's first-moment accumulator.
- line-search logic as a debugging model for divergence and oscillation.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 2.5 Notation, dimensions, and assumptions

In this section, condition number is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Gradient Descent, the phrase "Notation, dimensions, and assumptions" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **condition number** is the part of Gradient Descent that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where condition number can be computed directly and compared with theory.
- A logistic-regression or softmax objective where condition number affects optimization but the model remains interpretable.
- A transformer training diagnostic where condition number appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating condition number as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving condition
number, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes condition number visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about condition number is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- the basic training loop used by every neural-network optimizer.
- step-size stability for cross-entropy and mean-squared-error objectives.
- momentum as the ancestor of Adam's first-moment accumulator.
- line-search logic as a debugging model for divergence and oscillation.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 3. Core Theory I: Geometry and Guarantees

This block develops core theory i: geometry and guarantees for Gradient Descent. It keeps the
scope local to this section while pointing forward when a neighboring topic owns the full
treatment.

### 3.1 Geometry of exact line search

In this section, PL condition is treated as a concrete optimization object rather than a slogan.
The goal is to understand how it changes the objective, the update rule, the convergence story,
and the diagnostics a practitioner should inspect when training a modern model. For Gradient
Descent, the phrase "Geometry of exact line search" means a precise mathematical habit: state
the assumptions, write the update, identify what can be measured, and connect the result to a
real AI training decision.

> **Definition.**
>
> For this section, **PL condition** is the part of Gradient Descent that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where PL condition can be computed directly and compared with theory.
- A logistic-regression or softmax objective where PL condition affects optimization but the model remains interpretable.
- A transformer training diagnostic where PL condition appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating PL condition as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving PL
condition, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes PL condition visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about PL condition is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- the basic training loop used by every neural-network optimizer.
- step-size stability for cross-entropy and mean-squared-error objectives.
- momentum as the ancestor of Adam's first-moment accumulator.
- line-search logic as a debugging model for divergence and oscillation.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 3.2 Key inequality for backtracking line search

In this section, condition number is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Gradient Descent, the phrase "Key inequality for backtracking line search" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **condition number** is the part of Gradient Descent that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where condition number can be computed directly and compared with theory.
- A logistic-regression or softmax objective where condition number affects optimization but the model remains interpretable.
- A transformer training diagnostic where condition number appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating condition number as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving condition
number, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes condition number visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about condition number is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- the basic training loop used by every neural-network optimizer.
- step-size stability for cross-entropy and mean-squared-error objectives.
- momentum as the ancestor of Adam's first-moment accumulator.
- line-search logic as a debugging model for divergence and oscillation.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 3.3 Role of Armijo condition

In this section, Polyak momentum is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Gradient Descent, the phrase "Role of Armijo condition" means a precise mathematical habit:
state the assumptions, write the update, identify what can be measured, and connect the result
to a real AI training decision.

> **Definition.**
>
> For this section, **Polyak momentum** is the part of Gradient Descent that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Polyak momentum can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Polyak momentum affects optimization but the model remains interpretable.
- A transformer training diagnostic where Polyak momentum appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Polyak momentum as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving Polyak
momentum, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Polyak momentum visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Polyak momentum is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- the basic training loop used by every neural-network optimizer.
- step-size stability for cross-entropy and mean-squared-error objectives.
- momentum as the ancestor of Adam's first-moment accumulator.
- line-search logic as a debugging model for divergence and oscillation.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 3.4 Proof template and what the proof actually buys

In this section, Nesterov acceleration is treated as a concrete optimization object rather than
a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Gradient Descent, the phrase "Proof template and what the proof actually buys" means
a precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **Nesterov acceleration** is the part of Gradient Descent that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Nesterov acceleration can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Nesterov acceleration affects optimization but the model remains interpretable.
- A transformer training diagnostic where Nesterov acceleration appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Nesterov acceleration as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving Nesterov
acceleration, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Nesterov acceleration visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Nesterov acceleration is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- the basic training loop used by every neural-network optimizer.
- step-size stability for cross-entropy and mean-squared-error objectives.
- momentum as the ancestor of Adam's first-moment accumulator.
- line-search logic as a debugging model for divergence and oscillation.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 3.5 Failure modes when assumptions are removed

In this section, gradient flow is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Gradient Descent, the phrase "Failure modes when assumptions are removed" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **gradient flow** is the part of Gradient Descent that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where gradient flow can be computed directly and compared with theory.
- A logistic-regression or softmax objective where gradient flow affects optimization but the model remains interpretable.
- A transformer training diagnostic where gradient flow appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating gradient flow as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving gradient
flow, and use the section assumptions to bound the change in objective value. If the assumption
is geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes gradient flow visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about gradient flow is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- the basic training loop used by every neural-network optimizer.
- step-size stability for cross-entropy and mean-squared-error objectives.
- momentum as the ancestor of Adam's first-moment accumulator.
- line-search logic as a debugging model for divergence and oscillation.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 4. Core Theory II: Algorithms and Dynamics

This block develops core theory ii: algorithms and dynamics for Gradient Descent. It keeps the
scope local to this section while pointing forward when a neighboring topic owns the full
treatment.

### 4.1 Algorithmic update for Wolfe conditions

In this section, Nesterov acceleration is treated as a concrete optimization object rather than
a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Gradient Descent, the phrase "Algorithmic update for Wolfe conditions" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **Nesterov acceleration** is the part of Gradient Descent that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Nesterov acceleration can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Nesterov acceleration affects optimization but the model remains interpretable.
- A transformer training diagnostic where Nesterov acceleration appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Nesterov acceleration as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving Nesterov
acceleration, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Nesterov acceleration visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Nesterov acceleration is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- the basic training loop used by every neural-network optimizer.
- step-size stability for cross-entropy and mean-squared-error objectives.
- momentum as the ancestor of Adam's first-moment accumulator.
- line-search logic as a debugging model for divergence and oscillation.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 4.2 Stability role of convex convergence

In this section, gradient flow is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Gradient Descent, the phrase "Stability role of convex convergence" means a precise mathematical
habit: state the assumptions, write the update, identify what can be measured, and connect the
result to a real AI training decision.

> **Definition.**
>
> For this section, **gradient flow** is the part of Gradient Descent that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where gradient flow can be computed directly and compared with theory.
- A logistic-regression or softmax objective where gradient flow affects optimization but the model remains interpretable.
- A transformer training diagnostic where gradient flow appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating gradient flow as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving gradient
flow, and use the section assumptions to bound the change in objective value. If the assumption
is geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes gradient flow visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about gradient flow is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- the basic training loop used by every neural-network optimizer.
- step-size stability for cross-entropy and mean-squared-error objectives.
- momentum as the ancestor of Adam's first-moment accumulator.
- line-search logic as a debugging model for divergence and oscillation.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 4.3 Rate or complexity controlled by strongly convex convergence

In this section, oscillation is treated as a concrete optimization object rather than a slogan.
The goal is to understand how it changes the objective, the update rule, the convergence story,
and the diagnostics a practitioner should inspect when training a modern model. For Gradient
Descent, the phrase "Rate or complexity controlled by strongly convex convergence" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **oscillation** is the part of Gradient Descent that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where oscillation can be computed directly and compared with theory.
- A logistic-regression or softmax objective where oscillation affects optimization but the model remains interpretable.
- A transformer training diagnostic where oscillation appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating oscillation as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
oscillation, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes oscillation visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about oscillation is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- the basic training loop used by every neural-network optimizer.
- step-size stability for cross-entropy and mean-squared-error objectives.
- momentum as the ancestor of Adam's first-moment accumulator.
- line-search logic as a debugging model for divergence and oscillation.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 4.4 Diagnostic interpretation of the update path

In this section, edge of stability preview is treated as a concrete optimization object rather
than a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Gradient Descent, the phrase "Diagnostic interpretation of the update path" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **edge of stability preview** is the part of Gradient Descent that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where edge of stability preview can be computed directly and compared with theory.
- A logistic-regression or softmax objective where edge of stability preview affects optimization but the model remains interpretable.
- A transformer training diagnostic where edge of stability preview appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating edge of stability preview as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving edge of
stability preview, and use the section assumptions to bound the change in objective value. If
the assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes edge of stability preview visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about edge of stability preview is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- the basic training loop used by every neural-network optimizer.
- step-size stability for cross-entropy and mean-squared-error objectives.
- momentum as the ancestor of Adam's first-moment accumulator.
- line-search logic as a debugging model for divergence and oscillation.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 4.5 Connection to the next section in the chapter

In this section, gradient clipping preview is treated as a concrete optimization object rather
than a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Gradient Descent, the phrase "Connection to the next section in the chapter" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **gradient clipping preview** is the part of Gradient Descent that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where gradient clipping preview can be computed directly and compared with theory.
- A logistic-regression or softmax objective where gradient clipping preview affects optimization but the model remains interpretable.
- A transformer training diagnostic where gradient clipping preview appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating gradient clipping preview as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving gradient
clipping preview, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes gradient clipping preview visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about gradient clipping preview is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- the basic training loop used by every neural-network optimizer.
- step-size stability for cross-entropy and mean-squared-error objectives.
- momentum as the ancestor of Adam's first-moment accumulator.
- line-search logic as a debugging model for divergence and oscillation.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 5. Core Theory III: Practical Variants

This block develops core theory iii: practical variants for Gradient Descent. It keeps the scope
local to this section while pointing forward when a neighboring topic owns the full treatment.

### 5.1 Variant built around nonconvex stationarity

In this section, edge of stability preview is treated as a concrete optimization object rather
than a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Gradient Descent, the phrase "Variant built around nonconvex stationarity" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **edge of stability preview** is the part of Gradient Descent that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where edge of stability preview can be computed directly and compared with theory.
- A logistic-regression or softmax objective where edge of stability preview affects optimization but the model remains interpretable.
- A transformer training diagnostic where edge of stability preview appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating edge of stability preview as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving edge of
stability preview, and use the section assumptions to bound the change in objective value. If
the assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes edge of stability preview visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about edge of stability preview is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- the basic training loop used by every neural-network optimizer.
- step-size stability for cross-entropy and mean-squared-error objectives.
- momentum as the ancestor of Adam's first-moment accumulator.
- line-search logic as a debugging model for divergence and oscillation.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 5.2 Variant built around PL condition

In this section, gradient clipping preview is treated as a concrete optimization object rather
than a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Gradient Descent, the phrase "Variant built around PL condition" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **gradient clipping preview** is the part of Gradient Descent that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where gradient clipping preview can be computed directly and compared with theory.
- A logistic-regression or softmax objective where gradient clipping preview affects optimization but the model remains interpretable.
- A transformer training diagnostic where gradient clipping preview appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating gradient clipping preview as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving gradient
clipping preview, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes gradient clipping preview visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about gradient clipping preview is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- the basic training loop used by every neural-network optimizer.
- step-size stability for cross-entropy and mean-squared-error objectives.
- momentum as the ancestor of Adam's first-moment accumulator.
- line-search logic as a debugging model for divergence and oscillation.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 5.3 Variant built around condition number

In this section, linear regression by GD is treated as a concrete optimization object rather
than a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Gradient Descent, the phrase "Variant built around condition number" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **linear regression by GD** is the part of Gradient Descent that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where linear regression by GD can be computed directly and compared with theory.
- A logistic-regression or softmax objective where linear regression by GD affects optimization but the model remains interpretable.
- A transformer training diagnostic where linear regression by GD appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating linear regression by GD as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving linear
regression by GD, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes linear regression by GD visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about linear regression by GD is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- the basic training loop used by every neural-network optimizer.
- step-size stability for cross-entropy and mean-squared-error objectives.
- momentum as the ancestor of Adam's first-moment accumulator.
- line-search logic as a debugging model for divergence and oscillation.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 5.4 Implementation constraints and numerical stability

In this section, logistic regression by GD is treated as a concrete optimization object rather
than a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Gradient Descent, the phrase "Implementation constraints and numerical stability"
means a precise mathematical habit: state the assumptions, write the update, identify what can
be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **logistic regression by GD** is the part of Gradient Descent that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where logistic regression by GD can be computed directly and compared with theory.
- A logistic-regression or softmax objective where logistic regression by GD affects optimization but the model remains interpretable.
- A transformer training diagnostic where logistic regression by GD appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating logistic regression by GD as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving logistic
regression by GD, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes logistic regression by GD visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about logistic regression by GD is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- the basic training loop used by every neural-network optimizer.
- step-size stability for cross-entropy and mean-squared-error objectives.
- momentum as the ancestor of Adam's first-moment accumulator.
- line-search logic as a debugging model for divergence and oscillation.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 5.5 What belongs here versus neighboring sections

In this section, learning-rate diagnostics is treated as a concrete optimization object rather
than a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Gradient Descent, the phrase "What belongs here versus neighboring sections" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **learning-rate diagnostics** is the part of Gradient Descent that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where learning-rate diagnostics can be computed directly and compared with theory.
- A logistic-regression or softmax objective where learning-rate diagnostics affects optimization but the model remains interpretable.
- A transformer training diagnostic where learning-rate diagnostics appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating learning-rate diagnostics as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
learning-rate diagnostics, and use the section assumptions to bound the change in objective
value. If the assumption is geometric, the proof turns a picture into an inequality. If the
assumption is stochastic, the proof takes conditional expectation before applying the bound. If
the assumption is algorithmic, the proof checks that the proposed update is a descent,
projection, or preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes learning-rate diagnostics visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about learning-rate diagnostics is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- the basic training loop used by every neural-network optimizer.
- step-size stability for cross-entropy and mean-squared-error objectives.
- momentum as the ancestor of Adam's first-moment accumulator.
- line-search logic as a debugging model for divergence and oscillation.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 6. Advanced Topics

This block develops advanced topics for Gradient Descent. It keeps the scope local to this
section while pointing forward when a neighboring topic owns the full treatment.

### 6.1 Advanced view of Polyak momentum

In this section, logistic regression by GD is treated as a concrete optimization object rather
than a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Gradient Descent, the phrase "Advanced view of Polyak momentum" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **logistic regression by GD** is the part of Gradient Descent that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where logistic regression by GD can be computed directly and compared with theory.
- A logistic-regression or softmax objective where logistic regression by GD affects optimization but the model remains interpretable.
- A transformer training diagnostic where logistic regression by GD appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating logistic regression by GD as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving logistic
regression by GD, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes logistic regression by GD visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about logistic regression by GD is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- the basic training loop used by every neural-network optimizer.
- step-size stability for cross-entropy and mean-squared-error objectives.
- momentum as the ancestor of Adam's first-moment accumulator.
- line-search logic as a debugging model for divergence and oscillation.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 6.2 Advanced view of Nesterov acceleration

In this section, learning-rate diagnostics is treated as a concrete optimization object rather
than a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Gradient Descent, the phrase "Advanced view of Nesterov acceleration" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **learning-rate diagnostics** is the part of Gradient Descent that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where learning-rate diagnostics can be computed directly and compared with theory.
- A logistic-regression or softmax objective where learning-rate diagnostics affects optimization but the model remains interpretable.
- A transformer training diagnostic where learning-rate diagnostics appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating learning-rate diagnostics as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
learning-rate diagnostics, and use the section assumptions to bound the change in objective
value. If the assumption is geometric, the proof turns a picture into an inequality. If the
assumption is stochastic, the proof takes conditional expectation before applying the bound. If
the assumption is algorithmic, the proof checks that the proposed update is a descent,
projection, or preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes learning-rate diagnostics visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about learning-rate diagnostics is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- the basic training loop used by every neural-network optimizer.
- step-size stability for cross-entropy and mean-squared-error objectives.
- momentum as the ancestor of Adam's first-moment accumulator.
- line-search logic as a debugging model for divergence and oscillation.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 6.3 Advanced view of gradient flow

In this section, optimization loop design is treated as a concrete optimization object rather
than a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Gradient Descent, the phrase "Advanced view of gradient flow" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **optimization loop design** is the part of Gradient Descent that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where optimization loop design can be computed directly and compared with theory.
- A logistic-regression or softmax objective where optimization loop design affects optimization but the model remains interpretable.
- A transformer training diagnostic where optimization loop design appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating optimization loop design as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
optimization loop design, and use the section assumptions to bound the change in objective
value. If the assumption is geometric, the proof turns a picture into an inequality. If the
assumption is stochastic, the proof takes conditional expectation before applying the bound. If
the assumption is algorithmic, the proof checks that the proposed update is a descent,
projection, or preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes optimization loop design visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about optimization loop design is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- the basic training loop used by every neural-network optimizer.
- step-size stability for cross-entropy and mean-squared-error objectives.
- momentum as the ancestor of Adam's first-moment accumulator.
- line-search logic as a debugging model for divergence and oscillation.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 6.4 Infinite-dimensional or large-scale interpretation

In this section, gradient direction is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Gradient Descent, the phrase "Infinite-dimensional or large-scale interpretation" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **gradient direction** is the part of Gradient Descent that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where gradient direction can be computed directly and compared with theory.
- A logistic-regression or softmax objective where gradient direction affects optimization but the model remains interpretable.
- A transformer training diagnostic where gradient direction appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating gradient direction as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving gradient
direction, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes gradient direction visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about gradient direction is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- the basic training loop used by every neural-network optimizer.
- step-size stability for cross-entropy and mean-squared-error objectives.
- momentum as the ancestor of Adam's first-moment accumulator.
- line-search logic as a debugging model for divergence and oscillation.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 6.5 Open questions for frontier model training

In this section, descent lemma is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Gradient Descent, the phrase "Open questions for frontier model training" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **descent lemma** is the part of Gradient Descent that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where descent lemma can be computed directly and compared with theory.
- A logistic-regression or softmax objective where descent lemma affects optimization but the model remains interpretable.
- A transformer training diagnostic where descent lemma appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating descent lemma as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving descent
lemma, and use the section assumptions to bound the change in objective value. If the assumption
is geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes descent lemma visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about descent lemma is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- the basic training loop used by every neural-network optimizer.
- step-size stability for cross-entropy and mean-squared-error objectives.
- momentum as the ancestor of Adam's first-moment accumulator.
- line-search logic as a debugging model for divergence and oscillation.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 7. Applications in Machine Learning

This block develops applications in machine learning for Gradient Descent. It keeps the scope
local to this section while pointing forward when a neighboring topic owns the full treatment.

### 7.1 the basic training loop used by every neural-network optimizer

In this section, gradient direction is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Gradient Descent, the phrase "the basic training loop used by every neural-network optimizer"
means a precise mathematical habit: state the assumptions, write the update, identify what can
be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **gradient direction** is the part of Gradient Descent that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where gradient direction can be computed directly and compared with theory.
- A logistic-regression or softmax objective where gradient direction affects optimization but the model remains interpretable.
- A transformer training diagnostic where gradient direction appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating gradient direction as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving gradient
direction, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes gradient direction visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about gradient direction is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- the basic training loop used by every neural-network optimizer.
- step-size stability for cross-entropy and mean-squared-error objectives.
- momentum as the ancestor of Adam's first-moment accumulator.
- line-search logic as a debugging model for divergence and oscillation.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 7.2 step-size stability for cross-entropy and mean-squared-error objectives

In this section, descent lemma is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Gradient Descent, the phrase "step-size stability for cross-entropy and mean-squared-error
objectives" means a precise mathematical habit: state the assumptions, write the update,
identify what can be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **descent lemma** is the part of Gradient Descent that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where descent lemma can be computed directly and compared with theory.
- A logistic-regression or softmax objective where descent lemma affects optimization but the model remains interpretable.
- A transformer training diagnostic where descent lemma appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating descent lemma as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving descent
lemma, and use the section assumptions to bound the change in objective value. If the assumption
is geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes descent lemma visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about descent lemma is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- the basic training loop used by every neural-network optimizer.
- step-size stability for cross-entropy and mean-squared-error objectives.
- momentum as the ancestor of Adam's first-moment accumulator.
- line-search logic as a debugging model for divergence and oscillation.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 7.3 momentum as the ancestor of Adam's first-moment accumulator

In this section, constant step size is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Gradient Descent, the phrase "momentum as the ancestor of Adam's first-moment accumulator" means
a precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **constant step size** is the part of Gradient Descent that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where constant step size can be computed directly and compared with theory.
- A logistic-regression or softmax objective where constant step size affects optimization but the model remains interpretable.
- A transformer training diagnostic where constant step size appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating constant step size as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving constant
step size, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes constant step size visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about constant step size is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- the basic training loop used by every neural-network optimizer.
- step-size stability for cross-entropy and mean-squared-error objectives.
- momentum as the ancestor of Adam's first-moment accumulator.
- line-search logic as a debugging model for divergence and oscillation.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 7.4 line-search logic as a debugging model for divergence and oscillation

In this section, exact line search is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Gradient Descent, the phrase "line-search logic as a debugging model for divergence and
oscillation" means a precise mathematical habit: state the assumptions, write the update,
identify what can be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **exact line search** is the part of Gradient Descent that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where exact line search can be computed directly and compared with theory.
- A logistic-regression or softmax objective where exact line search affects optimization but the model remains interpretable.
- A transformer training diagnostic where exact line search appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating exact line search as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving exact line
search, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes exact line search visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about exact line search is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- the basic training loop used by every neural-network optimizer.
- step-size stability for cross-entropy and mean-squared-error objectives.
- momentum as the ancestor of Adam's first-moment accumulator.
- line-search logic as a debugging model for divergence and oscillation.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 7.5 Diagnostic checklist for real experiments

In this section, backtracking line search is treated as a concrete optimization object rather
than a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Gradient Descent, the phrase "Diagnostic checklist for real experiments" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **backtracking line search** is the part of Gradient Descent that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where backtracking line search can be computed directly and compared with theory.
- A logistic-regression or softmax objective where backtracking line search affects optimization but the model remains interpretable.
- A transformer training diagnostic where backtracking line search appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating backtracking line search as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
backtracking line search, and use the section assumptions to bound the change in objective
value. If the assumption is geometric, the proof turns a picture into an inequality. If the
assumption is stochastic, the proof takes conditional expectation before applying the bound. If
the assumption is algorithmic, the proof checks that the proposed update is a descent,
projection, or preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes backtracking line search visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about backtracking line search is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- the basic training loop used by every neural-network optimizer.
- step-size stability for cross-entropy and mean-squared-error objectives.
- momentum as the ancestor of Adam's first-moment accumulator.
- line-search logic as a debugging model for divergence and oscillation.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 8. Implementation and Diagnostics

This block develops implementation and diagnostics for Gradient Descent. It keeps the scope
local to this section while pointing forward when a neighboring topic owns the full treatment.

### 8.1 Minimal NumPy experiment for oscillation

In this section, exact line search is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Gradient Descent, the phrase "Minimal NumPy experiment for oscillation" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **exact line search** is the part of Gradient Descent that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where exact line search can be computed directly and compared with theory.
- A logistic-regression or softmax objective where exact line search affects optimization but the model remains interpretable.
- A transformer training diagnostic where exact line search appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating exact line search as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving exact line
search, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes exact line search visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about exact line search is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- the basic training loop used by every neural-network optimizer.
- step-size stability for cross-entropy and mean-squared-error objectives.
- momentum as the ancestor of Adam's first-moment accumulator.
- line-search logic as a debugging model for divergence and oscillation.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 8.2 Monitoring signal for edge of stability preview

In this section, backtracking line search is treated as a concrete optimization object rather
than a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Gradient Descent, the phrase "Monitoring signal for edge of stability preview" means
a precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **backtracking line search** is the part of Gradient Descent that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where backtracking line search can be computed directly and compared with theory.
- A logistic-regression or softmax objective where backtracking line search affects optimization but the model remains interpretable.
- A transformer training diagnostic where backtracking line search appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating backtracking line search as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
backtracking line search, and use the section assumptions to bound the change in objective
value. If the assumption is geometric, the proof turns a picture into an inequality. If the
assumption is stochastic, the proof takes conditional expectation before applying the bound. If
the assumption is algorithmic, the proof checks that the proposed update is a descent,
projection, or preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes backtracking line search visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about backtracking line search is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- the basic training loop used by every neural-network optimizer.
- step-size stability for cross-entropy and mean-squared-error objectives.
- momentum as the ancestor of Adam's first-moment accumulator.
- line-search logic as a debugging model for divergence and oscillation.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 8.3 Failure signature for gradient clipping preview

In this section, Armijo condition is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Gradient Descent, the phrase "Failure signature for gradient clipping preview" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **Armijo condition** is the part of Gradient Descent that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Armijo condition can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Armijo condition affects optimization but the model remains interpretable.
- A transformer training diagnostic where Armijo condition appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Armijo condition as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving Armijo
condition, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Armijo condition visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Armijo condition is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- the basic training loop used by every neural-network optimizer.
- step-size stability for cross-entropy and mean-squared-error objectives.
- momentum as the ancestor of Adam's first-moment accumulator.
- line-search logic as a debugging model for divergence and oscillation.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 8.4 Framework-level implementation pattern

In this section, Wolfe conditions is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Gradient Descent, the phrase "Framework-level implementation pattern" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **Wolfe conditions** is the part of Gradient Descent that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Wolfe conditions can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Wolfe conditions affects optimization but the model remains interpretable.
- A transformer training diagnostic where Wolfe conditions appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Wolfe conditions as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving Wolfe
conditions, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Wolfe conditions visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Wolfe conditions is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- the basic training loop used by every neural-network optimizer.
- step-size stability for cross-entropy and mean-squared-error objectives.
- momentum as the ancestor of Adam's first-moment accumulator.
- line-search logic as a debugging model for divergence and oscillation.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 8.5 Reproducibility and logging checklist

In this section, convex convergence is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Gradient Descent, the phrase "Reproducibility and logging checklist" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **convex convergence** is the part of Gradient Descent that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where convex convergence can be computed directly and compared with theory.
- A logistic-regression or softmax objective where convex convergence affects optimization but the model remains interpretable.
- A transformer training diagnostic where convex convergence appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating convex convergence as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving convex
convergence, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes convex convergence visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about convex convergence is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- the basic training loop used by every neural-network optimizer.
- step-size stability for cross-entropy and mean-squared-error objectives.
- momentum as the ancestor of Adam's first-moment accumulator.
- line-search logic as a debugging model for divergence and oscillation.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 9. Common Mistakes

| # | Mistake | Why It Is Wrong | Fix |
| --- | --- | --- | --- |
| 1 | Using a recipe without checking assumptions | Optimization guarantees depend on smoothness, convexity, stochasticity, or feasibility assumptions. | Write the assumptions next to the update rule before choosing hyperparameters. |
| 2 | Confusing objective decrease with validation improvement | The optimizer sees the training objective; validation behavior also depends on generalization and data split quality. | Track objective, train metric, validation metric, and update norm separately. |
| 3 | Treating all norms as interchangeable | The geometry changes when the norm changes, especially for constraints and regularizers. | State whether you use $\ell_1$, $\ell_2$, Frobenius, spectral, or another norm. |
| 4 | Ignoring scale | Learning rates, penalties, curvature, and gradient norms are all scale-sensitive. | Normalize units and inspect effective update size $\lVert \Delta\boldsymbol{\theta}\rVert_2 / \lVert\boldsymbol{\theta}\rVert_2$. |
| 5 | Overfitting to a single seed | Optimization can look stable for one seed and fail under another. | Run small seed sweeps for important claims. |
| 6 | Hiding instability behind smoothed plots | A moving average can hide spikes, divergence, and bad curvature events. | Plot raw metrics alongside smoothed metrics. |
| 7 | Using test data during tuning | This contaminates the final evaluation. | Reserve test data until after model and hyperparameter selection. |
| 8 | Assuming large models make theory irrelevant | Large models often make diagnostics more important because failures are expensive. | Use theory to decide what to log, not to pretend every theorem applies exactly. |
| 9 | Mixing optimizer state with model state carelessly | State corruption changes the effective algorithm. | Checkpoint parameters, gradients if needed, optimizer moments, scheduler state, and random seeds. |
| 10 | Not checking numerical precision | BF16, FP16, FP8, and accumulation choices can change the observed optimizer. | Cross-check suspicious runs against higher precision on a small batch. |

## 10. Exercises

1. **Exercise 1 [*] - Constant Step Size**
   (a) Define constant step size using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

2. **Exercise 2 [*] - Backtracking Line Search**
   (a) Define backtracking line search using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

3. **Exercise 3 [*] - Wolfe Conditions**
   (a) Define Wolfe conditions using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

4. **Exercise 4 [**] - Strongly Convex Convergence**
   (a) Define strongly convex convergence using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

5. **Exercise 5 [**] - Pl Condition**
   (a) Define PL condition using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

6. **Exercise 6 [**] - Polyak Momentum**
   (a) Define Polyak momentum using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

7. **Exercise 7 [**] - Gradient Flow**
   (a) Define gradient flow using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

8. **Exercise 8 [***] - Edge Of Stability Preview**
   (a) Define edge of stability preview using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

9. **Exercise 9 [***] - Linear Regression By Gd**
   (a) Define linear regression by GD using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

10. **Exercise 10 [***] - Learning-Rate Diagnostics**
   (a) Define learning-rate diagnostics using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
f(\boldsymbol{\theta}_{t+1}) \leq f(\boldsymbol{\theta}_t) - \frac{\eta}{2}\lVert \nabla f(\boldsymbol{\theta}_t)\rVert_2^2
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

## 11. Why This Matters for AI (2026 Perspective)

| Concept | AI Impact |
| --- | --- |
| gradient direction | the basic training loop used by every neural-network optimizer |
| descent lemma | step-size stability for cross-entropy and mean-squared-error objectives |
| constant step size | momentum as the ancestor of Adam's first-moment accumulator |
| exact line search | line-search logic as a debugging model for divergence and oscillation |
| backtracking line search | the basic training loop used by every neural-network optimizer |
| Armijo condition | step-size stability for cross-entropy and mean-squared-error objectives |
| Wolfe conditions | momentum as the ancestor of Adam's first-moment accumulator |
| convex convergence | line-search logic as a debugging model for divergence and oscillation |
| strongly convex convergence | the basic training loop used by every neural-network optimizer |
| nonconvex stationarity | step-size stability for cross-entropy and mean-squared-error objectives |

## 12. Conceptual Bridge

Gradient Descent sits inside a chain. Earlier sections give the calculus, probability, and
linear algebra needed to write the objective and interpret the update. Later sections use this
material to reason about noisy gradients, adaptive state, regularization, tuning, schedules, and
finally information-theoretic losses.

Backward link: [Convex Optimization](../01-Convex-Optimization/notes.md) supplies the immediate
prerequisite vocabulary.

Forward link: [Second-Order Methods](../03-Second-Order-Methods/notes.md) uses this section as a
building block.

```text
+------------------------------------------------------------+
| Chapter 8: Optimization                                    |
|    01-Convex-Optimization          Convex Optimization    |
| >> 02-Gradient-Descent             Gradient Descent       |
|    03-Second-Order-Methods         Second-Order Methods   |
|    04-Constrained-Optimization     Constrained Optimization |
|    05-Stochastic-Optimization      Stochastic Optimization |
|    06-Optimization-Landscape       Optimization Landscape |
|    07-Adaptive-Learning-Rate       Adaptive Learning Rate |
|    08-Regularization-Methods       Regularization Methods |
|    09-Hyperparameter-Optimization  Hyperparameter Optimization |
|    10-Learning-Rate-Schedules      Learning Rate Schedules |
+------------------------------------------------------------+
```

## Appendix A. Extended Derivation and Diagnostic Cards

## References

- Nocedal and Wright, Numerical Optimization.
- Bertsekas, Nonlinear Programming.
- Polyak, Introduction to Optimization.
- Nesterov, A Method for Solving the Convex Programming Problem.
- Goodfellow, Bengio, and Courville, Deep Learning.
- Bottou, Curtis, and Nocedal, Optimization Methods for Large-Scale Machine Learning.
- PyTorch optimizer and scheduler documentation.
- Optax documentation for composable optimizer transformations.
