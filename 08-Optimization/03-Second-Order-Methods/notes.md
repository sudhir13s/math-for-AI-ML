[Previous: Gradient Descent](../02-Gradient-Descent/notes.md) | [Back to Chapter 8: Optimization](../README.md) | [Next: Constrained Optimization](../04-Constrained-Optimization/notes.md)

---

# Second-Order Methods

> _"Curvature is expensive information, but sometimes it is exactly the information you are missing."_

## Overview

Second-Order Methods is part of the optimization spine of this curriculum. It explains how
mathematical assumptions become training behavior, and how training behavior becomes measurable
engineering evidence. The section is the canonical home for Hessian-aware methods: Newton,
damped Newton, Gauss-Newton, quasi-Newton, natural gradient, Hessian-vector products, and
curvature tradeoffs.

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
- The previous optimization section, [Gradient Descent](../02-Gradient-Descent/notes.md), is assumed as local context.

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Interactive derivations, numerical checks, and visual diagnostics for Second-Order Methods. |
| [exercises.ipynb](exercises.ipynb) | Graded implementation and proof exercises for Second-Order Methods. |

## Learning Objectives

- Define the canonical objects used in Second-Order Methods with repository notation.
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
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 Why Second-Order Methods matters for training systems](#11-why-secondorder-methods-matters-for-training-systems)
  - [1.2 The optimization object: parameters, objective, algorithm, and diagnostic](#12-the-optimization-object-parameters-objective-algorithm-and-diagnostic)
  - [1.3 Historical arc from classical optimization to modern AI](#13-historical-arc-from-classical-optimization-to-modern-ai)
  - [1.4 What this section treats as canonical scope](#14-what-this-section-treats-as-canonical-scope)
  - [1.5 A first mental model for LLM training](#15-a-first-mental-model-for-llm-training)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Primary definition: Hessian matrix](#21-primary-definition-hessian-matrix)
  - [2.2 Secondary definition: quadratic model](#22-secondary-definition-quadratic-model)
  - [2.3 Algorithmic object: Newton step](#23-algorithmic-object-newton-step)
  - [2.4 Examples, non-examples, and boundary cases](#24-examples-nonexamples-and-boundary-cases)
  - [2.5 Notation, dimensions, and assumptions](#25-notation-dimensions-and-assumptions)
- [3. Core Theory I: Geometry and Guarantees](#3-core-theory-i-geometry-and-guarantees)
  - [3.1 Geometry of Newton decrement](#31-geometry-of-newton-decrement)
  - [3.2 Key inequality for damped Newton](#32-key-inequality-for-damped-newton)
  - [3.3 Role of modified Cholesky](#33-role-of-modified-cholesky)
  - [3.4 Proof template and what the proof actually buys](#34-proof-template-and-what-the-proof-actually-buys)
  - [3.5 Failure modes when assumptions are removed](#35-failure-modes-when-assumptions-are-removed)
- [4. Core Theory II: Algorithms and Dynamics](#4-core-theory-ii-algorithms-and-dynamics)
  - [4.1 Algorithmic update for trust-region preview](#41-algorithmic-update-for-trustregion-preview)
  - [4.2 Stability role of Gauss-Newton](#42-stability-role-of-gaussnewton)
  - [4.3 Rate or complexity controlled by Levenberg-Marquardt](#43-rate-or-complexity-controlled-by-levenbergmarquardt)
  - [4.4 Diagnostic interpretation of the update path](#44-diagnostic-interpretation-of-the-update-path)
  - [4.5 Connection to the next section in the chapter](#45-connection-to-the-next-section-in-the-chapter)
- [5. Core Theory III: Practical Variants](#5-core-theory-iii-practical-variants)
  - [5.1 Variant built around secant equation](#51-variant-built-around-secant-equation)
  - [5.2 Variant built around BFGS](#52-variant-built-around-bfgs)
  - [5.3 Variant built around L-BFGS](#53-variant-built-around-lbfgs)
  - [5.4 Implementation constraints and numerical stability](#54-implementation-constraints-and-numerical-stability)
  - [5.5 What belongs here versus neighboring sections](#55-what-belongs-here-versus-neighboring-sections)
- [6. Advanced Topics](#6-advanced-topics)
  - [6.1 Advanced view of two-loop recursion](#61-advanced-view-of-twoloop-recursion)
  - [6.2 Advanced view of Hessian-vector products](#62-advanced-view-of-hessianvector-products)
  - [6.3 Advanced view of Fisher information](#63-advanced-view-of-fisher-information)
  - [6.4 Infinite-dimensional or large-scale interpretation](#64-infinitedimensional-or-largescale-interpretation)
  - [6.5 Open questions for frontier model training](#65-open-questions-for-frontier-model-training)
- [7. Applications in Machine Learning](#7-applications-in-machine-learning)
  - [7.1 K-FAC and natural-gradient style preconditioners for neural networks](#71-kfac-and-naturalgradient-style-preconditioners-for-neural-networks)
  - [7.2 L-BFGS for small-batch fine-tuning and classical ML objectives](#72-lbfgs-for-smallbatch-finetuning-and-classical-ml-objectives)
  - [7.3 Hessian-vector products for sharpness and interpretability diagnostics](#73-hessianvector-products-for-sharpness-and-interpretability-diagnostics)
  - [7.4 structured preconditioning proposals such as Shampoo, SOAP, and Muon-family methods](#74-structured-preconditioning-proposals-such-as-shampoo-soap-and-muonfamily-methods)
  - [7.5 Diagnostic checklist for real experiments](#75-diagnostic-checklist-for-real-experiments)
- [8. Implementation and Diagnostics](#8-implementation-and-diagnostics)
  - [8.1 Minimal NumPy experiment for natural gradient](#81-minimal-numpy-experiment-for-natural-gradient)
  - [8.2 Monitoring signal for K-FAC](#82-monitoring-signal-for-kfac)
  - [8.3 Failure signature for Shampoo](#83-failure-signature-for-shampoo)
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

This block develops intuition for Second-Order Methods. It keeps the scope local to this section
while pointing forward when a neighboring topic owns the full treatment.

### 1.1 Why Second-Order Methods matters for training systems

In this section, damped Newton is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Second-Order Methods, the phrase "Why Second-Order Methods matters for training systems" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **damped Newton** is the part of Second-Order Methods that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where damped Newton can be computed directly and compared with theory.
- A logistic-regression or softmax objective where damped Newton affects optimization but the model remains interpretable.
- A transformer training diagnostic where damped Newton appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating damped Newton as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - H_f(\boldsymbol{\theta}_t)^{-1}\nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving damped
Newton, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes damped Newton visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about damped Newton is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- K-FAC and natural-gradient style preconditioners for neural networks.
- L-BFGS for small-batch fine-tuning and classical ML objectives.
- Hessian-vector products for sharpness and interpretability diagnostics.
- structured preconditioning proposals such as Shampoo, SOAP, and Muon-family methods.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 1.2 The optimization object: parameters, objective, algorithm, and diagnostic

In this section, modified Cholesky is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Second-Order Methods, the phrase "The optimization object: parameters, objective, algorithm, and
diagnostic" means a precise mathematical habit: state the assumptions, write the update,
identify what can be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **modified Cholesky** is the part of Second-Order Methods that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where modified Cholesky can be computed directly and compared with theory.
- A logistic-regression or softmax objective where modified Cholesky affects optimization but the model remains interpretable.
- A transformer training diagnostic where modified Cholesky appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating modified Cholesky as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - H_f(\boldsymbol{\theta}_t)^{-1}\nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving modified
Cholesky, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes modified Cholesky visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about modified Cholesky is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- K-FAC and natural-gradient style preconditioners for neural networks.
- L-BFGS for small-batch fine-tuning and classical ML objectives.
- Hessian-vector products for sharpness and interpretability diagnostics.
- structured preconditioning proposals such as Shampoo, SOAP, and Muon-family methods.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 1.3 Historical arc from classical optimization to modern AI

In this section, trust-region preview is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Second-Order Methods, the phrase "Historical arc from classical optimization to modern AI" means
a precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **trust-region preview** is the part of Second-Order Methods that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where trust-region preview can be computed directly and compared with theory.
- A logistic-regression or softmax objective where trust-region preview affects optimization but the model remains interpretable.
- A transformer training diagnostic where trust-region preview appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating trust-region preview as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - H_f(\boldsymbol{\theta}_t)^{-1}\nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
trust-region preview, and use the section assumptions to bound the change in objective value. If
the assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes trust-region preview visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about trust-region preview is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- K-FAC and natural-gradient style preconditioners for neural networks.
- L-BFGS for small-batch fine-tuning and classical ML objectives.
- Hessian-vector products for sharpness and interpretability diagnostics.
- structured preconditioning proposals such as Shampoo, SOAP, and Muon-family methods.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 1.4 What this section treats as canonical scope

In this section, Gauss-Newton is treated as a concrete optimization object rather than a slogan.
The goal is to understand how it changes the objective, the update rule, the convergence story,
and the diagnostics a practitioner should inspect when training a modern model. For Second-Order
Methods, the phrase "What this section treats as canonical scope" means a precise mathematical
habit: state the assumptions, write the update, identify what can be measured, and connect the
result to a real AI training decision.

> **Definition.**
>
> For this section, **Gauss-Newton** is the part of Second-Order Methods that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Gauss-Newton can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Gauss-Newton affects optimization but the model remains interpretable.
- A transformer training diagnostic where Gauss-Newton appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Gauss-Newton as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - H_f(\boldsymbol{\theta}_t)^{-1}\nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
Gauss-Newton, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Gauss-Newton visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Gauss-Newton is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- K-FAC and natural-gradient style preconditioners for neural networks.
- L-BFGS for small-batch fine-tuning and classical ML objectives.
- Hessian-vector products for sharpness and interpretability diagnostics.
- structured preconditioning proposals such as Shampoo, SOAP, and Muon-family methods.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 1.5 A first mental model for LLM training

In this section, Levenberg-Marquardt is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Second-Order Methods, the phrase "A first mental model for LLM training" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **Levenberg-Marquardt** is the part of Second-Order Methods that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Levenberg-Marquardt can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Levenberg-Marquardt affects optimization but the model remains interpretable.
- A transformer training diagnostic where Levenberg-Marquardt appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Levenberg-Marquardt as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - H_f(\boldsymbol{\theta}_t)^{-1}\nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
Levenberg-Marquardt, and use the section assumptions to bound the change in objective value. If
the assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Levenberg-Marquardt visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Levenberg-Marquardt is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- K-FAC and natural-gradient style preconditioners for neural networks.
- L-BFGS for small-batch fine-tuning and classical ML objectives.
- Hessian-vector products for sharpness and interpretability diagnostics.
- structured preconditioning proposals such as Shampoo, SOAP, and Muon-family methods.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 2. Formal Definitions

This block develops formal definitions for Second-Order Methods. It keeps the scope local to
this section while pointing forward when a neighboring topic owns the full treatment.

### 2.1 Primary definition: Hessian matrix

In this section, Gauss-Newton is treated as a concrete optimization object rather than a slogan.
The goal is to understand how it changes the objective, the update rule, the convergence story,
and the diagnostics a practitioner should inspect when training a modern model. For Second-Order
Methods, the phrase "Primary definition: Hessian matrix" means a precise mathematical habit:
state the assumptions, write the update, identify what can be measured, and connect the result
to a real AI training decision.

> **Definition.**
>
> For this section, **Gauss-Newton** is the part of Second-Order Methods that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Gauss-Newton can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Gauss-Newton affects optimization but the model remains interpretable.
- A transformer training diagnostic where Gauss-Newton appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Gauss-Newton as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - H_f(\boldsymbol{\theta}_t)^{-1}\nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
Gauss-Newton, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Gauss-Newton visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Gauss-Newton is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- K-FAC and natural-gradient style preconditioners for neural networks.
- L-BFGS for small-batch fine-tuning and classical ML objectives.
- Hessian-vector products for sharpness and interpretability diagnostics.
- structured preconditioning proposals such as Shampoo, SOAP, and Muon-family methods.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 2.2 Secondary definition: quadratic model

In this section, Levenberg-Marquardt is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Second-Order Methods, the phrase "Secondary definition: quadratic model" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **Levenberg-Marquardt** is the part of Second-Order Methods that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Levenberg-Marquardt can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Levenberg-Marquardt affects optimization but the model remains interpretable.
- A transformer training diagnostic where Levenberg-Marquardt appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Levenberg-Marquardt as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - H_f(\boldsymbol{\theta}_t)^{-1}\nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
Levenberg-Marquardt, and use the section assumptions to bound the change in objective value. If
the assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Levenberg-Marquardt visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Levenberg-Marquardt is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- K-FAC and natural-gradient style preconditioners for neural networks.
- L-BFGS for small-batch fine-tuning and classical ML objectives.
- Hessian-vector products for sharpness and interpretability diagnostics.
- structured preconditioning proposals such as Shampoo, SOAP, and Muon-family methods.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 2.3 Algorithmic object: Newton step

In this section, secant equation is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Second-Order Methods, the phrase "Algorithmic object: Newton step" means a precise mathematical
habit: state the assumptions, write the update, identify what can be measured, and connect the
result to a real AI training decision.

> **Definition.**
>
> For this section, **secant equation** is the part of Second-Order Methods that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where secant equation can be computed directly and compared with theory.
- A logistic-regression or softmax objective where secant equation affects optimization but the model remains interpretable.
- A transformer training diagnostic where secant equation appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating secant equation as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - H_f(\boldsymbol{\theta}_t)^{-1}\nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving secant
equation, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes secant equation visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about secant equation is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- K-FAC and natural-gradient style preconditioners for neural networks.
- L-BFGS for small-batch fine-tuning and classical ML objectives.
- Hessian-vector products for sharpness and interpretability diagnostics.
- structured preconditioning proposals such as Shampoo, SOAP, and Muon-family methods.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 2.4 Examples, non-examples, and boundary cases

In this section, BFGS is treated as a concrete optimization object rather than a slogan. The
goal is to understand how it changes the objective, the update rule, the convergence story, and
the diagnostics a practitioner should inspect when training a modern model. For Second-Order
Methods, the phrase "Examples, non-examples, and boundary cases" means a precise mathematical
habit: state the assumptions, write the update, identify what can be measured, and connect the
result to a real AI training decision.

> **Definition.**
>
> For this section, **BFGS** is the part of Second-Order Methods that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where BFGS can be computed directly and compared with theory.
- A logistic-regression or softmax objective where BFGS affects optimization but the model remains interpretable.
- A transformer training diagnostic where BFGS appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating BFGS as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - H_f(\boldsymbol{\theta}_t)^{-1}\nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving BFGS, and
use the section assumptions to bound the change in objective value. If the assumption is
geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes BFGS visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about BFGS is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- K-FAC and natural-gradient style preconditioners for neural networks.
- L-BFGS for small-batch fine-tuning and classical ML objectives.
- Hessian-vector products for sharpness and interpretability diagnostics.
- structured preconditioning proposals such as Shampoo, SOAP, and Muon-family methods.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 2.5 Notation, dimensions, and assumptions

In this section, L-BFGS is treated as a concrete optimization object rather than a slogan. The
goal is to understand how it changes the objective, the update rule, the convergence story, and
the diagnostics a practitioner should inspect when training a modern model. For Second-Order
Methods, the phrase "Notation, dimensions, and assumptions" means a precise mathematical habit:
state the assumptions, write the update, identify what can be measured, and connect the result
to a real AI training decision.

> **Definition.**
>
> For this section, **L-BFGS** is the part of Second-Order Methods that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where L-BFGS can be computed directly and compared with theory.
- A logistic-regression or softmax objective where L-BFGS affects optimization but the model remains interpretable.
- A transformer training diagnostic where L-BFGS appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating L-BFGS as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - H_f(\boldsymbol{\theta}_t)^{-1}\nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving L-BFGS,
and use the section assumptions to bound the change in objective value. If the assumption is
geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes L-BFGS visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about L-BFGS is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- K-FAC and natural-gradient style preconditioners for neural networks.
- L-BFGS for small-batch fine-tuning and classical ML objectives.
- Hessian-vector products for sharpness and interpretability diagnostics.
- structured preconditioning proposals such as Shampoo, SOAP, and Muon-family methods.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 3. Core Theory I: Geometry and Guarantees

This block develops core theory i: geometry and guarantees for Second-Order Methods. It keeps
the scope local to this section while pointing forward when a neighboring topic owns the full
treatment.

### 3.1 Geometry of Newton decrement

In this section, BFGS is treated as a concrete optimization object rather than a slogan. The
goal is to understand how it changes the objective, the update rule, the convergence story, and
the diagnostics a practitioner should inspect when training a modern model. For Second-Order
Methods, the phrase "Geometry of Newton decrement" means a precise mathematical habit: state the
assumptions, write the update, identify what can be measured, and connect the result to a real
AI training decision.

> **Definition.**
>
> For this section, **BFGS** is the part of Second-Order Methods that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where BFGS can be computed directly and compared with theory.
- A logistic-regression or softmax objective where BFGS affects optimization but the model remains interpretable.
- A transformer training diagnostic where BFGS appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating BFGS as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - H_f(\boldsymbol{\theta}_t)^{-1}\nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving BFGS, and
use the section assumptions to bound the change in objective value. If the assumption is
geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes BFGS visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about BFGS is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- K-FAC and natural-gradient style preconditioners for neural networks.
- L-BFGS for small-batch fine-tuning and classical ML objectives.
- Hessian-vector products for sharpness and interpretability diagnostics.
- structured preconditioning proposals such as Shampoo, SOAP, and Muon-family methods.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 3.2 Key inequality for damped Newton

In this section, L-BFGS is treated as a concrete optimization object rather than a slogan. The
goal is to understand how it changes the objective, the update rule, the convergence story, and
the diagnostics a practitioner should inspect when training a modern model. For Second-Order
Methods, the phrase "Key inequality for damped Newton" means a precise mathematical habit: state
the assumptions, write the update, identify what can be measured, and connect the result to a
real AI training decision.

> **Definition.**
>
> For this section, **L-BFGS** is the part of Second-Order Methods that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where L-BFGS can be computed directly and compared with theory.
- A logistic-regression or softmax objective where L-BFGS affects optimization but the model remains interpretable.
- A transformer training diagnostic where L-BFGS appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating L-BFGS as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - H_f(\boldsymbol{\theta}_t)^{-1}\nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving L-BFGS,
and use the section assumptions to bound the change in objective value. If the assumption is
geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes L-BFGS visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about L-BFGS is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- K-FAC and natural-gradient style preconditioners for neural networks.
- L-BFGS for small-batch fine-tuning and classical ML objectives.
- Hessian-vector products for sharpness and interpretability diagnostics.
- structured preconditioning proposals such as Shampoo, SOAP, and Muon-family methods.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 3.3 Role of modified Cholesky

In this section, two-loop recursion is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Second-Order Methods, the phrase "Role of modified Cholesky" means a precise mathematical habit:
state the assumptions, write the update, identify what can be measured, and connect the result
to a real AI training decision.

> **Definition.**
>
> For this section, **two-loop recursion** is the part of Second-Order Methods that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where two-loop recursion can be computed directly and compared with theory.
- A logistic-regression or softmax objective where two-loop recursion affects optimization but the model remains interpretable.
- A transformer training diagnostic where two-loop recursion appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating two-loop recursion as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - H_f(\boldsymbol{\theta}_t)^{-1}\nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving two-loop
recursion, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes two-loop recursion visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about two-loop recursion is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- K-FAC and natural-gradient style preconditioners for neural networks.
- L-BFGS for small-batch fine-tuning and classical ML objectives.
- Hessian-vector products for sharpness and interpretability diagnostics.
- structured preconditioning proposals such as Shampoo, SOAP, and Muon-family methods.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 3.4 Proof template and what the proof actually buys

In this section, Hessian-vector products is treated as a concrete optimization object rather
than a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Second-Order Methods, the phrase "Proof template and what the proof actually buys"
means a precise mathematical habit: state the assumptions, write the update, identify what can
be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **Hessian-vector products** is the part of Second-Order Methods that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Hessian-vector products can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Hessian-vector products affects optimization but the model remains interpretable.
- A transformer training diagnostic where Hessian-vector products appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Hessian-vector products as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - H_f(\boldsymbol{\theta}_t)^{-1}\nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
Hessian-vector products, and use the section assumptions to bound the change in objective value.
If the assumption is geometric, the proof turns a picture into an inequality. If the assumption
is stochastic, the proof takes conditional expectation before applying the bound. If the
assumption is algorithmic, the proof checks that the proposed update is a descent, projection,
or preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Hessian-vector products visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Hessian-vector products is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- K-FAC and natural-gradient style preconditioners for neural networks.
- L-BFGS for small-batch fine-tuning and classical ML objectives.
- Hessian-vector products for sharpness and interpretability diagnostics.
- structured preconditioning proposals such as Shampoo, SOAP, and Muon-family methods.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 3.5 Failure modes when assumptions are removed

In this section, Fisher information is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Second-Order Methods, the phrase "Failure modes when assumptions are removed" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **Fisher information** is the part of Second-Order Methods that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Fisher information can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Fisher information affects optimization but the model remains interpretable.
- A transformer training diagnostic where Fisher information appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Fisher information as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - H_f(\boldsymbol{\theta}_t)^{-1}\nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving Fisher
information, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Fisher information visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Fisher information is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- K-FAC and natural-gradient style preconditioners for neural networks.
- L-BFGS for small-batch fine-tuning and classical ML objectives.
- Hessian-vector products for sharpness and interpretability diagnostics.
- structured preconditioning proposals such as Shampoo, SOAP, and Muon-family methods.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 4. Core Theory II: Algorithms and Dynamics

This block develops core theory ii: algorithms and dynamics for Second-Order Methods. It keeps
the scope local to this section while pointing forward when a neighboring topic owns the full
treatment.

### 4.1 Algorithmic update for trust-region preview

In this section, Hessian-vector products is treated as a concrete optimization object rather
than a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Second-Order Methods, the phrase "Algorithmic update for trust-region preview" means
a precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **Hessian-vector products** is the part of Second-Order Methods that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Hessian-vector products can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Hessian-vector products affects optimization but the model remains interpretable.
- A transformer training diagnostic where Hessian-vector products appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Hessian-vector products as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - H_f(\boldsymbol{\theta}_t)^{-1}\nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
Hessian-vector products, and use the section assumptions to bound the change in objective value.
If the assumption is geometric, the proof turns a picture into an inequality. If the assumption
is stochastic, the proof takes conditional expectation before applying the bound. If the
assumption is algorithmic, the proof checks that the proposed update is a descent, projection,
or preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Hessian-vector products visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Hessian-vector products is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- K-FAC and natural-gradient style preconditioners for neural networks.
- L-BFGS for small-batch fine-tuning and classical ML objectives.
- Hessian-vector products for sharpness and interpretability diagnostics.
- structured preconditioning proposals such as Shampoo, SOAP, and Muon-family methods.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 4.2 Stability role of Gauss-Newton

In this section, Fisher information is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Second-Order Methods, the phrase "Stability role of Gauss-Newton" means a precise mathematical
habit: state the assumptions, write the update, identify what can be measured, and connect the
result to a real AI training decision.

> **Definition.**
>
> For this section, **Fisher information** is the part of Second-Order Methods that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Fisher information can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Fisher information affects optimization but the model remains interpretable.
- A transformer training diagnostic where Fisher information appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Fisher information as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - H_f(\boldsymbol{\theta}_t)^{-1}\nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving Fisher
information, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Fisher information visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Fisher information is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- K-FAC and natural-gradient style preconditioners for neural networks.
- L-BFGS for small-batch fine-tuning and classical ML objectives.
- Hessian-vector products for sharpness and interpretability diagnostics.
- structured preconditioning proposals such as Shampoo, SOAP, and Muon-family methods.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 4.3 Rate or complexity controlled by Levenberg-Marquardt

In this section, natural gradient is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Second-Order Methods, the phrase "Rate or complexity controlled by Levenberg-Marquardt" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **natural gradient** is the part of Second-Order Methods that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where natural gradient can be computed directly and compared with theory.
- A logistic-regression or softmax objective where natural gradient affects optimization but the model remains interpretable.
- A transformer training diagnostic where natural gradient appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating natural gradient as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - H_f(\boldsymbol{\theta}_t)^{-1}\nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving natural
gradient, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes natural gradient visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about natural gradient is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- K-FAC and natural-gradient style preconditioners for neural networks.
- L-BFGS for small-batch fine-tuning and classical ML objectives.
- Hessian-vector products for sharpness and interpretability diagnostics.
- structured preconditioning proposals such as Shampoo, SOAP, and Muon-family methods.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 4.4 Diagnostic interpretation of the update path

In this section, K-FAC is treated as a concrete optimization object rather than a slogan. The
goal is to understand how it changes the objective, the update rule, the convergence story, and
the diagnostics a practitioner should inspect when training a modern model. For Second-Order
Methods, the phrase "Diagnostic interpretation of the update path" means a precise mathematical
habit: state the assumptions, write the update, identify what can be measured, and connect the
result to a real AI training decision.

> **Definition.**
>
> For this section, **K-FAC** is the part of Second-Order Methods that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where K-FAC can be computed directly and compared with theory.
- A logistic-regression or softmax objective where K-FAC affects optimization but the model remains interpretable.
- A transformer training diagnostic where K-FAC appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating K-FAC as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - H_f(\boldsymbol{\theta}_t)^{-1}\nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving K-FAC, and
use the section assumptions to bound the change in objective value. If the assumption is
geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes K-FAC visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about K-FAC is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- K-FAC and natural-gradient style preconditioners for neural networks.
- L-BFGS for small-batch fine-tuning and classical ML objectives.
- Hessian-vector products for sharpness and interpretability diagnostics.
- structured preconditioning proposals such as Shampoo, SOAP, and Muon-family methods.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 4.5 Connection to the next section in the chapter

In this section, Shampoo is treated as a concrete optimization object rather than a slogan. The
goal is to understand how it changes the objective, the update rule, the convergence story, and
the diagnostics a practitioner should inspect when training a modern model. For Second-Order
Methods, the phrase "Connection to the next section in the chapter" means a precise mathematical
habit: state the assumptions, write the update, identify what can be measured, and connect the
result to a real AI training decision.

> **Definition.**
>
> For this section, **Shampoo** is the part of Second-Order Methods that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Shampoo can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Shampoo affects optimization but the model remains interpretable.
- A transformer training diagnostic where Shampoo appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Shampoo as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - H_f(\boldsymbol{\theta}_t)^{-1}\nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving Shampoo,
and use the section assumptions to bound the change in objective value. If the assumption is
geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Shampoo visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Shampoo is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- K-FAC and natural-gradient style preconditioners for neural networks.
- L-BFGS for small-batch fine-tuning and classical ML objectives.
- Hessian-vector products for sharpness and interpretability diagnostics.
- structured preconditioning proposals such as Shampoo, SOAP, and Muon-family methods.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 5. Core Theory III: Practical Variants

This block develops core theory iii: practical variants for Second-Order Methods. It keeps the
scope local to this section while pointing forward when a neighboring topic owns the full
treatment.

### 5.1 Variant built around secant equation

In this section, K-FAC is treated as a concrete optimization object rather than a slogan. The
goal is to understand how it changes the objective, the update rule, the convergence story, and
the diagnostics a practitioner should inspect when training a modern model. For Second-Order
Methods, the phrase "Variant built around secant equation" means a precise mathematical habit:
state the assumptions, write the update, identify what can be measured, and connect the result
to a real AI training decision.

> **Definition.**
>
> For this section, **K-FAC** is the part of Second-Order Methods that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where K-FAC can be computed directly and compared with theory.
- A logistic-regression or softmax objective where K-FAC affects optimization but the model remains interpretable.
- A transformer training diagnostic where K-FAC appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating K-FAC as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - H_f(\boldsymbol{\theta}_t)^{-1}\nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving K-FAC, and
use the section assumptions to bound the change in objective value. If the assumption is
geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes K-FAC visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about K-FAC is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- K-FAC and natural-gradient style preconditioners for neural networks.
- L-BFGS for small-batch fine-tuning and classical ML objectives.
- Hessian-vector products for sharpness and interpretability diagnostics.
- structured preconditioning proposals such as Shampoo, SOAP, and Muon-family methods.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 5.2 Variant built around BFGS

In this section, Shampoo is treated as a concrete optimization object rather than a slogan. The
goal is to understand how it changes the objective, the update rule, the convergence story, and
the diagnostics a practitioner should inspect when training a modern model. For Second-Order
Methods, the phrase "Variant built around BFGS" means a precise mathematical habit: state the
assumptions, write the update, identify what can be measured, and connect the result to a real
AI training decision.

> **Definition.**
>
> For this section, **Shampoo** is the part of Second-Order Methods that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Shampoo can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Shampoo affects optimization but the model remains interpretable.
- A transformer training diagnostic where Shampoo appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Shampoo as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - H_f(\boldsymbol{\theta}_t)^{-1}\nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving Shampoo,
and use the section assumptions to bound the change in objective value. If the assumption is
geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Shampoo visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Shampoo is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- K-FAC and natural-gradient style preconditioners for neural networks.
- L-BFGS for small-batch fine-tuning and classical ML objectives.
- Hessian-vector products for sharpness and interpretability diagnostics.
- structured preconditioning proposals such as Shampoo, SOAP, and Muon-family methods.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 5.3 Variant built around L-BFGS

In this section, SOAP is treated as a concrete optimization object rather than a slogan. The
goal is to understand how it changes the objective, the update rule, the convergence story, and
the diagnostics a practitioner should inspect when training a modern model. For Second-Order
Methods, the phrase "Variant built around L-BFGS" means a precise mathematical habit: state the
assumptions, write the update, identify what can be measured, and connect the result to a real
AI training decision.

> **Definition.**
>
> For this section, **SOAP** is the part of Second-Order Methods that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where SOAP can be computed directly and compared with theory.
- A logistic-regression or softmax objective where SOAP affects optimization but the model remains interpretable.
- A transformer training diagnostic where SOAP appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating SOAP as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - H_f(\boldsymbol{\theta}_t)^{-1}\nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving SOAP, and
use the section assumptions to bound the change in objective value. If the assumption is
geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes SOAP visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about SOAP is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- K-FAC and natural-gradient style preconditioners for neural networks.
- L-BFGS for small-batch fine-tuning and classical ML objectives.
- Hessian-vector products for sharpness and interpretability diagnostics.
- structured preconditioning proposals such as Shampoo, SOAP, and Muon-family methods.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 5.4 Implementation constraints and numerical stability

In this section, Muon preview is treated as a concrete optimization object rather than a slogan.
The goal is to understand how it changes the objective, the update rule, the convergence story,
and the diagnostics a practitioner should inspect when training a modern model. For Second-Order
Methods, the phrase "Implementation constraints and numerical stability" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **Muon preview** is the part of Second-Order Methods that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Muon preview can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Muon preview affects optimization but the model remains interpretable.
- A transformer training diagnostic where Muon preview appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Muon preview as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - H_f(\boldsymbol{\theta}_t)^{-1}\nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving Muon
preview, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Muon preview visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Muon preview is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- K-FAC and natural-gradient style preconditioners for neural networks.
- L-BFGS for small-batch fine-tuning and classical ML objectives.
- Hessian-vector products for sharpness and interpretability diagnostics.
- structured preconditioning proposals such as Shampoo, SOAP, and Muon-family methods.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 5.5 What belongs here versus neighboring sections

In this section, curvature diagnostics is treated as a concrete optimization object rather than
a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Second-Order Methods, the phrase "What belongs here versus neighboring sections"
means a precise mathematical habit: state the assumptions, write the update, identify what can
be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **curvature diagnostics** is the part of Second-Order Methods that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where curvature diagnostics can be computed directly and compared with theory.
- A logistic-regression or softmax objective where curvature diagnostics affects optimization but the model remains interpretable.
- A transformer training diagnostic where curvature diagnostics appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating curvature diagnostics as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - H_f(\boldsymbol{\theta}_t)^{-1}\nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving curvature
diagnostics, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes curvature diagnostics visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about curvature diagnostics is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- K-FAC and natural-gradient style preconditioners for neural networks.
- L-BFGS for small-batch fine-tuning and classical ML objectives.
- Hessian-vector products for sharpness and interpretability diagnostics.
- structured preconditioning proposals such as Shampoo, SOAP, and Muon-family methods.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 6. Advanced Topics

This block develops advanced topics for Second-Order Methods. It keeps the scope local to this
section while pointing forward when a neighboring topic owns the full treatment.

### 6.1 Advanced view of two-loop recursion

In this section, Muon preview is treated as a concrete optimization object rather than a slogan.
The goal is to understand how it changes the objective, the update rule, the convergence story,
and the diagnostics a practitioner should inspect when training a modern model. For Second-Order
Methods, the phrase "Advanced view of two-loop recursion" means a precise mathematical habit:
state the assumptions, write the update, identify what can be measured, and connect the result
to a real AI training decision.

> **Definition.**
>
> For this section, **Muon preview** is the part of Second-Order Methods that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Muon preview can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Muon preview affects optimization but the model remains interpretable.
- A transformer training diagnostic where Muon preview appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Muon preview as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - H_f(\boldsymbol{\theta}_t)^{-1}\nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving Muon
preview, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Muon preview visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Muon preview is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- K-FAC and natural-gradient style preconditioners for neural networks.
- L-BFGS for small-batch fine-tuning and classical ML objectives.
- Hessian-vector products for sharpness and interpretability diagnostics.
- structured preconditioning proposals such as Shampoo, SOAP, and Muon-family methods.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 6.2 Advanced view of Hessian-vector products

In this section, curvature diagnostics is treated as a concrete optimization object rather than
a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Second-Order Methods, the phrase "Advanced view of Hessian-vector products" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **curvature diagnostics** is the part of Second-Order Methods that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where curvature diagnostics can be computed directly and compared with theory.
- A logistic-regression or softmax objective where curvature diagnostics affects optimization but the model remains interpretable.
- A transformer training diagnostic where curvature diagnostics appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating curvature diagnostics as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - H_f(\boldsymbol{\theta}_t)^{-1}\nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving curvature
diagnostics, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes curvature diagnostics visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about curvature diagnostics is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- K-FAC and natural-gradient style preconditioners for neural networks.
- L-BFGS for small-batch fine-tuning and classical ML objectives.
- Hessian-vector products for sharpness and interpretability diagnostics.
- structured preconditioning proposals such as Shampoo, SOAP, and Muon-family methods.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 6.3 Advanced view of Fisher information

In this section, large-model feasibility is treated as a concrete optimization object rather
than a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Second-Order Methods, the phrase "Advanced view of Fisher information" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **large-model feasibility** is the part of Second-Order Methods that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where large-model feasibility can be computed directly and compared with theory.
- A logistic-regression or softmax objective where large-model feasibility affects optimization but the model remains interpretable.
- A transformer training diagnostic where large-model feasibility appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating large-model feasibility as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - H_f(\boldsymbol{\theta}_t)^{-1}\nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
large-model feasibility, and use the section assumptions to bound the change in objective value.
If the assumption is geometric, the proof turns a picture into an inequality. If the assumption
is stochastic, the proof takes conditional expectation before applying the bound. If the
assumption is algorithmic, the proof checks that the proposed update is a descent, projection,
or preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes large-model feasibility visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about large-model feasibility is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- K-FAC and natural-gradient style preconditioners for neural networks.
- L-BFGS for small-batch fine-tuning and classical ML objectives.
- Hessian-vector products for sharpness and interpretability diagnostics.
- structured preconditioning proposals such as Shampoo, SOAP, and Muon-family methods.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 6.4 Infinite-dimensional or large-scale interpretation

In this section, Hessian matrix is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Second-Order Methods, the phrase "Infinite-dimensional or large-scale interpretation" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **Hessian matrix** is the part of Second-Order Methods that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Hessian matrix can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Hessian matrix affects optimization but the model remains interpretable.
- A transformer training diagnostic where Hessian matrix appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Hessian matrix as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - H_f(\boldsymbol{\theta}_t)^{-1}\nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving Hessian
matrix, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Hessian matrix visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Hessian matrix is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- K-FAC and natural-gradient style preconditioners for neural networks.
- L-BFGS for small-batch fine-tuning and classical ML objectives.
- Hessian-vector products for sharpness and interpretability diagnostics.
- structured preconditioning proposals such as Shampoo, SOAP, and Muon-family methods.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 6.5 Open questions for frontier model training

In this section, quadratic model is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Second-Order Methods, the phrase "Open questions for frontier model training" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **quadratic model** is the part of Second-Order Methods that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where quadratic model can be computed directly and compared with theory.
- A logistic-regression or softmax objective where quadratic model affects optimization but the model remains interpretable.
- A transformer training diagnostic where quadratic model appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating quadratic model as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - H_f(\boldsymbol{\theta}_t)^{-1}\nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving quadratic
model, and use the section assumptions to bound the change in objective value. If the assumption
is geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes quadratic model visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about quadratic model is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- K-FAC and natural-gradient style preconditioners for neural networks.
- L-BFGS for small-batch fine-tuning and classical ML objectives.
- Hessian-vector products for sharpness and interpretability diagnostics.
- structured preconditioning proposals such as Shampoo, SOAP, and Muon-family methods.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 7. Applications in Machine Learning

This block develops applications in machine learning for Second-Order Methods. It keeps the
scope local to this section while pointing forward when a neighboring topic owns the full
treatment.

### 7.1 K-FAC and natural-gradient style preconditioners for neural networks

In this section, Hessian matrix is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Second-Order Methods, the phrase "K-FAC and natural-gradient style preconditioners for neural
networks" means a precise mathematical habit: state the assumptions, write the update, identify
what can be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **Hessian matrix** is the part of Second-Order Methods that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Hessian matrix can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Hessian matrix affects optimization but the model remains interpretable.
- A transformer training diagnostic where Hessian matrix appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Hessian matrix as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - H_f(\boldsymbol{\theta}_t)^{-1}\nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving Hessian
matrix, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Hessian matrix visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Hessian matrix is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- K-FAC and natural-gradient style preconditioners for neural networks.
- L-BFGS for small-batch fine-tuning and classical ML objectives.
- Hessian-vector products for sharpness and interpretability diagnostics.
- structured preconditioning proposals such as Shampoo, SOAP, and Muon-family methods.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 7.2 L-BFGS for small-batch fine-tuning and classical ML objectives

In this section, quadratic model is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Second-Order Methods, the phrase "L-BFGS for small-batch fine-tuning and classical ML
objectives" means a precise mathematical habit: state the assumptions, write the update,
identify what can be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **quadratic model** is the part of Second-Order Methods that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where quadratic model can be computed directly and compared with theory.
- A logistic-regression or softmax objective where quadratic model affects optimization but the model remains interpretable.
- A transformer training diagnostic where quadratic model appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating quadratic model as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - H_f(\boldsymbol{\theta}_t)^{-1}\nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving quadratic
model, and use the section assumptions to bound the change in objective value. If the assumption
is geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes quadratic model visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about quadratic model is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- K-FAC and natural-gradient style preconditioners for neural networks.
- L-BFGS for small-batch fine-tuning and classical ML objectives.
- Hessian-vector products for sharpness and interpretability diagnostics.
- structured preconditioning proposals such as Shampoo, SOAP, and Muon-family methods.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 7.3 Hessian-vector products for sharpness and interpretability diagnostics

In this section, Newton step is treated as a concrete optimization object rather than a slogan.
The goal is to understand how it changes the objective, the update rule, the convergence story,
and the diagnostics a practitioner should inspect when training a modern model. For Second-Order
Methods, the phrase "Hessian-vector products for sharpness and interpretability diagnostics"
means a precise mathematical habit: state the assumptions, write the update, identify what can
be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **Newton step** is the part of Second-Order Methods that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Newton step can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Newton step affects optimization but the model remains interpretable.
- A transformer training diagnostic where Newton step appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Newton step as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - H_f(\boldsymbol{\theta}_t)^{-1}\nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving Newton
step, and use the section assumptions to bound the change in objective value. If the assumption
is geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Newton step visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Newton step is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- K-FAC and natural-gradient style preconditioners for neural networks.
- L-BFGS for small-batch fine-tuning and classical ML objectives.
- Hessian-vector products for sharpness and interpretability diagnostics.
- structured preconditioning proposals such as Shampoo, SOAP, and Muon-family methods.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 7.4 structured preconditioning proposals such as Shampoo, SOAP, and Muon-family methods

In this section, Newton decrement is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Second-Order Methods, the phrase "structured preconditioning proposals such as Shampoo, SOAP,
and Muon-family methods" means a precise mathematical habit: state the assumptions, write the
update, identify what can be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **Newton decrement** is the part of Second-Order Methods that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Newton decrement can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Newton decrement affects optimization but the model remains interpretable.
- A transformer training diagnostic where Newton decrement appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Newton decrement as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - H_f(\boldsymbol{\theta}_t)^{-1}\nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving Newton
decrement, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Newton decrement visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Newton decrement is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- K-FAC and natural-gradient style preconditioners for neural networks.
- L-BFGS for small-batch fine-tuning and classical ML objectives.
- Hessian-vector products for sharpness and interpretability diagnostics.
- structured preconditioning proposals such as Shampoo, SOAP, and Muon-family methods.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 7.5 Diagnostic checklist for real experiments

In this section, damped Newton is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Second-Order Methods, the phrase "Diagnostic checklist for real experiments" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **damped Newton** is the part of Second-Order Methods that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where damped Newton can be computed directly and compared with theory.
- A logistic-regression or softmax objective where damped Newton affects optimization but the model remains interpretable.
- A transformer training diagnostic where damped Newton appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating damped Newton as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - H_f(\boldsymbol{\theta}_t)^{-1}\nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving damped
Newton, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes damped Newton visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about damped Newton is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- K-FAC and natural-gradient style preconditioners for neural networks.
- L-BFGS for small-batch fine-tuning and classical ML objectives.
- Hessian-vector products for sharpness and interpretability diagnostics.
- structured preconditioning proposals such as Shampoo, SOAP, and Muon-family methods.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 8. Implementation and Diagnostics

This block develops implementation and diagnostics for Second-Order Methods. It keeps the scope
local to this section while pointing forward when a neighboring topic owns the full treatment.

### 8.1 Minimal NumPy experiment for natural gradient

In this section, Newton decrement is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Second-Order Methods, the phrase "Minimal NumPy experiment for natural gradient" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **Newton decrement** is the part of Second-Order Methods that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Newton decrement can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Newton decrement affects optimization but the model remains interpretable.
- A transformer training diagnostic where Newton decrement appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Newton decrement as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - H_f(\boldsymbol{\theta}_t)^{-1}\nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving Newton
decrement, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Newton decrement visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Newton decrement is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- K-FAC and natural-gradient style preconditioners for neural networks.
- L-BFGS for small-batch fine-tuning and classical ML objectives.
- Hessian-vector products for sharpness and interpretability diagnostics.
- structured preconditioning proposals such as Shampoo, SOAP, and Muon-family methods.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 8.2 Monitoring signal for K-FAC

In this section, damped Newton is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Second-Order Methods, the phrase "Monitoring signal for K-FAC" means a precise mathematical
habit: state the assumptions, write the update, identify what can be measured, and connect the
result to a real AI training decision.

> **Definition.**
>
> For this section, **damped Newton** is the part of Second-Order Methods that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where damped Newton can be computed directly and compared with theory.
- A logistic-regression or softmax objective where damped Newton affects optimization but the model remains interpretable.
- A transformer training diagnostic where damped Newton appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating damped Newton as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - H_f(\boldsymbol{\theta}_t)^{-1}\nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving damped
Newton, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes damped Newton visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about damped Newton is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- K-FAC and natural-gradient style preconditioners for neural networks.
- L-BFGS for small-batch fine-tuning and classical ML objectives.
- Hessian-vector products for sharpness and interpretability diagnostics.
- structured preconditioning proposals such as Shampoo, SOAP, and Muon-family methods.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 8.3 Failure signature for Shampoo

In this section, modified Cholesky is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Second-Order Methods, the phrase "Failure signature for Shampoo" means a precise mathematical
habit: state the assumptions, write the update, identify what can be measured, and connect the
result to a real AI training decision.

> **Definition.**
>
> For this section, **modified Cholesky** is the part of Second-Order Methods that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where modified Cholesky can be computed directly and compared with theory.
- A logistic-regression or softmax objective where modified Cholesky affects optimization but the model remains interpretable.
- A transformer training diagnostic where modified Cholesky appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating modified Cholesky as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - H_f(\boldsymbol{\theta}_t)^{-1}\nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving modified
Cholesky, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes modified Cholesky visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about modified Cholesky is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- K-FAC and natural-gradient style preconditioners for neural networks.
- L-BFGS for small-batch fine-tuning and classical ML objectives.
- Hessian-vector products for sharpness and interpretability diagnostics.
- structured preconditioning proposals such as Shampoo, SOAP, and Muon-family methods.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 8.4 Framework-level implementation pattern

In this section, trust-region preview is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Second-Order Methods, the phrase "Framework-level implementation pattern" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **trust-region preview** is the part of Second-Order Methods that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where trust-region preview can be computed directly and compared with theory.
- A logistic-regression or softmax objective where trust-region preview affects optimization but the model remains interpretable.
- A transformer training diagnostic where trust-region preview appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating trust-region preview as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - H_f(\boldsymbol{\theta}_t)^{-1}\nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
trust-region preview, and use the section assumptions to bound the change in objective value. If
the assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes trust-region preview visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about trust-region preview is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- K-FAC and natural-gradient style preconditioners for neural networks.
- L-BFGS for small-batch fine-tuning and classical ML objectives.
- Hessian-vector products for sharpness and interpretability diagnostics.
- structured preconditioning proposals such as Shampoo, SOAP, and Muon-family methods.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 8.5 Reproducibility and logging checklist

In this section, Gauss-Newton is treated as a concrete optimization object rather than a slogan.
The goal is to understand how it changes the objective, the update rule, the convergence story,
and the diagnostics a practitioner should inspect when training a modern model. For Second-Order
Methods, the phrase "Reproducibility and logging checklist" means a precise mathematical habit:
state the assumptions, write the update, identify what can be measured, and connect the result
to a real AI training decision.

> **Definition.**
>
> For this section, **Gauss-Newton** is the part of Second-Order Methods that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Gauss-Newton can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Gauss-Newton affects optimization but the model remains interpretable.
- A transformer training diagnostic where Gauss-Newton appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Gauss-Newton as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - H_f(\boldsymbol{\theta}_t)^{-1}\nabla f(\boldsymbol{\theta}_t)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
Gauss-Newton, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Gauss-Newton visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Gauss-Newton is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- K-FAC and natural-gradient style preconditioners for neural networks.
- L-BFGS for small-batch fine-tuning and classical ML objectives.
- Hessian-vector products for sharpness and interpretability diagnostics.
- structured preconditioning proposals such as Shampoo, SOAP, and Muon-family methods.

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

1. **Exercise 1 [*] - Newton Step**
   (a) Define Newton step using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

2. **Exercise 2 [*] - Damped Newton**
   (a) Define damped Newton using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

3. **Exercise 3 [*] - Trust-Region Preview**
   (a) Define trust-region preview using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

4. **Exercise 4 [**] - Levenberg-Marquardt**
   (a) Define Levenberg-Marquardt using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

5. **Exercise 5 [**] - Bfgs**
   (a) Define BFGS using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

6. **Exercise 6 [**] - Two-Loop Recursion**
   (a) Define two-loop recursion using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

7. **Exercise 7 [**] - Fisher Information**
   (a) Define Fisher information using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

8. **Exercise 8 [***] - K-Fac**
   (a) Define K-FAC using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

9. **Exercise 9 [***] - Soap**
   (a) Define SOAP using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

10. **Exercise 10 [***] - Curvature Diagnostics**
   (a) Define curvature diagnostics using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
H_f(\boldsymbol{\theta}_t)\mathbf{p}_t = -\nabla f(\boldsymbol{\theta}_t)
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

## 11. Why This Matters for AI (2026 Perspective)

| Concept | AI Impact |
| --- | --- |
| Hessian matrix | K-FAC and natural-gradient style preconditioners for neural networks |
| quadratic model | L-BFGS for small-batch fine-tuning and classical ML objectives |
| Newton step | Hessian-vector products for sharpness and interpretability diagnostics |
| Newton decrement | structured preconditioning proposals such as Shampoo, SOAP, and Muon-family methods |
| damped Newton | K-FAC and natural-gradient style preconditioners for neural networks |
| modified Cholesky | L-BFGS for small-batch fine-tuning and classical ML objectives |
| trust-region preview | Hessian-vector products for sharpness and interpretability diagnostics |
| Gauss-Newton | structured preconditioning proposals such as Shampoo, SOAP, and Muon-family methods |
| Levenberg-Marquardt | K-FAC and natural-gradient style preconditioners for neural networks |
| secant equation | L-BFGS for small-batch fine-tuning and classical ML objectives |

## 12. Conceptual Bridge

Second-Order Methods sits inside a chain. Earlier sections give the calculus, probability, and
linear algebra needed to write the objective and interpret the update. Later sections use this
material to reason about noisy gradients, adaptive state, regularization, tuning, schedules, and
finally information-theoretic losses.

Backward link: [Gradient Descent](../02-Gradient-Descent/notes.md) supplies the immediate
prerequisite vocabulary.

Forward link: [Constrained Optimization](../04-Constrained-Optimization/notes.md) uses this
section as a building block.

```text
+------------------------------------------------------------+
| Chapter 8: Optimization                                    |
|    01-Convex-Optimization          Convex Optimization    |
|    02-Gradient-Descent             Gradient Descent       |
| >> 03-Second-Order-Methods         Second-Order Methods   |
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
- Martens and Grosse, Optimizing Neural Networks with Kronecker-factored Approximate Curvature.
- Amari, Natural Gradient Works Efficiently in Learning.
- Gupta et al., Shampoo: Preconditioned Stochastic Tensor Optimization.
- Goodfellow, Bengio, and Courville, Deep Learning.
- Bottou, Curtis, and Nocedal, Optimization Methods for Large-Scale Machine Learning.
- PyTorch optimizer and scheduler documentation.
- Optax documentation for composable optimizer transformations.
