[Back to Chapter 8: Optimization](../README.md) | [Next: Gradient Descent](../02-Gradient-Descent/notes.md)

---

# Convex Optimization

> _"Convexity is the rare geometry where local reasoning becomes global truth."_

## Overview

Convex Optimization is part of the optimization spine of this curriculum. It explains how
mathematical assumptions become training behavior, and how training behavior becomes measurable
engineering evidence. The section is the canonical home for convex sets, convex functions,
smoothness, strong convexity, condition numbers, convex problem classes, and the first full view
of Lagrangian duality.

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

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Interactive derivations, numerical checks, and visual diagnostics for Convex Optimization. |
| [exercises.ipynb](exercises.ipynb) | Graded implementation and proof exercises for Convex Optimization. |

## Learning Objectives

- Define the canonical objects used in Convex Optimization with repository notation.
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
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 Why Convex Optimization matters for training systems](#11-why-convex-optimization-matters-for-training-systems)
  - [1.2 The optimization object: parameters, objective, algorithm, and diagnostic](#12-the-optimization-object-parameters-objective-algorithm-and-diagnostic)
  - [1.3 Historical arc from classical optimization to modern AI](#13-historical-arc-from-classical-optimization-to-modern-ai)
  - [1.4 What this section treats as canonical scope](#14-what-this-section-treats-as-canonical-scope)
  - [1.5 A first mental model for LLM training](#15-a-first-mental-model-for-llm-training)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Primary definition: convex sets](#21-primary-definition-convex-sets)
  - [2.2 Secondary definition: convex combinations](#22-secondary-definition-convex-combinations)
  - [2.3 Algorithmic object: convex functions](#23-algorithmic-object-convex-functions)
  - [2.4 Examples, non-examples, and boundary cases](#24-examples-nonexamples-and-boundary-cases)
  - [2.5 Notation, dimensions, and assumptions](#25-notation-dimensions-and-assumptions)
- [3. Core Theory I: Geometry and Guarantees](#3-core-theory-i-geometry-and-guarantees)
  - [3.1 Geometry of Jensen inequality](#31-geometry-of-jensen-inequality)
  - [3.2 Key inequality for first-order characterization](#32-key-inequality-for-firstorder-characterization)
  - [3.3 Role of second-order characterization](#33-role-of-secondorder-characterization)
  - [3.4 Proof template and what the proof actually buys](#34-proof-template-and-what-the-proof-actually-buys)
  - [3.5 Failure modes when assumptions are removed](#35-failure-modes-when-assumptions-are-removed)
- [4. Core Theory II: Algorithms and Dynamics](#4-core-theory-ii-algorithms-and-dynamics)
  - [4.1 Algorithmic update for smoothness](#41-algorithmic-update-for-smoothness)
  - [4.2 Stability role of strong convexity](#42-stability-role-of-strong-convexity)
  - [4.3 Rate or complexity controlled by condition number](#43-rate-or-complexity-controlled-by-condition-number)
  - [4.4 Diagnostic interpretation of the update path](#44-diagnostic-interpretation-of-the-update-path)
  - [4.5 Connection to the next section in the chapter](#45-connection-to-the-next-section-in-the-chapter)
- [5. Core Theory III: Practical Variants](#5-core-theory-iii-practical-variants)
  - [5.1 Variant built around convex problem classes](#51-variant-built-around-convex-problem-classes)
  - [5.2 Variant built around linear programs](#52-variant-built-around-linear-programs)
  - [5.3 Variant built around quadratic programs](#53-variant-built-around-quadratic-programs)
  - [5.4 Implementation constraints and numerical stability](#54-implementation-constraints-and-numerical-stability)
  - [5.5 What belongs here versus neighboring sections](#55-what-belongs-here-versus-neighboring-sections)
- [6. Advanced Topics](#6-advanced-topics)
  - [6.1 Advanced view of semidefinite programs](#61-advanced-view-of-semidefinite-programs)
  - [6.2 Advanced view of subgradients](#62-advanced-view-of-subgradients)
  - [6.3 Advanced view of proximal operators](#63-advanced-view-of-proximal-operators)
  - [6.4 Infinite-dimensional or large-scale interpretation](#64-infinitedimensional-or-largescale-interpretation)
  - [6.5 Open questions for frontier model training](#65-open-questions-for-frontier-model-training)
- [7. Applications in Machine Learning](#7-applications-in-machine-learning)
  - [7.1 logistic regression and softmax regression as convex baselines](#71-logistic-regression-and-softmax-regression-as-convex-baselines)
  - [7.2 support-vector machines through primal and dual convex programs](#72-supportvector-machines-through-primal-and-dual-convex-programs)
  - [7.3 nuclear-norm relaxations behind low-rank matrix recovery and LoRA intuition](#73-nuclearnorm-relaxations-behind-lowrank-matrix-recovery-and-lora-intuition)
  - [7.4 regularized empirical risk minimization with explicit certificates](#74-regularized-empirical-risk-minimization-with-explicit-certificates)
  - [7.5 Diagnostic checklist for real experiments](#75-diagnostic-checklist-for-real-experiments)
- [8. Implementation and Diagnostics](#8-implementation-and-diagnostics)
  - [8.1 Minimal NumPy experiment for Lagrangian duality](#81-minimal-numpy-experiment-for-lagrangian-duality)
  - [8.2 Monitoring signal for weak duality](#82-monitoring-signal-for-weak-duality)
  - [8.3 Failure signature for strong duality](#83-failure-signature-for-strong-duality)
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

This block develops intuition for Convex Optimization. It keeps the scope local to this section
while pointing forward when a neighboring topic owns the full treatment.

### 1.1 Why Convex Optimization matters for training systems

In this section, first-order characterization is treated as a concrete optimization object
rather than a slogan. The goal is to understand how it changes the objective, the update rule,
the convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Convex Optimization, the phrase "Why Convex Optimization matters for training
systems" means a precise mathematical habit: state the assumptions, write the update, identify
what can be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **first-order characterization** is the part of Convex Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where first-order characterization can be computed directly and compared with theory.
- A logistic-regression or softmax objective where first-order characterization affects optimization but the model remains interpretable.
- A transformer training diagnostic where first-order characterization appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating first-order characterization as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
f(\alpha \mathbf{x} + (1-\alpha)\mathbf{y}) \leq \alpha f(\mathbf{x}) + (1-\alpha)f(\mathbf{y})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
first-order characterization, and use the section assumptions to bound the change in objective
value. If the assumption is geometric, the proof turns a picture into an inequality. If the
assumption is stochastic, the proof takes conditional expectation before applying the bound. If
the assumption is algorithmic, the proof checks that the proposed update is a descent,
projection, or preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes first-order characterization visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about first-order characterization is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- logistic regression and softmax regression as convex baselines.
- support-vector machines through primal and dual convex programs.
- nuclear-norm relaxations behind low-rank matrix recovery and LoRA intuition.
- regularized empirical risk minimization with explicit certificates.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 1.2 The optimization object: parameters, objective, algorithm, and diagnostic

In this section, second-order characterization is treated as a concrete optimization object
rather than a slogan. The goal is to understand how it changes the objective, the update rule,
the convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Convex Optimization, the phrase "The optimization object: parameters, objective,
algorithm, and diagnostic" means a precise mathematical habit: state the assumptions, write the
update, identify what can be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **second-order characterization** is the part of Convex Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where second-order characterization can be computed directly and compared with theory.
- A logistic-regression or softmax objective where second-order characterization affects optimization but the model remains interpretable.
- A transformer training diagnostic where second-order characterization appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating second-order characterization as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
f(\alpha \mathbf{x} + (1-\alpha)\mathbf{y}) \leq \alpha f(\mathbf{x}) + (1-\alpha)f(\mathbf{y})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
second-order characterization, and use the section assumptions to bound the change in objective
value. If the assumption is geometric, the proof turns a picture into an inequality. If the
assumption is stochastic, the proof takes conditional expectation before applying the bound. If
the assumption is algorithmic, the proof checks that the proposed update is a descent,
projection, or preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes second-order characterization visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about second-order characterization is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- logistic regression and softmax regression as convex baselines.
- support-vector machines through primal and dual convex programs.
- nuclear-norm relaxations behind low-rank matrix recovery and LoRA intuition.
- regularized empirical risk minimization with explicit certificates.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 1.3 Historical arc from classical optimization to modern AI

In this section, smoothness is treated as a concrete optimization object rather than a slogan.
The goal is to understand how it changes the objective, the update rule, the convergence story,
and the diagnostics a practitioner should inspect when training a modern model. For Convex
Optimization, the phrase "Historical arc from classical optimization to modern AI" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **smoothness** is the part of Convex Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where smoothness can be computed directly and compared with theory.
- A logistic-regression or softmax objective where smoothness affects optimization but the model remains interpretable.
- A transformer training diagnostic where smoothness appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating smoothness as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
f(\alpha \mathbf{x} + (1-\alpha)\mathbf{y}) \leq \alpha f(\mathbf{x}) + (1-\alpha)f(\mathbf{y})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
smoothness, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes smoothness visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about smoothness is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- logistic regression and softmax regression as convex baselines.
- support-vector machines through primal and dual convex programs.
- nuclear-norm relaxations behind low-rank matrix recovery and LoRA intuition.
- regularized empirical risk minimization with explicit certificates.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 1.4 What this section treats as canonical scope

In this section, strong convexity is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Convex Optimization, the phrase "What this section treats as canonical scope" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **strong convexity** is the part of Convex Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where strong convexity can be computed directly and compared with theory.
- A logistic-regression or softmax objective where strong convexity affects optimization but the model remains interpretable.
- A transformer training diagnostic where strong convexity appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating strong convexity as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
f(\alpha \mathbf{x} + (1-\alpha)\mathbf{y}) \leq \alpha f(\mathbf{x}) + (1-\alpha)f(\mathbf{y})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving strong
convexity, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes strong convexity visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about strong convexity is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- logistic regression and softmax regression as convex baselines.
- support-vector machines through primal and dual convex programs.
- nuclear-norm relaxations behind low-rank matrix recovery and LoRA intuition.
- regularized empirical risk minimization with explicit certificates.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 1.5 A first mental model for LLM training

In this section, condition number is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Convex Optimization, the phrase "A first mental model for LLM training" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **condition number** is the part of Convex Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
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
f(\alpha \mathbf{x} + (1-\alpha)\mathbf{y}) \leq \alpha f(\mathbf{x}) + (1-\alpha)f(\mathbf{y})
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
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about condition number is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- logistic regression and softmax regression as convex baselines.
- support-vector machines through primal and dual convex programs.
- nuclear-norm relaxations behind low-rank matrix recovery and LoRA intuition.
- regularized empirical risk minimization with explicit certificates.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 2. Formal Definitions

This block develops formal definitions for Convex Optimization. It keeps the scope local to this
section while pointing forward when a neighboring topic owns the full treatment.

### 2.1 Primary definition: convex sets

In this section, strong convexity is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Convex Optimization, the phrase "Primary definition: convex sets" means a precise mathematical
habit: state the assumptions, write the update, identify what can be measured, and connect the
result to a real AI training decision.

> **Definition.**
>
> For this section, **strong convexity** is the part of Convex Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where strong convexity can be computed directly and compared with theory.
- A logistic-regression or softmax objective where strong convexity affects optimization but the model remains interpretable.
- A transformer training diagnostic where strong convexity appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating strong convexity as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
f(\alpha \mathbf{x} + (1-\alpha)\mathbf{y}) \leq \alpha f(\mathbf{x}) + (1-\alpha)f(\mathbf{y})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving strong
convexity, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes strong convexity visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about strong convexity is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- logistic regression and softmax regression as convex baselines.
- support-vector machines through primal and dual convex programs.
- nuclear-norm relaxations behind low-rank matrix recovery and LoRA intuition.
- regularized empirical risk minimization with explicit certificates.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 2.2 Secondary definition: convex combinations

In this section, condition number is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Convex Optimization, the phrase "Secondary definition: convex combinations" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **condition number** is the part of Convex Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
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
f(\alpha \mathbf{x} + (1-\alpha)\mathbf{y}) \leq \alpha f(\mathbf{x}) + (1-\alpha)f(\mathbf{y})
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
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about condition number is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- logistic regression and softmax regression as convex baselines.
- support-vector machines through primal and dual convex programs.
- nuclear-norm relaxations behind low-rank matrix recovery and LoRA intuition.
- regularized empirical risk minimization with explicit certificates.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 2.3 Algorithmic object: convex functions

In this section, convex problem classes is treated as a concrete optimization object rather than
a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Convex Optimization, the phrase "Algorithmic object: convex functions" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **convex problem classes** is the part of Convex Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where convex problem classes can be computed directly and compared with theory.
- A logistic-regression or softmax objective where convex problem classes affects optimization but the model remains interpretable.
- A transformer training diagnostic where convex problem classes appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating convex problem classes as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
f(\alpha \mathbf{x} + (1-\alpha)\mathbf{y}) \leq \alpha f(\mathbf{x}) + (1-\alpha)f(\mathbf{y})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving convex
problem classes, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes convex problem classes visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about convex problem classes is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- logistic regression and softmax regression as convex baselines.
- support-vector machines through primal and dual convex programs.
- nuclear-norm relaxations behind low-rank matrix recovery and LoRA intuition.
- regularized empirical risk minimization with explicit certificates.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 2.4 Examples, non-examples, and boundary cases

In this section, linear programs is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Convex Optimization, the phrase "Examples, non-examples, and boundary cases" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **linear programs** is the part of Convex Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where linear programs can be computed directly and compared with theory.
- A logistic-regression or softmax objective where linear programs affects optimization but the model remains interpretable.
- A transformer training diagnostic where linear programs appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating linear programs as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
f(\alpha \mathbf{x} + (1-\alpha)\mathbf{y}) \leq \alpha f(\mathbf{x}) + (1-\alpha)f(\mathbf{y})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving linear
programs, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes linear programs visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about linear programs is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- logistic regression and softmax regression as convex baselines.
- support-vector machines through primal and dual convex programs.
- nuclear-norm relaxations behind low-rank matrix recovery and LoRA intuition.
- regularized empirical risk minimization with explicit certificates.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 2.5 Notation, dimensions, and assumptions

In this section, quadratic programs is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Convex Optimization, the phrase "Notation, dimensions, and assumptions" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **quadratic programs** is the part of Convex Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where quadratic programs can be computed directly and compared with theory.
- A logistic-regression or softmax objective where quadratic programs affects optimization but the model remains interpretable.
- A transformer training diagnostic where quadratic programs appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating quadratic programs as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
f(\alpha \mathbf{x} + (1-\alpha)\mathbf{y}) \leq \alpha f(\mathbf{x}) + (1-\alpha)f(\mathbf{y})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving quadratic
programs, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes quadratic programs visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about quadratic programs is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- logistic regression and softmax regression as convex baselines.
- support-vector machines through primal and dual convex programs.
- nuclear-norm relaxations behind low-rank matrix recovery and LoRA intuition.
- regularized empirical risk minimization with explicit certificates.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 3. Core Theory I: Geometry and Guarantees

This block develops core theory i: geometry and guarantees for Convex Optimization. It keeps the
scope local to this section while pointing forward when a neighboring topic owns the full
treatment.

### 3.1 Geometry of Jensen inequality

In this section, linear programs is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Convex Optimization, the phrase "Geometry of Jensen inequality" means a precise mathematical
habit: state the assumptions, write the update, identify what can be measured, and connect the
result to a real AI training decision.

> **Definition.**
>
> For this section, **linear programs** is the part of Convex Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where linear programs can be computed directly and compared with theory.
- A logistic-regression or softmax objective where linear programs affects optimization but the model remains interpretable.
- A transformer training diagnostic where linear programs appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating linear programs as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
f(\alpha \mathbf{x} + (1-\alpha)\mathbf{y}) \leq \alpha f(\mathbf{x}) + (1-\alpha)f(\mathbf{y})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving linear
programs, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes linear programs visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about linear programs is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- logistic regression and softmax regression as convex baselines.
- support-vector machines through primal and dual convex programs.
- nuclear-norm relaxations behind low-rank matrix recovery and LoRA intuition.
- regularized empirical risk minimization with explicit certificates.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 3.2 Key inequality for first-order characterization

In this section, quadratic programs is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Convex Optimization, the phrase "Key inequality for first-order characterization" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **quadratic programs** is the part of Convex Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where quadratic programs can be computed directly and compared with theory.
- A logistic-regression or softmax objective where quadratic programs affects optimization but the model remains interpretable.
- A transformer training diagnostic where quadratic programs appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating quadratic programs as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
f(\alpha \mathbf{x} + (1-\alpha)\mathbf{y}) \leq \alpha f(\mathbf{x}) + (1-\alpha)f(\mathbf{y})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving quadratic
programs, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes quadratic programs visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about quadratic programs is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- logistic regression and softmax regression as convex baselines.
- support-vector machines through primal and dual convex programs.
- nuclear-norm relaxations behind low-rank matrix recovery and LoRA intuition.
- regularized empirical risk minimization with explicit certificates.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 3.3 Role of second-order characterization

In this section, semidefinite programs is treated as a concrete optimization object rather than
a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Convex Optimization, the phrase "Role of second-order characterization" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **semidefinite programs** is the part of Convex Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where semidefinite programs can be computed directly and compared with theory.
- A logistic-regression or softmax objective where semidefinite programs affects optimization but the model remains interpretable.
- A transformer training diagnostic where semidefinite programs appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating semidefinite programs as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
f(\alpha \mathbf{x} + (1-\alpha)\mathbf{y}) \leq \alpha f(\mathbf{x}) + (1-\alpha)f(\mathbf{y})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
semidefinite programs, and use the section assumptions to bound the change in objective value.
If the assumption is geometric, the proof turns a picture into an inequality. If the assumption
is stochastic, the proof takes conditional expectation before applying the bound. If the
assumption is algorithmic, the proof checks that the proposed update is a descent, projection,
or preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes semidefinite programs visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about semidefinite programs is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- logistic regression and softmax regression as convex baselines.
- support-vector machines through primal and dual convex programs.
- nuclear-norm relaxations behind low-rank matrix recovery and LoRA intuition.
- regularized empirical risk minimization with explicit certificates.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 3.4 Proof template and what the proof actually buys

In this section, subgradients is treated as a concrete optimization object rather than a slogan.
The goal is to understand how it changes the objective, the update rule, the convergence story,
and the diagnostics a practitioner should inspect when training a modern model. For Convex
Optimization, the phrase "Proof template and what the proof actually buys" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **subgradients** is the part of Convex Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where subgradients can be computed directly and compared with theory.
- A logistic-regression or softmax objective where subgradients affects optimization but the model remains interpretable.
- A transformer training diagnostic where subgradients appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating subgradients as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
f(\alpha \mathbf{x} + (1-\alpha)\mathbf{y}) \leq \alpha f(\mathbf{x}) + (1-\alpha)f(\mathbf{y})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
subgradients, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes subgradients visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about subgradients is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- logistic regression and softmax regression as convex baselines.
- support-vector machines through primal and dual convex programs.
- nuclear-norm relaxations behind low-rank matrix recovery and LoRA intuition.
- regularized empirical risk minimization with explicit certificates.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 3.5 Failure modes when assumptions are removed

In this section, proximal operators is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Convex Optimization, the phrase "Failure modes when assumptions are removed" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **proximal operators** is the part of Convex Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where proximal operators can be computed directly and compared with theory.
- A logistic-regression or softmax objective where proximal operators affects optimization but the model remains interpretable.
- A transformer training diagnostic where proximal operators appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating proximal operators as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
f(\alpha \mathbf{x} + (1-\alpha)\mathbf{y}) \leq \alpha f(\mathbf{x}) + (1-\alpha)f(\mathbf{y})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving proximal
operators, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes proximal operators visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about proximal operators is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- logistic regression and softmax regression as convex baselines.
- support-vector machines through primal and dual convex programs.
- nuclear-norm relaxations behind low-rank matrix recovery and LoRA intuition.
- regularized empirical risk minimization with explicit certificates.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 4. Core Theory II: Algorithms and Dynamics

This block develops core theory ii: algorithms and dynamics for Convex Optimization. It keeps
the scope local to this section while pointing forward when a neighboring topic owns the full
treatment.

### 4.1 Algorithmic update for smoothness

In this section, subgradients is treated as a concrete optimization object rather than a slogan.
The goal is to understand how it changes the objective, the update rule, the convergence story,
and the diagnostics a practitioner should inspect when training a modern model. For Convex
Optimization, the phrase "Algorithmic update for smoothness" means a precise mathematical habit:
state the assumptions, write the update, identify what can be measured, and connect the result
to a real AI training decision.

> **Definition.**
>
> For this section, **subgradients** is the part of Convex Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where subgradients can be computed directly and compared with theory.
- A logistic-regression or softmax objective where subgradients affects optimization but the model remains interpretable.
- A transformer training diagnostic where subgradients appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating subgradients as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
f(\alpha \mathbf{x} + (1-\alpha)\mathbf{y}) \leq \alpha f(\mathbf{x}) + (1-\alpha)f(\mathbf{y})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
subgradients, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes subgradients visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about subgradients is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- logistic regression and softmax regression as convex baselines.
- support-vector machines through primal and dual convex programs.
- nuclear-norm relaxations behind low-rank matrix recovery and LoRA intuition.
- regularized empirical risk minimization with explicit certificates.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 4.2 Stability role of strong convexity

In this section, proximal operators is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Convex Optimization, the phrase "Stability role of strong convexity" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **proximal operators** is the part of Convex Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where proximal operators can be computed directly and compared with theory.
- A logistic-regression or softmax objective where proximal operators affects optimization but the model remains interpretable.
- A transformer training diagnostic where proximal operators appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating proximal operators as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
f(\alpha \mathbf{x} + (1-\alpha)\mathbf{y}) \leq \alpha f(\mathbf{x}) + (1-\alpha)f(\mathbf{y})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving proximal
operators, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes proximal operators visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about proximal operators is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- logistic regression and softmax regression as convex baselines.
- support-vector machines through primal and dual convex programs.
- nuclear-norm relaxations behind low-rank matrix recovery and LoRA intuition.
- regularized empirical risk minimization with explicit certificates.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 4.3 Rate or complexity controlled by condition number

In this section, Lagrangian duality is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Convex Optimization, the phrase "Rate or complexity controlled by condition number" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **Lagrangian duality** is the part of Convex Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Lagrangian duality can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Lagrangian duality affects optimization but the model remains interpretable.
- A transformer training diagnostic where Lagrangian duality appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Lagrangian duality as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
f(\alpha \mathbf{x} + (1-\alpha)\mathbf{y}) \leq \alpha f(\mathbf{x}) + (1-\alpha)f(\mathbf{y})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving Lagrangian
duality, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Lagrangian duality visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Lagrangian duality is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- logistic regression and softmax regression as convex baselines.
- support-vector machines through primal and dual convex programs.
- nuclear-norm relaxations behind low-rank matrix recovery and LoRA intuition.
- regularized empirical risk minimization with explicit certificates.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 4.4 Diagnostic interpretation of the update path

In this section, weak duality is treated as a concrete optimization object rather than a slogan.
The goal is to understand how it changes the objective, the update rule, the convergence story,
and the diagnostics a practitioner should inspect when training a modern model. For Convex
Optimization, the phrase "Diagnostic interpretation of the update path" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **weak duality** is the part of Convex Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where weak duality can be computed directly and compared with theory.
- A logistic-regression or softmax objective where weak duality affects optimization but the model remains interpretable.
- A transformer training diagnostic where weak duality appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating weak duality as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
f(\alpha \mathbf{x} + (1-\alpha)\mathbf{y}) \leq \alpha f(\mathbf{x}) + (1-\alpha)f(\mathbf{y})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving weak
duality, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes weak duality visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about weak duality is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- logistic regression and softmax regression as convex baselines.
- support-vector machines through primal and dual convex programs.
- nuclear-norm relaxations behind low-rank matrix recovery and LoRA intuition.
- regularized empirical risk minimization with explicit certificates.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 4.5 Connection to the next section in the chapter

In this section, strong duality is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Convex Optimization, the phrase "Connection to the next section in the chapter" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **strong duality** is the part of Convex Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where strong duality can be computed directly and compared with theory.
- A logistic-regression or softmax objective where strong duality affects optimization but the model remains interpretable.
- A transformer training diagnostic where strong duality appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating strong duality as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
f(\alpha \mathbf{x} + (1-\alpha)\mathbf{y}) \leq \alpha f(\mathbf{x}) + (1-\alpha)f(\mathbf{y})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving strong
duality, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes strong duality visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about strong duality is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- logistic regression and softmax regression as convex baselines.
- support-vector machines through primal and dual convex programs.
- nuclear-norm relaxations behind low-rank matrix recovery and LoRA intuition.
- regularized empirical risk minimization with explicit certificates.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 5. Core Theory III: Practical Variants

This block develops core theory iii: practical variants for Convex Optimization. It keeps the
scope local to this section while pointing forward when a neighboring topic owns the full
treatment.

### 5.1 Variant built around convex problem classes

In this section, weak duality is treated as a concrete optimization object rather than a slogan.
The goal is to understand how it changes the objective, the update rule, the convergence story,
and the diagnostics a practitioner should inspect when training a modern model. For Convex
Optimization, the phrase "Variant built around convex problem classes" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **weak duality** is the part of Convex Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where weak duality can be computed directly and compared with theory.
- A logistic-regression or softmax objective where weak duality affects optimization but the model remains interpretable.
- A transformer training diagnostic where weak duality appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating weak duality as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
f(\alpha \mathbf{x} + (1-\alpha)\mathbf{y}) \leq \alpha f(\mathbf{x}) + (1-\alpha)f(\mathbf{y})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving weak
duality, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes weak duality visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about weak duality is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- logistic regression and softmax regression as convex baselines.
- support-vector machines through primal and dual convex programs.
- nuclear-norm relaxations behind low-rank matrix recovery and LoRA intuition.
- regularized empirical risk minimization with explicit certificates.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 5.2 Variant built around linear programs

In this section, strong duality is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Convex Optimization, the phrase "Variant built around linear programs" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **strong duality** is the part of Convex Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where strong duality can be computed directly and compared with theory.
- A logistic-regression or softmax objective where strong duality affects optimization but the model remains interpretable.
- A transformer training diagnostic where strong duality appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating strong duality as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
f(\alpha \mathbf{x} + (1-\alpha)\mathbf{y}) \leq \alpha f(\mathbf{x}) + (1-\alpha)f(\mathbf{y})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving strong
duality, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes strong duality visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about strong duality is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- logistic regression and softmax regression as convex baselines.
- support-vector machines through primal and dual convex programs.
- nuclear-norm relaxations behind low-rank matrix recovery and LoRA intuition.
- regularized empirical risk minimization with explicit certificates.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 5.3 Variant built around quadratic programs

In this section, Slater condition is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Convex Optimization, the phrase "Variant built around quadratic programs" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **Slater condition** is the part of Convex Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Slater condition can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Slater condition affects optimization but the model remains interpretable.
- A transformer training diagnostic where Slater condition appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Slater condition as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
f(\alpha \mathbf{x} + (1-\alpha)\mathbf{y}) \leq \alpha f(\mathbf{x}) + (1-\alpha)f(\mathbf{y})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving Slater
condition, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Slater condition visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Slater condition is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- logistic regression and softmax regression as convex baselines.
- support-vector machines through primal and dual convex programs.
- nuclear-norm relaxations behind low-rank matrix recovery and LoRA intuition.
- regularized empirical risk minimization with explicit certificates.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 5.4 Implementation constraints and numerical stability

In this section, optimality certificates is treated as a concrete optimization object rather
than a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Convex Optimization, the phrase "Implementation constraints and numerical stability"
means a precise mathematical habit: state the assumptions, write the update, identify what can
be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **optimality certificates** is the part of Convex Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where optimality certificates can be computed directly and compared with theory.
- A logistic-regression or softmax objective where optimality certificates affects optimization but the model remains interpretable.
- A transformer training diagnostic where optimality certificates appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating optimality certificates as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
f(\alpha \mathbf{x} + (1-\alpha)\mathbf{y}) \leq \alpha f(\mathbf{x}) + (1-\alpha)f(\mathbf{y})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving optimality
certificates, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes optimality certificates visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about optimality certificates is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- logistic regression and softmax regression as convex baselines.
- support-vector machines through primal and dual convex programs.
- nuclear-norm relaxations behind low-rank matrix recovery and LoRA intuition.
- regularized empirical risk minimization with explicit certificates.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 5.5 What belongs here versus neighboring sections

In this section, logistic regression convexity is treated as a concrete optimization object
rather than a slogan. The goal is to understand how it changes the objective, the update rule,
the convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Convex Optimization, the phrase "What belongs here versus neighboring sections" means
a precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **logistic regression convexity** is the part of Convex Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where logistic regression convexity can be computed directly and compared with theory.
- A logistic-regression or softmax objective where logistic regression convexity affects optimization but the model remains interpretable.
- A transformer training diagnostic where logistic regression convexity appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating logistic regression convexity as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
f(\alpha \mathbf{x} + (1-\alpha)\mathbf{y}) \leq \alpha f(\mathbf{x}) + (1-\alpha)f(\mathbf{y})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving logistic
regression convexity, and use the section assumptions to bound the change in objective value. If
the assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes logistic regression convexity visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about logistic regression convexity is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- logistic regression and softmax regression as convex baselines.
- support-vector machines through primal and dual convex programs.
- nuclear-norm relaxations behind low-rank matrix recovery and LoRA intuition.
- regularized empirical risk minimization with explicit certificates.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 6. Advanced Topics

This block develops advanced topics for Convex Optimization. It keeps the scope local to this
section while pointing forward when a neighboring topic owns the full treatment.

### 6.1 Advanced view of semidefinite programs

In this section, optimality certificates is treated as a concrete optimization object rather
than a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Convex Optimization, the phrase "Advanced view of semidefinite programs" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **optimality certificates** is the part of Convex Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where optimality certificates can be computed directly and compared with theory.
- A logistic-regression or softmax objective where optimality certificates affects optimization but the model remains interpretable.
- A transformer training diagnostic where optimality certificates appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating optimality certificates as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
f(\alpha \mathbf{x} + (1-\alpha)\mathbf{y}) \leq \alpha f(\mathbf{x}) + (1-\alpha)f(\mathbf{y})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving optimality
certificates, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes optimality certificates visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about optimality certificates is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- logistic regression and softmax regression as convex baselines.
- support-vector machines through primal and dual convex programs.
- nuclear-norm relaxations behind low-rank matrix recovery and LoRA intuition.
- regularized empirical risk minimization with explicit certificates.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 6.2 Advanced view of subgradients

In this section, logistic regression convexity is treated as a concrete optimization object
rather than a slogan. The goal is to understand how it changes the objective, the update rule,
the convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Convex Optimization, the phrase "Advanced view of subgradients" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **logistic regression convexity** is the part of Convex Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where logistic regression convexity can be computed directly and compared with theory.
- A logistic-regression or softmax objective where logistic regression convexity affects optimization but the model remains interpretable.
- A transformer training diagnostic where logistic regression convexity appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating logistic regression convexity as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
f(\alpha \mathbf{x} + (1-\alpha)\mathbf{y}) \leq \alpha f(\mathbf{x}) + (1-\alpha)f(\mathbf{y})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving logistic
regression convexity, and use the section assumptions to bound the change in objective value. If
the assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes logistic regression convexity visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about logistic regression convexity is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- logistic regression and softmax regression as convex baselines.
- support-vector machines through primal and dual convex programs.
- nuclear-norm relaxations behind low-rank matrix recovery and LoRA intuition.
- regularized empirical risk minimization with explicit certificates.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 6.3 Advanced view of proximal operators

In this section, nuclear norm relaxation is treated as a concrete optimization object rather
than a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Convex Optimization, the phrase "Advanced view of proximal operators" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **nuclear norm relaxation** is the part of Convex Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where nuclear norm relaxation can be computed directly and compared with theory.
- A logistic-regression or softmax objective where nuclear norm relaxation affects optimization but the model remains interpretable.
- A transformer training diagnostic where nuclear norm relaxation appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating nuclear norm relaxation as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
f(\alpha \mathbf{x} + (1-\alpha)\mathbf{y}) \leq \alpha f(\mathbf{x}) + (1-\alpha)f(\mathbf{y})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving nuclear
norm relaxation, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes nuclear norm relaxation visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about nuclear norm relaxation is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- logistic regression and softmax regression as convex baselines.
- support-vector machines through primal and dual convex programs.
- nuclear-norm relaxations behind low-rank matrix recovery and LoRA intuition.
- regularized empirical risk minimization with explicit certificates.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 6.4 Infinite-dimensional or large-scale interpretation

In this section, convex sets is treated as a concrete optimization object rather than a slogan.
The goal is to understand how it changes the objective, the update rule, the convergence story,
and the diagnostics a practitioner should inspect when training a modern model. For Convex
Optimization, the phrase "Infinite-dimensional or large-scale interpretation" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **convex sets** is the part of Convex Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where convex sets can be computed directly and compared with theory.
- A logistic-regression or softmax objective where convex sets affects optimization but the model remains interpretable.
- A transformer training diagnostic where convex sets appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating convex sets as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
f(\alpha \mathbf{x} + (1-\alpha)\mathbf{y}) \leq \alpha f(\mathbf{x}) + (1-\alpha)f(\mathbf{y})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving convex
sets, and use the section assumptions to bound the change in objective value. If the assumption
is geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes convex sets visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about convex sets is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- logistic regression and softmax regression as convex baselines.
- support-vector machines through primal and dual convex programs.
- nuclear-norm relaxations behind low-rank matrix recovery and LoRA intuition.
- regularized empirical risk minimization with explicit certificates.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 6.5 Open questions for frontier model training

In this section, convex combinations is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Convex Optimization, the phrase "Open questions for frontier model training" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **convex combinations** is the part of Convex Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where convex combinations can be computed directly and compared with theory.
- A logistic-regression or softmax objective where convex combinations affects optimization but the model remains interpretable.
- A transformer training diagnostic where convex combinations appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating convex combinations as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
f(\alpha \mathbf{x} + (1-\alpha)\mathbf{y}) \leq \alpha f(\mathbf{x}) + (1-\alpha)f(\mathbf{y})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving convex
combinations, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes convex combinations visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about convex combinations is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- logistic regression and softmax regression as convex baselines.
- support-vector machines through primal and dual convex programs.
- nuclear-norm relaxations behind low-rank matrix recovery and LoRA intuition.
- regularized empirical risk minimization with explicit certificates.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 7. Applications in Machine Learning

This block develops applications in machine learning for Convex Optimization. It keeps the scope
local to this section while pointing forward when a neighboring topic owns the full treatment.

### 7.1 logistic regression and softmax regression as convex baselines

In this section, convex sets is treated as a concrete optimization object rather than a slogan.
The goal is to understand how it changes the objective, the update rule, the convergence story,
and the diagnostics a practitioner should inspect when training a modern model. For Convex
Optimization, the phrase "logistic regression and softmax regression as convex baselines" means
a precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **convex sets** is the part of Convex Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where convex sets can be computed directly and compared with theory.
- A logistic-regression or softmax objective where convex sets affects optimization but the model remains interpretable.
- A transformer training diagnostic where convex sets appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating convex sets as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
f(\alpha \mathbf{x} + (1-\alpha)\mathbf{y}) \leq \alpha f(\mathbf{x}) + (1-\alpha)f(\mathbf{y})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving convex
sets, and use the section assumptions to bound the change in objective value. If the assumption
is geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes convex sets visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about convex sets is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- logistic regression and softmax regression as convex baselines.
- support-vector machines through primal and dual convex programs.
- nuclear-norm relaxations behind low-rank matrix recovery and LoRA intuition.
- regularized empirical risk minimization with explicit certificates.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 7.2 support-vector machines through primal and dual convex programs

In this section, convex combinations is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Convex Optimization, the phrase "support-vector machines through primal and dual convex
programs" means a precise mathematical habit: state the assumptions, write the update, identify
what can be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **convex combinations** is the part of Convex Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where convex combinations can be computed directly and compared with theory.
- A logistic-regression or softmax objective where convex combinations affects optimization but the model remains interpretable.
- A transformer training diagnostic where convex combinations appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating convex combinations as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
f(\alpha \mathbf{x} + (1-\alpha)\mathbf{y}) \leq \alpha f(\mathbf{x}) + (1-\alpha)f(\mathbf{y})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving convex
combinations, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes convex combinations visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about convex combinations is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- logistic regression and softmax regression as convex baselines.
- support-vector machines through primal and dual convex programs.
- nuclear-norm relaxations behind low-rank matrix recovery and LoRA intuition.
- regularized empirical risk minimization with explicit certificates.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 7.3 nuclear-norm relaxations behind low-rank matrix recovery and LoRA intuition

In this section, convex functions is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Convex Optimization, the phrase "nuclear-norm relaxations behind low-rank matrix recovery and
LoRA intuition" means a precise mathematical habit: state the assumptions, write the update,
identify what can be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **convex functions** is the part of Convex Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where convex functions can be computed directly and compared with theory.
- A logistic-regression or softmax objective where convex functions affects optimization but the model remains interpretable.
- A transformer training diagnostic where convex functions appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating convex functions as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
f(\alpha \mathbf{x} + (1-\alpha)\mathbf{y}) \leq \alpha f(\mathbf{x}) + (1-\alpha)f(\mathbf{y})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving convex
functions, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes convex functions visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about convex functions is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- logistic regression and softmax regression as convex baselines.
- support-vector machines through primal and dual convex programs.
- nuclear-norm relaxations behind low-rank matrix recovery and LoRA intuition.
- regularized empirical risk minimization with explicit certificates.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 7.4 regularized empirical risk minimization with explicit certificates

In this section, Jensen inequality is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Convex Optimization, the phrase "regularized empirical risk minimization with explicit
certificates" means a precise mathematical habit: state the assumptions, write the update,
identify what can be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **Jensen inequality** is the part of Convex Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Jensen inequality can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Jensen inequality affects optimization but the model remains interpretable.
- A transformer training diagnostic where Jensen inequality appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Jensen inequality as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
f(\alpha \mathbf{x} + (1-\alpha)\mathbf{y}) \leq \alpha f(\mathbf{x}) + (1-\alpha)f(\mathbf{y})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving Jensen
inequality, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Jensen inequality visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Jensen inequality is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- logistic regression and softmax regression as convex baselines.
- support-vector machines through primal and dual convex programs.
- nuclear-norm relaxations behind low-rank matrix recovery and LoRA intuition.
- regularized empirical risk minimization with explicit certificates.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 7.5 Diagnostic checklist for real experiments

In this section, first-order characterization is treated as a concrete optimization object
rather than a slogan. The goal is to understand how it changes the objective, the update rule,
the convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Convex Optimization, the phrase "Diagnostic checklist for real experiments" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **first-order characterization** is the part of Convex Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where first-order characterization can be computed directly and compared with theory.
- A logistic-regression or softmax objective where first-order characterization affects optimization but the model remains interpretable.
- A transformer training diagnostic where first-order characterization appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating first-order characterization as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
f(\alpha \mathbf{x} + (1-\alpha)\mathbf{y}) \leq \alpha f(\mathbf{x}) + (1-\alpha)f(\mathbf{y})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
first-order characterization, and use the section assumptions to bound the change in objective
value. If the assumption is geometric, the proof turns a picture into an inequality. If the
assumption is stochastic, the proof takes conditional expectation before applying the bound. If
the assumption is algorithmic, the proof checks that the proposed update is a descent,
projection, or preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes first-order characterization visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about first-order characterization is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- logistic regression and softmax regression as convex baselines.
- support-vector machines through primal and dual convex programs.
- nuclear-norm relaxations behind low-rank matrix recovery and LoRA intuition.
- regularized empirical risk minimization with explicit certificates.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 8. Implementation and Diagnostics

This block develops implementation and diagnostics for Convex Optimization. It keeps the scope
local to this section while pointing forward when a neighboring topic owns the full treatment.

### 8.1 Minimal NumPy experiment for Lagrangian duality

In this section, Jensen inequality is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Convex Optimization, the phrase "Minimal NumPy experiment for Lagrangian duality" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **Jensen inequality** is the part of Convex Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Jensen inequality can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Jensen inequality affects optimization but the model remains interpretable.
- A transformer training diagnostic where Jensen inequality appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Jensen inequality as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
f(\alpha \mathbf{x} + (1-\alpha)\mathbf{y}) \leq \alpha f(\mathbf{x}) + (1-\alpha)f(\mathbf{y})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving Jensen
inequality, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Jensen inequality visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Jensen inequality is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- logistic regression and softmax regression as convex baselines.
- support-vector machines through primal and dual convex programs.
- nuclear-norm relaxations behind low-rank matrix recovery and LoRA intuition.
- regularized empirical risk minimization with explicit certificates.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 8.2 Monitoring signal for weak duality

In this section, first-order characterization is treated as a concrete optimization object
rather than a slogan. The goal is to understand how it changes the objective, the update rule,
the convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Convex Optimization, the phrase "Monitoring signal for weak duality" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **first-order characterization** is the part of Convex Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where first-order characterization can be computed directly and compared with theory.
- A logistic-regression or softmax objective where first-order characterization affects optimization but the model remains interpretable.
- A transformer training diagnostic where first-order characterization appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating first-order characterization as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
f(\alpha \mathbf{x} + (1-\alpha)\mathbf{y}) \leq \alpha f(\mathbf{x}) + (1-\alpha)f(\mathbf{y})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
first-order characterization, and use the section assumptions to bound the change in objective
value. If the assumption is geometric, the proof turns a picture into an inequality. If the
assumption is stochastic, the proof takes conditional expectation before applying the bound. If
the assumption is algorithmic, the proof checks that the proposed update is a descent,
projection, or preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes first-order characterization visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about first-order characterization is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- logistic regression and softmax regression as convex baselines.
- support-vector machines through primal and dual convex programs.
- nuclear-norm relaxations behind low-rank matrix recovery and LoRA intuition.
- regularized empirical risk minimization with explicit certificates.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 8.3 Failure signature for strong duality

In this section, second-order characterization is treated as a concrete optimization object
rather than a slogan. The goal is to understand how it changes the objective, the update rule,
the convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Convex Optimization, the phrase "Failure signature for strong duality" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **second-order characterization** is the part of Convex Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where second-order characterization can be computed directly and compared with theory.
- A logistic-regression or softmax objective where second-order characterization affects optimization but the model remains interpretable.
- A transformer training diagnostic where second-order characterization appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating second-order characterization as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
f(\alpha \mathbf{x} + (1-\alpha)\mathbf{y}) \leq \alpha f(\mathbf{x}) + (1-\alpha)f(\mathbf{y})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
second-order characterization, and use the section assumptions to bound the change in objective
value. If the assumption is geometric, the proof turns a picture into an inequality. If the
assumption is stochastic, the proof takes conditional expectation before applying the bound. If
the assumption is algorithmic, the proof checks that the proposed update is a descent,
projection, or preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes second-order characterization visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about second-order characterization is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- logistic regression and softmax regression as convex baselines.
- support-vector machines through primal and dual convex programs.
- nuclear-norm relaxations behind low-rank matrix recovery and LoRA intuition.
- regularized empirical risk minimization with explicit certificates.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 8.4 Framework-level implementation pattern

In this section, smoothness is treated as a concrete optimization object rather than a slogan.
The goal is to understand how it changes the objective, the update rule, the convergence story,
and the diagnostics a practitioner should inspect when training a modern model. For Convex
Optimization, the phrase "Framework-level implementation pattern" means a precise mathematical
habit: state the assumptions, write the update, identify what can be measured, and connect the
result to a real AI training decision.

> **Definition.**
>
> For this section, **smoothness** is the part of Convex Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where smoothness can be computed directly and compared with theory.
- A logistic-regression or softmax objective where smoothness affects optimization but the model remains interpretable.
- A transformer training diagnostic where smoothness appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating smoothness as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
f(\alpha \mathbf{x} + (1-\alpha)\mathbf{y}) \leq \alpha f(\mathbf{x}) + (1-\alpha)f(\mathbf{y})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
smoothness, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes smoothness visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about smoothness is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- logistic regression and softmax regression as convex baselines.
- support-vector machines through primal and dual convex programs.
- nuclear-norm relaxations behind low-rank matrix recovery and LoRA intuition.
- regularized empirical risk minimization with explicit certificates.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 8.5 Reproducibility and logging checklist

In this section, strong convexity is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Convex Optimization, the phrase "Reproducibility and logging checklist" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **strong convexity** is the part of Convex Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where strong convexity can be computed directly and compared with theory.
- A logistic-regression or softmax objective where strong convexity affects optimization but the model remains interpretable.
- A transformer training diagnostic where strong convexity appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating strong convexity as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
f(\alpha \mathbf{x} + (1-\alpha)\mathbf{y}) \leq \alpha f(\mathbf{x}) + (1-\alpha)f(\mathbf{y})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving strong
convexity, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes strong convexity visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about strong convexity is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- logistic regression and softmax regression as convex baselines.
- support-vector machines through primal and dual convex programs.
- nuclear-norm relaxations behind low-rank matrix recovery and LoRA intuition.
- regularized empirical risk minimization with explicit certificates.

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

1. **Exercise 1 [*] - Convex Functions**
   (a) Define convex functions using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

2. **Exercise 2 [*] - First-Order Characterization**
   (a) Define first-order characterization using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

3. **Exercise 3 [*] - Smoothness**
   (a) Define smoothness using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

4. **Exercise 4 [**] - Condition Number**
   (a) Define condition number using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

5. **Exercise 5 [**] - Linear Programs**
   (a) Define linear programs using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

6. **Exercise 6 [**] - Semidefinite Programs**
   (a) Define semidefinite programs using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

7. **Exercise 7 [**] - Proximal Operators**
   (a) Define proximal operators using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

8. **Exercise 8 [***] - Weak Duality**
   (a) Define weak duality using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

9. **Exercise 9 [***] - Slater Condition**
   (a) Define Slater condition using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

10. **Exercise 10 [***] - Logistic Regression Convexity**
   (a) Define logistic regression convexity using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\mathbf{x}^* \in \arg\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

## 11. Why This Matters for AI (2026 Perspective)

| Concept | AI Impact |
| --- | --- |
| convex sets | logistic regression and softmax regression as convex baselines |
| convex combinations | support-vector machines through primal and dual convex programs |
| convex functions | nuclear-norm relaxations behind low-rank matrix recovery and LoRA intuition |
| Jensen inequality | regularized empirical risk minimization with explicit certificates |
| first-order characterization | logistic regression and softmax regression as convex baselines |
| second-order characterization | support-vector machines through primal and dual convex programs |
| smoothness | nuclear-norm relaxations behind low-rank matrix recovery and LoRA intuition |
| strong convexity | regularized empirical risk minimization with explicit certificates |
| condition number | logistic regression and softmax regression as convex baselines |
| convex problem classes | support-vector machines through primal and dual convex programs |

## 12. Conceptual Bridge

Convex Optimization sits inside a chain. Earlier sections give the calculus, probability, and
linear algebra needed to write the objective and interpret the update. Later sections use this
material to reason about noisy gradients, adaptive state, regularization, tuning, schedules, and
finally information-theoretic losses.

Backward link: Chapters 5-7 supply gradients, Hessians, probability, estimation, and empirical
risk.

Forward link: [Gradient Descent](../02-Gradient-Descent/notes.md) uses this section as a
building block.

```text
+------------------------------------------------------------+
| Chapter 8: Optimization                                    |
| >> 01-Convex-Optimization          Convex Optimization    |
|    02-Gradient-Descent             Gradient Descent       |
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

- Boyd and Vandenberghe, Convex Optimization.
- Nesterov, Introductory Lectures on Convex Optimization.
- Shalev-Shwartz and Ben-David, Understanding Machine Learning.
- Bubeck, Convex Optimization: Algorithms and Complexity.
- Goodfellow, Bengio, and Courville, Deep Learning.
- Bottou, Curtis, and Nocedal, Optimization Methods for Large-Scale Machine Learning.
- PyTorch optimizer and scheduler documentation.
- Optax documentation for composable optimizer transformations.
