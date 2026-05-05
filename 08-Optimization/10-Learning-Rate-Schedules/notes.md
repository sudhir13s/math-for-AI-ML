[Previous: Hyperparameter Optimization](../09-Hyperparameter-Optimization/notes.md) | [Back to Chapter 8: Optimization](../README.md) | [Next Chapter: Information Theory](../../09-Information-Theory/01-Entropy/notes.md)

---

# Learning Rate Schedules

> _"The learning rate is not just a knob; it is the tempo of training."_

## Overview

Learning Rate Schedules is part of the optimization spine of this curriculum. It explains how
mathematical assumptions become training behavior, and how training behavior becomes measurable
engineering evidence. The section is the canonical home for time-varying learning rates:
constant, step, exponential, polynomial, warmup, cosine, cyclic, one-cycle, WSD, cooldown, and
batch-size coupling.

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
- The previous optimization section, [Hyperparameter Optimization](../09-Hyperparameter-Optimization/notes.md), is assumed as local context.

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Interactive derivations, numerical checks, and visual diagnostics for Learning Rate Schedules. |
| [exercises.ipynb](exercises.ipynb) | Graded implementation and proof exercises for Learning Rate Schedules. |

## Learning Objectives

- Define the canonical objects used in Learning Rate Schedules with repository notation.
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
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 Why Learning Rate Schedules matters for training systems](#11-why-learning-rate-schedules-matters-for-training-systems)
  - [1.2 The optimization object: parameters, objective, algorithm, and diagnostic](#12-the-optimization-object-parameters-objective-algorithm-and-diagnostic)
  - [1.3 Historical arc from classical optimization to modern AI](#13-historical-arc-from-classical-optimization-to-modern-ai)
  - [1.4 What this section treats as canonical scope](#14-what-this-section-treats-as-canonical-scope)
  - [1.5 A first mental model for LLM training](#15-a-first-mental-model-for-llm-training)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Primary definition: schedule function](#21-primary-definition-schedule-function)
  - [2.2 Secondary definition: constant learning rate](#22-secondary-definition-constant-learning-rate)
  - [2.3 Algorithmic object: step decay](#23-algorithmic-object-step-decay)
  - [2.4 Examples, non-examples, and boundary cases](#24-examples-nonexamples-and-boundary-cases)
  - [2.5 Notation, dimensions, and assumptions](#25-notation-dimensions-and-assumptions)
- [3. Core Theory I: Geometry and Guarantees](#3-core-theory-i-geometry-and-guarantees)
  - [3.1 Geometry of exponential decay](#31-geometry-of-exponential-decay)
  - [3.2 Key inequality for polynomial decay](#32-key-inequality-for-polynomial-decay)
  - [3.3 Role of linear warmup](#33-role-of-linear-warmup)
  - [3.4 Proof template and what the proof actually buys](#34-proof-template-and-what-the-proof-actually-buys)
  - [3.5 Failure modes when assumptions are removed](#35-failure-modes-when-assumptions-are-removed)
- [4. Core Theory II: Algorithms and Dynamics](#4-core-theory-ii-algorithms-and-dynamics)
  - [4.1 Algorithmic update for warmup ratio](#41-algorithmic-update-for-warmup-ratio)
  - [4.2 Stability role of cosine annealing](#42-stability-role-of-cosine-annealing)
  - [4.3 Rate or complexity controlled by cosine with restarts](#43-rate-or-complexity-controlled-by-cosine-with-restarts)
  - [4.4 Diagnostic interpretation of the update path](#44-diagnostic-interpretation-of-the-update-path)
  - [4.5 Connection to the next section in the chapter](#45-connection-to-the-next-section-in-the-chapter)
- [5. Core Theory III: Practical Variants](#5-core-theory-iii-practical-variants)
  - [5.1 Variant built around cyclic learning rate](#51-variant-built-around-cyclic-learning-rate)
  - [5.2 Variant built around one-cycle policy](#52-variant-built-around-onecycle-policy)
  - [5.3 Variant built around linear decay](#53-variant-built-around-linear-decay)
  - [5.4 Implementation constraints and numerical stability](#54-implementation-constraints-and-numerical-stability)
  - [5.5 What belongs here versus neighboring sections](#55-what-belongs-here-versus-neighboring-sections)
- [6. Advanced Topics](#6-advanced-topics)
  - [6.1 Advanced view of inverse-square-root decay](#61-advanced-view-of-inversesquareroot-decay)
  - [6.2 Advanced view of WSD schedule](#62-advanced-view-of-wsd-schedule)
  - [6.3 Advanced view of cooldown](#63-advanced-view-of-cooldown)
  - [6.4 Infinite-dimensional or large-scale interpretation](#64-infinitedimensional-or-largescale-interpretation)
  - [6.5 Open questions for frontier model training](#65-open-questions-for-frontier-model-training)
- [7. Applications in Machine Learning](#7-applications-in-machine-learning)
  - [7.1 linear warmup plus cosine decay for transformer pretraining](#71-linear-warmup-plus-cosine-decay-for-transformer-pretraining)
  - [7.2 warmup-stable-decay schedules for long LLM runs](#72-warmupstabledecay-schedules-for-long-llm-runs)
  - [7.3 one-cycle schedules for fast supervised training](#73-onecycle-schedules-for-fast-supervised-training)
  - [7.4 batch-size and gradient-accumulation coupling in distributed training](#74-batchsize-and-gradientaccumulation-coupling-in-distributed-training)
  - [7.5 Diagnostic checklist for real experiments](#75-diagnostic-checklist-for-real-experiments)
- [8. Implementation and Diagnostics](#8-implementation-and-diagnostics)
  - [8.1 Minimal NumPy experiment for learning-rate rewinding](#81-minimal-numpy-experiment-for-learningrate-rewinding)
  - [8.2 Monitoring signal for batch-size scaling](#82-monitoring-signal-for-batchsize-scaling)
  - [8.3 Failure signature for gradient accumulation coupling](#83-failure-signature-for-gradient-accumulation-coupling)
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

This block develops intuition for Learning Rate Schedules. It keeps the scope local to this
section while pointing forward when a neighboring topic owns the full treatment.

### 1.1 Why Learning Rate Schedules matters for training systems

In this section, polynomial decay is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Learning Rate Schedules, the phrase "Why Learning Rate Schedules matters for training systems"
means a precise mathematical habit: state the assumptions, write the update, identify what can
be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **polynomial decay** is the part of Learning Rate Schedules that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where polynomial decay can be computed directly and compared with theory.
- A logistic-regression or softmax objective where polynomial decay affects optimization but the model remains interpretable.
- A transformer training diagnostic where polynomial decay appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating polynomial decay as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving polynomial
decay, and use the section assumptions to bound the change in objective value. If the assumption
is geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes polynomial decay visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about polynomial decay is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- linear warmup plus cosine decay for transformer pretraining.
- warmup-stable-decay schedules for long LLM runs.
- one-cycle schedules for fast supervised training.
- batch-size and gradient-accumulation coupling in distributed training.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 1.2 The optimization object: parameters, objective, algorithm, and diagnostic

In this section, linear warmup is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Learning Rate Schedules, the phrase "The optimization object: parameters, objective, algorithm,
and diagnostic" means a precise mathematical habit: state the assumptions, write the update,
identify what can be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **linear warmup** is the part of Learning Rate Schedules that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where linear warmup can be computed directly and compared with theory.
- A logistic-regression or softmax objective where linear warmup affects optimization but the model remains interpretable.
- A transformer training diagnostic where linear warmup appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating linear warmup as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving linear
warmup, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes linear warmup visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about linear warmup is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- linear warmup plus cosine decay for transformer pretraining.
- warmup-stable-decay schedules for long LLM runs.
- one-cycle schedules for fast supervised training.
- batch-size and gradient-accumulation coupling in distributed training.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 1.3 Historical arc from classical optimization to modern AI

In this section, warmup ratio is treated as a concrete optimization object rather than a slogan.
The goal is to understand how it changes the objective, the update rule, the convergence story,
and the diagnostics a practitioner should inspect when training a modern model. For Learning
Rate Schedules, the phrase "Historical arc from classical optimization to modern AI" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **warmup ratio** is the part of Learning Rate Schedules that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where warmup ratio can be computed directly and compared with theory.
- A logistic-regression or softmax objective where warmup ratio affects optimization but the model remains interpretable.
- A transformer training diagnostic where warmup ratio appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating warmup ratio as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving warmup
ratio, and use the section assumptions to bound the change in objective value. If the assumption
is geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes warmup ratio visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about warmup ratio is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- linear warmup plus cosine decay for transformer pretraining.
- warmup-stable-decay schedules for long LLM runs.
- one-cycle schedules for fast supervised training.
- batch-size and gradient-accumulation coupling in distributed training.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 1.4 What this section treats as canonical scope

In this section, cosine annealing is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Learning Rate Schedules, the phrase "What this section treats as canonical scope" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **cosine annealing** is the part of Learning Rate Schedules that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where cosine annealing can be computed directly and compared with theory.
- A logistic-regression or softmax objective where cosine annealing affects optimization but the model remains interpretable.
- A transformer training diagnostic where cosine annealing appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating cosine annealing as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving cosine
annealing, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes cosine annealing visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about cosine annealing is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- linear warmup plus cosine decay for transformer pretraining.
- warmup-stable-decay schedules for long LLM runs.
- one-cycle schedules for fast supervised training.
- batch-size and gradient-accumulation coupling in distributed training.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 1.5 A first mental model for LLM training

In this section, cosine with restarts is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Learning Rate Schedules, the phrase "A first mental model for LLM training" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **cosine with restarts** is the part of Learning Rate Schedules that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where cosine with restarts can be computed directly and compared with theory.
- A logistic-regression or softmax objective where cosine with restarts affects optimization but the model remains interpretable.
- A transformer training diagnostic where cosine with restarts appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating cosine with restarts as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving cosine
with restarts, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes cosine with restarts visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about cosine with restarts is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- linear warmup plus cosine decay for transformer pretraining.
- warmup-stable-decay schedules for long LLM runs.
- one-cycle schedules for fast supervised training.
- batch-size and gradient-accumulation coupling in distributed training.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 2. Formal Definitions

This block develops formal definitions for Learning Rate Schedules. It keeps the scope local to
this section while pointing forward when a neighboring topic owns the full treatment.

### 2.1 Primary definition: schedule function

In this section, cosine annealing is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Learning Rate Schedules, the phrase "Primary definition: schedule function" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **cosine annealing** is the part of Learning Rate Schedules that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where cosine annealing can be computed directly and compared with theory.
- A logistic-regression or softmax objective where cosine annealing affects optimization but the model remains interpretable.
- A transformer training diagnostic where cosine annealing appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating cosine annealing as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving cosine
annealing, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes cosine annealing visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about cosine annealing is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- linear warmup plus cosine decay for transformer pretraining.
- warmup-stable-decay schedules for long LLM runs.
- one-cycle schedules for fast supervised training.
- batch-size and gradient-accumulation coupling in distributed training.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 2.2 Secondary definition: constant learning rate

In this section, cosine with restarts is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Learning Rate Schedules, the phrase "Secondary definition: constant learning rate" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **cosine with restarts** is the part of Learning Rate Schedules that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where cosine with restarts can be computed directly and compared with theory.
- A logistic-regression or softmax objective where cosine with restarts affects optimization but the model remains interpretable.
- A transformer training diagnostic where cosine with restarts appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating cosine with restarts as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving cosine
with restarts, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes cosine with restarts visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about cosine with restarts is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- linear warmup plus cosine decay for transformer pretraining.
- warmup-stable-decay schedules for long LLM runs.
- one-cycle schedules for fast supervised training.
- batch-size and gradient-accumulation coupling in distributed training.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 2.3 Algorithmic object: step decay

In this section, cyclic learning rate is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Learning Rate Schedules, the phrase "Algorithmic object: step decay" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **cyclic learning rate** is the part of Learning Rate Schedules that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where cyclic learning rate can be computed directly and compared with theory.
- A logistic-regression or softmax objective where cyclic learning rate affects optimization but the model remains interpretable.
- A transformer training diagnostic where cyclic learning rate appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating cyclic learning rate as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving cyclic
learning rate, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes cyclic learning rate visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about cyclic learning rate is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- linear warmup plus cosine decay for transformer pretraining.
- warmup-stable-decay schedules for long LLM runs.
- one-cycle schedules for fast supervised training.
- batch-size and gradient-accumulation coupling in distributed training.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 2.4 Examples, non-examples, and boundary cases

In this section, one-cycle policy is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Learning Rate Schedules, the phrase "Examples, non-examples, and boundary cases" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **one-cycle policy** is the part of Learning Rate Schedules that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where one-cycle policy can be computed directly and compared with theory.
- A logistic-regression or softmax objective where one-cycle policy affects optimization but the model remains interpretable.
- A transformer training diagnostic where one-cycle policy appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating one-cycle policy as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving one-cycle
policy, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes one-cycle policy visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about one-cycle policy is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- linear warmup plus cosine decay for transformer pretraining.
- warmup-stable-decay schedules for long LLM runs.
- one-cycle schedules for fast supervised training.
- batch-size and gradient-accumulation coupling in distributed training.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 2.5 Notation, dimensions, and assumptions

In this section, linear decay is treated as a concrete optimization object rather than a slogan.
The goal is to understand how it changes the objective, the update rule, the convergence story,
and the diagnostics a practitioner should inspect when training a modern model. For Learning
Rate Schedules, the phrase "Notation, dimensions, and assumptions" means a precise mathematical
habit: state the assumptions, write the update, identify what can be measured, and connect the
result to a real AI training decision.

> **Definition.**
>
> For this section, **linear decay** is the part of Learning Rate Schedules that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where linear decay can be computed directly and compared with theory.
- A logistic-regression or softmax objective where linear decay affects optimization but the model remains interpretable.
- A transformer training diagnostic where linear decay appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating linear decay as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving linear
decay, and use the section assumptions to bound the change in objective value. If the assumption
is geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes linear decay visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about linear decay is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- linear warmup plus cosine decay for transformer pretraining.
- warmup-stable-decay schedules for long LLM runs.
- one-cycle schedules for fast supervised training.
- batch-size and gradient-accumulation coupling in distributed training.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 3. Core Theory I: Geometry and Guarantees

This block develops core theory i: geometry and guarantees for Learning Rate Schedules. It keeps
the scope local to this section while pointing forward when a neighboring topic owns the full
treatment.

### 3.1 Geometry of exponential decay

In this section, one-cycle policy is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Learning Rate Schedules, the phrase "Geometry of exponential decay" means a precise mathematical
habit: state the assumptions, write the update, identify what can be measured, and connect the
result to a real AI training decision.

> **Definition.**
>
> For this section, **one-cycle policy** is the part of Learning Rate Schedules that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where one-cycle policy can be computed directly and compared with theory.
- A logistic-regression or softmax objective where one-cycle policy affects optimization but the model remains interpretable.
- A transformer training diagnostic where one-cycle policy appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating one-cycle policy as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving one-cycle
policy, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes one-cycle policy visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about one-cycle policy is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- linear warmup plus cosine decay for transformer pretraining.
- warmup-stable-decay schedules for long LLM runs.
- one-cycle schedules for fast supervised training.
- batch-size and gradient-accumulation coupling in distributed training.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 3.2 Key inequality for polynomial decay

In this section, linear decay is treated as a concrete optimization object rather than a slogan.
The goal is to understand how it changes the objective, the update rule, the convergence story,
and the diagnostics a practitioner should inspect when training a modern model. For Learning
Rate Schedules, the phrase "Key inequality for polynomial decay" means a precise mathematical
habit: state the assumptions, write the update, identify what can be measured, and connect the
result to a real AI training decision.

> **Definition.**
>
> For this section, **linear decay** is the part of Learning Rate Schedules that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where linear decay can be computed directly and compared with theory.
- A logistic-regression or softmax objective where linear decay affects optimization but the model remains interpretable.
- A transformer training diagnostic where linear decay appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating linear decay as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving linear
decay, and use the section assumptions to bound the change in objective value. If the assumption
is geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes linear decay visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about linear decay is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- linear warmup plus cosine decay for transformer pretraining.
- warmup-stable-decay schedules for long LLM runs.
- one-cycle schedules for fast supervised training.
- batch-size and gradient-accumulation coupling in distributed training.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 3.3 Role of linear warmup

In this section, inverse-square-root decay is treated as a concrete optimization object rather
than a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Learning Rate Schedules, the phrase "Role of linear warmup" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **inverse-square-root decay** is the part of Learning Rate Schedules that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where inverse-square-root decay can be computed directly and compared with theory.
- A logistic-regression or softmax objective where inverse-square-root decay affects optimization but the model remains interpretable.
- A transformer training diagnostic where inverse-square-root decay appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating inverse-square-root decay as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
inverse-square-root decay, and use the section assumptions to bound the change in objective
value. If the assumption is geometric, the proof turns a picture into an inequality. If the
assumption is stochastic, the proof takes conditional expectation before applying the bound. If
the assumption is algorithmic, the proof checks that the proposed update is a descent,
projection, or preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes inverse-square-root decay visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about inverse-square-root decay is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- linear warmup plus cosine decay for transformer pretraining.
- warmup-stable-decay schedules for long LLM runs.
- one-cycle schedules for fast supervised training.
- batch-size and gradient-accumulation coupling in distributed training.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 3.4 Proof template and what the proof actually buys

In this section, WSD schedule is treated as a concrete optimization object rather than a slogan.
The goal is to understand how it changes the objective, the update rule, the convergence story,
and the diagnostics a practitioner should inspect when training a modern model. For Learning
Rate Schedules, the phrase "Proof template and what the proof actually buys" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **WSD schedule** is the part of Learning Rate Schedules that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where WSD schedule can be computed directly and compared with theory.
- A logistic-regression or softmax objective where WSD schedule affects optimization but the model remains interpretable.
- A transformer training diagnostic where WSD schedule appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating WSD schedule as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving WSD
schedule, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes WSD schedule visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about WSD schedule is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- linear warmup plus cosine decay for transformer pretraining.
- warmup-stable-decay schedules for long LLM runs.
- one-cycle schedules for fast supervised training.
- batch-size and gradient-accumulation coupling in distributed training.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 3.5 Failure modes when assumptions are removed

In this section, cooldown is treated as a concrete optimization object rather than a slogan. The
goal is to understand how it changes the objective, the update rule, the convergence story, and
the diagnostics a practitioner should inspect when training a modern model. For Learning Rate
Schedules, the phrase "Failure modes when assumptions are removed" means a precise mathematical
habit: state the assumptions, write the update, identify what can be measured, and connect the
result to a real AI training decision.

> **Definition.**
>
> For this section, **cooldown** is the part of Learning Rate Schedules that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where cooldown can be computed directly and compared with theory.
- A logistic-regression or softmax objective where cooldown affects optimization but the model remains interpretable.
- A transformer training diagnostic where cooldown appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating cooldown as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving cooldown,
and use the section assumptions to bound the change in objective value. If the assumption is
geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes cooldown visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about cooldown is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- linear warmup plus cosine decay for transformer pretraining.
- warmup-stable-decay schedules for long LLM runs.
- one-cycle schedules for fast supervised training.
- batch-size and gradient-accumulation coupling in distributed training.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 4. Core Theory II: Algorithms and Dynamics

This block develops core theory ii: algorithms and dynamics for Learning Rate Schedules. It
keeps the scope local to this section while pointing forward when a neighboring topic owns the
full treatment.

### 4.1 Algorithmic update for warmup ratio

In this section, WSD schedule is treated as a concrete optimization object rather than a slogan.
The goal is to understand how it changes the objective, the update rule, the convergence story,
and the diagnostics a practitioner should inspect when training a modern model. For Learning
Rate Schedules, the phrase "Algorithmic update for warmup ratio" means a precise mathematical
habit: state the assumptions, write the update, identify what can be measured, and connect the
result to a real AI training decision.

> **Definition.**
>
> For this section, **WSD schedule** is the part of Learning Rate Schedules that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where WSD schedule can be computed directly and compared with theory.
- A logistic-regression or softmax objective where WSD schedule affects optimization but the model remains interpretable.
- A transformer training diagnostic where WSD schedule appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating WSD schedule as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving WSD
schedule, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes WSD schedule visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about WSD schedule is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- linear warmup plus cosine decay for transformer pretraining.
- warmup-stable-decay schedules for long LLM runs.
- one-cycle schedules for fast supervised training.
- batch-size and gradient-accumulation coupling in distributed training.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 4.2 Stability role of cosine annealing

In this section, cooldown is treated as a concrete optimization object rather than a slogan. The
goal is to understand how it changes the objective, the update rule, the convergence story, and
the diagnostics a practitioner should inspect when training a modern model. For Learning Rate
Schedules, the phrase "Stability role of cosine annealing" means a precise mathematical habit:
state the assumptions, write the update, identify what can be measured, and connect the result
to a real AI training decision.

> **Definition.**
>
> For this section, **cooldown** is the part of Learning Rate Schedules that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where cooldown can be computed directly and compared with theory.
- A logistic-regression or softmax objective where cooldown affects optimization but the model remains interpretable.
- A transformer training diagnostic where cooldown appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating cooldown as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving cooldown,
and use the section assumptions to bound the change in objective value. If the assumption is
geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes cooldown visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about cooldown is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- linear warmup plus cosine decay for transformer pretraining.
- warmup-stable-decay schedules for long LLM runs.
- one-cycle schedules for fast supervised training.
- batch-size and gradient-accumulation coupling in distributed training.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 4.3 Rate or complexity controlled by cosine with restarts

In this section, learning-rate rewinding is treated as a concrete optimization object rather
than a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Learning Rate Schedules, the phrase "Rate or complexity controlled by cosine with
restarts" means a precise mathematical habit: state the assumptions, write the update, identify
what can be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **learning-rate rewinding** is the part of Learning Rate Schedules that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where learning-rate rewinding can be computed directly and compared with theory.
- A logistic-regression or softmax objective where learning-rate rewinding affects optimization but the model remains interpretable.
- A transformer training diagnostic where learning-rate rewinding appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating learning-rate rewinding as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
learning-rate rewinding, and use the section assumptions to bound the change in objective value.
If the assumption is geometric, the proof turns a picture into an inequality. If the assumption
is stochastic, the proof takes conditional expectation before applying the bound. If the
assumption is algorithmic, the proof checks that the proposed update is a descent, projection,
or preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes learning-rate rewinding visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about learning-rate rewinding is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- linear warmup plus cosine decay for transformer pretraining.
- warmup-stable-decay schedules for long LLM runs.
- one-cycle schedules for fast supervised training.
- batch-size and gradient-accumulation coupling in distributed training.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 4.4 Diagnostic interpretation of the update path

In this section, batch-size scaling is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Learning Rate Schedules, the phrase "Diagnostic interpretation of the update path" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **batch-size scaling** is the part of Learning Rate Schedules that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where batch-size scaling can be computed directly and compared with theory.
- A logistic-regression or softmax objective where batch-size scaling affects optimization but the model remains interpretable.
- A transformer training diagnostic where batch-size scaling appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating batch-size scaling as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving batch-size
scaling, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes batch-size scaling visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about batch-size scaling is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- linear warmup plus cosine decay for transformer pretraining.
- warmup-stable-decay schedules for long LLM runs.
- one-cycle schedules for fast supervised training.
- batch-size and gradient-accumulation coupling in distributed training.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 4.5 Connection to the next section in the chapter

In this section, gradient accumulation coupling is treated as a concrete optimization object
rather than a slogan. The goal is to understand how it changes the objective, the update rule,
the convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Learning Rate Schedules, the phrase "Connection to the next section in the chapter"
means a precise mathematical habit: state the assumptions, write the update, identify what can
be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **gradient accumulation coupling** is the part of Learning Rate Schedules that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where gradient accumulation coupling can be computed directly and compared with theory.
- A logistic-regression or softmax objective where gradient accumulation coupling affects optimization but the model remains interpretable.
- A transformer training diagnostic where gradient accumulation coupling appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating gradient accumulation coupling as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving gradient
accumulation coupling, and use the section assumptions to bound the change in objective value.
If the assumption is geometric, the proof turns a picture into an inequality. If the assumption
is stochastic, the proof takes conditional expectation before applying the bound. If the
assumption is algorithmic, the proof checks that the proposed update is a descent, projection,
or preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes gradient accumulation coupling visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about gradient accumulation coupling is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- linear warmup plus cosine decay for transformer pretraining.
- warmup-stable-decay schedules for long LLM runs.
- one-cycle schedules for fast supervised training.
- batch-size and gradient-accumulation coupling in distributed training.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 5. Core Theory III: Practical Variants

This block develops core theory iii: practical variants for Learning Rate Schedules. It keeps
the scope local to this section while pointing forward when a neighboring topic owns the full
treatment.

### 5.1 Variant built around cyclic learning rate

In this section, batch-size scaling is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Learning Rate Schedules, the phrase "Variant built around cyclic learning rate" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **batch-size scaling** is the part of Learning Rate Schedules that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where batch-size scaling can be computed directly and compared with theory.
- A logistic-regression or softmax objective where batch-size scaling affects optimization but the model remains interpretable.
- A transformer training diagnostic where batch-size scaling appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating batch-size scaling as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving batch-size
scaling, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes batch-size scaling visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about batch-size scaling is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- linear warmup plus cosine decay for transformer pretraining.
- warmup-stable-decay schedules for long LLM runs.
- one-cycle schedules for fast supervised training.
- batch-size and gradient-accumulation coupling in distributed training.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 5.2 Variant built around one-cycle policy

In this section, gradient accumulation coupling is treated as a concrete optimization object
rather than a slogan. The goal is to understand how it changes the objective, the update rule,
the convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Learning Rate Schedules, the phrase "Variant built around one-cycle policy" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **gradient accumulation coupling** is the part of Learning Rate Schedules that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where gradient accumulation coupling can be computed directly and compared with theory.
- A logistic-regression or softmax objective where gradient accumulation coupling affects optimization but the model remains interpretable.
- A transformer training diagnostic where gradient accumulation coupling appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating gradient accumulation coupling as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving gradient
accumulation coupling, and use the section assumptions to bound the change in objective value.
If the assumption is geometric, the proof turns a picture into an inequality. If the assumption
is stochastic, the proof takes conditional expectation before applying the bound. If the
assumption is algorithmic, the proof checks that the proposed update is a descent, projection,
or preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes gradient accumulation coupling visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about gradient accumulation coupling is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- linear warmup plus cosine decay for transformer pretraining.
- warmup-stable-decay schedules for long LLM runs.
- one-cycle schedules for fast supervised training.
- batch-size and gradient-accumulation coupling in distributed training.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 5.3 Variant built around linear decay

In this section, token-budget scheduling is treated as a concrete optimization object rather
than a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Learning Rate Schedules, the phrase "Variant built around linear decay" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **token-budget scheduling** is the part of Learning Rate Schedules that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where token-budget scheduling can be computed directly and compared with theory.
- A logistic-regression or softmax objective where token-budget scheduling affects optimization but the model remains interpretable.
- A transformer training diagnostic where token-budget scheduling appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating token-budget scheduling as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
token-budget scheduling, and use the section assumptions to bound the change in objective value.
If the assumption is geometric, the proof turns a picture into an inequality. If the assumption
is stochastic, the proof takes conditional expectation before applying the bound. If the
assumption is algorithmic, the proof checks that the proposed update is a descent, projection,
or preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes token-budget scheduling visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about token-budget scheduling is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- linear warmup plus cosine decay for transformer pretraining.
- warmup-stable-decay schedules for long LLM runs.
- one-cycle schedules for fast supervised training.
- batch-size and gradient-accumulation coupling in distributed training.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 5.4 Implementation constraints and numerical stability

In this section, optimizer-state interaction is treated as a concrete optimization object rather
than a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Learning Rate Schedules, the phrase "Implementation constraints and numerical
stability" means a precise mathematical habit: state the assumptions, write the update, identify
what can be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **optimizer-state interaction** is the part of Learning Rate Schedules that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where optimizer-state interaction can be computed directly and compared with theory.
- A logistic-regression or softmax objective where optimizer-state interaction affects optimization but the model remains interpretable.
- A transformer training diagnostic where optimizer-state interaction appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating optimizer-state interaction as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
optimizer-state interaction, and use the section assumptions to bound the change in objective
value. If the assumption is geometric, the proof turns a picture into an inequality. If the
assumption is stochastic, the proof takes conditional expectation before applying the bound. If
the assumption is algorithmic, the proof checks that the proposed update is a descent,
projection, or preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes optimizer-state interaction visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about optimizer-state interaction is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- linear warmup plus cosine decay for transformer pretraining.
- warmup-stable-decay schedules for long LLM runs.
- one-cycle schedules for fast supervised training.
- batch-size and gradient-accumulation coupling in distributed training.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 5.5 What belongs here versus neighboring sections

In this section, LLM pretraining schedule design is treated as a concrete optimization object
rather than a slogan. The goal is to understand how it changes the objective, the update rule,
the convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Learning Rate Schedules, the phrase "What belongs here versus neighboring sections"
means a precise mathematical habit: state the assumptions, write the update, identify what can
be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **LLM pretraining schedule design** is the part of Learning Rate Schedules that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where LLM pretraining schedule design can be computed directly and compared with theory.
- A logistic-regression or softmax objective where LLM pretraining schedule design affects optimization but the model remains interpretable.
- A transformer training diagnostic where LLM pretraining schedule design appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating LLM pretraining schedule design as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving LLM
pretraining schedule design, and use the section assumptions to bound the change in objective
value. If the assumption is geometric, the proof turns a picture into an inequality. If the
assumption is stochastic, the proof takes conditional expectation before applying the bound. If
the assumption is algorithmic, the proof checks that the proposed update is a descent,
projection, or preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes LLM pretraining schedule design visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about LLM pretraining schedule design is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- linear warmup plus cosine decay for transformer pretraining.
- warmup-stable-decay schedules for long LLM runs.
- one-cycle schedules for fast supervised training.
- batch-size and gradient-accumulation coupling in distributed training.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 6. Advanced Topics

This block develops advanced topics for Learning Rate Schedules. It keeps the scope local to
this section while pointing forward when a neighboring topic owns the full treatment.

### 6.1 Advanced view of inverse-square-root decay

In this section, optimizer-state interaction is treated as a concrete optimization object rather
than a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Learning Rate Schedules, the phrase "Advanced view of inverse-square-root decay"
means a precise mathematical habit: state the assumptions, write the update, identify what can
be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **optimizer-state interaction** is the part of Learning Rate Schedules that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where optimizer-state interaction can be computed directly and compared with theory.
- A logistic-regression or softmax objective where optimizer-state interaction affects optimization but the model remains interpretable.
- A transformer training diagnostic where optimizer-state interaction appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating optimizer-state interaction as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
optimizer-state interaction, and use the section assumptions to bound the change in objective
value. If the assumption is geometric, the proof turns a picture into an inequality. If the
assumption is stochastic, the proof takes conditional expectation before applying the bound. If
the assumption is algorithmic, the proof checks that the proposed update is a descent,
projection, or preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes optimizer-state interaction visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about optimizer-state interaction is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- linear warmup plus cosine decay for transformer pretraining.
- warmup-stable-decay schedules for long LLM runs.
- one-cycle schedules for fast supervised training.
- batch-size and gradient-accumulation coupling in distributed training.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 6.2 Advanced view of WSD schedule

In this section, LLM pretraining schedule design is treated as a concrete optimization object
rather than a slogan. The goal is to understand how it changes the objective, the update rule,
the convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Learning Rate Schedules, the phrase "Advanced view of WSD schedule" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **LLM pretraining schedule design** is the part of Learning Rate Schedules that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where LLM pretraining schedule design can be computed directly and compared with theory.
- A logistic-regression or softmax objective where LLM pretraining schedule design affects optimization but the model remains interpretable.
- A transformer training diagnostic where LLM pretraining schedule design appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating LLM pretraining schedule design as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving LLM
pretraining schedule design, and use the section assumptions to bound the change in objective
value. If the assumption is geometric, the proof turns a picture into an inequality. If the
assumption is stochastic, the proof takes conditional expectation before applying the bound. If
the assumption is algorithmic, the proof checks that the proposed update is a descent,
projection, or preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes LLM pretraining schedule design visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about LLM pretraining schedule design is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- linear warmup plus cosine decay for transformer pretraining.
- warmup-stable-decay schedules for long LLM runs.
- one-cycle schedules for fast supervised training.
- batch-size and gradient-accumulation coupling in distributed training.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 6.3 Advanced view of cooldown

In this section, fine-tuning schedule design is treated as a concrete optimization object rather
than a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Learning Rate Schedules, the phrase "Advanced view of cooldown" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **fine-tuning schedule design** is the part of Learning Rate Schedules that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where fine-tuning schedule design can be computed directly and compared with theory.
- A logistic-regression or softmax objective where fine-tuning schedule design affects optimization but the model remains interpretable.
- A transformer training diagnostic where fine-tuning schedule design appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating fine-tuning schedule design as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
fine-tuning schedule design, and use the section assumptions to bound the change in objective
value. If the assumption is geometric, the proof turns a picture into an inequality. If the
assumption is stochastic, the proof takes conditional expectation before applying the bound. If
the assumption is algorithmic, the proof checks that the proposed update is a descent,
projection, or preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes fine-tuning schedule design visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about fine-tuning schedule design is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- linear warmup plus cosine decay for transformer pretraining.
- warmup-stable-decay schedules for long LLM runs.
- one-cycle schedules for fast supervised training.
- batch-size and gradient-accumulation coupling in distributed training.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 6.4 Infinite-dimensional or large-scale interpretation

In this section, schedule function is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Learning Rate Schedules, the phrase "Infinite-dimensional or large-scale interpretation" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **schedule function** is the part of Learning Rate Schedules that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where schedule function can be computed directly and compared with theory.
- A logistic-regression or softmax objective where schedule function affects optimization but the model remains interpretable.
- A transformer training diagnostic where schedule function appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating schedule function as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving schedule
function, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes schedule function visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about schedule function is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- linear warmup plus cosine decay for transformer pretraining.
- warmup-stable-decay schedules for long LLM runs.
- one-cycle schedules for fast supervised training.
- batch-size and gradient-accumulation coupling in distributed training.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 6.5 Open questions for frontier model training

In this section, constant learning rate is treated as a concrete optimization object rather than
a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Learning Rate Schedules, the phrase "Open questions for frontier model training"
means a precise mathematical habit: state the assumptions, write the update, identify what can
be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **constant learning rate** is the part of Learning Rate Schedules that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where constant learning rate can be computed directly and compared with theory.
- A logistic-regression or softmax objective where constant learning rate affects optimization but the model remains interpretable.
- A transformer training diagnostic where constant learning rate appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating constant learning rate as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving constant
learning rate, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes constant learning rate visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about constant learning rate is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- linear warmup plus cosine decay for transformer pretraining.
- warmup-stable-decay schedules for long LLM runs.
- one-cycle schedules for fast supervised training.
- batch-size and gradient-accumulation coupling in distributed training.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 7. Applications in Machine Learning

This block develops applications in machine learning for Learning Rate Schedules. It keeps the
scope local to this section while pointing forward when a neighboring topic owns the full
treatment.

### 7.1 linear warmup plus cosine decay for transformer pretraining

In this section, schedule function is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Learning Rate Schedules, the phrase "linear warmup plus cosine decay for transformer
pretraining" means a precise mathematical habit: state the assumptions, write the update,
identify what can be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **schedule function** is the part of Learning Rate Schedules that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where schedule function can be computed directly and compared with theory.
- A logistic-regression or softmax objective where schedule function affects optimization but the model remains interpretable.
- A transformer training diagnostic where schedule function appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating schedule function as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving schedule
function, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes schedule function visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about schedule function is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- linear warmup plus cosine decay for transformer pretraining.
- warmup-stable-decay schedules for long LLM runs.
- one-cycle schedules for fast supervised training.
- batch-size and gradient-accumulation coupling in distributed training.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 7.2 warmup-stable-decay schedules for long LLM runs

In this section, constant learning rate is treated as a concrete optimization object rather than
a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Learning Rate Schedules, the phrase "warmup-stable-decay schedules for long LLM runs"
means a precise mathematical habit: state the assumptions, write the update, identify what can
be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **constant learning rate** is the part of Learning Rate Schedules that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where constant learning rate can be computed directly and compared with theory.
- A logistic-regression or softmax objective where constant learning rate affects optimization but the model remains interpretable.
- A transformer training diagnostic where constant learning rate appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating constant learning rate as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving constant
learning rate, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes constant learning rate visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about constant learning rate is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- linear warmup plus cosine decay for transformer pretraining.
- warmup-stable-decay schedules for long LLM runs.
- one-cycle schedules for fast supervised training.
- batch-size and gradient-accumulation coupling in distributed training.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 7.3 one-cycle schedules for fast supervised training

In this section, step decay is treated as a concrete optimization object rather than a slogan.
The goal is to understand how it changes the objective, the update rule, the convergence story,
and the diagnostics a practitioner should inspect when training a modern model. For Learning
Rate Schedules, the phrase "one-cycle schedules for fast supervised training" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **step decay** is the part of Learning Rate Schedules that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where step decay can be computed directly and compared with theory.
- A logistic-regression or softmax objective where step decay affects optimization but the model remains interpretable.
- A transformer training diagnostic where step decay appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating step decay as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving step
decay, and use the section assumptions to bound the change in objective value. If the assumption
is geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes step decay visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about step decay is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- linear warmup plus cosine decay for transformer pretraining.
- warmup-stable-decay schedules for long LLM runs.
- one-cycle schedules for fast supervised training.
- batch-size and gradient-accumulation coupling in distributed training.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 7.4 batch-size and gradient-accumulation coupling in distributed training

In this section, exponential decay is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Learning Rate Schedules, the phrase "batch-size and gradient-accumulation coupling in
distributed training" means a precise mathematical habit: state the assumptions, write the
update, identify what can be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **exponential decay** is the part of Learning Rate Schedules that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where exponential decay can be computed directly and compared with theory.
- A logistic-regression or softmax objective where exponential decay affects optimization but the model remains interpretable.
- A transformer training diagnostic where exponential decay appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating exponential decay as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
exponential decay, and use the section assumptions to bound the change in objective value. If
the assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes exponential decay visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about exponential decay is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- linear warmup plus cosine decay for transformer pretraining.
- warmup-stable-decay schedules for long LLM runs.
- one-cycle schedules for fast supervised training.
- batch-size and gradient-accumulation coupling in distributed training.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 7.5 Diagnostic checklist for real experiments

In this section, polynomial decay is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Learning Rate Schedules, the phrase "Diagnostic checklist for real experiments" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **polynomial decay** is the part of Learning Rate Schedules that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where polynomial decay can be computed directly and compared with theory.
- A logistic-regression or softmax objective where polynomial decay affects optimization but the model remains interpretable.
- A transformer training diagnostic where polynomial decay appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating polynomial decay as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving polynomial
decay, and use the section assumptions to bound the change in objective value. If the assumption
is geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes polynomial decay visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about polynomial decay is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- linear warmup plus cosine decay for transformer pretraining.
- warmup-stable-decay schedules for long LLM runs.
- one-cycle schedules for fast supervised training.
- batch-size and gradient-accumulation coupling in distributed training.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 8. Implementation and Diagnostics

This block develops implementation and diagnostics for Learning Rate Schedules. It keeps the
scope local to this section while pointing forward when a neighboring topic owns the full
treatment.

### 8.1 Minimal NumPy experiment for learning-rate rewinding

In this section, exponential decay is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Learning Rate Schedules, the phrase "Minimal NumPy experiment for learning-rate rewinding" means
a precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **exponential decay** is the part of Learning Rate Schedules that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where exponential decay can be computed directly and compared with theory.
- A logistic-regression or softmax objective where exponential decay affects optimization but the model remains interpretable.
- A transformer training diagnostic where exponential decay appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating exponential decay as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
exponential decay, and use the section assumptions to bound the change in objective value. If
the assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes exponential decay visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about exponential decay is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- linear warmup plus cosine decay for transformer pretraining.
- warmup-stable-decay schedules for long LLM runs.
- one-cycle schedules for fast supervised training.
- batch-size and gradient-accumulation coupling in distributed training.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 8.2 Monitoring signal for batch-size scaling

In this section, polynomial decay is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Learning Rate Schedules, the phrase "Monitoring signal for batch-size scaling" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **polynomial decay** is the part of Learning Rate Schedules that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where polynomial decay can be computed directly and compared with theory.
- A logistic-regression or softmax objective where polynomial decay affects optimization but the model remains interpretable.
- A transformer training diagnostic where polynomial decay appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating polynomial decay as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving polynomial
decay, and use the section assumptions to bound the change in objective value. If the assumption
is geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes polynomial decay visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about polynomial decay is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- linear warmup plus cosine decay for transformer pretraining.
- warmup-stable-decay schedules for long LLM runs.
- one-cycle schedules for fast supervised training.
- batch-size and gradient-accumulation coupling in distributed training.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 8.3 Failure signature for gradient accumulation coupling

In this section, linear warmup is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Learning Rate Schedules, the phrase "Failure signature for gradient accumulation coupling" means
a precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **linear warmup** is the part of Learning Rate Schedules that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where linear warmup can be computed directly and compared with theory.
- A logistic-regression or softmax objective where linear warmup affects optimization but the model remains interpretable.
- A transformer training diagnostic where linear warmup appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating linear warmup as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving linear
warmup, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes linear warmup visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about linear warmup is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- linear warmup plus cosine decay for transformer pretraining.
- warmup-stable-decay schedules for long LLM runs.
- one-cycle schedules for fast supervised training.
- batch-size and gradient-accumulation coupling in distributed training.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 8.4 Framework-level implementation pattern

In this section, warmup ratio is treated as a concrete optimization object rather than a slogan.
The goal is to understand how it changes the objective, the update rule, the convergence story,
and the diagnostics a practitioner should inspect when training a modern model. For Learning
Rate Schedules, the phrase "Framework-level implementation pattern" means a precise mathematical
habit: state the assumptions, write the update, identify what can be measured, and connect the
result to a real AI training decision.

> **Definition.**
>
> For this section, **warmup ratio** is the part of Learning Rate Schedules that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where warmup ratio can be computed directly and compared with theory.
- A logistic-regression or softmax objective where warmup ratio affects optimization but the model remains interpretable.
- A transformer training diagnostic where warmup ratio appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating warmup ratio as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving warmup
ratio, and use the section assumptions to bound the change in objective value. If the assumption
is geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes warmup ratio visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about warmup ratio is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- linear warmup plus cosine decay for transformer pretraining.
- warmup-stable-decay schedules for long LLM runs.
- one-cycle schedules for fast supervised training.
- batch-size and gradient-accumulation coupling in distributed training.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 8.5 Reproducibility and logging checklist

In this section, cosine annealing is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Learning Rate Schedules, the phrase "Reproducibility and logging checklist" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **cosine annealing** is the part of Learning Rate Schedules that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where cosine annealing can be computed directly and compared with theory.
- A logistic-regression or softmax objective where cosine annealing affects optimization but the model remains interpretable.
- A transformer training diagnostic where cosine annealing appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating cosine annealing as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\frac{\pi t}{T}\right)
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving cosine
annealing, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes cosine annealing visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about cosine annealing is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- linear warmup plus cosine decay for transformer pretraining.
- warmup-stable-decay schedules for long LLM runs.
- one-cycle schedules for fast supervised training.
- batch-size and gradient-accumulation coupling in distributed training.

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

1. **Exercise 1 [*] - Step Decay**
   (a) Define step decay using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

2. **Exercise 2 [*] - Polynomial Decay**
   (a) Define polynomial decay using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

3. **Exercise 3 [*] - Warmup Ratio**
   (a) Define warmup ratio using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

4. **Exercise 4 [**] - Cosine With Restarts**
   (a) Define cosine with restarts using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

5. **Exercise 5 [**] - One-Cycle Policy**
   (a) Define one-cycle policy using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

6. **Exercise 6 [**] - Inverse-Square-Root Decay**
   (a) Define inverse-square-root decay using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

7. **Exercise 7 [**] - Cooldown**
   (a) Define cooldown using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

8. **Exercise 8 [***] - Batch-Size Scaling**
   (a) Define batch-size scaling using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

9. **Exercise 9 [***] - Token-Budget Scheduling**
   (a) Define token-budget scheduling using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

10. **Exercise 10 [***] - Llm Pretraining Schedule Design**
   (a) Define LLM pretraining schedule design using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{u}_t
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

## 11. Why This Matters for AI (2026 Perspective)

| Concept | AI Impact |
| --- | --- |
| schedule function | linear warmup plus cosine decay for transformer pretraining |
| constant learning rate | warmup-stable-decay schedules for long LLM runs |
| step decay | one-cycle schedules for fast supervised training |
| exponential decay | batch-size and gradient-accumulation coupling in distributed training |
| polynomial decay | linear warmup plus cosine decay for transformer pretraining |
| linear warmup | warmup-stable-decay schedules for long LLM runs |
| warmup ratio | one-cycle schedules for fast supervised training |
| cosine annealing | batch-size and gradient-accumulation coupling in distributed training |
| cosine with restarts | linear warmup plus cosine decay for transformer pretraining |
| cyclic learning rate | warmup-stable-decay schedules for long LLM runs |

## 12. Conceptual Bridge

Learning Rate Schedules sits inside a chain. Earlier sections give the calculus, probability,
and linear algebra needed to write the objective and interpret the update. Later sections use
this material to reason about noisy gradients, adaptive state, regularization, tuning,
schedules, and finally information-theoretic losses.

Backward link: [Hyperparameter Optimization](../09-Hyperparameter-Optimization/notes.md)
supplies the immediate prerequisite vocabulary.

Forward link: Chapter 9 turns optimization objectives into information-theoretic quantities such
as entropy, KL divergence, cross-entropy, and Fisher information.

```text
+------------------------------------------------------------+
| Chapter 8: Optimization                                    |
|    01-Convex-Optimization          Convex Optimization    |
|    02-Gradient-Descent             Gradient Descent       |
|    03-Second-Order-Methods         Second-Order Methods   |
|    04-Constrained-Optimization     Constrained Optimization |
|    05-Stochastic-Optimization      Stochastic Optimization |
|    06-Optimization-Landscape       Optimization Landscape |
|    07-Adaptive-Learning-Rate       Adaptive Learning Rate |
|    08-Regularization-Methods       Regularization Methods |
|    09-Hyperparameter-Optimization  Hyperparameter Optimization |
| >> 10-Learning-Rate-Schedules      Learning Rate Schedules |
+------------------------------------------------------------+
```

## Appendix A. Extended Derivation and Diagnostic Cards

## References

- Smith, Cyclical Learning Rates for Training Neural Networks.
- Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.
- Vaswani et al., Attention Is All You Need.
- Recent work on warmup-stable-decay schedules for large language models.
- Goodfellow, Bengio, and Courville, Deep Learning.
- Bottou, Curtis, and Nocedal, Optimization Methods for Large-Scale Machine Learning.
- PyTorch optimizer and scheduler documentation.
- Optax documentation for composable optimizer transformations.
