[Previous: Stochastic Optimization](../05-Stochastic-Optimization/notes.md) | [Back to Chapter 8: Optimization](../README.md) | [Next: Adaptive Learning Rate](../07-Adaptive-Learning-Rate/notes.md)

---

# Optimization Landscape

> _"The loss surface is not the whole training story, but it is the terrain every update must cross."_

## Overview

Optimization Landscape is part of the optimization spine of this curriculum. It explains how
mathematical assumptions become training behavior, and how training behavior becomes measurable
engineering evidence. The section is the canonical home for critical points, saddles, Hessian
spectra, sharpness, flatness, mode connectivity, edge of stability, and nonconvex training-path
geometry.

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
- The previous optimization section, [Stochastic Optimization](../05-Stochastic-Optimization/notes.md), is assumed as local context.

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Interactive derivations, numerical checks, and visual diagnostics for Optimization Landscape. |
| [exercises.ipynb](exercises.ipynb) | Graded implementation and proof exercises for Optimization Landscape. |

## Learning Objectives

- Define the canonical objects used in Optimization Landscape with repository notation.
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
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 Why Optimization Landscape matters for training systems](#11-why-optimization-landscape-matters-for-training-systems)
  - [1.2 The optimization object: parameters, objective, algorithm, and diagnostic](#12-the-optimization-object-parameters-objective-algorithm-and-diagnostic)
  - [1.3 Historical arc from classical optimization to modern AI](#13-historical-arc-from-classical-optimization-to-modern-ai)
  - [1.4 What this section treats as canonical scope](#14-what-this-section-treats-as-canonical-scope)
  - [1.5 A first mental model for LLM training](#15-a-first-mental-model-for-llm-training)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Primary definition: critical point](#21-primary-definition-critical-point)
  - [2.2 Secondary definition: local minimum](#22-secondary-definition-local-minimum)
  - [2.3 Algorithmic object: saddle point](#23-algorithmic-object-saddle-point)
  - [2.4 Examples, non-examples, and boundary cases](#24-examples-nonexamples-and-boundary-cases)
  - [2.5 Notation, dimensions, and assumptions](#25-notation-dimensions-and-assumptions)
- [3. Core Theory I: Geometry and Guarantees](#3-core-theory-i-geometry-and-guarantees)
  - [3.1 Geometry of strict saddle](#31-geometry-of-strict-saddle)
  - [3.2 Key inequality for plateau](#32-key-inequality-for-plateau)
  - [3.3 Role of Hessian spectrum](#33-role-of-hessian-spectrum)
  - [3.4 Proof template and what the proof actually buys](#34-proof-template-and-what-the-proof-actually-buys)
  - [3.5 Failure modes when assumptions are removed](#35-failure-modes-when-assumptions-are-removed)
- [4. Core Theory II: Algorithms and Dynamics](#4-core-theory-ii-algorithms-and-dynamics)
  - [4.1 Algorithmic update for negative curvature](#41-algorithmic-update-for-negative-curvature)
  - [4.2 Stability role of degeneracy](#42-stability-role-of-degeneracy)
  - [4.3 Rate or complexity controlled by symmetry](#43-rate-or-complexity-controlled-by-symmetry)
  - [4.4 Diagnostic interpretation of the update path](#44-diagnostic-interpretation-of-the-update-path)
  - [4.5 Connection to the next section in the chapter](#45-connection-to-the-next-section-in-the-chapter)
- [5. Core Theory III: Practical Variants](#5-core-theory-iii-practical-variants)
  - [5.1 Variant built around overparameterization](#51-variant-built-around-overparameterization)
  - [5.2 Variant built around basin of attraction](#52-variant-built-around-basin-of-attraction)
  - [5.3 Variant built around barrier](#53-variant-built-around-barrier)
  - [5.4 Implementation constraints and numerical stability](#54-implementation-constraints-and-numerical-stability)
  - [5.5 What belongs here versus neighboring sections](#55-what-belongs-here-versus-neighboring-sections)
- [6. Advanced Topics](#6-advanced-topics)
  - [6.1 Advanced view of sharpness](#61-advanced-view-of-sharpness)
  - [6.2 Advanced view of flatness](#62-advanced-view-of-flatness)
  - [6.3 Advanced view of reparameterization caveat](#63-advanced-view-of-reparameterization-caveat)
  - [6.4 Infinite-dimensional or large-scale interpretation](#64-infinitedimensional-or-largescale-interpretation)
  - [6.5 Open questions for frontier model training](#65-open-questions-for-frontier-model-training)
- [7. Applications in Machine Learning](#7-applications-in-machine-learning)
  - [7.1 sharpness-aware minimization and flat-minimum heuristics](#71-sharpnessaware-minimization-and-flatminimum-heuristics)
  - [7.2 mode connectivity behind checkpoint averaging and model soups](#72-mode-connectivity-behind-checkpoint-averaging-and-model-soups)
  - [7.3 edge-of-stability behavior in large neural-network training](#73-edgeofstability-behavior-in-large-neuralnetwork-training)
  - [7.4 Hessian-spectrum diagnostics for loss spikes and instability](#74-hessianspectrum-diagnostics-for-loss-spikes-and-instability)
  - [7.5 Diagnostic checklist for real experiments](#75-diagnostic-checklist-for-real-experiments)
- [8. Implementation and Diagnostics](#8-implementation-and-diagnostics)
  - [8.1 Minimal NumPy experiment for mode connectivity](#81-minimal-numpy-experiment-for-mode-connectivity)
  - [8.2 Monitoring signal for linear interpolation](#82-monitoring-signal-for-linear-interpolation)
  - [8.3 Failure signature for curve finding](#83-failure-signature-for-curve-finding)
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

This block develops intuition for Optimization Landscape. It keeps the scope local to this
section while pointing forward when a neighboring topic owns the full treatment.

### 1.1 Why Optimization Landscape matters for training systems

In this section, plateau is treated as a concrete optimization object rather than a slogan. The
goal is to understand how it changes the objective, the update rule, the convergence story, and
the diagnostics a practitioner should inspect when training a modern model. For Optimization
Landscape, the phrase "Why Optimization Landscape matters for training systems" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **plateau** is the part of Optimization Landscape that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where plateau can be computed directly and compared with theory.
- A logistic-regression or softmax objective where plateau affects optimization but the model remains interpretable.
- A transformer training diagnostic where plateau appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating plateau as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\nabla f(\boldsymbol{\theta}^*)=\mathbf{0}, \qquad H_f(\boldsymbol{\theta}^*) \text{ determines local curvature}
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving plateau,
and use the section assumptions to bound the change in objective value. If the assumption is
geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes plateau visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about plateau is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- sharpness-aware minimization and flat-minimum heuristics.
- mode connectivity behind checkpoint averaging and model soups.
- edge-of-stability behavior in large neural-network training.
- Hessian-spectrum diagnostics for loss spikes and instability.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 1.2 The optimization object: parameters, objective, algorithm, and diagnostic

In this section, Hessian spectrum is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Optimization Landscape, the phrase "The optimization object: parameters, objective, algorithm,
and diagnostic" means a precise mathematical habit: state the assumptions, write the update,
identify what can be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **Hessian spectrum** is the part of Optimization Landscape that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Hessian spectrum can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Hessian spectrum affects optimization but the model remains interpretable.
- A transformer training diagnostic where Hessian spectrum appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Hessian spectrum as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\nabla f(\boldsymbol{\theta}^*)=\mathbf{0}, \qquad H_f(\boldsymbol{\theta}^*) \text{ determines local curvature}
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving Hessian
spectrum, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Hessian spectrum visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Hessian spectrum is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- sharpness-aware minimization and flat-minimum heuristics.
- mode connectivity behind checkpoint averaging and model soups.
- edge-of-stability behavior in large neural-network training.
- Hessian-spectrum diagnostics for loss spikes and instability.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 1.3 Historical arc from classical optimization to modern AI

In this section, negative curvature is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Optimization Landscape, the phrase "Historical arc from classical optimization to modern AI"
means a precise mathematical habit: state the assumptions, write the update, identify what can
be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **negative curvature** is the part of Optimization Landscape that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where negative curvature can be computed directly and compared with theory.
- A logistic-regression or softmax objective where negative curvature affects optimization but the model remains interpretable.
- A transformer training diagnostic where negative curvature appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating negative curvature as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\nabla f(\boldsymbol{\theta}^*)=\mathbf{0}, \qquad H_f(\boldsymbol{\theta}^*) \text{ determines local curvature}
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving negative
curvature, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes negative curvature visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about negative curvature is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- sharpness-aware minimization and flat-minimum heuristics.
- mode connectivity behind checkpoint averaging and model soups.
- edge-of-stability behavior in large neural-network training.
- Hessian-spectrum diagnostics for loss spikes and instability.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 1.4 What this section treats as canonical scope

In this section, degeneracy is treated as a concrete optimization object rather than a slogan.
The goal is to understand how it changes the objective, the update rule, the convergence story,
and the diagnostics a practitioner should inspect when training a modern model. For Optimization
Landscape, the phrase "What this section treats as canonical scope" means a precise mathematical
habit: state the assumptions, write the update, identify what can be measured, and connect the
result to a real AI training decision.

> **Definition.**
>
> For this section, **degeneracy** is the part of Optimization Landscape that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where degeneracy can be computed directly and compared with theory.
- A logistic-regression or softmax objective where degeneracy affects optimization but the model remains interpretable.
- A transformer training diagnostic where degeneracy appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating degeneracy as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\nabla f(\boldsymbol{\theta}^*)=\mathbf{0}, \qquad H_f(\boldsymbol{\theta}^*) \text{ determines local curvature}
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
degeneracy, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes degeneracy visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about degeneracy is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- sharpness-aware minimization and flat-minimum heuristics.
- mode connectivity behind checkpoint averaging and model soups.
- edge-of-stability behavior in large neural-network training.
- Hessian-spectrum diagnostics for loss spikes and instability.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 1.5 A first mental model for LLM training

In this section, symmetry is treated as a concrete optimization object rather than a slogan. The
goal is to understand how it changes the objective, the update rule, the convergence story, and
the diagnostics a practitioner should inspect when training a modern model. For Optimization
Landscape, the phrase "A first mental model for LLM training" means a precise mathematical
habit: state the assumptions, write the update, identify what can be measured, and connect the
result to a real AI training decision.

> **Definition.**
>
> For this section, **symmetry** is the part of Optimization Landscape that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where symmetry can be computed directly and compared with theory.
- A logistic-regression or softmax objective where symmetry affects optimization but the model remains interpretable.
- A transformer training diagnostic where symmetry appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating symmetry as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\nabla f(\boldsymbol{\theta}^*)=\mathbf{0}, \qquad H_f(\boldsymbol{\theta}^*) \text{ determines local curvature}
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving symmetry,
and use the section assumptions to bound the change in objective value. If the assumption is
geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes symmetry visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about symmetry is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- sharpness-aware minimization and flat-minimum heuristics.
- mode connectivity behind checkpoint averaging and model soups.
- edge-of-stability behavior in large neural-network training.
- Hessian-spectrum diagnostics for loss spikes and instability.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 2. Formal Definitions

This block develops formal definitions for Optimization Landscape. It keeps the scope local to
this section while pointing forward when a neighboring topic owns the full treatment.

### 2.1 Primary definition: critical point

In this section, degeneracy is treated as a concrete optimization object rather than a slogan.
The goal is to understand how it changes the objective, the update rule, the convergence story,
and the diagnostics a practitioner should inspect when training a modern model. For Optimization
Landscape, the phrase "Primary definition: critical point" means a precise mathematical habit:
state the assumptions, write the update, identify what can be measured, and connect the result
to a real AI training decision.

> **Definition.**
>
> For this section, **degeneracy** is the part of Optimization Landscape that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where degeneracy can be computed directly and compared with theory.
- A logistic-regression or softmax objective where degeneracy affects optimization but the model remains interpretable.
- A transformer training diagnostic where degeneracy appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating degeneracy as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\nabla f(\boldsymbol{\theta}^*)=\mathbf{0}, \qquad H_f(\boldsymbol{\theta}^*) \text{ determines local curvature}
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
degeneracy, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes degeneracy visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about degeneracy is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- sharpness-aware minimization and flat-minimum heuristics.
- mode connectivity behind checkpoint averaging and model soups.
- edge-of-stability behavior in large neural-network training.
- Hessian-spectrum diagnostics for loss spikes and instability.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 2.2 Secondary definition: local minimum

In this section, symmetry is treated as a concrete optimization object rather than a slogan. The
goal is to understand how it changes the objective, the update rule, the convergence story, and
the diagnostics a practitioner should inspect when training a modern model. For Optimization
Landscape, the phrase "Secondary definition: local minimum" means a precise mathematical habit:
state the assumptions, write the update, identify what can be measured, and connect the result
to a real AI training decision.

> **Definition.**
>
> For this section, **symmetry** is the part of Optimization Landscape that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where symmetry can be computed directly and compared with theory.
- A logistic-regression or softmax objective where symmetry affects optimization but the model remains interpretable.
- A transformer training diagnostic where symmetry appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating symmetry as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\nabla f(\boldsymbol{\theta}^*)=\mathbf{0}, \qquad H_f(\boldsymbol{\theta}^*) \text{ determines local curvature}
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving symmetry,
and use the section assumptions to bound the change in objective value. If the assumption is
geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes symmetry visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about symmetry is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- sharpness-aware minimization and flat-minimum heuristics.
- mode connectivity behind checkpoint averaging and model soups.
- edge-of-stability behavior in large neural-network training.
- Hessian-spectrum diagnostics for loss spikes and instability.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 2.3 Algorithmic object: saddle point

In this section, overparameterization is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Optimization Landscape, the phrase "Algorithmic object: saddle point" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **overparameterization** is the part of Optimization Landscape that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where overparameterization can be computed directly and compared with theory.
- A logistic-regression or softmax objective where overparameterization affects optimization but the model remains interpretable.
- A transformer training diagnostic where overparameterization appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating overparameterization as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\nabla f(\boldsymbol{\theta}^*)=\mathbf{0}, \qquad H_f(\boldsymbol{\theta}^*) \text{ determines local curvature}
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
overparameterization, and use the section assumptions to bound the change in objective value. If
the assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes overparameterization visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about overparameterization is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- sharpness-aware minimization and flat-minimum heuristics.
- mode connectivity behind checkpoint averaging and model soups.
- edge-of-stability behavior in large neural-network training.
- Hessian-spectrum diagnostics for loss spikes and instability.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 2.4 Examples, non-examples, and boundary cases

In this section, basin of attraction is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Optimization Landscape, the phrase "Examples, non-examples, and boundary cases" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **basin of attraction** is the part of Optimization Landscape that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where basin of attraction can be computed directly and compared with theory.
- A logistic-regression or softmax objective where basin of attraction affects optimization but the model remains interpretable.
- A transformer training diagnostic where basin of attraction appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating basin of attraction as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\nabla f(\boldsymbol{\theta}^*)=\mathbf{0}, \qquad H_f(\boldsymbol{\theta}^*) \text{ determines local curvature}
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving basin of
attraction, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes basin of attraction visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about basin of attraction is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- sharpness-aware minimization and flat-minimum heuristics.
- mode connectivity behind checkpoint averaging and model soups.
- edge-of-stability behavior in large neural-network training.
- Hessian-spectrum diagnostics for loss spikes and instability.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 2.5 Notation, dimensions, and assumptions

In this section, barrier is treated as a concrete optimization object rather than a slogan. The
goal is to understand how it changes the objective, the update rule, the convergence story, and
the diagnostics a practitioner should inspect when training a modern model. For Optimization
Landscape, the phrase "Notation, dimensions, and assumptions" means a precise mathematical
habit: state the assumptions, write the update, identify what can be measured, and connect the
result to a real AI training decision.

> **Definition.**
>
> For this section, **barrier** is the part of Optimization Landscape that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where barrier can be computed directly and compared with theory.
- A logistic-regression or softmax objective where barrier affects optimization but the model remains interpretable.
- A transformer training diagnostic where barrier appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating barrier as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\nabla f(\boldsymbol{\theta}^*)=\mathbf{0}, \qquad H_f(\boldsymbol{\theta}^*) \text{ determines local curvature}
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving barrier,
and use the section assumptions to bound the change in objective value. If the assumption is
geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes barrier visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about barrier is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- sharpness-aware minimization and flat-minimum heuristics.
- mode connectivity behind checkpoint averaging and model soups.
- edge-of-stability behavior in large neural-network training.
- Hessian-spectrum diagnostics for loss spikes and instability.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 3. Core Theory I: Geometry and Guarantees

This block develops core theory i: geometry and guarantees for Optimization Landscape. It keeps
the scope local to this section while pointing forward when a neighboring topic owns the full
treatment.

### 3.1 Geometry of strict saddle

In this section, basin of attraction is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Optimization Landscape, the phrase "Geometry of strict saddle" means a precise mathematical
habit: state the assumptions, write the update, identify what can be measured, and connect the
result to a real AI training decision.

> **Definition.**
>
> For this section, **basin of attraction** is the part of Optimization Landscape that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where basin of attraction can be computed directly and compared with theory.
- A logistic-regression or softmax objective where basin of attraction affects optimization but the model remains interpretable.
- A transformer training diagnostic where basin of attraction appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating basin of attraction as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\nabla f(\boldsymbol{\theta}^*)=\mathbf{0}, \qquad H_f(\boldsymbol{\theta}^*) \text{ determines local curvature}
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving basin of
attraction, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes basin of attraction visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about basin of attraction is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- sharpness-aware minimization and flat-minimum heuristics.
- mode connectivity behind checkpoint averaging and model soups.
- edge-of-stability behavior in large neural-network training.
- Hessian-spectrum diagnostics for loss spikes and instability.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 3.2 Key inequality for plateau

In this section, barrier is treated as a concrete optimization object rather than a slogan. The
goal is to understand how it changes the objective, the update rule, the convergence story, and
the diagnostics a practitioner should inspect when training a modern model. For Optimization
Landscape, the phrase "Key inequality for plateau" means a precise mathematical habit: state the
assumptions, write the update, identify what can be measured, and connect the result to a real
AI training decision.

> **Definition.**
>
> For this section, **barrier** is the part of Optimization Landscape that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where barrier can be computed directly and compared with theory.
- A logistic-regression or softmax objective where barrier affects optimization but the model remains interpretable.
- A transformer training diagnostic where barrier appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating barrier as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\nabla f(\boldsymbol{\theta}^*)=\mathbf{0}, \qquad H_f(\boldsymbol{\theta}^*) \text{ determines local curvature}
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving barrier,
and use the section assumptions to bound the change in objective value. If the assumption is
geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes barrier visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about barrier is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- sharpness-aware minimization and flat-minimum heuristics.
- mode connectivity behind checkpoint averaging and model soups.
- edge-of-stability behavior in large neural-network training.
- Hessian-spectrum diagnostics for loss spikes and instability.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 3.3 Role of Hessian spectrum

In this section, sharpness is treated as a concrete optimization object rather than a slogan.
The goal is to understand how it changes the objective, the update rule, the convergence story,
and the diagnostics a practitioner should inspect when training a modern model. For Optimization
Landscape, the phrase "Role of Hessian spectrum" means a precise mathematical habit: state the
assumptions, write the update, identify what can be measured, and connect the result to a real
AI training decision.

> **Definition.**
>
> For this section, **sharpness** is the part of Optimization Landscape that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where sharpness can be computed directly and compared with theory.
- A logistic-regression or softmax objective where sharpness affects optimization but the model remains interpretable.
- A transformer training diagnostic where sharpness appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating sharpness as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\nabla f(\boldsymbol{\theta}^*)=\mathbf{0}, \qquad H_f(\boldsymbol{\theta}^*) \text{ determines local curvature}
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving sharpness,
and use the section assumptions to bound the change in objective value. If the assumption is
geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes sharpness visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about sharpness is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- sharpness-aware minimization and flat-minimum heuristics.
- mode connectivity behind checkpoint averaging and model soups.
- edge-of-stability behavior in large neural-network training.
- Hessian-spectrum diagnostics for loss spikes and instability.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 3.4 Proof template and what the proof actually buys

In this section, flatness is treated as a concrete optimization object rather than a slogan. The
goal is to understand how it changes the objective, the update rule, the convergence story, and
the diagnostics a practitioner should inspect when training a modern model. For Optimization
Landscape, the phrase "Proof template and what the proof actually buys" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **flatness** is the part of Optimization Landscape that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where flatness can be computed directly and compared with theory.
- A logistic-regression or softmax objective where flatness affects optimization but the model remains interpretable.
- A transformer training diagnostic where flatness appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating flatness as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\nabla f(\boldsymbol{\theta}^*)=\mathbf{0}, \qquad H_f(\boldsymbol{\theta}^*) \text{ determines local curvature}
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving flatness,
and use the section assumptions to bound the change in objective value. If the assumption is
geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes flatness visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about flatness is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- sharpness-aware minimization and flat-minimum heuristics.
- mode connectivity behind checkpoint averaging and model soups.
- edge-of-stability behavior in large neural-network training.
- Hessian-spectrum diagnostics for loss spikes and instability.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 3.5 Failure modes when assumptions are removed

In this section, reparameterization caveat is treated as a concrete optimization object rather
than a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Optimization Landscape, the phrase "Failure modes when assumptions are removed" means
a precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **reparameterization caveat** is the part of Optimization Landscape that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where reparameterization caveat can be computed directly and compared with theory.
- A logistic-regression or softmax objective where reparameterization caveat affects optimization but the model remains interpretable.
- A transformer training diagnostic where reparameterization caveat appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating reparameterization caveat as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\nabla f(\boldsymbol{\theta}^*)=\mathbf{0}, \qquad H_f(\boldsymbol{\theta}^*) \text{ determines local curvature}
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
reparameterization caveat, and use the section assumptions to bound the change in objective
value. If the assumption is geometric, the proof turns a picture into an inequality. If the
assumption is stochastic, the proof takes conditional expectation before applying the bound. If
the assumption is algorithmic, the proof checks that the proposed update is a descent,
projection, or preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes reparameterization caveat visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about reparameterization caveat is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- sharpness-aware minimization and flat-minimum heuristics.
- mode connectivity behind checkpoint averaging and model soups.
- edge-of-stability behavior in large neural-network training.
- Hessian-spectrum diagnostics for loss spikes and instability.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 4. Core Theory II: Algorithms and Dynamics

This block develops core theory ii: algorithms and dynamics for Optimization Landscape. It keeps
the scope local to this section while pointing forward when a neighboring topic owns the full
treatment.

### 4.1 Algorithmic update for negative curvature

In this section, flatness is treated as a concrete optimization object rather than a slogan. The
goal is to understand how it changes the objective, the update rule, the convergence story, and
the diagnostics a practitioner should inspect when training a modern model. For Optimization
Landscape, the phrase "Algorithmic update for negative curvature" means a precise mathematical
habit: state the assumptions, write the update, identify what can be measured, and connect the
result to a real AI training decision.

> **Definition.**
>
> For this section, **flatness** is the part of Optimization Landscape that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where flatness can be computed directly and compared with theory.
- A logistic-regression or softmax objective where flatness affects optimization but the model remains interpretable.
- A transformer training diagnostic where flatness appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating flatness as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\nabla f(\boldsymbol{\theta}^*)=\mathbf{0}, \qquad H_f(\boldsymbol{\theta}^*) \text{ determines local curvature}
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving flatness,
and use the section assumptions to bound the change in objective value. If the assumption is
geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes flatness visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about flatness is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- sharpness-aware minimization and flat-minimum heuristics.
- mode connectivity behind checkpoint averaging and model soups.
- edge-of-stability behavior in large neural-network training.
- Hessian-spectrum diagnostics for loss spikes and instability.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 4.2 Stability role of degeneracy

In this section, reparameterization caveat is treated as a concrete optimization object rather
than a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Optimization Landscape, the phrase "Stability role of degeneracy" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **reparameterization caveat** is the part of Optimization Landscape that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where reparameterization caveat can be computed directly and compared with theory.
- A logistic-regression or softmax objective where reparameterization caveat affects optimization but the model remains interpretable.
- A transformer training diagnostic where reparameterization caveat appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating reparameterization caveat as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\nabla f(\boldsymbol{\theta}^*)=\mathbf{0}, \qquad H_f(\boldsymbol{\theta}^*) \text{ determines local curvature}
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
reparameterization caveat, and use the section assumptions to bound the change in objective
value. If the assumption is geometric, the proof turns a picture into an inequality. If the
assumption is stochastic, the proof takes conditional expectation before applying the bound. If
the assumption is algorithmic, the proof checks that the proposed update is a descent,
projection, or preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes reparameterization caveat visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about reparameterization caveat is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- sharpness-aware minimization and flat-minimum heuristics.
- mode connectivity behind checkpoint averaging and model soups.
- edge-of-stability behavior in large neural-network training.
- Hessian-spectrum diagnostics for loss spikes and instability.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 4.3 Rate or complexity controlled by symmetry

In this section, mode connectivity is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Optimization Landscape, the phrase "Rate or complexity controlled by symmetry" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **mode connectivity** is the part of Optimization Landscape that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where mode connectivity can be computed directly and compared with theory.
- A logistic-regression or softmax objective where mode connectivity affects optimization but the model remains interpretable.
- A transformer training diagnostic where mode connectivity appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating mode connectivity as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\nabla f(\boldsymbol{\theta}^*)=\mathbf{0}, \qquad H_f(\boldsymbol{\theta}^*) \text{ determines local curvature}
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving mode
connectivity, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes mode connectivity visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about mode connectivity is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- sharpness-aware minimization and flat-minimum heuristics.
- mode connectivity behind checkpoint averaging and model soups.
- edge-of-stability behavior in large neural-network training.
- Hessian-spectrum diagnostics for loss spikes and instability.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 4.4 Diagnostic interpretation of the update path

In this section, linear interpolation is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Optimization Landscape, the phrase "Diagnostic interpretation of the update path" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **linear interpolation** is the part of Optimization Landscape that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where linear interpolation can be computed directly and compared with theory.
- A logistic-regression or softmax objective where linear interpolation affects optimization but the model remains interpretable.
- A transformer training diagnostic where linear interpolation appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating linear interpolation as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\nabla f(\boldsymbol{\theta}^*)=\mathbf{0}, \qquad H_f(\boldsymbol{\theta}^*) \text{ determines local curvature}
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving linear
interpolation, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes linear interpolation visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about linear interpolation is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- sharpness-aware minimization and flat-minimum heuristics.
- mode connectivity behind checkpoint averaging and model soups.
- edge-of-stability behavior in large neural-network training.
- Hessian-spectrum diagnostics for loss spikes and instability.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 4.5 Connection to the next section in the chapter

In this section, curve finding is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Optimization Landscape, the phrase "Connection to the next section in the chapter" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **curve finding** is the part of Optimization Landscape that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where curve finding can be computed directly and compared with theory.
- A logistic-regression or softmax objective where curve finding affects optimization but the model remains interpretable.
- A transformer training diagnostic where curve finding appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating curve finding as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\nabla f(\boldsymbol{\theta}^*)=\mathbf{0}, \qquad H_f(\boldsymbol{\theta}^*) \text{ determines local curvature}
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving curve
finding, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes curve finding visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about curve finding is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- sharpness-aware minimization and flat-minimum heuristics.
- mode connectivity behind checkpoint averaging and model soups.
- edge-of-stability behavior in large neural-network training.
- Hessian-spectrum diagnostics for loss spikes and instability.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 5. Core Theory III: Practical Variants

This block develops core theory iii: practical variants for Optimization Landscape. It keeps the
scope local to this section while pointing forward when a neighboring topic owns the full
treatment.

### 5.1 Variant built around overparameterization

In this section, linear interpolation is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Optimization Landscape, the phrase "Variant built around overparameterization" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **linear interpolation** is the part of Optimization Landscape that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where linear interpolation can be computed directly and compared with theory.
- A logistic-regression or softmax objective where linear interpolation affects optimization but the model remains interpretable.
- A transformer training diagnostic where linear interpolation appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating linear interpolation as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\nabla f(\boldsymbol{\theta}^*)=\mathbf{0}, \qquad H_f(\boldsymbol{\theta}^*) \text{ determines local curvature}
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving linear
interpolation, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes linear interpolation visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about linear interpolation is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- sharpness-aware minimization and flat-minimum heuristics.
- mode connectivity behind checkpoint averaging and model soups.
- edge-of-stability behavior in large neural-network training.
- Hessian-spectrum diagnostics for loss spikes and instability.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 5.2 Variant built around basin of attraction

In this section, curve finding is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Optimization Landscape, the phrase "Variant built around basin of attraction" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **curve finding** is the part of Optimization Landscape that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where curve finding can be computed directly and compared with theory.
- A logistic-regression or softmax objective where curve finding affects optimization but the model remains interpretable.
- A transformer training diagnostic where curve finding appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating curve finding as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\nabla f(\boldsymbol{\theta}^*)=\mathbf{0}, \qquad H_f(\boldsymbol{\theta}^*) \text{ determines local curvature}
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving curve
finding, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes curve finding visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about curve finding is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- sharpness-aware minimization and flat-minimum heuristics.
- mode connectivity behind checkpoint averaging and model soups.
- edge-of-stability behavior in large neural-network training.
- Hessian-spectrum diagnostics for loss spikes and instability.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 5.3 Variant built around barrier

In this section, SWA is treated as a concrete optimization object rather than a slogan. The goal
is to understand how it changes the objective, the update rule, the convergence story, and the
diagnostics a practitioner should inspect when training a modern model. For Optimization
Landscape, the phrase "Variant built around barrier" means a precise mathematical habit: state
the assumptions, write the update, identify what can be measured, and connect the result to a
real AI training decision.

> **Definition.**
>
> For this section, **SWA** is the part of Optimization Landscape that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where SWA can be computed directly and compared with theory.
- A logistic-regression or softmax objective where SWA affects optimization but the model remains interpretable.
- A transformer training diagnostic where SWA appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating SWA as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\nabla f(\boldsymbol{\theta}^*)=\mathbf{0}, \qquad H_f(\boldsymbol{\theta}^*) \text{ determines local curvature}
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving SWA, and
use the section assumptions to bound the change in objective value. If the assumption is
geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes SWA visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about SWA is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- sharpness-aware minimization and flat-minimum heuristics.
- mode connectivity behind checkpoint averaging and model soups.
- edge-of-stability behavior in large neural-network training.
- Hessian-spectrum diagnostics for loss spikes and instability.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 5.4 Implementation constraints and numerical stability

In this section, model soups is treated as a concrete optimization object rather than a slogan.
The goal is to understand how it changes the objective, the update rule, the convergence story,
and the diagnostics a practitioner should inspect when training a modern model. For Optimization
Landscape, the phrase "Implementation constraints and numerical stability" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **model soups** is the part of Optimization Landscape that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where model soups can be computed directly and compared with theory.
- A logistic-regression or softmax objective where model soups affects optimization but the model remains interpretable.
- A transformer training diagnostic where model soups appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating model soups as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\nabla f(\boldsymbol{\theta}^*)=\mathbf{0}, \qquad H_f(\boldsymbol{\theta}^*) \text{ determines local curvature}
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving model
soups, and use the section assumptions to bound the change in objective value. If the assumption
is geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes model soups visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about model soups is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- sharpness-aware minimization and flat-minimum heuristics.
- mode connectivity behind checkpoint averaging and model soups.
- edge-of-stability behavior in large neural-network training.
- Hessian-spectrum diagnostics for loss spikes and instability.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 5.5 What belongs here versus neighboring sections

In this section, edge of stability is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Optimization Landscape, the phrase "What belongs here versus neighboring sections" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **edge of stability** is the part of Optimization Landscape that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where edge of stability can be computed directly and compared with theory.
- A logistic-regression or softmax objective where edge of stability affects optimization but the model remains interpretable.
- A transformer training diagnostic where edge of stability appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating edge of stability as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\nabla f(\boldsymbol{\theta}^*)=\mathbf{0}, \qquad H_f(\boldsymbol{\theta}^*) \text{ determines local curvature}
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving edge of
stability, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes edge of stability visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about edge of stability is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- sharpness-aware minimization and flat-minimum heuristics.
- mode connectivity behind checkpoint averaging and model soups.
- edge-of-stability behavior in large neural-network training.
- Hessian-spectrum diagnostics for loss spikes and instability.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 6. Advanced Topics

This block develops advanced topics for Optimization Landscape. It keeps the scope local to this
section while pointing forward when a neighboring topic owns the full treatment.

### 6.1 Advanced view of sharpness

In this section, model soups is treated as a concrete optimization object rather than a slogan.
The goal is to understand how it changes the objective, the update rule, the convergence story,
and the diagnostics a practitioner should inspect when training a modern model. For Optimization
Landscape, the phrase "Advanced view of sharpness" means a precise mathematical habit: state the
assumptions, write the update, identify what can be measured, and connect the result to a real
AI training decision.

> **Definition.**
>
> For this section, **model soups** is the part of Optimization Landscape that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where model soups can be computed directly and compared with theory.
- A logistic-regression or softmax objective where model soups affects optimization but the model remains interpretable.
- A transformer training diagnostic where model soups appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating model soups as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\nabla f(\boldsymbol{\theta}^*)=\mathbf{0}, \qquad H_f(\boldsymbol{\theta}^*) \text{ determines local curvature}
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving model
soups, and use the section assumptions to bound the change in objective value. If the assumption
is geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes model soups visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about model soups is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- sharpness-aware minimization and flat-minimum heuristics.
- mode connectivity behind checkpoint averaging and model soups.
- edge-of-stability behavior in large neural-network training.
- Hessian-spectrum diagnostics for loss spikes and instability.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 6.2 Advanced view of flatness

In this section, edge of stability is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Optimization Landscape, the phrase "Advanced view of flatness" means a precise mathematical
habit: state the assumptions, write the update, identify what can be measured, and connect the
result to a real AI training decision.

> **Definition.**
>
> For this section, **edge of stability** is the part of Optimization Landscape that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where edge of stability can be computed directly and compared with theory.
- A logistic-regression or softmax objective where edge of stability affects optimization but the model remains interpretable.
- A transformer training diagnostic where edge of stability appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating edge of stability as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\nabla f(\boldsymbol{\theta}^*)=\mathbf{0}, \qquad H_f(\boldsymbol{\theta}^*) \text{ determines local curvature}
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving edge of
stability, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes edge of stability visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about edge of stability is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- sharpness-aware minimization and flat-minimum heuristics.
- mode connectivity behind checkpoint averaging and model soups.
- edge-of-stability behavior in large neural-network training.
- Hessian-spectrum diagnostics for loss spikes and instability.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 6.3 Advanced view of reparameterization caveat

In this section, catapult dynamics is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Optimization Landscape, the phrase "Advanced view of reparameterization caveat" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **catapult dynamics** is the part of Optimization Landscape that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where catapult dynamics can be computed directly and compared with theory.
- A logistic-regression or softmax objective where catapult dynamics affects optimization but the model remains interpretable.
- A transformer training diagnostic where catapult dynamics appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating catapult dynamics as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\nabla f(\boldsymbol{\theta}^*)=\mathbf{0}, \qquad H_f(\boldsymbol{\theta}^*) \text{ determines local curvature}
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving catapult
dynamics, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes catapult dynamics visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about catapult dynamics is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- sharpness-aware minimization and flat-minimum heuristics.
- mode connectivity behind checkpoint averaging and model soups.
- edge-of-stability behavior in large neural-network training.
- Hessian-spectrum diagnostics for loss spikes and instability.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 6.4 Infinite-dimensional or large-scale interpretation

In this section, critical point is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Optimization Landscape, the phrase "Infinite-dimensional or large-scale interpretation" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **critical point** is the part of Optimization Landscape that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where critical point can be computed directly and compared with theory.
- A logistic-regression or softmax objective where critical point affects optimization but the model remains interpretable.
- A transformer training diagnostic where critical point appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating critical point as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\nabla f(\boldsymbol{\theta}^*)=\mathbf{0}, \qquad H_f(\boldsymbol{\theta}^*) \text{ determines local curvature}
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving critical
point, and use the section assumptions to bound the change in objective value. If the assumption
is geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes critical point visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about critical point is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- sharpness-aware minimization and flat-minimum heuristics.
- mode connectivity behind checkpoint averaging and model soups.
- edge-of-stability behavior in large neural-network training.
- Hessian-spectrum diagnostics for loss spikes and instability.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 6.5 Open questions for frontier model training

In this section, local minimum is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Optimization Landscape, the phrase "Open questions for frontier model training" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **local minimum** is the part of Optimization Landscape that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where local minimum can be computed directly and compared with theory.
- A logistic-regression or softmax objective where local minimum affects optimization but the model remains interpretable.
- A transformer training diagnostic where local minimum appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating local minimum as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\nabla f(\boldsymbol{\theta}^*)=\mathbf{0}, \qquad H_f(\boldsymbol{\theta}^*) \text{ determines local curvature}
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving local
minimum, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes local minimum visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about local minimum is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- sharpness-aware minimization and flat-minimum heuristics.
- mode connectivity behind checkpoint averaging and model soups.
- edge-of-stability behavior in large neural-network training.
- Hessian-spectrum diagnostics for loss spikes and instability.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 7. Applications in Machine Learning

This block develops applications in machine learning for Optimization Landscape. It keeps the
scope local to this section while pointing forward when a neighboring topic owns the full
treatment.

### 7.1 sharpness-aware minimization and flat-minimum heuristics

In this section, critical point is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Optimization Landscape, the phrase "sharpness-aware minimization and flat-minimum heuristics"
means a precise mathematical habit: state the assumptions, write the update, identify what can
be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **critical point** is the part of Optimization Landscape that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where critical point can be computed directly and compared with theory.
- A logistic-regression or softmax objective where critical point affects optimization but the model remains interpretable.
- A transformer training diagnostic where critical point appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating critical point as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\nabla f(\boldsymbol{\theta}^*)=\mathbf{0}, \qquad H_f(\boldsymbol{\theta}^*) \text{ determines local curvature}
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving critical
point, and use the section assumptions to bound the change in objective value. If the assumption
is geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes critical point visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about critical point is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- sharpness-aware minimization and flat-minimum heuristics.
- mode connectivity behind checkpoint averaging and model soups.
- edge-of-stability behavior in large neural-network training.
- Hessian-spectrum diagnostics for loss spikes and instability.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 7.2 mode connectivity behind checkpoint averaging and model soups

In this section, local minimum is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Optimization Landscape, the phrase "mode connectivity behind checkpoint averaging and model
soups" means a precise mathematical habit: state the assumptions, write the update, identify
what can be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **local minimum** is the part of Optimization Landscape that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where local minimum can be computed directly and compared with theory.
- A logistic-regression or softmax objective where local minimum affects optimization but the model remains interpretable.
- A transformer training diagnostic where local minimum appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating local minimum as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\nabla f(\boldsymbol{\theta}^*)=\mathbf{0}, \qquad H_f(\boldsymbol{\theta}^*) \text{ determines local curvature}
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving local
minimum, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes local minimum visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about local minimum is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- sharpness-aware minimization and flat-minimum heuristics.
- mode connectivity behind checkpoint averaging and model soups.
- edge-of-stability behavior in large neural-network training.
- Hessian-spectrum diagnostics for loss spikes and instability.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 7.3 edge-of-stability behavior in large neural-network training

In this section, saddle point is treated as a concrete optimization object rather than a slogan.
The goal is to understand how it changes the objective, the update rule, the convergence story,
and the diagnostics a practitioner should inspect when training a modern model. For Optimization
Landscape, the phrase "edge-of-stability behavior in large neural-network training" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **saddle point** is the part of Optimization Landscape that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where saddle point can be computed directly and compared with theory.
- A logistic-regression or softmax objective where saddle point affects optimization but the model remains interpretable.
- A transformer training diagnostic where saddle point appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating saddle point as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\nabla f(\boldsymbol{\theta}^*)=\mathbf{0}, \qquad H_f(\boldsymbol{\theta}^*) \text{ determines local curvature}
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving saddle
point, and use the section assumptions to bound the change in objective value. If the assumption
is geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes saddle point visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about saddle point is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- sharpness-aware minimization and flat-minimum heuristics.
- mode connectivity behind checkpoint averaging and model soups.
- edge-of-stability behavior in large neural-network training.
- Hessian-spectrum diagnostics for loss spikes and instability.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 7.4 Hessian-spectrum diagnostics for loss spikes and instability

In this section, strict saddle is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Optimization Landscape, the phrase "Hessian-spectrum diagnostics for loss spikes and
instability" means a precise mathematical habit: state the assumptions, write the update,
identify what can be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **strict saddle** is the part of Optimization Landscape that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where strict saddle can be computed directly and compared with theory.
- A logistic-regression or softmax objective where strict saddle affects optimization but the model remains interpretable.
- A transformer training diagnostic where strict saddle appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating strict saddle as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\nabla f(\boldsymbol{\theta}^*)=\mathbf{0}, \qquad H_f(\boldsymbol{\theta}^*) \text{ determines local curvature}
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving strict
saddle, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes strict saddle visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about strict saddle is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- sharpness-aware minimization and flat-minimum heuristics.
- mode connectivity behind checkpoint averaging and model soups.
- edge-of-stability behavior in large neural-network training.
- Hessian-spectrum diagnostics for loss spikes and instability.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 7.5 Diagnostic checklist for real experiments

In this section, plateau is treated as a concrete optimization object rather than a slogan. The
goal is to understand how it changes the objective, the update rule, the convergence story, and
the diagnostics a practitioner should inspect when training a modern model. For Optimization
Landscape, the phrase "Diagnostic checklist for real experiments" means a precise mathematical
habit: state the assumptions, write the update, identify what can be measured, and connect the
result to a real AI training decision.

> **Definition.**
>
> For this section, **plateau** is the part of Optimization Landscape that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where plateau can be computed directly and compared with theory.
- A logistic-regression or softmax objective where plateau affects optimization but the model remains interpretable.
- A transformer training diagnostic where plateau appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating plateau as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\nabla f(\boldsymbol{\theta}^*)=\mathbf{0}, \qquad H_f(\boldsymbol{\theta}^*) \text{ determines local curvature}
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving plateau,
and use the section assumptions to bound the change in objective value. If the assumption is
geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes plateau visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about plateau is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- sharpness-aware minimization and flat-minimum heuristics.
- mode connectivity behind checkpoint averaging and model soups.
- edge-of-stability behavior in large neural-network training.
- Hessian-spectrum diagnostics for loss spikes and instability.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 8. Implementation and Diagnostics

This block develops implementation and diagnostics for Optimization Landscape. It keeps the
scope local to this section while pointing forward when a neighboring topic owns the full
treatment.

### 8.1 Minimal NumPy experiment for mode connectivity

In this section, strict saddle is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Optimization Landscape, the phrase "Minimal NumPy experiment for mode connectivity" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **strict saddle** is the part of Optimization Landscape that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where strict saddle can be computed directly and compared with theory.
- A logistic-regression or softmax objective where strict saddle affects optimization but the model remains interpretable.
- A transformer training diagnostic where strict saddle appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating strict saddle as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\nabla f(\boldsymbol{\theta}^*)=\mathbf{0}, \qquad H_f(\boldsymbol{\theta}^*) \text{ determines local curvature}
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving strict
saddle, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes strict saddle visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about strict saddle is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- sharpness-aware minimization and flat-minimum heuristics.
- mode connectivity behind checkpoint averaging and model soups.
- edge-of-stability behavior in large neural-network training.
- Hessian-spectrum diagnostics for loss spikes and instability.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 8.2 Monitoring signal for linear interpolation

In this section, plateau is treated as a concrete optimization object rather than a slogan. The
goal is to understand how it changes the objective, the update rule, the convergence story, and
the diagnostics a practitioner should inspect when training a modern model. For Optimization
Landscape, the phrase "Monitoring signal for linear interpolation" means a precise mathematical
habit: state the assumptions, write the update, identify what can be measured, and connect the
result to a real AI training decision.

> **Definition.**
>
> For this section, **plateau** is the part of Optimization Landscape that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where plateau can be computed directly and compared with theory.
- A logistic-regression or softmax objective where plateau affects optimization but the model remains interpretable.
- A transformer training diagnostic where plateau appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating plateau as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\nabla f(\boldsymbol{\theta}^*)=\mathbf{0}, \qquad H_f(\boldsymbol{\theta}^*) \text{ determines local curvature}
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving plateau,
and use the section assumptions to bound the change in objective value. If the assumption is
geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes plateau visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about plateau is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- sharpness-aware minimization and flat-minimum heuristics.
- mode connectivity behind checkpoint averaging and model soups.
- edge-of-stability behavior in large neural-network training.
- Hessian-spectrum diagnostics for loss spikes and instability.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 8.3 Failure signature for curve finding

In this section, Hessian spectrum is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Optimization Landscape, the phrase "Failure signature for curve finding" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **Hessian spectrum** is the part of Optimization Landscape that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Hessian spectrum can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Hessian spectrum affects optimization but the model remains interpretable.
- A transformer training diagnostic where Hessian spectrum appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Hessian spectrum as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\nabla f(\boldsymbol{\theta}^*)=\mathbf{0}, \qquad H_f(\boldsymbol{\theta}^*) \text{ determines local curvature}
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving Hessian
spectrum, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Hessian spectrum visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Hessian spectrum is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- sharpness-aware minimization and flat-minimum heuristics.
- mode connectivity behind checkpoint averaging and model soups.
- edge-of-stability behavior in large neural-network training.
- Hessian-spectrum diagnostics for loss spikes and instability.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 8.4 Framework-level implementation pattern

In this section, negative curvature is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Optimization Landscape, the phrase "Framework-level implementation pattern" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **negative curvature** is the part of Optimization Landscape that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where negative curvature can be computed directly and compared with theory.
- A logistic-regression or softmax objective where negative curvature affects optimization but the model remains interpretable.
- A transformer training diagnostic where negative curvature appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating negative curvature as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\nabla f(\boldsymbol{\theta}^*)=\mathbf{0}, \qquad H_f(\boldsymbol{\theta}^*) \text{ determines local curvature}
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving negative
curvature, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes negative curvature visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about negative curvature is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- sharpness-aware minimization and flat-minimum heuristics.
- mode connectivity behind checkpoint averaging and model soups.
- edge-of-stability behavior in large neural-network training.
- Hessian-spectrum diagnostics for loss spikes and instability.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 8.5 Reproducibility and logging checklist

In this section, degeneracy is treated as a concrete optimization object rather than a slogan.
The goal is to understand how it changes the objective, the update rule, the convergence story,
and the diagnostics a practitioner should inspect when training a modern model. For Optimization
Landscape, the phrase "Reproducibility and logging checklist" means a precise mathematical
habit: state the assumptions, write the update, identify what can be measured, and connect the
result to a real AI training decision.

> **Definition.**
>
> For this section, **degeneracy** is the part of Optimization Landscape that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where degeneracy can be computed directly and compared with theory.
- A logistic-regression or softmax objective where degeneracy affects optimization but the model remains interpretable.
- A transformer training diagnostic where degeneracy appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating degeneracy as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\nabla f(\boldsymbol{\theta}^*)=\mathbf{0}, \qquad H_f(\boldsymbol{\theta}^*) \text{ determines local curvature}
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
degeneracy, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes degeneracy visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about degeneracy is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- sharpness-aware minimization and flat-minimum heuristics.
- mode connectivity behind checkpoint averaging and model soups.
- edge-of-stability behavior in large neural-network training.
- Hessian-spectrum diagnostics for loss spikes and instability.

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

1. **Exercise 1 [*] - Saddle Point**
   (a) Define saddle point using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

2. **Exercise 2 [*] - Plateau**
   (a) Define plateau using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

3. **Exercise 3 [*] - Negative Curvature**
   (a) Define negative curvature using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

4. **Exercise 4 [**] - Symmetry**
   (a) Define symmetry using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

5. **Exercise 5 [**] - Basin Of Attraction**
   (a) Define basin of attraction using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

6. **Exercise 6 [**] - Sharpness**
   (a) Define sharpness using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

7. **Exercise 7 [**] - Reparameterization Caveat**
   (a) Define reparameterization caveat using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

8. **Exercise 8 [***] - Linear Interpolation**
   (a) Define linear interpolation using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

9. **Exercise 9 [***] - Swa**
   (a) Define SWA using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

10. **Exercise 10 [***] - Edge Of Stability**
   (a) Define edge of stability using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\lambda_{\max}(H_f(\boldsymbol{\theta}_t)) \eta \approx 2
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

## 11. Why This Matters for AI (2026 Perspective)

| Concept | AI Impact |
| --- | --- |
| critical point | sharpness-aware minimization and flat-minimum heuristics |
| local minimum | mode connectivity behind checkpoint averaging and model soups |
| saddle point | edge-of-stability behavior in large neural-network training |
| strict saddle | Hessian-spectrum diagnostics for loss spikes and instability |
| plateau | sharpness-aware minimization and flat-minimum heuristics |
| Hessian spectrum | mode connectivity behind checkpoint averaging and model soups |
| negative curvature | edge-of-stability behavior in large neural-network training |
| degeneracy | Hessian-spectrum diagnostics for loss spikes and instability |
| symmetry | sharpness-aware minimization and flat-minimum heuristics |
| overparameterization | mode connectivity behind checkpoint averaging and model soups |

## 12. Conceptual Bridge

Optimization Landscape sits inside a chain. Earlier sections give the calculus, probability, and
linear algebra needed to write the objective and interpret the update. Later sections use this
material to reason about noisy gradients, adaptive state, regularization, tuning, schedules, and
finally information-theoretic losses.

Backward link: [Stochastic Optimization](../05-Stochastic-Optimization/notes.md) supplies the
immediate prerequisite vocabulary.

Forward link: [Adaptive Learning Rate](../07-Adaptive-Learning-Rate/notes.md) uses this section
as a building block.

```text
+------------------------------------------------------------+
| Chapter 8: Optimization                                    |
|    01-Convex-Optimization          Convex Optimization    |
|    02-Gradient-Descent             Gradient Descent       |
|    03-Second-Order-Methods         Second-Order Methods   |
|    04-Constrained-Optimization     Constrained Optimization |
|    05-Stochastic-Optimization      Stochastic Optimization |
| >> 06-Optimization-Landscape       Optimization Landscape |
|    07-Adaptive-Learning-Rate       Adaptive Learning Rate |
|    08-Regularization-Methods       Regularization Methods |
|    09-Hyperparameter-Optimization  Hyperparameter Optimization |
|    10-Learning-Rate-Schedules      Learning Rate Schedules |
+------------------------------------------------------------+
```

## Appendix A. Extended Derivation and Diagnostic Cards

## References

- Keskar et al., On Large-Batch Training for Deep Learning.
- Foret et al., Sharpness-Aware Minimization.
- Garipov et al., Loss Surfaces, Mode Connectivity, and Fast Ensembling.
- Cohen et al., Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability.
- Goodfellow, Bengio, and Courville, Deep Learning.
- Bottou, Curtis, and Nocedal, Optimization Methods for Large-Scale Machine Learning.
- PyTorch optimizer and scheduler documentation.
- Optax documentation for composable optimizer transformations.
