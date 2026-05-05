[Previous: Constrained Optimization](../04-Constrained-Optimization/notes.md) | [Back to Chapter 8: Optimization](../README.md) | [Next: Optimization Landscape](../06-Optimization-Landscape/notes.md)

---

# Stochastic Optimization

> _"Modern training does not remove noise from gradients; it learns how to use the noise without being ruled by it."_

## Overview

Stochastic Optimization is part of the optimization spine of this curriculum. It explains how
mathematical assumptions become training behavior, and how training behavior becomes measurable
engineering evidence. The section is the canonical home for stochastic gradient oracles,
minibatch estimators, variance, SGD convergence, variance reduction, batch-size scaling,
distributed SGD, and gradient-noise diagnostics.

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
- The previous optimization section, [Constrained Optimization](../04-Constrained-Optimization/notes.md), is assumed as local context.

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Interactive derivations, numerical checks, and visual diagnostics for Stochastic Optimization. |
| [exercises.ipynb](exercises.ipynb) | Graded implementation and proof exercises for Stochastic Optimization. |

## Learning Objectives

- Define the canonical objects used in Stochastic Optimization with repository notation.
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
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 Why Stochastic Optimization matters for training systems](#11-why-stochastic-optimization-matters-for-training-systems)
  - [1.2 The optimization object: parameters, objective, algorithm, and diagnostic](#12-the-optimization-object-parameters-objective-algorithm-and-diagnostic)
  - [1.3 Historical arc from classical optimization to modern AI](#13-historical-arc-from-classical-optimization-to-modern-ai)
  - [1.4 What this section treats as canonical scope](#14-what-this-section-treats-as-canonical-scope)
  - [1.5 A first mental model for LLM training](#15-a-first-mental-model-for-llm-training)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Primary definition: stochastic objective](#21-primary-definition-stochastic-objective)
  - [2.2 Secondary definition: empirical risk](#22-secondary-definition-empirical-risk)
  - [2.3 Algorithmic object: population risk](#23-algorithmic-object-population-risk)
  - [2.4 Examples, non-examples, and boundary cases](#24-examples-nonexamples-and-boundary-cases)
  - [2.5 Notation, dimensions, and assumptions](#25-notation-dimensions-and-assumptions)
- [3. Core Theory I: Geometry and Guarantees](#3-core-theory-i-geometry-and-guarantees)
  - [3.1 Geometry of unbiased gradient oracle](#31-geometry-of-unbiased-gradient-oracle)
  - [3.2 Key inequality for gradient variance](#32-key-inequality-for-gradient-variance)
  - [3.3 Role of minibatch estimator](#33-role-of-minibatch-estimator)
  - [3.4 Proof template and what the proof actually buys](#34-proof-template-and-what-the-proof-actually-buys)
  - [3.5 Failure modes when assumptions are removed](#35-failure-modes-when-assumptions-are-removed)
- [4. Core Theory II: Algorithms and Dynamics](#4-core-theory-ii-algorithms-and-dynamics)
  - [4.1 Algorithmic update for batch-size scaling](#41-algorithmic-update-for-batchsize-scaling)
  - [4.2 Stability role of critical batch size](#42-stability-role-of-critical-batch-size)
  - [4.3 Rate or complexity controlled by Robbins-Monro schedule](#43-rate-or-complexity-controlled-by-robbinsmonro-schedule)
  - [4.4 Diagnostic interpretation of the update path](#44-diagnostic-interpretation-of-the-update-path)
  - [4.5 Connection to the next section in the chapter](#45-connection-to-the-next-section-in-the-chapter)
- [5. Core Theory III: Practical Variants](#5-core-theory-iii-practical-variants)
  - [5.1 Variant built around SGD convergence](#51-variant-built-around-sgd-convergence)
  - [5.2 Variant built around strongly convex SGD](#52-variant-built-around-strongly-convex-sgd)
  - [5.3 Variant built around nonconvex SGD](#53-variant-built-around-nonconvex-sgd)
  - [5.4 Implementation constraints and numerical stability](#54-implementation-constraints-and-numerical-stability)
  - [5.5 What belongs here versus neighboring sections](#55-what-belongs-here-versus-neighboring-sections)
- [6. Advanced Topics](#6-advanced-topics)
  - [6.1 Advanced view of gradient noise scale](#61-advanced-view-of-gradient-noise-scale)
  - [6.2 Advanced view of SVRG](#62-advanced-view-of-svrg)
  - [6.3 Advanced view of SAGA](#63-advanced-view-of-saga)
  - [6.4 Infinite-dimensional or large-scale interpretation](#64-infinitedimensional-or-largescale-interpretation)
  - [6.5 Open questions for frontier model training](#65-open-questions-for-frontier-model-training)
- [7. Applications in Machine Learning](#7-applications-in-machine-learning)
  - [7.1 minibatch training for deep networks and transformers](#71-minibatch-training-for-deep-networks-and-transformers)
  - [7.2 batch-size and learning-rate coupling in large-scale pretraining](#72-batchsize-and-learningrate-coupling-in-largescale-pretraining)
  - [7.3 distributed gradient averaging under data parallelism](#73-distributed-gradient-averaging-under-data-parallelism)
  - [7.4 variance reduction ideas behind efficient fine-tuning and classical ML solvers](#74-variance-reduction-ideas-behind-efficient-finetuning-and-classical-ml-solvers)
  - [7.5 Diagnostic checklist for real experiments](#75-diagnostic-checklist-for-real-experiments)
- [8. Implementation and Diagnostics](#8-implementation-and-diagnostics)
  - [8.1 Minimal NumPy experiment for control variates](#81-minimal-numpy-experiment-for-control-variates)
  - [8.2 Monitoring signal for Polyak averaging](#82-monitoring-signal-for-polyak-averaging)
  - [8.3 Failure signature for distributed SGD](#83-failure-signature-for-distributed-sgd)
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

This block develops intuition for Stochastic Optimization. It keeps the scope local to this
section while pointing forward when a neighboring topic owns the full treatment.

### 1.1 Why Stochastic Optimization matters for training systems

In this section, gradient variance is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Stochastic Optimization, the phrase "Why Stochastic Optimization matters for training systems"
means a precise mathematical habit: state the assumptions, write the update, identify what can
be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **gradient variance** is the part of Stochastic Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where gradient variance can be computed directly and compared with theory.
- A logistic-regression or softmax objective where gradient variance affects optimization but the model remains interpretable.
- A transformer training diagnostic where gradient variance appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating gradient variance as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\mathbf{g}_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t}\nabla_{\boldsymbol{\theta}}\ell(\boldsymbol{\theta}_t; \mathbf{x}^{(i)}, y^{(i)})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving gradient
variance, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes gradient variance visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about gradient variance is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- minibatch training for deep networks and transformers.
- batch-size and learning-rate coupling in large-scale pretraining.
- distributed gradient averaging under data parallelism.
- variance reduction ideas behind efficient fine-tuning and classical ML solvers.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 1.2 The optimization object: parameters, objective, algorithm, and diagnostic

In this section, minibatch estimator is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Stochastic Optimization, the phrase "The optimization object: parameters, objective, algorithm,
and diagnostic" means a precise mathematical habit: state the assumptions, write the update,
identify what can be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **minibatch estimator** is the part of Stochastic Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where minibatch estimator can be computed directly and compared with theory.
- A logistic-regression or softmax objective where minibatch estimator affects optimization but the model remains interpretable.
- A transformer training diagnostic where minibatch estimator appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating minibatch estimator as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\mathbf{g}_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t}\nabla_{\boldsymbol{\theta}}\ell(\boldsymbol{\theta}_t; \mathbf{x}^{(i)}, y^{(i)})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving minibatch
estimator, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes minibatch estimator visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about minibatch estimator is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- minibatch training for deep networks and transformers.
- batch-size and learning-rate coupling in large-scale pretraining.
- distributed gradient averaging under data parallelism.
- variance reduction ideas behind efficient fine-tuning and classical ML solvers.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 1.3 Historical arc from classical optimization to modern AI

In this section, batch-size scaling is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Stochastic Optimization, the phrase "Historical arc from classical optimization to modern AI"
means a precise mathematical habit: state the assumptions, write the update, identify what can
be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **batch-size scaling** is the part of Stochastic Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
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
\mathbf{g}_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t}\nabla_{\boldsymbol{\theta}}\ell(\boldsymbol{\theta}_t; \mathbf{x}^{(i)}, y^{(i)})
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
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about batch-size scaling is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- minibatch training for deep networks and transformers.
- batch-size and learning-rate coupling in large-scale pretraining.
- distributed gradient averaging under data parallelism.
- variance reduction ideas behind efficient fine-tuning and classical ML solvers.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 1.4 What this section treats as canonical scope

In this section, critical batch size is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Stochastic Optimization, the phrase "What this section treats as canonical scope" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **critical batch size** is the part of Stochastic Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where critical batch size can be computed directly and compared with theory.
- A logistic-regression or softmax objective where critical batch size affects optimization but the model remains interpretable.
- A transformer training diagnostic where critical batch size appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating critical batch size as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\mathbf{g}_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t}\nabla_{\boldsymbol{\theta}}\ell(\boldsymbol{\theta}_t; \mathbf{x}^{(i)}, y^{(i)})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving critical
batch size, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes critical batch size visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about critical batch size is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- minibatch training for deep networks and transformers.
- batch-size and learning-rate coupling in large-scale pretraining.
- distributed gradient averaging under data parallelism.
- variance reduction ideas behind efficient fine-tuning and classical ML solvers.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 1.5 A first mental model for LLM training

In this section, Robbins-Monro schedule is treated as a concrete optimization object rather than
a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Stochastic Optimization, the phrase "A first mental model for LLM training" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **Robbins-Monro schedule** is the part of Stochastic Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Robbins-Monro schedule can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Robbins-Monro schedule affects optimization but the model remains interpretable.
- A transformer training diagnostic where Robbins-Monro schedule appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Robbins-Monro schedule as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\mathbf{g}_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t}\nabla_{\boldsymbol{\theta}}\ell(\boldsymbol{\theta}_t; \mathbf{x}^{(i)}, y^{(i)})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
Robbins-Monro schedule, and use the section assumptions to bound the change in objective value.
If the assumption is geometric, the proof turns a picture into an inequality. If the assumption
is stochastic, the proof takes conditional expectation before applying the bound. If the
assumption is algorithmic, the proof checks that the proposed update is a descent, projection,
or preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Robbins-Monro schedule visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Robbins-Monro schedule is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- minibatch training for deep networks and transformers.
- batch-size and learning-rate coupling in large-scale pretraining.
- distributed gradient averaging under data parallelism.
- variance reduction ideas behind efficient fine-tuning and classical ML solvers.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 2. Formal Definitions

This block develops formal definitions for Stochastic Optimization. It keeps the scope local to
this section while pointing forward when a neighboring topic owns the full treatment.

### 2.1 Primary definition: stochastic objective

In this section, critical batch size is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Stochastic Optimization, the phrase "Primary definition: stochastic objective" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **critical batch size** is the part of Stochastic Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where critical batch size can be computed directly and compared with theory.
- A logistic-regression or softmax objective where critical batch size affects optimization but the model remains interpretable.
- A transformer training diagnostic where critical batch size appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating critical batch size as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\mathbf{g}_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t}\nabla_{\boldsymbol{\theta}}\ell(\boldsymbol{\theta}_t; \mathbf{x}^{(i)}, y^{(i)})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving critical
batch size, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes critical batch size visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about critical batch size is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- minibatch training for deep networks and transformers.
- batch-size and learning-rate coupling in large-scale pretraining.
- distributed gradient averaging under data parallelism.
- variance reduction ideas behind efficient fine-tuning and classical ML solvers.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 2.2 Secondary definition: empirical risk

In this section, Robbins-Monro schedule is treated as a concrete optimization object rather than
a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Stochastic Optimization, the phrase "Secondary definition: empirical risk" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **Robbins-Monro schedule** is the part of Stochastic Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Robbins-Monro schedule can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Robbins-Monro schedule affects optimization but the model remains interpretable.
- A transformer training diagnostic where Robbins-Monro schedule appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Robbins-Monro schedule as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\mathbf{g}_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t}\nabla_{\boldsymbol{\theta}}\ell(\boldsymbol{\theta}_t; \mathbf{x}^{(i)}, y^{(i)})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
Robbins-Monro schedule, and use the section assumptions to bound the change in objective value.
If the assumption is geometric, the proof turns a picture into an inequality. If the assumption
is stochastic, the proof takes conditional expectation before applying the bound. If the
assumption is algorithmic, the proof checks that the proposed update is a descent, projection,
or preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Robbins-Monro schedule visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Robbins-Monro schedule is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- minibatch training for deep networks and transformers.
- batch-size and learning-rate coupling in large-scale pretraining.
- distributed gradient averaging under data parallelism.
- variance reduction ideas behind efficient fine-tuning and classical ML solvers.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 2.3 Algorithmic object: population risk

In this section, SGD convergence is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Stochastic Optimization, the phrase "Algorithmic object: population risk" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **SGD convergence** is the part of Stochastic Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where SGD convergence can be computed directly and compared with theory.
- A logistic-regression or softmax objective where SGD convergence affects optimization but the model remains interpretable.
- A transformer training diagnostic where SGD convergence appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating SGD convergence as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\mathbf{g}_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t}\nabla_{\boldsymbol{\theta}}\ell(\boldsymbol{\theta}_t; \mathbf{x}^{(i)}, y^{(i)})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving SGD
convergence, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes SGD convergence visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about SGD convergence is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- minibatch training for deep networks and transformers.
- batch-size and learning-rate coupling in large-scale pretraining.
- distributed gradient averaging under data parallelism.
- variance reduction ideas behind efficient fine-tuning and classical ML solvers.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 2.4 Examples, non-examples, and boundary cases

In this section, strongly convex SGD is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Stochastic Optimization, the phrase "Examples, non-examples, and boundary cases" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **strongly convex SGD** is the part of Stochastic Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where strongly convex SGD can be computed directly and compared with theory.
- A logistic-regression or softmax objective where strongly convex SGD affects optimization but the model remains interpretable.
- A transformer training diagnostic where strongly convex SGD appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating strongly convex SGD as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\mathbf{g}_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t}\nabla_{\boldsymbol{\theta}}\ell(\boldsymbol{\theta}_t; \mathbf{x}^{(i)}, y^{(i)})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving strongly
convex SGD, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes strongly convex SGD visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about strongly convex SGD is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- minibatch training for deep networks and transformers.
- batch-size and learning-rate coupling in large-scale pretraining.
- distributed gradient averaging under data parallelism.
- variance reduction ideas behind efficient fine-tuning and classical ML solvers.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 2.5 Notation, dimensions, and assumptions

In this section, nonconvex SGD is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Stochastic Optimization, the phrase "Notation, dimensions, and assumptions" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **nonconvex SGD** is the part of Stochastic Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where nonconvex SGD can be computed directly and compared with theory.
- A logistic-regression or softmax objective where nonconvex SGD affects optimization but the model remains interpretable.
- A transformer training diagnostic where nonconvex SGD appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating nonconvex SGD as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\mathbf{g}_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t}\nabla_{\boldsymbol{\theta}}\ell(\boldsymbol{\theta}_t; \mathbf{x}^{(i)}, y^{(i)})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving nonconvex
SGD, and use the section assumptions to bound the change in objective value. If the assumption
is geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes nonconvex SGD visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about nonconvex SGD is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- minibatch training for deep networks and transformers.
- batch-size and learning-rate coupling in large-scale pretraining.
- distributed gradient averaging under data parallelism.
- variance reduction ideas behind efficient fine-tuning and classical ML solvers.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 3. Core Theory I: Geometry and Guarantees

This block develops core theory i: geometry and guarantees for Stochastic Optimization. It keeps
the scope local to this section while pointing forward when a neighboring topic owns the full
treatment.

### 3.1 Geometry of unbiased gradient oracle

In this section, strongly convex SGD is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Stochastic Optimization, the phrase "Geometry of unbiased gradient oracle" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **strongly convex SGD** is the part of Stochastic Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where strongly convex SGD can be computed directly and compared with theory.
- A logistic-regression or softmax objective where strongly convex SGD affects optimization but the model remains interpretable.
- A transformer training diagnostic where strongly convex SGD appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating strongly convex SGD as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\mathbf{g}_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t}\nabla_{\boldsymbol{\theta}}\ell(\boldsymbol{\theta}_t; \mathbf{x}^{(i)}, y^{(i)})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving strongly
convex SGD, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes strongly convex SGD visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about strongly convex SGD is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- minibatch training for deep networks and transformers.
- batch-size and learning-rate coupling in large-scale pretraining.
- distributed gradient averaging under data parallelism.
- variance reduction ideas behind efficient fine-tuning and classical ML solvers.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 3.2 Key inequality for gradient variance

In this section, nonconvex SGD is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Stochastic Optimization, the phrase "Key inequality for gradient variance" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **nonconvex SGD** is the part of Stochastic Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where nonconvex SGD can be computed directly and compared with theory.
- A logistic-regression or softmax objective where nonconvex SGD affects optimization but the model remains interpretable.
- A transformer training diagnostic where nonconvex SGD appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating nonconvex SGD as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\mathbf{g}_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t}\nabla_{\boldsymbol{\theta}}\ell(\boldsymbol{\theta}_t; \mathbf{x}^{(i)}, y^{(i)})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving nonconvex
SGD, and use the section assumptions to bound the change in objective value. If the assumption
is geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes nonconvex SGD visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about nonconvex SGD is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- minibatch training for deep networks and transformers.
- batch-size and learning-rate coupling in large-scale pretraining.
- distributed gradient averaging under data parallelism.
- variance reduction ideas behind efficient fine-tuning and classical ML solvers.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 3.3 Role of minibatch estimator

In this section, gradient noise scale is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Stochastic Optimization, the phrase "Role of minibatch estimator" means a precise mathematical
habit: state the assumptions, write the update, identify what can be measured, and connect the
result to a real AI training decision.

> **Definition.**
>
> For this section, **gradient noise scale** is the part of Stochastic Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where gradient noise scale can be computed directly and compared with theory.
- A logistic-regression or softmax objective where gradient noise scale affects optimization but the model remains interpretable.
- A transformer training diagnostic where gradient noise scale appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating gradient noise scale as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\mathbf{g}_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t}\nabla_{\boldsymbol{\theta}}\ell(\boldsymbol{\theta}_t; \mathbf{x}^{(i)}, y^{(i)})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving gradient
noise scale, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes gradient noise scale visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about gradient noise scale is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- minibatch training for deep networks and transformers.
- batch-size and learning-rate coupling in large-scale pretraining.
- distributed gradient averaging under data parallelism.
- variance reduction ideas behind efficient fine-tuning and classical ML solvers.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 3.4 Proof template and what the proof actually buys

In this section, SVRG is treated as a concrete optimization object rather than a slogan. The
goal is to understand how it changes the objective, the update rule, the convergence story, and
the diagnostics a practitioner should inspect when training a modern model. For Stochastic
Optimization, the phrase "Proof template and what the proof actually buys" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **SVRG** is the part of Stochastic Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where SVRG can be computed directly and compared with theory.
- A logistic-regression or softmax objective where SVRG affects optimization but the model remains interpretable.
- A transformer training diagnostic where SVRG appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating SVRG as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\mathbf{g}_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t}\nabla_{\boldsymbol{\theta}}\ell(\boldsymbol{\theta}_t; \mathbf{x}^{(i)}, y^{(i)})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving SVRG, and
use the section assumptions to bound the change in objective value. If the assumption is
geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes SVRG visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about SVRG is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- minibatch training for deep networks and transformers.
- batch-size and learning-rate coupling in large-scale pretraining.
- distributed gradient averaging under data parallelism.
- variance reduction ideas behind efficient fine-tuning and classical ML solvers.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 3.5 Failure modes when assumptions are removed

In this section, SAGA is treated as a concrete optimization object rather than a slogan. The
goal is to understand how it changes the objective, the update rule, the convergence story, and
the diagnostics a practitioner should inspect when training a modern model. For Stochastic
Optimization, the phrase "Failure modes when assumptions are removed" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **SAGA** is the part of Stochastic Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where SAGA can be computed directly and compared with theory.
- A logistic-regression or softmax objective where SAGA affects optimization but the model remains interpretable.
- A transformer training diagnostic where SAGA appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating SAGA as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\mathbf{g}_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t}\nabla_{\boldsymbol{\theta}}\ell(\boldsymbol{\theta}_t; \mathbf{x}^{(i)}, y^{(i)})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving SAGA, and
use the section assumptions to bound the change in objective value. If the assumption is
geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes SAGA visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about SAGA is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- minibatch training for deep networks and transformers.
- batch-size and learning-rate coupling in large-scale pretraining.
- distributed gradient averaging under data parallelism.
- variance reduction ideas behind efficient fine-tuning and classical ML solvers.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 4. Core Theory II: Algorithms and Dynamics

This block develops core theory ii: algorithms and dynamics for Stochastic Optimization. It
keeps the scope local to this section while pointing forward when a neighboring topic owns the
full treatment.

### 4.1 Algorithmic update for batch-size scaling

In this section, SVRG is treated as a concrete optimization object rather than a slogan. The
goal is to understand how it changes the objective, the update rule, the convergence story, and
the diagnostics a practitioner should inspect when training a modern model. For Stochastic
Optimization, the phrase "Algorithmic update for batch-size scaling" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **SVRG** is the part of Stochastic Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where SVRG can be computed directly and compared with theory.
- A logistic-regression or softmax objective where SVRG affects optimization but the model remains interpretable.
- A transformer training diagnostic where SVRG appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating SVRG as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\mathbf{g}_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t}\nabla_{\boldsymbol{\theta}}\ell(\boldsymbol{\theta}_t; \mathbf{x}^{(i)}, y^{(i)})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving SVRG, and
use the section assumptions to bound the change in objective value. If the assumption is
geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes SVRG visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about SVRG is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- minibatch training for deep networks and transformers.
- batch-size and learning-rate coupling in large-scale pretraining.
- distributed gradient averaging under data parallelism.
- variance reduction ideas behind efficient fine-tuning and classical ML solvers.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 4.2 Stability role of critical batch size

In this section, SAGA is treated as a concrete optimization object rather than a slogan. The
goal is to understand how it changes the objective, the update rule, the convergence story, and
the diagnostics a practitioner should inspect when training a modern model. For Stochastic
Optimization, the phrase "Stability role of critical batch size" means a precise mathematical
habit: state the assumptions, write the update, identify what can be measured, and connect the
result to a real AI training decision.

> **Definition.**
>
> For this section, **SAGA** is the part of Stochastic Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where SAGA can be computed directly and compared with theory.
- A logistic-regression or softmax objective where SAGA affects optimization but the model remains interpretable.
- A transformer training diagnostic where SAGA appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating SAGA as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\mathbf{g}_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t}\nabla_{\boldsymbol{\theta}}\ell(\boldsymbol{\theta}_t; \mathbf{x}^{(i)}, y^{(i)})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving SAGA, and
use the section assumptions to bound the change in objective value. If the assumption is
geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes SAGA visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about SAGA is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- minibatch training for deep networks and transformers.
- batch-size and learning-rate coupling in large-scale pretraining.
- distributed gradient averaging under data parallelism.
- variance reduction ideas behind efficient fine-tuning and classical ML solvers.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 4.3 Rate or complexity controlled by Robbins-Monro schedule

In this section, control variates is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Stochastic Optimization, the phrase "Rate or complexity controlled by Robbins-Monro schedule"
means a precise mathematical habit: state the assumptions, write the update, identify what can
be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **control variates** is the part of Stochastic Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where control variates can be computed directly and compared with theory.
- A logistic-regression or softmax objective where control variates affects optimization but the model remains interpretable.
- A transformer training diagnostic where control variates appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating control variates as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\mathbf{g}_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t}\nabla_{\boldsymbol{\theta}}\ell(\boldsymbol{\theta}_t; \mathbf{x}^{(i)}, y^{(i)})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving control
variates, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes control variates visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about control variates is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- minibatch training for deep networks and transformers.
- batch-size and learning-rate coupling in large-scale pretraining.
- distributed gradient averaging under data parallelism.
- variance reduction ideas behind efficient fine-tuning and classical ML solvers.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 4.4 Diagnostic interpretation of the update path

In this section, Polyak averaging is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Stochastic Optimization, the phrase "Diagnostic interpretation of the update path" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **Polyak averaging** is the part of Stochastic Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Polyak averaging can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Polyak averaging affects optimization but the model remains interpretable.
- A transformer training diagnostic where Polyak averaging appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Polyak averaging as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\mathbf{g}_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t}\nabla_{\boldsymbol{\theta}}\ell(\boldsymbol{\theta}_t; \mathbf{x}^{(i)}, y^{(i)})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving Polyak
averaging, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Polyak averaging visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Polyak averaging is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- minibatch training for deep networks and transformers.
- batch-size and learning-rate coupling in large-scale pretraining.
- distributed gradient averaging under data parallelism.
- variance reduction ideas behind efficient fine-tuning and classical ML solvers.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 4.5 Connection to the next section in the chapter

In this section, distributed SGD is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Stochastic Optimization, the phrase "Connection to the next section in the chapter" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **distributed SGD** is the part of Stochastic Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where distributed SGD can be computed directly and compared with theory.
- A logistic-regression or softmax objective where distributed SGD affects optimization but the model remains interpretable.
- A transformer training diagnostic where distributed SGD appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating distributed SGD as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\mathbf{g}_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t}\nabla_{\boldsymbol{\theta}}\ell(\boldsymbol{\theta}_t; \mathbf{x}^{(i)}, y^{(i)})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
distributed SGD, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes distributed SGD visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about distributed SGD is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- minibatch training for deep networks and transformers.
- batch-size and learning-rate coupling in large-scale pretraining.
- distributed gradient averaging under data parallelism.
- variance reduction ideas behind efficient fine-tuning and classical ML solvers.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 5. Core Theory III: Practical Variants

This block develops core theory iii: practical variants for Stochastic Optimization. It keeps
the scope local to this section while pointing forward when a neighboring topic owns the full
treatment.

### 5.1 Variant built around SGD convergence

In this section, Polyak averaging is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Stochastic Optimization, the phrase "Variant built around SGD convergence" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **Polyak averaging** is the part of Stochastic Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Polyak averaging can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Polyak averaging affects optimization but the model remains interpretable.
- A transformer training diagnostic where Polyak averaging appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Polyak averaging as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\mathbf{g}_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t}\nabla_{\boldsymbol{\theta}}\ell(\boldsymbol{\theta}_t; \mathbf{x}^{(i)}, y^{(i)})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving Polyak
averaging, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Polyak averaging visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Polyak averaging is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- minibatch training for deep networks and transformers.
- batch-size and learning-rate coupling in large-scale pretraining.
- distributed gradient averaging under data parallelism.
- variance reduction ideas behind efficient fine-tuning and classical ML solvers.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 5.2 Variant built around strongly convex SGD

In this section, distributed SGD is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Stochastic Optimization, the phrase "Variant built around strongly convex SGD" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **distributed SGD** is the part of Stochastic Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where distributed SGD can be computed directly and compared with theory.
- A logistic-regression or softmax objective where distributed SGD affects optimization but the model remains interpretable.
- A transformer training diagnostic where distributed SGD appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating distributed SGD as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\mathbf{g}_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t}\nabla_{\boldsymbol{\theta}}\ell(\boldsymbol{\theta}_t; \mathbf{x}^{(i)}, y^{(i)})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
distributed SGD, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes distributed SGD visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about distributed SGD is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- minibatch training for deep networks and transformers.
- batch-size and learning-rate coupling in large-scale pretraining.
- distributed gradient averaging under data parallelism.
- variance reduction ideas behind efficient fine-tuning and classical ML solvers.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 5.3 Variant built around nonconvex SGD

In this section, gradient accumulation is treated as a concrete optimization object rather than
a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Stochastic Optimization, the phrase "Variant built around nonconvex SGD" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **gradient accumulation** is the part of Stochastic Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where gradient accumulation can be computed directly and compared with theory.
- A logistic-regression or softmax objective where gradient accumulation affects optimization but the model remains interpretable.
- A transformer training diagnostic where gradient accumulation appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating gradient accumulation as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\mathbf{g}_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t}\nabla_{\boldsymbol{\theta}}\ell(\boldsymbol{\theta}_t; \mathbf{x}^{(i)}, y^{(i)})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving gradient
accumulation, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes gradient accumulation visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about gradient accumulation is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- minibatch training for deep networks and transformers.
- batch-size and learning-rate coupling in large-scale pretraining.
- distributed gradient averaging under data parallelism.
- variance reduction ideas behind efficient fine-tuning and classical ML solvers.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 5.4 Implementation constraints and numerical stability

In this section, local SGD is treated as a concrete optimization object rather than a slogan.
The goal is to understand how it changes the objective, the update rule, the convergence story,
and the diagnostics a practitioner should inspect when training a modern model. For Stochastic
Optimization, the phrase "Implementation constraints and numerical stability" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **local SGD** is the part of Stochastic Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where local SGD can be computed directly and compared with theory.
- A logistic-regression or softmax objective where local SGD affects optimization but the model remains interpretable.
- A transformer training diagnostic where local SGD appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating local SGD as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\mathbf{g}_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t}\nabla_{\boldsymbol{\theta}}\ell(\boldsymbol{\theta}_t; \mathbf{x}^{(i)}, y^{(i)})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving local SGD,
and use the section assumptions to bound the change in objective value. If the assumption is
geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes local SGD visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about local SGD is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- minibatch training for deep networks and transformers.
- batch-size and learning-rate coupling in large-scale pretraining.
- distributed gradient averaging under data parallelism.
- variance reduction ideas behind efficient fine-tuning and classical ML solvers.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 5.5 What belongs here versus neighboring sections

In this section, federated averaging is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Stochastic Optimization, the phrase "What belongs here versus neighboring sections" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **federated averaging** is the part of Stochastic Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where federated averaging can be computed directly and compared with theory.
- A logistic-regression or softmax objective where federated averaging affects optimization but the model remains interpretable.
- A transformer training diagnostic where federated averaging appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating federated averaging as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\mathbf{g}_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t}\nabla_{\boldsymbol{\theta}}\ell(\boldsymbol{\theta}_t; \mathbf{x}^{(i)}, y^{(i)})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving federated
averaging, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes federated averaging visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about federated averaging is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- minibatch training for deep networks and transformers.
- batch-size and learning-rate coupling in large-scale pretraining.
- distributed gradient averaging under data parallelism.
- variance reduction ideas behind efficient fine-tuning and classical ML solvers.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 6. Advanced Topics

This block develops advanced topics for Stochastic Optimization. It keeps the scope local to
this section while pointing forward when a neighboring topic owns the full treatment.

### 6.1 Advanced view of gradient noise scale

In this section, local SGD is treated as a concrete optimization object rather than a slogan.
The goal is to understand how it changes the objective, the update rule, the convergence story,
and the diagnostics a practitioner should inspect when training a modern model. For Stochastic
Optimization, the phrase "Advanced view of gradient noise scale" means a precise mathematical
habit: state the assumptions, write the update, identify what can be measured, and connect the
result to a real AI training decision.

> **Definition.**
>
> For this section, **local SGD** is the part of Stochastic Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where local SGD can be computed directly and compared with theory.
- A logistic-regression or softmax objective where local SGD affects optimization but the model remains interpretable.
- A transformer training diagnostic where local SGD appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating local SGD as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\mathbf{g}_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t}\nabla_{\boldsymbol{\theta}}\ell(\boldsymbol{\theta}_t; \mathbf{x}^{(i)}, y^{(i)})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving local SGD,
and use the section assumptions to bound the change in objective value. If the assumption is
geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes local SGD visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about local SGD is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- minibatch training for deep networks and transformers.
- batch-size and learning-rate coupling in large-scale pretraining.
- distributed gradient averaging under data parallelism.
- variance reduction ideas behind efficient fine-tuning and classical ML solvers.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 6.2 Advanced view of SVRG

In this section, federated averaging is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Stochastic Optimization, the phrase "Advanced view of SVRG" means a precise mathematical habit:
state the assumptions, write the update, identify what can be measured, and connect the result
to a real AI training decision.

> **Definition.**
>
> For this section, **federated averaging** is the part of Stochastic Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where federated averaging can be computed directly and compared with theory.
- A logistic-regression or softmax objective where federated averaging affects optimization but the model remains interpretable.
- A transformer training diagnostic where federated averaging appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating federated averaging as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\mathbf{g}_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t}\nabla_{\boldsymbol{\theta}}\ell(\boldsymbol{\theta}_t; \mathbf{x}^{(i)}, y^{(i)})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving federated
averaging, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes federated averaging visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about federated averaging is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- minibatch training for deep networks and transformers.
- batch-size and learning-rate coupling in large-scale pretraining.
- distributed gradient averaging under data parallelism.
- variance reduction ideas behind efficient fine-tuning and classical ML solvers.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 6.3 Advanced view of SAGA

In this section, communication compression is treated as a concrete optimization object rather
than a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Stochastic Optimization, the phrase "Advanced view of SAGA" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **communication compression** is the part of Stochastic Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where communication compression can be computed directly and compared with theory.
- A logistic-regression or softmax objective where communication compression affects optimization but the model remains interpretable.
- A transformer training diagnostic where communication compression appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating communication compression as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\mathbf{g}_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t}\nabla_{\boldsymbol{\theta}}\ell(\boldsymbol{\theta}_t; \mathbf{x}^{(i)}, y^{(i)})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
communication compression, and use the section assumptions to bound the change in objective
value. If the assumption is geometric, the proof turns a picture into an inequality. If the
assumption is stochastic, the proof takes conditional expectation before applying the bound. If
the assumption is algorithmic, the proof checks that the proposed update is a descent,
projection, or preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes communication compression visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about communication compression is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- minibatch training for deep networks and transformers.
- batch-size and learning-rate coupling in large-scale pretraining.
- distributed gradient averaging under data parallelism.
- variance reduction ideas behind efficient fine-tuning and classical ML solvers.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 6.4 Infinite-dimensional or large-scale interpretation

In this section, LLM pretraining noise is treated as a concrete optimization object rather than
a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Stochastic Optimization, the phrase "Infinite-dimensional or large-scale
interpretation" means a precise mathematical habit: state the assumptions, write the update,
identify what can be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **LLM pretraining noise** is the part of Stochastic Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where LLM pretraining noise can be computed directly and compared with theory.
- A logistic-regression or softmax objective where LLM pretraining noise affects optimization but the model remains interpretable.
- A transformer training diagnostic where LLM pretraining noise appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating LLM pretraining noise as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\mathbf{g}_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t}\nabla_{\boldsymbol{\theta}}\ell(\boldsymbol{\theta}_t; \mathbf{x}^{(i)}, y^{(i)})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving LLM
pretraining noise, and use the section assumptions to bound the change in objective value. If
the assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes LLM pretraining noise visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about LLM pretraining noise is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- minibatch training for deep networks and transformers.
- batch-size and learning-rate coupling in large-scale pretraining.
- distributed gradient averaging under data parallelism.
- variance reduction ideas behind efficient fine-tuning and classical ML solvers.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 6.5 Open questions for frontier model training

In this section, stochastic objective is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Stochastic Optimization, the phrase "Open questions for frontier model training" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **stochastic objective** is the part of Stochastic Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where stochastic objective can be computed directly and compared with theory.
- A logistic-regression or softmax objective where stochastic objective affects optimization but the model remains interpretable.
- A transformer training diagnostic where stochastic objective appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating stochastic objective as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\mathbf{g}_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t}\nabla_{\boldsymbol{\theta}}\ell(\boldsymbol{\theta}_t; \mathbf{x}^{(i)}, y^{(i)})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving stochastic
objective, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes stochastic objective visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about stochastic objective is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- minibatch training for deep networks and transformers.
- batch-size and learning-rate coupling in large-scale pretraining.
- distributed gradient averaging under data parallelism.
- variance reduction ideas behind efficient fine-tuning and classical ML solvers.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 7. Applications in Machine Learning

This block develops applications in machine learning for Stochastic Optimization. It keeps the
scope local to this section while pointing forward when a neighboring topic owns the full
treatment.

### 7.1 minibatch training for deep networks and transformers

In this section, LLM pretraining noise is treated as a concrete optimization object rather than
a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Stochastic Optimization, the phrase "minibatch training for deep networks and
transformers" means a precise mathematical habit: state the assumptions, write the update,
identify what can be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **LLM pretraining noise** is the part of Stochastic Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where LLM pretraining noise can be computed directly and compared with theory.
- A logistic-regression or softmax objective where LLM pretraining noise affects optimization but the model remains interpretable.
- A transformer training diagnostic where LLM pretraining noise appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating LLM pretraining noise as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\mathbf{g}_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t}\nabla_{\boldsymbol{\theta}}\ell(\boldsymbol{\theta}_t; \mathbf{x}^{(i)}, y^{(i)})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving LLM
pretraining noise, and use the section assumptions to bound the change in objective value. If
the assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes LLM pretraining noise visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about LLM pretraining noise is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- minibatch training for deep networks and transformers.
- batch-size and learning-rate coupling in large-scale pretraining.
- distributed gradient averaging under data parallelism.
- variance reduction ideas behind efficient fine-tuning and classical ML solvers.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 7.2 batch-size and learning-rate coupling in large-scale pretraining

In this section, stochastic objective is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Stochastic Optimization, the phrase "batch-size and learning-rate coupling in large-scale
pretraining" means a precise mathematical habit: state the assumptions, write the update,
identify what can be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **stochastic objective** is the part of Stochastic Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where stochastic objective can be computed directly and compared with theory.
- A logistic-regression or softmax objective where stochastic objective affects optimization but the model remains interpretable.
- A transformer training diagnostic where stochastic objective appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating stochastic objective as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\mathbf{g}_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t}\nabla_{\boldsymbol{\theta}}\ell(\boldsymbol{\theta}_t; \mathbf{x}^{(i)}, y^{(i)})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving stochastic
objective, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes stochastic objective visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about stochastic objective is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- minibatch training for deep networks and transformers.
- batch-size and learning-rate coupling in large-scale pretraining.
- distributed gradient averaging under data parallelism.
- variance reduction ideas behind efficient fine-tuning and classical ML solvers.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 7.3 distributed gradient averaging under data parallelism

In this section, empirical risk is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Stochastic Optimization, the phrase "distributed gradient averaging under data parallelism"
means a precise mathematical habit: state the assumptions, write the update, identify what can
be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **empirical risk** is the part of Stochastic Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where empirical risk can be computed directly and compared with theory.
- A logistic-regression or softmax objective where empirical risk affects optimization but the model remains interpretable.
- A transformer training diagnostic where empirical risk appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating empirical risk as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\mathbf{g}_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t}\nabla_{\boldsymbol{\theta}}\ell(\boldsymbol{\theta}_t; \mathbf{x}^{(i)}, y^{(i)})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving empirical
risk, and use the section assumptions to bound the change in objective value. If the assumption
is geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes empirical risk visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about empirical risk is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- minibatch training for deep networks and transformers.
- batch-size and learning-rate coupling in large-scale pretraining.
- distributed gradient averaging under data parallelism.
- variance reduction ideas behind efficient fine-tuning and classical ML solvers.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 7.4 variance reduction ideas behind efficient fine-tuning and classical ML solvers

In this section, population risk is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Stochastic Optimization, the phrase "variance reduction ideas behind efficient fine-tuning and
classical ML solvers" means a precise mathematical habit: state the assumptions, write the
update, identify what can be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **population risk** is the part of Stochastic Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where population risk can be computed directly and compared with theory.
- A logistic-regression or softmax objective where population risk affects optimization but the model remains interpretable.
- A transformer training diagnostic where population risk appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating population risk as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\mathbf{g}_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t}\nabla_{\boldsymbol{\theta}}\ell(\boldsymbol{\theta}_t; \mathbf{x}^{(i)}, y^{(i)})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving population
risk, and use the section assumptions to bound the change in objective value. If the assumption
is geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes population risk visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about population risk is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- minibatch training for deep networks and transformers.
- batch-size and learning-rate coupling in large-scale pretraining.
- distributed gradient averaging under data parallelism.
- variance reduction ideas behind efficient fine-tuning and classical ML solvers.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 7.5 Diagnostic checklist for real experiments

In this section, unbiased gradient oracle is treated as a concrete optimization object rather
than a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Stochastic Optimization, the phrase "Diagnostic checklist for real experiments" means
a precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **unbiased gradient oracle** is the part of Stochastic Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where unbiased gradient oracle can be computed directly and compared with theory.
- A logistic-regression or softmax objective where unbiased gradient oracle affects optimization but the model remains interpretable.
- A transformer training diagnostic where unbiased gradient oracle appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating unbiased gradient oracle as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\mathbf{g}_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t}\nabla_{\boldsymbol{\theta}}\ell(\boldsymbol{\theta}_t; \mathbf{x}^{(i)}, y^{(i)})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving unbiased
gradient oracle, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes unbiased gradient oracle visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about unbiased gradient oracle is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- minibatch training for deep networks and transformers.
- batch-size and learning-rate coupling in large-scale pretraining.
- distributed gradient averaging under data parallelism.
- variance reduction ideas behind efficient fine-tuning and classical ML solvers.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 8. Implementation and Diagnostics

This block develops implementation and diagnostics for Stochastic Optimization. It keeps the
scope local to this section while pointing forward when a neighboring topic owns the full
treatment.

### 8.1 Minimal NumPy experiment for control variates

In this section, population risk is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Stochastic Optimization, the phrase "Minimal NumPy experiment for control variates" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **population risk** is the part of Stochastic Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where population risk can be computed directly and compared with theory.
- A logistic-regression or softmax objective where population risk affects optimization but the model remains interpretable.
- A transformer training diagnostic where population risk appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating population risk as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\mathbf{g}_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t}\nabla_{\boldsymbol{\theta}}\ell(\boldsymbol{\theta}_t; \mathbf{x}^{(i)}, y^{(i)})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving population
risk, and use the section assumptions to bound the change in objective value. If the assumption
is geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes population risk visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about population risk is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- minibatch training for deep networks and transformers.
- batch-size and learning-rate coupling in large-scale pretraining.
- distributed gradient averaging under data parallelism.
- variance reduction ideas behind efficient fine-tuning and classical ML solvers.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 8.2 Monitoring signal for Polyak averaging

In this section, unbiased gradient oracle is treated as a concrete optimization object rather
than a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Stochastic Optimization, the phrase "Monitoring signal for Polyak averaging" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **unbiased gradient oracle** is the part of Stochastic Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where unbiased gradient oracle can be computed directly and compared with theory.
- A logistic-regression or softmax objective where unbiased gradient oracle affects optimization but the model remains interpretable.
- A transformer training diagnostic where unbiased gradient oracle appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating unbiased gradient oracle as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\mathbf{g}_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t}\nabla_{\boldsymbol{\theta}}\ell(\boldsymbol{\theta}_t; \mathbf{x}^{(i)}, y^{(i)})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving unbiased
gradient oracle, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes unbiased gradient oracle visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about unbiased gradient oracle is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- minibatch training for deep networks and transformers.
- batch-size and learning-rate coupling in large-scale pretraining.
- distributed gradient averaging under data parallelism.
- variance reduction ideas behind efficient fine-tuning and classical ML solvers.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 8.3 Failure signature for distributed SGD

In this section, gradient variance is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Stochastic Optimization, the phrase "Failure signature for distributed SGD" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **gradient variance** is the part of Stochastic Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where gradient variance can be computed directly and compared with theory.
- A logistic-regression or softmax objective where gradient variance affects optimization but the model remains interpretable.
- A transformer training diagnostic where gradient variance appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating gradient variance as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\mathbf{g}_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t}\nabla_{\boldsymbol{\theta}}\ell(\boldsymbol{\theta}_t; \mathbf{x}^{(i)}, y^{(i)})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving gradient
variance, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes gradient variance visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about gradient variance is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- minibatch training for deep networks and transformers.
- batch-size and learning-rate coupling in large-scale pretraining.
- distributed gradient averaging under data parallelism.
- variance reduction ideas behind efficient fine-tuning and classical ML solvers.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 8.4 Framework-level implementation pattern

In this section, minibatch estimator is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Stochastic Optimization, the phrase "Framework-level implementation pattern" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **minibatch estimator** is the part of Stochastic Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where minibatch estimator can be computed directly and compared with theory.
- A logistic-regression or softmax objective where minibatch estimator affects optimization but the model remains interpretable.
- A transformer training diagnostic where minibatch estimator appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating minibatch estimator as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\mathbf{g}_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t}\nabla_{\boldsymbol{\theta}}\ell(\boldsymbol{\theta}_t; \mathbf{x}^{(i)}, y^{(i)})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving minibatch
estimator, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes minibatch estimator visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about minibatch estimator is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- minibatch training for deep networks and transformers.
- batch-size and learning-rate coupling in large-scale pretraining.
- distributed gradient averaging under data parallelism.
- variance reduction ideas behind efficient fine-tuning and classical ML solvers.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 8.5 Reproducibility and logging checklist

In this section, batch-size scaling is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Stochastic Optimization, the phrase "Reproducibility and logging checklist" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **batch-size scaling** is the part of Stochastic Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
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
\mathbf{g}_t = \frac{1}{B}\sum_{i \in \mathcal{B}_t}\nabla_{\boldsymbol{\theta}}\ell(\boldsymbol{\theta}_t; \mathbf{x}^{(i)}, y^{(i)})
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
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about batch-size scaling is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- minibatch training for deep networks and transformers.
- batch-size and learning-rate coupling in large-scale pretraining.
- distributed gradient averaging under data parallelism.
- variance reduction ideas behind efficient fine-tuning and classical ML solvers.

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

1. **Exercise 1 [*] - Population Risk**
   (a) Define population risk using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

2. **Exercise 2 [*] - Gradient Variance**
   (a) Define gradient variance using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

3. **Exercise 3 [*] - Batch-Size Scaling**
   (a) Define batch-size scaling using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

4. **Exercise 4 [**] - Robbins-Monro Schedule**
   (a) Define Robbins-Monro schedule using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

5. **Exercise 5 [**] - Strongly Convex Sgd**
   (a) Define strongly convex SGD using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

6. **Exercise 6 [**] - Gradient Noise Scale**
   (a) Define gradient noise scale using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

7. **Exercise 7 [**] - Saga**
   (a) Define SAGA using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

8. **Exercise 8 [***] - Polyak Averaging**
   (a) Define Polyak averaging using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

9. **Exercise 9 [***] - Gradient Accumulation**
   (a) Define gradient accumulation using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

10. **Exercise 10 [***] - Federated Averaging**
   (a) Define federated averaging using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \mathbf{g}_t
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

## 11. Why This Matters for AI (2026 Perspective)

| Concept | AI Impact |
| --- | --- |
| stochastic objective | minibatch training for deep networks and transformers |
| empirical risk | batch-size and learning-rate coupling in large-scale pretraining |
| population risk | distributed gradient averaging under data parallelism |
| unbiased gradient oracle | variance reduction ideas behind efficient fine-tuning and classical ML solvers |
| gradient variance | minibatch training for deep networks and transformers |
| minibatch estimator | batch-size and learning-rate coupling in large-scale pretraining |
| batch-size scaling | distributed gradient averaging under data parallelism |
| critical batch size | variance reduction ideas behind efficient fine-tuning and classical ML solvers |
| Robbins-Monro schedule | minibatch training for deep networks and transformers |
| SGD convergence | batch-size and learning-rate coupling in large-scale pretraining |

## 12. Conceptual Bridge

Stochastic Optimization sits inside a chain. Earlier sections give the calculus, probability,
and linear algebra needed to write the objective and interpret the update. Later sections use
this material to reason about noisy gradients, adaptive state, regularization, tuning,
schedules, and finally information-theoretic losses.

Backward link: [Constrained Optimization](../04-Constrained-Optimization/notes.md) supplies the
immediate prerequisite vocabulary.

Forward link: [Optimization Landscape](../06-Optimization-Landscape/notes.md) uses this section
as a building block.

```text
+------------------------------------------------------------+
| Chapter 8: Optimization                                    |
|    01-Convex-Optimization          Convex Optimization    |
|    02-Gradient-Descent             Gradient Descent       |
|    03-Second-Order-Methods         Second-Order Methods   |
|    04-Constrained-Optimization     Constrained Optimization |
| >> 05-Stochastic-Optimization      Stochastic Optimization |
|    06-Optimization-Landscape       Optimization Landscape |
|    07-Adaptive-Learning-Rate       Adaptive Learning Rate |
|    08-Regularization-Methods       Regularization Methods |
|    09-Hyperparameter-Optimization  Hyperparameter Optimization |
|    10-Learning-Rate-Schedules      Learning Rate Schedules |
+------------------------------------------------------------+
```

## Appendix A. Extended Derivation and Diagnostic Cards

## References

- Robbins and Monro, A Stochastic Approximation Method.
- Bottou, Curtis, and Nocedal, Optimization Methods for Large-Scale Machine Learning.
- Johnson and Zhang, Accelerating Stochastic Gradient Descent using Predictive Variance Reduction.
- Goyal et al., Accurate, Large Minibatch SGD.
- Goodfellow, Bengio, and Courville, Deep Learning.
- Bottou, Curtis, and Nocedal, Optimization Methods for Large-Scale Machine Learning.
- PyTorch optimizer and scheduler documentation.
- Optax documentation for composable optimizer transformations.
