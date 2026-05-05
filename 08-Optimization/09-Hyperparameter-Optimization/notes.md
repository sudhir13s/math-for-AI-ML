[Previous: Regularization Methods](../08-Regularization-Methods/notes.md) | [Back to Chapter 8: Optimization](../README.md) | [Next: Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md)

---

# Hyperparameter Optimization

> _"Hyperparameter optimization is experimental design under a compute budget."_

## Overview

Hyperparameter Optimization is part of the optimization spine of this curriculum. It explains
how mathematical assumptions become training behavior, and how training behavior becomes
measurable engineering evidence. The section is the canonical home for search spaces, grid and
random search, Bayesian optimization, acquisition functions, multi-fidelity methods, Hyperband,
ASHA, PBT, and leakage-aware tuning.

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
- The previous optimization section, [Regularization Methods](../08-Regularization-Methods/notes.md), is assumed as local context.

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Interactive derivations, numerical checks, and visual diagnostics for Hyperparameter Optimization. |
| [exercises.ipynb](exercises.ipynb) | Graded implementation and proof exercises for Hyperparameter Optimization. |

## Learning Objectives

- Define the canonical objects used in Hyperparameter Optimization with repository notation.
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
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 Why Hyperparameter Optimization matters for training systems](#11-why-hyperparameter-optimization-matters-for-training-systems)
  - [1.2 The optimization object: parameters, objective, algorithm, and diagnostic](#12-the-optimization-object-parameters-objective-algorithm-and-diagnostic)
  - [1.3 Historical arc from classical optimization to modern AI](#13-historical-arc-from-classical-optimization-to-modern-ai)
  - [1.4 What this section treats as canonical scope](#14-what-this-section-treats-as-canonical-scope)
  - [1.5 A first mental model for LLM training](#15-a-first-mental-model-for-llm-training)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Primary definition: configuration space](#21-primary-definition-configuration-space)
  - [2.2 Secondary definition: conditional parameter](#22-secondary-definition-conditional-parameter)
  - [2.3 Algorithmic object: log-uniform sampling](#23-algorithmic-object-loguniform-sampling)
  - [2.4 Examples, non-examples, and boundary cases](#24-examples-nonexamples-and-boundary-cases)
  - [2.5 Notation, dimensions, and assumptions](#25-notation-dimensions-and-assumptions)
- [3. Core Theory I: Geometry and Guarantees](#3-core-theory-i-geometry-and-guarantees)
  - [3.1 Geometry of grid search](#31-geometry-of-grid-search)
  - [3.2 Key inequality for random search](#32-key-inequality-for-random-search)
  - [3.3 Role of Sobol initialization](#33-role-of-sobol-initialization)
  - [3.4 Proof template and what the proof actually buys](#34-proof-template-and-what-the-proof-actually-buys)
  - [3.5 Failure modes when assumptions are removed](#35-failure-modes-when-assumptions-are-removed)
- [4. Core Theory II: Algorithms and Dynamics](#4-core-theory-ii-algorithms-and-dynamics)
  - [4.1 Algorithmic update for surrogate model](#41-algorithmic-update-for-surrogate-model)
  - [4.2 Stability role of Gaussian process](#42-stability-role-of-gaussian-process)
  - [4.3 Rate or complexity controlled by expected improvement](#43-rate-or-complexity-controlled-by-expected-improvement)
  - [4.4 Diagnostic interpretation of the update path](#44-diagnostic-interpretation-of-the-update-path)
  - [4.5 Connection to the next section in the chapter](#45-connection-to-the-next-section-in-the-chapter)
- [5. Core Theory III: Practical Variants](#5-core-theory-iii-practical-variants)
  - [5.1 Variant built around upper confidence bound](#51-variant-built-around-upper-confidence-bound)
  - [5.2 Variant built around Thompson sampling](#52-variant-built-around-thompson-sampling)
  - [5.3 Variant built around Bayesian optimization](#53-variant-built-around-bayesian-optimization)
  - [5.4 Implementation constraints and numerical stability](#54-implementation-constraints-and-numerical-stability)
  - [5.5 What belongs here versus neighboring sections](#55-what-belongs-here-versus-neighboring-sections)
- [6. Advanced Topics](#6-advanced-topics)
  - [6.1 Advanced view of successive halving](#61-advanced-view-of-successive-halving)
  - [6.2 Advanced view of Hyperband](#62-advanced-view-of-hyperband)
  - [6.3 Advanced view of ASHA](#63-advanced-view-of-asha)
  - [6.4 Infinite-dimensional or large-scale interpretation](#64-infinitedimensional-or-largescale-interpretation)
  - [6.5 Open questions for frontier model training](#65-open-questions-for-frontier-model-training)
- [7. Applications in Machine Learning](#7-applications-in-machine-learning)
  - [7.1 learning-rate, weight-decay, batch-size, and schedule tuning for LLM fine-tuning](#71-learningrate-weightdecay-batchsize-and-schedule-tuning-for-llm-finetuning)
  - [7.2 Hyperband and ASHA for neural architecture and training-budget search](#72-hyperband-and-asha-for-neural-architecture-and-trainingbudget-search)
  - [7.3 Bayesian optimization for expensive, low-dimensional continuous tuning](#73-bayesian-optimization-for-expensive-lowdimensional-continuous-tuning)
  - [7.4 validation leakage prevention in model-selection pipelines](#74-validation-leakage-prevention-in-modelselection-pipelines)
  - [7.5 Diagnostic checklist for real experiments](#75-diagnostic-checklist-for-real-experiments)
- [8. Implementation and Diagnostics](#8-implementation-and-diagnostics)
  - [8.1 Minimal NumPy experiment for BOHB](#81-minimal-numpy-experiment-for-bohb)
  - [8.2 Monitoring signal for population-based training](#82-monitoring-signal-for-populationbased-training)
  - [8.3 Failure signature for multi-objective tuning](#83-failure-signature-for-multiobjective-tuning)
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

This block develops intuition for Hyperparameter Optimization. It keeps the scope local to this
section while pointing forward when a neighboring topic owns the full treatment.

### 1.1 Why Hyperparameter Optimization matters for training systems

In this section, random search is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Hyperparameter Optimization, the phrase "Why Hyperparameter Optimization matters for training
systems" means a precise mathematical habit: state the assumptions, write the update, identify
what can be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **random search** is the part of Hyperparameter Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where random search can be computed directly and compared with theory.
- A logistic-regression or softmax objective where random search affects optimization but the model remains interpretable.
- A transformer training diagnostic where random search appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating random search as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\lambda}^* \in \arg\min_{\boldsymbol{\lambda}\in\Lambda} \mathcal{V}(\mathcal{A}(\boldsymbol{\lambda}), \mathcal{D}_{\mathrm{val}})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving random
search, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes random search visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about random search is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- learning-rate, weight-decay, batch-size, and schedule tuning for LLM fine-tuning.
- Hyperband and ASHA for neural architecture and training-budget search.
- Bayesian optimization for expensive, low-dimensional continuous tuning.
- validation leakage prevention in model-selection pipelines.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 1.2 The optimization object: parameters, objective, algorithm, and diagnostic

In this section, Sobol initialization is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Hyperparameter Optimization, the phrase "The optimization object: parameters, objective,
algorithm, and diagnostic" means a precise mathematical habit: state the assumptions, write the
update, identify what can be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **Sobol initialization** is the part of Hyperparameter Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Sobol initialization can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Sobol initialization affects optimization but the model remains interpretable.
- A transformer training diagnostic where Sobol initialization appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Sobol initialization as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\lambda}^* \in \arg\min_{\boldsymbol{\lambda}\in\Lambda} \mathcal{V}(\mathcal{A}(\boldsymbol{\lambda}), \mathcal{D}_{\mathrm{val}})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving Sobol
initialization, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Sobol initialization visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Sobol initialization is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- learning-rate, weight-decay, batch-size, and schedule tuning for LLM fine-tuning.
- Hyperband and ASHA for neural architecture and training-budget search.
- Bayesian optimization for expensive, low-dimensional continuous tuning.
- validation leakage prevention in model-selection pipelines.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 1.3 Historical arc from classical optimization to modern AI

In this section, surrogate model is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Hyperparameter Optimization, the phrase "Historical arc from classical optimization to modern
AI" means a precise mathematical habit: state the assumptions, write the update, identify what
can be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **surrogate model** is the part of Hyperparameter Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where surrogate model can be computed directly and compared with theory.
- A logistic-regression or softmax objective where surrogate model affects optimization but the model remains interpretable.
- A transformer training diagnostic where surrogate model appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating surrogate model as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\lambda}^* \in \arg\min_{\boldsymbol{\lambda}\in\Lambda} \mathcal{V}(\mathcal{A}(\boldsymbol{\lambda}), \mathcal{D}_{\mathrm{val}})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving surrogate
model, and use the section assumptions to bound the change in objective value. If the assumption
is geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes surrogate model visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about surrogate model is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- learning-rate, weight-decay, batch-size, and schedule tuning for LLM fine-tuning.
- Hyperband and ASHA for neural architecture and training-budget search.
- Bayesian optimization for expensive, low-dimensional continuous tuning.
- validation leakage prevention in model-selection pipelines.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 1.4 What this section treats as canonical scope

In this section, Gaussian process is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Hyperparameter Optimization, the phrase "What this section treats as canonical scope" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **Gaussian process** is the part of Hyperparameter Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Gaussian process can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Gaussian process affects optimization but the model remains interpretable.
- A transformer training diagnostic where Gaussian process appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Gaussian process as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\lambda}^* \in \arg\min_{\boldsymbol{\lambda}\in\Lambda} \mathcal{V}(\mathcal{A}(\boldsymbol{\lambda}), \mathcal{D}_{\mathrm{val}})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving Gaussian
process, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Gaussian process visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Gaussian process is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- learning-rate, weight-decay, batch-size, and schedule tuning for LLM fine-tuning.
- Hyperband and ASHA for neural architecture and training-budget search.
- Bayesian optimization for expensive, low-dimensional continuous tuning.
- validation leakage prevention in model-selection pipelines.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 1.5 A first mental model for LLM training

In this section, expected improvement is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Hyperparameter Optimization, the phrase "A first mental model for LLM training" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **expected improvement** is the part of Hyperparameter Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where expected improvement can be computed directly and compared with theory.
- A logistic-regression or softmax objective where expected improvement affects optimization but the model remains interpretable.
- A transformer training diagnostic where expected improvement appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating expected improvement as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\lambda}^* \in \arg\min_{\boldsymbol{\lambda}\in\Lambda} \mathcal{V}(\mathcal{A}(\boldsymbol{\lambda}), \mathcal{D}_{\mathrm{val}})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving expected
improvement, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes expected improvement visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about expected improvement is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- learning-rate, weight-decay, batch-size, and schedule tuning for LLM fine-tuning.
- Hyperband and ASHA for neural architecture and training-budget search.
- Bayesian optimization for expensive, low-dimensional continuous tuning.
- validation leakage prevention in model-selection pipelines.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 2. Formal Definitions

This block develops formal definitions for Hyperparameter Optimization. It keeps the scope local
to this section while pointing forward when a neighboring topic owns the full treatment.

### 2.1 Primary definition: configuration space

In this section, Gaussian process is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Hyperparameter Optimization, the phrase "Primary definition: configuration space" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **Gaussian process** is the part of Hyperparameter Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Gaussian process can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Gaussian process affects optimization but the model remains interpretable.
- A transformer training diagnostic where Gaussian process appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Gaussian process as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\lambda}^* \in \arg\min_{\boldsymbol{\lambda}\in\Lambda} \mathcal{V}(\mathcal{A}(\boldsymbol{\lambda}), \mathcal{D}_{\mathrm{val}})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving Gaussian
process, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Gaussian process visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Gaussian process is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- learning-rate, weight-decay, batch-size, and schedule tuning for LLM fine-tuning.
- Hyperband and ASHA for neural architecture and training-budget search.
- Bayesian optimization for expensive, low-dimensional continuous tuning.
- validation leakage prevention in model-selection pipelines.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 2.2 Secondary definition: conditional parameter

In this section, expected improvement is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Hyperparameter Optimization, the phrase "Secondary definition: conditional parameter" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **expected improvement** is the part of Hyperparameter Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where expected improvement can be computed directly and compared with theory.
- A logistic-regression or softmax objective where expected improvement affects optimization but the model remains interpretable.
- A transformer training diagnostic where expected improvement appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating expected improvement as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\lambda}^* \in \arg\min_{\boldsymbol{\lambda}\in\Lambda} \mathcal{V}(\mathcal{A}(\boldsymbol{\lambda}), \mathcal{D}_{\mathrm{val}})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving expected
improvement, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes expected improvement visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about expected improvement is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- learning-rate, weight-decay, batch-size, and schedule tuning for LLM fine-tuning.
- Hyperband and ASHA for neural architecture and training-budget search.
- Bayesian optimization for expensive, low-dimensional continuous tuning.
- validation leakage prevention in model-selection pipelines.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 2.3 Algorithmic object: log-uniform sampling

In this section, upper confidence bound is treated as a concrete optimization object rather than
a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Hyperparameter Optimization, the phrase "Algorithmic object: log-uniform sampling"
means a precise mathematical habit: state the assumptions, write the update, identify what can
be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **upper confidence bound** is the part of Hyperparameter Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where upper confidence bound can be computed directly and compared with theory.
- A logistic-regression or softmax objective where upper confidence bound affects optimization but the model remains interpretable.
- A transformer training diagnostic where upper confidence bound appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating upper confidence bound as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\lambda}^* \in \arg\min_{\boldsymbol{\lambda}\in\Lambda} \mathcal{V}(\mathcal{A}(\boldsymbol{\lambda}), \mathcal{D}_{\mathrm{val}})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving upper
confidence bound, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes upper confidence bound visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about upper confidence bound is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- learning-rate, weight-decay, batch-size, and schedule tuning for LLM fine-tuning.
- Hyperband and ASHA for neural architecture and training-budget search.
- Bayesian optimization for expensive, low-dimensional continuous tuning.
- validation leakage prevention in model-selection pipelines.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 2.4 Examples, non-examples, and boundary cases

In this section, Thompson sampling is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Hyperparameter Optimization, the phrase "Examples, non-examples, and boundary cases" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **Thompson sampling** is the part of Hyperparameter Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Thompson sampling can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Thompson sampling affects optimization but the model remains interpretable.
- A transformer training diagnostic where Thompson sampling appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Thompson sampling as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\lambda}^* \in \arg\min_{\boldsymbol{\lambda}\in\Lambda} \mathcal{V}(\mathcal{A}(\boldsymbol{\lambda}), \mathcal{D}_{\mathrm{val}})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving Thompson
sampling, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Thompson sampling visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Thompson sampling is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- learning-rate, weight-decay, batch-size, and schedule tuning for LLM fine-tuning.
- Hyperband and ASHA for neural architecture and training-budget search.
- Bayesian optimization for expensive, low-dimensional continuous tuning.
- validation leakage prevention in model-selection pipelines.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 2.5 Notation, dimensions, and assumptions

In this section, Bayesian optimization is treated as a concrete optimization object rather than
a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Hyperparameter Optimization, the phrase "Notation, dimensions, and assumptions" means
a precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **Bayesian optimization** is the part of Hyperparameter Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Bayesian optimization can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Bayesian optimization affects optimization but the model remains interpretable.
- A transformer training diagnostic where Bayesian optimization appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Bayesian optimization as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\lambda}^* \in \arg\min_{\boldsymbol{\lambda}\in\Lambda} \mathcal{V}(\mathcal{A}(\boldsymbol{\lambda}), \mathcal{D}_{\mathrm{val}})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving Bayesian
optimization, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Bayesian optimization visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Bayesian optimization is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- learning-rate, weight-decay, batch-size, and schedule tuning for LLM fine-tuning.
- Hyperband and ASHA for neural architecture and training-budget search.
- Bayesian optimization for expensive, low-dimensional continuous tuning.
- validation leakage prevention in model-selection pipelines.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 3. Core Theory I: Geometry and Guarantees

This block develops core theory i: geometry and guarantees for Hyperparameter Optimization. It
keeps the scope local to this section while pointing forward when a neighboring topic owns the
full treatment.

### 3.1 Geometry of grid search

In this section, Thompson sampling is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Hyperparameter Optimization, the phrase "Geometry of grid search" means a precise mathematical
habit: state the assumptions, write the update, identify what can be measured, and connect the
result to a real AI training decision.

> **Definition.**
>
> For this section, **Thompson sampling** is the part of Hyperparameter Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Thompson sampling can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Thompson sampling affects optimization but the model remains interpretable.
- A transformer training diagnostic where Thompson sampling appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Thompson sampling as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\lambda}^* \in \arg\min_{\boldsymbol{\lambda}\in\Lambda} \mathcal{V}(\mathcal{A}(\boldsymbol{\lambda}), \mathcal{D}_{\mathrm{val}})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving Thompson
sampling, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Thompson sampling visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Thompson sampling is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- learning-rate, weight-decay, batch-size, and schedule tuning for LLM fine-tuning.
- Hyperband and ASHA for neural architecture and training-budget search.
- Bayesian optimization for expensive, low-dimensional continuous tuning.
- validation leakage prevention in model-selection pipelines.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 3.2 Key inequality for random search

In this section, Bayesian optimization is treated as a concrete optimization object rather than
a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Hyperparameter Optimization, the phrase "Key inequality for random search" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **Bayesian optimization** is the part of Hyperparameter Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Bayesian optimization can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Bayesian optimization affects optimization but the model remains interpretable.
- A transformer training diagnostic where Bayesian optimization appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Bayesian optimization as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\lambda}^* \in \arg\min_{\boldsymbol{\lambda}\in\Lambda} \mathcal{V}(\mathcal{A}(\boldsymbol{\lambda}), \mathcal{D}_{\mathrm{val}})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving Bayesian
optimization, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Bayesian optimization visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Bayesian optimization is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- learning-rate, weight-decay, batch-size, and schedule tuning for LLM fine-tuning.
- Hyperband and ASHA for neural architecture and training-budget search.
- Bayesian optimization for expensive, low-dimensional continuous tuning.
- validation leakage prevention in model-selection pipelines.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 3.3 Role of Sobol initialization

In this section, successive halving is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Hyperparameter Optimization, the phrase "Role of Sobol initialization" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **successive halving** is the part of Hyperparameter Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where successive halving can be computed directly and compared with theory.
- A logistic-regression or softmax objective where successive halving affects optimization but the model remains interpretable.
- A transformer training diagnostic where successive halving appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating successive halving as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\lambda}^* \in \arg\min_{\boldsymbol{\lambda}\in\Lambda} \mathcal{V}(\mathcal{A}(\boldsymbol{\lambda}), \mathcal{D}_{\mathrm{val}})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving successive
halving, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes successive halving visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about successive halving is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- learning-rate, weight-decay, batch-size, and schedule tuning for LLM fine-tuning.
- Hyperband and ASHA for neural architecture and training-budget search.
- Bayesian optimization for expensive, low-dimensional continuous tuning.
- validation leakage prevention in model-selection pipelines.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 3.4 Proof template and what the proof actually buys

In this section, Hyperband is treated as a concrete optimization object rather than a slogan.
The goal is to understand how it changes the objective, the update rule, the convergence story,
and the diagnostics a practitioner should inspect when training a modern model. For
Hyperparameter Optimization, the phrase "Proof template and what the proof actually buys" means
a precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **Hyperband** is the part of Hyperparameter Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Hyperband can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Hyperband affects optimization but the model remains interpretable.
- A transformer training diagnostic where Hyperband appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Hyperband as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\lambda}^* \in \arg\min_{\boldsymbol{\lambda}\in\Lambda} \mathcal{V}(\mathcal{A}(\boldsymbol{\lambda}), \mathcal{D}_{\mathrm{val}})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving Hyperband,
and use the section assumptions to bound the change in objective value. If the assumption is
geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Hyperband visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Hyperband is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- learning-rate, weight-decay, batch-size, and schedule tuning for LLM fine-tuning.
- Hyperband and ASHA for neural architecture and training-budget search.
- Bayesian optimization for expensive, low-dimensional continuous tuning.
- validation leakage prevention in model-selection pipelines.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 3.5 Failure modes when assumptions are removed

In this section, ASHA is treated as a concrete optimization object rather than a slogan. The
goal is to understand how it changes the objective, the update rule, the convergence story, and
the diagnostics a practitioner should inspect when training a modern model. For Hyperparameter
Optimization, the phrase "Failure modes when assumptions are removed" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **ASHA** is the part of Hyperparameter Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where ASHA can be computed directly and compared with theory.
- A logistic-regression or softmax objective where ASHA affects optimization but the model remains interpretable.
- A transformer training diagnostic where ASHA appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating ASHA as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\lambda}^* \in \arg\min_{\boldsymbol{\lambda}\in\Lambda} \mathcal{V}(\mathcal{A}(\boldsymbol{\lambda}), \mathcal{D}_{\mathrm{val}})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving ASHA, and
use the section assumptions to bound the change in objective value. If the assumption is
geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes ASHA visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about ASHA is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- learning-rate, weight-decay, batch-size, and schedule tuning for LLM fine-tuning.
- Hyperband and ASHA for neural architecture and training-budget search.
- Bayesian optimization for expensive, low-dimensional continuous tuning.
- validation leakage prevention in model-selection pipelines.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 4. Core Theory II: Algorithms and Dynamics

This block develops core theory ii: algorithms and dynamics for Hyperparameter Optimization. It
keeps the scope local to this section while pointing forward when a neighboring topic owns the
full treatment.

### 4.1 Algorithmic update for surrogate model

In this section, Hyperband is treated as a concrete optimization object rather than a slogan.
The goal is to understand how it changes the objective, the update rule, the convergence story,
and the diagnostics a practitioner should inspect when training a modern model. For
Hyperparameter Optimization, the phrase "Algorithmic update for surrogate model" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **Hyperband** is the part of Hyperparameter Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Hyperband can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Hyperband affects optimization but the model remains interpretable.
- A transformer training diagnostic where Hyperband appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Hyperband as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\lambda}^* \in \arg\min_{\boldsymbol{\lambda}\in\Lambda} \mathcal{V}(\mathcal{A}(\boldsymbol{\lambda}), \mathcal{D}_{\mathrm{val}})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving Hyperband,
and use the section assumptions to bound the change in objective value. If the assumption is
geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Hyperband visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Hyperband is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- learning-rate, weight-decay, batch-size, and schedule tuning for LLM fine-tuning.
- Hyperband and ASHA for neural architecture and training-budget search.
- Bayesian optimization for expensive, low-dimensional continuous tuning.
- validation leakage prevention in model-selection pipelines.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 4.2 Stability role of Gaussian process

In this section, ASHA is treated as a concrete optimization object rather than a slogan. The
goal is to understand how it changes the objective, the update rule, the convergence story, and
the diagnostics a practitioner should inspect when training a modern model. For Hyperparameter
Optimization, the phrase "Stability role of Gaussian process" means a precise mathematical
habit: state the assumptions, write the update, identify what can be measured, and connect the
result to a real AI training decision.

> **Definition.**
>
> For this section, **ASHA** is the part of Hyperparameter Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where ASHA can be computed directly and compared with theory.
- A logistic-regression or softmax objective where ASHA affects optimization but the model remains interpretable.
- A transformer training diagnostic where ASHA appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating ASHA as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\lambda}^* \in \arg\min_{\boldsymbol{\lambda}\in\Lambda} \mathcal{V}(\mathcal{A}(\boldsymbol{\lambda}), \mathcal{D}_{\mathrm{val}})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving ASHA, and
use the section assumptions to bound the change in objective value. If the assumption is
geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes ASHA visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about ASHA is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- learning-rate, weight-decay, batch-size, and schedule tuning for LLM fine-tuning.
- Hyperband and ASHA for neural architecture and training-budget search.
- Bayesian optimization for expensive, low-dimensional continuous tuning.
- validation leakage prevention in model-selection pipelines.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 4.3 Rate or complexity controlled by expected improvement

In this section, BOHB is treated as a concrete optimization object rather than a slogan. The
goal is to understand how it changes the objective, the update rule, the convergence story, and
the diagnostics a practitioner should inspect when training a modern model. For Hyperparameter
Optimization, the phrase "Rate or complexity controlled by expected improvement" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **BOHB** is the part of Hyperparameter Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where BOHB can be computed directly and compared with theory.
- A logistic-regression or softmax objective where BOHB affects optimization but the model remains interpretable.
- A transformer training diagnostic where BOHB appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating BOHB as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\lambda}^* \in \arg\min_{\boldsymbol{\lambda}\in\Lambda} \mathcal{V}(\mathcal{A}(\boldsymbol{\lambda}), \mathcal{D}_{\mathrm{val}})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving BOHB, and
use the section assumptions to bound the change in objective value. If the assumption is
geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes BOHB visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about BOHB is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- learning-rate, weight-decay, batch-size, and schedule tuning for LLM fine-tuning.
- Hyperband and ASHA for neural architecture and training-budget search.
- Bayesian optimization for expensive, low-dimensional continuous tuning.
- validation leakage prevention in model-selection pipelines.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 4.4 Diagnostic interpretation of the update path

In this section, population-based training is treated as a concrete optimization object rather
than a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Hyperparameter Optimization, the phrase "Diagnostic interpretation of the update
path" means a precise mathematical habit: state the assumptions, write the update, identify what
can be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **population-based training** is the part of Hyperparameter Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where population-based training can be computed directly and compared with theory.
- A logistic-regression or softmax objective where population-based training affects optimization but the model remains interpretable.
- A transformer training diagnostic where population-based training appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating population-based training as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\lambda}^* \in \arg\min_{\boldsymbol{\lambda}\in\Lambda} \mathcal{V}(\mathcal{A}(\boldsymbol{\lambda}), \mathcal{D}_{\mathrm{val}})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
population-based training, and use the section assumptions to bound the change in objective
value. If the assumption is geometric, the proof turns a picture into an inequality. If the
assumption is stochastic, the proof takes conditional expectation before applying the bound. If
the assumption is algorithmic, the proof checks that the proposed update is a descent,
projection, or preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes population-based training visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about population-based training is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- learning-rate, weight-decay, batch-size, and schedule tuning for LLM fine-tuning.
- Hyperband and ASHA for neural architecture and training-budget search.
- Bayesian optimization for expensive, low-dimensional continuous tuning.
- validation leakage prevention in model-selection pipelines.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 4.5 Connection to the next section in the chapter

In this section, multi-objective tuning is treated as a concrete optimization object rather than
a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Hyperparameter Optimization, the phrase "Connection to the next section in the
chapter" means a precise mathematical habit: state the assumptions, write the update, identify
what can be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **multi-objective tuning** is the part of Hyperparameter Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where multi-objective tuning can be computed directly and compared with theory.
- A logistic-regression or softmax objective where multi-objective tuning affects optimization but the model remains interpretable.
- A transformer training diagnostic where multi-objective tuning appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating multi-objective tuning as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\lambda}^* \in \arg\min_{\boldsymbol{\lambda}\in\Lambda} \mathcal{V}(\mathcal{A}(\boldsymbol{\lambda}), \mathcal{D}_{\mathrm{val}})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
multi-objective tuning, and use the section assumptions to bound the change in objective value.
If the assumption is geometric, the proof turns a picture into an inequality. If the assumption
is stochastic, the proof takes conditional expectation before applying the bound. If the
assumption is algorithmic, the proof checks that the proposed update is a descent, projection,
or preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes multi-objective tuning visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about multi-objective tuning is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- learning-rate, weight-decay, batch-size, and schedule tuning for LLM fine-tuning.
- Hyperband and ASHA for neural architecture and training-budget search.
- Bayesian optimization for expensive, low-dimensional continuous tuning.
- validation leakage prevention in model-selection pipelines.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 5. Core Theory III: Practical Variants

This block develops core theory iii: practical variants for Hyperparameter Optimization. It
keeps the scope local to this section while pointing forward when a neighboring topic owns the
full treatment.

### 5.1 Variant built around upper confidence bound

In this section, population-based training is treated as a concrete optimization object rather
than a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Hyperparameter Optimization, the phrase "Variant built around upper confidence bound"
means a precise mathematical habit: state the assumptions, write the update, identify what can
be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **population-based training** is the part of Hyperparameter Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where population-based training can be computed directly and compared with theory.
- A logistic-regression or softmax objective where population-based training affects optimization but the model remains interpretable.
- A transformer training diagnostic where population-based training appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating population-based training as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\lambda}^* \in \arg\min_{\boldsymbol{\lambda}\in\Lambda} \mathcal{V}(\mathcal{A}(\boldsymbol{\lambda}), \mathcal{D}_{\mathrm{val}})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
population-based training, and use the section assumptions to bound the change in objective
value. If the assumption is geometric, the proof turns a picture into an inequality. If the
assumption is stochastic, the proof takes conditional expectation before applying the bound. If
the assumption is algorithmic, the proof checks that the proposed update is a descent,
projection, or preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes population-based training visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about population-based training is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- learning-rate, weight-decay, batch-size, and schedule tuning for LLM fine-tuning.
- Hyperband and ASHA for neural architecture and training-budget search.
- Bayesian optimization for expensive, low-dimensional continuous tuning.
- validation leakage prevention in model-selection pipelines.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 5.2 Variant built around Thompson sampling

In this section, multi-objective tuning is treated as a concrete optimization object rather than
a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Hyperparameter Optimization, the phrase "Variant built around Thompson sampling"
means a precise mathematical habit: state the assumptions, write the update, identify what can
be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **multi-objective tuning** is the part of Hyperparameter Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where multi-objective tuning can be computed directly and compared with theory.
- A logistic-regression or softmax objective where multi-objective tuning affects optimization but the model remains interpretable.
- A transformer training diagnostic where multi-objective tuning appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating multi-objective tuning as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\lambda}^* \in \arg\min_{\boldsymbol{\lambda}\in\Lambda} \mathcal{V}(\mathcal{A}(\boldsymbol{\lambda}), \mathcal{D}_{\mathrm{val}})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
multi-objective tuning, and use the section assumptions to bound the change in objective value.
If the assumption is geometric, the proof turns a picture into an inequality. If the assumption
is stochastic, the proof takes conditional expectation before applying the bound. If the
assumption is algorithmic, the proof checks that the proposed update is a descent, projection,
or preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes multi-objective tuning visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about multi-objective tuning is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- learning-rate, weight-decay, batch-size, and schedule tuning for LLM fine-tuning.
- Hyperband and ASHA for neural architecture and training-budget search.
- Bayesian optimization for expensive, low-dimensional continuous tuning.
- validation leakage prevention in model-selection pipelines.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 5.3 Variant built around Bayesian optimization

In this section, Pareto frontier is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Hyperparameter Optimization, the phrase "Variant built around Bayesian optimization" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **Pareto frontier** is the part of Hyperparameter Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Pareto frontier can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Pareto frontier affects optimization but the model remains interpretable.
- A transformer training diagnostic where Pareto frontier appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Pareto frontier as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\lambda}^* \in \arg\min_{\boldsymbol{\lambda}\in\Lambda} \mathcal{V}(\mathcal{A}(\boldsymbol{\lambda}), \mathcal{D}_{\mathrm{val}})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving Pareto
frontier, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Pareto frontier visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Pareto frontier is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- learning-rate, weight-decay, batch-size, and schedule tuning for LLM fine-tuning.
- Hyperband and ASHA for neural architecture and training-budget search.
- Bayesian optimization for expensive, low-dimensional continuous tuning.
- validation leakage prevention in model-selection pipelines.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 5.4 Implementation constraints and numerical stability

In this section, validation leakage is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Hyperparameter Optimization, the phrase "Implementation constraints and numerical stability"
means a precise mathematical habit: state the assumptions, write the update, identify what can
be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **validation leakage** is the part of Hyperparameter Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where validation leakage can be computed directly and compared with theory.
- A logistic-regression or softmax objective where validation leakage affects optimization but the model remains interpretable.
- A transformer training diagnostic where validation leakage appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating validation leakage as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\lambda}^* \in \arg\min_{\boldsymbol{\lambda}\in\Lambda} \mathcal{V}(\mathcal{A}(\boldsymbol{\lambda}), \mathcal{D}_{\mathrm{val}})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving validation
leakage, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes validation leakage visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about validation leakage is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- learning-rate, weight-decay, batch-size, and schedule tuning for LLM fine-tuning.
- Hyperband and ASHA for neural architecture and training-budget search.
- Bayesian optimization for expensive, low-dimensional continuous tuning.
- validation leakage prevention in model-selection pipelines.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 5.5 What belongs here versus neighboring sections

In this section, nested validation is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Hyperparameter Optimization, the phrase "What belongs here versus neighboring sections" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **nested validation** is the part of Hyperparameter Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where nested validation can be computed directly and compared with theory.
- A logistic-regression or softmax objective where nested validation affects optimization but the model remains interpretable.
- A transformer training diagnostic where nested validation appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating nested validation as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\lambda}^* \in \arg\min_{\boldsymbol{\lambda}\in\Lambda} \mathcal{V}(\mathcal{A}(\boldsymbol{\lambda}), \mathcal{D}_{\mathrm{val}})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving nested
validation, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes nested validation visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about nested validation is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- learning-rate, weight-decay, batch-size, and schedule tuning for LLM fine-tuning.
- Hyperband and ASHA for neural architecture and training-budget search.
- Bayesian optimization for expensive, low-dimensional continuous tuning.
- validation leakage prevention in model-selection pipelines.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 6. Advanced Topics

This block develops advanced topics for Hyperparameter Optimization. It keeps the scope local to
this section while pointing forward when a neighboring topic owns the full treatment.

### 6.1 Advanced view of successive halving

In this section, validation leakage is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Hyperparameter Optimization, the phrase "Advanced view of successive halving" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **validation leakage** is the part of Hyperparameter Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where validation leakage can be computed directly and compared with theory.
- A logistic-regression or softmax objective where validation leakage affects optimization but the model remains interpretable.
- A transformer training diagnostic where validation leakage appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating validation leakage as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\lambda}^* \in \arg\min_{\boldsymbol{\lambda}\in\Lambda} \mathcal{V}(\mathcal{A}(\boldsymbol{\lambda}), \mathcal{D}_{\mathrm{val}})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving validation
leakage, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes validation leakage visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about validation leakage is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- learning-rate, weight-decay, batch-size, and schedule tuning for LLM fine-tuning.
- Hyperband and ASHA for neural architecture and training-budget search.
- Bayesian optimization for expensive, low-dimensional continuous tuning.
- validation leakage prevention in model-selection pipelines.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 6.2 Advanced view of Hyperband

In this section, nested validation is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Hyperparameter Optimization, the phrase "Advanced view of Hyperband" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **nested validation** is the part of Hyperparameter Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where nested validation can be computed directly and compared with theory.
- A logistic-regression or softmax objective where nested validation affects optimization but the model remains interpretable.
- A transformer training diagnostic where nested validation appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating nested validation as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\lambda}^* \in \arg\min_{\boldsymbol{\lambda}\in\Lambda} \mathcal{V}(\mathcal{A}(\boldsymbol{\lambda}), \mathcal{D}_{\mathrm{val}})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving nested
validation, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes nested validation visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about nested validation is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- learning-rate, weight-decay, batch-size, and schedule tuning for LLM fine-tuning.
- Hyperband and ASHA for neural architecture and training-budget search.
- Bayesian optimization for expensive, low-dimensional continuous tuning.
- validation leakage prevention in model-selection pipelines.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 6.3 Advanced view of ASHA

In this section, LLM fine-tuning search is treated as a concrete optimization object rather than
a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Hyperparameter Optimization, the phrase "Advanced view of ASHA" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **LLM fine-tuning search** is the part of Hyperparameter Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where LLM fine-tuning search can be computed directly and compared with theory.
- A logistic-regression or softmax objective where LLM fine-tuning search affects optimization but the model remains interpretable.
- A transformer training diagnostic where LLM fine-tuning search appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating LLM fine-tuning search as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\lambda}^* \in \arg\min_{\boldsymbol{\lambda}\in\Lambda} \mathcal{V}(\mathcal{A}(\boldsymbol{\lambda}), \mathcal{D}_{\mathrm{val}})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving LLM
fine-tuning search, and use the section assumptions to bound the change in objective value. If
the assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes LLM fine-tuning search visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about LLM fine-tuning search is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- learning-rate, weight-decay, batch-size, and schedule tuning for LLM fine-tuning.
- Hyperband and ASHA for neural architecture and training-budget search.
- Bayesian optimization for expensive, low-dimensional continuous tuning.
- validation leakage prevention in model-selection pipelines.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 6.4 Infinite-dimensional or large-scale interpretation

In this section, configuration space is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Hyperparameter Optimization, the phrase "Infinite-dimensional or large-scale interpretation"
means a precise mathematical habit: state the assumptions, write the update, identify what can
be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **configuration space** is the part of Hyperparameter Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where configuration space can be computed directly and compared with theory.
- A logistic-regression or softmax objective where configuration space affects optimization but the model remains interpretable.
- A transformer training diagnostic where configuration space appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating configuration space as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\lambda}^* \in \arg\min_{\boldsymbol{\lambda}\in\Lambda} \mathcal{V}(\mathcal{A}(\boldsymbol{\lambda}), \mathcal{D}_{\mathrm{val}})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
configuration space, and use the section assumptions to bound the change in objective value. If
the assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes configuration space visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about configuration space is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- learning-rate, weight-decay, batch-size, and schedule tuning for LLM fine-tuning.
- Hyperband and ASHA for neural architecture and training-budget search.
- Bayesian optimization for expensive, low-dimensional continuous tuning.
- validation leakage prevention in model-selection pipelines.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 6.5 Open questions for frontier model training

In this section, conditional parameter is treated as a concrete optimization object rather than
a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Hyperparameter Optimization, the phrase "Open questions for frontier model training"
means a precise mathematical habit: state the assumptions, write the update, identify what can
be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **conditional parameter** is the part of Hyperparameter Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where conditional parameter can be computed directly and compared with theory.
- A logistic-regression or softmax objective where conditional parameter affects optimization but the model remains interpretable.
- A transformer training diagnostic where conditional parameter appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating conditional parameter as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\lambda}^* \in \arg\min_{\boldsymbol{\lambda}\in\Lambda} \mathcal{V}(\mathcal{A}(\boldsymbol{\lambda}), \mathcal{D}_{\mathrm{val}})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
conditional parameter, and use the section assumptions to bound the change in objective value.
If the assumption is geometric, the proof turns a picture into an inequality. If the assumption
is stochastic, the proof takes conditional expectation before applying the bound. If the
assumption is algorithmic, the proof checks that the proposed update is a descent, projection,
or preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes conditional parameter visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about conditional parameter is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- learning-rate, weight-decay, batch-size, and schedule tuning for LLM fine-tuning.
- Hyperband and ASHA for neural architecture and training-budget search.
- Bayesian optimization for expensive, low-dimensional continuous tuning.
- validation leakage prevention in model-selection pipelines.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 7. Applications in Machine Learning

This block develops applications in machine learning for Hyperparameter Optimization. It keeps
the scope local to this section while pointing forward when a neighboring topic owns the full
treatment.

### 7.1 learning-rate, weight-decay, batch-size, and schedule tuning for LLM fine-tuning

In this section, configuration space is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Hyperparameter Optimization, the phrase "learning-rate, weight-decay, batch-size, and schedule
tuning for LLM fine-tuning" means a precise mathematical habit: state the assumptions, write the
update, identify what can be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **configuration space** is the part of Hyperparameter Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where configuration space can be computed directly and compared with theory.
- A logistic-regression or softmax objective where configuration space affects optimization but the model remains interpretable.
- A transformer training diagnostic where configuration space appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating configuration space as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\lambda}^* \in \arg\min_{\boldsymbol{\lambda}\in\Lambda} \mathcal{V}(\mathcal{A}(\boldsymbol{\lambda}), \mathcal{D}_{\mathrm{val}})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
configuration space, and use the section assumptions to bound the change in objective value. If
the assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes configuration space visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about configuration space is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- learning-rate, weight-decay, batch-size, and schedule tuning for LLM fine-tuning.
- Hyperband and ASHA for neural architecture and training-budget search.
- Bayesian optimization for expensive, low-dimensional continuous tuning.
- validation leakage prevention in model-selection pipelines.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 7.2 Hyperband and ASHA for neural architecture and training-budget search

In this section, conditional parameter is treated as a concrete optimization object rather than
a slogan. The goal is to understand how it changes the objective, the update rule, the
convergence story, and the diagnostics a practitioner should inspect when training a modern
model. For Hyperparameter Optimization, the phrase "Hyperband and ASHA for neural architecture
and training-budget search" means a precise mathematical habit: state the assumptions, write the
update, identify what can be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **conditional parameter** is the part of Hyperparameter Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where conditional parameter can be computed directly and compared with theory.
- A logistic-regression or softmax objective where conditional parameter affects optimization but the model remains interpretable.
- A transformer training diagnostic where conditional parameter appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating conditional parameter as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\lambda}^* \in \arg\min_{\boldsymbol{\lambda}\in\Lambda} \mathcal{V}(\mathcal{A}(\boldsymbol{\lambda}), \mathcal{D}_{\mathrm{val}})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
conditional parameter, and use the section assumptions to bound the change in objective value.
If the assumption is geometric, the proof turns a picture into an inequality. If the assumption
is stochastic, the proof takes conditional expectation before applying the bound. If the
assumption is algorithmic, the proof checks that the proposed update is a descent, projection,
or preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes conditional parameter visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about conditional parameter is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- learning-rate, weight-decay, batch-size, and schedule tuning for LLM fine-tuning.
- Hyperband and ASHA for neural architecture and training-budget search.
- Bayesian optimization for expensive, low-dimensional continuous tuning.
- validation leakage prevention in model-selection pipelines.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 7.3 Bayesian optimization for expensive, low-dimensional continuous tuning

In this section, log-uniform sampling is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Hyperparameter Optimization, the phrase "Bayesian optimization for expensive, low-dimensional
continuous tuning" means a precise mathematical habit: state the assumptions, write the update,
identify what can be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **log-uniform sampling** is the part of Hyperparameter Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where log-uniform sampling can be computed directly and compared with theory.
- A logistic-regression or softmax objective where log-uniform sampling affects optimization but the model remains interpretable.
- A transformer training diagnostic where log-uniform sampling appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating log-uniform sampling as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\lambda}^* \in \arg\min_{\boldsymbol{\lambda}\in\Lambda} \mathcal{V}(\mathcal{A}(\boldsymbol{\lambda}), \mathcal{D}_{\mathrm{val}})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving
log-uniform sampling, and use the section assumptions to bound the change in objective value. If
the assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes log-uniform sampling visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about log-uniform sampling is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- learning-rate, weight-decay, batch-size, and schedule tuning for LLM fine-tuning.
- Hyperband and ASHA for neural architecture and training-budget search.
- Bayesian optimization for expensive, low-dimensional continuous tuning.
- validation leakage prevention in model-selection pipelines.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 7.4 validation leakage prevention in model-selection pipelines

In this section, grid search is treated as a concrete optimization object rather than a slogan.
The goal is to understand how it changes the objective, the update rule, the convergence story,
and the diagnostics a practitioner should inspect when training a modern model. For
Hyperparameter Optimization, the phrase "validation leakage prevention in model-selection
pipelines" means a precise mathematical habit: state the assumptions, write the update, identify
what can be measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **grid search** is the part of Hyperparameter Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where grid search can be computed directly and compared with theory.
- A logistic-regression or softmax objective where grid search affects optimization but the model remains interpretable.
- A transformer training diagnostic where grid search appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating grid search as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\lambda}^* \in \arg\min_{\boldsymbol{\lambda}\in\Lambda} \mathcal{V}(\mathcal{A}(\boldsymbol{\lambda}), \mathcal{D}_{\mathrm{val}})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving grid
search, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes grid search visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about grid search is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- learning-rate, weight-decay, batch-size, and schedule tuning for LLM fine-tuning.
- Hyperband and ASHA for neural architecture and training-budget search.
- Bayesian optimization for expensive, low-dimensional continuous tuning.
- validation leakage prevention in model-selection pipelines.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 7.5 Diagnostic checklist for real experiments

In this section, random search is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Hyperparameter Optimization, the phrase "Diagnostic checklist for real experiments" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **random search** is the part of Hyperparameter Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where random search can be computed directly and compared with theory.
- A logistic-regression or softmax objective where random search affects optimization but the model remains interpretable.
- A transformer training diagnostic where random search appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating random search as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\lambda}^* \in \arg\min_{\boldsymbol{\lambda}\in\Lambda} \mathcal{V}(\mathcal{A}(\boldsymbol{\lambda}), \mathcal{D}_{\mathrm{val}})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving random
search, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes random search visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about random search is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- learning-rate, weight-decay, batch-size, and schedule tuning for LLM fine-tuning.
- Hyperband and ASHA for neural architecture and training-budget search.
- Bayesian optimization for expensive, low-dimensional continuous tuning.
- validation leakage prevention in model-selection pipelines.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

## 8. Implementation and Diagnostics

This block develops implementation and diagnostics for Hyperparameter Optimization. It keeps the
scope local to this section while pointing forward when a neighboring topic owns the full
treatment.

### 8.1 Minimal NumPy experiment for BOHB

In this section, grid search is treated as a concrete optimization object rather than a slogan.
The goal is to understand how it changes the objective, the update rule, the convergence story,
and the diagnostics a practitioner should inspect when training a modern model. For
Hyperparameter Optimization, the phrase "Minimal NumPy experiment for BOHB" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **grid search** is the part of Hyperparameter Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where grid search can be computed directly and compared with theory.
- A logistic-regression or softmax objective where grid search affects optimization but the model remains interpretable.
- A transformer training diagnostic where grid search appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating grid search as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\lambda}^* \in \arg\min_{\boldsymbol{\lambda}\in\Lambda} \mathcal{V}(\mathcal{A}(\boldsymbol{\lambda}), \mathcal{D}_{\mathrm{val}})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving grid
search, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes grid search visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about grid search is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- learning-rate, weight-decay, batch-size, and schedule tuning for LLM fine-tuning.
- Hyperband and ASHA for neural architecture and training-budget search.
- Bayesian optimization for expensive, low-dimensional continuous tuning.
- validation leakage prevention in model-selection pipelines.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 8.2 Monitoring signal for population-based training

In this section, random search is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Hyperparameter Optimization, the phrase "Monitoring signal for population-based training" means
a precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **random search** is the part of Hyperparameter Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where random search can be computed directly and compared with theory.
- A logistic-regression or softmax objective where random search affects optimization but the model remains interpretable.
- A transformer training diagnostic where random search appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating random search as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\lambda}^* \in \arg\min_{\boldsymbol{\lambda}\in\Lambda} \mathcal{V}(\mathcal{A}(\boldsymbol{\lambda}), \mathcal{D}_{\mathrm{val}})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving random
search, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes random search visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about random search is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- learning-rate, weight-decay, batch-size, and schedule tuning for LLM fine-tuning.
- Hyperband and ASHA for neural architecture and training-budget search.
- Bayesian optimization for expensive, low-dimensional continuous tuning.
- validation leakage prevention in model-selection pipelines.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 8.3 Failure signature for multi-objective tuning

In this section, Sobol initialization is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Hyperparameter Optimization, the phrase "Failure signature for multi-objective tuning" means a
precise mathematical habit: state the assumptions, write the update, identify what can be
measured, and connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **Sobol initialization** is the part of Hyperparameter Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Sobol initialization can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Sobol initialization affects optimization but the model remains interpretable.
- A transformer training diagnostic where Sobol initialization appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Sobol initialization as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\lambda}^* \in \arg\min_{\boldsymbol{\lambda}\in\Lambda} \mathcal{V}(\mathcal{A}(\boldsymbol{\lambda}), \mathcal{D}_{\mathrm{val}})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving Sobol
initialization, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Sobol initialization visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Sobol initialization is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- learning-rate, weight-decay, batch-size, and schedule tuning for LLM fine-tuning.
- Hyperband and ASHA for neural architecture and training-budget search.
- Bayesian optimization for expensive, low-dimensional continuous tuning.
- validation leakage prevention in model-selection pipelines.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 8.4 Framework-level implementation pattern

In this section, surrogate model is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Hyperparameter Optimization, the phrase "Framework-level implementation pattern" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **surrogate model** is the part of Hyperparameter Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where surrogate model can be computed directly and compared with theory.
- A logistic-regression or softmax objective where surrogate model affects optimization but the model remains interpretable.
- A transformer training diagnostic where surrogate model appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating surrogate model as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\lambda}^* \in \arg\min_{\boldsymbol{\lambda}\in\Lambda} \mathcal{V}(\mathcal{A}(\boldsymbol{\lambda}), \mathcal{D}_{\mathrm{val}})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving surrogate
model, and use the section assumptions to bound the change in objective value. If the assumption
is geometric, the proof turns a picture into an inequality. If the assumption is stochastic, the
proof takes conditional expectation before applying the bound. If the assumption is algorithmic,
the proof checks that the proposed update is a descent, projection, or preconditioning step.
This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes surrogate model visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about surrogate model is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- learning-rate, weight-decay, batch-size, and schedule tuning for LLM fine-tuning.
- Hyperband and ASHA for neural architecture and training-budget search.
- Bayesian optimization for expensive, low-dimensional continuous tuning.
- validation leakage prevention in model-selection pipelines.

Local scope boundary:
This subsection may reference neighboring material, but the full canonical treatment stays in
its own folder. For example, stochastic gradient noise belongs to [Stochastic
Optimization](../05-Stochastic-Optimization/notes.md), external schedule shapes belong to
[Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md), and cross-entropy as an
information measure belongs to
[Cross-Entropy](../../09-Information-Theory/04-Cross-Entropy/notes.md).

### 8.5 Reproducibility and logging checklist

In this section, Gaussian process is treated as a concrete optimization object rather than a
slogan. The goal is to understand how it changes the objective, the update rule, the convergence
story, and the diagnostics a practitioner should inspect when training a modern model. For
Hyperparameter Optimization, the phrase "Reproducibility and logging checklist" means a precise
mathematical habit: state the assumptions, write the update, identify what can be measured, and
connect the result to a real AI training decision.

> **Definition.**
>
> For this section, **Gaussian process** is the part of Hyperparameter Optimization that controls how the objective, feasible region, or update rule behaves under the assumptions currently in force.
>
> Symbolically, we track it through $f$, $\boldsymbol{\theta}$, $\eta$, $\nabla f(\boldsymbol{\theta})$, and any auxiliary state used by the algorithm.

Examples:
- A small synthetic quadratic where Gaussian process can be computed directly and compared with theory.
- A logistic-regression or softmax objective where Gaussian process affects optimization but the model remains interpretable.
- A transformer training diagnostic where Gaussian process appears through gradient norms, update norms, curvature, or validation loss.

Non-examples:
- Treating Gaussian process as a hyperparameter recipe without checking the objective assumptions.
- Inferring global behavior from one noisy minibatch when the section requires a population or full-batch statement.

Useful formula:

$$
\boldsymbol{\lambda}^* \in \arg\min_{\boldsymbol{\lambda}\in\Lambda} \mathcal{V}(\mathcal{A}(\boldsymbol{\lambda}), \mathcal{D}_{\mathrm{val}})
$$

Proof sketch or reasoning pattern:

Start with the local model around $\boldsymbol{\theta}_t$, isolate the term involving Gaussian
process, and use the section assumptions to bound the change in objective value. If the
assumption is geometric, the proof turns a picture into an inequality. If the assumption is
stochastic, the proof takes conditional expectation before applying the bound. If the assumption
is algorithmic, the proof checks that the proposed update is a descent, projection, or
preconditioning step. This pattern is reusable across optimization theory.

Implementation consequence:
- Log a metric that makes Gaussian process visible; otherwise a training run can fail while the scalar loss hides the cause.
- Compare the measured update with the mathematical update below before blaming data or architecture.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

- Keep units straight: parameter norm, gradient norm, update norm, objective value, and validation metric are different objects.

Diagnostic questions:
- Which assumption about Gaussian process is most fragile in the current training setup?
- What number would you log to catch the failure one thousand steps before divergence?

AI connection:
- learning-rate, weight-decay, batch-size, and schedule tuning for LLM fine-tuning.
- Hyperband and ASHA for neural architecture and training-budget search.
- Bayesian optimization for expensive, low-dimensional continuous tuning.
- validation leakage prevention in model-selection pipelines.

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

1. **Exercise 1 [*] - Log-Uniform Sampling**
   (a) Define log-uniform sampling using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

2. **Exercise 2 [*] - Random Search**
   (a) Define random search using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

3. **Exercise 3 [*] - Surrogate Model**
   (a) Define surrogate model using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

4. **Exercise 4 [**] - Expected Improvement**
   (a) Define expected improvement using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

5. **Exercise 5 [**] - Thompson Sampling**
   (a) Define Thompson sampling using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

6. **Exercise 6 [**] - Successive Halving**
   (a) Define successive halving using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

7. **Exercise 7 [**] - Asha**
   (a) Define ASHA using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

8. **Exercise 8 [***] - Population-Based Training**
   (a) Define population-based training using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

9. **Exercise 9 [***] - Pareto Frontier**
   (a) Define Pareto frontier using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

10. **Exercise 10 [***] - Nested Validation**
   (a) Define nested validation using the notation of this repository.
   (b) Give three valid examples and two non-examples.
   (c) Derive the relevant update or inequality shown below.

$$
\boldsymbol{\lambda}_{t+1} = \arg\max_{\boldsymbol{\lambda}\in\Lambda} a_t(\boldsymbol{\lambda})
$$

   (d) Implement a NumPy check on a synthetic two-dimensional objective.
   (e) Explain what metric you would log in a real LLM or fine-tuning run.

## 11. Why This Matters for AI (2026 Perspective)

| Concept | AI Impact |
| --- | --- |
| configuration space | learning-rate, weight-decay, batch-size, and schedule tuning for LLM fine-tuning |
| conditional parameter | Hyperband and ASHA for neural architecture and training-budget search |
| log-uniform sampling | Bayesian optimization for expensive, low-dimensional continuous tuning |
| grid search | validation leakage prevention in model-selection pipelines |
| random search | learning-rate, weight-decay, batch-size, and schedule tuning for LLM fine-tuning |
| Sobol initialization | Hyperband and ASHA for neural architecture and training-budget search |
| surrogate model | Bayesian optimization for expensive, low-dimensional continuous tuning |
| Gaussian process | validation leakage prevention in model-selection pipelines |
| expected improvement | learning-rate, weight-decay, batch-size, and schedule tuning for LLM fine-tuning |
| upper confidence bound | Hyperband and ASHA for neural architecture and training-budget search |

## 12. Conceptual Bridge

Hyperparameter Optimization sits inside a chain. Earlier sections give the calculus,
probability, and linear algebra needed to write the objective and interpret the update. Later
sections use this material to reason about noisy gradients, adaptive state, regularization,
tuning, schedules, and finally information-theoretic losses.

Backward link: [Regularization Methods](../08-Regularization-Methods/notes.md) supplies the
immediate prerequisite vocabulary.

Forward link: [Learning Rate Schedules](../10-Learning-Rate-Schedules/notes.md) uses this
section as a building block.

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
| >> 09-Hyperparameter-Optimization  Hyperparameter Optimization |
|    10-Learning-Rate-Schedules      Learning Rate Schedules |
+------------------------------------------------------------+
```

## Appendix A. Extended Derivation and Diagnostic Cards

## References

- Bergstra and Bengio, Random Search for Hyper-Parameter Optimization.
- Snoek, Larochelle, and Adams, Practical Bayesian Optimization of Machine Learning Algorithms.
- Li et al., Hyperband.
- Jaderberg et al., Population Based Training of Neural Networks.
- Goodfellow, Bengio, and Courville, Deep Learning.
- Bottou, Curtis, and Nocedal, Optimization Methods for Large-Scale Machine Learning.
- PyTorch optimizer and scheduler documentation.
- Optax documentation for composable optimizer transformations.
