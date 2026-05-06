[Back to Curriculum](../../README.md) | [Previous: PAC Learning](../01-PAC-Learning/notes.md) | [Next: Bias Variance Tradeoff](../03-Bias-Variance-Tradeoff/notes.md)

---

# VC Dimension

> _"A class is complex when it can realize many worlds on the same finite sample."_

## Overview

VC dimension measures the combinatorial capacity of binary hypothesis classes and explains when infinite classes can still generalize.

Statistical learning theory is the chapter where finite samples meet future performance. It does not ask only whether a model performed well on observed data. It asks what assumptions let observed performance control unobserved risk.

This section is written in LaTeX Markdown. Inline mathematics uses `$...$`, and display equations use `$$...$$`. The notes emphasize theorem-level objects such as risk, hypothesis class, sample complexity, capacity, and confidence.

## Prerequisites

- [Sets and Logic](../../01-Mathematical-Foundations/02-Sets-and-Logic/notes.md)
- [PAC Learning](../01-PAC-Learning/notes.md)
- [Linear Algebra Basics](../../02-Linear-Algebra-Basics/01-Vectors-and-Spaces/notes.md)
- [Robustness and Distribution Shift](../../17-Evaluation-and-Reliability/03-Robustness-and-Distribution-Shift/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Executable demonstrations for vc dimension |
| [exercises.ipynb](exercises.ipynb) | Graded practice for vc dimension |

## Learning Objectives

After completing this section, you will be able to:

- Define true risk, empirical risk, and generalization gap
- State PAC guarantees using $\epsilon$ and $\delta$
- Derive finite-class sample complexity with union bounds
- Compute simple VC dimensions and growth functions
- Explain the bias-variance decomposition under squared loss
- Compare finite-class, VC, margin, norm, and Rademacher bounds
- Estimate simple Rademacher complexity by simulation
- Separate learnability theory from empirical benchmarking
- Identify when classical assumptions fail for modern AI systems
- Use learning-theory notation consistently with the repo guide

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 complexity as ability to shatter points](#11-complexity-as-ability-to-shatter-points)
  - [1.2 why parameter count is not enough](#12-why-parameter-count-is-not-enough)
  - [1.3 geometric examples](#13-geometric-examples)
  - [1.4 memorization versus generalization](#14-memorization-versus-generalization)
  - [1.5 VC theory as infinite-class PAC](#15-vc-theory-as-infiniteclass-pac)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 dichotomy](#21-dichotomy)
  - [2.2 shattering](#22-shattering)
  - [2.3 growth function $\Pi_{\mathcal{H}}(m)$](#23-growth-function)
  - [2.4 VC dimension $\operatorname{VCdim}(\mathcal{H})$](#24-vc-dimension)
  - [2.5 Sauer-Shelah lemma preview](#25-sauershelah-lemma-preview)
- [3. Computing VC Dimension](#3-computing-vc-dimension)
  - [3.1 thresholds on $\mathbb{R}$](#31-thresholds-on)
  - [3.2 intervals](#32-intervals)
  - [3.3 linear separators in $\mathbb{R}^d$](#33-linear-separators-in)
  - [3.4 axis-aligned rectangles](#34-axisaligned-rectangles)
  - [3.5 finite classes](#35-finite-classes)
- [4. Growth Functions](#4-growth-functions)
  - [4.1 dichotomy counts](#41-dichotomy-counts)
  - [4.2 polynomial versus exponential growth](#42-polynomial-versus-exponential-growth)
  - [4.3 Sauer-Shelah bound](#43-sauershelah-bound)
  - [4.4 break points](#44-break-points)
  - [4.5 sample complexity role](#45-sample-complexity-role)
- [5. VC Generalization](#5-vc-generalization)
  - [5.1 uniform convergence for VC classes](#51-uniform-convergence-for-vc-classes)
  - [5.2 realizable bound](#52-realizable-bound)
  - [5.3 agnostic bound](#53-agnostic-bound)
  - [5.4 structural risk minimization](#54-structural-risk-minimization)
  - [5.5 margin preview](#55-margin-preview)
- [6. ML and LLM Connections](#6-ml-and-llm-connections)
  - [6.1 why deep nets can shatter](#61-why-deep-nets-can-shatter)
  - [6.2 capacity control beyond VC](#62-capacity-control-beyond-vc)
  - [6.3 linear probes](#63-linear-probes)
  - [6.4 memorization audits](#64-memorization-audits)
  - [6.5 modern overparameterization caveat](#65-modern-overparameterization-caveat)
- [7. Common Mistakes](#7-common-mistakes)
- [8. Exercises](#8-exercises)
- [9. Why This Matters for AI](#9-why-this-matters-for-ai)
- [10. Conceptual Bridge](#10-conceptual-bridge)
- [References](#references)

---

## 1. Intuition

Intuition develops the part of vc dimension specified by the approved Chapter 21 table of contents. The emphasis is statistical learning theory, not generic statistics, optimization recipes, or benchmark operations.

### 1.1 complexity as ability to shatter points

Complexity as ability to shatter points is part of the canonical scope of VC Dimension. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is shattering, growth functions, Sauer-Shelah bounds, VC sample complexity, and capacity control beyond parameter count. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$\Pi_{\mathcal{H}}(m)=\max_{\mathbf{x}^{(1)},\ldots,\mathbf{x}^{(m)}}\lvert\{(h(\mathbf{x}^{(1)}),\ldots,h(\mathbf{x}^{(m)})):h\in\mathcal{H}\}\rvert.$$

The formula should be read operationally. For complexity as ability to shatter points, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of complexity as ability to shatter points:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for complexity as ability to shatter points is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

A useful ASCII picture for this subsection is:

```text
unknown distribution D
        | sample S
        v
 empirical learner h_S ----> empirical risk L_S(h_S)
        |
        v
 true deployment risk L_D(h_S)
```

The gap between the last two quantities is the reason this chapter exists. Chapter 17 measures it empirically with benchmark protocols. Chapter 21 studies when mathematics can control it before all future examples are observed.

Implementation note for the companion notebook: complexity as ability to shatter points will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using complexity as ability to shatter points responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 1.2 why parameter count is not enough

Why parameter count is not enough is part of the canonical scope of VC Dimension. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is shattering, growth functions, Sauer-Shelah bounds, VC sample complexity, and capacity control beyond parameter count. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$\operatorname{VCdim}(\mathcal{H})=\max\{m:\Pi_{\mathcal{H}}(m)=2^m\}.$$

The formula should be read operationally. For why parameter count is not enough, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of why parameter count is not enough:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for why parameter count is not enough is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

A useful ASCII picture for this subsection is:

```text
unknown distribution D
        | sample S
        v
 empirical learner h_S ----> empirical risk L_S(h_S)
        |
        v
 true deployment risk L_D(h_S)
```

The gap between the last two quantities is the reason this chapter exists. Chapter 17 measures it empirically with benchmark protocols. Chapter 21 studies when mathematics can control it before all future examples are observed.

Implementation note for the companion notebook: why parameter count is not enough will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using why parameter count is not enough responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 1.3 geometric examples

Geometric examples is part of the canonical scope of VC Dimension. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is shattering, growth functions, Sauer-Shelah bounds, VC sample complexity, and capacity control beyond parameter count. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$\Pi_{\mathcal{H}}(m)\le \sum_{i=0}^{d}\binom{m}{i}\le \left(\frac{em}{d}\right)^d.$$

The formula should be read operationally. For geometric examples, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of geometric examples:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for geometric examples is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

A useful ASCII picture for this subsection is:

```text
unknown distribution D
        | sample S
        v
 empirical learner h_S ----> empirical risk L_S(h_S)
        |
        v
 true deployment risk L_D(h_S)
```

The gap between the last two quantities is the reason this chapter exists. Chapter 17 measures it empirically with benchmark protocols. Chapter 21 studies when mathematics can control it before all future examples are observed.

Implementation note for the companion notebook: geometric examples will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using geometric examples responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 1.4 memorization versus generalization

Memorization versus generalization is part of the canonical scope of VC Dimension. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is shattering, growth functions, Sauer-Shelah bounds, VC sample complexity, and capacity control beyond parameter count. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$m = O\left(\frac{d\log(1/\epsilon)+\log(1/\delta)}{\epsilon}\right).$$

The formula should be read operationally. For memorization versus generalization, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of memorization versus generalization:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for memorization versus generalization is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

A useful ASCII picture for this subsection is:

```text
unknown distribution D
        | sample S
        v
 empirical learner h_S ----> empirical risk L_S(h_S)
        |
        v
 true deployment risk L_D(h_S)
```

The gap between the last two quantities is the reason this chapter exists. Chapter 17 measures it empirically with benchmark protocols. Chapter 21 studies when mathematics can control it before all future examples are observed.

Implementation note for the companion notebook: memorization versus generalization will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using memorization versus generalization responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 1.5 VC theory as infinite-class PAC

Vc theory as infinite-class pac is part of the canonical scope of VC Dimension. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is shattering, growth functions, Sauer-Shelah bounds, VC sample complexity, and capacity control beyond parameter count. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$\Pi_{\mathcal{H}}(m)=\max_{\mathbf{x}^{(1)},\ldots,\mathbf{x}^{(m)}}\lvert\{(h(\mathbf{x}^{(1)}),\ldots,h(\mathbf{x}^{(m)})):h\in\mathcal{H}\}\rvert.$$

The formula should be read operationally. For vc theory as infinite-class pac, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of vc theory as infinite-class pac:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for vc theory as infinite-class pac is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

A useful ASCII picture for this subsection is:

```text
unknown distribution D
        | sample S
        v
 empirical learner h_S ----> empirical risk L_S(h_S)
        |
        v
 true deployment risk L_D(h_S)
```

The gap between the last two quantities is the reason this chapter exists. Chapter 17 measures it empirically with benchmark protocols. Chapter 21 studies when mathematics can control it before all future examples are observed.

Implementation note for the companion notebook: vc theory as infinite-class pac will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using vc theory as infinite-class pac responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

## 2. Formal Definitions

Formal Definitions develops the part of vc dimension specified by the approved Chapter 21 table of contents. The emphasis is statistical learning theory, not generic statistics, optimization recipes, or benchmark operations.

### 2.1 dichotomy

Dichotomy is part of the canonical scope of VC Dimension. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is shattering, growth functions, Sauer-Shelah bounds, VC sample complexity, and capacity control beyond parameter count. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$\operatorname{VCdim}(\mathcal{H})=\max\{m:\Pi_{\mathcal{H}}(m)=2^m\}.$$

The formula should be read operationally. For dichotomy, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of dichotomy:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for dichotomy is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

A useful ASCII picture for this subsection is:

```text
unknown distribution D
        | sample S
        v
 empirical learner h_S ----> empirical risk L_S(h_S)
        |
        v
 true deployment risk L_D(h_S)
```

The gap between the last two quantities is the reason this chapter exists. Chapter 17 measures it empirically with benchmark protocols. Chapter 21 studies when mathematics can control it before all future examples are observed.

Implementation note for the companion notebook: dichotomy will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using dichotomy responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 2.2 shattering

Shattering is part of the canonical scope of VC Dimension. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is shattering, growth functions, Sauer-Shelah bounds, VC sample complexity, and capacity control beyond parameter count. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$\Pi_{\mathcal{H}}(m)\le \sum_{i=0}^{d}\binom{m}{i}\le \left(\frac{em}{d}\right)^d.$$

The formula should be read operationally. For shattering, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of shattering:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for shattering is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

A useful ASCII picture for this subsection is:

```text
unknown distribution D
        | sample S
        v
 empirical learner h_S ----> empirical risk L_S(h_S)
        |
        v
 true deployment risk L_D(h_S)
```

The gap between the last two quantities is the reason this chapter exists. Chapter 17 measures it empirically with benchmark protocols. Chapter 21 studies when mathematics can control it before all future examples are observed.

Implementation note for the companion notebook: shattering will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using shattering responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 2.3 growth function $\Pi_{\mathcal{H}}(m)$

Growth function $\pi_{\mathcal{h}}(m)$ is part of the canonical scope of VC Dimension. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is shattering, growth functions, Sauer-Shelah bounds, VC sample complexity, and capacity control beyond parameter count. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$m = O\left(\frac{d\log(1/\epsilon)+\log(1/\delta)}{\epsilon}\right).$$

The formula should be read operationally. For growth function $\pi_{\mathcal{h}}(m)$, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of growth function $\pi_{\mathcal{h}}(m)$:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for growth function $\pi_{\mathcal{h}}(m)$ is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

A useful ASCII picture for this subsection is:

```text
unknown distribution D
        | sample S
        v
 empirical learner h_S ----> empirical risk L_S(h_S)
        |
        v
 true deployment risk L_D(h_S)
```

The gap between the last two quantities is the reason this chapter exists. Chapter 17 measures it empirically with benchmark protocols. Chapter 21 studies when mathematics can control it before all future examples are observed.

Implementation note for the companion notebook: growth function $\pi_{\mathcal{h}}(m)$ will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using growth function $\pi_{\mathcal{h}}(m)$ responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 2.4 VC dimension $\operatorname{VCdim}(\mathcal{H})$

Vc dimension $\operatorname{vcdim}(\mathcal{h})$ is part of the canonical scope of VC Dimension. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is shattering, growth functions, Sauer-Shelah bounds, VC sample complexity, and capacity control beyond parameter count. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$\Pi_{\mathcal{H}}(m)=\max_{\mathbf{x}^{(1)},\ldots,\mathbf{x}^{(m)}}\lvert\{(h(\mathbf{x}^{(1)}),\ldots,h(\mathbf{x}^{(m)})):h\in\mathcal{H}\}\rvert.$$

The formula should be read operationally. For vc dimension $\operatorname{vcdim}(\mathcal{h})$, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of vc dimension $\operatorname{vcdim}(\mathcal{h})$:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for vc dimension $\operatorname{vcdim}(\mathcal{h})$ is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

A useful ASCII picture for this subsection is:

```text
unknown distribution D
        | sample S
        v
 empirical learner h_S ----> empirical risk L_S(h_S)
        |
        v
 true deployment risk L_D(h_S)
```

The gap between the last two quantities is the reason this chapter exists. Chapter 17 measures it empirically with benchmark protocols. Chapter 21 studies when mathematics can control it before all future examples are observed.

Implementation note for the companion notebook: vc dimension $\operatorname{vcdim}(\mathcal{h})$ will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using vc dimension $\operatorname{vcdim}(\mathcal{h})$ responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 2.5 Sauer-Shelah lemma preview

Sauer-shelah lemma preview is part of the canonical scope of VC Dimension. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is shattering, growth functions, Sauer-Shelah bounds, VC sample complexity, and capacity control beyond parameter count. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$\operatorname{VCdim}(\mathcal{H})=\max\{m:\Pi_{\mathcal{H}}(m)=2^m\}.$$

The formula should be read operationally. For sauer-shelah lemma preview, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of sauer-shelah lemma preview:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for sauer-shelah lemma preview is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

A useful ASCII picture for this subsection is:

```text
unknown distribution D
        | sample S
        v
 empirical learner h_S ----> empirical risk L_S(h_S)
        |
        v
 true deployment risk L_D(h_S)
```

The gap between the last two quantities is the reason this chapter exists. Chapter 17 measures it empirically with benchmark protocols. Chapter 21 studies when mathematics can control it before all future examples are observed.

Implementation note for the companion notebook: sauer-shelah lemma preview will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using sauer-shelah lemma preview responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

## 3. Computing VC Dimension

Computing VC Dimension develops the part of vc dimension specified by the approved Chapter 21 table of contents. The emphasis is statistical learning theory, not generic statistics, optimization recipes, or benchmark operations.

### 3.1 thresholds on $\mathbb{R}$

Thresholds on $\mathbb{r}$ is part of the canonical scope of VC Dimension. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is shattering, growth functions, Sauer-Shelah bounds, VC sample complexity, and capacity control beyond parameter count. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$\Pi_{\mathcal{H}}(m)\le \sum_{i=0}^{d}\binom{m}{i}\le \left(\frac{em}{d}\right)^d.$$

The formula should be read operationally. For thresholds on $\mathbb{r}$, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of thresholds on $\mathbb{r}$:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for thresholds on $\mathbb{r}$ is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

A useful ASCII picture for this subsection is:

```text
unknown distribution D
        | sample S
        v
 empirical learner h_S ----> empirical risk L_S(h_S)
        |
        v
 true deployment risk L_D(h_S)
```

The gap between the last two quantities is the reason this chapter exists. Chapter 17 measures it empirically with benchmark protocols. Chapter 21 studies when mathematics can control it before all future examples are observed.

Implementation note for the companion notebook: thresholds on $\mathbb{r}$ will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using thresholds on $\mathbb{r}$ responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 3.2 intervals

Intervals is part of the canonical scope of VC Dimension. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is shattering, growth functions, Sauer-Shelah bounds, VC sample complexity, and capacity control beyond parameter count. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$m = O\left(\frac{d\log(1/\epsilon)+\log(1/\delta)}{\epsilon}\right).$$

The formula should be read operationally. For intervals, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of intervals:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for intervals is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

A useful ASCII picture for this subsection is:

```text
unknown distribution D
        | sample S
        v
 empirical learner h_S ----> empirical risk L_S(h_S)
        |
        v
 true deployment risk L_D(h_S)
```

The gap between the last two quantities is the reason this chapter exists. Chapter 17 measures it empirically with benchmark protocols. Chapter 21 studies when mathematics can control it before all future examples are observed.

Implementation note for the companion notebook: intervals will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using intervals responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 3.3 linear separators in $\mathbb{R}^d$

Linear separators in $\mathbb{r}^d$ is part of the canonical scope of VC Dimension. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is shattering, growth functions, Sauer-Shelah bounds, VC sample complexity, and capacity control beyond parameter count. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$\Pi_{\mathcal{H}}(m)=\max_{\mathbf{x}^{(1)},\ldots,\mathbf{x}^{(m)}}\lvert\{(h(\mathbf{x}^{(1)}),\ldots,h(\mathbf{x}^{(m)})):h\in\mathcal{H}\}\rvert.$$

The formula should be read operationally. For linear separators in $\mathbb{r}^d$, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of linear separators in $\mathbb{r}^d$:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for linear separators in $\mathbb{r}^d$ is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

A useful ASCII picture for this subsection is:

```text
unknown distribution D
        | sample S
        v
 empirical learner h_S ----> empirical risk L_S(h_S)
        |
        v
 true deployment risk L_D(h_S)
```

The gap between the last two quantities is the reason this chapter exists. Chapter 17 measures it empirically with benchmark protocols. Chapter 21 studies when mathematics can control it before all future examples are observed.

Implementation note for the companion notebook: linear separators in $\mathbb{r}^d$ will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using linear separators in $\mathbb{r}^d$ responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 3.4 axis-aligned rectangles

Axis-aligned rectangles is part of the canonical scope of VC Dimension. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is shattering, growth functions, Sauer-Shelah bounds, VC sample complexity, and capacity control beyond parameter count. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$\operatorname{VCdim}(\mathcal{H})=\max\{m:\Pi_{\mathcal{H}}(m)=2^m\}.$$

The formula should be read operationally. For axis-aligned rectangles, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of axis-aligned rectangles:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for axis-aligned rectangles is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

A useful ASCII picture for this subsection is:

```text
unknown distribution D
        | sample S
        v
 empirical learner h_S ----> empirical risk L_S(h_S)
        |
        v
 true deployment risk L_D(h_S)
```

The gap between the last two quantities is the reason this chapter exists. Chapter 17 measures it empirically with benchmark protocols. Chapter 21 studies when mathematics can control it before all future examples are observed.

Implementation note for the companion notebook: axis-aligned rectangles will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using axis-aligned rectangles responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 3.5 finite classes

Finite classes is part of the canonical scope of VC Dimension. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is shattering, growth functions, Sauer-Shelah bounds, VC sample complexity, and capacity control beyond parameter count. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$\Pi_{\mathcal{H}}(m)\le \sum_{i=0}^{d}\binom{m}{i}\le \left(\frac{em}{d}\right)^d.$$

The formula should be read operationally. For finite classes, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of finite classes:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for finite classes is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

A useful ASCII picture for this subsection is:

```text
unknown distribution D
        | sample S
        v
 empirical learner h_S ----> empirical risk L_S(h_S)
        |
        v
 true deployment risk L_D(h_S)
```

The gap between the last two quantities is the reason this chapter exists. Chapter 17 measures it empirically with benchmark protocols. Chapter 21 studies when mathematics can control it before all future examples are observed.

Implementation note for the companion notebook: finite classes will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using finite classes responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

## 4. Growth Functions

Growth Functions develops the part of vc dimension specified by the approved Chapter 21 table of contents. The emphasis is statistical learning theory, not generic statistics, optimization recipes, or benchmark operations.

### 4.1 dichotomy counts

Dichotomy counts is part of the canonical scope of VC Dimension. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is shattering, growth functions, Sauer-Shelah bounds, VC sample complexity, and capacity control beyond parameter count. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$m = O\left(\frac{d\log(1/\epsilon)+\log(1/\delta)}{\epsilon}\right).$$

The formula should be read operationally. For dichotomy counts, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of dichotomy counts:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for dichotomy counts is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

A useful ASCII picture for this subsection is:

```text
unknown distribution D
        | sample S
        v
 empirical learner h_S ----> empirical risk L_S(h_S)
        |
        v
 true deployment risk L_D(h_S)
```

The gap between the last two quantities is the reason this chapter exists. Chapter 17 measures it empirically with benchmark protocols. Chapter 21 studies when mathematics can control it before all future examples are observed.

Implementation note for the companion notebook: dichotomy counts will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using dichotomy counts responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 4.2 polynomial versus exponential growth

Polynomial versus exponential growth is part of the canonical scope of VC Dimension. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is shattering, growth functions, Sauer-Shelah bounds, VC sample complexity, and capacity control beyond parameter count. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$\Pi_{\mathcal{H}}(m)=\max_{\mathbf{x}^{(1)},\ldots,\mathbf{x}^{(m)}}\lvert\{(h(\mathbf{x}^{(1)}),\ldots,h(\mathbf{x}^{(m)})):h\in\mathcal{H}\}\rvert.$$

The formula should be read operationally. For polynomial versus exponential growth, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of polynomial versus exponential growth:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for polynomial versus exponential growth is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

A useful ASCII picture for this subsection is:

```text
unknown distribution D
        | sample S
        v
 empirical learner h_S ----> empirical risk L_S(h_S)
        |
        v
 true deployment risk L_D(h_S)
```

The gap between the last two quantities is the reason this chapter exists. Chapter 17 measures it empirically with benchmark protocols. Chapter 21 studies when mathematics can control it before all future examples are observed.

Implementation note for the companion notebook: polynomial versus exponential growth will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using polynomial versus exponential growth responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 4.3 Sauer-Shelah bound

Sauer-shelah bound is part of the canonical scope of VC Dimension. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is shattering, growth functions, Sauer-Shelah bounds, VC sample complexity, and capacity control beyond parameter count. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$\operatorname{VCdim}(\mathcal{H})=\max\{m:\Pi_{\mathcal{H}}(m)=2^m\}.$$

The formula should be read operationally. For sauer-shelah bound, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of sauer-shelah bound:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for sauer-shelah bound is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

A useful ASCII picture for this subsection is:

```text
unknown distribution D
        | sample S
        v
 empirical learner h_S ----> empirical risk L_S(h_S)
        |
        v
 true deployment risk L_D(h_S)
```

The gap between the last two quantities is the reason this chapter exists. Chapter 17 measures it empirically with benchmark protocols. Chapter 21 studies when mathematics can control it before all future examples are observed.

Implementation note for the companion notebook: sauer-shelah bound will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using sauer-shelah bound responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 4.4 break points

Break points is part of the canonical scope of VC Dimension. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is shattering, growth functions, Sauer-Shelah bounds, VC sample complexity, and capacity control beyond parameter count. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$\Pi_{\mathcal{H}}(m)\le \sum_{i=0}^{d}\binom{m}{i}\le \left(\frac{em}{d}\right)^d.$$

The formula should be read operationally. For break points, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of break points:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for break points is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

A useful ASCII picture for this subsection is:

```text
unknown distribution D
        | sample S
        v
 empirical learner h_S ----> empirical risk L_S(h_S)
        |
        v
 true deployment risk L_D(h_S)
```

The gap between the last two quantities is the reason this chapter exists. Chapter 17 measures it empirically with benchmark protocols. Chapter 21 studies when mathematics can control it before all future examples are observed.

Implementation note for the companion notebook: break points will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using break points responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 4.5 sample complexity role

Sample complexity role is part of the canonical scope of VC Dimension. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is shattering, growth functions, Sauer-Shelah bounds, VC sample complexity, and capacity control beyond parameter count. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$m = O\left(\frac{d\log(1/\epsilon)+\log(1/\delta)}{\epsilon}\right).$$

The formula should be read operationally. For sample complexity role, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of sample complexity role:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for sample complexity role is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

A useful ASCII picture for this subsection is:

```text
unknown distribution D
        | sample S
        v
 empirical learner h_S ----> empirical risk L_S(h_S)
        |
        v
 true deployment risk L_D(h_S)
```

The gap between the last two quantities is the reason this chapter exists. Chapter 17 measures it empirically with benchmark protocols. Chapter 21 studies when mathematics can control it before all future examples are observed.

Implementation note for the companion notebook: sample complexity role will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using sample complexity role responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

## 5. VC Generalization

VC Generalization develops the part of vc dimension specified by the approved Chapter 21 table of contents. The emphasis is statistical learning theory, not generic statistics, optimization recipes, or benchmark operations.

### 5.1 uniform convergence for VC classes

Uniform convergence for vc classes is part of the canonical scope of VC Dimension. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is shattering, growth functions, Sauer-Shelah bounds, VC sample complexity, and capacity control beyond parameter count. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$\Pi_{\mathcal{H}}(m)=\max_{\mathbf{x}^{(1)},\ldots,\mathbf{x}^{(m)}}\lvert\{(h(\mathbf{x}^{(1)}),\ldots,h(\mathbf{x}^{(m)})):h\in\mathcal{H}\}\rvert.$$

The formula should be read operationally. For uniform convergence for vc classes, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of uniform convergence for vc classes:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for uniform convergence for vc classes is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

A useful ASCII picture for this subsection is:

```text
unknown distribution D
        | sample S
        v
 empirical learner h_S ----> empirical risk L_S(h_S)
        |
        v
 true deployment risk L_D(h_S)
```

The gap between the last two quantities is the reason this chapter exists. Chapter 17 measures it empirically with benchmark protocols. Chapter 21 studies when mathematics can control it before all future examples are observed.

Implementation note for the companion notebook: uniform convergence for vc classes will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using uniform convergence for vc classes responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 5.2 realizable bound

Realizable bound is part of the canonical scope of VC Dimension. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is shattering, growth functions, Sauer-Shelah bounds, VC sample complexity, and capacity control beyond parameter count. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$\operatorname{VCdim}(\mathcal{H})=\max\{m:\Pi_{\mathcal{H}}(m)=2^m\}.$$

The formula should be read operationally. For realizable bound, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of realizable bound:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for realizable bound is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

A useful ASCII picture for this subsection is:

```text
unknown distribution D
        | sample S
        v
 empirical learner h_S ----> empirical risk L_S(h_S)
        |
        v
 true deployment risk L_D(h_S)
```

The gap between the last two quantities is the reason this chapter exists. Chapter 17 measures it empirically with benchmark protocols. Chapter 21 studies when mathematics can control it before all future examples are observed.

Implementation note for the companion notebook: realizable bound will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using realizable bound responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 5.3 agnostic bound

Agnostic bound is part of the canonical scope of VC Dimension. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is shattering, growth functions, Sauer-Shelah bounds, VC sample complexity, and capacity control beyond parameter count. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$\Pi_{\mathcal{H}}(m)\le \sum_{i=0}^{d}\binom{m}{i}\le \left(\frac{em}{d}\right)^d.$$

The formula should be read operationally. For agnostic bound, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of agnostic bound:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for agnostic bound is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

A useful ASCII picture for this subsection is:

```text
unknown distribution D
        | sample S
        v
 empirical learner h_S ----> empirical risk L_S(h_S)
        |
        v
 true deployment risk L_D(h_S)
```

The gap between the last two quantities is the reason this chapter exists. Chapter 17 measures it empirically with benchmark protocols. Chapter 21 studies when mathematics can control it before all future examples are observed.

Implementation note for the companion notebook: agnostic bound will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using agnostic bound responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 5.4 structural risk minimization

Structural risk minimization is part of the canonical scope of VC Dimension. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is shattering, growth functions, Sauer-Shelah bounds, VC sample complexity, and capacity control beyond parameter count. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$m = O\left(\frac{d\log(1/\epsilon)+\log(1/\delta)}{\epsilon}\right).$$

The formula should be read operationally. For structural risk minimization, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of structural risk minimization:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for structural risk minimization is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

A useful ASCII picture for this subsection is:

```text
unknown distribution D
        | sample S
        v
 empirical learner h_S ----> empirical risk L_S(h_S)
        |
        v
 true deployment risk L_D(h_S)
```

The gap between the last two quantities is the reason this chapter exists. Chapter 17 measures it empirically with benchmark protocols. Chapter 21 studies when mathematics can control it before all future examples are observed.

Implementation note for the companion notebook: structural risk minimization will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using structural risk minimization responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 5.5 margin preview

Margin preview is part of the canonical scope of VC Dimension. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is shattering, growth functions, Sauer-Shelah bounds, VC sample complexity, and capacity control beyond parameter count. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$\Pi_{\mathcal{H}}(m)=\max_{\mathbf{x}^{(1)},\ldots,\mathbf{x}^{(m)}}\lvert\{(h(\mathbf{x}^{(1)}),\ldots,h(\mathbf{x}^{(m)})):h\in\mathcal{H}\}\rvert.$$

The formula should be read operationally. For margin preview, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of margin preview:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for margin preview is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

A useful ASCII picture for this subsection is:

```text
unknown distribution D
        | sample S
        v
 empirical learner h_S ----> empirical risk L_S(h_S)
        |
        v
 true deployment risk L_D(h_S)
```

The gap between the last two quantities is the reason this chapter exists. Chapter 17 measures it empirically with benchmark protocols. Chapter 21 studies when mathematics can control it before all future examples are observed.

Implementation note for the companion notebook: margin preview will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using margin preview responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

## 6. ML and LLM Connections

ML and LLM Connections develops the part of vc dimension specified by the approved Chapter 21 table of contents. The emphasis is statistical learning theory, not generic statistics, optimization recipes, or benchmark operations.

### 6.1 why deep nets can shatter

Why deep nets can shatter is part of the canonical scope of VC Dimension. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is shattering, growth functions, Sauer-Shelah bounds, VC sample complexity, and capacity control beyond parameter count. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$\operatorname{VCdim}(\mathcal{H})=\max\{m:\Pi_{\mathcal{H}}(m)=2^m\}.$$

The formula should be read operationally. For why deep nets can shatter, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of why deep nets can shatter:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for why deep nets can shatter is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

A useful ASCII picture for this subsection is:

```text
unknown distribution D
        | sample S
        v
 empirical learner h_S ----> empirical risk L_S(h_S)
        |
        v
 true deployment risk L_D(h_S)
```

The gap between the last two quantities is the reason this chapter exists. Chapter 17 measures it empirically with benchmark protocols. Chapter 21 studies when mathematics can control it before all future examples are observed.

Implementation note for the companion notebook: why deep nets can shatter will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using why deep nets can shatter responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 6.2 capacity control beyond VC

Capacity control beyond vc is part of the canonical scope of VC Dimension. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is shattering, growth functions, Sauer-Shelah bounds, VC sample complexity, and capacity control beyond parameter count. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$\Pi_{\mathcal{H}}(m)\le \sum_{i=0}^{d}\binom{m}{i}\le \left(\frac{em}{d}\right)^d.$$

The formula should be read operationally. For capacity control beyond vc, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of capacity control beyond vc:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for capacity control beyond vc is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

A useful ASCII picture for this subsection is:

```text
unknown distribution D
        | sample S
        v
 empirical learner h_S ----> empirical risk L_S(h_S)
        |
        v
 true deployment risk L_D(h_S)
```

The gap between the last two quantities is the reason this chapter exists. Chapter 17 measures it empirically with benchmark protocols. Chapter 21 studies when mathematics can control it before all future examples are observed.

Implementation note for the companion notebook: capacity control beyond vc will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using capacity control beyond vc responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 6.3 linear probes

Linear probes is part of the canonical scope of VC Dimension. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is shattering, growth functions, Sauer-Shelah bounds, VC sample complexity, and capacity control beyond parameter count. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$m = O\left(\frac{d\log(1/\epsilon)+\log(1/\delta)}{\epsilon}\right).$$

The formula should be read operationally. For linear probes, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of linear probes:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for linear probes is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

A useful ASCII picture for this subsection is:

```text
unknown distribution D
        | sample S
        v
 empirical learner h_S ----> empirical risk L_S(h_S)
        |
        v
 true deployment risk L_D(h_S)
```

The gap between the last two quantities is the reason this chapter exists. Chapter 17 measures it empirically with benchmark protocols. Chapter 21 studies when mathematics can control it before all future examples are observed.

Implementation note for the companion notebook: linear probes will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using linear probes responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 6.4 memorization audits

Memorization audits is part of the canonical scope of VC Dimension. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is shattering, growth functions, Sauer-Shelah bounds, VC sample complexity, and capacity control beyond parameter count. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$\Pi_{\mathcal{H}}(m)=\max_{\mathbf{x}^{(1)},\ldots,\mathbf{x}^{(m)}}\lvert\{(h(\mathbf{x}^{(1)}),\ldots,h(\mathbf{x}^{(m)})):h\in\mathcal{H}\}\rvert.$$

The formula should be read operationally. For memorization audits, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of memorization audits:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for memorization audits is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

A useful ASCII picture for this subsection is:

```text
unknown distribution D
        | sample S
        v
 empirical learner h_S ----> empirical risk L_S(h_S)
        |
        v
 true deployment risk L_D(h_S)
```

The gap between the last two quantities is the reason this chapter exists. Chapter 17 measures it empirically with benchmark protocols. Chapter 21 studies when mathematics can control it before all future examples are observed.

Implementation note for the companion notebook: memorization audits will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using memorization audits responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 6.5 modern overparameterization caveat

Modern overparameterization caveat is part of the canonical scope of VC Dimension. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is shattering, growth functions, Sauer-Shelah bounds, VC sample complexity, and capacity control beyond parameter count. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$\operatorname{VCdim}(\mathcal{H})=\max\{m:\Pi_{\mathcal{H}}(m)=2^m\}.$$

The formula should be read operationally. For modern overparameterization caveat, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of modern overparameterization caveat:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for modern overparameterization caveat is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

A useful ASCII picture for this subsection is:

```text
unknown distribution D
        | sample S
        v
 empirical learner h_S ----> empirical risk L_S(h_S)
        |
        v
 true deployment risk L_D(h_S)
```

The gap between the last two quantities is the reason this chapter exists. Chapter 17 measures it empirically with benchmark protocols. Chapter 21 studies when mathematics can control it before all future examples are observed.

Implementation note for the companion notebook: modern overparameterization caveat will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using modern overparameterization caveat responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

## 7. Common Mistakes

| # | Mistake | Why It Is Wrong | Fix |
| --- | --- | --- | --- |
| 1 | Confusing empirical risk with true risk | A low training or validation error is still an estimate from finite data. | Always state the sample, distributional assumption, and confidence level. |
| 2 | Treating PAC as an algorithm | PAC is a guarantee framework, not a specific optimizer. | Separate the learner, hypothesis class, loss, and sample-complexity statement. |
| 3 | Using parameter count as capacity | VC dimension and Rademacher complexity can differ sharply from raw parameter count. | Analyze the class behavior on samples, margins, norms, or data-dependent complexity. |
| 4 | Ignoring the confidence parameter | An error tolerance without a probability statement is not a PAC-style guarantee. | Track both $\epsilon$ and $\delta$ in every sample-complexity claim. |
| 5 | Assuming bounds must be tight to be useful | Loose bounds can still reveal dependence on sample size, class complexity, and confidence. | Interpret bounds qualitatively when numerical values are conservative. |
| 6 | Applying realizable results to noisy data | Consistency assumptions fail when labels are stochastic or corrupted. | Use agnostic learning and excess risk when noise is present. |
| 7 | Over-reading bias-variance curves | The classical U-shape does not fully explain interpolation and deep learning. | Use it as one decomposition, then connect to modern overparameterization carefully. |
| 8 | Replacing evaluation with theory | Theoretical guarantees do not remove the need for benchmark and production checks. | Use Chapter 17 evaluation as empirical evidence and Chapter 21 as mathematical context. |
| 9 | Mixing causal and statistical claims | Generalization bounds do not identify interventions or counterfactuals. | Leave causal claims to Chapter 22 and state distributional assumptions explicitly. |
| 10 | Forgetting the loss-composition step | Bounds for hypotheses may not directly apply to composed losses. | Bound the induced loss class or use contraction-style arguments when appropriate. |

## 8. Exercises

1. (*) Work through a learning-theory question for vc dimension.
   - (a) Define the sample, distribution, hypothesis class, and loss.
   - (b) State the relevant risk or complexity quantity.
   - (c) Derive or compute the bound requested by the problem.
   - (d) Interpret the result for an ML or LLM system.

2. (*) Work through a learning-theory question for vc dimension.
   - (a) Define the sample, distribution, hypothesis class, and loss.
   - (b) State the relevant risk or complexity quantity.
   - (c) Derive or compute the bound requested by the problem.
   - (d) Interpret the result for an ML or LLM system.

3. (*) Work through a learning-theory question for vc dimension.
   - (a) Define the sample, distribution, hypothesis class, and loss.
   - (b) State the relevant risk or complexity quantity.
   - (c) Derive or compute the bound requested by the problem.
   - (d) Interpret the result for an ML or LLM system.

4. (**) Work through a learning-theory question for vc dimension.
   - (a) Define the sample, distribution, hypothesis class, and loss.
   - (b) State the relevant risk or complexity quantity.
   - (c) Derive or compute the bound requested by the problem.
   - (d) Interpret the result for an ML or LLM system.

5. (**) Work through a learning-theory question for vc dimension.
   - (a) Define the sample, distribution, hypothesis class, and loss.
   - (b) State the relevant risk or complexity quantity.
   - (c) Derive or compute the bound requested by the problem.
   - (d) Interpret the result for an ML or LLM system.

6. (**) Work through a learning-theory question for vc dimension.
   - (a) Define the sample, distribution, hypothesis class, and loss.
   - (b) State the relevant risk or complexity quantity.
   - (c) Derive or compute the bound requested by the problem.
   - (d) Interpret the result for an ML or LLM system.

7. (***) Work through a learning-theory question for vc dimension.
   - (a) Define the sample, distribution, hypothesis class, and loss.
   - (b) State the relevant risk or complexity quantity.
   - (c) Derive or compute the bound requested by the problem.
   - (d) Interpret the result for an ML or LLM system.

8. (***) Work through a learning-theory question for vc dimension.
   - (a) Define the sample, distribution, hypothesis class, and loss.
   - (b) State the relevant risk or complexity quantity.
   - (c) Derive or compute the bound requested by the problem.
   - (d) Interpret the result for an ML or LLM system.

9. (***) Work through a learning-theory question for vc dimension.
   - (a) Define the sample, distribution, hypothesis class, and loss.
   - (b) State the relevant risk or complexity quantity.
   - (c) Derive or compute the bound requested by the problem.
   - (d) Interpret the result for an ML or LLM system.

10. (***) Work through a learning-theory question for vc dimension.
   - (a) Define the sample, distribution, hypothesis class, and loss.
   - (b) State the relevant risk or complexity quantity.
   - (c) Derive or compute the bound requested by the problem.
   - (d) Interpret the result for an ML or LLM system.

## 9. Why This Matters for AI

| Concept | AI Impact |
| --- | --- |
| PAC guarantee | Clarifies what sample size can and cannot certify |
| VC dimension | Explains capacity beyond naive parameter counting |
| Bias-variance decomposition | Separates approximation, estimation, and noise effects |
| Generalization gap | Connects training behavior to future deployment risk |
| Rademacher complexity | Gives a data-dependent view of capacity |
| Confidence parameter | Prevents overconfident claims from small samples |
| Margin or norm bound | Links geometry and regularization to generalization |
| Theory-practice gap | Teaches caution when applying classical theorems to foundation models |

## 10. Conceptual Bridge

VC Dimension belongs in the research-frontier phase because modern AI systems force us to ask why enormous models generalize from finite data. Earlier chapters gave probability, statistics, optimization, evaluation, and production practice. This chapter turns those ingredients into mathematical learnability questions.

The backward bridge is concentration and risk estimation. Chapter 6 supplies probability tools, Chapter 7 supplies statistical estimation language, Chapter 8 supplies optimization procedures, and Chapter 17 supplies empirical evaluation discipline. Chapter 21 asks when those observed quantities can support future-risk claims.

The forward bridge is causal inference. Generalization bounds still reason about distributions, not interventions. Chapter 22 will ask what happens when the data-generating process changes because an action is taken. That is a different kind of uncertainty.

```text
+--------------------------------------------------------------+
| probability -> statistics -> evaluation -> learning theory   |
|      sample S       empirical risk       true risk           |
|      class H        capacity             confidence          |
| learning theory -> causal inference -> research frontiers    |
+--------------------------------------------------------------+
```

## References

- Vapnik and Chervonenkis. On the uniform convergence of relative frequencies of events to their probabilities. https://link.springer.com/article/10.1007/BF01065038
- Blumer et al.. Learnability and the Vapnik-Chervonenkis dimension. https://dl.acm.org/doi/10.1145/76359.76371
- Shalev-Shwartz and Ben-David. Understanding Machine Learning. https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/
- Mohri, Rostamizadeh, and Talwalkar. Foundations of Machine Learning. https://cs.nyu.edu/~mohri/mlbook/
