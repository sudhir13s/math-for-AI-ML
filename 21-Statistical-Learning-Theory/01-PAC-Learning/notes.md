[Back to Curriculum](../../README.md) | [Previous: Wavelets](../../20-Fourier-Analysis-and-Signal-Processing/05-Wavelets/notes.md) | [Next: VC Dimension](../02-VC-Dimension/notes.md)

---

# PAC Learning

> _"A learner is useful when its future error can be bounded before the future is seen."_

## Overview

PAC learning formalizes learnability by asking how many samples are enough to make small true error probable.

Statistical learning theory is the chapter where finite samples meet future performance. It does not ask only whether a model performed well on observed data. It asks what assumptions let observed performance control unobserved risk.

This section is written in LaTeX Markdown. Inline mathematics uses `$...$`, and display equations use `$$...$$`. The notes emphasize theorem-level objects such as risk, hypothesis class, sample complexity, capacity, and confidence.

## Prerequisites

- [Concentration Inequalities](../../06-Probability-Theory/05-Concentration-Inequalities/notes.md)
- [Estimation Theory](../../07-Statistics/02-Estimation-Theory/notes.md)
- [Hypothesis Testing](../../07-Statistics/03-Hypothesis-Testing/notes.md)
- [Capability Benchmarks](../../17-Evaluation-and-Reliability/01-Capability-Benchmarks/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Executable demonstrations for pac learning |
| [exercises.ipynb](exercises.ipynb) | Graded practice for pac learning |

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
  - [1.1 learning as selecting a hypothesis from data](#11-learning-as-selecting-a-hypothesis-from-data)
  - [1.2 probably approximately correct guarantee](#12-probably-approximately-correct-guarantee)
  - [1.3 error confidence and sample size](#13-error-confidence-and-sample-size)
  - [1.4 why PAC is distribution-free](#14-why-pac-is-distributionfree)
  - [1.5 what PAC does not promise](#15-what-pac-does-not-promise)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 instance space $\mathcal{X}$](#21-instance-space)
  - [2.2 label space $\mathcal{Y}$](#22-label-space)
  - [2.3 hypothesis class $\mathcal{H}$](#23-hypothesis-class)
  - [2.4 true risk $L_{\mathcal{D}}(h)$ and empirical risk $L_S(h)$](#24-true-risk-and-empirical-risk)
  - [2.5 PAC learner with $(\epsilon,\delta)$](#25-pac-learner-with)
- [3. Realizable PAC Learning](#3-realizable-pac-learning)
  - [3.1 consistency assumption](#31-consistency-assumption)
  - [3.2 finite hypothesis class bound](#32-finite-hypothesis-class-bound)
  - [3.3 union bound proof sketch](#33-union-bound-proof-sketch)
  - [3.4 sample complexity $m_{\mathcal{H}}(\epsilon,\delta)$](#34-sample-complexity)
  - [3.5 consistent ERM](#35-consistent-erm)
- [4. Agnostic PAC Learning](#4-agnostic-pac-learning)
  - [4.1 Bayes error and approximation error](#41-bayes-error-and-approximation-error)
  - [4.2 excess risk](#42-excess-risk)
  - [4.3 agnostic ERM](#43-agnostic-erm)
  - [4.4 finite-class agnostic bound](#44-finiteclass-agnostic-bound)
  - [4.5 noisy labels](#45-noisy-labels)
- [5. Sample Complexity](#5-sample-complexity)
  - [5.1 dependence on $\epsilon$](#51-dependence-on)
  - [5.2 dependence on $\delta$](#52-dependence-on)
  - [5.3 logarithmic class-size dependence](#53-logarithmic-classsize-dependence)
  - [5.4 infinite-class motivation](#54-infiniteclass-motivation)
  - [5.5 AI-scale interpretation](#55-aiscale-interpretation)
- [6. Applications in ML and LLMs](#6-applications-in-ml-and-llms)
  - [6.1 classifier selection](#61-classifier-selection)
  - [6.2 prompt classifier reliability](#62-prompt-classifier-reliability)
  - [6.3 safety classifier sample needs](#63-safety-classifier-sample-needs)
  - [6.4 evaluation-set sizing preview](#64-evaluationset-sizing-preview)
  - [6.5 limits for deep nets](#65-limits-for-deep-nets)
- [7. Common Mistakes](#7-common-mistakes)
- [8. Exercises](#8-exercises)
- [9. Why This Matters for AI](#9-why-this-matters-for-ai)
- [10. Conceptual Bridge](#10-conceptual-bridge)
- [References](#references)

---

## 1. Intuition

Intuition develops the part of pac learning specified by the approved Chapter 21 table of contents. The emphasis is statistical learning theory, not generic statistics, optimization recipes, or benchmark operations.

### 1.1 learning as selecting a hypothesis from data

Learning as selecting a hypothesis from data is part of the canonical scope of PAC Learning. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is probably approximately correct guarantees, finite-class sample complexity, realizable and agnostic learning, and distribution-free learnability. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$L_{\mathcal{D}}(h)=P_{(\mathbf{x},y)\sim\mathcal{D}}[h(\mathbf{x})\ne y].$$

The formula should be read operationally. For learning as selecting a hypothesis from data, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of learning as selecting a hypothesis from data:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for learning as selecting a hypothesis from data is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: learning as selecting a hypothesis from data will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using learning as selecting a hypothesis from data responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 1.2 probably approximately correct guarantee

Probably approximately correct guarantee is part of the canonical scope of PAC Learning. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is probably approximately correct guarantees, finite-class sample complexity, realizable and agnostic learning, and distribution-free learnability. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$L_S(h)=\frac{1}{m}\sum_{i=1}^{m}\mathbb{1}[h(\mathbf{x}^{(i)})\ne y^{(i)}].$$

The formula should be read operationally. For probably approximately correct guarantee, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of probably approximately correct guarantee:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for probably approximately correct guarantee is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: probably approximately correct guarantee will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using probably approximately correct guarantee responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 1.3 error confidence and sample size

Error confidence and sample size is part of the canonical scope of PAC Learning. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is probably approximately correct guarantees, finite-class sample complexity, realizable and agnostic learning, and distribution-free learnability. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$m \ge \frac{1}{\epsilon}\left(\log\lvert\mathcal{H}\rvert + \log\frac{1}{\delta}\right).$$

The formula should be read operationally. For error confidence and sample size, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of error confidence and sample size:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for error confidence and sample size is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: error confidence and sample size will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using error confidence and sample size responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 1.4 why PAC is distribution-free

Why pac is distribution-free is part of the canonical scope of PAC Learning. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is probably approximately correct guarantees, finite-class sample complexity, realizable and agnostic learning, and distribution-free learnability. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$P[L_{\mathcal{D}}(h_S)\le \epsilon] \ge 1-\delta.$$

The formula should be read operationally. For why pac is distribution-free, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of why pac is distribution-free:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for why pac is distribution-free is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: why pac is distribution-free will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using why pac is distribution-free responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 1.5 what PAC does not promise

What pac does not promise is part of the canonical scope of PAC Learning. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is probably approximately correct guarantees, finite-class sample complexity, realizable and agnostic learning, and distribution-free learnability. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$L_{\mathcal{D}}(h)=P_{(\mathbf{x},y)\sim\mathcal{D}}[h(\mathbf{x})\ne y].$$

The formula should be read operationally. For what pac does not promise, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of what pac does not promise:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for what pac does not promise is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: what pac does not promise will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using what pac does not promise responsibly:

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

Formal Definitions develops the part of pac learning specified by the approved Chapter 21 table of contents. The emphasis is statistical learning theory, not generic statistics, optimization recipes, or benchmark operations.

### 2.1 instance space $\mathcal{X}$

Instance space $\mathcal{x}$ is part of the canonical scope of PAC Learning. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is probably approximately correct guarantees, finite-class sample complexity, realizable and agnostic learning, and distribution-free learnability. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$L_S(h)=\frac{1}{m}\sum_{i=1}^{m}\mathbb{1}[h(\mathbf{x}^{(i)})\ne y^{(i)}].$$

The formula should be read operationally. For instance space $\mathcal{x}$, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of instance space $\mathcal{x}$:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for instance space $\mathcal{x}$ is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: instance space $\mathcal{x}$ will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using instance space $\mathcal{x}$ responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 2.2 label space $\mathcal{Y}$

Label space $\mathcal{y}$ is part of the canonical scope of PAC Learning. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is probably approximately correct guarantees, finite-class sample complexity, realizable and agnostic learning, and distribution-free learnability. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$m \ge \frac{1}{\epsilon}\left(\log\lvert\mathcal{H}\rvert + \log\frac{1}{\delta}\right).$$

The formula should be read operationally. For label space $\mathcal{y}$, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of label space $\mathcal{y}$:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for label space $\mathcal{y}$ is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: label space $\mathcal{y}$ will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using label space $\mathcal{y}$ responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 2.3 hypothesis class $\mathcal{H}$

Hypothesis class $\mathcal{h}$ is part of the canonical scope of PAC Learning. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is probably approximately correct guarantees, finite-class sample complexity, realizable and agnostic learning, and distribution-free learnability. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$P[L_{\mathcal{D}}(h_S)\le \epsilon] \ge 1-\delta.$$

The formula should be read operationally. For hypothesis class $\mathcal{h}$, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of hypothesis class $\mathcal{h}$:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for hypothesis class $\mathcal{h}$ is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: hypothesis class $\mathcal{h}$ will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using hypothesis class $\mathcal{h}$ responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 2.4 true risk $L_{\mathcal{D}}(h)$ and empirical risk $L_S(h)$

True risk $l_{\mathcal{d}}(h)$ and empirical risk $l_s(h)$ is part of the canonical scope of PAC Learning. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is probably approximately correct guarantees, finite-class sample complexity, realizable and agnostic learning, and distribution-free learnability. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$L_{\mathcal{D}}(h)=P_{(\mathbf{x},y)\sim\mathcal{D}}[h(\mathbf{x})\ne y].$$

The formula should be read operationally. For true risk $l_{\mathcal{d}}(h)$ and empirical risk $l_s(h)$, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of true risk $l_{\mathcal{d}}(h)$ and empirical risk $l_s(h)$:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for true risk $l_{\mathcal{d}}(h)$ and empirical risk $l_s(h)$ is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: true risk $l_{\mathcal{d}}(h)$ and empirical risk $l_s(h)$ will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using true risk $l_{\mathcal{d}}(h)$ and empirical risk $l_s(h)$ responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 2.5 PAC learner with $(\epsilon,\delta)$

Pac learner with $(\epsilon,\delta)$ is part of the canonical scope of PAC Learning. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is probably approximately correct guarantees, finite-class sample complexity, realizable and agnostic learning, and distribution-free learnability. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$L_S(h)=\frac{1}{m}\sum_{i=1}^{m}\mathbb{1}[h(\mathbf{x}^{(i)})\ne y^{(i)}].$$

The formula should be read operationally. For pac learner with $(\epsilon,\delta)$, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of pac learner with $(\epsilon,\delta)$:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for pac learner with $(\epsilon,\delta)$ is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: pac learner with $(\epsilon,\delta)$ will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using pac learner with $(\epsilon,\delta)$ responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

## 3. Realizable PAC Learning

Realizable PAC Learning develops the part of pac learning specified by the approved Chapter 21 table of contents. The emphasis is statistical learning theory, not generic statistics, optimization recipes, or benchmark operations.

### 3.1 consistency assumption

Consistency assumption is part of the canonical scope of PAC Learning. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is probably approximately correct guarantees, finite-class sample complexity, realizable and agnostic learning, and distribution-free learnability. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$m \ge \frac{1}{\epsilon}\left(\log\lvert\mathcal{H}\rvert + \log\frac{1}{\delta}\right).$$

The formula should be read operationally. For consistency assumption, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of consistency assumption:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for consistency assumption is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: consistency assumption will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using consistency assumption responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 3.2 finite hypothesis class bound

Finite hypothesis class bound is part of the canonical scope of PAC Learning. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is probably approximately correct guarantees, finite-class sample complexity, realizable and agnostic learning, and distribution-free learnability. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$P[L_{\mathcal{D}}(h_S)\le \epsilon] \ge 1-\delta.$$

The formula should be read operationally. For finite hypothesis class bound, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of finite hypothesis class bound:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for finite hypothesis class bound is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: finite hypothesis class bound will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using finite hypothesis class bound responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 3.3 union bound proof sketch

Union bound proof sketch is part of the canonical scope of PAC Learning. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is probably approximately correct guarantees, finite-class sample complexity, realizable and agnostic learning, and distribution-free learnability. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$L_{\mathcal{D}}(h)=P_{(\mathbf{x},y)\sim\mathcal{D}}[h(\mathbf{x})\ne y].$$

The formula should be read operationally. For union bound proof sketch, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of union bound proof sketch:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for union bound proof sketch is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: union bound proof sketch will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using union bound proof sketch responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 3.4 sample complexity $m_{\mathcal{H}}(\epsilon,\delta)$

Sample complexity $m_{\mathcal{h}}(\epsilon,\delta)$ is part of the canonical scope of PAC Learning. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is probably approximately correct guarantees, finite-class sample complexity, realizable and agnostic learning, and distribution-free learnability. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$L_S(h)=\frac{1}{m}\sum_{i=1}^{m}\mathbb{1}[h(\mathbf{x}^{(i)})\ne y^{(i)}].$$

The formula should be read operationally. For sample complexity $m_{\mathcal{h}}(\epsilon,\delta)$, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of sample complexity $m_{\mathcal{h}}(\epsilon,\delta)$:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for sample complexity $m_{\mathcal{h}}(\epsilon,\delta)$ is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: sample complexity $m_{\mathcal{h}}(\epsilon,\delta)$ will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using sample complexity $m_{\mathcal{h}}(\epsilon,\delta)$ responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 3.5 consistent ERM

Consistent erm is part of the canonical scope of PAC Learning. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is probably approximately correct guarantees, finite-class sample complexity, realizable and agnostic learning, and distribution-free learnability. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$m \ge \frac{1}{\epsilon}\left(\log\lvert\mathcal{H}\rvert + \log\frac{1}{\delta}\right).$$

The formula should be read operationally. For consistent erm, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of consistent erm:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for consistent erm is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: consistent erm will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using consistent erm responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

## 4. Agnostic PAC Learning

Agnostic PAC Learning develops the part of pac learning specified by the approved Chapter 21 table of contents. The emphasis is statistical learning theory, not generic statistics, optimization recipes, or benchmark operations.

### 4.1 Bayes error and approximation error

Bayes error and approximation error is part of the canonical scope of PAC Learning. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is probably approximately correct guarantees, finite-class sample complexity, realizable and agnostic learning, and distribution-free learnability. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$P[L_{\mathcal{D}}(h_S)\le \epsilon] \ge 1-\delta.$$

The formula should be read operationally. For bayes error and approximation error, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of bayes error and approximation error:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for bayes error and approximation error is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: bayes error and approximation error will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using bayes error and approximation error responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 4.2 excess risk

Excess risk is part of the canonical scope of PAC Learning. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is probably approximately correct guarantees, finite-class sample complexity, realizable and agnostic learning, and distribution-free learnability. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$L_{\mathcal{D}}(h)=P_{(\mathbf{x},y)\sim\mathcal{D}}[h(\mathbf{x})\ne y].$$

The formula should be read operationally. For excess risk, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of excess risk:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for excess risk is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: excess risk will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using excess risk responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 4.3 agnostic ERM

Agnostic erm is part of the canonical scope of PAC Learning. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is probably approximately correct guarantees, finite-class sample complexity, realizable and agnostic learning, and distribution-free learnability. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$L_S(h)=\frac{1}{m}\sum_{i=1}^{m}\mathbb{1}[h(\mathbf{x}^{(i)})\ne y^{(i)}].$$

The formula should be read operationally. For agnostic erm, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of agnostic erm:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for agnostic erm is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: agnostic erm will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using agnostic erm responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 4.4 finite-class agnostic bound

Finite-class agnostic bound is part of the canonical scope of PAC Learning. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is probably approximately correct guarantees, finite-class sample complexity, realizable and agnostic learning, and distribution-free learnability. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$m \ge \frac{1}{\epsilon}\left(\log\lvert\mathcal{H}\rvert + \log\frac{1}{\delta}\right).$$

The formula should be read operationally. For finite-class agnostic bound, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of finite-class agnostic bound:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for finite-class agnostic bound is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: finite-class agnostic bound will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using finite-class agnostic bound responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 4.5 noisy labels

Noisy labels is part of the canonical scope of PAC Learning. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is probably approximately correct guarantees, finite-class sample complexity, realizable and agnostic learning, and distribution-free learnability. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$P[L_{\mathcal{D}}(h_S)\le \epsilon] \ge 1-\delta.$$

The formula should be read operationally. For noisy labels, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of noisy labels:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for noisy labels is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: noisy labels will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using noisy labels responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

## 5. Sample Complexity

Sample Complexity develops the part of pac learning specified by the approved Chapter 21 table of contents. The emphasis is statistical learning theory, not generic statistics, optimization recipes, or benchmark operations.

### 5.1 dependence on $\epsilon$

Dependence on $\epsilon$ is part of the canonical scope of PAC Learning. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is probably approximately correct guarantees, finite-class sample complexity, realizable and agnostic learning, and distribution-free learnability. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$L_{\mathcal{D}}(h)=P_{(\mathbf{x},y)\sim\mathcal{D}}[h(\mathbf{x})\ne y].$$

The formula should be read operationally. For dependence on $\epsilon$, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of dependence on $\epsilon$:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for dependence on $\epsilon$ is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: dependence on $\epsilon$ will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using dependence on $\epsilon$ responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 5.2 dependence on $\delta$

Dependence on $\delta$ is part of the canonical scope of PAC Learning. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is probably approximately correct guarantees, finite-class sample complexity, realizable and agnostic learning, and distribution-free learnability. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$L_S(h)=\frac{1}{m}\sum_{i=1}^{m}\mathbb{1}[h(\mathbf{x}^{(i)})\ne y^{(i)}].$$

The formula should be read operationally. For dependence on $\delta$, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of dependence on $\delta$:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for dependence on $\delta$ is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: dependence on $\delta$ will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using dependence on $\delta$ responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 5.3 logarithmic class-size dependence

Logarithmic class-size dependence is part of the canonical scope of PAC Learning. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is probably approximately correct guarantees, finite-class sample complexity, realizable and agnostic learning, and distribution-free learnability. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$m \ge \frac{1}{\epsilon}\left(\log\lvert\mathcal{H}\rvert + \log\frac{1}{\delta}\right).$$

The formula should be read operationally. For logarithmic class-size dependence, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of logarithmic class-size dependence:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for logarithmic class-size dependence is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: logarithmic class-size dependence will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using logarithmic class-size dependence responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 5.4 infinite-class motivation

Infinite-class motivation is part of the canonical scope of PAC Learning. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is probably approximately correct guarantees, finite-class sample complexity, realizable and agnostic learning, and distribution-free learnability. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$P[L_{\mathcal{D}}(h_S)\le \epsilon] \ge 1-\delta.$$

The formula should be read operationally. For infinite-class motivation, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of infinite-class motivation:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for infinite-class motivation is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: infinite-class motivation will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using infinite-class motivation responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 5.5 AI-scale interpretation

Ai-scale interpretation is part of the canonical scope of PAC Learning. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is probably approximately correct guarantees, finite-class sample complexity, realizable and agnostic learning, and distribution-free learnability. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$L_{\mathcal{D}}(h)=P_{(\mathbf{x},y)\sim\mathcal{D}}[h(\mathbf{x})\ne y].$$

The formula should be read operationally. For ai-scale interpretation, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of ai-scale interpretation:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for ai-scale interpretation is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: ai-scale interpretation will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using ai-scale interpretation responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

## 6. Applications in ML and LLMs

Applications in ML and LLMs develops the part of pac learning specified by the approved Chapter 21 table of contents. The emphasis is statistical learning theory, not generic statistics, optimization recipes, or benchmark operations.

### 6.1 classifier selection

Classifier selection is part of the canonical scope of PAC Learning. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is probably approximately correct guarantees, finite-class sample complexity, realizable and agnostic learning, and distribution-free learnability. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$L_S(h)=\frac{1}{m}\sum_{i=1}^{m}\mathbb{1}[h(\mathbf{x}^{(i)})\ne y^{(i)}].$$

The formula should be read operationally. For classifier selection, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of classifier selection:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for classifier selection is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: classifier selection will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using classifier selection responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 6.2 prompt classifier reliability

Prompt classifier reliability is part of the canonical scope of PAC Learning. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is probably approximately correct guarantees, finite-class sample complexity, realizable and agnostic learning, and distribution-free learnability. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$m \ge \frac{1}{\epsilon}\left(\log\lvert\mathcal{H}\rvert + \log\frac{1}{\delta}\right).$$

The formula should be read operationally. For prompt classifier reliability, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of prompt classifier reliability:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for prompt classifier reliability is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: prompt classifier reliability will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using prompt classifier reliability responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 6.3 safety classifier sample needs

Safety classifier sample needs is part of the canonical scope of PAC Learning. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is probably approximately correct guarantees, finite-class sample complexity, realizable and agnostic learning, and distribution-free learnability. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$P[L_{\mathcal{D}}(h_S)\le \epsilon] \ge 1-\delta.$$

The formula should be read operationally. For safety classifier sample needs, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of safety classifier sample needs:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for safety classifier sample needs is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: safety classifier sample needs will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using safety classifier sample needs responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 6.4 evaluation-set sizing preview

Evaluation-set sizing preview is part of the canonical scope of PAC Learning. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is probably approximately correct guarantees, finite-class sample complexity, realizable and agnostic learning, and distribution-free learnability. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$L_{\mathcal{D}}(h)=P_{(\mathbf{x},y)\sim\mathcal{D}}[h(\mathbf{x})\ne y].$$

The formula should be read operationally. For evaluation-set sizing preview, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of evaluation-set sizing preview:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for evaluation-set sizing preview is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: evaluation-set sizing preview will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using evaluation-set sizing preview responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 6.5 limits for deep nets

Limits for deep nets is part of the canonical scope of PAC Learning. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is probably approximately correct guarantees, finite-class sample complexity, realizable and agnostic learning, and distribution-free learnability. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$L_S(h)=\frac{1}{m}\sum_{i=1}^{m}\mathbb{1}[h(\mathbf{x}^{(i)})\ne y^{(i)}].$$

The formula should be read operationally. For limits for deep nets, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of limits for deep nets:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for limits for deep nets is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: limits for deep nets will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using limits for deep nets responsibly:

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

1. (*) Work through a learning-theory question for pac learning.
   - (a) Define the sample, distribution, hypothesis class, and loss.
   - (b) State the relevant risk or complexity quantity.
   - (c) Derive or compute the bound requested by the problem.
   - (d) Interpret the result for an ML or LLM system.

2. (*) Work through a learning-theory question for pac learning.
   - (a) Define the sample, distribution, hypothesis class, and loss.
   - (b) State the relevant risk or complexity quantity.
   - (c) Derive or compute the bound requested by the problem.
   - (d) Interpret the result for an ML or LLM system.

3. (*) Work through a learning-theory question for pac learning.
   - (a) Define the sample, distribution, hypothesis class, and loss.
   - (b) State the relevant risk or complexity quantity.
   - (c) Derive or compute the bound requested by the problem.
   - (d) Interpret the result for an ML or LLM system.

4. (**) Work through a learning-theory question for pac learning.
   - (a) Define the sample, distribution, hypothesis class, and loss.
   - (b) State the relevant risk or complexity quantity.
   - (c) Derive or compute the bound requested by the problem.
   - (d) Interpret the result for an ML or LLM system.

5. (**) Work through a learning-theory question for pac learning.
   - (a) Define the sample, distribution, hypothesis class, and loss.
   - (b) State the relevant risk or complexity quantity.
   - (c) Derive or compute the bound requested by the problem.
   - (d) Interpret the result for an ML or LLM system.

6. (**) Work through a learning-theory question for pac learning.
   - (a) Define the sample, distribution, hypothesis class, and loss.
   - (b) State the relevant risk or complexity quantity.
   - (c) Derive or compute the bound requested by the problem.
   - (d) Interpret the result for an ML or LLM system.

7. (***) Work through a learning-theory question for pac learning.
   - (a) Define the sample, distribution, hypothesis class, and loss.
   - (b) State the relevant risk or complexity quantity.
   - (c) Derive or compute the bound requested by the problem.
   - (d) Interpret the result for an ML or LLM system.

8. (***) Work through a learning-theory question for pac learning.
   - (a) Define the sample, distribution, hypothesis class, and loss.
   - (b) State the relevant risk or complexity quantity.
   - (c) Derive or compute the bound requested by the problem.
   - (d) Interpret the result for an ML or LLM system.

9. (***) Work through a learning-theory question for pac learning.
   - (a) Define the sample, distribution, hypothesis class, and loss.
   - (b) State the relevant risk or complexity quantity.
   - (c) Derive or compute the bound requested by the problem.
   - (d) Interpret the result for an ML or LLM system.

10. (***) Work through a learning-theory question for pac learning.
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

PAC Learning belongs in the research-frontier phase because modern AI systems force us to ask why enormous models generalize from finite data. Earlier chapters gave probability, statistics, optimization, evaluation, and production practice. This chapter turns those ingredients into mathematical learnability questions.

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

- Valiant. A Theory of the Learnable. https://dl.acm.org/doi/10.1145/1968.1972
- Shalev-Shwartz and Ben-David. Understanding Machine Learning: From Theory to Algorithms. https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/
- Mohri, Rostamizadeh, and Talwalkar. Foundations of Machine Learning. https://cs.nyu.edu/~mohri/mlbook/
- Blumer et al.. Learnability and the Vapnik-Chervonenkis dimension. https://dl.acm.org/doi/10.1145/76359.76371
