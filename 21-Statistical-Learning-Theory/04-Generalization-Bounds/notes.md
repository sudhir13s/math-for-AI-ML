[Back to Curriculum](../../README.md) | [Previous: Bias Variance Tradeoff](../03-Bias-Variance-Tradeoff/notes.md) | [Next: Rademacher Complexity](../05-Rademacher-Complexity/notes.md)

---

# Generalization Bounds

> _"A bound is a contract between what was measured and what can be expected."_

## Overview

Generalization bounds quantify when empirical performance is likely to approximate population performance.

Statistical learning theory is the chapter where finite samples meet future performance. It does not ask only whether a model performed well on observed data. It asks what assumptions let observed performance control unobserved risk.

This section is written in LaTeX Markdown. Inline mathematics uses `$...$`, and display equations use `$$...$$`. The notes emphasize theorem-level objects such as risk, hypothesis class, sample complexity, capacity, and confidence.

## Prerequisites

- [Concentration Inequalities](../../06-Probability-Theory/05-Concentration-Inequalities/notes.md)
- [PAC Learning](../01-PAC-Learning/notes.md)
- [VC Dimension](../02-VC-Dimension/notes.md)
- [Capability Benchmarks](../../17-Evaluation-and-Reliability/01-Capability-Benchmarks/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Executable demonstrations for generalization bounds |
| [exercises.ipynb](exercises.ipynb) | Graded practice for generalization bounds |

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
  - [1.1 from empirical risk to true risk](#11-from-empirical-risk-to-true-risk)
  - [1.2 concentration as bridge](#12-concentration-as-bridge)
  - [1.3 uniform bounds](#13-uniform-bounds)
  - [1.4 margin and norm bounds](#14-margin-and-norm-bounds)
  - [1.5 why bounds can be loose but useful](#15-why-bounds-can-be-loose-but-useful)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 true risk and empirical risk](#21-true-risk-and-empirical-risk)
  - [2.2 generalization gap](#22-generalization-gap)
  - [2.3 uniform convergence](#23-uniform-convergence)
  - [2.4 confidence parameter $\delta$](#24-confidence-parameter)
  - [2.5 bound template](#25-bound-template)
- [3. Finite-Class Bounds](#3-finiteclass-bounds)
  - [3.1 Hoeffding inequality recall](#31-hoeffding-inequality-recall)
  - [3.2 union bound over $\mathcal{H}$](#32-union-bound-over)
  - [3.3 realizable bound](#33-realizable-bound)
  - [3.4 agnostic bound](#34-agnostic-bound)
  - [3.5 sample-size calculator](#35-samplesize-calculator)
- [4. Complexity-Based Bounds](#4-complexitybased-bounds)
  - [4.1 VC bound](#41-vc-bound)
  - [4.2 Rademacher preview](#42-rademacher-preview)
  - [4.3 covering number preview](#43-covering-number-preview)
  - [4.4 compression preview](#44-compression-preview)
  - [4.5 stability preview](#45-stability-preview)
- [5. Margin and Norm Bounds](#5-margin-and-norm-bounds)
  - [5.1 margins for classifiers](#51-margins-for-classifiers)
  - [5.2 perceptron and SVM intuition](#52-perceptron-and-svm-intuition)
  - [5.3 norm-based neural network bounds](#53-normbased-neural-network-bounds)
  - [5.4 PAC-Bayes preview](#54-pacbayes-preview)
  - [5.5 calibration to practice](#55-calibration-to-practice)
- [6. Interpreting Bounds in AI](#6-interpreting-bounds-in-ai)
  - [6.1 why deep-net bounds are loose](#61-why-deepnet-bounds-are-loose)
  - [6.2 relative guarantees](#62-relative-guarantees)
  - [6.3 evaluation-set confidence](#63-evaluationset-confidence)
  - [6.4 data scaling laws](#64-data-scaling-laws)
  - [6.5 safety-critical conservatism](#65-safetycritical-conservatism)
- [7. Common Mistakes](#7-common-mistakes)
- [8. Exercises](#8-exercises)
- [9. Why This Matters for AI](#9-why-this-matters-for-ai)
- [10. Conceptual Bridge](#10-conceptual-bridge)
- [References](#references)

---

## 1. Intuition

Intuition develops the part of generalization bounds specified by the approved Chapter 21 table of contents. The emphasis is statistical learning theory, not generic statistics, optimization recipes, or benchmark operations.

### 1.1 from empirical risk to true risk

From empirical risk to true risk is part of the canonical scope of Generalization Bounds. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is finite-class, VC, margin, norm, stability, compression, and PAC-Bayes style bounds as tools for interpreting empirical risk. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$\operatorname{gen}(h)=L_{\mathcal{D}}(h)-L_S(h).$$

The formula should be read operationally. For from empirical risk to true risk, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of from empirical risk to true risk:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for from empirical risk to true risk is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: from empirical risk to true risk will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using from empirical risk to true risk responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 1.2 concentration as bridge

Concentration as bridge is part of the canonical scope of Generalization Bounds. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is finite-class, VC, margin, norm, stability, compression, and PAC-Bayes style bounds as tools for interpreting empirical risk. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$P\left(\lvert L_{\mathcal{D}}(h)-L_S(h)\rvert>\epsilon\right)\le 2e^{-2m\epsilon^2}.$$

The formula should be read operationally. For concentration as bridge, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of concentration as bridge:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for concentration as bridge is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: concentration as bridge will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using concentration as bridge responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 1.3 uniform bounds

Uniform bounds is part of the canonical scope of Generalization Bounds. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is finite-class, VC, margin, norm, stability, compression, and PAC-Bayes style bounds as tools for interpreting empirical risk. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$\sup_{h\in\mathcal{H}}\lvert L_{\mathcal{D}}(h)-L_S(h)\rvert \le B(m,\mathcal{H},\delta).$$

The formula should be read operationally. For uniform bounds, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of uniform bounds:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for uniform bounds is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: uniform bounds will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using uniform bounds responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 1.4 margin and norm bounds

Margin and norm bounds is part of the canonical scope of Generalization Bounds. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is finite-class, VC, margin, norm, stability, compression, and PAC-Bayes style bounds as tools for interpreting empirical risk. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$L_{\mathcal{D}}(h)\le L_S(h)+\sqrt{\frac{\log\lvert\mathcal{H}\rvert+\log(2/\delta)}{2m}}.$$

The formula should be read operationally. For margin and norm bounds, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of margin and norm bounds:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for margin and norm bounds is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: margin and norm bounds will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using margin and norm bounds responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 1.5 why bounds can be loose but useful

Why bounds can be loose but useful is part of the canonical scope of Generalization Bounds. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is finite-class, VC, margin, norm, stability, compression, and PAC-Bayes style bounds as tools for interpreting empirical risk. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$\operatorname{gen}(h)=L_{\mathcal{D}}(h)-L_S(h).$$

The formula should be read operationally. For why bounds can be loose but useful, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of why bounds can be loose but useful:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for why bounds can be loose but useful is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: why bounds can be loose but useful will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using why bounds can be loose but useful responsibly:

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

Formal Definitions develops the part of generalization bounds specified by the approved Chapter 21 table of contents. The emphasis is statistical learning theory, not generic statistics, optimization recipes, or benchmark operations.

### 2.1 true risk and empirical risk

True risk and empirical risk is part of the canonical scope of Generalization Bounds. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is finite-class, VC, margin, norm, stability, compression, and PAC-Bayes style bounds as tools for interpreting empirical risk. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$P\left(\lvert L_{\mathcal{D}}(h)-L_S(h)\rvert>\epsilon\right)\le 2e^{-2m\epsilon^2}.$$

The formula should be read operationally. For true risk and empirical risk, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of true risk and empirical risk:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for true risk and empirical risk is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: true risk and empirical risk will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using true risk and empirical risk responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 2.2 generalization gap

Generalization gap is part of the canonical scope of Generalization Bounds. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is finite-class, VC, margin, norm, stability, compression, and PAC-Bayes style bounds as tools for interpreting empirical risk. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$\sup_{h\in\mathcal{H}}\lvert L_{\mathcal{D}}(h)-L_S(h)\rvert \le B(m,\mathcal{H},\delta).$$

The formula should be read operationally. For generalization gap, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of generalization gap:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for generalization gap is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: generalization gap will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using generalization gap responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 2.3 uniform convergence

Uniform convergence is part of the canonical scope of Generalization Bounds. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is finite-class, VC, margin, norm, stability, compression, and PAC-Bayes style bounds as tools for interpreting empirical risk. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$L_{\mathcal{D}}(h)\le L_S(h)+\sqrt{\frac{\log\lvert\mathcal{H}\rvert+\log(2/\delta)}{2m}}.$$

The formula should be read operationally. For uniform convergence, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of uniform convergence:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for uniform convergence is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: uniform convergence will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using uniform convergence responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 2.4 confidence parameter $\delta$

Confidence parameter $\delta$ is part of the canonical scope of Generalization Bounds. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is finite-class, VC, margin, norm, stability, compression, and PAC-Bayes style bounds as tools for interpreting empirical risk. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$\operatorname{gen}(h)=L_{\mathcal{D}}(h)-L_S(h).$$

The formula should be read operationally. For confidence parameter $\delta$, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of confidence parameter $\delta$:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for confidence parameter $\delta$ is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: confidence parameter $\delta$ will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using confidence parameter $\delta$ responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 2.5 bound template

Bound template is part of the canonical scope of Generalization Bounds. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is finite-class, VC, margin, norm, stability, compression, and PAC-Bayes style bounds as tools for interpreting empirical risk. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$P\left(\lvert L_{\mathcal{D}}(h)-L_S(h)\rvert>\epsilon\right)\le 2e^{-2m\epsilon^2}.$$

The formula should be read operationally. For bound template, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of bound template:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for bound template is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: bound template will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using bound template responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

## 3. Finite-Class Bounds

Finite-Class Bounds develops the part of generalization bounds specified by the approved Chapter 21 table of contents. The emphasis is statistical learning theory, not generic statistics, optimization recipes, or benchmark operations.

### 3.1 Hoeffding inequality recall

Hoeffding inequality recall is part of the canonical scope of Generalization Bounds. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is finite-class, VC, margin, norm, stability, compression, and PAC-Bayes style bounds as tools for interpreting empirical risk. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$\sup_{h\in\mathcal{H}}\lvert L_{\mathcal{D}}(h)-L_S(h)\rvert \le B(m,\mathcal{H},\delta).$$

The formula should be read operationally. For hoeffding inequality recall, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of hoeffding inequality recall:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for hoeffding inequality recall is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: hoeffding inequality recall will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using hoeffding inequality recall responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 3.2 union bound over $\mathcal{H}$

Union bound over $\mathcal{h}$ is part of the canonical scope of Generalization Bounds. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is finite-class, VC, margin, norm, stability, compression, and PAC-Bayes style bounds as tools for interpreting empirical risk. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$L_{\mathcal{D}}(h)\le L_S(h)+\sqrt{\frac{\log\lvert\mathcal{H}\rvert+\log(2/\delta)}{2m}}.$$

The formula should be read operationally. For union bound over $\mathcal{h}$, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of union bound over $\mathcal{h}$:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for union bound over $\mathcal{h}$ is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: union bound over $\mathcal{h}$ will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using union bound over $\mathcal{h}$ responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 3.3 realizable bound

Realizable bound is part of the canonical scope of Generalization Bounds. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is finite-class, VC, margin, norm, stability, compression, and PAC-Bayes style bounds as tools for interpreting empirical risk. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$\operatorname{gen}(h)=L_{\mathcal{D}}(h)-L_S(h).$$

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

### 3.4 agnostic bound

Agnostic bound is part of the canonical scope of Generalization Bounds. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is finite-class, VC, margin, norm, stability, compression, and PAC-Bayes style bounds as tools for interpreting empirical risk. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$P\left(\lvert L_{\mathcal{D}}(h)-L_S(h)\rvert>\epsilon\right)\le 2e^{-2m\epsilon^2}.$$

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

### 3.5 sample-size calculator

Sample-size calculator is part of the canonical scope of Generalization Bounds. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is finite-class, VC, margin, norm, stability, compression, and PAC-Bayes style bounds as tools for interpreting empirical risk. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$\sup_{h\in\mathcal{H}}\lvert L_{\mathcal{D}}(h)-L_S(h)\rvert \le B(m,\mathcal{H},\delta).$$

The formula should be read operationally. For sample-size calculator, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of sample-size calculator:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for sample-size calculator is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: sample-size calculator will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using sample-size calculator responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

## 4. Complexity-Based Bounds

Complexity-Based Bounds develops the part of generalization bounds specified by the approved Chapter 21 table of contents. The emphasis is statistical learning theory, not generic statistics, optimization recipes, or benchmark operations.

### 4.1 VC bound

Vc bound is part of the canonical scope of Generalization Bounds. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is finite-class, VC, margin, norm, stability, compression, and PAC-Bayes style bounds as tools for interpreting empirical risk. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$L_{\mathcal{D}}(h)\le L_S(h)+\sqrt{\frac{\log\lvert\mathcal{H}\rvert+\log(2/\delta)}{2m}}.$$

The formula should be read operationally. For vc bound, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of vc bound:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for vc bound is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: vc bound will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using vc bound responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 4.2 Rademacher preview

Rademacher preview is part of the canonical scope of Generalization Bounds. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is finite-class, VC, margin, norm, stability, compression, and PAC-Bayes style bounds as tools for interpreting empirical risk. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$\operatorname{gen}(h)=L_{\mathcal{D}}(h)-L_S(h).$$

The formula should be read operationally. For rademacher preview, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of rademacher preview:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for rademacher preview is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: rademacher preview will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using rademacher preview responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 4.3 covering number preview

Covering number preview is part of the canonical scope of Generalization Bounds. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is finite-class, VC, margin, norm, stability, compression, and PAC-Bayes style bounds as tools for interpreting empirical risk. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$P\left(\lvert L_{\mathcal{D}}(h)-L_S(h)\rvert>\epsilon\right)\le 2e^{-2m\epsilon^2}.$$

The formula should be read operationally. For covering number preview, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of covering number preview:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for covering number preview is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: covering number preview will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using covering number preview responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 4.4 compression preview

Compression preview is part of the canonical scope of Generalization Bounds. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is finite-class, VC, margin, norm, stability, compression, and PAC-Bayes style bounds as tools for interpreting empirical risk. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$\sup_{h\in\mathcal{H}}\lvert L_{\mathcal{D}}(h)-L_S(h)\rvert \le B(m,\mathcal{H},\delta).$$

The formula should be read operationally. For compression preview, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of compression preview:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for compression preview is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: compression preview will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using compression preview responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 4.5 stability preview

Stability preview is part of the canonical scope of Generalization Bounds. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is finite-class, VC, margin, norm, stability, compression, and PAC-Bayes style bounds as tools for interpreting empirical risk. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$L_{\mathcal{D}}(h)\le L_S(h)+\sqrt{\frac{\log\lvert\mathcal{H}\rvert+\log(2/\delta)}{2m}}.$$

The formula should be read operationally. For stability preview, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of stability preview:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for stability preview is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: stability preview will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using stability preview responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

## 5. Margin and Norm Bounds

Margin and Norm Bounds develops the part of generalization bounds specified by the approved Chapter 21 table of contents. The emphasis is statistical learning theory, not generic statistics, optimization recipes, or benchmark operations.

### 5.1 margins for classifiers

Margins for classifiers is part of the canonical scope of Generalization Bounds. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is finite-class, VC, margin, norm, stability, compression, and PAC-Bayes style bounds as tools for interpreting empirical risk. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$\operatorname{gen}(h)=L_{\mathcal{D}}(h)-L_S(h).$$

The formula should be read operationally. For margins for classifiers, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of margins for classifiers:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for margins for classifiers is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: margins for classifiers will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using margins for classifiers responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 5.2 perceptron and SVM intuition

Perceptron and svm intuition is part of the canonical scope of Generalization Bounds. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is finite-class, VC, margin, norm, stability, compression, and PAC-Bayes style bounds as tools for interpreting empirical risk. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$P\left(\lvert L_{\mathcal{D}}(h)-L_S(h)\rvert>\epsilon\right)\le 2e^{-2m\epsilon^2}.$$

The formula should be read operationally. For perceptron and svm intuition, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of perceptron and svm intuition:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for perceptron and svm intuition is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: perceptron and svm intuition will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using perceptron and svm intuition responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 5.3 norm-based neural network bounds

Norm-based neural network bounds is part of the canonical scope of Generalization Bounds. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is finite-class, VC, margin, norm, stability, compression, and PAC-Bayes style bounds as tools for interpreting empirical risk. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$\sup_{h\in\mathcal{H}}\lvert L_{\mathcal{D}}(h)-L_S(h)\rvert \le B(m,\mathcal{H},\delta).$$

The formula should be read operationally. For norm-based neural network bounds, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of norm-based neural network bounds:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for norm-based neural network bounds is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: norm-based neural network bounds will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using norm-based neural network bounds responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 5.4 PAC-Bayes preview

Pac-bayes preview is part of the canonical scope of Generalization Bounds. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is finite-class, VC, margin, norm, stability, compression, and PAC-Bayes style bounds as tools for interpreting empirical risk. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$L_{\mathcal{D}}(h)\le L_S(h)+\sqrt{\frac{\log\lvert\mathcal{H}\rvert+\log(2/\delta)}{2m}}.$$

The formula should be read operationally. For pac-bayes preview, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of pac-bayes preview:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for pac-bayes preview is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: pac-bayes preview will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using pac-bayes preview responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 5.5 calibration to practice

Calibration to practice is part of the canonical scope of Generalization Bounds. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is finite-class, VC, margin, norm, stability, compression, and PAC-Bayes style bounds as tools for interpreting empirical risk. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$\operatorname{gen}(h)=L_{\mathcal{D}}(h)-L_S(h).$$

The formula should be read operationally. For calibration to practice, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of calibration to practice:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for calibration to practice is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: calibration to practice will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using calibration to practice responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

## 6. Interpreting Bounds in AI

Interpreting Bounds in AI develops the part of generalization bounds specified by the approved Chapter 21 table of contents. The emphasis is statistical learning theory, not generic statistics, optimization recipes, or benchmark operations.

### 6.1 why deep-net bounds are loose

Why deep-net bounds are loose is part of the canonical scope of Generalization Bounds. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is finite-class, VC, margin, norm, stability, compression, and PAC-Bayes style bounds as tools for interpreting empirical risk. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$P\left(\lvert L_{\mathcal{D}}(h)-L_S(h)\rvert>\epsilon\right)\le 2e^{-2m\epsilon^2}.$$

The formula should be read operationally. For why deep-net bounds are loose, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of why deep-net bounds are loose:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for why deep-net bounds are loose is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: why deep-net bounds are loose will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using why deep-net bounds are loose responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 6.2 relative guarantees

Relative guarantees is part of the canonical scope of Generalization Bounds. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is finite-class, VC, margin, norm, stability, compression, and PAC-Bayes style bounds as tools for interpreting empirical risk. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$\sup_{h\in\mathcal{H}}\lvert L_{\mathcal{D}}(h)-L_S(h)\rvert \le B(m,\mathcal{H},\delta).$$

The formula should be read operationally. For relative guarantees, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of relative guarantees:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for relative guarantees is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: relative guarantees will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using relative guarantees responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 6.3 evaluation-set confidence

Evaluation-set confidence is part of the canonical scope of Generalization Bounds. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is finite-class, VC, margin, norm, stability, compression, and PAC-Bayes style bounds as tools for interpreting empirical risk. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$L_{\mathcal{D}}(h)\le L_S(h)+\sqrt{\frac{\log\lvert\mathcal{H}\rvert+\log(2/\delta)}{2m}}.$$

The formula should be read operationally. For evaluation-set confidence, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of evaluation-set confidence:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for evaluation-set confidence is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: evaluation-set confidence will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using evaluation-set confidence responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 6.4 data scaling laws

Data scaling laws is part of the canonical scope of Generalization Bounds. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is finite-class, VC, margin, norm, stability, compression, and PAC-Bayes style bounds as tools for interpreting empirical risk. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$\operatorname{gen}(h)=L_{\mathcal{D}}(h)-L_S(h).$$

The formula should be read operationally. For data scaling laws, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of data scaling laws:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for data scaling laws is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: data scaling laws will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using data scaling laws responsibly:

- State the sample space and label space.
- State the hypothesis or function class.
- State the loss and risk definition.
- State whether the setting is realizable or agnostic.
- Track both accuracy tolerance and confidence.
- Identify whether the bound is distribution-free or data-dependent.
- Separate the theorem from the empirical measurement.

For AI systems, this discipline prevents a common confusion: empirical success is evidence, but learnability theory explains which kinds of evidence should scale with sample size, class capacity, margins, norms, and noise.

The subsection also prepares the later material. PAC learning motivates VC dimension. VC dimension motivates generalization bounds. Bias-variance decomposition gives a different error accounting. Rademacher complexity gives a data-dependent complexity view.

### 6.5 safety-critical conservatism

Safety-critical conservatism is part of the canonical scope of Generalization Bounds. The purpose is to understand when finite data can justify a claim about unseen examples, not to replace empirical evaluation or production monitoring.

In this subsection the working scope is finite-class, VC, margin, norm, stability, compression, and PAC-Bayes style bounds as tools for interpreting empirical risk. We use a distribution $\mathcal{D}$, a sample $S$, a hypothesis class $\mathcal{H}$, and a loss-derived risk. The core question is whether the behavior on $S$ can control the behavior under $\mathcal{D}$.

$$P\left(\lvert L_{\mathcal{D}}(h)-L_S(h)\rvert>\epsilon\right)\le 2e^{-2m\epsilon^2}.$$

The formula should be read operationally. For safety-critical conservatism, a learner is not certified by a story about model architecture. It is certified by assumptions, a class of hypotheses, a loss, a sample size, and a probability statement.

| Theory object | Meaning | AI interpretation |
| --- | --- | --- |
| $\mathcal{D}$ | Unknown data distribution | User prompts, images, tokens, labels, or tasks the system will face |
| $S$ | Finite training or evaluation sample | The observed examples available to the learner or auditor |
| $\mathcal{H}$ | Hypothesis class | Classifiers, probes, reward models, safety filters, or predictors |
| $L_S(h)$ | Empirical risk | Error measured on the observed sample |
| $L_{\mathcal{D}}(h)$ | True risk | Error on the distribution that matters after deployment |

Three examples of safety-critical conservatism:

1. A binary safety classifier is evaluated on a sample of labeled prompts, but the team needs a bound on future violation-detection error.
2. A linear probe is trained on hidden states, and learning theory asks how much the probe's validation behavior depends on sample size and class capacity.
3. A small model is fine-tuned on limited domain data, and the practitioner wants to separate approximation error from estimation error.

Two non-examples are just as important:

1. A leaderboard rank without a distributional statement is not a learnability guarantee.
2. A production incident report without a hypothesis class, loss, or sampling assumption is not a statistical learning theorem.

The proof habit for safety-critical conservatism is to identify the random object first. Sometimes the randomness is the sample $S$. Sometimes it is Rademacher signs. Sometimes it is label noise. Once the random object is explicit, concentration and symmetrization tools can be used without hand-waving.

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

Implementation note for the companion notebook: safety-critical conservatism will be demonstrated with synthetic finite samples. The code will not depend on external datasets; it will compute bounds, simulate class behavior, or plot risk decompositions so the theorem-level object is visible.

The modern AI caution is that very large models often violate the cleanest textbook assumptions. That does not make the mathematics useless. It means the reader should distinguish theorem-level guarantees from diagnostic metaphors and engineering heuristics.

Checklist for using safety-critical conservatism responsibly:

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

1. (*) Work through a learning-theory question for generalization bounds.
   - (a) Define the sample, distribution, hypothesis class, and loss.
   - (b) State the relevant risk or complexity quantity.
   - (c) Derive or compute the bound requested by the problem.
   - (d) Interpret the result for an ML or LLM system.

2. (*) Work through a learning-theory question for generalization bounds.
   - (a) Define the sample, distribution, hypothesis class, and loss.
   - (b) State the relevant risk or complexity quantity.
   - (c) Derive or compute the bound requested by the problem.
   - (d) Interpret the result for an ML or LLM system.

3. (*) Work through a learning-theory question for generalization bounds.
   - (a) Define the sample, distribution, hypothesis class, and loss.
   - (b) State the relevant risk or complexity quantity.
   - (c) Derive or compute the bound requested by the problem.
   - (d) Interpret the result for an ML or LLM system.

4. (**) Work through a learning-theory question for generalization bounds.
   - (a) Define the sample, distribution, hypothesis class, and loss.
   - (b) State the relevant risk or complexity quantity.
   - (c) Derive or compute the bound requested by the problem.
   - (d) Interpret the result for an ML or LLM system.

5. (**) Work through a learning-theory question for generalization bounds.
   - (a) Define the sample, distribution, hypothesis class, and loss.
   - (b) State the relevant risk or complexity quantity.
   - (c) Derive or compute the bound requested by the problem.
   - (d) Interpret the result for an ML or LLM system.

6. (**) Work through a learning-theory question for generalization bounds.
   - (a) Define the sample, distribution, hypothesis class, and loss.
   - (b) State the relevant risk or complexity quantity.
   - (c) Derive or compute the bound requested by the problem.
   - (d) Interpret the result for an ML or LLM system.

7. (***) Work through a learning-theory question for generalization bounds.
   - (a) Define the sample, distribution, hypothesis class, and loss.
   - (b) State the relevant risk or complexity quantity.
   - (c) Derive or compute the bound requested by the problem.
   - (d) Interpret the result for an ML or LLM system.

8. (***) Work through a learning-theory question for generalization bounds.
   - (a) Define the sample, distribution, hypothesis class, and loss.
   - (b) State the relevant risk or complexity quantity.
   - (c) Derive or compute the bound requested by the problem.
   - (d) Interpret the result for an ML or LLM system.

9. (***) Work through a learning-theory question for generalization bounds.
   - (a) Define the sample, distribution, hypothesis class, and loss.
   - (b) State the relevant risk or complexity quantity.
   - (c) Derive or compute the bound requested by the problem.
   - (d) Interpret the result for an ML or LLM system.

10. (***) Work through a learning-theory question for generalization bounds.
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

Generalization Bounds belongs in the research-frontier phase because modern AI systems force us to ask why enormous models generalize from finite data. Earlier chapters gave probability, statistics, optimization, evaluation, and production practice. This chapter turns those ingredients into mathematical learnability questions.

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

- Mohri, Rostamizadeh, and Talwalkar. Foundations of Machine Learning. https://cs.nyu.edu/~mohri/mlbook/
- Shalev-Shwartz and Ben-David. Understanding Machine Learning. https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/
- Vapnik. Statistical Learning Theory. https://www.wiley.com/en-us/Statistical+Learning+Theory-p-9780471030034
- McAllester. PAC-Bayesian Model Averaging. https://dl.acm.org/doi/10.1145/307400.307435
