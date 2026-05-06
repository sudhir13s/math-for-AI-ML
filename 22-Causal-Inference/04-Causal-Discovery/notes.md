[Back to Curriculum](../../README.md) | [Previous: Counterfactuals](../03-Counterfactuals/notes.md) | [Next: Nash Equilibria](../../23-Game-Theory/01-Nash-Equilibria/notes.md)

---

# Causal Discovery

> _"Causal discovery turns data into candidate mechanisms, never into assumption-free truth."_

## Overview

Causal discovery studies when causal graph structure can be inferred from observational, interventional, or multi-environment data under explicit assumptions.

Causal inference is the part of the curriculum that separates observing from doing. It asks which assumptions allow a learner to move from associations in data to claims about interventions, alternatives, and mechanisms.

This section is written in LaTeX Markdown. Inline mathematics uses `$...$`, and display equations use `$$...$$`. The notes emphasize graph assumptions, intervention notation, counterfactual semantics, and the estimand-estimator split.

## Prerequisites

- [Structural Causal Models](../01-Structural-Causal-Models/notes.md)
- [Graph Algorithms](../../11-Graph-Theory/02-Graph-Algorithms/notes.md)
- [Generalization Bounds](../../21-Statistical-Learning-Theory/04-Generalization-Bounds/notes.md)
- [Constrained Optimization](../../08-Optimization/04-Constrained-Optimization/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Executable demonstrations for causal discovery |
| [exercises.ipynb](exercises.ipynb) | Graded practice for causal discovery |

## Learning Objectives

After completing this section, you will be able to:

- Define SCMs, structural assignments, and intervention distributions
- Distinguish conditioning from intervention using the do-operator
- Apply d-separation to simple causal graphs
- State backdoor and frontdoor adjustment formulas
- Separate causal estimands from statistical estimators
- Compute ATE, ATT, and simple counterfactual quantities
- Explain abduction, action, and prediction in SCM counterfactuals
- Describe constraint-based and score-based causal discovery
- Identify assumptions behind causal discovery algorithms
- Connect causal inference to robust ML, fairness, recommendation, and LLM agents

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 learning causal graphs from data](#11-learning-causal-graphs-from-data)
  - [1.2 why discovery is impossible without assumptions](#12-why-discovery-is-impossible-without-assumptions)
  - [1.3 Markov equivalence](#13-markov-equivalence)
  - [1.4 interventions break equivalence](#14-interventions-break-equivalence)
  - [1.5 discovery as hypothesis generation](#15-discovery-as-hypothesis-generation)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 DAG and adjacency matrix $A$](#21-dag-and-adjacency-matrix)
  - [2.2 conditional independence oracle](#22-conditional-independence-oracle)
  - [2.3 Markov equivalence class](#23-markov-equivalence-class)
  - [2.4 CPDAG](#24-cpdag)
  - [2.5 acyclicity constraint](#25-acyclicity-constraint)
- [3. Constraint-Based Discovery](#3-constraintbased-discovery)
  - [3.1 PC algorithm skeleton](#31-pc-algorithm-skeleton)
  - [3.2 conditional independence tests](#32-conditional-independence-tests)
  - [3.3 v-structures](#33-vstructures)
  - [3.4 orientation rules](#34-orientation-rules)
  - [3.5 faithfulness and finite-sample failure](#35-faithfulness-and-finitesample-failure)
- [4. Score-Based Discovery](#4-scorebased-discovery)
  - [4.1 likelihood and BIC scores](#41-likelihood-and-bic-scores)
  - [4.2 greedy equivalence search](#42-greedy-equivalence-search)
  - [4.3 super-exponential DAG search](#43-superexponential-dag-search)
  - [4.4 NOTEARS continuous optimization](#44-notears-continuous-optimization)
  - [4.5 regularization and sparsity](#45-regularization-and-sparsity)
- [5. Functional and Invariant Methods](#5-functional-and-invariant-methods)
  - [5.1 LiNGAM](#51-lingam)
  - [5.2 additive noise models](#52-additive-noise-models)
  - [5.3 invariant causal prediction](#53-invariant-causal-prediction)
  - [5.4 causal discovery across environments](#54-causal-discovery-across-environments)
  - [5.5 time-series causal discovery preview](#55-timeseries-causal-discovery-preview)
- [6. Evaluation and ML Applications](#6-evaluation-and-ml-applications)
  - [6.1 structural Hamming distance](#61-structural-hamming-distance)
  - [6.2 structural intervention distance preview](#62-structural-intervention-distance-preview)
  - [6.3 synthetic benchmarks](#63-synthetic-benchmarks)
  - [6.4 causal feature selection](#64-causal-feature-selection)
  - [6.5 LLM-assisted causal hypothesis generation with human review](#65-llmassisted-causal-hypothesis-generation-with-human-review)
- [7. Common Mistakes](#7-common-mistakes)
- [8. Exercises](#8-exercises)
- [9. Why This Matters for AI](#9-why-this-matters-for-ai)
- [10. Conceptual Bridge](#10-conceptual-bridge)
- [References](#references)

---

## 1. Intuition

Intuition develops the part of causal discovery specified by the approved Chapter 22 table of contents. The treatment is causal, not merely predictive: the central objects are mechanisms, interventions, assumptions, and counterfactuals.

### 1.1 learning causal graphs from data

Learning causal graphs from data belongs to the canonical scope of Causal Discovery. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is constraint-based, score-based, functional, invariant, and optimization-based causal graph discovery with clear assumptions and evaluation metrics. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$A_{ij}=1 \iff X_i \to X_j.$$

The formula gives a compact handle on learning causal graphs from data. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of learning causal graphs from data:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for learning causal graphs from data is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, learning causal graphs from data is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using learning causal graphs from data responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, learning causal graphs from data is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 1.2 why discovery is impossible without assumptions

Why discovery is impossible without assumptions belongs to the canonical scope of Causal Discovery. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is constraint-based, score-based, functional, invariant, and optimization-based causal graph discovery with clear assumptions and evaluation metrics. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$h(W)=\operatorname{tr}(e^{W\odot W})-d=0.$$

The formula gives a compact handle on why discovery is impossible without assumptions. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of why discovery is impossible without assumptions:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for why discovery is impossible without assumptions is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, why discovery is impossible without assumptions is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using why discovery is impossible without assumptions responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, why discovery is impossible without assumptions is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 1.3 Markov equivalence

Markov equivalence belongs to the canonical scope of Causal Discovery. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is constraint-based, score-based, functional, invariant, and optimization-based causal graph discovery with clear assumptions and evaluation metrics. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$\operatorname{SHD}(G,\widehat{G})=\#\{\text{edge additions, deletions, reversals}\}.$$

The formula gives a compact handle on markov equivalence. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of markov equivalence:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for markov equivalence is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, markov equivalence is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using markov equivalence responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, markov equivalence is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 1.4 interventions break equivalence

Interventions break equivalence belongs to the canonical scope of Causal Discovery. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is constraint-based, score-based, functional, invariant, and optimization-based causal graph discovery with clear assumptions and evaluation metrics. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$X_j=f_j(\operatorname{pa}_j)+N_j,\qquad N_j \perp\!\!\!\perp \operatorname{pa}_j.$$

The formula gives a compact handle on interventions break equivalence. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of interventions break equivalence:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for interventions break equivalence is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, interventions break equivalence is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using interventions break equivalence responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, interventions break equivalence is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 1.5 discovery as hypothesis generation

Discovery as hypothesis generation belongs to the canonical scope of Causal Discovery. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is constraint-based, score-based, functional, invariant, and optimization-based causal graph discovery with clear assumptions and evaluation metrics. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$A_{ij}=1 \iff X_i \to X_j.$$

The formula gives a compact handle on discovery as hypothesis generation. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of discovery as hypothesis generation:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for discovery as hypothesis generation is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, discovery as hypothesis generation is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using discovery as hypothesis generation responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, discovery as hypothesis generation is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

## 2. Formal Definitions

Formal Definitions develops the part of causal discovery specified by the approved Chapter 22 table of contents. The treatment is causal, not merely predictive: the central objects are mechanisms, interventions, assumptions, and counterfactuals.

### 2.1 DAG and adjacency matrix $A$

Dag and adjacency matrix $a$ belongs to the canonical scope of Causal Discovery. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is constraint-based, score-based, functional, invariant, and optimization-based causal graph discovery with clear assumptions and evaluation metrics. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$h(W)=\operatorname{tr}(e^{W\odot W})-d=0.$$

The formula gives a compact handle on dag and adjacency matrix $a$. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of dag and adjacency matrix $a$:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for dag and adjacency matrix $a$ is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, dag and adjacency matrix $a$ is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using dag and adjacency matrix $a$ responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, dag and adjacency matrix $a$ is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 2.2 conditional independence oracle

Conditional independence oracle belongs to the canonical scope of Causal Discovery. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is constraint-based, score-based, functional, invariant, and optimization-based causal graph discovery with clear assumptions and evaluation metrics. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$\operatorname{SHD}(G,\widehat{G})=\#\{\text{edge additions, deletions, reversals}\}.$$

The formula gives a compact handle on conditional independence oracle. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of conditional independence oracle:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for conditional independence oracle is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, conditional independence oracle is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using conditional independence oracle responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, conditional independence oracle is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 2.3 Markov equivalence class

Markov equivalence class belongs to the canonical scope of Causal Discovery. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is constraint-based, score-based, functional, invariant, and optimization-based causal graph discovery with clear assumptions and evaluation metrics. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$X_j=f_j(\operatorname{pa}_j)+N_j,\qquad N_j \perp\!\!\!\perp \operatorname{pa}_j.$$

The formula gives a compact handle on markov equivalence class. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of markov equivalence class:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for markov equivalence class is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, markov equivalence class is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using markov equivalence class responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, markov equivalence class is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 2.4 CPDAG

Cpdag belongs to the canonical scope of Causal Discovery. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is constraint-based, score-based, functional, invariant, and optimization-based causal graph discovery with clear assumptions and evaluation metrics. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$A_{ij}=1 \iff X_i \to X_j.$$

The formula gives a compact handle on cpdag. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of cpdag:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for cpdag is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, cpdag is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using cpdag responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, cpdag is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 2.5 acyclicity constraint

Acyclicity constraint belongs to the canonical scope of Causal Discovery. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is constraint-based, score-based, functional, invariant, and optimization-based causal graph discovery with clear assumptions and evaluation metrics. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$h(W)=\operatorname{tr}(e^{W\odot W})-d=0.$$

The formula gives a compact handle on acyclicity constraint. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of acyclicity constraint:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for acyclicity constraint is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, acyclicity constraint is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using acyclicity constraint responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, acyclicity constraint is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

## 3. Constraint-Based Discovery

Constraint-Based Discovery develops the part of causal discovery specified by the approved Chapter 22 table of contents. The treatment is causal, not merely predictive: the central objects are mechanisms, interventions, assumptions, and counterfactuals.

### 3.1 PC algorithm skeleton

Pc algorithm skeleton belongs to the canonical scope of Causal Discovery. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is constraint-based, score-based, functional, invariant, and optimization-based causal graph discovery with clear assumptions and evaluation metrics. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$\operatorname{SHD}(G,\widehat{G})=\#\{\text{edge additions, deletions, reversals}\}.$$

The formula gives a compact handle on pc algorithm skeleton. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of pc algorithm skeleton:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for pc algorithm skeleton is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, pc algorithm skeleton is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using pc algorithm skeleton responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, pc algorithm skeleton is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 3.2 conditional independence tests

Conditional independence tests belongs to the canonical scope of Causal Discovery. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is constraint-based, score-based, functional, invariant, and optimization-based causal graph discovery with clear assumptions and evaluation metrics. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$X_j=f_j(\operatorname{pa}_j)+N_j,\qquad N_j \perp\!\!\!\perp \operatorname{pa}_j.$$

The formula gives a compact handle on conditional independence tests. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of conditional independence tests:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for conditional independence tests is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, conditional independence tests is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using conditional independence tests responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, conditional independence tests is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 3.3 v-structures

V-structures belongs to the canonical scope of Causal Discovery. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is constraint-based, score-based, functional, invariant, and optimization-based causal graph discovery with clear assumptions and evaluation metrics. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$A_{ij}=1 \iff X_i \to X_j.$$

The formula gives a compact handle on v-structures. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of v-structures:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for v-structures is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, v-structures is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using v-structures responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, v-structures is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 3.4 orientation rules

Orientation rules belongs to the canonical scope of Causal Discovery. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is constraint-based, score-based, functional, invariant, and optimization-based causal graph discovery with clear assumptions and evaluation metrics. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$h(W)=\operatorname{tr}(e^{W\odot W})-d=0.$$

The formula gives a compact handle on orientation rules. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of orientation rules:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for orientation rules is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, orientation rules is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using orientation rules responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, orientation rules is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 3.5 faithfulness and finite-sample failure

Faithfulness and finite-sample failure belongs to the canonical scope of Causal Discovery. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is constraint-based, score-based, functional, invariant, and optimization-based causal graph discovery with clear assumptions and evaluation metrics. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$\operatorname{SHD}(G,\widehat{G})=\#\{\text{edge additions, deletions, reversals}\}.$$

The formula gives a compact handle on faithfulness and finite-sample failure. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of faithfulness and finite-sample failure:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for faithfulness and finite-sample failure is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, faithfulness and finite-sample failure is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using faithfulness and finite-sample failure responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, faithfulness and finite-sample failure is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

## 4. Score-Based Discovery

Score-Based Discovery develops the part of causal discovery specified by the approved Chapter 22 table of contents. The treatment is causal, not merely predictive: the central objects are mechanisms, interventions, assumptions, and counterfactuals.

### 4.1 likelihood and BIC scores

Likelihood and bic scores belongs to the canonical scope of Causal Discovery. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is constraint-based, score-based, functional, invariant, and optimization-based causal graph discovery with clear assumptions and evaluation metrics. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$X_j=f_j(\operatorname{pa}_j)+N_j,\qquad N_j \perp\!\!\!\perp \operatorname{pa}_j.$$

The formula gives a compact handle on likelihood and bic scores. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of likelihood and bic scores:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for likelihood and bic scores is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, likelihood and bic scores is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using likelihood and bic scores responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, likelihood and bic scores is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 4.2 greedy equivalence search

Greedy equivalence search belongs to the canonical scope of Causal Discovery. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is constraint-based, score-based, functional, invariant, and optimization-based causal graph discovery with clear assumptions and evaluation metrics. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$A_{ij}=1 \iff X_i \to X_j.$$

The formula gives a compact handle on greedy equivalence search. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of greedy equivalence search:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for greedy equivalence search is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, greedy equivalence search is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using greedy equivalence search responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, greedy equivalence search is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 4.3 super-exponential DAG search

Super-exponential dag search belongs to the canonical scope of Causal Discovery. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is constraint-based, score-based, functional, invariant, and optimization-based causal graph discovery with clear assumptions and evaluation metrics. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$h(W)=\operatorname{tr}(e^{W\odot W})-d=0.$$

The formula gives a compact handle on super-exponential dag search. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of super-exponential dag search:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for super-exponential dag search is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, super-exponential dag search is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using super-exponential dag search responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, super-exponential dag search is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 4.4 NOTEARS continuous optimization

Notears continuous optimization belongs to the canonical scope of Causal Discovery. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is constraint-based, score-based, functional, invariant, and optimization-based causal graph discovery with clear assumptions and evaluation metrics. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$\operatorname{SHD}(G,\widehat{G})=\#\{\text{edge additions, deletions, reversals}\}.$$

The formula gives a compact handle on notears continuous optimization. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of notears continuous optimization:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for notears continuous optimization is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, notears continuous optimization is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using notears continuous optimization responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, notears continuous optimization is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 4.5 regularization and sparsity

Regularization and sparsity belongs to the canonical scope of Causal Discovery. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is constraint-based, score-based, functional, invariant, and optimization-based causal graph discovery with clear assumptions and evaluation metrics. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$X_j=f_j(\operatorname{pa}_j)+N_j,\qquad N_j \perp\!\!\!\perp \operatorname{pa}_j.$$

The formula gives a compact handle on regularization and sparsity. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of regularization and sparsity:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for regularization and sparsity is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, regularization and sparsity is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using regularization and sparsity responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, regularization and sparsity is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

## 5. Functional and Invariant Methods

Functional and Invariant Methods develops the part of causal discovery specified by the approved Chapter 22 table of contents. The treatment is causal, not merely predictive: the central objects are mechanisms, interventions, assumptions, and counterfactuals.

### 5.1 LiNGAM

Lingam belongs to the canonical scope of Causal Discovery. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is constraint-based, score-based, functional, invariant, and optimization-based causal graph discovery with clear assumptions and evaluation metrics. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$A_{ij}=1 \iff X_i \to X_j.$$

The formula gives a compact handle on lingam. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of lingam:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for lingam is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, lingam is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using lingam responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, lingam is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 5.2 additive noise models

Additive noise models belongs to the canonical scope of Causal Discovery. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is constraint-based, score-based, functional, invariant, and optimization-based causal graph discovery with clear assumptions and evaluation metrics. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$h(W)=\operatorname{tr}(e^{W\odot W})-d=0.$$

The formula gives a compact handle on additive noise models. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of additive noise models:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for additive noise models is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, additive noise models is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using additive noise models responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, additive noise models is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 5.3 invariant causal prediction

Invariant causal prediction belongs to the canonical scope of Causal Discovery. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is constraint-based, score-based, functional, invariant, and optimization-based causal graph discovery with clear assumptions and evaluation metrics. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$\operatorname{SHD}(G,\widehat{G})=\#\{\text{edge additions, deletions, reversals}\}.$$

The formula gives a compact handle on invariant causal prediction. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of invariant causal prediction:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for invariant causal prediction is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, invariant causal prediction is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using invariant causal prediction responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, invariant causal prediction is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 5.4 causal discovery across environments

Causal discovery across environments belongs to the canonical scope of Causal Discovery. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is constraint-based, score-based, functional, invariant, and optimization-based causal graph discovery with clear assumptions and evaluation metrics. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$X_j=f_j(\operatorname{pa}_j)+N_j,\qquad N_j \perp\!\!\!\perp \operatorname{pa}_j.$$

The formula gives a compact handle on causal discovery across environments. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of causal discovery across environments:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for causal discovery across environments is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, causal discovery across environments is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using causal discovery across environments responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, causal discovery across environments is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 5.5 time-series causal discovery preview

Time-series causal discovery preview belongs to the canonical scope of Causal Discovery. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is constraint-based, score-based, functional, invariant, and optimization-based causal graph discovery with clear assumptions and evaluation metrics. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$A_{ij}=1 \iff X_i \to X_j.$$

The formula gives a compact handle on time-series causal discovery preview. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of time-series causal discovery preview:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for time-series causal discovery preview is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, time-series causal discovery preview is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using time-series causal discovery preview responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, time-series causal discovery preview is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

## 6. Evaluation and ML Applications

Evaluation and ML Applications develops the part of causal discovery specified by the approved Chapter 22 table of contents. The treatment is causal, not merely predictive: the central objects are mechanisms, interventions, assumptions, and counterfactuals.

### 6.1 structural Hamming distance

Structural hamming distance belongs to the canonical scope of Causal Discovery. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is constraint-based, score-based, functional, invariant, and optimization-based causal graph discovery with clear assumptions and evaluation metrics. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$h(W)=\operatorname{tr}(e^{W\odot W})-d=0.$$

The formula gives a compact handle on structural hamming distance. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of structural hamming distance:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for structural hamming distance is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, structural hamming distance is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using structural hamming distance responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, structural hamming distance is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 6.2 structural intervention distance preview

Structural intervention distance preview belongs to the canonical scope of Causal Discovery. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is constraint-based, score-based, functional, invariant, and optimization-based causal graph discovery with clear assumptions and evaluation metrics. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$\operatorname{SHD}(G,\widehat{G})=\#\{\text{edge additions, deletions, reversals}\}.$$

The formula gives a compact handle on structural intervention distance preview. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of structural intervention distance preview:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for structural intervention distance preview is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, structural intervention distance preview is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using structural intervention distance preview responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, structural intervention distance preview is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 6.3 synthetic benchmarks

Synthetic benchmarks belongs to the canonical scope of Causal Discovery. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is constraint-based, score-based, functional, invariant, and optimization-based causal graph discovery with clear assumptions and evaluation metrics. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$X_j=f_j(\operatorname{pa}_j)+N_j,\qquad N_j \perp\!\!\!\perp \operatorname{pa}_j.$$

The formula gives a compact handle on synthetic benchmarks. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of synthetic benchmarks:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for synthetic benchmarks is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, synthetic benchmarks is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using synthetic benchmarks responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, synthetic benchmarks is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 6.4 causal feature selection

Causal feature selection belongs to the canonical scope of Causal Discovery. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is constraint-based, score-based, functional, invariant, and optimization-based causal graph discovery with clear assumptions and evaluation metrics. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$A_{ij}=1 \iff X_i \to X_j.$$

The formula gives a compact handle on causal feature selection. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of causal feature selection:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for causal feature selection is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, causal feature selection is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using causal feature selection responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, causal feature selection is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 6.5 LLM-assisted causal hypothesis generation with human review

Llm-assisted causal hypothesis generation with human review belongs to the canonical scope of Causal Discovery. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is constraint-based, score-based, functional, invariant, and optimization-based causal graph discovery with clear assumptions and evaluation metrics. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$h(W)=\operatorname{tr}(e^{W\odot W})-d=0.$$

The formula gives a compact handle on llm-assisted causal hypothesis generation with human review. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of llm-assisted causal hypothesis generation with human review:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for llm-assisted causal hypothesis generation with human review is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, llm-assisted causal hypothesis generation with human review is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using llm-assisted causal hypothesis generation with human review responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, llm-assisted causal hypothesis generation with human review is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

## 7. Common Mistakes

| # | Mistake | Why It Is Wrong | Fix |
| --- | --- | --- | --- |
| 1 | Equating correlation with causation | Conditional association can arise from confounding, selection, or collider bias. | State the causal graph and the target intervention before interpreting associations. |
| 2 | Conditioning on colliders | A collider can open a spurious path when conditioned on. | Use d-separation and adjustment criteria, not variable-importance intuition alone. |
| 3 | Forgetting the estimand-estimator split | Identification is a symbolic question; estimation is a statistical question. | First derive the causal estimand, then choose an estimator and diagnostics. |
| 4 | Using do-calculus without assumptions | The rules operate on a causal graph whose assumptions are supplied by the analyst. | Make graph assumptions explicit and discuss unobserved variables. |
| 5 | Treating counterfactuals as factual labels | Only one potential outcome is observed for each unit. | Use consistency, exchangeability, and sensitivity analysis carefully. |
| 6 | Assuming discovery is assumption-free | Many graphs can imply the same observational distribution. | Report equivalence classes, required assumptions, and intervention needs. |
| 7 | Confusing prediction robustness with causal invariance | A predictive feature can be stable in one dataset and noncausal under intervention. | Use environment shifts and mechanism assumptions to justify causal claims. |
| 8 | Ignoring positivity or overlap | Causal effects cannot be estimated where treatment assignments have no support. | Inspect propensity or support before using adjustment formulas. |
| 9 | Letting ML hide causal design | Flexible nuisance models do not create identification. | Use ML after identification, with cross-fitting or regularization as estimation tools. |
| 10 | Overtrusting LLM causal explanations | Language models can narrate plausible mechanisms without evidence. | Use LLMs for hypothesis generation, then require graph, data, and domain checks. |

## 8. Exercises

1. (*) Work through a causal-inference task for causal discovery.
   - (a) State the causal query using intervention or counterfactual notation.
   - (b) Draw or describe the relevant graph and assumptions.
   - (c) Decide whether the estimand is identifiable from the available data.
   - (d) Give an estimator or diagnostic only after identification is clear.
   - (e) Explain the AI or LLM system implication.

2. (*) Work through a causal-inference task for causal discovery.
   - (a) State the causal query using intervention or counterfactual notation.
   - (b) Draw or describe the relevant graph and assumptions.
   - (c) Decide whether the estimand is identifiable from the available data.
   - (d) Give an estimator or diagnostic only after identification is clear.
   - (e) Explain the AI or LLM system implication.

3. (*) Work through a causal-inference task for causal discovery.
   - (a) State the causal query using intervention or counterfactual notation.
   - (b) Draw or describe the relevant graph and assumptions.
   - (c) Decide whether the estimand is identifiable from the available data.
   - (d) Give an estimator or diagnostic only after identification is clear.
   - (e) Explain the AI or LLM system implication.

4. (**) Work through a causal-inference task for causal discovery.
   - (a) State the causal query using intervention or counterfactual notation.
   - (b) Draw or describe the relevant graph and assumptions.
   - (c) Decide whether the estimand is identifiable from the available data.
   - (d) Give an estimator or diagnostic only after identification is clear.
   - (e) Explain the AI or LLM system implication.

5. (**) Work through a causal-inference task for causal discovery.
   - (a) State the causal query using intervention or counterfactual notation.
   - (b) Draw or describe the relevant graph and assumptions.
   - (c) Decide whether the estimand is identifiable from the available data.
   - (d) Give an estimator or diagnostic only after identification is clear.
   - (e) Explain the AI or LLM system implication.

6. (**) Work through a causal-inference task for causal discovery.
   - (a) State the causal query using intervention or counterfactual notation.
   - (b) Draw or describe the relevant graph and assumptions.
   - (c) Decide whether the estimand is identifiable from the available data.
   - (d) Give an estimator or diagnostic only after identification is clear.
   - (e) Explain the AI or LLM system implication.

7. (***) Work through a causal-inference task for causal discovery.
   - (a) State the causal query using intervention or counterfactual notation.
   - (b) Draw or describe the relevant graph and assumptions.
   - (c) Decide whether the estimand is identifiable from the available data.
   - (d) Give an estimator or diagnostic only after identification is clear.
   - (e) Explain the AI or LLM system implication.

8. (***) Work through a causal-inference task for causal discovery.
   - (a) State the causal query using intervention or counterfactual notation.
   - (b) Draw or describe the relevant graph and assumptions.
   - (c) Decide whether the estimand is identifiable from the available data.
   - (d) Give an estimator or diagnostic only after identification is clear.
   - (e) Explain the AI or LLM system implication.

9. (***) Work through a causal-inference task for causal discovery.
   - (a) State the causal query using intervention or counterfactual notation.
   - (b) Draw or describe the relevant graph and assumptions.
   - (c) Decide whether the estimand is identifiable from the available data.
   - (d) Give an estimator or diagnostic only after identification is clear.
   - (e) Explain the AI or LLM system implication.

10. (***) Work through a causal-inference task for causal discovery.
   - (a) State the causal query using intervention or counterfactual notation.
   - (b) Draw or describe the relevant graph and assumptions.
   - (c) Decide whether the estimand is identifiable from the available data.
   - (d) Give an estimator or diagnostic only after identification is clear.
   - (e) Explain the AI or LLM system implication.

## 9. Why This Matters for AI

| Concept | AI Impact |
| --- | --- |
| SCM | Encodes which mechanisms should stay stable under policy or data changes |
| Do-operator | Separates observing a model behavior from changing an input, policy, or tool |
| Adjustment | Identifies which variables should be controlled for and which should not |
| Counterfactual | Supports recourse, fairness, and unit-level explanation |
| Causal discovery | Generates candidate mechanism graphs when domain knowledge is incomplete |
| Positivity | Prevents extrapolating treatment effects into unsupported regions |
| Hidden confounding | Warns when observational logs cannot support a causal claim |
| Estimand-estimator split | Keeps flexible ML estimators from hiding causal assumptions |

## 10. Conceptual Bridge

Causal Discovery follows statistical learning theory because learning theory explains how observed samples support future prediction claims. Causal inference asks a different question: what happens when an action changes the system that generated those samples?

The backward bridge is risk and uncertainty. Chapter 21 provides language for finite-sample generalization. Chapter 22 adds intervention semantics, graph assumptions, and counterfactual worlds. A causal claim is not just a better prediction; it is a claim about a modified data-generating mechanism.

The forward bridge is game theory. Once multiple agents adapt to interventions, the causal question becomes strategic: actions change incentives, incentives change behavior, and behavior changes the causal system. Chapter 23 will study that interaction explicitly.

```text
+--------------------------------------------------------------+
| Chapter 21: prediction under finite samples                  |
| Chapter 22: intervention, counterfactuals, causal discovery  |
| Chapter 23: strategic interaction and adversarial systems    |
+--------------------------------------------------------------+
```

## References

- Spirtes, Glymour, and Scheines. Causation, Prediction, and Search. https://mitpress.mit.edu/9780262194402/causation-prediction-and-search/
- Peters, Janzing, and Scholkopf. Elements of Causal Inference. https://tile.loc.gov/storage-services/master/gdc/gdcebookspublic/20/20/71/97/58/2020719758/2020719758.pdf
- Zheng et al.. DAGs with NO TEARS: Continuous Optimization for Structure Learning. https://arxiv.org/abs/1803.01422
- Peters et al.. Causal discovery with continuous additive noise models. https://jmlr.org/papers/v15/peters14a.html
