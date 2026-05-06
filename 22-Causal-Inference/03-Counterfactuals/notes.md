[Back to Curriculum](../../README.md) | [Previous: Do Calculus](../02-Do-Calculus/notes.md) | [Next: Causal Discovery](../04-Causal-Discovery/notes.md)

---

# Counterfactuals

> _"A counterfactual asks the model to remember what happened and imagine what would have happened instead."_

## Overview

Counterfactual reasoning studies unit-level alternatives by combining factual evidence with a causal model of interventions.

Causal inference is the part of the curriculum that separates observing from doing. It asks which assumptions allow a learner to move from associations in data to claims about interventions, alternatives, and mechanisms.

This section is written in LaTeX Markdown. Inline mathematics uses `$...$`, and display equations use `$$...$$`. The notes emphasize graph assumptions, intervention notation, counterfactual semantics, and the estimand-estimator split.

## Prerequisites

- [Structural Causal Models](../01-Structural-Causal-Models/notes.md)
- [Do Calculus](../02-Do-Calculus/notes.md)
- [Bayesian Inference](../../07-Statistics/04-Bayesian-Inference/notes.md)
- [Error Analysis and Ablations](../../17-Evaluation-and-Reliability/04-Error-Analysis-and-Ablations/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Executable demonstrations for counterfactuals |
| [exercises.ipynb](exercises.ipynb) | Graded practice for counterfactuals |

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
  - [1.1 what-if reasoning](#11-whatif-reasoning)
  - [1.2 Pearl's ladder of causation](#12-pearls-ladder-of-causation)
  - [1.3 unit-level vs population-level questions](#13-unitlevel-vs-populationlevel-questions)
  - [1.4 why counterfactuals need a model](#14-why-counterfactuals-need-a-model)
  - [1.5 counterfactuals in decisions and explanations](#15-counterfactuals-in-decisions-and-explanations)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 potential outcomes $Y(1),Y(0)$](#21-potential-outcomes)
  - [2.2 SCM counterfactual $Y_x(\mathbf{u})$](#22-scm-counterfactual)
  - [2.3 factual evidence $E=e$](#23-factual-evidence)
  - [2.4 consistency](#24-consistency)
  - [2.5 SUTVA and interference limits](#25-sutva-and-interference-limits)
- [3. Potential Outcomes Framework](#3-potential-outcomes-framework)
  - [3.1 treatment assignment $A$](#31-treatment-assignment)
  - [3.2 ATE ATT and CATE](#32-ate-att-and-cate)
  - [3.3 ignorability and exchangeability](#33-ignorability-and-exchangeability)
  - [3.4 positivity](#34-positivity)
  - [3.5 observed vs missing potential outcomes](#35-observed-vs-missing-potential-outcomes)
- [4. SCM Counterfactual Computation](#4-scm-counterfactual-computation)
  - [4.1 abduction](#41-abduction)
  - [4.2 action](#42-action)
  - [4.3 prediction](#43-prediction)
  - [4.4 twin networks](#44-twin-networks)
  - [4.5 linear counterfactual examples](#45-linear-counterfactual-examples)
- [5. Counterfactual Identification](#5-counterfactual-identification)
  - [5.1 randomized experiments](#51-randomized-experiments)
  - [5.2 observational identification assumptions](#52-observational-identification-assumptions)
  - [5.3 effect of treatment on treated](#53-effect-of-treatment-on-treated)
  - [5.4 mediation and natural effects preview](#54-mediation-and-natural-effects-preview)
  - [5.5 sensitivity to unobserved confounding](#55-sensitivity-to-unobserved-confounding)
- [6. AI Applications](#6-ai-applications)
  - [6.1 counterfactual fairness](#61-counterfactual-fairness)
  - [6.2 algorithmic recourse](#62-algorithmic-recourse)
  - [6.3 offline policy evaluation](#63-offline-policy-evaluation)
  - [6.4 personalized treatment and recommendation](#64-personalized-treatment-and-recommendation)
  - [6.5 limits of LLM-generated counterfactual explanations](#65-limits-of-llmgenerated-counterfactual-explanations)
- [7. Common Mistakes](#7-common-mistakes)
- [8. Exercises](#8-exercises)
- [9. Why This Matters for AI](#9-why-this-matters-for-ai)
- [10. Conceptual Bridge](#10-conceptual-bridge)
- [References](#references)

---

## 1. Intuition

Intuition develops the part of counterfactuals specified by the approved Chapter 22 table of contents. The treatment is causal, not merely predictive: the central objects are mechanisms, interventions, assumptions, and counterfactuals.

### 1.1 what-if reasoning

What-if reasoning belongs to the canonical scope of Counterfactuals. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is potential outcomes, SCM counterfactuals, abduction-action-prediction, twin networks, treatment effects, recourse, and fairness. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$\operatorname{ATE}=\mathbb{E}[Y(1)-Y(0)].$$

The formula gives a compact handle on what-if reasoning. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of what-if reasoning:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for what-if reasoning is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, what-if reasoning is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using what-if reasoning responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, what-if reasoning is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 1.2 Pearl's ladder of causation

Pearl's ladder of causation belongs to the canonical scope of Counterfactuals. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is potential outcomes, SCM counterfactuals, abduction-action-prediction, twin networks, treatment effects, recourse, and fairness. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$\operatorname{ATT}=\mathbb{E}[Y(1)-Y(0) \mid A=1].$$

The formula gives a compact handle on pearl's ladder of causation. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of pearl's ladder of causation:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for pearl's ladder of causation is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, pearl's ladder of causation is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using pearl's ladder of causation responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, pearl's ladder of causation is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 1.3 unit-level vs population-level questions

Unit-level vs population-level questions belongs to the canonical scope of Counterfactuals. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is potential outcomes, SCM counterfactuals, abduction-action-prediction, twin networks, treatment effects, recourse, and fairness. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$Y_x(\mathbf{u})=Y_{M_x}(\mathbf{u}).$$

The formula gives a compact handle on unit-level vs population-level questions. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of unit-level vs population-level questions:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for unit-level vs population-level questions is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, unit-level vs population-level questions is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using unit-level vs population-level questions responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, unit-level vs population-level questions is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 1.4 why counterfactuals need a model

Why counterfactuals need a model belongs to the canonical scope of Counterfactuals. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is potential outcomes, SCM counterfactuals, abduction-action-prediction, twin networks, treatment effects, recourse, and fairness. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P(Y_x=y \mid E=e)=\sum_{\mathbf{u}}\mathbb{1}[Y_x(\mathbf{u})=y]P(\mathbf{u} \mid E=e).$$

The formula gives a compact handle on why counterfactuals need a model. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of why counterfactuals need a model:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for why counterfactuals need a model is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, why counterfactuals need a model is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using why counterfactuals need a model responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, why counterfactuals need a model is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 1.5 counterfactuals in decisions and explanations

Counterfactuals in decisions and explanations belongs to the canonical scope of Counterfactuals. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is potential outcomes, SCM counterfactuals, abduction-action-prediction, twin networks, treatment effects, recourse, and fairness. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$\operatorname{ATE}=\mathbb{E}[Y(1)-Y(0)].$$

The formula gives a compact handle on counterfactuals in decisions and explanations. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of counterfactuals in decisions and explanations:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for counterfactuals in decisions and explanations is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, counterfactuals in decisions and explanations is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using counterfactuals in decisions and explanations responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, counterfactuals in decisions and explanations is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

## 2. Formal Definitions

Formal Definitions develops the part of counterfactuals specified by the approved Chapter 22 table of contents. The treatment is causal, not merely predictive: the central objects are mechanisms, interventions, assumptions, and counterfactuals.

### 2.1 potential outcomes $Y(1),Y(0)$

Potential outcomes $y(1),y(0)$ belongs to the canonical scope of Counterfactuals. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is potential outcomes, SCM counterfactuals, abduction-action-prediction, twin networks, treatment effects, recourse, and fairness. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$\operatorname{ATT}=\mathbb{E}[Y(1)-Y(0) \mid A=1].$$

The formula gives a compact handle on potential outcomes $y(1),y(0)$. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of potential outcomes $y(1),y(0)$:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for potential outcomes $y(1),y(0)$ is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, potential outcomes $y(1),y(0)$ is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using potential outcomes $y(1),y(0)$ responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, potential outcomes $y(1),y(0)$ is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 2.2 SCM counterfactual $Y_x(\mathbf{u})$

Scm counterfactual $y_x(\mathbf{u})$ belongs to the canonical scope of Counterfactuals. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is potential outcomes, SCM counterfactuals, abduction-action-prediction, twin networks, treatment effects, recourse, and fairness. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$Y_x(\mathbf{u})=Y_{M_x}(\mathbf{u}).$$

The formula gives a compact handle on scm counterfactual $y_x(\mathbf{u})$. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of scm counterfactual $y_x(\mathbf{u})$:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for scm counterfactual $y_x(\mathbf{u})$ is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, scm counterfactual $y_x(\mathbf{u})$ is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using scm counterfactual $y_x(\mathbf{u})$ responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, scm counterfactual $y_x(\mathbf{u})$ is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 2.3 factual evidence $E=e$

Factual evidence $e=e$ belongs to the canonical scope of Counterfactuals. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is potential outcomes, SCM counterfactuals, abduction-action-prediction, twin networks, treatment effects, recourse, and fairness. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P(Y_x=y \mid E=e)=\sum_{\mathbf{u}}\mathbb{1}[Y_x(\mathbf{u})=y]P(\mathbf{u} \mid E=e).$$

The formula gives a compact handle on factual evidence $e=e$. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of factual evidence $e=e$:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for factual evidence $e=e$ is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, factual evidence $e=e$ is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using factual evidence $e=e$ responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, factual evidence $e=e$ is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 2.4 consistency

Consistency belongs to the canonical scope of Counterfactuals. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is potential outcomes, SCM counterfactuals, abduction-action-prediction, twin networks, treatment effects, recourse, and fairness. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$\operatorname{ATE}=\mathbb{E}[Y(1)-Y(0)].$$

The formula gives a compact handle on consistency. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of consistency:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for consistency is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, consistency is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using consistency responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, consistency is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 2.5 SUTVA and interference limits

Sutva and interference limits belongs to the canonical scope of Counterfactuals. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is potential outcomes, SCM counterfactuals, abduction-action-prediction, twin networks, treatment effects, recourse, and fairness. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$\operatorname{ATT}=\mathbb{E}[Y(1)-Y(0) \mid A=1].$$

The formula gives a compact handle on sutva and interference limits. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of sutva and interference limits:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for sutva and interference limits is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, sutva and interference limits is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using sutva and interference limits responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, sutva and interference limits is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

## 3. Potential Outcomes Framework

Potential Outcomes Framework develops the part of counterfactuals specified by the approved Chapter 22 table of contents. The treatment is causal, not merely predictive: the central objects are mechanisms, interventions, assumptions, and counterfactuals.

### 3.1 treatment assignment $A$

Treatment assignment $a$ belongs to the canonical scope of Counterfactuals. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is potential outcomes, SCM counterfactuals, abduction-action-prediction, twin networks, treatment effects, recourse, and fairness. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$Y_x(\mathbf{u})=Y_{M_x}(\mathbf{u}).$$

The formula gives a compact handle on treatment assignment $a$. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of treatment assignment $a$:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for treatment assignment $a$ is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, treatment assignment $a$ is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using treatment assignment $a$ responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, treatment assignment $a$ is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 3.2 ATE ATT and CATE

Ate att and cate belongs to the canonical scope of Counterfactuals. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is potential outcomes, SCM counterfactuals, abduction-action-prediction, twin networks, treatment effects, recourse, and fairness. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P(Y_x=y \mid E=e)=\sum_{\mathbf{u}}\mathbb{1}[Y_x(\mathbf{u})=y]P(\mathbf{u} \mid E=e).$$

The formula gives a compact handle on ate att and cate. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of ate att and cate:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for ate att and cate is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, ate att and cate is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using ate att and cate responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, ate att and cate is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 3.3 ignorability and exchangeability

Ignorability and exchangeability belongs to the canonical scope of Counterfactuals. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is potential outcomes, SCM counterfactuals, abduction-action-prediction, twin networks, treatment effects, recourse, and fairness. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$\operatorname{ATE}=\mathbb{E}[Y(1)-Y(0)].$$

The formula gives a compact handle on ignorability and exchangeability. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of ignorability and exchangeability:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for ignorability and exchangeability is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, ignorability and exchangeability is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using ignorability and exchangeability responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, ignorability and exchangeability is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 3.4 positivity

Positivity belongs to the canonical scope of Counterfactuals. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is potential outcomes, SCM counterfactuals, abduction-action-prediction, twin networks, treatment effects, recourse, and fairness. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$\operatorname{ATT}=\mathbb{E}[Y(1)-Y(0) \mid A=1].$$

The formula gives a compact handle on positivity. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of positivity:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for positivity is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, positivity is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using positivity responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, positivity is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 3.5 observed vs missing potential outcomes

Observed vs missing potential outcomes belongs to the canonical scope of Counterfactuals. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is potential outcomes, SCM counterfactuals, abduction-action-prediction, twin networks, treatment effects, recourse, and fairness. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$Y_x(\mathbf{u})=Y_{M_x}(\mathbf{u}).$$

The formula gives a compact handle on observed vs missing potential outcomes. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of observed vs missing potential outcomes:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for observed vs missing potential outcomes is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, observed vs missing potential outcomes is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using observed vs missing potential outcomes responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, observed vs missing potential outcomes is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

## 4. SCM Counterfactual Computation

SCM Counterfactual Computation develops the part of counterfactuals specified by the approved Chapter 22 table of contents. The treatment is causal, not merely predictive: the central objects are mechanisms, interventions, assumptions, and counterfactuals.

### 4.1 abduction

Abduction belongs to the canonical scope of Counterfactuals. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is potential outcomes, SCM counterfactuals, abduction-action-prediction, twin networks, treatment effects, recourse, and fairness. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P(Y_x=y \mid E=e)=\sum_{\mathbf{u}}\mathbb{1}[Y_x(\mathbf{u})=y]P(\mathbf{u} \mid E=e).$$

The formula gives a compact handle on abduction. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of abduction:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for abduction is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, abduction is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using abduction responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, abduction is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 4.2 action

Action belongs to the canonical scope of Counterfactuals. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is potential outcomes, SCM counterfactuals, abduction-action-prediction, twin networks, treatment effects, recourse, and fairness. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$\operatorname{ATE}=\mathbb{E}[Y(1)-Y(0)].$$

The formula gives a compact handle on action. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of action:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for action is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, action is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using action responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, action is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 4.3 prediction

Prediction belongs to the canonical scope of Counterfactuals. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is potential outcomes, SCM counterfactuals, abduction-action-prediction, twin networks, treatment effects, recourse, and fairness. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$\operatorname{ATT}=\mathbb{E}[Y(1)-Y(0) \mid A=1].$$

The formula gives a compact handle on prediction. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of prediction:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for prediction is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, prediction is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using prediction responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, prediction is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 4.4 twin networks

Twin networks belongs to the canonical scope of Counterfactuals. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is potential outcomes, SCM counterfactuals, abduction-action-prediction, twin networks, treatment effects, recourse, and fairness. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$Y_x(\mathbf{u})=Y_{M_x}(\mathbf{u}).$$

The formula gives a compact handle on twin networks. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of twin networks:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for twin networks is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, twin networks is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using twin networks responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, twin networks is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 4.5 linear counterfactual examples

Linear counterfactual examples belongs to the canonical scope of Counterfactuals. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is potential outcomes, SCM counterfactuals, abduction-action-prediction, twin networks, treatment effects, recourse, and fairness. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P(Y_x=y \mid E=e)=\sum_{\mathbf{u}}\mathbb{1}[Y_x(\mathbf{u})=y]P(\mathbf{u} \mid E=e).$$

The formula gives a compact handle on linear counterfactual examples. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of linear counterfactual examples:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for linear counterfactual examples is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, linear counterfactual examples is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using linear counterfactual examples responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, linear counterfactual examples is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

## 5. Counterfactual Identification

Counterfactual Identification develops the part of counterfactuals specified by the approved Chapter 22 table of contents. The treatment is causal, not merely predictive: the central objects are mechanisms, interventions, assumptions, and counterfactuals.

### 5.1 randomized experiments

Randomized experiments belongs to the canonical scope of Counterfactuals. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is potential outcomes, SCM counterfactuals, abduction-action-prediction, twin networks, treatment effects, recourse, and fairness. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$\operatorname{ATE}=\mathbb{E}[Y(1)-Y(0)].$$

The formula gives a compact handle on randomized experiments. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of randomized experiments:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for randomized experiments is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, randomized experiments is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using randomized experiments responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, randomized experiments is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 5.2 observational identification assumptions

Observational identification assumptions belongs to the canonical scope of Counterfactuals. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is potential outcomes, SCM counterfactuals, abduction-action-prediction, twin networks, treatment effects, recourse, and fairness. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$\operatorname{ATT}=\mathbb{E}[Y(1)-Y(0) \mid A=1].$$

The formula gives a compact handle on observational identification assumptions. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of observational identification assumptions:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for observational identification assumptions is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, observational identification assumptions is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using observational identification assumptions responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, observational identification assumptions is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 5.3 effect of treatment on treated

Effect of treatment on treated belongs to the canonical scope of Counterfactuals. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is potential outcomes, SCM counterfactuals, abduction-action-prediction, twin networks, treatment effects, recourse, and fairness. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$Y_x(\mathbf{u})=Y_{M_x}(\mathbf{u}).$$

The formula gives a compact handle on effect of treatment on treated. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of effect of treatment on treated:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for effect of treatment on treated is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, effect of treatment on treated is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using effect of treatment on treated responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, effect of treatment on treated is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 5.4 mediation and natural effects preview

Mediation and natural effects preview belongs to the canonical scope of Counterfactuals. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is potential outcomes, SCM counterfactuals, abduction-action-prediction, twin networks, treatment effects, recourse, and fairness. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P(Y_x=y \mid E=e)=\sum_{\mathbf{u}}\mathbb{1}[Y_x(\mathbf{u})=y]P(\mathbf{u} \mid E=e).$$

The formula gives a compact handle on mediation and natural effects preview. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of mediation and natural effects preview:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for mediation and natural effects preview is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, mediation and natural effects preview is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using mediation and natural effects preview responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, mediation and natural effects preview is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 5.5 sensitivity to unobserved confounding

Sensitivity to unobserved confounding belongs to the canonical scope of Counterfactuals. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is potential outcomes, SCM counterfactuals, abduction-action-prediction, twin networks, treatment effects, recourse, and fairness. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$\operatorname{ATE}=\mathbb{E}[Y(1)-Y(0)].$$

The formula gives a compact handle on sensitivity to unobserved confounding. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of sensitivity to unobserved confounding:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for sensitivity to unobserved confounding is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, sensitivity to unobserved confounding is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using sensitivity to unobserved confounding responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, sensitivity to unobserved confounding is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

## 6. AI Applications

AI Applications develops the part of counterfactuals specified by the approved Chapter 22 table of contents. The treatment is causal, not merely predictive: the central objects are mechanisms, interventions, assumptions, and counterfactuals.

### 6.1 counterfactual fairness

Counterfactual fairness belongs to the canonical scope of Counterfactuals. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is potential outcomes, SCM counterfactuals, abduction-action-prediction, twin networks, treatment effects, recourse, and fairness. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$\operatorname{ATT}=\mathbb{E}[Y(1)-Y(0) \mid A=1].$$

The formula gives a compact handle on counterfactual fairness. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of counterfactual fairness:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for counterfactual fairness is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, counterfactual fairness is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using counterfactual fairness responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, counterfactual fairness is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 6.2 algorithmic recourse

Algorithmic recourse belongs to the canonical scope of Counterfactuals. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is potential outcomes, SCM counterfactuals, abduction-action-prediction, twin networks, treatment effects, recourse, and fairness. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$Y_x(\mathbf{u})=Y_{M_x}(\mathbf{u}).$$

The formula gives a compact handle on algorithmic recourse. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of algorithmic recourse:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for algorithmic recourse is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, algorithmic recourse is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using algorithmic recourse responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, algorithmic recourse is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 6.3 offline policy evaluation

Offline policy evaluation belongs to the canonical scope of Counterfactuals. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is potential outcomes, SCM counterfactuals, abduction-action-prediction, twin networks, treatment effects, recourse, and fairness. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P(Y_x=y \mid E=e)=\sum_{\mathbf{u}}\mathbb{1}[Y_x(\mathbf{u})=y]P(\mathbf{u} \mid E=e).$$

The formula gives a compact handle on offline policy evaluation. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of offline policy evaluation:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for offline policy evaluation is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, offline policy evaluation is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using offline policy evaluation responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, offline policy evaluation is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 6.4 personalized treatment and recommendation

Personalized treatment and recommendation belongs to the canonical scope of Counterfactuals. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is potential outcomes, SCM counterfactuals, abduction-action-prediction, twin networks, treatment effects, recourse, and fairness. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$\operatorname{ATE}=\mathbb{E}[Y(1)-Y(0)].$$

The formula gives a compact handle on personalized treatment and recommendation. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of personalized treatment and recommendation:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for personalized treatment and recommendation is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, personalized treatment and recommendation is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using personalized treatment and recommendation responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, personalized treatment and recommendation is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 6.5 limits of LLM-generated counterfactual explanations

Limits of llm-generated counterfactual explanations belongs to the canonical scope of Counterfactuals. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is potential outcomes, SCM counterfactuals, abduction-action-prediction, twin networks, treatment effects, recourse, and fairness. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$\operatorname{ATT}=\mathbb{E}[Y(1)-Y(0) \mid A=1].$$

The formula gives a compact handle on limits of llm-generated counterfactual explanations. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of limits of llm-generated counterfactual explanations:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for limits of llm-generated counterfactual explanations is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, limits of llm-generated counterfactual explanations is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using limits of llm-generated counterfactual explanations responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, limits of llm-generated counterfactual explanations is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

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

1. (*) Work through a causal-inference task for counterfactuals.
   - (a) State the causal query using intervention or counterfactual notation.
   - (b) Draw or describe the relevant graph and assumptions.
   - (c) Decide whether the estimand is identifiable from the available data.
   - (d) Give an estimator or diagnostic only after identification is clear.
   - (e) Explain the AI or LLM system implication.

2. (*) Work through a causal-inference task for counterfactuals.
   - (a) State the causal query using intervention or counterfactual notation.
   - (b) Draw or describe the relevant graph and assumptions.
   - (c) Decide whether the estimand is identifiable from the available data.
   - (d) Give an estimator or diagnostic only after identification is clear.
   - (e) Explain the AI or LLM system implication.

3. (*) Work through a causal-inference task for counterfactuals.
   - (a) State the causal query using intervention or counterfactual notation.
   - (b) Draw or describe the relevant graph and assumptions.
   - (c) Decide whether the estimand is identifiable from the available data.
   - (d) Give an estimator or diagnostic only after identification is clear.
   - (e) Explain the AI or LLM system implication.

4. (**) Work through a causal-inference task for counterfactuals.
   - (a) State the causal query using intervention or counterfactual notation.
   - (b) Draw or describe the relevant graph and assumptions.
   - (c) Decide whether the estimand is identifiable from the available data.
   - (d) Give an estimator or diagnostic only after identification is clear.
   - (e) Explain the AI or LLM system implication.

5. (**) Work through a causal-inference task for counterfactuals.
   - (a) State the causal query using intervention or counterfactual notation.
   - (b) Draw or describe the relevant graph and assumptions.
   - (c) Decide whether the estimand is identifiable from the available data.
   - (d) Give an estimator or diagnostic only after identification is clear.
   - (e) Explain the AI or LLM system implication.

6. (**) Work through a causal-inference task for counterfactuals.
   - (a) State the causal query using intervention or counterfactual notation.
   - (b) Draw or describe the relevant graph and assumptions.
   - (c) Decide whether the estimand is identifiable from the available data.
   - (d) Give an estimator or diagnostic only after identification is clear.
   - (e) Explain the AI or LLM system implication.

7. (***) Work through a causal-inference task for counterfactuals.
   - (a) State the causal query using intervention or counterfactual notation.
   - (b) Draw or describe the relevant graph and assumptions.
   - (c) Decide whether the estimand is identifiable from the available data.
   - (d) Give an estimator or diagnostic only after identification is clear.
   - (e) Explain the AI or LLM system implication.

8. (***) Work through a causal-inference task for counterfactuals.
   - (a) State the causal query using intervention or counterfactual notation.
   - (b) Draw or describe the relevant graph and assumptions.
   - (c) Decide whether the estimand is identifiable from the available data.
   - (d) Give an estimator or diagnostic only after identification is clear.
   - (e) Explain the AI or LLM system implication.

9. (***) Work through a causal-inference task for counterfactuals.
   - (a) State the causal query using intervention or counterfactual notation.
   - (b) Draw or describe the relevant graph and assumptions.
   - (c) Decide whether the estimand is identifiable from the available data.
   - (d) Give an estimator or diagnostic only after identification is clear.
   - (e) Explain the AI or LLM system implication.

10. (***) Work through a causal-inference task for counterfactuals.
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

Counterfactuals follows statistical learning theory because learning theory explains how observed samples support future prediction claims. Causal inference asks a different question: what happens when an action changes the system that generated those samples?

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

- Rubin. Estimating Causal Effects of Treatments in Randomized and Nonrandomized Studies. https://dash.harvard.edu/entities/publication/73120378-82c0-6bd4-e053-0100007fdf3b
- Holland. Statistics and Causal Inference. https://www.ets.org/research/policy_research_reports/publications/article/1986/ajqr.html
- Imbens and Rubin. Causal Inference for Statistics, Social, and Biomedical Sciences. https://www.cambridge.org/core/books/causal-inference-for-statistics-social-and-biomedical-sciences/71126BE90C58F1A431FE9B2DD07938AB
- Pearl. Causality: Models, Reasoning, and Inference. https://www.cambridge.org/core/books/causality/6836DD2F4FD4A767DE97BBECDD1655F5
