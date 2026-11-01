[Back to Curriculum](../../README.md) | [Previous: Rademacher Complexity](../../21-Statistical-Learning-Theory/05-Rademacher-Complexity/notes.md) | [Next: Do Calculus](../02-Do-Calculus/notes.md)

---

# Structural Causal Models

> _"Causality begins when a distribution is given a mechanism."_

## Overview

Structural causal models encode how variables are generated so interventions can be represented as changes to mechanisms, not merely changes to observations.

Causal inference is the part of the curriculum that separates observing from doing. It asks which assumptions allow a learner to move from associations in data to claims about interventions, alternatives, and mechanisms.

This section is written in LaTeX Markdown. Inline mathematics uses `$...$`, and display equations use `$$...$$`. The notes emphasize graph assumptions, intervention notation, counterfactual semantics, and the estimand-estimator split.

## Prerequisites

- [Joint Distributions](../../06-Probability-Theory/03-Joint-Distributions/notes.md)
- [Bayesian Inference](../../07-Statistics/04-Bayesian-Inference/notes.md)
- [Graph Basics and Representations](../../11-Graph-Theory/01-Graph-Basics-and-Representations/notes.md)
- [Rademacher Complexity](../../21-Statistical-Learning-Theory/05-Rademacher-Complexity/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Executable demonstrations for structural causal models |
| [exercises.ipynb](exercises.ipynb) | Graded practice for structural causal models |

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
  - [1.1 correlation vs causation](#11-correlation-vs-causation)
  - [1.2 mechanisms as stable assignments](#12-mechanisms-as-stable-assignments)
  - [1.3 DAGs as causal assumptions](#13-dags-as-causal-assumptions)
  - [1.4 interventions as model surgery](#14-interventions-as-model-surgery)
  - [1.5 why SCMs matter for ML distribution shift](#15-why-scms-matter-for-ml-distribution-shift)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 exogenous variables $\mathbf{U}$](#21-exogenous-variables)
  - [2.2 endogenous variables $\mathbf{V}$](#22-endogenous-variables)
  - [2.3 structural assignments $V_i=f_i(\operatorname{pa}_i,U_i)$](#23-structural-assignments)
  - [2.4 causal graph $G$](#24-causal-graph)
  - [2.5 observational interventional and counterfactual distributions](#25-observational-interventional-and-counterfactual-distributions)
- [3. Graph Semantics](#3-graph-semantics)
  - [3.1 parents descendants and ancestors](#31-parents-descendants-and-ancestors)
  - [3.2 paths and blocked paths](#32-paths-and-blocked-paths)
  - [3.3 d-separation](#33-dseparation)
  - [3.4 causal Markov property](#34-causal-markov-property)
  - [3.5 latent confounding and bidirected edges](#35-latent-confounding-and-bidirected-edges)
- [4. Structural Equations](#4-structural-equations)
  - [4.1 linear SCMs](#41-linear-scms)
  - [4.2 nonlinear SCMs](#42-nonlinear-scms)
  - [4.3 independent noise terms](#43-independent-noise-terms)
  - [4.4 modularity and autonomy](#44-modularity-and-autonomy)
  - [4.5 Markovian vs semi-Markovian models](#45-markovian-vs-semimarkovian-models)
- [5. Causal Effects Preview](#5-causal-effects-preview)
  - [5.1 total causal effect](#51-total-causal-effect)
  - [5.2 intervention distribution $P(Y \mid \operatorname{do}(X=x))$](#52-intervention-distribution)
  - [5.3 backdoor adjustment preview](#53-backdoor-adjustment-preview)
  - [5.4 frontdoor preview](#54-frontdoor-preview)
  - [5.5 estimand vs estimator](#55-estimand-vs-estimator)
- [6. ML and LLM Applications](#6-ml-and-llm-applications)
  - [6.1 causal representation learning](#61-causal-representation-learning)
  - [6.2 robust prediction under shift](#62-robust-prediction-under-shift)
  - [6.3 recommender interventions](#63-recommender-interventions)
  - [6.4 fairness and proxy variables](#64-fairness-and-proxy-variables)
  - [6.5 causal logging for LLM agents](#65-causal-logging-for-llm-agents)
- [7. Common Mistakes](#7-common-mistakes)
- [8. Exercises](#8-exercises)
- [9. Why This Matters for AI](#9-why-this-matters-for-ai)
- [10. Conceptual Bridge](#10-conceptual-bridge)
- [References](#references)

---

## 1. Intuition

Intuition develops the part of structural causal models specified by the approved Chapter 22 table of contents. The treatment is causal, not merely predictive: the central objects are mechanisms, interventions, assumptions, and counterfactuals.

### 1.1 correlation vs causation

Correlation vs causation belongs to the canonical scope of Structural Causal Models. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is structural assignments, causal graphs, d-separation, interventions, Markovian assumptions, and SCM links to robust ML. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$M=(\mathbf{U},\mathbf{V},\mathbf{F},P(\mathbf{U})).$$

The formula gives a compact handle on correlation vs causation. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of correlation vs causation:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for correlation vs causation is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, correlation vs causation is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using correlation vs causation responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, correlation vs causation is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 1.2 mechanisms as stable assignments

Mechanisms as stable assignments belongs to the canonical scope of Structural Causal Models. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is structural assignments, causal graphs, d-separation, interventions, Markovian assumptions, and SCM links to robust ML. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$V_i=f_i(\operatorname{pa}_i,U_i).$$

The formula gives a compact handle on mechanisms as stable assignments. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of mechanisms as stable assignments:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for mechanisms as stable assignments is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, mechanisms as stable assignments is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using mechanisms as stable assignments responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, mechanisms as stable assignments is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 1.3 DAGs as causal assumptions

Dags as causal assumptions belongs to the canonical scope of Structural Causal Models. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is structural assignments, causal graphs, d-separation, interventions, Markovian assumptions, and SCM links to robust ML. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P(\mathbf{v})=\prod_{i=1}^{d}P(v_i \mid \operatorname{pa}_i).$$

The formula gives a compact handle on dags as causal assumptions. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of dags as causal assumptions:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for dags as causal assumptions is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, dags as causal assumptions is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using dags as causal assumptions responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, dags as causal assumptions is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 1.4 interventions as model surgery

Interventions as model surgery belongs to the canonical scope of Structural Causal Models. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is structural assignments, causal graphs, d-separation, interventions, Markovian assumptions, and SCM links to robust ML. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P_M(Y \mid \operatorname{do}(X=x))=P_{M_x}(Y).$$

The formula gives a compact handle on interventions as model surgery. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of interventions as model surgery:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for interventions as model surgery is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, interventions as model surgery is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using interventions as model surgery responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, interventions as model surgery is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 1.5 why SCMs matter for ML distribution shift

Why scms matter for ml distribution shift belongs to the canonical scope of Structural Causal Models. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is structural assignments, causal graphs, d-separation, interventions, Markovian assumptions, and SCM links to robust ML. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$M=(\mathbf{U},\mathbf{V},\mathbf{F},P(\mathbf{U})).$$

The formula gives a compact handle on why scms matter for ml distribution shift. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of why scms matter for ml distribution shift:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for why scms matter for ml distribution shift is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, why scms matter for ml distribution shift is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using why scms matter for ml distribution shift responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, why scms matter for ml distribution shift is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

## 2. Formal Definitions

Formal Definitions develops the part of structural causal models specified by the approved Chapter 22 table of contents. The treatment is causal, not merely predictive: the central objects are mechanisms, interventions, assumptions, and counterfactuals.

### 2.1 exogenous variables $\mathbf{U}$

Exogenous variables $\mathbf{u}$ belongs to the canonical scope of Structural Causal Models. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is structural assignments, causal graphs, d-separation, interventions, Markovian assumptions, and SCM links to robust ML. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$V_i=f_i(\operatorname{pa}_i,U_i).$$

The formula gives a compact handle on exogenous variables $\mathbf{u}$. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of exogenous variables $\mathbf{u}$:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for exogenous variables $\mathbf{u}$ is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, exogenous variables $\mathbf{u}$ is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using exogenous variables $\mathbf{u}$ responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, exogenous variables $\mathbf{u}$ is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 2.2 endogenous variables $\mathbf{V}$

Endogenous variables $\mathbf{v}$ belongs to the canonical scope of Structural Causal Models. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is structural assignments, causal graphs, d-separation, interventions, Markovian assumptions, and SCM links to robust ML. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P(\mathbf{v})=\prod_{i=1}^{d}P(v_i \mid \operatorname{pa}_i).$$

The formula gives a compact handle on endogenous variables $\mathbf{v}$. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of endogenous variables $\mathbf{v}$:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for endogenous variables $\mathbf{v}$ is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, endogenous variables $\mathbf{v}$ is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using endogenous variables $\mathbf{v}$ responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, endogenous variables $\mathbf{v}$ is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 2.3 structural assignments $V_i=f_i(\operatorname{pa}_i,U_i)$

Structural assignments $v_i=f_i(\operatorname{pa}_i,u_i)$ belongs to the canonical scope of Structural Causal Models. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is structural assignments, causal graphs, d-separation, interventions, Markovian assumptions, and SCM links to robust ML. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P_M(Y \mid \operatorname{do}(X=x))=P_{M_x}(Y).$$

The formula gives a compact handle on structural assignments $v_i=f_i(\operatorname{pa}_i,u_i)$. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of structural assignments $v_i=f_i(\operatorname{pa}_i,u_i)$:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for structural assignments $v_i=f_i(\operatorname{pa}_i,u_i)$ is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, structural assignments $v_i=f_i(\operatorname{pa}_i,u_i)$ is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using structural assignments $v_i=f_i(\operatorname{pa}_i,u_i)$ responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, structural assignments $v_i=f_i(\operatorname{pa}_i,u_i)$ is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 2.4 causal graph $G$

Causal graph $g$ belongs to the canonical scope of Structural Causal Models. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is structural assignments, causal graphs, d-separation, interventions, Markovian assumptions, and SCM links to robust ML. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$M=(\mathbf{U},\mathbf{V},\mathbf{F},P(\mathbf{U})).$$

The formula gives a compact handle on causal graph $g$. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of causal graph $g$:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for causal graph $g$ is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, causal graph $g$ is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using causal graph $g$ responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, causal graph $g$ is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 2.5 observational interventional and counterfactual distributions

Observational interventional and counterfactual distributions belongs to the canonical scope of Structural Causal Models. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is structural assignments, causal graphs, d-separation, interventions, Markovian assumptions, and SCM links to robust ML. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$V_i=f_i(\operatorname{pa}_i,U_i).$$

The formula gives a compact handle on observational interventional and counterfactual distributions. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of observational interventional and counterfactual distributions:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for observational interventional and counterfactual distributions is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, observational interventional and counterfactual distributions is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using observational interventional and counterfactual distributions responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, observational interventional and counterfactual distributions is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

## 3. Graph Semantics

Graph Semantics develops the part of structural causal models specified by the approved Chapter 22 table of contents. The treatment is causal, not merely predictive: the central objects are mechanisms, interventions, assumptions, and counterfactuals.

### 3.1 parents descendants and ancestors

Parents descendants and ancestors belongs to the canonical scope of Structural Causal Models. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is structural assignments, causal graphs, d-separation, interventions, Markovian assumptions, and SCM links to robust ML. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P(\mathbf{v})=\prod_{i=1}^{d}P(v_i \mid \operatorname{pa}_i).$$

The formula gives a compact handle on parents descendants and ancestors. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of parents descendants and ancestors:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for parents descendants and ancestors is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, parents descendants and ancestors is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using parents descendants and ancestors responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, parents descendants and ancestors is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 3.2 paths and blocked paths

Paths and blocked paths belongs to the canonical scope of Structural Causal Models. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is structural assignments, causal graphs, d-separation, interventions, Markovian assumptions, and SCM links to robust ML. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P_M(Y \mid \operatorname{do}(X=x))=P_{M_x}(Y).$$

The formula gives a compact handle on paths and blocked paths. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of paths and blocked paths:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for paths and blocked paths is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, paths and blocked paths is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using paths and blocked paths responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, paths and blocked paths is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 3.3 d-separation

D-separation belongs to the canonical scope of Structural Causal Models. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is structural assignments, causal graphs, d-separation, interventions, Markovian assumptions, and SCM links to robust ML. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$M=(\mathbf{U},\mathbf{V},\mathbf{F},P(\mathbf{U})).$$

The formula gives a compact handle on d-separation. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of d-separation:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for d-separation is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, d-separation is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using d-separation responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, d-separation is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 3.4 causal Markov property

Causal markov property belongs to the canonical scope of Structural Causal Models. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is structural assignments, causal graphs, d-separation, interventions, Markovian assumptions, and SCM links to robust ML. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$V_i=f_i(\operatorname{pa}_i,U_i).$$

The formula gives a compact handle on causal markov property. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of causal markov property:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for causal markov property is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, causal markov property is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using causal markov property responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, causal markov property is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 3.5 latent confounding and bidirected edges

Latent confounding and bidirected edges belongs to the canonical scope of Structural Causal Models. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is structural assignments, causal graphs, d-separation, interventions, Markovian assumptions, and SCM links to robust ML. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P(\mathbf{v})=\prod_{i=1}^{d}P(v_i \mid \operatorname{pa}_i).$$

The formula gives a compact handle on latent confounding and bidirected edges. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of latent confounding and bidirected edges:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for latent confounding and bidirected edges is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, latent confounding and bidirected edges is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using latent confounding and bidirected edges responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, latent confounding and bidirected edges is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

## 4. Structural Equations

Structural Equations develops the part of structural causal models specified by the approved Chapter 22 table of contents. The treatment is causal, not merely predictive: the central objects are mechanisms, interventions, assumptions, and counterfactuals.

### 4.1 linear SCMs

Linear scms belongs to the canonical scope of Structural Causal Models. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is structural assignments, causal graphs, d-separation, interventions, Markovian assumptions, and SCM links to robust ML. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P_M(Y \mid \operatorname{do}(X=x))=P_{M_x}(Y).$$

The formula gives a compact handle on linear scms. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of linear scms:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for linear scms is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, linear scms is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using linear scms responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, linear scms is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 4.2 nonlinear SCMs

Nonlinear scms belongs to the canonical scope of Structural Causal Models. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is structural assignments, causal graphs, d-separation, interventions, Markovian assumptions, and SCM links to robust ML. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$M=(\mathbf{U},\mathbf{V},\mathbf{F},P(\mathbf{U})).$$

The formula gives a compact handle on nonlinear scms. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of nonlinear scms:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for nonlinear scms is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, nonlinear scms is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using nonlinear scms responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, nonlinear scms is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 4.3 independent noise terms

Independent noise terms belongs to the canonical scope of Structural Causal Models. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is structural assignments, causal graphs, d-separation, interventions, Markovian assumptions, and SCM links to robust ML. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$V_i=f_i(\operatorname{pa}_i,U_i).$$

The formula gives a compact handle on independent noise terms. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of independent noise terms:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for independent noise terms is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, independent noise terms is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using independent noise terms responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, independent noise terms is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 4.4 modularity and autonomy

Modularity and autonomy belongs to the canonical scope of Structural Causal Models. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is structural assignments, causal graphs, d-separation, interventions, Markovian assumptions, and SCM links to robust ML. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P(\mathbf{v})=\prod_{i=1}^{d}P(v_i \mid \operatorname{pa}_i).$$

The formula gives a compact handle on modularity and autonomy. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of modularity and autonomy:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for modularity and autonomy is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, modularity and autonomy is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using modularity and autonomy responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, modularity and autonomy is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 4.5 Markovian vs semi-Markovian models

Markovian vs semi-markovian models belongs to the canonical scope of Structural Causal Models. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is structural assignments, causal graphs, d-separation, interventions, Markovian assumptions, and SCM links to robust ML. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P_M(Y \mid \operatorname{do}(X=x))=P_{M_x}(Y).$$

The formula gives a compact handle on markovian vs semi-markovian models. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of markovian vs semi-markovian models:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for markovian vs semi-markovian models is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, markovian vs semi-markovian models is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using markovian vs semi-markovian models responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, markovian vs semi-markovian models is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

## 5. Causal Effects Preview

Causal Effects Preview develops the part of structural causal models specified by the approved Chapter 22 table of contents. The treatment is causal, not merely predictive: the central objects are mechanisms, interventions, assumptions, and counterfactuals.

### 5.1 total causal effect

Total causal effect belongs to the canonical scope of Structural Causal Models. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is structural assignments, causal graphs, d-separation, interventions, Markovian assumptions, and SCM links to robust ML. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$M=(\mathbf{U},\mathbf{V},\mathbf{F},P(\mathbf{U})).$$

The formula gives a compact handle on total causal effect. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of total causal effect:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for total causal effect is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, total causal effect is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using total causal effect responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, total causal effect is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 5.2 intervention distribution $P(Y \mid \operatorname{do}(X=x))$

Intervention distribution $p(y \mid \operatorname{do}(x=x))$ belongs to the canonical scope of Structural Causal Models. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is structural assignments, causal graphs, d-separation, interventions, Markovian assumptions, and SCM links to robust ML. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$V_i=f_i(\operatorname{pa}_i,U_i).$$

The formula gives a compact handle on intervention distribution $p(y \mid \operatorname{do}(x=x))$. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of intervention distribution $p(y \mid \operatorname{do}(x=x))$:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for intervention distribution $p(y \mid \operatorname{do}(x=x))$ is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, intervention distribution $p(y \mid \operatorname{do}(x=x))$ is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using intervention distribution $p(y \mid \operatorname{do}(x=x))$ responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, intervention distribution $p(y \mid \operatorname{do}(x=x))$ is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 5.3 backdoor adjustment preview

Backdoor adjustment preview belongs to the canonical scope of Structural Causal Models. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is structural assignments, causal graphs, d-separation, interventions, Markovian assumptions, and SCM links to robust ML. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P(\mathbf{v})=\prod_{i=1}^{d}P(v_i \mid \operatorname{pa}_i).$$

The formula gives a compact handle on backdoor adjustment preview. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of backdoor adjustment preview:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for backdoor adjustment preview is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, backdoor adjustment preview is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using backdoor adjustment preview responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, backdoor adjustment preview is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 5.4 frontdoor preview

Frontdoor preview belongs to the canonical scope of Structural Causal Models. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is structural assignments, causal graphs, d-separation, interventions, Markovian assumptions, and SCM links to robust ML. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P_M(Y \mid \operatorname{do}(X=x))=P_{M_x}(Y).$$

The formula gives a compact handle on frontdoor preview. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of frontdoor preview:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for frontdoor preview is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, frontdoor preview is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using frontdoor preview responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, frontdoor preview is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 5.5 estimand vs estimator

Estimand vs estimator belongs to the canonical scope of Structural Causal Models. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is structural assignments, causal graphs, d-separation, interventions, Markovian assumptions, and SCM links to robust ML. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$M=(\mathbf{U},\mathbf{V},\mathbf{F},P(\mathbf{U})).$$

The formula gives a compact handle on estimand vs estimator. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of estimand vs estimator:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for estimand vs estimator is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, estimand vs estimator is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using estimand vs estimator responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, estimand vs estimator is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

## 6. ML and LLM Applications

ML and LLM Applications develops the part of structural causal models specified by the approved Chapter 22 table of contents. The treatment is causal, not merely predictive: the central objects are mechanisms, interventions, assumptions, and counterfactuals.

### 6.1 causal representation learning

Causal representation learning belongs to the canonical scope of Structural Causal Models. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is structural assignments, causal graphs, d-separation, interventions, Markovian assumptions, and SCM links to robust ML. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$V_i=f_i(\operatorname{pa}_i,U_i).$$

The formula gives a compact handle on causal representation learning. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of causal representation learning:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for causal representation learning is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, causal representation learning is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using causal representation learning responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, causal representation learning is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 6.2 robust prediction under shift

Robust prediction under shift belongs to the canonical scope of Structural Causal Models. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is structural assignments, causal graphs, d-separation, interventions, Markovian assumptions, and SCM links to robust ML. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P(\mathbf{v})=\prod_{i=1}^{d}P(v_i \mid \operatorname{pa}_i).$$

The formula gives a compact handle on robust prediction under shift. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of robust prediction under shift:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for robust prediction under shift is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, robust prediction under shift is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using robust prediction under shift responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, robust prediction under shift is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 6.3 recommender interventions

Recommender interventions belongs to the canonical scope of Structural Causal Models. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is structural assignments, causal graphs, d-separation, interventions, Markovian assumptions, and SCM links to robust ML. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P_M(Y \mid \operatorname{do}(X=x))=P_{M_x}(Y).$$

The formula gives a compact handle on recommender interventions. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of recommender interventions:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for recommender interventions is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, recommender interventions is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using recommender interventions responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, recommender interventions is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 6.4 fairness and proxy variables

Fairness and proxy variables belongs to the canonical scope of Structural Causal Models. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is structural assignments, causal graphs, d-separation, interventions, Markovian assumptions, and SCM links to robust ML. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$M=(\mathbf{U},\mathbf{V},\mathbf{F},P(\mathbf{U})).$$

The formula gives a compact handle on fairness and proxy variables. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of fairness and proxy variables:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for fairness and proxy variables is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, fairness and proxy variables is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using fairness and proxy variables responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, fairness and proxy variables is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 6.5 causal logging for LLM agents

Causal logging for llm agents belongs to the canonical scope of Structural Causal Models. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is structural assignments, causal graphs, d-separation, interventions, Markovian assumptions, and SCM links to robust ML. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$V_i=f_i(\operatorname{pa}_i,U_i).$$

The formula gives a compact handle on causal logging for llm agents. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of causal logging for llm agents:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for causal logging for llm agents is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, causal logging for llm agents is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using causal logging for llm agents responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, causal logging for llm agents is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

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

1. (*) Work through a causal-inference task for structural causal models.
   - (a) State the causal query using intervention or counterfactual notation.
   - (b) Draw or describe the relevant graph and assumptions.
   - (c) Decide whether the estimand is identifiable from the available data.
   - (d) Give an estimator or diagnostic only after identification is clear.
   - (e) Explain the AI or LLM system implication.

2. (*) Work through a causal-inference task for structural causal models.
   - (a) State the causal query using intervention or counterfactual notation.
   - (b) Draw or describe the relevant graph and assumptions.
   - (c) Decide whether the estimand is identifiable from the available data.
   - (d) Give an estimator or diagnostic only after identification is clear.
   - (e) Explain the AI or LLM system implication.

3. (*) Work through a causal-inference task for structural causal models.
   - (a) State the causal query using intervention or counterfactual notation.
   - (b) Draw or describe the relevant graph and assumptions.
   - (c) Decide whether the estimand is identifiable from the available data.
   - (d) Give an estimator or diagnostic only after identification is clear.
   - (e) Explain the AI or LLM system implication.

4. (**) Work through a causal-inference task for structural causal models.
   - (a) State the causal query using intervention or counterfactual notation.
   - (b) Draw or describe the relevant graph and assumptions.
   - (c) Decide whether the estimand is identifiable from the available data.
   - (d) Give an estimator or diagnostic only after identification is clear.
   - (e) Explain the AI or LLM system implication.

5. (**) Work through a causal-inference task for structural causal models.
   - (a) State the causal query using intervention or counterfactual notation.
   - (b) Draw or describe the relevant graph and assumptions.
   - (c) Decide whether the estimand is identifiable from the available data.
   - (d) Give an estimator or diagnostic only after identification is clear.
   - (e) Explain the AI or LLM system implication.

6. (**) Work through a causal-inference task for structural causal models.
   - (a) State the causal query using intervention or counterfactual notation.
   - (b) Draw or describe the relevant graph and assumptions.
   - (c) Decide whether the estimand is identifiable from the available data.
   - (d) Give an estimator or diagnostic only after identification is clear.
   - (e) Explain the AI or LLM system implication.

7. (***) Work through a causal-inference task for structural causal models.
   - (a) State the causal query using intervention or counterfactual notation.
   - (b) Draw or describe the relevant graph and assumptions.
   - (c) Decide whether the estimand is identifiable from the available data.
   - (d) Give an estimator or diagnostic only after identification is clear.
   - (e) Explain the AI or LLM system implication.

8. (***) Work through a causal-inference task for structural causal models.
   - (a) State the causal query using intervention or counterfactual notation.
   - (b) Draw or describe the relevant graph and assumptions.
   - (c) Decide whether the estimand is identifiable from the available data.
   - (d) Give an estimator or diagnostic only after identification is clear.
   - (e) Explain the AI or LLM system implication.

9. (***) Work through a causal-inference task for structural causal models.
   - (a) State the causal query using intervention or counterfactual notation.
   - (b) Draw or describe the relevant graph and assumptions.
   - (c) Decide whether the estimand is identifiable from the available data.
   - (d) Give an estimator or diagnostic only after identification is clear.
   - (e) Explain the AI or LLM system implication.

10. (***) Work through a causal-inference task for structural causal models.
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

Structural Causal Models follows statistical learning theory because learning theory explains how observed samples support future prediction claims. Causal inference asks a different question: what happens when an action changes the system that generated those samples?

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

- Pearl. Causality: Models, Reasoning, and Inference. https://www.cambridge.org/core/books/causality/6836DD2F4FD4A767DE97BBECDD1655F5
- Pearl. Causal inference in statistics: An overview. https://projecteuclid.org/journals/statistics-surveys/volume-3/issue-none/Causal-inference-in-statistics-An-overview/10.1214/09-SS057.pdf
- Peters, Janzing, and Scholkopf. Elements of Causal Inference. https://tile.loc.gov/storage-services/master/gdc/gdcebookspublic/20/20/71/97/58/2020719758/2020719758.pdf
- Pearl, Glymour, and Jewell. Causal Inference in Statistics: A Primer. https://www.wiley.com/en-us/Causal+Inference+in+Statistics%3A+A+Primer-p-9781119186847
