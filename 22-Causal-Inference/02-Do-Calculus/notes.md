[Back to Curriculum](../../README.md) | [Previous: Structural Causal Models](../01-Structural-Causal-Models/notes.md) | [Next: Counterfactuals](../03-Counterfactuals/notes.md)

---

# Do Calculus

> _"The do-operator marks the difference between watching a system and changing it."_

## Overview

Do-calculus gives graph-based rules for transforming interventional causal queries into estimable observational expressions when assumptions permit identification.

Causal inference is the part of the curriculum that separates observing from doing. It asks which assumptions allow a learner to move from associations in data to claims about interventions, alternatives, and mechanisms.

This section is written in LaTeX Markdown. Inline mathematics uses `$...$`, and display equations use `$$...$$`. The notes emphasize graph assumptions, intervention notation, counterfactual semantics, and the estimand-estimator split.

## Prerequisites

- [Structural Causal Models](../01-Structural-Causal-Models/notes.md)
- [Conditional Probability](../../06-Probability-Theory/03-Joint-Distributions/notes.md)
- [Graph Algorithms](../../11-Graph-Theory/02-Graph-Algorithms/notes.md)
- [Hypothesis Testing](../../07-Statistics/03-Hypothesis-Testing/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Executable demonstrations for do calculus |
| [exercises.ipynb](exercises.ipynb) | Graded practice for do calculus |

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
  - [1.1 conditioning is not intervening](#11-conditioning-is-not-intervening)
  - [1.2 identification as expression rewriting](#12-identification-as-expression-rewriting)
  - [1.3 why graphs are needed](#13-why-graphs-are-needed)
  - [1.4 what do-calculus can and cannot prove](#14-what-docalculus-can-and-cannot-prove)
  - [1.5 observational data plus assumptions](#15-observational-data-plus-assumptions)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 do-operator $\operatorname{do}(X=x)$](#21-dooperator)
  - [2.2 mutilated graph $G_{\overline{X}}$](#22-mutilated-graph)
  - [2.3 causal effect $P(Y \mid \operatorname{do}(X=x))$](#23-causal-effect)
  - [2.4 identifiability](#24-identifiability)
  - [2.5 adjustment set](#25-adjustment-set)
- [3. Backdoor and Frontdoor](#3-backdoor-and-frontdoor)
  - [3.1 backdoor paths](#31-backdoor-paths)
  - [3.2 backdoor criterion](#32-backdoor-criterion)
  - [3.3 adjustment formula](#33-adjustment-formula)
  - [3.4 frontdoor criterion](#34-frontdoor-criterion)
  - [3.5 confounding and mediators](#35-confounding-and-mediators)
- [4. The Three Rules](#4-the-three-rules)
  - [4.1 insertion and deletion of observations](#41-insertion-and-deletion-of-observations)
  - [4.2 action observation exchange](#42-action-observation-exchange)
  - [4.3 insertion and deletion of actions](#43-insertion-and-deletion-of-actions)
  - [4.4 graph conditions for each rule](#44-graph-conditions-for-each-rule)
  - [4.5 proof intuition from d-separation](#45-proof-intuition-from-dseparation)
- [5. Identification Workflows](#5-identification-workflows)
  - [5.1 manual graph simplification](#51-manual-graph-simplification)
  - [5.2 ID algorithm preview](#52-id-algorithm-preview)
  - [5.3 c-components and hedges](#53-ccomponents-and-hedges)
  - [5.4 Tian-Pearl identification condition preview](#54-tianpearl-identification-condition-preview)
  - [5.5 when an effect is not identifiable](#55-when-an-effect-is-not-identifiable)
- [6. Estimation After Identification](#6-estimation-after-identification)
  - [6.1 plug-in adjustment](#61-plugin-adjustment)
  - [6.2 inverse-propensity weighting preview](#62-inversepropensity-weighting-preview)
  - [6.3 doubly robust preview](#63-doubly-robust-preview)
  - [6.4 positivity and overlap](#64-positivity-and-overlap)
  - [6.5 nuisance ML and cross-fitting preview](#65-nuisance-ml-and-crossfitting-preview)
- [7. ML and LLM Applications](#7-ml-and-llm-applications)
  - [7.1 causal effect of ranking policies](#71-causal-effect-of-ranking-policies)
  - [7.2 logged bandit interventions](#72-logged-bandit-interventions)
  - [7.3 prompt and tool interventions](#73-prompt-and-tool-interventions)
  - [7.4 safety policy changes](#74-safety-policy-changes)
  - [7.5 causal debugging of model behavior](#75-causal-debugging-of-model-behavior)
- [8. Common Mistakes](#8-common-mistakes)
- [9. Exercises](#9-exercises)
- [10. Why This Matters for AI](#10-why-this-matters-for-ai)
- [11. Conceptual Bridge](#11-conceptual-bridge)
- [References](#references)

---

## 1. Intuition

Intuition develops the part of do calculus specified by the approved Chapter 22 table of contents. The treatment is causal, not merely predictive: the central objects are mechanisms, interventions, assumptions, and counterfactuals.

### 1.1 conditioning is not intervening

Conditioning is not intervening belongs to the canonical scope of Do Calculus. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is do-operator semantics, mutilated graphs, backdoor and frontdoor criteria, identification rules, and post-identification estimation. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P(Y \mid \operatorname{do}(X=x)) \ne P(Y \mid X=x) \quad \text{in general}.$$

The formula gives a compact handle on conditioning is not intervening. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of conditioning is not intervening:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for conditioning is not intervening is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, conditioning is not intervening is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using conditioning is not intervening responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, conditioning is not intervening is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 1.2 identification as expression rewriting

Identification as expression rewriting belongs to the canonical scope of Do Calculus. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is do-operator semantics, mutilated graphs, backdoor and frontdoor criteria, identification rules, and post-identification estimation. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P(Y \mid \operatorname{do}(X=x))=\sum_z P(Y \mid X=x,Z=z)P(Z=z).$$

The formula gives a compact handle on identification as expression rewriting. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of identification as expression rewriting:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for identification as expression rewriting is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, identification as expression rewriting is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using identification as expression rewriting responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, identification as expression rewriting is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 1.3 why graphs are needed

Why graphs are needed belongs to the canonical scope of Do Calculus. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is do-operator semantics, mutilated graphs, backdoor and frontdoor criteria, identification rules, and post-identification estimation. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P(Y \mid \operatorname{do}(X=x))=\sum_z P(Z=z \mid X=x)\sum_{x'}P(Y \mid X=x',Z=z)P(X=x').$$

The formula gives a compact handle on why graphs are needed. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of why graphs are needed:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for why graphs are needed is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, why graphs are needed is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using why graphs are needed responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, why graphs are needed is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 1.4 what do-calculus can and cannot prove

What do-calculus can and cannot prove belongs to the canonical scope of Do Calculus. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is do-operator semantics, mutilated graphs, backdoor and frontdoor criteria, identification rules, and post-identification estimation. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$\operatorname{ATE}=\mathbb{E}[Y \mid \operatorname{do}(X=1)]-\mathbb{E}[Y \mid \operatorname{do}(X=0)].$$

The formula gives a compact handle on what do-calculus can and cannot prove. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of what do-calculus can and cannot prove:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for what do-calculus can and cannot prove is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, what do-calculus can and cannot prove is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using what do-calculus can and cannot prove responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, what do-calculus can and cannot prove is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 1.5 observational data plus assumptions

Observational data plus assumptions belongs to the canonical scope of Do Calculus. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is do-operator semantics, mutilated graphs, backdoor and frontdoor criteria, identification rules, and post-identification estimation. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P(Y \mid \operatorname{do}(X=x)) \ne P(Y \mid X=x) \quad \text{in general}.$$

The formula gives a compact handle on observational data plus assumptions. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of observational data plus assumptions:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for observational data plus assumptions is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, observational data plus assumptions is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using observational data plus assumptions responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, observational data plus assumptions is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

## 2. Formal Definitions

Formal Definitions develops the part of do calculus specified by the approved Chapter 22 table of contents. The treatment is causal, not merely predictive: the central objects are mechanisms, interventions, assumptions, and counterfactuals.

### 2.1 do-operator $\operatorname{do}(X=x)$

Do-operator $\operatorname{do}(x=x)$ belongs to the canonical scope of Do Calculus. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is do-operator semantics, mutilated graphs, backdoor and frontdoor criteria, identification rules, and post-identification estimation. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P(Y \mid \operatorname{do}(X=x))=\sum_z P(Y \mid X=x,Z=z)P(Z=z).$$

The formula gives a compact handle on do-operator $\operatorname{do}(x=x)$. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of do-operator $\operatorname{do}(x=x)$:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for do-operator $\operatorname{do}(x=x)$ is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, do-operator $\operatorname{do}(x=x)$ is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using do-operator $\operatorname{do}(x=x)$ responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, do-operator $\operatorname{do}(x=x)$ is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 2.2 mutilated graph $G_{\overline{X}}$

Mutilated graph $g_{\overline{x}}$ belongs to the canonical scope of Do Calculus. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is do-operator semantics, mutilated graphs, backdoor and frontdoor criteria, identification rules, and post-identification estimation. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P(Y \mid \operatorname{do}(X=x))=\sum_z P(Z=z \mid X=x)\sum_{x'}P(Y \mid X=x',Z=z)P(X=x').$$

The formula gives a compact handle on mutilated graph $g_{\overline{x}}$. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of mutilated graph $g_{\overline{x}}$:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for mutilated graph $g_{\overline{x}}$ is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, mutilated graph $g_{\overline{x}}$ is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using mutilated graph $g_{\overline{x}}$ responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, mutilated graph $g_{\overline{x}}$ is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 2.3 causal effect $P(Y \mid \operatorname{do}(X=x))$

Causal effect $p(y \mid \operatorname{do}(x=x))$ belongs to the canonical scope of Do Calculus. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is do-operator semantics, mutilated graphs, backdoor and frontdoor criteria, identification rules, and post-identification estimation. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$\operatorname{ATE}=\mathbb{E}[Y \mid \operatorname{do}(X=1)]-\mathbb{E}[Y \mid \operatorname{do}(X=0)].$$

The formula gives a compact handle on causal effect $p(y \mid \operatorname{do}(x=x))$. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of causal effect $p(y \mid \operatorname{do}(x=x))$:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for causal effect $p(y \mid \operatorname{do}(x=x))$ is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, causal effect $p(y \mid \operatorname{do}(x=x))$ is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using causal effect $p(y \mid \operatorname{do}(x=x))$ responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, causal effect $p(y \mid \operatorname{do}(x=x))$ is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 2.4 identifiability

Identifiability belongs to the canonical scope of Do Calculus. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is do-operator semantics, mutilated graphs, backdoor and frontdoor criteria, identification rules, and post-identification estimation. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P(Y \mid \operatorname{do}(X=x)) \ne P(Y \mid X=x) \quad \text{in general}.$$

The formula gives a compact handle on identifiability. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of identifiability:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for identifiability is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, identifiability is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using identifiability responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, identifiability is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 2.5 adjustment set

Adjustment set belongs to the canonical scope of Do Calculus. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is do-operator semantics, mutilated graphs, backdoor and frontdoor criteria, identification rules, and post-identification estimation. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P(Y \mid \operatorname{do}(X=x))=\sum_z P(Y \mid X=x,Z=z)P(Z=z).$$

The formula gives a compact handle on adjustment set. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of adjustment set:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for adjustment set is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, adjustment set is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using adjustment set responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, adjustment set is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

## 3. Backdoor and Frontdoor

Backdoor and Frontdoor develops the part of do calculus specified by the approved Chapter 22 table of contents. The treatment is causal, not merely predictive: the central objects are mechanisms, interventions, assumptions, and counterfactuals.

### 3.1 backdoor paths

Backdoor paths belongs to the canonical scope of Do Calculus. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is do-operator semantics, mutilated graphs, backdoor and frontdoor criteria, identification rules, and post-identification estimation. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P(Y \mid \operatorname{do}(X=x))=\sum_z P(Z=z \mid X=x)\sum_{x'}P(Y \mid X=x',Z=z)P(X=x').$$

The formula gives a compact handle on backdoor paths. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of backdoor paths:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for backdoor paths is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, backdoor paths is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using backdoor paths responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, backdoor paths is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 3.2 backdoor criterion

Backdoor criterion belongs to the canonical scope of Do Calculus. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is do-operator semantics, mutilated graphs, backdoor and frontdoor criteria, identification rules, and post-identification estimation. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$\operatorname{ATE}=\mathbb{E}[Y \mid \operatorname{do}(X=1)]-\mathbb{E}[Y \mid \operatorname{do}(X=0)].$$

The formula gives a compact handle on backdoor criterion. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of backdoor criterion:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for backdoor criterion is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, backdoor criterion is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using backdoor criterion responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, backdoor criterion is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 3.3 adjustment formula

Adjustment formula belongs to the canonical scope of Do Calculus. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is do-operator semantics, mutilated graphs, backdoor and frontdoor criteria, identification rules, and post-identification estimation. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P(Y \mid \operatorname{do}(X=x)) \ne P(Y \mid X=x) \quad \text{in general}.$$

The formula gives a compact handle on adjustment formula. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of adjustment formula:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for adjustment formula is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, adjustment formula is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using adjustment formula responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, adjustment formula is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 3.4 frontdoor criterion

Frontdoor criterion belongs to the canonical scope of Do Calculus. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is do-operator semantics, mutilated graphs, backdoor and frontdoor criteria, identification rules, and post-identification estimation. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P(Y \mid \operatorname{do}(X=x))=\sum_z P(Y \mid X=x,Z=z)P(Z=z).$$

The formula gives a compact handle on frontdoor criterion. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of frontdoor criterion:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for frontdoor criterion is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, frontdoor criterion is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using frontdoor criterion responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, frontdoor criterion is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 3.5 confounding and mediators

Confounding and mediators belongs to the canonical scope of Do Calculus. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is do-operator semantics, mutilated graphs, backdoor and frontdoor criteria, identification rules, and post-identification estimation. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P(Y \mid \operatorname{do}(X=x))=\sum_z P(Z=z \mid X=x)\sum_{x'}P(Y \mid X=x',Z=z)P(X=x').$$

The formula gives a compact handle on confounding and mediators. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of confounding and mediators:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for confounding and mediators is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, confounding and mediators is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using confounding and mediators responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, confounding and mediators is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

## 4. The Three Rules

The Three Rules develops the part of do calculus specified by the approved Chapter 22 table of contents. The treatment is causal, not merely predictive: the central objects are mechanisms, interventions, assumptions, and counterfactuals.

### 4.1 insertion and deletion of observations

Insertion and deletion of observations belongs to the canonical scope of Do Calculus. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is do-operator semantics, mutilated graphs, backdoor and frontdoor criteria, identification rules, and post-identification estimation. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$\operatorname{ATE}=\mathbb{E}[Y \mid \operatorname{do}(X=1)]-\mathbb{E}[Y \mid \operatorname{do}(X=0)].$$

The formula gives a compact handle on insertion and deletion of observations. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of insertion and deletion of observations:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for insertion and deletion of observations is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, insertion and deletion of observations is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using insertion and deletion of observations responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, insertion and deletion of observations is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 4.2 action observation exchange

Action observation exchange belongs to the canonical scope of Do Calculus. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is do-operator semantics, mutilated graphs, backdoor and frontdoor criteria, identification rules, and post-identification estimation. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P(Y \mid \operatorname{do}(X=x)) \ne P(Y \mid X=x) \quad \text{in general}.$$

The formula gives a compact handle on action observation exchange. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of action observation exchange:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for action observation exchange is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, action observation exchange is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using action observation exchange responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, action observation exchange is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 4.3 insertion and deletion of actions

Insertion and deletion of actions belongs to the canonical scope of Do Calculus. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is do-operator semantics, mutilated graphs, backdoor and frontdoor criteria, identification rules, and post-identification estimation. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P(Y \mid \operatorname{do}(X=x))=\sum_z P(Y \mid X=x,Z=z)P(Z=z).$$

The formula gives a compact handle on insertion and deletion of actions. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of insertion and deletion of actions:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for insertion and deletion of actions is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, insertion and deletion of actions is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using insertion and deletion of actions responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, insertion and deletion of actions is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 4.4 graph conditions for each rule

Graph conditions for each rule belongs to the canonical scope of Do Calculus. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is do-operator semantics, mutilated graphs, backdoor and frontdoor criteria, identification rules, and post-identification estimation. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P(Y \mid \operatorname{do}(X=x))=\sum_z P(Z=z \mid X=x)\sum_{x'}P(Y \mid X=x',Z=z)P(X=x').$$

The formula gives a compact handle on graph conditions for each rule. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of graph conditions for each rule:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for graph conditions for each rule is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, graph conditions for each rule is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using graph conditions for each rule responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, graph conditions for each rule is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 4.5 proof intuition from d-separation

Proof intuition from d-separation belongs to the canonical scope of Do Calculus. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is do-operator semantics, mutilated graphs, backdoor and frontdoor criteria, identification rules, and post-identification estimation. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$\operatorname{ATE}=\mathbb{E}[Y \mid \operatorname{do}(X=1)]-\mathbb{E}[Y \mid \operatorname{do}(X=0)].$$

The formula gives a compact handle on proof intuition from d-separation. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of proof intuition from d-separation:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for proof intuition from d-separation is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, proof intuition from d-separation is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using proof intuition from d-separation responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, proof intuition from d-separation is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

## 5. Identification Workflows

Identification Workflows develops the part of do calculus specified by the approved Chapter 22 table of contents. The treatment is causal, not merely predictive: the central objects are mechanisms, interventions, assumptions, and counterfactuals.

### 5.1 manual graph simplification

Manual graph simplification belongs to the canonical scope of Do Calculus. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is do-operator semantics, mutilated graphs, backdoor and frontdoor criteria, identification rules, and post-identification estimation. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P(Y \mid \operatorname{do}(X=x)) \ne P(Y \mid X=x) \quad \text{in general}.$$

The formula gives a compact handle on manual graph simplification. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of manual graph simplification:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for manual graph simplification is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, manual graph simplification is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using manual graph simplification responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, manual graph simplification is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 5.2 ID algorithm preview

Id algorithm preview belongs to the canonical scope of Do Calculus. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is do-operator semantics, mutilated graphs, backdoor and frontdoor criteria, identification rules, and post-identification estimation. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P(Y \mid \operatorname{do}(X=x))=\sum_z P(Y \mid X=x,Z=z)P(Z=z).$$

The formula gives a compact handle on id algorithm preview. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of id algorithm preview:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for id algorithm preview is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, id algorithm preview is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using id algorithm preview responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, id algorithm preview is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 5.3 c-components and hedges

C-components and hedges belongs to the canonical scope of Do Calculus. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is do-operator semantics, mutilated graphs, backdoor and frontdoor criteria, identification rules, and post-identification estimation. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P(Y \mid \operatorname{do}(X=x))=\sum_z P(Z=z \mid X=x)\sum_{x'}P(Y \mid X=x',Z=z)P(X=x').$$

The formula gives a compact handle on c-components and hedges. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of c-components and hedges:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for c-components and hedges is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, c-components and hedges is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using c-components and hedges responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, c-components and hedges is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 5.4 Tian-Pearl identification condition preview

Tian-pearl identification condition preview belongs to the canonical scope of Do Calculus. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is do-operator semantics, mutilated graphs, backdoor and frontdoor criteria, identification rules, and post-identification estimation. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$\operatorname{ATE}=\mathbb{E}[Y \mid \operatorname{do}(X=1)]-\mathbb{E}[Y \mid \operatorname{do}(X=0)].$$

The formula gives a compact handle on tian-pearl identification condition preview. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of tian-pearl identification condition preview:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for tian-pearl identification condition preview is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, tian-pearl identification condition preview is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using tian-pearl identification condition preview responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, tian-pearl identification condition preview is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 5.5 when an effect is not identifiable

When an effect is not identifiable belongs to the canonical scope of Do Calculus. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is do-operator semantics, mutilated graphs, backdoor and frontdoor criteria, identification rules, and post-identification estimation. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P(Y \mid \operatorname{do}(X=x)) \ne P(Y \mid X=x) \quad \text{in general}.$$

The formula gives a compact handle on when an effect is not identifiable. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of when an effect is not identifiable:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for when an effect is not identifiable is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, when an effect is not identifiable is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using when an effect is not identifiable responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, when an effect is not identifiable is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

## 6. Estimation After Identification

Estimation After Identification develops the part of do calculus specified by the approved Chapter 22 table of contents. The treatment is causal, not merely predictive: the central objects are mechanisms, interventions, assumptions, and counterfactuals.

### 6.1 plug-in adjustment

Plug-in adjustment belongs to the canonical scope of Do Calculus. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is do-operator semantics, mutilated graphs, backdoor and frontdoor criteria, identification rules, and post-identification estimation. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P(Y \mid \operatorname{do}(X=x))=\sum_z P(Y \mid X=x,Z=z)P(Z=z).$$

The formula gives a compact handle on plug-in adjustment. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of plug-in adjustment:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for plug-in adjustment is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, plug-in adjustment is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using plug-in adjustment responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, plug-in adjustment is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 6.2 inverse-propensity weighting preview

Inverse-propensity weighting preview belongs to the canonical scope of Do Calculus. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is do-operator semantics, mutilated graphs, backdoor and frontdoor criteria, identification rules, and post-identification estimation. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P(Y \mid \operatorname{do}(X=x))=\sum_z P(Z=z \mid X=x)\sum_{x'}P(Y \mid X=x',Z=z)P(X=x').$$

The formula gives a compact handle on inverse-propensity weighting preview. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of inverse-propensity weighting preview:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for inverse-propensity weighting preview is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, inverse-propensity weighting preview is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using inverse-propensity weighting preview responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, inverse-propensity weighting preview is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 6.3 doubly robust preview

Doubly robust preview belongs to the canonical scope of Do Calculus. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is do-operator semantics, mutilated graphs, backdoor and frontdoor criteria, identification rules, and post-identification estimation. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$\operatorname{ATE}=\mathbb{E}[Y \mid \operatorname{do}(X=1)]-\mathbb{E}[Y \mid \operatorname{do}(X=0)].$$

The formula gives a compact handle on doubly robust preview. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of doubly robust preview:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for doubly robust preview is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, doubly robust preview is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using doubly robust preview responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, doubly robust preview is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 6.4 positivity and overlap

Positivity and overlap belongs to the canonical scope of Do Calculus. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is do-operator semantics, mutilated graphs, backdoor and frontdoor criteria, identification rules, and post-identification estimation. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P(Y \mid \operatorname{do}(X=x)) \ne P(Y \mid X=x) \quad \text{in general}.$$

The formula gives a compact handle on positivity and overlap. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of positivity and overlap:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for positivity and overlap is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, positivity and overlap is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using positivity and overlap responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, positivity and overlap is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 6.5 nuisance ML and cross-fitting preview

Nuisance ml and cross-fitting preview belongs to the canonical scope of Do Calculus. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is do-operator semantics, mutilated graphs, backdoor and frontdoor criteria, identification rules, and post-identification estimation. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P(Y \mid \operatorname{do}(X=x))=\sum_z P(Y \mid X=x,Z=z)P(Z=z).$$

The formula gives a compact handle on nuisance ml and cross-fitting preview. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of nuisance ml and cross-fitting preview:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for nuisance ml and cross-fitting preview is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, nuisance ml and cross-fitting preview is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using nuisance ml and cross-fitting preview responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, nuisance ml and cross-fitting preview is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

## 7. ML and LLM Applications

ML and LLM Applications develops the part of do calculus specified by the approved Chapter 22 table of contents. The treatment is causal, not merely predictive: the central objects are mechanisms, interventions, assumptions, and counterfactuals.

### 7.1 causal effect of ranking policies

Causal effect of ranking policies belongs to the canonical scope of Do Calculus. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is do-operator semantics, mutilated graphs, backdoor and frontdoor criteria, identification rules, and post-identification estimation. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P(Y \mid \operatorname{do}(X=x))=\sum_z P(Z=z \mid X=x)\sum_{x'}P(Y \mid X=x',Z=z)P(X=x').$$

The formula gives a compact handle on causal effect of ranking policies. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of causal effect of ranking policies:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for causal effect of ranking policies is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, causal effect of ranking policies is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using causal effect of ranking policies responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, causal effect of ranking policies is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 7.2 logged bandit interventions

Logged bandit interventions belongs to the canonical scope of Do Calculus. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is do-operator semantics, mutilated graphs, backdoor and frontdoor criteria, identification rules, and post-identification estimation. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$\operatorname{ATE}=\mathbb{E}[Y \mid \operatorname{do}(X=1)]-\mathbb{E}[Y \mid \operatorname{do}(X=0)].$$

The formula gives a compact handle on logged bandit interventions. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of logged bandit interventions:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for logged bandit interventions is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, logged bandit interventions is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using logged bandit interventions responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, logged bandit interventions is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 7.3 prompt and tool interventions

Prompt and tool interventions belongs to the canonical scope of Do Calculus. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is do-operator semantics, mutilated graphs, backdoor and frontdoor criteria, identification rules, and post-identification estimation. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P(Y \mid \operatorname{do}(X=x)) \ne P(Y \mid X=x) \quad \text{in general}.$$

The formula gives a compact handle on prompt and tool interventions. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of prompt and tool interventions:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for prompt and tool interventions is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, prompt and tool interventions is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using prompt and tool interventions responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, prompt and tool interventions is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 7.4 safety policy changes

Safety policy changes belongs to the canonical scope of Do Calculus. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is do-operator semantics, mutilated graphs, backdoor and frontdoor criteria, identification rules, and post-identification estimation. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P(Y \mid \operatorname{do}(X=x))=\sum_z P(Y \mid X=x,Z=z)P(Z=z).$$

The formula gives a compact handle on safety policy changes. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of safety policy changes:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for safety policy changes is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, safety policy changes is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using safety policy changes responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, safety policy changes is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

### 7.5 causal debugging of model behavior

Causal debugging of model behavior belongs to the canonical scope of Do Calculus. The central move in causal inference is to distinguish a statistical relation from a claim about what would happen under an intervention.

For this subsection, the working scope is do-operator semantics, mutilated graphs, backdoor and frontdoor criteria, identification rules, and post-identification estimation. The mathematical objects are variables, mechanisms, graphs, interventions, and assumptions. A causal claim is incomplete until all five are visible.

$$P(Y \mid \operatorname{do}(X=x))=\sum_z P(Z=z \mid X=x)\sum_{x'}P(Y \mid X=x',Z=z)P(X=x').$$

The formula gives a compact handle on causal debugging of model behavior. It should not be read as a purely algebraic identity. In causal inference, equations encode assumptions about mechanisms, missing variables, and which parts of the world remain stable under intervention.

| Causal object | Meaning | AI interpretation |
| --- | --- | --- |
| Variable | Quantity in the causal system | Prompt feature, user action, treatment, tool call, exposure, label, reward |
| Mechanism | Assignment that generates a variable | Data pipeline, recommender policy, human behavior, model routing rule |
| Graph | Qualitative causal assumptions | What can affect what, and which paths may confound effects |
| Intervention | Replacement of a mechanism | A/B rollout, policy switch, prompt template change, retrieval update |
| Counterfactual | Unit-level alternate world | What this user or model trace would have done under another action |

Three examples of causal debugging of model behavior:

1. A recommender team wants the causal effect of ranking a document higher, not merely the correlation between rank and clicks.
2. An LLM platform changes a safety policy and wants to estimate whether refusals changed because of the policy or because user prompts shifted.
3. A fairness auditor asks whether a proxy feature transmits an impermissible causal path into a model decision.

Two non-examples expose the boundary:

1. A high predictive coefficient is not a causal effect unless the graph and intervention assumptions justify it.
2. A plausible narrative produced by a language model is not a counterfactual unless it is grounded in a causal model.

The proof habit for causal debugging of model behavior is to name the graph operation. Conditioning restricts a distribution. Intervention replaces a mechanism. Counterfactual reasoning updates exogenous uncertainty from evidence, changes a mechanism, then predicts.

```text
observed association:      P(Y | X=x)
intervention question:     P(Y | do(X=x))
counterfactual question:   P(Y_x | E=e)
discovery question:        which G could have generated P(V)?
```

In machine learning, causal debugging of model behavior is valuable because models are often deployed under interventions: ranking changes, policy changes, safety filters, tool-use gates, data collection changes, and human feedback loops. Prediction alone does not tell us which change caused which downstream behavior.

Notebook implementation will use synthetic SCMs and small graphs. This keeps the examples executable while preserving the conceptual split between identification and estimation.

Checklist for using causal debugging of model behavior responsibly:

- State the causal question before choosing a method.
- Draw or describe the assumed causal graph.
- Mark observed, latent, treatment, outcome, and adjustment variables.
- Separate intervention notation from conditioning notation.
- Decide whether the query is identifiable before estimating it.
- Report assumptions that cannot be tested from the observed data alone.
- Use ML as an estimation aid, not as a substitute for causal design.

This chapter follows the boundary set by Chapter 21. Statistical learning theory controls prediction error under distributional assumptions. Causal inference asks what happens when the distribution changes because something is done.

Modern AI systems make this distinction unavoidable. A foundation model can predict which action historically followed a context, but a decision system needs to know what would happen if it took a different action in that context.

Thus, causal debugging of model behavior is not an abstract philosophical add-on. It is a production and research tool for deciding which model, prompt, policy, feature, or intervention actually changed an outcome.

A final diagnostic question is whether the claim would survive a policy change. If the answer depends only on a historical correlation, it belongs in predictive modeling. If the answer depends on what mechanism is replaced and which paths remain active, it belongs in causal inference.

| Diagnostic question | Causal discipline it tests |
| --- | --- |
| What is being changed? | Intervention target |
| Which mechanism is replaced? | SCM modularity |
| Which paths transmit the effect? | Graph semantics |
| Which variables are merely observed? | Conditioning versus intervention |
| Which quantities are unobserved? | Confounding and counterfactual uncertainty |

## 8. Common Mistakes

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

## 9. Exercises

1. (*) Work through a causal-inference task for do calculus.
   - (a) State the causal query using intervention or counterfactual notation.
   - (b) Draw or describe the relevant graph and assumptions.
   - (c) Decide whether the estimand is identifiable from the available data.
   - (d) Give an estimator or diagnostic only after identification is clear.
   - (e) Explain the AI or LLM system implication.

2. (*) Work through a causal-inference task for do calculus.
   - (a) State the causal query using intervention or counterfactual notation.
   - (b) Draw or describe the relevant graph and assumptions.
   - (c) Decide whether the estimand is identifiable from the available data.
   - (d) Give an estimator or diagnostic only after identification is clear.
   - (e) Explain the AI or LLM system implication.

3. (*) Work through a causal-inference task for do calculus.
   - (a) State the causal query using intervention or counterfactual notation.
   - (b) Draw or describe the relevant graph and assumptions.
   - (c) Decide whether the estimand is identifiable from the available data.
   - (d) Give an estimator or diagnostic only after identification is clear.
   - (e) Explain the AI or LLM system implication.

4. (**) Work through a causal-inference task for do calculus.
   - (a) State the causal query using intervention or counterfactual notation.
   - (b) Draw or describe the relevant graph and assumptions.
   - (c) Decide whether the estimand is identifiable from the available data.
   - (d) Give an estimator or diagnostic only after identification is clear.
   - (e) Explain the AI or LLM system implication.

5. (**) Work through a causal-inference task for do calculus.
   - (a) State the causal query using intervention or counterfactual notation.
   - (b) Draw or describe the relevant graph and assumptions.
   - (c) Decide whether the estimand is identifiable from the available data.
   - (d) Give an estimator or diagnostic only after identification is clear.
   - (e) Explain the AI or LLM system implication.

6. (**) Work through a causal-inference task for do calculus.
   - (a) State the causal query using intervention or counterfactual notation.
   - (b) Draw or describe the relevant graph and assumptions.
   - (c) Decide whether the estimand is identifiable from the available data.
   - (d) Give an estimator or diagnostic only after identification is clear.
   - (e) Explain the AI or LLM system implication.

7. (***) Work through a causal-inference task for do calculus.
   - (a) State the causal query using intervention or counterfactual notation.
   - (b) Draw or describe the relevant graph and assumptions.
   - (c) Decide whether the estimand is identifiable from the available data.
   - (d) Give an estimator or diagnostic only after identification is clear.
   - (e) Explain the AI or LLM system implication.

8. (***) Work through a causal-inference task for do calculus.
   - (a) State the causal query using intervention or counterfactual notation.
   - (b) Draw or describe the relevant graph and assumptions.
   - (c) Decide whether the estimand is identifiable from the available data.
   - (d) Give an estimator or diagnostic only after identification is clear.
   - (e) Explain the AI or LLM system implication.

9. (***) Work through a causal-inference task for do calculus.
   - (a) State the causal query using intervention or counterfactual notation.
   - (b) Draw or describe the relevant graph and assumptions.
   - (c) Decide whether the estimand is identifiable from the available data.
   - (d) Give an estimator or diagnostic only after identification is clear.
   - (e) Explain the AI or LLM system implication.

10. (***) Work through a causal-inference task for do calculus.
   - (a) State the causal query using intervention or counterfactual notation.
   - (b) Draw or describe the relevant graph and assumptions.
   - (c) Decide whether the estimand is identifiable from the available data.
   - (d) Give an estimator or diagnostic only after identification is clear.
   - (e) Explain the AI or LLM system implication.

## 10. Why This Matters for AI

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

## 11. Conceptual Bridge

Do Calculus follows statistical learning theory because learning theory explains how observed samples support future prediction claims. Causal inference asks a different question: what happens when an action changes the system that generated those samples?

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

- Pearl. Causal diagrams for empirical research. https://academic.oup.com/biomet/article-abstract/82/4/669/251647
- Pearl. Causal inference in statistics: An overview. https://projecteuclid.org/journals/statistics-surveys/volume-3/issue-none/Causal-inference-in-statistics-An-overview/10.1214/09-SS057.pdf
- Tian and Pearl. A General Identification Condition for Causal Effects. https://8www.aaai.org/Library/AAAI/2002/aaai02-085.php
- Hernan and Robins. Causal Inference: What If. https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/
