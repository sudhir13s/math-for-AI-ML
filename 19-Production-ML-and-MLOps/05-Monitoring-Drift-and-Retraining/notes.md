[Back to Curriculum](../../README.md) | [Previous: Model Serving and Inference Optimization](../04-Model-Serving-and-Inference-Optimization/notes.md) | [Next: LLM Evaluation Observability and Guardrails](../06-LLM-Evaluation-Observability-and-Guardrails/notes.md)

---

# Monitoring Drift and Retraining

> _"Monitoring is the production derivative of evaluation: it measures how fast reality is moving away."_

## Overview

Drift monitoring and retraining convert production telemetry into controlled model maintenance instead of reactive firefighting.

Production ML and MLOps are the mathematical discipline of keeping a learned system useful after it leaves the notebook. The model is only one artifact in a larger graph of data, code, configuration, evaluation, deployment, monitoring, and response actions.

This chapter uses LaTeX Markdown throughout. Inline mathematics uses `$...$`, and display equations use `$$...$$`. The central habit is to turn production behavior into explicit objects: versions, hashes, traces, thresholds, queues, contracts, and release decisions.

## Prerequisites

- [Robustness and Distribution Shift](../../17-Evaluation-and-Reliability/03-Robustness-and-Distribution-Shift/notes.md)
- [Calibration and Uncertainty](../../17-Evaluation-and-Reliability/02-Calibration-and-Uncertainty/notes.md)
- [Online Experimentation and AB Testing](../../17-Evaluation-and-Reliability/05-Online-Experimentation-and-AB-Testing/notes.md)
- [Model Serving and Inference Optimization](../04-Model-Serving-and-Inference-Optimization/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Executable demonstrations for monitoring drift and retraining |
| [exercises.ipynb](exercises.ipynb) | Graded practice for monitoring drift and retraining |

## Learning Objectives

After completing this section, you will be able to:

- Define production ML artifacts using mathematical notation
- Represent dependencies as auditable graphs and contracts
- Compute simple production statistics with synthetic data
- Separate offline evaluation from online monitoring
- Design release gates that combine quality, safety, latency, and cost
- Explain how versioning enables rollback and reproducibility
- Diagnose drift, skew, and production regressions
- Connect LLM traces to evaluations, guardrails, and retraining data
- Identify operational failure modes before they become incidents
- Build lightweight notebook simulations of production ML behavior

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 models decay because the world changes](#11-models-decay-because-the-world-changes)
  - [1.2 monitoring versus evaluation boundary](#12-monitoring-versus-evaluation-boundary)
  - [1.3 data model and business signals](#13-data-model-and-business-signals)
  - [1.4 alert fatigue](#14-alert-fatigue)
  - [1.5 retraining as a control loop](#15-retraining-as-a-control-loop)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 reference distribution $p_{\mathrm{ref}}$](#21-reference-distribution)
  - [2.2 current distribution $p_t$](#22-current-distribution)
  - [2.3 drift statistic](#23-drift-statistic)
  - [2.4 alert threshold $\tau$](#24-alert-threshold)
  - [2.5 retraining policy](#25-retraining-policy)
- [3. Monitoring Signals](#3-monitoring-signals)
  - [3.1 input drift](#31-input-drift)
  - [3.2 prediction drift](#32-prediction-drift)
  - [3.3 label and performance drift](#33-label-and-performance-drift)
  - [3.4 latency and cost](#34-latency-and-cost)
  - [3.5 data quality](#35-data-quality)
- [4. Drift Detection Math](#4-drift-detection-math)
  - [4.1 population stability index](#41-population-stability-index)
  - [4.2 KL and JS divergence](#42-kl-and-js-divergence)
  - [4.3 Wasserstein distance](#43-wasserstein-distance)
  - [4.4 two-sample tests](#44-twosample-tests)
  - [4.5 embedding drift](#45-embedding-drift)
- [5. Alerting and Diagnosis](#5-alerting-and-diagnosis)
  - [5.1 threshold tuning](#51-threshold-tuning)
  - [5.2 severity levels](#52-severity-levels)
  - [5.3 slice analysis](#53-slice-analysis)
  - [5.4 false positives](#54-false-positives)
  - [5.5 root-cause workflow](#55-rootcause-workflow)
- [6. Retraining Systems](#6-retraining-systems)
  - [6.1 scheduled retraining](#61-scheduled-retraining)
  - [6.2 trigger-based retraining](#62-triggerbased-retraining)
  - [6.3 champion challenger](#63-champion-challenger)
  - [6.4 rollback-safe retraining](#64-rollbacksafe-retraining)
  - [6.5 data freshness gates](#65-data-freshness-gates)
- [7. LLM Production Drift](#7-llm-production-drift)
  - [7.1 prompt distribution drift](#71-prompt-distribution-drift)
  - [7.2 retrieval corpus drift](#72-retrieval-corpus-drift)
  - [7.3 judge drift](#73-judge-drift)
  - [7.4 cost and latency drift](#74-cost-and-latency-drift)
  - [7.5 behavior regression](#75-behavior-regression)
- [8. Common Mistakes](#8-common-mistakes)
- [9. Exercises](#9-exercises)
- [10. Why This Matters for AI](#10-why-this-matters-for-ai)
- [11. Conceptual Bridge](#11-conceptual-bridge)
- [References](#references)

---

## 1. Intuition

Intuition develops the part of monitoring drift and retraining assigned by the approved Chapter 19 table of contents. The treatment is production-focused: every idea is connected to a versioned artifact, measurable signal, release decision, or incident response.

### 1.1 models decay because the world changes

Models decay because the world changes is part of the canonical scope of Monitoring Drift and Retraining. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is production monitoring signals, drift statistics, alerting, diagnosis, retraining policies, and LLM production drift. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$S_t = d(p_t, p_{\mathrm{ref}}).$$

The formula is intentionally simple. It says that models decay because the world changes should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of models decay because the world changes in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Models decay because the world changes is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for models decay because the world changes:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of models decay because the world changes executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for models decay because the world changes should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 1.2 monitoring versus evaluation boundary

Monitoring versus evaluation boundary is part of the canonical scope of Monitoring Drift and Retraining. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is production monitoring signals, drift statistics, alerting, diagnosis, retraining policies, and LLM production drift. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\operatorname{PSI} = \sum_{b=1}^{B}(q_b-p_b)\log\frac{q_b}{p_b}.$$

The formula is intentionally simple. It says that monitoring versus evaluation boundary should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of monitoring versus evaluation boundary in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Monitoring versus evaluation boundary is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for monitoring versus evaluation boundary:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of monitoring versus evaluation boundary executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for monitoring versus evaluation boundary should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 1.3 data model and business signals

Data model and business signals is part of the canonical scope of Monitoring Drift and Retraining. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is production monitoring signals, drift statistics, alerting, diagnosis, retraining policies, and LLM production drift. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$D_{\mathrm{JS}}(p\Vert q)=\frac{1}{2}D_{\mathrm{KL}}(p\Vert m)+\frac{1}{2}D_{\mathrm{KL}}(q\Vert m), \quad m=\frac{p+q}{2}.$$

The formula is intentionally simple. It says that data model and business signals should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of data model and business signals in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Data model and business signals is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for data model and business signals:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of data model and business signals executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for data model and business signals should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 1.4 alert fatigue

Alert fatigue is part of the canonical scope of Monitoring Drift and Retraining. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is production monitoring signals, drift statistics, alerting, diagnosis, retraining policies, and LLM production drift. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\pi_{\mathrm{retrain}}(t)=\mathbb{1}[S_t>\tau]\mathbb{1}[Q_t=1].$$

The formula is intentionally simple. It says that alert fatigue should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of alert fatigue in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Alert fatigue is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for alert fatigue:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of alert fatigue executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for alert fatigue should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 1.5 retraining as a control loop

Retraining as a control loop is part of the canonical scope of Monitoring Drift and Retraining. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is production monitoring signals, drift statistics, alerting, diagnosis, retraining policies, and LLM production drift. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$S_t = d(p_t, p_{\mathrm{ref}}).$$

The formula is intentionally simple. It says that retraining as a control loop should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of retraining as a control loop in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Retraining as a control loop is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for retraining as a control loop:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of retraining as a control loop executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for retraining as a control loop should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

## 2. Formal Definitions

Formal Definitions develops the part of monitoring drift and retraining assigned by the approved Chapter 19 table of contents. The treatment is production-focused: every idea is connected to a versioned artifact, measurable signal, release decision, or incident response.

### 2.1 reference distribution $p_{\mathrm{ref}}$

Reference distribution $p_{\mathrm{ref}}$ is part of the canonical scope of Monitoring Drift and Retraining. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is production monitoring signals, drift statistics, alerting, diagnosis, retraining policies, and LLM production drift. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\operatorname{PSI} = \sum_{b=1}^{B}(q_b-p_b)\log\frac{q_b}{p_b}.$$

The formula is intentionally simple. It says that reference distribution $p_{\mathrm{ref}}$ should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of reference distribution $p_{\mathrm{ref}}$ in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Reference distribution $p_{\mathrm{ref}}$ is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for reference distribution $p_{\mathrm{ref}}$:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of reference distribution $p_{\mathrm{ref}}$ executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for reference distribution $p_{\mathrm{ref}}$ should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 2.2 current distribution $p_t$

Current distribution $p_t$ is part of the canonical scope of Monitoring Drift and Retraining. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is production monitoring signals, drift statistics, alerting, diagnosis, retraining policies, and LLM production drift. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$D_{\mathrm{JS}}(p\Vert q)=\frac{1}{2}D_{\mathrm{KL}}(p\Vert m)+\frac{1}{2}D_{\mathrm{KL}}(q\Vert m), \quad m=\frac{p+q}{2}.$$

The formula is intentionally simple. It says that current distribution $p_t$ should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of current distribution $p_t$ in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Current distribution $p_t$ is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for current distribution $p_t$:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of current distribution $p_t$ executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for current distribution $p_t$ should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 2.3 drift statistic

Drift statistic is part of the canonical scope of Monitoring Drift and Retraining. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is production monitoring signals, drift statistics, alerting, diagnosis, retraining policies, and LLM production drift. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\pi_{\mathrm{retrain}}(t)=\mathbb{1}[S_t>\tau]\mathbb{1}[Q_t=1].$$

The formula is intentionally simple. It says that drift statistic should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of drift statistic in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Drift statistic is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for drift statistic:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of drift statistic executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for drift statistic should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 2.4 alert threshold $\tau$

Alert threshold $\tau$ is part of the canonical scope of Monitoring Drift and Retraining. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is production monitoring signals, drift statistics, alerting, diagnosis, retraining policies, and LLM production drift. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$S_t = d(p_t, p_{\mathrm{ref}}).$$

The formula is intentionally simple. It says that alert threshold $\tau$ should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of alert threshold $\tau$ in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Alert threshold $\tau$ is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for alert threshold $\tau$:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of alert threshold $\tau$ executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for alert threshold $\tau$ should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 2.5 retraining policy

Retraining policy is part of the canonical scope of Monitoring Drift and Retraining. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is production monitoring signals, drift statistics, alerting, diagnosis, retraining policies, and LLM production drift. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\operatorname{PSI} = \sum_{b=1}^{B}(q_b-p_b)\log\frac{q_b}{p_b}.$$

The formula is intentionally simple. It says that retraining policy should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of retraining policy in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Retraining policy is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for retraining policy:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of retraining policy executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for retraining policy should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

## 3. Monitoring Signals

Monitoring Signals develops the part of monitoring drift and retraining assigned by the approved Chapter 19 table of contents. The treatment is production-focused: every idea is connected to a versioned artifact, measurable signal, release decision, or incident response.

### 3.1 input drift

Input drift is part of the canonical scope of Monitoring Drift and Retraining. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is production monitoring signals, drift statistics, alerting, diagnosis, retraining policies, and LLM production drift. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$D_{\mathrm{JS}}(p\Vert q)=\frac{1}{2}D_{\mathrm{KL}}(p\Vert m)+\frac{1}{2}D_{\mathrm{KL}}(q\Vert m), \quad m=\frac{p+q}{2}.$$

The formula is intentionally simple. It says that input drift should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of input drift in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Input drift is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for input drift:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of input drift executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for input drift should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 3.2 prediction drift

Prediction drift is part of the canonical scope of Monitoring Drift and Retraining. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is production monitoring signals, drift statistics, alerting, diagnosis, retraining policies, and LLM production drift. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\pi_{\mathrm{retrain}}(t)=\mathbb{1}[S_t>\tau]\mathbb{1}[Q_t=1].$$

The formula is intentionally simple. It says that prediction drift should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of prediction drift in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Prediction drift is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for prediction drift:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of prediction drift executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for prediction drift should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 3.3 label and performance drift

Label and performance drift is part of the canonical scope of Monitoring Drift and Retraining. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is production monitoring signals, drift statistics, alerting, diagnosis, retraining policies, and LLM production drift. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$S_t = d(p_t, p_{\mathrm{ref}}).$$

The formula is intentionally simple. It says that label and performance drift should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of label and performance drift in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Label and performance drift is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for label and performance drift:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of label and performance drift executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for label and performance drift should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 3.4 latency and cost

Latency and cost is part of the canonical scope of Monitoring Drift and Retraining. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is production monitoring signals, drift statistics, alerting, diagnosis, retraining policies, and LLM production drift. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\operatorname{PSI} = \sum_{b=1}^{B}(q_b-p_b)\log\frac{q_b}{p_b}.$$

The formula is intentionally simple. It says that latency and cost should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of latency and cost in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Latency and cost is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for latency and cost:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of latency and cost executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for latency and cost should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 3.5 data quality

Data quality is part of the canonical scope of Monitoring Drift and Retraining. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is production monitoring signals, drift statistics, alerting, diagnosis, retraining policies, and LLM production drift. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$D_{\mathrm{JS}}(p\Vert q)=\frac{1}{2}D_{\mathrm{KL}}(p\Vert m)+\frac{1}{2}D_{\mathrm{KL}}(q\Vert m), \quad m=\frac{p+q}{2}.$$

The formula is intentionally simple. It says that data quality should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of data quality in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Data quality is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for data quality:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of data quality executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for data quality should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

## 4. Drift Detection Math

Drift Detection Math develops the part of monitoring drift and retraining assigned by the approved Chapter 19 table of contents. The treatment is production-focused: every idea is connected to a versioned artifact, measurable signal, release decision, or incident response.

### 4.1 population stability index

Population stability index is part of the canonical scope of Monitoring Drift and Retraining. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is production monitoring signals, drift statistics, alerting, diagnosis, retraining policies, and LLM production drift. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\pi_{\mathrm{retrain}}(t)=\mathbb{1}[S_t>\tau]\mathbb{1}[Q_t=1].$$

The formula is intentionally simple. It says that population stability index should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of population stability index in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Population stability index is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for population stability index:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of population stability index executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for population stability index should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 4.2 KL and JS divergence

Kl and js divergence is part of the canonical scope of Monitoring Drift and Retraining. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is production monitoring signals, drift statistics, alerting, diagnosis, retraining policies, and LLM production drift. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$S_t = d(p_t, p_{\mathrm{ref}}).$$

The formula is intentionally simple. It says that kl and js divergence should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of kl and js divergence in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Kl and js divergence is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for kl and js divergence:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of kl and js divergence executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for kl and js divergence should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 4.3 Wasserstein distance

Wasserstein distance is part of the canonical scope of Monitoring Drift and Retraining. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is production monitoring signals, drift statistics, alerting, diagnosis, retraining policies, and LLM production drift. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\operatorname{PSI} = \sum_{b=1}^{B}(q_b-p_b)\log\frac{q_b}{p_b}.$$

The formula is intentionally simple. It says that wasserstein distance should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of wasserstein distance in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Wasserstein distance is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for wasserstein distance:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of wasserstein distance executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for wasserstein distance should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 4.4 two-sample tests

Two-sample tests is part of the canonical scope of Monitoring Drift and Retraining. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is production monitoring signals, drift statistics, alerting, diagnosis, retraining policies, and LLM production drift. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$D_{\mathrm{JS}}(p\Vert q)=\frac{1}{2}D_{\mathrm{KL}}(p\Vert m)+\frac{1}{2}D_{\mathrm{KL}}(q\Vert m), \quad m=\frac{p+q}{2}.$$

The formula is intentionally simple. It says that two-sample tests should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of two-sample tests in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Two-sample tests is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for two-sample tests:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of two-sample tests executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for two-sample tests should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 4.5 embedding drift

Embedding drift is part of the canonical scope of Monitoring Drift and Retraining. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is production monitoring signals, drift statistics, alerting, diagnosis, retraining policies, and LLM production drift. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\pi_{\mathrm{retrain}}(t)=\mathbb{1}[S_t>\tau]\mathbb{1}[Q_t=1].$$

The formula is intentionally simple. It says that embedding drift should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of embedding drift in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Embedding drift is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for embedding drift:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of embedding drift executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for embedding drift should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

## 5. Alerting and Diagnosis

Alerting and Diagnosis develops the part of monitoring drift and retraining assigned by the approved Chapter 19 table of contents. The treatment is production-focused: every idea is connected to a versioned artifact, measurable signal, release decision, or incident response.

### 5.1 threshold tuning

Threshold tuning is part of the canonical scope of Monitoring Drift and Retraining. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is production monitoring signals, drift statistics, alerting, diagnosis, retraining policies, and LLM production drift. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$S_t = d(p_t, p_{\mathrm{ref}}).$$

The formula is intentionally simple. It says that threshold tuning should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of threshold tuning in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Threshold tuning is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for threshold tuning:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of threshold tuning executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for threshold tuning should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 5.2 severity levels

Severity levels is part of the canonical scope of Monitoring Drift and Retraining. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is production monitoring signals, drift statistics, alerting, diagnosis, retraining policies, and LLM production drift. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\operatorname{PSI} = \sum_{b=1}^{B}(q_b-p_b)\log\frac{q_b}{p_b}.$$

The formula is intentionally simple. It says that severity levels should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of severity levels in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Severity levels is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for severity levels:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of severity levels executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for severity levels should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 5.3 slice analysis

Slice analysis is part of the canonical scope of Monitoring Drift and Retraining. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is production monitoring signals, drift statistics, alerting, diagnosis, retraining policies, and LLM production drift. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$D_{\mathrm{JS}}(p\Vert q)=\frac{1}{2}D_{\mathrm{KL}}(p\Vert m)+\frac{1}{2}D_{\mathrm{KL}}(q\Vert m), \quad m=\frac{p+q}{2}.$$

The formula is intentionally simple. It says that slice analysis should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of slice analysis in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Slice analysis is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for slice analysis:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of slice analysis executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for slice analysis should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 5.4 false positives

False positives is part of the canonical scope of Monitoring Drift and Retraining. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is production monitoring signals, drift statistics, alerting, diagnosis, retraining policies, and LLM production drift. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\pi_{\mathrm{retrain}}(t)=\mathbb{1}[S_t>\tau]\mathbb{1}[Q_t=1].$$

The formula is intentionally simple. It says that false positives should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of false positives in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. False positives is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for false positives:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of false positives executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for false positives should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 5.5 root-cause workflow

Root-cause workflow is part of the canonical scope of Monitoring Drift and Retraining. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is production monitoring signals, drift statistics, alerting, diagnosis, retraining policies, and LLM production drift. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$S_t = d(p_t, p_{\mathrm{ref}}).$$

The formula is intentionally simple. It says that root-cause workflow should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of root-cause workflow in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Root-cause workflow is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for root-cause workflow:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of root-cause workflow executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for root-cause workflow should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

## 6. Retraining Systems

Retraining Systems develops the part of monitoring drift and retraining assigned by the approved Chapter 19 table of contents. The treatment is production-focused: every idea is connected to a versioned artifact, measurable signal, release decision, or incident response.

### 6.1 scheduled retraining

Scheduled retraining is part of the canonical scope of Monitoring Drift and Retraining. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is production monitoring signals, drift statistics, alerting, diagnosis, retraining policies, and LLM production drift. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\operatorname{PSI} = \sum_{b=1}^{B}(q_b-p_b)\log\frac{q_b}{p_b}.$$

The formula is intentionally simple. It says that scheduled retraining should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of scheduled retraining in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Scheduled retraining is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for scheduled retraining:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of scheduled retraining executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for scheduled retraining should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 6.2 trigger-based retraining

Trigger-based retraining is part of the canonical scope of Monitoring Drift and Retraining. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is production monitoring signals, drift statistics, alerting, diagnosis, retraining policies, and LLM production drift. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$D_{\mathrm{JS}}(p\Vert q)=\frac{1}{2}D_{\mathrm{KL}}(p\Vert m)+\frac{1}{2}D_{\mathrm{KL}}(q\Vert m), \quad m=\frac{p+q}{2}.$$

The formula is intentionally simple. It says that trigger-based retraining should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of trigger-based retraining in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Trigger-based retraining is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for trigger-based retraining:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of trigger-based retraining executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for trigger-based retraining should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 6.3 champion challenger

Champion challenger is part of the canonical scope of Monitoring Drift and Retraining. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is production monitoring signals, drift statistics, alerting, diagnosis, retraining policies, and LLM production drift. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\pi_{\mathrm{retrain}}(t)=\mathbb{1}[S_t>\tau]\mathbb{1}[Q_t=1].$$

The formula is intentionally simple. It says that champion challenger should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of champion challenger in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Champion challenger is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for champion challenger:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of champion challenger executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for champion challenger should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 6.4 rollback-safe retraining

Rollback-safe retraining is part of the canonical scope of Monitoring Drift and Retraining. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is production monitoring signals, drift statistics, alerting, diagnosis, retraining policies, and LLM production drift. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$S_t = d(p_t, p_{\mathrm{ref}}).$$

The formula is intentionally simple. It says that rollback-safe retraining should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of rollback-safe retraining in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Rollback-safe retraining is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for rollback-safe retraining:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of rollback-safe retraining executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for rollback-safe retraining should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 6.5 data freshness gates

Data freshness gates is part of the canonical scope of Monitoring Drift and Retraining. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is production monitoring signals, drift statistics, alerting, diagnosis, retraining policies, and LLM production drift. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\operatorname{PSI} = \sum_{b=1}^{B}(q_b-p_b)\log\frac{q_b}{p_b}.$$

The formula is intentionally simple. It says that data freshness gates should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of data freshness gates in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Data freshness gates is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for data freshness gates:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of data freshness gates executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for data freshness gates should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

## 7. LLM Production Drift

LLM Production Drift develops the part of monitoring drift and retraining assigned by the approved Chapter 19 table of contents. The treatment is production-focused: every idea is connected to a versioned artifact, measurable signal, release decision, or incident response.

### 7.1 prompt distribution drift

Prompt distribution drift is part of the canonical scope of Monitoring Drift and Retraining. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is production monitoring signals, drift statistics, alerting, diagnosis, retraining policies, and LLM production drift. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$D_{\mathrm{JS}}(p\Vert q)=\frac{1}{2}D_{\mathrm{KL}}(p\Vert m)+\frac{1}{2}D_{\mathrm{KL}}(q\Vert m), \quad m=\frac{p+q}{2}.$$

The formula is intentionally simple. It says that prompt distribution drift should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of prompt distribution drift in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Prompt distribution drift is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for prompt distribution drift:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of prompt distribution drift executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for prompt distribution drift should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 7.2 retrieval corpus drift

Retrieval corpus drift is part of the canonical scope of Monitoring Drift and Retraining. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is production monitoring signals, drift statistics, alerting, diagnosis, retraining policies, and LLM production drift. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\pi_{\mathrm{retrain}}(t)=\mathbb{1}[S_t>\tau]\mathbb{1}[Q_t=1].$$

The formula is intentionally simple. It says that retrieval corpus drift should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of retrieval corpus drift in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Retrieval corpus drift is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for retrieval corpus drift:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of retrieval corpus drift executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for retrieval corpus drift should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 7.3 judge drift

Judge drift is part of the canonical scope of Monitoring Drift and Retraining. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is production monitoring signals, drift statistics, alerting, diagnosis, retraining policies, and LLM production drift. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$S_t = d(p_t, p_{\mathrm{ref}}).$$

The formula is intentionally simple. It says that judge drift should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of judge drift in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Judge drift is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for judge drift:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of judge drift executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for judge drift should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 7.4 cost and latency drift

Cost and latency drift is part of the canonical scope of Monitoring Drift and Retraining. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is production monitoring signals, drift statistics, alerting, diagnosis, retraining policies, and LLM production drift. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\operatorname{PSI} = \sum_{b=1}^{B}(q_b-p_b)\log\frac{q_b}{p_b}.$$

The formula is intentionally simple. It says that cost and latency drift should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of cost and latency drift in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Cost and latency drift is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for cost and latency drift:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of cost and latency drift executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for cost and latency drift should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 7.5 behavior regression

Behavior regression is part of the canonical scope of Monitoring Drift and Retraining. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is production monitoring signals, drift statistics, alerting, diagnosis, retraining policies, and LLM production drift. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$D_{\mathrm{JS}}(p\Vert q)=\frac{1}{2}D_{\mathrm{KL}}(p\Vert m)+\frac{1}{2}D_{\mathrm{KL}}(q\Vert m), \quad m=\frac{p+q}{2}.$$

The formula is intentionally simple. It says that behavior regression should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of behavior regression in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Behavior regression is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for behavior regression:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of behavior regression executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for behavior regression should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

## 8. Common Mistakes

| # | Mistake | Why It Is Wrong | Fix |
| --- | --- | --- | --- |
| 1 | Treating production metadata as optional | Without metadata, failures cannot be attributed to a dataset, run, endpoint, prompt, or release. | Make identifiers, hashes, versions, and owners part of the production contract. |
| 2 | Optimizing one metric in isolation | Single metrics hide tail latency, subgroup failure, safety regressions, and cost explosions. | Use metric hierarchies with guardrails and release gates. |
| 3 | Comparing runs without controlling variance | A one-run improvement can be noise, seed luck, or validation leakage. | Use repeated runs, confidence intervals, paired comparisons, and frozen evaluation sets. |
| 4 | Letting dashboards replace decisions | A dashboard can display signals without encoding what action should follow. | Tie every alert to an owner, severity, runbook, and rollback or retraining policy. |
| 5 | Ignoring training-serving skew | The model learns one feature distribution and serves on another. | Use shared transformations, point-in-time joins, contract tests, and skew monitors. |
| 6 | Deploying without rollback evidence | A rollback is impossible if the previous artifacts and dependencies are not recoverable. | Keep model, data, config, endpoint, and environment versions in the registry. |
| 7 | Using raw thresholds without calibration | Bad thresholds create alert floods or missed incidents. | Tune thresholds on historical incidents and measure false positives and false negatives. |
| 8 | Conflating evaluation, monitoring, and alignment | Offline evals, online telemetry, and safety policy answer different questions. | Keep chapter boundaries clear and connect them through release gates. |
| 9 | Forgetting cost as a reliability metric | A system that is accurate but unaffordable fails in production. | Track tokens, GPU time, cache hit rate, and cost per successful task. |
| 10 | Overfitting production fixes to one incident | A narrow patch can pass the incident case while worsening the broader distribution. | Convert incidents into regression tests, then run full capability and safety suites. |

## 9. Exercises

1. (*) Design a production ML check related to monitoring drift and retraining.
   - (a) Define the object being checked using mathematical notation.
   - (b) State the metric, predicate, or threshold used to decide pass/fail.
   - (c) Explain which artifact versions must be logged.
   - (d) Give one failure case and one rollback or escalation action.

2. (*) Design a production ML check related to monitoring drift and retraining.
   - (a) Define the object being checked using mathematical notation.
   - (b) State the metric, predicate, or threshold used to decide pass/fail.
   - (c) Explain which artifact versions must be logged.
   - (d) Give one failure case and one rollback or escalation action.

3. (*) Design a production ML check related to monitoring drift and retraining.
   - (a) Define the object being checked using mathematical notation.
   - (b) State the metric, predicate, or threshold used to decide pass/fail.
   - (c) Explain which artifact versions must be logged.
   - (d) Give one failure case and one rollback or escalation action.

4. (**) Design a production ML check related to monitoring drift and retraining.
   - (a) Define the object being checked using mathematical notation.
   - (b) State the metric, predicate, or threshold used to decide pass/fail.
   - (c) Explain which artifact versions must be logged.
   - (d) Give one failure case and one rollback or escalation action.

5. (**) Design a production ML check related to monitoring drift and retraining.
   - (a) Define the object being checked using mathematical notation.
   - (b) State the metric, predicate, or threshold used to decide pass/fail.
   - (c) Explain which artifact versions must be logged.
   - (d) Give one failure case and one rollback or escalation action.

6. (**) Design a production ML check related to monitoring drift and retraining.
   - (a) Define the object being checked using mathematical notation.
   - (b) State the metric, predicate, or threshold used to decide pass/fail.
   - (c) Explain which artifact versions must be logged.
   - (d) Give one failure case and one rollback or escalation action.

7. (***) Design a production ML check related to monitoring drift and retraining.
   - (a) Define the object being checked using mathematical notation.
   - (b) State the metric, predicate, or threshold used to decide pass/fail.
   - (c) Explain which artifact versions must be logged.
   - (d) Give one failure case and one rollback or escalation action.

8. (***) Design a production ML check related to monitoring drift and retraining.
   - (a) Define the object being checked using mathematical notation.
   - (b) State the metric, predicate, or threshold used to decide pass/fail.
   - (c) Explain which artifact versions must be logged.
   - (d) Give one failure case and one rollback or escalation action.

9. (***) Design a production ML check related to monitoring drift and retraining.
   - (a) Define the object being checked using mathematical notation.
   - (b) State the metric, predicate, or threshold used to decide pass/fail.
   - (c) Explain which artifact versions must be logged.
   - (d) Give one failure case and one rollback or escalation action.

10. (***) Design a production ML check related to monitoring drift and retraining.
   - (a) Define the object being checked using mathematical notation.
   - (b) State the metric, predicate, or threshold used to decide pass/fail.
   - (c) Explain which artifact versions must be logged.
   - (d) Give one failure case and one rollback or escalation action.

## 10. Why This Matters for AI

| Concept | AI Impact |
| --- | --- |
| Versioned artifacts | Make model behavior reproducible after a production incident |
| Lineage graphs | Reveal which upstream data, prompt, feature, or code change caused a downstream regression |
| Release gates | Prevent models from shipping on quality alone while safety, latency, or cost fails |
| Drift statistics | Convert changing user behavior into measurable maintenance signals |
| LLM traces | Explain failures across prompts, retrieval, tools, guardrails, and generated responses |
| Contracts | Catch invalid data before it silently corrupts training or serving |
| Registries | Preserve rollback candidates and promotion evidence |
| Observability | Turns production behavior into data for future evaluation and retraining |

## 11. Conceptual Bridge

Monitoring Drift and Retraining sits after the chapters on data construction, evaluation, and alignment because production systems combine all three. Chapter 16 explains how reliable datasets are assembled. Chapter 17 explains how models are measured. Chapter 18 explains how desired behavior and safety constraints are specified. Chapter 19 asks whether those ideas survive contact with changing data, users, services, and costs.

The backward bridge is operational memory. If a model fails today, the team must recover the data, code, environment, model, endpoint, prompt, retriever, guardrail, and metric definitions that produced the behavior. That is why the notation in this chapter emphasizes hashes, graphs, traces, thresholds, and predicates.

The forward bridge is broader mathematical maturity. Later chapters return to signal processing, learning theory, causal inference, game theory, measure theory, and geometry. Production ML uses those ideas under constraints: bounded latency, incomplete labels, shifting distributions, and costly human attention.

```text
+--------------------------------------------------------------+
| Chapter 16: data construction and governance                 |
| Chapter 17: evaluation and reliability                       |
| Chapter 18: alignment and safety                             |
| Chapter 19: production ML and MLOps                          |
|   artifact -> endpoint -> telemetry -> alert -> retrain      |
| Chapter 20+: mathematical tools for deeper modeling          |
+--------------------------------------------------------------+
```

## References

- Evidently. Data drift metrics. https://docs.evidentlyai.com/metrics/preset_data_drift
- OpenTelemetry. Observability signals. https://opentelemetry.io/docs/what-is-opentelemetry/
- Google Cloud. MLOps continuous delivery for ML. https://cloud.google.com/solutions/machine-learning/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning
- Sculley et al.. Hidden Technical Debt in Machine Learning Systems. https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-syst
