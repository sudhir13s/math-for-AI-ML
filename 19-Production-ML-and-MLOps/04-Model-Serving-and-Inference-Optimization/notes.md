[Back to Curriculum](../../README.md) | [Previous: Feature Stores and Data Contracts](../03-Feature-Stores-and-Data-Contracts/notes.md) | [Next: Monitoring Drift and Retraining](../05-Monitoring-Drift-and-Retraining/notes.md)

---

# Model Serving and Inference Optimization

> _"A model is production-ready only when its predictions arrive on time, at cost, and under control."_

## Overview

Serving mathematics connects model quality to latency, throughput, utilization, and rollback-safe release decisions.

Production ML and MLOps are the mathematical discipline of keeping a learned system useful after it leaves the notebook. The model is only one artifact in a larger graph of data, code, configuration, evaluation, deployment, monitoring, and response actions.

This chapter uses LaTeX Markdown throughout. Inline mathematics uses `$...$`, and display equations use `$$...$$`. The central habit is to turn production behavior into explicit objects: versions, hashes, traces, thresholds, queues, contracts, and release decisions.

## Prerequisites

- [Numerical Linear Algebra](../../10-Numerical-Methods/02-Numerical-Linear-Algebra/notes.md)
- [Learning Rate Schedules](../../08-Optimization/10-Learning-Rate-Schedules/notes.md)
- [Feature Stores and Data Contracts](../03-Feature-Stores-and-Data-Contracts/notes.md)
- [Online Experimentation and AB Testing](../../17-Evaluation-and-Reliability/05-Online-Experimentation-and-AB-Testing/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Executable demonstrations for model serving and inference optimization |
| [exercises.ipynb](exercises.ipynb) | Graded practice for model serving and inference optimization |

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
  - [1.1 training optimizes loss while serving optimizes systems constraints](#11-training-optimizes-loss-while-serving-optimizes-systems-constraints)
  - [1.2 latency throughput cost triangle](#12-latency-throughput-cost-triangle)
  - [1.3 batch versus online inference](#13-batch-versus-online-inference)
  - [1.4 model endpoints as contracts](#14-model-endpoints-as-contracts)
  - [1.5 failure budgets](#15-failure-budgets)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 service $s(\mathbf{x})$](#21-service)
  - [2.2 latency distribution](#22-latency-distribution)
  - [2.3 throughput $q$](#23-throughput)
  - [2.4 utilization $\rho$](#24-utilization)
  - [2.5 service-level objectives and service-level agreements](#25-servicelevel-objectives-and-servicelevel-agreements)
- [3. Serving Architectures](#3-serving-architectures)
  - [3.1 REST and gRPC endpoints](#31-rest-and-grpc-endpoints)
  - [3.2 batch serving](#32-batch-serving)
  - [3.3 streaming inference](#33-streaming-inference)
  - [3.4 model routers](#34-model-routers)
  - [3.5 shadow and canary deployments](#35-shadow-and-canary-deployments)
- [4. Inference Optimization](#4-inference-optimization)
  - [4.1 batching](#41-batching)
  - [4.2 caching](#42-caching)
  - [4.3 quantization preview](#43-quantization-preview)
  - [4.4 distillation preview](#44-distillation-preview)
  - [4.5 hardware-aware scheduling](#45-hardwareaware-scheduling)
- [5. Queueing and Capacity](#5-queueing-and-capacity)
  - [5.1 arrival rates](#51-arrival-rates)
  - [5.2 Little's law](#52-littles-law)
  - [5.3 tail latency](#53-tail-latency)
  - [5.4 autoscaling](#54-autoscaling)
  - [5.5 overload control](#55-overload-control)
- [6. Deployment Safety](#6-deployment-safety)
  - [6.1 versioned endpoints](#61-versioned-endpoints)
  - [6.2 blue-green deploys](#62-bluegreen-deploys)
  - [6.3 canary metrics](#63-canary-metrics)
  - [6.4 rollback triggers](#64-rollback-triggers)
  - [6.5 incident playbooks](#65-incident-playbooks)
- [7. LLM Serving](#7-llm-serving)
  - [7.1 token latency](#71-token-latency)
  - [7.2 KV cache](#72-kv-cache)
  - [7.3 prompt batching](#73-prompt-batching)
  - [7.4 model routing](#74-model-routing)
  - [7.5 cost per answer](#75-cost-per-answer)
- [8. Common Mistakes](#8-common-mistakes)
- [9. Exercises](#9-exercises)
- [10. Why This Matters for AI](#10-why-this-matters-for-ai)
- [11. Conceptual Bridge](#11-conceptual-bridge)
- [References](#references)

---

## 1. Intuition

Intuition develops the part of model serving and inference optimization assigned by the approved Chapter 19 table of contents. The treatment is production-focused: every idea is connected to a versioned artifact, measurable signal, release decision, or incident response.

### 1.1 training optimizes loss while serving optimizes systems constraints

Training optimizes loss while serving optimizes systems constraints is part of the canonical scope of Model Serving and Inference Optimization. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is serving architectures, inference optimization, queueing, capacity planning, deployment safety, and LLM serving economics. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\rho = \frac{\lambda}{\mu}, \qquad 0 \le \rho < 1.$$

The formula is intentionally simple. It says that training optimizes loss while serving optimizes systems constraints should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of training optimizes loss while serving optimizes systems constraints in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Training optimizes loss while serving optimizes systems constraints is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for training optimizes loss while serving optimizes systems constraints:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of training optimizes loss while serving optimizes systems constraints executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for training optimizes loss while serving optimizes systems constraints should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 1.2 latency throughput cost triangle

Latency throughput cost triangle is part of the canonical scope of Model Serving and Inference Optimization. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is serving architectures, inference optimization, queueing, capacity planning, deployment safety, and LLM serving economics. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$L = \lambda W.$$

The formula is intentionally simple. It says that latency throughput cost triangle should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of latency throughput cost triangle in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Latency throughput cost triangle is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for latency throughput cost triangle:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of latency throughput cost triangle executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for latency throughput cost triangle should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 1.3 batch versus online inference

Batch versus online inference is part of the canonical scope of Model Serving and Inference Optimization. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is serving architectures, inference optimization, queueing, capacity planning, deployment safety, and LLM serving economics. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$p95 = \inf\{t : F_T(t) \ge 0.95\}.$$

The formula is intentionally simple. It says that batch versus online inference should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of batch versus online inference in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Batch versus online inference is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for batch versus online inference:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of batch versus online inference executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for batch versus online inference should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 1.4 model endpoints as contracts

Model endpoints as contracts is part of the canonical scope of Model Serving and Inference Optimization. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is serving architectures, inference optimization, queueing, capacity planning, deployment safety, and LLM serving economics. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\operatorname{cost}(y) = c_{\mathrm{in}}n_{\mathrm{in}} + c_{\mathrm{out}}n_{\mathrm{out}} + c_{\mathrm{gpu}}T.$$

The formula is intentionally simple. It says that model endpoints as contracts should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of model endpoints as contracts in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Model endpoints as contracts is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for model endpoints as contracts:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of model endpoints as contracts executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for model endpoints as contracts should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 1.5 failure budgets

Failure budgets is part of the canonical scope of Model Serving and Inference Optimization. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is serving architectures, inference optimization, queueing, capacity planning, deployment safety, and LLM serving economics. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\rho = \frac{\lambda}{\mu}, \qquad 0 \le \rho < 1.$$

The formula is intentionally simple. It says that failure budgets should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of failure budgets in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Failure budgets is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for failure budgets:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of failure budgets executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for failure budgets should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

## 2. Formal Definitions

Formal Definitions develops the part of model serving and inference optimization assigned by the approved Chapter 19 table of contents. The treatment is production-focused: every idea is connected to a versioned artifact, measurable signal, release decision, or incident response.

### 2.1 service $s(\mathbf{x})$

Service $s(\mathbf{x})$ is part of the canonical scope of Model Serving and Inference Optimization. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is serving architectures, inference optimization, queueing, capacity planning, deployment safety, and LLM serving economics. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$L = \lambda W.$$

The formula is intentionally simple. It says that service $s(\mathbf{x})$ should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of service $s(\mathbf{x})$ in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Service $s(\mathbf{x})$ is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for service $s(\mathbf{x})$:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of service $s(\mathbf{x})$ executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for service $s(\mathbf{x})$ should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 2.2 latency distribution

Latency distribution is part of the canonical scope of Model Serving and Inference Optimization. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is serving architectures, inference optimization, queueing, capacity planning, deployment safety, and LLM serving economics. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$p95 = \inf\{t : F_T(t) \ge 0.95\}.$$

The formula is intentionally simple. It says that latency distribution should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of latency distribution in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Latency distribution is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for latency distribution:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of latency distribution executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for latency distribution should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 2.3 throughput $q$

Throughput $q$ is part of the canonical scope of Model Serving and Inference Optimization. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is serving architectures, inference optimization, queueing, capacity planning, deployment safety, and LLM serving economics. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\operatorname{cost}(y) = c_{\mathrm{in}}n_{\mathrm{in}} + c_{\mathrm{out}}n_{\mathrm{out}} + c_{\mathrm{gpu}}T.$$

The formula is intentionally simple. It says that throughput $q$ should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of throughput $q$ in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Throughput $q$ is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for throughput $q$:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of throughput $q$ executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for throughput $q$ should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 2.4 utilization $\rho$

Utilization $\rho$ is part of the canonical scope of Model Serving and Inference Optimization. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is serving architectures, inference optimization, queueing, capacity planning, deployment safety, and LLM serving economics. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\rho = \frac{\lambda}{\mu}, \qquad 0 \le \rho < 1.$$

The formula is intentionally simple. It says that utilization $\rho$ should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of utilization $\rho$ in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Utilization $\rho$ is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for utilization $\rho$:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of utilization $\rho$ executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for utilization $\rho$ should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 2.5 service-level objectives and service-level agreements

Service-level objectives and service-level agreements is part of the canonical scope of Model Serving and Inference Optimization. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is serving architectures, inference optimization, queueing, capacity planning, deployment safety, and LLM serving economics. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$L = \lambda W.$$

The formula is intentionally simple. It says that service-level objectives and service-level agreements should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of service-level objectives and service-level agreements in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Service-level objectives and service-level agreements is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for service-level objectives and service-level agreements:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of service-level objectives and service-level agreements executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for service-level objectives and service-level agreements should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

## 3. Serving Architectures

Serving Architectures develops the part of model serving and inference optimization assigned by the approved Chapter 19 table of contents. The treatment is production-focused: every idea is connected to a versioned artifact, measurable signal, release decision, or incident response.

### 3.1 REST and gRPC endpoints

Rest and grpc endpoints is part of the canonical scope of Model Serving and Inference Optimization. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is serving architectures, inference optimization, queueing, capacity planning, deployment safety, and LLM serving economics. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$p95 = \inf\{t : F_T(t) \ge 0.95\}.$$

The formula is intentionally simple. It says that rest and grpc endpoints should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of rest and grpc endpoints in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Rest and grpc endpoints is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for rest and grpc endpoints:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of rest and grpc endpoints executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for rest and grpc endpoints should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 3.2 batch serving

Batch serving is part of the canonical scope of Model Serving and Inference Optimization. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is serving architectures, inference optimization, queueing, capacity planning, deployment safety, and LLM serving economics. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\operatorname{cost}(y) = c_{\mathrm{in}}n_{\mathrm{in}} + c_{\mathrm{out}}n_{\mathrm{out}} + c_{\mathrm{gpu}}T.$$

The formula is intentionally simple. It says that batch serving should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of batch serving in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Batch serving is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for batch serving:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of batch serving executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for batch serving should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 3.3 streaming inference

Streaming inference is part of the canonical scope of Model Serving and Inference Optimization. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is serving architectures, inference optimization, queueing, capacity planning, deployment safety, and LLM serving economics. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\rho = \frac{\lambda}{\mu}, \qquad 0 \le \rho < 1.$$

The formula is intentionally simple. It says that streaming inference should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of streaming inference in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Streaming inference is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for streaming inference:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of streaming inference executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for streaming inference should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 3.4 model routers

Model routers is part of the canonical scope of Model Serving and Inference Optimization. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is serving architectures, inference optimization, queueing, capacity planning, deployment safety, and LLM serving economics. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$L = \lambda W.$$

The formula is intentionally simple. It says that model routers should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of model routers in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Model routers is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for model routers:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of model routers executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for model routers should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 3.5 shadow and canary deployments

Shadow and canary deployments is part of the canonical scope of Model Serving and Inference Optimization. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is serving architectures, inference optimization, queueing, capacity planning, deployment safety, and LLM serving economics. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$p95 = \inf\{t : F_T(t) \ge 0.95\}.$$

The formula is intentionally simple. It says that shadow and canary deployments should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of shadow and canary deployments in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Shadow and canary deployments is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for shadow and canary deployments:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of shadow and canary deployments executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for shadow and canary deployments should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

## 4. Inference Optimization

Inference Optimization develops the part of model serving and inference optimization assigned by the approved Chapter 19 table of contents. The treatment is production-focused: every idea is connected to a versioned artifact, measurable signal, release decision, or incident response.

### 4.1 batching

Batching is part of the canonical scope of Model Serving and Inference Optimization. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is serving architectures, inference optimization, queueing, capacity planning, deployment safety, and LLM serving economics. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\operatorname{cost}(y) = c_{\mathrm{in}}n_{\mathrm{in}} + c_{\mathrm{out}}n_{\mathrm{out}} + c_{\mathrm{gpu}}T.$$

The formula is intentionally simple. It says that batching should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of batching in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Batching is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for batching:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of batching executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for batching should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 4.2 caching

Caching is part of the canonical scope of Model Serving and Inference Optimization. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is serving architectures, inference optimization, queueing, capacity planning, deployment safety, and LLM serving economics. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\rho = \frac{\lambda}{\mu}, \qquad 0 \le \rho < 1.$$

The formula is intentionally simple. It says that caching should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of caching in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Caching is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for caching:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of caching executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for caching should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 4.3 quantization preview

Quantization preview is part of the canonical scope of Model Serving and Inference Optimization. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is serving architectures, inference optimization, queueing, capacity planning, deployment safety, and LLM serving economics. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$L = \lambda W.$$

The formula is intentionally simple. It says that quantization preview should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of quantization preview in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Quantization preview is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for quantization preview:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of quantization preview executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for quantization preview should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 4.4 distillation preview

Distillation preview is part of the canonical scope of Model Serving and Inference Optimization. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is serving architectures, inference optimization, queueing, capacity planning, deployment safety, and LLM serving economics. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$p95 = \inf\{t : F_T(t) \ge 0.95\}.$$

The formula is intentionally simple. It says that distillation preview should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of distillation preview in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Distillation preview is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for distillation preview:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of distillation preview executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for distillation preview should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 4.5 hardware-aware scheduling

Hardware-aware scheduling is part of the canonical scope of Model Serving and Inference Optimization. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is serving architectures, inference optimization, queueing, capacity planning, deployment safety, and LLM serving economics. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\operatorname{cost}(y) = c_{\mathrm{in}}n_{\mathrm{in}} + c_{\mathrm{out}}n_{\mathrm{out}} + c_{\mathrm{gpu}}T.$$

The formula is intentionally simple. It says that hardware-aware scheduling should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of hardware-aware scheduling in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Hardware-aware scheduling is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for hardware-aware scheduling:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of hardware-aware scheduling executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for hardware-aware scheduling should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

## 5. Queueing and Capacity

Queueing and Capacity develops the part of model serving and inference optimization assigned by the approved Chapter 19 table of contents. The treatment is production-focused: every idea is connected to a versioned artifact, measurable signal, release decision, or incident response.

### 5.1 arrival rates

Arrival rates is part of the canonical scope of Model Serving and Inference Optimization. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is serving architectures, inference optimization, queueing, capacity planning, deployment safety, and LLM serving economics. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\rho = \frac{\lambda}{\mu}, \qquad 0 \le \rho < 1.$$

The formula is intentionally simple. It says that arrival rates should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of arrival rates in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Arrival rates is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for arrival rates:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of arrival rates executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for arrival rates should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 5.2 Little's law

Little's law is part of the canonical scope of Model Serving and Inference Optimization. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is serving architectures, inference optimization, queueing, capacity planning, deployment safety, and LLM serving economics. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$L = \lambda W.$$

The formula is intentionally simple. It says that little's law should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of little's law in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Little's law is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for little's law:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of little's law executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for little's law should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 5.3 tail latency

Tail latency is part of the canonical scope of Model Serving and Inference Optimization. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is serving architectures, inference optimization, queueing, capacity planning, deployment safety, and LLM serving economics. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$p95 = \inf\{t : F_T(t) \ge 0.95\}.$$

The formula is intentionally simple. It says that tail latency should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of tail latency in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Tail latency is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for tail latency:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of tail latency executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for tail latency should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 5.4 autoscaling

Autoscaling is part of the canonical scope of Model Serving and Inference Optimization. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is serving architectures, inference optimization, queueing, capacity planning, deployment safety, and LLM serving economics. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\operatorname{cost}(y) = c_{\mathrm{in}}n_{\mathrm{in}} + c_{\mathrm{out}}n_{\mathrm{out}} + c_{\mathrm{gpu}}T.$$

The formula is intentionally simple. It says that autoscaling should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of autoscaling in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Autoscaling is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for autoscaling:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of autoscaling executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for autoscaling should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 5.5 overload control

Overload control is part of the canonical scope of Model Serving and Inference Optimization. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is serving architectures, inference optimization, queueing, capacity planning, deployment safety, and LLM serving economics. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\rho = \frac{\lambda}{\mu}, \qquad 0 \le \rho < 1.$$

The formula is intentionally simple. It says that overload control should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of overload control in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Overload control is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for overload control:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of overload control executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for overload control should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

## 6. Deployment Safety

Deployment Safety develops the part of model serving and inference optimization assigned by the approved Chapter 19 table of contents. The treatment is production-focused: every idea is connected to a versioned artifact, measurable signal, release decision, or incident response.

### 6.1 versioned endpoints

Versioned endpoints is part of the canonical scope of Model Serving and Inference Optimization. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is serving architectures, inference optimization, queueing, capacity planning, deployment safety, and LLM serving economics. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$L = \lambda W.$$

The formula is intentionally simple. It says that versioned endpoints should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of versioned endpoints in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Versioned endpoints is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for versioned endpoints:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of versioned endpoints executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for versioned endpoints should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 6.2 blue-green deploys

Blue-green deploys is part of the canonical scope of Model Serving and Inference Optimization. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is serving architectures, inference optimization, queueing, capacity planning, deployment safety, and LLM serving economics. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$p95 = \inf\{t : F_T(t) \ge 0.95\}.$$

The formula is intentionally simple. It says that blue-green deploys should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of blue-green deploys in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Blue-green deploys is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for blue-green deploys:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of blue-green deploys executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for blue-green deploys should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 6.3 canary metrics

Canary metrics is part of the canonical scope of Model Serving and Inference Optimization. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is serving architectures, inference optimization, queueing, capacity planning, deployment safety, and LLM serving economics. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\operatorname{cost}(y) = c_{\mathrm{in}}n_{\mathrm{in}} + c_{\mathrm{out}}n_{\mathrm{out}} + c_{\mathrm{gpu}}T.$$

The formula is intentionally simple. It says that canary metrics should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of canary metrics in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Canary metrics is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for canary metrics:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of canary metrics executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for canary metrics should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 6.4 rollback triggers

Rollback triggers is part of the canonical scope of Model Serving and Inference Optimization. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is serving architectures, inference optimization, queueing, capacity planning, deployment safety, and LLM serving economics. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\rho = \frac{\lambda}{\mu}, \qquad 0 \le \rho < 1.$$

The formula is intentionally simple. It says that rollback triggers should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of rollback triggers in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Rollback triggers is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for rollback triggers:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of rollback triggers executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for rollback triggers should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 6.5 incident playbooks

Incident playbooks is part of the canonical scope of Model Serving and Inference Optimization. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is serving architectures, inference optimization, queueing, capacity planning, deployment safety, and LLM serving economics. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$L = \lambda W.$$

The formula is intentionally simple. It says that incident playbooks should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of incident playbooks in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Incident playbooks is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for incident playbooks:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of incident playbooks executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for incident playbooks should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

## 7. LLM Serving

LLM Serving develops the part of model serving and inference optimization assigned by the approved Chapter 19 table of contents. The treatment is production-focused: every idea is connected to a versioned artifact, measurable signal, release decision, or incident response.

### 7.1 token latency

Token latency is part of the canonical scope of Model Serving and Inference Optimization. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is serving architectures, inference optimization, queueing, capacity planning, deployment safety, and LLM serving economics. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$p95 = \inf\{t : F_T(t) \ge 0.95\}.$$

The formula is intentionally simple. It says that token latency should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of token latency in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Token latency is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for token latency:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of token latency executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for token latency should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 7.2 KV cache

Kv cache is part of the canonical scope of Model Serving and Inference Optimization. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is serving architectures, inference optimization, queueing, capacity planning, deployment safety, and LLM serving economics. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\operatorname{cost}(y) = c_{\mathrm{in}}n_{\mathrm{in}} + c_{\mathrm{out}}n_{\mathrm{out}} + c_{\mathrm{gpu}}T.$$

The formula is intentionally simple. It says that kv cache should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of kv cache in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Kv cache is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for kv cache:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of kv cache executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for kv cache should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 7.3 prompt batching

Prompt batching is part of the canonical scope of Model Serving and Inference Optimization. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is serving architectures, inference optimization, queueing, capacity planning, deployment safety, and LLM serving economics. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$\rho = \frac{\lambda}{\mu}, \qquad 0 \le \rho < 1.$$

The formula is intentionally simple. It says that prompt batching should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of prompt batching in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Prompt batching is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for prompt batching:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of prompt batching executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for prompt batching should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 7.4 model routing

Model routing is part of the canonical scope of Model Serving and Inference Optimization. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is serving architectures, inference optimization, queueing, capacity planning, deployment safety, and LLM serving economics. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$L = \lambda W.$$

The formula is intentionally simple. It says that model routing should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of model routing in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Model routing is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for model routing:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of model routing executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for model routing should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

| Failure question | Production test | Response |
| --- | --- | --- |
| Is the artifact stale? | Compare event time to freshness limit | Warn, block, or backfill |
| Is the artifact malformed? | Evaluate schema and semantic contract | Reject before serving or training |
| Is the artifact inconsistent? | Compare current statistic with reference statistic | Investigate drift or skew |
| Is the artifact unauditable? | Check for missing version, owner, or lineage edge | Stop promotion until metadata exists |
| Is the artifact too costly? | Track latency, tokens, storage, or compute | Route, cache, batch, or downscale |

The production design pattern is therefore not just to calculate a value. It is to calculate a value, compare it with a declared rule, log the evidence, and make the next action unambiguous. That four-step pattern will reappear across all Chapter 19 notebooks.

### 7.5 cost per answer

Cost per answer is part of the canonical scope of Model Serving and Inference Optimization. In production ML, the useful question is not only whether the model can be trained, but whether the surrounding artifact, signal, or control can be named, versioned, measured, and recovered after a failure.

For this section, the working object is serving architectures, inference optimization, queueing, capacity planning, deployment safety, and LLM serving economics. The notation below treats production systems as mathematical objects because that is how incidents become diagnosable. A dataset, feature, run, trace, or endpoint that lacks a stable identifier cannot be compared across time.

$$p95 = \inf\{t : F_T(t) \ge 0.95\}.$$

The formula is intentionally simple. It says that cost per answer should be reduced to a measurable object before anyone argues about dashboards or tools. Once the object is measurable, the system can decide whether to accept, warn, rollback, retrain, or escalate.

| Production object | Mathematical role | Operational consequence |
| --- | --- | --- |
| Identifier | A stable key in a set or graph | Lets teams join logs, artifacts, and incidents |
| Version | A time-indexed element such as $v_t$ | Makes old and new behavior comparable |
| Metric | A function $m: \mathcal{X} \to \mathbb{R}$ | Turns behavior into a release or alert signal |
| Contract | A predicate $C(\cdot)$ | Rejects invalid inputs before the model absorbs them |
| Owner | A decision variable outside the model | Prevents silent failure after detection |

Examples of cost per answer in a real system:

1. A production pipeline records the input version, transformation code hash, model version, and endpoint version before serving predictions.
2. An LLM application logs prompt version, retrieval index version, tool span, latency, token count, and guardrail action for each trace.
3. A release gate compares the candidate model against the current model on quality, safety, latency, and cost before promotion.

Non-examples that often look similar but fail the production contract:

1. A manually named file like `final_dataset.csv` with no hash, schema, lineage, or owner.
2. A metric screenshot pasted into chat without the run id, evaluation dataset, seed, or model artifact.
3. A dashboard alert with no threshold rationale, no escalation rule, and no rollback candidate.

The AI connection is concrete. Modern ML and LLM systems are compound systems: data pipelines, feature stores, model registries, inference servers, retrievers, tools, evaluators, and safety layers. Cost per answer is one place where the compound system either becomes observable or becomes technical debt.

Operational checklist for cost per answer:

- State the artifact or signal being controlled.
- Give it a stable id and version.
- Define the metric or predicate that decides whether it is valid.
- Log the dependency chain needed to reproduce it.
- Attach an owner and a response action.
- Test the check in continuous integration or release gating.

A useful mental model is to treat every production ML component as a function with preconditions and postconditions. If $u$ is the upstream artifact and $z$ is the downstream artifact, the production question is whether the relation $u \mapsto z$ can be replayed and audited.

$$z = T(u; c, e),$$

where $T$ is the transformation, $c$ is code or configuration, and $e$ is the execution environment. The hidden technical debt appears when any of $u$, $c$, or $e$ is missing from the record.

In notebooks, this subsection will be represented with small synthetic arrays, graphs, traces, or counters rather than external services. The point is not to mimic a vendor tool. The point is to make the mathematics of cost per answer executable enough to test.

Boundary note: this chapter assumes the evaluation methods from Chapter 17, the safety policy ideas from Chapter 18, and the data documentation work from Chapter 16. Here we focus on the production machinery that makes those ideas run repeatedly.

Failure analysis for cost per answer should be written before the incident occurs. A good production note asks what can be stale, missing, corrupted, delayed, unaudited, or too expensive. Each answer should correspond to one observable signal and one response action.

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

1. (*) Design a production ML check related to model serving and inference optimization.
   - (a) Define the object being checked using mathematical notation.
   - (b) State the metric, predicate, or threshold used to decide pass/fail.
   - (c) Explain which artifact versions must be logged.
   - (d) Give one failure case and one rollback or escalation action.

2. (*) Design a production ML check related to model serving and inference optimization.
   - (a) Define the object being checked using mathematical notation.
   - (b) State the metric, predicate, or threshold used to decide pass/fail.
   - (c) Explain which artifact versions must be logged.
   - (d) Give one failure case and one rollback or escalation action.

3. (*) Design a production ML check related to model serving and inference optimization.
   - (a) Define the object being checked using mathematical notation.
   - (b) State the metric, predicate, or threshold used to decide pass/fail.
   - (c) Explain which artifact versions must be logged.
   - (d) Give one failure case and one rollback or escalation action.

4. (**) Design a production ML check related to model serving and inference optimization.
   - (a) Define the object being checked using mathematical notation.
   - (b) State the metric, predicate, or threshold used to decide pass/fail.
   - (c) Explain which artifact versions must be logged.
   - (d) Give one failure case and one rollback or escalation action.

5. (**) Design a production ML check related to model serving and inference optimization.
   - (a) Define the object being checked using mathematical notation.
   - (b) State the metric, predicate, or threshold used to decide pass/fail.
   - (c) Explain which artifact versions must be logged.
   - (d) Give one failure case and one rollback or escalation action.

6. (**) Design a production ML check related to model serving and inference optimization.
   - (a) Define the object being checked using mathematical notation.
   - (b) State the metric, predicate, or threshold used to decide pass/fail.
   - (c) Explain which artifact versions must be logged.
   - (d) Give one failure case and one rollback or escalation action.

7. (***) Design a production ML check related to model serving and inference optimization.
   - (a) Define the object being checked using mathematical notation.
   - (b) State the metric, predicate, or threshold used to decide pass/fail.
   - (c) Explain which artifact versions must be logged.
   - (d) Give one failure case and one rollback or escalation action.

8. (***) Design a production ML check related to model serving and inference optimization.
   - (a) Define the object being checked using mathematical notation.
   - (b) State the metric, predicate, or threshold used to decide pass/fail.
   - (c) Explain which artifact versions must be logged.
   - (d) Give one failure case and one rollback or escalation action.

9. (***) Design a production ML check related to model serving and inference optimization.
   - (a) Define the object being checked using mathematical notation.
   - (b) State the metric, predicate, or threshold used to decide pass/fail.
   - (c) Explain which artifact versions must be logged.
   - (d) Give one failure case and one rollback or escalation action.

10. (***) Design a production ML check related to model serving and inference optimization.
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

Model Serving and Inference Optimization sits after the chapters on data construction, evaluation, and alignment because production systems combine all three. Chapter 16 explains how reliable datasets are assembled. Chapter 17 explains how models are measured. Chapter 18 explains how desired behavior and safety constraints are specified. Chapter 19 asks whether those ideas survive contact with changing data, users, services, and costs.

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

- NVIDIA. Triton Inference Server user guide. https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/
- KServe. Model inference serving documentation. https://kserve.github.io/website/latest/
- Google Cloud. MLOps continuous delivery for ML. https://cloud.google.com/solutions/machine-learning/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning
- OpenTelemetry. Metrics, traces, and logs overview. https://opentelemetry.io/docs/what-is-opentelemetry/
